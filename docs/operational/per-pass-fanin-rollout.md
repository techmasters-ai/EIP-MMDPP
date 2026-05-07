# Per-Pass Celery Fan-In: Rollout Guide

## What changed

`derive_ontology_graph` went from a single monolithic Celery task with an 8-hour
soft time limit that ran all 12 ontology passes inline, to a thin dispatcher that
fans out per-pass Celery tasks backed by a DB state machine. The dispatcher writes
a summary `StageRun` (RUNNING), claims each pass in `dispatched_phases` JSONB, and
publishes one `run_extract_pass` task per pass to the broker. Each `run_extract_pass`
task executes exactly one ontology pass — HTTP call to docling-graph, StageRun write,
`pipeline_pass_outputs` row — then terminates. A separate `finalize_ontology_graph_passes`
task fires after all per-pass tasks resolve and performs the required-pass gate,
merge/resolve, three-phase graph import, and summary StageRun terminalization.

A beat-scheduled reconciler (`reconcile_ontology_graph_runs`) is the safety net for
stranded phases: it re-dispatches phases stuck in `claimed` state for more than
`PHASE_CLAIM_STALE_SECONDS` (default 30 s) and phases stuck in `dispatched` state
for more than `2 × PASS_SOFT_TIME_LIMIT` seconds without a matching
`pipeline_pass_outputs` row (indicating a worker crash after broker accept but before
task completion).

Key user-visible changes from this architecture:

- The graph_only-stuck-at-PARTIAL_COMPLETE bug is fixed (mode-scoped `REQUIRED_STAGES`
  check in `finalize_document` now correctly sees the summary StageRun written by
  the fan-in merge task, because the stage_name is the same as the legacy monolithic path).
- Failed optional passes degrade gracefully — runs still complete with the successful
  passes' graph data present; the document reaches PARTIAL_COMPLETE rather than FAILED.
- Cancel-mid-extraction is safe: `save_pass_output` swallows only the specific
  `pipeline_pass_outputs_pipeline_run_id_fkey` FK violation (indicating the run was
  deleted while the task was running) and re-raises all other integrity errors.

---

## Pre-deploy checklist

1. The migration `0019_add_pipeline_pass_outputs_and_dispatched_phases.py` is additive
   and safe to apply ahead of code (it adds `ingest.pipeline_pass_outputs` and the
   `dispatched_phases` JSONB column to `pipeline_runs` without touching existing columns
   or rows). The legacy monolithic path does not read or write these; applying the
   migration before the code deploy causes no behavioral change on the running system.

2. Ensure no in-flight `derive_ontology_graph` extractions are running at deploy time.
   The new dispatcher returns immediately after publishing per-pass tasks; existing
   inline extractions running under the old code will not be migrated automatically.
   If any runs are in-flight, either:
   - Let them complete before deploying, or
   - Plan for a `graph_only` reingest of those documents after deploy.

---

## Deploy steps

### 1. Apply migration (idempotent if already applied)

```bash
docker exec eip-mmdpp-api-1 alembic upgrade head
```

Verify:

```bash
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -c \
  "\d ingest.pipeline_pass_outputs"
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -c \
  "SELECT column_name FROM information_schema.columns
   WHERE table_schema = 'ingest' AND table_name = 'pipeline_runs'
     AND column_name = 'dispatched_phases';"
```

Both queries should return results (the table and the column must exist).

### 2. Rebuild and recreate the 4 services that need new code

```bash
docker compose -f docker-compose.yml build api worker worker-graph beat
docker compose -f docker-compose.yml up -d --no-build --no-deps api worker worker-graph beat
```

`arcadedb`, `postgres`, `redis`, `minio`, `docling`, `docling-graph`, `worker-embed`,
and `worker-ingest` are unchanged and do not need a rebuild.

### 3. Verify the beat schedule includes the reconciler

```bash
docker exec eip-mmdpp-beat-1 celery -A app.workers.celery_app inspect scheduled \
  2>&1 | grep reconcile_ontology_graph
```

The output should list `reconcile_ontology_graph_runs` in the scheduled tasks.
If it does not appear, check that the beat container has the new code (confirm
the image was rebuilt in step 2).

### 4. Smoke test with a small completed document

Use a small radar or missile PDF that has been ingested before:

```bash
curl -X POST http://localhost:8005/v1/documents/<doc_id>/reingest \
     -H "Content-Type: application/json" -d '{"mode": "graph_only"}'
```

Watch logs in parallel:

```bash
docker logs -f eip-mmdpp-worker-graph-1 &
docker logs -f eip-mmdpp-worker-1 &
```

Expected log sequence:

1. `worker-graph` logs: `[dispatcher] dispatching N passes for run <run_id>`
2. `worker` (or `worker-graph`) logs: per-pass `run_extract_pass` task starts and
   completes for each pass.
3. `worker-graph` logs: `[fan-in] all N passes terminal for run <run_id>; proceeding to merge`
4. `worker-graph` logs: merge/resolve + graph import complete.
5. Document status reaches COMPLETE (or PARTIAL_COMPLETE if any optional pass failed).

---

## Rollback

To revert to the legacy monolithic path:

1. Revert the branch's commits (or deploy the prior image tag).
2. Redeploy `worker-graph`, `worker`, `api`, and `beat` from the prior image.
3. The `pipeline_pass_outputs` table and `dispatched_phases` column remain in the DB
   but are unreferenced by the legacy code (it does not read or write them). The
   CASCADE relationship means `pipeline_pass_outputs` rows are cleaned up
   automatically when their parent `pipeline_runs` row is deleted.

Documents that were mid-extraction under the new code at rollback time will need a
`graph_only` reingest after rollback (same as the deploy case above).

---

## Operational considerations

- **Reconciler frequency**: `RECONCILER_PERIOD_SECONDS=60` (default). Lower values
  reduce stuck-state recovery latency at the cost of more frequent DB queries.
  Higher values reduce DB load at the cost of longer recovery windows. The reconciler
  query is lightweight (indexed on `status='PROCESSING'` + `mode`).

- **Per-document concurrency cap**: `PASS_CONCURRENCY_PER_DOCUMENT=2` (default).
  Worker-graph total LLM load = `cap × concurrent_documents`. With 4 documents
  ingesting simultaneously and cap=2, expect ~8 concurrent LLM calls. Raise the cap
  only if Ollama has sufficient capacity; lowering it reduces throughput but protects
  the LLM backend from saturation.

- **Pass soft time limit**: `PASS_SOFT_TIME_LIMIT=3600` (1 hour, default). If a pass
  exceeds this limit, Celery soft-kills the task by raising `SoftTimeLimitExceeded`
  inside the worker. The `guard_stage_run` wrapper in `run_extract_pass` catches this
  and terminalizes the per-pass StageRun as FAILED. The reconciler will not re-dispatch
  a pass that already has a terminal `pipeline_pass_outputs` row.

- **Worker queue routing**: `run_extract_pass` and `finalize_ontology_graph_passes`
  are routed to the `graph` queue (same as `derive_ontology_graph`). No queue
  configuration changes are required; the existing `worker-graph` service handles all
  three tasks.

---

## Known issues and carry-forward

### 1. Concurrent dispatcher race on summary StageRun

When two `derive_ontology_graph` dispatcher tasks fire for the same `(run_id, attempt)`
— for example, after a Celery task redelivery following a worker restart — only one
can successfully insert the summary StageRun. The second crashes with an
`IntegrityError` on the `uq_stage_runs_summary_row` partial unique index. The first
dispatcher's run continues normally; the second produces an ERROR log line and the
Celery task is marked as failed on the broker, but no data loss occurs (the winning
dispatcher's per-pass tasks are still in flight).

**Mitigation (not yet implemented):** wrap the StageRun INSERT in
`ON CONFLICT DO NOTHING` or catch `IntegrityError` in the dispatcher's body so the
second redelivery becomes a no-op. Search for `TODO(per-pass-fanin)` in the codebase
for the relevant marker.

### 2. `stage_run_id` always NULL on `pipeline_pass_outputs` rows

`_write_stage_run` returns `None` (it does not return the inserted row's ID). As a
result, the `stage_run_id` FK column on `pipeline_pass_outputs` is always NULL.
Operators must query by `(pipeline_run_id, pass_name, attempt)` to find the matching
`stage_runs` row rather than joining via the FK:

```sql
SELECT sr.*
  FROM ingest.stage_runs sr
  JOIN ingest.pipeline_pass_outputs ppo
    ON sr.pipeline_run_id = ppo.pipeline_run_id
   AND sr.pass_name       = ppo.pass_name
   AND sr.attempt         = ppo.attempt
 WHERE ppo.pipeline_run_id = '<run_id>';
```

### 3. JSONB size for large extract_pass_response_json

Typical pass outputs are under 100 KB. Large radar or missile documents may produce
1–5 MB per row in `pipeline_pass_outputs.extract_pass_response_json`. PostgreSQL
TOAST handles this transparently, but aggregate table size and query performance may
degrade for very large corpora. Monitor table bloat with:

```sql
SELECT pg_size_pretty(pg_total_relation_size('ingest.pipeline_pass_outputs'));
```

If the table grows beyond a few GB, consider splitting the raw response JSON into a
separate overflow table and keeping only aggregate counts in the primary row.

---

## Verification

After deploy and smoke test, inspect a successful run's state:

```sql
-- Pipeline run + dispatched_phases JSONB (one entry per pass)
SELECT id, status, mode, dispatched_phases
  FROM ingest.pipeline_runs
 WHERE id = '<run_id>';

-- Per-pass terminal records (one row per (run, pass))
SELECT pass_name, execution_status, yield_status,
       primary_entities_extracted, relationships_extracted
  FROM ingest.pipeline_pass_outputs
 WHERE pipeline_run_id = '<run_id>'
 ORDER BY pass_name;

-- Per-attempt audit (multiple rows per pass on retries)
SELECT pass_name, attempt, execution_status, finished_at
  FROM ingest.stage_runs
 WHERE pipeline_run_id = '<run_id>'
   AND pass_name IS NOT NULL
 ORDER BY pass_name, attempt;
```

Expected on a happy-path `graph_only` run for a typical radar or missile document:

- `dispatched_phases` has all entity passes + system_links + merge entries with
  `state=completed` and `result=succeeded` (or `result=skipped` for authorized
  SKIPPED passes).
- `pipeline_pass_outputs` has one row per pass that resolved; all
  `execution_status=COMPLETE` (some `yield_status=EMPTY` for off-domain passes is
  normal and expected).
- `stage_runs` has at least one row per pass with `execution_status=COMPLETE`.
- The summary `stage_runs` row (where `pass_name IS NULL` and
  `stage_name = 'derive_ontology_graph'`) has `execution_status=COMPLETE`.
