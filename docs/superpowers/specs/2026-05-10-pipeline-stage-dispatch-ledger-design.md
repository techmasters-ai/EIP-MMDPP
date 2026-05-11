# Pipeline Stage Dispatch Ledger — Design

**Status:** Approved (revised after sixth review pass)
**Date:** 2026-05-10
**Author:** Josh (with Claude)
**Scope:** v1 covers sequential stages 1–9 of the ingest pipeline. Per-pass fan-in inside stage 9 and the post-merge tail (stages 10–12) are out of scope and remain on their existing chain mechanisms.

## Problem

Two documents in the current SA-2_Sources ingest (`Radar Basics.pdf`, `radar2_waveform1.pdf`) completed up through `derive_image_embeddings` (stage 7) but never started `derive_document_anchors` (stage 8). The `pipeline_run` rows are stuck in `PROCESSING` with no Celery task queued, no in-flight worker, and no path to recovery short of a manual `/v1/documents/{id}/reingest`.

Root cause: the sequential ingest pipeline relies on Celery's in-memory `chain(...)` (`app/workers/pipeline.py:2369`). When a worker accepts task N and dies before publishing task N+1 — or when Celery silently drops the chord/chain callback under certain Redis edge cases — the pipeline orphans. There is no durable record that "task N+1 was supposed to run next," so nothing can republish it.

The existing `periodic_stale_run_sweep` (`app/workers/pipeline.py:1814`) detects stale `RUNNING` stage_runs, marks them `FAILED`, bumps `documents.retry_count`, and re-dispatches via `start_ingest_pipeline` (lines 1918–1922). That recovers some failures but not the orphan-after-success case: when stage N successfully writes `COMPLETE` but stage N+1 was never published, the sweeper sees nothing to sweep.

## Goal

Replace the in-memory chain with a database-backed work queue (the "dispatch ledger") so that **a worker death between stages cannot strand a pipeline_run.** Every handoff is committed to Postgres before any Celery publish is assumed durable, and the terminal `COMPLETE` write is committed in the same transaction as the successor's `PENDING` insert.

This is the only goal. Throughput, observability improvements, and per-pass fan-in are explicit non-goals for v1.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Data model | Extend existing `ingest.stage_runs` summary rows | Avoids a second source of truth. Ledger summary rows always carry `attempt=1`; a new `dispatch_attempt` column tracks retries. With `attempt=1` constant, the existing partial unique index `(pipeline_run_id, stage_name, attempt) WHERE pass_name IS NULL` effectively enforces uniqueness on `(pipeline_run_id, stage_name)` for ledger rows |
| Dispatch path | Replace the Celery chain entirely | Single dispatch path; one mental model |
| Cadence | Beat-only, every 5s | Adds ~2.5s avg per handoff (~20s mean over 8 stages); no inline-trigger code path |
| Idempotency | Skip if summary stage_run is COMPLETE; else re-run | Stages already have purge/overwrite semantics; rely on them rather than partial-state tracking |
| Deploy posture | Drain — old chain finishes for in-flight runs; new code applies to new runs | Smallest blast radius |
| Scope (v1) | Sequential stages 1→9 only | Stage 9 fan-in keeps `dispatched_phases` JSONB; post-merge tail keeps inline `celery_chain` |
| Wrapper integration | Extend existing `guard_stage_run` decorator | Existing pattern (decorator, `wrapper.stage_name` marker for tests, `CeleryRetry` passthrough) already handles 80% of the contract |
| Terminal-write ownership | Wrapper-only; body's `_update_stage_run("COMPLETE")` is intercepted | Closes the death window between body's COMPLETE commit and wrapper's successor insert |

## Architecture

State machine on `ingest.stage_runs` (summary rows, `pass_name IS NULL`):

```
                    [ dispatcher every 5s ]
                              │
                              ▼
   PENDING ──── claim ───▶ DISPATCHED ── worker picks up ──▶ RUNNING
      ▲                       │                                │
      │                       │ stale > threshold              │ success
      │                       └──────────────────────┐         ▼
      │                                              │      COMPLETE ─── single tx:
      │                                              │                   inserts next-stage PENDING
      │                                              │                   AND flips self → COMPLETE
      │                                              │                       │
      │  stale RUNNING > threshold (dispatch_attempt+=1) │                   │
      ├──────────────────────────────────────────────┘                       │
      │                                                                      │
      └─── retryable failure (Tx-4: status=PENDING, available_at advances) ──┘
                              ▲
                              │
                              └─── terminal failure (max attempts) ─▶ FAILED ─▶ pipeline_run = FAILED
```

Boundary of v1:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  NEW DISPATCHER OWNS                                                            │
│   prepare_document ─▶ detect_and_translate ─▶ derive_document_metadata          │
│      ─▶ purge_document_derivations ─▶ derive_picture_descriptions               │
│      ─▶ derive_text_embeddings ─▶ derive_image_embeddings                       │
│      ─▶ derive_document_anchors ─▶ derive_ontology_graph (entry only)           │
└──────────────────────────────────────── (handoff) ──────────────────────────────┘
                                          │
                                          ▼
              ┌──────────────────────────────────────────────────┐
              │ UNCHANGED — per-pass fan-in (dispatched_phases   │
              │ JSONB on pipeline_runs, derive_ontology_graph_   │
              │ pass × N, derive_ontology_graph_merge)           │
              └──────────────────────────────────────────────────┘
                                          │
                                          ▼
              ┌──────────────────────────────────────────────────┐
              │ UNCHANGED — celery_chain inside merge:           │
              │   collect_derivations ─▶ derive_structure_links  │
              │     ─▶ derive_canonicalization                   │
              └──────────────────────────────────────────────────┘
```

> **Note on stage names.** The persisted `stage_runs.stage_name` for the text-embedding stage is `derive_text_embeddings` (the `@guard_stage_run("derive_text_embeddings")` decoration on `pipeline.py:4829`), even though the function is `derive_text_chunks_and_embeddings`. The ledger keys on the persisted name throughout this design.

### Components added/changed

- `app/models/ingest.py` — new columns on `StageRun`; `DISPATCHED` added to status enum
- `alembic/versions/0020_stage_dispatch_columns.py` *(new)* — migration: 5 columns + 1 partial index, additive
- `app/workers/_stage_lifecycle.py` *(new)* — `_LifecycleCtx` dataclass and `_CTX` ContextVar
- `app/workers/dispatcher.py` *(new)* — `dispatch_pending_pipeline_stages` Celery task + helpers
- `app/workers/pipeline.py` —
  - extend `guard_stage_run` with `lifecycle`/`next_stage`/`next_task`/`intercept_terminal` kwargs
  - extend `_update_stage_run` with the lifecycle-context interception block
  - add `STAGE_SUCCESSORS` table + `_resolve_queue` helper
  - add `_seed_first_stage` helper
  - rewrite `start_ingest_pipeline` and `reingest_graph_only` to seed-and-go
  - extend `_sweep_stale_runs`: stale-RUNNING resets to PENDING in same run; stale-DISPATCHED resets to PENDING; remove the `start_ingest_pipeline` re-dispatch branch for ledger-managed stages
- `app/workers/celery_app.py` — beat schedule entry for the dispatcher (5s); add `app.workers.dispatcher` to `include`
- `app/config.py` — new setting `max_stage_dispatches` (default 5); existing `stale_stage_run_threshold_seconds` (default 34200 = ~9.5h) and `stale_dispatched_threshold_seconds` (new, default 600 = 10min) are referenced by the new sweeper logic

## Schema changes

Single migration, additive only.

```python
# alembic/versions/0020_stage_dispatch_columns.py

def upgrade():
    op.add_column("stage_runs",
        sa.Column("queue_name", sa.String(64), nullable=True),
        schema="ingest")
    op.add_column("stage_runs",
        sa.Column("task_name", sa.String(255), nullable=True),
        schema="ingest")
    op.add_column("stage_runs",
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        schema="ingest")
    op.add_column("stage_runs",
        sa.Column("available_at",
                  sa.DateTime(timezone=True),
                  nullable=True,
                  server_default=sa.text("NOW()")),
        schema="ingest")
    op.add_column("stage_runs",
        sa.Column("dispatched_at",
                  sa.DateTime(timezone=True), nullable=True),
        schema="ingest")
    # Ledger-only retry counter. Distinct from `attempt`, which legacy code
    # mutates per Celery retry. Ledger summary rows always have attempt=1;
    # dispatch_attempt is the field Tx-4 and the sweeper increment.
    op.add_column("stage_runs",
        sa.Column("dispatch_attempt",
                  sa.Integer(),
                  nullable=False,
                  server_default=sa.text("1")),
        schema="ingest")

    # Dispatcher's hot path: scan for actionable PENDING summary rows.
    # task_name IS NOT NULL excludes incidental PENDING rows that legacy code
    # may have created without dispatch metadata.
    op.create_index(
        "ix_stage_runs_dispatcher_claim",
        "stage_runs",
        ["available_at"],
        unique=False,
        postgresql_where=sa.text(
            "status = 'PENDING' AND pass_name IS NULL AND task_name IS NOT NULL"
        ),
        schema="ingest",
    )
```

**No backfill** in the migration. Per the drain-the-chain rollout, existing PROCESSING runs keep running on the old chain code path; new runs go through the dispatcher.

### Status enum

The existing `status` column (`String(50)`, no DB constraint) gains one valid value: `DISPATCHED`. Code-level constant change in `app/models/ingest.py`:

```python
class StageRunStatus(str, Enum):
    PENDING    = "PENDING"      # ledger row exists, awaiting dispatcher
    DISPATCHED = "DISPATCHED"   # NEW — published to Celery, not yet running
    RUNNING    = "RUNNING"
    COMPLETE   = "COMPLETE"
    FAILED     = "FAILED"
```

### Existing partial unique indexes (unaffected)

- `uq_stage_runs_run_pass_attempt` (`pass_name IS NOT NULL`) — per-pass rows; v1 never inserts these
- `uq_stage_runs_summary_row` (`pass_name IS NULL`) — index on `(pipeline_run_id, stage_name, attempt)`. **Ledger semantics:** all ledger inserts use `attempt=1` and never mutate `attempt`. Retries increment `dispatch_attempt` instead. With `attempt=1` invariant for ledger rows, the existing 3-column partial index effectively enforces uniqueness on `(pipeline_run_id, stage_name)` for ledger summary rows, without breaking legacy code paths that still write `attempt=N` and create one summary row per Celery attempt. ON CONFLICT uses index inference, not constraint name (the index is unnamed-constraint).

### Column purpose

| Column | Purpose |
|---|---|
| `queue_name` | Resolved from `celery_app.conf.task_routes` at insert time. Informational; the dispatcher relies on `task_routes` at publish time, not on this column |
| `task_name` | Fully-qualified task name (`app.workers.pipeline.derive_document_anchors`). Acts as the "is-this-a-ledger-row?" marker — non-ledger code paths never set it |
| `celery_task_id` | Returned by `apply_async()`; used by stale-DISPATCHED detection and observability |
| `available_at` | Earliest dispatcher should pick this row. `NOW()` for normal handoffs, `NOW() + backoff` for retries |
| `dispatched_at` | Set when dispatcher transitions PENDING→DISPATCHED. Reset to NULL on retry |
| `dispatch_attempt` | Ledger retry counter. Starts at 1, incremented by Tx-4 and the stale-RUNNING sweeper. Distinct from `attempt`, which legacy code mutates per Celery retry. Capped at `settings.max_stage_dispatches`; exceeding the cap terminalizes the stage and the pipeline_run |

## Successor table and queue resolution

Single source of truth in `app/workers/pipeline.py`. Keys are persisted stage names; values pair the next persisted name with the next task path.

```python
@dataclass(frozen=True)
class StageEdge:
    next_stage: str | None    # persisted stage_name (matches @guard_stage_run)
    next_task:  str | None    # fully-qualified Celery task path

STAGE_SUCCESSORS: dict[str, StageEdge] = {
    "prepare_document":            StageEdge("detect_and_translate",        "app.workers.pipeline.detect_and_translate"),
    "detect_and_translate":        StageEdge("derive_document_metadata",    "app.workers.pipeline.derive_document_metadata"),
    "derive_document_metadata":    StageEdge("purge_document_derivations",  "app.workers.pipeline.purge_document_derivations"),
    "purge_document_derivations":  StageEdge("derive_picture_descriptions", "app.workers.pipeline.derive_picture_descriptions"),
    "derive_picture_descriptions": StageEdge("derive_text_embeddings",      "app.workers.pipeline.derive_text_chunks_and_embeddings"),
    "derive_text_embeddings":      StageEdge("derive_image_embeddings",     "app.workers.pipeline.derive_image_embeddings"),
    "derive_image_embeddings":     StageEdge("derive_document_anchors",     "app.workers.pipeline.derive_document_anchors"),
    "derive_document_anchors":     StageEdge("derive_ontology_graph",       "app.workers.pipeline.derive_ontology_graph"),
    "derive_ontology_graph":       StageEdge(None, None),
}

# Subset distinctions used by the stale-RUNNING sweeper:
LEDGER_SEQUENTIAL_STAGES = [s for s in STAGE_SUCCESSORS if s != "derive_ontology_graph"]
LEDGER_FANOUT_STAGES     = ["derive_ontology_graph"]
```

`derive_ontology_graph` is in `STAGE_SUCCESSORS` (the dispatcher and lifecycle wrapper both target it) but is excluded from `LEDGER_SEQUENTIAL_STAGES`. The stale-RUNNING sweeper's new ledger logic targets only `LEDGER_SEQUENTIAL_STAGES`. Stage 9's RUNNING state — which legitimately persists for 30+ minutes while per-pass fan-in is active — is owned by the existing `reconcile_ontology_graph_runs` reconciler (`pipeline.py:6668`), unchanged from today.

Queue resolution comes from a single helper that consults both `celery_app.conf.task_routes` and, as fallback, the task's `queue=` decorator argument. Not every ledger stage is in `task_routes` — `detect_and_translate`, `derive_document_metadata`, and `derive_picture_descriptions` route to the default `celery` queue without explicit `task_routes` entries, and `derive_text_chunks_and_embeddings` / `derive_image_embeddings` / `derive_document_anchors` / `derive_ontology_graph` use both `task_routes` *and* `queue=` decorators. The helper unifies the two sources:

```python
def _resolve_queue(task_name: str) -> str:
    """Return the queue Celery will actually route this task to."""
    # 1. Explicit task_routes entry wins (matches Celery's own precedence).
    routes = celery_app.conf.task_routes or {}
    entry = routes.get(task_name)
    if entry and entry.get("queue"):
        return entry["queue"]
    # 2. Decorator-level queue= argument.
    task = celery_app.tasks.get(task_name)
    if task is not None:
        decorator_queue = getattr(task, "queue", None)
        if decorator_queue:
            return decorator_queue
    # 3. Celery's broker default (`celery_app.conf.task_default_queue`,
    #    which is "celery" unless overridden).
    return celery_app.conf.task_default_queue or "celery"
```

The dispatcher does NOT pass `queue=` to `apply_async`; routing precedence inside Celery already follows the same order this helper does, so observability written via `_resolve_queue` matches the runtime destination. The `queue_name` column on the ledger row is set at insert time purely for observability.

## Stage-task contract

The lifecycle is folded into the existing `guard_stage_run` decorator. Stages 1–9 add `lifecycle=True` and (for stages 1–8) successor metadata.

```python
# app/workers/pipeline.py — stage example

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60, queue="embed")
@guard_stage_run(
    "derive_image_embeddings",
    lifecycle=True,
    next_stage="derive_document_anchors",
    next_task="app.workers.pipeline.derive_document_anchors",
)
def derive_image_embeddings(self, document_id: str, run_id: str | None = None) -> dict:
    return _run_image_embeddings(document_id, run_id)   # body unchanged
```

Stage 9:

```python
@guard_stage_run(
    "derive_ontology_graph",
    lifecycle=True,
    next_stage=None,
    next_task=None,
    intercept_terminal=False,   # merge owns COMPLETE
)
```

### Lifecycle context

```python
# app/workers/_stage_lifecycle.py  (new)

@dataclass
class _LifecycleCtx:
    pipeline_run_id: str        # always stored as str
    stage_name: str
    dispatch_attempt: int       # ledger retry counter (NOT Celery retry counter)
    intercept_terminal: bool    # False for stage 9; True otherwise
    next_stage: str | None
    next_task:  str | None
    pending_status:  str | None = None   # "COMPLETE" | "FAILED" | None
    pending_metrics: dict | None = None
    pending_error:   str | None = None

_CTX: ContextVar[_LifecycleCtx | None] = ContextVar(
    "stage_lifecycle_ctx", default=None,
)
```

**Critical ordering:** `_CTX` is set **only after** CLAIM successfully transitioned a ledger row to RUNNING (1 row updated). The 5 zero-row CLAIM outcomes return early without ever touching `_CTX`, and the legacy-run path (no ledger row at all) runs the body inline without `_CTX`. This guarantees `_CTX` is non-`None` only when there is a live ledger row in RUNNING state that this wrapper invocation owns.

`_CTX.reset(token)` is called in a `finally` block. Celery's prefork worker reuses the same process across many task invocations and does **not** reset Python ContextVars between tasks; without the explicit reset, the ContextVar would leak into subsequent task runs in the same worker process and cause `_update_stage_run` interception to fire on the wrong stage/run.

```python
def wrapper(self, document_id, run_id=None, *args, **kwargs):
    # Non-lifecycle invocation — pass through to legacy body unchanged.
    if not (lifecycle and run_id):
        return fn(self, document_id, run_id, *args, **kwargs)

    # Tx-1 CLAIM + follow-up SELECT to disambiguate 5 zero-row outcomes.
    claim = _claim_tx1(
        run_id=run_id,
        stage_name=stage_name,
        celery_task_id=self.request.id,
        is_celery_retry=(self.request.retries > 0),
    )

    if claim.outcome == "legacy":
        # No ledger row at all (pre-deploy chain run). Body owns its own writes.
        return fn(self, document_id, run_id, *args, **kwargs)

    if claim.outcome != "proceed":
        # already_complete | concurrent_running | stale_pending | terminal_failed
        return claim.early_result

    # CLAIM succeeded; row is now RUNNING. Set _CTX only at this point.
    ctx = _LifecycleCtx(
        pipeline_run_id=str(run_id),         # normalize at construction
        stage_name=stage_name,
        dispatch_attempt=claim.dispatch_attempt,
        intercept_terminal=intercept_terminal,
        next_stage=next_stage,
        next_task=next_task,
    )
    token = _CTX.set(ctx)
    try:
        result = fn(self, document_id, run_id, *args, **kwargs)
        if intercept_terminal:
            _finalize_after_body(ctx, result)
        return result
    except CeleryRetry:
        raise                                   # row stays RUNNING; Celery republishes
    except Exception as exc:
        if intercept_terminal:
            _tx4_finalize_failure(
                ctx,
                error=str(exc),
                celery_retries=self.request.retries,
                max_retries=self.max_retries,
            )
        raise
    finally:
        _CTX.reset(token)

# guard_stage_run decorator additionally sets:
wrapper.stage_name = stage_name      # existing marker (pre-design)
wrapper._lifecycle = lifecycle       # NEW — read by module-load assertion
```

### Tx-1: CLAIM (stage entry)

```sql
-- Bind :is_celery_retry = (self.request.retries > 0)
UPDATE ingest.stage_runs
SET status         = 'RUNNING',
    started_at     = COALESCE(started_at, NOW()),
    celery_task_id = :celery_task_id    -- always overwrite with current attempt's id
WHERE pipeline_run_id = :run_id
  AND stage_name      = :stage_name
  AND pass_name       IS NULL
  AND (
        status IN ('DISPATCHED', 'PENDING')
     OR (status = 'RUNNING' AND :is_celery_retry)   -- Celery-retry re-entry branch
  )
RETURNING id, attempt, dispatch_attempt;
```

Six outcomes, packaged into the `_claim_tx1` return value (`outcome` ∈ {`proceed`, `legacy`, `already_complete`, `concurrent_running`, `stale_pending`, `terminal_failed`}). When the UPDATE returns 0 rows, the wrapper issues a single follow-up SELECT to disambiguate:

```python
if rowcount == 0:
    row = db.execute(text("""
        SELECT status FROM ingest.stage_runs
        WHERE pipeline_run_id = :run_id
          AND stage_name      = :stage_name
          AND pass_name       IS NULL
    """), {"run_id": run_id, "stage_name": stage_name}).first()
    if row is None:
        outcome = "no_row"          # legacy run
    else:
        outcome = row.status        # "COMPLETE" | "RUNNING" | "FAILED" | "PENDING"
```

| UPDATE rowcount | Follow-up SELECT | `outcome` | Wrapper action | `claim.early_result` |
|---|---|---|---|---|
| 1 | (not run) | `proceed` | Set `_CTX`, run body | n/a (body runs) |
| 0 | row.status = `COMPLETE` | `already_complete` | Return early; `_CTX` never set | `{"stage": stage_name, "status": "skipped", "reason": "already_complete"}` |
| 0 | row.status = `RUNNING` (and `is_celery_retry=False`) | `concurrent_running` | Return early without touching anything; `_CTX` never set | `None` |
| 0 | row.status = `PENDING` | `stale_pending` | Return; dispatcher will republish on next tick; `_CTX` never set | `None` |
| 0 | row.status = `FAILED` | `terminal_failed` | Return; do not run body; `_CTX` never set | `{"stage": stage_name, "status": "terminal_failed", "reason": "stage_previously_failed"}` |
| 0 | no row | `legacy` | Run body inline; existing inline `_update_stage_run` writes commit through to DB as today; `_CTX` never set | n/a (body runs) |

`None` is preferred for `concurrent_running` and `stale_pending` so that observability won't conflate "wrapper skipped because of race" with "stage genuinely chose to skip" (which would produce a dict via the legitimate `{"status":"skipped",...}` body-return path that the wrapper then treats as Tx-3 success). `terminal_failed` uses the distinct status value `"terminal_failed"` rather than `"skipped"` to keep log-grep distinct: `"skipped"` is exclusively the success-advance signal (either legitimate body-return or `already_complete` race recovery), while `"terminal_failed"` marks a no-op return for a stage of an already-FAILED pipeline_run.

**Note on `concurrent_running` and broker re-delivery.** With `task_acks_late=True` + `task_reject_on_worker_lost=True` (set in `app/workers/celery_app.py:65-66`), the broker can rarely re-deliver a task message after the original worker started but before it ack'd. In that rare case the second worker sees status=RUNNING and `is_celery_retry=False` (retries=0 for the redelivery), and falls into the `concurrent_running` branch — the early-return without touching anything is the safe outcome the design needs in both the redelivery case and the genuine-concurrent-execution case.

Because `_CTX` is set *only* on `proceed`, the contradiction the previous draft had — wrapper sets `_CTX` before CLAIM, then the table claimed certain outcomes wouldn't set `_CTX` — is gone.

The `pass_name IS NULL` clause keeps the wrapper from ever touching per-pass rows under stage 9.

**Critical rule:** `_CTX` is only ever set when a summary ledger row exists for this run/stage. In the legacy-run branch (no row), `_CTX` stays `None` for the lifetime of the body, so existing pre-deploy chain tasks have their `COMPLETE` writes commit through to the DB exactly as today. This is what allows the "drain the chain" rollout to be safe.

### Tx-3: ENQUEUE_NEXT (stage success)

After the body returns successfully, the wrapper runs **one SQLAlchemy transaction** wrapping two statements. The transaction boundary is explicit and the bind dicts make the `_CTX` plumbing load-bearing:

```python
def _tx3_complete_and_enqueue_next(ctx: _LifecycleCtx) -> None:
    db = _get_db()
    try:
        with db.begin():       # one transaction; one commit on context exit
            db.execute(text(<3a INSERT>), {
                "run_id":     ctx.pipeline_run_id,
                "next_stage": ctx.next_stage,
                "next_queue": _resolve_queue(ctx.next_task),
                "next_task":  ctx.next_task,
            })
            db.execute(text(<3b UPDATE>), {
                "run_id":     ctx.pipeline_run_id,
                "stage_name": ctx.stage_name,
                "metrics":    ctx.pending_metrics,   # populated by interception
            })
    finally:
        db.close()
```

The metrics passed to Tx-3b are precisely what the stage body wrote via `_update_stage_run("COMPLETE", metrics=...)` — the interception stashed them in `ctx.pending_metrics` rather than committing them, and Tx-3b is now the durable write. The `_resolve_queue` lookup uses `ctx.next_task` (a fully-qualified task path) and returns the actual runtime queue for that task (see "Successor table and queue resolution").

The SQL for the two statements:

```sql
-- 3a: insert successor PENDING row (idempotent on the partial unique index)
-- attempt is hardcoded to 1; ledger retries increment dispatch_attempt, not attempt.
INSERT INTO ingest.stage_runs
    (id, pipeline_run_id, stage_name, attempt, status,
     queue_name, task_name, available_at, dispatch_attempt)
VALUES (gen_random_uuid(), :run_id, :next_stage, 1, 'PENDING',
        :next_queue, :next_task, NOW(), 1)
ON CONFLICT (pipeline_run_id, stage_name, attempt)
WHERE pass_name IS NULL
DO NOTHING;

-- 3b: flip self to COMPLETE with metrics that the body stashed in _CTX
UPDATE ingest.stage_runs
SET status      = 'COMPLETE',
    finished_at = NOW(),
    metrics     = :metrics
WHERE pipeline_run_id = :run_id
  AND stage_name      = :stage_name
  AND pass_name       IS NULL;
```

Both statements run inside the `with db.begin():` block and commit together. Any exception inside the block rolls both back atomically — the database guarantee the design depends on.

Conflict target uses index inference (`(pipeline_run_id, stage_name, attempt) WHERE pass_name IS NULL`), matching the actual partial unique index from migration 0016. There is no `ON CONFLICT ON CONSTRAINT <name>` because the underlying object is an index, not a named constraint.

Because every ledger insert uses `attempt=1` and Tx-4 increments `dispatch_attempt` (never `attempt`), there is exactly one ledger summary row per `(pipeline_run_id, stage_name)`. Re-publication after Tx-4 finds the existing row and updates it; it cannot create a duplicate.

The single commit is the durability guarantee: either the next stage is queued **and** this one is marked done, or neither happens.

### Tx-4: failure handling (called from wrapper's exception path or success-with-failure-dict path)

Retryable (`dispatch_attempt + 1 <= settings.max_stage_dispatches`):

```sql
UPDATE ingest.stage_runs
SET status           = 'PENDING',
    dispatch_attempt = dispatch_attempt + 1,
    available_at     = NOW() + (:backoff_seconds || ' seconds')::interval,
    started_at       = NULL,
    dispatched_at    = NULL,
    error_message    = :error_message
WHERE pipeline_run_id = :run_id AND stage_name = :stage_name AND pass_name IS NULL;
```

Terminal (`dispatch_attempt + 1 > settings.max_stage_dispatches`):

```sql
UPDATE ingest.stage_runs
SET status           = 'FAILED',
    dispatch_attempt = dispatch_attempt + 1,
    finished_at      = NOW(),
    error_message    = :err
WHERE pipeline_run_id = :run_id AND stage_name = :stage_name AND pass_name IS NULL;

UPDATE ingest.pipeline_runs
SET status        = 'FAILED',
    finished_at   = NOW(),
    error_message = :err
WHERE id = :run_id AND status = 'PROCESSING';
```

Tx-4 never touches `attempt` — that stays 1 for the ledger row's lifetime. `error_message` (existing column) carries the last error across retries. No new column added beyond `dispatch_attempt`.

### Body return-value contract — `_finalize_after_body`

The wrapper's success path calls `_finalize_after_body(ctx, result)`. This is the function whose call appears in the wrapper skeleton above:

```python
def _finalize_after_body(ctx: _LifecycleCtx, result) -> None:
    """Decide between Tx-3 (success, enqueue next stage) and Tx-4 (failure)
    based on the body's return value and any `_update_stage_run` interception
    that occurred during the body.
    """
    if not ctx.intercept_terminal:        # stage 9: merge owns finalization
        return

    if ctx.pending_status == "FAILED" or (
        isinstance(result, dict) and result.get("status") in ("FAILED", "failed")
    ):
        _tx4_finalize_failure(ctx, error=ctx.pending_error or "stage returned failure status")
        return

    # Normal success — including bodies that return {"status":"skipped",...}.
    # A skipped result is a legitimate no-op completion (e.g., translation disabled,
    # no pictures to describe). Pipeline must advance. Skip metadata is preserved
    # in the row's `metrics` column via the wrapper's interception of the body's
    # `_update_stage_run("COMPLETE", metrics={"skipped": True, "reason": "..."})`
    # call.
    _tx3_complete_and_enqueue_next(ctx)
```

The only way to halt the pipeline from inside a stage body is to raise an exception (Tx-4) or to return a dict with `status in {"FAILED","failed"}` (also Tx-4). A `skipped` return is treated as success — three existing stages (`detect_and_translate`, `derive_document_metadata`, `derive_picture_descriptions`) routinely return `{"status":"skipped",...}` on legitimate no-op completions and the pipeline must advance for them.

This catches the swallow-and-return-failure pattern that exists in some stages without forcing a body refactor.

### `_update_stage_run` interception

The single function modification that closes the premature-`COMPLETE` window. ID normalization is critical because callers pass `pipeline_run_id` as both `str` and `UUID`.

```python
def _update_stage_run(db, pipeline_run_id, stage_name, status,
                     attempt=1, metrics=None, error=None):
    ctx = _CTX.get()
    if (
        ctx is not None
        and str(ctx.pipeline_run_id) == str(pipeline_run_id)   # normalize both sides
        and ctx.stage_name == stage_name
        and ctx.intercept_terminal
    ):
        if status == "RUNNING":
            return                          # wrapper already wrote RUNNING via CLAIM
        if status in ("COMPLETE", "FAILED"):
            ctx.pending_status  = status
            ctx.pending_metrics = metrics
            ctx.pending_error   = error
            return                          # do NOT commit; wrapper finalizes in Tx-3 / Tx-4
    # legacy / non-intercepted path: existing implementation runs unchanged
    <existing implementation>
```

Three rules implicit in the predicate:

1. **ID normalization.** `str(ctx.pipeline_run_id) == str(pipeline_run_id)` covers callers that pass `UUID` objects (e.g., from a `Document` model's `.id` field) and callers that pass strings (Celery task args). Without this, a body that internally converts to UUID before calling `_update_stage_run` would bypass interception and re-open the death window.
2. **Legacy mode never sets `_CTX`.** The CLAIM step above only enters the `_CTX.set()` branch when a summary ledger row exists. For pre-deploy chain runs (no row), `_CTX` is `None` for the body's entire lifetime, so the branch above is a no-op and the existing `_update_stage_run` implementation commits the COMPLETE write to the DB exactly as today.
3. **Stage 9 never intercepts.** `ctx.intercept_terminal=False` for the `derive_ontology_graph` entry task, so its existing `_update_summary_stage_run("RUNNING")` write commits, and `derive_ontology_graph_merge` keeps owning the eventual `_update_summary_stage_run("COMPLETE")` write.

### Death window the design closes

```
Old (pre-design) failure mode:
  T0  body: _update_stage_run("COMPLETE")           ── COMMITS row=COMPLETE
  T1  body returns
  T2  Celery serializer publishes next chain link  ── could fail silently
  T3  worker dies
  Result: row=COMPLETE, no next task queued, sweeper has nothing to sweep.

New (design) failure mode:
  T0  body: _update_stage_run("COMPLETE")           ── INTERCEPTED, stashed in _CTX
  T1  body returns
  T2  wrapper begins Tx-3
  T3  worker dies before Tx-3 commit
  Result: row stays RUNNING. Stale-RUNNING sweeper resets to PENDING with
          attempt+=1. Dispatcher republishes within 5s of next beat tick.
```

### Wrapper guarantees

1. **No orphans on success path.** Successor PENDING is committed in the same tx as `self → COMPLETE`.
2. **No orphans on worker death.** Body crashes between RUNNING and COMPLETE → row stays RUNNING → sweeper resets to PENDING → dispatcher republishes.
3. **No double-execution.** CLAIM is a conditional UPDATE; only one worker can flip DISPATCHED→RUNNING. Celery retry re-entry is single-threaded by Celery's scheduler under normal conditions. Under pathological conditions (`task_acks_late=True` + `task_reject_on_worker_lost=True`, set in `app/workers/celery_app.py:65-66`, can cause broker-redelivery after worker crash mid-`self.retry()`), the design *tolerates* the resulting concurrent re-entry rather than preventing it at the CLAIM level: both re-entrant workers would see RUNNING + `is_celery_retry=True`, both would pass CLAIM. This is the same race that exists in the pre-design code today; the pre-design implementation has not produced visible duplicate-execution incidents, and v1 inherits the same posture. A follow-up could tighten CLAIM with `AND celery_task_id = :prior_task_id` once the wrapper has access to the prior id, but is out of scope for v1.
4. **No premature COMPLETE.** A lifecycle-wrapped stage cannot leave the ledger in a state where `status='COMPLETE'` but the successor row does not exist. The body's terminal-status writes are intercepted; the only commit that flips the row to COMPLETE also inserts the successor.
5. **No double-retry under Celery `self.retry()`.** When a stage raises `CeleryRetry`, the wrapper passes through unchanged; the row stays RUNNING; Celery's scheduler republishes; the next attempt's CLAIM re-enters via the `is_celery_retry` branch. Tx-4 only fires for non-`CeleryRetry` exceptions or for return-dict-as-failure.

## Dispatcher loop

Beat-scheduled Celery task, runs on the default `celery` queue.

### Beat entry

```python
# app/workers/celery_app.py — beat_schedule additions
"dispatch-pending-pipeline-stages": {
    "task": "app.workers.dispatcher.dispatch_pending_pipeline_stages",
    "schedule": 5.0,
    "options": {"queue": "celery"},
},
```

Also: `app.workers.dispatcher` must be added to the `include=` list passed to `Celery()` (or imported by the existing `__init__.py` import set) so the beat task registers at worker boot.

### Task body

```python
# app/workers/dispatcher.py  (new file)

from app.services.redis_utils import get_redis   # existing helper

DISPATCH_BATCH_LIMIT = 50
DISPATCH_LOCK_KEY    = "dispatcher:pipeline_stages"
DISPATCH_LOCK_TTL    = 30  # seconds

@celery_app.task(bind=True, name="app.workers.dispatcher.dispatch_pending_pipeline_stages")
def dispatch_pending_pipeline_stages(self) -> dict:
    redis = get_redis()
    if not redis.set(DISPATCH_LOCK_KEY, self.request.id, nx=True, ex=DISPATCH_LOCK_TTL):
        return {"skipped": "another dispatcher tick is in flight"}
    try:
        return _run_dispatch_tick()
    finally:
        _release_lock_if_owner(redis, DISPATCH_LOCK_KEY, self.request.id)


def _release_lock_if_owner(redis, key: str, token: str) -> None:
    """Lua-CAS release: only the holder of the lock can release it."""
    redis.eval(
        "if redis.call('get', KEYS[1]) == ARGV[1] then "
        "return redis.call('del', KEYS[1]) else return 0 end",
        1, key, token,
    )
```

`get_redis` is the existing helper at `app/services/redis_utils.py`. No new Redis-connection plumbing is introduced.

### Claim query

Single SQL statement scans + atomically claims using `FOR UPDATE SKIP LOCKED`.

```python
def _run_dispatch_tick() -> dict:
    db = _get_db()
    try:
        rows = db.execute(text("""
            WITH claimable AS (
                SELECT sr.id
                FROM ingest.stage_runs sr
                JOIN ingest.pipeline_runs pr ON pr.id = sr.pipeline_run_id
                WHERE sr.status      = 'PENDING'
                  AND sr.pass_name   IS NULL
                  AND sr.task_name   IS NOT NULL   -- ledger rows only
                  AND sr.available_at <= NOW()
                  AND pr.status      = 'PROCESSING'
                ORDER BY sr.available_at ASC
                LIMIT :limit
                FOR UPDATE OF sr SKIP LOCKED
            )
            UPDATE ingest.stage_runs sr
            SET status        = 'DISPATCHED',
                dispatched_at = NOW()
            FROM claimable c
            WHERE sr.id = c.id
            RETURNING sr.id, sr.pipeline_run_id, sr.stage_name,
                      sr.task_name, sr.queue_name,
                      (SELECT pr.document_id
                       FROM ingest.pipeline_runs pr
                       WHERE pr.id = sr.pipeline_run_id);
        """), {"limit": DISPATCH_BATCH_LIMIT}).fetchall()
        db.commit()
    finally:
        db.close()

    published = 0
    for row in rows:
        try:
            _publish(row.task_name, row.document_id, row.pipeline_run_id, row.id)
            published += 1
        except Exception as exc:
            _undo_claim(row.id, error=str(exc))
            logger.exception("dispatcher: failed to publish stage_run=%s", row.id)

    return {"claimed": len(rows), "published": published}

# Race note: if apply_async() succeeds (broker has accepted the message) and a
# subsequent line inside _publish raises (e.g., the celery_task_id write fails
# on a transient DB blip), _undo_claim is invoked. Three subcases:
#   - Worker had not yet picked up the message → row at status='DISPATCHED'.
#     _undo_claim's WHERE status='DISPATCHED' matches → flips row to PENDING.
#     Broker delivers the original message to a worker → CLAIM sees PENDING and
#     is_celery_retry=False → proceeds → body runs once. A subsequent dispatcher
#     tick would see status=RUNNING (the worker's CLAIM moved it) and skip via
#     SKIP LOCKED + the WHERE-clause filter.
#   - Worker picked up the message between DISPATCHED and _undo_claim, ran CLAIM,
#     row is now status='RUNNING'. _undo_claim's WHERE status='DISPATCHED' finds
#     0 rows → no-op. Body runs once.
#   - Worker is mid-body (RUNNING and writing to other tables) when _undo_claim
#     fires. Same as above: _undo_claim is a no-op because status != 'DISPATCHED'.
# In every subcase exactly one body execution happens. PENDING ↔ RUNNING are
# mutually exclusive states, so no second dispatcher tick can re-publish while
# a worker is mid-body.


def _undo_claim(stage_run_id: uuid.UUID, *, error: str) -> None:
    """Return a stage_run from DISPATCHED back to PENDING after a publish failure.

    Does NOT bump dispatch_attempt — the stage body never ran; this isn't a real
    attempt. available_at is left at its prior value so the next tick retries
    immediately.
    """
    db = _get_db()
    try:
        db.execute(text("""
            UPDATE ingest.stage_runs
            SET status        = 'PENDING',
                dispatched_at = NULL,
                error_message = COALESCE(error_message, '') || ' publish_failed: ' || :err
            WHERE id = :id AND status = 'DISPATCHED'
        """), {"id": stage_run_id, "err": error})
        db.commit()
    finally:
        db.close()
```

### Publish helper

```python
def _publish(task_name, document_id, run_id, stage_run_id):
    task = celery_app.tasks[task_name]
    # No queue= override — task_routes is the single source of truth
    result = task.apply_async(
        args=[str(document_id), str(run_id)],
        headers={"stage_run_id": str(stage_run_id)},
    )
    db = _get_db()
    try:
        db.execute(text("""
            UPDATE ingest.stage_runs
            SET celery_task_id = :tid
            WHERE id = :id AND status = 'DISPATCHED'
        """), {"tid": result.id, "id": stage_run_id})
        db.commit()
    finally:
        db.close()
```

### Race protection summary

| Mechanism | What it protects against |
|---|---|
| `LIMIT 50` per tick | Slow tick cannot deplete the queue table or hold long row-locks |
| `FOR UPDATE SKIP LOCKED` | Two concurrent ticks read disjoint rows |
| `UPDATE ... RETURNING` in a single statement | Claim is committed before publish; no PENDING-but-being-claimed limbo |
| Redis advisory lock | Prevents wasteful concurrent ticks |
| `_undo_claim` on publish failure | Broker outages don't strand rows in DISPATCHED |

### Stale-row sweeping (extends `_sweep_stale_runs`)

The existing sweeper (`pipeline.py:1970` (entry point) / `pipeline.py:1814` (helper body)) is the only place in the codebase that's allowed to repair stale ledger state. v1 changes its behavior in four ways:

1. **Stale RUNNING for `LEDGER_SEQUENTIAL_STAGES` (stages 1–8): reset to PENDING or terminalize based on dispatch cap.** Stage 9 is *excluded* — it legitimately stays RUNNING for the entire per-pass fan-in window, and is the existing `reconcile_ontology_graph_runs` reconciler's responsibility. A single CTE handles retryable and terminal cases atomically:

   ```sql
   WITH stale AS (
       SELECT sr.id, sr.pipeline_run_id,
              sr.dispatch_attempt + 1 AS next_attempt
       FROM ingest.stage_runs sr
       JOIN ingest.pipeline_runs pr ON pr.id = sr.pipeline_run_id
       WHERE sr.status     = 'RUNNING'
         AND sr.pass_name  IS NULL
         AND sr.stage_name = ANY(:ledger_sequential_stages)
         AND sr.started_at < NOW() - make_interval(secs => :threshold)
         AND pr.status     = 'PROCESSING'
   ),
   retryable AS (
       UPDATE ingest.stage_runs sr
       SET status           = 'PENDING',
           dispatch_attempt = s.next_attempt,
           started_at       = NULL,
           dispatched_at    = NULL,
           available_at     = NOW(),
           error_message    = COALESCE(sr.error_message, '')
                              || ' stale; reset by sweeper'
       FROM stale s
       WHERE sr.id = s.id AND s.next_attempt <= :max_dispatches
       RETURNING sr.id
   ),
   terminal AS (
       UPDATE ingest.stage_runs sr
       SET status           = 'FAILED',
           finished_at      = NOW(),
           dispatch_attempt = s.next_attempt,
           error_message    = COALESCE(sr.error_message, '')
                              || ' stale; max dispatches reached'
       FROM stale s
       WHERE sr.id = s.id AND s.next_attempt > :max_dispatches
       RETURNING sr.pipeline_run_id
   )
   UPDATE ingest.pipeline_runs pr
   SET status        = 'FAILED',
       finished_at   = NOW(),
       error_message = COALESCE(pr.error_message, '')
                       || ' stage exceeded max dispatches'
   FROM terminal t
   WHERE pr.id = t.pipeline_run_id AND pr.status = 'PROCESSING';
   ```

   Parameters: `:ledger_sequential_stages = LEDGER_SEQUENTIAL_STAGES`, `:threshold = settings.stale_stage_run_threshold_seconds`, `:max_dispatches = settings.max_stage_dispatches`.

2. **Stale DISPATCHED for all ledger stages (including stage 9): reset to PENDING (no dispatch_attempt bump).** The stage didn't actually run, so this isn't a real attempt. Stage 9 entry rarely sits in DISPATCHED for 10+ minutes (the entry task is short), so this case is rare but valid.

   ```sql
   UPDATE ingest.stage_runs
   SET status        = 'PENDING',
       dispatched_at = NULL,
       error_message = COALESCE(error_message, '') || ' stale; reset by dispatcher sweeper'
   WHERE status        = 'DISPATCHED'
     AND pass_name     IS NULL
     AND task_name     IS NOT NULL
     AND dispatched_at < NOW() - make_interval(secs => :threshold);
   ```

   Threshold: `settings.stale_dispatched_threshold_seconds` (default 600).

3. **Stage 9 stale-RUNNING is owned by `reconcile_ontology_graph_runs`** (`pipeline.py:6668`), unchanged from today. That reconciler understands per-pass state and is the only code path that should mutate stage 9's summary row while it's RUNNING.

4. **Stale RUNNING for stages outside `STAGE_SUCCESSORS`** (per-pass rows, post-merge stages 10–12): existing FAILED+`start_ingest_pipeline` path is preserved.

### Stale threshold constraint

`stale_stage_run_threshold_seconds` must exceed two things simultaneously for every ledger-wrapped stage:

1. **The stage's `time_limit`** — the maximum wall-clock duration of a single attempt. A healthy long-running attempt cannot be preempted by the sweeper while it's still inside Celery's hard time limit. Current per-stage time limits (`app/config.py`):
   - `picture_desc_time_limit = 21600` (6 hours)
   - `translation_time_limit = 8100` (~2.25 hours)
   - `embed_time_limit = 4500` (75 minutes)

2. **The stage's full retry envelope** — `max_retries × default_retry_delay`, the time over which Celery's own retry machinery may legitimately keep the row in RUNNING/PENDING limbo.

Today's setting `stale_stage_run_threshold_seconds = 34200` (~9.5 hours) was calibrated against the largest `time_limit` (`picture_desc_time_limit + margin`). v1 **keeps this setting unchanged** and uses it for the new ledger sweeper too. Reusing the existing value avoids introducing a parallel knob to keep in sync.

The startup assertion in `celery_app.py` fails loud if for any task whose name appears in `STAGE_SUCCESSORS`:

```python
# task.run is the registered task's call target — the wrapped function whose
# `stage_name` and `_lifecycle` markers were set by guard_stage_run.
envelope = task.time_limit + task.max_retries * task.default_retry_delay
assert settings.stale_stage_run_threshold_seconds >= envelope, (
    f"stale_stage_run_threshold_seconds ({settings.stale_stage_run_threshold_seconds}) "
    f"must exceed envelope ({envelope}) for ledger stage {task.run.stage_name}"
)
```

Setting the threshold to 30 minutes (as an earlier draft of this spec proposed) would have preempted healthy long-running attempts of `derive_picture_descriptions`, `translation`, and `embed` mid-execution — a critical bug caught in third review.

### Observability

Three log lines per tick, structured:

```
DISPATCHER_TICK claimed=N published=M elapsed_ms=X
DISPATCHER_PUBLISH stage_run=… stage=… document=… queue=… celery_task=…
DISPATCHER_PUBLISH_FAIL stage_run=… error=…
```

### Out-of-scope for the dispatcher

- Does not touch per-pass rows (`pass_name IS NOT NULL`)
- Does not retry on stage failure — that's the lifecycle wrapper's Tx-4 job
- Does not understand stage order — order is enforced by Tx-3 only inserting the next row after current success
- Does not run on the `graph` queue

## Entry-point + reingest changes

### `start_ingest_pipeline`

Before (`app/workers/pipeline.py:2369`):

```python
pipeline = chain(
    prepare_document.si(document_id, run_id),
    detect_and_translate.si(document_id, run_id),
    ...
    derive_ontology_graph.si(document_id, run_id),
)
result = pipeline.apply_async()
return IngestDispatchResult(pipeline_run_id=run_id, celery_task_id=result.id)
```

After:

```python
run_id = _create_pipeline_run(db, document_id, mode="full", ...)
_seed_first_stage(
    db,
    pipeline_run_id=run_id,
    stage_name="prepare_document",
    task_name="app.workers.pipeline.prepare_document",
)
db.commit()
return IngestDispatchResult(pipeline_run_id=run_id, celery_task_id="")
```

`celery_task_id=""` matches existing behavior when `start_ingest_pipeline` returns the row of an already-active run, so no caller code breaks.

### `_seed_first_stage` helper

```python
def _seed_first_stage(db, *, pipeline_run_id, stage_name, task_name):
    queue = _resolve_queue(task_name)
    db.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             queue_name, task_name, available_at)
        VALUES (gen_random_uuid(), :run_id, :stage, 1, 'PENDING',
                :queue, :task, NOW())
        ON CONFLICT (pipeline_run_id, stage_name, attempt)
        WHERE pass_name IS NULL
        DO NOTHING
    """), {"run_id": pipeline_run_id, "stage": stage_name,
           "queue": queue, "task": task_name})
```

### `reingest_graph_only`

Before:

```python
result = celery_chain(
    derive_document_anchors.si(doc_id_str, run_id),
    derive_ontology_graph.si(doc_id_str, run_id),
).apply_async()
```

After:

```python
_seed_first_stage(
    db,
    pipeline_run_id=run_id,
    stage_name="derive_document_anchors",
    task_name="app.workers.pipeline.derive_document_anchors",
)
db.commit()
return {"pipeline_run_id": run_id, "celery_task_id": ""}
```

This is the path that resumes the two stalled docs (`Radar Basics.pdf`, `radar2_waveform1.pdf`). One row insert, one DB commit, dispatcher republishes within 5s. No chain to lose.

### Watcher

`app/workers/watcher.py:178-191` calls `start_ingest_pipeline(document_id)` and stores both `pipeline_run_id` and `celery_task_id` on the `Document` row. Under the new design, `celery_task_id` will always be the empty string `""` for newly-watcher-ingested documents (because the new entry point returns `celery_task_id=""`). The `Document.celery_task_id` column is `Optional[str]` (`app/models/ingest.py:78`), so empty-string writes are accepted. This matches the existing fallback behavior when `start_ingest_pipeline` returns the row of an already-active run (which also produces `celery_task_id=""`). Watcher code itself needs no changes.

**Downstream consumers of `Document.celery_task_id`:** any code that filters or joins on a non-empty value should treat `""` and `NULL` equivalently. The implementation task includes auditing this column's consumers before merging.

### Modes that don't change in v1

- `mode="full"` — covered above
- `mode="embeddings_only"` — still uses old chain; **out of scope**, listed in follow-ups
- `mode="graph_only"` — covered above

### Drain semantics

All `pipeline_runs` in `PROCESSING` at deploy time finish under the **old chain** code path. The CLAIM step's "no row at all" branch detects this and runs the body without setting `_CTX`, so existing inline `_update_stage_run` writes commit through to the DB exactly as today.

Once all pre-deploy PROCESSING runs drain (or are stale-swept by the existing 30-min RUNNING-stale sweep, whose behavior is unchanged for these legacy stages), the legacy code path is dead but harmless.

The two stalled docs require one manual `/v1/documents/{id}/reingest {"mode":"graph_only"}` after deploy. They go through the new path on first reingest and self-heal from then on.

## Rollout follow-ups (out of v1, additive)

Each reuses `_seed_first_stage`, `STAGE_SUCCESSORS`, the lifecycle integration in `guard_stage_run`, and the dispatcher — no new infrastructure required.

1. Migrate `mode="embeddings_only"` reingest to seed-and-dispatch
2. Migrate the post-merge `celery_chain(collect_derivations, derive_structure_links, derive_canonicalization, finalize_document)` inside `derive_ontology_graph_merge`
3. Migrate the per-pass fan-in (`dispatched_phases` JSONB) to per-pass dispatch rows using the existing `(pipeline_run_id, stage_name, pass_name, attempt)` partial unique index

## Testing

### Test layers

```
Layer 1 — Unit (pytest, no broker)
Layer 2 — DB integration (real Postgres, no broker)
Layer 3 — Pipeline integration (real Postgres + real broker)
```

### Headline test (Layer 3) — the regression test for this work

```python
def test_worker_death_between_stages_resumes_via_dispatcher(
    pg_session, celery_worker, small_pdf_fixture
):
    """Kill a worker mid-pipeline; verify the dispatcher resumes the run."""
    doc_id = ingest_fixture(small_pdf_fixture)

    wait_until_stage_complete(pg_session, doc_id, "derive_image_embeddings",
                              timeout_s=120)

    next_row = read_stage_run(pg_session, doc_id, "derive_document_anchors")
    assert next_row.status == "PENDING"
    assert next_row.celery_task_id is None
    assert broker_queue_depth("graph") == 0

    celery_worker.kill_and_restart()

    wait_until_stage_complete(pg_session, doc_id, "derive_document_anchors",
                              timeout_s=15)

    next_row = read_stage_run(pg_session, doc_id, "derive_document_anchors")
    assert next_row.status == "COMPLETE"
    assert next_row.celery_task_id is not None
```

### Death-window tests (Layer 2)

```python
def test_death_after_body_complete_call_before_wrapper_tx3(monkeypatch):
    """Body's _update_stage_run('COMPLETE') is intercepted; if wrapper dies
    before Tx-3 commits, ledger row stays RUNNING and is recoverable."""
    monkeypatch.setattr(pipeline, "_tx3_complete_and_enqueue_next",
                        lambda ctx: (_ for _ in ()).throw(SimulatedWorkerDeath()))
    drive_synthetic_stage_with_complete_call()
    row = read_stage_run(...)
    assert row.status == "RUNNING"
    assert row.finished_at is None
    assert no_successor_exists(...)

def test_death_after_tx3a_before_tx3b_rolls_back_both(monkeypatch):
    """Force a DB error between INSERT successor and UPDATE self → both roll back."""
    # Patch Tx-3b to raise; assert the INSERT also rolls back atomically.

def test_legacy_stage_unaffected_when_no_lifecycle_ctx():
    """Stage with lifecycle=False (or pre-deploy run with no ledger row) sees
    _update_stage_run('COMPLETE') write through to the DB as today."""
```

### Coverage matrix

| Test | What it asserts |
|---|---|
| `test_stage_successors_form_a_dag` | Every key in `STAGE_SUCCESSORS` reachable from `prepare_document`; no cycles |
| `test_resolve_queue_returns_actual_runtime_queue` | For every task in `STAGE_SUCCESSORS`, `_resolve_queue` returns the same queue name that an `apply_async()` call (without `queue=` override) would route to. Covers task_routes entries (`prepare_document → ingest`, `derive_image_embeddings → embed`, etc.) AND decorator-only fallbacks (`detect_and_translate → celery`, `derive_document_metadata → celery`, `derive_picture_descriptions → celery`) AND the broker default |
| `test_claim_first_dispatched_then_complete_skips_body` | Re-dispatch after success → wrapper returns early, `_CTX` not set |
| `test_claim_running_concurrent_no_retry_skips_body` | RUNNING + `is_celery_retry=False` → skip, `_CTX` not set |
| `test_claim_running_celery_retry_proceeds` | RUNNING + `is_celery_retry=True` → CLAIM succeeds, body runs |
| `test_claim_no_row_runs_body_without_ctx` | No summary row at all (legacy run) → body runs, `_CTX` is None |
| `test_update_stage_run_intercepts_complete_inside_ctx` | Body's `_update_stage_run("COMPLETE")` stashes metrics in `_CTX`; no DB commit |
| `test_update_stage_run_normalizes_uuid_vs_str` | ctx.pipeline_run_id is UUID, body passes str → still intercepts |
| `test_update_stage_run_passes_through_outside_ctx` | `_CTX is None` → existing implementation runs unchanged |
| `test_enqueue_next_atomic_with_self_complete` | Force DB error after Tx-3a but before Tx-3b → both writes roll back |
| `test_enqueue_next_idempotent_under_concurrent_dispatch` | Two workers reach Tx-3 → only one PENDING successor exists |
| `test_failure_retryable_path` | `dispatch_attempt += 1`, `attempt` unchanged, `available_at` advances, `started_at` resets to NULL |
| `test_failure_terminal_path` | After `dispatch_attempt > settings.max_stage_dispatches`: stage_run = FAILED, pipeline_run = FAILED, `attempt` still 1 |
| `test_attempt_column_never_mutated_for_ledger_rows` | Drive 5 retries on a ledger stage; assert `attempt = 1` on the summary row throughout, `dispatch_attempt` increments |
| `test_tx3a_idempotent_after_dispatch_attempt_bump` | Force a Tx-4 retry (dispatch_attempt=2), then have an upstream stage's Tx-3a re-fire with attempt=1 → existing row found via index conflict; no duplicate created |
| `test_celery_retry_passthrough_no_tx4` | Body raises CeleryRetry → wrapper passes through, no Tx-4, row stays RUNNING |
| `test_return_dict_status_failed_triggers_tx4` | Body returns `{"status":"FAILED",...}` → Tx-4 runs |
| `test_return_dict_status_skipped_advances_pipeline` | Body returns `{"status":"skipped","reason":"disabled"}` → wrapper enqueues successor and marks self COMPLETE; pipeline_run stays PROCESSING; skip metadata preserved in `metrics` |
| `test_three_real_stages_skipped_paths_advance` | Run a fixture where `translation_enabled=False`, no metadata available, and no pictures → all three stages return skipped, pipeline advances through them, downstream stages run |
| `test_dispatcher_skip_locked_disjoint_claims` | Two concurrent ticks against 100 PENDING rows → claimed_a + claimed_b == 100 |
| `test_dispatcher_publish_failure_undoes_claim` | Mock publish raises → row returns to PENDING |
| `test_dispatcher_no_queue_override` | `apply_async` called without `queue=`; routing comes from `task_routes` |
| `test_dispatcher_ignores_pass_name_rows` | Insert `pass_name='radar_modulation'` PENDING → never claimed |
| `test_dispatcher_ignores_runs_not_processing` | pipeline_run.status='FAILED' → not claimed |
| `test_dispatcher_ignores_rows_without_task_name` | Insert summary row with `task_name=NULL` PENDING → never claimed |
| `test_dispatcher_respects_available_at` | available_at = NOW + 60s → not claimed |
| `test_stale_running_resets_to_pending_with_dispatch_attempt_bump` | RUNNING > threshold for `LEDGER_SEQUENTIAL_STAGES` member → status='PENDING', `dispatch_attempt+=1`, `available_at=NOW()`, `attempt` unchanged |
| `test_stale_running_terminalizes_at_max_dispatches` | `dispatch_attempt` already at max → status='FAILED', pipeline_run='FAILED' in same sweep |
| `test_stale_running_excludes_stage_9` | RUNNING > threshold for `derive_ontology_graph` → sweeper does NOT touch the row; reconcile_ontology_graph_runs is the only owner |
| `test_stale_running_outside_ledger_uses_legacy_path` | RUNNING > threshold for non-ledger stage → existing FAILED+redispatch path |
| `test_stale_dispatched_resets_to_pending` | DISPATCHED > threshold → status='PENDING', `dispatch_attempt` unchanged |
| `test_stale_dispatched_ignores_rows_without_task_name` | DISPATCHED row with `task_name=NULL` (shouldn't exist, defense-in-depth) → not reset |
| `test_stale_threshold_exceeds_celery_retry_envelope` | Startup assertion fails if any wrapped stage's max_retries × default_retry_delay > threshold |
| `test_drain_legacy_run_completes_without_lifecycle` | Pre-deploy PROCESSING run with no ledger row finishes normally; existing inline writes commit |
| `test_lifecycle_enforcement_assertion` | Define a `STAGE_SUCCESSORS` key whose registered task lacks `task.run._lifecycle is True` → `RuntimeError` at import |
| `test_ctx_var_reset_between_tasks` | Run two synthetic tasks back-to-back in the same process; assert `_CTX.get() is None` at the start of the second wrapper invocation regardless of what the first did |
| `test_reingest_graph_only_seeds_pending_then_dispatcher_publishes` | Call `reingest_graph_only` from a test fixture; assert one PENDING row for `derive_document_anchors` exists with `task_name` set; dispatcher tick publishes it; row reaches RUNNING within 5s |
| `test_claim_zero_rows_disambiguation` | Force each of the 5 zero-row outcomes (COMPLETE / RUNNING+no-retry / PENDING / FAILED / no-row); assert correct `outcome` value and `_CTX` is never set |
| `test_ctx_never_set_on_zero_row_claim` | For each zero-row outcome, assert `_CTX.get() is None` both during the early-return and after the wrapper exits |
| `test_undo_claim_returns_to_pending_without_attempt_bump` | Publish failure path: `_undo_claim` writes status='PENDING', `dispatched_at=NULL`, `dispatch_attempt` unchanged |
| `test_startup_assertion_envelope_exceeds_threshold` | Patch a stage's `time_limit` to threshold+1; module load must raise `RuntimeError` naming the offending stage |

### Explicitly NOT tested in v1

- Per-pass fan-in (out of scope; existing tests for `derive_ontology_graph_pass` still cover it)
- `mode="embeddings_only"` reingest path (still on old chain in v1)
- Performance under sustained 1k-doc load (correctness first; load-test follow-up)
- Backfill of in-flight runs (drain rollout means nothing to backfill-test)

## Live-system verification after deploy

This section covers checks the implementer must perform post-deploy. Functional equivalence to the old chain is verified here rather than in unit tests because the old chain code path is removed by this change — there's no in-process branch to A/B against.

```bash
# 1. New ingest goes through the dispatcher.
docker logs eip-mmdpp-beat-1 --since 1m | grep DISPATCHER_TICK

# 2. The two stalled docs resume after manual reingest.
curl -sX POST http://localhost:8005/v1/documents/<id>/reingest \
     -H "content-type: application/json" \
     -d '{"mode":"graph_only"}'
# Watch /v1/documents/<id>/stages — derive_document_anchors PENDING within 1s,
# DISPATCHED within 5s.

# 3. Chain bypass is real.
docker exec eip-mmdpp-redis-1 redis-cli LLEN celery
# Small bursts at tick time only.

# 4. Stale-DISPATCHED detection alive.
docker logs eip-mmdpp-worker-1 --since 30m | grep "stale; reset by dispatcher sweeper"
# Empty on healthy systems.

# 5. Functional equivalence to old chain (manual A/B).
#    Capture a baseline from the most recent pre-deploy successful ingest of
#    each document type (PDF, jpg, txt, handwritten). For each, record:
#      - Final stage_runs row count and per-stage metrics
#      - text_chunks and image_chunks counts
#      - ArcadeDB element/edge counts for that document
#    Re-ingest the same documents post-deploy and diff. The intermediate
#    stage_runs row sequence WILL differ (one row per stage now, vs one per
#    Celery attempt before), but terminal counts and downstream graph state
#    must match within ±1 (rounding/timing noise on counts).

# 6. Single-doc latency observation (AC12 verification).
#    Time a fresh ingest of one small-fixture document end-to-end.
#    Expected: within ~20s of pre-deploy median for the same doc.
```

Steps 5 and 6 replace the originally-planned `test_functional_equivalence_to_old_chain` and AC12 unit assertions — both are properties that can only be observed in a live environment, not unit-tested against a code branch that no longer exists.

## Observability summary

- Per-stage timings already on `stage_runs` (`started_at`, `finished_at`, `metrics`). Unchanged.
- New: `available_at`, `dispatched_at` give end-to-end latency breakdown:
  - **dispatch latency** = `dispatched_at − available_at` (≤ 5s in steady state)
  - **accept latency** = `started_at − dispatched_at` (< 1s; > 30s indicates worker overload)
- `error_message` populates on retry and persists across attempts via Tx-4
- Three dispatcher log lines cover tick-level visibility

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Beat is the only trigger; if beat dies, all new ingests stall | Existing `eip-mmdpp-beat-1` already runs scan-watch-directories, community-detection, and stale-sweep; beat being down is an existing top-priority alert |
| Dispatcher publish loop could amplify a broken broker | `_undo_claim` returns rows to PENDING; LIMIT 50 per tick caps amplification |
| Stage author forgets to add `lifecycle=True` to `guard_stage_run` | Module-load assertion: iterate `STAGE_SUCCESSORS` keys; for each persisted stage name `s`, locate the registered Celery task whose `task.run.stage_name == s` (the marker `guard_stage_run` sets via `wrapper.stage_name = stage_name`) and assert `task.run._lifecycle is True`. Raise `RuntimeError` at import time on any mismatch |
| Stale-RUNNING sweeper preempts Celery's own retries or healthy long-running attempts | Startup assertion: `stale_stage_run_threshold_seconds` ≥ `time_limit + max_retries × default_retry_delay` for every ledger-wrapped stage. Existing setting (34200s / 9.5h) is reused unchanged — it was calibrated against `picture_desc_time_limit=21600` |
| `_CTX` ContextVar leaks across Celery task invocations in the same prefork worker | Wrapper uses `try/finally` around the body; `_CTX.reset(token)` is in the `finally` block. Test `test_ctx_var_reset_between_tasks` asserts |
| Watcher writes empty `celery_task_id` to `Document` | Empty-string writes are already accepted (existing fallback behavior). Audit `Document.celery_task_id` consumers during implementation; treat `""` and `NULL` equivalently |
| Concurrent `start_ingest_pipeline` for same document | Existing `FOR UPDATE` guard on PROCESSING runs prevents this; `_seed_first_stage` is idempotent via `ON CONFLICT DO NOTHING` |
| Migration creates index that locks the table | Partial index on summary rows is small; `CREATE INDEX CONCURRENTLY` not needed at current table size, can be added if it grows |
| `_update_stage_run` ID type mismatch silently bypasses interception | Comparison uses `str()` on both sides; test `test_update_stage_run_normalizes_uuid_vs_str` covers both directions |
| Legacy-mode `_CTX` leak | CLAIM only sets `_CTX` when a summary row exists; test `test_drain_legacy_run_completes_without_lifecycle` asserts |

## Acceptance criteria

1. Killing a worker between stages 1–8 of a fresh ingest no longer strands the pipeline_run; the dispatcher resumes within 10s of the next beat tick.
2. Killing a worker after a stage body's `_update_stage_run("COMPLETE")` call but before the wrapper's Tx-3 commit leaves the ledger row in `RUNNING` (not `COMPLETE`); the stale-RUNNING sweeper recovers it within `stale_stage_run_threshold_seconds`.
3. `reingest_graph_only` writes a single PENDING row and returns; the dispatcher publishes within 5s.
4. Stages that return `{"status":"skipped", ...}` (today: `detect_and_translate` with translation disabled, `derive_document_metadata` when nothing to derive, `derive_picture_descriptions` when no pictures) advance the pipeline — successor row is inserted, downstream stages run, pipeline_run reaches COMPLETE.
5. A ledger summary row's `attempt` column remains `1` across the row's entire lifetime, regardless of how many ledger retries happen; `dispatch_attempt` is the field that increments.
6. There is exactly one ledger summary row per `(pipeline_run_id, stage_name)` for any pipeline_run that uses the ledger code path. Re-publication after a Tx-4 retry finds the existing row and updates it; it cannot create a duplicate.
7. Stage 9's summary row is never modified by the ledger sweeper while in RUNNING state — only `reconcile_ontology_graph_runs` (existing) may modify it during fan-in.
8. All existing pipeline tests still pass.
9. The headline regression test, the death-window tests, the skipped-advances tests, and the attempt-invariance test all pass.
10. Documents that complete successfully under the new model produce identical *terminal* `pipeline_runs`, downstream graph state, and chunk/embedding state to documents under the old chain. The *intermediate* stage_runs row sequence differs by design (one row per stage instead of one per Celery attempt; no transient `COMPLETE`-without-successor window), and tests assert the new sequence. Functional equivalence to the old chain is verified in **Live-system verification step 5** rather than unit-tested, because the old chain code path is removed by this change.
11. Pre-deploy PROCESSING runs complete without modification under the legacy code path; their existing inline `_update_stage_run` writes commit through to the DB as today.
12. No measurable change in successful-document end-to-end latency beyond the expected ~20s mean (8 handoffs × 2.5s avg dispatcher latency). Verified in **Live-system verification step 6** rather than unit-tested.
