# Incident: Engagement re-collection run de4d0c11 — slow-degraded run misdiagnosed as wedge, killed by operator signal

Date: 2026-06-11. Status: root-caused (3-agent forensic investigation, read-only).
Run: `de4d0c11-b47a-4ff8-a12b-02b52cc4a85b` (Engagement doc, 175 chunks, graph_only,
air_defense_v3, part of the Phase-3 guarded-ranker re-collection).

## What actually happened (corrected narrative)

1. **The run was never wedged.** Both identity passes (radar_identity on worker-1,
   missile_identity on worker-graph-1) dispatched normally at 12:53 and ran
   continuously: ~469 Ollama completions between 12:53 and 20:22 (49–78/hour).
   Worker-side silence is **normal by design** — each pass is ONE synchronous
   `httpx.post` to docling-graph (`pipeline.py:4507`) that blocks until the whole
   pass finishes server-side; healthy passes produce the identical 3 log lines.
2. **The capacity math guaranteed degradation.** 175 chunks → ~174 batches per
   pass × 2 passes × 2 threads each on a 2-host gemma4:31b pool. At observed
   batch rates this cannot fit inside `DOCLING_GRAPH_BATCH_HARD_TIMEOUT_SECONDS=10800`.
3. **BATCH_HARD_TIMEOUT is mis-implemented** (patch
   `docker/docling-graph/patches/0001-orchestrator-batch-hard-timeout.patch`,
   `delta/orchestrator.py:552`): `as_completed(futures, timeout=10800)` is a
   **total-elapsed ceiling**, not the per-batch hung-future watchdog the comment
   describes. At 15:53 it fired for BOTH passes, mislabeled queued-but-healthy
   batches as "hung … silent-deadlock class" (missile 73/174, radar 103/174),
   cancelled them, and degraded the remainder to a **sequential** split-retry
   tail — strictly slower than doing nothing. Every large doc will hit this.
4. **Truncation pathology burned the budget**: 16 events of gemma4:31b streaming
   1300–1520 s and delivering len=0 at the 32768 `num_predict` cap, each followed
   by a 65536-token retry (~40–50 min wasted per event). Consistent with the
   known unconstrained-decoding issue (`format` param no-op on /v1).
5. **Every watchdog was configured between 80 h and 200 h**, so nothing flagged
   7.5 h as abnormal:
   | Layer | Setting | Live value |
   |---|---|---|
   | worker → docling-graph HTTP | `DOCLING_GRAPH_TIMEOUT` (.env:253) | 720000 s = **200 h** |
   | celery pass task soft limit | `PASS_SOFT_TIME_LIMIT` (.env:324) | 288000 s = **80 h** (comment says "8h → 80h"; possibly a 10× typo) |
   | celery hard limit | — | none |
   | reconciler stale-dispatched | `2 × pass_soft_time_limit` (pipeline.py:9441) | **160 h** |
   | docling-graph per-LLM-call read | `DOCLING_GRAPH_LLM_TIMEOUT` (.env:226) | 720000 s = **200 h** (inverts the batch-watchdog patch's own precondition by 400×) |
   | batch ceiling | `DOCLING_GRAPH_BATCH_HARD_TIMEOUT_SECONDS` (.env:292) | 10800 s = 3 h (the only *short* limit — and the mis-scoped one) |
   The only effective per-call protection is the pool client's 1800 s stream
   wall timeout, which fired and retried correctly 5 times.
6. **The dispatch-race warning was benign.** `mark_phase_dispatched: state has
   changed under us` fires on essentially every pass dispatch (88 occurrences in
   3 days, including clean runs). Not causal.
7. **Operator error (mine) ended the run.** Diagnostic mistakes: (a) 0.4 s
   responses from tiny Ollama probes were read as "pool idle" — they only prove
   the server accepts new requests (latency ≠ occupancy); (b) worker-side
   silence + zero DB rows were read as a stall, but both are expected during a
   long pass (`pipeline_pass_outputs` is inserted only at completion, with
   `created_at` **backdated** to pass start); (c) docling-graph logs — the one
   place the work was visible — were never checked. SIGUSR1 (celery's
   soft-limit signal) was sent to both pool processes: missile_identity had
   already finished naturally 23 s earlier (coincidence, not rescue);
   radar_identity was killed mid-`sock.recv` with ~71 batches left — it would
   most likely have completed. The failure handler then ran terminal cleanup,
   **deleting the run's 175 ExtractionChunk rows**, and the run went FAILED.
8. **Orphaned server-side work continued**: docling-graph kept executing
   radar_identity's batches after the client died (no cancellation propagation),
   contending with the next run (4f53397c) on the gemma pool.

## Immediate mitigation (idle window between doc 8 and the Engagement retry)

Config-only, reversible: raise `DOCLING_GRAPH_BATCH_HARD_TIMEOUT_SECONDS`
10800 → 28800 (8 h) so the Engagement retry runs parallel end-to-end instead of
degrading at 3 h (per-call hang protection remains the 1800 s stream wall
timeout, which works). Deploy = edit `.env` + `docker compose -p eip-mmdpp up -d
--force-recreate docling-graph` strictly while no run is PROCESSING; pause/relaunch
the retry driver around it. Expected effect: retry completes in ~4–5 h instead
of ~7.5 h, with no mislabeled-hung cancellations.

## Follow-up fixes (file with the walltime follow-ups; NOT mid-collection work)

1. **Per-batch watchdog semantics** in the DG fork's orchestrator: replace the
   `as_completed(total)` ceiling with per-future deadlines (or scale the ceiling
   with `ceil(batches/threads) × p95_batch_seconds`). The current behavior
   punishes exactly the docs that need the most time.
2. **Rationalize the timeout ladder**: worker HTTP and LLM read timeouts of
   200 h and a soft limit of 80 h make the system unobservable and the reconciler
   decorative. Proposal: `DOCLING_GRAPH_TIMEOUT` ≈ 12 h, `PASS_SOFT_TIME_LIMIT`
   back to a value that means something, reconciler threshold decoupled from it.
3. **Progress heartbeat**: docling-graph should surface per-pass batch progress
   (done/total) to the worker or DB (e.g. periodic stage_runs.metrics update)
   so a 7-hour pass is distinguishable from a dead one without reading container
   logs.
4. **Terminal cleanup must not delete the run-shared chunk index** while sibling
   passes of the same run are live (radar's failure deleted the index 23 s after
   missile's success); scope cleanup to truly-terminal runs.
5. **Cancellation propagation**: client death should cancel the server-side
   pass (request-scoped cancellation or run-id kill endpoint) instead of
   orphaning hours of GPU work.
6. **`created_at` backdating** on pipeline_pass_outputs misleads forensics;
   record both pass_started_at and inserted_at.
7. **stage_runs bookkeeping**: umbrella row left RUNNING forever; pass rows
   inconsistent (COMPLETE row with started_at NULL; FAILED row with pass_name
   NULL). Tighten the state writes.
8. **Truncation pathology** (16 × ~45 min wasted): the known `format`-param /
   unconstrained-decoding issue; fix tracked separately
   (response_format json_schema migration).

## Operator lessons (encoded in memory)

- Latency probes do not measure pool occupancy; check docling-graph extract-pass
  START/END lines and the Ollama completion rate instead.
- Worker silence + empty pass_outputs during a pass is the designed behavior of
  the single-blocking-POST architecture — not a stall signature.
- If a pass must be killed, prefer restarting workers (acks_late redelivery
  re-runs cleanly) over signaling pool processes; and know that the failure
  path currently deletes the run's chunk index.
