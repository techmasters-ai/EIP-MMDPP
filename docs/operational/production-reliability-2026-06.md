# Production Reliability Issues — extraction pipeline (2026-06)

**Scope:** standalone tracker for production reliability defects, separate from
the guarded-ranker chunk-selection plan. **Production = this deployment**
(operator-confirmed): the live `eip-mmdpp` stack bind-mounts the worktree
(`walltime/c0-telemetry`) and reads the live `.env`; there is no separate prod
host. All settings below verified in the RUNNING containers (`docker exec
printenv`), not just files.

**Deploy constraint:** a re-collection run (Engagement, `66a2afef`) is in flight.
Worker/docling-graph restart or rebuild would disrupt it. Therefore: code/config
changes are PREPARED + COMMITTED now, and DEPLOYED (worker restart +
docling-graph rebuild + in-container verify) in the window AFTER Engagement
completes — batched with the guarded-ranker fallback-fix deploy and the 3-small-
doc re-run.

Severity legend: **S1** = silent multi-hour/day stall or data loss; **S2** =
degraded throughput / wasted GPU; **S3** = cosmetic / debuggability.

---

## R1 — BATCH_HARD_TIMEOUT is a total-elapsed ceiling, not a per-batch watchdog  [S1/S2, code]

**Root cause (verified).** `docker/docling-graph/patches/0001-orchestrator-batch-hard-timeout.patch`
wraps the batch loop in `as_completed(futures, timeout=_per_batch_timeout)` and
`future.result(timeout=_per_batch_timeout)`. `as_completed`'s `timeout` is the
**total** wall-clock budget for the entire iteration over ALL futures — not
per-future. The patch's own comment ("The per-batch ceiling here MUST be >= the
OllamaPool client's read-timeout") shows the author believed it was per-batch.

**Effect.** A pass with N batches on P workers needs ≈ ⌈N/P⌉ × (per-batch time)
wall-clock. For a 175-chunk doc: 174 batches / 2 workers × (60–1800 s/batch) ≫
the live ceiling (10800 s = 3 h; raised to 28800 s = 8 h on 2026-06-12 as a
mitigation). When the total ceiling trips, ALL not-yet-completed futures —
including healthy queued ones — are cancelled, labelled "hung … silent-deadlock
class", and degraded to the **sequential** split-retry loop, which is strictly
slower than letting them run. Observed firing on the original Engagement run
(both identity passes, 15:53), undisturbed doc 8 (SA2_RU missile_identity, 21/67
mislabelled), and absent post-mitigation on the Engagement retry only because
the ceiling was raised above this doc's total.

**Why the mitigation is a band-aid.** Raising the ceiling (3 h → 8 h) just moves
the cliff; any doc whose aggregate batch time exceeds the ceiling still
mis-degrades. The 21-doc collection and production full-bundle docs will hit it.

**Fix design (no-progress / inter-arrival watchdog).** Replace the total-elapsed
ceiling with a *stall* detector: track time since the last future completion;
abort only if NO future has completed within a `no_progress_timeout` (≥ the
1800 s OllamaPool stream wall-timeout, e.g. 2400 s). A slow-but-progressing pass
never trips it; a genuinely deadlocked pool (every worker stuck, no completions)
trips within one window. Cancel only the still-running futures on trip. This
decouples the watchdog from doc size. Couples with R2: once per-LLM-call
timeouts are finite (they are — 1800 s stream wall-timeout enforces it), a
future cannot hang forever, so the watchdog is a backstop, not the primary
guard.
- Implementation: rework the patch's `try/except _FuturesTimeoutError` block to a
  loop that waits on `as_completed` in `no_progress_timeout` slices and breaks on
  a stall (no completion in a full slice). Extract the wait logic into a pure,
  host-testable helper so the inter-arrival semantics are unit-tested
  independent of the container.
- Config: rename intent — keep `DOCLING_GRAPH_BATCH_HARD_TIMEOUT_SECONDS` as the
  no-progress window (document the changed meaning), default 2400.
- Test: host unit test of the helper with fake futures (steady completions → no
  trip; one slow-but-progressing → no trip; all-stalled → trip after one window).
  In-container build + a large-doc smoke at deploy time.

**Status:** PREPARED this turn (patch + helper + host test, committed). Build +
in-container verify + deploy gated on Engagement completion.

---

## R2 — No effective timeout/watchdog ladder; a stuck pass hangs for ~days  [S1, config + code]

**Root cause (verified in-container).** `PASS_SOFT_TIME_LIMIT=288000` (80 h),
`DOCLING_GRAPH_TIMEOUT=720000` (200 h, worker→DG HTTP),
`DOCLING_GRAPH_LLM_TIMEOUT=720000` (200 h per call),
`CELERY_VISIBILITY_TIMEOUT=360000` (100 h); reconciler stale-dispatched
threshold = `2 × pass_soft_time_limit` = 160 h
(`pipeline.py:9441`), and the reconciler is **timestamp-only** — no Celery
liveness, heartbeat, or progress check (`pipeline.py:9418`). So nothing
distinguishes a stuck pass from a slow one, and the first watchdog that could
act does so at 80–200 h.

**Effect.** A genuinely deadlocked pass sits for days; even the operator's
manual SIGUSR1 was the only thing that moved the original Engagement run.

**Why we cannot just lower the timeouts.** Large docs legitimately need long
passes — Engagement's identity passes ran ~10 h each; total ~24 h. Naively
setting `PASS_SOFT_TIME_LIMIT` to e.g. 4 h would kill healthy large-doc passes.
The timeout ladder and the "stuck vs slow" problem are the same problem: **we
need a progress signal, not a shorter deadline.**

**Fix design (progress heartbeat + progress-aware staleness).**
1. **Heartbeat:** docling-graph emits per-pass batch progress (done/total +
   timestamp) periodically during a pass — written to `ingest.stage_runs.metrics`
   (or a dedicated `progress_json`) via a lightweight callback, OR the worker
   polls a DG progress endpoint. (R1's batch loop already has the completion
   events to drive this.)
2. **Progress-aware reconciler:** stale = "no progress advance in T minutes"
   (T ≈ 2 × no_progress_window), not "age > 160 h". Reclaim/fail a pass whose
   progress hasn't moved, regardless of absolute age.
3. **Then** the absolute timeouts can come down to sane backstops (e.g. HTTP
   12 h, soft-limit decoupled) without endangering slow-but-progressing docs.
4. **Visibility:** a 24 h `progress` view per active run so a human can see
   done/total without reading container logs.

**Status:** DESIGN only this turn. Heartbeat emission lives in the docling-graph
fork + worker; needs in-container iteration. Recommend implementing in the
post-Engagement window (it also makes future large-doc runs observable). Do NOT
lower the timeout ladder until the heartbeat exists.

---

## R3 — Truncation pathology: unconstrained decoding on the `/v1` endpoint  [S2, config/model]

**Root cause.** gemma4:31b on the OpenAI-compat `/v1` endpoint ignores the
native `format` param (see memory `project_format_param_noop_on_v1`); decoding
runs unconstrained and frequently streams to the `num_predict` cap (32768)
producing `len=0`, then retries at 65536 — ~16 events on the Engagement run,
~45 min each.

**Effect.** Large fraction of large-doc wall time burned on truncation retries.

**Fix design.** Migrate extraction to `response_format` with a JSON **schema**
(`json_schema`, not `json_object`) on the `/v1` path so decoding is constrained;
and/or cap `num_ctx`/`num_predict` sanely. This is a known, separate workstream
(its own spec) touching the LLM client + per-pass schema emission.

**Status:** DESIGN/refer-out. Not implemented here. File as its own effort.

---

## R4 — Failure cleanup deletes the run's ExtractionChunk index  [S3, working-as-designed]

**Re-assessment.** `_terminalize_doc_and_run` (`pipeline.py:1657`) calls
`cleanup_extraction_index(run_id)` which deletes all ExtractionChunk rows for the
run. Audited behavior: **COMPLETE runs retain their chunks** (export-safe — the
2026-06-13 dataset audit confirmed all completed runs keep chunk_text); cleanup
fires on the failure/PARTIAL_COMPLETE terminalize paths (`:4180`, `:8978`). For a
genuinely dead run this is correct hygiene (working data; re-ingest rebuilds the
index). The incident's deletion was of a run that had FAILED (my SIGUSR1 caused
the failure) — cleanup behaved as designed.

**Optional improvement (low priority).** For debuggability, preserve chunks on
failure (let the hourly janitor reap by age) instead of immediate delete, so a
failed run can be inspected. Not a correctness fix; no data-loss for COMPLETE
runs.

**Status:** no action required for the plan; optional follow-up.

---

## R5 — Dispatch-race warning + reconciler liveness (rolled into R2)  [S3]

`mark_phase_dispatched: state has changed under us` is benign noise (fires on
~every dispatch; 88 occurrences in 3 days incl. clean runs). The reconciler's
lack of liveness is the substantive part — covered by R2.

---

## Deploy plan (post-Engagement window, batched)

When `66a2afef` reaches terminal:
1. `git` already carries: guarded-ranker fallback fix (`8a5dcf4`), R1 watchdog
   patch (this turn). Worker restart picks up worker-side code; docling-graph
   rebuild picks up the R1 patch.
2. Sequence (while NO run is PROCESSING): `docker compose -p eip-mmdpp build
   docling-graph` → verify build succeeds (a bad R1 patch fails the build here,
   old image keeps running — no silent breakage) → `up -d --force-recreate
   docling-graph` → `docker restart eip-mmdpp-worker-1 eip-mmdpp-worker-graph-1`
   → verify `StartedAt` + in-container env + a large-doc smoke.
3. Re-run the 3 small docs (SNR-75, V-75, Dvina) so they capture gate-union +
   row_cosines (guarded-ranker fallback fix) — uniform dataset_v2.
4. Then export dataset_v2 + check_gate_coverage + eval (Phase 3).

R2 (heartbeat) and R3 (json_schema) are larger and tracked as their own efforts.
