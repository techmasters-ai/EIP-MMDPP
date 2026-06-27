# R2 — Progress-Aware Watchdog (full heartbeat) — Design Spec

Date: 2026-06-14
Status: approved (user, 2026-06-14)
Worktree: `.worktrees/walltime-c0-telemetry` (branch `walltime/c0-telemetry`)
Tracker: `docs/operational/production-reliability-2026-06.md` §R2
Predecessor: R1 (no-progress batch watchdog) deployed 2026-06-14 (`f880232`).

## 0. Problem & constraints

**Problem (R2).** Production (= this deployment) has no effective timeout/watchdog
ladder: `PASS_SOFT_TIME_LIMIT`=80h, worker→DG `DOCLING_GRAPH_TIMEOUT`=200h, and
the reconciler's stale-dispatched threshold = `2 × pass_soft_time_limit` = 160h,
with NO progress signal. A genuinely stuck pass hangs for days; the reconciler
(`reconcile_ontology_graph_runs`, `pipeline.py:9418`) is purely timestamp-driven
and cannot tell "stuck" from "slow." Naively lowering the timeouts would kill
legitimately long large-doc passes (Engagement identity passes ran ~10h each).

**Enabling fact.** R1 (deployed) caps *in-batch-loop* hangs (no-progress watchdog,
3600s). The remaining "hangs for days" cases are hangs OUTSIDE the batch loop
(worker↔DG connection, post-loop merge/clean, an Ollama call dodging the stream
timeout) and dead celery tasks. A progress heartbeat distinguishes stuck from
slow precisely; sane absolute backstops become viable because R1 removes the
pathological slow-degradation that forced the 80–200h ladder.

**Constraints.** Prepare + commit + unit-test NOW; **no container restart** while
the Engagement re-run + 3 small docs are processing. Deploy when the dataset
lands. Every behavior change is **flag-gated default-off or passive**, so neither
committing nor an accidental deploy can disturb the in-flight passes.

**Architecture fact (verified).** docling-graph has NO postgres/redis access (no
deps, no env). It cannot write progress to the DB directly. uvicorn runs a single
process (Dockerfile CMD has no `--workers`), so an in-memory registry is shareable
within DG. The worker calls `/extract-pass` synchronously and is blocked for the
whole pass, so it cannot poll progress mid-pass — a separate poller (with DB
access) bridges DG → postgres.

## 1. Component 1 — DG progress registry + `/progress` endpoint (passive, always-on)

- `docker/docling-graph/app/schemas.py`: add `pipeline_run_id: str | None = None`
  to `ExtractPassRequest` (registry key alongside the existing `pass_name`).
- New module `docker/docling-graph/app/progress_registry.py`: a module-level
  `dict[(run_id, pass_name)] → {done:int, total:int, started_at:float,
  updated_at:float}` guarded by a `threading.Lock` (the batch pool is a
  ThreadPoolExecutor). API: `start(run_id, pass_name, total)`,
  `advance(run_id, pass_name)` (done += 1, updated_at = now), `clear(run_id,
  pass_name)`, `snapshot() → list[dict]`.
- Hook into the **R1 batch-drain loop** (the patch's
  `_drain_futures_with_progress_watchdog` consumer in `orchestrator.py`): `start`
  before the loop (total = `len(batch_plan)`), `advance` on each completed batch,
  `clear` in a `finally`. The orchestrator must receive `run_id`/`pass_name` —
  thread them from the request through `build_pipeline_config`/the extractor (or
  pass a callback). If threading the ids into the orchestrator is too invasive,
  fall back to a per-request callback the endpoint registers.
- `GET /progress` (docling-graph `main.py`): returns `snapshot()` — list of
  `{run_id, pass_name, done, total, updated_at, age_s}`. Read-only.
- **Phase-transition heartbeat (important):** the batch loop is only part of a
  pass — after it come merge/normalize/clean/sanitize, which do NOT advance the
  batch counter. To prevent a false "no-progress" reading during those phases,
  the registry must bump `updated_at` (without changing done) at each phase
  transition (e.g. a `touch(run_id, pass_name, phase="merging")` call). For
  realistic docs these post-loop phases are seconds-to-minutes (<< the 2h
  threshold), but the touch makes the heartbeat robust for very large docs and
  records the current phase for observability.
- **Safety:** purely additive; never alters extraction. Always-on after deploy.

## 2. Component 2 — progress poller (beat task; flag-gated)

- New celery task `poll_extraction_progress` (`app/workers/pipeline.py`), beat
  schedule in `app/workers/celery_app.py`, interval ~`reconciler_period_seconds`.
- For each PROCESSING ontology-graph run: GET `{internal_dg_url}/progress`, match
  `(run_id, pass_name)` entries, and write `{done, total, updated_at}` into the
  matching `ingest.stage_runs.metrics->'progress'` (JSON merge — no migration;
  reuse the existing metrics-update path).
- Flag `DG_PROGRESS_POLLER_ENABLED` (default **false**): task returns
  `{"status":"disabled"}` immediately when off.
- **Safety:** read-only on DG; writes only the `metrics.progress` sub-key; no-op
  when the flag is off.

## 3. Component 3 — progress-aware reconciler (flag-gated, default OFF)

- Flag `RECONCILER_PROGRESS_AWARE` (default **false**). When **false**, the
  reconciler behaves EXACTLY as today (absolute threshold only) — so committing
  and deploying this code cannot reclaim/kill any in-flight pass.
- When **true**, for a dispatched pass the staleness rule becomes:
  - if `metrics.progress.updated_at` exists and has advanced within
    `RECONCILER_NO_PROGRESS_THRESHOLD_S` (default 7200s = 2h, > R1's 1h window) →
    NOT stale (progressing), regardless of absolute age;
  - if `metrics.progress.updated_at` is older than the threshold AND not
    pending-retry → stale → reclaim/fail;
  - if NO progress data exists (poller off, or a pre-heartbeat run) → fall back to
    the absolute `stale_dispatched` threshold (Component 4).
- Implementation: factor the staleness decision into a pure helper
  `is_dispatched_phase_stale(progress, now, *, no_progress_s, absolute_s,
  progress_aware) → bool` so it is unit-tested independent of the DB scan.
- **Double-dispatch safety (critical).** Reclaiming a dispatched phase while the
  original worker is still blocked in its `/extract-pass` HTTP call causes a
  duplicate pass run. This is acceptable ONLY when the original is genuinely
  stuck (no progress) — which is exactly the condition Component 3 fires on. To
  keep false reclaims near-zero: (1) the no-progress threshold (2h) is set far
  above the poller interval (~60s) and the longest legit non-batch gap, so a
  lagging poller or a merge phase never trips it; (2) reclaim reuses the existing
  `reclaim_stale_phase` + phase state machine (idempotent claim), so a reclaim
  that races a late completion is resolved by that machinery, not by a blind
  re-launch. The progress-blind worker→DG HTTP timeout (Component 4, 18h) only
  bounds how long the STUCK original wastes its slot after reclaim — it is the
  one value that ignores progress, so it must stay above the largest legitimate
  pass; revisit it if a real pass ever approaches it.

## 4. Component 4 — sane backstop timeouts (config; deploy-gated)

Viable because R1 caps in-loop hangs and Component 3 protects slow-but-progressing
passes. Values (live `.env` + `.env.example`, parity):

| Var | Current | New | Rationale |
|---|---|---|---|
| `DOCLING_GRAPH_TIMEOUT` | 720000 (200h) | 64800 (18h) | worker↔DG HTTP backstop; > longest legit pass (~10h post-R1) + margin |
| `DOCLING_GRAPH_LLM_TIMEOUT` | 720000 (200h) | 64800 (18h) | per-call; the 1800s stream wall-timeout is the real guard, this is a ceiling |
| `PASS_SOFT_TIME_LIMIT` | 288000 (80h) | 72000 (20h) | > HTTP backstop so the HTTP timeout fails the pass cleanly first |
| reconciler stale-dispatched | `2×pass_soft`=160h | `RECONCILER_STALE_DISPATCHED_S`=86400 (24h) | decouple from pass_soft; fixed catch-all even when progress-aware off |

`CELERY_VISIBILITY_TIMEOUT` (100h) stays — must remain > the longest task
`time_limit` (`GRAPH_TIME_LIMIT`=90h) to avoid broker-redelivery double-runs.
The in-flight ~10h passes sit under all new backstops, so even an accidental
recreate cannot kill them.

## 5. Flag-gating / safe rollout

- Component 1 (registry + `/progress`): always-on, passive — no extraction change.
- Component 2 (poller): `DG_PROGRESS_POLLER_ENABLED=false`.
- Component 3 (reconciler): `RECONCILER_PROGRESS_AWARE=false` → inert.
- Component 4 (backstops): config, applied at the dataset-deploy recreate.

Rollout at the dataset-deploy: rebuild docling-graph (registry + R1 hook) +
recreate workers/api with new code, **flags OFF**, backstop timeouts applied →
on the NEXT run, verify `/progress` emits and the poller writes
`metrics.progress` → flip `DG_PROGRESS_POLLER_ENABLED` then
`RECONCILER_PROGRESS_AWARE` on once confirmed.

## 6. Testing (host, no deploy)

- **Registry:** unit-test start/advance/clear/snapshot + lock behavior with fake
  completions; assert done monotonic, updated_at advances, clear removes.
- **R1-loop hook:** extend the R1 watchdog test — driving fake batch completions
  advances the registry; a stall does not.
- **Poller:** unit-test GET-`/progress` → `metrics.progress` mapping with a mocked
  DG response + mocked DB; flag-off → no-op.
- **Reconciler helper (highest risk):** `is_dispatched_phase_stale` —
  progressing→not stale; no-progress>T→stale; flag off→ABSOLUTE only (today's
  behavior); no-data→absolute fallback; pending-retry respected.
- **Byte-identical default:** existing reconciler tests pass UNCHANGED with
  `RECONCILER_PROGRESS_AWARE=false`.
- **Config parity:** `.env` and `.env.example` agree on every R2 var; new flags
  default to the safe value.

## 7. Files

- `docker/docling-graph/app/schemas.py` — `ExtractPassRequest.pipeline_run_id`
- `docker/docling-graph/app/progress_registry.py` — new registry module
- `docker/docling-graph/app/main.py` — `GET /progress`
- `docker/docling-graph/patches/0001-orchestrator-batch-hard-timeout.patch` (or a
  new patch) — registry start/advance/clear in the R1 drain loop
- `app/workers/pipeline.py` — worker passes `pipeline_run_id`;
  `poll_extraction_progress`; progress-aware reconciler + helper
- `app/workers/celery_app.py` — poller beat schedule
- `app/config.py` + `.env` + `.env.example` — flags + backstop timeouts
- `tests/unit/...` — registry, poller, reconciler-helper, config-parity tests
- docling-graph host tests for the registry/loop hook (run via `run_dg_lineage`)

## 8. Deploy (when dataset lands)

Per §5. Rebuild docling-graph + recreate workers/api with flags OFF + backstops;
verify `/progress` + poller on the next run; flip the flags on. Update
`docs/operational/production-reliability-2026-06.md` §R2 to "implemented +
deployed."

## 9. Out of scope

- Lowering `CELERY_VISIBILITY_TIMEOUT` (must stay above task limits).
- The R3 truncation/json_schema migration (separate effort).
- Any change to the guarded-ranker selection path (this is orchestration only).
