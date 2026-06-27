# R2 Progress-Aware Watchdog — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the pipeline a progress heartbeat so the reconciler can reclaim a genuinely-stuck pass in hours (not 160h) without killing slow-but-progressing large-doc passes — per `docs/superpowers/specs/2026-06-14-r2-progress-aware-watchdog-design.md`.

**Architecture:** docling-graph (no DB access, single uvicorn process) keeps an in-memory progress registry fed by the R1 batch-drain loop and serves it at `GET /progress`. A flag-gated celery beat poller bridges `/progress` → `ingest.stage_runs.metrics->'progress'`. A flag-gated, default-off progress-aware branch in the reconciler reads that progress to distinguish stuck from slow. Sane backstop timeouts are applied **only after** progress-awareness is verified on.

**Tech Stack:** FastAPI/uvicorn (docling-graph fork), Celery beat (worker), SQLAlchemy/Postgres (`ingest.stage_runs`), pydantic-settings, pytest.

**User decisions (already made):** "Full heartbeat now" (all four components). "prepare + commit + unit-test now; NO container restart while the Engagement re-run + 3 small docs are processing; deploy when the dataset lands." "All behavior changes flag-gated default-off or passive." (Refinement, this plan: backstop timeout reduction is gated behind progress-aware being verified-on, not in the initial deploy.)

**Worktree:** `/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry` (branch `walltime/c0-telemetry`). All paths relative to it.

---

## STANDING RULES

1. **NO container build/restart/recreate** for Tasks 1–6 (re-run in flight). Tasks 1–6 are pure code + unit tests + commit. Deploy is Task 7 (single gated step when the dataset lands) and Task 8 (gated flag-flip + backstop reduction after verification).
2. **Flag/passive default-safe:** every behavior change is either passive (registry, `/progress`) or behind a flag defaulting to today's behavior (`DG_PROGRESS_POLLER_ENABLED=false`, `RECONCILER_PROGRESS_AWARE=false`). Committing or an accidental deploy must not reclaim/kill any in-flight pass.
3. **New env vars** mirrored into BOTH `.env` and `.env.example` with a default + comment (standing rule); compose needs `--force-recreate` to pick them up.
4. **docling-graph is COPY-based** → its changes need an image rebuild (Task 7). Worker/api bind-mount `app/` → restart/recreate reloads them.
5. **JSONB is not mutation-tracked in place:** always `m = dict(row.metrics or {}); m[...] = ...; row.metrics = m`.

## File structure

| File | Responsibility | Task |
|---|---|---|
| `docker/docling-graph/app/_progress_registry.py` (new) | process-global progress dict + lock; start/advance/touch/clear/snapshot | T1 |
| `docker/docling-graph/app/schemas.py` | `ExtractPassRequest.pipeline_run_id` | T2 |
| `docker/docling-graph/app/main.py` | thread-local `_pass_progress_overrides`; set/reset in `run_extraction_pass`; pass `run_id` from endpoint; `GET /progress` | T2, T4 |
| `docker/docling-graph/patches/0001-orchestrator-batch-hard-timeout.patch` | registry start/advance/clear in the R1 drain loop (import-guarded) | T3 |
| `app/workers/pipeline.py` | worker body `pipeline_run_id`; `poll_extraction_progress` task; progress-aware reconciler + `is_dispatched_phase_stale` helper; decouple stale_dispatched | T2, T5, T6 |
| `app/workers/celery_app.py` | beat entry for the poller | T5 |
| `app/config.py` + `.env` + `.env.example` | flags + poll interval (T5); backstop timeout reductions (T8) | T5, T6, T8 |
| `tests/unit/test_progress_registry.py`, `test_poll_extraction_progress.py`, `test_reconcile_ontology_graph_runs.py` (extend), DG host test | tests | T1,T5,T6,T3 |

---

### Task 1: DG progress registry module

**Goal:** A pure, process-global progress registry (dict + lock) that the batch loop updates and `/progress` reads — host-testable with no docling_graph deps.

**Files:**
- Create: `docker/docling-graph/app/_progress_registry.py`
- Test: `docker/docling-graph/tests/test_progress_registry.py`

**Acceptance Criteria:**
- [ ] `start(run_id, pass_name, total)`, `advance(run_id, pass_name)`, `touch(run_id, pass_name, phase)`, `clear(run_id, pass_name)`, `snapshot()` exported; all lock-guarded
- [ ] `advance` increments `done` and bumps `updated_at`; `touch` bumps `updated_at` (+ records `phase`) without changing `done`; `clear` removes the entry
- [ ] `snapshot()` returns `list[{run_id, pass_name, done, total, phase, started_at, updated_at, age_s}]` (age_s = now − updated_at)
- [ ] Concurrent start/advance on two keys don't interfere (test with two threads)
- [ ] Module imports with only stdlib (`time`, `threading`) — no docling_graph imports

**Verify:** `python3 -m pytest docker/docling-graph/tests/test_progress_registry.py -v` → all PASS

**Steps:**

- [ ] **Step 1: Failing test** `docker/docling-graph/tests/test_progress_registry.py`:

```python
import importlib.util, threading, time
from pathlib import Path

_MOD = Path(__file__).resolve().parent.parent / "app" / "_progress_registry.py"
_spec = importlib.util.spec_from_file_location("dg_progress_registry", _MOD)
reg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(reg)


def setup_function(_):
    reg._REGISTRY.clear()


def test_start_advance_clear():
    reg.start("run1", "radar_identity", total=3)
    snap = {(s["run_id"], s["pass_name"]): s for s in reg.snapshot()}
    assert snap[("run1", "radar_identity")]["done"] == 0
    assert snap[("run1", "radar_identity")]["total"] == 3
    reg.advance("run1", "radar_identity")
    reg.advance("run1", "radar_identity")
    assert {(s["run_id"], s["pass_name"]): s["done"] for s in reg.snapshot()}[("run1", "radar_identity")] == 2
    reg.clear("run1", "radar_identity")
    assert reg.snapshot() == []


def test_touch_bumps_updated_at_not_done():
    reg.start("r", "p", total=5)
    reg.advance("r", "p")
    before = [s for s in reg.snapshot() if s["pass_name"] == "p"][0]
    time.sleep(0.01)
    reg.touch("r", "p", phase="merging")
    after = [s for s in reg.snapshot() if s["pass_name"] == "p"][0]
    assert after["done"] == before["done"] == 1
    assert after["updated_at"] > before["updated_at"]
    assert after["phase"] == "merging"


def test_two_keys_isolated_under_threads():
    def work(rid):
        reg.start(rid, "p", total=50)
        for _ in range(50):
            reg.advance(rid, "p")
    t1 = threading.Thread(target=work, args=("a",)); t2 = threading.Thread(target=work, args=("b",))
    t1.start(); t2.start(); t1.join(); t2.join()
    done = {(s["run_id"]): s["done"] for s in reg.snapshot()}
    assert done["a"] == 50 and done["b"] == 50


def test_age_s_monotonic():
    reg.start("r", "p", total=1)
    time.sleep(0.02)
    assert [s for s in reg.snapshot()][0]["age_s"] >= 0.02
```

- [ ] **Step 2: Run → FAIL** (`No such file`/import error).

- [ ] **Step 3: Implement** `docker/docling-graph/app/_progress_registry.py`:

```python
"""Process-global per-pass progress registry for the R2 watchdog (passive).

The docling-graph batch loop calls start/advance/touch/clear; GET /progress
serves snapshot(). docling-graph is a single uvicorn process but runs up to
DOCLING_GRAPH_MAX_CONCURRENT_EXTRACTIONS passes concurrently on separate
threads, so the shared dict is lock-guarded. No docling_graph imports — pure
stdlib so it loads in standalone unit tests and inside the patched library."""
from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
# key: (run_id, pass_name) -> {done, total, phase, started_at, updated_at}
_REGISTRY: dict[tuple[str, str], dict] = {}


def start(run_id: str, pass_name: str, total: int) -> None:
    now = time.time()
    with _LOCK:
        _REGISTRY[(run_id, pass_name)] = {
            "done": 0, "total": int(total), "phase": "batches",
            "started_at": now, "updated_at": now,
        }


def advance(run_id: str, pass_name: str) -> None:
    with _LOCK:
        e = _REGISTRY.get((run_id, pass_name))
        if e is not None:
            e["done"] += 1
            e["updated_at"] = time.time()


def touch(run_id: str, pass_name: str, phase: str) -> None:
    with _LOCK:
        e = _REGISTRY.get((run_id, pass_name))
        if e is not None:
            e["phase"] = phase
            e["updated_at"] = time.time()


def clear(run_id: str, pass_name: str) -> None:
    with _LOCK:
        _REGISTRY.pop((run_id, pass_name), None)


def snapshot() -> list[dict]:
    now = time.time()
    with _LOCK:
        return [
            {"run_id": rid, "pass_name": pn, "done": e["done"], "total": e["total"],
             "phase": e["phase"], "started_at": e["started_at"],
             "updated_at": e["updated_at"], "age_s": now - e["updated_at"]}
            for (rid, pn), e in _REGISTRY.items()
        ]
```

- [ ] **Step 4: Run → PASS.** **Step 5: Commit** `feat(docling-graph): R2 progress registry module (passive, lock-guarded)`. Register the test in `scripts/run_tests.sh` `run_dg_lineage()` explicit list (next to `test_value_grounding_mirror.py`).

---

### Task 2: Thread `pipeline_run_id` through the wire + thread-local bridge

**Goal:** `pipeline_run_id` flows worker → `/extract-pass` → `run_extraction_pass` → a thread-local the patched batch loop can read, so the registry can key by `(run_id, pass_name)`.

**Files:**
- Modify: `docker/docling-graph/app/schemas.py:134` (add field)
- Modify: `docker/docling-graph/app/main.py` (thread-local + set/reset + endpoint passes run_id + `run_extraction_pass` param)
- Modify: `app/workers/pipeline.py:4342-4418` (`_build_extract_pass_request` adds `pipeline_run_id`) + `:803-811` (call site passes it)
- Test: `tests/unit/test_extract_pass_request_body.py` (or extend an existing worker-request test)

**Acceptance Criteria:**
- [ ] `ExtractPassRequest` gains `pipeline_run_id: Optional[str] = None` (extra='forbid' no longer rejects it)
- [ ] worker `_build_extract_pass_request` sets `body["pipeline_run_id"]` when the kwarg is provided; omits it when None (byte-identical body when absent)
- [ ] `run_extraction_pass` accepts `pipeline_run_id` param; sets `_pass_progress_overrides.run_id`/`.pass_name` before `run_pipeline(config)` and resets them in the existing `finally` (main.py ~1072-1081)
- [ ] endpoint passes `body.pipeline_run_id` into `run_extraction_pass`
- [ ] thread-local default is safe (`getattr(..., "run_id", None)` → None) when not set

**Verify:** `python3 -m pytest tests/unit/test_extract_pass_request_body.py docker/docling-graph/tests/test_progress_registry.py -v` → PASS; `python3 -c "from docker.docling_graph.app.schemas import ExtractPassRequest"` is not importable standalone — instead assert the field via a worker-body unit test (below)

**Steps:**

- [ ] **Step 1: Failing worker-body test** `tests/unit/test_extract_pass_request_body.py`:

```python
import pytest
pytestmark = pytest.mark.unit
from types import SimpleNamespace
from app.workers.pipeline import _build_extract_pass_request

def _pass_def():
    return SimpleNamespace(name="radar_identity", execution=None)

def test_run_id_included_when_provided():
    body = _build_extract_pass_request(
        bundle_key="air_defense_v3", pass_def=_pass_def(), doc_json={"x": 1},
        upstream_refs=None, document_id="doc-1", pipeline_run_id="run-1",
    )
    assert body["pipeline_run_id"] == "run-1"

def test_run_id_omitted_when_none():
    body = _build_extract_pass_request(
        bundle_key="air_defense_v3", pass_def=_pass_def(), doc_json={"x": 1},
        upstream_refs=None, document_id="doc-1", pipeline_run_id=None,
    )
    assert "pipeline_run_id" not in body  # byte-identical to today when absent
```

- [ ] **Step 2: Run → FAIL** (unexpected kwarg / key absent).
- [ ] **Step 3: Implement.**
  - `schemas.py:134` ExtractPassRequest, after `document_id` field: `pipeline_run_id: Optional[str] = Field(default=None, description="Pipeline run UUID for progress-registry correlation (R2).")`
  - `pipeline.py` `_build_extract_pass_request` signature add `pipeline_run_id: str | None = None`; after the base `body` dict: `if pipeline_run_id: body["pipeline_run_id"] = pipeline_run_id`. At the call site in `_execute_pass_attempt` (it already has `pipeline_run_id` in scope), pass `pipeline_run_id=pipeline_run_id`.
  - `main.py`: near the existing `_pass_orchestrator_overrides = _threading...local()` (~:143) add `_pass_progress_overrides = _threading_for_dg_orch_patch.local()`. In `run_extraction_pass` add a `pipeline_run_id: str | None = None` param; just before `context = run_pipeline(config)` (~:1037): `_pass_progress_overrides.run_id = pipeline_run_id; _pass_progress_overrides.pass_name = pass_name`; in the existing `finally` (~:1072-1081) add `_pass_progress_overrides.run_id = None; _pass_progress_overrides.pass_name = None`. At the endpoint call (~:1647) add `pipeline_run_id=body.pipeline_run_id` to the `run_extraction_pass(...)` args.
- [ ] **Step 4: Run → PASS.** **Step 5: Commit** `feat: thread pipeline_run_id worker->extract-pass->run_extraction_pass + DG progress thread-local (passive)`.

---

### Task 3: Hook the registry into the R1 batch-drain loop (patch)

**Goal:** The patched parallel batch loop calls `registry.start/advance/touch/clear` keyed by the thread-local `(run_id, pass_name)`, import-guarded so standalone lib tests still pass.

**Files:**
- Modify: `docker/docling-graph/patches/0001-orchestrator-batch-hard-timeout.patch`
- Test: extend `docker/docling-graph/tests/test_orchestrator_progress_watchdog.py`

**Acceptance Criteria:**
- [ ] Patch still applies cleanly (`patch -p1 --dry-run --fuzz=0` exit 0, no FAILED/offset) — the hard build-safety gate
- [ ] In the `parallel_workers > 1` branch: `start(run_id, pass_name, total=len(futures))` before the drain while-loop, `advance(run_id, pass_name)` on each successful `next(_drainer)`, `clear(run_id, pass_name)` in a `finally`/after the hung block; `run_id`/`pass_name` read from `_pass_progress_overrides` (thread-local) via a guarded import
- [ ] Import is `try: from app._progress_registry import start as _pr_start, advance as _pr_advance, clear as _pr_clear; from app.main import _pass_progress_overrides except Exception: <no-op stubs>` so the patch is import-safe in standalone lib tests
- [ ] Registry calls are wrapped so a registry failure never breaks extraction (best-effort)

**Verify:** `patch -p1 --dry-run` on a temp copy → exit 0; `python3 -m pytest docker/docling-graph/tests/test_orchestrator_progress_watchdog.py -v` → PASS

**Steps:**

- [ ] **Step 1:** Read the current patch + the verbatim drain block (this plan's context). Add, at the top of the `parallel_workers > 1` branch (after `_no_progress_seconds = ...`):

```python
                try:
                    from app._progress_registry import (
                        start as _pr_start, advance as _pr_advance, clear as _pr_clear,
                    )
                    from app.main import _pass_progress_overrides as _pr_ovr
                    _pr_run = getattr(_pr_ovr, "run_id", None)
                    _pr_pass = getattr(_pr_ovr, "pass_name", None)
                except Exception:  # standalone lib test / app not importable
                    _pr_start = _pr_advance = _pr_clear = None
                    _pr_run = _pr_pass = None

                def _pr_safe(fn, *a):
                    if fn is not None and _pr_run and _pr_pass:
                        try:
                            fn(_pr_run, _pr_pass, *a)
                        except Exception:
                            pass
```

  Then `_pr_safe(_pr_start, len(futures))` immediately before `_drainer = _drain_futures_with_progress_watchdog(...)`; `_pr_safe(_pr_advance)` inside the `while True` loop right after a successful `future, original_batch_idx, result = next(_drainer)`; and wrap the whole parallel block so `_pr_safe(_pr_clear)` runs in a `finally` (or place `clear` after the `if _hung_futures:` block — clear takes no extra arg, so call `_pr_clear(_pr_run, _pr_pass)` directly via a 0-extra-arg `_pr_safe`). Note: `clear`/`start`/`advance` have different arity — define `_pr_safe(fn, *a)` to forward `*a` so `_pr_safe(_pr_start, total)`, `_pr_safe(_pr_advance)`, `_pr_safe(_pr_clear)` all work.

- [ ] **Step 1b: Post-batch phase heartbeat (touch).** Immediately after the drain block (where the orchestrator proceeds to normalize/merge/clean — i.e. after the `if _hung_futures:` block, before `if not successful_results:`), add `_pr_safe(_pr_touch, "merging")` so `updated_at` keeps advancing through the non-batch phases (add `touch as _pr_touch` to the guarded import; `_pr_safe(_pr_touch, "merging")` forwards "merging" as the phase arg). This keeps the no-progress signal honest on very large docs whose merge phase is slow; for realistic docs the 2h threshold already covers it, so this is robustness, not a correctness dependency.
- [ ] **Step 2: Regenerate + verify the patch applies** against a fresh copy of the base orchestrator (the Task-1-style temp-tree `patch -p1 --dry-run --fuzz=0`); must be exit 0 with no fuzz. If hunk offsets shifted, regenerate via `diff -u`.
- [ ] **Step 3: Extend the watchdog test** to assert the registry hook: monkeypatch a fake `_progress_registry` + `_pass_progress_overrides` into `sys.modules`/the loaded module, drive fake completions, assert `advance` called N times + `clear` called once; and that with the import failing (no app module) the loop still drains (import-guard path).
- [ ] **Step 4: Run → PASS.** **Step 5: Commit** `feat(docling-graph): R2 — feed progress registry from the batch-drain loop (import-guarded; patch verified applies)`.

---

### Task 4: `GET /progress` endpoint

**Goal:** docling-graph serves the registry snapshot for the poller.

**Files:**
- Modify: `docker/docling-graph/app/main.py` (new route)
- Test: `docker/docling-graph/tests/test_progress_endpoint.py`

**Acceptance Criteria:**
- [ ] `GET /progress` returns `{"passes": snapshot()}` (200); optional `?pipeline_run_id=` filters to that run
- [ ] Read-only; no extraction impact; uses the same module-global registry as Task 1
- [ ] Returns `{"passes": []}` when nothing active

**Verify:** `python3 -m pytest docker/docling-graph/tests/test_progress_endpoint.py -v` → PASS (uses FastAPI TestClient against the route, registry seeded directly)

**Steps:**
- [ ] **Step 1:** Failing test: seed `_progress_registry` via `start/advance`, call the route function (or TestClient), assert the JSON shape + filter. If importing `app.main` standalone is heavy, test the route's pure handler factored as `def _progress_payload(run_id_filter=None)` and unit-test that.
- [ ] **Step 2:** Implement `@app.get("/progress")` in `main.py`: `from app import _progress_registry as _pr`; `snap = _pr.snapshot()`; if `pipeline_run_id` query param, filter `s["run_id"] == pipeline_run_id`; `return {"passes": snap}`.
- [ ] **Step 3: Run → PASS.** **Commit** `feat(docling-graph): GET /progress endpoint (R2, read-only)`.

---

### Task 5: Progress poller (beat task, flag-gated) + config/env

**Goal:** A flag-gated poller writes DG `/progress` into `stage_runs.metrics->'progress'`.

**Files:**
- Modify: `app/config.py` (after `reconciler_period_seconds`, ~:325): `dg_progress_poller_enabled: bool = False`, `extraction_progress_poll_seconds: int = 30`
- Modify: `.env` + `.env.example` (after `RECONCILER_PERIOD_SECONDS`): `DG_PROGRESS_POLLER_ENABLED=false`, `EXTRACTION_PROGRESS_POLL_SECONDS=30` (+ comments)
- Modify: `app/workers/pipeline.py` (new `poll_extraction_progress` task)
- Modify: `app/workers/celery_app.py:82-126` (beat entry)
- Test: `tests/unit/test_poll_extraction_progress.py`

**Acceptance Criteria:**
- [ ] Flag off (default) → task returns `{"status": "disabled"}`, makes NO HTTP call and NO DB write
- [ ] Flag on → GETs `f"{settings.docling_graph_base_url}/progress"` (timeout ~`settings.vector_router_chunk_scope_timeout_s`-style small value, fail-open: log WARNING + return on any exception) and for each entry matching a PROCESSING run's pass, merges `{done, total, phase, updated_at}` into the latest StageRun's `metrics['progress']` via copy-mutate-reassign
- [ ] Beat entry registered, mirroring `reconcile-ontology-graph-runs` style (queue `graph`, schedule `settings.extraction_progress_poll_seconds`)
- [ ] `.env`/`.env.example` parity for both new vars

**Verify:** `python3 -m pytest tests/unit/test_poll_extraction_progress.py -v` → PASS

**Steps:**
- [ ] **Step 1:** Failing tests: (a) flag-off → no `httpx`/no DB (assert mocks not called); (b) flag-on with a mocked `/progress` payload + mocked StageRun query → asserts `row.metrics['progress'] == {done,total,phase,updated_at}` written via reassignment; (c) HTTP error → fail-open (no raise, no write).
- [ ] **Step 2:** Implement the task in `pipeline.py`:

```python
@celery_app.task(bind=True, queue="graph", soft_time_limit=60,
                 name="app.workers.pipeline.poll_extraction_progress")
def poll_extraction_progress(self) -> dict:
    if not settings.dg_progress_poller_enabled:
        return {"status": "disabled"}
    import httpx
    from app.models.ingest import StageRun
    url = f"{settings.docling_graph_base_url}/progress"
    try:
        with httpx.Client(timeout=settings.vector_router_chunk_scope_timeout_s) as c:
            passes = c.get(url).json().get("passes", [])
    except Exception:
        logger.warning("poll_extraction_progress: GET %s failed", url, exc_info=True)
        return {"status": "poll_failed"}
    written = 0
    with get_sync_session() as session:
        for p in passes:
            row = (session.query(StageRun)
                   .filter(StageRun.pipeline_run_id == p["run_id"],
                           StageRun.stage_name == "derive_ontology_graph",
                           StageRun.pass_name == p["pass_name"])
                   .order_by(StageRun.attempt.desc()).first())
            if row is None:
                continue
            m = dict(row.metrics or {})
            m["progress"] = {"done": p["done"], "total": p["total"],
                             "phase": p.get("phase"), "updated_at": p["updated_at"]}
            row.metrics = m
            written += 1
        session.commit()
    return {"status": "ok", "written": written}
```

  Add config fields + `.env`/`.env.example` lines + the beat entry (mirror `reconcile-ontology-graph-runs`).
- [ ] **Step 3: Run → PASS.** **Commit** `feat(worker): R2 progress poller (flag-gated default-off) + /progress->stage_runs.metrics`.

---

### Task 6: Progress-aware reconciler (flag-gated, default OFF) + decouple stale_dispatched

**Goal:** When `RECONCILER_PROGRESS_AWARE` is on, a dispatched pass making progress is not reclaimed regardless of absolute age; when off, behavior is byte-identical to today. Decouple the absolute threshold to a setting.

**Files:**
- Modify: `app/config.py` (~after :325): `reconciler_progress_aware: bool = False`, `reconciler_no_progress_threshold_s: int = 7200`, `reconciler_stale_dispatched_s: int = 86400`
- Modify: `.env` + `.env.example`: `RECONCILER_PROGRESS_AWARE=false`, `RECONCILER_NO_PROGRESS_THRESHOLD_S=7200`, `RECONCILER_STALE_DISPATCHED_S=86400` (+ comments)
- Modify: `app/workers/pipeline.py:9441` (decouple) + `:9602-9633` (progress-aware gate) + summary key + a pure helper
- Test: `tests/unit/test_reconcile_ontology_graph_runs.py` (extend, incl. byte-identical-default)

**Acceptance Criteria:**
- [ ] `pipeline.py:9441` `stale_dispatched_threshold_s = settings.reconciler_stale_dispatched_s` (was `2 * settings.pass_soft_time_limit`)
- [ ] New pure helper `is_dispatched_phase_progressing(metrics, now_ts, no_progress_s) -> bool`: True iff `metrics.progress.updated_at` exists and `now_ts - updated_at < no_progress_s`
- [ ] In the dispatched branch, AFTER the age-stale gate (`age_s >= threshold`) and the pending-retry gate, BEFORE `reclaim_stale_phase`: when `settings.reconciler_progress_aware` is True, read the latest StageRun.metrics for `(run, pass)` (the `_has_pending_retry_for_pass` query shape) and if `is_dispatched_phase_progressing(...)` → `summary["skipped_making_progress"].append(phase_key); continue` (skip reclaim)
- [ ] `RECONCILER_PROGRESS_AWARE=false` (default) → reclaim path byte-identical to today (no metrics read, existing tests pass unchanged)
- [ ] New summary key `skipped_making_progress` mirrored into the init dict (:9443) + no-actions tuple (:9702) + log line (:9711)

**Verify:** `DATABASE_URL_SYNC=postgresql+psycopg2://eip_test:eip_test_secret@localhost:5438/eip_test python3 -m pytest tests/unit/test_reconcile_ontology_graph_runs.py -v` → PASS (incl. new progress-aware + byte-identical tests)

**Steps:**
- [ ] **Step 1: Failing tests** in `test_reconcile_ontology_graph_runs.py` (extend `_seed_stage_run` to accept a `metrics` dict → write the JSONB column): (a) `test_progress_aware_skips_when_progressing`: flag on, dispatched 3h ago, latest StageRun.metrics.progress.updated_at = 1min ago → `_PHASE_A in result["skipped_making_progress"]`, revoke NOT called; (b) `test_progress_aware_reclaims_when_no_progress`: flag on, progress.updated_at = 3h ago → reclaimed; (c) `test_progress_aware_no_metrics_falls_back_to_absolute`: flag on, no progress metrics → reclaimed (absolute); (d) `test_flag_off_byte_identical`: flag off + fresh progress metrics present → still reclaims (today's behavior, metrics ignored).
- [ ] **Step 2:** Implement the pure helper (module-level), the decouple at :9441, the flag-gated gate in the 9602–9633 window (reuse the `_has_pending_retry_for_pass` query shape to fetch the latest StageRun, read `row.metrics`), the summary-key wiring, config fields, `.env`/`.env.example` lines.
- [ ] **Step 3: Run → PASS** incl. the existing reconciler suite unchanged. **Step 4: Commit** `feat(worker): R2 progress-aware reconciler (flag default-off, byte-identical) + decouple stale_dispatched_s`.

**PHASE GATE (prepare-complete):** Tasks 1–6 committed, all unit tests green, NOTHING deployed. Present to the user before Task 7 (deploy).

---

### Task 7: Deploy heartbeat code, flags OFF, current timeouts (USER GATE)

**Goal:** When the dataset lands, deploy the R2 code with all flags OFF and the CURRENT (huge) backstop timeouts unchanged; verify `/progress` emits and the poller writes `metrics.progress` on a real run — WITHOUT changing reconciler/extraction behavior.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:** none modified — operational (rebuild + recreate + verify).

**Acceptance Criteria:**
- [ ] Dataset has landed (Engagement re-run + 3 small docs terminal) and NO run is PROCESSING before deploy
- [ ] docling-graph rebuilt (registry + `/progress` + patch hook) — patch applies in the build (build exit 0); `GET /progress` returns 200 on the running container
- [ ] worker + api force-recreated; `printenv DG_PROGRESS_POLLER_ENABLED`→false, `RECONCILER_PROGRESS_AWARE`→false in-container
- [ ] On the NEXT real run (or by flipping ONLY `DG_PROGRESS_POLLER_ENABLED=true` and recreating the worker), `ingest.stage_runs.metrics->'progress'` shows advancing `done/total` for an active pass; reconciler behavior unchanged (no `skipped_making_progress`, no unexpected reclaims)
- [ ] `RECONCILER_PROGRESS_AWARE` stays false in this task

**Verify:** `curl -s localhost:8002/progress` → `{"passes":[...]}`; `psql ... -c "SELECT pass_name, metrics->'progress' FROM ingest.stage_runs WHERE pipeline_run_id='<active>'"` shows advancing done.

**Steps:** rebuild docling-graph; recreate worker+api (flags off); confirm env + `/progress`; on a probe run, flip poller on (recreate worker) and confirm `metrics.progress` advances; leave progress-aware OFF. Capture outputs.

```json:metadata
{"userGate": true, "tags": ["user-gate"], "files": [], "verifyCommand": "curl -s localhost:8002/progress", "acceptanceCriteria": ["dataset landed, nothing PROCESSING", "docling-graph rebuilt, /progress 200", "worker+api recreated, flags off in-container", "metrics.progress advances on a real run with poller on", "reconciler_progress_aware stays false"]}
```

---

### Task 8: Flip progress-aware on, then reduce backstop timeouts (USER GATE)

**Goal:** After the heartbeat is verified, enable the progress-aware reconciler, then reduce the absolute backstop timeouts — in that order, so the tighter absolute timeout is never active without the progress guard.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:** `.env` + `.env.example` (timeout reductions, applied LAST); docling-graph `config_builder.py`/compose for `DOCLING_GRAPH_LLM_TIMEOUT` (no Settings field exists — env/compose only).

**Acceptance Criteria:**
- [ ] `RECONCILER_PROGRESS_AWARE=true` set + workers recreated; a deliberately-stuck pass (or a fixture run) is reclaimed via `skipped_making_progress`→no, reclaim→yes only when progress truly stalls; a progressing pass is NOT reclaimed
- [ ] THEN, and only then, reduce: `DOCLING_GRAPH_TIMEOUT` 720000→64800, `PASS_SOFT_TIME_LIMIT` 288000→72000, `RECONCILER_STALE_DISPATCHED_S` 86400 (already), `DOCLING_GRAPH_LLM_TIMEOUT` 720000→64800 (env/compose), in BOTH `.env` and `.env.example`; recreate; confirm in-container
- [ ] No legitimately-long in-flight pass is killed by the new backstops (verify against the largest doc's observed per-pass time + margin)
- [ ] `docs/operational/production-reliability-2026-06.md` §R2 updated to "implemented + deployed"

**Verify:** in-container `printenv` shows the new values + `RECONCILER_PROGRESS_AWARE=true`; reconciler log shows `skipped_making_progress` for a progressing pass.

**Steps:** flip `RECONCILER_PROGRESS_AWARE=true` (recreate worker) → verify progressing-pass-not-reclaimed on a live/fixture run → reduce the four timeout values in both env files (+ DG compose/config_builder for LLM_TIMEOUT) → recreate → confirm → update the R2 doc.

```json:metadata
{"userGate": true, "tags": ["user-gate"], "files": [".env.example", "docs/operational/production-reliability-2026-06.md"], "verifyCommand": "docker exec eip-mmdpp-worker-graph-1 printenv RECONCILER_PROGRESS_AWARE DOCLING_GRAPH_TIMEOUT PASS_SOFT_TIME_LIMIT", "acceptanceCriteria": ["progress-aware on; progressing pass NOT reclaimed, stalled pass reclaimed", "timeouts reduced in both env files only AFTER progress-aware verified", "no legit pass killed by new backstops", "R2 doc updated to implemented+deployed"], "requireEvidenceTokens": [["progress-aware-off","before"], ["progress-aware-on","after"]]}
```
