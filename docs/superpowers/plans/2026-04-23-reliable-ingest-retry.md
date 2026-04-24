# Reliable Ingest Retry Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pipeline failures loud (exception-raising) at the task level so Celery's built-in retry machinery handles restarts automatically; keep the periodic sweeper as a narrow last-resort safety net for the residual silent-death case.

**Architecture:** Three layers, ordered by responsibility:

1. **Realistic per-stage timeouts.** Every `@celery_app.task` gets a `soft_time_limit` and `time_limit` sized to the observed P99 of that stage + margin. A task that hangs past its soft limit raises `SoftTimeLimitExceeded`, which the existing `try/except` branches convert into `self.retry(exc=...)`. No more tasks running forever silently.
2. **Universal `guard_stage_run` + silent-swallow audit.** The decorator introduced in the 2026-04-23 stale-run plan is applied to every pipeline task (today it's only on `prepare_document`). Every `except Exception:` path that currently logs-and-continues is audited and converted to either (a) explicit `raise` so Celery retries, or (b) documented, intentional swallow with a code comment justifying it. Failures propagate exceptions; Celery retries. Stage_runs never silently stay RUNNING.
3. **Sweeper becomes an auto-restart safety net.** `_sweep_stale_runs` is rewritten: instead of marking rows FAILED and stopping there, it (a) marks the `pipeline_run` FAILED, (b) bumps `ingest.documents.retry_count`, (c) if under `settings.max_doc_retry_count`, dispatches `start_ingest_pipeline(doc_id)` to create a fresh run, (d) if at the cap, leaves the doc at `status='FAILED'` permanently for operator triage. The sweeper threshold is raised to `max(*_time_limit) + 30min` so it only ever fires after Celery's own timeout machinery should have already killed and retried the task. The sweeper catches the case where nothing else fired — the narrow silent-death residual.

**Tech Stack:** Python 3.11, Celery 5+ (prefork + Redis broker/beat), SQLAlchemy (sync session), Alembic, Pydantic-Settings, pytest.

**Relationship to prior plan (2026-04-23-stale-stage-run-sweeper.md):**
The prior plan shipped Tasks 0–5 (commits `61c0cff..9942acd`) — settings scaffold, `_sweep_stale_runs`, beat entry, `guard_stage_run`, applied to `prepare_document`, lifecycle audit comment. This plan *extends* that work: it keeps the committed decorator + sweeper skeleton but replaces the sweeper's body (passive marking → active restart), applies the decorator universally, audits silent swallows, and tunes timeouts to realistic values. The prior plan's threshold constant is reused; its value changes via env.

**Context from 2026-04-23 ingest (the motivating incident):**
A 21-doc batch ingest ran the prior sweeper live. One translate task (`chinese_research_paper.pdf`) was genuinely slow (100 non-English elements across 8 batches, ~30 min of real work against a loaded remote Ollama). The sweeper with its 15-min `started_at` threshold classified it as dead and marked its `pipeline_run` FAILED, **while the worker continued producing real results into an orphaned run that could no longer advance the chain.** This proved threshold-on-start_time cannot distinguish slow-alive from truly-dead. The correct remedy is: (a) let realistic per-stage time limits kill genuinely-hung tasks via exceptions, (b) let Celery retry on those exceptions, (c) only fall back to the sweeper when that entire chain fails — and when it does, the sweeper needs to actually restart work, not just surface it.

**Out of scope (explicitly deferred):**
- Heartbeat-based detection (`last_heartbeat_at` column + background updater). No longer needed once the three layers above work. Keep as a fallback design note only; do NOT implement in this plan.
- Per-stage sweeper thresholds. Single global threshold sized to the slowest stage is sufficient once timeouts enforce per-stage limits.
- UI / API surface for operator-driven retry of a permanently-failed doc. Operators can reingest by re-uploading; a proper UI retry endpoint is a separate feature.

---

## File Structure

**Create:**
- `alembic/versions/<ts>_add_retry_count_to_documents.py` — migration adding `retry_count INTEGER NOT NULL DEFAULT 0` to `ingest.documents`
- `tests/pipeline/test_sweeper_autorestart.py` — unit tests for restart + retry-cap behavior in `_sweep_stale_runs`
- `tests/pipeline/test_guard_stage_run_universal.py` — assertion tests proving every `@celery_app.task` in `pipeline.py` is wrapped

**Modify:**
- `.env` + `env.example` — revised `*_TIME_LIMIT` + `*_SOFT_TIME_LIMIT` to realistic values; raise `STALE_STAGE_RUN_THRESHOLD_SECONDS`; add `MAX_DOC_RETRY_COUNT`
- `app/config.py` — add `max_doc_retry_count: int = 3` default; revise the per-stage time-limit defaults
- `app/models/ingest.py` — add `retry_count: Mapped[int]` column on `Document`
- `app/workers/pipeline.py` — apply `@guard_stage_run("<stage_name>")` to every remaining pipeline task; audit and fix silent-swallow patterns; rewrite `_sweep_stale_runs` body to auto-restart with retry cap

Each file has one clear responsibility. The decorator lives next to the stage-update helper it calls (already in place). The sweeper body stays in the same function the prior plan created; only its semantics change.

---

## Chunk 1: Make failures loud

### Task 0: Set realistic timeouts across every pipeline stage

**Goal:** Every task has a `soft_time_limit` / `time_limit` that reflects its observed P99 plus a ~30% buffer. Above those, `SoftTimeLimitExceeded` raises and Celery retries.

**Rationale:** During 2026-04-23 we 10×'d every timeout to protect a single run, which pushed TRANSLATION_TIME_LIMIT to ≈125 days — effectively disabling the time-limit layer entirely. Those values were panic-ballast for one run, not the sustainable configuration. But the pre-panic values in this repo were *also* intentionally high because certain stages legitimately take **hours** on big docs (graph extraction with many passes, picture descriptions across 100+ images, translate across hundreds of non-English elements). The correct shape is: generous enough to never cut off legitimate work, but finite enough that `SoftTimeLimitExceeded` can still fire on genuinely-hung tasks and the sweeper's threshold isn't effectively infinite.

**Calibration principle:** Set each `*_TIME_LIMIT` at roughly 2× the slowest legitimate run you've observed for that stage. If you haven't observed an extreme case, err high — an overly-long timeout costs a few wasted hours of GPU on a genuinely-stuck task (which is rare); an overly-short timeout costs a *dead* run on legitimate work (which is what today's chinese_research_paper sweep was).

**Files:**
- Modify: `.env`
- Modify: `env.example`
- Modify: `app/config.py` (defaults only; env overrides remain)

Proposed starting values (calibrated for this corpus's observed-worst + generous headroom; **operator should tune per workload**):

| Stage | `*_SOFT_TIME_LIMIT` | `*_TIME_LIMIT` | Note |
|---|---:|---:|---|
| prepare_document | 3600 (1h) | 4500 (75m) | Docling + OCR on large PDFs; 67s observed but safe ceiling |
| detect_and_translate | 7200 (2h) | 8100 (135m) | 100+ non-English elements × minutes each; 25m observed + margin |
| derive_document_metadata | 1800 (30m) | 2400 (40m) | 4 parallel LLM calls; 2m observed. **Adds new settings fields** `doc_analysis_soft_time_limit` / `doc_analysis_time_limit` — currently the decorator derives limits from `settings.doc_analysis_timeout + 60/+120` at pipeline.py:3170, which cannot produce these numbers without the timeout being raised. This plan replaces that derivation with direct settings fields. |
| derive_picture_descriptions | 18000 (5h) | 21600 (6h) | N images × 30-60s each VLM; scales w/ image count |
| derive_text_chunks_and_embeddings (EMBED_*) | 3600 (1h) | 4500 (75m) | Batched embeds; fast but can stall on large docs |
| derive_image_embeddings (EMBED_*) | — | — | Shares EMBED_* budget |
| derive_ontology_graph (GRAPH_*) | 21600 (6h) | 25200 (7h) | Multi-pass LLM × chunks; the biggest sink |
| derive_structure_links (GRAPH_*) | — | — | Shares GRAPH_* budget; deterministic, fast |
| derive_canonicalization (GRAPH_*) | — | — | Shares GRAPH_* budget |
| finalize_document | 600 (10m) | 900 (15m) | Small ArcadeDB writes |

`STALE_STAGE_RUN_THRESHOLD_SECONDS` = max(time_limit values above) + 1800 = 25200 + 1800 = **27000s (7.5h)** — the sweeper only ever fires after Celery's timeout should have already killed + retried. Accept that silent-death detection latency is bounded by the longest legitimate stage (graph extraction), not 15 min. This is the direct consequence of the hours-long stages.

If you observe legitimate runs exceeding these starting values, raise them in `.env` and bump `STALE_STAGE_RUN_THRESHOLD_SECONDS` to match. Do NOT leave stale values that are smaller than your real workload.

`MAX_DOC_RETRY_COUNT` = 3 — sweeper-triggered chain retries capped at 3; beyond that the doc is permanently FAILED.

`TRANSLATION_TIMEOUT` (HTTP client timeout inside translate code, not Celery's) should be larger than any single batch call to Ollama: 1800s (30m) accommodates batch-N-elements on a loaded model.

`DOCLING_TIMEOUT_SECONDS`, `DOCLING_GRAPH_TIMEOUT`, `DOCLING_GRAPH_LLM_TIMEOUT`, `PICTURE_DESCRIPTION_TIMEOUT` — HTTP client timeouts (not Celery). Set slightly below the corresponding Celery soft time limit so the HTTP layer times out first (producing a clean `httpx.ReadTimeout`, which task bodies convert to `self.retry(exc=...)`).

- [ ] **Step 1: Compile the full revised env block**

Produce this exact block for use in `.env` and `env.example`:

```
# Celery per-stage soft/hard time limits (seconds). Soft raises
# SoftTimeLimitExceeded so the task can clean up + retry; hard SIGKILLs the
# child process. Sized to accommodate the SLOWEST legitimate run of this
# stage seen on this corpus + generous margin. Prefer overly-long here —
# cutting off legit work is worse than a rare zombie task waiting for sweep.
# Calibrate per your workload; raise in .env if you see real work exceeding.
PREPARE_SOFT_TIME_LIMIT=3600
PREPARE_TIME_LIMIT=4500
TRANSLATION_SOFT_TIME_LIMIT=7200
TRANSLATION_TIME_LIMIT=8100
DOC_ANALYSIS_SOFT_TIME_LIMIT=1800
DOC_ANALYSIS_TIME_LIMIT=2400
PICTURE_DESC_SOFT_TIME_LIMIT=18000
PICTURE_DESC_TIME_LIMIT=21600
EMBED_SOFT_TIME_LIMIT=3600
EMBED_TIME_LIMIT=4500
GRAPH_SOFT_TIME_LIMIT=21600
GRAPH_TIME_LIMIT=25200
FINALIZE_SOFT_TIME_LIMIT=600
FINALIZE_TIME_LIMIT=900

# HTTP client timeouts (seconds) on calls to docling / docling-graph / ollama.
# Set SLIGHTLY below the corresponding Celery soft limit so HTTP-level timeouts
# fire first (producing clean httpx.ReadTimeout, not SoftTimeLimitExceeded
# which doesn't tell you what was actually slow).
DOCLING_TIMEOUT_SECONDS=3300
DOCLING_GRAPH_TIMEOUT=21000
DOCLING_GRAPH_LLM_TIMEOUT=21000
PICTURE_DESCRIPTION_TIMEOUT=17400
TRANSLATION_TIMEOUT=1800
DOC_ANALYSIS_TIMEOUT=1500

# Docling internal (non-Celery) knobs
DOCLING_HEALTH_TIMEOUT=15.0
DOCLING_LOCK_TIMEOUT=3600
PREPARE_SINGLEFLIGHT_TIMEOUT=4200

# Celery Redis visibility timeout (seconds). Must be > longest Celery time
# limit so in-flight long tasks aren't redelivered to another worker while
# still running. max(TIME_LIMIT) = 25200, so 28800 (8h) is safe.
CELERY_VISIBILITY_TIMEOUT=28800

# Sweeper threshold (seconds). Must be > max(*_TIME_LIMIT) + buffer so the
# sweeper only fires after Celery's own time-limit machinery should have
# already killed + retried the task. max TIME_LIMIT = 25200 -> 27000.
# Silent-death detection latency is bounded by the longest legit stage
# (graph extraction @ 7h). This is the direct consequence of hours-long
# legitimate work; there is no way around it without heartbeats.
STALE_STAGE_RUN_THRESHOLD_SECONDS=27000

# Max times the sweeper will restart a document before giving up and marking
# it permanently FAILED. 3 catches transient issues without looping on
# genuinely-broken docs.
MAX_DOC_RETRY_COUNT=3
```

- [ ] **Step 2: Apply to `.env`**

Replace every existing `*_TIME_LIMIT`, `*_SOFT_TIME_LIMIT`, HTTP timeout, and the sweeper/visibility constants with the block above. Use a single pass so the file ends up consistent; do not leave stale 10× values interleaved with new ones.

- [ ] **Step 3: Apply to `env.example`**

Same block, same replacement policy. `env.example` is the source of truth new contributors will start from — keep it readable and matching `.env` line-for-line on these vars.

- [ ] **Step 4: Update Pydantic defaults + add new fields in `app/config.py`**

For every field whose default is now wildly off the env value, update the default to match the env block above. This matters because missing env vars fall back to the default. Add these new fields:

```python
    # Celery time limits for derive_document_metadata. Previously derived from
    # doc_analysis_timeout + 60/+120 inline in pipeline.py; made explicit here
    # so the env block can control them directly.
    doc_analysis_soft_time_limit: int = 1800
    doc_analysis_time_limit: int = 2400

    # Max times the stale-run sweeper will restart a document before marking
    # it permanently FAILED. See docs/superpowers/plans/2026-04-23-reliable-ingest-retry.md.
    max_doc_retry_count: int = 3
```

- [ ] **Step 4b: Rewire `derive_document_metadata` decorator to use the new fields**

In `app/workers/pipeline.py` around line 3170, replace:

```python
soft_time_limit=settings.doc_analysis_timeout + 60,
time_limit=settings.doc_analysis_timeout + 120,
```

with:

```python
soft_time_limit=settings.doc_analysis_soft_time_limit,
time_limit=settings.doc_analysis_time_limit,
```

`doc_analysis_timeout` stays as the HTTP client timeout for individual LLM calls within the task body; it just no longer doubles as the Celery limit.

- [ ] **Step 5: Verify env loads cleanly**

```bash
.venv/bin/python -c "from app.config import get_settings; s = get_settings(); print('picture_desc_time_limit:', s.picture_desc_time_limit); print('stale_stage_run_threshold_seconds:', s.stale_stage_run_threshold_seconds); print('max_doc_retry_count:', s.max_doc_retry_count)"
```

Expected: `21600`, `27000`, `3`.

- [ ] **Step 6: Commit**

```bash
git add .env env.example app/config.py
git commit -m "feat(config): realistic per-stage timeouts + MAX_DOC_RETRY_COUNT"
```

Note: `.env` may be gitignored in this repo. If so, stage only `env.example` and `app/config.py`; document the `.env` change in the commit body.

---

### Task 1: Apply `guard_stage_run` to every pipeline task

**Goal:** Every `@celery_app.task` in `app/workers/pipeline.py` is wrapped with `@guard_stage_run("<stage_name>")` immediately below its `@celery_app.task` decorator. Today only `prepare_document` is wrapped.

**Rationale:** The decorator's contract is: if the task raises anything other than `CeleryRetry`/`SoftTimeLimitExceeded`, write `stage_runs.status='FAILED'` with the exception repr, log a full traceback, and re-raise so Celery's retry machinery sees it. Applied universally, this closes the "exception in task body left stage_run stuck at RUNNING" failure class for every stage, not just `prepare_document`.

**Files:**
- Modify: `app/workers/pipeline.py` — add one `@guard_stage_run("<stage>")` line above each pipeline `@celery_app.task` `def` line
- Test: `tests/pipeline/test_guard_stage_run_universal.py` — programmatic assertion that every pipeline task is wrapped

**CANONICAL STAGE-NAME MAPPING — the decorator argument MUST match the stage string the task body already writes into `ingest.stage_runs`, not the function name.** For most tasks they happen to match; one does not. Using the wrong string creates orphan `stage_runs` rows under a name no other part of the system recognizes.

| Task function | Canonical `stage_name` argument | Note |
|---|---|---|
| `prepare_document` | `"prepare_document"` | Already wrapped — verify |
| `detect_and_translate` | `"detect_and_translate"` | |
| `derive_document_metadata` | `"derive_document_metadata"` | |
| `derive_picture_descriptions` | `"derive_picture_descriptions"` | |
| `purge_document_derivations` | `"purge_document_derivations"` | |
| `derive_text_chunks_and_embeddings` | **`"derive_text_embeddings"`** | **MISMATCH — function name ≠ stage name.** Task body writes `_update_stage_run(db, run_id, "derive_text_embeddings", ...)` (see pipeline.py around line 3842). `finalize_document` reads this canonical name (see line ~5488). Using `"derive_text_chunks_and_embeddings"` creates split bookkeeping. |
| `derive_image_embeddings` | `"derive_image_embeddings"` | |
| `derive_document_anchors` | `"derive_document_anchors"` | |
| `derive_ontology_graph` | `"derive_ontology_graph"` | Task writes multiple per-pass `stage_runs` via `_write_stage_run`; the guard's write on uncaught exception is a new summary row under this name, which is correct. |
| `derive_structure_links` | `"derive_structure_links"` | |
| `collect_derivations` | `"collect_derivations"` | Task does not natively write a stage_run; guard's write on exception is the first row. Acceptable. |
| `derive_canonicalization` | `"derive_canonicalization"` | |
| `finalize_document` | `"finalize_document"` | |

Tasks NOT to wrap (not pipeline-doc tasks; may have different error semantics): `_chord_error_handler`, `scan_watch_directories`, `periodic_stale_run_sweep`, `index_trusted_submission`, `run_community_detection_task`.

**Verify line numbers before editing** — the prior stale-run plan added code above these tasks, shifting their absolute line numbers. Use `grep -nE "^def (prepare_document|derive_|detect_|purge_|collect_|finalize_)" app/workers/pipeline.py` to find the current positions.

- [ ] **Step 0: Expose `stage_name` as an attribute on the `guard_stage_run` wrapper**

The coverage test (Step 1 below) needs to assert which stage_name argument each task was decorated with. The current decorator (committed in `79e2eb3`) captures stage_name in a closure, invisible to introspection. Add one line to the decorator so the test can check it:

In `app/workers/pipeline.py`, inside `guard_stage_run`, right before `return wrapper`, add:

```python
        wrapper.stage_name = stage_name  # surfaced for test introspection
        return wrapper
```

This is a ~2-line change to the existing decorator; no behavioral change to the task's runtime.

- [ ] **Step 1: Write a failing coverage test**

Create `tests/pipeline/test_guard_stage_run_universal.py`:

```python
"""Assert every pipeline stage task is wrapped by guard_stage_run.

The guard's contract is that any uncaught exception marks the stage_run
FAILED and re-raises so Celery retries. Applied to prepare_document only
in the prior plan; this test locks in universal coverage.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

# Pipeline tasks that must be wrapped. If you add a new pipeline task,
# add it here AND add the decorator application in the same PR.
# Each entry is (task_function_name, expected_stage_name_passed_to_guard_stage_run).
# Function name and stage_name intentionally differ for one task — see plan.
PIPELINE_TASKS = [
    ("prepare_document", "prepare_document"),
    ("derive_document_metadata", "derive_document_metadata"),
    ("detect_and_translate", "detect_and_translate"),
    ("derive_picture_descriptions", "derive_picture_descriptions"),
    ("purge_document_derivations", "purge_document_derivations"),
    ("derive_text_chunks_and_embeddings", "derive_text_embeddings"),
    ("derive_image_embeddings", "derive_image_embeddings"),
    ("derive_document_anchors", "derive_document_anchors"),
    ("derive_ontology_graph", "derive_ontology_graph"),
    ("derive_structure_links", "derive_structure_links"),
    ("collect_derivations", "collect_derivations"),
    ("derive_canonicalization", "derive_canonicalization"),
    ("finalize_document", "finalize_document"),
]


@pytest.mark.parametrize("task_name,expected_stage", PIPELINE_TASKS)
def test_pipeline_task_has_guard_stage_run(task_name, expected_stage):
    """Every pipeline-doc task is wrapped by guard_stage_run with the CANONICAL stage name.

    The guard stashes its stage_name argument on the wrapper as an attribute so
    that wiring can be verified without running the task. This catches the
    known derive_text_chunks_and_embeddings mismatch case: the function name
    is derive_text_chunks_and_embeddings but the correct stage_name is
    derive_text_embeddings.
    """
    import app.workers.pipeline as pipeline
    task = getattr(pipeline, task_name)
    assert hasattr(task.run, "__wrapped__"), (
        f"{task_name} is not wrapped by guard_stage_run — its failures will "
        "leave stage_run rows at RUNNING if an exception escapes the body."
    )
    assert getattr(task.run, "stage_name", None) == expected_stage, (
        f"{task_name} is wrapped with stage_name="
        f"{getattr(task.run, 'stage_name', None)!r} but should use "
        f"{expected_stage!r} to match the canonical _update_stage_run calls."
    )
```

- [ ] **Step 2: Run the test, confirm most cases fail**

```bash
.venv/bin/pytest tests/pipeline/test_guard_stage_run_universal.py -v
```

Expected: 1 pass (`prepare_document`), 12 fails.

- [ ] **Step 3: Apply the decorator to each task**

For each task in the list above, find the `@celery_app.task(...)` line and insert `@guard_stage_run("<stage_name>")` on the line immediately below it (above `def <task_name>(...)`). **The `<stage_name>` argument MUST come from the second column of the CANONICAL STAGE-NAME MAPPING table above, NOT the function name.** These match for most tasks but deliberately differ for `derive_text_chunks_and_embeddings` (function name) → `"derive_text_embeddings"` (stage_name). Using the function name there creates split stage_runs bookkeeping that `finalize_document` can't read.

Exact example for `derive_document_metadata` (function name == stage name):
```python
@celery_app.task(
    bind=True, max_retries=2, default_retry_delay=60,
    soft_time_limit=settings.doc_analysis_soft_time_limit,
    time_limit=settings.doc_analysis_time_limit,
)
@guard_stage_run("derive_document_metadata")
def derive_document_metadata(self, document_id: str, run_id: str | None = None) -> dict:
```

Do all 12 in one pass; the test will tell you which you missed.

- [ ] **Step 4: Run the test, confirm all pass**

```bash
.venv/bin/pytest tests/pipeline/test_guard_stage_run_universal.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Run the existing pipeline suite to confirm no regressions**

```bash
.venv/bin/pytest tests/pipeline/ tests/unit/test_ingest_pipeline_coverage.py -v
```

Any pre-existing failures from before this plan stay as-is (document them in the commit body). New failures MUST be investigated before committing — the decorator changes task signatures via `functools.wraps`, which occasionally trips tests that introspect tasks.

- [ ] **Step 6: Commit**

```bash
git add tests/pipeline/test_guard_stage_run_universal.py app/workers/pipeline.py
git commit -m "feat(pipeline): apply guard_stage_run to every pipeline task"
```

---

### Task 2: Audit and fix silent-swallow patterns in `pipeline.py`

**Goal:** Every `except Exception` AND every `except SoftTimeLimitExceeded` that currently logs-and-returns without re-raising is either converted to `self.retry(exc=...)` (so Celery's retry layer activates) or has a code comment documenting why swallowing is correct. Failures propagate exceptions.

**Rationale:** Two classes of silent-failure to hunt:
1. `grep -c "except Exception" app/workers/pipeline.py` returns 55. Many are best-effort (MinIO cache, optional markdown regen); some are real bugs — e.g., HTTP calls wrapped in `try/except/logger.warning/pass`.
2. `except SoftTimeLimitExceeded` currently appears in several tasks with the pattern `log → mark stage_run FAILED → return {"status": "timeout"}`. Observed at `detect_and_translate` (pipeline.py ~3485) and `derive_image_embeddings` (~4412). **These do not retry.** The doc is marked PARTIAL_COMPLETE and the pipeline moves on. That violates the plan's acceptance criterion 5 — a soft-timeout should trigger Celery's built-in retry, not silently degrade the doc.

**Files:**
- Modify: `app/workers/pipeline.py` (throughout)
- No new test file; existing `test_guard_stage_run_universal.py` + integration validation in Task 6 covers the behavior change.

**Audit methodology:**

Do **two passes**. Pass 1 audits `except Exception`; Pass 2 audits `except SoftTimeLimitExceeded`. For each location, classify into one of:

| Category | Action |
|---|---|
| **A. Best-effort side work** (e.g., MinIO cache upload, optional markdown regen) | Keep swallow, add one-line comment: `# best-effort: main work already succeeded` |
| **B. Silent failure of the main stage work** (e.g., ArcadeDB write, embedding call) | Convert to `raise` or `self.retry(exc=e)` |
| **C. Cleanup in `finally`** (DB rollback, lock release) | Keep swallow, no comment needed — `finally` intent is obvious |
| **D. Swallow that forwards to a retry counter but doesn't actually retry** | Convert to `self.retry(exc=e)` so Celery owns retry state |
| **E. `SoftTimeLimitExceeded` caught + return** (observed in detect_and_translate + derive_image_embeddings) | Convert to: mark stage_run FAILED for visibility, then `raise self.retry(exc=exc)` so Celery retries with its configured delay. Do NOT just return. |

Example transform for Category E (detect_and_translate ~3485):

```python
# before
except SoftTimeLimitExceeded:
    logger.warning("detect_and_translate: soft time limit for %s — marking FAILED", document_id)
    if run_id:
        try:
            _update_stage_run(db, run_id, "detect_and_translate", "FAILED",
                              attempt=self.request.retries + 1, error="soft time limit exceeded")
            db.commit()
        except Exception:
            pass
    _update_document_status(document_id, STATUS_PARTIAL_COMPLETE, stage="detect_and_translate")
    return {"stage": "detect_and_translate", "status": "timeout"}

# after
except SoftTimeLimitExceeded as exc:
    logger.warning("detect_and_translate: soft time limit for %s — retrying via Celery", document_id)
    if run_id:
        try:
            _update_stage_run(db, run_id, "detect_and_translate", "FAILED",
                              attempt=self.request.retries + 1, error="soft time limit exceeded")
            db.commit()
        except Exception:
            logger.exception("stage_run FAILED write also failed")
    raise self.retry(exc=exc)  # Celery re-dispatches with default_retry_delay
```

- [ ] **Step 1: Enumerate every except-Exception location**

```bash
grep -n "except Exception" app/workers/pipeline.py > /tmp/swallow-audit.txt
wc -l /tmp/swallow-audit.txt
```

Expected: ~55 lines. Work from the top of the file downward so commits stay small.

- [ ] **Step 2: Classify and fix, one function at a time**

For each pipeline task function (use the list from Task 1):
1. Read every `except` block in the function body.
2. Classify per the table above.
3. If Category B: convert. Typical transform:
   ```python
   # before
   try:
       result = httpx.post(url, ...).raise_for_status()
   except Exception as e:
       logger.warning("upload failed: %s", e)
       return  # silent!

   # after
   try:
       result = httpx.post(url, ...).raise_for_status()
   except (httpx.ReadTimeout, httpx.ConnectError) as e:
       raise self.retry(exc=e)  # Celery handles backoff + retry
   ```
4. If Category A: add the justifying comment, no code change.
5. If Category D: convert to explicit `self.retry(exc=e)` pattern.

Commit after each function so the diff is small and reviewable. Commit message format:
```
fix(pipeline.<task_name>): raise on X instead of swallowing
```

- [ ] **Step 3: Run the pipeline test suite after each function**

```bash
.venv/bin/pytest tests/pipeline/ tests/unit/test_ingest_pipeline_coverage.py -v
```

If a previously-passing test breaks, the behavior change matters — either update the test to the new semantics or revert the audit change.

- [ ] **Step 4: Final sweep — grep for remaining bare `except:` (no class) and `except BaseException`**

```bash
grep -nE "except:$|except BaseException" app/workers/pipeline.py
```

Expected: zero matches. `except:` without a class catches `KeyboardInterrupt` / `SystemExit` / `SoftTimeLimitExceeded` and is always wrong for pipeline code. If any found, fix.

- [ ] **Step 5: Commit final audit note**

After all per-function commits, add a summary commit:

```bash
git commit --allow-empty -m "chore(pipeline): silent-swallow audit complete

Audited 55 except-Exception locations. Classified and either:
- documented as intentional best-effort (category A), or
- converted to raise / self.retry (categories B/D).
Remaining swallows are in finally blocks (C) where intent is obvious."
```

---

## Chunk 2: Sweeper as restart-capable safety net

### Task 3: Add `retry_count` column to `ingest.documents`

**Files:**
- Create: `alembic/versions/<timestamp>_add_retry_count_to_documents.py`
- Modify: `app/models/ingest.py` (add `retry_count` field to `Document`)

- [ ] **Step 1: Generate the alembic migration file**

```bash
docker compose exec api alembic revision -m "add_retry_count_to_documents"
```

Or, if the repo uses `.venv/bin/alembic` from the host:
```bash
.venv/bin/alembic revision -m "add_retry_count_to_documents"
```

Note the generated filename under `alembic/versions/`.

- [ ] **Step 2: Fill in the migration body**

Edit the new migration file. The `upgrade()` and `downgrade()` should read:

```python
from alembic import op
import sqlalchemy as sa

# revision = "<auto-filled>"
# down_revision = "<auto-filled — whatever the current head is>"

def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        schema="ingest",
    )
    # Drop the server_default once backfilled — future inserts use the ORM default.
    op.alter_column("documents", "retry_count", server_default=None, schema="ingest")


def downgrade() -> None:
    op.drop_column("documents", "retry_count", schema="ingest")
```

- [ ] **Step 3: Add the column to the SQLAlchemy model**

In `app/models/ingest.py`, locate the `Document` class and add:

```python
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
        doc="Times the sweeper has restarted this doc. Capped at settings.max_doc_retry_count.",
    )
```

Place it near other counter-like columns (e.g., where `pipeline_status` lives) for discoverability.

- [ ] **Step 4: Apply the migration**

```bash
docker compose exec api alembic upgrade head
```

Or equivalent invocation. Verify:

```bash
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -c "\d ingest.documents" | grep retry_count
```

Expected: `retry_count | integer | not null | 0`.

- [ ] **Step 5: Commit**

```bash
git add alembic/versions/ app/models/ingest.py
git commit -m "feat(schema): add ingest.documents.retry_count for sweeper auto-restart cap"
```

---

### Task 4: Rewrite `_sweep_stale_runs` to auto-restart with retry cap

**Goal:** When the sweeper finds a stage_run stuck past threshold, it (a) marks the stage_run + pipeline_run FAILED, (b) bumps `documents.retry_count`, (c) if under cap, resets the doc to `PENDING` and dispatches `start_ingest_pipeline(doc_id)`, (d) if at cap, leaves the doc at `FAILED` permanently.

**Files:**
- Modify: `app/workers/pipeline.py::_sweep_stale_runs` — replace the body
- Test: `tests/pipeline/test_sweeper_autorestart.py` — new file

**Concurrency note:** The sweeper only acts on rows where both (a) stage_run status is still `RUNNING` and (b) pipeline_run status is still `PROCESSING`. This makes it idempotent across multiple concurrent beat-scheduled invocations — the first sweep flips pipeline_run to `FAILED`, subsequent sweeps find nothing to act on for that row.

**Transactional ordering (critical):** `start_ingest_pipeline` performs its own duplicate-dispatch check via `SELECT ... FOR UPDATE WHERE status='PROCESSING'` (pipeline.py ~line 1736). It runs in its own DB session, so **uncommitted writes from the sweeper are invisible to it.** The sweeper MUST commit its `pipeline_run PROCESSING → FAILED` transition BEFORE calling `start_ingest_pipeline`, otherwise the dispatch guard sees the pre-sweep `PROCESSING` row and refuses to dispatch. This means the sweeper cannot wrap the full "mark + dispatch + reset" in a single transaction — the failure → dispatch handoff is split across two DB transactions, and a dispatch failure must be handled by a compensating transaction.

- [ ] **Step 1: Write failing tests**

Create `tests/pipeline/test_sweeper_autorestart.py`:

```python
"""Tests for the auto-restart behavior of _sweep_stale_runs.

Design contract: on sweep of a stale row, mark stage_run + pipeline_run
FAILED, bump documents.retry_count, and if under settings.max_doc_retry_count,
re-dispatch start_ingest_pipeline(doc_id) with a fresh pipeline_run.
If at cap, mark the document pipeline_status='FAILED' permanently.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestSweeperAutorestart:
    """Mock SQL sequence must match _sweep_stale_runs() body.

    Per-row execute calls in the main db session:
      1. SELECT fetchall -> [(sr_id, pr_id, doc_id, stage_name), ...]   (once, up front)
      2. UPDATE stage_runs                                              (per row)
      3. UPDATE pipeline_runs, read .rowcount                           (per row)
      4. UPDATE documents ... RETURNING retry_count, read .scalar()     (per row; skipped if step 3 rowcount=0)
      5. (If over cap:) UPDATE documents (mark FAILED)                  (per row over cap)

    After the main-session commit, dispatch loop:
      6. start_ingest_pipeline(doc_id) (patched)
      7. (If dispatch raises) _get_db() returns a fresh MagicMock used for the
         compensation transaction — another UPDATE + commit/close.
    """

    def _main_db(self, rows, rowcounts, new_retry_counts):
        """Build the mock db used by the primary transaction.

        `rowcounts` = list[int] aligned to `rows`, value each UPDATE pipeline_runs returns.
        `new_retry_counts` = list[int|None] aligned to `rows`, value each UPDATE ... RETURNING
        retry_count returns (None means doc disappeared mid-sweep).
        """
        db = MagicMock()
        effects = [MagicMock(fetchall=MagicMock(return_value=rows))]
        for row, rowcount, new_rc in zip(rows, rowcounts, new_retry_counts):
            effects.append(MagicMock())  # UPDATE stage_runs
            effects.append(MagicMock(rowcount=rowcount))  # UPDATE pipeline_runs
            if rowcount == 0:
                continue  # skip retry_count bump
            effects.append(MagicMock(scalar=MagicMock(return_value=new_rc)))  # bump RETURNING
            if new_rc is None:
                continue
            if new_rc > 3:  # over the fixed max_retry=3 below
                effects.append(MagicMock())  # UPDATE documents (mark permanent FAILED)
        db.execute.side_effect = effects
        return db

    def test_sweep_marks_failed_and_redispatches_when_under_cap(self):
        from app.workers.pipeline import _sweep_stale_runs

        doc_id = uuid.uuid4()
        rows = [(uuid.uuid4(), uuid.uuid4(), doc_id, "detect_and_translate")]
        db = self._main_db(rows, rowcounts=[1], new_retry_counts=[1])

        with patch("app.workers.pipeline._get_db", return_value=db), \
             patch("app.workers.pipeline.settings") as mock_settings, \
             patch("app.workers.pipeline.start_ingest_pipeline") as mock_dispatch:
            mock_settings.stale_stage_run_threshold_seconds = 27000
            mock_settings.max_doc_retry_count = 3
            swept = _sweep_stale_runs()

        assert swept == 1
        # Dispatch must happen AFTER the main-session commit, otherwise
        # start_ingest_pipeline's duplicate-dispatch guard sees stale PROCESSING state.
        assert db.commit.called
        mock_dispatch.assert_called_once()
        (dispatched_doc,), _ = mock_dispatch.call_args
        assert str(dispatched_doc) == str(doc_id)

    def test_sweep_marks_permanently_failed_at_cap(self):
        from app.workers.pipeline import _sweep_stale_runs

        doc_id = uuid.uuid4()
        rows = [(uuid.uuid4(), uuid.uuid4(), doc_id, "derive_picture_descriptions")]
        # bump returns 4, which is > max_retry_count=3 -> permanent FAILED path
        db = self._main_db(rows, rowcounts=[1], new_retry_counts=[4])

        with patch("app.workers.pipeline._get_db", return_value=db), \
             patch("app.workers.pipeline.settings") as mock_settings, \
             patch("app.workers.pipeline.start_ingest_pipeline") as mock_dispatch:
            mock_settings.stale_stage_run_threshold_seconds = 27000
            mock_settings.max_doc_retry_count = 3
            swept = _sweep_stale_runs()

        assert swept == 1
        mock_dispatch.assert_not_called()

    def test_sweep_returns_zero_when_nothing_stale(self):
        from app.workers.pipeline import _sweep_stale_runs

        db = self._main_db(rows=[], rowcounts=[], new_retry_counts=[])
        with patch("app.workers.pipeline._get_db", return_value=db), \
             patch("app.workers.pipeline.settings") as mock_settings, \
             patch("app.workers.pipeline.start_ingest_pipeline") as mock_dispatch:
            mock_settings.stale_stage_run_threshold_seconds = 27000
            mock_settings.max_doc_retry_count = 3
            swept = _sweep_stale_runs()

        assert swept == 0
        mock_dispatch.assert_not_called()

    def test_sweep_does_not_redispatch_if_pipeline_run_already_not_processing(self):
        """If a prior sweep already flipped pipeline_run to FAILED, rowcount=0 and
        we skip both the retry_count bump and the dispatch."""
        from app.workers.pipeline import _sweep_stale_runs

        rows = [(uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), "foo")]
        db = self._main_db(rows, rowcounts=[0], new_retry_counts=[None])

        with patch("app.workers.pipeline._get_db", return_value=db), \
             patch("app.workers.pipeline.settings") as mock_settings, \
             patch("app.workers.pipeline.start_ingest_pipeline") as mock_dispatch:
            mock_settings.stale_stage_run_threshold_seconds = 27000
            mock_settings.max_doc_retry_count = 3
            _sweep_stale_runs()

        mock_dispatch.assert_not_called()

    def test_sweep_compensates_on_dispatch_failure(self):
        """When start_ingest_pipeline raises after the main commit, the sweeper
        runs a compensating transaction that reverts retry_count and marks
        the document FAILED."""
        from app.workers.pipeline import _sweep_stale_runs

        doc_id = uuid.uuid4()
        rows = [(uuid.uuid4(), uuid.uuid4(), doc_id, "prepare_document")]
        main_db = self._main_db(rows, rowcounts=[1], new_retry_counts=[1])
        comp_db = MagicMock()  # used by the compensating transaction

        # _get_db is called twice: once for main session, once for compensation.
        get_db_returns = iter([main_db, comp_db])

        with patch("app.workers.pipeline._get_db", side_effect=lambda: next(get_db_returns)), \
             patch("app.workers.pipeline.settings") as mock_settings, \
             patch("app.workers.pipeline.start_ingest_pipeline", side_effect=RuntimeError("dispatch broke")):
            mock_settings.stale_stage_run_threshold_seconds = 27000
            mock_settings.max_doc_retry_count = 3
            _sweep_stale_runs()

        # Compensation transaction committed
        assert comp_db.execute.called
        assert comp_db.commit.called
```

- [ ] **Step 2: Run tests and confirm most fail (current sweeper doesn't redispatch)**

```bash
.venv/bin/pytest tests/pipeline/test_sweeper_autorestart.py -v
```

Expected: 1 pass (zero-stale case), 3 fails.

- [ ] **Step 3: Replace the body of `_sweep_stale_runs`**

In `app/workers/pipeline.py`, replace the existing function body with:

```python
def _sweep_stale_runs() -> int:
    """Sweep stale RUNNING stage_runs and auto-restart their documents.

    For each stage_run at status='RUNNING' older than
    settings.stale_stage_run_threshold_seconds:

      1. Mark the stage_run FAILED + pipeline_run FAILED + bump retry_count.
         **Commit these writes before calling start_ingest_pipeline** — the
         dispatch guard in start_ingest_pipeline runs in its own DB session
         and won't see uncommitted changes (pipeline.py:1736).
      2. If new retry_count <= settings.max_doc_retry_count, call
         start_ingest_pipeline(doc_id). On success it creates a fresh
         pipeline_run + chain in its own transaction. On exception we run
         a compensating transaction: revert retry_count, mark doc FAILED.
      3. If new retry_count > cap, the initial transaction sets
         pipeline_status='FAILED' and we do NOT dispatch.

    The failure → dispatch handoff is split across two transactions by
    design; that's required for the dispatch guard to see the FAILED row.

    Returns the number of stage_runs swept.
    """
    from sqlalchemy import text

    threshold = settings.stale_stage_run_threshold_seconds
    max_retry = settings.max_doc_retry_count

    db = _get_db()
    try:
        stale_rows = db.execute(
            text(
                """
                SELECT sr.id, sr.pipeline_run_id, pr.document_id, sr.stage_name
                FROM ingest.stage_runs sr
                JOIN ingest.pipeline_runs pr ON pr.id = sr.pipeline_run_id
                WHERE sr.status = 'RUNNING'
                  AND sr.started_at < NOW() - make_interval(secs => :threshold)
                  AND pr.status = 'PROCESSING'
                """
            ),
            {"threshold": threshold},
        ).fetchall()

        if not stale_rows:
            return 0

        # Per-row plan: first transaction does failure bookkeeping + retry_count bump.
        # If redispatch is warranted, happens AFTER commit. Dispatch failures run a
        # separate compensating transaction.
        to_dispatch: list[tuple] = []  # (document_id, stage_name, new_retry_count)

        for stage_run_id, pipeline_run_id, document_id, stage_name in stale_rows:
            # --- Transaction 1: mark failures + decide retry path ---

            # 1a. Mark stage_run FAILED
            db.execute(
                text(
                    """
                    UPDATE ingest.stage_runs
                    SET status = 'FAILED',
                        finished_at = NOW(),
                        error_message = COALESCE(error_message, '') || 'stale; swept by periodic_stale_run_sweep'
                    WHERE id = :id
                    """
                ),
                {"id": stage_run_id},
            )

            # 1b. Atomically flip pipeline_run PROCESSING -> FAILED.
            pr_update = db.execute(
                text(
                    """
                    UPDATE ingest.pipeline_runs
                    SET status = 'FAILED',
                        finished_at = COALESCE(finished_at, NOW()),
                        error_message = COALESCE(error_message, '') || 'stale; swept by periodic_stale_run_sweep'
                    WHERE id = :id AND status = 'PROCESSING'
                    """
                ),
                {"id": pipeline_run_id},
            )
            if pr_update.rowcount == 0:
                # Already FAILED from a prior sweep; don't double-dispatch.
                continue

            # 1c. Bump retry_count atomically.
            bump = db.execute(
                text(
                    """
                    UPDATE ingest.documents
                    SET retry_count = retry_count + 1
                    WHERE id = :doc_id
                    RETURNING retry_count
                    """
                ),
                {"doc_id": document_id},
            ).scalar()

            if bump is None:
                logger.warning(
                    "sweeper: document %s disappeared before retry bump; skipping",
                    document_id,
                )
                continue

            # 1d. If over cap, mark document permanently FAILED in this same tx.
            #     failed_stages is ARRAY(String) — see app/models/ingest.py:67.
            if bump > max_retry:
                db.execute(
                    text(
                        """
                        UPDATE ingest.documents
                        SET pipeline_status = 'FAILED',
                            pipeline_stage = :stage,
                            failed_stages =
                                COALESCE(failed_stages, ARRAY[]::text[])
                                || ARRAY[:stage]::text[]
                        WHERE id = :doc_id
                        """
                    ),
                    {"doc_id": document_id, "stage": stage_name},
                )
                logger.error(
                    "sweeper: document=%s exhausted retries (%d > %d) — permanently FAILED",
                    document_id, bump, max_retry,
                )
            else:
                # Under cap: defer dispatch until after this tx commits.
                to_dispatch.append((document_id, stage_name, bump))

        # --- Commit failure bookkeeping. After this, start_ingest_pipeline
        # can see pipeline_run.status='FAILED' from its own session. ---
        db.commit()

        # --- Dispatch phase (separate implicit transactions). ---
        swept = len(stale_rows)
        for document_id, stage_name, bump in to_dispatch:
            try:
                start_ingest_pipeline(str(document_id))
                logger.warning(
                    "sweeper: redispatched document=%s stage_failed=%s retry=%d/%d",
                    document_id, stage_name, bump, max_retry,
                )
            except Exception:
                # Compensating transaction: revert the retry_count bump and
                # mark the document FAILED so the next sweep doesn't re-pick-it.
                logger.exception(
                    "sweeper: redispatch failed for document=%s; "
                    "marking FAILED and reverting retry_count",
                    document_id,
                )
                comp = _get_db()
                try:
                    comp.execute(
                        text(
                            """
                            UPDATE ingest.documents
                            SET retry_count = GREATEST(retry_count - 1, 0),
                                pipeline_status = 'FAILED',
                                pipeline_stage = :stage,
                                failed_stages =
                                    COALESCE(failed_stages, ARRAY[]::text[])
                                    || ARRAY[:stage]::text[]
                            WHERE id = :doc_id
                            """
                        ),
                        {"doc_id": document_id, "stage": stage_name},
                    )
                    comp.commit()
                except Exception:
                    logger.exception(
                        "sweeper: compensation write also failed for %s; operator must triage",
                        document_id,
                    )
                    comp.rollback()
                finally:
                    comp.close()

        return swept
    except Exception:
        logger.exception("_sweep_stale_runs: rollback due to error")
        db.rollback()
        return 0
    finally:
        db.close()
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
.venv/bin/pytest tests/pipeline/test_sweeper_autorestart.py tests/pipeline/test_stale_run_sweeper.py -v
```

Expected: all tests from BOTH files pass. The older `test_stale_run_sweeper.py::TestSweepStaleRuns` tests may need minor updates — their mocks only accounted for 3 db.execute calls; the new function issues more. Update those mocks to tolerate the new call count (use `db.execute.side_effect = [...]` with enough MagicMock rows), or switch them to `return_value=MagicMock(...)` to accept any number of calls.

- [ ] **Step 5: Run the full pipeline suite for regressions**

```bash
.venv/bin/pytest tests/pipeline/ tests/unit/test_ingest_pipeline_coverage.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/pipeline/test_sweeper_autorestart.py tests/pipeline/test_stale_run_sweeper.py app/workers/pipeline.py
git commit -m "feat(pipeline): sweeper auto-restarts failed docs with retry_count cap"
```

---

## Chunk 3: Validation

### Task 5: End-to-end integration check against a deliberately-induced orphan

**Files:** none modified; this task is manual validation using SQL + Celery inspect.

Requires: stack running (`docker compose up -d`), `.env` reflecting new timeouts + threshold, worker + beat rebuilt.

- [ ] **Step 1: Verify the beat schedule picked up the new threshold**

```bash
docker exec eip-mmdpp-beat-1 python -c "
import sys; sys.path.insert(0, '/app')
from app.workers.celery_app import celery_app
from app.config import get_settings
print('threshold:', get_settings().stale_stage_run_threshold_seconds)
print('max_retry:', get_settings().max_doc_retry_count)
for name, entry in celery_app.conf.beat_schedule.items():
    print(f'{name}: {entry[\"schedule\"]}')
"
```

Expected: `threshold: 27000`, `max_retry: 3`, `periodic-stale-run-sweep: 0:10:00`.

- [ ] **Step 2: Induce a synthetic orphan**

Pick any existing document in the DB (or upload a throwaway one). Insert a stale pipeline_run + stage_run for it with `started_at` well past the threshold. With `STALE_STAGE_RUN_THRESHOLD_SECONDS=27000` (7.5h), an 8-hour-old row will trip it:

```sql
-- Replace <doc_id> with a real document UUID
WITH new_run AS (
  INSERT INTO ingest.pipeline_runs (document_id, status, started_at)
  VALUES ('<doc_id>', 'PROCESSING', NOW() - INTERVAL '8 hours')
  RETURNING id
)
INSERT INTO ingest.stage_runs (pipeline_run_id, stage_name, status, started_at, attempt)
SELECT id, 'prepare_document', 'RUNNING', NOW() - INTERVAL '8 hours', 1 FROM new_run;
```

- [ ] **Step 3: Trigger the sweeper immediately (don't wait 10 min)**

```bash
docker compose exec worker python -c "from app.workers.pipeline import _sweep_stale_runs; print('swept:', _sweep_stale_runs())"
```

Expected: `swept: 1`.

- [ ] **Step 4: Verify the sweeper's effects**

Check the synthetic rows and the target document:

```sql
SELECT id, status, error_message FROM ingest.pipeline_runs ORDER BY started_at DESC LIMIT 5;
SELECT id, stage_name, status, error_message FROM ingest.stage_runs ORDER BY started_at DESC LIMIT 5;
SELECT id, filename, pipeline_status, retry_count FROM ingest.documents WHERE id = '<doc_id>';
```

Expected:
- The synthetic `pipeline_runs` row is `FAILED` with the `swept by periodic_stale_run_sweep` marker.
- The synthetic `stage_runs` row is `FAILED`.
- A **new** `pipeline_runs` row exists for the same document (from the sweeper's redispatch), status = `PROCESSING`.
- `documents.retry_count` = 1.
- `documents.pipeline_status` = `PROCESSING` (the fresh chain has already started).

- [ ] **Step 5: Verify retry cap by repeating the orphan induction 3 more times**

Repeat Step 2's INSERT and Step 3's trigger three more times, manually setting `documents.retry_count` between runs to simulate prior retries. Or simpler: set `retry_count` directly to 3 in the DB, then induce one more orphan + trigger.

After the 4th sweep:
- `documents.pipeline_status` should be `FAILED`.
- The sweeper logs an `ERROR` line: `"sweeper: document=... exhausted retries (4 > 3) — permanently FAILED"`.
- No new `pipeline_runs` row is created.

- [ ] **Step 6: Cleanup**

Delete synthetic rows:
```sql
DELETE FROM ingest.stage_runs WHERE error_message LIKE '%swept by periodic_stale_run_sweep%';
DELETE FROM ingest.pipeline_runs WHERE error_message LIKE '%swept by periodic_stale_run_sweep%';
UPDATE ingest.documents SET retry_count = 0 WHERE id = '<doc_id>';
```

- [ ] **Step 7: Final sanity — run full test suite one more time**

```bash
.venv/bin/pytest tests/pipeline/ tests/unit/test_ingest_pipeline_coverage.py -v
```

All tests should pass except any pre-existing failures documented in prior commits.

---

## Acceptance Criteria

1. **Every pipeline task** has a realistic `soft_time_limit` + `time_limit` and is wrapped by `@guard_stage_run("<stage>")`. `test_guard_stage_run_universal.py` passes for all 13 pipeline tasks.
2. **No task in `pipeline.py` silently swallows exceptions** in the main success path. Category-A swallows in best-effort side work carry a code comment justifying the choice. `grep -nE "except:$|except BaseException" app/workers/pipeline.py` returns zero matches.
3. **Sweeper threshold** = `max(*_time_limit) + 30min`. Under normal operation the sweeper never fires because Celery's time-limit machinery kills runaway tasks first via `SoftTimeLimitExceeded`.
4. **Sweeper auto-restart** works: a synthetic orphan is marked FAILED, `documents.retry_count` increments, and a fresh `pipeline_run` is created via `start_ingest_pipeline`. After `max_doc_retry_count` sweeps on the same document, the document's `pipeline_status` flips to `FAILED` permanently and no further redispatch happens.
5. **Celery retries engage** for real exceptions: any `httpx.ReadTimeout` / `SoftTimeLimitExceeded` / other non-retry-signal exception raised from a task body triggers `self.retry(exc=...)` which adds the task back to the queue with the task's configured `default_retry_delay`.

## Rollback

This plan is layered — each layer is independently revertible:

- **Timeouts are too tight** — revert `.env` values to prior (larger) numbers. No code change needed. Worker restart picks up the new values on next `docker compose up -d --force-recreate worker`.
- **Universal `guard_stage_run` causes a regression** — revert the specific task's decorator (one-line change). The decorator is independent per task.
- **Silent-swallow audit caused a regression** — revert that specific function's commit. Other fixes stay.
- **Sweeper auto-restart misbehaves** — revert the `_sweep_stale_runs` body commit; the prior version (mark-and-surface-only) is the prior plan's `ae49090` state. Or just remove the `periodic-stale-run-sweep` entry from `beat_schedule` to disable entirely while leaving the function in place for manual invocation.
- **`retry_count` column causes a migration problem** — `alembic downgrade -1`. The model field tolerates a missing column at import time only if you also revert the model change.

Each layer can be rolled back in < 5 minutes.

## Notes for the Executor

- **Do not implement Task 4 (heartbeat) from the prior plan.** This plan makes it unnecessary. Keep the deferred-note in that plan as historical context.
- **Commit after every function** in Task 2 (silent-swallow audit). That file is large; atomic commits make review and revert sane.
- **Do not skip the test updates** in Task 4 Step 4 — the older `test_stale_run_sweeper.py` tests were written against a simpler sweeper and need their mocks widened. Leaving them failing would hide real regressions.
