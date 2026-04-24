# Ingest Hardening v2 — Close Gaps from 2026-04-24 Run

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every gap observed in the 2026-04-24 ingest run. Make failures surface consistently, make terminal states unconditional on exception paths, fix the two docs that didn't complete, eliminate the ArcadeDB EXTRACTED_FROM silent-loss bug, raise realistic timeouts, and switch to split-worker mode as the default to get 3× concurrent VLM processing.

**Architecture:** Four-layer structure that replaces the conditional safety nets from the prior plan with unconditional ones:

1. **Unconditional terminal state.** `guard_stage_run` currently re-raises unhandled exceptions and relies on `self.request.retries >= self.max_retries` to decide whether to flip the doc to PARTIAL_COMPLETE. That check is false on the FIRST failure of any task with `max_retries=1` (like `derive_document_anchors`), which is exactly the scenario that left the `.txt` doc stuck at PROCESSING. The correct invariant: **if code reaches the guard's `except Exception`, the task decided not to retry** — if it had, it would have called `self.retry(exc=...)` which raises `CeleryRetry` (already passed through untouched). Therefore guard terminalizes unconditionally on `except Exception`. This simultaneously fixes (a) the stuck-PROCESSING bug, (b) soft-timeout exhaustion (Celery's `MaxRetriesExceededError` from `self.retry(...)` *is* an `Exception` and will reach the guard), and (c) any future task that fails without calling `self.retry`.

2. **Correct audit for `derive_ontology_graph`.** The 2026-04-23 plan's audit missed this task's helper `_derive_ontology_graph_bundle_passes`, where `except Exception` (catching `SoftTimeLimitExceeded`) calls `_terminalize_failure` and re-raises without `self.retry`. Add an explicit `except SoftTimeLimitExceeded` branch *before* the generic `except Exception` that calls `self.retry(exc=exc)`. On exhaustion, `MaxRetriesExceededError` propagates to the guard (which now always terminalizes). Pair with per-call HTTP timeouts well below the full stage budget so a single bad pass doesn't consume the whole retry budget.

3. **Legacy extraction writes a schema-valid stub.** `.txt` files fall through to legacy extraction in `prepare_document`, which writes markdown but not `docling_document.json`. Downstream `derive_document_anchors` calls `DoclingDocument.model_validate(...)` (in `docling_anchors.py:317`), so the stub must be a valid `DoclingDocument`, not a hand-crafted dict. Build it via `DoclingDocument(...)`/`add_text(...)` and `model_dump(...)`, tested by round-tripping through both `model_validate` and `walk`.

4. **Correct edge-creation path + propagation.** The observed `Record #XX:Y not found` errors are from `batch_create_entity_chunk_edges_sync` (in `app/services/arcadedb_graph.py:2168`) called from `derive_structure_links` (in `app/workers/pipeline.py:5448`), **not** from `_import_graph_phase_structural_edges`. The client raises `httpx.HTTPStatusError` with the error body on `exc.response.text`. Add an in-batch retry that matches the ArcadeDB "Record #N:M not found" error pattern from `exc.response.text`, backs off, and retries up to 3 attempts. **Crucially**, after retry exhaustion the batch helper must raise rather than log-and-return-count — and `derive_structure_links` must propagate that raise (today it catches the batch exception and only logs a warning, which is why the 500s in the run were silent).

**Tech Stack:** Python 3.11, Celery 5+ (prefork + Redis broker/beat), SQLAlchemy (sync session), Alembic, Pydantic-Settings, ArcadeDB via custom `httpx`-based client, docling_core DoclingDocument model, pytest, docker compose.

**Relationship to prior plans:**
- `2026-04-23-stale-stage-run-sweeper.md` (shipped) — `guard_stage_run` + `_sweep_stale_runs` + beat entry.
- `2026-04-23-reliable-ingest-retry.md` (shipped) — realistic timeouts + `retry_count` + auto-restart sweeper + universal `guard_stage_run` + 5-task SoftTimeLimitExceeded audit.
- **This plan** replaces the prior plan's conditional terminalization with unconditional, finishes the audit (adds `_derive_ontology_graph_bundle_passes`), ships a schema-valid legacy stub, fixes the correct ArcadeDB edge path with propagation, defaults to split-worker topology, and cleans up config/manage.sh drift.

**Out of scope (explicit):**
- Shortening the `PICTURE_DESCRIPTION_PROMPT` — user is handling manually.
- Per-pass kill-signal timeout inside `derive_ontology_graph` (would need signal-based interrupt; defer until data shows a doc needing it).
- Heartbeat-based detection (prior plan's deferred item).
- Confirmed-count return from `batch_create_entity_chunk_edges_sync` (would require rewriting the batch `sqlscript` to `LET / RETURN` the created RIDs; separate follow-up).

**Note on `.env`:** this repo keeps `.env` local-only (gitignored). Each commit that lists `.env` changes in its step-by-step text updates the file on-disk but does NOT include it in `git add`. Only `env.example` goes into version control. Operators with existing `.env` files should hand-merge the new values from the step block, or diff against the prior commit's `env.example`.

**2026-04-24 run facts that motivated this plan:**
- 21 docs, 10.5h wall time.
- 19/21 COMPLETE, 1 PARTIAL_COMPLETE, 1 stuck PROCESSING.
- Picture descriptions: 7h16m cumulative (59% of total), max 3h54m on one doc.
- Graph extraction: 3h54m cumulative across successful runs (max 2h10m); the one FAILED `derive_ontology_graph` tripped the 6h soft limit, proving that the *failed* ceiling is real even though every *successful* run was well under it.
- Zero sweeper restarts were needed — retries + guard handled everything real.
- 5+ ArcadeDB 500s on EXTRACTED_FROM edge creation (silent data loss — confirmed in `batch_create_entity_chunk_edges_sync`, not structural-edges path).

---

## File Structure

**Tests (extend existing files where present; create only the legacy artifact test):**
- `tests/pipeline/test_guard_stage_run.py` — append unconditional-terminalization cases + `_terminalize_doc_and_run` helper tests
- `tests/unit/test_derive_ontology_graph_bundle_passes.py` — add soft-timeout → self.retry case (create this file only if it doesn't exist; otherwise append)
- `tests/unit/test_arcadedb_graph.py` — append batch-edge retry + propagation cases
- `tests/pipeline/test_legacy_extraction_artifact.py` — **new** file; legacy stub round-trips through `DoclingDocument.model_validate` + `docling_anchors.walk`
- `tests/pipeline/test_pipeline_chain.py` or `tests/unit/test_ingest_pipeline_coverage.py` — extend with `collect_derivations` StageRun assertion (Task 6)

**Modify (source):**
- `app/workers/pipeline.py`
  - `guard_stage_run` — unconditional terminalize in `except Exception`
  - `_derive_ontology_graph_bundle_passes` — explicit `except SoftTimeLimitExceeded` with `self.retry`
  - `prepare_document` legacy branch — build + upload schema-valid stub
  - `derive_structure_links` — propagate batch-edge failure instead of swallow
  - investigate `collect_derivations` StageRun write (see Task 7)
- `app/services/arcadedb_graph.py::batch_create_entity_chunk_edges_sync` — in-batch retry on `Record #...not found` pattern + confirmed-count return
- `app/config.py` — sync defaults to env values (timeouts + sweeper threshold)
- `.env` + `env.example` — bump graph budgets, set per-call timeout well below task budget, set `WORKER_*_CONCURRENCY` defaults
- `docker-compose.yml` — ensure split-profile services read `WORKER_*_CONCURRENCY` env vars
- `manage.sh` — `--start` defaults to split; `--start-mixed` for legacy; update `--worker-status`, `--restart`, `--logs`, `--help` to enumerate split containers

---

## Chunk 1: Guard + terminalization (foundation)

### Task 0: Make `guard_stage_run` terminalize unconditionally on Exception

**Why this is first:** Every other task in this plan relies on the invariant that an exception reaching the guard means terminal failure. The prior plan's conditional `retries >= max_retries` check is incorrect for tasks where `max_retries=1` and the generic `except` branch doesn't call `self.retry` (observed on `derive_document_anchors`: exception on first attempt leaves `retries=0 < max_retries=1`, guard skipped terminalization, doc stuck at PROCESSING).

**Correctness argument:**
- `CeleryRetry`: raised by `self.retry(...)` when retries remain. Guard passes through untouched. Celery re-queues.
- `SoftTimeLimitExceeded`: raised by Celery's soft-timeout signal. Our task-body handlers all call `self.retry(exc=exc)` (this plan adds the missing one in Task 2). **Celery 5 behavior for `self.retry(exc=exc)` on retry exhaustion: it re-raises the provided `exc`, not `MaxRetriesExceededError`.** So after the final retry, the task body re-raises `SoftTimeLimitExceeded`. If the guard passes it through (as the prior plan did), the exception escapes terminalization and the doc stays stuck. **Therefore, remove `SoftTimeLimitExceeded` from the guard's pass-through tuple.** Soft-timeout always reaches `except Exception` in the guard, which terminalizes. On the retry-remaining case, the task body calls `self.retry(exc=exc)` which raises `CeleryRetry` (still pass-through) before the `SoftTimeLimitExceeded` can propagate.
- `MaxRetriesExceededError`: raised by `self.retry()` ONLY when no `exc` is passed. Safe to let it reach `except Exception` and terminalize — that path isn't used in this codebase (all our handlers pass `exc`).
- Anything else: if a task raises and reaches the guard's `except Exception`, by construction it did NOT call `self.retry`, so it is terminal.

**Two must-do nuances (from review):**
1. **Preserve existing terminal statuses.** `prepare_document` sets `STATUS_FAILED` before raising on deterministic failures (`app/workers/pipeline.py` ~line 3224/3241/3254). The guard must NOT overwrite an already-`FAILED` or `COMPLETE` document with `PARTIAL_COMPLETE` — that would downgrade a definitive failure to something softer. Only terminalize documents currently in a non-terminal state (`PENDING` or `PROCESSING`).
2. **Terminalize the PipelineRun too, not just the document.** If the chain aborts before `finalize_document`, `ingest.pipeline_runs.status` stays `PROCESSING` indefinitely. Add a helper `_terminalize_doc_and_run(document_id, run_id, status)` that atomically flips the document AND the owning pipeline_run when both are in non-terminal state.

Files:
- Modify: `app/workers/pipeline.py::guard_stage_run`
- Add: `app/workers/pipeline.py::_terminalize_doc_and_run` helper (new, near `_update_document_pipeline_status`)
- Modify: `tests/pipeline/test_guard_stage_run.py`

- [ ] **Step 1: Add failing tests to existing `tests/pipeline/test_guard_stage_run.py`**

Append:
```python
class TestUnconditionalTerminalization:
    DOC_ID = "11111111-1111-1111-1111-111111111111"
    RUN_ID = "22222222-2222-2222-2222-222222222222"

    def _fake_task(self, retries=0, max_retries=2):
        from unittest.mock import MagicMock
        s = MagicMock()
        s.request.retries = retries
        s.max_retries = max_retries
        return s

    def test_terminalizes_on_first_failure_for_max_retries_1(self):
        """Reproduces the 0005_wildweasels stuck-PROCESSING bug:
        a task with max_retries=1 that raises on first attempt must terminalize
        BOTH the document and its PipelineRun."""
        from unittest.mock import patch
        from app.workers.pipeline import guard_stage_run

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise RuntimeError("boom")

        with patch("app.workers.pipeline._get_db"), \
             patch("app.workers.pipeline._update_stage_run"), \
             patch("app.workers.pipeline._terminalize_doc_and_run") as m_term:
            import pytest as _p
            with _p.raises(RuntimeError):
                task(self._fake_task(retries=0, max_retries=1), self.DOC_ID, run_id=self.RUN_ID)

        m_term.assert_called_once()
        args, _ = m_term.call_args
        assert args == (self.DOC_ID, self.RUN_ID, "PARTIAL_COMPLETE")

    def test_does_not_overwrite_existing_failed_status(self):
        """prepare_document sets STATUS_FAILED on deterministic failure before
        raising. Guard must NOT downgrade that to PARTIAL_COMPLETE.
        (The preserve-check is inside _terminalize_doc_and_run — separate test below.)"""
        from unittest.mock import patch
        from app.workers.pipeline import guard_stage_run

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise RuntimeError("deterministic failure")

        with patch("app.workers.pipeline._get_db"), \
             patch("app.workers.pipeline._update_stage_run"), \
             patch("app.workers.pipeline._terminalize_doc_and_run") as m_term:
            import pytest as _p
            with _p.raises(RuntimeError):
                task(self._fake_task(retries=0, max_retries=3), self.DOC_ID, run_id=self.RUN_ID)

        m_term.assert_called_once()

    def test_celery_retry_still_passes_through_no_terminalization(self):
        from unittest.mock import patch
        from celery.exceptions import Retry as CeleryRetry
        from app.workers.pipeline import guard_stage_run

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise CeleryRetry()

        with patch("app.workers.pipeline._update_stage_run"), \
             patch("app.workers.pipeline._terminalize_doc_and_run") as m_term:
            import pytest as _p
            with _p.raises(CeleryRetry):
                task(self._fake_task(), self.DOC_ID, run_id=self.RUN_ID)

        m_term.assert_not_called()


class TestTerminalizeDocAndRun:
    """Helper-level tests: _terminalize_doc_and_run must preserve existing
    terminal statuses and must update BOTH the document and the pipeline_run.
    Note: uses real UUIDs because the helper calls uuid.UUID(...)."""

    DOC_ID = "11111111-1111-1111-1111-111111111111"
    RUN_ID = "22222222-2222-2222-2222-222222222222"

    def test_preserves_existing_failed_document_status(self):
        from unittest.mock import MagicMock, patch
        from app.workers.pipeline import _terminalize_doc_and_run

        db = MagicMock()
        doc = MagicMock(); doc.pipeline_status = "FAILED"
        run = MagicMock(); run.status = "PROCESSING"
        db.get.side_effect = [doc, run]

        with patch("app.workers.pipeline._get_db", return_value=db):
            _terminalize_doc_and_run(self.DOC_ID, self.RUN_ID, "PARTIAL_COMPLETE")

        # Existing FAILED is preserved even though pipeline_run still flips.
        assert doc.pipeline_status == "FAILED"
        assert run.status == "FAILED"  # run flipped from PROCESSING

    def test_preserves_existing_pending_human_review(self):
        """PENDING_HUMAN_REVIEW is also terminal — don't downgrade it."""
        from unittest.mock import MagicMock, patch
        from app.workers.pipeline import _terminalize_doc_and_run

        db = MagicMock()
        doc = MagicMock(); doc.pipeline_status = "PENDING_HUMAN_REVIEW"
        run = MagicMock(); run.status = "PROCESSING"
        db.get.side_effect = [doc, run]

        with patch("app.workers.pipeline._get_db", return_value=db):
            _terminalize_doc_and_run(self.DOC_ID, self.RUN_ID, "PARTIAL_COMPLETE")

        assert doc.pipeline_status == "PENDING_HUMAN_REVIEW"

    def test_flips_processing_document_and_pipeline_run(self):
        from unittest.mock import MagicMock, patch
        from app.workers.pipeline import _terminalize_doc_and_run

        db = MagicMock()
        doc = MagicMock(); doc.pipeline_status = "PROCESSING"
        run = MagicMock(); run.status = "PROCESSING"
        db.get.side_effect = [doc, run]

        with patch("app.workers.pipeline._get_db", return_value=db):
            _terminalize_doc_and_run(self.DOC_ID, self.RUN_ID, "PARTIAL_COMPLETE")

        assert doc.pipeline_status == "PARTIAL_COMPLETE"
        assert run.status == "FAILED"  # pipeline_run is FAILED not PARTIAL
        assert run.finished_at is not None
        assert db.commit.called

    def test_preserves_pipeline_run_already_failed(self):
        from unittest.mock import MagicMock, patch
        from app.workers.pipeline import _terminalize_doc_and_run

        db = MagicMock()
        doc = MagicMock(); doc.pipeline_status = "PROCESSING"
        run = MagicMock(); run.status = "FAILED"  # already failed
        db.get.side_effect = [doc, run]

        with patch("app.workers.pipeline._get_db", return_value=db):
            _terminalize_doc_and_run(self.DOC_ID, self.RUN_ID, "PARTIAL_COMPLETE")

        # doc flips; run stays FAILED (no overwrite)
        assert doc.pipeline_status == "PARTIAL_COMPLETE"
        assert run.status == "FAILED"
```

- [ ] **Step 2: Run tests, confirm 2 fail (the two non-CeleryRetry terminalization cases)**

```bash
.venv/bin/pytest tests/pipeline/test_guard_stage_run.py::TestUnconditionalTerminalization -v
```

- [ ] **Step 3: Add the `_terminalize_doc_and_run` helper**

In `app/workers/pipeline.py`, near the existing `_update_document_pipeline_status` (around line 1042). Mirror that function's pattern of **local imports** for the ORM models (the existing helper does `from app.models.ingest import Document` locally):

```python
# Terminal statuses the guard must NOT overwrite. Matches the union of
# terminal states used elsewhere in the pipeline (see STATUS_* constants
# at pipeline.py:~1300). PENDING_HUMAN_REVIEW is terminal-ish — the doc
# is awaiting operator action and guard must not downgrade it.
_TERMINAL_DOC_STATUSES = {
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_PARTIAL_COMPLETE,
    STATUS_PENDING_REVIEW,
}


def _terminalize_doc_and_run(document_id: str, run_id: str | None, doc_status: str) -> None:
    """Flip a document and its owning PipelineRun to terminal states.

    Preserves existing terminal document statuses — if the doc is already
    FAILED / COMPLETE / PARTIAL_COMPLETE / PENDING_HUMAN_REVIEW, do NOT
    overwrite with a softer value (e.g., prepare_document sets STATUS_FAILED
    on deterministic failure; guard shouldn't downgrade that).

    The pipeline_run always moves to FAILED when it was PROCESSING,
    regardless of the doc_status argument (a PARTIAL_COMPLETE document
    still corresponds to a FAILED run — the chain didn't reach
    finalize_document).
    """
    from datetime import datetime as dt
    from app.models.ingest import Document, PipelineRun  # local imports match the pattern in _update_document_pipeline_status

    db = _get_db()
    try:
        doc = db.get(Document, uuid.UUID(str(document_id)))
        if doc is not None and doc.pipeline_status not in _TERMINAL_DOC_STATUSES:
            doc.pipeline_status = doc_status

        if run_id:
            run = db.get(PipelineRun, uuid.UUID(str(run_id)))
            if run is not None and run.status == "PROCESSING":
                run.status = "FAILED"
                if run.finished_at is None:
                    run.finished_at = dt.utcnow()

        db.commit()
    except Exception:
        logger.exception(
            "_terminalize_doc_and_run: failed for document=%s run_id=%s",
            document_id, run_id,
        )
        db.rollback()
    finally:
        db.close()
```

- [ ] **Step 3b: Remove `SoftTimeLimitExceeded` from the guard pass-through**

In `app/workers/pipeline.py::guard_stage_run`, change:
```python
            except (CeleryRetry, SoftTimeLimitExceeded):
                raise
```
to:
```python
            except CeleryRetry:
                raise
            # SoftTimeLimitExceeded is NOT pass-through. If the task body calls
            # self.retry(exc=SoftTimeLimitExceeded), Celery re-raises the
            # exception ITSELF on retry exhaustion (not MaxRetriesExceededError).
            # Letting it fall through to `except Exception` ensures terminalization
            # on the final attempt.
```

- [ ] **Step 3c: Update the existing test that asserted soft-timeout pass-through**

`tests/pipeline/test_guard_stage_run.py::test_soft_time_limit_passes_through_untouched` (around line 73) currently asserts the opposite invariant. Replace it with a test that asserts **soft-timeout reaches terminalization on exhaustion**:

```python
def test_soft_time_limit_terminalizes_on_final_attempt(self):
    """SoftTimeLimitExceeded reaching the guard (i.e. not converted to
    CeleryRetry) means retries are exhausted — must terminalize."""
    from unittest.mock import patch
    from celery.exceptions import SoftTimeLimitExceeded
    from app.workers.pipeline import guard_stage_run

    @guard_stage_run("fake_stage")
    def task(self_, document_id, run_id=None):
        raise SoftTimeLimitExceeded()

    with patch("app.workers.pipeline._get_db"), \
         patch("app.workers.pipeline._update_stage_run"), \
         patch("app.workers.pipeline._terminalize_doc_and_run") as m_term:
        import pytest as _p
        with _p.raises(SoftTimeLimitExceeded):
            task(self._fake_task_self(), "11111111-1111-1111-1111-111111111111", run_id="22222222-2222-2222-2222-222222222222")

    m_term.assert_called_once()
```

- [ ] **Step 4: Modify `guard_stage_run` — use the new helper unconditionally**

In `app/workers/pipeline.py::guard_stage_run`, replace the conditional retry-count check inside `except Exception as exc:` with a single call to the helper:

```python
            except Exception as exc:
                logger.exception(
                    "guard_stage_run: %s raised unhandled exception "
                    "(document_id=%s run_id=%s)",
                    stage_name, document_id, run_id,
                )
                if run_id:
                    try:
                        db = _get_db()
                        try:
                            _update_stage_run(
                                db, run_id, stage_name, "FAILED",
                                attempt=self.request.retries + 1,
                                error=f"unhandled exception: {exc!r}",
                            )
                            db.commit()
                        finally:
                            db.close()
                    except Exception:
                        logger.exception(
                            "guard_stage_run: FAILED-status write also failed "
                            "for run_id=%s stage=%s",
                            run_id, stage_name,
                        )
                # Unconditional terminalization: reaching this branch means the
                # task did not call self.retry() (which raises CeleryRetry and is
                # pass-through). The helper preserves existing terminal statuses
                # (FAILED set by prepare_document on deterministic failure won't
                # be downgraded) and flips the pipeline_run when still PROCESSING.
                _terminalize_doc_and_run(document_id, run_id, "PARTIAL_COMPLETE")
                raise
```

Remove any prior conditional `retries >= max_retries` logic.

- [ ] **Step 5: Run tests + regression**

```bash
.venv/bin/pytest tests/pipeline/ tests/unit/test_ingest_pipeline_coverage.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/pipeline/test_guard_stage_run.py app/workers/pipeline.py
git commit -m "fix(pipeline): guard_stage_run terminalizes both doc and pipeline_run; preserves existing FAILED"
```

---

## Chunk 2: Task-specific fixes

### Task 1: Raise graph time limits + sync Pydantic defaults

**Why:** one doc needed >6h on graph extraction in the 2026-04-24 run (failed at the 6h soft limit). Bump soft→8h, hard→9h, and make per-call HTTP timeout (`DOCLING_GRAPH_TIMEOUT`) tight relative to the full stage budget so a single runaway LLM call can't consume the whole 8h budget.

**Reviewer-requested clarifications addressed:**
- Successful graph runs peaked at 2h10m. The 6h limit was tripped by a single *failed* run on one doc (Engagement and Fire Control Radars). 8h gives that case margin without approving infinite runs.
- **Timeout math with per-pass retries.** `_run_single_pass` has its own retry loop with `pass_max_retries=3` (default in `app/config.py:126`). So a single pass can issue up to 3 LLM calls before giving up. If one call takes `DOCLING_GRAPH_TIMEOUT` seconds, the worst case for *one pass* is `3 × DOCLING_GRAPH_TIMEOUT`. A doc typically runs 2-4 passes. To fit within the 8h (28800s) stage budget while allowing realistic pass counts and pass-internal retries:
  - `DOCLING_GRAPH_TIMEOUT = 1800s` (30 min per LLM call)
  - `pass_max_retries = 3` (unchanged)
  - Worst case per pass: 3 × 1800 = 5400s = 90 min
  - Four passes × 90 min = 6h; fits in 8h budget with room for orchestration overhead
  - This is much tighter than the previous 4h per-call that would have allowed a single runaway call to consume 12h (review finding H2).
- **Service-side config must also update.** `DOCLING_GRAPH_LLM_TIMEOUT` is read by the docling-graph *service* (separate container, separate config). The compose fallback at `docker-compose.yml:162` reads `${DOCLING_GRAPH_LLM_TIMEOUT:-10800}` — missing `.env` would silently give 3h. Must update the fallback default AND the service-side default in `docker/docling-graph/app/config_builder.py` to match.

**Config defaults also synced:** `app/config.py` currently has `celery_visibility_timeout=10800` and `docling_graph_timeout=10800`. Update to match the env block so a missing `.env` doesn't revert to old values.

**Files:**
- Modify: `app/config.py` (graph_soft_time_limit, graph_time_limit, docling_graph_timeout, celery_visibility_timeout, stale_stage_run_threshold_seconds defaults — NOT docling_graph_llm_timeout; that lives service-side)
- Modify: `.env`, `env.example` (matching values)
- Modify: `docker-compose.yml:162` (fallback for DOCLING_GRAPH_LLM_TIMEOUT — consumed by the docling-graph service)
- Modify: `docker/docling-graph/app/config_builder.py` (service-side default for DOCLING_GRAPH_LLM_TIMEOUT)

- [ ] **Step 1: Update `app/config.py` defaults**

```python
    # Graph extraction is the slowest stage — soft=8h, hard=9h cover observed slow docs
    graph_soft_time_limit: int = 28800
    graph_time_limit: int = 32400

    # Per-call HTTP timeout for docling-graph /extract-pass. 30 min per LLM call;
    # with pass_max_retries=3 inside _run_single_pass, worst case per pass = 90 min,
    # leaving budget for ~4 passes within the 8h stage limit.
    #
    # Note: DOCLING_GRAPH_LLM_TIMEOUT is NOT an app-side setting here. It's read
    # by the separate docling-graph service — update there via
    # docker/docling-graph/app/config_builder.py and the docker-compose.yml
    # fallback at line 162. See Steps 3 + 4 below.
    docling_graph_timeout: int = 1800

    # Must exceed longest Celery time_limit so long tasks aren't redelivered.
    celery_visibility_timeout: int = 36000  # 10h

    # Sweeper only fires after Celery's own timeout should have killed + retried.
    stale_stage_run_threshold_seconds: int = 34200  # 9h30m
```

- [ ] **Step 2: Update `.env` and `env.example`**

```
GRAPH_SOFT_TIME_LIMIT=28800
GRAPH_TIME_LIMIT=32400
DOCLING_GRAPH_TIMEOUT=1800
DOCLING_GRAPH_LLM_TIMEOUT=1800
CELERY_VISIBILITY_TIMEOUT=36000
STALE_STAGE_RUN_THRESHOLD_SECONDS=34200
```

`.env` is local-only (gitignored); only `env.example` is committed. Note this in each commit that modifies `.env`.

- [ ] **Step 3: Update `docker-compose.yml` fallback**

At line ~162:
```yaml
      DOCLING_GRAPH_LLM_TIMEOUT: ${DOCLING_GRAPH_LLM_TIMEOUT:-1800}
```
(was `:-10800`).

- [ ] **Step 4: Update `docker/docling-graph/app/config_builder.py`**

Find the default for `DOCLING_GRAPH_LLM_TIMEOUT` (typically read via `os.environ.get("DOCLING_GRAPH_LLM_TIMEOUT", "10800")` or similar). Change the fallback to `"1800"`. If the service defines a Pydantic settings model, update that field's default.

- [ ] **Step 5: Verify**

```bash
.venv/bin/python -c "from app.config import get_settings; s=get_settings(); print(s.graph_time_limit, s.graph_soft_time_limit, s.docling_graph_timeout, s.celery_visibility_timeout, s.stale_stage_run_threshold_seconds)"
```
Expected: `32400 28800 1800 36000 34200`.

(Confirming `DOCLING_GRAPH_LLM_TIMEOUT` takes effect in the docling-graph service is a service-side verification — after container rebuild, `docker compose exec docling-graph env | grep DOCLING_GRAPH_LLM_TIMEOUT` should show `1800`.)

- [ ] **Step 6: Commit**

```bash
git add app/config.py env.example docker-compose.yml docker/docling-graph/app/config_builder.py
git commit -m "feat(config): graph stage 8h/9h + per-call 30m across app and docling-graph service"
```

Note: `.env` is not staged (gitignored); the env block lives only on the operator's disk.

---

### Task 2: `derive_ontology_graph` soft-timeout → `self.retry(exc=exc)`

**Why:** `_derive_ontology_graph_bundle_passes`'s `except Exception` branch catches `SoftTimeLimitExceeded` and calls `_terminalize_failure` without engaging Celery's retry budget. With Task 0's unconditional guard-terminalization now in place, we keep `_terminalize_failure` for non-timeout cases but route `SoftTimeLimitExceeded` through `self.retry(exc=exc)`. On retries-exhausted, `self.retry` raises `MaxRetriesExceededError` (an `Exception`), which propagates to guard (which now terminalizes unconditionally — from Task 0).

**Files:**
- Modify: `app/workers/pipeline.py::_derive_ontology_graph_bundle_passes` (around line 4950)
- Modify or create: `tests/unit/test_derive_ontology_graph_bundle_passes.py`

- [ ] **Step 1: Write the failing test**

If the file doesn't exist, create it; otherwise append. `_terminalize_failure` is **nested inside** `_derive_ontology_graph_bundle_passes`, so it CANNOT be patched at module level. Instead, patch the module-level things it eventually calls (`_update_document_pipeline_status`, `_attempt_rollback`) plus the early trigger:

```python
"""_derive_ontology_graph_bundle_passes must convert SoftTimeLimitExceeded
into self.retry(), not silently flip the doc to PARTIAL_COMPLETE."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import Retry as CeleryRetry, SoftTimeLimitExceeded

pytestmark = pytest.mark.unit


def test_soft_time_limit_in_helper_triggers_self_retry():
    from app.workers.pipeline import _derive_ontology_graph_bundle_passes

    self_mock = MagicMock()
    self_mock.retry.side_effect = CeleryRetry()
    self_mock.request.retries = 0

    # Patch the module-level helpers the nested _terminalize_failure calls,
    # NOT _terminalize_failure itself (it's a closure inside the helper).
    # Also trigger SoftTimeLimitExceeded early by making load_bundle_manifest raise.
    with patch("app.workers.pipeline.load_bundle_manifest",
               side_effect=SoftTimeLimitExceeded()), \
         patch("app.workers.pipeline._update_document_pipeline_status"), \
         patch("app.workers.pipeline._attempt_rollback"), \
         patch("app.workers.pipeline._get_db"):
        with pytest.raises(CeleryRetry):
            _derive_ontology_graph_bundle_passes(
                self_mock, str(uuid.uuid4()), str(uuid.uuid4())
            )

    self_mock.retry.assert_called_once()
    _, kwargs = self_mock.retry.call_args
    assert isinstance(kwargs.get("exc"), SoftTimeLimitExceeded)
```

- [ ] **Step 2: Confirm test fails**

```bash
.venv/bin/pytest tests/unit/test_derive_ontology_graph_bundle_passes.py -v
```

- [ ] **Step 3: Add the `except SoftTimeLimitExceeded` branch BEFORE `except Exception`**

In `_derive_ontology_graph_bundle_passes`, between the existing `except IngestFailed` and `except Exception`:

```python
    except SoftTimeLimitExceeded as exc:
        logger.warning(
            "derive_ontology_graph: soft time limit for run=%s doc=%s — retrying via Celery",
            pipeline_run_id, run_document_id,
        )
        # Mark the summary row FAILED for visibility; let self.retry engage the
        # Celery retry budget (max_retries=2). On exhaustion self.retry raises
        # MaxRetriesExceededError which reaches guard_stage_run (which now
        # terminalizes unconditionally — see Task 0 of this plan).
        try:
            db_soft = _get_db()
            try:
                row = db_soft.get(StageRun, stage_summary_id)
                if row:
                    from datetime import datetime as dt
                    row.status = "FAILED"
                    row.error_message = "soft_time_limit_exceeded"
                    row.finished_at = dt.utcnow()
                db_soft.commit()
            finally:
                db_soft.close()
        except Exception:
            logger.exception("stage_run FAILED write also failed")
        raise self.retry(exc=exc)
```

Note: `self` is the Celery task instance passed positionally by `derive_ontology_graph` — verify the helper signature accepts it.

- [ ] **Step 4: Test + regress**

```bash
.venv/bin/pytest tests/unit/test_derive_ontology_graph_bundle_passes.py tests/pipeline/ -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_derive_ontology_graph_bundle_passes.py app/workers/pipeline.py
git commit -m "fix(pipeline): derive_ontology_graph soft-timeout triggers self.retry()"
```

---

### Task 3: Legacy extraction writes a *valid* `DoclingDocument` stub

**Why:** `derive_document_anchors` calls `DoclingDocument.model_validate(...)` at `app/services/docling_anchors.py:317`. A hand-crafted dict fails validation. The stub must be a real `DoclingDocument`, built via the library API and serialized with `model_dump(...)`.

**Files:**
- Modify: `app/workers/pipeline.py::prepare_document` legacy branch
- Modify: `app/services/docling_anchors.py` (no code change; just depend on the model)
- Create: `tests/pipeline/test_legacy_extraction_artifact.py`
- Reuse: `app/services/storage.py::upload_bytes_sync`

**Stub contract:** a `DoclingDocument` with name = doc UUID, a single text item containing the full extracted markdown, empty pictures/tables/furniture, one page. The test round-trips through both `DoclingDocument.model_validate(json)` AND `docling_anchors.walk(doc)` to prove both consumers work.

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_legacy_extraction_artifact.py`:
```python
"""Legacy extraction must produce a stub docling_document.json that:
  1. Round-trips through DoclingDocument.model_validate()
  2. Is walkable by docling_anchors.walk()
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


def test_stub_validates_as_docling_document():
    from app.workers.pipeline import _build_legacy_docling_document_json

    stub = _build_legacy_docling_document_json(
        document_id="11111111-1111-1111-1111-111111111111",
        text="hello world from a .txt fallback",
    )

    # Contract 1: the stub (a dict) must model_validate as a real DoclingDocument.
    from docling_core.types.doc import DoclingDocument
    doc = DoclingDocument.model_validate(stub)
    assert doc.name == "11111111-1111-1111-1111-111111111111"
    assert any("hello world" in t.text for t in doc.texts)


def test_stub_walks_via_docling_anchors():
    """The downstream walker must not crash on a legacy stub. walk() accepts
    a JSON dict (NOT a validated DoclingDocument) and returns a MergedExtraction.
    Signature from app/services/docling_anchors.py:290:
        walk(docling_doc_json: dict, document_uuid: str,
             pipeline_run_id: str, ontology: dict, *,
             source_storage_key: str | None = None) -> MergedExtraction
    """
    from app.workers.pipeline import _build_legacy_docling_document_json
    from app.services.docling_anchors import walk as _walk

    stub = _build_legacy_docling_document_json(
        document_id="22222222-2222-2222-2222-222222222222",
        text="a tiny markdown body",
    )
    # walk() internally calls DoclingDocument.model_validate(stub) — if the
    # stub shape is wrong, walk() raises during validation. Success here proves
    # both schema validity AND walker tolerance.
    result = _walk(
        stub,
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333",
        {},  # minimal ontology
    )
    assert result is not None  # MergedExtraction is returned
    # Legacy text-only stub has no pictures/tables — walker should return an
    # extraction with zero FIGURE/TABLE entities and possibly zero SECTIONs.
    assert hasattr(result, "entities")
```

- [ ] **Step 2: Confirm failing test**

```bash
.venv/bin/pytest tests/pipeline/test_legacy_extraction_artifact.py -v
```

- [ ] **Step 3: Implement the helper**

Add to `app/workers/pipeline.py` near other `_build_*` helpers:

```python
def _build_legacy_docling_document_json(document_id: str, text: str) -> dict:
    """Construct a minimal, schema-valid DoclingDocument for text-only fallback.

    Legacy extraction in prepare_document is triggered for non-PDF mimes
    (text/plain, etc.) which Docling doesn't support. Downstream stages like
    derive_document_anchors call DoclingDocument.model_validate(...) on the
    stored JSON artifact, so a hand-crafted dict fails validation. This helper
    uses the DoclingDocument API to guarantee schema validity.
    """
    from docling_core.types.doc import DoclingDocument, DocItemLabel

    doc = DoclingDocument(name=str(document_id))
    if text:
        doc.add_text(label=DocItemLabel.TEXT, text=text)
    # Serialize through model_dump so the on-disk form is the same shape
    # DoclingDocument.model_validate produces. mode="json" yields JSON-safe types.
    return doc.export_to_dict()
```

(Call `export_to_dict()` if that is the idiomatic API. If not, use `model_dump(mode="json")`. Confirm which method the installed `docling_core` version offers by running `python -c "from docling_core.types.doc import DoclingDocument; print([m for m in dir(DoclingDocument) if 'dict' in m.lower() or 'dump' in m.lower()])"` before coding.)

In `prepare_document`'s legacy branch (search for `using legacy extraction`, ~line 2868), the existing code uses `_fb_base` and `settings.minio_bucket_derived` — NOT `bucket`/`base_key`. Match that naming. Also `fallback_md` is defined inside the existing broad try/except — move the stub write **outside** that swallowing block so failures propagate.

Concretely, restructure the legacy branch to:

```python
        if mime_type not in _DOCLING_MIMES:
            logger.info("prepare_document: %s not supported by Docling (mime=%s), using legacy extraction", document_id, mime_type)
            _legacy_extract(db, document_id, doc, file_bytes)
            db.commit()

            # Aggregate extracted text once — used for BOTH the legacy markdown
            # and the stub docling_document.json below.
            fallback_md = ""
            try:
                from app.models.ingest import DocumentElement
                from sqlalchemy import select as sql_select
                elems = db.execute(
                    sql_select(DocumentElement.content_text)
                    .where(DocumentElement.document_id == uuid.UUID(document_id))
                    .order_by(DocumentElement.element_order)
                ).scalars().all()
                fallback_md = "\n\n".join(t for t in elems if t and t.strip())
                if fallback_md:
                    fallback_md = _normalize_text(fallback_md)
            except Exception:
                logger.exception("prepare_document: failed to aggregate legacy text for %s", document_id)

            _fb_base = f"artifacts/{document_id}"
            from app.services.storage import upload_bytes_sync

            # Markdown is best-effort (existing behavior) — keep the swallow.
            if fallback_md:
                try:
                    upload_bytes_sync(
                        fallback_md.encode("utf-8"),
                        settings.minio_bucket_derived,
                        f"{_fb_base}/docling_document.md",
                        content_type="text/markdown; charset=utf-8",
                    )
                    logger.info(
                        "prepare_document: persisted legacy markdown for %s (%d chars)",
                        document_id, len(fallback_md),
                    )
                except Exception as _fb_err:
                    logger.warning(
                        "prepare_document: failed to persist legacy markdown for %s: %s",
                        document_id, _fb_err,
                    )

            # Stub docling_document.json is REQUIRED by downstream stages.
            # Always write, even when fallback_md is empty. Do NOT wrap in
            # try/except — failures must propagate so guard_stage_run terminalizes.
            stub = _build_legacy_docling_document_json(document_id, fallback_md or "")
            upload_bytes_sync(
                json.dumps(stub, ensure_ascii=False).encode("utf-8"),
                settings.minio_bucket_derived,
                f"{_fb_base}/docling_document.json",
                content_type="application/json; charset=utf-8",
            )
            logger.info(
                "prepare_document: persisted legacy docling_document.json stub for %s",
                document_id,
            )

            _update_stage_run(db, run_id, "prepare_document", "COMPLETE", attempt=self.request.retries + 1, metrics={"fallback": True, "reason": "unsupported_format"})
            db.commit()
            return document_id
```

- [ ] **Step 4: Test + regression**

```bash
.venv/bin/pytest tests/pipeline/test_legacy_extraction_artifact.py tests/pipeline/ -v
```

- [ ] **Step 5: End-to-end spot-check (manual, not a test)**

Upload a small `.txt` via the API. Verify:
- `derive_document_anchors` does not raise `NoSuchKey`.
- Final `pipeline_status = COMPLETE`.

- [ ] **Step 6: Commit**

```bash
git add tests/pipeline/test_legacy_extraction_artifact.py app/workers/pipeline.py
git commit -m "fix(pipeline): legacy extraction writes schema-valid DoclingDocument stub"
```

---

### Task 4: `batch_create_entity_chunk_edges_sync` retries on ArcadeDB "Record not found" + propagates exhaustion

**Why:** The 2026-04-24 run logged 5+ HTTP 500s with error bodies matching `Record #N:M not found` during EXTRACTED_FROM batch creation. The client raises `httpx.HTTPStatusError` (not `requests.HTTPError`) and the body is at `exc.response.text`. Today the batch helper returns an attempted count (not confirmed), and `derive_structure_links` catches the batch exception and only logs a warning — which is why the failures were silent and docs showed COMPLETE despite missing edges.

**Two-part fix:**
1. In `batch_create_entity_chunk_edges_sync`, wrap each batch in retry-on-RecordNotFound (match regex `Record #\d+:\d+ not found` against `exc.response.text`), up to 3 attempts with 200ms backoff. After exhaustion, **re-raise** the underlying `httpx.HTTPStatusError`. (Confirmed count — counting edges the server acknowledged — is descoped: the current `sqlscript` doesn't `RETURN` edge RIDs, so we'd have to rewrite the script. For this plan, retain the attempted-count return value but make sure failures propagate; confirmed-count is a follow-up.)
2. In `derive_structure_links`, **do not catch and log** batch failures. Let them propagate to `guard_stage_run` which will terminalize the doc (Task 0 ensures this works correctly).

**Files:**
- Modify: `app/services/arcadedb_graph.py::batch_create_entity_chunk_edges_sync` (add retry wrapper around command_sync; keep attempted-count return — confirmed count is a separate follow-up that requires rewriting the batch sqlscript with `LET / RETURN`)
- Modify: `app/workers/pipeline.py::derive_structure_links` (search for `batch_create_entity_chunk_edges_sync` call and its surrounding try/except — remove the log-and-swallow)
- Modify: `tests/unit/test_arcadedb_graph.py` (append — do NOT create separate file)

- [ ] **Step 1: Inspect current batch call + the catch in derive_structure_links**

```bash
grep -nE "batch_create_entity_chunk_edges_sync" app/services/arcadedb_graph.py app/workers/pipeline.py
```

Read the specific function body at `app/services/arcadedb_graph.py:2168` and the catch at `app/workers/pipeline.py:5453` (line numbers per review). Document the current semantics in the commit message.

- [ ] **Step 2: Append failing tests to `tests/unit/test_arcadedb_graph.py`**

```python
class TestBatchEntityChunkEdgeRetry:
    """Class is ArcadeDBGraphStore (capital DB). Method takes EntityChunkEdge
    objects, not dicts. The client already unwraps resp.json()['result'] so
    command_sync returns a list directly."""

    def _http_status_error(self, body_text):
        import httpx
        resp = httpx.Response(
            status_code=500,
            request=httpx.Request("POST", "http://test"),
            text=body_text,
        )
        return httpx.HTTPStatusError("500", request=resp.request, response=resp)

    def _sample_edge(self):
        """EntityChunkEdge fields per app/services/graph_store.py:107:
            entity_name: str
            entity_type: str
            chunk_rid: str
            entity_id: str | None = None
            source_rid: str | None = None
        No target_rid, no document_id — the doc_id is passed to the batch method
        separately, and target is derived from chunk_rid server-side.
        """
        from app.services.graph_store import EntityChunkEdge
        return EntityChunkEdge(
            entity_name="APG-77",
            entity_type="RADAR_SYSTEM",
            chunk_rid="#40:0",
            entity_id="E1",
            source_rid="#37:9",
        )

    def test_retries_on_record_not_found(self):
        from unittest.mock import MagicMock
        from app.services.arcadedb_graph import ArcadeDBGraphStore

        gs = ArcadeDBGraphStore.__new__(ArcadeDBGraphStore)
        gs._database = "test_db"
        gs._client = MagicMock()

        calls = {"n": 0}
        def fake_command_sync(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise self._http_status_error(
                    '{"error":"Error on transaction commit","detail":"Record #37:9 not found"}'
                )
            # Client unwraps resp.json()["result"] — return the list directly
            return [{"@rid": "#50:0"}]

        gs._client.command_sync.side_effect = fake_command_sync
        result = gs.batch_create_entity_chunk_edges_sync(
            document_id="doc-uuid",
            edges=[self._sample_edge()],
        )
        assert calls["n"] == 2  # retried once
        # `result` is the attempted count (confirmed count is a separate follow-up)
        assert result == 1

    def test_raises_after_exhausting_retries(self):
        from unittest.mock import MagicMock
        from app.services.arcadedb_graph import ArcadeDBGraphStore
        import httpx, pytest

        gs = ArcadeDBGraphStore.__new__(ArcadeDBGraphStore)
        gs._database = "test_db"
        gs._client = MagicMock()
        gs._client.command_sync.side_effect = self._http_status_error(
            '{"error":"tx commit","detail":"Record #37:9 not found"}'
        )

        with pytest.raises(httpx.HTTPStatusError):
            gs.batch_create_entity_chunk_edges_sync(
                document_id="doc-uuid",
                edges=[self._sample_edge()],
            )
```

Read the real `EntityChunkEdge` definition + the real `batch_create_entity_chunk_edges_sync` signature at `app/services/arcadedb_graph.py:18` and `:2168` before finalizing the fixture.

- [ ] **Step 3: Implement retry + propagation in the batch helper**

Wrap each per-edge or per-chunk-of-edges command in a retry:
```python
import httpx, re, time
_RECORD_NOT_FOUND = re.compile(r"Record #\d+:\d+\s+not found", re.IGNORECASE)

def _retry_on_record_not_found(fn, *args, max_attempts=3, **kwargs):
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            body = getattr(getattr(exc, "response", None), "text", "") or ""
            if _RECORD_NOT_FOUND.search(body) and attempt + 1 < max_attempts:
                logger.warning(
                    "arcadedb edge: RecordNotFound on attempt %d — retrying",
                    attempt + 1,
                )
                time.sleep(0.2 * (attempt + 1))
                last_exc = exc
                continue
            raise
    # should not reach here, but just in case
    if last_exc:
        raise last_exc
```

Call it around each edge-creation command. Keep the existing attempted-count return value (don't change the method's semantics beyond adding retry + propagation). Confirmed-count (via `LET / RETURN` rewrite of the sqlscript) is a separate follow-up and NOT in scope for this plan.

- [ ] **Step 4: In `derive_structure_links`, remove the batch-exception swallow**

Find the current try/except around `batch_create_entity_chunk_edges_sync` (line ~5448-5453). Remove the `except Exception: logger.warning(...)` so that exceptions propagate naturally into `guard_stage_run`.

- [ ] **Step 5: Run tests + regression**

```bash
.venv/bin/pytest tests/unit/test_arcadedb_graph.py tests/pipeline/ -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_arcadedb_graph.py app/services/arcadedb_graph.py app/workers/pipeline.py
git commit -m "fix(graph): batch edge creation retries on NotFound + propagates exhaustion"
```

---

## Chunk 3: Operations — split-worker + misc

### Task 5: `./manage.sh --start` defaults to split-worker; update every related flag

**Why:** Single-worker at c=2 bottlenecked the 2026-04-24 run (picture_desc alone consumed 7h16m cumulative). Split mode (`worker-ingest c=3`, `worker-graph c=2`, `worker-embed c=2`) runs 3 docs' picture_desc concurrently instead of 2 and isolates long LLM tasks from short stages. Estimated 40-50% wall-time reduction.

**Reviewer-requested:** `--start` alone is not enough. `--worker-status`, `--restart`, `--logs`, and `--help` must all know the split-mode container names (`eip-mmdpp-worker-ingest-1`, `-embed-1`, `-graph-1` instead of `eip-mmdpp-worker-1`).

**Files:**
- Modify: `manage.sh` — flag semantics + status/restart/logs/help
- Modify: `docker-compose.yml` — confirm split-profile services read `WORKER_*_CONCURRENCY` env vars
- Modify: `.env`, `env.example` — set/expose `WORKER_INGEST_CONCURRENCY=3`, `WORKER_GRAPH_CONCURRENCY=2`, `WORKER_EMBED_CONCURRENCY=2`

- [ ] **Step 1: Inspect current split-profile**

```bash
grep -A8 "worker-ingest:\|worker-graph:\|worker-embed:" docker-compose.yml | head -80
grep -nE "--concurrency" docker-compose.yml | head -10
```

Confirm the split services exist and their concurrency flag references env vars. If any are hardcoded (e.g., `--concurrency=1`), switch to `--concurrency=${WORKER_*_CONCURRENCY:-N}`.

- [ ] **Step 2: Set the env vars**

Add to `.env` and `env.example`:
```
WORKER_INGEST_CONCURRENCY=3
WORKER_GRAPH_CONCURRENCY=2
WORKER_EMBED_CONCURRENCY=2
```

- [ ] **Step 3: Swap `manage.sh` flag semantics**

Near line 470:
```
  --start)          cmd_start split ;;   # default to split-worker mode
  --start-mixed)    cmd_start ;;          # legacy single-worker
  --start-split)    cmd_start split ;;   # alias, backward compat
```

- [ ] **Step 4: Update `--worker-status`, `--restart`, `--logs`, `--help`**

For `--worker-status` (around line 367): instead of `docker compose exec worker celery ...`, iterate over the three split containers and run status for each. Fall back to single-worker if split containers aren't present (detect via `docker ps`).

For `--restart` (around line 265): add awareness of split profile. If user started with split, restart split containers; if mixed, restart single. Easiest: use `docker compose ps --services --filter status=running` to see what's actually up and restart those.

Update `--help` to document: `--start`, `--start-mixed`, `--start-split`, and call out the default change.

- [ ] **Step 5: Manual verification**

```bash
./manage.sh --blow-away     # user confirms
./manage.sh --start
docker ps --filter "name=eip-mmdpp-worker" --format '{{.Names}}'
```
Expected: `eip-mmdpp-worker-ingest-1`, `eip-mmdpp-worker-embed-1`, `eip-mmdpp-worker-graph-1`.

```bash
./manage.sh --worker-status
./manage.sh --restart
./manage.sh --logs worker-ingest
```
All should work without errors.

- [ ] **Step 6: Commit**

```bash
git add manage.sh docker-compose.yml env.example
git commit -m "feat(ops): default to split-worker; update status/restart/logs/help"
```

---

### Task 6: Audit `collect_derivations` StageRun emission

**Why:** `collect_derivations` is in the full chain (`app/workers/pipeline.py:1919`) and referenced by `finalize_document`'s required-stages check (`app/workers/pipeline.py:5602`). But the function body (line 5500+) does not write a successful `stage_runs` row. If `finalize_document` requires a COMPLETE row for it, the run would always claim a missing stage; if it doesn't, the REQUIRED_STAGES list is misleading.

**Files:**
- Modify: `app/workers/pipeline.py::collect_derivations` OR `REQUIRED_STAGES` definition

- [ ] **Step 1: Read both sites**

```bash
grep -nE "REQUIRED_STAGES|collect_derivations" app/workers/pipeline.py | head -20
```

Read function body at ~5500 and `finalize_document`'s check at ~5602. **Recommendation per review:** option (a) — write a COMPLETE stage_run — is cleaner since `finalize_document` already requires the stage. Option (b) (remove from REQUIRED) is a fallback if the function is truly a no-op.

- [ ] **Step 2: Add stage_run write + test**

In `collect_derivations` (line ~5500), at function entry write a `RUNNING` stage_run, and on success write `COMPLETE`:
```python
def collect_derivations(self, document_id: str, run_id: str | None = None) -> None:
    logger.info("collect_derivations: document_id=%s run_id=%s", document_id, run_id)
    db = _get_db()
    try:
        if run_id:
            _update_stage_run(db, run_id, "collect_derivations", "RUNNING",
                              attempt=self.request.retries + 1)
            db.commit()
        # ... existing body ...
        if run_id:
            _update_stage_run(db, run_id, "collect_derivations", "COMPLETE",
                              attempt=self.request.retries + 1)
            db.commit()
    finally:
        db.close()
```

Append a test to `tests/pipeline/test_pipeline_chain.py` (or create a new class within it) asserting that after calling `collect_derivations`, a `stage_runs` row with `stage_name='collect_derivations'` and `status='COMPLETE'` exists for the given `run_id`. Additional assertion: `finalize_document`'s required-stages check no longer reports `collect_derivations` as missing when its row exists.

- [ ] **Step 3: Regression test**

```bash
.venv/bin/pytest tests/pipeline/ tests/unit/test_ingest_pipeline_coverage.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/pipeline/test_pipeline_chain.py app/workers/pipeline.py
git commit -m "fix(pipeline): collect_derivations writes COMPLETE StageRun for finalize check"
```

---

## Chunk 4: End-to-end validation

### Task 7: Full reingest of the 21-doc corpus

**Expected outcomes vs 2026-04-24 baseline:**
- **21/21 COMPLETE** (previous: 19/21 + 1 PARTIAL + 1 stuck PROCESSING)
- **Wall time < 6h** (previous: 10.5h) — driven by split-worker + 3-way VLM concurrency
- **Zero stuck-PROCESSING** docs (Task 0 guarantees terminal state on exception)
- **Zero unrecovered ArcadeDB 500s** (Task 4 retries transparently; surfaces on exhaustion)
- **`.txt` file** completes through `derive_document_anchors` (Task 3 stub)
- **Graph-heavy doc** that needed >6h last run completes within 8h or fails loudly (Task 1+2)

Requires: user does reingest; plan executor only monitors + reports.

- [ ] **Step 1: Push + blow away**

```bash
git push
./manage.sh --blow-away  # user confirms
```

- [ ] **Step 2: Start (split default)**

```bash
./manage.sh --start
```

- [ ] **Step 3: User re-uploads the 21 docs**

- [ ] **Step 4: Monitor**

Re-arm monitors same as prior runs: worker failures, API 4xx/5xx, docling errors, docling-graph errors, ArcadeDB SEVERE, sweeper fires.

- [ ] **Step 5: Post-run audit queries**

```sql
-- Terminal status check
SELECT pipeline_status, retry_count, COUNT(*) FROM ingest.documents GROUP BY 1,2;
-- Expected: 21 COMPLETE / 0 / 21

-- Stage completions
SELECT stage_name, status, COUNT(*) FROM ingest.stage_runs
WHERE pass_name IS NULL GROUP BY 1,2 ORDER BY 1,2;
-- Expected: each stage 21 COMPLETE, 0 FAILED, 0 RUNNING

-- Edge integrity (rough — requires sampling a few docs against ArcadeDB)
-- Pick one COMPLETE doc and confirm its EXTRACTED_FROM count is non-zero and consistent
```

Check worker logs for ArcadeDB 500s that retried successfully (these are expected and acceptable):
```bash
# Count successful retry attempts (informational — some retries are OK)
docker logs eip-mmdpp-worker-graph-1 2>&1 | grep -cE "RecordNotFound on attempt"

# Count UNRECOVERED NotFound errors (should be 0 — a propagated failure would
# have terminalized the doc via guard_stage_run)
docker logs eip-mmdpp-worker-graph-1 2>&1 \
  | grep -E "httpx.HTTPStatusError.*Record #.*not found" \
  | wc -l
```

Check wall time vs 10.5h baseline:
```sql
SELECT ROUND(EXTRACT(EPOCH FROM (MAX(finished_at) - MIN(started_at)))/60.0, 1) AS wall_min
FROM ingest.stage_runs;
```

---

## Acceptance Criteria

1. `guard_stage_run` terminalizes the document to PARTIAL_COMPLETE on **any** unhandled exception (not conditional on retries), verified by the new `test_terminalizes_on_first_failure_for_max_retries_1` case.
2. `_derive_ontology_graph_bundle_passes` handles `SoftTimeLimitExceeded` via `self.retry(exc=exc)`; tests assert `self.retry` was called, not just `_terminalize_failure`.
3. Legacy extraction writes a `DoclingDocument`-valid stub JSON; tests assert `DoclingDocument.model_validate(...)` succeeds AND `docling_anchors.walk(...)` runs without crashing.
4. `batch_create_entity_chunk_edges_sync` retries on `httpx.HTTPStatusError` whose `response.text` matches `Record #\d+:\d+ not found`, up to 3 attempts, and raises on exhaustion. `derive_structure_links` no longer catches this exception.
5. `./manage.sh --start` launches split-worker containers (`worker-ingest c=3`, `worker-graph c=2`, `worker-embed c=2`). `--worker-status`, `--restart`, `--logs`, `--help` all know the split names. `--start-mixed` preserves the old behavior.
6. `collect_derivations` is consistent — either writes a COMPLETE stage_run or is absent from `REQUIRED_STAGES`.
7. Config defaults in `app/config.py` match the `.env` values for graph timeouts, sweeper threshold, and visibility timeout.
8. End-to-end reingest of the 21-doc corpus: 21/21 COMPLETE, wall time <6h, 0 stuck PROCESSING, 0 unrecovered ArcadeDB 500s, 0 NoSuchKey from anchors.

## Rollback

Each task is independently revertible. Priority revert paths:
- **Guard regression** (Task 0) → revert the `except Exception` change; stuck-PROCESSING behavior returns but other bugs don't regress.
- **Edge retry breaks something** (Task 4) → revert `batch_create_entity_chunk_edges_sync` + the catch removal in `derive_structure_links`; silent edge loss returns but docs stop failing loudly.
- **Split-worker default causes ops confusion** (Task 5) → `manage.sh --start-mixed` restores single-worker without reverting any code.

## Notes for the Executor

- **Do Task 0 first** — every other task's tests rely on guard's new unconditional terminalization behavior.
- **Task 3's stub must round-trip through `DoclingDocument.model_validate`** — do not ship a dict literal; use the library API.
- **Task 4's mock must use `httpx.HTTPStatusError` with `response.text`** matching the exact pattern ArcadeDB returns (`Record #N:M not found`), not `requests.HTTPError` and not a generic "Record not found" substring.
- **Task 7 is the only task where wall-time is a meaningful acceptance metric** — earlier tasks are correctness-only.
