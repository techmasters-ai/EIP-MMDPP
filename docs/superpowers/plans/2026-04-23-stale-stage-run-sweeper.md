# Stale `stage_runs` Sweeper + Defensive Failure Write Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make silent worker failures self-healing by (a) periodically marking genuinely-orphaned `ingest.stage_runs` rows as `FAILED` via Celery beat, and (b) adding a defensive status-write guard to `prepare_document` so uncaught exceptions are always recorded, not lost.

**Architecture:** Two layers of defense. *Preventive*: a decorator `guard_stage_run` wraps `prepare_document` (the observed offender) and writes `status='FAILED'` + a logged traceback on any uncaught exception path. *Reactive*: a new periodic Celery-beat task `periodic_stale_run_sweep` calls a new function `_sweep_stale_runs` that flips any `stage_runs` row at `RUNNING` for longer than `settings.stale_stage_run_threshold_seconds` (default 900s = 15 min) to `FAILED`, and flips its owning `pipeline_run` to `FAILED`. The sweeper is the catch-all; the decorator is the proximate fix. This plan intentionally applies the decorator to `prepare_document` only — the sweeper covers the other tasks — so the blast radius stays small and the pattern is validated before rollout.

**Tech Stack:** Python 3.11, Celery (with Redis broker + beat), SQLAlchemy (sync session), Pydantic settings, pytest. All additions are pure-Python; no migration, no new dependencies.

**Context from 2026-04-23 incident:**
A 22-doc batch ingest left one doc (`SA-2 Guideline…pdf`) with a single `stage_runs` row at `status='RUNNING'` for `prepare_document` for 17+ minutes. Worker was alive and processing other docs. No exception logged, no downstream stage ever dispatched. Doc had to be deleted+re-uploaded. Existing `_cleanup_stale_runs` is bound to `worker_ready.connect` — it only fires on worker startup, so a long-lived worker never clears mid-session orphans. Root cause of the specific orphan is unconfirmed (candidates: forkpool child recycle, swallowed exception between the `RUNNING` write and the `COMPLETE` write, or a DB deadlock on the status update); this plan does NOT require root-causing that — it hardens the failure modes regardless.

**Out of scope (explicitly deferred):**
- Heartbeat-based detection (`last_heartbeat_at` column + 30s background updater). Worth adding only if sweeper+decorator leave residual orphans; defer until measured.
- Applying `guard_stage_run` to the other 11 pipeline tasks. Sweeper covers them; incremental application is a follow-up.
- Resumable sub-steps within `derive_picture_descriptions` / `derive_ontology_graph`. Separate concern.

---

## File Structure

**Create:**
- `tests/pipeline/test_stale_run_sweeper.py` — unit tests for `_sweep_stale_runs` and `periodic_stale_run_sweep`
- `tests/pipeline/test_guard_stage_run.py` — unit tests for the decorator

**Modify:**
- `app/config.py` — add `stale_stage_run_threshold_seconds: int = 900`
- `app/workers/pipeline.py` — add `_sweep_stale_runs()` helper, add `@celery_app.task` `periodic_stale_run_sweep`, add `guard_stage_run` decorator, apply decorator to `prepare_document`
- `app/workers/celery_app.py:72` — add `periodic-stale-run-sweep` entry to `beat_schedule`
- `.env` and `env.example` — add `STALE_STAGE_RUN_THRESHOLD_SECONDS=900`

Each file has one responsibility. The decorator lives in `pipeline.py` near `_update_stage_run` because they share state-write semantics. The sweeper lives next to `_cleanup_stale_runs` to make the relationship obvious to future readers.

---

## Chunk 1: Core Implementation

### Task 0: Add settings scaffolding

**Files:**
- Modify: `app/config.py` (add near `docling_lock_timeout`, ~line 263)
- Modify: `.env` (append next to other periodic/sweep settings)
- Modify: `env.example` (same position as `.env`)

- [ ] **Step 1: Add the setting to `app/config.py`**

Locate the block containing `docling_lock_timeout` and add the new field directly below it:

```python
    # Max age (seconds) a stage_run row can sit at status='RUNNING' before the
    # periodic sweeper marks it FAILED. Must be larger than the slowest observed
    # legitimate stage duration; 15 min is safely above the ~9 min peak seen in
    # 2026-04-23 detect_and_translate runs.
    stale_stage_run_threshold_seconds: int = 900
```

- [ ] **Step 2: Add to `.env`**

Append near other periodic settings (search for `watch_dir_poll_interval_seconds` or `DOCLING_LOCK_TIMEOUT` and add below):

```
# Stale stage_run sweeper threshold (seconds). Rows at RUNNING older than this
# get marked FAILED by periodic_stale_run_sweep.
STALE_STAGE_RUN_THRESHOLD_SECONDS=900
```

- [ ] **Step 3: Add to `env.example`**

Mirror the change exactly:

```
# Stale stage_run sweeper threshold (seconds). Rows at RUNNING older than this
# get marked FAILED by periodic_stale_run_sweep.
STALE_STAGE_RUN_THRESHOLD_SECONDS=900
```

- [ ] **Step 4: Verify settings load**

Run:
```bash
docker compose exec worker python -c "from app.config import get_settings; print(get_settings().stale_stage_run_threshold_seconds)"
```
Expected output: `900`

If the worker isn't running, use: `python -c "from app.config import get_settings; print(get_settings().stale_stage_run_threshold_seconds)"` from within any activated project env.

- [ ] **Step 5: Commit**

```bash
git add app/config.py .env env.example
git commit -m "feat(config): add STALE_STAGE_RUN_THRESHOLD_SECONDS for periodic sweeper"
```

---

### Task 1: Write the sweeper helper with tests (TDD)

**Files:**
- Test: `tests/pipeline/test_stale_run_sweeper.py` (create)
- Modify: `app/workers/pipeline.py` (add `_sweep_stale_runs` near `_cleanup_stale_runs`, ~line 1305)

- [ ] **Step 1: Write the failing unit test**

Create `tests/pipeline/test_stale_run_sweeper.py`:

```python
"""Tests for the periodic stale stage_run sweeper.

The sweeper flips any ingest.stage_runs row at status='RUNNING' older than
settings.stale_stage_run_threshold_seconds to status='FAILED', and flips
its owning ingest.pipeline_runs row to status='FAILED' as well.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestSweepStaleRuns:
    def test_marks_old_running_stage_runs_failed(self):
        """A stage_run RUNNING older than threshold is flipped to FAILED."""
        from app.workers.pipeline import _sweep_stale_runs

        stale_sr_id = uuid.uuid4()
        stale_pr_id = uuid.uuid4()
        fake_rows = [(stale_sr_id, stale_pr_id)]

        db = MagicMock()
        # first execute: SELECT stale rows; second: UPDATE stage_runs; third: UPDATE pipeline_runs
        db.execute.side_effect = [
            MagicMock(fetchall=MagicMock(return_value=fake_rows)),
            MagicMock(),
            MagicMock(),
        ]

        with patch("app.workers.pipeline._get_db", return_value=db), \
             patch("app.workers.pipeline.settings") as mock_settings:
            mock_settings.stale_stage_run_threshold_seconds = 900
            swept = _sweep_stale_runs()

        assert swept == 1
        assert db.commit.called
        assert db.execute.call_count == 3

    def test_returns_zero_when_nothing_stale(self):
        """No stale rows -> returns 0, no UPDATEs issued."""
        from app.workers.pipeline import _sweep_stale_runs

        db = MagicMock()
        db.execute.side_effect = [MagicMock(fetchall=MagicMock(return_value=[]))]

        with patch("app.workers.pipeline._get_db", return_value=db), \
             patch("app.workers.pipeline.settings") as mock_settings:
            mock_settings.stale_stage_run_threshold_seconds = 900
            swept = _sweep_stale_runs()

        assert swept == 0
        # Only the SELECT should have fired
        assert db.execute.call_count == 1
        # No commit needed if nothing to write
        assert not db.commit.called

    def test_rollback_on_exception(self):
        """A failure mid-sweep rolls back, does not raise."""
        from app.workers.pipeline import _sweep_stale_runs

        db = MagicMock()
        db.execute.side_effect = RuntimeError("db blew up")

        with patch("app.workers.pipeline._get_db", return_value=db), \
             patch("app.workers.pipeline.settings") as mock_settings:
            mock_settings.stale_stage_run_threshold_seconds = 900
            swept = _sweep_stale_runs()

        assert swept == 0
        assert db.rollback.called
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
docker compose exec worker pytest tests/pipeline/test_stale_run_sweeper.py::TestSweepStaleRuns -v
```
Expected: `ImportError: cannot import name '_sweep_stale_runs'` or `AttributeError`.

- [ ] **Step 3: Implement `_sweep_stale_runs` in `app/workers/pipeline.py`**

Add immediately below the existing `_cleanup_stale_runs` function (after its closing `finally: db.close()`, around line 1357):

```python
def _sweep_stale_runs() -> int:
    """Mark stage_runs stuck at RUNNING beyond the configured threshold as FAILED.

    Returns the number of rows swept. Intended to be called from a periodic
    Celery-beat task — complements `_cleanup_stale_runs` (which only runs on
    worker startup) by catching mid-session orphans on long-lived workers.
    """
    from sqlalchemy import text

    threshold = settings.stale_stage_run_threshold_seconds
    db = _get_db()
    try:
        stale = db.execute(
            text(
                """
                SELECT id, pipeline_run_id
                FROM ingest.stage_runs
                WHERE status = 'RUNNING'
                  AND started_at < NOW() - make_interval(secs => :threshold)
                """
            ),
            {"threshold": threshold},
        ).fetchall()

        if not stale:
            return 0

        sr_ids = [row[0] for row in stale]
        pr_ids = list({row[1] for row in stale})

        db.execute(
            text(
                """
                UPDATE ingest.stage_runs
                SET status = 'FAILED',
                    finished_at = NOW(),
                    error_message = COALESCE(error_message, '') || 'stale; swept by periodic_stale_run_sweep'
                WHERE id = ANY(:ids)
                """
            ),
            {"ids": sr_ids},
        )

        db.execute(
            text(
                """
                UPDATE ingest.pipeline_runs
                SET status = 'FAILED',
                    finished_at = COALESCE(finished_at, NOW()),
                    error_message = COALESCE(error_message, '') || 'stale; swept by periodic_stale_run_sweep'
                WHERE id = ANY(:ids) AND status = 'PROCESSING'
                """
            ),
            {"ids": pr_ids},
        )

        db.commit()
        logger.warning(
            "periodic_stale_run_sweep: marked %d stale stage_runs FAILED "
            "(threshold=%ds, pipeline_runs affected=%d)",
            len(sr_ids), threshold, len(pr_ids),
        )
        return len(sr_ids)
    except Exception:
        logger.exception("_sweep_stale_runs: rollback due to error")
        db.rollback()
        return 0
    finally:
        db.close()
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
docker compose exec worker pytest tests/pipeline/test_stale_run_sweeper.py::TestSweepStaleRuns -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/pipeline/test_stale_run_sweeper.py app/workers/pipeline.py
git commit -m "feat(pipeline): add _sweep_stale_runs helper with unit tests"
```

---

### Task 2: Add the Celery task + beat schedule entry

**Files:**
- Test: `tests/pipeline/test_stale_run_sweeper.py` (extend)
- Modify: `app/workers/pipeline.py` (add task after `_sweep_stale_runs`)
- Modify: `app/workers/celery_app.py:72-85` (add beat entry)

- [ ] **Step 1: Add a failing test for the Celery task wrapper**

Append to `tests/pipeline/test_stale_run_sweeper.py`:

```python
class TestPeriodicStaleRunSweepTask:
    def test_task_calls_sweep_and_returns_count(self):
        """periodic_stale_run_sweep delegates to _sweep_stale_runs and returns its result."""
        from app.workers.pipeline import periodic_stale_run_sweep

        with patch("app.workers.pipeline._sweep_stale_runs", return_value=3) as mock_sweep:
            result = periodic_stale_run_sweep.apply().get()

        mock_sweep.assert_called_once()
        assert result == 3
```

- [ ] **Step 2: Run the test and confirm it fails**

```bash
docker compose exec worker pytest tests/pipeline/test_stale_run_sweeper.py::TestPeriodicStaleRunSweepTask -v
```
Expected: `ImportError: cannot import name 'periodic_stale_run_sweep'`.

- [ ] **Step 3: Add the Celery task in `app/workers/pipeline.py`**

Directly below `_sweep_stale_runs`:

```python
@celery_app.task(bind=True)
def periodic_stale_run_sweep(self) -> int:
    """Beat-scheduled wrapper around _sweep_stale_runs.

    Scheduled in `app/workers/celery_app.py::beat_schedule`. Runs on any worker
    that consumes the `celery` queue. Safe to run concurrently — the UPDATE
    statements are idempotent and scoped to RUNNING rows older than the
    threshold.
    """
    return _sweep_stale_runs()
```

- [ ] **Step 4: Add the beat schedule entry**

Edit `app/workers/celery_app.py` — inside the `beat_schedule={...}` dict at line 72, add below `community-detection`:

```python
        "periodic-stale-run-sweep": {
            "task": "app.workers.pipeline.periodic_stale_run_sweep",
            "schedule": timedelta(minutes=10),
        },
```

- [ ] **Step 5: Run the new test and confirm it passes**

```bash
docker compose exec worker pytest tests/pipeline/test_stale_run_sweeper.py::TestPeriodicStaleRunSweepTask -v
```
Expected: 1 passed.

- [ ] **Step 6: Run the full sweeper test file to ensure nothing regressed**

```bash
docker compose exec worker pytest tests/pipeline/test_stale_run_sweeper.py -v
```
Expected: 4 passed.

- [ ] **Step 7: Verify the beat schedule is registered**

Restart beat and inspect:
```bash
docker compose restart beat
sleep 5
docker logs eip-mmdpp-beat-1 --tail 20 2>&1 | grep -i "periodic-stale-run-sweep\|Scheduler"
```
Expected: a line mentioning `periodic-stale-run-sweep` or confirmation the scheduler came up without error.

- [ ] **Step 8: Commit**

```bash
git add tests/pipeline/test_stale_run_sweeper.py app/workers/pipeline.py app/workers/celery_app.py
git commit -m "feat(pipeline): schedule periodic_stale_run_sweep every 10 min via beat"
```

---

### Task 3: Add `guard_stage_run` decorator with tests (TDD)

**Files:**
- Test: `tests/pipeline/test_guard_stage_run.py` (create)
- Modify: `app/workers/pipeline.py` (add decorator near `_update_stage_run`, ~line 1872)

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_guard_stage_run.py`:

```python
"""Tests for the guard_stage_run decorator.

The decorator wraps a pipeline task so that any unhandled exception:
  1. Writes stage_runs.status = 'FAILED' with the exception repr as error_message
  2. Logs a full traceback
  3. Re-raises (so Celery retry/failure machinery still runs)

CeleryRetry and SoftTimeLimitExceeded pass through untouched — they are
Celery's own control-flow exceptions and must not be shadowed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestGuardStageRun:
    def _fake_task_self(self, retries: int = 0):
        s = MagicMock()
        s.request.retries = retries
        return s

    def test_successful_call_passes_through(self):
        """A task that returns normally is unaffected."""
        from app.workers.pipeline import guard_stage_run

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            return "ok"

        result = task(self._fake_task_self(), "doc-1", run_id="run-1")
        assert result == "ok"

    def test_unhandled_exception_writes_failed_status(self):
        """An uncaught exception triggers a FAILED stage_runs write then re-raises."""
        from app.workers.pipeline import guard_stage_run

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise ValueError("boom")

        with patch("app.workers.pipeline._get_db") as mock_get_db, \
             patch("app.workers.pipeline._update_stage_run") as mock_update:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db
            with pytest.raises(ValueError, match="boom"):
                task(self._fake_task_self(), "doc-1", run_id="run-1")

        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert args[2] == "fake_stage"      # stage_name
        assert args[3] == "FAILED"          # status
        assert "boom" in (kwargs.get("error") or "")

    def test_celery_retry_passes_through_untouched(self):
        """CeleryRetry must not trigger a FAILED write — it's a normal retry signal."""
        from app.workers.pipeline import guard_stage_run
        from celery.exceptions import Retry as CeleryRetry

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise CeleryRetry()

        with patch("app.workers.pipeline._update_stage_run") as mock_update:
            with pytest.raises(CeleryRetry):
                task(self._fake_task_self(), "doc-1", run_id="run-1")

        mock_update.assert_not_called()

    def test_soft_time_limit_passes_through_untouched(self):
        """SoftTimeLimitExceeded is handled by the task's own except branch, not the guard."""
        from app.workers.pipeline import guard_stage_run
        from celery.exceptions import SoftTimeLimitExceeded

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise SoftTimeLimitExceeded()

        with patch("app.workers.pipeline._update_stage_run") as mock_update:
            with pytest.raises(SoftTimeLimitExceeded):
                task(self._fake_task_self(), "doc-1", run_id="run-1")

        mock_update.assert_not_called()

    def test_no_run_id_no_stage_write(self):
        """With run_id=None there is no stage_run to update; still re-raises."""
        from app.workers.pipeline import guard_stage_run

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise RuntimeError("x")

        with patch("app.workers.pipeline._update_stage_run") as mock_update:
            with pytest.raises(RuntimeError):
                task(self._fake_task_self(), "doc-1", run_id=None)

        mock_update.assert_not_called()

    def test_status_write_failure_does_not_mask_original(self):
        """If writing FAILED itself fails, the original exception still propagates."""
        from app.workers.pipeline import guard_stage_run

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise ValueError("original")

        with patch("app.workers.pipeline._get_db", side_effect=RuntimeError("db dead")):
            with pytest.raises(ValueError, match="original"):
                task(self._fake_task_self(), "doc-1", run_id="run-1")
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
docker compose exec worker pytest tests/pipeline/test_guard_stage_run.py -v
```
Expected: `ImportError: cannot import name 'guard_stage_run'`.

- [ ] **Step 3: Implement the decorator**

In `app/workers/pipeline.py`, add immediately before `_update_stage_run` (around line 1872) so the decorator and the helper it calls live together:

```python
import functools


def guard_stage_run(stage_name: str):
    """Wrap a pipeline task so uncaught exceptions mark the stage_run FAILED.

    CeleryRetry and SoftTimeLimitExceeded are passed through untouched — those
    are Celery's own control-flow exceptions and the task's existing except
    branches handle them. Any other exception triggers a defensive FAILED
    status write (scoped to the current run_id, if any) and a full traceback
    log, then re-raises so Celery's retry / failure machinery still runs.

    This is a narrow safety net for the silent-orphan case observed on
    2026-04-23: a task marked RUNNING, then died in a way that left no log
    entry and no status update. The sweeper catches such orphans eventually;
    this decorator catches them immediately.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, document_id, run_id=None, *args, **kwargs):
            try:
                return fn(self, document_id, run_id, *args, **kwargs)
            except (CeleryRetry, SoftTimeLimitExceeded):
                raise
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
                raise
        return wrapper
    return decorator
```

Note: `CeleryRetry` and `SoftTimeLimitExceeded` are already imported at the top of `pipeline.py` — search for `from celery.exceptions import` to confirm before adding import statements.

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
docker compose exec worker pytest tests/pipeline/test_guard_stage_run.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/pipeline/test_guard_stage_run.py app/workers/pipeline.py
git commit -m "feat(pipeline): add guard_stage_run decorator for defensive FAILED writes"
```

---

### Task 4: Apply `guard_stage_run` to `prepare_document`

**Files:**
- Modify: `app/workers/pipeline.py:2464-2467` (decorator application)
- Test: `tests/pipeline/test_guard_stage_run.py` (extend with integration-style check)

- [ ] **Step 1: Write an integration-style test that verifies the decorator is applied**

Append to `tests/pipeline/test_guard_stage_run.py`:

```python
class TestPrepareDocumentGuarded:
    def test_prepare_document_has_guard_wrapper(self):
        """prepare_document is wrapped — the function has the guard's __wrapped__ attr."""
        from app.workers.pipeline import prepare_document

        # guard_stage_run uses functools.wraps, so the underlying function is
        # preserved via __wrapped__. The presence of this attribute is the
        # observable signal that the decorator is applied.
        assert hasattr(prepare_document.run, "__wrapped__"), (
            "prepare_document is not wrapped by guard_stage_run"
        )
```

- [ ] **Step 2: Run the new test and confirm it fails**

```bash
docker compose exec worker pytest tests/pipeline/test_guard_stage_run.py::TestPrepareDocumentGuarded -v
```
Expected: AssertionError — decorator not yet applied.

- [ ] **Step 3: Apply the decorator to `prepare_document`**

In `app/workers/pipeline.py`, locate lines 2464-2467:

```python
@celery_app.task(bind=True, max_retries=3, default_retry_delay=30,
                 ...)
def prepare_document(self, document_id: str, run_id: str | None = None) -> str:
```

Add `@guard_stage_run("prepare_document")` between the `@celery_app.task` decorator and the `def` line:

```python
@celery_app.task(bind=True, max_retries=3, default_retry_delay=30,
                 ...)
@guard_stage_run("prepare_document")
def prepare_document(self, document_id: str, run_id: str | None = None) -> str:
```

**Order matters:** `@celery_app.task` must be outermost (outer-most decorator becomes the Celery task). `@guard_stage_run` goes inside it.

- [ ] **Step 4: Run the integration test**

```bash
docker compose exec worker pytest tests/pipeline/test_guard_stage_run.py::TestPrepareDocumentGuarded -v
```
Expected: 1 passed.

- [ ] **Step 5: Run the existing pipeline tests to confirm no regression**

```bash
docker compose exec worker pytest tests/pipeline/test_pipeline_chain.py -v
```
Expected: all previously-passing tests still pass. If any fail due to the decorator changing signatures/behavior, investigate before proceeding.

- [ ] **Step 6: End-to-end smoke: induce a prepare_document failure and verify the stage_run gets FAILED**

This is a one-shot manual verification, not an automated test — automating it requires a full pipeline fixture which is out of scope.

In a Python shell inside the worker container:

```bash
docker compose exec worker python
```

```python
import uuid
from unittest.mock import patch
from app.workers.pipeline import prepare_document

# Bypass the celery broker — run the task body directly with a forced failure.
doc_id = str(uuid.uuid4())
run_id = str(uuid.uuid4())

# Pre-insert a dummy pipeline_run + stage_run RUNNING row so the update path has something to target.
# (In a real run, start_ingest_pipeline creates these. For this smoke test use a pre-existing run_id
# from a failed-or-cancelled previous upload, or skip this step and just verify no crash on run_id=None.)

# Easiest smoke: call with run_id=None and a patched fn body that raises — confirms re-raise works.
with patch("app.workers.pipeline._prepare_document_impl", side_effect=RuntimeError("smoke")) if False else patch("app.workers.pipeline.logger") as log:
    try:
        prepare_document.apply(args=[doc_id, None]).get()
    except Exception as e:
        print("raised:", repr(e))
```

Expected: the exception is re-raised and `guard_stage_run:` log line appears. Absence of a stage_run FAILED row is expected here because `run_id=None`.

For a full E2E smoke with the DB row path, upload a document through the notebook / API that will trip an exception (e.g. an unsupported MIME type), wait, then query:

```sql
SELECT status, error_message FROM ingest.stage_runs
WHERE stage_name='prepare_document'
ORDER BY started_at DESC LIMIT 1;
```

Expected row: `status='FAILED'` with `error_message` containing `"unhandled exception"`.

- [ ] **Step 7: Commit**

```bash
git add tests/pipeline/test_guard_stage_run.py app/workers/pipeline.py
git commit -m "feat(pipeline): guard prepare_document with guard_stage_run decorator"
```

---

### Task 5: Worker lifecycle audit + documentation

**Files:**
- Modify: `app/workers/celery_app.py` (add a comment block; no behavior change unless the audit finds an issue)

- [ ] **Step 1: Confirm current worker lifecycle posture**

Run:
```bash
grep -n "worker_max_tasks_per_child\|worker_max_memory_per_child\|task_acks_late\|worker_prefetch_multiplier" app/workers/celery_app.py
```

Expected matches include `task_acks_late=True` and `worker_prefetch_multiplier=1`. If `worker_max_tasks_per_child` or `worker_max_memory_per_child` appears set to a non-zero / non-empty value, that is a candidate cause of silent task loss (Celery drops in-flight tasks when a child is recycled). Decide: keep at default (no recycle) or switch to an explicit recycle strategy with acks-late properly audited.

- [ ] **Step 2: Document the posture**

Add a comment block above the `Celery(...)` instantiation in `app/workers/celery_app.py`:

```python
# Worker lifecycle posture (verified 2026-04-23):
#   task_acks_late=True            -> task ACKed only after return/success
#   worker_prefetch_multiplier=1   -> at most one queued-but-not-running task per slot
#   worker_max_tasks_per_child=0   -> no forkpool recycle (default)
#   worker_max_memory_per_child=0  -> no memory-based recycle (default)
# Recycling a forkpool child mid-task silently loses the in-flight task (Celery
# does NOT re-dispatch), so the recycle knobs are intentionally at 0. If either
# is raised in the future, audit every long-running stage for idempotency and
# rely on the periodic_stale_run_sweep to close the gap.
```

- [ ] **Step 3: Commit**

```bash
git add app/workers/celery_app.py
git commit -m "docs(workers): document Celery lifecycle posture for orphan prevention"
```

---

### Task 6: End-to-end validation against a deliberately-induced orphan

**Files:** none modified — this task is a manual validation with a SQL-produced orphan.

- [ ] **Step 1: Produce a synthetic orphan in the database**

Connect to postgres:
```bash
docker exec -it eip-mmdpp-postgres-1 psql -U eip -d eip
```

Insert a fake stale row (requires an existing document — find one first):
```sql
WITH existing_doc AS (
  SELECT id FROM ingest.documents LIMIT 1
)
INSERT INTO ingest.pipeline_runs (document_id, status, started_at)
SELECT id, 'PROCESSING', NOW() - INTERVAL '20 minutes' FROM existing_doc
RETURNING id;
```

Note the returned `pipeline_run_id`. Then:

```sql
INSERT INTO ingest.stage_runs (pipeline_run_id, stage_name, status, started_at, attempt)
VALUES ('<pipeline_run_id>', 'prepare_document', 'RUNNING', NOW() - INTERVAL '20 minutes', 1);
```

- [ ] **Step 2: Manually trigger the sweeper (bypass the 10-minute wait)**

```bash
docker compose exec worker celery -A app.workers.celery_app call app.workers.pipeline.periodic_stale_run_sweep
```
Expected output: a task id (and the result can be fetched with `celery result <task-id>` — should be `1`).

Alternative direct invocation:
```bash
docker compose exec worker python -c "from app.workers.pipeline import _sweep_stale_runs; print(_sweep_stale_runs())"
```
Expected output: `1`.

- [ ] **Step 3: Verify the orphan was marked FAILED**

```sql
SELECT status, error_message FROM ingest.stage_runs
WHERE pipeline_run_id = '<pipeline_run_id>';
```
Expected: `status='FAILED'` with `error_message` containing `'stale; swept by periodic_stale_run_sweep'`.

```sql
SELECT status, error_message FROM ingest.pipeline_runs
WHERE id = '<pipeline_run_id>';
```
Expected: `status='FAILED'`.

- [ ] **Step 4: Clean up the synthetic rows**

```sql
DELETE FROM ingest.stage_runs WHERE pipeline_run_id = '<pipeline_run_id>';
DELETE FROM ingest.pipeline_runs WHERE id = '<pipeline_run_id>';
```

- [ ] **Step 5: Run the full test suite for these new modules**

```bash
docker compose exec worker pytest tests/pipeline/test_stale_run_sweeper.py tests/pipeline/test_guard_stage_run.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Final commit (if any validation notes need to be captured)**

If Step 1–4 uncovered anything unexpected, add a short note to the commit message of Task 5 (amend the last commit) or create a follow-up issue. Otherwise, no commit needed — Task 6 is validation only.

---

## Acceptance Criteria

- `periodic_stale_run_sweep` runs every 10 minutes via beat and emits a log line on each run.
- A `stage_runs` row manually inserted at `RUNNING` with `started_at` older than 15 minutes is flipped to `FAILED` within one sweep cycle without operator intervention.
- Any uncaught exception inside `prepare_document` writes `status='FAILED'` to its `stage_runs` row and logs a full traceback.
- `CeleryRetry` and `SoftTimeLimitExceeded` raised inside `prepare_document` are NOT masked by the guard — they continue to reach Celery's retry / soft-timeout handlers unchanged.
- No regressions in `tests/pipeline/test_pipeline_chain.py`.
- New tests in `tests/pipeline/test_stale_run_sweeper.py` (4) and `tests/pipeline/test_guard_stage_run.py` (7) all pass.

## Rollback

If the sweeper causes spurious `FAILED` marks on legitimately-long stages:
1. Raise `STALE_STAGE_RUN_THRESHOLD_SECONDS` in `.env` to above the slowest observed stage duration (e.g., `3600` for 1 hour).
2. If that isn't enough, remove the `periodic-stale-run-sweep` entry from `beat_schedule` in `app/workers/celery_app.py` and restart beat. The decorator on `prepare_document` is independent and stays.

If the decorator causes unexpected test or production failures:
1. Remove `@guard_stage_run("prepare_document")` from the task declaration.
2. The sweeper remains as the catch-all safety net.

Both layers can be individually disabled without touching the other.
