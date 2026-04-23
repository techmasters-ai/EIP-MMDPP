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
