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

    def test_soft_time_limit_terminalizes_on_final_attempt(self):
        """SoftTimeLimitExceeded reaching the guard (i.e. not converted to
        CeleryRetry by the task's own handler) means retries are exhausted.
        Celery 5's self.retry(exc=SoftTimeLimitExceeded) re-raises the
        original exc on exhaustion, so the guard must treat it as terminal
        rather than passing through."""
        from app.workers.pipeline import guard_stage_run
        from celery.exceptions import SoftTimeLimitExceeded

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise SoftTimeLimitExceeded()

        with patch("app.workers.pipeline._get_db"), \
             patch("app.workers.pipeline._update_stage_run"), \
             patch("app.workers.pipeline._terminalize_doc_and_run") as mock_term:
            with pytest.raises(SoftTimeLimitExceeded):
                task(
                    self._fake_task_self(),
                    "11111111-1111-1111-1111-111111111111",
                    run_id="22222222-2222-2222-2222-222222222222",
                )

        mock_term.assert_called_once()

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
        """Reproduces the 0005_wildweasels stuck-PROCESSING bug: task with
        max_retries=1 that raises on first attempt must terminalize."""
        from app.workers.pipeline import guard_stage_run

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise RuntimeError("boom")

        with patch("app.workers.pipeline._get_db"), \
             patch("app.workers.pipeline._update_stage_run"), \
             patch("app.workers.pipeline._terminalize_doc_and_run") as m_term:
            with pytest.raises(RuntimeError):
                task(self._fake_task(retries=0, max_retries=1), self.DOC_ID, run_id=self.RUN_ID)

        m_term.assert_called_once()
        args, _ = m_term.call_args
        assert args == (self.DOC_ID, self.RUN_ID, "PARTIAL_COMPLETE")

    def test_terminalizes_regardless_of_retry_count(self):
        """Reaching the generic except branch means the task chose not to retry
        (didn't call self.retry). Always terminalize."""
        from app.workers.pipeline import guard_stage_run

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise ValueError("nope")

        with patch("app.workers.pipeline._get_db"), \
             patch("app.workers.pipeline._update_stage_run"), \
             patch("app.workers.pipeline._terminalize_doc_and_run") as m_term:
            with pytest.raises(ValueError):
                task(self._fake_task(retries=0, max_retries=99), self.DOC_ID, run_id=self.RUN_ID)

        m_term.assert_called_once()

    def test_celery_retry_still_passes_through_no_terminalization(self):
        from celery.exceptions import Retry as CeleryRetry
        from app.workers.pipeline import guard_stage_run

        @guard_stage_run("fake_stage")
        def task(self_, document_id, run_id=None):
            raise CeleryRetry()

        with patch("app.workers.pipeline._update_stage_run"), \
             patch("app.workers.pipeline._terminalize_doc_and_run") as m_term:
            with pytest.raises(CeleryRetry):
                task(self._fake_task(), self.DOC_ID, run_id=self.RUN_ID)

        m_term.assert_not_called()


class TestTerminalizeDocAndRun:
    """Helper-level tests: preserve existing terminal statuses,
    update BOTH the document and the pipeline_run."""

    DOC_ID = "11111111-1111-1111-1111-111111111111"
    RUN_ID = "22222222-2222-2222-2222-222222222222"

    def test_preserves_existing_failed_document_status(self):
        from unittest.mock import MagicMock
        from app.workers.pipeline import _terminalize_doc_and_run

        db = MagicMock()
        doc = MagicMock(); doc.pipeline_status = "FAILED"
        run = MagicMock(); run.status = "PROCESSING"
        db.get.side_effect = [doc, run]

        with patch("app.workers.pipeline._get_db", return_value=db):
            _terminalize_doc_and_run(self.DOC_ID, self.RUN_ID, "PARTIAL_COMPLETE")

        assert doc.pipeline_status == "FAILED"
        assert run.status == "FAILED"

    def test_preserves_existing_pending_human_review(self):
        """PENDING_HUMAN_REVIEW is terminal — don't downgrade."""
        from unittest.mock import MagicMock
        from app.workers.pipeline import _terminalize_doc_and_run

        db = MagicMock()
        doc = MagicMock(); doc.pipeline_status = "PENDING_HUMAN_REVIEW"
        run = MagicMock(); run.status = "PROCESSING"
        db.get.side_effect = [doc, run]

        with patch("app.workers.pipeline._get_db", return_value=db):
            _terminalize_doc_and_run(self.DOC_ID, self.RUN_ID, "PARTIAL_COMPLETE")

        assert doc.pipeline_status == "PENDING_HUMAN_REVIEW"

    def test_flips_processing_document_and_pipeline_run(self):
        from unittest.mock import MagicMock
        from app.workers.pipeline import _terminalize_doc_and_run

        db = MagicMock()
        doc = MagicMock(); doc.pipeline_status = "PROCESSING"
        run = MagicMock(); run.status = "PROCESSING"
        db.get.side_effect = [doc, run]

        with patch("app.workers.pipeline._get_db", return_value=db):
            _terminalize_doc_and_run(self.DOC_ID, self.RUN_ID, "PARTIAL_COMPLETE")

        assert doc.pipeline_status == "PARTIAL_COMPLETE"
        assert run.status == "FAILED"
        assert run.finished_at is not None
        assert db.commit.called

    def test_preserves_pipeline_run_already_failed(self):
        from unittest.mock import MagicMock
        from app.workers.pipeline import _terminalize_doc_and_run

        db = MagicMock()
        doc = MagicMock(); doc.pipeline_status = "PROCESSING"
        run = MagicMock(); run.status = "FAILED"
        db.get.side_effect = [doc, run]

        with patch("app.workers.pipeline._get_db", return_value=db):
            _terminalize_doc_and_run(self.DOC_ID, self.RUN_ID, "PARTIAL_COMPLETE")

        assert doc.pipeline_status == "PARTIAL_COMPLETE"
        assert run.status == "FAILED"
