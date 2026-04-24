"""Tests for the periodic stale stage_run sweeper.

The sweeper flips any ingest.stage_runs row at status='RUNNING' older than
settings.stale_stage_run_threshold_seconds to status='FAILED', and flips
its owning ingest.pipeline_runs row to status='FAILED' as well.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestSweepStaleRuns:
    def test_marks_old_running_stage_runs_failed(self):
        """A stage_run RUNNING older than threshold is flipped to FAILED + pipeline_run FAILED + retry bump."""
        from app.workers.pipeline import _sweep_stale_runs

        stale_sr_id = uuid.uuid4()
        stale_pr_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        # New SELECT returns (stage_run_id, pipeline_run_id, document_id, stage_name)
        fake_rows = [(stale_sr_id, stale_pr_id, doc_id, "prepare_document")]

        db = MagicMock()
        # SELECT -> UPDATE stage_runs -> UPDATE pipeline_runs (rowcount=1) ->
        # UPDATE documents RETURNING retry_count (scalar=1)
        db.execute.side_effect = [
            MagicMock(fetchall=MagicMock(return_value=fake_rows)),
            MagicMock(),
            MagicMock(rowcount=1),
            MagicMock(scalar=MagicMock(return_value=1)),
        ]

        with patch("app.workers.pipeline._get_db", return_value=db), \
             patch("app.workers.pipeline.settings") as mock_settings, \
             patch("app.workers.pipeline.start_ingest_pipeline"):
            mock_settings.stale_stage_run_threshold_seconds = 27000
            mock_settings.max_doc_retry_count = 3
            swept = _sweep_stale_runs()

        assert swept == 1
        assert db.commit.called

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


class TestPeriodicStaleRunSweepTask:
    def test_task_calls_sweep_and_returns_count(self):
        """periodic_stale_run_sweep delegates to _sweep_stale_runs and returns its result."""
        from app.workers.pipeline import periodic_stale_run_sweep

        with patch("app.workers.pipeline._sweep_stale_runs", return_value=3) as mock_sweep:
            result = periodic_stale_run_sweep.apply().get()

        mock_sweep.assert_called_once()
        assert result == 3
