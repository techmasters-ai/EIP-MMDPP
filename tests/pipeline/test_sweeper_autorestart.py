"""Tests for the auto-restart behavior of _sweep_stale_runs.

Contract: on sweep of a stale row, mark stage_run + pipeline_run FAILED,
bump documents.retry_count, COMMIT, then call start_ingest_pipeline(doc_id)
if under cap (the main-session commit is required so the dispatch guard in
start_ingest_pipeline sees the FAILED state from its own session). On
dispatch failure, run a compensating transaction in a fresh session.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestSweeperAutorestart:
    """Mock SQL sequence must match _sweep_stale_runs() body.

    Per-row execute calls in the main db session:
      1. SELECT fetchall -> [(sr_id, pr_id, doc_id, stage_name), ...]
      2. UPDATE stage_runs                                              (per row)
      3. UPDATE pipeline_runs, read .rowcount                           (per row)
      4. UPDATE documents ... RETURNING retry_count, read .scalar()     (per row; skipped if step 3 rowcount=0)
      5. (If over cap:) UPDATE documents (mark permanent FAILED)        (per row over cap)

    After the main-session commit, dispatch loop:
      6. start_ingest_pipeline(doc_id) (patched)
      7. (If dispatch raises) _get_db() returns a fresh MagicMock for the
         compensation transaction — UPDATE + commit/close.
    """

    def _main_db(self, rows, rowcounts, new_retry_counts, max_retry=3):
        """Build the mock db used by the primary transaction.

        `rowcounts` = list[int] aligned to `rows`, value each UPDATE pipeline_runs returns.
        `new_retry_counts` = list[int|None] aligned to `rows`, value each UPDATE RETURNING returns.
        """
        db = MagicMock()
        effects = [MagicMock(fetchall=MagicMock(return_value=rows))]
        for row, rowcount, new_rc in zip(rows, rowcounts, new_retry_counts):
            effects.append(MagicMock())  # UPDATE stage_runs
            effects.append(MagicMock(rowcount=rowcount))  # UPDATE pipeline_runs
            if rowcount == 0:
                continue  # skip retry_count bump
            effects.append(MagicMock(scalar=MagicMock(return_value=new_rc)))
            if new_rc is None:
                continue
            if new_rc > max_retry:
                effects.append(MagicMock())  # mark permanent FAILED
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
        # Dispatch must happen AFTER the main-session commit so the dispatch
        # guard sees our pipeline_run FAILED transition.
        assert db.commit.called
        mock_dispatch.assert_called_once()
        (dispatched_doc,), _ = mock_dispatch.call_args
        assert str(dispatched_doc) == str(doc_id)

    def test_sweep_marks_permanently_failed_at_cap(self):
        from app.workers.pipeline import _sweep_stale_runs

        doc_id = uuid.uuid4()
        rows = [(uuid.uuid4(), uuid.uuid4(), doc_id, "derive_picture_descriptions")]
        # bump returns 4 which > max_retry=3 -> permanent FAILED path
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
        """pipeline_runs UPDATE returns rowcount=0 if another sweep already flipped
        it to FAILED; we must not double-bump retry_count or double-dispatch."""
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
        """If start_ingest_pipeline raises after the main commit, run a
        compensating transaction in a fresh session that reverts retry_count
        and marks the document FAILED."""
        from app.workers.pipeline import _sweep_stale_runs

        doc_id = uuid.uuid4()
        rows = [(uuid.uuid4(), uuid.uuid4(), doc_id, "prepare_document")]
        main_db = self._main_db(rows, rowcounts=[1], new_retry_counts=[1])
        comp_db = MagicMock()

        get_db_returns = iter([main_db, comp_db])

        with patch("app.workers.pipeline._get_db", side_effect=lambda: next(get_db_returns)), \
             patch("app.workers.pipeline.settings") as mock_settings, \
             patch("app.workers.pipeline.start_ingest_pipeline", side_effect=RuntimeError("dispatch broke")):
            mock_settings.stale_stage_run_threshold_seconds = 27000
            mock_settings.max_doc_retry_count = 3
            _sweep_stale_runs()

        assert comp_db.execute.called
        assert comp_db.commit.called
