"""Unit tests for directory watcher.

Tests _is_file_stable and _scan_directory with mocked filesystem and DB.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

pytest.importorskip("celery", reason="celery not installed")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _is_file_stable
# ---------------------------------------------------------------------------

class TestIsFileStable:
    @patch("app.workers.watcher.time.sleep")
    def test_stable_file_returns_true(self, mock_sleep):
        from app.workers.watcher import _is_file_stable
        mock_path = MagicMock(spec=Path)
        stat_result = MagicMock()
        stat_result.st_size = 1024
        mock_path.stat.return_value = stat_result
        assert _is_file_stable(mock_path) is True
        mock_sleep.assert_called_once()

    @patch("app.workers.watcher.time.sleep")
    def test_growing_file_returns_false(self, mock_sleep):
        from app.workers.watcher import _is_file_stable
        mock_path = MagicMock(spec=Path)
        stat1 = MagicMock()
        stat1.st_size = 100
        stat2 = MagicMock()
        stat2.st_size = 200
        mock_path.stat.side_effect = [stat1, stat2]
        assert _is_file_stable(mock_path) is False

    @patch("app.workers.watcher.time.sleep")
    def test_zero_size_returns_false(self, mock_sleep):
        from app.workers.watcher import _is_file_stable
        mock_path = MagicMock(spec=Path)
        stat_result = MagicMock()
        stat_result.st_size = 0
        mock_path.stat.return_value = stat_result
        assert _is_file_stable(mock_path) is False

    def test_file_not_found_returns_false(self):
        from app.workers.watcher import _is_file_stable
        mock_path = MagicMock(spec=Path)
        mock_path.stat.side_effect = FileNotFoundError("gone")
        assert _is_file_stable(mock_path) is False

    def test_os_error_returns_false(self):
        from app.workers.watcher import _is_file_stable
        mock_path = MagicMock(spec=Path)
        mock_path.stat.side_effect = OSError("disk error")
        assert _is_file_stable(mock_path) is False


# ---------------------------------------------------------------------------
# _scan_directory
# ---------------------------------------------------------------------------

class TestScanDirectory:
    def test_skips_nonexistent_dir(self):
        from app.workers.watcher import _scan_directory
        db = MagicMock()
        watch_dir = MagicMock()
        watch_dir.path = "/nonexistent/path"
        with patch("app.workers.watcher.Path") as MockPath:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = False
            MockPath.return_value = mock_dir
            _scan_directory(db, watch_dir)
        # No files processed, no db calls
        db.add.assert_not_called()

    def test_skips_non_matching_pattern(self):
        from app.workers.watcher import _scan_directory
        db = MagicMock()
        watch_dir = MagicMock()
        watch_dir.path = "/data"
        watch_dir.file_patterns = ["*.pdf"]

        with patch("app.workers.watcher.Path") as MockPath:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            mock_dir.is_dir.return_value = True
            mock_file = MagicMock()
            mock_file.is_file.return_value = True
            mock_file.name = "readme.txt"
            mock_dir.iterdir.return_value = [mock_file]
            MockPath.return_value = mock_dir
            _scan_directory(db, watch_dir)
        db.add.assert_not_called()

    @patch("app.workers.watcher._is_file_stable", return_value=False)
    def test_skips_unstable_file(self, mock_stable):
        from app.workers.watcher import _scan_directory
        db = MagicMock()
        watch_dir = MagicMock()
        watch_dir.path = "/data"
        watch_dir.file_patterns = ["*.pdf"]

        with patch("app.workers.watcher.Path") as MockPath:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            mock_dir.is_dir.return_value = True
            mock_file = MagicMock()
            mock_file.is_file.return_value = True
            mock_file.name = "test.pdf"
            mock_dir.iterdir.return_value = [mock_file]
            MockPath.return_value = mock_dir
            _scan_directory(db, watch_dir)
        db.add.assert_not_called()

    @patch("app.workers.watcher._is_file_stable", return_value=True)
    def test_skips_already_processed_hash(self, mock_stable):
        from app.workers.watcher import _scan_directory
        db = MagicMock()
        watch_dir = MagicMock()
        watch_dir.path = "/data"
        watch_dir.id = "wd-1"
        watch_dir.file_patterns = ["*.pdf"]

        with patch("app.workers.watcher.Path") as MockPath:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            mock_dir.is_dir.return_value = True
            mock_file = MagicMock()
            mock_file.is_file.return_value = True
            mock_file.name = "test.pdf"
            mock_file.read_bytes.return_value = b"pdf data"
            mock_dir.iterdir.return_value = [mock_file]
            MockPath.return_value = mock_dir
            # Return existing WatchLog entry (hash already seen)
            db.execute.return_value.scalar_one_or_none.return_value = MagicMock()
            _scan_directory(db, watch_dir)
        # File was already processed, should not add new records
        db.add.assert_not_called()

    @patch("app.workers.watcher._is_file_stable", return_value=True)
    def test_read_error_skipped(self, mock_stable):
        from app.workers.watcher import _scan_directory
        db = MagicMock()
        watch_dir = MagicMock()
        watch_dir.path = "/data"
        watch_dir.file_patterns = ["*.pdf"]

        with patch("app.workers.watcher.Path") as MockPath:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            mock_dir.is_dir.return_value = True
            mock_file = MagicMock()
            mock_file.is_file.return_value = True
            mock_file.name = "test.pdf"
            mock_file.read_bytes.side_effect = PermissionError("denied")
            mock_dir.iterdir.return_value = [mock_file]
            MockPath.return_value = mock_dir
            _scan_directory(db, watch_dir)
        db.add.assert_not_called()

    @patch("app.workers.watcher._is_file_stable", return_value=True)
    @patch("app.services.storage.upload_bytes_sync")
    @patch("app.workers.pipeline.start_ingest_pipeline")
    def test_dispatch_result_unpacked_into_celery_task_id(
        self, mock_dispatch, mock_upload, mock_stable,
    ):
        """start_ingest_pipeline returns IngestDispatchResult (a dataclass).

        The watcher must store dispatch.celery_task_id as a string on
        Document.celery_task_id — not the dataclass itself, which would
        produce an unusable repr string.
        """
        from app.workers.dispatch_types import IngestDispatchResult
        from app.workers.watcher import _scan_directory

        fake_dispatch = IngestDispatchResult(
            pipeline_run_id="pipeline-run-uuid-001",
            celery_task_id="celery-task-uuid-abc",
        )
        mock_dispatch.return_value = fake_dispatch

        db = MagicMock()
        watch_dir = MagicMock()
        watch_dir.path = "/data"
        watch_dir.id = "wd-1"
        watch_dir.source_id = "source-uuid-1"
        watch_dir.file_patterns = ["*.pdf"]

        # No existing WatchLog entry — file is new.
        db.execute.return_value.scalar_one_or_none.return_value = None
        # db.get(Source, ...) returns a truthy source.
        db.get.return_value = MagicMock()

        with patch("app.workers.watcher.Path") as MockPath:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            mock_dir.is_dir.return_value = True
            mock_file = MagicMock()
            mock_file.is_file.return_value = True
            mock_file.name = "test.pdf"
            mock_file.read_bytes.return_value = b"pdf data"
            mock_dir.iterdir.return_value = [mock_file]
            MockPath.return_value = mock_dir

            _scan_directory(db, watch_dir)

        mock_dispatch.assert_called_once()

        # Scan every UPDATE statement for one that sets celery_task_id and
        # pull the bound parameter value. A correctly-written watcher sets
        # it to the string from dispatch.celery_task_id; the buggy version
        # stores the whole IngestDispatchResult dataclass.
        from sqlalchemy.sql.dml import Update
        found_value = "SENTINEL_NOT_FOUND"
        for call in db.execute.call_args_list:
            if not call.args:
                continue
            stmt = call.args[0]
            if not isinstance(stmt, Update):
                continue
            compiled = stmt.compile()
            params = compiled.params
            if "celery_task_id" in params:
                found_value = params["celery_task_id"]
                break

        assert found_value != "SENTINEL_NOT_FOUND", (
            "No UPDATE statement set celery_task_id."
        )
        assert isinstance(found_value, str), (
            f"celery_task_id stored as {type(found_value).__name__}="
            f"{found_value!r}; expected str from dispatch.celery_task_id."
        )
        assert found_value == "celery-task-uuid-abc", (
            f"celery_task_id = {found_value!r}; expected "
            "'celery-task-uuid-abc' (the string from dispatch.celery_task_id)."
        )
