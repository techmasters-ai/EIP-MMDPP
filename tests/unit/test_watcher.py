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
