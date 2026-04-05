"""Unit tests for community Celery tasks.

All external dependencies (Redis, Postgres, ArcadeDB, Celery) are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lock():
    """Return a MagicMock with release() as a stub — mimics redis.Redis.lock()."""
    lock = MagicMock()
    lock.release = MagicMock()
    return lock


def _mock_settings(**overrides):
    defaults = dict(
        community_detection_enabled=True,
        community_detection_interval_minutes=60,
        celery_broker_url="redis://localhost:6379/0",
    )
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


def _mock_session():
    """SQLAlchemy session mock usable as a context manager."""
    s = MagicMock()
    s.__enter__ = MagicMock(return_value=s)
    s.__exit__ = MagicMock(return_value=False)
    return s


# ---------------------------------------------------------------------------
# run_community_detection_task  (call via .run() — Celery convention)
# ---------------------------------------------------------------------------

def test_task_skips_when_detection_disabled():
    from app.workers.community_tasks import run_community_detection_task

    settings = _mock_settings(community_detection_enabled=False)

    with patch("app.workers.community_tasks.get_settings", return_value=settings):
        result = run_community_detection_task.run(mode="incremental")

    assert result["status"] == "skipped"
    assert result["reason"] == "detection disabled"


def test_task_skips_when_lock_not_acquired():
    from app.workers.community_tasks import run_community_detection_task

    settings = _mock_settings()

    with (
        patch("app.workers.community_tasks.get_settings", return_value=settings),
        patch("app.workers.community_tasks.redis_lock", return_value=None),
    ):
        result = run_community_detection_task.run(mode="incremental")

    assert result["status"] == "skipped"
    assert result["reason"] == "detection already running"


def test_task_releases_lock_on_success():
    from app.workers.community_tasks import run_community_detection_task

    settings = _mock_settings()
    lock = _make_lock()
    detection_result = {
        "status": "COMPLETE",
        "total_communities": 3,
        "reports_generated": 2,
        "reports_reused": 1,
    }

    with (
        patch("app.workers.community_tasks.get_settings", return_value=settings),
        patch("app.workers.community_tasks.redis_lock", return_value=lock),
        patch("app.workers.community_tasks._record_run_start"),
        patch("app.workers.community_tasks._record_run_complete"),
        patch("app.db.session.get_graph_store", return_value=MagicMock()),
        patch("asyncio.run", return_value=detection_result),
    ):
        result = run_community_detection_task.run(mode="incremental", run_id="test-run-id")

    assert result["status"] == "COMPLETE"
    lock.release.assert_called_once()


def test_task_releases_lock_on_failure():
    from app.workers.community_tasks import run_community_detection_task

    settings = _mock_settings()
    lock = _make_lock()

    with (
        patch("app.workers.community_tasks.get_settings", return_value=settings),
        patch("app.workers.community_tasks.redis_lock", return_value=lock),
        patch("app.workers.community_tasks._record_run_start"),
        patch("app.workers.community_tasks._record_run_failed"),
        patch("app.db.session.get_graph_store", return_value=MagicMock()),
        patch("asyncio.run", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError),
    ):
        run_community_detection_task.run(mode="incremental", run_id="test-run-id")

    # Lock must be released even when the task raises
    lock.release.assert_called_once()


# ---------------------------------------------------------------------------
# _record helpers — patch the lazy import target (app.db.session)
# ---------------------------------------------------------------------------

def test_record_run_start_calls_sql():
    from app.workers.community_tasks import _record_run_start

    sess = _mock_session()
    with patch("app.db.session.get_sync_session", return_value=sess):
        _record_run_start("run-123", "incremental")

    sess.execute.assert_called_once()
    sess.commit.assert_called_once()


def test_record_run_complete_calls_sql():
    from app.workers.community_tasks import _record_run_complete

    sess = _mock_session()
    result = {"total_communities": 5, "reports_generated": 4, "reports_reused": 1}
    with patch("app.db.session.get_sync_session", return_value=sess):
        _record_run_complete("run-123", result)

    sess.execute.assert_called_once()
    sess.commit.assert_called_once()


def test_record_run_failed_truncates_long_error():
    """Error messages longer than 1000 chars must be truncated before INSERT."""
    from app.workers.community_tasks import _record_run_failed

    captured_params: dict = {}

    sess = _mock_session()
    sess.execute = MagicMock(side_effect=lambda stmt, params: captured_params.update(params))

    long_error = "x" * 2000
    with patch("app.db.session.get_sync_session", return_value=sess):
        _record_run_failed("run-123", long_error)

    assert len(captured_params["err"]) == 1000
