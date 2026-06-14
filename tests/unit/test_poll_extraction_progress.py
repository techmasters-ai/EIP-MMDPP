import pytest
pytestmark = pytest.mark.unit
from unittest.mock import MagicMock, patch

# Session API: poll_extraction_progress uses the synchronous ORM Session
# returned by get_sync_session() as a context manager, and queries via
# session.query(StageRun).filter(...).order_by(...).first() — matching the
# _finalize... helpers in app/workers/pipeline.py (NOT the db.execute(select())
# style used by _has_pending_retry_for_pass). The mock chain below mirrors
# that exact call shape.


def _run():
    from app.workers.pipeline import poll_extraction_progress
    return poll_extraction_progress.run()  # bind=True task; .run() invokes the body


def test_disabled_is_noop():
    with patch("app.workers.pipeline.settings") as s:
        s.dg_progress_poller_enabled = False
        with patch("httpx.Client") as hc:
            assert _run() == {"status": "disabled"}
            hc.assert_not_called()


def test_poll_failure_fail_open():
    with patch("app.workers.pipeline.settings") as s:
        s.dg_progress_poller_enabled = True
        s.docling_graph_base_url = "http://dg:8002"
        s.vector_router_chunk_scope_timeout_s = 5.0
        with patch("httpx.Client", side_effect=Exception("boom")):
            assert _run()["status"] == "poll_failed"


def test_writes_metrics_progress():
    fake_row = MagicMock(); fake_row.metrics = {"existing": 1}
    # build a fake session whose query(...).filter(...).order_by(...).first() -> fake_row
    sess = MagicMock()
    sess.query.return_value.filter.return_value.order_by.return_value.first.return_value = fake_row
    cm = MagicMock(); cm.__enter__.return_value = sess; cm.__exit__.return_value = False
    resp = MagicMock(); resp.json.return_value = {"passes": [
        {"run_id": "r1", "pass_name": "radar_identity", "done": 3, "total": 10, "phase": "batches", "updated_at": 123.0}]}
    client_cm = MagicMock(); client_cm.__enter__.return_value.get.return_value = resp
    with patch("app.workers.pipeline.settings") as s, \
         patch("httpx.Client", return_value=client_cm), \
         patch("app.workers.pipeline.get_sync_session", return_value=cm):
        s.dg_progress_poller_enabled = True
        s.docling_graph_base_url = "http://dg:8002"
        s.vector_router_chunk_scope_timeout_s = 5.0
        out = _run()
    assert out == {"status": "ok", "written": 1}
    # metrics reassigned (copy-mutate-reassign), progress merged, 'existing' preserved
    assert fake_row.metrics["progress"] == {"done": 3, "total": 10, "phase": "batches", "updated_at": 123.0}
    assert fake_row.metrics["existing"] == 1
    sess.commit.assert_called_once()
