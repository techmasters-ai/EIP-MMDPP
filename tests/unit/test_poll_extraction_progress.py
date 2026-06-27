import pytest
pytestmark = pytest.mark.unit
from unittest.mock import MagicMock, patch

# Session API: poll_extraction_progress uses the synchronous ORM Session
# returned by get_sync_session() as a context manager, and queries via
# session.query(StageRun).filter(...).order_by(...).first() — matching the
# _finalize... helpers in app/workers/pipeline.py (NOT the db.execute(select())
# style used by _has_pending_retry_for_pass). The mock chain below mirrors
# that exact call shape.
#
# STORAGE-LOCATION FIX (root cause): the per-pass StageRun row
# (stage_name='derive_ontology_graph', pass_name=<X>) does NOT exist while a
# pass is running — it is created only at pass COMPLETION by _write_stage_run
# (called from derive_ontology_graph_pass step 5). The summary row
# (pass_name IS NULL, status='RUNNING') is the only row that exists for the
# whole stage, so progress is stored on the SUMMARY row keyed by pass_name:
#     summary.metrics["progress"] = {"<pass_name>": {...}, ...}
# Two concurrent passes share one summary row, so per-pass keying is required
# (a flat shape would let passes clobber each other).


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


def _make_session(summary_rows: dict):
    """Build a fake Session whose query(...).filter(...).order_by(...).first()
    returns the SUMMARY row for the run_id captured from the filter call.

    ``summary_rows`` maps run_id -> fake StageRun (the SUMMARY row).  The poller
    groups passes by run_id, so each distinct run_id yields one .first() call.
    """
    sess = MagicMock()

    # Capture run_id from the StageRun.pipeline_run_id == p["run_id"] filter so
    # the right summary row is returned. The poller filters on
    # pipeline_run_id == run_id, stage_name, pass_name IS NULL, and issues one
    # fresh query(...) per run_id.
    state = {"run_id": None}

    def _query(*args, **kwargs):
        state["run_id"] = None  # fresh query → reset captured run_id
        return q

    def _filter(*args, **kwargs):
        # Find the BinaryExpression pipeline_run_id == <run_id> (NOT the
        # stage_name == ... condition, which also carries a string literal) and
        # pull its right-hand literal value.
        for a in args:
            left = getattr(a, "left", None)
            key = getattr(left, "key", None)
            if key == "pipeline_run_id":
                val = getattr(getattr(a, "right", None), "value", None)
                if isinstance(val, str):
                    state["run_id"] = val
        return q

    q = MagicMock()
    q.filter.side_effect = _filter
    q.order_by.return_value = q
    q.first.side_effect = lambda: summary_rows.get(state["run_id"])
    sess.query.side_effect = _query
    return sess


def test_writes_metrics_progress_on_summary_row():
    """Single pass on one run lands at metrics['progress'][pass_name]; the
    existing top-level metrics are preserved (copy-mutate-reassign)."""
    summary = MagicMock(); summary.metrics = {"existing": 1}
    sess = _make_session({"r1": summary})
    cm = MagicMock(); cm.__enter__.return_value = sess; cm.__exit__.return_value = False
    resp = MagicMock(); resp.json.return_value = {"passes": [
        {"run_id": "r1", "pass_name": "radar_identity", "done": 3, "total": 10,
         "phase": "batches", "updated_at": 123.0}]}
    client_cm = MagicMock(); client_cm.__enter__.return_value.get.return_value = resp
    with patch("app.workers.pipeline.settings") as s, \
         patch("httpx.Client", return_value=client_cm), \
         patch("app.workers.pipeline.get_sync_session", return_value=cm):
        s.dg_progress_poller_enabled = True
        s.docling_graph_base_url = "http://dg:8002"
        s.vector_router_chunk_scope_timeout_s = 5.0
        out = _run()
    # `written` counts pass entries written.
    assert out == {"status": "ok", "written": 1}
    # progress is keyed by pass_name on the SUMMARY row.
    assert summary.metrics["progress"] == {
        "radar_identity": {"done": 3, "total": 10, "phase": "batches", "updated_at": 123.0}
    }
    # top-level metrics preserved (copy-mutate-reassign, not in-place).
    assert summary.metrics["existing"] == 1
    sess.commit.assert_called_once()


def test_two_passes_one_run_both_land_no_clobber():
    """Two concurrent passes on the SAME run must BOTH land under their own
    pass_name key on the shared summary row (regression guard against a flat
    shape that would clobber)."""
    summary = MagicMock(); summary.metrics = {}
    sess = _make_session({"r1": summary})
    cm = MagicMock(); cm.__enter__.return_value = sess; cm.__exit__.return_value = False
    resp = MagicMock(); resp.json.return_value = {"passes": [
        {"run_id": "r1", "pass_name": "radar_identity", "done": 3, "total": 10,
         "phase": "batches", "updated_at": 100.0},
        {"run_id": "r1", "pass_name": "missile_airframe", "done": 1, "total": 4,
         "phase": "merge", "updated_at": 200.0}]}
    client_cm = MagicMock(); client_cm.__enter__.return_value.get.return_value = resp
    with patch("app.workers.pipeline.settings") as s, \
         patch("httpx.Client", return_value=client_cm), \
         patch("app.workers.pipeline.get_sync_session", return_value=cm):
        s.dg_progress_poller_enabled = True
        s.docling_graph_base_url = "http://dg:8002"
        s.vector_router_chunk_scope_timeout_s = 5.0
        out = _run()
    assert out == {"status": "ok", "written": 2}
    prog = summary.metrics["progress"]
    # BOTH passes present — neither clobbered the other.
    assert prog["radar_identity"] == {"done": 3, "total": 10, "phase": "batches", "updated_at": 100.0}
    assert prog["missile_airframe"] == {"done": 1, "total": 4, "phase": "merge", "updated_at": 200.0}


def test_no_per_pass_row_required():
    """When the SUMMARY row is absent for a run, the pass is skipped — but a NO
    per-pass StageRun row is ever required (the poller queries pass_name IS NULL
    only). With a summary row present, the write succeeds without any per-pass
    row existing."""
    summary = MagicMock(); summary.metrics = None  # None metrics handled (dict(None or {}))
    sess = _make_session({"r1": summary})
    cm = MagicMock(); cm.__enter__.return_value = sess; cm.__exit__.return_value = False
    resp = MagicMock(); resp.json.return_value = {"passes": [
        {"run_id": "r1", "pass_name": "radar_identity", "done": 2, "total": 5,
         "phase": "batches", "updated_at": 50.0},
        # This run has NO summary row → skipped (and never needs a per-pass row).
        {"run_id": "r_missing", "pass_name": "other", "done": 1, "total": 2,
         "phase": "batches", "updated_at": 60.0}]}
    client_cm = MagicMock(); client_cm.__enter__.return_value.get.return_value = resp
    with patch("app.workers.pipeline.settings") as s, \
         patch("httpx.Client", return_value=client_cm), \
         patch("app.workers.pipeline.get_sync_session", return_value=cm):
        s.dg_progress_poller_enabled = True
        s.docling_graph_base_url = "http://dg:8002"
        s.vector_router_chunk_scope_timeout_s = 5.0
        out = _run()
    # Only the run with a summary row was written; the missing-row run skipped.
    assert out == {"status": "ok", "written": 1}
    assert summary.metrics["progress"] == {
        "radar_identity": {"done": 2, "total": 5, "phase": "batches", "updated_at": 50.0}
    }
    sess.commit.assert_called_once()
