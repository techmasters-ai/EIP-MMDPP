"""_run_dispatch_tick claims PENDING ledger rows and publishes Celery tasks."""
import uuid
from unittest.mock import patch
from sqlalchemy import text
from app.workers.dispatcher import _run_dispatch_tick

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _seed_pending_ledger_row(db_session, stage="prepare_document", run_status="PROCESSING"):
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, retry_count,
             uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',0,:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, :st)
    """), {"r": run, "d": doc, "st": run_status})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, queue_name, available_at, dispatch_attempt)
        VALUES (gen_random_uuid(), :r, :s, 1, 'PENDING',
                'app.workers.pipeline.prepare_document', 'ingest', NOW(), 1)
    """), {"r": run, "s": stage})
    db_session.flush()
    return str(run)


def test_claims_pending_row_and_publishes(db_session, patched_get_db):
    run_id = _seed_pending_ledger_row(db_session)

    with patch("app.workers.dispatcher._publish") as pub:
        result = _run_dispatch_tick()
        assert pub.called

    assert result["claimed"] >= 1
    row = db_session.execute(text(
        "SELECT status, dispatched_at FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "DISPATCHED"
    assert row.dispatched_at is not None


def test_does_not_claim_rows_without_task_name(db_session, patched_get_db):
    """task_name IS NULL -> not a ledger row -> never claimed."""
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, retry_count,
             uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',0,:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status, dispatch_attempt)
        VALUES (gen_random_uuid(), :r, 'unknown_stage', 1, 'PENDING', 1)
    """), {"r": run})
    db_session.flush()

    with patch("app.workers.dispatcher._publish") as pub:
        _run_dispatch_tick()
    row = db_session.execute(text(
        "SELECT status FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run}).first()
    assert row.status == "PENDING"
    pub.assert_not_called()


def test_does_not_claim_when_pipeline_run_not_processing(db_session, patched_get_db):
    run_id = _seed_pending_ledger_row(db_session, run_status="FAILED")
    with patch("app.workers.dispatcher._publish") as pub:
        _run_dispatch_tick()
    row = db_session.execute(text(
        "SELECT status FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "PENDING"
    pub.assert_not_called()


def test_does_not_claim_pass_name_rows(db_session, patched_get_db):
    """pass_name IS NOT NULL -> per-pass row -> dispatcher must ignore."""
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, retry_count,
             uploaded_by, pipeline_status)
        VALUES (:d,:s,'x.pdf','application/pdf',0,'b','k',0,:u,'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             pass_name, task_name, dispatch_attempt)
        VALUES (gen_random_uuid(), :r, 'derive_ontology_graph', 1, 'PENDING',
                'radar_modulation',
                'app.workers.pipeline.derive_ontology_graph_pass', 1)
    """), {"r": run})
    db_session.flush()

    with patch("app.workers.dispatcher._publish") as pub:
        _run_dispatch_tick()
    pub.assert_not_called()


def test_respects_available_at_future(db_session, patched_get_db):
    run_id = _seed_pending_ledger_row(db_session)
    db_session.execute(text(
        "UPDATE ingest.stage_runs SET available_at = NOW() + INTERVAL '60 seconds' "
        "WHERE pipeline_run_id = :r"
    ), {"r": run_id})
    db_session.flush()
    with patch("app.workers.dispatcher._publish") as pub:
        _run_dispatch_tick()
    pub.assert_not_called()


def test_undo_claim_on_publish_failure(db_session, patched_get_db):
    run_id = _seed_pending_ledger_row(db_session)

    def boom(*a, **kw):
        raise RuntimeError("broker down")

    with patch("app.workers.dispatcher._publish", side_effect=boom):
        _run_dispatch_tick()

    row = db_session.execute(text(
        "SELECT status, error_message FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "PENDING"
    assert "broker down" in (row.error_message or "")
