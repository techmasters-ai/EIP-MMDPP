"""DISPATCHED rows older than threshold reset to PENDING; dispatch_attempt unchanged."""
import uuid
from sqlalchemy import text
from app.workers.pipeline import _sweep_stale_runs

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _seed_dispatched(db_session, dispatched_secs_ago=900):
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
             task_name, dispatch_attempt, dispatched_at)
        VALUES (gen_random_uuid(), :r, 'prepare_document', 1, 'DISPATCHED',
                'app.workers.pipeline.prepare_document', 2,
                NOW() - make_interval(secs => :secs))
    """), {"r": run, "secs": dispatched_secs_ago})
    db_session.flush()
    return str(run)


def test_stale_dispatched_resets_to_pending_no_attempt_bump(db_session, patched_get_db):
    run_id = _seed_dispatched(db_session, dispatched_secs_ago=700)
    _sweep_stale_runs()
    row = db_session.execute(text("""
        SELECT status, dispatched_at IS NULL AS cleared,
               dispatch_attempt
        FROM ingest.stage_runs WHERE pipeline_run_id = :r
    """), {"r": run_id}).first()
    assert row.status == "PENDING"
    assert row.cleared
    assert row.dispatch_attempt == 2     # unchanged


def test_fresh_dispatched_not_swept(db_session, patched_get_db):
    run_id = _seed_dispatched(db_session, dispatched_secs_ago=60)
    _sweep_stale_runs()
    row = db_session.execute(text(
        "SELECT status FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "DISPATCHED"
