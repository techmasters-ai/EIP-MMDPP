"""start_ingest_pipeline seeds the first ledger row instead of chain.apply_async."""
import uuid
from unittest.mock import patch
from sqlalchemy import text
from app.workers.pipeline import start_ingest_pipeline

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _create_document(db_session):
    src, doc = uuid.uuid4(), uuid.uuid4()
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
    db_session.commit()
    return str(doc)


def test_seeds_prepare_document_pending_no_chain(db_session, patched_get_db):
    doc_id = _create_document(db_session)

    with patch("app.workers.pipeline.chain") as chain_mock:
        result = start_ingest_pipeline(doc_id)
        chain_mock.assert_not_called()       # old chain path is dead

    # First ledger row exists in PENDING.
    row = db_session.execute(text("""
        SELECT status, task_name, queue_name
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r
    """), {"r": result.pipeline_run_id}).first()
    assert row is not None, "expected a stage_runs row for the new pipeline_run"
    assert row.status == "PENDING"
    assert row.task_name == "app.workers.pipeline.prepare_document"
    assert row.queue_name == "ingest"
    assert result.celery_task_id == ""        # no apply_async happened
