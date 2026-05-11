"""_seed_first_stage inserts a PENDING ledger row idempotently."""
import uuid
from sqlalchemy import text
from app.workers.pipeline import _seed_first_stage


_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _new_pipeline_run(db_session) -> str:
    """Create a minimal source+document+pipeline_run for testing.

    Includes every NOT NULL column on each table:
    - sources.created_by
    - documents.storage_bucket, documents.storage_key, documents.retry_count
    """
    doc_id = uuid.uuid4()
    run_id = uuid.uuid4()
    src_id = uuid.uuid4()
    db_session.execute(text("""
        INSERT INTO ingest.sources (id, name, created_by)
        VALUES (:s, 'test-source', :u)
    """), {"s": src_id, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key,
             uploaded_by, pipeline_status, retry_count)
        VALUES (:d, :s, 'x.pdf', 'application/pdf', 0,
                'test-bucket', 'test/x.pdf',
                :u, 'PROCESSING', 0)
    """), {"d": doc_id, "s": src_id, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run_id, "d": doc_id})
    db_session.commit()
    return str(run_id)


def test_seed_first_stage_inserts_pending_row(db_session):
    run_id = _new_pipeline_run(db_session)
    _seed_first_stage(
        db_session,
        pipeline_run_id=run_id,
        stage_name="prepare_document",
        task_name="app.workers.pipeline.prepare_document",
    )
    db_session.commit()

    row = db_session.execute(text("""
        SELECT status, attempt, dispatch_attempt, queue_name, task_name,
               available_at IS NOT NULL AS has_available_at
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'prepare_document'
    """), {"r": run_id}).first()
    assert row is not None
    assert row.status == "PENDING"
    assert row.attempt == 1                  # ledger invariant
    assert row.dispatch_attempt == 1
    assert row.queue_name == "ingest"        # resolved via task_routes
    assert row.task_name == "app.workers.pipeline.prepare_document"
    assert row.has_available_at is True


def test_seed_first_stage_is_idempotent(db_session):
    """Calling twice does not create a duplicate row."""
    run_id = _new_pipeline_run(db_session)
    _seed_first_stage(
        db_session,
        pipeline_run_id=run_id,
        stage_name="prepare_document",
        task_name="app.workers.pipeline.prepare_document",
    )
    db_session.commit()
    _seed_first_stage(
        db_session,
        pipeline_run_id=run_id,
        stage_name="prepare_document",
        task_name="app.workers.pipeline.prepare_document",
    )
    db_session.commit()

    count = db_session.execute(text("""
        SELECT COUNT(*) FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'prepare_document' AND pass_name IS NULL
    """), {"r": run_id}).scalar_one()
    assert count == 1
