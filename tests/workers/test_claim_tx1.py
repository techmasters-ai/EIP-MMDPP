"""Tx-1 CLAIM atomically transitions ledger rows to RUNNING with 6 outcomes."""
import uuid
from sqlalchemy import text
from app.workers.pipeline import _claim_tx1


_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _setup_run_and_stage(db_session, status: str | None = "PENDING"):
    """Create pipeline_run + (optionally) ledger row.

    Includes every NOT NULL column: sources.created_by,
    documents.storage_bucket, documents.storage_key, documents.retry_count.
    """
    src_id, doc_id, run_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text("""
        INSERT INTO ingest.sources (id, name, created_by)
        VALUES (:s, 'test', :u)
    """), {"s": src_id, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, retry_count,
             uploaded_by, pipeline_status)
        VALUES (:d, :s, 'x.pdf', 'application/pdf', 0,
                'test-bucket', 'test/x.pdf', 0,
                :u, 'PROCESSING')
    """), {"d": doc_id, "s": src_id, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run_id, "d": doc_id})
    if status is not None:
        db_session.execute(text("""
            INSERT INTO ingest.stage_runs
                (id, pipeline_run_id, stage_name, attempt, status,
                 task_name, dispatch_attempt)
            VALUES (gen_random_uuid(), :r, 'prepare_document', 1, :st,
                    'app.workers.pipeline.prepare_document', 1)
        """), {"r": run_id, "st": status})
    db_session.commit()
    return str(run_id)


def test_claim_proceed_from_pending(db_session):
    run_id = _setup_run_and_stage(db_session, status="PENDING")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-1", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "proceed"
    assert result.dispatch_attempt == 1
    row = db_session.execute(text(
        "SELECT status, celery_task_id FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.status == "RUNNING"
    assert row.celery_task_id == "t-1"


def test_claim_proceed_from_dispatched(db_session):
    run_id = _setup_run_and_stage(db_session, status="DISPATCHED")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-2", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "proceed"


def test_claim_already_complete_returns_skip_dict(db_session):
    run_id = _setup_run_and_stage(db_session, status="COMPLETE")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-3", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "already_complete"
    assert result.early_result == {
        "stage": "prepare_document",
        "status": "skipped",
        "reason": "already_complete",
    }


def test_claim_concurrent_running_no_retry_returns_none(db_session):
    run_id = _setup_run_and_stage(db_session, status="RUNNING")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-4", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "concurrent_running"
    assert result.early_result is None


def test_claim_celery_retry_proceeds_on_running(db_session):
    """is_celery_retry=True allows re-entry on RUNNING (same task republished)."""
    run_id = _setup_run_and_stage(db_session, status="RUNNING")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-5", is_celery_retry=True)
    db_session.commit()
    assert result.outcome == "proceed"
    row = db_session.execute(text(
        "SELECT celery_task_id FROM ingest.stage_runs WHERE pipeline_run_id = :r"
    ), {"r": run_id}).first()
    assert row.celery_task_id == "t-5"  # overwritten with current attempt's id


def test_claim_terminal_failed_returns_distinct_dict(db_session):
    run_id = _setup_run_and_stage(db_session, status="FAILED")
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-7", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "terminal_failed"
    assert result.early_result == {
        "stage": "prepare_document",
        "status": "terminal_failed",     # distinct from "skipped"
        "reason": "stage_previously_failed",
    }


def test_claim_legacy_no_row(db_session):
    run_id = _setup_run_and_stage(db_session, status=None)  # NO ledger row
    result = _claim_tx1(db_session, run_id, "prepare_document",
                        celery_task_id="t-8", is_celery_retry=False)
    db_session.commit()
    assert result.outcome == "legacy"
    assert result.early_result is None
