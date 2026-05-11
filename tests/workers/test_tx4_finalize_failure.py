"""Tx-4 retries within cap, terminalizes past it, propagates to pipeline_run."""
import uuid

from sqlalchemy import text

from app.workers._stage_lifecycle import _LifecycleCtx
from app.workers.pipeline import _tx4_finalize_failure

_TEST_USER = "00000000-0000-0000-0000-000000000001"


def _setup_running(db_session, dispatch_attempt=1, stage="prepare_document"):
    src, doc, run = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    db_session.execute(text(
        "INSERT INTO ingest.sources (id, name, created_by) VALUES (:s,'t',:u)"
    ), {"s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.documents
            (id, source_id, filename, mime_type, file_size_bytes,
             storage_bucket, storage_key, retry_count,
             uploaded_by, pipeline_status)
        VALUES (:d, :s, 'x.pdf', 'application/pdf', 0,
                'test-bucket', 'test/x.pdf', 0,
                :u, 'PROCESSING')
    """), {"d": doc, "s": src, "u": _TEST_USER})
    db_session.execute(text("""
        INSERT INTO ingest.pipeline_runs (id, document_id, status)
        VALUES (:r, :d, 'PROCESSING')
    """), {"r": run, "d": doc})
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt, started_at)
        VALUES (gen_random_uuid(), :r, :s, 1, 'RUNNING',
                'app.workers.pipeline.prepare_document', :da, NOW())
    """), {"r": run, "s": stage, "da": dispatch_attempt})
    db_session.flush()  # use flush, not commit, since we share transaction
    return str(run)


def _ctx(run_id, stage="prepare_document", dispatch_attempt=1):
    """Build a ctx with the dispatch_attempt the row was inserted with."""
    return _LifecycleCtx(
        pipeline_run_id=run_id, stage_name=stage,
        dispatch_attempt=dispatch_attempt, intercept_terminal=True,
        next_stage=None, next_task=None,
    )


def test_tx4_retryable_bumps_dispatch_attempt_and_sets_pending(db_session, patched_get_db):
    run_id = _setup_running(db_session, dispatch_attempt=1)
    _tx4_finalize_failure(
        _ctx(run_id, dispatch_attempt=1),
        error="boom",
        celery_retries=0,
        max_retries=0,
        backoff_seconds=60,
    )
    row = db_session.execute(text("""
        SELECT status, dispatch_attempt, attempt,
               started_at IS NULL AS started_cleared,
               dispatched_at IS NULL AS dispatched_cleared,
               available_at > NOW() AS in_future,
               error_message LIKE '%boom%' AS has_err
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'prepare_document'
    """), {"r": run_id}).first()
    assert row.status == "PENDING"
    assert row.dispatch_attempt == 2
    assert row.attempt == 1               # ledger invariant: attempt never mutates
    assert row.started_cleared
    assert row.dispatched_cleared
    assert row.in_future
    assert row.has_err


def test_tx4_terminal_at_cap_marks_failed_and_propagates_to_pipeline_run(
    db_session, patched_get_db, monkeypatch
):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "max_stage_dispatches", 3)

    # dispatch_attempt=3 already → next=4 > cap=3 → terminal
    run_id = _setup_running(db_session, dispatch_attempt=3)
    _tx4_finalize_failure(
        _ctx(run_id, dispatch_attempt=3), error="exhausted",
        celery_retries=0, max_retries=0, backoff_seconds=60,
    )
    row = db_session.execute(text("""
        SELECT status, dispatch_attempt, finished_at IS NOT NULL AS done
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r
    """), {"r": run_id}).first()
    assert row.status == "FAILED"
    assert row.dispatch_attempt == 4
    assert row.done

    pr = db_session.execute(text(
        "SELECT status, error_message FROM ingest.pipeline_runs WHERE id = :r"
    ), {"r": run_id}).first()
    assert pr.status == "FAILED"
    assert "exhausted" in (pr.error_message or "")


def test_tx4_does_not_overwrite_already_failed_pipeline_run(
    db_session, patched_get_db, monkeypatch
):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "max_stage_dispatches", 1)
    run_id = _setup_running(db_session, dispatch_attempt=1)
    db_session.execute(text("""
        UPDATE ingest.pipeline_runs
        SET status = 'FAILED', error_message = 'earlier failure'
        WHERE id = :r
    """), {"r": run_id})
    db_session.flush()

    _tx4_finalize_failure(
        _ctx(run_id, dispatch_attempt=1), error="later failure",
        celery_retries=0, max_retries=0, backoff_seconds=60,
    )
    pr = db_session.execute(text(
        "SELECT error_message FROM ingest.pipeline_runs WHERE id = :r"
    ), {"r": run_id}).first()
    assert pr.error_message == "earlier failure"  # WHERE status='PROCESSING' guard preserved earlier error
