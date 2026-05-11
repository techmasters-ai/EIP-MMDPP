"""Tx-3 atomically inserts the successor PENDING row and marks self COMPLETE."""
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.workers._stage_lifecycle import _LifecycleCtx
from app.workers.pipeline import _tx3_complete_and_enqueue_next

_TEST_USER = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def patched_get_db(db_session, monkeypatch):
    """Make `_get_db()` return a sub-session sharing the test's Connection.

    The default `_get_db` (`get_sync_session`) opens a fresh pool connection,
    which cannot see rows the test's `db_session` fixture wrote inside its
    uncommitted outer transaction and instead deadlocks on the parent-row
    FK lock. Joining the test connection via `join_transaction_mode=
    "create_savepoint"` makes the helper's `with db.begin():` resolve to a
    SAVEPOINT, so Tx-3 still exercises real atomic-rollback semantics while
    the test's rollback cleans everything up.
    """
    bind = db_session.get_bind()  # underlying Connection (transaction-joined)

    def fake_get_db():
        return Session(bind=bind, join_transaction_mode="create_savepoint")

    monkeypatch.setattr("app.workers.pipeline._get_db", fake_get_db)


def _setup(db_session, stage="purge_document_derivations"):
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
    # Seed self as RUNNING (post-CLAIM state) — Tx-3 flips it to COMPLETE.
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt, started_at)
        VALUES (gen_random_uuid(), :r, :s, 1, 'RUNNING',
                'app.workers.pipeline.purge_document_derivations', 1, NOW())
    """), {"r": run_id, "s": stage})
    db_session.flush()
    return str(run_id)


def test_tx3_atomic_complete_and_successor_insert(db_session, patched_get_db):
    run_id = _setup(db_session, stage="purge_document_derivations")
    ctx = _LifecycleCtx(
        pipeline_run_id=run_id,
        stage_name="purge_document_derivations",
        dispatch_attempt=1,
        intercept_terminal=True,
        next_stage="derive_picture_descriptions",
        next_task="app.workers.pipeline.derive_picture_descriptions",
        pending_metrics={"some_metric": 7},
    )
    _tx3_complete_and_enqueue_next(ctx)

    # Self row → COMPLETE with metrics persisted.
    self_row = db_session.execute(text("""
        SELECT status, metrics, finished_at IS NOT NULL AS done
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'purge_document_derivations'
    """), {"r": run_id}).first()
    assert self_row.status == "COMPLETE"
    assert self_row.metrics == {"some_metric": 7}
    assert self_row.done

    # Successor row inserted PENDING with task_name + queue resolved.
    # Note: derive_picture_descriptions has queue="ingest" on its decorator,
    # NOT default "celery" (verified during Task 6 implementation).
    next_row = db_session.execute(text("""
        SELECT status, task_name, queue_name, attempt, dispatch_attempt
        FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_picture_descriptions'
    """), {"r": run_id}).first()
    assert next_row.status == "PENDING"
    assert next_row.task_name == "app.workers.pipeline.derive_picture_descriptions"
    assert next_row.queue_name == "ingest"   # actual decorator routing
    assert next_row.attempt == 1
    assert next_row.dispatch_attempt == 1


def test_tx3a_idempotent_on_concurrent_re_run(db_session, patched_get_db):
    """If the successor row already exists, Tx-3a is a no-op via ON CONFLICT."""
    run_id = _setup(db_session, stage="purge_document_derivations")
    db_session.execute(text("""
        INSERT INTO ingest.stage_runs
            (id, pipeline_run_id, stage_name, attempt, status,
             task_name, dispatch_attempt, available_at)
        VALUES (gen_random_uuid(), :r, 'derive_picture_descriptions', 1, 'PENDING',
                'app.workers.pipeline.derive_picture_descriptions', 1, NOW())
    """), {"r": run_id})
    db_session.flush()

    ctx = _LifecycleCtx(
        pipeline_run_id=run_id,
        stage_name="purge_document_derivations",
        dispatch_attempt=1,
        intercept_terminal=True,
        next_stage="derive_picture_descriptions",
        next_task="app.workers.pipeline.derive_picture_descriptions",
    )
    _tx3_complete_and_enqueue_next(ctx)

    count = db_session.execute(text("""
        SELECT COUNT(*) FROM ingest.stage_runs
        WHERE pipeline_run_id = :r AND stage_name = 'derive_picture_descriptions'
    """), {"r": run_id}).scalar_one()
    assert count == 1
