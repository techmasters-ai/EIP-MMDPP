"""Migration 0020 adds dispatch ledger columns + partial index to stage_runs."""
from sqlalchemy import inspect, text


def test_migration_0020_adds_expected_columns(db_session):
    """All six new columns exist with correct types and defaults."""
    inspector = inspect(db_session.bind)
    cols = {c["name"]: c for c in inspector.get_columns("stage_runs", schema="ingest")}

    assert "queue_name" in cols and "VARCHAR" in str(cols["queue_name"]["type"]).upper()
    assert "task_name" in cols and "VARCHAR" in str(cols["task_name"]["type"]).upper()
    assert "celery_task_id" in cols and "VARCHAR" in str(cols["celery_task_id"]["type"]).upper()
    assert "available_at" in cols
    assert "dispatched_at" in cols
    assert "dispatch_attempt" in cols
    assert cols["dispatch_attempt"]["nullable"] is False


def test_migration_0020_creates_partial_index(db_session):
    """Dispatcher's hot-path partial index exists with correct predicate."""
    result = db_session.execute(text("""
        SELECT indexdef FROM pg_indexes
        WHERE schemaname = 'ingest' AND indexname = 'ix_stage_runs_dispatcher_claim'
    """)).scalar_one_or_none()
    assert result is not None
    assert "status = 'PENDING'" in result
    assert "pass_name IS NULL" in result
    assert "task_name IS NOT NULL" in result
