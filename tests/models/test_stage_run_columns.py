"""StageRun model exposes the new dispatch-ledger columns and DISPATCHED status."""
from app.models.ingest import StageRun, StageRunStatus


def test_stage_run_has_new_columns():
    """ORM mapper recognises all six new columns."""
    cols = {c.name for c in StageRun.__table__.columns}
    assert {
        "queue_name",
        "task_name",
        "celery_task_id",
        "available_at",
        "dispatched_at",
        "dispatch_attempt",
    }.issubset(cols)


def test_stage_run_status_enum_has_dispatched():
    """DISPATCHED is a valid StageRunStatus value."""
    assert StageRunStatus.DISPATCHED.value == "DISPATCHED"
    # Verify all five statuses
    assert {s.value for s in StageRunStatus} == {
        "PENDING", "DISPATCHED", "RUNNING", "COMPLETE", "FAILED"
    }


def test_dispatch_attempt_column_invariants():
    """dispatch_attempt must be NOT NULL with server_default=1.

    Downstream code (Tx-4, sweeper) relies on these invariants; a future edit
    that relaxes either breaks the retry-cap contract silently.
    """
    col = StageRun.__table__.columns["dispatch_attempt"]
    assert col.nullable is False
    assert col.server_default is not None
    # The server_default is a sa.text(...) TextClause; coerce to str to check.
    assert "1" in str(col.server_default.arg)
