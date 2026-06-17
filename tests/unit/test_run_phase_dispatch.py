"""Integration tests for app.services.run_phase_dispatch.

Uses a real PostgreSQL session via the conftest ``db_session`` fixture.
Run with:

    DATABASE_URL_SYNC=postgresql+psycopg2://eip_test:eip_test_secret@localhost:5438/eip_test \
        pytest tests/unit/test_run_phase_dispatch.py -v

Each test covers exactly one behavioral contract of the compare-and-reset state
machine.  Parent rows (Source → Document → PipelineRun) are created via the
shared ``pipeline_run_factory`` conftest fixture and rolled back after each test.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.run_phase_dispatch import (
    PhaseEntry,
    claim_phase,
    is_run_cancelled,
    mark_phase_dispatched,
    mark_phase_terminal,
    read_phase_state,
    reclaim_stale_phase,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PHASE = "entity_pass_radar_identity"


def _seed_phase(db: Session, run_id: uuid.UUID, phase_name: str, entry: dict) -> None:
    """Directly write a phase entry into dispatched_phases, bypassing the helpers.

    Used by tests that need to pre-seed a specific state (e.g., 'dispatched' with
    a back-dated timestamp) without going through the normal lifecycle.
    """
    import json

    db.execute(
        text(
            """
            UPDATE ingest.pipeline_runs
            SET dispatched_phases = jsonb_set(
                dispatched_phases,
                ARRAY[:phase],
                CAST(:entry AS jsonb),
                true
            )
            WHERE id = :run_id
            """
        ),
        {"run_id": str(run_id), "phase": phase_name, "entry": json.dumps(entry)},
    )
    db.flush()


def _read_raw(db: Session, run_id: uuid.UUID, phase_name: str) -> dict | None:
    """Read the raw JSONB dict for a phase (bypasses the dataclass parser)."""
    row = db.execute(
        text(
            "SELECT dispatched_phases->:phase FROM ingest.pipeline_runs WHERE id = :run_id"
        ),
        {"run_id": str(run_id), "phase": phase_name},
    ).fetchone()
    if row is None:
        return None
    return row[0]


# ---------------------------------------------------------------------------
# Phase 1 — Helper unit tests
# ---------------------------------------------------------------------------


def test_claim_first_caller_wins(db_session: Session, pipeline_run_factory):
    """claim_phase returns True; entry has state='claimed', populated claimed_at, null task_id."""
    run_id = pipeline_run_factory()

    result = claim_phase(db_session, run_id, _PHASE)
    db_session.flush()

    assert result is True

    entry = read_phase_state(db_session, run_id, _PHASE)
    assert entry is not None
    assert entry.state == "claimed"
    assert entry.result is None
    assert entry.task_id is None
    assert isinstance(entry.claimed_at, datetime)
    assert entry.dispatched_at is None
    assert entry.completed_at is None


def test_claim_returns_false_if_already_claimed(db_session: Session, pipeline_run_factory):
    """Second claim_phase for the same (run, phase) returns False; entry unchanged."""
    run_id = pipeline_run_factory()

    first = claim_phase(db_session, run_id, _PHASE)
    db_session.flush()
    assert first is True

    before = _read_raw(db_session, run_id, _PHASE)

    second = claim_phase(db_session, run_id, _PHASE)
    db_session.flush()

    assert second is False
    # Entry is bit-for-bit unchanged
    after = _read_raw(db_session, run_id, _PHASE)
    assert after == before


def test_mark_dispatched_advances_state(db_session: Session, pipeline_run_factory):
    """claim → mark_dispatched: state='dispatched', task_id populated, claimed_at preserved."""
    run_id = pipeline_run_factory()

    claim_phase(db_session, run_id, _PHASE)
    db_session.flush()

    claimed_entry = read_phase_state(db_session, run_id, _PHASE)
    claimed_at_before = claimed_entry.claimed_at

    result = mark_phase_dispatched(db_session, run_id, _PHASE, task_id="celery-task-abc123")
    db_session.flush()

    assert result is True

    entry = read_phase_state(db_session, run_id, _PHASE)
    assert entry is not None
    assert entry.state == "dispatched"
    assert entry.task_id == "celery-task-abc123"
    assert isinstance(entry.dispatched_at, datetime)
    assert entry.result is None
    assert entry.completed_at is None
    # claimed_at is preserved across the dispatched transition
    assert entry.claimed_at == claimed_at_before


def test_mark_dispatched_no_op_when_state_changed(db_session: Session, pipeline_run_factory):
    """mark_phase_dispatched with expected_state='claimed' when actual state='completed' → False."""
    run_id = pipeline_run_factory()

    # Pre-seed directly as 'completed' to simulate a state-change race
    _seed_phase(
        db_session,
        run_id,
        _PHASE,
        {
            "state": "completed",
            "result": "succeeded",
            "task_id": "old-task",
            "claimed_at": datetime.now(timezone.utc).isoformat(),
            "dispatched_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    result = mark_phase_dispatched(
        db_session, run_id, _PHASE, task_id="new-task", expected_state="claimed"
    )
    db_session.flush()

    assert result is False

    # Entry unchanged — task_id is still 'old-task', state still 'completed'
    entry = read_phase_state(db_session, run_id, _PHASE)
    assert entry.state == "completed"
    assert entry.task_id == "old-task"


def test_mark_terminal_records_result_succeeded(db_session: Session, pipeline_run_factory):
    """claim → dispatch → terminal(succeeded): state='completed', result='succeeded'."""
    run_id = pipeline_run_factory()

    claim_phase(db_session, run_id, _PHASE)
    db_session.flush()
    mark_phase_dispatched(db_session, run_id, _PHASE, task_id="t1")
    db_session.flush()

    result = mark_phase_terminal(db_session, run_id, _PHASE, result="succeeded")
    db_session.flush()

    assert result is True

    entry = read_phase_state(db_session, run_id, _PHASE)
    assert entry is not None
    assert entry.state == "completed"
    assert entry.result == "succeeded"
    assert isinstance(entry.completed_at, datetime)


def test_mark_terminal_records_result_failed(db_session: Session, pipeline_run_factory):
    """Full lifecycle with result='failed'."""
    run_id = pipeline_run_factory()

    claim_phase(db_session, run_id, _PHASE)
    db_session.flush()
    mark_phase_dispatched(db_session, run_id, _PHASE, task_id="t2")
    db_session.flush()

    result = mark_phase_terminal(db_session, run_id, _PHASE, result="failed")
    db_session.flush()

    assert result is True
    entry = read_phase_state(db_session, run_id, _PHASE)
    assert entry.state == "completed"
    assert entry.result == "failed"
    assert entry.completed_at is not None


def test_mark_terminal_records_result_skipped(db_session: Session, pipeline_run_factory):
    """Full lifecycle with result='skipped'."""
    run_id = pipeline_run_factory()

    claim_phase(db_session, run_id, _PHASE)
    db_session.flush()
    mark_phase_dispatched(db_session, run_id, _PHASE, task_id="t3")
    db_session.flush()

    result = mark_phase_terminal(db_session, run_id, _PHASE, result="skipped")
    db_session.flush()

    assert result is True
    entry = read_phase_state(db_session, run_id, _PHASE)
    assert entry.state == "completed"
    assert entry.result == "skipped"


def test_mark_terminal_idempotent(db_session: Session, pipeline_run_factory):
    """Second mark_phase_terminal returns False; entry unchanged from first call."""
    run_id = pipeline_run_factory()

    claim_phase(db_session, run_id, _PHASE)
    db_session.flush()
    mark_phase_dispatched(db_session, run_id, _PHASE, task_id="t4")
    db_session.flush()

    first = mark_phase_terminal(db_session, run_id, _PHASE, result="succeeded")
    db_session.flush()
    assert first is True

    entry_after_first = _read_raw(db_session, run_id, _PHASE)

    second = mark_phase_terminal(db_session, run_id, _PHASE, result="failed")
    db_session.flush()
    assert second is False

    # Entry is unchanged — result is still 'succeeded', not 'failed'
    entry_after_second = _read_raw(db_session, run_id, _PHASE)
    assert entry_after_second == entry_after_first
    assert entry_after_second["result"] == "succeeded"


# ---------------------------------------------------------------------------
# Phase 2 — Reclaim tests
# ---------------------------------------------------------------------------


def test_reclaim_stale_claimed_compare_and_reset(db_session: Session, pipeline_run_factory):
    """Stale claimed entry is removed; fresh claim with recent timestamp is NOT removed."""
    run_id = pipeline_run_factory()

    # Pre-seed as 'claimed' with claimed_at 60 seconds ago
    stale_claimed_at = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    _seed_phase(
        db_session,
        run_id,
        _PHASE,
        {
            "state": "claimed",
            "result": None,
            "task_id": None,
            "claimed_at": stale_claimed_at,
            "dispatched_at": None,
            "completed_at": None,
        },
    )

    # Reclaim with threshold=30s — the entry is 60s old, so it should be removed
    reclaimed = reclaim_stale_phase(
        db_session, run_id, _PHASE, claim_threshold_s=30, dispatch_threshold_s=3600
    )
    db_session.flush()
    assert reclaimed is True

    # Phase should be absent now
    raw = _read_raw(db_session, run_id, _PHASE)
    assert raw is None

    # Now claim again with a fresh timestamp (now)
    won = claim_phase(db_session, run_id, _PHASE)
    db_session.flush()
    assert won is True

    # Fresh claim's claimed_at is ~now; reclaim with threshold=30s should NOT remove it
    reclaimed_again = reclaim_stale_phase(
        db_session, run_id, _PHASE, claim_threshold_s=30, dispatch_threshold_s=3600
    )
    db_session.flush()
    assert reclaimed_again is False

    # Entry still present
    entry = read_phase_state(db_session, run_id, _PHASE)
    assert entry is not None
    assert entry.state == "claimed"


def test_reclaim_stale_dispatched_revokes_and_resets(
    db_session: Session, pipeline_run_factory, monkeypatch
):
    """Stale dispatched entry: celery revoke is called with the task_id; phase removed.

    Two-phase positive+negative: first call with stale dispatched_at succeeds;
    second call with fresh dispatched_at returns False without firing revoke or
    removing the entry.
    """
    run_id = pipeline_run_factory()

    # --- Positive case: dispatched_at 2 hours ago, threshold 1 hour ---
    stale_dispatched_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    claimed_at = (datetime.now(timezone.utc) - timedelta(hours=2, minutes=1)).isoformat()
    _seed_phase(
        db_session,
        run_id,
        _PHASE,
        {
            "state": "dispatched",
            "result": None,
            "task_id": "abc",
            "claimed_at": claimed_at,
            "dispatched_at": stale_dispatched_at,
            "completed_at": None,
        },
    )

    # Mock the celery_app to capture the revoke call. reclaim_stale_phase
    # imports celery_app lazily (`from app.workers.celery_app import celery_app`
    # inside the function), so there is no module-level
    # app.services.run_phase_dispatch.celery_app attribute to patch — patch the
    # source module attribute the lazy import binds against.
    mock_celery = MagicMock()
    monkeypatch.setattr("app.workers.celery_app.celery_app", mock_celery)

    reclaimed = reclaim_stale_phase(
        db_session,
        run_id,
        _PHASE,
        claim_threshold_s=30,
        dispatch_threshold_s=3600,
    )
    db_session.flush()

    assert reclaimed is True

    # Verify revoke was called with the correct task_id
    mock_celery.control.revoke.assert_called_once_with("abc", terminate=True, signal="SIGTERM")

    # Phase should be absent
    raw = _read_raw(db_session, run_id, _PHASE)
    assert raw is None

    # --- Negative case: re-seed with a fresh dispatched_at (10s ago), same threshold ---
    fresh_dispatched_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    fresh_claimed_at = (datetime.now(timezone.utc) - timedelta(seconds=15)).isoformat()
    _seed_phase(
        db_session,
        run_id,
        _PHASE,
        {
            "state": "dispatched",
            "result": None,
            "task_id": "fresh-task",
            "claimed_at": fresh_claimed_at,
            "dispatched_at": fresh_dispatched_at,
            "completed_at": None,
        },
    )

    mock_celery.reset_mock()

    reclaimed_again = reclaim_stale_phase(
        db_session,
        run_id,
        _PHASE,
        claim_threshold_s=30,
        dispatch_threshold_s=3600,
    )
    db_session.flush()

    assert reclaimed_again is False
    # revoke must NOT have been called for the fresh entry
    mock_celery.control.revoke.assert_not_called()
    # Entry still present and unchanged
    entry = read_phase_state(db_session, run_id, _PHASE)
    assert entry is not None
    assert entry.state == "dispatched"
    assert entry.task_id == "fresh-task"


# ---------------------------------------------------------------------------
# Phase 3 — Read + cancel
# ---------------------------------------------------------------------------


def test_read_phase_state_returns_none_when_absent(db_session: Session, pipeline_run_factory):
    """read_phase_state returns None for a run that exists but has no phase entry."""
    run_id = pipeline_run_factory()

    result = read_phase_state(db_session, run_id, _PHASE)
    assert result is None


def test_read_phase_state_returns_none_when_run_missing(db_session: Session):
    """Pass a run_id that doesn't exist; verify None (no exception)."""
    result = read_phase_state(db_session, uuid.uuid4(), _PHASE)
    assert result is None


def test_read_phase_state_round_trips_dataclass(db_session: Session, pipeline_run_factory):
    """read_phase_state parses all fields correctly from the JSONB store."""
    run_id = pipeline_run_factory()

    now = datetime.now(timezone.utc)
    claimed_at = now - timedelta(minutes=5)
    dispatched_at = now - timedelta(minutes=3)
    completed_at = now - timedelta(minutes=1)

    _seed_phase(
        db_session,
        run_id,
        _PHASE,
        {
            "state": "completed",
            "result": "succeeded",
            "task_id": "task-xyz-789",
            "claimed_at": claimed_at.isoformat(),
            "dispatched_at": dispatched_at.isoformat(),
            "completed_at": completed_at.isoformat(),
        },
    )

    entry = read_phase_state(db_session, run_id, _PHASE)

    assert isinstance(entry, PhaseEntry)
    assert entry.state == "completed"
    assert entry.result == "succeeded"
    assert entry.task_id == "task-xyz-789"
    # Timestamps round-trip through ISO-8601 — compare up to microseconds
    assert abs((entry.claimed_at - claimed_at).total_seconds()) < 0.001
    assert abs((entry.dispatched_at - dispatched_at).total_seconds()) < 0.001
    assert abs((entry.completed_at - completed_at).total_seconds()) < 0.001


def test_is_run_cancelled_true_when_status_failed(db_session: Session, pipeline_run_factory):
    """is_run_cancelled returns True for a run with status='FAILED'."""
    run_id = pipeline_run_factory(status="FAILED")

    assert is_run_cancelled(db_session, run_id) is True


def test_is_run_cancelled_true_when_run_missing(db_session: Session):
    """is_run_cancelled returns True for a UUID that does not exist in pipeline_runs."""
    nonexistent_id = uuid.uuid4()

    assert is_run_cancelled(db_session, nonexistent_id) is True


def test_is_run_cancelled_false_when_status_processing(db_session: Session, pipeline_run_factory):
    """is_run_cancelled returns False for a run with status='PROCESSING'."""
    run_id = pipeline_run_factory(status="PROCESSING")

    assert is_run_cancelled(db_session, run_id) is False
