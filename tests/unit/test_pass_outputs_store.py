"""Integration tests for app.services.pass_outputs_store.

Uses a real PostgreSQL session (Option A) via the conftest ``db_session``
fixture.  Run with:

    DATABASE_URL_SYNC=postgresql+psycopg2://eip_test:eip_test_secret@localhost:5438/eip_test \
        pytest tests/unit/test_pass_outputs_store.py -v

Each test function covers exactly one behavioral contract.  Parent rows
(Source, Document, PipelineRun) are created inside each test via the shared
``pipeline_run_factory`` conftest fixture; they are rolled back after each test
by the ``db_session`` transaction-rollback fixture.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.services.pass_outputs_store import (
    _is_pipeline_run_fk_violation,
    count_completed_passes,
    count_terminal_passes,
    is_pass_already_resolved,
    load_completed_pass_outputs,
    load_pass_output,
    save_pass_output,
)


def _minimal_response() -> dict:
    return {"status": "ok", "pass": "radar_domain"}


def _full_response() -> dict:
    return {
        "status": "ok",
        "pass": "radar_domain",
        "entities": [{"name": "SA-2", "type": "RADAR_SYSTEM"}],
        "unicode": "日本語テスト",
        "nested": {"deep": [1, 2, {"key": "val"}]},
    }


def _save(
    db: Session,
    pipeline_run_id: uuid.UUID,
    pass_name: str = "radar_domain",
    *,
    execution_status: str = "COMPLETE",
    yield_status: str | None = "HIT",
    skip_reason: str | None = None,
    diagnostics: dict | None = None,
    response: dict | None = None,
    attempt: int = 1,
) -> None:
    save_pass_output(
        db,
        pipeline_run_id=pipeline_run_id,
        stage_run_id=None,
        pass_name=pass_name,
        attempt=attempt,
        execution_status=execution_status,
        skip_reason=skip_reason,
        yield_status=yield_status,
        extract_pass_response=response or _minimal_response(),
        primary_entities_extracted=3,
        bridge_entities_extracted=1,
        relationships_extracted=2,
        relationships_rejected=0,
        diagnostics=diagnostics or {},
        field_provenance=[{"field": "name", "source": "llm"}],
    )
    db.flush()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_then_load_round_trip(db_session: Session, pipeline_run_factory):
    """Full save with all fields → load returns same data; JSON round-trip is faithful."""
    run_id = pipeline_run_factory()
    response = _full_response()

    save_pass_output(
        db_session,
        pipeline_run_id=run_id,
        stage_run_id=None,
        pass_name="radar_domain",
        attempt=2,
        execution_status="COMPLETE",
        skip_reason=None,
        yield_status="HIT",
        extract_pass_response=response,
        primary_entities_extracted=5,
        bridge_entities_extracted=2,
        relationships_extracted=4,
        relationships_rejected=1,
        diagnostics={"retry_exhausted": False, "parser": "json_block"},
        field_provenance=[{"field": "range_km", "source": "llm", "confidence": 0.91}],
    )
    db_session.flush()

    row = load_pass_output(db_session, run_id, "radar_domain")

    assert row is not None
    assert row.pass_name == "radar_domain"
    assert row.attempt == 2
    assert row.execution_status == "COMPLETE"
    assert row.yield_status == "HIT"
    assert row.primary_entities_extracted == 5
    assert row.bridge_entities_extracted == 2
    assert row.relationships_extracted == 4
    assert row.relationships_rejected == 1

    # JSON round-trip — nested dicts, lists, unicode must survive intact
    stored = row.extract_pass_response_json
    assert stored["entities"] == response["entities"]
    assert stored["unicode"] == "日本語テスト"
    assert stored["nested"] == {"deep": [1, 2, {"key": "val"}]}

    assert row.diagnostics_json == {"retry_exhausted": False, "parser": "json_block"}
    assert row.field_provenance_json == [
        {"field": "range_km", "source": "llm", "confidence": 0.91}
    ]


def test_empty_yield_is_completed(db_session: Session, pipeline_run_factory):
    """COMPLETE + EMPTY yield → is_pass_already_resolved=True, count_completed=1.

    Off-domain entity passes legitimately produce no entities; EMPTY yield
    is still a COMPLETE execution and must count as such for merge inputs.
    """
    run_id = pipeline_run_factory()
    _save(db_session, run_id, "missile_kinematics", execution_status="COMPLETE", yield_status="EMPTY")

    assert is_pass_already_resolved(db_session, run_id, "missile_kinematics") is True
    assert count_completed_passes(db_session, run_id) == 1


def test_skipped_pass_with_reason_persists(db_session: Session, pipeline_run_factory):
    """SKIPPED + skip_reason round-trips; resolved=True; completed=0; terminal=1."""
    run_id = pipeline_run_factory()
    save_pass_output(
        db_session,
        pipeline_run_id=run_id,
        stage_run_id=None,
        pass_name="system_links",
        attempt=1,
        execution_status="SKIPPED",
        skip_reason="NO_UPSTREAM_ENDPOINTS",
        yield_status=None,
        extract_pass_response={},
        primary_entities_extracted=None,
        bridge_entities_extracted=None,
        relationships_extracted=None,
        relationships_rejected=None,
        diagnostics={},
        field_provenance=[],
    )
    db_session.flush()

    row = load_pass_output(db_session, run_id, "system_links")
    assert row is not None
    assert row.execution_status == "SKIPPED"
    assert row.skip_reason == "NO_UPSTREAM_ENDPOINTS"

    assert is_pass_already_resolved(db_session, run_id, "system_links") is True
    # SKIPPED does NOT count as COMPLETE — no output for merge
    assert count_completed_passes(db_session, run_id) == 0
    assert count_terminal_passes(db_session, run_id) == 1


def test_failed_pass_persists_with_diagnostics(db_session: Session, pipeline_run_factory):
    """FAILED + retry_exhausted diagnostics round-trips; resolved=True; completed=0; terminal=1."""
    run_id = pipeline_run_factory()
    diagnostics = {"retry_exhausted": True, "last_error": "LLM timeout", "attempts": 3}

    save_pass_output(
        db_session,
        pipeline_run_id=run_id,
        stage_run_id=None,
        pass_name="radar_timing",
        attempt=3,
        execution_status="FAILED",
        skip_reason=None,
        yield_status=None,
        extract_pass_response={},
        primary_entities_extracted=None,
        bridge_entities_extracted=None,
        relationships_extracted=None,
        relationships_rejected=None,
        diagnostics=diagnostics,
        field_provenance=[],
    )
    db_session.flush()

    row = load_pass_output(db_session, run_id, "radar_timing")
    assert row is not None
    assert row.execution_status == "FAILED"
    assert row.attempt == 3
    assert row.diagnostics_json["retry_exhausted"] is True
    assert row.diagnostics_json["attempts"] == 3

    assert is_pass_already_resolved(db_session, run_id, "radar_timing") is True
    assert count_completed_passes(db_session, run_id) == 0
    assert count_terminal_passes(db_session, run_id) == 1


def test_count_terminal_includes_failed_and_skipped(db_session: Session, pipeline_run_factory):
    """Mixed COMPLETE/FAILED/SKIPPED: count_terminal = total; count_completed = COMPLETE only."""
    run_id = pipeline_run_factory()

    _save(db_session, run_id, "radar_domain", execution_status="COMPLETE")
    _save(db_session, run_id, "radar_timing", execution_status="COMPLETE")
    _save(db_session, run_id, "missile_kinematics", execution_status="FAILED",
          yield_status=None, diagnostics={"retry_exhausted": True})
    _save(db_session, run_id, "system_links", execution_status="SKIPPED",
          yield_status=None, skip_reason="NO_UPSTREAM_ENDPOINTS")

    assert count_terminal_passes(db_session, run_id) == 4
    assert count_completed_passes(db_session, run_id) == 2


def test_count_terminal_with_pass_names_filter(db_session: Session, pipeline_run_factory):
    """count_terminal_passes with pass_names only counts matching rows."""
    run_id = pipeline_run_factory()

    _save(db_session, run_id, "radar_domain", execution_status="COMPLETE")
    _save(db_session, run_id, "radar_timing", execution_status="COMPLETE")
    _save(db_session, run_id, "missile_kinematics", execution_status="FAILED",
          yield_status=None, diagnostics={"retry_exhausted": True})
    _save(db_session, run_id, "system_links", execution_status="SKIPPED",
          yield_status=None, skip_reason="NO_UPSTREAM_ENDPOINTS")

    # Only ask about the two entity passes
    result = count_terminal_passes(
        db_session, run_id, pass_names=["radar_domain", "missile_kinematics"]
    )
    assert result == 2

    # Empty list → 0
    assert count_terminal_passes(db_session, run_id, pass_names=[]) == 0

    # Non-existent pass → 0
    assert count_terminal_passes(db_session, run_id, pass_names=["nonexistent"]) == 0


def test_load_completed_excludes_failed_and_skipped(db_session: Session, pipeline_run_factory):
    """load_completed_pass_outputs returns ONLY COMPLETE rows."""
    run_id = pipeline_run_factory()

    _save(db_session, run_id, "radar_domain", execution_status="COMPLETE")
    _save(db_session, run_id, "radar_timing", execution_status="FAILED",
          yield_status=None, diagnostics={"retry_exhausted": True})
    _save(db_session, run_id, "system_links", execution_status="SKIPPED",
          yield_status=None, skip_reason="NO_UPSTREAM_ENDPOINTS")

    completed = load_completed_pass_outputs(db_session, run_id)

    assert set(completed.keys()) == {"radar_domain"}
    assert completed["radar_domain"].execution_status == "COMPLETE"


def test_save_overwrites_same_run_pass(db_session: Session, pipeline_run_factory):
    """Upsert: COMPLETE for (run, pass) then FAILED for same key → FAILED wins."""
    run_id = pipeline_run_factory()

    _save(db_session, run_id, "radar_domain", execution_status="COMPLETE", yield_status="HIT")
    # Defensive: same pass terminalizes again with FAILED (should overwrite)
    _save(db_session, run_id, "radar_domain", execution_status="FAILED",
          yield_status=None, diagnostics={"retry_exhausted": True})

    row = load_pass_output(db_session, run_id, "radar_domain")
    assert row is not None
    assert row.execution_status == "FAILED"
    # Only one row exists (unique constraint enforced by upsert)
    assert count_terminal_passes(db_session, run_id) == 1
    assert count_completed_passes(db_session, run_id) == 0


def test_load_pass_output_returns_none_when_absent(db_session: Session, pipeline_run_factory):
    """load_pass_output returns None when no row exists yet."""
    run_id = pipeline_run_factory()

    result = load_pass_output(db_session, run_id, "nonexistent_pass")
    assert result is None


def test_count_returns_zero_when_no_rows(db_session: Session, pipeline_run_factory):
    """All count helpers return 0 when no pass_output rows exist for the run."""
    run_id = pipeline_run_factory()

    assert count_completed_passes(db_session, run_id) == 0
    assert count_terminal_passes(db_session, run_id) == 0
    assert count_terminal_passes(db_session, run_id, pass_names=["radar_domain"]) == 0
    assert is_pass_already_resolved(db_session, run_id, "radar_domain") is False


# ---------------------------------------------------------------------------
# FK-swallow / cancel-safety tests (Task 10)
# ---------------------------------------------------------------------------


def test_save_swallows_specific_pipeline_run_fk_violation(
    db_session: Session, pipeline_run_factory
):
    """save_pass_output discards the row and returns cleanly when the parent
    pipeline_runs row has been hard-deleted (cancel_document path).

    Uses a mocked db.execute to inject a synthetic ForeignKeyViolation for
    the specific pipeline_run_id FK constraint.  This avoids the real-DB
    approach (which would roll back the outer test transaction and discard
    the pipeline_run_factory setup) while still exercising the full
    save_pass_output try/except + rollback path.

    Verifies:
    - No exception is raised.
    - The call returns None (cleanly).
    - db.rollback() was called exactly once.
    """
    from unittest.mock import patch, MagicMock, call
    from psycopg2.errors import ForeignKeyViolation

    run_id = pipeline_run_factory()

    # Build a synthetic ForeignKeyViolation for the pipeline_run_id FK.
    fake_diag = MagicMock()
    fake_diag.constraint_name = "pipeline_pass_outputs_pipeline_run_id_fkey"
    fake_orig = MagicMock(spec=ForeignKeyViolation)
    fake_orig.diag = fake_diag
    fk_exc = IntegrityError("stmt", {}, fake_orig)
    fk_exc.orig = fake_orig

    # Patch db.execute to raise the FK violation, and spy on db.rollback.
    with patch.object(db_session, "execute", side_effect=fk_exc):
        with patch.object(db_session, "rollback") as mock_rollback:
            result = save_pass_output(
                db_session,
                pipeline_run_id=run_id,
                stage_run_id=None,
                pass_name="radar_timing",
                attempt=1,
                execution_status="COMPLETE",
                skip_reason=None,
                yield_status="HIT",
                extract_pass_response={"status": "ok"},
                primary_entities_extracted=0,
                bridge_entities_extracted=0,
                relationships_extracted=0,
                relationships_rejected=0,
                diagnostics={},
                field_provenance=[],
            )

    # Call returns None (cleanly swallowed) and rollback was called once.
    assert result is None
    mock_rollback.assert_called_once()


def test_helper_returns_true_for_pipeline_run_fk(db_session: Session):
    """_is_pipeline_run_fk_violation returns True for the exact target constraint."""
    from psycopg2.errors import ForeignKeyViolation

    fake_diag = MagicMock()
    fake_diag.constraint_name = "pipeline_pass_outputs_pipeline_run_id_fkey"

    fake_orig = MagicMock(spec=ForeignKeyViolation)
    fake_orig.diag = fake_diag

    exc = IntegrityError("stmt", {}, fake_orig)
    exc.orig = fake_orig

    assert _is_pipeline_run_fk_violation(exc) is True


def test_helper_returns_false_for_other_fk(db_session: Session):
    """_is_pipeline_run_fk_violation returns False for a different FK constraint."""
    from psycopg2.errors import ForeignKeyViolation

    fake_diag = MagicMock()
    fake_diag.constraint_name = "pipeline_pass_outputs_stage_run_id_fkey"

    fake_orig = MagicMock(spec=ForeignKeyViolation)
    fake_orig.diag = fake_diag

    exc = IntegrityError("stmt", {}, fake_orig)
    exc.orig = fake_orig

    assert _is_pipeline_run_fk_violation(exc) is False


def test_helper_returns_false_for_unique_violation(db_session: Session):
    """_is_pipeline_run_fk_violation returns False for a UniqueViolation."""
    from psycopg2.errors import UniqueViolation

    fake_orig = MagicMock(spec=UniqueViolation)

    exc = IntegrityError("stmt", {}, fake_orig)
    exc.orig = fake_orig

    assert _is_pipeline_run_fk_violation(exc) is False
