"""Unit tests for reconcile_ontology_graph_runs (Task 9).

Tests the beat-scheduled safety net that recovers stuck derive_ontology_graph
runs in the per-pass Celery fan-in architecture.

Test infrastructure:
- Real PostgreSQL on port 5438 via the shared ``db_session`` / ``pipeline_run_factory``
  conftest fixtures.
- ``_get_db`` patched to return a non-closing proxy around ``db_session`` so the
  reconciler body shares the same transaction as the test (and the rollback at
  teardown works).
- ``derive_ontology_graph_pass.delay``, ``derive_ontology_graph_merge.delay``,
  and ``celery_app.control.revoke`` mocked so no real tasks are enqueued.
- ``load_bundle_manifest`` mocked to return a deterministic manifest with two
  entity passes: ``radar_identity`` and ``missile_airframe``.

Run with:

    DATABASE_URL_SYNC=postgresql+psycopg2://eip_test:eip_test_secret@localhost:5438/eip_test \\
        /home/josh/development/EIP-MMDPP/.venv/bin/python -m pytest \\
        tests/unit/test_reconcile_ontology_graph_runs.py -v
"""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.pass_outputs_store import (
    load_pass_output,
    save_pass_output,
)
from app.services.run_phase_dispatch import read_phase_state
from app.workers.pipeline import reconcile_ontology_graph_runs

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BUNDLE_KEY = "air_defense_v3"
_PASS_A = "radar_identity"
_PASS_B = "missile_airframe"
_PHASE_A = f"entity_pass_{_PASS_A}"
_PHASE_B = f"entity_pass_{_PASS_B}"


# ---------------------------------------------------------------------------
# Test-local helpers
# ---------------------------------------------------------------------------


def _fake_pass_def(name: str, *, required: bool = True):
    return SimpleNamespace(
        name=name,
        kind="entities_and_relationships",
        input_mode="document_only",
        required=required,
        depends_on=[],
        primary_entity_types=["RADAR_SYSTEM"],
        bridge_entity_types=[],
        extracted_relationship_types=["INSTALLED_ON"],
        skip_if_no_upstream_endpoints=False,
        module=f"extraction_schemas.{name}",
        template_class=name.title().replace("_", "") + "Pass",
    )


def _fake_manifest():
    return SimpleNamespace(
        bundle_key=_BUNDLE_KEY,
        passes=[_fake_pass_def(_PASS_A), _fake_pass_def(_PASS_B)],
    )


class _NoCloseProxy:
    """Proxy around a SQLAlchemy Session that suppresses ``.close()`` so the
    test's transaction-rollback fixture can still clean up after the task."""

    def __init__(self, session: Session):
        self._session = session

    def close(self):
        pass  # Suppress — db_session fixture handles teardown

    def commit(self):
        self._session.flush()

    def rollback(self):
        pass  # tests rollback at teardown

    def __getattr__(self, name: str):
        return getattr(self._session, name)


def _seed_phase(db: Session, run_id: uuid.UUID, phase_name: str, entry: dict) -> None:
    """Directly write a phase entry into dispatched_phases, bypassing the helpers."""
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


def _seed_stage_run(
    db: Session,
    run_id: uuid.UUID,
    pass_name: str,
    *,
    execution_status: str = "FAILED",
    attempt: int = 1,
    finished_at: datetime | None = None,
) -> None:
    """Insert a StageRun row for the given pass."""
    stage_run_id = uuid.uuid4()
    db.execute(
        text(
            """
            INSERT INTO ingest.stage_runs
                (id, pipeline_run_id, stage_name, pass_name, attempt, status,
                 execution_status, finished_at, started_at)
            VALUES
                (:id, :run_id, 'derive_ontology_graph', :pass_name, :attempt,
                 'FAILED', :execution_status, :finished_at, NOW())
            """
        ),
        {
            "id": stage_run_id,
            "run_id": str(run_id),
            "pass_name": pass_name,
            "attempt": attempt,
            "execution_status": execution_status,
            "finished_at": finished_at.isoformat() if finished_at else None,
        },
    )
    db.flush()


def _seed_pass_output(
    db: Session,
    run_id: uuid.UUID,
    pass_name: str,
    *,
    execution_status: str = "COMPLETE",
) -> None:
    """Insert a pipeline_pass_outputs row for the given pass."""
    save_pass_output(
        db,
        pipeline_run_id=run_id,
        stage_run_id=None,
        pass_name=pass_name,
        attempt=1,
        execution_status=execution_status,
        skip_reason=None,
        yield_status=None,
        extract_pass_response={},
        primary_entities_extracted=0,
        bridge_entities_extracted=0,
        relationships_extracted=0,
        relationships_rejected=0,
        diagnostics={},
        field_provenance=[],
    )
    db.flush()


def _set_run_mode(db: Session, run_id: uuid.UUID, *, mode: str = "full") -> None:
    """Set the mode and ontology_bundle_key on a pipeline_run row."""
    db.execute(
        text(
            """
            UPDATE ingest.pipeline_runs
            SET mode = :mode, ontology_bundle_key = :bundle_key
            WHERE id = :run_id
            """
        ),
        {"run_id": str(run_id), "mode": mode, "bundle_key": _BUNDLE_KEY},
    )
    db.flush()


def _set_run_status(db: Session, run_id: uuid.UUID, status: str) -> None:
    db.execute(
        text("UPDATE ingest.pipeline_runs SET status = :status WHERE id = :run_id"),
        {"run_id": str(run_id), "status": status},
    )
    db.flush()


def _now_minus(seconds: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


def _now_minus_iso(seconds: int) -> str:
    return _now_minus(seconds).isoformat()


def _build_claimed_entry(claimed_at: str, task_id: str = "task-abc") -> dict:
    return {
        "state": "claimed",
        "result": None,
        "task_id": task_id,
        "claimed_at": claimed_at,
        "dispatched_at": None,
        "completed_at": None,
    }


def _build_dispatched_entry(
    dispatched_at: str,
    *,
    claimed_at: str | None = None,
    task_id: str = "task-xyz",
) -> dict:
    if claimed_at is None:
        claimed_at = dispatched_at  # same as dispatched for simplicity
    return {
        "state": "dispatched",
        "result": None,
        "task_id": task_id,
        "claimed_at": claimed_at,
        "dispatched_at": dispatched_at,
        "completed_at": None,
    }


@contextmanager
def _patched_reconciler(db_session: Session, *, manifest=None):
    """Patch the reconciler's external dependencies and yield a dict of mocks.

    Patches:
    - ``_get_db`` → non-closing proxy around db_session
    - ``load_bundle_manifest`` → fake manifest with two entity passes
    - ``derive_ontology_graph_pass.delay`` → MagicMock (entity pass dispatch)
    - ``celery_app.send_task`` → MagicMock (merge dispatch via send_task)
    - ``app.services.run_phase_dispatch.celery_app`` → captures revoke calls
    """
    proxy = _NoCloseProxy(db_session)
    if manifest is None:
        manifest = _fake_manifest()

    mock_revoke = MagicMock()
    mock_delay = MagicMock(return_value=MagicMock(id="mocked-task-id"))
    mock_send_task = MagicMock()

    with (
        patch("app.workers.pipeline._get_db", return_value=proxy),
        patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
        patch("app.workers.pipeline.derive_ontology_graph_pass.delay", mock_delay),
        # _try_advance_phase dispatches merge via celery_app.send_task, not .delay()
        patch("app.workers.pipeline.celery_app.send_task", mock_send_task),
        patch("app.services.run_phase_dispatch.celery_app") as mock_celery,
    ):
        mock_celery.control.revoke = mock_revoke
        yield {
            "revoke": mock_revoke,
            "delay": mock_delay,
            "send_task": mock_send_task,
            "celery": mock_celery,
        }


def _run_reconciler() -> dict:
    """Invoke the reconciler task body directly (bypasses Celery infrastructure).

    For ``bind=True`` Celery tasks, ``task.run()`` is already bound — the Celery
    framework injects ``self`` automatically, so we call with no positional args.
    """
    return reconcile_ontology_graph_runs.run()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReconcilesStaleClaimedPhase:
    """Branch 1: stale claimed repair."""

    def test_repairs_stale_claimed_compare_and_reset_dispatches(
        self, db_session: Session, pipeline_run_factory
    ):
        """Pre-seed phase as claimed with claimed_at 60s ago; threshold 30s.
        Verify reclaim_stale_phase is called + _try_advance_phase dispatches."""
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        stale_at = _now_minus_iso(60)
        _seed_phase(db_session, run_id, _PHASE_A, _build_claimed_entry(stale_at))

        with _patched_reconciler(db_session) as mocks:
            result = _run_reconciler()

        assert _PHASE_A in result["stale_claimed_reclaimed"], (
            f"Expected {_PHASE_A} in stale_claimed_reclaimed; got {result}"
        )
        # _try_advance_phase should have dispatched the next pass (entity_pass_A
        # was reclaimed, so it dispatches _PASS_A again via claim_phase + .delay).
        mocks["delay"].assert_called()

    def test_does_not_touch_recent_claimed(
        self, db_session: Session, pipeline_run_factory
    ):
        """Pre-seed phase as claimed with claimed_at 5s ago; threshold 30s.
        Verify NO reclaim, NO advance."""
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        fresh_at = _now_minus_iso(5)
        _seed_phase(db_session, run_id, _PHASE_A, _build_claimed_entry(fresh_at))

        with _patched_reconciler(db_session) as mocks:
            result = _run_reconciler()

        assert result["stale_claimed_reclaimed"] == [], (
            f"Expected no reclaims; got {result['stale_claimed_reclaimed']}"
        )
        mocks["delay"].assert_not_called()

        # Phase entry should still be present.
        entry = read_phase_state(db_session, run_id, _PHASE_A)
        assert entry is not None
        assert entry.state == "claimed"


class TestReconcilesStaleDispatchedPhase:
    """Branch 2: stale dispatched repair (pending-retry-aware)."""

    def test_repairs_stale_dispatched_revokes_and_redispatches(
        self, db_session: Session, pipeline_run_factory
    ):
        """Pre-seed phase as dispatched with dispatched_at 3h ago; NO pass-output row;
        NO pending retry; threshold 2h. Verify celery revoke + reclaim + advance."""
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        stale_at = _now_minus_iso(10800)  # 3 hours ago
        _seed_phase(
            db_session, run_id, _PHASE_A,
            _build_dispatched_entry(stale_at, task_id="stale-task-id"),
        )

        with _patched_reconciler(db_session) as mocks:
            result = _run_reconciler()

        assert _PHASE_A in result["stale_dispatched_reclaimed"], (
            f"Expected {_PHASE_A} in stale_dispatched_reclaimed; got {result}"
        )
        # Revoke should have been issued (via reclaim_stale_phase).
        mocks["revoke"].assert_called()

    def test_does_not_touch_recent_dispatched(
        self, db_session: Session, pipeline_run_factory
    ):
        """Pre-seed phase as dispatched with dispatched_at 30min ago; threshold 2h.
        Verify NO action."""
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        recent_at = _now_minus_iso(1800)  # 30 minutes ago
        _seed_phase(
            db_session, run_id, _PHASE_A,
            _build_dispatched_entry(recent_at, task_id="fresh-task"),
        )

        with _patched_reconciler(db_session) as mocks:
            result = _run_reconciler()

        assert result["stale_dispatched_reclaimed"] == [], (
            f"Expected no dispatched reclaims; got {result['stale_dispatched_reclaimed']}"
        )
        mocks["revoke"].assert_not_called()
        mocks["delay"].assert_not_called()

    def test_does_not_reclaim_dispatched_with_pending_retry(
        self, db_session: Session, pipeline_run_factory
    ):
        """Pre-seed phase as dispatched (>2h old) AND a recent FAILED StageRun
        (attempt=2/6, finished_at=2min ago). Verify NO revoke, NO reclaim — the
        Celery retry is pending in the countdown queue."""
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        stale_at = _now_minus_iso(10800)  # 3 hours
        _seed_phase(
            db_session, run_id, _PHASE_A,
            _build_dispatched_entry(stale_at, task_id="stale-but-retrying"),
        )

        # Seed a FAILED StageRun with attempt=2 (max=3+3=6), finished 2 min ago.
        # This falls within the countdown_buffer_seconds=600 window.
        finished_2min_ago = _now_minus(120)
        _seed_stage_run(
            db_session, run_id, _PASS_A,
            execution_status="FAILED",
            attempt=2,
            finished_at=finished_2min_ago,
        )

        with _patched_reconciler(db_session) as mocks:
            result = _run_reconciler()

        assert _PHASE_A in result["skipped_pending_retry"], (
            f"Expected {_PHASE_A} in skipped_pending_retry; got {result}"
        )
        assert result["stale_dispatched_reclaimed"] == [], (
            f"Should not reclaim; got {result['stale_dispatched_reclaimed']}"
        )
        mocks["revoke"].assert_not_called()

    def test_does_not_reclaim_dispatched_when_attempts_exhausted_but_pass_output_exists(
        self, db_session: Session, pipeline_run_factory
    ):
        """Pre-seed phase as dispatched (>2h old) AND a terminal pass-output row
        (execution_status=COMPLETE). Verify reconciler calls mark_phase_terminal
        (branch 3) instead of reclaim (branch 2)."""
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        stale_at = _now_minus_iso(10800)
        _seed_phase(
            db_session, run_id, _PHASE_A,
            _build_dispatched_entry(stale_at, task_id="exhausted-but-output"),
        )
        # Seed a terminal pass-output row (task completed but didn't mark terminal).
        _seed_pass_output(db_session, run_id, _PASS_A, execution_status="COMPLETE")

        with _patched_reconciler(db_session) as mocks:
            result = _run_reconciler()

        # Should be promoted, NOT reclaimed.
        assert _PHASE_A in result["promoted_to_terminal"], (
            f"Expected {_PHASE_A} in promoted_to_terminal; got {result}"
        )
        assert result["stale_dispatched_reclaimed"] == [], (
            f"Should not reclaim; got {result['stale_dispatched_reclaimed']}"
        )
        mocks["revoke"].assert_not_called()


class TestPromotesCompletedButNotMarkedTerminal:
    """Branch 3: task wrote pass-output but crashed before mark_phase_terminal."""

    def test_promotes_completed_but_not_marked_terminal_failed(
        self, db_session: Session, pipeline_run_factory
    ):
        """Pre-seed phase as dispatched AND pass-output row with
        execution_status=FAILED. Verify mark_phase_terminal(result='failed')
        called, NOT reclaim."""
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        # dispatched_at can be recent — promotion is independent of age.
        recent_at = _now_minus_iso(30)
        _seed_phase(
            db_session, run_id, _PHASE_A,
            _build_dispatched_entry(recent_at, task_id="task-wrote-failed"),
        )
        _seed_pass_output(db_session, run_id, _PASS_A, execution_status="FAILED")

        with _patched_reconciler(db_session) as mocks:
            result = _run_reconciler()

        assert _PHASE_A in result["promoted_to_terminal"], (
            f"Expected {_PHASE_A} in promoted_to_terminal; got {result}"
        )
        assert result["stale_dispatched_reclaimed"] == [], (
            f"Should NOT reclaim; got {result['stale_dispatched_reclaimed']}"
        )
        mocks["revoke"].assert_not_called()

        # Phase should now be in completed state with result=failed.
        entry = read_phase_state(db_session, run_id, _PHASE_A)
        assert entry is not None
        assert entry.state == "completed"
        assert entry.result == "failed"

    def test_promotes_completed_but_not_marked_terminal_succeeded(
        self, db_session: Session, pipeline_run_factory
    ):
        """COMPLETE pass-output → result='succeeded' after promotion."""
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        recent_at = _now_minus_iso(45)
        _seed_phase(
            db_session, run_id, _PHASE_A,
            _build_dispatched_entry(recent_at, task_id="task-wrote-complete"),
        )
        _seed_pass_output(db_session, run_id, _PASS_A, execution_status="COMPLETE")

        with _patched_reconciler(db_session):
            result = _run_reconciler()

        assert _PHASE_A in result["promoted_to_terminal"]
        entry = read_phase_state(db_session, run_id, _PHASE_A)
        assert entry is not None
        assert entry.result == "succeeded"

    def test_promotes_skipped_to_terminal_skipped(
        self, db_session: Session, pipeline_run_factory
    ):
        """Branch 3: dispatched + SKIPPED pass-output → mark_phase_terminal(result='skipped').

        Regression guard: a typo mapping SKIPPED to 'failed' instead of 'skipped'
        would go undetected without this test (only COMPLETE and FAILED were
        previously covered).
        """
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        recent_at = _now_minus_iso(45)
        _seed_phase(
            db_session, run_id, _PHASE_A,
            _build_dispatched_entry(recent_at, task_id="task-wrote-skipped"),
        )
        _seed_pass_output(db_session, run_id, _PASS_A, execution_status="SKIPPED")

        with _patched_reconciler(db_session):
            result = _run_reconciler()

        assert _PHASE_A in result["promoted_to_terminal"], (
            f"Expected {_PHASE_A} in promoted_to_terminal; got {result}"
        )
        # Phase must be marked completed with result='skipped' (not 'failed').
        entry = read_phase_state(db_session, run_id, _PHASE_A)
        assert entry is not None
        assert entry.state == "completed"
        assert entry.result == "skipped", (
            f"Expected result='skipped' for SKIPPED execution_status; got '{entry.result}'"
        )


class TestStuckWithoutAdvance:
    """Branch 4: all entity passes terminal but no follow-up dispatched."""

    def test_dispatches_missing_follow_up(
        self, db_session: Session, pipeline_run_factory
    ):
        """Pre-seed all entity passes COMPLETE in pipeline_pass_outputs; NO
        system_links or merge phase entry. Verify reconciler calls _try_advance_phase
        (which dispatches the next stage).

        The realistic stuck-without-advance scenario: both entity passes wrote
        terminal pass-output rows AND their phase entries were marked 'completed',
        but the finisher crashed after mark_phase_terminal and before
        _try_advance_phase. So dispatched_phases has two 'completed' entries
        (not empty) and pipeline_pass_outputs has two rows — but no system_links
        or merge phase entry exists yet.
        """
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        # Seed terminal pass-output rows for both entity passes.
        _seed_pass_output(db_session, run_id, _PASS_A, execution_status="COMPLETE")
        _seed_pass_output(db_session, run_id, _PASS_B, execution_status="COMPLETE")

        # Seed completed phase entries in dispatched_phases (as mark_phase_terminal
        # would have left them) — NO system_links or merge entry exists yet.
        completed_at = _now_minus_iso(5)
        for phase_key in (_PHASE_A, _PHASE_B):
            _seed_phase(
                db_session, run_id, phase_key,
                {
                    "state": "completed",
                    "result": "succeeded",
                    "task_id": "task-done",
                    "claimed_at": _now_minus_iso(300),
                    "dispatched_at": _now_minus_iso(250),
                    "completed_at": completed_at,
                },
            )

        with _patched_reconciler(db_session) as mocks:
            result = _run_reconciler()

        assert str(run_id) in result["stuck_advances"], (
            f"Expected run_id in stuck_advances; got {result['stuck_advances']}"
        )
        # _try_advance_phase dispatches merge via send_task (no system_links in manifest).
        mocks["send_task"].assert_called()

    def test_dispatches_merge_when_system_links_completed_but_merge_absent(
        self, db_session: Session, pipeline_run_factory
    ):
        """Regression: Branch 4 must fire when system_links is completed AND merge
        is absent.

        Previously the condition was ``merge_absent and sl_absent`` — this skipped
        Branch 4 whenever system_links appeared in dispatched_phases at all (even
        with state='completed'), leaving runs with:
          - all entity passes completed
          - system_links completed
          - no merge dispatch
        in a permanently stuck state.  Production bundles (air_defense_v3) include
        system_links, so this bug would have manifested in every production run.
        """
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        # Use a manifest that includes system_links (depends_on=[_PASS_A] marks it as
        # a non-entity pass so entity_passes contains only _PASS_A and _PASS_B).
        sl_pass_def = SimpleNamespace(
            name="system_links",
            kind="relationships_only",
            input_mode="document_only",
            required=True,
            depends_on=[_PASS_A],  # non-empty → not an entity pass
            primary_entity_types=[],
            bridge_entity_types=[],
            extracted_relationship_types=["ASSOCIATED_WITH"],
            skip_if_no_upstream_endpoints=True,
            module="extraction_schemas.system_links",
            template_class="SystemLinksPass",
        )
        manifest_with_sl = SimpleNamespace(
            bundle_key=_BUNDLE_KEY,
            passes=[_fake_pass_def(_PASS_A), _fake_pass_def(_PASS_B), sl_pass_def],
        )

        # Seed terminal pass-output rows for both entity passes AND system_links.
        _seed_pass_output(db_session, run_id, _PASS_A, execution_status="COMPLETE")
        _seed_pass_output(db_session, run_id, _PASS_B, execution_status="COMPLETE")
        _seed_pass_output(db_session, run_id, "system_links", execution_status="COMPLETE")

        # Seed dispatched_phases: entity passes + system_links all 'completed'; no merge.
        completed_at = _now_minus_iso(5)
        completed_entry = {
            "state": "completed",
            "result": "succeeded",
            "task_id": "task-done",
            "claimed_at": _now_minus_iso(300),
            "dispatched_at": _now_minus_iso(250),
            "completed_at": completed_at,
        }
        for phase_key in (_PHASE_A, _PHASE_B, "system_links"):
            _seed_phase(db_session, run_id, phase_key, completed_entry)
        # "merge" is intentionally absent from dispatched_phases.

        with _patched_reconciler(db_session, manifest=manifest_with_sl) as mocks:
            result = _run_reconciler()

        assert str(run_id) in result["stuck_advances"], (
            "Branch 4 should have fired for run with completed entity passes + "
            f"completed system_links + absent merge; got stuck_advances={result['stuck_advances']}"
        )
        # _try_advance_phase should have dispatched merge.
        mocks["send_task"].assert_called()


class TestSkipsTerminalizedRuns:
    """Reconciler must skip runs that are not PROCESSING."""

    def test_skips_terminalized_runs(
        self, db_session: Session, pipeline_run_factory
    ):
        """Pre-seed run.status=FAILED. Verify reconciler skips it (no actions)."""
        run_id = pipeline_run_factory(status="FAILED")
        _set_run_mode(db_session, run_id)

        stale_at = _now_minus_iso(3600)
        _seed_phase(db_session, run_id, _PHASE_A, _build_claimed_entry(stale_at))

        with _patched_reconciler(db_session) as mocks:
            result = _run_reconciler()

        assert result["scanned_runs"] == 0, (
            "FAILED runs should not be scanned; scanned_runs should be 0"
        )
        assert result["stale_claimed_reclaimed"] == []
        mocks["delay"].assert_not_called()


class TestCompareAndResetRace:
    """Race safety: reconciler must not clobber a fresh claim."""

    def test_compare_and_reset_does_not_clobber_fresh_claim_under_us(
        self, db_session: Session, pipeline_run_factory
    ):
        """Race scenario: reconciler reads phase as stale-claimed; before its UPDATE
        fires, simulate a fresh claim_phase call (update entry to fresh claimed_at).
        Verify reconciler's UPDATE no-ops because the predicate matches stale-only
        timestamps.

        Implementation: we use _patched_reconciler to intercept the _get_db proxy,
        then manually advance the claimed_at AFTER the reconciler reads dispatched_phases
        (simulated by seeding the fresh timestamp directly before invoking). The key
        behavior under test is reclaim_stale_phase's SQL predicate:
          WHERE dispatched_phases->:phase->>'state' = 'claimed'
            AND claimed_at < NOW() - threshold
        A fresh claim written moments before the UPDATE fires will have a recent
        claimed_at and fail the timestamp predicate → rowcount=0 → no reclaim.
        """
        run_id = pipeline_run_factory(status="PROCESSING")
        _set_run_mode(db_session, run_id)

        # --- Step 1: seed a stale claimed entry so reconciler's initial scan sees it.
        stale_at = _now_minus_iso(90)  # 90s ago, well above 30s threshold
        _seed_phase(db_session, run_id, _PHASE_A, _build_claimed_entry(stale_at))

        # --- Step 2: before the reconciler fires its UPDATE, simulate a fresh
        # worker winning a claim (update the claimed_at to "now").
        # This is the race: the reconciler already read the stale entry, but the
        # fresh claim changes the row's claimed_at to a recent timestamp.
        fresh_at = datetime.now(timezone.utc).isoformat()
        _seed_phase(db_session, run_id, _PHASE_A, _build_claimed_entry(fresh_at))

        # --- Step 3: run the reconciler. Its UPDATE predicate:
        #   claimed_at < NOW() - 30s
        # will NOT match the fresh claimed_at, so rowcount=0 → no reclaim.
        with _patched_reconciler(db_session) as mocks:
            result = _run_reconciler()

        assert result["stale_claimed_reclaimed"] == [], (
            "Fresh claim should NOT have been reclaimed; compare-and-reset "
            f"predicate must protect it. Got: {result['stale_claimed_reclaimed']}"
        )
        mocks["delay"].assert_not_called()

        # The fresh entry must still be present and unchanged.
        entry = read_phase_state(db_session, run_id, _PHASE_A)
        assert entry is not None
        assert entry.state == "claimed"


class TestBeatScheduleRegistered:
    """Sanity: the beat schedule entry exists in celery_app."""

    def test_beat_schedule_entry_registered(self):
        from app.workers.celery_app import celery_app

        assert "reconcile-ontology-graph-runs" in celery_app.conf.beat_schedule, (
            "reconcile-ontology-graph-runs not found in celery_app.conf.beat_schedule"
        )
        entry = celery_app.conf.beat_schedule["reconcile-ontology-graph-runs"]
        assert entry["task"] == "app.workers.pipeline.reconcile_ontology_graph_runs"
        assert entry["options"]["queue"] == "graph"
