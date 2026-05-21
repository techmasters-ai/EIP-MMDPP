"""Regression test for Fix 1: reconciler stuck-without-advance fast path.

Scenario: ALL entity passes have terminal pipeline_pass_outputs rows BUT
dispatched_phases still has those passes in state='dispatched' (the task
wrote pass_output but crashed before mark_phase_terminal).

Before Fix 1:
  - Branch 4 fast path fires (count_terminal_passes sees the terminal rows).
  - Branch 4 calls _try_advance_phase.
  - _try_advance_phase checks dispatched_phases → identity passes still
    "dispatched" (not "completed") → identity_all_terminal == False → returns.
  - Branch 4 hits `continue` → skips Branch 3 (the per-phase promotion loop).
  - Next reconcile cycle: identical outcome. Reconciler loops forever.

After Fix 1 (reorder — Branch 3 loop runs BEFORE Branch 4):
  - Per-phase loop runs first, sees dispatched + pass_output row → promotes
    dispatched_phases entries to "completed" via mark_phase_terminal.
  - Branch 4 fast path then runs with up-to-date dispatched_phases.
  - _try_advance_phase sees identity_all_terminal == True → advances.

This test is hermetic (no real DB, no Celery) following the pattern in
test_try_advance_phase_by_phase.py.

Run standalone:
    python3 -m pytest tests/unit/test_reconciler_stuck_recovery.py -v
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Helpers (hermetic, mirrors test_try_advance_phase_by_phase.py pattern)
# ---------------------------------------------------------------------------

_RUN_ID = str(uuid.uuid4())
_DOC_ID = str(uuid.uuid4())
_BUNDLE_KEY = "test_stuck_bundle"
_PASS_A = "radar_identity"
_PASS_B = "missile_identity"
_PHASE_A = f"entity_pass_{_PASS_A}"
_PHASE_B = f"entity_pass_{_PASS_B}"


def _make_pass_def(name: str, phase: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        phase=phase,
        required=True,
        depends_on=[],
        input_mode="document_only",
    )


def _make_manifest(pass_defs: list) -> SimpleNamespace:
    return SimpleNamespace(bundle_key=_BUNDLE_KEY, passes=pass_defs)


def _dispatched_entry(phase_key: str) -> dict:
    """A phase entry that is still in 'dispatched' state (task not yet marked terminal)."""
    return {
        "state": "dispatched",
        "result": None,
        "task_id": f"task-{phase_key}",
        "claimed_at": "2026-01-01T00:00:00+00:00",
        "dispatched_at": "2026-01-01T00:01:00+00:00",
        "completed_at": None,
    }


def _make_pass_output(execution_status: str) -> SimpleNamespace:
    return SimpleNamespace(execution_status=execution_status)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------

class TestReconcilerStuckRecovery:
    """Fix 1 regression: stuck-without-advance fast path must NOT skip per-phase promotion."""

    def test_dispatched_phase_with_terminal_pass_output_gets_promoted_and_advances(self):
        """Core regression: all entity passes have terminal pass_output rows AND
        dispatched_phases entries still in 'dispatched' state.

        Before fix: Branch 4 fires → _try_advance_phase fails identity gate →
        continue skips Branch 3 → stuck forever.

        After fix (reorder): Branch 3 runs first → promotes dispatched_phases
        to 'completed' → Branch 4 _try_advance_phase sees correct state →
        advances.

        We test by verifying that mark_phase_terminal is called for each
        dispatched-but-pass-output-terminal phase, AND that _try_advance_phase
        is called (via the send_task mock picking up the merge dispatch).
        """
        from app.workers.pipeline import reconcile_ontology_graph_runs

        manifest = _make_manifest([
            _make_pass_def(_PASS_A, "identity"),
            _make_pass_def(_PASS_B, "identity"),
        ])

        # dispatched_phases: both entity passes are still "dispatched"
        # (task wrote pass_output but crashed before mark_phase_terminal).
        dispatched_phases_state = {
            _PHASE_A: _dispatched_entry(_PHASE_A),
            _PHASE_B: _dispatched_entry(_PHASE_B),
        }

        # Fake PipelineRun with PROCESSING status.
        fake_run = MagicMock()
        fake_run.id = uuid.UUID(_RUN_ID)
        fake_run.document_id = uuid.UUID(_DOC_ID)
        fake_run.status = "PROCESSING"
        fake_run.mode = "full"
        fake_run.ontology_bundle_key = _BUNDLE_KEY
        fake_run.dispatched_phases = dispatched_phases_state

        # DB mock: scalars() returns the fake run, then empty on subsequent queries.
        mock_db = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = [fake_run]

        # pass_output mock: both passes have terminal rows (COMPLETE).
        pass_output_complete = _make_pass_output("COMPLETE")

        # count_terminal_passes mock: returns len(entity_passes) to trigger Branch 4.
        mock_count_terminal = MagicMock(return_value=2)

        # load_pass_output: returns terminal rows for both entity passes.
        mock_load_pass_output = MagicMock(return_value=pass_output_complete)

        # mark_phase_terminal mock: records calls.
        mock_mark_terminal = MagicMock(return_value=True)

        # _try_advance_phase mock: records calls (we verify it IS called).
        mock_try_advance = MagicMock()

        # _has_pending_retry_for_pass mock: returns False (no pending retry).
        mock_pending = MagicMock(return_value=False)

        # send_task mock (merge dispatch inside _try_advance_phase if real).
        mock_send_task = MagicMock()

        with (
            patch("app.workers.pipeline._get_db", return_value=mock_db),
            patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
            patch("app.workers.pipeline.count_terminal_passes", mock_count_terminal),
            patch("app.workers.pipeline.load_pass_output", mock_load_pass_output),
            patch("app.workers.pipeline.mark_phase_terminal", mock_mark_terminal),
            patch("app.workers.pipeline._try_advance_phase", mock_try_advance),
            patch("app.workers.pipeline._has_pending_retry_for_pass", mock_pending),
            patch("app.workers.pipeline.celery_app.send_task", mock_send_task),
        ):
            result = reconcile_ontology_graph_runs.run()

        # mark_phase_terminal MUST have been called for the dispatched-but-output-terminal phases.
        assert mock_mark_terminal.call_count >= 1, (
            "Expected mark_phase_terminal to be called for at least one phase with "
            f"terminal pass_output; call_count={mock_mark_terminal.call_count}"
        )

        # _try_advance_phase MUST have been called (promotion unlocks advance).
        assert mock_try_advance.call_count >= 1, (
            "Expected _try_advance_phase to be called after phase promotion; "
            f"call_count={mock_try_advance.call_count}"
        )

        # promoted_to_terminal must include the phase keys.
        assert len(result["promoted_to_terminal"]) >= 1, (
            f"Expected at least 1 promoted phase; got: {result['promoted_to_terminal']}"
        )

    def test_dispatched_phase_with_no_pass_output_does_not_promote(self):
        """Sanity: if no pass_output row exists, no premature promotion occurs.

        The per-phase loop should check load_pass_output → None → skip promotion.
        """
        from app.workers.pipeline import reconcile_ontology_graph_runs

        manifest = _make_manifest([
            _make_pass_def(_PASS_A, "identity"),
            _make_pass_def(_PASS_B, "identity"),
        ])

        # Only PASS_A is dispatched; PASS_B is not yet started.
        dispatched_phases_state = {
            _PHASE_A: _dispatched_entry(_PHASE_A),
        }

        fake_run = MagicMock()
        fake_run.id = uuid.UUID(_RUN_ID)
        fake_run.document_id = uuid.UUID(_DOC_ID)
        fake_run.status = "PROCESSING"
        fake_run.mode = "full"
        fake_run.ontology_bundle_key = _BUNDLE_KEY
        fake_run.dispatched_phases = dispatched_phases_state

        mock_db = MagicMock()
        mock_db.execute.return_value.scalars.return_value.all.return_value = [fake_run]

        # No pass_output row exists yet.
        mock_load_pass_output = MagicMock(return_value=None)
        mock_mark_terminal = MagicMock(return_value=True)
        mock_try_advance = MagicMock()
        mock_pending = MagicMock(return_value=False)
        # count_terminal_passes returns 0 (no terminal rows) → Branch 4 does not fire.
        mock_count_terminal = MagicMock(return_value=0)

        with (
            patch("app.workers.pipeline._get_db", return_value=mock_db),
            patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest),
            patch("app.workers.pipeline.count_terminal_passes", mock_count_terminal),
            patch("app.workers.pipeline.load_pass_output", mock_load_pass_output),
            patch("app.workers.pipeline.mark_phase_terminal", mock_mark_terminal),
            patch("app.workers.pipeline._try_advance_phase", mock_try_advance),
            patch("app.workers.pipeline._has_pending_retry_for_pass", mock_pending),
        ):
            result = reconcile_ontology_graph_runs.run()

        # No promotion should have occurred.
        assert mock_mark_terminal.call_count == 0, (
            f"mark_phase_terminal should NOT have been called; count={mock_mark_terminal.call_count}"
        )
        assert result["promoted_to_terminal"] == [], (
            f"Expected empty promoted_to_terminal; got {result['promoted_to_terminal']}"
        )
