"""C1.6 — _try_advance_phase phase-gated branching tests.

Task 1.6.5: integration test covering all four Branches + transitions.

Tests are fully hermetic: no real DB, no real Celery.  All I/O is mocked
at the ``app.workers.pipeline`` module level.

Design choices
--------------
Each test builds:
  - A fake ``PipelineRun`` whose ``dispatched_phases`` dict describes the
    current state of the world (which passes are in-flight, completed, etc.).
  - A fake ``BundleManifest`` with ``PassManifest`` objects carrying ``phase``.
  - Appropriate mocks for ``count_terminal_passes``, ``read_phase_state``,
    ``load_pass_output``, ``_claim_and_dispatch_pass``, ``claim_phase``,
    ``mark_phase_dispatched``, and ``celery_app.send_task``.

We test six scenarios:
  (a) Branch 1 only — identity in-flight < cap → dispatches next identity.
  (b) Branch 1 → Branch 2 transition — all identity terminal, Branch 2 fires.
  (c) Branch 2 ordering guard — identity in-flight → Branch 2 does NOT fire.
  (d) Branch 2 → Branch 3 transition — all field_group terminal → Branch 3.
  (e) Branch 3 → Branch 4 transition — system_links terminal → Branch 4 (merge).
  (f) Back-compat: identity + relationship only (no field_group) → Branch 2
      skipped; identity-terminal triggers Branch 3 directly.

Run standalone (no DB required):
    python3 -m pytest tests/unit/test_try_advance_phase_by_phase.py -v
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_ID = str(uuid.uuid4())
_DOC_ID = str(uuid.uuid4())
_BUNDLE_KEY = "test_phase_bundle"


def _make_pass_def(name: str, phase: str, *, depends_on: list[str] | None = None) -> SimpleNamespace:
    """Return a SimpleNamespace mimicking PassManifest."""
    return SimpleNamespace(
        name=name,
        phase=phase,
        required=(phase == "relationship"),
        depends_on=list(depends_on or []),
        input_mode=(
            "document_plus_entity_refs" if phase == "relationship" else "document_only"
        ),
    )


def _make_manifest(pass_defs: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(bundle_key=_BUNDLE_KEY, passes=pass_defs)


def _make_run(dispatched_phases: dict | None = None) -> MagicMock:
    run = MagicMock()
    run.ontology_bundle_key = _BUNDLE_KEY
    run.dispatched_phases = dispatched_phases or {}
    return run


def _make_db(run: MagicMock) -> MagicMock:
    db = MagicMock()
    db.get.return_value = run
    # expire + refresh are no-ops
    db.expire.return_value = None
    db.refresh.return_value = None
    return db


def _invoke_advance(db, doc_id: str, run_id: str) -> None:
    """Call _try_advance_phase directly."""
    from app.workers.pipeline import _try_advance_phase
    _try_advance_phase(db, doc_id, run_id)


# ---------------------------------------------------------------------------
# Standard manifests used across tests
# ---------------------------------------------------------------------------

# Full 3-phase manifest: 2 identity + 2 field_group + 1 relationship
_FULL_PASSES = [
    _make_pass_def("radar_identity", "identity"),
    _make_pass_def("missile_identity", "identity"),
    _make_pass_def("radar_power_rf", "field_group"),
    _make_pass_def("missile_kinematics", "field_group"),
    _make_pass_def("system_links", "relationship", depends_on=[
        "radar_identity", "missile_identity", "radar_power_rf", "missile_kinematics",
    ]),
]

# Identity-only + relationship (no field_group) — back-compat case
_NO_FG_PASSES = [
    _make_pass_def("radar_identity", "identity"),
    _make_pass_def("missile_identity", "identity"),
    _make_pass_def("system_links", "relationship", depends_on=[
        "radar_identity", "missile_identity",
    ]),
]


# ---------------------------------------------------------------------------
# (a) Branch 1 — identity in-flight < cap → dispatch next identity
# ---------------------------------------------------------------------------

class TestBranch1DispatchesNextIdentity:

    def _run(self, dispatched_phases: dict, concurrency_cap: int = 4):
        manifest = _make_manifest(_FULL_PASSES)
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None):
            dispatched_calls.append(pass_name)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document",
                   concurrency_cap):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        return dispatched_calls

    def test_dispatches_first_identity_when_none_in_flight(self):
        """With no passes dispatched yet, Branch 1 dispatches the first identity pass."""
        calls = self._run({})
        assert len(calls) == 1
        assert calls[0] in ("radar_identity", "missile_identity"), (
            f"Expected an identity pass, got {calls[0]!r}"
        )

    def test_dispatches_second_identity_when_first_in_flight(self):
        """One identity in-flight, cap=4 → second identity dispatched."""
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "dispatched"},
        }
        calls = self._run(dispatched_phases, concurrency_cap=4)
        assert len(calls) == 1
        assert calls[0] == "missile_identity"

    def test_no_dispatch_when_both_identity_in_flight(self):
        """Both identity passes in-flight at cap=2 → no dispatch (cap reached)."""
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "dispatched"},
            "entity_pass_missile_identity": {"state": "dispatched"},
        }
        calls = self._run(dispatched_phases, concurrency_cap=2)
        assert len(calls) == 0

    def test_branch1_dispatches_only_identity_not_field_group(self):
        """With identity still pending, Branch 1 dispatches identity only.

        With radar_identity still in-flight and missile_identity not yet
        dispatched, Branch 1 dispatches missile_identity (the next identity
        pass) rather than any field_group pass.  field_group passes are NOT
        dispatched until all identity passes terminate (Branch 2).
        """
        # One identity in-flight, one identity still un-dispatched.
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "dispatched"},
        }
        calls = self._run(dispatched_phases, concurrency_cap=4)
        # Branch 1 must dispatch missile_identity (identity), not field_group.
        assert calls == ["missile_identity"], (
            f"Branch 1 should dispatch missile_identity; got {calls}"
        )
        assert not any(
            n in ("radar_power_rf", "missile_kinematics") for n in calls
        ), f"Branch 1 dispatched a field_group pass: {calls}"


# ---------------------------------------------------------------------------
# (b) Branch 1 → Branch 2 transition
# ---------------------------------------------------------------------------

class TestBranch1ToBranch2Transition:
    """When all identity passes are terminal, Branch 1 must not dispatch;
    Branch 2 fires and dispatches a field_group pass."""

    def test_branch2_fires_when_all_identity_terminal(self):
        """Both identity passes completed → Branch 2 dispatches a field_group pass."""
        manifest = _make_manifest(_FULL_PASSES)
        # All identity passes completed; no field_group passes yet.
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
            "entity_pass_missile_identity": {"state": "completed"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None):
            dispatched_calls.append(pass_name)
            return True

        # count_terminal_passes: only count field_group passes (0 so far)
        def fake_count_terminal(db_, run_id, names):
            return 0  # no field_group passes completed yet

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes",
                   side_effect=fake_count_terminal), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert len(dispatched_calls) == 1
        assert dispatched_calls[0] in ("radar_power_rf", "missile_kinematics"), (
            f"Expected a field_group pass from Branch 2, got {dispatched_calls!r}"
        )

    def test_branch1_returns_without_dispatch_when_all_identity_complete(self):
        """With all identity completed AND one field_group in-flight at cap,
        Branch 1 should not try to dispatch anything (all identity done,
        cap consumed by in-flight field_group)."""
        manifest = _make_manifest(_FULL_PASSES)
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
            "entity_pass_missile_identity": {"state": "completed"},
            "entity_pass_radar_power_rf": {"state": "dispatched"},
            "entity_pass_missile_kinematics": {"state": "dispatched"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None):
            dispatched_calls.append(pass_name)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 2):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        # At cap=2 with 2 field_group in-flight: no dispatch expected.
        assert len(dispatched_calls) == 0

    def test_branch2_fires_after_failed_optional_identity_pass(self):
        """Regression: a FAILED-optional identity pass still counts as terminal
        in the dispatched_phases JSONB (mark_phase_terminal writes
        state='completed' regardless of `result`). Branch 2 must fire when all
        identity passes are terminal — including any that failed-optional —
        because the identity guard reads `state == "completed"`, not `result`.
        (Required identity-pass failures raise IngestFailed before
        _try_advance_phase is called, so the FAILED-required path is not
        exercised here.)
        """
        manifest = _make_manifest(_FULL_PASSES)
        # radar_identity FAILED (optional) — mark_phase_terminal still wrote
        # state="completed" with result="failed". missile_identity succeeded.
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed", "result": "failed"},
            "entity_pass_missile_identity": {"state": "completed", "result": "succeeded"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None):
            dispatched_calls.append(pass_name)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        # Branch 2 fired — a field_group pass was dispatched even though
        # one identity pass failed-optional.
        assert len(dispatched_calls) == 1
        assert dispatched_calls[0] in ("radar_power_rf", "missile_kinematics"), (
            f"Expected a field_group pass from Branch 2 after FAILED-optional "
            f"identity, got {dispatched_calls!r}"
        )


# ---------------------------------------------------------------------------
# (c) Branch 2 ordering guard — identity still in-flight → no field_group dispatch
# ---------------------------------------------------------------------------

class TestBranch2OrderingGuard:
    """Field_group passes must NOT be dispatched while any identity pass is in-flight."""

    def test_no_field_group_while_identity_in_flight(self):
        """One identity in-flight → Branch 2 does NOT fire (identity not all terminal)."""
        manifest = _make_manifest(_FULL_PASSES)
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "dispatched"},  # in-flight
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None):
            dispatched_calls.append(pass_name)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        # Branch 1 should dispatch missile_identity (next undispatched identity).
        # radar_power_rf / missile_kinematics should NOT be dispatched.
        assert not any(
            n in ("radar_power_rf", "missile_kinematics") for n in dispatched_calls
        ), f"Branch 2 fired while identity was still in-flight: {dispatched_calls}"

    def test_no_field_group_when_one_identity_not_yet_dispatched(self):
        """Even with cap space, field_group must wait until identity finishes."""
        manifest = _make_manifest(_FULL_PASSES)
        # missile_identity not yet dispatched at all.
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None):
            dispatched_calls.append(pass_name)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        # Branch 1 should dispatch missile_identity (still un-dispatched identity).
        assert dispatched_calls == ["missile_identity"], (
            f"Expected only missile_identity; got {dispatched_calls}"
        )


# ---------------------------------------------------------------------------
# (d) Branch 2 → Branch 3 transition
# ---------------------------------------------------------------------------

class TestBranch2ToBranch3Transition:
    """When all field_group passes terminalize, Branch 3 dispatches system_links."""

    def test_system_links_dispatched_after_all_field_group_terminal(self):
        manifest = _make_manifest(_FULL_PASSES)
        # All identity + all field_group completed.
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
            "entity_pass_missile_identity": {"state": "completed"},
            "entity_pass_radar_power_rf": {"state": "completed"},
            "entity_pass_missile_kinematics": {"state": "completed"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None):
            dispatched_calls.append(pass_name)
            return True

        # All 4 non-relationship passes are terminal.
        def fake_count_terminal(db_, run_id, names):
            return len(names)  # all terminal

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes",
                   side_effect=fake_count_terminal), \
             patch("app.workers.pipeline.read_phase_state", return_value=None), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert dispatched_calls == ["system_links"], (
            f"Expected system_links dispatch from Branch 3; got {dispatched_calls!r}"
        )

    def test_system_links_not_dispatched_when_field_group_still_in_flight(self):
        """Branch 3 must not fire while field_group passes are still running."""
        manifest = _make_manifest(_FULL_PASSES)
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
            "entity_pass_missile_identity": {"state": "completed"},
            "entity_pass_radar_power_rf": {"state": "dispatched"},  # still running
            "entity_pass_missile_kinematics": {"state": "completed"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None):
            dispatched_calls.append(pass_name)
            return True

        # 1 out of 2 field_group passes completed.
        def fake_count_terminal(db_, run_id, names):
            return 1

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes",
                   side_effect=fake_count_terminal), \
             patch("app.workers.pipeline.read_phase_state", return_value=None), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert "system_links" not in dispatched_calls, (
            f"system_links should NOT be dispatched while field_group still running; "
            f"got {dispatched_calls!r}"
        )


# ---------------------------------------------------------------------------
# (e) Branch 3 → Branch 4 transition (merge)
# ---------------------------------------------------------------------------

class TestBranch3ToBranch4Transition:
    """When system_links terminalizes, Branch 4 dispatches merge."""

    def test_merge_dispatched_after_system_links_terminal(self):
        manifest = _make_manifest(_FULL_PASSES)
        # Everything completed including system_links.
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
            "entity_pass_missile_identity": {"state": "completed"},
            "entity_pass_radar_power_rf": {"state": "completed"},
            "entity_pass_missile_kinematics": {"state": "completed"},
            "system_links": {"state": "completed"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        # All passes terminal.
        def fake_count_terminal(db_, run_id, names):
            return len(names)

        # system_links pass output is terminal.
        sl_output = SimpleNamespace(execution_status="COMPLETE")

        def fake_read_phase_state(db_, run_id, phase):
            # system_links already completed; merge not yet dispatched.
            if phase == "system_links":
                return "completed"
            return None  # merge not dispatched yet

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass") as mock_dispatch, \
             patch("app.workers.pipeline.count_terminal_passes",
                   side_effect=fake_count_terminal), \
             patch("app.workers.pipeline.read_phase_state",
                   side_effect=fake_read_phase_state), \
             patch("app.workers.pipeline.load_pass_output", return_value=sl_output), \
             patch("app.workers.pipeline.claim_phase", return_value=True), \
             patch("app.workers.pipeline.mark_phase_dispatched"), \
             patch("app.workers.pipeline.celery_app") as mock_celery, \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        # merge should be dispatched via send_task
        assert mock_celery.send_task.called, "celery_app.send_task not called for merge"
        task_name = mock_celery.send_task.call_args[0][0]
        assert "derive_ontology_graph_merge" in task_name, (
            f"Expected merge task; got {task_name!r}"
        )
        assert mock_dispatch.call_count == 0, (
            "No entity-pass dispatch should happen when merging"
        )

    def test_merge_not_dispatched_if_system_links_not_yet_terminal(self):
        """Branch 4 must not fire if system_links is still in-flight."""
        manifest = _make_manifest(_FULL_PASSES)
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
            "entity_pass_missile_identity": {"state": "completed"},
            "entity_pass_radar_power_rf": {"state": "completed"},
            "entity_pass_missile_kinematics": {"state": "completed"},
            "system_links": {"state": "dispatched"},  # still running
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        # system_links has no terminal pass output yet.
        def fake_load_pass_output(db_, run_id, pass_name):
            return None

        def fake_count_terminal(db_, run_id, names):
            return len(names)

        def fake_read_phase_state(db_, run_id, phase):
            # system_links is dispatched (in-flight); merge not started.
            if phase == "system_links":
                return "dispatched"  # already dispatched, not completed
            return None

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass") as mock_dispatch, \
             patch("app.workers.pipeline.count_terminal_passes",
                   side_effect=fake_count_terminal), \
             patch("app.workers.pipeline.read_phase_state",
                   side_effect=fake_read_phase_state), \
             patch("app.workers.pipeline.load_pass_output",
                   side_effect=fake_load_pass_output), \
             patch("app.workers.pipeline.claim_phase", return_value=True), \
             patch("app.workers.pipeline.celery_app") as mock_celery, \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert not mock_celery.send_task.called, (
            "merge should NOT be dispatched while system_links still in-flight"
        )


# ---------------------------------------------------------------------------
# (f) Back-compat: identity + relationship only (no field_group passes)
# ---------------------------------------------------------------------------

class TestBackCompatNoFieldGroupPasses:
    """When a bundle has no field_group passes, Branch 2 is skipped and
    identity-terminal directly triggers Branch 3 (system_links dispatch)."""

    def test_system_links_dispatched_directly_after_identity_terminal(self):
        manifest = _make_manifest(_NO_FG_PASSES)
        # Both identity completed; system_links not yet dispatched.
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
            "entity_pass_missile_identity": {"state": "completed"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None):
            dispatched_calls.append(pass_name)
            return True

        # 2 identity passes terminal; 0 field_group passes exist.
        def fake_count_terminal(db_, run_id, names):
            return len(names)

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes",
                   side_effect=fake_count_terminal), \
             patch("app.workers.pipeline.read_phase_state", return_value=None), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert dispatched_calls == ["system_links"], (
            f"Expected system_links dispatch; got {dispatched_calls!r}"
        )

    def test_no_field_group_dispatch_in_no_fg_bundle(self):
        """In a no-field_group bundle, no field_group passes should ever be dispatched."""
        manifest = _make_manifest(_NO_FG_PASSES)
        # identity still in flight.
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "dispatched"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None):
            dispatched_calls.append(pass_name)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        # Should dispatch missile_identity (identity, not yet dispatched).
        assert dispatched_calls == ["missile_identity"], (
            f"Expected missile_identity; got {dispatched_calls!r}"
        )
        # Definitely no field_group dispatches.
        assert not any(
            n in ("radar_power_rf", "missile_kinematics") for n in dispatched_calls
        )

    def test_merge_fires_after_system_links_in_no_fg_bundle(self):
        """In no-field_group bundle, merge fires after system_links terminates."""
        manifest = _make_manifest(_NO_FG_PASSES)
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
            "entity_pass_missile_identity": {"state": "completed"},
            "system_links": {"state": "completed"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        sl_output = SimpleNamespace(execution_status="COMPLETE")

        def fake_count_terminal(db_, run_id, names):
            return len(names)

        def fake_read_phase_state(db_, run_id, phase):
            # system_links completed; merge not yet dispatched.
            if phase == "system_links":
                return "completed"
            return None

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass") as mock_dispatch, \
             patch("app.workers.pipeline.count_terminal_passes",
                   side_effect=fake_count_terminal), \
             patch("app.workers.pipeline.read_phase_state",
                   side_effect=fake_read_phase_state), \
             patch("app.workers.pipeline.load_pass_output", return_value=sl_output), \
             patch("app.workers.pipeline.claim_phase", return_value=True), \
             patch("app.workers.pipeline.mark_phase_dispatched"), \
             patch("app.workers.pipeline.celery_app") as mock_celery, \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert mock_celery.send_task.called, "merge should be dispatched"
        task_name = mock_celery.send_task.call_args[0][0]
        assert "derive_ontology_graph_merge" in task_name


# ---------------------------------------------------------------------------
# (g) One-dispatch-per-finisher rule preserved
# ---------------------------------------------------------------------------

class TestOneDispatchPerFinisher:
    """_try_advance_phase must return after exactly one dispatch."""

    def test_only_one_dispatch_per_call(self):
        """With many eligible passes, only one is dispatched per invocation."""
        manifest = _make_manifest(_FULL_PASSES)
        # Nothing dispatched yet.
        dispatched_phases = {}
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None):
            dispatched_calls.append(pass_name)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert len(dispatched_calls) == 1, (
            f"Expected exactly 1 dispatch but got {len(dispatched_calls)}: {dispatched_calls}"
        )
