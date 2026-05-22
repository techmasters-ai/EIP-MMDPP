"""C1.6r — _try_advance_phase concurrent entity dispatch tests.

C1.6r removes the strict identity-first serialization gate.  Identity and
field_group passes share the per-document concurrency budget.  Relationship
passes are still gated on ALL entity passes (identity ∪ field_group) being
terminal.  Merge is still gated on relationship terminal.

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

Invariants tested (C1.6r):
  (a) Branch 1 (entity): identity and field_group share the cap; identity-
      first ordering is scheduling preference only, NOT a hard gate.
  (b) Concurrent entity dispatch: field_group passes are eligible while
      identity passes are still in-flight (no serialization gate).
  (c) All-entity-terminal gate: system_links is NOT dispatched until ALL
      entity passes (identity ∪ field_group) are terminal.
  (d) Relationship-terminal-before-merge: merge is NOT dispatched until
      system_links is terminal.
  (e) Back-compat: identity + relationship only (no field_group) → Branch 2
      (system_links) fires directly after identity terminates.
  (f) FAILED-optional entity pass counts as terminal (any phase).
  (g) One dispatch per finisher.

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
# (a) Branch 1 — entity (identity + field_group) dispatch
# ---------------------------------------------------------------------------

class TestBranch1EntityDispatch:

    def _run(self, dispatched_phases: dict, concurrency_cap: int = 4):
        manifest = _make_manifest(_FULL_PASSES)
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None, **kwargs):
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
            f"Expected an identity pass (first in list), got {calls[0]!r}"
        )

    def test_dispatches_second_identity_when_first_in_flight(self):
        """One identity in-flight, cap=4 → second identity dispatched (identity first in list)."""
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "dispatched"},
        }
        calls = self._run(dispatched_phases, concurrency_cap=4)
        assert len(calls) == 1
        assert calls[0] == "missile_identity"

    def test_no_dispatch_when_at_cap(self):
        """Both identity passes in-flight at cap=2 → no dispatch (cap reached)."""
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "dispatched"},
            "entity_pass_missile_identity": {"state": "dispatched"},
        }
        calls = self._run(dispatched_phases, concurrency_cap=2)
        assert len(calls) == 0

    def test_field_group_dispatched_when_identity_in_flight_and_cap_allows(self):
        """C1.6r: field_group passes are eligible while identity is in-flight.

        With radar_identity in-flight, missile_identity terminal, and cap=4,
        Branch 1 dispatches the next un-started entity pass from the ordered
        list (identity first, then field_group).  missile_identity is
        terminal, so the next eligible pass is radar_power_rf (field_group).
        This confirms the strict serialization gate is removed.
        """
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "dispatched"},   # in-flight
            "entity_pass_missile_identity": {"state": "completed"},   # terminal
        }
        calls = self._run(dispatched_phases, concurrency_cap=4)
        # Branch 1 should dispatch the next un-started entity: radar_power_rf
        # (first field_group pass, since missile_identity is already terminal and
        # radar_identity is in-flight).
        assert len(calls) == 1
        assert calls[0] in ("radar_power_rf", "missile_kinematics"), (
            f"Expected a field_group pass (identity in-flight but cap allows); got {calls}"
        )


# ---------------------------------------------------------------------------
# (b) Concurrent entity dispatch confirmed (field_group while identity runs)
# ---------------------------------------------------------------------------

class TestConcurrentEntityDispatch:
    """C1.6r: field_group passes are eligible alongside identity — no serialization gate."""

    def test_field_group_eligible_while_identity_in_flight(self):
        """Both identity passes in-flight, cap=4 → field_group dispatched next."""
        manifest = _make_manifest(_FULL_PASSES)
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "dispatched"},
            "entity_pass_missile_identity": {"state": "dispatched"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None, **kwargs):
            dispatched_calls.append(pass_name)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        # identity passes in-flight, cap=4 → field_group is next in list
        assert len(dispatched_calls) == 1
        assert dispatched_calls[0] in ("radar_power_rf", "missile_kinematics"), (
            f"Expected field_group dispatch while identity in-flight; got {dispatched_calls}"
        )

    def test_failed_optional_entity_pass_counts_as_terminal(self):
        """Regression: FAILED-optional entity pass (any phase) counts as terminal.

        mark_phase_terminal writes state='completed' regardless of `result`.
        Branch 1 must treat a FAILED-optional entity pass as terminal and
        continue dispatching remaining passes — whether the failed pass was
        an identity or field_group pass.
        (Required identity-pass failures raise IngestFailed before
        _try_advance_phase is called, so the FAILED-required path is not
        exercised here.)
        """
        manifest = _make_manifest(_FULL_PASSES)
        # radar_identity FAILED (optional) — still terminal; missile_identity succeeded.
        # No field_group dispatched yet.
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed", "result": "failed"},
            "entity_pass_missile_identity": {"state": "completed", "result": "succeeded"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None, **kwargs):
            dispatched_calls.append(pass_name)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        # FAILED-optional identity counts as terminal; next entity pass dispatched.
        assert len(dispatched_calls) == 1
        assert dispatched_calls[0] in ("radar_power_rf", "missile_kinematics"), (
            f"Expected a field_group pass after FAILED-optional identity; got {dispatched_calls!r}"
        )

    def test_no_dispatch_when_all_entity_in_flight_at_cap(self):
        """All 4 entity passes in-flight at cap=4 → no dispatch."""
        manifest = _make_manifest(_FULL_PASSES)
        dispatched_phases = {
            "entity_pass_radar_identity":     {"state": "dispatched"},
            "entity_pass_missile_identity":   {"state": "dispatched"},
            "entity_pass_radar_power_rf":     {"state": "dispatched"},
            "entity_pass_missile_kinematics": {"state": "dispatched"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None, **kwargs):
            dispatched_calls.append(pass_name)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert len(dispatched_calls) == 0, (
            f"Expected no dispatch at cap with all entity in-flight; got {dispatched_calls}"
        )


# ---------------------------------------------------------------------------
# (c) All-entity-terminal gate — system_links gated on identity ∪ field_group
# ---------------------------------------------------------------------------

class TestAllEntityTerminalBeforeRelationship:
    """system_links must NOT be dispatched until ALL entity passes
    (identity ∪ field_group) are terminal."""

    def test_system_links_dispatched_after_all_entity_terminal(self):
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

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None, **kwargs):
            dispatched_calls.append(pass_name)
            return True

        # All 4 entity passes are terminal.
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
            f"Expected system_links dispatch from Branch 2; got {dispatched_calls!r}"
        )

    def test_system_links_not_dispatched_when_field_group_still_in_flight(self):
        """Branch 2 must not fire while any field_group pass is still running."""
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

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None, **kwargs):
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
            f"system_links should NOT be dispatched while entity passes still running; "
            f"got {dispatched_calls!r}"
        )

    def test_system_links_not_dispatched_when_identity_still_in_flight(self):
        """Branch 2 must not fire while any identity pass is still running."""
        manifest = _make_manifest(_FULL_PASSES)
        # identity still running; field_group completed
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "dispatched"},  # still running
            "entity_pass_missile_identity": {"state": "completed"},
            "entity_pass_radar_power_rf": {"state": "completed"},
            "entity_pass_missile_kinematics": {"state": "completed"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None, **kwargs):
            dispatched_calls.append(pass_name)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.read_phase_state", return_value=None), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert "system_links" not in dispatched_calls, (
            f"system_links should NOT be dispatched while identity still in-flight; "
            f"got {dispatched_calls!r}"
        )


# ---------------------------------------------------------------------------
# (d) Relationship-terminal-before-merge (Branch 2 → Branch 3 transition)
# ---------------------------------------------------------------------------

class TestRelationshipTerminalBeforeMerge:
    """When system_links terminalizes, Branch 3 dispatches merge."""

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
        """Branch 3 must not fire if system_links is still in-flight."""
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
# (e) Back-compat: identity + relationship only (no field_group passes)
# ---------------------------------------------------------------------------

class TestBackCompatNoFieldGroupPasses:
    """When a bundle has no field_group passes, Branch 1 dispatches identity
    passes; Branch 2 (system_links) fires directly after all identity terminal.
    """

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

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None, **kwargs):
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

    def test_identity_dispatched_in_no_fg_bundle(self):
        """In a no-field_group bundle, identity passes are dispatched normally."""
        manifest = _make_manifest(_NO_FG_PASSES)
        # radar_identity still in flight.
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "dispatched"},
        }
        run = _make_run(dispatched_phases)
        db = _make_db(run)

        dispatched_calls = []

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None, **kwargs):
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
            f"Expected missile_identity; got {dispatched_calls}"
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

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name, queued_counter=None, **kwargs):
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


# ---------------------------------------------------------------------------
# IMPORTANT #1 — build_index_failed propagated through _try_advance_phase
# ---------------------------------------------------------------------------


class TestTryAdvancePhaseVRIndexFailed:
    """_try_advance_phase must read vr_index_built from PipelineRun.metrics
    and pass the correct build_index_failed value to _claim_and_dispatch_pass.

    Without the fix, _try_advance_phase hardcodes build_index_failed=False.
    With the fix, it reads run.metrics.get("vr_index_built") and passes
    build_index_failed=not vr_index_built.
    """

    def _make_run_with_metrics(self, metrics: dict | None, dispatched_phases: dict | None = None):
        run = MagicMock()
        run.ontology_bundle_key = _BUNDLE_KEY
        run.dispatched_phases = dispatched_phases or {}
        run.metrics = metrics
        return run

    def test_try_advance_phase_passes_build_index_failed_true_when_vr_index_not_built(self):
        """When run.metrics['vr_index_built'] is False, _try_advance_phase must pass
        build_index_failed=True to _claim_and_dispatch_pass.
        """
        # One identity in-flight; field_group pass is next
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
            "entity_pass_missile_identity": {"state": "completed"},
        }
        run = self._make_run_with_metrics(
            {"vr_index_built": False},
            dispatched_phases=dispatched_phases,
        )
        db = _make_db(run)
        manifest = _make_manifest(_FULL_PASSES)

        captured_kwargs = {}

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name,
                                    queued_counter=None, **kwargs):
            captured_kwargs.update(kwargs)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert captured_kwargs.get("build_index_failed") is True, (
            "When vr_index_built=False, build_index_failed must be True in dispatch call; "
            f"got captured_kwargs={captured_kwargs!r}"
        )

    def test_try_advance_phase_passes_build_index_failed_false_when_vr_index_built(self):
        """When run.metrics['vr_index_built'] is True, _try_advance_phase must pass
        build_index_failed=False to _claim_and_dispatch_pass.
        """
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
            "entity_pass_missile_identity": {"state": "completed"},
        }
        run = self._make_run_with_metrics(
            {"vr_index_built": True},
            dispatched_phases=dispatched_phases,
        )
        db = _make_db(run)
        manifest = _make_manifest(_FULL_PASSES)

        captured_kwargs = {}

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name,
                                    queued_counter=None, **kwargs):
            captured_kwargs.update(kwargs)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert captured_kwargs.get("build_index_failed") is False, (
            "When vr_index_built=True, build_index_failed must be False in dispatch call; "
            f"got captured_kwargs={captured_kwargs!r}"
        )

    def test_try_advance_phase_defaults_to_false_when_no_metrics(self):
        """When run.metrics is None/empty (old runs before this fix), default to
        build_index_failed=False (safe: endpoint will fail-open if unavailable).
        """
        dispatched_phases = {
            "entity_pass_radar_identity": {"state": "completed"},
            "entity_pass_missile_identity": {"state": "completed"},
        }
        run = self._make_run_with_metrics(None, dispatched_phases=dispatched_phases)
        db = _make_db(run)
        manifest = _make_manifest(_FULL_PASSES)

        captured_kwargs = {}

        def fake_claim_and_dispatch(db_, doc_id, run_id, pass_name,
                                    queued_counter=None, **kwargs):
            captured_kwargs.update(kwargs)
            return True

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._claim_and_dispatch_pass",
                   side_effect=fake_claim_and_dispatch), \
             patch("app.workers.pipeline.count_terminal_passes", return_value=0), \
             patch("app.workers.pipeline.settings.pass_concurrency_per_document", 4):
            _invoke_advance(db, _DOC_ID, _RUN_ID)

        assert captured_kwargs.get("build_index_failed") is False, (
            "When run.metrics is None, build_index_failed must default to False; "
            f"got captured_kwargs={captured_kwargs!r}"
        )
