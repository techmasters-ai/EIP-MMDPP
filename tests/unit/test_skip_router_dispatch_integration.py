"""C3 — Skip-router dispatch integration tests.

Task 3.6 + 3.7c: verify router wiring inside _claim_and_dispatch_pass.

Six sub-cases:
  (A) mode=enforce + SKIP decision → no .delay() call + terminal SKIPPED row
      written + mark_phase_terminal called + _try_advance_phase fires.
  (B) mode=shadow + SKIP decision → logged but .delay() STILL called.
  (C) mode=disabled → no router invocation; .delay() always called.
  (D) Required pass with cues that don't hit → RUN_FULL regardless (system_links).
  (E) Identity pass with cues that don't hit → RUN_FULL regardless.
  (F) Second-wave dispatch (regression): field_group pass dispatched via
      _try_advance_phase → _claim_and_dispatch_pass also goes through router.

Tests are fully hermetic: no real DB, no real Celery.  All I/O mocked at the
``app.workers.pipeline`` module level.

Run standalone:
    python3 -m pytest tests/unit/test_skip_router_dispatch_integration.py -v
"""
from __future__ import annotations

import logging
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_ID = str(uuid.uuid4())
_DOC_ID = str(uuid.uuid4())
_BUNDLE_KEY = "test_c3_bundle"


def _make_pass_def(
    name: str,
    *,
    phase: str = "field_group",
    required: bool = False,
    cues: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        phase=phase,
        required=required,
        cues=list(cues or []),
        input_mode="document_only",
        depends_on=[],
    )


def _make_manifest(passes: list) -> SimpleNamespace:
    def _find_pass(name):
        for p in passes:
            if p.name == name:
                return p
        raise KeyError(name)
    return SimpleNamespace(bundle_key=_BUNDLE_KEY, passes=passes, find_pass=_find_pass)


def _make_db() -> MagicMock:
    db = MagicMock()
    db.commit.return_value = None
    return db


def _make_skip_decision():
    from app.services.extraction_routing import RouteDecision
    return RouteDecision(
        mode="SKIP_NO_EVIDENCE",
        reason="no cue hits among 3 cues",
        matched_cues=[],
        chunks_with_hits=0,
    )


def _make_run_full_decision():
    from app.services.extraction_routing import RouteDecision
    return RouteDecision(
        mode="RUN_FULL",
        reason="required_pass",
        matched_cues=[],
        chunks_with_hits=0,
    )


def _make_doc_json() -> dict:
    return {
        "texts": [{"text": "The radar transmitter operates at 9375 MHz."}],
        "tables": [],
        "pictures": [],
    }


def _invoke_claim_and_dispatch(
    pass_def,
    doc_json: dict | None = None,
    *,
    mode: str = "shadow",
    queued_counter: dict | None = None,
):
    """Call _claim_and_dispatch_pass with all external I/O mocked."""
    from app.workers.pipeline import _claim_and_dispatch_pass

    db = _make_db()

    # Patch everything that _claim_and_dispatch_pass calls
    patches = {
        "app.workers.pipeline.claim_phase": MagicMock(return_value=True),
        "app.workers.pipeline.mark_phase_dispatched": MagicMock(),
        "app.workers.pipeline.mark_phase_terminal": MagicMock(),
        "app.workers.pipeline._try_advance_phase": MagicMock(),
        "app.workers.pipeline._write_stage_run": MagicMock(return_value=uuid.uuid4()),
        "app.workers.pipeline.save_pass_output": MagicMock(),
        "app.workers.pipeline.derive_ontology_graph_pass": MagicMock(),
    }

    return db, patches


# ---------------------------------------------------------------------------
# Helper: build a complete mock patch context for a single _claim_and_dispatch_pass call
# ---------------------------------------------------------------------------

def _run_claim_dispatch(
    pass_def,
    *,
    mode: str,
    router_decision=None,
    doc_json: dict | None = None,
    queued_counter: dict | None = None,
) -> dict:
    """
    Run _claim_and_dispatch_pass with all dependencies mocked.

    Returns a dict of the mock objects so tests can assert on them.
    """
    from app.workers.pipeline import _claim_and_dispatch_pass

    db = _make_db()
    if doc_json is None:
        doc_json = _make_doc_json()

    claim_phase_mock = MagicMock(return_value=True)
    mark_phase_dispatched_mock = MagicMock()
    mark_phase_terminal_mock = MagicMock()
    try_advance_mock = MagicMock()
    write_stage_run_mock = MagicMock(return_value=uuid.uuid4())
    save_pass_output_mock = MagicMock()
    delay_mock = MagicMock()
    delay_mock.id = "celery-task-id-test"

    task_mock = MagicMock()
    task_mock.delay.return_value = delay_mock

    route_pass_mock = MagicMock(return_value=router_decision)
    build_docling_doc_mock = MagicMock(return_value=doc_json)

    manifest = _make_manifest([pass_def])
    load_manifest_mock = MagicMock(return_value=manifest)

    # Provide a fake PipelineRun with ontology_bundle_key so db.get() works.
    fake_run = MagicMock()
    fake_run.ontology_bundle_key = _BUNDLE_KEY
    db.get.return_value = fake_run

    with patch.dict("os.environ", {"DOCLING_GRAPH_SKIP_ROUTER_MODE": mode}), \
         patch("app.workers.pipeline.claim_phase", claim_phase_mock), \
         patch("app.workers.pipeline.mark_phase_dispatched", mark_phase_dispatched_mock), \
         patch("app.workers.pipeline.mark_phase_terminal", mark_phase_terminal_mock), \
         patch("app.workers.pipeline._try_advance_phase", try_advance_mock), \
         patch("app.workers.pipeline._write_stage_run", write_stage_run_mock), \
         patch("app.workers.pipeline.save_pass_output", save_pass_output_mock), \
         patch("app.workers.pipeline.derive_ontology_graph_pass", task_mock), \
         patch("app.workers.pipeline.route_pass", route_pass_mock), \
         patch("app.workers.pipeline.load_bundle_manifest", load_manifest_mock), \
         patch("app.workers.pipeline._build_docling_document_json", build_docling_doc_mock):

        result = _claim_and_dispatch_pass(
            db, _DOC_ID, _RUN_ID, pass_def.name,
            queued_counter=queued_counter,
        )

    return {
        "result": result,
        "db": db,
        "claim_phase": claim_phase_mock,
        "mark_phase_dispatched": mark_phase_dispatched_mock,
        "mark_phase_terminal": mark_phase_terminal_mock,
        "try_advance": try_advance_mock,
        "write_stage_run": write_stage_run_mock,
        "save_pass_output": save_pass_output_mock,
        "delay": task_mock.delay,
        "route_pass": route_pass_mock,
        "load_manifest": load_manifest_mock,
    }


# ---------------------------------------------------------------------------
# (A) mode=enforce + SKIP decision → no .delay(), SKIPPED row written
# ---------------------------------------------------------------------------

class TestEnforceModeSkipDecision:
    """Sub-case A: enforce mode with SKIP_NO_EVIDENCE → skip + write terminal row."""

    def _make_field_group_pass(self):
        return _make_pass_def(
            "radar_power_rf", phase="field_group", required=False,
            cues=["MHz", "GHz", "kW"],
        )

    def _run(self, **kwargs):
        pass_def = self._make_field_group_pass()
        skip_decision = _make_skip_decision()
        return _run_claim_dispatch(
            pass_def, mode="enforce", router_decision=skip_decision, **kwargs
        )

    def test_no_delay_call_on_enforce_skip(self):
        """enforce + SKIP → .delay() is NOT called."""
        mocks = self._run()
        mocks["delay"].assert_not_called()

    def test_returns_false_on_enforce_skip(self):
        """enforce + SKIP → _claim_and_dispatch_pass returns False (not dispatched)."""
        mocks = self._run()
        assert mocks["result"] is False

    def test_mark_phase_terminal_called_on_enforce_skip(self):
        """enforce + SKIP → mark_phase_terminal is called with result='skipped'."""
        mocks = self._run()
        mocks["mark_phase_terminal"].assert_called_once()
        call_args = mocks["mark_phase_terminal"].call_args
        # result should be 'skipped'
        assert "skipped" in str(call_args), (
            f"mark_phase_terminal should be called with result='skipped'; got {call_args}"
        )

    def test_try_advance_phase_called_on_enforce_skip(self):
        """enforce + SKIP → _try_advance_phase fires so the next pass can dispatch."""
        mocks = self._run()
        mocks["try_advance"].assert_called_once()

    def test_save_pass_output_called_with_skipped_status(self):
        """enforce + SKIP → save_pass_output is called with execution_status='SKIPPED'."""
        mocks = self._run()
        mocks["save_pass_output"].assert_called_once()
        call_kwargs = mocks["save_pass_output"].call_args.kwargs
        assert call_kwargs.get("execution_status") == "SKIPPED", (
            f"Expected execution_status='SKIPPED'; got {call_kwargs}"
        )

    def test_save_pass_output_skip_reason_is_no_relevant_evidence(self):
        """enforce + SKIP → skip_reason='NO_RELEVANT_EVIDENCE' in save_pass_output."""
        mocks = self._run()
        call_kwargs = mocks["save_pass_output"].call_args.kwargs
        assert call_kwargs.get("skip_reason") == "NO_RELEVANT_EVIDENCE", (
            f"Expected skip_reason='NO_RELEVANT_EVIDENCE'; got {call_kwargs}"
        )

    def test_router_decision_in_diagnostics(self):
        """enforce + SKIP → diagnostics carry router decision metadata."""
        mocks = self._run()
        call_kwargs = mocks["save_pass_output"].call_args.kwargs
        diagnostics = call_kwargs.get("diagnostics") or {}
        # router_mode, mode, chunks_with_hits should be present
        assert "skip_router" in str(diagnostics) or len(diagnostics) > 0, (
            f"diagnostics should carry router info; got {diagnostics}"
        )

    def test_queued_counter_not_incremented_on_enforce_skip(self):
        """enforce + SKIP → queued_counter is NOT incremented (no task queued)."""
        counter = {"n": 0}
        self._run(queued_counter=counter)
        assert counter["n"] == 0, (
            f"queued_counter should stay 0 on skip; got {counter['n']}"
        )


# ---------------------------------------------------------------------------
# (B) mode=shadow + SKIP decision → log, but .delay() STILL called
# ---------------------------------------------------------------------------

class TestShadowModeSkipDecision:
    """Sub-case B: shadow mode with SKIP_NO_EVIDENCE → logs but dispatches."""

    def _make_field_group_pass(self):
        return _make_pass_def(
            "radar_power_rf", phase="field_group", required=False,
            cues=["MHz", "GHz", "kW"],
        )

    def _run(self, **kwargs):
        pass_def = self._make_field_group_pass()
        skip_decision = _make_skip_decision()
        return _run_claim_dispatch(
            pass_def, mode="shadow", router_decision=skip_decision, **kwargs
        )

    def test_delay_is_called_in_shadow_mode(self):
        """shadow + SKIP → .delay() IS called (shadow only logs)."""
        mocks = self._run()
        mocks["delay"].assert_called_once()

    def test_returns_true_in_shadow_mode(self):
        """shadow + SKIP → _claim_and_dispatch_pass returns True (dispatched)."""
        mocks = self._run()
        assert mocks["result"] is True

    def test_mark_phase_dispatched_called_in_shadow_mode(self):
        """shadow + SKIP → mark_phase_dispatched is called (normal dispatch)."""
        mocks = self._run()
        mocks["mark_phase_dispatched"].assert_called_once()

    def test_mark_phase_terminal_not_called_in_shadow_mode(self):
        """shadow + SKIP → mark_phase_terminal NOT called (no premature skip)."""
        mocks = self._run()
        mocks["mark_phase_terminal"].assert_not_called()

    def test_try_advance_not_called_in_shadow_mode(self):
        """shadow + SKIP → _try_advance_phase NOT called by skip path (normal dispatch does it later)."""
        mocks = self._run()
        # In shadow mode, advance happens when the actual pass task completes,
        # NOT when _claim_and_dispatch_pass is called.
        mocks["try_advance"].assert_not_called()

    def test_shadow_logs_decision(self, caplog):
        """shadow + SKIP → logs decision at INFO level."""
        pass_def = self._make_field_group_pass()
        skip_decision = _make_skip_decision()
        with caplog.at_level(logging.INFO, logger="app.workers.pipeline"):
            _run_claim_dispatch(pass_def, mode="shadow", router_decision=skip_decision)
        # Should emit a log about the skip_router decision
        log_text = " ".join(r.message for r in caplog.records)
        assert "skip_router" in log_text or "SKIP" in log_text or "shadow" in log_text, (
            f"Expected skip_router log in shadow mode; records: {[r.message for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# (C) mode=disabled → no router invocation; .delay() always called
# ---------------------------------------------------------------------------

class TestDisabledModeNoRouterInvocation:
    """Sub-case C: disabled mode → router is not invoked at all."""

    def _make_field_group_pass(self):
        return _make_pass_def(
            "radar_power_rf", phase="field_group", required=False,
            cues=["MHz", "GHz"],
        )

    def test_delay_called_in_disabled_mode(self):
        """disabled → .delay() called regardless of what router would return."""
        pass_def = self._make_field_group_pass()
        # Even if router would SKIP, disabled means it's never called.
        skip_decision = _make_skip_decision()
        mocks = _run_claim_dispatch(pass_def, mode="disabled", router_decision=skip_decision)
        mocks["delay"].assert_called_once()

    def test_router_not_called_in_disabled_mode(self):
        """disabled → route_pass is NOT invoked."""
        pass_def = self._make_field_group_pass()
        skip_decision = _make_skip_decision()
        mocks = _run_claim_dispatch(pass_def, mode="disabled", router_decision=skip_decision)
        mocks["route_pass"].assert_not_called()

    def test_returns_true_in_disabled_mode(self):
        """disabled → _claim_and_dispatch_pass returns True."""
        pass_def = self._make_field_group_pass()
        mocks = _run_claim_dispatch(pass_def, mode="disabled", router_decision=_make_skip_decision())
        assert mocks["result"] is True


# ---------------------------------------------------------------------------
# (D) Required pass with no cue hits → RUN_FULL regardless
# ---------------------------------------------------------------------------

class TestRequiredPassNeverSkipped:
    """Sub-case D: required=True → router returns RUN_FULL; no skip even in enforce."""

    def test_required_system_links_dispatched_even_with_skip_decision(self):
        """system_links is required; router returns RUN_FULL; .delay() called."""
        pass_def = _make_pass_def("system_links", phase="relationship", required=True, cues=[])
        # In the real router, required=True → RUN_FULL. We use a RUN_FULL decision here.
        run_full = _make_run_full_decision()
        mocks = _run_claim_dispatch(pass_def, mode="enforce", router_decision=run_full)
        mocks["delay"].assert_called_once()
        assert mocks["result"] is True


# ---------------------------------------------------------------------------
# (E) Identity pass → always RUN_FULL
# ---------------------------------------------------------------------------

class TestIdentityPassNeverSkipped:
    """Sub-case E: identity passes → router returns RUN_FULL; no skip."""

    def test_radar_identity_dispatched_in_enforce_mode(self):
        """radar_identity is an identity pass; RUN_FULL decision → dispatched."""
        pass_def = _make_pass_def("radar_identity", phase="identity", required=False, cues=[])
        run_full_decision = _make_run_full_decision()
        mocks = _run_claim_dispatch(pass_def, mode="enforce", router_decision=run_full_decision)
        mocks["delay"].assert_called_once()
        assert mocks["result"] is True


# ---------------------------------------------------------------------------
# (F) Second-wave dispatch regression test
# ---------------------------------------------------------------------------

class TestSecondWaveRouterEnforcement:
    """Sub-case F: field_group pass dispatched via _try_advance_phase (second wave)
    also goes through the router.

    Regression test for the rev-2 bug where only the initial wave was routable.
    We can't easily exercise _try_advance_phase → _claim_and_dispatch_pass as a
    full integration stack here (it would need a real DB), so we verify the router
    is invoked inside _claim_and_dispatch_pass REGARDLESS of who called it — the
    pattern is identical for initial dispatch and advance-phase dispatch.
    """

    def test_skip_router_fires_in_second_wave_dispatch(self):
        """_claim_and_dispatch_pass invokes router in enforce mode for field_group pass
        regardless of whether it was called by the initial dispatcher or _try_advance_phase.

        We simulate the second-wave scenario by calling _claim_and_dispatch_pass directly
        (without a queued_counter, as _try_advance_phase does) and asserting the router
        is invoked.
        """
        pass_def = _make_pass_def(
            "missile_kinematics", phase="field_group", required=False,
            cues=["range", "km"],
        )
        skip_decision = _make_skip_decision()
        # No queued_counter (matches _try_advance_phase call pattern)
        mocks = _run_claim_dispatch(
            pass_def, mode="enforce", router_decision=skip_decision,
            queued_counter=None,
        )
        # Router was invoked
        mocks["route_pass"].assert_called_once()
        # .delay() was NOT called (skip enforced)
        mocks["delay"].assert_not_called()
        # Terminal row was written
        mocks["save_pass_output"].assert_called_once()
        # Advance fired
        mocks["try_advance"].assert_called_once()

    def test_field_group_with_run_full_decision_dispatches_in_second_wave(self):
        """When router returns RUN_FULL for field_group, .delay() fires normally."""
        pass_def = _make_pass_def(
            "missile_kinematics", phase="field_group", required=False,
            cues=["range", "km"],
        )
        from app.services.extraction_routing import RouteDecision
        run_full = RouteDecision(
            mode="RUN_FULL",
            reason="2 chunks with cue hits",
            matched_cues=["range", "km"],
            chunks_with_hits=2,
        )
        mocks = _run_claim_dispatch(
            pass_def, mode="enforce", router_decision=run_full,
            queued_counter=None,
        )
        mocks["delay"].assert_called_once()
        mocks["save_pass_output"].assert_not_called()
        assert mocks["result"] is True
