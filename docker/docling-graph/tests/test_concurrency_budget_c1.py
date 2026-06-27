"""C1 concurrency-budget warning — unit tests for the helper that
inspects the three concurrency knobs and emits a startup warning when
they're misconfigured relative to the LLM backend slot budget.

The helper lives in ``docker/docling-graph/app/_concurrency_budget.py``
and is called from the FastAPI lifespan. Loaded by file path to avoid
the repo-root ``app/`` package shadowing the docling-graph one.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path


def _load_budget_module():
    if "_dg_concurrency_budget" in sys.modules:
        return sys.modules["_dg_concurrency_budget"]
    path = Path(__file__).resolve().parent.parent / "app" / "_concurrency_budget.py"
    spec = importlib.util.spec_from_file_location("_dg_concurrency_budget", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dg_concurrency_budget"] = module
    spec.loader.exec_module(module)
    return module


class TestComputeEffectiveInFlight:
    def test_returns_product_of_max_concurrent_and_parallel_workers(self):
        m = _load_budget_module()
        assert m.compute_effective_in_flight(
            max_concurrent_extractions=2,
            parallel_workers=4,
        ) == 8

    def test_with_one_pass_at_a_time(self):
        m = _load_budget_module()
        assert m.compute_effective_in_flight(
            max_concurrent_extractions=1,
            parallel_workers=4,
        ) == 4

    def test_zero_workers_yields_zero(self):
        """Defensive: if parallel_workers is 0 (rare misconfig), effective
        in-flight is 0 — not a crash."""
        m = _load_budget_module()
        assert m.compute_effective_in_flight(
            max_concurrent_extractions=2,
            parallel_workers=0,
        ) == 0


class TestEmitConcurrencyBudgetWarning:
    """``emit_concurrency_budget_warning(settings, logger)`` examines the
    three knobs and logs at the appropriate level."""

    def _capture(self, caplog, level=logging.DEBUG):
        caplog.set_level(level, logger="_dg_concurrency_budget")
        return caplog

    def test_warns_when_cap_unset_and_effective_in_flight_above_threshold(self, caplog):
        m = _load_budget_module()
        self._capture(caplog)
        m.emit_concurrency_budget_warning(
            max_concurrent_extractions=2,
            parallel_workers=4,
            llm_max_in_flight=0,  # unset / disabled
        )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a WARNING when llm_max_in_flight=0 and effective>1"
        msg = warnings[0].getMessage()
        assert "effective_in_flight" in msg or "8" in msg
        assert "DOCLING_GRAPH_LLM_MAX_IN_FLIGHT" in msg

    def test_warns_when_cap_set_but_below_effective(self, caplog):
        m = _load_budget_module()
        self._capture(caplog)
        m.emit_concurrency_budget_warning(
            max_concurrent_extractions=2,
            parallel_workers=4,
            llm_max_in_flight=4,  # cap below 2*4=8
        )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected WARNING when cap below effective in-flight"
        msg = warnings[0].getMessage()
        # Message must mention both numbers so an operator can act
        assert "4" in msg and "8" in msg

    def test_info_when_cap_meets_or_exceeds_effective(self, caplog):
        m = _load_budget_module()
        self._capture(caplog)
        m.emit_concurrency_budget_warning(
            max_concurrent_extractions=2,
            parallel_workers=4,
            llm_max_in_flight=8,
        )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert not warnings, "no warning expected when cap >= effective"
        assert infos, "expected INFO confirming healthy budget"

    def test_message_includes_formula(self, caplog):
        """Operators reading the warning should see the formula so they
        can reason about which knob to change."""
        m = _load_budget_module()
        self._capture(caplog)
        m.emit_concurrency_budget_warning(
            max_concurrent_extractions=2,
            parallel_workers=4,
            llm_max_in_flight=0,
        )
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "DOCLING_GRAPH_MAX_CONCURRENT_EXTRACTIONS" in msgs
        assert "DOCLING_GRAPH_PARALLEL_WORKERS" in msgs
