"""C0 service-side telemetry — unit test for the ``PhaseTimer`` helper that
the docling-graph service uses to record per-phase wall time into a
diagnostics dict.

The full /extract-pass HTTP response shape (8 service-side metric keys) is
verified by a container-side smoke run during Task #7; that integration
check cannot run in the host environment because the ``docling_graph``
library is only installed inside the docling-graph container image.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest


def _load_telemetry_module():
    """Load docker/docling-graph/app/_telemetry.py directly by file path.

    Going through ``from app._telemetry import ...`` resolves to the repo-root
    ``app/`` package because pytest's rootdir wins on sys.path. Loading by
    file path bypasses that and matches the pattern used by test_alias_map.py.
    """
    if "_dg_telemetry" in sys.modules:
        return sys.modules["_dg_telemetry"]
    path = Path(__file__).resolve().parent.parent / "app" / "_telemetry.py"
    spec = importlib.util.spec_from_file_location("_dg_telemetry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dg_telemetry"] = module
    spec.loader.exec_module(module)
    return module


class TestPhaseTimer:
    """Context manager that records the elapsed milliseconds of a code
    block into a dict, keyed by phase name."""

    def test_records_elapsed_ms_in_sink_under_key(self):
        PhaseTimer = _load_telemetry_module().PhaseTimer

        sink: dict = {}
        with PhaseTimer(sink, "sanitize_ms"):
            time.sleep(0.005)

        assert "sanitize_ms" in sink
        assert isinstance(sink["sanitize_ms"], float)
        # Should be at least ~5ms; allow generous lower bound for slow CI
        assert sink["sanitize_ms"] >= 1.0
        # Bound: 5ms sleep shouldn't take more than 500ms on any sane host
        assert sink["sanitize_ms"] < 500.0

    def test_multiple_phases_recorded_independently(self):
        PhaseTimer = _load_telemetry_module().PhaseTimer

        sink: dict = {}
        with PhaseTimer(sink, "sanitize_ms"):
            time.sleep(0.002)
        with PhaseTimer(sink, "postprocess_ms"):
            time.sleep(0.003)

        assert "sanitize_ms" in sink
        assert "postprocess_ms" in sink
        # Different keys, different durations
        assert sink["sanitize_ms"] != sink["postprocess_ms"]

    def test_records_zero_or_more_on_no_work(self):
        PhaseTimer = _load_telemetry_module().PhaseTimer

        sink: dict = {}
        with PhaseTimer(sink, "noop_ms"):
            pass

        assert sink["noop_ms"] >= 0.0
        # A no-op block should be sub-millisecond on any modern machine
        assert sink["noop_ms"] < 5.0

    def test_records_even_when_block_raises(self):
        """If the timed block raises, the timing must still be recorded so
        we don't lose diagnostic data on failure paths."""
        PhaseTimer = _load_telemetry_module().PhaseTimer

        sink: dict = {}
        with pytest.raises(ValueError, match="boom"):
            with PhaseTimer(sink, "failed_phase_ms"):
                time.sleep(0.002)
                raise ValueError("boom")

        assert "failed_phase_ms" in sink
        assert sink["failed_phase_ms"] >= 1.0

    def test_overwrites_existing_key(self):
        """A second time through the same phase overwrites — last write wins.
        This is the expected behavior for a per-pass diagnostics dict."""
        PhaseTimer = _load_telemetry_module().PhaseTimer

        sink: dict = {"sanitize_ms": 999.0}
        with PhaseTimer(sink, "sanitize_ms"):
            pass
        assert sink["sanitize_ms"] < 5.0
