"""P6 success-path payload trim — unit tests for the helper that strips
``library_log`` from raw_response_payload.diagnostics on COMPLETE outcomes
before persistence.

Why: library_log is the full captured stdout/stderr of run_pipeline. On a
successful pass it's mostly dead weight in pipeline_pass_outputs (bloats
extract_pass_response_json and slows system_links rehydration). On FAILED
/ SKIPPED outcomes it's forensic gold and must be preserved.

Contract: helper returns a SHALLOW-COPIED payload — caller-visible mutation
of the original outcome.raw_response_payload is forbidden, since downstream
log emitters in main.py still read it after _save_terminal_pass_output
returns.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


LIBRARY_LOG_SAMPLE = (
    "[LlmBackend] Initialized with: gemma4:31b\n"
    "[DeltaExtraction] Calling LLM (batch mode)...\n"
    "[GraphConverter] Final graph: 23 nodes, 4 edges\n"
)


def _make_outcome(*, status: str, library_log=LIBRARY_LOG_SAMPLE, extra_diag=None,
                  payload_keys_extra=None):
    from app.workers.pipeline import PassAttemptOutcome

    diag: dict = {"sanitize_ms": 1.0, "run_pipeline_ms": 2.0}
    if library_log is not None:
        diag["library_log"] = library_log
    if extra_diag:
        diag.update(extra_diag)

    payload: dict = {
        "pass_output": {"radar_systems": [{"system_name": "Fan Song"}]},
        "metadata": {"node_count": 1, "edge_count": 0},
        "diagnostics": diag,
    }
    if payload_keys_extra:
        payload.update(payload_keys_extra)

    return PassAttemptOutcome(
        execution_status=status,
        skip_reason=None,
        yield_status="HIT" if status == "COMPLETE" else None,
        pass_result=None,
        raw_response_payload=payload,
        counts=None,
        error=None,
        worker_diagnostics={"pass_wall_ms": 100.0},
    )


class TestTrimResponseForPersistence:
    def test_drops_library_log_on_complete(self):
        from app.workers.pipeline import _trim_response_for_persistence

        outcome = _make_outcome(status="COMPLETE")
        trimmed = _trim_response_for_persistence(outcome)

        assert isinstance(trimmed, dict)
        assert "library_log" not in trimmed["diagnostics"]
        # Other diagnostics preserved
        assert trimmed["diagnostics"]["sanitize_ms"] == 1.0
        assert trimmed["diagnostics"]["run_pipeline_ms"] == 2.0
        # pass_output / metadata preserved verbatim
        assert trimmed["pass_output"] == {"radar_systems": [{"system_name": "Fan Song"}]}
        assert trimmed["metadata"]["node_count"] == 1

    def test_preserves_library_log_on_failed(self):
        from app.workers.pipeline import _trim_response_for_persistence

        outcome = _make_outcome(status="FAILED")
        trimmed = _trim_response_for_persistence(outcome)

        assert trimmed["diagnostics"]["library_log"] == LIBRARY_LOG_SAMPLE

    def test_preserves_library_log_on_skipped(self):
        """SKIPPED outcomes have raw_response_payload=None typically, but
        defensively: if a SKIPPED outcome ever carries a payload with
        library_log, it should be preserved (skips are quasi-forensic)."""
        from app.workers.pipeline import _trim_response_for_persistence

        outcome = _make_outcome(status="SKIPPED")
        trimmed = _trim_response_for_persistence(outcome)
        assert trimmed["diagnostics"]["library_log"] == LIBRARY_LOG_SAMPLE

    def test_handles_none_payload(self):
        """Transport-failed outcomes have raw_response_payload=None."""
        from app.workers.pipeline import (
            _trim_response_for_persistence,
            PassAttemptOutcome,
        )

        outcome = PassAttemptOutcome(
            execution_status="FAILED",
            skip_reason=None,
            yield_status=None,
            pass_result=None,
            raw_response_payload=None,
            counts=None,
            error=RuntimeError("boom"),
            worker_diagnostics={"pass_wall_ms": 1.0},
        )
        trimmed = _trim_response_for_persistence(outcome)
        # Helper must not crash; returns empty dict or None
        assert trimmed in (None, {})

    def test_handles_payload_with_no_diagnostics_key(self):
        from app.workers.pipeline import (
            _trim_response_for_persistence,
            PassAttemptOutcome,
        )

        outcome = PassAttemptOutcome(
            execution_status="COMPLETE",
            skip_reason=None,
            yield_status="HIT",
            pass_result=None,
            raw_response_payload={"pass_output": {}},
            counts=None,
            error=None,
            worker_diagnostics={"pass_wall_ms": 1.0},
        )
        trimmed = _trim_response_for_persistence(outcome)
        # Helper doesn't crash; pass_output preserved
        assert trimmed.get("pass_output") == {}

    def test_does_not_mutate_input_payload(self):
        """The original outcome.raw_response_payload must be intact after
        trimming — main.py's logging code reads it after _save_terminal_
        pass_output returns."""
        from app.workers.pipeline import _trim_response_for_persistence

        outcome = _make_outcome(status="COMPLETE")
        original_payload_id = id(outcome.raw_response_payload)
        original_diag_id = id(outcome.raw_response_payload["diagnostics"])

        _ = _trim_response_for_persistence(outcome)

        # Both dicts still have library_log (helper returned a copy, didn't mutate)
        assert "library_log" in outcome.raw_response_payload["diagnostics"]
        assert outcome.raw_response_payload["diagnostics"]["library_log"] == LIBRARY_LOG_SAMPLE
        # And the dicts themselves are the same objects (we didn't replace them)
        assert id(outcome.raw_response_payload) == original_payload_id
        assert id(outcome.raw_response_payload["diagnostics"]) == original_diag_id

    def test_trim_only_removes_library_log_not_other_diag_keys(self):
        """P6 scope is library_log only — service_identity_gate,
        service_postprocess, failed_batch_traces, and C0 timings must all
        survive the trim on COMPLETE."""
        from app.workers.pipeline import _trim_response_for_persistence

        outcome = _make_outcome(
            status="COMPLETE",
            extra_diag={
                "service_identity_gate": {"dropped_entities_by_field": {}},
                "service_postprocess": {"some_stat": 42},
                "failed_batch_traces": {"batch_3": {"_load_error": "x"}},
                "chunk_count": 10,
                "batch_count": 3,
            },
        )
        trimmed = _trim_response_for_persistence(outcome)
        diag = trimmed["diagnostics"]

        assert "library_log" not in diag
        # Everything else survives
        assert diag["service_identity_gate"] == {"dropped_entities_by_field": {}}
        assert diag["service_postprocess"] == {"some_stat": 42}
        assert diag["failed_batch_traces"] == {"batch_3": {"_load_error": "x"}}
        assert diag["chunk_count"] == 10
        assert diag["batch_count"] == 3
        assert diag["sanitize_ms"] == 1.0
