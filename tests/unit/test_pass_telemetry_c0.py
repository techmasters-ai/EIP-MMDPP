"""C0 telemetry — unit tests for the 13-metric per-pass diagnostics shape.

Phase 0 of the wall-time-reduction plan: every pass must record per-pass
timings, byte counts, chunk_count, and batch_count in the diagnostics
JSONB column so later phases can be A/B'd against the bdde417 baseline.

Worker captures 5 metrics, service captures 8. The merge happens in the
worker before persistence. These tests lock in the shape and key set; they
do NOT pin numeric values (those are environment-dependent).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


SERVICE_METRIC_KEYS = {
    "sanitize_ms",
    "table_overlay_ms",
    "table_normalization_ms",
    "run_pipeline_ms",
    "postprocess_ms",
    "field_provenance_ms",
    "chunk_count",
    "batch_count",
}

WORKER_METRIC_KEYS = {
    "doc_json_load_ms",
    "request_bytes",
    "service_queue_wait_ms",
    "response_bytes",
    "pass_wall_ms",
}

ALL_METRIC_KEYS = SERVICE_METRIC_KEYS | WORKER_METRIC_KEYS


class TestBuildPassDiagnosticsDict:
    """Helper that merges service-side + worker-side + override extras into
    the final diagnostics dict written to pipeline_pass_outputs.diagnostics.
    """

    def _make_outcome(self, *, service_diag: dict | None, worker_diag: dict | None):
        from app.workers.pipeline import PassAttemptOutcome

        return PassAttemptOutcome(
            execution_status="COMPLETE",
            skip_reason=None,
            yield_status="HIT",
            pass_result=None,
            raw_response_payload={"diagnostics": service_diag} if service_diag is not None else {},
            counts=None,
            error=None,
            worker_diagnostics=worker_diag,
        )

    def test_returns_dict_with_all_13_metric_keys(self):
        from app.workers.pipeline import _build_pass_diagnostics_dict

        outcome = self._make_outcome(
            service_diag={k: 1.0 for k in SERVICE_METRIC_KEYS},
            worker_diag={k: 1.0 for k in WORKER_METRIC_KEYS},
        )
        diag = _build_pass_diagnostics_dict(outcome, override_extra=None)

        for key in ALL_METRIC_KEYS:
            assert key in diag, f"diagnostics missing required C0 key: {key}"

    def test_preserves_existing_service_diagnostics_keys(self):
        """Existing service-side diagnostics (service_identity_gate,
        service_postprocess, etc.) must not be clobbered."""
        from app.workers.pipeline import _build_pass_diagnostics_dict

        outcome = self._make_outcome(
            service_diag={
                "service_identity_gate": {"dropped_entities_by_field": {}},
                "service_postprocess": {"some_stat": 42},
                "sanitize_ms": 1.0,
            },
            worker_diag={"pass_wall_ms": 2.0},
        )
        diag = _build_pass_diagnostics_dict(outcome, override_extra=None)

        assert diag["service_identity_gate"] == {"dropped_entities_by_field": {}}
        assert diag["service_postprocess"] == {"some_stat": 42}

    def test_override_extra_takes_precedence(self):
        from app.workers.pipeline import _build_pass_diagnostics_dict

        outcome = self._make_outcome(
            service_diag={"sanitize_ms": 99.0},
            worker_diag={"pass_wall_ms": 100.0, "request_bytes": 5},
        )
        diag = _build_pass_diagnostics_dict(
            outcome,
            override_extra={"retry_exhausted": True, "request_bytes": 999},
        )

        assert diag["retry_exhausted"] is True
        # override wins over both service and worker
        assert diag["request_bytes"] == 999

    def test_handles_missing_worker_diagnostics_gracefully(self):
        """A FAILED outcome may have worker_diagnostics=None — the helper
        must still return the service-side diagnostics without raising."""
        from app.workers.pipeline import _build_pass_diagnostics_dict

        outcome = self._make_outcome(
            service_diag={"sanitize_ms": 1.0},
            worker_diag=None,
        )
        diag = _build_pass_diagnostics_dict(outcome, override_extra=None)

        assert diag == {"sanitize_ms": 1.0}

    def test_handles_missing_service_diagnostics_gracefully(self):
        """A transport-error outcome has raw_response_payload=None — helper
        must still return worker-side diagnostics without raising."""
        from app.workers.pipeline import _build_pass_diagnostics_dict, PassAttemptOutcome

        outcome = PassAttemptOutcome(
            execution_status="FAILED",
            skip_reason=None,
            yield_status=None,
            pass_result=None,
            raw_response_payload=None,
            counts=None,
            error=RuntimeError("boom"),
            worker_diagnostics={"pass_wall_ms": 7.0},
        )
        diag = _build_pass_diagnostics_dict(outcome, override_extra=None)

        assert diag == {"pass_wall_ms": 7.0}


class TestExecutePassAttemptWorkerDiagnostics:
    """_execute_pass_attempt must capture the 5 worker-side metrics on every
    COMPLETE outcome and stash them on outcome.worker_diagnostics."""

    def _common_kwargs(self):
        pass_def = SimpleNamespace(
            name="radar_identity",
            kind="entities_and_relationships",
            input_mode="document_only",
            required=True,
            depends_on=[],
            skip_if_no_upstream_endpoints=False,
            primary_entity_types=["RADAR_SYSTEM"],
            bridge_entity_types=[],
            extracted_relationship_types=["INSTALLED_ON"],
            module="extraction_schemas.radar_identity",
            template_class="RadarIdentityPass",
        )
        manifest = SimpleNamespace(bundle_key="air_defense_v3", passes=[pass_def])
        return dict(
            pipeline_run_id="run-1",
            pass_def=pass_def,
            manifest=manifest,
            ontology={"validation_matrix": []},
            bundle_key="air_defense_v3",
            doc_json={"texts": [], "tables": [], "pictures": [], "body": {"children": []}},
            upstream_refs={},
            document_id="doc-1",
        )

    def test_complete_outcome_carries_worker_diagnostics(self):
        """The function captures its 4 measurable metrics (request_bytes,
        response_bytes, service_queue_wait_ms, pass_wall_ms) and merges any
        caller_worker_diag passed in (which is how doc_json_load_ms — owned
        by the outer caller that loaded the doc — flows through to the final
        diagnostics dict).
        """
        from app.workers.pipeline import _execute_pass_attempt

        fake_raw = {
            "pass_output": {},
            "metadata": {"node_count": 1, "edge_count": 0},
            "diagnostics": {"sanitize_ms": 1.0, "run_pipeline_ms": 2.0},
        }
        fake_pass_result = SimpleNamespace(
            pass_name="radar_identity",
            template_instance=SimpleNamespace(),
            metadata=SimpleNamespace(schema_size_chars=100, structured_output_mode="strict"),
            pre_merge_rejections=[],
            relationships=[],
        )

        kwargs = self._common_kwargs()
        kwargs["caller_worker_diag"] = {"doc_json_load_ms": 42.0}

        with patch("app.workers.pipeline._call_extract_pass", return_value=fake_raw), \
             patch("app.workers.pipeline._parse_pass_response", return_value=fake_pass_result), \
             patch("app.workers.pipeline._count_pass_output", return_value={
                 "primary_entities_extracted": 1,
                 "bridge_entities_extracted": 0,
                 "relationships_extracted": 0,
                 "relationships_rejected": 0,
                 "schema_size_chars": 100,
                 "structured_output_mode": "strict",
                 "salvaged": False,
             }), \
             patch("app.workers.pipeline.classify_yield", return_value="HIT"):
            outcome = _execute_pass_attempt(**kwargs)

        assert outcome.execution_status == "COMPLETE"
        assert outcome.worker_diagnostics is not None
        # Caller-supplied metric flows through
        assert outcome.worker_diagnostics["doc_json_load_ms"] == 42.0
        # Internally-captured metrics present and non-negative
        for key in ("request_bytes", "response_bytes", "service_queue_wait_ms", "pass_wall_ms"):
            assert key in outcome.worker_diagnostics, f"missing internally-captured key: {key}"
            assert isinstance(outcome.worker_diagnostics[key], (int, float))
            assert outcome.worker_diagnostics[key] >= 0
        # All 5 worker keys present in the union
        assert WORKER_METRIC_KEYS.issubset(outcome.worker_diagnostics.keys())

    def test_failed_outcome_still_carries_partial_worker_diagnostics(self):
        """When _call_extract_pass raises, the worker still captures pass_wall_ms
        + caller_worker_diag — even though request/response_bytes may be 0 and
        service_queue_wait_ms is undefined (response never arrived)."""
        from app.workers.pipeline import _execute_pass_attempt, PassRetryable

        kwargs = self._common_kwargs()
        kwargs["caller_worker_diag"] = {"doc_json_load_ms": 7.5}

        with patch("app.workers.pipeline._call_extract_pass", side_effect=PassRetryable("boom")):
            outcome = _execute_pass_attempt(**kwargs)

        assert outcome.execution_status == "FAILED"
        assert outcome.worker_diagnostics is not None
        assert outcome.worker_diagnostics["doc_json_load_ms"] == 7.5
        assert "pass_wall_ms" in outcome.worker_diagnostics
        assert outcome.worker_diagnostics["pass_wall_ms"] >= 0
