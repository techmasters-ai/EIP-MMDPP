"""Unit tests for _execute_pass_attempt — single-attempt helper (Task 4).

Verifies that _execute_pass_attempt returns the correct PassAttemptOutcome
for all exit paths: SKIPPED, COMPLETE, and the three FAILED sub-cases
(PassRetryable, PassTransportError, PassTerminal during HTTP call, and
PassTerminal during parse).

Mock pattern mirrors tests/unit/test_run_single_pass.py — no real DB
or HTTP connections; all infrastructure patched at the module level.
"""
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest


# --- Shared fixtures (copied from test_run_single_pass.py) ------------------

def _fake_pass_def(
    *,
    name="radar_domain",
    kind="entities_and_relationships",
    input_mode="document_only",
    required=True,
    depends_on=None,
    skip_if_no_upstream_endpoints=False,
    primary=("RADAR_SYSTEM",),
    bridge=(),
    rels=("INSTALLED_ON",),
    module="extraction_schemas.radar_domain",
    template_class="RadarDomainPass",
):
    return SimpleNamespace(
        name=name,
        kind=kind,
        input_mode=input_mode,
        required=required,
        depends_on=list(depends_on or []),
        skip_if_no_upstream_endpoints=skip_if_no_upstream_endpoints,
        primary_entity_types=list(primary),
        bridge_entity_types=list(bridge),
        extracted_relationship_types=list(rels),
        module=module,
        template_class=template_class,
    )


def _fake_manifest(passes):
    return SimpleNamespace(
        bundle_key="air_defense_v3",
        passes=passes,
    )


MINIMAL_ONTOLOGY = {
    "validation_matrix": [
        {"source": "RADAR_SYSTEM", "relationship": "INSTALLED_ON", "target": "PLATFORM"},
    ],
}


def _common_call_kwargs(pass_def=None, upstream_refs=None, manifest=None):
    """Return the standard kwargs for _execute_pass_attempt."""
    if pass_def is None:
        pass_def = _fake_pass_def()
    if upstream_refs is None:
        upstream_refs = {}
    if manifest is None:
        manifest = _fake_manifest([pass_def])
    return dict(
        pipeline_run_id="run-1",
        pass_def=pass_def,
        manifest=manifest,
        ontology=MINIMAL_ONTOLOGY,
        bundle_key="air_defense_v3",
        doc_json={"text": "..."},
        upstream_refs=upstream_refs,
        document_id="doc-1",
    )


# --- Test cases -------------------------------------------------------------

class TestExecutePassAttempt:

    # 1. Skip outcome
    def test_skip_outcome_when_no_upstream_endpoints(self):
        """When _should_skip returns True, outcome is SKIPPED with
        skip_reason='NO_UPSTREAM_ENDPOINTS' and no HTTP call is made."""
        from app.workers.pipeline import _execute_pass_attempt

        pass_def = _fake_pass_def(
            kind="relationships_only",
            skip_if_no_upstream_endpoints=True,
            depends_on=["radar_domain"],
            rels=["INSTALLED_ON"],
        )

        with patch("app.workers.pipeline._call_extract_pass") as mock_call:
            outcome = _execute_pass_attempt(**_common_call_kwargs(pass_def=pass_def))

        assert outcome.execution_status == "SKIPPED"
        assert outcome.skip_reason == "NO_UPSTREAM_ENDPOINTS"
        assert outcome.pass_result is None
        assert outcome.raw_response_payload is None
        assert outcome.counts is None
        assert outcome.error is None
        assert mock_call.call_count == 0

    # 2. Complete outcome — raw payload captured
    def test_complete_outcome_carries_raw_payload(self):
        """On success, raw_response_payload must match exactly what
        _call_extract_pass returned."""
        from app.workers.pipeline import _execute_pass_attempt

        pass_def = _fake_pass_def()
        fake_raw = {"pass_output": {"nodes": []}, "metadata": {"node_count": 0}}
        fake_pass_result = SimpleNamespace(
            pass_name=pass_def.name,
            template_instance=SimpleNamespace(),
            metadata=SimpleNamespace(schema_size_chars=1000, structured_output_mode="strict"),
            pre_merge_rejections=[],
            relationships=[],
        )

        with patch("app.workers.pipeline._call_extract_pass", return_value=fake_raw), \
             patch("app.workers.pipeline._parse_pass_response", return_value=fake_pass_result), \
             patch("app.workers.pipeline._count_pass_output", return_value={
                 "primary_entities_extracted": 1,
                 "bridge_entities_extracted": 0,
                 "relationships_extracted": 0,
                 "relationships_rejected": 0,
                 "schema_size_chars": 1000,
                 "structured_output_mode": "strict",
                 "salvaged": False,
             }), \
             patch("app.workers.pipeline.classify_yield", return_value="HIT"):
            outcome = _execute_pass_attempt(**_common_call_kwargs(pass_def=pass_def))

        assert outcome.execution_status == "COMPLETE"
        assert outcome.raw_response_payload is fake_raw

    # 3. Complete outcome — pass_result + pre_merge_walk + counts populated
    def test_complete_outcome_carries_pass_result(self):
        """On success, pass_result.pre_merge_walk is set and counts is non-None."""
        from app.workers.pipeline import _execute_pass_attempt

        pass_def = _fake_pass_def()
        fake_raw = {"pass_output": {}, "metadata": {}}
        fake_pass_result = SimpleNamespace(
            pass_name=pass_def.name,
            template_instance=SimpleNamespace(),
            metadata=SimpleNamespace(schema_size_chars=500, structured_output_mode="strict"),
            pre_merge_rejections=[],
            relationships=[],
        )
        fake_counts = {
            "primary_entities_extracted": 2,
            "bridge_entities_extracted": 0,
            "relationships_extracted": 1,
            "relationships_rejected": 0,
            "schema_size_chars": 500,
            "structured_output_mode": "strict",
            "salvaged": False,
        }

        with patch("app.workers.pipeline._call_extract_pass", return_value=fake_raw), \
             patch("app.workers.pipeline._parse_pass_response", return_value=fake_pass_result), \
             patch("app.workers.pipeline._count_pass_output", return_value=fake_counts), \
             patch("app.workers.pipeline.classify_yield", return_value="HIT"):
            outcome = _execute_pass_attempt(**_common_call_kwargs(pass_def=pass_def))

        assert outcome.execution_status == "COMPLETE"
        assert outcome.pass_result is fake_pass_result
        # pre_merge_walk must have been attached by the helper
        assert hasattr(outcome.pass_result, "pre_merge_walk")
        assert outcome.pass_result.pre_merge_walk is not None
        assert outcome.counts is not None
        assert outcome.error is None

    # 4. Failed outcome — PassRetryable during HTTP call
    def test_failed_outcome_on_pass_retryable(self):
        """When _call_extract_pass raises PassRetryable, outcome is FAILED
        with the error instance, and pass_result + raw_response_payload are None."""
        from app.workers.pipeline import _execute_pass_attempt, PassRetryable

        pass_def = _fake_pass_def()
        exc = PassRetryable("upstream timeout")

        with patch("app.workers.pipeline._call_extract_pass", side_effect=exc):
            outcome = _execute_pass_attempt(**_common_call_kwargs(pass_def=pass_def))

        assert outcome.execution_status == "FAILED"
        assert isinstance(outcome.error, PassRetryable)
        assert outcome.error is exc
        assert outcome.pass_result is None
        assert outcome.raw_response_payload is None
        assert outcome.counts is None
        assert outcome.skip_reason is None

    # 5. Failed outcome — PassTransportError during HTTP call
    def test_failed_outcome_on_pass_transport_error(self):
        """When _call_extract_pass raises PassTransportError, outcome is FAILED
        and error is both a PassTransportError AND a PassRetryable (subclass)."""
        from app.workers.pipeline import _execute_pass_attempt, PassTransportError, PassRetryable

        pass_def = _fake_pass_def()
        exc = PassTransportError("connection refused")

        with patch("app.workers.pipeline._call_extract_pass", side_effect=exc):
            outcome = _execute_pass_attempt(**_common_call_kwargs(pass_def=pass_def))

        assert outcome.execution_status == "FAILED"
        assert isinstance(outcome.error, PassTransportError)
        # Subclass relationship must hold so the wrapper's isinstance check works
        assert isinstance(outcome.error, PassRetryable)
        assert outcome.pass_result is None
        assert outcome.raw_response_payload is None

    # 6. Failed outcome — PassTerminal during HTTP call
    def test_failed_outcome_on_pass_terminal_during_call(self):
        """When _call_extract_pass raises PassTerminal, outcome is FAILED,
        error is PassTerminal, and raw_response_payload is None (no response
        was received)."""
        from app.workers.pipeline import _execute_pass_attempt, PassTerminal

        pass_def = _fake_pass_def()
        exc = PassTerminal("HTTP 400 bad request")

        with patch("app.workers.pipeline._call_extract_pass", side_effect=exc):
            outcome = _execute_pass_attempt(**_common_call_kwargs(pass_def=pass_def))

        assert outcome.execution_status == "FAILED"
        assert isinstance(outcome.error, PassTerminal)
        assert outcome.raw_response_payload is None
        assert outcome.pass_result is None

    # 7. Failed outcome — PassTerminal during parse (raw_payload IS captured)
    def test_failed_outcome_on_pass_terminal_during_parse(self):
        """When _call_extract_pass succeeds but _parse_pass_response raises
        PassTerminal, the raw payload must still be captured in
        raw_response_payload — useful forensic data for debugging."""
        from app.workers.pipeline import _execute_pass_attempt, PassTerminal

        pass_def = _fake_pass_def()
        fake_raw = {"pass_output": {"malformed": True}, "metadata": {}}
        exc = PassTerminal("template validation failed")

        with patch("app.workers.pipeline._call_extract_pass", return_value=fake_raw), \
             patch("app.workers.pipeline._parse_pass_response", side_effect=exc):
            outcome = _execute_pass_attempt(**_common_call_kwargs(pass_def=pass_def))

        assert outcome.execution_status == "FAILED"
        assert isinstance(outcome.error, PassTerminal)
        # Raw payload MUST be present — distinguishes this from a call-level failure
        assert outcome.raw_response_payload is fake_raw
        assert outcome.pass_result is None
        assert outcome.counts is None
