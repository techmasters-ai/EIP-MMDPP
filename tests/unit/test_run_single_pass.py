"""Spec §5.5 + §6.4 — per-pass dispatcher, skip logic, and required-pass gate.

Task 4.3 of the extraction-refactor plan. Mocks the HTTP helper
(_call_extract_pass) and the DB helpers so tests don't touch real
infrastructure. Every test is scoped to exactly one behavior from the
plan's 12-case list.
"""
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest


# --- Shared fixtures --------------------------------------------------------

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


# --- _should_skip -----------------------------------------------------------

class TestShouldSkip:

    def test_entities_pass_never_skipped(self):
        from app.workers.pipeline import _should_skip
        p = _fake_pass_def(kind="entities_and_relationships", skip_if_no_upstream_endpoints=False)
        assert _should_skip(p, {}, MINIMAL_ONTOLOGY) is False

    def test_relationships_only_without_flag_not_skipped(self):
        from app.workers.pipeline import _should_skip
        p = _fake_pass_def(kind="relationships_only", skip_if_no_upstream_endpoints=False)
        assert _should_skip(p, {}, MINIMAL_ONTOLOGY) is False

    def test_relationships_only_empty_upstream_skipped(self):
        from app.workers.pipeline import _should_skip
        p = _fake_pass_def(
            kind="relationships_only",
            skip_if_no_upstream_endpoints=True,
            depends_on=["radar_domain"],
            rels=["INSTALLED_ON"],
        )
        assert _should_skip(p, {}, MINIMAL_ONTOLOGY) is True

    def test_relationships_only_satisfiable_triple_not_skipped(self):
        from app.workers.pipeline import _should_skip
        p = _fake_pass_def(
            kind="relationships_only",
            skip_if_no_upstream_endpoints=True,
            depends_on=["radar_domain", "missile_domain"],
            rels=["INSTALLED_ON"],
        )
        upstream_refs = {
            "ref-1": SimpleNamespace(entity_type="RADAR_SYSTEM", pass_origin="radar_domain"),
            "ref-2": SimpleNamespace(entity_type="PLATFORM", pass_origin="radar_domain"),
        }
        assert _should_skip(p, upstream_refs, MINIMAL_ONTOLOGY) is False

    def test_relationships_only_no_valid_triple_skipped(self):
        """Upstream refs exist but no (source, rel, target) in validation_matrix matches."""
        from app.workers.pipeline import _should_skip
        p = _fake_pass_def(
            kind="relationships_only",
            skip_if_no_upstream_endpoints=True,
            depends_on=["radar_domain"],
            rels=["INSTALLED_ON"],
        )
        upstream_refs = {
            "ref-1": SimpleNamespace(entity_type="RADAR_SYSTEM", pass_origin="radar_domain"),
            # no PLATFORM — the INSTALLED_ON triple is not satisfiable
        }
        assert _should_skip(p, upstream_refs, MINIMAL_ONTOLOGY) is True


# --- _run_single_pass ------------------------------------------------------

class TestRunSinglePass:

    def test_skipped_writes_stage_run_and_returns(self):
        """When _should_skip returns True, write a SKIPPED StageRun and return without calling the HTTP path."""
        from app.workers.pipeline import _run_single_pass

        pass_def = _fake_pass_def(
            kind="relationships_only",
            skip_if_no_upstream_endpoints=True,
            depends_on=["radar_domain"],
            rels=["INSTALLED_ON"],
            required=True,
        )
        manifest = _fake_manifest([pass_def])
        pass_results: dict = {}

        with patch("app.workers.pipeline._write_stage_run") as mock_write, \
             patch("app.workers.pipeline._call_extract_pass") as mock_call:
            _run_single_pass(
                pipeline_run_id="run-1",
                pass_def=pass_def,
                manifest=manifest,
                ontology=MINIMAL_ONTOLOGY,
                bundle_key="air_defense_v3",
                doc_json={},
                pass_results=pass_results,
                upstream_refs={},
                document_id="doc-1",
            )

        assert mock_call.call_count == 0
        assert mock_write.call_count == 1
        kwargs = mock_write.call_args.kwargs
        assert kwargs["execution_status"] == "SKIPPED"
        assert kwargs["skip_reason"] == "NO_UPSTREAM_ENDPOINTS"
        assert pass_def.name not in pass_results

    def test_happy_path_writes_complete_and_populates_results(self):
        from app.workers.pipeline import _run_single_pass

        pass_def = _fake_pass_def()
        manifest = _fake_manifest([pass_def])
        pass_results: dict = {}

        # Mock _call_extract_pass → _parse_pass_response
        fake_pass_result = SimpleNamespace(
            pass_name=pass_def.name,
            template_instance=SimpleNamespace(),
            metadata=SimpleNamespace(schema_size_chars=1000, structured_output_mode="strict"),
            pre_merge_rejections=[],
            relationships=[],
        )

        with patch("app.workers.pipeline._call_extract_pass") as mock_call, \
             patch("app.workers.pipeline._parse_pass_response", return_value=fake_pass_result), \
             patch("app.workers.pipeline._write_stage_run") as mock_write, \
             patch("app.workers.pipeline._count_pass_output", return_value={
                 "primary_entities_extracted": 2,
                 "bridge_entities_extracted": 0,
                 "relationships_extracted": 1,
                 "relationships_rejected": 0,
                 "schema_size_chars": 1000,
                 "structured_output_mode": "strict",
                 "salvaged": False,
             }), \
             patch("app.workers.pipeline.classify_yield", return_value="HIT"):
            mock_call.return_value = {"pass_output": {}, "metadata": {}}
            _run_single_pass(
                pipeline_run_id="run-1",
                pass_def=pass_def,
                manifest=manifest,
                ontology=MINIMAL_ONTOLOGY,
                bundle_key="air_defense_v3",
                doc_json={"text": "..."},
                pass_results=pass_results,
                upstream_refs={},
                document_id="doc-1",
            )

        assert pass_def.name in pass_results
        assert mock_write.call_args.kwargs["execution_status"] == "COMPLETE"
        assert mock_write.call_args.kwargs["yield_status"] == "HIT"

    def test_retryable_retries_up_to_max(self):
        """PassRetryable → retry until max, then give up."""
        from app.workers.pipeline import _run_single_pass, PassRetryable

        pass_def = _fake_pass_def(required=False)
        manifest = _fake_manifest([pass_def])
        pass_results: dict = {}

        # Always raise retryable
        with patch("app.workers.pipeline._call_extract_pass", side_effect=PassRetryable("timeout")), \
             patch("app.workers.pipeline._write_stage_run") as mock_write, \
             patch("app.workers.pipeline._backoff"):  # skip real backoff sleep
            _run_single_pass(
                pipeline_run_id="run-1",
                pass_def=pass_def,
                manifest=manifest,
                ontology=MINIMAL_ONTOLOGY,
                bundle_key="air_defense_v3",
                doc_json={},
                pass_results=pass_results,
                upstream_refs={},
                document_id="doc-1",
            )

        # One StageRun row per attempt
        assert mock_write.call_count == 3  # pass_max_retries default
        for i, call in enumerate(mock_write.call_args_list, start=1):
            assert call.kwargs["attempt"] == i
            assert call.kwargs["execution_status"] == "FAILED"

    def test_retryable_exhausted_raises_ingest_failed_if_required(self):
        from app.workers.pipeline import _run_single_pass, PassRetryable, IngestFailed

        pass_def = _fake_pass_def(required=True)
        manifest = _fake_manifest([pass_def])

        with patch("app.workers.pipeline._call_extract_pass", side_effect=PassRetryable("timeout")), \
             patch("app.workers.pipeline._write_stage_run"), \
             patch("app.workers.pipeline._backoff"):
            with pytest.raises(IngestFailed, match="exhausted retries"):
                _run_single_pass(
                    pipeline_run_id="run-1",
                    pass_def=pass_def,
                    manifest=manifest,
                    ontology=MINIMAL_ONTOLOGY,
                    bundle_key="air_defense_v3",
                    doc_json={},
                    pass_results={},
                    upstream_refs={},
                    document_id="doc-1",
                )

    def test_transport_error_does_not_burn_business_retry_budget(self):
        """PassTransportError uses its own counter — business attempt stays at 1
        across all transport retries until either transport budget is exhausted
        or a non-transport response arrives.
        """
        from app.workers.pipeline import _run_single_pass, PassTransportError

        pass_def = _fake_pass_def(required=False)
        manifest = _fake_manifest([pass_def])

        with patch(
            "app.workers.pipeline._call_extract_pass",
            side_effect=PassTransportError("server disconnected"),
        ), patch("app.workers.pipeline._write_stage_run") as mock_write, \
             patch("app.workers.pipeline._backoff"):
            _run_single_pass(
                pipeline_run_id="run-1",
                pass_def=pass_def,
                manifest=manifest,
                ontology=MINIMAL_ONTOLOGY,
                bundle_key="air_defense_v3",
                doc_json={},
                pass_results={},
                upstream_refs={},
                document_id="doc-1",
            )

        # Default pass_max_transport_retries=3 → 3 StageRun rows, all attempt=1
        assert mock_write.call_count == 3
        for call in mock_write.call_args_list:
            assert call.kwargs["attempt"] == 1
            assert call.kwargs["execution_status"] == "FAILED"
            assert "transport-retry" in call.kwargs["error"]

    def test_transport_error_exhausted_raises_if_required(self):
        from app.workers.pipeline import (
            _run_single_pass, PassTransportError, IngestFailed,
        )

        pass_def = _fake_pass_def(required=True)
        manifest = _fake_manifest([pass_def])

        with patch(
            "app.workers.pipeline._call_extract_pass",
            side_effect=PassTransportError("name resolution failed"),
        ), patch("app.workers.pipeline._write_stage_run"), \
             patch("app.workers.pipeline._backoff"):
            with pytest.raises(IngestFailed, match="exhausted transport retries"):
                _run_single_pass(
                    pipeline_run_id="run-1",
                    pass_def=pass_def,
                    manifest=manifest,
                    ontology=MINIMAL_ONTOLOGY,
                    bundle_key="air_defense_v3",
                    doc_json={},
                    pass_results={},
                    upstream_refs={},
                    document_id="doc-1",
                )

    def test_terminal_error_raises_immediately_if_required(self):
        from app.workers.pipeline import _run_single_pass, PassTerminal, IngestFailed

        pass_def = _fake_pass_def(required=True)
        manifest = _fake_manifest([pass_def])

        with patch("app.workers.pipeline._call_extract_pass", side_effect=PassTerminal("4xx")), \
             patch("app.workers.pipeline._write_stage_run") as mock_write:
            with pytest.raises(IngestFailed, match="terminal failure"):
                _run_single_pass(
                    pipeline_run_id="run-1",
                    pass_def=pass_def,
                    manifest=manifest,
                    ontology=MINIMAL_ONTOLOGY,
                    bundle_key="air_defense_v3",
                    doc_json={},
                    pass_results={},
                    upstream_refs={},
                    document_id="doc-1",
                )

        # Only one StageRun row — no retry
        assert mock_write.call_count == 1
        assert mock_write.call_args.kwargs["attempt"] == 1
        assert mock_write.call_args.kwargs["execution_status"] == "FAILED"


# --- check_required_pass_gate -----------------------------------------------

class TestCheckRequiredPassGate:

    def test_all_complete_passes_gate(self):
        from app.workers.pipeline import check_required_pass_gate

        manifest = _fake_manifest([
            _fake_pass_def(name="radar_domain", required=True),
            _fake_pass_def(name="missile_domain", required=True),
        ])

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._get_db") as mock_get_db:
            session = MagicMock()
            mock_get_db.return_value = session

            # All passes have a COMPLETE StageRun
            def fake_query(model):
                q = MagicMock()
                q.filter.return_value = q
                q.order_by.return_value = q
                q.first.return_value = SimpleNamespace(
                    execution_status="COMPLETE",
                    skip_reason=None,
                    error_message=None,
                )
                return q
            session.query = fake_query
            session.get = MagicMock(return_value=SimpleNamespace(ontology_bundle_key="air_defense_v3"))

            result = check_required_pass_gate("00000000-0000-0000-0000-000000000001")

        assert result.passed is True
        assert result.failures == []

    def test_failed_pass_fails_gate(self):
        from app.workers.pipeline import check_required_pass_gate

        manifest = _fake_manifest([
            _fake_pass_def(name="radar_domain", required=True),
        ])

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._get_db") as mock_get_db:
            session = MagicMock()
            mock_get_db.return_value = session
            q = MagicMock()
            q.filter.return_value = q
            q.order_by.return_value = q
            q.first.return_value = SimpleNamespace(
                execution_status="FAILED",
                skip_reason=None,
                error_message="timeout",
            )
            session.query = MagicMock(return_value=q)
            session.get = MagicMock(return_value=SimpleNamespace(ontology_bundle_key="air_defense_v3"))

            result = check_required_pass_gate("00000000-0000-0000-0000-000000000001")

        assert result.passed is False
        assert len(result.failures) == 1
        assert "radar_domain" in result.failures[0][0]

    def test_authorized_skip_passes_gate(self):
        from app.workers.pipeline import check_required_pass_gate

        manifest = _fake_manifest([
            _fake_pass_def(name="system_links", required=True),
        ])

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._get_db") as mock_get_db:
            session = MagicMock()
            mock_get_db.return_value = session
            q = MagicMock()
            q.filter.return_value = q
            q.order_by.return_value = q
            q.first.return_value = SimpleNamespace(
                execution_status="SKIPPED",
                skip_reason="NO_UPSTREAM_ENDPOINTS",
                error_message=None,
            )
            session.query = MagicMock(return_value=q)
            session.get = MagicMock(return_value=SimpleNamespace(ontology_bundle_key="air_defense_v3"))

            result = check_required_pass_gate("00000000-0000-0000-0000-000000000001")

        assert result.passed is True

    def test_empty_anchor_set_skip_passes_gate(self):
        from app.workers.pipeline import check_required_pass_gate

        manifest = _fake_manifest([
            _fake_pass_def(name="system_links", required=True),
        ])

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._get_db") as mock_get_db:
            session = MagicMock()
            mock_get_db.return_value = session
            q = MagicMock()
            q.filter.return_value = q
            q.order_by.return_value = q
            q.first.return_value = SimpleNamespace(
                execution_status="SKIPPED",
                skip_reason="EMPTY_ANCHOR_SET",
                error_message=None,
            )
            session.query = MagicMock(return_value=q)
            session.get = MagicMock(return_value=SimpleNamespace(ontology_bundle_key="air_defense_v3"))

            result = check_required_pass_gate("00000000-0000-0000-0000-000000000001")

        assert result.passed is True

    def test_unauthorized_skip_fails_gate(self):
        from app.workers.pipeline import check_required_pass_gate

        manifest = _fake_manifest([
            _fake_pass_def(name="radar_domain", required=True),
        ])

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._get_db") as mock_get_db:
            session = MagicMock()
            mock_get_db.return_value = session
            q = MagicMock()
            q.filter.return_value = q
            q.order_by.return_value = q
            q.first.return_value = SimpleNamespace(
                execution_status="SKIPPED",
                skip_reason="WEIRD_REASON",
                error_message=None,
            )
            session.query = MagicMock(return_value=q)
            session.get = MagicMock(return_value=SimpleNamespace(ontology_bundle_key="air_defense_v3"))

            result = check_required_pass_gate("00000000-0000-0000-0000-000000000001")

        assert result.passed is False
        assert "WEIRD_REASON" in result.failures[0][1]

    def test_missing_stage_run_raises_worker_invariant_error(self):
        from app.workers.pipeline import check_required_pass_gate, WorkerInvariantError

        manifest = _fake_manifest([
            _fake_pass_def(name="radar_domain", required=True),
        ])

        with patch("app.workers.pipeline.load_bundle_manifest", return_value=manifest), \
             patch("app.workers.pipeline._get_db") as mock_get_db:
            session = MagicMock()
            mock_get_db.return_value = session
            q = MagicMock()
            q.filter.return_value = q
            q.order_by.return_value = q
            q.first.return_value = None  # no StageRun
            session.query = MagicMock(return_value=q)
            session.get = MagicMock(return_value=SimpleNamespace(ontology_bundle_key="air_defense_v3"))

            with pytest.raises(WorkerInvariantError, match="radar_domain"):
                check_required_pass_gate("00000000-0000-0000-0000-000000000001")


# --- TestPassResultUpstreamRefsAttachment -----------------------------------

class TestPassResultUpstreamRefsAttachment:
    """After Task 4, document_plus_entity_refs passes must have
    pass_result.upstream_refs populated with LogicalIdentity values so
    merge_and_resolve can match from_ref_id / to_ref_id. document_only
    passes must NOT have upstream_refs attached."""

    ONTOLOGY = {
        "entity_types": [
            {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"],
             "identity_scope": "global"},
            {"name": "MISSILE_SYSTEM", "identity_fields": ["system_name"],
             "identity_scope": "global"},
        ],
        "validation_matrix": [
            {"source": "RADAR_SYSTEM", "relationship": "ASSOCIATED_WITH",
             "target": "MISSILE_SYSTEM"},
        ],
    }

    def _fake_pass_result(self):
        return SimpleNamespace(
            pass_name="system_links",
            template_instance=SimpleNamespace(),
            metadata=SimpleNamespace(schema_size_chars=500,
                                     structured_output_mode="strict"),
            pre_merge_rejections=[],
            relationships=[],
            upstream_refs=None,  # _run_single_pass should populate this
        )

    def test_document_plus_entity_refs_pass_attaches_upstream_refs(self):
        """system_links is document_plus_entity_refs; after the parsed
        result returns, pass_result.upstream_refs must be a dict of
        LogicalIdentity objects keyed by ref_id (filtered through
        _select_upstream_refs_for_pass)."""
        from app.workers.pipeline import _run_single_pass
        from app.services.extraction_merge import LogicalIdentity

        system_links_def = _fake_pass_def(
            name="system_links",
            kind="relationships_only",
            input_mode="document_plus_entity_refs",
            required=True,
            depends_on=["radar_domain", "missile_domain"],
            primary=(),  # relationships_only — no primary entity output
            bridge=(),
            rels=("ASSOCIATED_WITH",),
        )
        manifest = _fake_manifest([system_links_def])
        pass_results: dict = {}

        # Upstream refs accumulated by earlier (mocked) passes.
        accumulated_refs = {
            "E001": SimpleNamespace(
                pass_origin="radar_domain",
                entity_type="RADAR_SYSTEM",
                identity_values={"system_name": "Fan Song"},
                display_label="Fan Song",
            ),
            "E002": SimpleNamespace(
                pass_origin="missile_domain",
                entity_type="MISSILE_SYSTEM",
                identity_values={"system_name": "SA-2"},
                display_label="SA-2",
            ),
        }

        fake_pass_result = self._fake_pass_result()

        with patch("app.workers.pipeline._call_extract_pass") as mock_call, \
             patch("app.workers.pipeline._parse_pass_response",
                   return_value=fake_pass_result), \
             patch("app.workers.pipeline._write_stage_run"), \
             patch("app.workers.pipeline._count_pass_output", return_value={
                 "primary_entities_extracted": 0,
                 "bridge_entities_extracted": 0,
                 "relationships_extracted": 0,
                 "relationships_rejected": 0,
                 "schema_size_chars": 500,
                 "structured_output_mode": "strict",
                 "salvaged": False,
             }), \
             patch("app.workers.pipeline.classify_yield", return_value="HIT"):
            mock_call.return_value = {"pass_output": {}, "metadata": {}}
            _run_single_pass(
                pipeline_run_id="run-1",
                pass_def=system_links_def,
                manifest=manifest,
                ontology=self.ONTOLOGY,
                bundle_key="air_defense_v3",
                doc_json={},
                pass_results=pass_results,
                upstream_refs=accumulated_refs,
                document_id="doc-1",
            )

        # The pass was recorded.
        assert "system_links" in pass_results
        result = pass_results["system_links"]

        # Both refs attached as LogicalIdentity (both pass the shared
        # validity rule AND are valid endpoints for ASSOCIATED_WITH).
        # Bug #59 fix: keyed by ref_id only; cross-pass canonicalization
        # is handled by identity_aliases inside merge_and_resolve.
        assert result.upstream_refs is not None
        assert set(result.upstream_refs.keys()) == {"E001", "E002"}
        assert all(
            isinstance(v, LogicalIdentity)
            for v in result.upstream_refs.values()
        )
        assert result.upstream_refs["E001"].entity_type == "RADAR_SYSTEM"
        assert result.upstream_refs["E002"].entity_type == "MISSILE_SYSTEM"

    def test_document_only_pass_does_not_attach_upstream_refs(self):
        """radar_domain is document_only — pass_result.upstream_refs must
        remain None/absent even when accumulated upstream refs exist."""
        from app.workers.pipeline import _run_single_pass

        radar_def = _fake_pass_def(
            name="radar_domain",
            kind="entities_and_relationships",
            input_mode="document_only",
            primary=("RADAR_SYSTEM",),
            rels=("INSTALLED_ON",),
        )
        manifest = _fake_manifest([radar_def])
        pass_results: dict = {}

        accumulated_refs = {
            "E001": SimpleNamespace(
                pass_origin="reference",
                entity_type="SECTION",
                identity_values={"heading": "Intro"},
                display_label="Intro",
            ),
        }

        fake_pass_result = self._fake_pass_result()
        fake_pass_result.pass_name = "radar_domain"

        with patch("app.workers.pipeline._call_extract_pass") as mock_call, \
             patch("app.workers.pipeline._parse_pass_response",
                   return_value=fake_pass_result), \
             patch("app.workers.pipeline._write_stage_run"), \
             patch("app.workers.pipeline._count_pass_output", return_value={
                 "primary_entities_extracted": 0,
                 "bridge_entities_extracted": 0,
                 "relationships_extracted": 0,
                 "relationships_rejected": 0,
                 "schema_size_chars": 500,
                 "structured_output_mode": "strict",
                 "salvaged": False,
             }), \
             patch("app.workers.pipeline.classify_yield", return_value="HIT"):
            mock_call.return_value = {"pass_output": {}, "metadata": {}}
            _run_single_pass(
                pipeline_run_id="run-1",
                pass_def=radar_def,
                manifest=manifest,
                ontology=self.ONTOLOGY,
                bundle_key="air_defense_v3",
                doc_json={},
                pass_results=pass_results,
                upstream_refs=accumulated_refs,
                document_id="doc-1",
            )

        result = pass_results["radar_domain"]
        # document_only passes do not consume upstream refs, so the post-
        # parse attach block must not populate this field.
        assert result.upstream_refs is None


# --- _call_extract_pass: stub-on-error handling ----------------------------

class TestCallExtractPass:
    """When the docling-graph service catches an internal exception, it
    returns 200 with ``diagnostics.pipeline_error`` set and an empty
    pass_output (a ``stub`` response). The worker must surface this as
    PassRetryable so the retry loop can have another go, instead of treating
    the empty stub as a clean success.
    """

    def _make_response(self, *, status_code=200, payload):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json = MagicMock(return_value=payload)
        resp.text = ""
        return resp

    def test_pipeline_error_stub_raises_pass_retryable(self):
        from app.workers.pipeline import _call_extract_pass, PassRetryable

        stub_payload = {
            "pass_output": {},
            "metadata": {"node_count": 0, "edge_count": 0},
            "diagnostics": {
                "pipeline_error": {
                    "type": "PipelineError",
                    "message": "stage 'Extraction' failed",
                },
            },
        }
        with patch(
            "app.workers.pipeline.httpx.post",
            return_value=self._make_response(payload=stub_payload),
        ):
            with pytest.raises(PassRetryable, match="service pipeline_error"):
                _call_extract_pass(
                    {"bundle_key": "air_defense_v3", "pass_name": "missile_identity",
                     "document_id": "doc-1"},
                    timeout=10.0,
                )

    def test_clean_zero_yield_does_not_raise(self):
        """Empty result without a pipeline_error marker is a legitimate
        outcome (off-domain pass) — log only, do not retry.
        """
        from app.workers.pipeline import _call_extract_pass

        clean_empty = {
            "pass_output": {},
            "metadata": {"node_count": 0, "edge_count": 0},
            "diagnostics": {},
        }
        with patch(
            "app.workers.pipeline.httpx.post",
            return_value=self._make_response(payload=clean_empty),
        ):
            result = _call_extract_pass(
                {"bundle_key": "air_defense_v3", "pass_name": "missile_identity",
                 "document_id": "doc-1"},
                timeout=10.0,
            )
            assert result == clean_empty


# ---------------------------------------------------------------------------
# Plan Task 9 — _parse_pass_response reads response_json["table_overlay"]
# onto PassResult.table_overlay (Mechanism A1, spec §4.4 + §5.5).
# ---------------------------------------------------------------------------


def test_parse_pass_response_reads_table_overlay():
    """When response_json carries a table_overlay key, _parse_pass_response
    must populate PassResult.table_overlay with a parsed TableOverlay
    instance. None when key is missing or value is null."""
    from app.workers.pipeline import _parse_pass_response
    from app.services.table_overlay import TableOverlay

    pass_def = type("PD", (), {
        "name": "missile_propulsion",
        "module": "extraction_schemas.missile_propulsion",
        "template_class": "MissilePropulsionPass",
    })()
    manifest = type("M", (), {"bundle_key": "air_defense_v3"})()

    response_json = {
        "bundle_key": "air_defense_v3",
        "pass_name": "missile_propulsion",
        "pass_output": {"missile_systems": []},
        "metadata": {"node_count": 0, "edge_count": 0},
        "provenance": [],
        "field_provenance": [],
        "table_overlay": {
            "alias_map_by_entity_type": {
                "MISSILE_SYSTEM": {"SA-75": "1D"},
            },
            "facts": [{
                "canonical_entity": "1D",
                "entity_type": "MISSILE_SYSTEM",
                "schema_field": "booster_mass_kg",
                "value": 1135.0,
                "source_label": "Weight kg",
                "section_ctx": "1st Stage",
                "pass_name": "missile_propulsion",
                "raw_text": "1135",
            }],
            "cross_entity_hints": [],
        },
    }

    result = _parse_pass_response(response_json, pass_def, manifest)
    assert isinstance(result.table_overlay, TableOverlay)
    assert result.table_overlay.alias_map_by_entity_type == {
        "MISSILE_SYSTEM": {"SA-75": "1D"},
    }
    assert len(result.table_overlay.facts) == 1
    assert result.table_overlay.facts[0].canonical_entity == "1D"


def test_parse_pass_response_table_overlay_missing_is_none():
    from app.workers.pipeline import _parse_pass_response
    pass_def = type("PD", (), {
        "name": "missile_propulsion",
        "module": "extraction_schemas.missile_propulsion",
        "template_class": "MissilePropulsionPass",
    })()
    manifest = type("M", (), {"bundle_key": "air_defense_v3"})()
    response_json = {
        "bundle_key": "air_defense_v3", "pass_name": "missile_propulsion",
        "pass_output": {"missile_systems": []},
        "metadata": {}, "provenance": [], "field_provenance": [],
        # No table_overlay key
    }
    result = _parse_pass_response(response_json, pass_def, manifest)
    assert result.table_overlay is None


def test_parse_pass_response_malformed_table_overlay_is_dropped(caplog):
    """A malformed payload (e.g., wrong field types) must not crash;
    log a WARNING and set table_overlay=None."""
    import logging
    from app.workers.pipeline import _parse_pass_response
    pass_def = type("PD", (), {
        "name": "missile_propulsion",
        "module": "extraction_schemas.missile_propulsion",
        "template_class": "MissilePropulsionPass",
    })()
    manifest = type("M", (), {"bundle_key": "air_defense_v3"})()
    response_json = {
        "bundle_key": "air_defense_v3", "pass_name": "missile_propulsion",
        "pass_output": {"missile_systems": []},
        "metadata": {}, "provenance": [], "field_provenance": [],
        "table_overlay": {"alias_map_by_entity_type": "not a dict"},  # bogus
    }
    with caplog.at_level(logging.WARNING, logger="app.workers.pipeline"):
        result = _parse_pass_response(response_json, pass_def, manifest)
    assert result.table_overlay is None
    assert any(
        "table_overlay" in rec.message for rec in caplog.records
        if rec.levelno >= logging.WARNING
    )
