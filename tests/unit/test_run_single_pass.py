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
