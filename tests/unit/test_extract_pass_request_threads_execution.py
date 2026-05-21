"""C2 — Worker plumbing: _build_extract_pass_request threads execution fields.

Verifies that when pass_def.execution carries override values, they are
correctly plumbed onto the /extract-pass HTTP body.

Run standalone:
    python3 -m pytest tests/unit/test_extract_pass_request_threads_execution.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_pass_def(
    name: str = "radar_identity",
    *,
    execution=None,
) -> SimpleNamespace:
    """Minimal pass_def SimpleNamespace mirroring PassManifest fields."""
    return SimpleNamespace(
        name=name,
        required=False,
        kind="entities",
        input_mode="document_only",
        module="extraction_schemas.radar_identity",
        template_class="RadarIdentityPass",
        primary_entity_types=["RADAR_SYSTEM"],
        bridge_entity_types=[],
        extracted_relationship_types=[],
        depends_on=[],
        skip_if_no_upstream_endpoints=False,
        phase="identity",
        execution=execution,
    )


def _make_execution(
    *,
    llm_batch_token_size=None,
    temperature=None,
    max_tokens=None,
    chunk_max_tokens=None,
):
    return SimpleNamespace(
        llm_batch_token_size=llm_batch_token_size,
        temperature=temperature,
        max_tokens=max_tokens,
        chunk_max_tokens=chunk_max_tokens,
    )


DOC_JSON = {"texts": [], "tables": [], "pictures": [], "body": {"children": []}}


class TestBuildExtractPassRequestExecutionThreading:

    def test_no_execution_block_no_overrides_in_body(self):
        """When execution is None, no override fields are added to the body."""
        from app.workers.pipeline import _build_extract_pass_request

        body = _build_extract_pass_request(
            bundle_key="air_defense_v3",
            pass_def=_make_pass_def(),
            doc_json=DOC_JSON,
            upstream_refs=None,
            document_id="doc-1",
        )
        assert "llm_batch_token_size" not in body
        assert "temperature" not in body
        assert "max_tokens" not in body
        assert "chunk_max_tokens" not in body

    def test_execution_llm_batch_token_size_threaded_onto_body(self):
        """execution.llm_batch_token_size appears in the POST body."""
        from app.workers.pipeline import _build_extract_pass_request

        exec_prof = _make_execution(llm_batch_token_size=2048)
        body = _build_extract_pass_request(
            bundle_key="air_defense_v3",
            pass_def=_make_pass_def(execution=exec_prof),
            doc_json=DOC_JSON,
            upstream_refs=None,
            document_id="doc-1",
        )
        assert body["llm_batch_token_size"] == 2048

    def test_execution_temperature_threaded_onto_body(self):
        """execution.temperature appears in the POST body."""
        from app.workers.pipeline import _build_extract_pass_request

        exec_prof = _make_execution(temperature=0.3)
        body = _build_extract_pass_request(
            bundle_key="air_defense_v3",
            pass_def=_make_pass_def(execution=exec_prof),
            doc_json=DOC_JSON,
            upstream_refs=None,
            document_id="doc-1",
        )
        assert body["temperature"] == pytest.approx(0.3)

    def test_execution_max_tokens_threaded_onto_body(self):
        """execution.max_tokens appears in the POST body."""
        from app.workers.pipeline import _build_extract_pass_request

        exec_prof = _make_execution(max_tokens=8192)
        body = _build_extract_pass_request(
            bundle_key="air_defense_v3",
            pass_def=_make_pass_def(execution=exec_prof),
            doc_json=DOC_JSON,
            upstream_refs=None,
            document_id="doc-1",
        )
        assert body["max_tokens"] == 8192

    def test_execution_chunk_max_tokens_threaded_onto_body(self):
        """execution.chunk_max_tokens appears in the POST body."""
        from app.workers.pipeline import _build_extract_pass_request

        exec_prof = _make_execution(chunk_max_tokens=256)
        body = _build_extract_pass_request(
            bundle_key="air_defense_v3",
            pass_def=_make_pass_def(execution=exec_prof),
            doc_json=DOC_JSON,
            upstream_refs=None,
            document_id="doc-1",
        )
        assert body["chunk_max_tokens"] == 256

    def test_none_values_in_execution_are_not_threaded(self):
        """Fields that are None in execution should NOT appear in the body."""
        from app.workers.pipeline import _build_extract_pass_request

        exec_prof = _make_execution(llm_batch_token_size=2048)  # others None
        body = _build_extract_pass_request(
            bundle_key="air_defense_v3",
            pass_def=_make_pass_def(execution=exec_prof),
            doc_json=DOC_JSON,
            upstream_refs=None,
            document_id="doc-1",
        )
        assert body["llm_batch_token_size"] == 2048
        assert "temperature" not in body
        assert "max_tokens" not in body
        assert "chunk_max_tokens" not in body

    def test_all_four_execution_fields_threaded(self):
        """All four fields present in execution → all appear in body."""
        from app.workers.pipeline import _build_extract_pass_request

        exec_prof = _make_execution(
            llm_batch_token_size=2048,
            temperature=0.2,
            max_tokens=8192,
            chunk_max_tokens=256,
        )
        body = _build_extract_pass_request(
            bundle_key="air_defense_v3",
            pass_def=_make_pass_def(execution=exec_prof),
            doc_json=DOC_JSON,
            upstream_refs=None,
            document_id="doc-1",
        )
        assert body["llm_batch_token_size"] == 2048
        assert body["temperature"] == pytest.approx(0.2)
        assert body["max_tokens"] == 8192
        assert body["chunk_max_tokens"] == 256
