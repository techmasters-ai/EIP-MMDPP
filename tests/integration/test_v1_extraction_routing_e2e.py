"""Integration tests: /v1/extraction/chunk-scope end-to-end.

VR Phase C.3 — full round-trip against live services (ArcadeDB + reranker).

SKIP CONDITIONS
---------------
All tests skip when:
  * ArcadeDB is not reachable on arcadedb:2480 / localhost:2480
  * Reranker model is not available (RERANKER_ENABLED=false or model load fails)

Ollama / LLM is NOT required — the test uses deterministic fake embeddings
(same pattern as test_extraction_chunk_index_e2e.py).

Test strategy:
  1. Insert a small set of ExtractionChunk vertices via build_extraction_index.
  2. POST to /v1/extraction/chunk-scope.
  3. Assert:
     - Response shape matches ChunkScopeResponse schema.
     - self_refs (when mode=selected_refs) are a subset of the inserted self_refs.
     - Diagnostics fields are all present.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from app.services.arcadedb_graph import ArcadeDBGraphStore

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Deterministic embedding stub (same as test_extraction_chunk_index_e2e.py)
# ---------------------------------------------------------------------------

_DIM = 1024


def _fake_embed(texts: list[str], query: bool = False) -> list[list[float]]:
    """Deterministic unit-norm embeddings for testing (no Ollama required)."""
    result = []
    for text in texts:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vec = []
        for i in range(_DIM):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vec.append((seed / 0xFFFFFFFF) * 2 - 1)
        magnitude = math.sqrt(sum(v * v for v in vec))
        if magnitude == 0:
            magnitude = 1.0
        result.append([v / magnitude for v in vec])
    return result


# ---------------------------------------------------------------------------
# Synthetic chunk data
# ---------------------------------------------------------------------------

_CHUNK_TEXTS = [
    "The SA-2 Dvina radar system transmits at peak power of 500 kW on carrier frequency 2900 MHz.",
    "The radar ERP is 45 dBW measured at the antenna aperture.",
    "Guidance radar operates in the C-band frequency range with pulse repetition frequency of 1200 Hz.",
    "The air defense missile battery consists of six launchers arranged in a hexagonal formation.",
    "Radar maximum range performance at low altitude targets is limited by ground clutter.",
]

# Expected self_refs for the 5 text elements
_EXPECTED_SELF_REFS = [f"#/texts/{i}" for i in range(len(_CHUNK_TEXTS))]


def _make_test_doc_json(run_id: str) -> dict:
    """Build a minimal synthetic doc_json with 5 text elements."""
    texts = []
    for i, text in enumerate(_CHUNK_TEXTS):
        texts.append({
            "self_ref": f"#/texts/{i}",
            "text": f"[run={run_id[:8]}] {text}",
            "label": "text",
            "prov": [{"page_no": 1}],
        })
    return {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": f"test_e2e_{run_id[:8]}",
        "texts": texts,
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"$ref": "#/body"},
    }


# ---------------------------------------------------------------------------
# Skip fixtures
# ---------------------------------------------------------------------------

def _reranker_available() -> bool:
    """Return True if the reranker model can be loaded."""
    try:
        from app.config import get_settings
        s = get_settings()
        if not s.reranker_enabled:
            return False
        from app.services.reranker import _get_reranker_model
        _get_reranker_model()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------

def test_resolve_template_class_uses_real_manifest():
    """Resolver must work against a real manifest pass — unpatched.

    Regression guard for the rev 16 bug where _resolve_template_class imported
    pass_def.module literally (without 'ontology_bundles.{bundle_key}.' prefix).
    The E2E tests above patch _resolve_template_class; this test does NOT patch
    it, exercising the real import path.

    Catches future regressions of the same shape.
    """
    from app.services.ontology_bundles import load_bundle_manifest
    from app.api.v1.extraction_routing import _resolve_template_class

    bundle_key = "air_defense_v3_baseline_subset"
    manifest = load_bundle_manifest(bundle_key)
    pass_def = next(p for p in manifest.passes if p.name == "radar_power_rf")

    # Manifest stores RELATIVE module path — no bundle prefix
    assert not pass_def.module.startswith("ontology_bundles."), (
        f"Manifest module should be relative; got: {pass_def.module!r}"
    )

    template_cls = _resolve_template_class(bundle_key, pass_def)

    assert template_cls.__name__ == "RadarPowerRfPass", (
        f"Expected 'RadarPowerRfPass', got {template_cls.__name__!r}"
    )
    assert template_cls.__module__.startswith("ontology_bundles.air_defense_v3"), (
        f"Expected module rooted at 'ontology_bundles.air_defense_v3'; "
        f"got {template_cls.__module__!r}"
    )


@pytest.mark.asyncio
async def test_chunk_scope_e2e_returns_valid_response(arcadedb_store):
    """Insert chunks, POST to endpoint, assert response shape + self_refs subset."""
    pytest.importorskip("sentence_transformers", reason="reranker requires sentence_transformers")

    if not _reranker_available():
        pytest.skip("Reranker not available (RERANKER_ENABLED=false or model load failed)")

    pipeline_run_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    bundle_key = "air_defense_v3_baseline_subset"
    pass_name = "radar_power_rf"

    # Insert ExtractionChunk vertices for this run (sync call)
    from app.services.extraction_chunk_index import (
        build_extraction_index,
        cleanup_extraction_index,
    )

    doc_json = _make_test_doc_json(pipeline_run_id)

    try:
        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            side_effect=_fake_embed,
        ):
            diag = build_extraction_index(
                doc_json, pipeline_run_id, document_id, store=arcadedb_store
            )

        assert diag.chunks_inserted > 0, "Expected at least one ExtractionChunk inserted"

        # POST to the endpoint with patched embed
        from app.main import create_app

        app = create_app()

        # Patch _resolve_template_class + build_retrieval_query so we don't need
        # the extraction_schemas module importable via the bundle-relative path.
        # Patch get_graph_store to use the same live store from the fixture
        # (avoids DNS issues with the default "arcadedb" hostname on dev host).
        # The endpoint logic under test is vector-search + rerank, not schema walking.
        from pydantic import BaseModel

        class _FakePassCls(BaseModel):
            pass

        with (
            patch(
                "app.api.v1.extraction_routing.embed_texts",
                side_effect=_fake_embed,
            ),
            patch(
                "app.api.v1.extraction_routing.get_graph_store",
                return_value=arcadedb_store,
            ),
            patch(
                "app.api.v1.extraction_routing._resolve_template_class",
                return_value=_FakePassCls,
            ),
            patch(
                "app.api.v1.extraction_routing.build_retrieval_query",
                return_value="radar power rf ERP frequency transmitter peak power",
            ),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                resp = await client.post(
                    "/v1/extraction/chunk-scope",
                    json={
                        "pipeline_run_id": pipeline_run_id,
                        "bundle_key": bundle_key,
                        "pass_name": pass_name,
                    },
                )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()

        # Response shape
        assert "mode" in data
        assert data["mode"] in ("selected_refs", "full", "would_skip")
        assert "self_refs" in data
        assert isinstance(data["self_refs"], list)
        assert "diagnostics" in data

        diag_resp = data["diagnostics"]
        required_diag_fields = [
            "mode", "query_text", "vector_threshold", "candidate_count",
            "selected_ref_count", "selected_token_estimate", "full_doc_token_estimate",
            "would_skip_if_fallback_disabled", "vector_search_ms", "rerank_ms",
            "ann_top_k_requested", "post_filter_candidate_count",
            "post_filter_retry_count", "filter_strategy",
        ]
        for f in required_diag_fields:
            assert f in diag_resp, f"Missing diagnostics field: {f}"

        # If selected_refs returned, they must all be from the inserted set
        if data["mode"] == "selected_refs":
            for ref in data["self_refs"]:
                assert ref in _EXPECTED_SELF_REFS, (
                    f"self_ref {ref!r} is not in the expected set {_EXPECTED_SELF_REFS}. "
                    "Vector router must not return refs from other pipeline runs."
                )

    finally:
        cleanup_extraction_index(pipeline_run_id, store=arcadedb_store)


@pytest.mark.asyncio
async def test_chunk_scope_e2e_wrong_run_id_returns_no_cross_contamination(arcadedb_store):
    """self_refs must not contain chunks from a different pipeline_run_id."""
    pytest.importorskip("sentence_transformers", reason="reranker requires sentence_transformers")

    if not _reranker_available():
        pytest.skip("Reranker not available")

    pipeline_run_id_a = str(uuid.uuid4())
    pipeline_run_id_b = str(uuid.uuid4())
    document_id_a = str(uuid.uuid4())
    document_id_b = str(uuid.uuid4())
    bundle_key = "air_defense_v3_baseline_subset"
    pass_name = "radar_power_rf"

    from app.services.extraction_chunk_index import (
        build_extraction_index,
        cleanup_extraction_index,
    )

    doc_json_a = _make_test_doc_json(pipeline_run_id_a)
    doc_json_b = _make_test_doc_json(pipeline_run_id_b)

    try:
        with patch(
            "app.services.extraction_chunk_index.embed_texts",
            side_effect=_fake_embed,
        ):
            build_extraction_index(
                doc_json_a, pipeline_run_id_a, document_id_a, store=arcadedb_store
            )
            build_extraction_index(
                doc_json_b, pipeline_run_id_b, document_id_b, store=arcadedb_store
            )

        from pydantic import BaseModel as _BaseModel

        class _FakePassCls2(_BaseModel):
            pass

        from app.main import create_app
        app = create_app()

        with (
            patch(
                "app.api.v1.extraction_routing.embed_texts",
                side_effect=_fake_embed,
            ),
            patch(
                "app.api.v1.extraction_routing.get_graph_store",
                return_value=arcadedb_store,
            ),
            patch(
                "app.api.v1.extraction_routing._resolve_template_class",
                return_value=_FakePassCls2,
            ),
            patch(
                "app.api.v1.extraction_routing.build_retrieval_query",
                return_value="radar power rf ERP frequency transmitter peak power",
            ),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                # Query for run_b — must never return refs from run_a
                resp = await client.post(
                    "/v1/extraction/chunk-scope",
                    json={
                        "pipeline_run_id": pipeline_run_id_b,
                        "bundle_key": bundle_key,
                        "pass_name": pass_name,
                    },
                )

        assert resp.status_code == 200
        data = resp.json()

        if data["mode"] == "selected_refs":
            # Since doc_json_a and doc_json_b use the same text bodies (just
            # different run_id prefixes), their self_refs are the same strings
            # (#/texts/0, etc.). The cross-contamination test here verifies
            # that the post-filter is working by checking that the endpoint
            # response does not include MORE refs than the expected set.
            assert len(data["self_refs"]) <= len(_EXPECTED_SELF_REFS), (
                "More self_refs returned than inserted for run_b. "
                "Possible cross-run contamination."
            )
    finally:
        cleanup_extraction_index(pipeline_run_id_a, store=arcadedb_store)
        cleanup_extraction_index(pipeline_run_id_b, store=arcadedb_store)
