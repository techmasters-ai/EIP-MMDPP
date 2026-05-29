"""Unit tests for /v1/extraction/chunk-scope endpoint — VR Phase C.3.

All external dependencies are mocked (manifest load, embed_texts,
search_extraction_chunks, reranker.rerank). No live ArcadeDB, Ollama,
or reranker model required.

TDD discipline: tests were written before the implementation module.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

_PIPELINE_RUN_ID = str(uuid.uuid4())
_BUNDLE_KEY = "air_defense_v3_baseline_subset"
_PASS_NAME = "radar_power_rf"

_VALID_BODY = {
    "pipeline_run_id": _PIPELINE_RUN_ID,
    "bundle_key": _BUNDLE_KEY,
    "pass_name": _PASS_NAME,
}


@dataclass
class _FakeResult:
    """Minimal mock for GraphEntityResult used by the endpoint."""
    node_id: str = "rid:0:0"
    name: str = "chunk"
    entity_type: str = "ExtractionChunk"
    score: float = 0.75
    properties: dict = field(default_factory=dict)


@dataclass
class _FakeSearchDiag:
    ann_top_k_requested: int = 500
    post_filter_candidate_count: int = 5
    post_filter_retry_count: int = 0
    filter_strategy: str = "overfetch_post_filter"
    short_fetch: bool = False


def _make_result(self_ref: str, chunk_text: str, score: float = 0.75) -> _FakeResult:
    return _FakeResult(
        node_id=f"rid:0:{self_ref}",
        score=score,
        properties={"self_ref": self_ref, "chunk_text": chunk_text, "pipeline_run_id": _PIPELINE_RUN_ID},
    )


def _make_reranked(candidates: list[_FakeResult]) -> list[dict]:
    """Build a rerank output list from fake results (sorted by reranker_score desc)."""
    return [
        {
            "content_text": r.properties.get("chunk_text", ""),
            "self_ref": r.properties.get("self_ref"),
            "vector_score": r.score,
            "reranker_score": r.score - 0.01,  # slightly different from vector
        }
        for r in candidates
    ]


def _make_pass_def(
    name: str = _PASS_NAME,
    retrieval: Any = None,
    module: str = "ontology_bundles.air_defense_v3_baseline_subset.extraction_schemas.radar_power_rf",
    template_class: str = "RadarPowerRfPass",
    phase: str = "field_group",
):
    """Build a minimal PassManifest-like mock."""
    pd = MagicMock()
    pd.name = name
    pd.phase = phase
    pd.module = module
    pd.template_class = template_class
    if retrieval is None:
        # Default retrieval profile
        rp = MagicMock()
        rp.min_similarity = 0.45
        rp.top_n_candidates = 50
        rp.top_k = 20
        rp.fallback_to_full = True
        # C6: set field_query_top_k=0 so existing per-element-mode tests
        # never enter the multi-channel branch (condition requires > 0).
        rp.field_query_top_k = 0
        pd.retrieval = rp
    else:
        pd.retrieval = retrieval
    return pd


def _make_manifest(passes=None):
    """Build a minimal BundleManifest-like mock."""
    m = MagicMock()
    if passes is None:
        passes = [_make_pass_def()]
    m.passes = passes
    return m


# ---------------------------------------------------------------------------
# App fixture — creates a fresh FastAPI app for testing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    from app.main import create_app
    return create_app()


@pytest.fixture
async def client(app):
    """Async HTTP client wired to the FastAPI app (no DB dependency needed here)."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _patch_manifest(manifest=None):
    """Patch load_bundle_manifest to return *manifest* (or a default)."""
    if manifest is None:
        manifest = _make_manifest()
    return patch(
        "app.api.v1.extraction_routing.load_bundle_manifest",
        return_value=manifest,
    )


def _patch_embed(vector=None):
    """Patch embed_texts to return a list containing *vector*."""
    if vector is None:
        vector = [0.1] * 384
    return patch(
        "app.api.v1.extraction_routing.embed_texts",
        return_value=[vector],
    )


def _patch_search(results=None, diag=None):
    """Patch search_extraction_chunks to return (*results*, *diag*)."""
    if results is None:
        results = []
    if diag is None:
        diag = _FakeSearchDiag()
    return patch(
        "app.api.v1.extraction_routing.search_extraction_chunks",
        new=AsyncMock(return_value=(results, diag)),
    )


def _patch_rerank(reranked=None):
    """Patch reranker.rerank to return *reranked*."""
    if reranked is None:
        reranked = []
    return patch(
        "app.api.v1.extraction_routing.rrk.rerank",
        return_value=reranked,
    )


def _patch_full_doc_estimate(value: int = 1000):
    """Patch the async full-doc token estimate coroutine."""
    return patch(
        "app.api.v1.extraction_routing._async_full_doc_token_estimate",
        new=AsyncMock(return_value=value),
    )


def _patch_template_class():
    """Patch _resolve_template_class to return a trivial mock class."""
    from pydantic import BaseModel
    class _FakeCls(BaseModel):
        pass
    return patch(
        "app.api.v1.extraction_routing._resolve_template_class",
        return_value=_FakeCls,
    )


def _patch_build_query(text: str = "radar power rf query"):
    """Patch build_retrieval_profile to return a mock PassRetrievalSignals.

    C6 wiring replaced build_retrieval_query with build_retrieval_profile;
    query_text now comes from signals.entity_query. This helper patches the
    new call so all per-element-mode tests continue to control the query text.
    build_retrieval_query is no longer called in the endpoint and does not
    need to be patched (kept for reference only).
    """
    signals_mock = MagicMock()
    signals_mock.entity_query = text
    signals_mock.field_queries = ()
    return patch(
        "app.api.v1.extraction_routing.build_retrieval_profile",
        return_value=signals_mock,
    )


def _patch_graph_store():
    """Patch get_graph_store to return an AsyncMock store."""
    store = AsyncMock()
    store.vector_search = AsyncMock(return_value=[])
    return patch(
        "app.api.v1.extraction_routing.get_graph_store",
        return_value=store,
    ), store


def _patch_unknown_bundle():
    """Patch load_bundle_manifest to raise UnknownBundleError."""
    from app.services.ontology_templates import UnknownBundleError
    return patch(
        "app.api.v1.extraction_routing.load_bundle_manifest",
        side_effect=UnknownBundleError("no such bundle"),
    )


# ---------------------------------------------------------------------------
# 1. Request validation — missing required fields → 422
# ---------------------------------------------------------------------------

class TestRequestValidation:
    @pytest.mark.asyncio
    async def test_request_validation_rejects_missing_fields(self, client):
        """POST with body missing required fields → 422."""
        resp = await client.post("/v1/extraction/chunk-scope", json={})
        assert resp.status_code == 422, resp.text

    @pytest.mark.asyncio
    async def test_request_validation_rejects_unknown_keys(self, client):
        """Body with extra keys → 422 (ConfigDict extra='forbid')."""
        body = {**_VALID_BODY, "unexpected_key": "oops"}
        resp = await client.post("/v1/extraction/chunk-scope", json=body)
        assert resp.status_code == 422, resp.text

    @pytest.mark.asyncio
    async def test_request_validation_rejects_partial_body(self, client):
        """POST with only pipeline_run_id missing → 422."""
        resp = await client.post(
            "/v1/extraction/chunk-scope",
            json={"bundle_key": _BUNDLE_KEY, "pass_name": _PASS_NAME},
        )
        assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# 2. Unknown bundle → 404
# ---------------------------------------------------------------------------

class TestUnknownBundle:
    @pytest.mark.asyncio
    async def test_unknown_bundle_returns_404(self, client):
        with _patch_unknown_bundle():
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)
        assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# 3. Unknown pass_name → 404
# ---------------------------------------------------------------------------

class TestUnknownPassName:
    @pytest.mark.asyncio
    async def test_unknown_pass_name_returns_404(self, client):
        manifest = _make_manifest(passes=[_make_pass_def(name="some_other_pass")])
        with _patch_manifest(manifest):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)
        assert resp.status_code == 404, resp.text
        assert "pass_name" in resp.json()["detail"] or "radar_power_rf" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 4. Pass with no retrieval block → mode=full + fallback_reason="pass_not_routable"
# ---------------------------------------------------------------------------

class TestPassNotRoutable:
    @pytest.mark.asyncio
    async def test_pass_with_no_retrieval_block_returns_mode_full(self, client):
        """Identity/required/relationship pass routed to endpoint → defensive mode=full."""
        pass_def = _make_pass_def(phase="identity")
        pass_def.retrieval = None
        manifest = _make_manifest(passes=[pass_def])
        with _patch_manifest(manifest):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "full"
        assert data["diagnostics"]["fallback_reason"] == "pass_not_routable"
        assert data["self_refs"] == []


# ---------------------------------------------------------------------------
# 5. Empty retrieval + fallback_to_full=true → mode=full
# ---------------------------------------------------------------------------

class TestEmptyRetrievalFallback:
    @pytest.mark.asyncio
    async def test_empty_retrieval_fallback_to_full_returns_mode_full(self, client):
        rp = MagicMock()
        rp.min_similarity = 0.45
        rp.top_n_candidates = 50
        rp.top_k = 20
        rp.fallback_to_full = True
        pass_def = _make_pass_def(retrieval=rp)
        manifest = _make_manifest(passes=[pass_def])

        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_search(results=[], diag=_FakeSearchDiag()),
            _patch_full_doc_estimate(2000),
            _patch_template_class(),
            _patch_build_query(),
            _patch_graph_store()[0],
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "full"
        assert data["diagnostics"]["fallback_reason"] == "no_chunks_above_threshold"
        assert data["diagnostics"]["would_skip_if_fallback_disabled"] is True
        assert data["self_refs"] == []

    @pytest.mark.asyncio
    async def test_empty_retrieval_no_fallback_returns_mode_would_skip(self, client):
        """No chunks; pass has fallback_to_full=false → mode=would_skip."""
        rp = MagicMock()
        rp.min_similarity = 0.45
        rp.top_n_candidates = 50
        rp.top_k = 20
        rp.fallback_to_full = False
        pass_def = _make_pass_def(retrieval=rp)
        manifest = _make_manifest(passes=[pass_def])

        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_search(results=[], diag=_FakeSearchDiag()),
            _patch_full_doc_estimate(2000),
            _patch_template_class(),
            _patch_build_query(),
            _patch_graph_store()[0],
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "would_skip"
        assert data["diagnostics"]["would_skip_if_fallback_disabled"] is True
        assert data["self_refs"] == []


# ---------------------------------------------------------------------------
# 6. Reranker error → mode=full, NOT vector-only
# ---------------------------------------------------------------------------

class TestRerankerError:
    @pytest.mark.asyncio
    async def test_reranker_error_returns_mode_full_not_vector_only(self, client):
        """reranker.rerank raises → mode=full + fallback_reason='reranker_unavailable'."""
        results = [_make_result("#/texts/0", "radar ERP 45 dBW", 0.81)]
        pass_def = _make_pass_def()
        manifest = _make_manifest(passes=[pass_def])

        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_search(results=results, diag=_FakeSearchDiag(post_filter_candidate_count=1)),
            _patch_full_doc_estimate(2000),
            _patch_template_class(),
            _patch_build_query(),
            _patch_graph_store()[0],
            patch(
                "app.api.v1.extraction_routing.rrk.rerank",
                side_effect=RuntimeError("model unavailable"),
            ),
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "full", (
            f"Expected mode=full on reranker error, got mode={data['mode']!r}. "
            "Vector-only fallback is wrong — rev 10 M6."
        )
        assert data["diagnostics"]["fallback_reason"] == "reranker_unavailable"
        assert data["self_refs"] == []
        # Confirm would_skip_if_fallback_disabled is False (we had candidates)
        assert data["diagnostics"]["would_skip_if_fallback_disabled"] is False


# ---------------------------------------------------------------------------
# 7. chunk_text → content_text mapping at reranker call boundary
# ---------------------------------------------------------------------------

class TestChunkTextMapping:
    @pytest.mark.asyncio
    async def test_chunk_text_to_content_text_mapping(self, client):
        """Verify candidate dict passed to rerank has 'content_text', not 'chunk_text'."""
        results = [_make_result("#/texts/0", "radar ERP 45 dBW", 0.80)]
        pass_def = _make_pass_def()
        manifest = _make_manifest(passes=[pass_def])

        captured_candidates: list[dict] = []

        def _spy_rerank(query, candidates, top_k=10):
            captured_candidates.extend(candidates)
            # Return them unchanged with a reranker_score added
            return [
                {**c, "reranker_score": 0.9}
                for c in candidates
            ]

        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_search(results=results, diag=_FakeSearchDiag(post_filter_candidate_count=1)),
            _patch_full_doc_estimate(2000),
            _patch_template_class(),
            _patch_build_query(),
            _patch_graph_store()[0],
            patch("app.api.v1.extraction_routing.rrk.rerank", side_effect=_spy_rerank),
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        assert len(captured_candidates) == 1
        cand = captured_candidates[0]
        assert "content_text" in cand, (
            "Candidate dict must use 'content_text' key, not 'chunk_text' (rev 9 M2)"
        )
        assert "chunk_text" not in cand, (
            "Candidate dict must NOT pass 'chunk_text' key to reranker (rev 9 M2)"
        )
        assert cand["content_text"] == "radar ERP 45 dBW"


# ---------------------------------------------------------------------------
# 8. Happy path: top-K by rerank score → selected_refs
# ---------------------------------------------------------------------------

class TestSelectedRefs:
    @pytest.mark.asyncio
    async def test_selected_refs_returns_top_k_by_rerank_score(self, client):
        """Happy path: mock rerank to return reordered list; assert top-K self_refs."""
        # 5 results; reranker reverses order and top_k=3 in profile
        results = [_make_result(f"#/texts/{i}", f"chunk text {i}", score=0.7) for i in range(5)]
        pass_def = _make_pass_def()
        pass_def.retrieval.top_k = 3

        manifest = _make_manifest(passes=[pass_def])

        # Reranker returns the chunks in reverse order (last chunk scored highest)
        reranked_output = [
            {
                "content_text": f"chunk text {i}",
                "self_ref": f"#/texts/{i}",
                "vector_score": 0.7,
                "reranker_score": i * 0.1 + 0.5,
            }
            for i in range(4, -1, -1)  # 4, 3, 2, 1, 0
        ]

        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_search(results=results, diag=_FakeSearchDiag(post_filter_candidate_count=5)),
            _patch_full_doc_estimate(5000),
            _patch_template_class(),
            _patch_build_query(),
            _patch_graph_store()[0],
            patch("app.api.v1.extraction_routing.rrk.rerank", return_value=reranked_output),
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "selected_refs"
        # top_k=3 from profile; reranked_output is already top_k items from mock
        assert len(data["self_refs"]) == 3
        # First self_ref should be highest-reranked (index 4)
        assert data["self_refs"][0] == "#/texts/4"
        assert data["self_refs"][1] == "#/texts/3"
        assert data["self_refs"][2] == "#/texts/2"
        assert data["text_by_ref"] == {
            "#/texts/4": "chunk text 4",
            "#/texts/3": "chunk text 3",
            "#/texts/2": "chunk text 2",
        }


# ---------------------------------------------------------------------------
# 9. All ChunkScopeDiagnostics fields populated in success path
# ---------------------------------------------------------------------------

class TestDiagnosticsPopulated:
    @pytest.mark.asyncio
    async def test_diagnostics_populated_all_fields(self, client):
        """Verify every ChunkScopeDiagnostics field is non-None in the success path."""
        results = [_make_result("#/texts/0", "some chunk text", 0.78)]
        pass_def = _make_pass_def()
        manifest = _make_manifest(passes=[pass_def])
        reranked_output = [
            {
                "content_text": "some chunk text",
                "self_ref": "#/texts/0",
                "vector_score": 0.78,
                "reranker_score": 0.85,
            }
        ]

        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_search(results=results, diag=_FakeSearchDiag(post_filter_candidate_count=1)),
            _patch_full_doc_estimate(1000),
            _patch_template_class(),
            _patch_build_query("radar power rf query"),
            _patch_graph_store()[0],
            patch("app.api.v1.extraction_routing.rrk.rerank", return_value=reranked_output),
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        diag = resp.json()["diagnostics"]

        # Fields that must always be present and non-None in success path
        required_fields = [
            "mode", "query_text", "vector_threshold", "candidate_count",
            "selected_ref_count", "selected_token_estimate", "full_doc_token_estimate",
            "would_skip_if_fallback_disabled", "vector_search_ms", "rerank_ms",
            "ann_top_k_requested", "post_filter_candidate_count",
            "post_filter_retry_count", "filter_strategy",
        ]
        for f in required_fields:
            assert f in diag, f"Missing field: {f}"
            assert diag[f] is not None, f"Field {f!r} must be non-None in success path"

        # Score ranges may be None when no results, but must be present when there are results
        assert diag["vector_score_range"] is not None, "vector_score_range should be populated"
        assert diag["rerank_score_range"] is not None, "rerank_score_range should be populated"

        assert diag["mode"] == "selected_refs"
        assert diag["query_text"] == "radar power rf query"
        assert diag["vector_threshold"] == 0.45
        assert diag["candidate_count"] == 1
        assert diag["selected_ref_count"] == 1
        assert diag["would_skip_if_fallback_disabled"] is False
        assert diag["filter_strategy"] == "overfetch_post_filter"


# ---------------------------------------------------------------------------
# 10. Narrowing-ineffective warning at ratio > 0.80
# ---------------------------------------------------------------------------

class TestNarrowingIneffectiveWarning:
    @pytest.mark.asyncio
    async def test_selected_token_estimate_ratio_warning(self, client, caplog):
        """When selected_token_estimate / full_doc_token_estimate > 0.80, WARNING emitted."""
        # Make chunk_text very long relative to full_doc_estimate
        long_text = "x" * 4000  # ~1000 tokens
        results = [_make_result("#/texts/0", long_text, 0.78)]
        pass_def = _make_pass_def()
        manifest = _make_manifest(passes=[pass_def])
        reranked_output = [
            {
                "content_text": long_text,
                "self_ref": "#/texts/0",
                "vector_score": 0.78,
                "reranker_score": 0.85,
            }
        ]

        # full_doc = 1000 tokens; selected ≈ 1000 tokens → ratio ≈ 1.0 > 0.80
        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_search(results=results, diag=_FakeSearchDiag(post_filter_candidate_count=1)),
            _patch_full_doc_estimate(1000),
            _patch_template_class(),
            _patch_build_query(),
            _patch_graph_store()[0],
            patch("app.api.v1.extraction_routing.rrk.rerank", return_value=reranked_output),
            caplog.at_level(logging.WARNING, logger="app.api.v1.extraction_routing"),
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "narrowing INEFFECTIVE" in m or "INEFFECTIVE" in m
            for m in warning_msgs
        ), f"Expected narrowing-ineffective WARNING; got: {warning_msgs}"


# ---------------------------------------------------------------------------
# 11. embed_texts called via run_in_executor
# ---------------------------------------------------------------------------

class TestAsyncEmbedViaExecutor:
    @pytest.mark.asyncio
    async def test_async_embed_via_run_in_executor(self, client):
        """Verify the endpoint succeeds when embed_texts is called (sync function wired
        through loop.run_in_executor in the implementation).

        Minor #7 (rev 16): Patching asyncio.AbstractEventLoop.run_in_executor to assert
        wrapping is not feasible without significant asyncio internals coupling. This
        test asserts the functional outcome: the endpoint returns 200 with valid
        selected_refs data, which proves embed_texts ran successfully from within the
        async endpoint (i.e., the executor path did not crash).
        """
        results = [_make_result("#/texts/0", "chunk text", 0.78)]
        pass_def = _make_pass_def()
        manifest = _make_manifest(passes=[pass_def])
        reranked_output = [
            {
                "content_text": "chunk text",
                "self_ref": "#/texts/0",
                "vector_score": 0.78,
                "reranker_score": 0.9,
            }
        ]

        executor_calls: list[str] = []
        _original_embed = None

        def _tracking_embed(texts, query=False):
            return [[0.1] * 384]

        # Wrap run_in_executor to track that it was called with a callable
        _original_run_in_executor = asyncio.AbstractEventLoop.run_in_executor

        async def _spy_run_in_executor(self_loop, executor, func, *args):
            if getattr(func, "__qualname__", "").endswith("_embed"):
                executor_calls.append("embed")
            elif callable(func) and "embed" in getattr(func, "__qualname__", ""):
                executor_calls.append("embed")
            return _original_run_in_executor(self_loop, executor, func, *args)

        with (
            _patch_manifest(manifest),
            patch("app.api.v1.extraction_routing.embed_texts", side_effect=_tracking_embed),
            _patch_search(results=results, diag=_FakeSearchDiag(post_filter_candidate_count=1)),
            _patch_full_doc_estimate(2000),
            _patch_template_class(),
            _patch_build_query(),
            _patch_graph_store()[0],
            patch("app.api.v1.extraction_routing.rrk.rerank", return_value=reranked_output),
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        # Functional assertion: 200 + valid data proves embed_texts ran
        # within the executor path without crashing.
        data = resp.json()
        assert data["mode"] == "selected_refs"
        assert data["self_refs"] == ["#/texts/0"]


# ---------------------------------------------------------------------------
# 12. Diagnostics on empty-retrieval paths
# ---------------------------------------------------------------------------

class TestEmptyRetrievalDiagnostics:
    @pytest.mark.asyncio
    async def test_empty_retrieval_diagnostics_have_correct_fields(self, client):
        """Empty retrieval path should still populate all diagnostic fields."""
        rp = MagicMock()
        rp.min_similarity = 0.55
        rp.top_n_candidates = 30
        rp.top_k = 10
        rp.fallback_to_full = True
        pass_def = _make_pass_def(retrieval=rp)
        manifest = _make_manifest(passes=[pass_def])
        diag_data = _FakeSearchDiag(
            ann_top_k_requested=300,
            post_filter_candidate_count=0,
            post_filter_retry_count=0,
        )

        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_search(results=[], diag=diag_data),
            _patch_full_doc_estimate(3000),
            _patch_template_class(),
            _patch_build_query("some query"),
            _patch_graph_store()[0],
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "full"
        diag = data["diagnostics"]
        assert diag["candidate_count"] == 0
        assert diag["selected_ref_count"] == 0
        assert diag["ann_top_k_requested"] == 300
        assert diag["post_filter_candidate_count"] == 0
        assert diag["post_filter_retry_count"] == 0
        assert diag["filter_strategy"] == "overfetch_post_filter"
        assert diag["full_doc_token_estimate"] == 3000
        assert diag["vector_threshold"] == 0.55
        assert diag["would_skip_if_fallback_disabled"] is True


# ---------------------------------------------------------------------------
# 13. Important #1 — template resolution failure → structured mode=full (rev 16)
# ---------------------------------------------------------------------------

class TestTemplateResolutionError:
    @pytest.mark.asyncio
    async def test_template_resolution_error_returns_mode_full(self, client):
        """_resolve_template_class raises → structured ChunkScopeResponse mode=full,
        fallback_reason='template_resolution_error' (NOT a bare 500).

        This test is RED until Important #1 fix is applied.
        """
        pass_def = _make_pass_def()
        manifest = _make_manifest(passes=[pass_def])

        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_full_doc_estimate(0),
            _patch_graph_store()[0],
            patch(
                "app.api.v1.extraction_routing._resolve_template_class",
                side_effect=ImportError("no module named 'ontology_bundles.bad_module'"),
            ),
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, (
            f"Expected 200 structured response on template resolution error, "
            f"got {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert data["mode"] == "full", (
            f"Expected mode=full on template resolution error, got mode={data['mode']!r}"
        )
        assert data["diagnostics"]["fallback_reason"] == "template_resolution_error", (
            f"Expected fallback_reason='template_resolution_error', "
            f"got {data['diagnostics']['fallback_reason']!r}"
        )
        assert data["self_refs"] == []


# ---------------------------------------------------------------------------
# 14. Important #2 — short_fetch propagates from search_diag to response (rev 16)
# ---------------------------------------------------------------------------

class TestShortFetchPropagation:
    @pytest.mark.asyncio
    async def test_short_fetch_true_propagates_to_response_diagnostics(self, client, caplog):
        """When search_diag.short_fetch=True, response.diagnostics.short_fetch must be True
        and a WARNING must be logged.

        This test is RED until Important #2 fix is applied.
        """
        results = [_make_result(f"#/texts/{i}", f"chunk text {i}", score=0.70) for i in range(3)]
        pass_def = _make_pass_def()
        manifest = _make_manifest(passes=[pass_def])

        # short_fetch=True simulates: post-filter survivors < desired_top_n after retry
        short_fetch_diag = _FakeSearchDiag(
            ann_top_k_requested=2000,
            post_filter_candidate_count=3,
            post_filter_retry_count=1,
            short_fetch=True,
        )

        reranked_output = [
            {
                "content_text": f"chunk text {i}",
                "self_ref": f"#/texts/{i}",
                "vector_score": 0.70,
                "reranker_score": 0.80 - i * 0.01,
            }
            for i in range(3)
        ]

        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_search(results=results, diag=short_fetch_diag),
            _patch_full_doc_estimate(5000),
            _patch_template_class(),
            _patch_build_query(),
            _patch_graph_store()[0],
            patch("app.api.v1.extraction_routing.rrk.rerank", return_value=reranked_output),
            caplog.at_level(logging.WARNING, logger="app.api.v1.extraction_routing"),
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "selected_refs", (
            f"short_fetch=True should NOT change mode; expected selected_refs, got {data['mode']!r}"
        )
        assert data["diagnostics"]["short_fetch"] is True, (
            "diagnostics.short_fetch must be True when search_diag.short_fetch=True"
        )
        # Check WARNING was logged
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("short-fetch" in m.lower() or "short_fetch" in m.lower() for m in warning_msgs), (
            f"Expected short-fetch WARNING in logs; got: {warning_msgs}"
        )

    @pytest.mark.asyncio
    async def test_short_fetch_false_default_in_response(self, client):
        """When search_diag.short_fetch=False (default), response.diagnostics.short_fetch=False."""
        results = [_make_result("#/texts/0", "chunk text", 0.78)]
        pass_def = _make_pass_def()
        manifest = _make_manifest(passes=[pass_def])
        reranked_output = [
            {
                "content_text": "chunk text",
                "self_ref": "#/texts/0",
                "vector_score": 0.78,
                "reranker_score": 0.90,
            }
        ]

        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_search(results=results, diag=_FakeSearchDiag(short_fetch=False)),
            _patch_full_doc_estimate(1000),
            _patch_template_class(),
            _patch_build_query(),
            _patch_graph_store()[0],
            patch("app.api.v1.extraction_routing.rrk.rerank", return_value=reranked_output),
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        assert resp.json()["diagnostics"]["short_fetch"] is False


# ---------------------------------------------------------------------------
# 15. HIGH (rev 17) — _resolve_template_class is bundle-aware (unpatched test)
# ---------------------------------------------------------------------------

class TestResolveBundleAware:
    def test_resolve_template_class_uses_real_manifest(self):
        """Resolver must work against a real manifest pass — not patched.

        Regression guard for the rev 16 bug where _resolve_template_class imported
        pass_def.module literally without the 'ontology_bundles.{bundle_key}.' prefix.
        The manifest stores RELATIVE module paths (e.g. 'extraction_schemas.radar_power_rf');
        the worker (pipeline.py:_parse_pass_response ~line 3603) prefixes them with
        'ontology_bundles.{bundle_key}.' before importing.  The endpoint must do the same.
        """
        from app.services.ontology_bundles import load_bundle_manifest
        from app.api.v1.extraction_routing import _resolve_template_class

        bundle_key = "air_defense_v3_baseline_subset"
        manifest = load_bundle_manifest(bundle_key)
        pass_def = next(p for p in manifest.passes if p.name == "radar_power_rf")

        # Confirm the manifest stores a RELATIVE path (no bundle prefix)
        assert not pass_def.module.startswith("ontology_bundles."), (
            f"Manifest module path should be relative, got: {pass_def.module!r}"
        )

        template_cls = _resolve_template_class(bundle_key, pass_def)

        # Must resolve to the real pydantic class — not a mock or fallback
        assert template_cls.__name__ == "RadarPowerRfPass", (
            f"Expected class name 'RadarPowerRfPass', got {template_cls.__name__!r}"
        )
        assert template_cls.__module__.startswith("ontology_bundles.air_defense_v3"), (
            f"Expected module starting with 'ontology_bundles.air_defense_v3', "
            f"got {template_cls.__module__!r}"
        )


# ---------------------------------------------------------------------------
# 16. MED (rev 17) — empty selected_refs after rerank → mode=full
# ---------------------------------------------------------------------------

class TestEmptySelectedRefsAfterRerank:
    @pytest.mark.asyncio
    async def test_empty_selected_refs_after_rerank_returns_mode_full(self, client):
        """When rerank returns candidates that all lack self_ref keys, endpoint must
        return mode=full + fallback_reason='no_selected_refs_after_rerank' instead of
        the contract-invalid mode=selected_refs with self_refs=[].

        Rev 17 MED guard.
        """
        # Candidates have content_text but NO self_ref key
        results = [_make_result("#/texts/0", "radar ERP 45 dBW", 0.81)]
        pass_def = _make_pass_def()
        manifest = _make_manifest(passes=[pass_def])

        reranked_no_self_ref = [
            {
                "content_text": "radar ERP 45 dBW",
                # deliberately omit self_ref to trigger the guard
                "vector_score": 0.81,
                "reranker_score": 0.88,
            }
        ]

        with (
            _patch_manifest(manifest),
            _patch_embed(),
            _patch_search(results=results, diag=_FakeSearchDiag(post_filter_candidate_count=1)),
            _patch_full_doc_estimate(2000),
            _patch_template_class(),
            _patch_build_query(),
            _patch_graph_store()[0],
            patch("app.api.v1.extraction_routing.rrk.rerank", return_value=reranked_no_self_ref),
        ):
            resp = await client.post("/v1/extraction/chunk-scope", json=_VALID_BODY)

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "full", (
            f"Expected mode=full when no valid self_refs after rerank, got {data['mode']!r}"
        )
        assert data["diagnostics"]["fallback_reason"] == "no_selected_refs_after_rerank", (
            f"Expected fallback_reason='no_selected_refs_after_rerank', "
            f"got {data['diagnostics']['fallback_reason']!r}"
        )
        assert data["self_refs"] == []


# ---------------------------------------------------------------------------
# 17. LOW (rev 17) — _async_full_doc_token_estimate uses non-vector SQL
# ---------------------------------------------------------------------------

class TestNonVectorTokenEstimate:
    @pytest.mark.asyncio
    async def test_full_doc_token_estimate_uses_sql_not_hnsw(self):
        """_async_full_doc_token_estimate must use direct SQL (not HNSW vector_search).

        Rev 17 LOW: switched from zero-vector probe (filter-starvation risk, same
        shape as C.1 bug) to SELECT sum(chunk_text.size()) by pipeline_run_id.

        This test mocks store._client.query and asserts it's called with an SQL
        SELECT statement — NOT a vector_search call.  Also asserts the token
        estimate is computed from the SQL result.
        """
        from unittest.mock import AsyncMock, MagicMock
        from app.api.v1.extraction_routing import _async_full_doc_token_estimate

        pipeline_run_id = "test-run-123"

        # Simulate ArcadeDB returning total_chars = 4002 (→ 1334 tokens via //3)
        mock_client = MagicMock()
        mock_client.query = AsyncMock(return_value=[{"total_chars": 4002}])

        mock_store = MagicMock()
        mock_store._client = mock_client
        mock_store._database = "testdb"
        # vector_search should NOT be called; attach an AsyncMock so a call would be detectable
        mock_store.vector_search = AsyncMock(side_effect=AssertionError(
            "vector_search called — full_doc_token_estimate must use SQL, not HNSW"
        ))

        result = await _async_full_doc_token_estimate(pipeline_run_id, mock_store)

        # Verify SQL query was used
        mock_client.query.assert_called_once()
        call_args = mock_client.query.call_args
        sql_cmd = call_args.args[2] if len(call_args.args) >= 3 else call_args.kwargs.get("command", "")
        assert "SUM" in sql_cmd.upper() or "sum" in sql_cmd, (
            f"Expected SUM aggregate in SQL; got: {sql_cmd!r}"
        )
        assert "ExtractionChunk" in sql_cmd, (
            f"Expected ExtractionChunk in SQL; got: {sql_cmd!r}"
        )
        assert pipeline_run_id in str(call_args) or "run_id" in str(call_args), (
            f"Expected run_id parameter in SQL call; call_args={call_args}"
        )

        # Token estimate: 4002 chars // 3 = 1334 tokens (technical-text ratio)
        assert result == 1334, f"Expected token estimate 1334 (4002 chars // 3), got {result}"

    @pytest.mark.asyncio
    async def test_full_doc_token_estimate_returns_zero_on_empty_result(self):
        """When SQL returns no rows (empty run), estimator returns 0."""
        from unittest.mock import AsyncMock, MagicMock
        from app.api.v1.extraction_routing import _async_full_doc_token_estimate

        mock_client = MagicMock()
        mock_client.query = AsyncMock(return_value=[])

        mock_store = MagicMock()
        mock_store._client = mock_client
        mock_store._database = "testdb"
        mock_store.vector_search = AsyncMock(side_effect=AssertionError("HNSW called"))

        result = await _async_full_doc_token_estimate("some-run", mock_store)
        assert result == 0

    @pytest.mark.asyncio
    async def test_full_doc_token_estimate_returns_zero_on_sql_error(self):
        """When SQL query raises, estimator returns 0 (fail-safe, not fail-hard)."""
        from unittest.mock import AsyncMock, MagicMock
        from app.api.v1.extraction_routing import _async_full_doc_token_estimate

        mock_client = MagicMock()
        mock_client.query = AsyncMock(side_effect=ConnectionError("ArcadeDB unreachable"))

        mock_store = MagicMock()
        mock_store._client = mock_client
        mock_store._database = "testdb"
        mock_store.vector_search = AsyncMock(side_effect=AssertionError("HNSW called"))

        result = await _async_full_doc_token_estimate("some-run", mock_store)
        assert result == 0

    @pytest.mark.asyncio
    async def test_full_doc_token_estimate_uses_arcadedb_native_string_size(self):
        """SQL must use ArcadeDB-native `chunk_text.size()`, not SQL `LENGTH()`.

        Regression for the bug where the shipped query used LENGTH(chunk_text)
        which ArcadeDB rejects as "Unknown function name 'LENGTH'", the except
        clause swallowed the error, and the function silently returned 0 — which
        in turn disabled the narrowing-ineffective WARNING for every pass.
        """
        from unittest.mock import AsyncMock, MagicMock
        from app.api.v1.extraction_routing import _async_full_doc_token_estimate

        mock_client = MagicMock()
        mock_client.query = AsyncMock(return_value=[{"total_chars": 4000}])
        mock_store = MagicMock()
        mock_store._client = mock_client
        mock_store._database = "testdb"

        await _async_full_doc_token_estimate("any-run", mock_store)

        sql_cmd = mock_client.query.call_args.args[2]
        assert "LENGTH(" not in sql_cmd.upper(), (
            f"SQL must not use LENGTH() — ArcadeDB does not implement it; got: {sql_cmd!r}"
        )
        assert ".size()" in sql_cmd, (
            f"SQL must use ArcadeDB-native chunk_text.size(); got: {sql_cmd!r}"
        )
