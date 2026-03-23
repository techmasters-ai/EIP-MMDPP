"""Comprehensive unit tests for all query/retrieval functionality.

Covers:
- _text_vector_search (basic text search via Qdrant)
- _image_vector_search (image search via Qdrant)
- _multi_modal_pipeline (hybrid text+image search)
- _graphrag_local_query, _graphrag_global_query, _graphrag_drift_query, _graphrag_basic_query
- _merge_seed_results (result merging)
- _build_qdrant_filters (filter builder)
- _apply_reranker (cross-encoder reranking)
- graphrag_service.local_search, global_search, drift_search, basic_search
- compute_fusion_score (scoring)
- deduplicate_results / diversify_results (dedup)
- _populate_image_urls (image URL population)
"""

import sys
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("asyncpg", reason="asyncpg not installed")


# ---------------------------------------------------------------------------
# Stub heavy external deps that aren't in the test venv (pandas, graphrag, etc.)
# This must happen before any app module that depends on them is imported.
# ---------------------------------------------------------------------------

class _AutoStubModule(types.ModuleType):
    """Module stub that auto-creates MagicMock attributes on access."""

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        mock = MagicMock()
        setattr(self, name, mock)
        return mock


# pandas stub (graphrag_service, graphrag_bridge import it at top level)
if "pandas" not in sys.modules:
    _pd_stub = _AutoStubModule("pandas")
    _pd_stub.DataFrame = MagicMock  # type: ignore[attr-defined]
    _pd_stub.read_parquet = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    sys.modules["pandas"] = _pd_stub

# graphrag ecosystem stubs
for _mod_name in (
    "graphrag", "graphrag.api", "graphrag.api.prompt_tune",
    "graphrag.config", "graphrag.config.enums", "graphrag.config.models",
    "graphrag.config.models.cluster_graph_config",
    "graphrag.config.models.drift_search_config",
    "graphrag.config.models.graph_rag_config",
    "graphrag.config.models.local_search_config",
    "graphrag.config.models.reporting_config",
    "graphrag.config.models.llm_config",
    "graphrag.config.models.llm_parameters_config",
    "graphrag.config.models.parallelization_parameters_config",
    "graphrag.config.models.embeddings_config",
    "graphrag.config.models.text_embedding_config",
    "graphrag_cache", "graphrag_cache.cache_config",
    "graphrag_llm", "graphrag_llm.config", "graphrag_llm.config.model_config",
    "graphrag_llm.embedding", "graphrag_llm.embedding.lite_llm_embedding",
    "graphrag_storage", "graphrag_storage.storage_config",
    "graphrag_vectors", "graphrag_vectors.vector_store_config",
    "lancedb",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _AutoStubModule(_mod_name)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_body(**kwargs):
    """Build a UnifiedQueryRequest with sensible defaults."""
    from app.schemas.retrieval import UnifiedQueryRequest

    defaults = {"query_text": "radar system specifications"}
    defaults.update(kwargs)
    return UnifiedQueryRequest(**defaults)


def _make_item(**kwargs):
    """Build a QueryResultItem with sensible defaults."""
    from app.schemas.retrieval import QueryResultItem

    defaults = dict(
        chunk_id=uuid.uuid4(),
        artifact_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        score=0.85,
        modality="text",
        content_text="Some chunk text about radar.",
        classification="UNCLASSIFIED",
    )
    defaults.update(kwargs)
    return QueryResultItem(**defaults)


def _mock_settings(**overrides):
    """Build a mock Settings object with retrieval-relevant defaults."""
    s = MagicMock()
    s.reranker_enabled = False
    s.reranker_top_n = 20
    s.retrieval_min_score_threshold = 0.25
    s.retrieval_diversity_oversample_factor = 8
    s.retrieval_diversity_max_candidates = 800
    s.retrieval_semantic_weight = 0.65
    s.retrieval_doc_structure_weight = 0.20
    s.retrieval_ontology_weight = 0.15
    s.retrieval_hop_penalty_base = 0.92
    s.retrieval_mil_id_bonus = 0.03
    s.retrieval_cross_modal_decay = 0.85
    s.retrieval_ontology_decay = 0.75
    s.retrieval_doc_expand_k = 5
    s.retrieval_doc_max_hops = 2
    s.retrieval_ontology_expand_k = 5
    s.retrieval_weight_next_chunk = 0.90
    s.retrieval_weight_same_section = 0.88
    s.retrieval_weight_same_artifact = 0.82
    s.retrieval_weight_same_page = 0.78
    # GraphRAG settings
    s.graphrag_indexing_enabled = True
    s.graphrag_llm_provider = "ollama"
    s.graphrag_llm_model = "llama3.2"
    s.graphrag_llm_api_base = "http://ollama:11434/v1"
    s.graphrag_api_key = ""
    s.graphrag_embedding_model = "nomic-embed-text"
    s.graphrag_data_dir = "/tmp/test_graphrag"
    s.graphrag_community_level = 2
    s.graphrag_max_cluster_size = 10
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_qdrant_hit(chunk_id=None, score=0.8, modality="text", **extra_payload):
    """Build a mock Qdrant search hit dict."""
    cid = chunk_id or str(uuid.uuid4())
    payload = {
        "chunk_id": cid,
        "artifact_id": str(uuid.uuid4()),
        "document_id": str(uuid.uuid4()),
        "modality": modality,
        "chunk_text": "Radar system test content.",
        "page_number": 1,
        "classification": "UNCLASSIFIED",
    }
    payload.update(extra_payload)
    return {"score": score, "payload": payload}


# ---------------------------------------------------------------------------
# 1. _text_vector_search
# ---------------------------------------------------------------------------

class TestTextVectorSearch:
    """Tests for _text_vector_search (basic text search via Qdrant)."""

    @pytest.mark.asyncio
    async def test_returns_results_from_qdrant(self):
        from app.api.v1.retrieval import _text_vector_search

        hits = [_make_qdrant_hit(score=0.9), _make_qdrant_hit(score=0.7)]

        with patch("app.api.v1.retrieval.get_qdrant_async_client"), \
             patch("app.api.v1.retrieval._build_qdrant_filters", return_value=None), \
             patch("app.services.embedding.embed_texts", return_value=[[0.1] * 384]), \
             patch("app.services.qdrant_store.search_text_vectors_async", new_callable=AsyncMock, return_value=hits), \
             patch("app.config.get_settings", return_value=_mock_settings()):
            db = AsyncMock()
            body = _make_body(top_k=5, include_context=True)
            results = await _text_vector_search(db, body)

        assert len(results) == 2
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_no_query_text_returns_empty(self):
        from app.api.v1.retrieval import _text_vector_search
        from app.schemas.retrieval import UnifiedQueryRequest

        db = AsyncMock()
        body = UnifiedQueryRequest.model_construct(
            query_text=None, query_image="base64data", strategy="basic",
            modality_filter="all", top_k=5, include_context=True,
        )
        results = await _text_vector_search(db, body)
        assert results == []

    @pytest.mark.asyncio
    async def test_filters_below_min_score(self):
        from app.api.v1.retrieval import _text_vector_search

        hits = [_make_qdrant_hit(score=0.1), _make_qdrant_hit(score=0.8)]

        with patch("app.api.v1.retrieval.get_qdrant_async_client"), \
             patch("app.api.v1.retrieval._build_qdrant_filters", return_value=None), \
             patch("app.services.embedding.embed_texts", return_value=[[0.1] * 384]), \
             patch("app.services.qdrant_store.search_text_vectors_async", new_callable=AsyncMock, return_value=hits), \
             patch("app.config.get_settings", return_value=_mock_settings()):
            db = AsyncMock()
            body = _make_body(top_k=10, include_context=True)
            results = await _text_vector_search(db, body)

        # The hit with score 0.1 should be filtered out (min threshold = 0.25)
        assert all(r.score >= 0.25 for r in results)

    @pytest.mark.asyncio
    async def test_strips_content_text_when_not_requested(self):
        from app.api.v1.retrieval import _text_vector_search

        hits = [_make_qdrant_hit(score=0.9)]

        with patch("app.api.v1.retrieval.get_qdrant_async_client"), \
             patch("app.api.v1.retrieval._build_qdrant_filters", return_value=None), \
             patch("app.services.embedding.embed_texts", return_value=[[0.1] * 384]), \
             patch("app.services.qdrant_store.search_text_vectors_async", new_callable=AsyncMock, return_value=hits), \
             patch("app.config.get_settings", return_value=_mock_settings()):
            db = AsyncMock()
            body = _make_body(top_k=5, include_context=False)
            results = await _text_vector_search(db, body)

        assert len(results) == 1
        assert results[0].content_text is None

    @pytest.mark.asyncio
    async def test_respects_top_k(self):
        from app.api.v1.retrieval import _text_vector_search

        hits = [_make_qdrant_hit(score=0.9 - i * 0.02) for i in range(20)]

        with patch("app.api.v1.retrieval.get_qdrant_async_client"), \
             patch("app.api.v1.retrieval._build_qdrant_filters", return_value=None), \
             patch("app.services.embedding.embed_texts", return_value=[[0.1] * 384]), \
             patch("app.services.qdrant_store.search_text_vectors_async", new_callable=AsyncMock, return_value=hits), \
             patch("app.config.get_settings", return_value=_mock_settings()):
            db = AsyncMock()
            body = _make_body(top_k=3, include_context=True)
            results = await _text_vector_search(db, body)

        assert len(results) <= 3


# ---------------------------------------------------------------------------
# 2. _image_vector_search
# ---------------------------------------------------------------------------

class TestImageVectorSearch:
    """Tests for _image_vector_search (CLIP image search via Qdrant)."""

    @pytest.mark.asyncio
    async def test_text_to_clip_search(self):
        from app.api.v1.retrieval import _image_vector_search

        hits = [_make_qdrant_hit(score=0.75, modality="image")]

        with patch("app.api.v1.retrieval.get_qdrant_async_client"), \
             patch("app.api.v1.retrieval._build_qdrant_filters", return_value=None), \
             patch("app.services.embedding.embed_text_for_clip", return_value=[0.1] * 512), \
             patch("app.services.qdrant_store.search_image_vectors_async", new_callable=AsyncMock, return_value=hits), \
             patch("app.config.get_settings", return_value=_mock_settings()):
            db = AsyncMock()
            body = _make_body(top_k=5, include_context=True)
            results = await _image_vector_search(db, body)

        assert len(results) == 1
        assert results[0].modality == "image"

    @pytest.mark.asyncio
    async def test_no_query_returns_empty(self):
        from app.api.v1.retrieval import _image_vector_search
        from app.schemas.retrieval import UnifiedQueryRequest

        db = AsyncMock()
        body = UnifiedQueryRequest.model_construct(
            query_text=None, query_image=None, strategy="basic",
            modality_filter="all", top_k=5, include_context=True,
        )
        results = await _image_vector_search(db, body)
        assert results == []

    @pytest.mark.asyncio
    async def test_filters_below_min_score_image(self):
        from app.api.v1.retrieval import _image_vector_search

        hits = [
            _make_qdrant_hit(score=0.1, modality="image"),
            _make_qdrant_hit(score=0.6, modality="image"),
        ]

        with patch("app.api.v1.retrieval.get_qdrant_async_client"), \
             patch("app.api.v1.retrieval._build_qdrant_filters", return_value=None), \
             patch("app.services.embedding.embed_text_for_clip", return_value=[0.1] * 512), \
             patch("app.services.qdrant_store.search_image_vectors_async", new_callable=AsyncMock, return_value=hits), \
             patch("app.config.get_settings", return_value=_mock_settings()):
            db = AsyncMock()
            body = _make_body(top_k=10, include_context=True)
            results = await _image_vector_search(db, body)

        assert all(r.score >= 0.25 for r in results)


# ---------------------------------------------------------------------------
# 3. _multi_modal_pipeline
# ---------------------------------------------------------------------------

class TestMultiModalPipeline:
    """Tests for _multi_modal_pipeline (hybrid mode)."""

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._apply_reranker", side_effect=lambda results, body: results)
    @patch("app.api.v1.retrieval._rescore_expanded_chunks", new_callable=AsyncMock, side_effect=lambda e, q: e)
    @patch("app.api.v1.retrieval._expand_seeds", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.retrieval._image_vector_search", new_callable=AsyncMock)
    @patch("app.api.v1.retrieval._text_vector_search", new_callable=AsyncMock)
    async def test_merges_text_and_image(self, mock_text, mock_image, mock_expand, mock_rescore, mock_reranker):
        from app.api.v1.retrieval import _multi_modal_pipeline

        text_item = _make_item(score=0.9, modality="text")
        image_item = _make_item(score=0.8, modality="image")
        mock_text.return_value = [text_item]
        mock_image.return_value = [image_item]

        db = AsyncMock()
        body = _make_body(strategy="hybrid", mode="multi_modal", top_k=10)
        results = await _multi_modal_pipeline(db, body)

        assert len(results) == 2
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._apply_reranker", side_effect=lambda results, body: results)
    @patch("app.api.v1.retrieval._rescore_expanded_chunks", new_callable=AsyncMock, side_effect=lambda e, q: e)
    @patch("app.api.v1.retrieval._expand_seeds", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.retrieval._image_vector_search", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.retrieval._text_vector_search", new_callable=AsyncMock)
    async def test_text_only_filter(self, mock_text, mock_image, mock_expand, mock_rescore, mock_reranker):
        from app.api.v1.retrieval import _multi_modal_pipeline
        from app.schemas.retrieval import ModalityFilter

        text_item = _make_item(score=0.9, modality="text")
        image_item = _make_item(score=0.8, modality="image")
        mock_text.return_value = [text_item, image_item]

        db = AsyncMock()
        body = _make_body(strategy="hybrid", modality_filter=ModalityFilter.text, top_k=10)
        results = await _multi_modal_pipeline(db, body)

        assert all(r.modality in ("text", "table") for r in results)

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._apply_reranker", side_effect=lambda results, body: results)
    @patch("app.api.v1.retrieval._rescore_expanded_chunks", new_callable=AsyncMock, side_effect=lambda e, q: e)
    @patch("app.api.v1.retrieval._expand_seeds", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.retrieval._image_vector_search", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.retrieval._text_vector_search", new_callable=AsyncMock)
    async def test_image_only_filter(self, mock_text, mock_image, mock_expand, mock_rescore, mock_reranker):
        from app.api.v1.retrieval import _multi_modal_pipeline
        from app.schemas.retrieval import ModalityFilter

        text_item = _make_item(score=0.9, modality="text")
        image_item = _make_item(score=0.8, modality="image")
        mock_text.return_value = [text_item, image_item]

        db = AsyncMock()
        body = _make_body(strategy="hybrid", modality_filter=ModalityFilter.image, top_k=10)
        results = await _multi_modal_pipeline(db, body)

        assert all(r.modality in ("image", "schematic") for r in results)

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._apply_reranker", side_effect=lambda results, body: results)
    @patch("app.api.v1.retrieval._rescore_expanded_chunks", new_callable=AsyncMock, side_effect=lambda e, q: e)
    @patch("app.api.v1.retrieval._expand_seeds", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.retrieval._image_vector_search", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.retrieval._text_vector_search", new_callable=AsyncMock, return_value=[])
    async def test_empty_search_returns_empty(self, mock_text, mock_image, mock_expand, mock_rescore, mock_reranker):
        from app.api.v1.retrieval import _multi_modal_pipeline

        db = AsyncMock()
        body = _make_body(strategy="hybrid", mode="multi_modal", top_k=10)
        results = await _multi_modal_pipeline(db, body)

        assert results == []

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._apply_reranker", side_effect=lambda results, body: results)
    @patch("app.api.v1.retrieval._rescore_expanded_chunks", new_callable=AsyncMock, side_effect=lambda e, q: e)
    @patch("app.api.v1.retrieval._expand_seeds", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.retrieval._image_vector_search", new_callable=AsyncMock, return_value=[])
    @patch("app.api.v1.retrieval._text_vector_search", new_callable=AsyncMock)
    async def test_caps_at_top_k(self, mock_text, mock_image, mock_expand, mock_rescore, mock_reranker):
        from app.api.v1.retrieval import _multi_modal_pipeline

        items = [_make_item(score=0.9 - i * 0.01) for i in range(20)]
        mock_text.return_value = items

        db = AsyncMock()
        body = _make_body(strategy="hybrid", mode="multi_modal", top_k=5)
        results = await _multi_modal_pipeline(db, body)

        assert len(results) <= 5


# ---------------------------------------------------------------------------
# 4. GraphRAG query functions (_graphrag_*_query in retrieval.py)
# ---------------------------------------------------------------------------

class TestGraphRAGLocalQuery:
    """Tests for _graphrag_local_query."""

    @pytest.mark.asyncio
    async def test_no_query_text_returns_empty(self):
        from app.api.v1.retrieval import _graphrag_local_query
        from app.schemas.retrieval import UnifiedQueryRequest

        body = UnifiedQueryRequest.model_construct(
            query_text=None, query_image="base64data", strategy="graphrag_local",
            modality_filter="all", top_k=10, include_context=True,
        )
        db = AsyncMock()
        results = await _graphrag_local_query(db, body)
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_graphrag_result(self):
        from app.api.v1.retrieval import _graphrag_local_query

        mock_local = MagicMock(return_value={
            "response": "The S-400 system capabilities include...",
            "context": {"entities": ["S-400"]},
        })

        with patch("app.services.graphrag_service.local_search", mock_local):
            db = AsyncMock()
            body = _make_body(strategy="graphrag_local")
            results = await _graphrag_local_query(db, body)

        assert len(results) == 1
        assert results[0].modality == "graphrag_response"
        assert "S-400" in results[0].content_text
        assert results[0].context["source"] == "graphrag_local"
        assert results[0].score == 1.0

    @pytest.mark.asyncio
    async def test_empty_response_raises_404(self):
        from app.api.v1.retrieval import _graphrag_local_query
        from fastapi import HTTPException

        mock_local = MagicMock(return_value={"response": "", "context": {}})

        with patch("app.services.graphrag_service.local_search", mock_local):
            db = AsyncMock()
            body = _make_body(strategy="graphrag_local")
            with pytest.raises(HTTPException) as exc_info:
                await _graphrag_local_query(db, body)
            assert exc_info.value.status_code == 404


class TestGraphRAGGlobalQuery:
    """Tests for _graphrag_global_query."""

    @pytest.mark.asyncio
    async def test_no_query_text_returns_empty(self):
        from app.api.v1.retrieval import _graphrag_global_query
        from app.schemas.retrieval import UnifiedQueryRequest

        body = UnifiedQueryRequest.model_construct(
            query_text=None, query_image="base64data", strategy="graphrag_global",
            modality_filter="all", top_k=10, include_context=True,
        )
        db = AsyncMock()
        results = await _graphrag_global_query(db, body)
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_graphrag_result(self):
        from app.api.v1.retrieval import _graphrag_global_query

        mock_global = MagicMock(return_value={
            "response": "Across all communities, threat assessment...",
            "context": {"reports": ["r1"]},
        })

        with patch("app.services.graphrag_service.global_search", mock_global):
            db = AsyncMock()
            body = _make_body(strategy="graphrag_global")
            results = await _graphrag_global_query(db, body)

        assert len(results) == 1
        assert results[0].context["source"] == "graphrag_global"

    @pytest.mark.asyncio
    async def test_empty_response_raises_409(self):
        from app.api.v1.retrieval import _graphrag_global_query
        from fastapi import HTTPException

        mock_global = MagicMock(return_value={"response": "", "context": {}})

        with patch("app.services.graphrag_service.global_search", mock_global):
            db = AsyncMock()
            body = _make_body(strategy="graphrag_global")
            with pytest.raises(HTTPException) as exc_info:
                await _graphrag_global_query(db, body)
            assert exc_info.value.status_code == 409


class TestGraphRAGDriftQuery:
    """Tests for _graphrag_drift_query."""

    @pytest.mark.asyncio
    async def test_no_query_text_returns_empty(self):
        from app.api.v1.retrieval import _graphrag_drift_query
        from app.schemas.retrieval import UnifiedQueryRequest

        body = UnifiedQueryRequest.model_construct(
            query_text=None, query_image="base64data", strategy="graphrag_drift",
            modality_filter="all", top_k=10, include_context=True,
        )
        db = AsyncMock()
        results = await _graphrag_drift_query(db, body)
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_drift_result(self):
        from app.api.v1.retrieval import _graphrag_drift_query

        mock_drift = MagicMock(return_value={
            "response": "DRIFT analysis of guidance methods...",
            "context": {"entities": ["guidance"]},
        })

        with patch("app.services.graphrag_service.drift_search", mock_drift):
            db = AsyncMock()
            body = _make_body(strategy="graphrag_drift")
            results = await _graphrag_drift_query(db, body)

        assert len(results) == 1
        assert results[0].context["source"] == "graphrag_drift"

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_list(self):
        from app.api.v1.retrieval import _graphrag_drift_query

        mock_drift = MagicMock(return_value={"response": "", "context": {}})

        with patch("app.services.graphrag_service.drift_search", mock_drift):
            db = AsyncMock()
            body = _make_body(strategy="graphrag_drift")
            results = await _graphrag_drift_query(db, body)
            assert results == []


class TestGraphRAGBasicQuery:
    """Tests for _graphrag_basic_query."""

    @pytest.mark.asyncio
    async def test_no_query_text_returns_empty(self):
        from app.api.v1.retrieval import _graphrag_basic_query
        from app.schemas.retrieval import UnifiedQueryRequest

        body = UnifiedQueryRequest.model_construct(
            query_text=None, query_image="base64data", strategy="graphrag_basic",
            modality_filter="all", top_k=10, include_context=True,
        )
        db = AsyncMock()
        results = await _graphrag_basic_query(db, body)
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_basic_result(self):
        from app.api.v1.retrieval import _graphrag_basic_query

        mock_basic = MagicMock(return_value={
            "response": "Based on text units: missile range data...",
            "context": {"chunks": ["c1"]},
        })

        with patch("app.services.graphrag_service.basic_search", mock_basic):
            db = AsyncMock()
            body = _make_body(strategy="graphrag_basic")
            results = await _graphrag_basic_query(db, body)

        assert len(results) == 1
        assert results[0].context["source"] == "graphrag_basic"
        assert results[0].score == 1.0

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_list(self):
        from app.api.v1.retrieval import _graphrag_basic_query

        mock_basic = MagicMock(return_value={"response": "", "context": {}})

        with patch("app.services.graphrag_service.basic_search", mock_basic):
            db = AsyncMock()
            body = _make_body(strategy="graphrag_basic")
            results = await _graphrag_basic_query(db, body)
            assert results == []


# ---------------------------------------------------------------------------
# 5. _merge_seed_results
# ---------------------------------------------------------------------------

class TestMergeSeedResultsCoverage:
    """Additional merge tests beyond test_retrieval_pipeline.py."""

    def test_multiple_lists_merged(self):
        from app.api.v1.retrieval import _merge_seed_results

        a = _make_item(chunk_id=uuid.uuid4(), score=0.8)
        b = _make_item(chunk_id=uuid.uuid4(), score=0.7)
        result = _merge_seed_results([[a], [b]])
        assert len(result) == 2

    def test_all_exceptions_returns_empty(self):
        from app.api.v1.retrieval import _merge_seed_results

        result = _merge_seed_results([Exception("fail1"), RuntimeError("fail2")])
        assert result == []

    def test_mixed_exceptions_and_results(self):
        from app.api.v1.retrieval import _merge_seed_results

        a = _make_item(chunk_id=uuid.uuid4(), score=0.9)
        result = _merge_seed_results([Exception("fail"), [a]])
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_duplicate_across_lists_keeps_highest(self):
        from app.api.v1.retrieval import _merge_seed_results

        cid = uuid.uuid4()
        low = _make_item(chunk_id=cid, score=0.3)
        high = _make_item(chunk_id=cid, score=0.95)
        result = _merge_seed_results([[low], [high]])
        assert len(result) == 1
        assert abs(result[0].score - 0.95) < 1e-6


# ---------------------------------------------------------------------------
# 6. _build_qdrant_filters
# ---------------------------------------------------------------------------

class TestBuildQdrantFiltersCoverage:
    """Additional Qdrant filter builder tests."""

    def test_empty_filters_object_returns_none(self):
        from app.api.v1.retrieval import _build_qdrant_filters
        from app.schemas.retrieval import QueryFilters

        body = _make_body(filters=QueryFilters())
        assert _build_qdrant_filters(body) is None

    def test_classification_only(self):
        from app.api.v1.retrieval import _build_qdrant_filters
        from app.schemas.retrieval import QueryFilters

        body = _make_body(filters=QueryFilters(classification="CUI"))
        result = _build_qdrant_filters(body)
        assert result == {"classification": "CUI"}

    def test_single_modality_is_string(self):
        from app.api.v1.retrieval import _build_qdrant_filters
        from app.schemas.retrieval import QueryFilters

        body = _make_body(filters=QueryFilters(modalities=["image"]))
        result = _build_qdrant_filters(body)
        assert result["modality"] == "image"

    def test_multiple_modalities_is_list(self):
        from app.api.v1.retrieval import _build_qdrant_filters
        from app.schemas.retrieval import QueryFilters

        body = _make_body(filters=QueryFilters(modalities=["text", "table", "image"]))
        result = _build_qdrant_filters(body)
        assert result["modality"] == ["text", "table", "image"]


# ---------------------------------------------------------------------------
# 7. _apply_reranker
# ---------------------------------------------------------------------------

class TestApplyReranker:
    """Tests for _apply_reranker."""

    def test_disabled_reranker_returns_original(self):
        from app.api.v1.retrieval import _apply_reranker

        with patch("app.config.get_settings", return_value=_mock_settings(reranker_enabled=False)):
            items = [_make_item(score=0.8), _make_item(score=0.6)]
            body = _make_body()
            result = _apply_reranker(items, body)
            assert result == items

    def test_no_query_text_returns_original(self):
        from app.api.v1.retrieval import _apply_reranker
        from app.schemas.retrieval import UnifiedQueryRequest

        with patch("app.config.get_settings", return_value=_mock_settings(reranker_enabled=True)):
            items = [_make_item(score=0.8)]
            body = UnifiedQueryRequest.model_construct(
                query_text=None, query_image="base64data", strategy="basic",
                modality_filter="all", top_k=10, include_context=True,
            )
            result = _apply_reranker(items, body)
            assert result == items

    def test_reranker_reorders_results(self):
        from app.api.v1.retrieval import _apply_reranker

        cid1 = str(uuid.uuid4())
        cid2 = str(uuid.uuid4())
        items = [
            _make_item(chunk_id=uuid.UUID(cid1), score=0.8, content_text="Low relevance"),
            _make_item(chunk_id=uuid.UUID(cid2), score=0.6, content_text="High relevance"),
        ]

        # Reranker returns cid2 first with higher reranker_score
        mock_rerank_fn = MagicMock(return_value=[
            {"chunk_id": cid2, "content_text": "High relevance", "reranker_score": 0.95, "modality": "text"},
            {"chunk_id": cid1, "content_text": "Low relevance", "reranker_score": 0.4, "modality": "text"},
        ])

        with patch("app.config.get_settings", return_value=_mock_settings(reranker_enabled=True, reranker_top_n=20)), \
             patch("app.services.reranker.rerank", mock_rerank_fn):
            body = _make_body(top_k=10)
            result = _apply_reranker(items, body)

        assert len(result) == 2
        assert str(result[0].chunk_id) == cid2
        assert result[0].score == 0.95


# ---------------------------------------------------------------------------
# 8. GraphRAG service functions (patching at lower level)
# ---------------------------------------------------------------------------

class TestGraphRAGServiceLocalSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.build_graphrag_config")
    @patch("app.services.graphrag_service.get_settings")
    def test_success(self, mock_gs, mock_config, mock_load):
        from app.services.graphrag_service import local_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = MagicMock()

        with patch("app.services.graphrag_service._run_local_search") as mock_run:
            mock_run.return_value = ("Local answer", {"entities": []})
            result = local_search("test query")

        assert result["response"] == "Local answer"

    @patch("app.services.graphrag_service._load_search_data", side_effect=Exception("boom"))
    @patch("app.services.graphrag_service.get_settings")
    def test_exception_returns_empty(self, mock_gs, mock_load):
        from app.services.graphrag_service import local_search

        mock_gs.return_value = _mock_settings()
        result = local_search("test query")
        assert result["response"] == ""
        assert result["context"] == {}


class TestGraphRAGServiceGlobalSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.build_graphrag_config")
    @patch("app.services.graphrag_service.get_settings")
    def test_success(self, mock_gs, mock_config, mock_load):
        from app.services.graphrag_service import global_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = MagicMock()

        with patch("app.services.graphrag_service._run_global_search") as mock_run:
            mock_run.return_value = ("Global answer", {"reports": []})
            result = global_search("broad question")

        assert result["response"] == "Global answer"

    @patch("app.services.graphrag_service._load_search_data", side_effect=RuntimeError("no data"))
    @patch("app.services.graphrag_service.get_settings")
    def test_exception_returns_empty(self, mock_gs, mock_load):
        from app.services.graphrag_service import global_search

        mock_gs.return_value = _mock_settings()
        result = global_search("test")
        assert result["response"] == ""


class TestGraphRAGServiceDriftSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.build_graphrag_config")
    @patch("app.services.graphrag_service.get_settings")
    def test_success(self, mock_gs, mock_config, mock_load):
        from app.services.graphrag_service import drift_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = MagicMock()

        with patch("app.services.graphrag_service._run_drift_search") as mock_run:
            mock_run.return_value = ("Drift answer", {"entities": []})
            result = drift_search("guidance question")

        assert result["response"] == "Drift answer"

    @patch("app.services.graphrag_service._load_search_data", side_effect=Exception("crash"))
    @patch("app.services.graphrag_service.get_settings")
    def test_exception_returns_empty(self, mock_gs, mock_load):
        from app.services.graphrag_service import drift_search

        mock_gs.return_value = _mock_settings()
        result = drift_search("test")
        assert result["response"] == ""


class TestGraphRAGServiceBasicSearch:
    @patch("app.services.graphrag_service._load_search_data")
    @patch("app.services.graphrag_service.build_graphrag_config")
    @patch("app.services.graphrag_service.get_settings")
    def test_success(self, mock_gs, mock_config, mock_load):
        from app.services.graphrag_service import basic_search

        mock_gs.return_value = _mock_settings()
        mock_load.return_value = MagicMock()

        with patch("app.services.graphrag_service._run_basic_search") as mock_run:
            mock_run.return_value = ("Basic answer", {"chunks": []})
            result = basic_search("simple query")

        assert result["response"] == "Basic answer"

    @patch("app.services.graphrag_service._load_search_data", side_effect=Exception("oops"))
    @patch("app.services.graphrag_service.get_settings")
    def test_exception_returns_empty(self, mock_gs, mock_load):
        from app.services.graphrag_service import basic_search

        mock_gs.return_value = _mock_settings()
        result = basic_search("test")
        assert result["response"] == ""


# ---------------------------------------------------------------------------
# 9. compute_fusion_score
# ---------------------------------------------------------------------------

class TestComputeFusionScoreCoverage:
    """Additional fusion score tests beyond test_retrieval_helpers.py."""

    def _call(self, **kwargs):
        from app.api.v1._retrieval_helpers import compute_fusion_score
        return compute_fusion_score(**kwargs)

    def test_all_components(self):
        """Score with semantic + doc_structure + ontology is higher than semantic alone."""
        base = self._call(semantic_score=0.8)
        full = self._call(
            semantic_score=0.8,
            doc_structure_weight=0.9,
            doc_structure_hops=1,
            ontology_rel_type="IS_VARIANT_OF",
            ontology_hops=1,
        )
        assert full > base

    def test_higher_doc_weight_higher_score(self):
        low = self._call(semantic_score=0.8, doc_structure_weight=0.3, doc_structure_hops=1)
        high = self._call(semantic_score=0.8, doc_structure_weight=0.9, doc_structure_hops=1)
        assert high > low

    def test_no_mil_bonus_when_no_match(self):
        """No bonus when query MIL IDs don't appear in content."""
        no_match = self._call(
            semantic_score=0.8,
            content_text="Unrelated content text",
            query_text="MIL-STD-810G testing",
        )
        plain = self._call(semantic_score=0.8)
        assert abs(no_match - plain) < 1e-6


# ---------------------------------------------------------------------------
# 10. deduplicate_results / diversify_results
# ---------------------------------------------------------------------------

class TestDiversifyResults:
    """Tests for diversify_results (content-level dedup)."""

    def test_same_text_same_doc_page_deduped(self):
        from app.api.v1._retrieval_helpers import diversify_results

        doc_id = uuid.uuid4()
        a = _make_item(chunk_id=uuid.uuid4(), document_id=doc_id, page_number=1,
                        content_text="Same text here", score=0.9, modality="text")
        b = _make_item(chunk_id=uuid.uuid4(), document_id=doc_id, page_number=1,
                        content_text="Same text here", score=0.5, modality="text")
        result = diversify_results([a, b])
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_different_pages_not_deduped(self):
        from app.api.v1._retrieval_helpers import diversify_results

        doc_id = uuid.uuid4()
        a = _make_item(chunk_id=uuid.uuid4(), document_id=doc_id, page_number=1,
                        content_text="Same text", score=0.9, modality="text")
        b = _make_item(chunk_id=uuid.uuid4(), document_id=doc_id, page_number=2,
                        content_text="Same text", score=0.5, modality="text")
        result = diversify_results([a, b])
        assert len(result) == 2

    def test_image_modality_passes_through(self):
        from app.api.v1._retrieval_helpers import diversify_results

        a = _make_item(modality="image", content_text="Same caption", score=0.9)
        b = _make_item(modality="image", content_text="Same caption", score=0.7)
        result = diversify_results([a, b])
        # Images pass through regardless of content
        assert len(result) == 2

    def test_none_content_text_passes_through(self):
        from app.api.v1._retrieval_helpers import diversify_results

        a = _make_item(content_text=None, score=0.9, modality="text")
        b = _make_item(content_text=None, score=0.7, modality="text")
        result = diversify_results([a, b])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 11. _populate_image_urls
# ---------------------------------------------------------------------------

class TestPopulateImageUrls:
    """Tests for _populate_image_urls."""

    @pytest.mark.asyncio
    async def test_sets_url_for_image_modality(self):
        from app.api.v1.retrieval import _populate_image_urls

        cid = uuid.uuid4()
        item = _make_item(chunk_id=cid, modality="image")
        db = AsyncMock()
        await _populate_image_urls(db, [item])
        assert item.image_url == f"/v1/images/{cid}"

    @pytest.mark.asyncio
    async def test_sets_url_for_schematic_modality(self):
        from app.api.v1.retrieval import _populate_image_urls

        cid = uuid.uuid4()
        item = _make_item(chunk_id=cid, modality="schematic")
        db = AsyncMock()
        await _populate_image_urls(db, [item])
        assert item.image_url == f"/v1/images/{cid}"

    @pytest.mark.asyncio
    async def test_no_url_for_text_modality(self):
        from app.api.v1.retrieval import _populate_image_urls

        item = _make_item(modality="text")
        db = AsyncMock()
        await _populate_image_urls(db, [item])
        assert item.image_url is None

    @pytest.mark.asyncio
    async def test_no_url_for_graphrag_response(self):
        from app.api.v1.retrieval import _populate_image_urls

        item = _make_item(modality="graphrag_response", chunk_id=None)
        db = AsyncMock()
        await _populate_image_urls(db, [item])
        assert item.image_url is None

    @pytest.mark.asyncio
    async def test_empty_list_no_error(self):
        from app.api.v1.retrieval import _populate_image_urls

        db = AsyncMock()
        await _populate_image_urls(db, [])
        # Should not raise


# ---------------------------------------------------------------------------
# 12. unified_query endpoint routing
# ---------------------------------------------------------------------------

class TestUnifiedQueryRouting:
    """Tests verifying that unified_query dispatches to the correct handler."""

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._populate_image_urls", new_callable=AsyncMock)
    @patch("app.api.v1.retrieval._text_vector_search", new_callable=AsyncMock)
    async def test_basic_strategy_calls_text_vector_search(self, mock_text, mock_urls):
        from app.api.v1.retrieval import unified_query

        mock_text.return_value = [_make_item(score=0.9)]
        mock_urls.return_value = None

        body = _make_body(strategy="basic", top_k=5, include_context=False)
        db = AsyncMock()
        response = await unified_query(body, db)

        mock_text.assert_awaited_once()
        assert response.strategy == "basic"
        assert response.total == 1

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._populate_image_urls", new_callable=AsyncMock)
    @patch("app.api.v1.retrieval._multi_modal_pipeline", new_callable=AsyncMock)
    async def test_hybrid_strategy_calls_multi_modal(self, mock_mm, mock_urls):
        from app.api.v1.retrieval import unified_query

        mock_mm.return_value = [_make_item(score=0.85)]
        mock_urls.return_value = None

        body = _make_body(strategy="hybrid", mode="multi_modal", top_k=5, include_context=False)
        db = AsyncMock()
        response = await unified_query(body, db)

        mock_mm.assert_awaited_once()
        assert response.strategy == "hybrid"

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._populate_image_urls", new_callable=AsyncMock)
    @patch("app.api.v1.retrieval._graphrag_local_query", new_callable=AsyncMock)
    async def test_graphrag_local_strategy(self, mock_local, mock_urls):
        from app.api.v1.retrieval import unified_query

        mock_local.return_value = [_make_item(score=1.0, modality="graphrag_response")]
        mock_urls.return_value = None

        body = _make_body(strategy="graphrag_local", top_k=5, include_context=False)
        db = AsyncMock()
        response = await unified_query(body, db)

        mock_local.assert_awaited_once()
        assert response.strategy == "graphrag_local"

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._populate_image_urls", new_callable=AsyncMock)
    @patch("app.api.v1.retrieval._graphrag_global_query", new_callable=AsyncMock)
    async def test_graphrag_global_strategy(self, mock_global, mock_urls):
        from app.api.v1.retrieval import unified_query

        mock_global.return_value = [_make_item(score=1.0, modality="graphrag_response")]
        mock_urls.return_value = None

        body = _make_body(strategy="graphrag_global", top_k=5, include_context=False)
        db = AsyncMock()
        response = await unified_query(body, db)

        mock_global.assert_awaited_once()
        assert response.strategy == "graphrag_global"

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._populate_image_urls", new_callable=AsyncMock)
    @patch("app.api.v1.retrieval._graphrag_drift_query", new_callable=AsyncMock)
    async def test_graphrag_drift_strategy(self, mock_drift, mock_urls):
        from app.api.v1.retrieval import unified_query

        mock_drift.return_value = [_make_item(score=1.0, modality="graphrag_response")]
        mock_urls.return_value = None

        body = _make_body(strategy="graphrag_drift", top_k=5, include_context=False)
        db = AsyncMock()
        response = await unified_query(body, db)

        mock_drift.assert_awaited_once()
        assert response.strategy == "graphrag_drift"

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._populate_image_urls", new_callable=AsyncMock)
    @patch("app.api.v1.retrieval._graphrag_basic_query", new_callable=AsyncMock)
    async def test_graphrag_basic_strategy(self, mock_basic, mock_urls):
        from app.api.v1.retrieval import unified_query

        mock_basic.return_value = [_make_item(score=1.0, modality="graphrag_response")]
        mock_urls.return_value = None

        body = _make_body(strategy="graphrag_basic", top_k=5, include_context=False)
        db = AsyncMock()
        response = await unified_query(body, db)

        mock_basic.assert_awaited_once()
        assert response.strategy == "graphrag_basic"

    @pytest.mark.asyncio
    @patch("app.api.v1.retrieval._populate_image_urls", new_callable=AsyncMock)
    @patch("app.api.v1.retrieval._text_vector_search", new_callable=AsyncMock, side_effect=RuntimeError("fail"))
    async def test_exception_returns_empty_results(self, mock_text, mock_urls):
        from app.api.v1.retrieval import unified_query

        mock_urls.return_value = None

        body = _make_body(strategy="basic", top_k=5, include_context=False)
        db = AsyncMock()
        response = await unified_query(body, db)

        assert response.total == 0
        assert response.results == []
