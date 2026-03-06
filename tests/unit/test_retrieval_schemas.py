"""Unit tests for retrieval Pydantic schemas.

Tests QueryStrategy/ModalityFilter enums, UnifiedQueryRequest validation
(defaults, boundaries, model_validator, backward-compat mode mapping),
UnifiedQueryResponse shape, and QueryResultItem defaults.
"""

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# QueryStrategy enum
# ---------------------------------------------------------------------------

class TestQueryStrategy:
    def test_all_strategy_values_exist(self):
        from app.schemas.retrieval import QueryStrategy
        expected = {"basic", "hybrid", "memory", "graphrag_local", "graphrag_global"}
        actual = {m.value for m in QueryStrategy}
        assert actual == expected

    def test_has_five_members(self):
        from app.schemas.retrieval import QueryStrategy
        assert len(QueryStrategy) == 5

    def test_strategy_string_values(self):
        from app.schemas.retrieval import QueryStrategy
        assert QueryStrategy.basic.value == "basic"
        assert QueryStrategy.hybrid.value == "hybrid"
        assert QueryStrategy.memory.value == "memory"
        assert QueryStrategy.graphrag_local.value == "graphrag_local"
        assert QueryStrategy.graphrag_global.value == "graphrag_global"

    def test_strategy_is_str_enum(self):
        from app.schemas.retrieval import QueryStrategy
        assert isinstance(QueryStrategy.basic, str)
        assert QueryStrategy.basic == "basic"


# ---------------------------------------------------------------------------
# ModalityFilter enum
# ---------------------------------------------------------------------------

class TestModalityFilter:
    def test_all_modality_values(self):
        from app.schemas.retrieval import ModalityFilter
        expected = {"all", "text", "image"}
        actual = {m.value for m in ModalityFilter}
        assert actual == expected

    def test_has_three_members(self):
        from app.schemas.retrieval import ModalityFilter
        assert len(ModalityFilter) == 3


# ---------------------------------------------------------------------------
# UnifiedQueryRequest validation
# ---------------------------------------------------------------------------

class TestUnifiedQueryRequest:
    def _make(self, **kwargs):
        from app.schemas.retrieval import UnifiedQueryRequest
        return UnifiedQueryRequest(**kwargs)

    def test_default_strategy_is_basic(self):
        from app.schemas.retrieval import QueryStrategy
        req = self._make(query_text="test")
        assert req.strategy == QueryStrategy.basic

    def test_default_modality_filter_is_all(self):
        from app.schemas.retrieval import ModalityFilter
        req = self._make(query_text="test")
        assert req.modality_filter == ModalityFilter.all

    def test_default_top_k_is_10(self):
        req = self._make(query_text="test")
        assert req.top_k == 10

    def test_default_include_context_is_true(self):
        req = self._make(query_text="test")
        assert req.include_context is True

    def test_top_k_min_is_1(self):
        with pytest.raises(ValidationError):
            self._make(query_text="test", top_k=0)

    def test_top_k_max_is_100(self):
        with pytest.raises(ValidationError):
            self._make(query_text="test", top_k=101)

    def test_top_k_boundary_1(self):
        req = self._make(query_text="test", top_k=1)
        assert req.top_k == 1

    def test_top_k_boundary_100(self):
        req = self._make(query_text="test", top_k=100)
        assert req.top_k == 100

    def test_requires_at_least_one_query(self):
        with pytest.raises(ValidationError, match="(?i)at least one"):
            self._make()

    def test_query_text_only_valid(self):
        req = self._make(query_text="some text")
        assert req.query_text == "some text"
        assert req.query_image is None

    def test_query_image_only_valid(self):
        req = self._make(query_image="base64data")
        assert req.query_image == "base64data"
        assert req.query_text is None

    def test_both_queries_valid(self):
        req = self._make(query_text="text", query_image="img")
        assert req.query_text == "text"
        assert req.query_image == "img"

    def test_query_text_max_length(self):
        with pytest.raises(ValidationError):
            self._make(query_text="x" * 4097)

    def test_query_text_at_max_length(self):
        req = self._make(query_text="x" * 4096)
        assert len(req.query_text) == 4096

    def test_filters_default_none(self):
        req = self._make(query_text="test")
        assert req.filters is None

    def test_strategy_accepts_valid_string(self):
        from app.schemas.retrieval import QueryStrategy
        req = self._make(query_text="test", strategy="hybrid")
        assert req.strategy == QueryStrategy.hybrid

    def test_strategy_rejects_invalid_string(self):
        with pytest.raises(ValidationError):
            self._make(query_text="test", strategy="invalid_strategy")

    # Backward-compat: legacy mode field
    def test_legacy_mode_text_basic_maps_to_basic(self):
        from app.schemas.retrieval import QueryStrategy, ModalityFilter
        req = self._make(query_text="test", mode="text_basic")
        assert req.strategy == QueryStrategy.basic
        assert req.modality_filter == ModalityFilter.all

    def test_legacy_mode_text_only_maps_to_hybrid_text(self):
        from app.schemas.retrieval import QueryStrategy, ModalityFilter
        req = self._make(query_text="test", mode="text_only")
        assert req.strategy == QueryStrategy.hybrid
        assert req.modality_filter == ModalityFilter.text

    def test_legacy_mode_images_only_maps_to_hybrid_image(self):
        from app.schemas.retrieval import QueryStrategy, ModalityFilter
        req = self._make(query_text="test", mode="images_only")
        assert req.strategy == QueryStrategy.hybrid
        assert req.modality_filter == ModalityFilter.image

    def test_legacy_mode_multi_modal_maps_to_hybrid_all(self):
        from app.schemas.retrieval import QueryStrategy, ModalityFilter
        req = self._make(query_text="test", mode="multi_modal")
        assert req.strategy == QueryStrategy.hybrid
        assert req.modality_filter == ModalityFilter.all

    def test_legacy_mode_memory(self):
        from app.schemas.retrieval import QueryStrategy
        req = self._make(query_text="test", mode="memory")
        assert req.strategy == QueryStrategy.memory

    def test_legacy_mode_graphrag_local(self):
        from app.schemas.retrieval import QueryStrategy
        req = self._make(query_text="test", mode="graphrag_local")
        assert req.strategy == QueryStrategy.graphrag_local

    def test_legacy_mode_graphrag_global(self):
        from app.schemas.retrieval import QueryStrategy
        req = self._make(query_text="test", mode="graphrag_global")
        assert req.strategy == QueryStrategy.graphrag_global


# ---------------------------------------------------------------------------
# UnifiedQueryResponse
# ---------------------------------------------------------------------------

class TestUnifiedQueryResponse:
    def test_response_has_required_fields(self):
        from app.schemas.retrieval import UnifiedQueryResponse
        resp = UnifiedQueryResponse(strategy="basic", modality_filter="all", results=[], total=0)
        assert resp.strategy == "basic"
        assert resp.modality_filter == "all"
        assert resp.results == []
        assert resp.total == 0

    def test_response_accepts_results_list(self):
        from app.schemas.retrieval import UnifiedQueryResponse, QueryResultItem
        item = QueryResultItem(score=0.9, modality="text")
        resp = UnifiedQueryResponse(strategy="hybrid", modality_filter="all", results=[item], total=1)
        assert len(resp.results) == 1
        assert resp.total == 1

    def test_response_no_sections_attribute(self):
        from app.schemas.retrieval import UnifiedQueryResponse
        resp = UnifiedQueryResponse(strategy="basic", modality_filter="all", results=[], total=0)
        assert not hasattr(resp, "sections")

    def test_response_optional_query_fields(self):
        from app.schemas.retrieval import UnifiedQueryResponse
        resp = UnifiedQueryResponse(
            strategy="basic", modality_filter="all", results=[], total=0,
            query_text="test", query_image="img",
        )
        assert resp.query_text == "test"
        assert resp.query_image == "img"


# ---------------------------------------------------------------------------
# QueryResultItem
# ---------------------------------------------------------------------------

class TestQueryResultItem:
    def test_default_classification_is_unclassified(self):
        from app.schemas.retrieval import QueryResultItem
        item = QueryResultItem(score=0.5, modality="text")
        assert item.classification == "UNCLASSIFIED"

    def test_context_field_none_by_default(self):
        from app.schemas.retrieval import QueryResultItem
        item = QueryResultItem(score=0.5, modality="text")
        assert item.context is None

    def test_context_field_accepts_dict(self):
        from app.schemas.retrieval import QueryResultItem
        ctx = {"source": "ontology", "rel_type": "IS_SUBSYSTEM_OF"}
        item = QueryResultItem(score=0.5, modality="text", context=ctx)
        assert item.context == ctx

    def test_chunk_id_optional(self):
        from app.schemas.retrieval import QueryResultItem
        item = QueryResultItem(score=0.5, modality="text")
        assert item.chunk_id is None

    def test_content_text_optional(self):
        from app.schemas.retrieval import QueryResultItem
        item = QueryResultItem(score=0.5, modality="text")
        assert item.content_text is None

    def test_page_number_optional(self):
        from app.schemas.retrieval import QueryResultItem
        item = QueryResultItem(score=0.5, modality="text")
        assert item.page_number is None

    def test_accepts_all_modality_values(self):
        from app.schemas.retrieval import QueryResultItem
        for mod in ("text", "image", "table", "schematic"):
            item = QueryResultItem(score=0.5, modality=mod)
            assert item.modality == mod

    def test_image_url_default_none(self):
        from app.schemas.retrieval import QueryResultItem
        item = QueryResultItem(score=0.5, modality="image")
        assert item.image_url is None

    def test_image_url_accepts_string(self):
        from app.schemas.retrieval import QueryResultItem
        item = QueryResultItem(
            score=0.5, modality="image",
            image_url="https://minio:9000/eip-derived/artifacts/test.png?X-Amz-Signature=abc",
        )
        assert item.image_url is not None
        assert "minio" in item.image_url
