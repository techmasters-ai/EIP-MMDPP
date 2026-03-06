"""Unit tests for retrieval Pydantic schemas.

Tests QueryMode enum values, UnifiedQueryRequest validation (defaults,
boundaries, model_validator), UnifiedQueryResponse shape, and
QueryResultItem defaults.
"""

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# QueryMode enum
# ---------------------------------------------------------------------------

class TestQueryMode:
    def test_all_mode_values_exist(self):
        from app.schemas.retrieval import QueryMode
        expected = {"text_basic", "text_only", "images_only", "multi_modal", "memory"}
        actual = {m.value for m in QueryMode}
        assert actual == expected

    def test_has_five_members(self):
        from app.schemas.retrieval import QueryMode
        assert len(QueryMode) == 5

    def test_mode_string_values(self):
        from app.schemas.retrieval import QueryMode
        assert QueryMode.text_basic.value == "text_basic"
        assert QueryMode.text_only.value == "text_only"
        assert QueryMode.images_only.value == "images_only"
        assert QueryMode.multi_modal.value == "multi_modal"
        assert QueryMode.memory.value == "memory"

    def test_mode_is_str_enum(self):
        from app.schemas.retrieval import QueryMode
        assert isinstance(QueryMode.text_basic, str)
        assert QueryMode.text_basic == "text_basic"


# ---------------------------------------------------------------------------
# UnifiedQueryRequest validation
# ---------------------------------------------------------------------------

class TestUnifiedQueryRequest:
    def _make(self, **kwargs):
        from app.schemas.retrieval import UnifiedQueryRequest
        return UnifiedQueryRequest(**kwargs)

    def test_default_mode_is_text_basic(self):
        from app.schemas.retrieval import QueryMode
        req = self._make(query_text="test")
        assert req.mode == QueryMode.text_basic

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

    def test_mode_accepts_valid_string(self):
        from app.schemas.retrieval import QueryMode
        req = self._make(query_text="test", mode="multi_modal")
        assert req.mode == QueryMode.multi_modal

    def test_mode_rejects_invalid_string(self):
        with pytest.raises(ValidationError):
            self._make(query_text="test", mode="invalid_mode")


# ---------------------------------------------------------------------------
# UnifiedQueryResponse
# ---------------------------------------------------------------------------

class TestUnifiedQueryResponse:
    def test_response_has_required_fields(self):
        from app.schemas.retrieval import UnifiedQueryResponse
        resp = UnifiedQueryResponse(mode="text_basic", results=[], total=0)
        assert resp.mode == "text_basic"
        assert resp.results == []
        assert resp.total == 0

    def test_response_accepts_results_list(self):
        from app.schemas.retrieval import UnifiedQueryResponse, QueryResultItem
        item = QueryResultItem(score=0.9, modality="text")
        resp = UnifiedQueryResponse(mode="multi_modal", results=[item], total=1)
        assert len(resp.results) == 1
        assert resp.total == 1

    def test_response_no_sections_attribute(self):
        from app.schemas.retrieval import UnifiedQueryResponse
        resp = UnifiedQueryResponse(mode="text_basic", results=[], total=0)
        assert not hasattr(resp, "sections")

    def test_response_optional_query_fields(self):
        from app.schemas.retrieval import UnifiedQueryResponse
        resp = UnifiedQueryResponse(
            mode="text_basic", results=[], total=0,
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
