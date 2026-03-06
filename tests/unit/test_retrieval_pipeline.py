"""Unit tests for retrieval pipeline helpers.

Tests _build_qdrant_filters and _merge_seed_results from retrieval.py.
"""

import uuid
from unittest.mock import MagicMock

import pytest

pytest.importorskip("asyncpg", reason="asyncpg not installed")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_body(**kwargs):
    """Build a mock UnifiedQueryRequest."""
    from app.schemas.retrieval import UnifiedQueryRequest
    defaults = {
        "query_text": "test query",
        "mode": "multi_modal",
    }
    defaults.update(kwargs)
    return UnifiedQueryRequest(**defaults)


def _make_result_item(**kwargs):
    """Build a QueryResultItem."""
    from app.schemas.retrieval import QueryResultItem
    defaults = {
        "chunk_id": str(uuid.uuid4()),
        "score": 0.5,
        "modality": "text",
    }
    defaults.update(kwargs)
    return QueryResultItem(**defaults)


# ---------------------------------------------------------------------------
# _build_qdrant_filters
# ---------------------------------------------------------------------------

class TestBuildQdrantFilters:
    def test_no_filters_returns_none(self):
        from app.api.v1.retrieval import _build_qdrant_filters
        body = _make_body()
        assert _build_qdrant_filters(body) is None

    def test_classification_filter(self):
        from app.api.v1.retrieval import _build_qdrant_filters
        from app.schemas.retrieval import QueryFilters
        body = _make_body(filters=QueryFilters(classification="SECRET"))
        result = _build_qdrant_filters(body)
        assert result is not None
        assert result["classification"] == "SECRET"

    def test_document_ids_filter(self):
        from app.api.v1.retrieval import _build_qdrant_filters
        from app.schemas.retrieval import QueryFilters
        did = uuid.uuid4()
        body = _make_body(filters=QueryFilters(document_ids=[did]))
        result = _build_qdrant_filters(body)
        assert result["document_id"] == [str(did)]

    def test_multiple_document_ids_filter(self):
        from app.api.v1.retrieval import _build_qdrant_filters
        from app.schemas.retrieval import QueryFilters
        ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
        body = _make_body(filters=QueryFilters(document_ids=ids))
        result = _build_qdrant_filters(body)
        assert result["document_id"] == [str(d) for d in ids]

    def test_single_modality_filter(self):
        from app.api.v1.retrieval import _build_qdrant_filters
        from app.schemas.retrieval import QueryFilters
        body = _make_body(filters=QueryFilters(modalities=["text"]))
        result = _build_qdrant_filters(body)
        assert result["modality"] == "text"

    def test_multiple_modalities_included_as_list(self):
        from app.api.v1.retrieval import _build_qdrant_filters
        from app.schemas.retrieval import QueryFilters
        body = _make_body(filters=QueryFilters(modalities=["text", "image"]))
        result = _build_qdrant_filters(body)
        assert result is not None
        assert result["modality"] == ["text", "image"]

    def test_combined_filters(self):
        from app.api.v1.retrieval import _build_qdrant_filters
        from app.schemas.retrieval import QueryFilters
        body = _make_body(filters=QueryFilters(
            classification="UNCLASSIFIED",
            modalities=["text"],
        ))
        result = _build_qdrant_filters(body)
        assert result["classification"] == "UNCLASSIFIED"
        assert result["modality"] == "text"


# ---------------------------------------------------------------------------
# _merge_seed_results
# ---------------------------------------------------------------------------

class TestMergeSeedResults:
    def test_empty_input(self):
        from app.api.v1.retrieval import _merge_seed_results
        assert _merge_seed_results([]) == []

    def test_single_list_passed_through(self):
        from app.api.v1.retrieval import _merge_seed_results
        items = [_make_result_item()]
        result = _merge_seed_results([items])
        assert len(result) == 1

    def test_exception_results_skipped(self):
        from app.api.v1.retrieval import _merge_seed_results
        items = [_make_result_item()]
        result = _merge_seed_results([Exception("fail"), items])
        assert len(result) == 1

    def test_keeps_highest_score_per_chunk_id(self):
        from app.api.v1.retrieval import _merge_seed_results
        cid = str(uuid.uuid4())
        low = _make_result_item(chunk_id=cid, score=0.3)
        high = _make_result_item(chunk_id=cid, score=0.9)
        result = _merge_seed_results([[low], [high]])
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_none_chunk_ids_unique(self):
        from app.api.v1.retrieval import _merge_seed_results
        a = _make_result_item(chunk_id=None, score=0.5)
        b = _make_result_item(chunk_id=None, score=0.6)
        result = _merge_seed_results([[a, b]])
        # None chunk_ids use id() so both should be present
        assert len(result) == 2
