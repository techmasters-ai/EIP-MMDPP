"""Unit tests for pure helper functions in app.api.v1._retrieval_helpers.

These tests exercise deduplicate_results, build_text_filters,
build_image_filters, and score decay constants — all of which have
no DB dependency and can be tested in isolation.
"""

import uuid

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(**kwargs):
    """Build a QueryResultItem with sensible defaults."""
    from app.schemas.retrieval import QueryResultItem

    defaults = dict(
        score=0.85,
        modality="text",
        content_text="Some chunk text.",
        classification="UNCLASSIFIED",
    )
    defaults.update(kwargs)
    return QueryResultItem(**defaults)


def _make_request(**kwargs):
    """Build a UnifiedQueryRequest with sensible defaults."""
    from app.schemas.retrieval import UnifiedQueryRequest

    defaults = dict(query_text="test query")
    defaults.update(kwargs)
    return UnifiedQueryRequest(**defaults)


# ---------------------------------------------------------------------------
# deduplicate_results
# ---------------------------------------------------------------------------

class TestDeduplicateResults:
    def _call(self, results):
        from app.api.v1._retrieval_helpers import deduplicate_results
        return deduplicate_results(results)

    def test_empty_list_returns_empty(self):
        assert self._call([]) == []

    def test_single_item_unchanged(self):
        item = _make_item(chunk_id=uuid.uuid4())
        result = self._call([item])
        assert len(result) == 1
        assert result[0].chunk_id == item.chunk_id

    def test_duplicate_chunk_ids_keeps_highest_score(self):
        cid = uuid.uuid4()
        low = _make_item(chunk_id=cid, score=0.5)
        high = _make_item(chunk_id=cid, score=0.9)
        result = self._call([low, high])
        assert len(result) == 1
        assert abs(result[0].score - 0.9) < 1e-6

    def test_duplicate_keeps_highest_regardless_of_order(self):
        cid = uuid.uuid4()
        high = _make_item(chunk_id=cid, score=0.9)
        low = _make_item(chunk_id=cid, score=0.5)
        result = self._call([high, low])
        assert len(result) == 1
        assert abs(result[0].score - 0.9) < 1e-6

    def test_none_chunk_ids_treated_independently(self):
        a = _make_item(chunk_id=None, score=0.8)
        b = _make_item(chunk_id=None, score=0.6)
        result = self._call([a, b])
        assert len(result) == 2

    def test_mixed_duplicates_and_uniques(self):
        cid1 = uuid.uuid4()
        cid2 = uuid.uuid4()
        cid3 = uuid.uuid4()
        items = [
            _make_item(chunk_id=cid1, score=0.5),
            _make_item(chunk_id=cid1, score=0.9),  # dup of cid1
            _make_item(chunk_id=cid2, score=0.7),
            _make_item(chunk_id=cid2, score=0.3),  # dup of cid2
            _make_item(chunk_id=cid3, score=0.6),   # unique
        ]
        result = self._call(items)
        assert len(result) == 3
        scores = {str(r.chunk_id): r.score for r in result}
        assert abs(scores[str(cid1)] - 0.9) < 1e-6
        assert abs(scores[str(cid2)] - 0.7) < 1e-6
        assert abs(scores[str(cid3)] - 0.6) < 1e-6

    def test_preserves_context_of_highest_scored(self):
        cid = uuid.uuid4()
        low = _make_item(chunk_id=cid, score=0.4, context={"source": "low"})
        high = _make_item(chunk_id=cid, score=0.8, context={"source": "high"})
        result = self._call([low, high])
        assert result[0].context["source"] == "high"


# ---------------------------------------------------------------------------
# build_text_filters
# ---------------------------------------------------------------------------

class TestBuildTextFilters:
    def _call(self, **kwargs):
        from app.api.v1._retrieval_helpers import build_text_filters
        body = _make_request(**kwargs)
        return build_text_filters(body)

    def test_no_filters_returns_empty_string(self):
        assert self._call() == ""

    def test_classification_filter(self):
        from app.schemas.retrieval import QueryFilters
        result = self._call(filters=QueryFilters(classification="SECRET"))
        assert "tc.classification" in result
        assert "SECRET" in result

    def test_document_ids_filter(self):
        from app.schemas.retrieval import QueryFilters
        doc_id = uuid.uuid4()
        result = self._call(filters=QueryFilters(document_ids=[doc_id]))
        assert "tc.document_id IN" in result
        assert str(doc_id) in result

    def test_both_filters(self):
        from app.schemas.retrieval import QueryFilters
        doc_id = uuid.uuid4()
        result = self._call(
            filters=QueryFilters(classification="CUI", document_ids=[doc_id])
        )
        assert "tc.classification" in result
        assert "tc.document_id IN" in result


# ---------------------------------------------------------------------------
# build_image_filters
# ---------------------------------------------------------------------------

class TestBuildImageFilters:
    def _call(self, **kwargs):
        from app.api.v1._retrieval_helpers import build_image_filters
        body = _make_request(**kwargs)
        return build_image_filters(body)

    def test_no_filters_returns_empty_string(self):
        assert self._call() == ""

    def test_classification_filter(self):
        from app.schemas.retrieval import QueryFilters
        result = self._call(filters=QueryFilters(classification="CUI"))
        assert "ic.classification" in result
        assert "CUI" in result

    def test_document_ids_filter(self):
        from app.schemas.retrieval import QueryFilters
        doc_id = uuid.uuid4()
        result = self._call(filters=QueryFilters(document_ids=[doc_id]))
        assert "ic.document_id IN" in result
        assert str(doc_id) in result


# ---------------------------------------------------------------------------
# Score decay constants
# ---------------------------------------------------------------------------

class TestScoreDecayConstants:
    def test_cross_modal_decay_value(self):
        from app.api.v1._retrieval_helpers import CROSS_MODAL_DECAY
        assert abs(CROSS_MODAL_DECAY - 0.85) < 1e-6

    def test_ontology_decay_value(self):
        from app.api.v1._retrieval_helpers import ONTOLOGY_DECAY
        assert abs(ONTOLOGY_DECAY - 0.75) < 1e-6
