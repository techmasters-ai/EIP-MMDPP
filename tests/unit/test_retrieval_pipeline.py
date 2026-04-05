"""Unit tests for retrieval pipeline helpers.

Tests _merge_seed_results from retrieval.py.
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
