from __future__ import annotations
import uuid
from unittest.mock import patch
import app.api.v1.retrieval as R
from app.schemas.retrieval import QueryResultItem, UnifiedQueryRequest


def _items():
    a = QueryResultItem(chunk_id=uuid.uuid4(), score=0.4, modality="text", content_text="a")
    b = QueryResultItem(chunk_id=uuid.uuid4(), score=0.2, modality="text", content_text="b")
    return [a, b]


def test_blend_alpha(monkeypatch):
    from app.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.setenv("RETRIEVAL_RERANK_BLEND_ALPHA", "0.6")
    get_settings.cache_clear()
    items = _items()
    def fake_rerank(query, candidates, top_k=10):
        out = []
        for i, c in enumerate(candidates):
            c = dict(c); c["reranker_score"] = 1.0 if i == 0 else 0.0; out.append(c)
        return out
    with patch("app.services.reranker.rerank", fake_rerank):
        body = UnifiedQueryRequest(query_text="q", top_k=2)
        out = R._apply_reranker(items, body)
    by_id = {str(o.chunk_id): o for o in out}
    a = by_id[str(items[0].chunk_id)]
    # 0.6*1.0 + 0.4*0.4 = 0.76
    assert abs(a.score - 0.76) < 1e-6
    get_settings.cache_clear()
