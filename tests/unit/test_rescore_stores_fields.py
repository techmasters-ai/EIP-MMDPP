from __future__ import annotations
from unittest.mock import patch
import numpy as np
import app.api.v1.retrieval as R
from app.schemas.retrieval import QueryResultItem
import uuid


async def test_rescore_stores_raw_cosine_and_fused():
    item = QueryResultItem(chunk_id=uuid.uuid4(), score=0.99, modality="text",
                           content_text="Fan Song radar associated with SA-2",
                           context={"source": "ontology_relation", "rel_type": "ASSOCIATED_WITH"})
    with patch("app.services.embedding.embed_texts",
               lambda texts, query=False: [np.ones(4) / 2.0 for _ in texts]):
        out = await R._rescore_expanded_chunks([item], "SA-2 guidance radar")
    ctx = out[0].context
    assert "raw_cosine" in ctx
    assert "fused_score_pre_rerank" in ctx
    assert ctx["fused_score_pre_rerank"] == out[0].score
    assert 0.0 <= ctx["raw_cosine"] <= 1.0
