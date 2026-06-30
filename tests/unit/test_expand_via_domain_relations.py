from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock
import uuid
import app.api.v1.retrieval as R
from app.schemas.retrieval import QueryResultItem


def _item():
    return QueryResultItem(chunk_id=uuid.uuid4(), score=0.0, modality="text", content_text="Fan Song radar")


async def test_domain_expansion_stamps_ontology_relation(monkeypatch):
    gs = MagicMock()
    gs.get_related_entity_chunks = AsyncMock(return_value=[
        {"target_chunk_id": "c1", "target_chunk_type": "text_chunk",
         "rel_type": "ASSOCIATED_WITH", "related_entity": "Fan Song"},
    ])
    monkeypatch.setattr(R, "get_graph_store", lambda: gs)
    monkeypatch.setattr(R, "_lookup_chunk_by_type", AsyncMock(return_value=_item()))
    out = await R._expand_via_domain_relations("c-sa2", 0.7, True, "SA-2")
    assert len(out) == 1
    ctx = out[0].context
    assert ctx["source"] == "ontology_relation"
    assert ctx["rel_type"] == "ASSOCIATED_WITH"
    assert ctx["related_entity"] == "Fan Song"
    assert ctx["source_chunk_id"] == "c-sa2"


async def test_expand_seeds_respects_master_flag(monkeypatch):
    from app.config import get_settings
    get_settings.cache_clear()
    monkeypatch.setenv("RETRIEVAL_DOMAIN_EXPANSION_ENABLED", "false")
    get_settings.cache_clear()
    called = AsyncMock(return_value=[])
    monkeypatch.setattr(R, "_expand_via_domain_relations", called)
    monkeypatch.setattr(R, "_expand_via_doc_structure", AsyncMock(return_value=[_item()]))
    monkeypatch.setattr(R, "_expand_via_ontology", AsyncMock(return_value=[]))
    seed = _item()
    await R._expand_seeds(MagicMock(), [seed], True, "SA-2")
    called.assert_not_called()
    get_settings.cache_clear()
