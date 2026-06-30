from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock


def _store_with_rows(rows, seed_type="TextChunk"):
    from app.services.arcadedb_graph import ArcadeDBGraphStore
    client = MagicMock()
    client.query = AsyncMock(side_effect=[
        [{"@rid": "#12:0"}],          # _resolve_rid
        [{"node_type": seed_type}],   # @type lookup
        rows,                          # MATCH
    ])
    return ArcadeDBGraphStore(client, "db"), client


async def test_returns_related_chunks_with_true_rel_type():
    rows = [{
        "chunk_rid": "#20:1", "chunk_type": "TextChunk", "chunk_id": "c-fan-song",
        "document_id": "d1", "text": "Fan Song ...", "modality": "text",
        "related_entity": "Fan Song", "rel_type": "ASSOCIATED_WITH",
    }]
    store, _ = _store_with_rows(rows)
    out = await store.get_related_entity_chunks("c-sa2", ["ASSOCIATED_WITH", "CUES"], 5)
    assert len(out) == 1
    assert out[0]["chunk_id"] == "c-fan-song"
    assert out[0]["rel_type"] == "ASSOCIATED_WITH"
    assert out[0]["related_entity"] == "Fan Song"


async def test_dedup_keeps_strongest_relation():
    rows = [
        {"chunk_rid": "#20:1", "chunk_id": "c1", "rel_type": "ASSOCIATED_WITH", "related_entity": "X"},
        {"chunk_rid": "#20:1", "chunk_id": "c1", "rel_type": "VARIANT_OF", "related_entity": "X"},
    ]
    store, _ = _store_with_rows(rows)
    out = await store.get_related_entity_chunks("c-sa2", ["ASSOCIATED_WITH", "VARIANT_OF"], 5)
    assert len(out) == 1
    assert out[0]["rel_type"] == "VARIANT_OF"


async def test_limit_and_non_chunk_seed():
    store, _ = _store_with_rows([{"chunk_rid": f"#20:{i}", "chunk_id": f"c{i}",
                                  "rel_type": "CUES", "related_entity": "X"} for i in range(10)])
    out = await store.get_related_entity_chunks("c", ["CUES"], 3)
    assert len(out) == 3
    from app.services.arcadedb_graph import ArcadeDBGraphStore
    c2 = MagicMock(); c2.query = AsyncMock(side_effect=[[{"@rid": "#9:0"}], [{"node_type": "Entity"}]])
    s2 = ArcadeDBGraphStore(c2, "db")
    assert await s2.get_related_entity_chunks("e", ["CUES"], 5) == []
