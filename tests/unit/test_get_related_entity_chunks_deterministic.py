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


async def test_equal_weight_ties_broken_by_chunk_id_stable_across_input_order():
    """Two equal-weight rows (same rel_type => same weight) must come back in
    the SAME order regardless of the order ArcadeDB happened to return them in.
    Ties are broken deterministically by ascending chunk_id."""
    rows_a = [
        {"chunk_rid": "#20:1", "chunk_id": "c-bravo", "rel_type": "CUES", "related_entity": "X"},
        {"chunk_rid": "#20:2", "chunk_id": "c-alpha", "rel_type": "CUES", "related_entity": "Y"},
    ]
    rows_b = list(reversed(rows_a))  # same rows, opposite arrival order

    store_a, _ = _store_with_rows(rows_a)
    store_b, _ = _store_with_rows(rows_b)

    out_a = await store_a.get_related_entity_chunks("c-seed", ["CUES"], 5)
    out_b = await store_b.get_related_entity_chunks("c-seed", ["CUES"], 5)

    ids_a = [r["chunk_id"] for r in out_a]
    ids_b = [r["chunk_id"] for r in out_b]

    # Identical output order regardless of input order
    assert ids_a == ids_b
    # Tie broken by ascending chunk_id
    assert ids_a == ["c-alpha", "c-bravo"]


async def test_limit_truncation_is_deterministic_on_ties():
    """When equal-weight candidates exceed the limit, the kept subset must be
    the deterministic chunk_id-sorted prefix, not an arbitrary DB-order slice."""
    rows = [
        {"chunk_rid": f"#20:{i}", "chunk_id": cid, "rel_type": "CUES", "related_entity": "X"}
        for i, cid in enumerate(["c5", "c1", "c3", "c2", "c4"])
    ]
    store_fwd, _ = _store_with_rows(rows)
    store_rev, _ = _store_with_rows(list(reversed(rows)))

    out_fwd = await store_fwd.get_related_entity_chunks("c-seed", ["CUES"], 3)
    out_rev = await store_rev.get_related_entity_chunks("c-seed", ["CUES"], 3)

    ids_fwd = [r["chunk_id"] for r in out_fwd]
    ids_rev = [r["chunk_id"] for r in out_rev]

    assert ids_fwd == ids_rev
    assert ids_fwd == ["c1", "c2", "c3"]
