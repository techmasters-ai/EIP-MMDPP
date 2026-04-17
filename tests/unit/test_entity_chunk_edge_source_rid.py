"""Phase 8 Task 53b — EntityChunkEdge.source_rid + .entity_id routing.

The batch writer uses a direct RID-to-RID CREATE EDGE when
``source_rid`` is populated (no more name+type LIMIT-1 subquery
that silently attaches to the wrong vertex when two entities share
name + type). When ``source_rid`` is None, falls back to the legacy
subquery path with a WARNING.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def _make_store(client=None):
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    if client is None:
        client = MagicMock()
        client.command_sync = MagicMock(return_value=[])
        client.query_sync = MagicMock(return_value=[])

    store = ArcadeDBGraphStore(client=client, database="testdb")
    store._validation_matrix = set()
    return store


def test_entity_chunk_edge_dataclass_has_entity_id_and_source_rid_with_defaults():
    """New fields default to None so legacy callers that construct
    EntityChunkEdge with only (entity_name, entity_type, chunk_rid)
    still work."""
    from app.services.graph_store import EntityChunkEdge
    ece = EntityChunkEdge(
        entity_name="Tombstone", entity_type="RADAR_SYSTEM", chunk_rid="#13:0",
    )
    assert ece.entity_id is None
    assert ece.source_rid is None

    # With new fields populated:
    ece2 = EntityChunkEdge(
        entity_name="Tombstone", entity_type="RADAR_SYSTEM", chunk_rid="#13:0",
        entity_id="v1::RADAR_SYSTEM::system_name='Tombstone'",
        source_rid="#42:17",
    )
    assert ece2.entity_id == "v1::RADAR_SYSTEM::system_name='Tombstone'"
    assert ece2.source_rid == "#42:17"


def test_batch_writer_uses_direct_rid_to_rid_when_source_rid_populated():
    """source_rid set → direct CREATE EDGE FROM :src TO :rid with
    entity_id persisted as an edge property. No subquery."""
    from app.services.graph_store import EntityChunkEdge

    store = _make_store()
    edges = [EntityChunkEdge(
        entity_name="Tombstone", entity_type="RADAR_SYSTEM", chunk_rid="#13:0",
        entity_id="v1::RADAR_SYSTEM::system_name='Tombstone'",
        source_rid="#42:17",
    )]
    store.batch_create_entity_chunk_edges_sync(edges)

    assert store._client.command_sync.call_count == 1
    _db, _lang, sql, params = store._client.command_sync.call_args.args
    assert "FROM :src_0 TO :rid_0" in sql
    assert "entity_id = :eid_0" in sql
    # No LIMIT 1 subquery.
    assert "LIMIT 1" not in sql
    assert "WHERE name" not in sql
    assert params["src_0"] == "#42:17"
    assert params["rid_0"] == "#13:0"
    assert params["eid_0"] == "v1::RADAR_SYSTEM::system_name='Tombstone'"


def test_batch_writer_falls_back_to_subquery_when_source_rid_missing_with_warning(caplog):
    """source_rid=None → legacy path (subquery keyed on name+type LIMIT 1)
    with a WARNING so operators can see the migration pressure."""
    from app.services.graph_store import EntityChunkEdge

    store = _make_store()
    edges = [EntityChunkEdge(
        entity_name="Tombstone", entity_type="RADAR_SYSTEM", chunk_rid="#13:0",
    )]

    with caplog.at_level(logging.WARNING, logger="app.services.arcadedb_graph"):
        store.batch_create_entity_chunk_edges_sync(edges)

    _db, _lang, sql, params = store._client.command_sync.call_args.args
    assert "LIMIT 1" in sql
    assert "WHERE name = :name_0" in sql
    assert params["name_0"] == "Tombstone"
    assert any("legacy name+type subquery path" in r.message for r in caplog.records)


def test_batch_writer_handles_mixed_edges_direct_and_fallback():
    """Edges with source_rid use the direct path; edges without use the
    subquery path; both coexist in one sqlscript call."""
    from app.services.graph_store import EntityChunkEdge

    store = _make_store()
    edges = [
        EntityChunkEdge(
            entity_name="Tombstone", entity_type="RADAR_SYSTEM", chunk_rid="#13:0",
            entity_id="eid-1", source_rid="#42:17",
        ),
        EntityChunkEdge(
            entity_name="LegacyEntity", entity_type="RADAR_SYSTEM", chunk_rid="#13:1",
            # No entity_id / source_rid — legacy fallback
        ),
    ]

    store.batch_create_entity_chunk_edges_sync(edges)

    _db, _lang, sql, _params = store._client.command_sync.call_args.args
    assert "FROM :src_0 TO :rid_0" in sql
    assert "FROM (SELECT FROM RADAR_SYSTEM WHERE name = :name_1" in sql


def test_batch_writer_empty_edges_noops():
    from app.services.graph_store import EntityChunkEdge

    store = _make_store()
    assert store.batch_create_entity_chunk_edges_sync([]) == 0
    assert store._client.command_sync.call_count == 0


def test_derive_structure_links_suppresses_fallback_by_entity_id_not_name():
    """Regression test for the same-name same-type bug fix: the primary
    mention path adds an entity_id to mentioned_entity_ids; fallback
    only runs for nodes whose entity_id is NOT already in that set —
    distinguishing two SECTIONs with the same 'Overview' heading but
    different section_numbers."""
    import inspect
    from app.workers.pipeline import derive_structure_links

    source = inspect.getsource(derive_structure_links)
    assert "mentioned_entity_ids" in source
    assert "entity_ids_needing_fallback" in source
    # The old name-based suppression set must be gone
    assert "mentioned_entities: set[str]" not in source
