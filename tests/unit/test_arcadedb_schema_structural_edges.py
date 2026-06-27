import asyncio

from unittest.mock import AsyncMock, MagicMock

from app.services.arcadedb_schema import _STRUCTURAL_EDGE_TYPES


def test_anchor_structural_edges_declared():
    """HAS_SECTION/HAS_FIGURE/HAS_TABLE/CHILD_OF must be declared."""
    for label in ("HAS_SECTION", "HAS_FIGURE", "HAS_TABLE", "CHILD_OF"):
        assert label in _STRUCTURAL_EDGE_TYPES, f"{label} missing from _STRUCTURAL_EDGE_TYPES"


def test_belongs_to_is_structural():
    """BELONGS_TO (Document -> Collection) is a structural edge — it carries no
    domain lineage and must be excluded from the gate's domain-edge partition."""
    assert "BELONGS_TO" in _STRUCTURAL_EDGE_TYPES


def _all_ddl() -> str:
    """Run ensure_schema against a mock client and concatenate every issued DDL
    script so individual CREATE statements can be asserted."""
    from app.services.arcadedb_schema import ensure_schema

    client = MagicMock()
    client.command = AsyncMock()
    asyncio.run(ensure_schema(client, database="test_db"))
    parts: list[str] = []
    for call in client.command.await_args_list:
        if call.args:
            parts.append(str(call.args[-1]))
    return "\n".join(parts)


def test_schema_emits_collection_vertex_type():
    assert "CREATE VERTEX TYPE Collection" in _all_ddl()


def test_schema_emits_belongs_to_edge_type():
    assert "CREATE EDGE TYPE BELONGS_TO" in _all_ddl()


def test_schema_emits_collection_unique_index_on_source_id():
    ddl = _all_ddl()
    # Mirror the Document(document_id) UNIQUE index shape.
    assert "Collection (source_id) UNIQUE" in ddl


def test_collection_is_not_an_entity_vertex_class():
    """Collection is structural infrastructure, NOT an ontology entity — it must
    never appear in the ontology entity_types vertex-class list."""
    from ontology_bundles.air_defense_v3.introspect import build_entity_types_list

    names = {e["name"] for e in build_entity_types_list()}
    assert "Collection" not in names
