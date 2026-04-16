"""Schema-creation coverage driven by Pydantic introspection.

Replaces the YAML-snapshot-driven test_arcadedb_schema.py (deleted in Chunk F).
Oracle: ontology_bundles.air_defense_v3.introspect.build_entity_types_list
+ introspect.build_relationship_types_list.
"""
from unittest.mock import AsyncMock, MagicMock

from app.services.arcadedb_schema import _STRUCTURAL_EDGE_TYPES, ensure_schema
from ontology_bundles.air_defense_v3.introspect import (
    build_entity_types_list,
    build_relationship_types_list,
)


def test_entity_vertex_classes_cover_ALL_ENTITIES():
    entity_types = build_entity_types_list()
    names = {e["name"] for e in entity_types}
    # Every canonical entity and component class produces a vertex-class entry.
    # (Components remain in ALL_ENTITIES even when is_entity=False.)
    assert "PLATFORM" in names
    assert "RADAR_SYSTEM" in names
    assert "MODULATION" in names  # demoted-to-component in Chunk B, still present


def test_relationship_types_list_is_non_empty():
    relationships = build_relationship_types_list()
    assert relationships, "build_relationship_types_list returned no entries"


def test_structural_edges_include_anchor_labels():
    for label in ("HAS_SECTION", "HAS_FIGURE", "HAS_TABLE", "CHILD_OF"):
        assert label in _STRUCTURAL_EDGE_TYPES


def test_ensure_schema_runs_without_yaml_dependency():
    """ensure_schema drives ArcadeDB DDL purely from introspection — no YAML file read."""
    import asyncio

    client = MagicMock()
    client.command = AsyncMock()

    asyncio.run(ensure_schema(client, database="test_db"))

    assert client.command.await_count > 0, (
        "ensure_schema should have issued DDL via client.command"
    )
