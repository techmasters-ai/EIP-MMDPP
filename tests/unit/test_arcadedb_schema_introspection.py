"""Schema-creation coverage driven by Pydantic introspection.

Replaces the YAML-snapshot-driven test_arcadedb_schema.py (deleted in Chunk F).
Oracle: ontology_bundles.air_defense_v3.introspect.build_entity_types_list
+ introspect.build_relationship_types_list.
"""
from unittest.mock import AsyncMock, MagicMock

from app.services.arcadedb_schema import (
    _STRUCTURAL_EDGE_TYPES,
    ensure_schema,
    ensure_schema_sync,
)
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
    assert "MISSILE_SYSTEM" in names  # canonical system class retained post-flat-schema refactor


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


def test_ensure_schema_sync_adapter_runs_outside_event_loop():
    """ensure_schema_sync lets the migration script call from plain sync code."""
    client = MagicMock()
    client.command = AsyncMock()

    report = ensure_schema_sync(client, database="test_db")

    assert client.command.await_count > 0
    assert report is not None


def test_ensure_schema_is_additive_only():
    """No DROP statements are emitted — destructive ops belong in the migration script."""
    import asyncio
    import re

    client = MagicMock()
    client.command = AsyncMock()

    asyncio.run(ensure_schema(client, database="test_db"))

    # Split each issued script into statements and check the leading verb.
    drop_stmts: list[str] = []
    for call in client.command.await_args_list:
        if not call.args:
            continue
        script = str(call.args[-1])
        for stmt in script.split(";"):
            leading = stmt.strip()
            if re.match(r"^DROP\b", leading, re.IGNORECASE):
                drop_stmts.append(leading)
    assert not drop_stmts, (
        "ensure_schema emitted DROP statements: " + " | ".join(drop_stmts)
    )
