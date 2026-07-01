"""Task 2 (case-insensitive-entity-identity): schema DDL for normalized
``<field>_key`` properties + UNIQUE indexes.

Pure unit tests over ``sync_schema_from_ontology`` with a minimal, hand-built
ontology dict (no live DB, no full air_defense_v3 bundle) — mirrors the
mock-client pattern used by ``tests/unit/test_arcadedb_schema_structural_edges.py``.

Oracle for the property/index naming convention: ``_key_fields`` in
``app/services/arcadedb_graph.py`` (Task 1's write layer), which persists
``{f"{k}_key": norm(v)}`` for every identity field except ``document_id``.
This test asserts the schema builder (Task 2) declares that exact column
name and indexes on it.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.services.arcadedb_schema import sync_schema_from_ontology

_ONTOLOGY: dict = {
    "entity_types": [
        # Global-scoped domain type, single identity field.
        {
            "name": "RADAR_SYSTEM",
            "identity_fields": ["system_name"],
            "identity_scope": "global",
            "properties": {"properties": {"system_name": {"type": "string"}}},
        },
        # Document-scoped domain type — document_id is appended by the
        # builder itself (NOT listed in identity_fields), matching the real
        # air_defense_v3 SECTION/FIGURE/TABLE pattern.
        {
            "name": "SECTION",
            "identity_fields": ["section_number"],
            "identity_scope": "document",
            "properties": {"properties": {"section_number": {"type": "string"}}},
        },
        # Component/content class — no identity fields, is_entity=False in
        # the real bundle. Introspection omits "identity_fields" entirely
        # for these (see introspect.py:_build_entity_entry).
        {
            "name": "COMPONENT_LABEL",
            "properties": {"properties": {"label": {"type": "string"}}},
        },
    ],
    "relationship_types": [],
}


def _run_sync_and_collect_ddl() -> str:
    client = MagicMock()
    client.command = AsyncMock()
    asyncio.run(sync_schema_from_ontology(client, database="test_db", ontology=_ONTOLOGY))
    parts: list[str] = []
    for call in client.command.await_args_list:
        if call.args:
            parts.append(str(call.args[-1]))
    return "\n".join(parts)


def test_global_type_emits_key_property_and_unique_index():
    ddl = _run_sync_and_collect_ddl()
    assert "CREATE PROPERTY RADAR_SYSTEM.system_name_key IF NOT EXISTS STRING" in ddl
    assert "ON RADAR_SYSTEM (system_name_key, entity_type) UNIQUE" in ddl


def test_document_scoped_type_emits_key_property_and_unique_index_with_document_id():
    ddl = _run_sync_and_collect_ddl()
    assert "CREATE PROPERTY SECTION.section_number_key IF NOT EXISTS STRING" in ddl
    assert "ON SECTION (section_number_key, document_id, entity_type) UNIQUE" in ddl


def test_key_property_ddl_precedes_key_index_ddl():
    ddl = _run_sync_and_collect_ddl()
    prop_pos = ddl.index("CREATE PROPERTY RADAR_SYSTEM.system_name_key")
    index_pos = ddl.index("ON RADAR_SYSTEM (system_name_key, entity_type) UNIQUE")
    assert prop_pos < index_pos, (
        "CREATE PROPERTY for <field>_key must be emitted before the "
        "CREATE INDEX that references it"
    )

    doc_prop_pos = ddl.index("CREATE PROPERTY SECTION.section_number_key")
    doc_index_pos = ddl.index("ON SECTION (section_number_key, document_id, entity_type) UNIQUE")
    assert doc_prop_pos < doc_index_pos


def test_raw_field_unique_index_is_kept_alongside_key_index():
    """The pre-existing raw-field UNIQUE index must NOT be removed — it's
    additive, not a replacement (see arcadedb_schema.py Phase 6b comment)."""
    ddl = _run_sync_and_collect_ddl()
    assert "ON RADAR_SYSTEM (system_name, entity_type) UNIQUE" in ddl
    assert "ON SECTION (section_number, document_id, entity_type) UNIQUE" in ddl


def test_component_class_has_no_key_property_or_index():
    ddl = _run_sync_and_collect_ddl()
    assert "COMPONENT_LABEL.label_key" not in ddl
    assert "_key" not in "\n".join(
        line for line in ddl.splitlines() if "COMPONENT_LABEL" in line
    )
