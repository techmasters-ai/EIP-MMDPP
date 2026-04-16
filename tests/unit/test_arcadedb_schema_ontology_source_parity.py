"""Parity tests for ``arcadedb_schema`` / ``arcadedb_graph`` under both
``ONTOLOGY_SOURCE`` modes.

Plan v32 Task 26 (Phase 4, CRITICAL). Registers ArcadeDB vertex/edge
types. Without live ArcadeDB we cannot assert against a schema query,
but we can prove the ``registered-type set`` derived from the ontology
dict is identical across sources. If introspection has a gap the YAML
backend would emit a type/property the Pydantic backend wouldn't —
this test pins that contract.
"""
from __future__ import annotations

import pytest

from app.services.arcadedb_schema import _YAML_TO_ARCADE, _safe_type_name
from app.services.ontology_templates import invalidate_ontology_cache, load_ontology


@pytest.fixture(autouse=True)
def _flush_cache():
    invalidate_ontology_cache()
    yield
    invalidate_ontology_cache()


def _collect_schema_inputs(ontology: dict) -> dict:
    """Mirror the `arcadedb_schema.sync_ontology_schema` DDL-planning
    logic as a pure function — no client, no side effects."""
    entity_types = []
    entity_property_ddls = []
    fulltext_ddls = []
    upsert_ddls = []
    bucket_types = []
    for entity_def in ontology.get("entity_types", []):
        etype = _safe_type_name(entity_def["name"])
        entity_types.append(etype)
        props_schema = entity_def.get("properties", {}).get("properties", {})
        for prop_name, prop_def in sorted(props_schema.items()):
            yaml_type = prop_def.get("type", "string")
            arcade_type = _YAML_TO_ARCADE.get(yaml_type, "STRING")
            entity_property_ddls.append(f"{etype}.{prop_name}:{arcade_type}")
        fulltext_ddls.append(f"{etype}(name)")
        upsert_ddls.append(f"{etype}(name,entity_type)")
        bucket_types.append(etype)

    rel_types = []
    for rel_def in ontology.get("relationship_types", []):
        rel_types.append(rel_def["name"])

    return {
        "entity_types": sorted(entity_types),
        "entity_property_ddls": sorted(entity_property_ddls),
        "fulltext_ddls": sorted(fulltext_ddls),
        "upsert_ddls": sorted(upsert_ddls),
        "bucket_types": sorted(bucket_types),
        "rel_types": sorted(rel_types),
    }


def test_schema_ddl_inputs_parity(monkeypatch):
    """Same set of registered types and properties under both sources."""
    monkeypatch.setenv("ONTOLOGY_SOURCE", "yaml")
    yaml_ontology = load_ontology()
    monkeypatch.setenv("ONTOLOGY_SOURCE", "pydantic")
    pyd_ontology = load_ontology()

    yaml_inputs = _collect_schema_inputs(yaml_ontology)
    pyd_inputs = _collect_schema_inputs(pyd_ontology)

    assert pyd_inputs == yaml_inputs


def test_arcadedb_graph_entity_type_list_parity(monkeypatch):
    """arcadedb_graph reads `e["name"]` from entity_types; the list must be
    the same set across modes. Relationship types must also match
    (tested by name-set equality)."""
    monkeypatch.setenv("ONTOLOGY_SOURCE", "yaml")
    yaml_ont = load_ontology()
    monkeypatch.setenv("ONTOLOGY_SOURCE", "pydantic")
    pyd_ont = load_ontology()

    yaml_entity_names = {e["name"] for e in yaml_ont["entity_types"]}
    pyd_entity_names = {e["name"] for e in pyd_ont["entity_types"]}
    assert yaml_entity_names == pyd_entity_names

    yaml_rel_names = {r["name"] for r in yaml_ont["relationship_types"]}
    pyd_rel_names = {r["name"] for r in pyd_ont["relationship_types"]}
    assert yaml_rel_names == pyd_rel_names


def test_validation_matrix_parity_for_graph_validator(monkeypatch):
    """arcadedb_graph's triple validator loads validation_matrix. Both
    sources must produce the same triple set (dedup already handled by
    the canonicalizer)."""
    monkeypatch.setenv("ONTOLOGY_SOURCE", "yaml")
    yaml_ont = load_ontology()
    monkeypatch.setenv("ONTOLOGY_SOURCE", "pydantic")
    pyd_ont = load_ontology()

    yaml_triples = {
        (e["source"], e["relationship"], e["target"])
        for e in yaml_ont["validation_matrix"]
    }
    pyd_triples = {
        (e["source"], e["relationship"], e["target"])
        for e in pyd_ont["validation_matrix"]
    }
    assert yaml_triples == pyd_triples
