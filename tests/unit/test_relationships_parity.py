"""Parity tests: relationships.py mirrors ontology.yaml 1:1.

Gates Plan v32 Tasks 10 (RelationshipType Enum) and 11 (RelationshipMetadata
registry). Breaks when the two drift.
"""
from __future__ import annotations

import yaml
from pathlib import Path

import pytest

from ontology_bundles.air_defense_v3.relationships import (
    RelationshipType,
    RelationshipMetadata,
    RELATIONSHIP_METADATA,
)


@pytest.fixture(scope="module")
def yaml_ontology() -> dict:
    path = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "fixtures"
        / "ontology"
        / "air_defense_v3_snapshot.yaml"
    )
    with path.open() as f:
        return yaml.safe_load(f)


def test_relationship_type_enum_mirrors_yaml(yaml_ontology):
    yaml_names = {r["name"] for r in yaml_ontology["relationship_types"]}
    enum_names = {m.value for m in RelationshipType}
    missing_in_enum = yaml_names - enum_names
    extra_in_enum = enum_names - yaml_names
    assert not missing_in_enum, (
        f"RelationshipType missing YAML names: {sorted(missing_in_enum)}"
    )
    assert not extra_in_enum, (
        f"RelationshipType has names not in YAML: {sorted(extra_in_enum)}"
    )


def test_relationship_metadata_registry_covers_all_types():
    enum_members = set(RelationshipType)
    registry_keys = set(RELATIONSHIP_METADATA.keys())
    assert enum_members == registry_keys, (
        f"Missing: {enum_members - registry_keys}; "
        f"Extra: {registry_keys - enum_members}"
    )


def test_relationship_metadata_matches_yaml_fields(yaml_ontology):
    yaml_by_name = {r["name"]: r for r in yaml_ontology["relationship_types"]}
    for rt, meta in RELATIONSHIP_METADATA.items():
        y = yaml_by_name[rt.value]
        assert meta.label == y.get("label", rt.value), (
            f"{rt.value}: label drift. Pydantic={meta.label!r}, "
            f"YAML={y.get('label')!r}"
        )
        assert meta.description == y.get("description", ""), (
            f"{rt.value}: description drift."
        )
        assert meta.source_type == y.get("source_type"), (
            f"{rt.value}: source_type drift. Pydantic={meta.source_type!r}, "
            f"YAML={y.get('source_type')!r}"
        )
        assert meta.target_type == y.get("target_type"), (
            f"{rt.value}: target_type drift."
        )
        assert meta.cardinality == y.get("cardinality"), (
            f"{rt.value}: cardinality drift. Pydantic={meta.cardinality!r}, "
            f"YAML={y.get('cardinality')!r}"
        )


def test_relationship_metadata_is_frozen():
    sample = next(iter(RELATIONSHIP_METADATA.values()))
    with pytest.raises((TypeError, ValueError)):
        sample.label = "tampered"  # type: ignore[misc]
