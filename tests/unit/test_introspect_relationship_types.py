"""Parity tests for ``introspect.build_relationship_types_list`` vs ``ontology.yaml``.

Plan v32 Task 21 (Phase 3). The Pydantic introspection output must be
deep-equal to the legacy YAML ``relationship_types`` list (per-entry
dict comparison). ``cardinality`` is omitted when unset — YAML elides
the key for IS_A / INSTANCE_OF / ALIAS_OF; Pydantic mirrors that.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ontology_bundles.air_defense_v3.introspect import build_relationship_types_list

ONTOLOGY_YAML = (
    Path(__file__).resolve().parents[2]
    / "ontology_bundles"
    / "air_defense_v3"
    / "ontology.yaml"
)


@pytest.fixture(scope="module")
def yaml_relationship_types() -> list[dict]:
    with ONTOLOGY_YAML.open() as f:
        return yaml.safe_load(f)["relationship_types"]


@pytest.fixture(scope="module")
def pyd_relationship_types() -> list[dict]:
    return build_relationship_types_list()


def test_relationship_count_matches(yaml_relationship_types, pyd_relationship_types):
    assert len(pyd_relationship_types) == len(yaml_relationship_types)


def test_relationship_names_match(yaml_relationship_types, pyd_relationship_types):
    assert {e["name"] for e in pyd_relationship_types} == {
        e["name"] for e in yaml_relationship_types
    }


def test_per_relationship_parity(yaml_relationship_types, pyd_relationship_types):
    pyd_by_name = {e["name"]: e for e in pyd_relationship_types}
    for yml in yaml_relationship_types:
        name = yml["name"]
        pyd = pyd_by_name[name]
        assert pyd == yml, (
            f"Parity mismatch for {name}.\nYAML: {yml}\nPydantic: {pyd}"
        )


def test_cardinality_elided_when_absent(pyd_relationship_types):
    """IS_A / INSTANCE_OF / ALIAS_OF have no cardinality in YAML; the key
    must be absent from the introspected dict (not None)."""
    pyd_by_name = {e["name"]: e for e in pyd_relationship_types}
    for elided in ("IS_A", "INSTANCE_OF", "ALIAS_OF"):
        assert "cardinality" not in pyd_by_name[elided], (
            f"{elided} must omit the cardinality key"
        )


def test_cardinality_emitted_when_present(pyd_relationship_types):
    pyd_by_name = {e["name"]: e for e in pyd_relationship_types}
    assert pyd_by_name["PART_OF"]["cardinality"] == "many_to_one"
    assert pyd_by_name["CONTAINS"]["cardinality"] == "one_to_many"


def test_null_source_target_emitted_as_none(pyd_relationship_types):
    """YAML ``source_type: null`` round-trips to Python None; introspection
    emits None for source_type/target_type (always present, sometimes None)."""
    pyd_by_name = {e["name"]: e for e in pyd_relationship_types}
    is_a = pyd_by_name["IS_A"]
    assert is_a["source_type"] is None
    assert is_a["target_type"] is None


def test_scoped_source_target_types_preserved(pyd_relationship_types):
    """HAS_STAGE has both source_type and target_type set."""
    pyd_by_name = {e["name"]: e for e in pyd_relationship_types}
    has_stage = pyd_by_name["HAS_STAGE"]
    assert has_stage["source_type"] == "PROPULSION_STACK"
    assert has_stage["target_type"] == "PROPULSION_STAGE"
