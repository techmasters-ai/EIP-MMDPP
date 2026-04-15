"""Parity tests for ``introspect.build_entity_types_list`` vs ``ontology.yaml``.

Plan v32 Task 20 (Phase 3). The Pydantic introspection output must be
deep-equal to the legacy YAML ``entity_types`` list (per-entity dict
comparison). Downstream consumers read the same shape via
``load_ontology()``; any drift between sources would corrupt the
``ONTOLOGY_SOURCE`` feature flag contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ontology_bundles.air_defense_v3.introspect import build_entity_types_list

ONTOLOGY_YAML = (
    Path(__file__).resolve().parents[2]
    / "ontology_bundles"
    / "air_defense_v3"
    / "ontology.yaml"
)


@pytest.fixture(scope="module")
def yaml_entity_types() -> list[dict]:
    with ONTOLOGY_YAML.open() as f:
        return yaml.safe_load(f)["entity_types"]


@pytest.fixture(scope="module")
def pyd_entity_types() -> list[dict]:
    return build_entity_types_list()


def test_entity_count_matches(yaml_entity_types, pyd_entity_types):
    assert len(pyd_entity_types) == len(yaml_entity_types)


def test_entity_names_match(yaml_entity_types, pyd_entity_types):
    assert {e["name"] for e in pyd_entity_types} == {
        e["name"] for e in yaml_entity_types
    }


def test_per_entity_parity(yaml_entity_types, pyd_entity_types):
    pyd_by_name = {e["name"]: e for e in pyd_entity_types}
    for yml in yaml_entity_types:
        name = yml["name"]
        pyd = pyd_by_name[name]
        assert pyd == yml, (
            f"Parity mismatch for {name}.\n"
            f"YAML: {yml}\nPydantic: {pyd}"
        )


def test_typed_edge_fields_excluded_from_properties(pyd_entity_types):
    """Entities with typed edges (e.g. RADAR_SYSTEM) must not leak edge field
    names into ``properties``. The set of properties is strictly scalar."""
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity

    edge_field_names = {
        fname
        for fname, finfo in RadarSystemEntity.model_fields.items()
        if isinstance(finfo.json_schema_extra, dict)
        and finfo.json_schema_extra.get("edge_label")
    }
    assert edge_field_names, (
        "test fixture stale: RADAR_SYSTEM is expected to declare edges"
    )

    pyd_by_name = {e["name"]: e for e in pyd_entity_types}
    props = pyd_by_name["RADAR_SYSTEM"]["properties"]["properties"]
    overlap = edge_field_names & props.keys()
    assert not overlap, f"edges leaked into properties: {overlap}"


def test_system_confidence_excluded_from_non_assertion(pyd_entity_types):
    """The generator adds ``confidence`` as a system field to every
    ``is_entity=True`` class that doesn't already have one from YAML.
    Introspection must filter those out — otherwise YAML parity breaks."""
    pyd_by_name = {e["name"]: e for e in pyd_entity_types}
    document_props = pyd_by_name["DOCUMENT"]["properties"]["properties"]
    assert "confidence" not in document_props


def test_assertion_domain_confidence_preserved(pyd_entity_types):
    """ASSERTION declares ``confidence`` as a domain property in YAML
    (example 0.85). Introspection must emit that exact schema, not the
    generator's system-field override."""
    pyd_by_name = {e["name"]: e for e in pyd_entity_types}
    props = pyd_by_name["ASSERTION"]["properties"]["properties"]
    assert props["confidence"] == {
        "type": "number",
        "description": "Extraction confidence score (0.0 to 1.0)",
        "example": 0.85,
    }


def test_singular_example_mapping(pyd_entity_types):
    """Pydantic ``examples=[a, b, c]`` maps to YAML ``example: a`` (first only)."""
    pyd_by_name = {e["name"]: e for e in pyd_entity_types}
    section_heading = pyd_by_name["SECTION"]["properties"]["properties"]["heading"]
    assert section_heading["example"] == "Chapter 3: Maintenance Procedures"
    # The Pydantic field has 2 examples (identity field duplication); only the first surfaces.


def test_enum_preserved_in_declared_order(pyd_entity_types):
    """Enum values ride in the declared YAML order."""
    pyd_by_name = {e["name"]: e for e in pyd_entity_types}
    source_type = pyd_by_name["DOCUMENT"]["properties"]["properties"]["source_type"]
    assert source_type["enum"] == [
        "MANUAL",
        "REPORT",
        "BRIEFING",
        "NOTE",
        "SCHEMATIC",
        "DRAWING",
        "SPECIFICATION",
        "CHECKLIST",
    ]


def test_pattern_preserved(pyd_entity_types):
    pyd_by_name = {e["name"]: e for e in pyd_entity_types}
    part_number = pyd_by_name["COMPONENT"]["properties"]["properties"]["part_number"]
    assert part_number["pattern"] == r"^[A-Z0-9][A-Z0-9\-/]{2,20}$"


def test_no_description_elided_when_absent(pyd_entity_types):
    """When a Pydantic field has no description, the ``description`` key
    is omitted — not emitted as ``null``."""
    for entry in pyd_entity_types:
        props = entry["properties"]["properties"]
        for _, prop in props.items():
            if "description" in prop:
                assert prop["description"] is not None


def test_propulsion_stack_empty_identity_scope_document(pyd_entity_types):
    """PROPULSION_STACK has identity_fields=[] and identity_scope=document
    (both keys emitted even though identity_fields is empty)."""
    pyd_by_name = {e["name"]: e for e in pyd_entity_types}
    stack = pyd_by_name["PROPULSION_STACK"]
    assert stack["identity_fields"] == []
    assert stack["identity_scope"] == "document"


def test_document_omits_identity_keys(pyd_entity_types):
    """DOCUMENT has graph_id_fields=[] and identity_scope='global' →
    neither key present in YAML → neither key emitted."""
    pyd_by_name = {e["name"]: e for e in pyd_entity_types}
    document = pyd_by_name["DOCUMENT"]
    assert "identity_fields" not in document
    assert "identity_scope" not in document
