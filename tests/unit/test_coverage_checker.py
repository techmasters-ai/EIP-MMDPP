"""Unit tests for tools/extraction_coverage — one rule at a time.

Each test builds synthetic in-memory dicts and calls the matching rule
function directly.  Integration is exercised only by
test_check_bundle_passes_on_real_air_defense_v3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest
from pydantic import BaseModel, Field

from tools.extraction_coverage.rules import (
    SCHEMA_SIZE_THRESHOLD_CHARS,
    _check_bridge_scope_consistency,
    _check_coverage_subset,
    _check_display_label_present,
    _check_empty_identity_global_warning,
    _check_extraction_subset_of_ontology,
    _check_identity_fields_completeness,
    _check_identity_scope_required,
    _check_manifest_entities_in_coverage,
    _check_manifest_relationships_in_coverage,
    _check_recursive_partial_safety,
    _check_relationships_in_validation_matrix,
    _check_schema_size,
    check_bundle,
)
from tools.extraction_coverage.manifest_consistency import check_manifest_self_consistency

# ---------------------------------------------------------------------------
# Minimal synthetic fixtures
# ---------------------------------------------------------------------------

def _make_ontology(entities=None, relationships=None, validation_matrix=None):
    entities = entities or []
    return {
        "entity_types": entities,
        "relationship_types": relationships or [],
        "validation_matrix": validation_matrix or [],
    }


def _make_entity(name, identity_fields=None, identity_scope=None, properties=None):
    e: dict = {"name": name}
    if identity_fields is not None:
        e["identity_fields"] = identity_fields
    if identity_scope is not None:
        e["identity_scope"] = identity_scope
    if properties is not None:
        e["properties"] = properties
    return e


def _make_manifest(passes=None):
    return {"passes": passes or []}


def _make_pass(
    name="pass1",
    primary=None,
    bridge=None,
    rels=None,
    depends_on=None,
    input_mode="document_only",
    module="extraction_schemas.radar_domain",
    template_class="RadarDomainPass",
):
    return {
        "name": name,
        "primary_entity_types": primary or [],
        "bridge_entity_types": bridge or [],
        "extracted_relationship_types": rels or [],
        "depends_on": depends_on or [],
        "input_mode": input_mode,
        "module": module,
        "template_class": template_class,
    }


def _make_coverage(extract=None, derive=None, rel_extract=None):
    return {
        "entity_types": {
            "extract": extract or [],
            "derive": derive or [],
        },
        "relationship_types": {
            "extract": rel_extract or [],
        },
    }


# ---------------------------------------------------------------------------
# Rule 1 — coverage subset of ontology
# ---------------------------------------------------------------------------

def test_rule1_missing_coverage_type_fails():
    ontology = _make_ontology(entities=[_make_entity("RADAR_SYSTEM")])
    coverage = _make_coverage(extract=["RADAR_SYSTEM", "GHOST_TYPE"])
    errors = _check_coverage_subset(ontology, coverage)
    assert any("GHOST_TYPE" in e for e in errors)


def test_rule1_happy_path():
    ontology = _make_ontology(entities=[_make_entity("RADAR_SYSTEM"), _make_entity("PLATFORM")])
    coverage = _make_coverage(extract=["RADAR_SYSTEM"], derive=["PLATFORM"])
    errors = _check_coverage_subset(ontology, coverage)
    assert errors == []


def test_rule1_derive_also_checked():
    ontology = _make_ontology(entities=[_make_entity("RADAR_SYSTEM")])
    coverage = _make_coverage(derive=["DERIVED_ONLY_TYPE"])
    errors = _check_coverage_subset(ontology, coverage)
    assert any("DERIVED_ONLY_TYPE" in e for e in errors)


# ---------------------------------------------------------------------------
# Rule 2 — manifest entity_types in coverage.extract
# ---------------------------------------------------------------------------

def test_rule2_manifest_entity_missing_from_coverage_fails():
    coverage = _make_coverage(extract=["RADAR_SYSTEM"])
    manifest = _make_manifest(passes=[
        _make_pass(primary=["RADAR_SYSTEM", "MISSING_TYPE"])
    ])
    errors = _check_manifest_entities_in_coverage(manifest, coverage)
    assert any("MISSING_TYPE" in e for e in errors)


def test_rule2_bridge_entity_also_checked():
    coverage = _make_coverage(extract=["RADAR_SYSTEM"])
    manifest = _make_manifest(passes=[
        _make_pass(primary=["RADAR_SYSTEM"], bridge=["BRIDGE_NOT_IN_COVERAGE"])
    ])
    errors = _check_manifest_entities_in_coverage(manifest, coverage)
    assert any("BRIDGE_NOT_IN_COVERAGE" in e for e in errors)


def test_rule2_happy_path():
    coverage = _make_coverage(extract=["RADAR_SYSTEM", "PLATFORM"])
    manifest = _make_manifest(passes=[
        _make_pass(primary=["RADAR_SYSTEM"], bridge=["PLATFORM"])
    ])
    errors = _check_manifest_entities_in_coverage(manifest, coverage)
    assert errors == []


# ---------------------------------------------------------------------------
# Rule 3 — manifest relationship_types in coverage.extract
# ---------------------------------------------------------------------------

def test_rule3_manifest_relationship_missing_from_coverage_fails():
    coverage = _make_coverage(rel_extract=["INSTALLED_ON"])
    manifest = _make_manifest(passes=[
        _make_pass(rels=["INSTALLED_ON", "GHOST_REL"])
    ])
    errors = _check_manifest_relationships_in_coverage(manifest, coverage)
    assert any("GHOST_REL" in e for e in errors)


def test_rule3_happy_path():
    coverage = _make_coverage(rel_extract=["INSTALLED_ON", "HAS_ANTENNA"])
    manifest = _make_manifest(passes=[
        _make_pass(rels=["INSTALLED_ON"])
    ])
    errors = _check_manifest_relationships_in_coverage(manifest, coverage)
    assert errors == []


# ---------------------------------------------------------------------------
# Rule 4 — extracted relationships have validation_matrix rows
# ---------------------------------------------------------------------------

def test_rule4_relationship_not_in_validation_matrix_fails():
    ontology = _make_ontology(
        validation_matrix=[{"relationship": "INSTALLED_ON"}],
    )
    manifest = _make_manifest(passes=[
        _make_pass(rels=["INSTALLED_ON", "UNMAPPED_REL"])
    ])
    errors = _check_relationships_in_validation_matrix(manifest, ontology)
    assert any("UNMAPPED_REL" in e for e in errors)
    assert not any("INSTALLED_ON" in e for e in errors)


def test_rule4_happy_path():
    ontology = _make_ontology(
        validation_matrix=[
            {"relationship": "INSTALLED_ON"},
            {"relationship": "HAS_ANTENNA"},
        ],
    )
    manifest = _make_manifest(passes=[
        _make_pass(rels=["INSTALLED_ON", "HAS_ANTENNA"])
    ])
    errors = _check_relationships_in_validation_matrix(manifest, ontology)
    assert errors == []


# ---------------------------------------------------------------------------
# Rule 5 — schema size threshold
# ---------------------------------------------------------------------------

class _SmallModel(BaseModel):
    x: Optional[str] = None


class _HugeModel(BaseModel):
    pass


def test_rule5_schema_size_threshold_enforced():
    # Build a model whose JSON schema exceeds the threshold by padding field names.
    import json as _json

    # Craft a model dynamically with many long field names to blow the threshold.
    field_defs = {
        f"field_{'a' * 50}_{i}": (Optional[str], None)
        for i in range(300)
    }
    from pydantic import create_model
    BigModel = create_model("BigModel", **field_defs)  # type: ignore[call-overload]

    schema_str = _json.dumps(BigModel.model_json_schema())
    # Only run this test if the schema is actually large
    if len(schema_str) > SCHEMA_SIZE_THRESHOLD_CHARS:
        errors = _check_schema_size(BigModel)
        assert errors, "Expected an error for oversized schema"
        assert "BigModel" in errors[0]
    else:
        pytest.skip(
            f"Could not synthesise a schema larger than {SCHEMA_SIZE_THRESHOLD_CHARS} chars"
        )


def test_rule5_small_schema_passes():
    errors = _check_schema_size(_SmallModel)
    assert errors == []


# ---------------------------------------------------------------------------
# Rule 6 — recursive partial-safety
# ---------------------------------------------------------------------------

class _OptionalInner(BaseModel):
    val: Optional[str] = None


class _OptionalOuter(BaseModel):
    inner: Optional[_OptionalInner] = None
    items: list[_OptionalInner] = Field(default_factory=list)


class _RequiredField(BaseModel):
    required_name: str  # no default — should fail


class _RequiredInner(BaseModel):
    bad: str  # no default


class _OuterWithBadInner(BaseModel):
    inner: Optional[_RequiredInner] = None


def test_rule6_required_field_fails():
    errors = _check_recursive_partial_safety(_RequiredField)
    assert any("required_name" in e for e in errors)


def test_rule6_required_inner_field_fails():
    errors = _check_recursive_partial_safety(_OuterWithBadInner)
    assert any("bad" in e for e in errors)


def test_rule6_optional_fields_pass():
    errors = _check_recursive_partial_safety(_OptionalOuter)
    assert errors == []


# ---------------------------------------------------------------------------
# Rule 8 — extraction field names ⊆ ontology entity properties
# ---------------------------------------------------------------------------

def test_rule8_extra_field_not_in_ontology_fails():
    # Build an ontology with RADAR_SYSTEM having only "system_name"
    ontology = _make_ontology(entities=[
        _make_entity(
            "RADAR_SYSTEM",
            properties={"type": "object", "properties": {"system_name": {}}},
        ),
    ])

    # Build a template class containing RadarSystemEntity with an extra field
    class RadarSystemEntity(BaseModel):
        system_name: Optional[str] = None
        alien_field: Optional[str] = None  # not in ontology

    class RadarDomainPass(BaseModel):
        radar_systems: list[RadarSystemEntity] = Field(default_factory=list)

    errors = _check_extraction_subset_of_ontology(RadarDomainPass, ontology)
    assert any("alien_field" in e for e in errors)


def test_rule8_all_fields_in_ontology_passes():
    ontology = _make_ontology(entities=[
        _make_entity(
            "RADAR_SYSTEM",
            properties={"type": "object", "properties": {"system_name": {}, "nomenclature": {}}},
        ),
    ])

    class RadarSystemEntity(BaseModel):
        system_name: Optional[str] = None
        nomenclature: Optional[str] = None
        confidence: Optional[float] = None  # system field — exempt

    class RadarDomainPass(BaseModel):
        radar_systems: list[RadarSystemEntity] = Field(default_factory=list)

    errors = _check_extraction_subset_of_ontology(RadarDomainPass, ontology)
    assert errors == []


def test_rule8_relationship_class_skipped():
    """Classes with 'Relationship' or 'Link' in the name must not be checked."""
    ontology = _make_ontology()

    class SomeRelationship(BaseModel):
        from_type: Optional[str] = None
        to_type: Optional[str] = None
        ghost_field: Optional[str] = None  # would fail if checked

    class TopLevel(BaseModel):
        rels: list[SomeRelationship] = Field(default_factory=list)

    errors = _check_extraction_subset_of_ontology(TopLevel, ontology)
    assert errors == []


# ---------------------------------------------------------------------------
# Rule 9 — identity_fields declared on every extract-bucket entity
# ---------------------------------------------------------------------------

def test_rule9_missing_identity_fields_fails():
    ontology = _make_ontology(entities=[
        _make_entity("RADAR_SYSTEM"),  # no identity_fields key
    ])
    coverage = _make_coverage(extract=["RADAR_SYSTEM"])
    errors = _check_identity_fields_completeness(ontology, coverage)
    assert any("RADAR_SYSTEM" in e for e in errors)
    assert any("identity_fields" in e for e in errors)


def test_rule9_empty_identity_fields_is_ok():
    """Empty list is valid (content-hash fallback)."""
    ontology = _make_ontology(entities=[
        _make_entity("PROPULSION_STACK", identity_fields=[]),
    ])
    coverage = _make_coverage(extract=["PROPULSION_STACK"])
    errors = _check_identity_fields_completeness(ontology, coverage)
    assert errors == []


def test_rule9_has_identity_fields_passes():
    ontology = _make_ontology(entities=[
        _make_entity("RADAR_SYSTEM", identity_fields=["system_name"]),
    ])
    coverage = _make_coverage(extract=["RADAR_SYSTEM"])
    errors = _check_identity_fields_completeness(ontology, coverage)
    assert errors == []


# ---------------------------------------------------------------------------
# Rule 11 — identity_scope ∈ {"document", "global"}
# ---------------------------------------------------------------------------

def test_rule11_unknown_identity_scope_fails():
    ontology = _make_ontology(entities=[
        _make_entity("RADAR_SYSTEM", identity_fields=["system_name"], identity_scope="cluster"),
    ])
    coverage = _make_coverage(extract=["RADAR_SYSTEM"])
    errors = _check_identity_scope_required(ontology, coverage)
    assert any("RADAR_SYSTEM" in e for e in errors)
    assert any("cluster" in e for e in errors)


def test_rule11_missing_identity_scope_fails():
    ontology = _make_ontology(entities=[
        _make_entity("RADAR_SYSTEM", identity_fields=["system_name"]),
        # no identity_scope key → scope is None → invalid
    ])
    coverage = _make_coverage(extract=["RADAR_SYSTEM"])
    errors = _check_identity_scope_required(ontology, coverage)
    assert errors  # None is not in {"document", "global"}


def test_rule11_valid_scopes_pass():
    ontology = _make_ontology(entities=[
        _make_entity("RADAR_SYSTEM", identity_fields=["system_name"], identity_scope="global"),
        _make_entity("WAVEFORM", identity_fields=["waveform_name"], identity_scope="document"),
    ])
    coverage = _make_coverage(extract=["RADAR_SYSTEM", "WAVEFORM"])
    errors = _check_identity_scope_required(ontology, coverage)
    assert errors == []


# ---------------------------------------------------------------------------
# Rule 12 — [WARN] identity_fields=[] + identity_scope=global
# ---------------------------------------------------------------------------

def test_rule12_empty_identity_global_returns_warning():
    ontology = _make_ontology(entities=[
        _make_entity("PROPULSION_STACK", identity_fields=[], identity_scope="global"),
    ])
    coverage = _make_coverage(extract=["PROPULSION_STACK"])
    warnings = _check_empty_identity_global_warning(ontology, coverage)
    assert warnings, "Expected a warning for identity_fields=[] + identity_scope=global"
    assert any("PROPULSION_STACK" in w for w in warnings)


def test_rule12_no_warning_when_document_scope():
    ontology = _make_ontology(entities=[
        _make_entity("PROPULSION_STACK", identity_fields=[], identity_scope="document"),
    ])
    coverage = _make_coverage(extract=["PROPULSION_STACK"])
    warnings = _check_empty_identity_global_warning(ontology, coverage)
    assert warnings == []


def test_rule12_no_warning_when_nonempty_fields():
    ontology = _make_ontology(entities=[
        _make_entity("RADAR_SYSTEM", identity_fields=["system_name"], identity_scope="global"),
    ])
    coverage = _make_coverage(extract=["RADAR_SYSTEM"])
    warnings = _check_empty_identity_global_warning(ontology, coverage)
    assert warnings == []


# ---------------------------------------------------------------------------
# Rule 13 — bridge entities have exactly one ontology declaration
# ---------------------------------------------------------------------------

def test_rule13_bridge_entity_missing_fails():
    ontology = _make_ontology(entities=[])  # no entries
    coverage = _make_coverage(extract=["PLATFORM"])
    manifest = _make_manifest(passes=[
        _make_pass(bridge=["PLATFORM"])
    ])
    errors = _check_bridge_scope_consistency(ontology, coverage, manifest)
    assert any("PLATFORM" in e for e in errors)
    assert any("not declared" in e for e in errors)


def test_rule13_duplicate_bridge_declaration_fails():
    ontology = _make_ontology(entities=[
        _make_entity("PLATFORM", identity_fields=["name"], identity_scope="global"),
        _make_entity("PLATFORM", identity_fields=["name"], identity_scope="document"),
    ])
    coverage = _make_coverage(extract=["PLATFORM"])
    manifest = _make_manifest(passes=[
        _make_pass(bridge=["PLATFORM"])
    ])
    errors = _check_bridge_scope_consistency(ontology, coverage, manifest)
    assert any("PLATFORM" in e for e in errors)
    assert any("2 declarations" in e for e in errors)


def test_rule13_happy_path():
    ontology = _make_ontology(entities=[
        _make_entity("PLATFORM", identity_fields=["name"], identity_scope="global"),
    ])
    coverage = _make_coverage(extract=["PLATFORM"])
    manifest = _make_manifest(passes=[
        _make_pass(bridge=["PLATFORM"])
    ])
    errors = _check_bridge_scope_consistency(ontology, coverage, manifest)
    assert errors == []


# ---------------------------------------------------------------------------
# Manifest self-consistency
# ---------------------------------------------------------------------------

def test_manifest_self_consistency_duplicate_pass_name_fails():
    manifest = _make_manifest(passes=[
        _make_pass(name="pass1"),
        _make_pass(name="pass1"),  # duplicate
    ])
    coverage = _make_coverage()
    errors = check_manifest_self_consistency(manifest, coverage, bundle_key="test_bundle")
    assert any("duplicate" in e for e in errors)
    assert any("pass1" in e for e in errors)


def test_manifest_self_consistency_depends_on_later_pass_fails():
    manifest = _make_manifest(passes=[
        _make_pass(name="pass2", depends_on=["pass1"]),  # pass1 not yet seen
        _make_pass(name="pass1"),
    ])
    coverage = _make_coverage()
    errors = check_manifest_self_consistency(manifest, coverage, bundle_key="test_bundle")
    assert any("depends_on" in e for e in errors)


def test_manifest_self_consistency_disjoint_entities_enforced():
    manifest = _make_manifest(passes=[
        _make_pass(name="pass1", primary=["RADAR_SYSTEM"]),
        _make_pass(name="pass2", primary=["RADAR_SYSTEM"]),  # same primary, not a bridge
    ])
    coverage = _make_coverage()
    errors = check_manifest_self_consistency(manifest, coverage, bundle_key="test_bundle")
    assert any("RADAR_SYSTEM" in e and "multiple passes" in e for e in errors)


def test_manifest_self_consistency_bridge_entity_allowed_in_multiple_passes():
    manifest = _make_manifest(passes=[
        _make_pass(name="pass1", bridge=["PLATFORM"]),
        _make_pass(name="pass2", bridge=["PLATFORM"]),  # bridge → OK
    ])
    coverage = _make_coverage()
    errors = check_manifest_self_consistency(manifest, coverage, bundle_key="test_bundle")
    # No "multiple passes" error about PLATFORM
    assert not any("PLATFORM" in e and "multiple passes" in e for e in errors)


def test_manifest_self_consistency_document_plus_entity_refs_needs_depends_on_fails():
    manifest = _make_manifest(passes=[
        _make_pass(name="pass1", input_mode="document_plus_entity_refs", depends_on=[]),
    ])
    coverage = _make_coverage()
    errors = check_manifest_self_consistency(manifest, coverage, bundle_key="test_bundle")
    assert any("document_plus_entity_refs" in e for e in errors)


def test_manifest_self_consistency_happy_path():
    manifest = _make_manifest(passes=[
        _make_pass(name="pass1", primary=["RADAR_SYSTEM"]),
        _make_pass(name="pass2", primary=["MISSILE_SYSTEM"], depends_on=["pass1"],
                   input_mode="document_plus_entity_refs"),
    ])
    coverage = _make_coverage()
    # Only importability check can fail here (we use a real module path below)
    # We skip the importability check by using the real air_defense_v3 modules
    # but here we just check the structural rules don't flag anything structural.
    errors_structural = [
        e for e in check_manifest_self_consistency(manifest, coverage, bundle_key="test_bundle")
        if "failed to import" not in e and "has no class" not in e and "missing module" not in e
    ]
    assert errors_structural == []


# ---------------------------------------------------------------------------
# Integration: real air_defense_v3 bundle must pass all rules
# ---------------------------------------------------------------------------

def test_check_bundle_passes_on_real_air_defense_v3():
    repo_root = Path(__file__).resolve().parent.parent.parent
    bundle_path = repo_root / "ontology_bundles" / "air_defense_v3"
    assert bundle_path.exists(), f"Bundle not found: {bundle_path}"
    errors, warnings = check_bundle(bundle_path)
    # Print for debugging if there are failures
    for e in errors:
        print(f"ERROR: {e}")
    for w in warnings:
        print(f"WARN:  {w}")
    assert errors == [], f"air_defense_v3 bundle failed {len(errors)} rule(s):\n" + "\n".join(errors)
