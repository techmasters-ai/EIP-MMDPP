"""Tests for json_schema_extra constructors used by canonical entities.

Exercises the four-bucket convention from the flat-schema profile
refactor spec §3.3: every domain field on RadarSystemEntity /
MissileSystemEntity falls into exactly one of profile-mapped,
system_metadata, identity, or system_field.
"""
from pydantic import BaseModel
from ontology_bundles.air_defense_v3.entities import (
    profile_field, metadata_field, identity_field,
)


def test_profile_field_sets_profile_sections_and_subgroup():
    class M(BaseModel):
        gain_dbi: float | None = profile_field(
            sections=["rf_parameters", "performance"],
            subgroup="transmit",
            description="Antenna gain in dBi.",
            examples=[35.0],
            default=None,
        )

    info = M.model_fields["gain_dbi"]
    extra = info.json_schema_extra or {}
    assert extra["profile_sections"] == ["rf_parameters", "performance"]
    assert extra["profile_subgroup"] == "transmit"
    assert info.description == "Antenna gain in dBi."
    assert info.examples == [35.0]


def test_metadata_field_marks_system_metadata():
    class M(BaseModel):
        responsible_agency: str | None = metadata_field(
            description="Agency that owns the parametric record.",
            default=None,
        )

    info = M.model_fields["responsible_agency"]
    extra = info.json_schema_extra or {}
    assert extra["profile_sections"] == []
    assert extra["system_metadata"] is True


def test_identity_field_marks_identity():
    class M(BaseModel):
        nomenclature: str | None = identity_field(
            description="Formal designator (e.g. 5N63S).",
            default=None,
        )

    info = M.model_fields["nomenclature"]
    extra = info.json_schema_extra or {}
    assert extra["identity_field"] is True
    assert extra["profile_sections"] == []


def test_radar_system_entity_every_field_is_bucketed():
    """Every domain field on canonical RadarSystemEntity falls into
    profile-mapped / system_metadata / identity / system_field
    (spec §3.3 four-bucket contract)."""
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity

    graph_id_fields = set(
        RadarSystemEntity.model_config.get("graph_id_fields", []) or []
    )

    misclassified = []
    for fname, finfo in RadarSystemEntity.model_fields.items():
        if fname in graph_id_fields:
            continue
        extra = finfo.json_schema_extra or {}
        if not isinstance(extra, dict):
            extra = {}
        is_profile = bool(extra.get("profile_sections"))
        is_metadata = extra.get("system_metadata") is True
        is_identity = extra.get("identity_field") is True
        is_system = extra.get("system_field") is True
        is_edge = bool(extra.get("edge_label"))
        if not (is_profile or is_metadata or is_identity or is_system or is_edge):
            misclassified.append(fname)

    assert not misclassified, (
        f"RadarSystemEntity fields not in any of the four buckets "
        f"(profile-mapped / system_metadata / identity / system_field) "
        f"or edges: {misclassified}"
    )
