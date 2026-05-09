"""Phase 8 Task 50: ExtractionProvenance schema tests.

Pure schema-shape tests — no running service required. Uses the
``dg_schemas`` fixture so the docling-graph ``app/schemas.py`` is
loaded instead of the repo-root ``app/schemas`` package (sys.path
ambiguity between the two ``app`` packages is handled in conftest).
"""
from __future__ import annotations

import pytest


def test_extraction_provenance_importable(dg_schemas):
    assert hasattr(dg_schemas, "ExtractionProvenance")


def test_extraction_provenance_requires_all_four_core_fields(dg_schemas):
    """Core contract: instance_id / ontology_name / identity_values /
    element_uid are all required. Missing any → ValidationError.
    Plan §8 Task 51 strengthened element_uid to required (the downstream
    derive_structure_links mention path resolves chunks by element_uid,
    never by page alone)."""
    from pydantic import ValidationError

    # Fully populated → OK
    prov = dg_schemas.ExtractionProvenance(
        instance_id="uuid-1",
        ontology_name="RADAR_SYSTEM",
        identity_values={"system_name": "Tombstone"},
        element_uid="elem-1",
    )
    assert prov.instance_id == "uuid-1"
    assert prov.ontology_name == "RADAR_SYSTEM"
    assert prov.identity_values == {"system_name": "Tombstone"}
    assert prov.element_uid == "elem-1"
    assert prov.page is None
    assert prov.chunk_index is None

    for missing in ("instance_id", "ontology_name", "identity_values", "element_uid"):
        kwargs = {
            "instance_id": "uuid-1",
            "ontology_name": "RADAR_SYSTEM",
            "identity_values": {"system_name": "Tombstone"},
            "element_uid": "elem-1",
        }
        del kwargs[missing]
        with pytest.raises(ValidationError):
            dg_schemas.ExtractionProvenance(**kwargs)


def test_extraction_provenance_page_and_chunk_index_optional(dg_schemas):
    prov = dg_schemas.ExtractionProvenance(
        instance_id="uuid-1",
        ontology_name="SECTION",
        identity_values={"section_number": "3.2"},
        element_uid="elem-2",
        page=42,
        chunk_index=7,
    )
    assert prov.page == 42
    assert prov.chunk_index == 7


def test_extraction_provenance_accepts_empty_identity_values(dg_schemas):
    """Components (is_entity=False) have empty identity tuples — the
    Phase 8 schema must not block that shape. Per-instance tracking via
    instance_id alone disambiguates them in downstream aggregation."""
    prov = dg_schemas.ExtractionProvenance(
        instance_id="uuid-component-1",
        ontology_name="PROPULSION_STACK",
        identity_values={},
        element_uid="elem-3",
    )
    assert prov.identity_values == {}


def test_extract_pass_response_carries_provenance_list_defaulting_to_empty(dg_schemas):
    """ExtractPassResponse gains a provenance: list[ExtractionProvenance]
    field that defaults to []. Additive — existing callers don't care."""
    resp = dg_schemas.ExtractPassResponse(
        bundle_key="air_defense_v3",
        pass_name="radar_domain",
        pass_output={"radar_systems": []},
    )
    assert resp.provenance == []


def test_extract_pass_response_roundtrip_with_provenance(dg_schemas):
    """The new field is serialization-compatible (json round-trip)."""
    resp = dg_schemas.ExtractPassResponse(
        bundle_key="air_defense_v3",
        pass_name="radar_domain",
        pass_output={"radar_systems": []},
        provenance=[
            dg_schemas.ExtractionProvenance(
                instance_id="uuid-1",
                ontology_name="RADAR_SYSTEM",
                identity_values={"system_name": "Tombstone"},
                element_uid="elem-1",
            )
        ],
    )
    dumped = resp.model_dump_json()
    reloaded = dg_schemas.ExtractPassResponse.model_validate_json(dumped)
    assert len(reloaded.provenance) == 1
    assert reloaded.provenance[0].element_uid == "elem-1"


# ---------------------------------------------------------------------------
# Task 9 — additive evidence fields (backward compat + accepts new fields)
# ---------------------------------------------------------------------------


def test_extraction_provenance_backward_compat(dg_schemas):
    """Old callers that don't pass evidence_ids / page_numbers / evidence_text
    must still get valid objects with empty/None defaults."""
    prov = dg_schemas.ExtractionProvenance(
        instance_id="uuid-bc",
        ontology_name="RADAR_SYSTEM",
        identity_values={"system_name": "Tombstone"},
        element_uid="#/texts/5",
    )
    assert prov.evidence_ids == []
    assert prov.page_numbers == []
    assert prov.evidence_text is None


def test_extraction_provenance_accepts_new_fields(dg_schemas):
    """New callers can populate the three additive fields."""
    prov = dg_schemas.ExtractionProvenance(
        instance_id="uuid-new",
        ontology_name="MISSILE_SYSTEM",
        identity_values={"system_name": "PAC-3"},
        element_uid="#/texts/10",
        evidence_ids=["#/texts/10", "#/texts/11"],
        page_numbers=[3, 4],
        evidence_text="The PAC-3 system intercepts ballistic missiles.",
    )
    assert prov.evidence_ids == ["#/texts/10", "#/texts/11"]
    assert prov.page_numbers == [3, 4]
    assert prov.evidence_text == "The PAC-3 system intercepts ballistic missiles."
