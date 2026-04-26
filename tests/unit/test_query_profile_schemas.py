"""Schemas added by the flat-schema profile refactor (spec §6)."""


def test_query_profile_field_entry_minimal():
    from app.schemas.query_profiles import QueryProfileFieldEntry
    entry = QueryProfileFieldEntry(
        name="gain_dbi", label="Gain (dBi)", value=35.0,
    )
    assert entry.name == "gain_dbi"
    assert entry.value == 35.0
    assert entry.evidence == []   # default empty


def test_query_profile_field_entry_with_metadata():
    from app.schemas.query_profiles import QueryProfileFieldEntry
    entry = QueryProfileFieldEntry(
        name="scan_type", label="Scan Type", value="phased-array",
        description="Scan mechanism / mode.",
        examples=["phased-array", "mechanical"],
    )
    assert entry.description == "Scan mechanism / mode."
    assert "mechanical" in entry.examples


def test_query_profile_field_group_groups_entries():
    from app.schemas.query_profiles import QueryProfileFieldGroup, QueryProfileFieldEntry
    grp = QueryProfileFieldGroup(
        subgroup="antenna", subgroup_label="Antenna",
        fields=[
            QueryProfileFieldEntry(name="gain_dbi", label="Gain", value=35.0),
            QueryProfileFieldEntry(name="beamwidth_az_deg", label="BW Az", value=1.5),
        ],
    )
    assert len(grp.fields) == 2
    assert grp.subgroup == "antenna"


def test_section_response_default_field_groups_empty():
    from app.schemas.query_profiles import QueryProfileSectionResponse
    from app.schemas.graph_store import GraphEntityResult

    resp = QueryProfileSectionResponse(
        registry_id=None, profile_id="p", profile_label="P",
        resolved_root=GraphEntityResult(name="X", entity_type="RADAR_SYSTEM"),
        total=0,
    )
    assert resp.field_groups == []
    assert resp.related_systems == []
    assert resp.items == []


def test_dossier_section_carries_kind_discriminator():
    from app.schemas.query_profiles import QueryProfileDossierSection
    sec = QueryProfileDossierSection(
        profile_id="system_rf_parameters", profile_label="RF",
        kind="section_properties",
    )
    assert sec.kind == "section_properties"
    assert sec.field_groups == []
    assert sec.items == []
