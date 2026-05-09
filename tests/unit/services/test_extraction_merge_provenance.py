"""Tests for Task 12.5: worker-side mirror of new service-side provenance fields.

Covers:
- ExtractionProvenance gains evidence_ids / page_numbers / evidence_text
- FieldEvidenceRow gains evidence_id / page / document_id
- New ExtractionRelationshipProvenance dataclass
- _parse_pass_response wires all new fields end-to-end
"""


def test_extraction_provenance_mirror_accepts_evidence_fields():
    """Worker-side ExtractionProvenance dataclass must accept the new
    evidence_ids / page_numbers / evidence_text fields from the service."""
    from app.services.extraction_merge import ExtractionProvenance
    p = ExtractionProvenance(
        instance_id="i1",
        ontology_name="RADAR_SYSTEM",
        identity_values={},
        element_uid="#/texts/12",
        evidence_ids=["#/texts/12"],
        page_numbers=[3],
        evidence_text="AN/MPQ-65 radar...",
    )
    assert p.evidence_ids == ["#/texts/12"]


def test_extraction_provenance_mirror_backward_compat():
    from app.services.extraction_merge import ExtractionProvenance
    p = ExtractionProvenance(
        instance_id="i1",
        ontology_name="RADAR_SYSTEM",
        identity_values={},
        element_uid="#/texts/12",
    )
    assert p.evidence_ids == []
    assert p.page_numbers == []
    assert p.evidence_text is None


def test_field_evidence_row_mirror_accepts_evidence_id_page_doc():
    from app.services.extraction_merge import FieldEvidenceRow
    r = FieldEvidenceRow(
        chunk_id=None,
        snippet="diameter: 0.37 m",
        element_uid="#/texts/12",
        value=0.37,
        evidence_id="#/texts/12",
        page=3,
        document_id="doc-uuid",
    )
    assert r.evidence_id == "#/texts/12"
    assert r.page == 3
    assert r.document_id == "doc-uuid"


def test_field_evidence_row_backward_compat():
    from app.services.extraction_merge import FieldEvidenceRow
    r = FieldEvidenceRow(
        chunk_id=None,
        snippet="some text",
        element_uid="#/texts/0",
    )
    assert r.evidence_id is None
    assert r.page is None
    assert r.document_id is None


def test_extraction_relationship_provenance_mirror_exists():
    from app.services.extraction_merge import ExtractionRelationshipProvenance
    r = ExtractionRelationshipProvenance(
        relationship_type="HAS_PART",
        source_instance_id="i1",
        target_instance_id="i2",
        evidence_ids=["#/texts/0"],
    )
    assert r.relationship_type == "HAS_PART"
    assert r.source_instance_id == "i1"
    assert r.target_instance_id == "i2"
    assert r.evidence_ids == ["#/texts/0"]
    assert r.self_refs == []
    assert r.page_numbers == []
    assert r.supporting_snippet is None


def test_parse_pass_response_extracts_relationship_provenance():
    """_parse_pass_response must consume the new relationship_provenance
    list from the service response."""
    from app.workers.pipeline import _parse_pass_response

    pass_def = type("PD", (), {
        "name": "missile_propulsion",
        "module": "extraction_schemas.missile_propulsion",
        "template_class": "MissilePropulsionPass",
    })()
    manifest = type("M", (), {"bundle_key": "air_defense_v3"})()

    raw = {
        "bundle_key": "air_defense_v3",
        "pass_name": "missile_propulsion",
        "pass_output": {"missile_systems": []},
        "metadata": {},
        "provenance": [],
        "field_provenance": [],
        "relationship_provenance": [
            {
                "relationship_type": "REL",
                "source_instance_id": "i1",
                "target_instance_id": "i2",
                "evidence_ids": ["#/texts/0"],
            },
        ],
    }
    parsed = _parse_pass_response(raw, pass_def, manifest)
    rels = getattr(parsed, "relationship_provenance", None) or []
    assert len(rels) == 1
    assert rels[0].relationship_type == "REL"
    assert rels[0].source_instance_id == "i1"
    assert rels[0].target_instance_id == "i2"
    assert rels[0].evidence_ids == ["#/texts/0"]


def test_parse_pass_response_extracts_provenance_new_fields():
    """_parse_pass_response must pass through evidence_ids/page_numbers/
    evidence_text into ExtractionProvenance rows."""
    from app.workers.pipeline import _parse_pass_response

    pass_def = type("PD", (), {
        "name": "missile_propulsion",
        "module": "extraction_schemas.missile_propulsion",
        "template_class": "MissilePropulsionPass",
    })()
    manifest = type("M", (), {"bundle_key": "air_defense_v3"})()

    raw = {
        "bundle_key": "air_defense_v3",
        "pass_name": "missile_propulsion",
        "pass_output": {"missile_systems": []},
        "metadata": {},
        "field_provenance": [],
        "relationship_provenance": [],
        "provenance": [
            {
                "instance_id": "inst-1",
                "ontology_name": "MISSILE_SYSTEM",
                "identity_values": {"system_name": "PAC-3"},
                "element_uid": "#/texts/7",
                "evidence_ids": ["#/texts/7", "#/texts/8"],
                "page_numbers": [3, 4],
                "evidence_text": "PAC-3 missile system...",
            },
        ],
    }
    parsed = _parse_pass_response(raw, pass_def, manifest)
    assert len(parsed.provenance) == 1
    prov = parsed.provenance[0]
    assert prov.evidence_ids == ["#/texts/7", "#/texts/8"]
    assert prov.page_numbers == [3, 4]
    assert prov.evidence_text == "PAC-3 missile system..."
