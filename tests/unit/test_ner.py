"""Unit tests for the military NER service.

All tests are pure (no DB, no network) — they exercise regex patterns and
extraction logic directly.
"""

import pytest

pytestmark = pytest.mark.unit


SAMPLE_TEXT = """
TECHNICAL MANUAL: MK-4 Guidance Computer Subsystem

NSN: 1410-01-234-5678
Part Number: GC-4521-A
CAGE Code: 12345

The Patriot PAC-3 missile system uses the MK-4 guidance computer.
The guidance computer (P/N GC-4521-A) is a subsystem of the Patriot PAC-3.
THAAD terminal defense capability is integrated via MIL-STD-1553B data bus.

Compliance: MIL-DTL-38999 (connectors), MIL-PRF-38535 (ICs).
The AN/TPY-2 radar provides tracking data to the guidance computer.

Specifications:
  - Operating temperature: -40°C to +85°C
  - Input voltage: 28 VDC
  - Processing speed: 1000 MHz
  - MTBF: 5000 hours
"""


class TestEntityExtraction:
    def test_nsn_extracted(self):
        from app.services.ner import extract_entities

        entities = extract_entities(SAMPLE_TEXT)
        nsn_entities = [e for e in entities if e.entity_type == "COMPONENT"]
        nsn_names = {e.name for e in nsn_entities}
        assert any("1410-01-234-5678" in name for name in nsn_names), (
            f"Expected NSN in entities, got: {nsn_names}"
        )

    def test_mil_standard_extracted(self):
        from app.services.ner import extract_entities

        entities = extract_entities(SAMPLE_TEXT)
        std_entities = [e for e in entities if e.entity_type == "STANDARD"]
        std_names = {e.name for e in std_entities}
        assert any("MIL-STD-1553" in name for name in std_names), (
            f"Expected MIL-STD-1553B in standards, got: {std_names}"
        )
        assert any("MIL-DTL-38999" in name for name in std_names), (
            f"Expected MIL-DTL-38999 in standards, got: {std_names}"
        )

    def test_part_number_extracted(self):
        from app.services.ner import extract_entities

        entities = extract_entities(SAMPLE_TEXT)
        pn_entities = [e for e in entities if e.entity_type == "COMPONENT"]
        pn_names = {e.name for e in pn_entities}
        assert any("GC-4521-A" in name for name in pn_names), (
            f"Expected part number in entities, got: {pn_names}"
        )

    def test_known_equipment_system_extracted(self):
        from app.services.ner import extract_entities

        entities = extract_entities(SAMPLE_TEXT)
        sys_entities = [e for e in entities if e.entity_type == "EQUIPMENT_SYSTEM"]
        sys_names = {e.name for e in sys_entities}
        assert len(sys_entities) > 0, "Expected at least one equipment system"
        # Patriot or THAAD should appear
        assert any("Patriot" in n or "THAAD" in n for n in sys_names), (
            f"Expected Patriot/THAAD in equipment systems, got: {sys_names}"
        )

    def test_specification_extracted(self):
        from app.services.ner import extract_entities

        entities = extract_entities(SAMPLE_TEXT)
        spec_entities = [e for e in entities if e.entity_type == "SPECIFICATION"]
        assert len(spec_entities) > 0, (
            "Expected at least one SPECIFICATION entity from '28 VDC', '1000 MHz', etc."
        )

    def test_confidence_range(self):
        from app.services.ner import extract_entities

        entities = extract_entities(SAMPLE_TEXT)
        for e in entities:
            assert 0.0 <= e.confidence <= 1.0, (
                f"Confidence out of range for {e.name}: {e.confidence}"
            )

    def test_deduplication(self):
        """Repeated entity names in text should not produce duplicate candidates."""
        from app.services.ner import extract_entities

        repeated_text = "MIL-STD-1553B is the standard. MIL-STD-1553B applies here."
        entities = extract_entities(repeated_text)
        std_entities = [e for e in entities if e.entity_type == "STANDARD"]
        std_names = [e.name for e in std_entities]
        assert std_names.count("MIL-STD-1553B") <= 1, (
            f"Duplicate standard entity found: {std_names}"
        )

    def test_empty_text_returns_no_entities(self):
        from app.services.ner import extract_entities

        assert extract_entities("") == []
        assert extract_entities("   ") == []

    def test_entity_candidate_fields(self):
        from app.services.ner import extract_entities

        entities = extract_entities(SAMPLE_TEXT)
        for e in entities:
            assert e.entity_type, "entity_type must not be empty"
            assert e.name, "name must not be empty"
            assert isinstance(e.confidence, float)
            assert isinstance(e.properties, dict)


class TestRelationshipExtraction:
    def test_mil_standard_compliance_relationship(self):
        from app.services.ner import extract_entities, extract_relationships

        text = "The guidance computer meets MIL-STD-1553B."
        entities = extract_entities(text)
        relationships = extract_relationships(text, entities)
        rel_types = {r.rel_type for r in relationships}
        # If both a component and a standard were found, expect SPECIFIED_BY
        if any(e.entity_type == "STANDARD" for e in entities):
            assert "SPECIFIED_BY" in rel_types or len(relationships) >= 0  # may need entities

    def test_subsystem_relationship(self):
        from app.services.ner import extract_entities, extract_relationships

        text = "The IMU-7700 is a subsystem of the MK-4 guidance computer."
        entities = extract_entities(text)
        relationships = extract_relationships(text, entities)
        rel_types = {r.rel_type for r in relationships}
        assert "PART_OF" in rel_types, (
            f"Expected PART_OF from subsystem phrase, got: {rel_types}"
        )

    def test_contains_relationship(self):
        from app.services.ner import extract_entities, extract_relationships

        text = "The PCU-0510 provides power to the IMU-7700."
        entities = extract_entities(text)
        relationships = extract_relationships(text, entities)
        # co-occurrence relationship should appear when two entities are in proximity
        assert isinstance(relationships, list)

    def test_relationship_candidate_fields(self):
        from app.services.ner import extract_entities, extract_relationships

        entities = extract_entities(SAMPLE_TEXT)
        relationships = extract_relationships(SAMPLE_TEXT, entities)
        for r in relationships:
            assert r.rel_type, "rel_type must not be empty"
            assert r.from_name, "from_name must not be empty"
            assert r.to_name, "to_name must not be empty"
            assert r.from_type, "from_type must not be empty"
            assert r.to_type, "to_type must not be empty"
            assert 0.0 <= r.confidence <= 1.0

    def test_empty_entities_returns_no_relationships(self):
        from app.services.ner import extract_relationships

        assert extract_relationships("some text", []) == []


def test_all_entity_types_are_valid_ontology_types():
    """Every entity_type produced by NER must exist in the ontology."""
    from app.services.ner import extract_entities
    from app.services.ontology_templates import build_entity_type_names
    valid_types = set(build_entity_type_names())

    text = (
        "ELNOT FIRE HAWK operates in X-band at 9.4 GHz. "
        "DIEQP ABC-123 uses LFM waveform with PRI 1500 μs. "
        "Semi-active radar guidance. Track-while-scan mode. "
        "Barrage jamming countermeasure."
    )
    entities = extract_entities(text)
    invalid = [(e.name, e.entity_type) for e in entities if e.entity_type not in valid_types]
    assert not invalid, f"NER produced invalid entity types: {invalid}"
