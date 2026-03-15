"""Unit tests for ontology template helpers.

All tests are pure (no DB, no network) — they exercise YAML loading and
type-name helpers.  Extraction prompt and Pydantic model tests have moved
to the Docling-Graph service.
"""

import pytest

pytestmark = pytest.mark.unit


class TestLoadOntology:
    def test_loads_ontology_yaml(self):
        from app.services.ontology_templates import load_ontology

        ontology = load_ontology()
        assert "version" in ontology
        assert "entity_types" in ontology
        assert "relationship_types" in ontology

    def test_entity_types_present(self):
        from app.services.ontology_templates import load_ontology

        ontology = load_ontology()
        names = [et["name"] for et in ontology["entity_types"]]
        assert "EQUIPMENT_SYSTEM" in names
        assert "COMPONENT" in names
        assert "STANDARD" in names

    def test_relationship_types_present(self):
        from app.services.ontology_templates import load_ontology

        ontology = load_ontology()
        names = [rt["name"] for rt in ontology["relationship_types"]]
        assert "PART_OF" in names
        assert "CONTAINS" in names
        assert "SPECIFIED_BY" in names


class TestPropertyEnrichment:
    """Validate that all ontology properties have descriptions, examples, and patterns."""

    def test_all_properties_have_descriptions(self):
        from app.services.ontology_templates import load_ontology

        ontology = load_ontology()
        missing = []
        for et in ontology["entity_types"]:
            props = et["properties"]["properties"]
            for prop_name, prop_def in props.items():
                if "description" not in prop_def:
                    missing.append(f"{et['name']}.{prop_name}")
        assert missing == [], f"Properties missing descriptions: {missing}"

    def test_structured_properties_have_examples(self):
        from app.services.ontology_templates import load_ontology

        ontology = load_ontology()
        missing = []
        for et in ontology["entity_types"]:
            props = et["properties"]["properties"]
            for prop_name, prop_def in props.items():
                if "pattern" in prop_def and "example" not in prop_def:
                    missing.append(f"{et['name']}.{prop_name}")
        assert missing == [], f"Properties with patterns but no examples: {missing}"


class TestValidationMatrix:
    def test_validation_matrix_loads(self):
        from app.services.ontology_templates import load_validation_matrix

        matrix = load_validation_matrix()
        assert len(matrix) > 0, "Validation matrix should not be empty"
        # Check a known valid triple
        assert ("SUBSYSTEM", "PART_OF", "EQUIPMENT_SYSTEM") in matrix

    def test_validation_matrix_covers_all_relationships(self):
        from app.services.ontology_templates import (
            build_relationship_type_names,
            load_validation_matrix,
        )

        matrix = load_validation_matrix()
        matrix_rels = {rel for _, rel, _ in matrix}
        ontology_rels = set(build_relationship_type_names())

        missing = ontology_rels - matrix_rels
        assert missing == set(), (
            f"Relationship types missing from validation matrix: {sorted(missing)}"
        )


class TestHelpers:
    def test_build_entity_type_names(self):
        from app.services.ontology_templates import build_entity_type_names

        names = build_entity_type_names()
        assert isinstance(names, list)
        assert "EQUIPMENT_SYSTEM" in names
        assert len(names) >= 10

    def test_build_relationship_type_names(self):
        from app.services.ontology_templates import build_relationship_type_names

        names = build_relationship_type_names()
        assert isinstance(names, list)
        assert "CONTAINS" in names
        assert len(names) >= 5
