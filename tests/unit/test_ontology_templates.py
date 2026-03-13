"""Unit tests for ontology template generation.

All tests are pure (no DB, no network) — they exercise YAML loading and
Pydantic model generation.
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
        assert "IS_SUBSYSTEM_OF" in names
        assert "CONTAINS" in names
        assert "MEETS_STANDARD" in names


class TestGenerateEntityTemplates:
    def test_generates_models_for_all_entity_types(self):
        from app.services.ontology_templates import generate_entity_templates, load_ontology

        ontology = load_ontology()
        templates = generate_entity_templates(ontology)
        entity_names = [et["name"] for et in ontology["entity_types"]]

        assert len(templates) == len(entity_names)
        for name in entity_names:
            assert name in templates

    def test_generated_model_has_expected_fields(self):
        from app.services.ontology_templates import generate_entity_templates, load_ontology

        ontology = load_ontology()
        templates = generate_entity_templates(ontology)

        equip = templates["EQUIPMENT_SYSTEM"]
        fields = equip.model_fields
        assert "name" in fields
        assert "designation" in fields
        assert "status" in fields

    def test_generated_model_is_instantiable(self):
        from app.services.ontology_templates import generate_entity_templates, load_ontology

        ontology = load_ontology()
        templates = generate_entity_templates(ontology)

        instance = templates["EQUIPMENT_SYSTEM"](
            name="Patriot PAC-3", designation="MIM-104F"
        )
        assert instance.name == "Patriot PAC-3"
        assert instance.designation == "MIM-104F"

    def test_generated_model_has_entity_metadata(self):
        from app.services.ontology_templates import generate_entity_templates, load_ontology

        ontology = load_ontology()
        templates = generate_entity_templates(ontology)

        config = templates["COMPONENT"].model_config
        extra = config.get("json_schema_extra", {})
        assert extra.get("is_entity") is True
        assert extra.get("entity_type") == "COMPONENT"

    def test_fields_are_optional(self):
        from app.services.ontology_templates import generate_entity_templates, load_ontology

        ontology = load_ontology()
        templates = generate_entity_templates(ontology)

        # All fields should default to None
        instance = templates["COMPONENT"]()
        assert instance.name is None
        assert instance.part_number is None


class TestRelationshipTemplates:
    def test_generates_relationship_dicts(self):
        from app.services.ontology_templates import generate_relationship_templates, load_ontology

        ontology = load_ontology()
        rels = generate_relationship_templates(ontology)
        assert len(rels) > 0
        names = [r["name"] for r in rels]
        assert "IS_SUBSYSTEM_OF" in names

    def test_relationship_has_required_keys(self):
        from app.services.ontology_templates import generate_relationship_templates, load_ontology

        ontology = load_ontology()
        rels = generate_relationship_templates(ontology)
        for rel in rels:
            assert "name" in rel
            assert "description" in rel
            assert "source_type" in rel
            assert "target_type" in rel
            assert "cardinality" in rel


class TestExtractionPrompt:
    def test_build_extraction_prompt(self):
        from app.services.ontology_templates import build_extraction_prompt, load_ontology

        ontology = load_ontology()
        prompt = build_extraction_prompt(ontology, "The Patriot PAC-3 system uses MIL-STD-1553B.")
        assert "EQUIPMENT_SYSTEM" in prompt
        assert "IS_SUBSYSTEM_OF" in prompt
        assert "Patriot PAC-3" in prompt

    def test_prompt_truncates_long_text(self):
        from app.services.ontology_templates import build_extraction_prompt, load_ontology

        ontology = load_ontology()
        long_text = "A" * 20000
        prompt = build_extraction_prompt(ontology, long_text, max_text_length=100)
        assert "[...truncated...]" in prompt

    def test_compact_prompt_is_shorter(self):
        from app.services.ontology_templates import build_extraction_prompt, load_ontology

        ontology = load_ontology()
        text = "The Patriot PAC-3 system uses MIL-STD-1553B."
        full_prompt = build_extraction_prompt(ontology, text, compact_ontology=False)
        compact_prompt = build_extraction_prompt(ontology, text, compact_ontology=True)
        assert len(compact_prompt) < len(full_prompt)


class TestFewShotPrompt:
    def test_few_shot_includes_example(self):
        from app.services.ontology_templates import build_extraction_prompt, load_ontology

        ontology = load_ontology()
        prompt = build_extraction_prompt(ontology, "Test text", few_shot=True)
        assert "## Example" in prompt
        assert "Patriot PAC-3" in prompt
        assert "EQUIPMENT_SYSTEM" in prompt

    def test_no_few_shot_excludes_example(self):
        from app.services.ontology_templates import build_extraction_prompt, load_ontology

        ontology = load_ontology()
        prompt = build_extraction_prompt(ontology, "Test text", few_shot=False)
        assert "## Example" not in prompt

    def test_few_shot_with_compact(self):
        from app.services.ontology_templates import build_extraction_prompt, load_ontology

        ontology = load_ontology()
        prompt = build_extraction_prompt(
            ontology, "Test text", compact_ontology=True, few_shot=True,
        )
        assert "## Example" in prompt
        assert len(prompt) < len(
            build_extraction_prompt(ontology, "Test text", few_shot=True)
        )

    def test_prompt_few_shot_uses_valid_types(self):
        from app.services.ontology_templates import build_extraction_prompt, load_ontology, build_entity_type_names, build_relationship_type_names

        ontology = load_ontology()
        prompt = build_extraction_prompt(ontology, "test text", few_shot=True)
        # Few-shot example should use EQUIPMENT_SYSTEM, not made-up types
        assert "Patriot PAC-3" in prompt
        assert "EQUIPMENT_SYSTEM" in prompt
        # The old invalid few-shot example with AN/MPQ-53 should be gone
        assert "AN/MPQ-53" not in prompt

    def test_prompt_includes_property_descriptions(self):
        from app.services.ontology_templates import build_extraction_prompt, load_ontology

        ontology = load_ontology()
        prompt = build_extraction_prompt(ontology, "test text")
        assert "National Stock Number" in prompt or "CAGE" in prompt


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
        # Check a known valid triple from the legacy relations
        assert ("SUBSYSTEM", "IS_SUBSYSTEM_OF", "EQUIPMENT_SYSTEM") in matrix

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
