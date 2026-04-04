import pytest
import yaml
from pathlib import Path
from pydantic import BaseModel

ONTOLOGY_PATH = Path(__file__).parent.parent.parent.parent / "ontology" / "ontology.yaml"


def _load_ontology():
    with open(ONTOLOGY_PATH) as f:
        return yaml.safe_load(f)


class TestGraphIdFieldsDerivation:
    def test_entity_with_name_property(self):
        # PLATFORM has 'name' → ["name"]
        from app.template_builder import derive_graph_id_fields
        props = {"name": {"type": "string"}, "platform_type": {"type": "string"}}
        assert derive_graph_id_fields("PLATFORM", props) == ["name"]

    def test_entity_with_system_name(self):
        # RADAR_SYSTEM has 'system_name' → ["system_name"]
        from app.template_builder import derive_graph_id_fields
        props = {"system_name": {"type": "string"}, "nomenclature": {"type": "string"}}
        assert derive_graph_id_fields("RADAR_SYSTEM", props) == ["system_name"]

    def test_entity_with_id_suffix(self):
        # DOCUMENT has 'document_id' → ["document_id"]
        from app.template_builder import derive_graph_id_fields
        props = {"document_id": {"type": "string"}, "title": {"type": "string"}}
        assert derive_graph_id_fields("DOCUMENT", props) == ["document_id"]

    def test_entity_with_figure_id(self):
        from app.template_builder import derive_graph_id_fields
        props = {"figure_id": {"type": "string"}, "caption": {"type": "string"}}
        assert derive_graph_id_fields("FIGURE", props) == ["figure_id"]

    def test_entity_with_table_id(self):
        from app.template_builder import derive_graph_id_fields
        props = {"table_id": {"type": "string"}, "caption": {"type": "string"}}
        assert derive_graph_id_fields("TABLE", props) == ["table_id"]

    def test_entity_with_heading(self):
        from app.template_builder import derive_graph_id_fields
        props = {"heading": {"type": "string"}, "page_start": {"type": "integer"}}
        assert derive_graph_id_fields("SECTION", props) == ["heading"]

    def test_entity_with_composite_identity(self):
        from app.template_builder import derive_graph_id_fields
        props = {"parameter": {"type": "string"}, "value": {"type": "string"}, "unit": {"type": "string"}}
        assert derive_graph_id_fields("SPECIFICATION", props) == ["parameter", "value"]

    def test_fallback_to_first_property(self):
        from app.template_builder import derive_graph_id_fields
        props = {"custom_field": {"type": "string"}, "other": {"type": "integer"}}
        assert derive_graph_id_fields("CUSTOM_TYPE", props) == ["custom_field"]

    def test_priority_name_over_id_suffix(self):
        from app.template_builder import derive_graph_id_fields
        props = {"name": {"type": "string"}, "component_id": {"type": "string"}}
        assert derive_graph_id_fields("COMPONENT", props) == ["name"]


class TestBuildTemplates:
    def test_builds_model_for_entity_type(self):
        from app.template_builder import build_templates
        ontology = _load_ontology()
        templates = build_templates(ontology)
        assert "RADAR_SYSTEM" in templates
        assert issubclass(templates["RADAR_SYSTEM"], BaseModel)

    def test_model_has_graph_id_fields(self):
        from app.template_builder import build_templates
        ontology = _load_ontology()
        templates = build_templates(ontology)
        model_cls = templates["RADAR_SYSTEM"]
        assert model_cls.model_config.get("graph_id_fields") == ["system_name"]

    def test_model_has_typed_fields(self):
        from app.template_builder import build_templates
        ontology = _load_ontology()
        templates = build_templates(ontology)
        model_cls = templates["RADAR_SYSTEM"]
        fields = model_cls.model_fields
        assert "system_name" in fields
        assert "nomenclature" in fields
        assert "radar_type" in fields

    def test_all_fields_optional_except_identity(self):
        from app.template_builder import build_templates
        ontology = _load_ontology()
        templates = build_templates(ontology)
        model_cls = templates["RADAR_SYSTEM"]
        fields = model_cls.model_fields
        # system_name is identity — required (no default)
        assert fields["system_name"].is_required()
        # nomenclature is not identity — optional
        assert not fields["nomenclature"].is_required()

    def test_field_descriptions_from_ontology(self):
        from app.template_builder import build_templates
        ontology = _load_ontology()
        templates = build_templates(ontology)
        model_cls = templates["RADAR_SYSTEM"]
        desc = model_cls.model_fields["system_name"].description
        assert desc and len(desc) > 0

    def test_all_ontology_entity_types_have_templates(self):
        from app.template_builder import build_templates
        ontology = _load_ontology()
        templates = build_templates(ontology)
        # Only entity_types with properties (not relationship types)
        entity_types = [
            et for et in ontology["entity_types"]
            if et.get("properties", {}).get("properties")
        ]
        for et in entity_types:
            name = et["name"]
            safe_name = "TABLE_REF" if name == "TABLE" else name
            assert safe_name in templates, f"Missing template for {name}"

    def test_reserved_word_table_mapped(self):
        from app.template_builder import build_templates
        ontology = _load_ontology()
        templates = build_templates(ontology)
        assert "TABLE" not in templates
        assert "TABLE_REF" in templates

    def test_model_instantiation(self):
        from app.template_builder import build_templates
        ontology = _load_ontology()
        templates = build_templates(ontology)
        model_cls = templates["RADAR_SYSTEM"]
        instance = model_cls(system_name="Tombstone")
        assert instance.system_name == "Tombstone"


class TestEdgeFields:
    def test_radar_system_has_installed_on_edge(self):
        from app.template_builder import build_templates_with_edges
        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)
        model_cls = templates["RADAR_SYSTEM"]
        assert "installed_on" in model_cls.model_fields
        field = model_cls.model_fields["installed_on"]
        assert field.json_schema_extra == {"edge_label": "INSTALLED_ON"}

    def test_radar_system_has_multiple_edges(self):
        from app.template_builder import build_templates_with_edges
        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)
        model_cls = templates["RADAR_SYSTEM"]
        fields = model_cls.model_fields
        # Multiple edges from validation_matrix
        assert "has_antenna" in fields
        assert "uses_waveform" in fields
        assert "has_performance" in fields

    def test_one_to_many_edge_is_list(self):
        from app.template_builder import build_templates_with_edges
        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)
        model_cls = templates["RADAR_SYSTEM"]
        field = model_cls.model_fields["has_antenna"]
        # one_to_many → default_factory=list
        assert field.default_factory is not None
        assert field.default_factory() == []

    def test_one_to_one_edge_is_optional(self):
        from app.template_builder import build_templates_with_edges
        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)
        model_cls = templates["RADAR_SYSTEM"]
        # HAS_PROCESSING_CHAIN is one_to_one → Optional (not required)
        field = model_cls.model_fields["has_processing_chain"]
        assert not field.is_required()
        assert field.default is None

    def test_no_edge_for_invalid_source(self):
        from app.template_builder import build_templates_with_edges
        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)
        # IF_AMPLIFIER has PART_OF → RECEIVER in validation_matrix
        model_cls = templates["IF_AMPLIFIER"]
        assert "part_of" in model_cls.model_fields

    def test_specification_has_specified_by_edge(self):
        from app.template_builder import build_templates_with_edges
        ontology = _load_ontology()
        templates = build_templates_with_edges(ontology)
        # EQUIPMENT_SYSTEM has SPECIFIED_BY edge
        model_cls = templates["EQUIPMENT_SYSTEM"]
        assert "specified_by" in model_cls.model_fields
        field = model_cls.model_fields["specified_by"]
        assert field.json_schema_extra == {"edge_label": "SPECIFIED_BY"}
