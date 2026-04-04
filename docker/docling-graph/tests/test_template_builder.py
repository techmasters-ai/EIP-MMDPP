import pytest


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
