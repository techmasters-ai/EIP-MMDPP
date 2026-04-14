"""Tests for _render_upstream_entities_preamble() in the docling-graph service.

All EntityRef inputs are constructed via dg_app_module.EntityRef so these
tests work whether or not the module is loaded in the combined test suite.
"""


def test_empty_list_returns_empty_string(dg_app_module):
    fn = dg_app_module._render_upstream_entities_preamble
    assert fn([]) == ""


def test_none_returns_empty_string(dg_app_module):
    fn = dg_app_module._render_upstream_entities_preamble
    assert fn(None) == ""


def test_single_entity_shape(dg_app_module):
    EntityRef = dg_app_module.EntityRef
    fn = dg_app_module._render_upstream_entities_preamble
    entity = EntityRef(
        ref_id="E001",
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "Fan Song"},
        display_label="Fan Song",
    )
    result = fn([entity])
    assert "Upstream entities:" in result
    assert "[E001] RADAR_SYSTEM \u2014 Fan Song" in result
    assert "Only emit from_ref_id and to_ref_id values from the list above" in result


def test_multiple_entities_preserve_input_order(dg_app_module):
    EntityRef = dg_app_module.EntityRef
    fn = dg_app_module._render_upstream_entities_preamble
    entities = [
        EntityRef(
            ref_id="E001",
            entity_type="RADAR_SYSTEM",
            identity_values={"system_name": "Fan Song"},
            display_label="Fan Song",
        ),
        EntityRef(
            ref_id="E002",
            entity_type="MISSILE_SYSTEM",
            identity_values={"system_name": "SA-2"},
            display_label="SA-2",
        ),
        EntityRef(
            ref_id="E003",
            entity_type="RADAR_SYSTEM",
            identity_values={"system_name": "Spoon Rest"},
            display_label="Spoon Rest",
        ),
    ]
    result = fn(entities)
    pos_e001 = result.index("[E001]")
    pos_e002 = result.index("[E002]")
    pos_e003 = result.index("[E003]")
    assert pos_e001 < pos_e002 < pos_e003, "Entities must appear in input order"


def test_missing_display_label_falls_back_to_entity_type(dg_app_module):
    EntityRef = dg_app_module.EntityRef
    fn = dg_app_module._render_upstream_entities_preamble
    entity = EntityRef(
        ref_id="E001",
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "Fan Song"},
        display_label=None,
    )
    result = fn([entity])
    # em-dash should NOT appear when display_label is None
    assert "\u2014" not in result
    assert "[E001] RADAR_SYSTEM" in result


def test_env_flag_disables_preamble(dg_app_module, monkeypatch):
    EntityRef = dg_app_module.EntityRef
    fn = dg_app_module._render_upstream_entities_preamble
    monkeypatch.setenv("DOCLING_GRAPH_UPSTREAM_PREAMBLE", "false")
    entity = EntityRef(
        ref_id="E001",
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "Fan Song"},
        display_label="Fan Song",
    )
    result = fn([entity])
    assert result == ""
