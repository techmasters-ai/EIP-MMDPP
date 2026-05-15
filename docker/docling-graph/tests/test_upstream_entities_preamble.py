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
    assert "REF=E001 | TYPE=RADAR_SYSTEM | Primary=Fan Song" in result
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
    pos_e001 = result.index("REF=E001")
    pos_e002 = result.index("REF=E002")
    pos_e003 = result.index("REF=E003")
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
    assert "REF=E001 | TYPE=RADAR_SYSTEM" in result


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


# --- Item 2.4: alias rendering tests -------------------------------------

def test_aliases_render_as_inline_primary_aliases_format(dg_app_module):
    """When an EntityRef carries aliases, the preamble emits the
    inline 'Primary | Aliases: a, b, c' format expected by the LLM."""
    EntityRef = dg_app_module.EntityRef
    fn = dg_app_module._render_upstream_entities_preamble
    entity = EntityRef(
        ref_id="E020",
        entity_type="MISSILE_SYSTEM",
        identity_values={"system_name": "1D"},
        display_label="1D",
        aliases=["SA-75", "SA-2A", "Guideline"],
    )
    result = fn([entity])
    assert (
        "REF=E020 | TYPE=MISSILE_SYSTEM | Primary=1D | Aliases: SA-75, SA-2A, Guideline"
        in result
    )


def test_no_aliases_uses_primary_format(dg_app_module):
    """When aliases is empty/None, the 'Primary=' format is used without
    the Aliases segment (v8a renamed from older emdash format)."""
    EntityRef = dg_app_module.EntityRef
    fn = dg_app_module._render_upstream_entities_preamble
    entity = EntityRef(
        ref_id="E001",
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "Fan Song"},
        display_label="Fan Song",
        aliases=None,
    )
    result = fn([entity])
    assert "REF=E001 | TYPE=RADAR_SYSTEM | Primary=Fan Song" in result
    assert "Aliases:" not in result


def test_aliases_filtered_when_duplicating_display_label(dg_app_module):
    """An alias equal to the display_label is filtered to avoid
    'Primary: X | Aliases: X' duplication noise."""
    EntityRef = dg_app_module.EntityRef
    fn = dg_app_module._render_upstream_entities_preamble
    entity = EntityRef(
        ref_id="E001",
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "Fan Song"},
        display_label="Fan Song",
        aliases=["Fan Song", "SNR-75"],   # "Fan Song" duplicates display_label
    )
    result = fn([entity])
    assert "Aliases: SNR-75" in result
    # The duplicate must not appear in the Aliases segment
    assert "Aliases: Fan Song, SNR-75" not in result


def test_aliases_instruction_mentions_alias_matching(dg_app_module):
    """The instruction line at the bottom should tell the LLM it can
    match either Primary or Aliases."""
    EntityRef = dg_app_module.EntityRef
    fn = dg_app_module._render_upstream_entities_preamble
    entity = EntityRef(
        ref_id="E020",
        entity_type="MISSILE_SYSTEM",
        identity_values={"system_name": "1D"},
        display_label="1D",
        aliases=["SA-75"],
    )
    result = fn([entity])
    assert "Match either the Primary name or any of the Aliases" in result


def test_dict_form_entity_also_renders_aliases(dg_app_module):
    """The preamble accepts both EntityRef instances and plain dicts.
    Aliases should work via either input form."""
    fn = dg_app_module._render_upstream_entities_preamble
    entity = {
        "ref_id": "E020",
        "entity_type": "MISSILE_SYSTEM",
        "display_label": "1D",
        "aliases": ["SA-75", "SA-2A"],
    }
    result = fn([entity])
    assert "REF=E020 | TYPE=MISSILE_SYSTEM | Primary=1D | Aliases: SA-75, SA-2A" in result


# --- v8a bracket-bug fix tests ----------------------------------------------

def test_preamble_does_not_wrap_ref_id_in_square_brackets(dg_app_module):
    """v8a fix: the LLM was copying surrounding brackets into ref_ids in its
    output (producing '[RADAR_SYSTEM:Fan Song]' instead of
    'RADAR_SYSTEM:Fan Song'). The preamble must use a format that doesn't
    visually wrap the ref_id."""
    EntityRef = dg_app_module.EntityRef
    fn = dg_app_module._render_upstream_entities_preamble
    entity = EntityRef(
        ref_id="RADAR_SYSTEM:Fan Song",
        entity_type="RADAR_SYSTEM",
        identity_values={"system_name": "Fan Song"},
        display_label="Fan Song",
    )
    result = fn([entity])
    assert "[RADAR_SYSTEM:Fan Song]" not in result, (
        f"Preamble still wraps ref_id in brackets — LLM will copy them into "
        f"output as malformed ids. Got:\n{result}"
    )


def test_preamble_uses_ref_equals_format(dg_app_module):
    """v8a fix: switch to 'REF=<ref_id>' style so the LLM doesn't include the
    delimiters in its emitted ref_ids."""
    EntityRef = dg_app_module.EntityRef
    fn = dg_app_module._render_upstream_entities_preamble
    entity = EntityRef(
        ref_id="MISSILE_SYSTEM:1D",
        entity_type="MISSILE_SYSTEM",
        identity_values={"system_name": "1D"},
        display_label="1D",
        aliases=["SA-75", "SA-2A"],
    )
    result = fn([entity])
    assert "REF=MISSILE_SYSTEM:1D" in result, (
        f"Expected 'REF=<ref_id>' format in preamble. Got:\n{result}"
    )
