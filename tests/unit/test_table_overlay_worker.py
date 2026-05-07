"""Worker-side unit tests for app.services.table_overlay (spec §8.3)."""
from unittest.mock import MagicMock
import pytest


def _make_pass_result(entity_type, instances):
    """Build a minimal PassResult-like with iter_entities_of_type."""
    pr = MagicMock()
    def _iter(et):
        if et != entity_type:
            return iter([])
        return iter(instances)
    pr.iter_entities_of_type = _iter
    return pr


def _missile_inst(name):
    """Real Pydantic missile instance, not a MagicMock."""
    from ontology_bundles.air_defense_v3.extraction_schemas import (
        missile_propulsion,
    )
    return missile_propulsion.MissilePropulsionRecord(system_name=name)


def test_identity_rewrite_empty_alias_map_is_noop():
    from app.services.table_overlay import apply_identity_rewrite
    inst = _missile_inst("SA-75")
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    ontology = {"entity_types": [{"name": "MISSILE_SYSTEM"}]}
    stats = apply_identity_rewrite(pass_results, {}, ontology)
    assert stats.rewrites == 0
    assert inst.system_name == "SA-75"


def test_identity_rewrite_rewrites_alias_to_canonical():
    from app.services.table_overlay import apply_identity_rewrite
    a = _missile_inst("SA-75")
    b = _missile_inst("SA-2A")
    pr = _make_pass_result("MISSILE_SYSTEM", [a, b])
    pass_results = {"missile_propulsion": pr}
    ontology = {"entity_types": [{"name": "MISSILE_SYSTEM"}]}
    alias_map = {"MISSILE_SYSTEM": {"SA-75": "1D", "SA-2A": "1D"}}
    stats = apply_identity_rewrite(pass_results, alias_map, ontology)
    assert stats.rewrites == 2
    assert a.system_name == "1D"
    assert b.system_name == "1D"


def test_identity_rewrite_entity_type_scoped():
    """alias 'COMMON' under MISSILE_SYSTEM rewrites missiles only;
    radar instances unaffected."""
    from app.services.table_overlay import apply_identity_rewrite
    from ontology_bundles.air_defense_v3.extraction_schemas import (
        missile_propulsion, radar_antenna,
    )
    m = missile_propulsion.MissilePropulsionRecord(system_name="COMMON")
    r = radar_antenna.RadarAntennaRecord(system_name="COMMON")
    pr_m = _make_pass_result("MISSILE_SYSTEM", [m])
    pr_r = _make_pass_result("RADAR_SYSTEM", [r])
    pass_results = {"missile_propulsion": pr_m, "radar_antenna": pr_r}
    ontology = {"entity_types": [
        {"name": "MISSILE_SYSTEM"}, {"name": "RADAR_SYSTEM"},
    ]}
    alias_map = {"MISSILE_SYSTEM": {"COMMON": "REWRITTEN"}}
    stats = apply_identity_rewrite(pass_results, alias_map, ontology)
    assert m.system_name == "REWRITTEN"
    assert r.system_name == "COMMON"  # untouched
    assert stats.rewrites == 1


def _make_table_fact(**kwargs):
    """Build TableFact from the canonical worker-side home (Task 6)."""
    from app.services.table_overlay import TableFact
    defaults = dict(
        canonical_entity="1D", entity_type="MISSILE_SYSTEM",
        schema_field="booster_mass_kg", value=1135.0,
        source_label="Weight kg", section_ctx="1st Stage",
        pass_name="missile_propulsion", raw_text="1135",
    )
    defaults.update(kwargs)
    return TableFact(**defaults)


def test_field_overlay_validation_runs_field_validator():
    """Spec §5.3 step (c): cls.model_validate must execute
    _v_booster_mass_kg = field_validator(...)(coerce_optional_float).
    Pass value as a STRING; expected coerced to float."""
    from app.services.table_overlay import apply_field_overlay
    inst = _missile_inst("1D")
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(value="1135")  # string, not float
    stats = apply_field_overlay(pass_results, [fact])
    assert stats.applied == 1
    assert stats.matches_touched == 1
    assert stats.skipped_validation_fail == 0
    assert isinstance(inst.booster_mass_kg, float)
    assert inst.booster_mass_kg == 1135.0


def test_field_overlay_unknown_field_precheck():
    from app.services.table_overlay import apply_field_overlay
    inst = _missile_inst("1D")
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(schema_field="totally_bogus_field")
    stats = apply_field_overlay(pass_results, [fact])
    assert stats.applied == 0
    assert stats.skipped_unknown_field == 1
    # Instance unchanged
    assert getattr(inst, "totally_bogus_field", "ABSENT") == "ABSENT"


def test_field_overlay_table_wins_overrides_populated():
    from app.services.table_overlay import apply_field_overlay
    inst = _missile_inst("1D")
    inst.booster_mass_kg = 970.0  # LLM wrong
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(value=1135.0)
    stats = apply_field_overlay(pass_results, [fact])
    assert stats.applied == 1
    assert stats.conflicts_overridden == 1
    assert inst.booster_mass_kg == 1135.0


def test_field_overlay_fans_out_to_all_matching():
    from app.services.table_overlay import apply_field_overlay
    a = _missile_inst("1D")
    b = _missile_inst("1D")  # post-rewrite duplicate
    pr = _make_pass_result("MISSILE_SYSTEM", [a, b])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(value=1135.0)
    stats = apply_field_overlay(pass_results, [fact])
    assert stats.applied == 2  # fact-instance count under fan-out
    assert stats.matches_touched == 1  # fact landed on >=1 (incremented once)
    assert a.booster_mass_kg == 1135.0
    assert b.booster_mass_kg == 1135.0


def test_field_overlay_entity_type_scope():
    from app.services.table_overlay import apply_field_overlay
    from ontology_bundles.air_defense_v3.extraction_schemas import radar_antenna
    radar = radar_antenna.RadarAntennaRecord(system_name="1D")
    pr = _make_pass_result("RADAR_SYSTEM", [radar])
    pass_results = {"radar_antenna": pr}
    fact = _make_table_fact()  # entity_type="MISSILE_SYSTEM"
    stats = apply_field_overlay(pass_results, [fact])
    # MISSILE_SYSTEM fact must NOT land on a RADAR_SYSTEM instance
    assert stats.applied == 0
    assert stats.skipped_no_entity == 1


def test_field_overlay_validation_failure_keeps_instance_unchanged():
    """Spec §5.3 step (c)+(d): when cls.model_validate(...) raises (here
    via validate_missile_system_name's ValueError on empty/None),
    skipped_validation_fail++ and the instance keeps its prior value
    AND siblings in the same fan-out are independent.

    Uses system_name + value=None because numeric-field validators on
    these schemas (coerce_optional_float, coerce_optional_text) are
    permissive — they coerce unparseable input to None instead of
    raising. system_name's validator
    (_missile_shared.py:validate_missile_system_name) is the one
    field-level validator on these schemas that genuinely raises."""
    from app.services.table_overlay import apply_field_overlay
    inst = _missile_inst("1D")
    inst.booster_mass_kg = 970.0  # canary — must be unchanged after fail
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(schema_field="system_name", value=None)
    stats = apply_field_overlay(pass_results, [fact])
    assert stats.applied == 0
    assert stats.skipped_validation_fail == 1
    # Per-pair atomicity: prior LLM canary value untouched.
    assert inst.booster_mass_kg == 970.0
    # And the system_name was NOT changed to None.
    assert inst.system_name == "1D"


def test_field_overlay_only_touches_fields_with_facts():
    """Scoped table_wins: a field with no fact is never touched."""
    from app.services.table_overlay import apply_field_overlay
    inst = _missile_inst("1D")
    inst.booster_mass_kg = 970.0
    inst.sustain_mass_kg = 555.0
    pr = _make_pass_result("MISSILE_SYSTEM", [inst])
    pass_results = {"missile_propulsion": pr}
    fact = _make_table_fact(schema_field="booster_mass_kg", value=1135.0)
    apply_field_overlay(pass_results, [fact])
    assert inst.booster_mass_kg == 1135.0
    assert inst.sustain_mass_kg == 555.0  # untouched
