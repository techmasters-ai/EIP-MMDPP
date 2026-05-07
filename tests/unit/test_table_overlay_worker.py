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
