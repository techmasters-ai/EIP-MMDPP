# tests/unit/test_extraction_pass_signal_config.py
from app.services.extraction_pass_signal_config import derive_pass_signal_config

def test_kinematics_dimensions():
    cfg = derive_pass_signal_config("air_defense_v3")
    assert cfg["missile_kinematics"].dimensions == {"length", "angle"}

def test_guidance_categorical_and_image():
    cfg = derive_pass_signal_config("air_defense_v3")
    g = cfg["missile_guidance"]
    assert g.categorical_fields == {"guidance_type", "seeker_type"}
    assert g.has_image_field is True

def test_antenna_image_and_dims():
    cfg = derive_pass_signal_config("air_defense_v3")
    a = cfg["radar_antenna"]
    assert a.has_image_field is True
    assert a.dimensions == {"length", "angle", "gain"}

def test_identity_passes_excluded():
    cfg = derive_pass_signal_config("air_defense_v3")
    assert "radar_identity" not in cfg
    assert "missile_identity" not in cfg
    assert len(cfg) == 9  # exactly the 9 field_group (routable) passes

def test_iter_routable_pass_fields_unwraps_records():
    from app.services.ontology_bundles import iter_routable_pass_fields
    routable = dict(iter_routable_pass_fields("air_defense_v3"))
    assert len(routable) == 9
    container_names = {"radar_systems", "missile_systems", "systems", "records"}
    for pass_name, fields in routable.items():
        assert not (set(fields) & container_names), f"{pass_name} got container-level names: {fields}"
        # sanity: real record fields present
        assert any("_" in f for f in fields), f"{pass_name} fields look wrong: {fields}"
