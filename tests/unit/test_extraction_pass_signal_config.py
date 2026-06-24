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
