# tests/unit/test_extraction_signal_detectors.py
from app.services.extraction_signal_detectors import (
    measurement_present, categorical_present, image_present)

def test_measurement_pass_specific():
    assert measurement_present({"length", "angle"}, "max range 2500 km") is True
    assert measurement_present({"mass"}, "2500 km") is False

def test_measurement_rejects_designators():
    assert measurement_present({"length"}, "S-75M and V-88, 5Ya23 variant") is False

def test_measurement_spelled_and_imperial():
    assert measurement_present({"length"}, "about 40 feet tall") is True
    assert measurement_present({"velocity"}, "4500 meters per second") is True

def test_measurement_mass_rejects_designators_keeps_real():
    # designators must NOT fire the mass signal
    assert measurement_present({"mass"}, "the Yak-130G aircraft") is False
    assert measurement_present({"mass"}, "engage the T-90 at close range") is False
    # real mass measurements still fire
    assert measurement_present({"mass"}, "warhead weighs 190 kg") is True
    assert measurement_present({"mass"}, "payload of 2 tonnes") is True

def test_categorical():
    assert categorical_present({"guidance_type"}, "uses semi-active radar homing") is True
    assert categorical_present({"guidance_type"}, "the warhead weighs 200 kg") is False

def test_image():
    assert image_present(["#/pictures/2", "#/texts/9"]) is True
    assert image_present(["#/texts/9"]) is False
    assert image_present(None) is False

def test_empty_inputs():
    assert categorical_present(set(), "semi-active radar homing") is False
    assert measurement_present(set(), "2500 km") is False
    assert image_present([]) is False

def test_categorical_unknown_field():
    assert categorical_present({"nonexistent_field"}, "semi-active radar homing") is False

def test_categorical_multi_field():
    assert categorical_present({"scan_type", "guidance_type"}, "phased array active radar homing") is True

def test_categorical_other_fields():
    assert categorical_present({"scan_type"}, "phased array radar antenna") is True
    assert categorical_present({"system_status"}, "system is operational") is True
    assert categorical_present({"emitter_function"}, "early warning radar") is True

def test_time_signal_rejects_prose_us():
    assert measurement_present({"time"}, "the system provides us more range") is False
    assert measurement_present({"time"}, "burn time 22 seconds") is True
