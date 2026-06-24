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

def test_categorical():
    assert categorical_present({"guidance_type"}, "uses semi-active radar homing") is True
    assert categorical_present({"guidance_type"}, "the warhead weighs 200 kg") is False

def test_image():
    assert image_present(["#/pictures/2", "#/texts/9"]) is True
    assert image_present(["#/texts/9"]) is False
    assert image_present(None) is False
