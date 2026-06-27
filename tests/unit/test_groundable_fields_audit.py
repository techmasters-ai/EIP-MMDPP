"""Tests for groundable-fields audit: new unit suffixes, plural word forms,
and the no-gap invariant across the air_defense_v3 bundle.

Companion to ``scripts/audit_groundable_fields.py``.
"""
from app.services.field_value_grounding import units_for
from scripts.audit_groundable_fields import UNITLESS_OK, _is_numeric_annotation, audit_bundle


def test_new_suffixes_ground():
    assert units_for("scan_period_sec") == ["s", "sec", "second", "seconds"]
    assert units_for("nominal_pri_usec") == ["µs", "us", "microsec"]
    assert units_for("gain_dbi") == ["dbi"]
    assert units_for("muzzle_velocity_kmh") == ["km/h", "kph"]  # longest-suffix-wins intact


def test_plural_word_forms_present():
    # token-bounded matching (Task 1) killed substring plurals; lists must carry them
    assert "metres" in units_for("body_length_m") and "meters" in units_for("body_length_m")


def test_no_silent_suffix_gaps():
    rows = audit_bundle("air_defense_v3")
    gaps = [r for r in rows if r.numeric and not r.units and r.field not in UNITLESS_OK]
    assert gaps == [], f"numeric fields with no unit suffix and not allowlisted: {gaps}"
    assert not any(r.field == "<resolution_error>" for r in rows)


def test_is_numeric_annotation_pep604():
    # PEP 604 union syntax (Python ≥ 3.10)
    assert _is_numeric_annotation(float | None) is True
    assert _is_numeric_annotation(bool | None) is False
