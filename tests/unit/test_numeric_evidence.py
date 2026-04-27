"""Phase B Session 1 — shared numeric-evidence predicate (spec §4.8).

Both the auto-evidence resolver and the radar-postprocessing
"unsupported numeric clearer" must use the same logic for deciding
whether a numeric value's stringified form (with unit-aware variants)
appears in batch evidence text.
"""
import importlib.util
import pathlib
import sys

_SERVICE_ROOT = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "docker" / "docling-graph" / "app"
)


def _load(modname: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ne = _load("_dgp_numeric_evidence", _SERVICE_ROOT / "_numeric_evidence.py")
value_is_supported_by_text = _ne.value_is_supported_by_text
value_match_candidates = _ne.value_match_candidates
normalize_text = _ne.normalize_text


def test_normalize_text_collapses_whitespace_and_casefolds():
    assert normalize_text("  Fan   SONG  ") == "fan song"


def test_value_match_candidates_for_float_with_dbi_suffix():
    forms = value_match_candidates(35.0, "gain_dbi")
    norm = [normalize_text(f) for f in forms]
    assert any("35" in n for n in norm)
    assert any("dbi" in n for n in norm)


def test_value_match_candidates_for_int_with_mhz_suffix():
    """Same-unit candidates AND cross-unit converted variants.

    Per cross-unit-conversion-followup-todo.md (Option A landed): the
    helper emits cross-magnitude conversions when consistent with the
    canonical field. For value 3000 in nominal_rf_mhz:
    - "3000 mhz"     — canonical (same-unit case variants)
    - "3 ghz"        — correct conversion (3000 MHz = 3 GHz)
    - "3000000 khz"  — correct conversion (3000 MHz = 3,000,000 kHz)

    Same-magnitude wrong-unit forms are STILL never emitted:
    - "3000 ghz"     — would mean 3000 GHz = 3 THz (wrong)
    - "3000 khz"     — would mean 3000 kHz = 3 MHz (wrong)
    """
    forms = value_match_candidates(3000, "nominal_rf_mhz")
    norm = [normalize_text(f) for f in forms]
    # Canonical (same-unit case variants).
    assert any("3000 mhz" in n for n in norm)
    # Cross-unit (correct conversions).
    assert any("3 ghz" in n for n in norm), (
        "value 3000 MHz should yield '3 GHz' candidate after Option A"
    )
    assert any("3000000 khz" in n for n in norm), (
        "value 3000 MHz should yield '3000000 kHz' candidate after Option A"
    )
    # Same-magnitude wrong-unit forms are STILL forbidden.
    assert not any("3000 ghz" in n for n in norm), (
        "must not emit '3000 GHz' for value 3000 in MHz field — that's "
        "physically wrong (3000 GHz = 3 THz)"
    )
    assert not any("3000 khz" in n for n in norm), (
        "must not emit '3000 kHz' for value 3000 in MHz field — that's "
        "physically wrong (3000 kHz = 3 MHz)"
    )


def test_value_match_candidates_for_int_with_kw_suffix_megawatts():
    """Cross-unit conversion for the harder-test Tombstone case.

    "1.4 megawatts" in source text → tx_peak_power_kw = 1400. For value
    1400 in tx_peak_power_kw, the helper must emit "1.4 megawatts" /
    "1.4 MW" / "1400000 W" candidates so the evidence gate preserves it.
    """
    forms = value_match_candidates(1400, "tx_peak_power_kw")
    norm = [normalize_text(f) for f in forms]
    assert any("1400 kw" in n for n in norm)         # canonical
    assert any("1.4 mw" in n for n in norm)          # MW abbreviation
    assert any("1.4 megawatts" in n for n in norm)   # full word
    assert any("1400000 w" in n for n in norm)       # base SI
    # Wrong-direction: "1400 megawatts" would mean 1.4 GW
    assert not any("1400 mw" in n for n in norm)
    assert not any("1400 megawatts" in n for n in norm)


def test_value_match_candidates_for_kg_suffix_tonnes():
    """Cross-unit conversion for the missile total_mass_kg case.

    A doc that says "1.5 tonnes" gives total_mass_kg = 1500. The helper
    must emit "1.5 tonnes" so the evidence gate preserves it.
    """
    forms = value_match_candidates(1500, "total_mass_kg")
    norm = [normalize_text(f) for f in forms]
    assert any("1500 kg" in n for n in norm)         # canonical
    assert any("1.5 tonnes" in n for n in norm)      # cross-unit
    assert any("1.5 tonne" in n for n in norm)       # singular variant
    assert any("1.5 metric tons" in n for n in norm)
    # Wrong-direction
    assert not any("1500 tonnes" in n for n in norm)


def test_value_match_candidates_for_km_suffix_meters():
    """Cross-unit conversion for missile range / radar coverage.

    A doc that says "43000 m" gives max_intercept_km = 43. The helper
    must emit "43000 m" so the evidence gate preserves it.
    """
    forms = value_match_candidates(43, "max_intercept_km")
    norm = [normalize_text(f) for f in forms]
    assert any("43 km" in n for n in norm)           # canonical
    assert any("43000 m" in n for n in norm)         # SI base
    assert any("43000 meters" in n for n in norm)
    assert not any("43 m" in n for n in norm)        # wrong direction


def test_value_is_supported_by_text_string_field():
    assert value_is_supported_by_text(
        "PHASED-ARRAY", "scan_type",
        "The radar uses a phased-array scan type.",
    )
    assert not value_is_supported_by_text(
        "ELECTRONIC", "scan_type",
        "The radar uses a phased-array scan type.",
    )


def test_value_is_supported_by_text_numeric_with_unit():
    assert value_is_supported_by_text(
        35.0, "gain_dbi", "The antenna gain is 35 dBi nominal.",
    )
    assert value_is_supported_by_text(
        3000.0, "nominal_rf_mhz", "Operates at 3000 MHz.",
    )
    assert value_is_supported_by_text(
        600.0, "tx_peak_power_kw", "Transmitter peak power is 600 kW.",
    )


def test_value_is_supported_by_text_numeric_no_match():
    """Unsupported numeric values return False so the caller can null them."""
    assert not value_is_supported_by_text(
        9999.0, "gain_dbi", "The antenna gain is 35 dBi.",
    )


def test_value_is_supported_by_text_none_value():
    """None values are vacuously supported (the caller's null-check
    happens upstream of this predicate)."""
    assert value_is_supported_by_text(None, "gain_dbi", "any text")


# ---------------------------------------------------------------------------
# Cross-unit conversion (Option A from cross-unit-conversion-followup-todo.md)
# ---------------------------------------------------------------------------


def test_value_is_supported_by_text_cross_unit_frequency_GHz():
    """value 10000.0 in nominal_rf_mhz, evidence "operates at 10 GHz" → True.

    This is the radar harder-test Tombstone case: Tombstone's frequency
    is stated in source as "10 GHz" but the canonical field is MHz, so
    the LLM must emit 10000.0. The §4.8 evidence gate must preserve it.
    """
    assert value_is_supported_by_text(
        10000.0, "nominal_rf_mhz", "operates at a nominal carrier frequency of 10 GHz",
    )
    assert value_is_supported_by_text(
        10000.0, "nominal_rf_mhz", "Operates at 10 GHz.",
    )


def test_value_is_supported_by_text_cross_unit_power_megawatts():
    """value 1400.0 in tx_peak_power_kw, evidence "1.4 megawatts" → True.

    The other half of the Tombstone harder-test case. "1.4 megawatts"
    must yield tx_peak_power_kw=1400 preserved.
    """
    assert value_is_supported_by_text(
        1400.0, "tx_peak_power_kw", "reported peak transmitter output of 1.4 megawatts",
    )
    assert value_is_supported_by_text(
        1400.0, "tx_peak_power_kw", "peak power 1.4 MW",
    )


def test_value_is_supported_by_text_cross_unit_mass_tonnes():
    """value 1500.0 in total_mass_kg, evidence "1.5 tonnes" → True.

    Missile-flavored cross-unit case. The missile postprocessor's
    _mechanically_supported_missile_fields() handles "WEIGHT: X LBS"
    (lbs → kg) case-specifically; this generalizes to tonnes via the
    new conversion table.
    """
    assert value_is_supported_by_text(
        1500.0, "total_mass_kg", "total launch weight 1.5 tonnes",
    )
    assert value_is_supported_by_text(
        1500.0, "total_mass_kg", "1.5 metric tons",
    )


def test_value_is_supported_by_text_cross_unit_negative():
    """Cross-unit conversion does NOT make the predicate accept unrelated values.

    Adding cross-unit conversion must not introduce false positives for
    value/evidence pairs that should still be rejected.
    """
    # 999 MHz != 10 GHz (= 10000 MHz). Neither "999" nor "0.999 GHz"
    # appears in the evidence → must reject.
    assert not value_is_supported_by_text(
        999.0, "nominal_rf_mhz", "operates at 10 GHz",
    )
    # 9999 kg != 1.5 tonnes (= 1500 kg). Neither "9999" nor "9.999 tonnes"
    # appears in the evidence → must reject.
    assert not value_is_supported_by_text(
        9999.0, "total_mass_kg", "1.5 tonnes",
    )
    # 88 km != 43 km. Neither "88" nor "88000 m" appears in evidence → reject.
    assert not value_is_supported_by_text(
        88.0, "max_intercept_km", "max range 43 km",
    )
    # NOTE on a deliberately-unsaved pre-existing case:
    # `value_is_supported_by_text(3000.0, "nominal_rf_mhz", "operates at 3000 GHz")`
    # returns True because the bare number "3000" appears in the evidence
    # ("3000 GHz"). That's not a cross-unit issue — the bare-number match
    # is orthogonal to the unit, and it existed pre-Option A. It's
    # tracked as a separate concern (bare-number false positive on
    # adversarial wrong-unit text) — not this commit's scope.
