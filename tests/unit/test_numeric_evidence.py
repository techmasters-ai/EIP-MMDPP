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
    """Same-unit candidates only — no cross-unit alternates.

    The helper does NOT generate '3000 GHz' as an alternate for '3000 MHz'
    because '3000 GHz' is a different physical magnitude (3 THz vs 3 GHz);
    matching across units without converting the value would silently
    accept wrong values. If you need cross-unit support, convert at the
    LLM-emission step or add explicit unit-conversion logic — do NOT
    paper over it here. Tracked in the plan's "Out of scope" section.
    """
    forms = value_match_candidates(3000, "nominal_rf_mhz")
    norm = [normalize_text(f) for f in forms]
    assert any("3000 mhz" in n for n in norm)
    # Negative assertion: no cross-unit alternates.
    assert not any("3000 ghz" in n for n in norm), (
        "value_match_candidates must not emit cross-unit alternates "
        "without value conversion — see helper docstring."
    )
    assert not any("3000 khz" in n for n in norm)


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
