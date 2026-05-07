"""Tests for coerce_value (spec §5.6 / D3 + D4)."""
import sys
from pathlib import Path

# Mirror the path setup used by test_table_facts_resolve.py so that
# `from app._alias_map import ...` inside _table_facts resolves to the
# docling-graph app package, not the repo-root app package.
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_APP_PATH = _SERVICE_ROOT / "app"
sys.path.insert(0, str(_APP_PATH))
sys.path.insert(0, str(_SERVICE_ROOT))

# Ensure the docling-graph app package is cached in sys.modules under 'app'
# BEFORE importing _table_facts (which does lazy `from app._alias_map import`
# inside function bodies). Without this, the repo-root app/ may win the race.
import importlib as _il
if "app" not in sys.modules or not hasattr(sys.modules["app"], "__path__") or \
        str(_APP_PATH) not in sys.modules["app"].__path__:
    # Evict any stale repo-root 'app' entries and load the DG one.
    for _k in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
        del sys.modules[_k]
    _il.import_module("app")

import _table_facts as _tf  # noqa: E402


def _load():
    return _tf


# --- numeric, single value -------------------------------------------------

def test_numeric_single_value_explicit_unit():
    tf = _load()
    out = tf.coerce_value("1135 kg", "booster_mass_kg")
    assert len(out) == 1
    assert out[0].value == 1135.0
    assert out[0].unit_inferred == "kg"
    assert out[0].conversion_factor == 1.0


def test_numeric_single_value_implied_unit_from_field_suffix():
    """No unit in cell; field suffix _kg implies mass_kg unit class."""
    tf = _load()
    out = tf.coerce_value("1135", "booster_mass_kg")
    assert len(out) == 1
    assert out[0].value == 1135.0


def test_numeric_unit_conversion_mm_to_m_explicit():
    """Cell has explicit unit '10726 mm' -> 10.726 m via mm conversion."""
    tf = _load()
    out = tf.coerce_value("10726 mm", "body_length_m")
    assert len(out) == 1
    assert abs(out[0].value - 10.726) < 1e-6
    assert out[0].conversion_factor == 0.001


def test_numeric_unit_conversion_mm_to_m_via_row_label():
    """Cell '10726' (no unit) + row_label 'Length mm' -> mm implied -> 10.726 m."""
    tf = _load()
    out = tf.coerce_value("10726", "body_length_m", row_label="Length mm")
    assert len(out) == 1
    assert abs(out[0].value - 10.726) < 1e-6


def test_bare_length_infers_mm_when_magnitude_requires_it():
    tf = _load()
    out = tf.coerce_value("10726", "body_length_m", row_label="Length")
    assert len(out) == 1
    assert abs(out[0].value - 10.726) < 1e-6
    assert out[0].unit_inferred == "mm"


def test_bare_length_keeps_m_when_magnitude_requires_it():
    tf = _load()
    out = tf.coerce_value("10.726", "body_length_m", row_label="Length")
    assert len(out) == 1
    assert out[0].value == 10.726
    assert out[0].unit_inferred == "m"


def test_bare_diameter_infers_mm_when_magnitude_requires_it():
    tf = _load()
    out = tf.coerce_value("654", "body_diameter_m", row_label="Diameter")
    assert len(out) == 1
    assert abs(out[0].value - 0.654) < 1e-6
    assert out[0].unit_inferred == "mm"


def test_bare_range_infers_m_when_magnitude_requires_it():
    tf = _load()
    out = tf.coerce_value("29000", "max_intercept_km", row_label="Max Range")
    assert len(out) == 1
    assert out[0].value == 29.0
    assert out[0].unit_inferred == "m"


def test_bare_range_keeps_km_when_magnitude_requires_it():
    tf = _load()
    out = tf.coerce_value("29", "max_intercept_km", row_label="Max Range")
    assert len(out) == 1
    assert out[0].value == 29.0
    assert out[0].unit_inferred == "km"


def test_bare_altitude_infers_m_when_magnitude_requires_it():
    tf = _load()
    out = tf.coerce_value("22000", "max_altitude_km", row_label="Altitude")
    assert len(out) == 1
    assert out[0].value == 22.0
    assert out[0].unit_inferred == "m"


def test_bare_min_altitude_infers_m_when_km_is_implausible():
    tf = _load()
    out = tf.coerce_value("100", "min_altitude_km", row_label="Min Altitude")
    assert len(out) == 1
    assert out[0].value == 0.1
    assert out[0].unit_inferred == "m"


def test_bare_risky_length_skips_when_no_interpretation_is_plausible():
    tf = _load()
    assert tf.coerce_value("150", "body_length_m", row_label="Length") == []


def test_numeric_unit_from_row_label_kg():
    """Cell '1135' + row_label '1st Stage Weight kg' -> kg implied."""
    tf = _load()
    out = tf.coerce_value("1135", "booster_mass_kg", row_label="1st Stage Weight kg")
    assert len(out) == 1
    assert out[0].value == 1135.0
    assert out[0].unit_inferred == "kg"


def test_numeric_no_unit_anywhere_falls_back_to_canonical():
    tf = _load()
    out = tf.coerce_value("1135", "booster_mass_kg")
    assert len(out) == 1
    assert out[0].value == 1135.0


def test_numeric_usec_suffix_bare_value_defaults_to_usec():
    tf = _load()
    out = tf.coerce_value("2500", "nominal_pri_usec", row_label="PRI")
    assert len(out) == 1
    assert out[0].value == 2500.0
    assert out[0].unit_inferred == "usec"


def test_numeric_ms_to_usec_explicit():
    tf = _load()
    out = tf.coerce_value("2.5 ms", "nominal_pri_usec")
    assert len(out) == 1
    assert out[0].value == 2500.0
    assert out[0].unit_inferred == "ms"


def test_numeric_dbm_to_dbw_uses_offset_conversion():
    tf = _load()
    out = tf.coerce_value("100 dBm", "erp_dbw")
    assert len(out) == 1
    assert out[0].value == 70.0
    assert out[0].unit_inferred == "dbm"


def test_multi_value_alternatives_slash():
    """'X/Y' -> two facts, multi_value emitted."""
    tf = _load()
    out = tf.coerce_value("1135/1028", "booster_mass_kg")
    assert len(out) == 2
    values = sorted(p.value for p in out)
    assert values == [1028.0, 1135.0]


def test_multi_value_range_endash_collapses_to_midpoint():
    """'4–6 sec' (en-dash range) -> ONE fact at midpoint 5.0."""
    tf = _load()
    out = tf.coerce_value("4–6 sec", "booster_time_sec")
    assert len(out) == 1
    assert out[0].value == 5.0


def test_multi_value_range_to_word_collapses_to_midpoint():
    tf = _load()
    out = tf.coerce_value("4 to 6 sec", "booster_time_sec")
    assert len(out) == 1
    assert out[0].value == 5.0


def test_ambiguous_hyphen_range_via_xLessThanY():
    """'29-34' with X<Y -> range, midpoint."""
    tf = _load()
    out = tf.coerce_value("29-34", "max_intercept_km")
    assert len(out) == 1
    assert out[0].value == 31.5


def test_ambiguous_hyphen_alternatives_via_xGreaterThanY():
    """'1135-1028' with X>Y -> alternatives, two facts."""
    tf = _load()
    out = tf.coerce_value("1135-1028", "booster_mass_kg")
    assert len(out) == 2


def test_string_field_passthrough():
    """String-typed schema fields pass cell text through verbatim."""
    tf = _load()
    out = tf.coerce_value("dual-pulse Mark 104 sustainer", "sustain_thrust")
    assert len(out) == 1
    assert out[0].value == "dual-pulse Mark 104 sustainer"
    assert out[0].unit_inferred is None


def test_stop_words_return_empty_list():
    tf = _load()
    for stop in ["", "TBD", "—", "N/A", "unknown", "?", "-", "--"]:
        assert tf.coerce_value(stop, "booster_mass_kg") == [], f"{stop!r} not stopped"


def test_unparseable_returns_empty():
    tf = _load()
    assert tf.coerce_value("not a number", "booster_mass_kg") == []


def test_unknown_unit_returns_empty():
    """'1135 furlongs' has no entry in mass_kg unit table -> skip."""
    tf = _load()
    assert tf.coerce_value("1135 furlongs", "booster_mass_kg") == []
