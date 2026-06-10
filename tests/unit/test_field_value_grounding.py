"""Unit tests for the pure value→chunk matcher (app/services/field_value_grounding).

This is the matcher originally validated in
``scripts/build_extracted_from_groundtruth.py`` (the chunk-9 "50 km" ground-truth
builder), moved into a shared module. These are the move-over tests:
  * adjacent prose match ("50 km")
  * table same-chunk match (number + detached-unit header)
  * the trailing-period case '2391.' near 'kg'
  * no false single-digit matches in the table tier
  * unit-from-suffix (longest suffix wins; unitless → no units)
Plus coverage for the ``match_value_to_chunks`` convenience wrapper used by the
worker lineage path.
"""
import json
from pathlib import Path

import pytest

from app.services.field_value_grounding import (
    ADJACENT,
    SAME_CHUNK,
    has_unit_token,
    match_value_to_chunks,
    nfc,
    num_variants,
    units_for,
    value_in_chunk,
)


# --- units_for (unit-from-suffix) -------------------------------------------
def test_units_for_basic_suffixes():
    assert units_for("max_intercept_km") == ["km", "км"]
    assert units_for("erp_dbw") == ["dbw"]
    assert units_for("nominal_rf_mhz") == ["mhz"]
    assert units_for("max_launch_angle_deg") == ["deg", "°", "degree", "degrees", "град"]
    assert units_for("total_mass_kg") == ["kg", "кг"]


def test_units_for_longest_suffix_wins():
    """``_kmh`` must resolve to km/h variants, NOT km (the ``_km`` suffix is a
    shorter tail of the same name)."""
    assert units_for("muzzle_velocity_kmh") == ["km/h", "kph"]
    assert units_for("muzzle_velocity_kmh") != ["km", "км"]


def test_units_for_unitless_returns_empty():
    assert units_for("system_name") == []
    assert units_for("guidance") == []
    assert units_for("") == []


# --- num_variants ------------------------------------------------------------
def test_num_variants_int_and_float_forms():
    assert num_variants(50.0) == {"50", "50.0"}
    assert num_variants(50) == {"50"}
    # a true float keeps its %g + raw str forms.
    assert "2391" in num_variants(2391.0)


def test_num_variants_non_numeric_empty():
    assert num_variants("radio command") == set()
    assert num_variants(None) == set()


# --- value_in_chunk: ADJACENT (prose) ---------------------------------------
def test_adjacent_prose_match():
    """'50 km' in prose → ADJACENT (high confidence)."""
    text = nfc("The missile can intercept targets up to 50 km away.")
    assert value_in_chunk(num_variants(50.0), units_for("max_intercept_km"), text) == ADJACENT


def test_adjacent_cyrillic_unit_match():
    """The NFC/casefold path matches the Cyrillic 'км' too (50км)."""
    text = nfc("дальность 50км")
    assert value_in_chunk(num_variants(50.0), units_for("max_intercept_km"), text) == ADJACENT


def test_adjacent_hyphen_and_paren_separators():
    """A value joined to its unit by a hyphen / paren still counts as adjacent."""
    assert value_in_chunk({"50"}, ["km"], nfc("a 50-km range")) == ADJACENT
    assert value_in_chunk({"50"}, ["km"], nfc("range 50(km)")) == ADJACENT


# --- value_in_chunk: SAME_CHUNK (table, detached unit) ----------------------
def test_table_same_chunk_match():
    """A number present AND a unit header elsewhere in the chunk → SAME_CHUNK."""
    text = nfc("Weight | 7 | = 2391 | header units = kg")
    assert value_in_chunk(num_variants(2391.0), ["kg", "кг"], text) == SAME_CHUNK


def test_trailing_period_2391_near_kg():
    """The '2391.' (trailing period, sentence-final) near a 'kg' header still
    matches in the table tier — the value boundary regex tolerates the period."""
    text = nfc("Launch mass listed as 2391. The unit column is kg.")
    assert value_in_chunk(num_variants(2391.0), ["kg", "кг"], text) == SAME_CHUNK


def test_no_false_single_digit_table_match():
    """The table tier requires >=2 digits, so a coincidental single-digit number
    ('7') does NOT match against a detached 'kg' header (avoids false hits)."""
    text = nfc("Row 7 of the spec sheet, units in kg")
    assert value_in_chunk(num_variants(7.0), ["kg", "кг"], text) is None


def test_no_match_when_value_absent():
    text = nfc("no relevant numbers here, just kg as a header")
    assert value_in_chunk(num_variants(50.0), ["kg", "кг"], text) is None


def test_no_match_when_substring_not_boundaried():
    """'50' must not match inside '2502' (boundary guards on both tiers)."""
    assert value_in_chunk({"50"}, ["km"], nfc("altitude 2502 m")) is None


# --- match_value_to_chunks (the worker-facing wrapper) ----------------------
def test_match_value_to_chunks_adjacent_only():
    chunks = {
        "c8": "intercept up to 50 km",   # adjacent → match
        "c18": "see table 4 for details",  # no value
    }
    assert match_value_to_chunks("max_intercept_km", 50.0, chunks) == ["c8"]


def test_match_value_to_chunks_orders_adjacent_before_same_chunk():
    """Adjacent (prose) hits rank ahead of detached-unit table hits."""
    chunks = {
        "c_table": "Weight | 2391 | ... | kg",   # SAME_CHUNK
        "c_prose": "launch mass is 2391 kg",       # ADJACENT
    }
    assert match_value_to_chunks("total_mass_kg", 2391.0, chunks) == ["c_prose", "c_table"]


def test_match_value_to_chunks_unitless_returns_empty():
    chunks = {"c1": "the value 50 appears here"}
    assert match_value_to_chunks("guidance", 50.0, chunks) == []


def test_match_value_to_chunks_non_numeric_returns_empty():
    chunks = {"c1": "radio command guidance, 50 km"}
    assert match_value_to_chunks("max_intercept_km", "radio command", chunks) == []


def test_match_value_to_chunks_no_hit_returns_empty():
    # neither chunk contains the value 50: c1 has no number, c2 has the unit
    # 'km' but a DIFFERENT number (43), so no chunk grounds the value.
    chunks = {"c1": "no matching value here", "c2": "the range is 43 km"}
    assert match_value_to_chunks("max_intercept_km", 50.0, chunks) == []


# --- fixture-driven tests (shared with docling-graph mirror) -----------------

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "docker" / "docling-graph" / "tests" / "fixtures" / "unit_matcher_cases.json"
)
_CASES = json.loads(_FIXTURE.read_text())


@pytest.mark.parametrize("case", _CASES["unit_token_cases"])
def test_unit_token_cases(case):
    assert has_unit_token(nfc(case["text"]), case["units"]) is case["expect"]


@pytest.mark.parametrize("case", _CASES["value_in_chunk_cases"])
def test_value_in_chunk_cases(case):
    got = value_in_chunk(case["num_strs"], case["units"], nfc(case["text"]))
    assert got == case["expect"]
