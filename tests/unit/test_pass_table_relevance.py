"""Tests for the per-pass + per-table relevance decision used by the v9
table-aware chunking pipeline.

The decision logic (is_table_relevant_for_pass) determines whether a
normalized non-OTHER table should have its raw `#/tables/N` $ref replaced
in-place by synth refs (synth-only mode for that table) or left alone with
synth refs appended at the end of body.children (v9 behavior for tables
that don't match the active pass).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.services.table_normalization import normalize_tables
from app.services.table_normalization._pipeline_hooks import (
    PASS_TABLE_ROW_ALIASES,
    RAW_ONLY_PASSES,
    SYNTH_ELIGIBLE_PASSES,
    is_table_relevant_for_pass,
)
from app.services.table_normalization.models import Shape


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def _doc(fixture: dict) -> dict:
    return {"tables": [fixture], "texts": []}


def _fake_normalized(row_labels: list[str], shape: Shape = Shape.COLUMN_MAJOR) -> SimpleNamespace:
    """Build a minimal stand-in for NormalizedTable with the row labels we care about."""
    cells = [SimpleNamespace(row_label=lbl) for lbl in row_labels]
    return SimpleNamespace(cells=cells, shape=shape)


# --- relevance positive cases (per pass) -----------------------------------

def test_kinematics_relevant_when_table_has_range_or_alt_rows():
    nt = _fake_normalized(["Max Range", "Min Range", "Max Alt", "Min Alt"])
    assert is_table_relevant_for_pass("missile_kinematics", nt) is True


def test_airframe_relevant_when_table_has_length_or_weight_rows():
    nt = _fake_normalized(["Length", "Diameter", "Weight"])
    assert is_table_relevant_for_pass("missile_airframe", nt) is True


def test_speed_timing_relevant_when_table_has_vmax_rows():
    nt = _fake_normalized(["Vmax appr tgt", "Vmax reced tgt"])
    assert is_table_relevant_for_pass("missile_speed_timing", nt) is True


def test_propulsion_relevant_when_table_has_stage_section_rows():
    nt = _fake_normalized(["1st Stage", "2nd Stage", "Booster", "Sustainer"])
    assert is_table_relevant_for_pass("missile_propulsion", nt) is True


def test_radar_timing_relevant_when_table_has_pri_or_pulse_rows():
    nt = _fake_normalized(["PRI", "Pulse Width", "Scan Period"])
    assert is_table_relevant_for_pass("radar_timing", nt) is True


def test_radar_antenna_relevant_when_table_has_gain_or_beamwidth_rows():
    nt = _fake_normalized(["Antenna Gain", "Azimuth Beamwidth", "Elevation Beamwidth"])
    assert is_table_relevant_for_pass("radar_antenna", nt) is True


def test_radar_power_rf_relevant_when_table_has_freq_or_power_rows():
    nt = _fake_normalized(["Operating Frequency", "Peak Power", "ERP"])
    assert is_table_relevant_for_pass("radar_power_rf", nt) is True


def test_radar_modulation_relevant_when_table_has_chirp_or_code_rows():
    nt = _fake_normalized(["Chirp Bandwidth", "Code Length", "Pulses per Dwell"])
    assert is_table_relevant_for_pass("radar_modulation", nt) is True


# --- relevance negative cases ---------------------------------------------

def test_identity_passes_never_relevant():
    """Identity passes are in RAW_ONLY — table relevance returns False even
    if the table has matching rows for some other pass."""
    nt = _fake_normalized(["Max Range", "Min Alt"])
    assert is_table_relevant_for_pass("missile_identity", nt) is False
    assert is_table_relevant_for_pass("radar_identity", nt) is False
    assert is_table_relevant_for_pass("system_links", nt) is False


def test_kinematics_not_relevant_for_radar_table():
    """A table containing only radar rows is not relevant to kinematics."""
    nt = _fake_normalized(["Antenna Gain", "Beamwidth", "PRI"])
    assert is_table_relevant_for_pass("missile_kinematics", nt) is False


def test_radar_antenna_not_relevant_for_missile_kinematics_table():
    nt = _fake_normalized(["Max Range", "Min Alt", "Max Alt"])
    assert is_table_relevant_for_pass("radar_antenna", nt) is False


def test_other_shape_table_never_relevant():
    """Shape.OTHER tables fall through to the raw-markdown render — synth-only
    treatment would be a no-op anyway."""
    nt = _fake_normalized(["Max Range", "Min Alt"], shape=Shape.OTHER)
    assert is_table_relevant_for_pass("missile_kinematics", nt) is False


def test_table_with_no_matching_rows_not_relevant():
    """A normalized table with row labels that don't match any pass alias."""
    nt = _fake_normalized(["Unknown Field 1", "Some Other Label"])
    assert is_table_relevant_for_pass("missile_kinematics", nt) is False


def test_unknown_pass_name_not_relevant():
    nt = _fake_normalized(["Max Range"])
    assert is_table_relevant_for_pass("totally_made_up_pass", nt) is False


def test_case_insensitive_match():
    """Row labels should match aliases case-insensitively."""
    nt = _fake_normalized(["MAX RANGE", "min alt"])
    assert is_table_relevant_for_pass("missile_kinematics", nt) is True


# --- policy invariants -----------------------------------------------------

def test_synth_eligible_and_raw_only_are_disjoint():
    """A pass can't be both synth-eligible and raw-only."""
    assert SYNTH_ELIGIBLE_PASSES.isdisjoint(RAW_ONLY_PASSES)


def test_every_synth_eligible_pass_has_row_aliases():
    """Every pass declared synth-eligible needs at least one row alias,
    otherwise table relevance always returns False and the pass is silently
    raw-only."""
    for pass_name in SYNTH_ELIGIBLE_PASSES:
        assert pass_name in PASS_TABLE_ROW_ALIASES, (
            f"{pass_name} is in SYNTH_ELIGIBLE_PASSES but has no entry in "
            f"PASS_TABLE_ROW_ALIASES — add row aliases or remove from the set."
        )
        assert PASS_TABLE_ROW_ALIASES[pass_name], (
            f"{pass_name} has an empty alias set"
        )


def test_real_sa2_table_kinematics_is_relevant():
    """End-to-end with the actual SA-2 sample fixture: missile_kinematics
    should see the SA-2 spec table as relevant (it has Max Range, Min Alt, etc)."""
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    assert is_table_relevant_for_pass("missile_kinematics", nt) is True


def test_real_sa2_table_airframe_is_relevant():
    """SA-2 table has Length/Diameter/Weight rows → airframe relevant."""
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    assert is_table_relevant_for_pass("missile_airframe", nt) is True


def test_real_sa2_table_radar_passes_not_relevant():
    """SA-2 table has no radar-specific rows → radar passes not relevant."""
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    assert is_table_relevant_for_pass("radar_timing", nt) is False
    assert is_table_relevant_for_pass("radar_antenna", nt) is False


def test_real_sa2_table_identity_passes_not_relevant():
    """Identity passes are blanket-excluded regardless of table content."""
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    assert is_table_relevant_for_pass("missile_identity", nt) is False
    assert is_table_relevant_for_pass("radar_identity", nt) is False
