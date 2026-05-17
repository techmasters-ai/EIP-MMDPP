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


def _fake_normalized(
    row_labels: list[str],
    *,
    identity_labels: list[str] | None = None,
    caption: str | None = None,
    shape: Shape = Shape.COLUMN_MAJOR,
) -> SimpleNamespace:
    """Build a minimal stand-in for NormalizedTable.

    - `row_labels`: spec/field rows that go into `cells` (and `rows`).
    - `identity_labels`: identity rows that go into `rows` only (mirrors
      the normalizer's behavior of excluding identity rows from spec cells).
    - `caption`: optional table caption for caption-fallback path.
    """
    identity_labels = identity_labels or []
    cells = [SimpleNamespace(row_label=lbl) for lbl in row_labels]
    rows = (
        [SimpleNamespace(label=lbl) for lbl in identity_labels]
        + [SimpleNamespace(label=lbl) for lbl in row_labels]
    )
    return SimpleNamespace(cells=cells, rows=rows, shape=shape, caption=caption)


# --- relevance positive cases (per pass) -----------------------------------
# All positives require BOTH (a) field row labels matching the pass aliases
# AND (b) identity context for the pass's entity type.

def test_kinematics_relevant_when_table_has_range_or_alt_rows_AND_missile_identity():
    nt = _fake_normalized(
        ["Max Range", "Min Range", "Max Alt", "Min Alt"],
        identity_labels=["Missile Type", "NATO Designation"],
    )
    assert is_table_relevant_for_pass("missile_kinematics", nt) is True


def test_airframe_relevant_when_table_has_length_or_weight_rows_AND_missile_identity():
    nt = _fake_normalized(
        ["Length", "Diameter", "Weight"],
        identity_labels=["Missile Designation"],
    )
    assert is_table_relevant_for_pass("missile_airframe", nt) is True


def test_speed_timing_relevant_when_table_has_vmax_rows_AND_missile_identity():
    nt = _fake_normalized(
        ["Vmax appr tgt", "Vmax reced tgt"],
        identity_labels=["Industry Designation"],
    )
    assert is_table_relevant_for_pass("missile_speed_timing", nt) is True


def test_propulsion_relevant_when_table_has_stage_section_rows_AND_missile_identity():
    nt = _fake_normalized(
        ["1st Stage", "2nd Stage", "Booster", "Sustainer"],
        identity_labels=["Military Designation"],
    )
    assert is_table_relevant_for_pass("missile_propulsion", nt) is True


def test_radar_timing_relevant_when_table_has_pri_or_pulse_rows_AND_radar_identity():
    nt = _fake_normalized(
        ["PRI", "Pulse Width", "Scan Period"],
        identity_labels=["Radar", "Nomenclature"],
    )
    assert is_table_relevant_for_pass("radar_timing", nt) is True


def test_radar_antenna_relevant_when_table_has_gain_or_beamwidth_rows_AND_radar_identity():
    nt = _fake_normalized(
        ["Antenna Gain", "Azimuth Beamwidth", "Elevation Beamwidth"],
        identity_labels=["Emitter"],
    )
    assert is_table_relevant_for_pass("radar_antenna", nt) is True


def test_radar_power_rf_relevant_when_table_has_freq_or_power_rows_AND_radar_identity():
    nt = _fake_normalized(
        ["Operating Frequency", "Peak Power", "ERP"],
        identity_labels=["Radar Type"],
    )
    assert is_table_relevant_for_pass("radar_power_rf", nt) is True


def test_radar_modulation_relevant_when_table_has_chirp_or_code_rows_AND_radar_identity():
    nt = _fake_normalized(
        ["Chirp Bandwidth", "Code Length", "Pulses per Dwell"],
        identity_labels=["ELNOT"],
    )
    assert is_table_relevant_for_pass("radar_modulation", nt) is True


# --- caption fallback (when no identity row but caption identifies type) ---

def test_caption_identifies_missile_when_no_identity_row():
    nt = _fake_normalized(
        ["Max Range", "Max Alt"],
        identity_labels=[],
        caption="Long-range surface-to-air missile performance",
    )
    assert is_table_relevant_for_pass("missile_kinematics", nt) is True


def test_caption_identifies_radar_when_no_identity_row():
    nt = _fake_normalized(
        ["Operating Frequency", "Peak Power"],
        identity_labels=[],
        caption="Phased-array radar transmitter characteristics",
    )
    assert is_table_relevant_for_pass("radar_power_rf", nt) is True


# --- false-positive prevention (per handoff requirements) ------------------

def test_logistics_weight_table_NOT_relevant_to_missile_airframe():
    """A logistics table with `Weight` row but no missile identity context
    must not be classified as missile_airframe."""
    nt = _fake_normalized(
        ["Weight", "Volume", "Container Count"],
        identity_labels=["Shipment ID", "Origin", "Destination"],
        caption="Container shipment manifest",
    )
    assert is_table_relevant_for_pass("missile_airframe", nt) is False


def test_communications_frequency_table_NOT_relevant_to_radar_power_rf():
    """A comms table with `Frequency` row but identifying a Radio Channel
    must not be classified as radar_power_rf."""
    nt = _fake_normalized(
        ["Frequency", "Bandwidth"],
        identity_labels=["Radio Channel", "Callsign"],
        caption="Tactical voice radio channel assignments",
    )
    assert is_table_relevant_for_pass("radar_power_rf", nt) is False


def test_platform_range_table_NOT_relevant_to_missile_kinematics():
    """A platform/vehicle table with `Range` row but no missile identity
    must not be classified as missile_kinematics."""
    nt = _fake_normalized(
        ["Range", "Top Speed"],
        identity_labels=["Vehicle Type", "Tail Number"],
        caption="Aircraft platform performance",
    )
    assert is_table_relevant_for_pass("missile_kinematics", nt) is False


def test_field_rows_only_NO_identity_NO_caption_NOT_relevant():
    """A table with matching field rows but no identity context at all
    (no identity rows AND no caption) must not be classified as relevant."""
    nt = _fake_normalized(
        ["Max Range", "Max Alt", "Min Alt"],
        identity_labels=[],
        caption=None,
    )
    assert is_table_relevant_for_pass("missile_kinematics", nt) is False


# --- relevance negative cases ---------------------------------------------

def test_identity_passes_never_relevant():
    """Identity passes are in RAW_ONLY — table relevance returns False even
    if the table has matching rows + identity context for the other passes."""
    nt = _fake_normalized(
        ["Max Range", "Min Alt"],
        identity_labels=["Missile Type"],
    )
    assert is_table_relevant_for_pass("missile_identity", nt) is False
    assert is_table_relevant_for_pass("radar_identity", nt) is False
    assert is_table_relevant_for_pass("system_links", nt) is False


def test_kinematics_not_relevant_for_radar_table():
    """A table whose identity is a radar (not a missile) is not relevant to
    kinematics even if it has range-like field rows."""
    nt = _fake_normalized(
        ["Antenna Gain", "Beamwidth", "PRI"],
        identity_labels=["Radar", "Nomenclature"],
    )
    assert is_table_relevant_for_pass("missile_kinematics", nt) is False


def test_radar_antenna_not_relevant_for_missile_kinematics_table():
    nt = _fake_normalized(
        ["Max Range", "Min Alt", "Max Alt"],
        identity_labels=["Missile Type"],
    )
    assert is_table_relevant_for_pass("radar_antenna", nt) is False


def test_other_shape_table_never_relevant():
    """Shape.OTHER tables fall through to the raw-markdown render — synth-only
    treatment would be a no-op anyway."""
    nt = _fake_normalized(
        ["Max Range", "Min Alt"],
        identity_labels=["Missile Type"],
        shape=Shape.OTHER,
    )
    assert is_table_relevant_for_pass("missile_kinematics", nt) is False


def test_table_with_no_matching_rows_not_relevant():
    """A normalized table with row labels that don't match any pass alias."""
    nt = _fake_normalized(
        ["Unknown Field 1", "Some Other Label"],
        identity_labels=["Missile Type"],
    )
    assert is_table_relevant_for_pass("missile_kinematics", nt) is False


def test_unknown_pass_name_not_relevant():
    nt = _fake_normalized(
        ["Max Range"],
        identity_labels=["Missile Type"],
    )
    assert is_table_relevant_for_pass("totally_made_up_pass", nt) is False


def test_case_insensitive_match():
    """Row labels and identity hints should match case-insensitively."""
    nt = _fake_normalized(
        ["MAX RANGE", "min alt"],
        identity_labels=["MISSILE TYPE", "NATO DESIGNATION"],
    )
    assert is_table_relevant_for_pass("missile_kinematics", nt) is True


def test_unit_suffix_stripped_from_row_labels():
    """`Max Range (m)` should normalize to `max range` and match the alias."""
    nt = _fake_normalized(
        ["Max Range (m)", "Min Alt [km]", "Max Alt km"],
        identity_labels=["Missile Type"],
    )
    assert is_table_relevant_for_pass("missile_kinematics", nt) is True


# --- Rec 3 part C — radar_power_rf relevance test cases --------------------
# Per the handoff: these specific cases must hold to prevent false-positive
# synth rewriting on cross-domain tables that happen to contain frequency-
# like row labels.

def test_radar_power_rf_relevant_with_radar_identity_and_frequency():
    """Radar table with `Radar` identity hint + RF field row → relevant."""
    nt = _fake_normalized(
        ["Frequency", "Bandwidth"],
        identity_labels=["Radar"],
    )
    assert is_table_relevant_for_pass("radar_power_rf", nt) is True


def test_radar_power_rf_relevant_with_emitter_identity_and_peak_power():
    """Radar table with `Emitter` identity hint + `Peak Power` row → relevant."""
    nt = _fake_normalized(
        ["Peak Power", "Bandwidth"],
        identity_labels=["Emitter"],
    )
    assert is_table_relevant_for_pass("radar_power_rf", nt) is True


def test_missile_table_with_frequency_row_NOT_relevant_to_radar_power_rf():
    """Missile-identified table with a `Frequency` row → not relevant.
    Prevents missile launch-detection-frequency type tables (or radar-
    cued missile freq tables) from triggering radar_power_rf rewriting."""
    nt = _fake_normalized(
        ["Frequency", "Bandwidth"],
        identity_labels=["Missile Type", "NATO Designation"],
    )
    assert is_table_relevant_for_pass("radar_power_rf", nt) is False


def test_comms_table_with_frequency_but_no_radar_identity_NOT_relevant():
    """Communications channel/callsign table with Frequency rows but no
    radar identity → not relevant. Prevents tactical-voice-radio tables
    from polluting radar_power_rf."""
    nt = _fake_normalized(
        ["Frequency", "Bandwidth"],
        identity_labels=["Radio Channel", "Callsign", "Network ID"],
        caption="Tactical voice radio assignments",
    )
    assert is_table_relevant_for_pass("radar_power_rf", nt) is False


def test_radar_identity_only_table_with_no_rf_rows_NOT_relevant_to_radar_power_rf():
    """A radar-identified table that contains only identity rows + non-RF
    field rows (e.g. only antenna data, no power/freq) → not relevant
    to radar_power_rf. Per-pass alias narrowness."""
    nt = _fake_normalized(
        ["Antenna Gain", "Beamwidth"],
        identity_labels=["Radar", "Emitter"],
    )
    # Antenna-shaped → relevant to radar_antenna, NOT radar_power_rf
    assert is_table_relevant_for_pass("radar_antenna", nt) is True
    assert is_table_relevant_for_pass("radar_power_rf", nt) is False


def test_electronics_specifications_with_freq_NOT_relevant_to_radar_power_rf():
    """Generic electronics-specifications table that mentions Frequency
    (e.g., crystal oscillator specs) without ANY radar identity context →
    not relevant. Guards against the broadest false-positive class."""
    nt = _fake_normalized(
        ["Operating Frequency", "Power Consumption"],
        identity_labels=["Component Part Number", "Manufacturer"],
        caption="Crystal oscillator specifications",
    )
    assert is_table_relevant_for_pass("radar_power_rf", nt) is False


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
    """End-to-end with the actual sample fixture: missile_kinematics should
    see the spec table as relevant — has both range/alt rows AND missile
    identity rows (Industry / Military / NATO Designation, Missile Type)."""
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    assert is_table_relevant_for_pass("missile_kinematics", nt) is True


def test_real_sa2_table_airframe_is_relevant():
    """Sample table has Length/Diameter/Weight rows + missile identity → relevant."""
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    assert is_table_relevant_for_pass("missile_airframe", nt) is True


def test_real_sa2_table_radar_passes_not_relevant():
    """Sample table is missile-identified — radar passes not relevant even
    though row labels like `Range` might overlap with radar field aliases."""
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    assert is_table_relevant_for_pass("radar_timing", nt) is False
    assert is_table_relevant_for_pass("radar_antenna", nt) is False


def test_real_sa2_table_identity_passes_not_relevant():
    """Identity passes are blanket-excluded regardless of table content."""
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    assert is_table_relevant_for_pass("missile_identity", nt) is False
    assert is_table_relevant_for_pass("radar_identity", nt) is False
