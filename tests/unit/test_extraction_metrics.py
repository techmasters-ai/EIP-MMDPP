"""Tests for `app.services.extraction_metrics` — categorized fill metrics.

Validates that radar_power_rf metric output distinguishes RF schema fills
from extra out-of-schema crumbs (the false-positive that prompted
Recommendation 3 part A).
"""
from __future__ import annotations

from app.services.extraction_metrics import (
    categorize_entity_fills,
    declared_schema_fields,
    summarize_pass_metrics,
)


# --- declared_schema_fields contract ----------------------------------------

def test_radar_power_rf_declares_only_rf_fields():
    """radar_power_rf schema is intentionally narrow — only RF carrier +
    transmit power fields. emitter_function / nomenclature MUST NOT
    appear in the declared set."""
    non_id, identity = declared_schema_fields("radar_power_rf")
    assert non_id == {"erp_dbw", "tx_peak_power_kw", "nominal_rf_mhz"}
    assert identity == {"system_name"}
    # Field-category isolation: identity ∩ non_identity = ∅
    assert non_id & identity == set()


def test_radar_identity_declares_role_fields():
    non_id, identity = declared_schema_fields("radar_identity")
    assert "emitter_function" in non_id
    assert "nomenclature" in non_id
    assert "scan_type" in non_id
    assert identity == {"system_name"}


def test_missile_kinematics_declares_only_engagement_envelope_fields():
    non_id, _ = declared_schema_fields("missile_kinematics")
    assert non_id == {
        "min_intercept_km", "max_intercept_km",
        "min_altitude_km", "max_altitude_km",
        "max_launch_angle_deg",
    }


def test_unknown_pass_returns_empty_schema_fields():
    non_id, identity = declared_schema_fields("totally_made_up_pass")
    assert non_id == set()
    assert identity == {"system_name"}  # always-identity fallback


# --- categorize_entity_fills per-entity behavior ---------------------------

def test_radar_power_rf_extra_fields_categorized_as_extra():
    """A radar_power_rf entity emitted with extra emitter_function +
    nomenclature crumbs must categorize those as `extra_fills`, NOT as
    `schema_fills` — that's the whole point of Rec 3 part A."""
    entity = {
        "system_name": "R1",
        "erp_dbw": 60.0,            # schema
        "tx_peak_power_kw": 600.0,  # schema
        "nominal_rf_mhz": None,     # not filled
        "emitter_function": "FIRE_CONTROL",  # EXTRA — merge crumb
        "nomenclature": "AN/MPQ-65",         # EXTRA — merge crumb
    }
    result = categorize_entity_fills(entity, "radar_power_rf")
    assert result["schema_fills"] == 2     # erp_dbw + tx_peak_power_kw
    assert result["identity_fills"] == 1   # system_name
    assert result["extra_fills"] == 2      # emitter_function + nomenclature
    assert result["schema_total_possible"] == 3


def test_all_schema_fields_filled():
    entity = {
        "system_name": "R1",
        "erp_dbw": 60.0,
        "tx_peak_power_kw": 600.0,
        "nominal_rf_mhz": 2900.0,
    }
    result = categorize_entity_fills(entity, "radar_power_rf")
    assert result["schema_fills"] == 3
    assert result["extra_fills"] == 0


def test_empty_string_and_none_count_as_not_filled():
    entity = {
        "system_name": "R1",
        "erp_dbw": None,
        "tx_peak_power_kw": "",
        "nominal_rf_mhz": [],
        "emitter_function": {},
    }
    result = categorize_entity_fills(entity, "radar_power_rf")
    assert result["schema_fills"] == 0
    assert result["identity_fills"] == 1
    assert result["extra_fills"] == 0


def test_private_underscore_fields_ignored():
    """Fields starting with `_` (Pydantic internals, etc.) don't count."""
    entity = {"system_name": "R1", "_internal": "anything", "erp_dbw": 50.0}
    result = categorize_entity_fills(entity, "radar_power_rf")
    assert result["extra_fills"] == 0
    assert result["schema_fills"] == 1


# --- summarize_pass_metrics aggregate behavior ----------------------------

def test_summarize_radar_power_rf_separates_rf_from_extras():
    """Models the actual SA-2 latest-run shape: 42 entities, 0 RF fills,
    24 extra fills on emitter_function + nomenclature. The metric MUST
    report schema_fills_total=0 — not 24, which would falsely claim
    RF extraction is working."""
    entities = []
    # 22 with emitter_function only (no RF)
    for i in range(22):
        entities.append({
            "system_name": f"R{i}",
            "emitter_function": "SEARCH",
        })
    # 2 with emitter_function + nomenclature (no RF)
    for i in range(22, 24):
        entities.append({
            "system_name": f"R{i}",
            "emitter_function": "SEARCH",
            "nomenclature": "AN/MPQ-65",
        })
    # 18 with system_name only
    for i in range(24, 42):
        entities.append({"system_name": f"R{i}"})

    summary = summarize_pass_metrics(entities, "radar_power_rf")
    assert summary["entity_count"] == 42
    # The headline: NO RF fields filled, regardless of extra crumbs
    assert summary["schema_fills_total"] == 0
    assert summary["schema_fill_rate"] == 0.0
    # All system_name's filled
    assert summary["identity_fills_total"] == 42
    # Extras: 22 emitter_function + 2 emitter_function + 2 nomenclature = 26
    assert summary["extra_fills_total"] == 26
    # Per-field tally for the actual RF fields
    assert summary["per_field"] == {
        "erp_dbw": 0,
        "tx_peak_power_kw": 0,
        "nominal_rf_mhz": 0,
    }


def test_summarize_handles_empty_entity_list():
    summary = summarize_pass_metrics([], "radar_power_rf")
    assert summary["entity_count"] == 0
    assert summary["schema_fills_total"] == 0
    assert summary["schema_fill_rate"] == 0.0


def test_summarize_handles_real_partial_fills():
    """Some RF data present + some extras."""
    entities = [
        {"system_name": "R1", "erp_dbw": 50.0},
        {"system_name": "R2", "erp_dbw": 55.0, "tx_peak_power_kw": 1000.0},
        {"system_name": "R3", "nominal_rf_mhz": 2900.0},
        {"system_name": "R4", "emitter_function": "FIRE_CONTROL"},  # extras only
    ]
    summary = summarize_pass_metrics(entities, "radar_power_rf")
    assert summary["entity_count"] == 4
    assert summary["schema_fills_total"] == 4  # 1 + 2 + 1 + 0
    assert summary["extra_fills_total"] == 1   # R4's emitter_function
    assert summary["per_field"] == {
        "erp_dbw": 2,
        "tx_peak_power_kw": 1,
        "nominal_rf_mhz": 1,
    }
    # Fill rate: 4 / (4 * 3) = 0.333
    assert summary["schema_fill_rate"] == 0.3333
