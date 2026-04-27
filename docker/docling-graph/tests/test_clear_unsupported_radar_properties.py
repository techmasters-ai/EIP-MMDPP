"""Regression test for _clear_unsupported_radar_properties (spec §4.8).

The function was unconditionally nulling 18 numeric fields. After the
refactor, numeric values that appear in evidence_text (with same-unit-
suffix variants) are preserved; only unsupported values are nulled.
"""
import importlib.util
import pathlib
import sys

_SERVICE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "app"


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# Load the docling-graph app's _numeric_evidence first under "app._numeric_evidence"
# so the file-path-loaded evidence_gate.py's `from app._numeric_evidence import ...`
# resolves to it (mirrors the test_auto_field_evidence.py pattern).
_load("app._numeric_evidence", _SERVICE_ROOT / "_numeric_evidence.py")
_eg = _load("_dgp_evidence_gate", _SERVICE_ROOT / "evidence_gate.py")
_clear = _eg._clear_unsupported_radar_properties


def test_supported_numeric_is_preserved():
    item = {"system_name": "Fan Song", "gain_dbi": 35.0}
    evidence = "The Fan Song antenna gain is 35 dBi nominal."
    cleared = _clear(item, evidence)
    assert "gain_dbi" not in cleared, f"gain_dbi should be preserved; cleared={cleared}"
    assert item["gain_dbi"] == 35.0


def test_unsupported_numeric_is_nulled():
    item = {"system_name": "Fan Song", "gain_dbi": 9999.0}
    evidence = "The Fan Song antenna gain is 35 dBi nominal."
    cleared = _clear(item, evidence)
    assert "gain_dbi" in cleared
    assert item["gain_dbi"] is None


def test_supported_value_with_unit_suffix():
    item = {"system_name": "Fan Song", "nominal_rf_mhz": 3000.0}
    evidence = "Fan Song operates at 3000 MHz."
    cleared = _clear(item, evidence)
    assert "nominal_rf_mhz" not in cleared


def test_text_field_branch_unchanged():
    """Text-field clearing must still work — the exact-quote check is preserved."""
    item = {"system_name": "Fan Song", "nomenclature": "5N62E"}
    evidence = "Fan Song E (5N62E) — Soviet Fire-Control Radar"
    cleared = _clear(item, evidence)
    assert "nomenclature" not in cleared
    assert item["nomenclature"] == "5N62E"


def test_text_field_unsupported_still_nulled():
    item = {"system_name": "Fan Song", "nomenclature": "ABC-999"}
    evidence = "Fan Song operates at 3000 MHz."
    cleared = _clear(item, evidence)
    assert "nomenclature" in cleared
    assert item["nomenclature"] is None


def test_evidence_gate_radar_fields_matches_field_groups():
    """Drift guard: EVIDENCE_GATE_RADAR_FIELDS must equal the union of
    all non-identity, non-text fields across the 4 numeric/parameter
    sub-pass groups (power_rf, antenna, timing, modulation) plus
    'confidence'.

    Set equality (not substring match) — catches both missing fields
    AND extra fields. If a new field is added to RADAR_FIELD_GROUPS
    without updating the constant, this test fails immediately.

    Text-typed fields in those groups (dwell_time, inter_pulse,
    intra_pulse_mop) are handled by the exact-quote branch of
    _clear_unsupported_radar_properties, not the numeric
    value_is_supported_by_text branch, so they are excluded from
    EVIDENCE_GATE_RADAR_FIELDS by design. They live alongside the
    other identity-side text fields (nomenclature, elnot, asrd, etc.)
    in the function's local exact_text_fields tuple.
    """
    EVIDENCE_GATE_RADAR_FIELDS = _eg.EVIDENCE_GATE_RADAR_FIELDS
    from ontology_bundles.air_defense_v3.extraction_schemas._field_groups import (
        RADAR_FIELD_GROUPS,
    )

    # Text-typed fields handled by the exact-quote branch, not the numeric branch.
    text_fields_in_numeric_groups = {"dwell_time", "inter_pulse", "intra_pulse_mop"}

    expected = (
        set(RADAR_FIELD_GROUPS["radar_power_rf"])
        | set(RADAR_FIELD_GROUPS["radar_antenna"])
        | set(RADAR_FIELD_GROUPS["radar_timing"])
        | set(RADAR_FIELD_GROUPS["radar_modulation"])
    ) - {"system_name"} - text_fields_in_numeric_groups
    expected.add("confidence")

    actual = set(EVIDENCE_GATE_RADAR_FIELDS)
    missing = expected - actual
    extra = actual - expected
    assert missing == set() and extra == set(), (
        f"EVIDENCE_GATE_RADAR_FIELDS drift detected.\n"
        f"  missing from constant (would be silently nulled): {sorted(missing)}\n"
        f"  extra in constant (not in any field group): {sorted(extra)}\n"
        f"Update either RADAR_FIELD_GROUPS or the constant."
    )
