"""Tests for Step 5 partial: max_launch_angle_deg evidence support
+ removal of the unconditional hard-clear.

Source statements that must populate `max_launch_angle_deg`:
- "launched the missile at 60 degrees" → 60
- "Max launch angle: 60°"              → 60
- "Launch angle: 75 degrees"           → 75
"""
import importlib.util
import sys
from pathlib import Path

_SERVICE_APP_ROOT = Path(__file__).resolve().parent.parent / "app"

_NUM_EV_SPEC = importlib.util.spec_from_file_location(
    "app._numeric_evidence", _SERVICE_APP_ROOT / "_numeric_evidence.py"
)
_NUM_EV_MOD = importlib.util.module_from_spec(_NUM_EV_SPEC)
sys.modules["app._numeric_evidence"] = _NUM_EV_MOD
assert _NUM_EV_SPEC.loader is not None
_NUM_EV_SPEC.loader.exec_module(_NUM_EV_MOD)

_MODULE_PATH = _SERVICE_APP_ROOT / "evidence_gate.py"
_SPEC = importlib.util.spec_from_file_location("docling_graph_evidence_gate", _MODULE_PATH)
_EVIDENCE_GATE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_EVIDENCE_GATE)


# ===== Evidence parsing =====

def test_launched_at_n_degrees_prose():
    """Dvina pattern: 'launched the missile at 60 degrees' → 60."""
    evidence = _EVIDENCE_GATE.normalize_evidence_text(
        "The S-75's launcher rotated 360 degrees and usually launched the missile at 60 degrees."
    )
    fields = _EVIDENCE_GATE._mechanically_supported_missile_fields(
        evidence, system_name="S-75",
    )
    assert fields.get("max_launch_angle_deg") == 60.0


def test_max_launch_angle_label_phrase():
    """'Max launch angle: 75°' or 'Max launch angle: 75 degrees' → 75."""
    evidence = _EVIDENCE_GATE.normalize_evidence_text(
        "S-75 specifications: Max launch angle: 75 degrees"
    )
    fields = _EVIDENCE_GATE._mechanically_supported_missile_fields(
        evidence, system_name="S-75",
    )
    assert fields.get("max_launch_angle_deg") == 75.0


def test_launch_angle_degree_symbol():
    """'45°' (Unicode degree sign) → 45."""
    evidence = _EVIDENCE_GATE.normalize_evidence_text(
        "S-75 launch elevation 45° from horizontal"
    )
    fields = _EVIDENCE_GATE._mechanically_supported_missile_fields(
        evidence, system_name="S-75",
    )
    assert fields.get("max_launch_angle_deg") == 45.0


def test_no_angle_in_evidence_returns_nothing():
    """No angle in evidence → field is NOT in the returned dict."""
    evidence = _EVIDENCE_GATE.normalize_evidence_text(
        "The S-75 is a surface-to-air missile."
    )
    fields = _EVIDENCE_GATE._mechanically_supported_missile_fields(
        evidence, system_name="S-75",
    )
    assert "max_launch_angle_deg" not in fields


def test_rotation_360_does_not_set_launch_angle():
    """Source phrase 'rotated 360 degrees' refers to launcher azimuth,
    NOT launch angle. Must not populate max_launch_angle_deg.
    (Prefer matching 'launch'-anchored or 'launched'-anchored phrasing.)"""
    evidence = _EVIDENCE_GATE.normalize_evidence_text(
        "The launcher could be transported on four wheels and rotated 360 degrees."
    )
    fields = _EVIDENCE_GATE._mechanically_supported_missile_fields(
        evidence, system_name="S-75",
    )
    # If 360 leaked through, that would be wrong — 360-degree rotation
    # is the azimuth coverage, not the missile launch angle.
    assert fields.get("max_launch_angle_deg") != 360


# ===== Hard-clear removal =====

def test_clear_unsupported_missile_preserves_angle_when_evidence_present():
    """The unconditional hard-clear must NOT null max_launch_angle_deg
    when evidence supports it."""
    evidence = _EVIDENCE_GATE.normalize_evidence_text(
        "The S-75's launcher rotated 360 degrees and usually launched the missile at 60 degrees."
    )
    item = {"system_name": "S-75", "max_launch_angle_deg": 60.0}
    cleared = _EVIDENCE_GATE._clear_unsupported_missile_properties(item, evidence)
    assert item.get("max_launch_angle_deg") == 60.0
    assert "max_launch_angle_deg" not in cleared


def test_clear_unsupported_missile_clears_angle_when_no_evidence():
    """When NO angle phrase is in evidence, the field is cleared. This is
    the same evidence-discipline as other numeric fields."""
    evidence = _EVIDENCE_GATE.normalize_evidence_text(
        "The S-75 is a surface-to-air missile."
    )
    item = {"system_name": "S-75", "max_launch_angle_deg": 60.0}
    cleared = _EVIDENCE_GATE._clear_unsupported_missile_properties(item, evidence)
    assert item.get("max_launch_angle_deg") is None
    assert "max_launch_angle_deg" in cleared


def test_mechanical_override_takes_priority():
    """When the LLM emitted a different value than the evidence supports,
    the mechanical extraction wins (matches existing behavior for
    min/max_intercept_km, min/max_altitude_km, etc.)."""
    evidence = _EVIDENCE_GATE.normalize_evidence_text(
        "S-75 specifications: Max launch angle: 75 degrees"
    )
    item = {"system_name": "S-75", "max_launch_angle_deg": 60.0}  # LLM said 60
    _EVIDENCE_GATE._clear_unsupported_missile_properties(item, evidence)
    # Mechanical evidence wins → 75
    assert item.get("max_launch_angle_deg") == 75.0
