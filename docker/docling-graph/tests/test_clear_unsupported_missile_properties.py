"""Regression test for _clear_unsupported_missile_properties (spec §4.8 missile pattern).

The function preserves these behaviors:
1. Mechanical-support extraction ("WEIGHT: X LBS" → total_mass_kg via lb→kg
   plus synth-table-block "Min Alt: 3000" / "Max Alt: 30000" → km).
2. Mechanical override for 5 fields: min/max_intercept_km,
   min/max_altitude_km, total_mass_kg.
3. Unconditional-null for 2 fields (max_launch_angle_deg, missile_photo).
   2026-05-16: min_altitude_km removed from the unconditional-null set;
   it now goes through the mechanical-support + evidence-verify path so
   `Min Alt: 3000` in a synth table block can fill it (1000 m → 1.0 km).

The CHANGED behavior: the strict_null_fields tuple loop is replaced with
value_is_supported_by_text verification — values that appear in evidence
are preserved instead of unconditionally nulled.
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


# Pre-register _numeric_evidence under app._numeric_evidence so the
# file-path-loaded evidence_gate.py's `from app._numeric_evidence import ...`
# resolves to it (mirrors radar's pattern in test_clear_unsupported_radar_properties.py).
_load("app._numeric_evidence", _SERVICE_ROOT / "_numeric_evidence.py")
_eg = _load("_dgp_evidence_gate", _SERVICE_ROOT / "evidence_gate.py")
_clear = _eg._clear_unsupported_missile_properties


def test_supported_numeric_is_preserved():
    """body_length_m=7.5 with evidence "length 7.5 m" → preserved."""
    item = {"system_name": "5V55K", "body_length_m": 7.5}
    evidence = "The 5V55K missile body length is 7.5 m."
    cleared = _clear(item, evidence)
    assert "body_length_m" not in cleared, f"should preserve; cleared={cleared}"
    assert item["body_length_m"] == 7.5


def test_unsupported_numeric_is_nulled():
    """body_length_m=999.0 with evidence "length 7.5 m" → nulled."""
    item = {"system_name": "5V55K", "body_length_m": 999.0}
    evidence = "The 5V55K missile body length is 7.5 m."
    cleared = _clear(item, evidence)
    assert "body_length_m" in cleared
    assert item["body_length_m"] is None


def test_mechanical_override_path():
    """LLM emits total_mass_kg=9999, evidence has "WEIGHT: 2300 LBS" → mechanical override wins."""
    item = {"system_name": "5V55K", "total_mass_kg": 9999.0}
    evidence = "WEIGHT: 2300 LBS"
    cleared = _clear(item, evidence)
    # Mechanical conversion: 2300 / 2.205 ≈ 1043.1
    assert item["total_mass_kg"] is not None
    assert 1040 <= item["total_mass_kg"] <= 1046, (
        f"expected ~1043.1 from mechanical conversion; got {item['total_mass_kg']}"
    )


def test_mechanical_override_absent_supported():
    """No mechanical extract; LLM value supported in evidence → preserved via verification branch."""
    item = {"system_name": "5V55K", "max_intercept_km": 43.0}
    evidence = "5V55K maximum intercept range is 43 km."
    cleared = _clear(item, evidence)
    assert "max_intercept_km" not in cleared, f"should preserve; cleared={cleared}"
    assert item["max_intercept_km"] == 43.0


def test_mechanical_override_absent_unsupported():
    """No mechanical extract; LLM value NOT in evidence → nulled."""
    item = {"system_name": "5V55K", "max_intercept_km": 999.0}
    evidence = "5V55K maximum intercept range is 43 km."
    cleared = _clear(item, evidence)
    assert "max_intercept_km" in cleared
    assert item["max_intercept_km"] is None


def test_cross_unit_tonnes_to_kg():
    """Cross-unit conversion (Option A from cross-unit-conversion-followup-todo.md).

    The missile postprocessor's mechanical-override path for total_mass_kg
    falls through to value_is_supported_by_text when no "WEIGHT: X LBS"
    match exists. After Option A, "1.5 tonnes" in source text is now a
    valid alternate for total_mass_kg=1500 — the cross-unit candidate
    "1.5 tonnes" appears in candidate list.
    """
    item = {"system_name": "5V55K", "total_mass_kg": 1500.0}
    evidence = "5V55K total launch weight 1.5 tonnes"
    cleared = _clear(item, evidence)
    assert "total_mass_kg" not in cleared, (
        f"total_mass_kg=1500 should be preserved when source says '1.5 tonnes'; "
        f"cleared={cleared}"
    )
    assert item["total_mass_kg"] == 1500.0


def test_cross_unit_meters_to_kilometers():
    """Cross-unit case for missile range fields stated in meters."""
    item = {"system_name": "5V55K", "max_intercept_km": 43.0}
    evidence = "5V55K maximum intercept range is 43000 m"
    cleared = _clear(item, evidence)
    assert "max_intercept_km" not in cleared, (
        f"max_intercept_km=43 should be preserved when source says '43000 m'; "
        f"cleared={cleared}"
    )
    assert item["max_intercept_km"] == 43.0


def test_text_field_preserved_by_exact_branch():
    """nomenclature stated verbatim → preserved."""
    item = {"system_name": "5V55K", "nomenclature": "5V55K"}
    evidence = "5V55K (formal designation 5V55K) is a Soviet SAM."
    cleared = _clear(item, evidence)
    assert "nomenclature" not in cleared
    assert item["nomenclature"] == "5V55K"


def test_max_launch_angle_deg_preserved_when_evidence_supports():
    """Step 5 (2026-05-19): max_launch_angle_deg evidence parser now reads
    anchored angle phrases. The Session 1 unconditional-null contract was
    superseded — explicit launch-angle phrases preserve the value."""
    item = {"system_name": "5V55K", "max_launch_angle_deg": 85.0}
    evidence = "5V55K maximum launch angle: 85 degrees."
    cleared = _clear(item, evidence)
    assert "max_launch_angle_deg" not in cleared
    assert item["max_launch_angle_deg"] == 85.0


def test_max_launch_angle_deg_still_clears_when_no_evidence():
    """When evidence has NO anchored launch-angle phrase, the field still
    clears — same evidence-discipline as other numeric fields."""
    item = {"system_name": "5V55K", "max_launch_angle_deg": 85.0}
    evidence = "5V55K is a Russian SAM missile."  # no launch-angle phrase
    cleared = _clear(item, evidence)
    assert "max_launch_angle_deg" in cleared
    assert item["max_launch_angle_deg"] is None


# --- 2026-05-16: min_altitude_km mechanical support tests -----------------
# All fixtures include both (a) an ENTITY block with a missile-identity row
# matching the item's system_name (entity-scoping requirement) AND (b) the
# UNITS preamble (unit-evidence gate for bare-number inference). Generic
# fixture names — no document-specific identities.

def test_min_altitude_km_from_synth_block_metres():
    """`Min Alt: 1000` in a synth table block with the SI UNITS preamble
    converts to 1.0 km via the bare-number SI-base assumption."""
    item = {"system_name": "M1"}
    evidence = (
        "UNITS: Numeric values in this block are in SI base units\n"
        "ENTITY:\n- Missile Type: M1\n"
        "GENERAL:\n- Min Alt: 1000\n- Max Alt: 30000\n"
    )
    cleared = _clear(item, evidence)
    assert "min_altitude_km" not in cleared
    assert item["min_altitude_km"] == 1.0
    assert item["max_altitude_km"] == 30.0


def test_min_altitude_km_sub_kilometre_value():
    """Min Alt: 50 metres → 0.05 km (smallest values we expect from SAM specs)."""
    item = {"system_name": "M2"}
    evidence = (
        "UNITS: Numeric values in this block are in SI base units\n"
        "ENTITY:\n- Missile Type: M2\n"
        "GENERAL:\n- Min Alt: 50\n"
    )
    cleared = _clear(item, evidence)
    assert "min_altitude_km" not in cleared
    assert item["min_altitude_km"] == 0.05


def test_min_altitude_km_explicit_km_unit():
    """`Min Altitude: 3 km` → 3.0 with an explicit `km` suffix — no UNITS
    preamble needed because the unit is on the row itself."""
    item = {"system_name": "M3"}
    evidence = (
        "ENTITY:\n- Missile Type: M3\n"
        "GENERAL:\n- Min Altitude: 3 km\n"
    )
    cleared = _clear(item, evidence)
    assert "min_altitude_km" not in cleared
    assert item["min_altitude_km"] == 3.0


def test_min_altitude_km_no_longer_unconditionally_nulled():
    """The exact regression we're fixing: a value emitted by the LLM that is
    supported by evidence (entity-scoped + unit-gated) should be preserved,
    not hard-cleared."""
    item = {"system_name": "M4", "min_altitude_km": 1.0}
    evidence = (
        "UNITS: Numeric values in this block are in SI base units\n"
        "ENTITY:\n- Missile Type: M4\n"
        "GENERAL:\n- Min Alt: 1000\n"
    )
    cleared = _clear(item, evidence)
    assert "min_altitude_km" not in cleared
    assert item["min_altitude_km"] == 1.0


def test_min_altitude_km_unsupported_value_still_clears():
    """If the LLM emits a min_altitude_km value that has NO matching Min Alt
    row in evidence, it should still clear (evidence gate works both ways)."""
    item = {"system_name": "Made-Up", "min_altitude_km": 99.0}
    evidence = "Some unrelated prose about a missile but no altitude data."
    cleared = _clear(item, evidence)
    assert "min_altitude_km" in cleared
    assert item["min_altitude_km"] is None


# --- Generalization tests: entity scoping + unit-evidence gating ----------

def test_min_altitude_km_bare_no_units_preamble_NOT_inferred():
    """Bare `Min Alt: 1000` with NO units preamble and NO km/m suffix
    must NOT be mechanically filled — no unit evidence to authorize the
    metres assumption."""
    item = {"system_name": "M5"}
    evidence = (
        "ENTITY:\n- Missile Type: M5\n"
        "GENERAL:\n- Min Alt: 1000\n"
    )
    cleared = _clear(item, evidence)
    # min_altitude_km should NOT be in item (not mechanically filled) AND
    # not in `cleared` (because nothing was emitted to clear).
    assert item.get("min_altitude_km") is None


def test_min_altitude_km_entity_scoping_no_cross_contamination():
    """Two missiles in the same evidence text — each should get its own
    `Min Alt`, not the first one applied to both."""
    evidence = (
        "UNITS: Numeric values in this block are in SI base units\n"
        "ENTITY:\n- Missile Type: M-ALPHA\n"
        "GENERAL:\n- Min Alt: 1000\n"
        "ENTITY:\n- Missile Type: M-BETA\n"
        "GENERAL:\n- Min Alt: 100\n"
    )
    alpha = {"system_name": "M-ALPHA"}
    beta  = {"system_name": "M-BETA"}
    _clear(alpha, evidence)
    _clear(beta,  evidence)
    assert alpha["min_altitude_km"] == 1.0
    assert beta["min_altitude_km"] == 0.1, (
        f"cross-contamination: beta should have 0.1 (from its own block), "
        f"got {beta.get('min_altitude_km')}"
    )


def test_min_altitude_km_no_entity_block_no_inference():
    """If the entity has no matching synth-block in evidence (e.g., a missile
    extracted from prose), mechanical support is skipped entirely — no
    false positive from a different entity's Min Alt row."""
    item = {"system_name": "M-PROSE-ONLY"}
    evidence = (
        "UNITS: Numeric values in this block are in SI base units\n"
        "ENTITY:\n- Missile Type: M-OTHER\n"
        "GENERAL:\n- Min Alt: 1000\n"
    )
    _clear(item, evidence)
    assert item.get("min_altitude_km") is None


def test_min_altitude_km_alias_resolves_entity_block():
    """If `system_name` doesn't match the identity row but `nomenclature`
    does, the entity block still resolves via the alias path."""
    item = {"system_name": "Canonical-X", "nomenclature": "Alt-Y"}
    evidence = (
        "UNITS: Numeric values in this block are in SI base units\n"
        "ENTITY:\n- Missile Type: Alt-Y\n"
        "GENERAL:\n- Min Alt: 500\n"
    )
    _clear(item, evidence)
    assert item.get("min_altitude_km") == 0.5


def test_min_altitude_km_production_normalized_evidence():
    """Evidence that's been through normalize_evidence_text() (uppercased,
    whitespace-collapsed, no newlines) still parses correctly."""
    raw = (
        "UNITS: Numeric values in this block are in SI base units\n"
        "ENTITY:\n- Missile Type: M6\n"
        "GENERAL:\n- Min Alt: 1000\n- Max Alt: 30000\n"
    )
    item = {"system_name": "M6"}
    _clear(item, _eg.normalize_evidence_text(raw))
    assert item.get("min_altitude_km") == 1.0
    assert item.get("max_altitude_km") == 30.0


def test_evidence_gate_missile_fields_matches_field_groups():
    """Drift guard: EVIDENCE_GATE_MISSILE_FIELDS must equal the
    evidence-verified subset of MISSILE_FIELD_GROUPS plus 'confidence'.

    Excluded from the constant on purpose:
    - missile_kinematics: most go through mechanical-override (min/max_intercept_km,
      max_altitude_km), and min_altitude_km/max_launch_angle_deg are
      unconditionally nulled.
    - missile_airframe `total_mass_kg`: mechanical-override path.
    - missile_guidance string fields (guidance_type, seeker_type): enum-explicit branch.
    - `missile_photo`: unconditionally nulled per Session 1 contract.
    - missile_propulsion string thrusts: exact-text branch.
    """
    EVIDENCE_GATE_MISSILE_FIELDS = _eg.EVIDENCE_GATE_MISSILE_FIELDS
    from ontology_bundles.air_defense_v3.extraction_schemas._field_groups import (
        MISSILE_FIELD_GROUPS,
    )

    expected = (
        set(MISSILE_FIELD_GROUPS["missile_airframe"])
        | set(MISSILE_FIELD_GROUPS["missile_speed_timing"])
        | set(MISSILE_FIELD_GROUPS["missile_propulsion"])
    ) - {"system_name"}
    expected.discard("total_mass_kg")
    expected.discard("ejector_thrust")
    expected.discard("booster_thrust")
    expected.discard("sustain_thrust")
    expected.add("confidence")

    actual = set(EVIDENCE_GATE_MISSILE_FIELDS)
    missing = expected - actual
    extra = actual - expected
    assert missing == set() and extra == set(), (
        f"EVIDENCE_GATE_MISSILE_FIELDS drift detected.\n"
        f"  missing from constant (would be silently nulled): {sorted(missing)}\n"
        f"  extra in constant (not in any field group's verification subset): {sorted(extra)}\n"
    )
