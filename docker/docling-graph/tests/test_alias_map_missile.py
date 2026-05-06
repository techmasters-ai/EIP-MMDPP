"""Per-pass alias-resolution tests for missile passes (spec §5.5).

Verifies that the headline aliases for each missile sub-pass are populated
correctly. These tests anchor the production behavior — if any of these
break, the synthesizer cannot recover the GT scorecard targets.
"""
import importlib.util
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _facts():
    return _load("docling_graph_service_table_facts", _APP_DIR / "_table_facts.py")


def _aliases():
    return _load("docling_graph_service_alias_map", _APP_DIR / "_alias_map.py")


def _resolve(label: str, section: str | None, pass_name: str) -> str | None:
    am = _aliases()
    tf = _facts()
    return am.ALIAS_MAP.get((tf.normalize_label(label), section, pass_name))


# --- missile_kinematics ----------------------------------------------------

def test_kinematics_max_range_km_aliases():
    assert _resolve("Max Range km", None, "missile_kinematics") == "max_intercept_km"
    assert _resolve("Max Range m", None, "missile_kinematics") == "max_intercept_km"
    assert _resolve("Range", None, "missile_kinematics") == "max_intercept_km"


def test_kinematics_min_range_aliases():
    assert _resolve("Min Range", None, "missile_kinematics") == "min_intercept_km"
    assert _resolve("Min Range km", None, "missile_kinematics") == "min_intercept_km"


def test_kinematics_altitude_aliases():
    """Only the full word 'Altitude' resolves — 'Alt' abbreviation is not in §12b prose."""
    assert _resolve("Max Altitude", None, "missile_kinematics") == "max_altitude_km"
    assert _resolve("Max Altitude km", None, "missile_kinematics") == "max_altitude_km"
    assert _resolve("Min Altitude", None, "missile_kinematics") == "min_altitude_km"
    # "Max Alt" deliberately does NOT resolve (would require §12b prose update).
    assert _resolve("Max Alt", None, "missile_kinematics") is None


def test_kinematics_pass_isolation():
    """Range labels do NOT resolve for other passes."""
    assert _resolve("Max Range km", None, "missile_propulsion") is None
    assert _resolve("Max Range km", None, "missile_airframe") is None


# --- missile_airframe ------------------------------------------------------

def test_airframe_length_aliases():
    assert _resolve("Length", None, "missile_airframe") == "body_length_m"
    assert _resolve("Length mm", None, "missile_airframe") == "body_length_m"
    assert _resolve("Body Length", None, "missile_airframe") == "body_length_m"


def test_airframe_diameter_aliases():
    assert _resolve("Diameter", None, "missile_airframe") == "body_diameter_m"
    assert _resolve("Diameter mm", None, "missile_airframe") == "body_diameter_m"
    assert _resolve("Body Diameter", None, "missile_airframe") == "body_diameter_m"


def test_airframe_total_mass_aliases():
    """Total Weight kg / Weight kg without section context maps to total_mass_kg."""
    assert _resolve("Total Weight kg", None, "missile_airframe") == "total_mass_kg"
    assert _resolve("Weight kg", None, "missile_airframe") == "total_mass_kg"
    assert _resolve("Mass kg", None, "missile_airframe") == "total_mass_kg"


# --- missile_speed_timing --------------------------------------------------

def test_speed_timing_max_speed_aliases():
    assert _resolve("Max Speed m/s", None, "missile_speed_timing") == "max_speed_mps"
    assert _resolve("Max Speed", None, "missile_speed_timing") == "max_speed_mps"


# --- missile_propulsion ----------------------------------------------------

def test_propulsion_booster_mass_kg_under_1st_stage():
    """The headline acceptance case — Weight kg under 1st Stage maps to
    booster_mass_kg only when active pass is missile_propulsion."""
    assert _resolve("Weight kg", "1st Stage", "missile_propulsion") == "booster_mass_kg"
    assert _resolve("Weight kg", "Booster", "missile_propulsion") == "booster_mass_kg"


def test_propulsion_sustain_mass_kg_under_2nd_stage():
    assert _resolve("Weight kg", "2nd Stage", "missile_propulsion") == "sustain_mass_kg"
    assert _resolve("Weight kg", "Sustainer", "missile_propulsion") == "sustain_mass_kg"
    assert _resolve("Weight kg", "Sustain", "missile_propulsion") == "sustain_mass_kg"


def test_propulsion_section_isolation():
    """Weight kg without section context does NOT resolve in propulsion pass —
    must have explicit stage section."""
    assert _resolve("Weight kg", None, "missile_propulsion") is None


def test_propulsion_embedded_label_resolves():
    """The SA-2 PDF puts the section in the label itself: '1st Stage Weight kg'.
    detect_section_context strips the section keyword (Task 10) so the
    resolver sees only the bare label 'Weight kg' with section_ctx='1st Stage'."""
    assert _resolve("Weight kg", "1st Stage", "missile_propulsion") == "booster_mass_kg"


def test_propulsion_burn_time_aliases():
    assert _resolve("Time sec", "1st Stage", "missile_propulsion") == "booster_time_sec"
    assert _resolve("Burn Time", "1st Stage", "missile_propulsion") == "booster_time_sec"
    assert _resolve("Time sec", "2nd Stage", "missile_propulsion") == "sustain_time_sec"


def test_propulsion_thrust_aliases():
    assert _resolve("Thrust", "1st Stage", "missile_propulsion") == "booster_thrust"
    assert _resolve("Thrust", "2nd Stage", "missile_propulsion") == "sustain_thrust"
    assert _resolve("Thrust", "Ejector", "missile_propulsion") == "ejector_thrust"
