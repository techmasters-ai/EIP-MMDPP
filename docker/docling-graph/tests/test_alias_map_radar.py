"""Per-pass alias-resolution tests for radar passes (spec §5.5)."""
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


# --- radar_power_rf --------------------------------------------------------

def test_radar_power_rf_frequency():
    assert _resolve("Frequency MHz", None, "radar_power_rf") == "nominal_rf_mhz"
    assert _resolve("Operating Frequency", None, "radar_power_rf") == "nominal_rf_mhz"
    assert _resolve("Carrier Frequency", None, "radar_power_rf") == "nominal_rf_mhz"


def test_radar_power_rf_peak_power():
    assert _resolve("Peak Power", None, "radar_power_rf") == "tx_peak_power_kw"
    assert _resolve("Tx Power", None, "radar_power_rf") == "tx_peak_power_kw"


# --- radar_timing ----------------------------------------------------------

def test_radar_timing_pri_aliases():
    assert _resolve("PRI", None, "radar_timing") == "nominal_pri_usec"
    assert _resolve("Pulse Repetition Interval", None, "radar_timing") == "nominal_pri_usec"


def test_radar_timing_pulse_width():
    assert _resolve("PW", None, "radar_timing") == "nominal_pd_usec"
    assert _resolve("Pulse Width", None, "radar_timing") == "nominal_pd_usec"
    assert _resolve("Pulse Duration", None, "radar_timing") == "nominal_pd_usec"


# --- radar_antenna ---------------------------------------------------------

def test_radar_antenna_gain():
    assert _resolve("Antenna Gain", None, "radar_antenna") == "gain_dbi"


def test_radar_antenna_beamwidth():
    assert _resolve("Azimuth Beamwidth", None, "radar_antenna") == "beamwidth_az_deg"
    assert _resolve("Elevation Beamwidth", None, "radar_antenna") == "beamwidth_el_deg"


# --- radar_modulation ------------------------------------------------------

def test_radar_modulation_chirp_bandwidth():
    assert _resolve("Chirp Bandwidth", None, "radar_modulation") == "frequency_excursion_mhz"


# --- Pass isolation --------------------------------------------------------

def test_radar_pass_isolation():
    """Radar labels do NOT resolve in missile passes."""
    assert _resolve("PRI", None, "missile_propulsion") is None
    assert _resolve("Antenna Gain", None, "missile_kinematics") is None
