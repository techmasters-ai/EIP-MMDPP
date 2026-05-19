"""Step 7: precision filters reject auxiliary equipment from radar_identity.

Generic rules — no equipment names. Operate on context labels in
evidence_text near the entity's emitted name. If the context labels the
entity as one of the known non-radar equipment classes, drop it from the
radar_identity output AND emit a diagnostic.

Acceptance per the next-steps plan:
- 5E93 (Power generator) must NOT emit as RADAR_SYSTEM
- 5L22 (Chemical decontamination van / Fuel tank) must NOT emit
- AKKORD (Training emulator) must NOT emit
- Tsikloida (Radio relay van) must NOT emit
- PR-11AM (Transloader) must NOT emit
- Existing radars (Fan Song, Spoon Rest, Side Net) must continue to emit
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

normalize = _EVIDENCE_GATE.normalize_evidence_text


# ===== Unit tests: filter predicate =====

class TestNonRadarContext:
    """The predicate `_is_non_radar_context(name, evidence)` returns True
    when the entity's emitted name is labeled as auxiliary equipment in
    evidence_text. Returns False for genuine radars."""

    def test_power_generator(self):
        ev = normalize("ESP-90 (3 x 5E93) Power generator")
        assert _EVIDENCE_GATE._is_non_radar_context("5E93", ev) is True

    def test_chemical_decontamination_van(self):
        ev = normalize("5L22 Chemical decontamination van")
        assert _EVIDENCE_GATE._is_non_radar_context("5L22", ev) is True

    def test_fuel_tank(self):
        ev = normalize("5L22A Fuel Tank (TG-02)")
        assert _EVIDENCE_GATE._is_non_radar_context("5L22A", ev) is True

    def test_oxidiser_tank(self):
        ev = normalize("5L62A Oxidiser Tank (AK-27P)")
        assert _EVIDENCE_GATE._is_non_radar_context("5L62A", ev) is True

    def test_training_emulator(self):
        ev = normalize("AKKORD Training Emulator (OdAZ-828 semitrailer)")
        assert _EVIDENCE_GATE._is_non_radar_context("AKKORD", ev) is True

    def test_radio_relay_van(self):
        ev = normalize("5Ya61/62/63 Tsikloida Radio relay van")
        assert _EVIDENCE_GATE._is_non_radar_context("Tsikloida", ev) is True

    def test_transporter_transloader(self):
        ev = normalize("PR-11AM Transporter/transloader")
        assert _EVIDENCE_GATE._is_non_radar_context("PR-11AM", ev) is True

    def test_launcher_single_rail(self):
        ev = normalize("SM-90 Launcher, Single Rail")
        assert _EVIDENCE_GATE._is_non_radar_context("SM-90", ev) is True


class TestRadarContext:
    """Genuine radar entities must NOT trigger the filter even if their
    context is dense with adjacent auxiliary mentions."""

    def test_fan_song_engagement_radar_kept(self):
        ev = normalize("RSNA-75/SNR-75 Fan Song Engagement Radar")
        assert _EVIDENCE_GATE._is_non_radar_context("Fan Song", ev) is False

    def test_spoon_rest_acquisition_radar_kept(self):
        ev = normalize("NITEL P-18-2/P-18M Spoon Rest D/E Acquisition Radar")
        assert _EVIDENCE_GATE._is_non_radar_context("Spoon Rest", ev) is False

    def test_side_net_heightfinding_radar_kept(self):
        ev = normalize("PRV-10 Konus / PRV-11 Vershina / Side Net Heightfinding Radars")
        assert _EVIDENCE_GATE._is_non_radar_context("Side Net", ev) is False

    def test_radar_in_radar_van_context_kept(self):
        """`Radar head van` is radar context — must not false-trigger
        the 'van' marker."""
        ev = normalize("SNR-75 PV Cabin / Fan Song Radar head van")
        assert _EVIDENCE_GATE._is_non_radar_context("SNR-75", ev) is False


class TestNotInEvidenceReturnsFalse:
    """If the entity name doesn't appear in evidence_text at all, return
    False (let the LLM emission stand — no information to act on)."""

    def test_entity_not_in_evidence(self):
        ev = normalize("Some other text without this entity name.")
        assert _EVIDENCE_GATE._is_non_radar_context("Nonexistent", ev) is False


# ===== Integration: _postprocess_air_defense_radars drops non-radars =====

class TestPostprocessDropsNonRadars:
    """End-to-end: when the radar_identity LLM emits non-radar entities
    (because the evidence chunk includes a 'S-75 Battery Components'
    table), the postprocess drops them and records a diagnostic."""

    def _postprocess(self, pass_output, evidence):
        return _EVIDENCE_GATE._postprocess_air_defense_radars(pass_output, evidence)

    def test_drops_power_generator(self):
        ev = normalize("Fan Song Engagement Radar. ESP-90 (3 x 5E93) Power generator.")
        pass_output = {
            "radar_systems": [
                {"system_name": "Fan Song", "emitter_function": "FIRE_CONTROL"},
                {"system_name": "5E93", "emitter_function": "FIRE_CONTROL"},
            ],
        }
        out, stats = self._postprocess(pass_output, ev)
        names = [r["system_name"] for r in out["radar_systems"]]
        assert "Fan Song" in names
        assert "5E93" not in names, f"5E93 should be dropped; got {names}"
        # Diagnostic captures the drop with reason
        assert "non_radar_dropped" in stats
        dropped = stats["non_radar_dropped"]
        assert any(d.get("system_name") == "5E93" for d in dropped)

    def test_drops_training_emulator_and_radio_relay(self):
        ev = normalize(
            "Spoon Rest Acquisition Radar. AKKORD Training Emulator. "
            "5Ya61/62/63 Tsikloida Radio relay van."
        )
        pass_output = {
            "radar_systems": [
                {"system_name": "Spoon Rest"},
                {"system_name": "AKKORD"},
                {"system_name": "Tsikloida"},
            ],
        }
        out, stats = self._postprocess(pass_output, ev)
        names = [r["system_name"] for r in out["radar_systems"]]
        assert names == ["Spoon Rest"], (
            f"Only Spoon Rest should survive; got {names}"
        )
        dropped_names = {d.get("system_name") for d in stats.get("non_radar_dropped", [])}
        assert {"AKKORD", "Tsikloida"}.issubset(dropped_names)

    def test_preserves_all_radars_when_none_non_radar(self):
        ev = normalize(
            "RSNA-75/SNR-75 Fan Song Engagement Radar. "
            "P-18 Spoon Rest Acquisition Radar."
        )
        pass_output = {
            "radar_systems": [
                {"system_name": "Fan Song"},
                {"system_name": "Spoon Rest"},
            ],
        }
        out, stats = self._postprocess(pass_output, ev)
        assert len(out["radar_systems"]) == 2
        assert "non_radar_dropped" not in stats
