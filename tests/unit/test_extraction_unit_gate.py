"""G1 unit-signature gate — TDD tests (Task 5).

Tests written BEFORE implementation per TDD contract.
Run:
    python3 -m pytest tests/unit/test_extraction_unit_gate.py -v
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from app.services.field_value_grounding import nfc

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "docker" / "docling-graph" / "tests" / "fixtures" / "unit_matcher_cases.json"
)
_CASES = json.loads(_FIXTURE.read_text())


# ---------------------------------------------------------------------------
# 1. signature_for_fields
# ---------------------------------------------------------------------------

class TestSignatureForFields:

    def test_signature_for_fields_unions_units(self):
        """Fields with unit suffixes contribute their synonyms; system_name
        (non-unit) contributes nothing."""
        from app.services.extraction_unit_gate import signature_for_fields

        sig = signature_for_fields(["max_intercept_km", "max_launch_angle_deg", "system_name"])
        assert "km" in sig and "км" in sig and "deg" in sig and "°" in sig
        assert "kw" not in sig

    def test_signature_deduplicates(self):
        """Two fields with the same unit suffix yield each synonym once."""
        from app.services.extraction_unit_gate import signature_for_fields

        sig = signature_for_fields(["range_km", "max_intercept_km"])
        assert sig.count("km") == 1

    def test_signature_is_sorted_tuple(self):
        """Return type is a sorted tuple (stable, hashable, reproducible)."""
        from app.services.extraction_unit_gate import signature_for_fields

        sig = signature_for_fields(["max_intercept_km", "erp_dbw", "nominal_rf_mhz"])
        assert isinstance(sig, tuple)
        assert list(sig) == sorted(sig)

    def test_empty_fields_gives_empty_tuple(self):
        from app.services.extraction_unit_gate import signature_for_fields

        assert signature_for_fields([]) == ()
        assert signature_for_fields(["system_name", "confidence"]) == ()

    def test_unknown_field_contributes_nothing(self):
        from app.services.extraction_unit_gate import signature_for_fields

        sig = signature_for_fields(["no_unit_here", "also_unknown"])
        assert sig == ()


# ---------------------------------------------------------------------------
# 2. chunk_passes_unit_gate — basic predicate behaviour
# ---------------------------------------------------------------------------

class TestChunkPassesUnitGate:

    def test_gate_requires_digit_and_unit(self):
        """chunk must contain BOTH a digit AND a unit token."""
        from app.services.extraction_unit_gate import chunk_passes_unit_gate, signature_for_fields

        sig = signature_for_fields(["max_intercept_km"])
        assert chunk_passes_unit_gate(nfc("range is 50 km"), sig) is True
        # unit present but no digit
        assert chunk_passes_unit_gate(nfc("range in km is unknown"), sig) is False
        # digit present but wrong unit
        assert chunk_passes_unit_gate(nfc("range is 50 miles"), sig) is False
        # digit present but no unit at all
        assert chunk_passes_unit_gate(nfc("count is 42"), sig) is False

    def test_gate_empty_signature_returns_false(self):
        """Empty signature → always False (no unit tokens to match)."""
        from app.services.extraction_unit_gate import chunk_passes_unit_gate

        assert chunk_passes_unit_gate(nfc("50 km"), ()) is False

    def test_gate_empty_text_returns_false(self):
        """Empty / None text → always False."""
        from app.services.extraction_unit_gate import chunk_passes_unit_gate

        assert chunk_passes_unit_gate("", ("km",)) is False

    def test_gate_cyrillic_unit(self):
        """Cyrillic unit synonym км must also fire the gate."""
        from app.services.extraction_unit_gate import chunk_passes_unit_gate, signature_for_fields

        sig = signature_for_fields(["max_intercept_km"])
        assert chunk_passes_unit_gate(nfc("дальность 50 км"), sig) is True

    def test_gate_unit_without_digit(self):
        """Unit token present but no digit → False."""
        from app.services.extraction_unit_gate import chunk_passes_unit_gate, signature_for_fields

        sig = signature_for_fields(["nominal_rf_mhz"])
        assert chunk_passes_unit_gate(nfc("frequency units are mhz"), sig) is False

    def test_gate_digit_without_matching_unit(self):
        """Digit present but no unit from signature → False."""
        from app.services.extraction_unit_gate import chunk_passes_unit_gate, signature_for_fields

        sig = signature_for_fields(["nominal_rf_mhz"])  # mhz, ghz, khz
        assert chunk_passes_unit_gate(nfc("weight is 50 kg"), sig) is False


# ---------------------------------------------------------------------------
# 3. Gate is superset of label: fixture-driven
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", [c for c in _CASES["value_in_chunk_cases"] if c["expect"] is not None]
)
def test_gate_superset_of_label(case):
    """By construction: any text the label can ground must fire the gate.

    The label requires digit + unit (ADJACENT or SAME_CHUNK), and so does the
    gate.  This property guarantees the gate never silently drops a labelable
    chunk from the candidate pool.
    """
    from app.services.extraction_unit_gate import chunk_passes_unit_gate

    assert chunk_passes_unit_gate(nfc(case["text"]), tuple(case["units"])) is True


# ---------------------------------------------------------------------------
# 4. Signals-level: build_retrieval_profile populates unit_signature
# ---------------------------------------------------------------------------

class TestBuildRetrievalProfileUnitSignature:

    def test_radar_power_rf_unit_signature_content(self):
        """radar_power_rf has erp_dbw / tx_peak_power_kw / nominal_rf_mhz fields.

        units_for("erp_dbw") → ["dbw"], units_for("tx_peak_power_kw") → ["kw"],
        units_for("nominal_rf_mhz") → ["mhz"].  All three must appear; "km"
        (a distance unit) must NOT appear.
        """
        from app.services.extraction_query_builder import build_retrieval_profile

        RadarPowerRfPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf",
            "RadarPowerRfPass",
        )
        signals = build_retrieval_profile(None, RadarPowerRfPass)
        sig = signals.unit_signature

        # Must include the kw, mhz, dbw tokens (casefolded via nfc)
        assert "kw" in sig, f"expected 'kw' in {sig}"
        assert "mhz" in sig, f"expected 'mhz' in {sig}"
        # dbw comes from erp_dbw field
        assert "dbw" in sig, f"expected 'dbw' in {sig}"

        # Must NOT include a distance unit (not a field in this pass)
        assert "km" not in sig, f"unexpected 'km' in {sig}"

    def test_default_unit_signature_is_empty_tuple_when_no_record_cls(self):
        """A pass with no resolvable Record class gets unit_signature=()."""
        from pydantic import BaseModel, ConfigDict
        from app.services.extraction_query_builder import build_retrieval_profile

        class EmptyPass(BaseModel):
            model_config = ConfigDict(extra="ignore")

        signals = build_retrieval_profile(None, EmptyPass)
        assert signals.unit_signature == ()

    def test_unit_signature_is_a_tuple(self):
        """unit_signature field must be a tuple (hashable, frozen-safe)."""
        from app.services.extraction_query_builder import build_retrieval_profile

        RadarPowerRfPass = _load_pass_cls(
            "ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf",
            "RadarPowerRfPass",
        )
        signals = build_retrieval_profile(None, RadarPowerRfPass)
        assert isinstance(signals.unit_signature, tuple)


# ---------------------------------------------------------------------------
# 5. RetrievalProfile.unit_gate field
# ---------------------------------------------------------------------------

class TestRetrievalProfileUnitGate:

    def test_unit_gate_defaults_to_false(self):
        """unit_gate must default to False — legacy pools are byte-identical."""
        from app.services.ontology_bundles import RetrievalProfile

        rp = RetrievalProfile()
        assert rp.unit_gate is False

    def test_unit_gate_can_be_set_true(self):
        """unit_gate=True must be accepted by the model."""
        from app.services.ontology_bundles import RetrievalProfile

        rp = RetrievalProfile(unit_gate=True)
        assert rp.unit_gate is True

    def test_retrieval_profile_extra_forbid_still_enforced(self):
        """extra='forbid' contract must survive the new field addition."""
        from pydantic import ValidationError
        from app.services.ontology_bundles import RetrievalProfile

        with pytest.raises(ValidationError):
            RetrievalProfile(nonexistent_key="boom")

    def test_existing_bundles_unit_gate_defaults(self):
        """Real bundle manifests do not set unit_gate → must read back False."""
        from app.services.ontology_bundles import load_bundle_manifest

        for bundle_key in ("air_defense_v3", "air_defense_v3_baseline_subset"):
            m = load_bundle_manifest(bundle_key)
            for pass_def in m.passes:
                rp = pass_def.retrieval
                if rp is None:
                    continue
                assert rp.unit_gate is False, (
                    f"{bundle_key}/{pass_def.name} unit_gate should default False"
                )


# ---------------------------------------------------------------------------
# Helpers (local)
# ---------------------------------------------------------------------------

def _load_pass_cls(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)
