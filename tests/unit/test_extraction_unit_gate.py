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

    def test_gate_fires_where_same_chunk_label_tier_rejects(self):
        """INTENTIONAL gate-label gap (Task 7 / Task 5 polish).

        The SAME_CHUNK label tier requires >=2 digits in the number to avoid
        coincidental single-digit hits, so a single-digit value with a
        DETACHED unit is NOT labelable via SAME_CHUNK.  The gate has no value
        knowledge — digit + signature unit token is enough — so it force-keeps
        the chunk anyway.  This is correct BY DESIGN: the gate is the recall
        FLOOR (cheap, value-free, must never drop a potentially groundable
        chunk); the label re-checks precision downstream and rejects what it
        cannot ground.
        """
        from app.services.extraction_unit_gate import chunk_passes_unit_gate
        from app.services.field_value_grounding import value_in_chunk

        # Adjacent single-digit form: gate fires.
        assert chunk_passes_unit_gate(nfc("weight 7 kg"), ("kg",)) is True

        # Detached single-digit form: gate STILL fires (recall floor) ...
        detached = nfc("weight 7 and the unit column header says kg")
        assert chunk_passes_unit_gate(detached, ("kg",)) is True
        # ... although the SAME_CHUNK label tier rejects it (single digit "7"
        # fails the >=2-digit rule; no ADJACENT match either).
        assert value_in_chunk({"7"}, ("kg",), detached) is None


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
# 6. count_unit_tokens — Task 8 structural text feature
# ---------------------------------------------------------------------------

class TestCountUnitTokens:
    """count_unit_tokens(text_nfc, signature) → int, capped at 20.

    Counts DISTINCT matching synonyms, not occurrences. Uses the same
    _compiled_unit_re matcher as chunk_passes_unit_gate so semantics
    can never drift between the gate and this counter.
    """

    def test_import(self):
        from app.services.extraction_unit_gate import count_unit_tokens  # noqa: F401

    def test_basic_single_synonym_match(self):
        """Single synonym present → 1."""
        from app.services.extraction_unit_gate import count_unit_tokens

        assert count_unit_tokens(nfc("range is 50 km"), ("km",)) == 1

    def test_two_synonyms_matched(self):
        """Two distinct synonyms present → 2."""
        from app.services.extraction_unit_gate import count_unit_tokens

        assert count_unit_tokens(nfc("50 kw at 60 mhz"), ("kw", "mhz")) == 2

    def test_distinct_not_occurrences(self):
        """Synonym appears multiple times but counts only once (distinct)."""
        from app.services.extraction_unit_gate import count_unit_tokens

        # "kw" appears twice; should count as 1 (distinct synonym, not occurrence)
        assert count_unit_tokens(nfc("50 kw and 60 kw"), ("kw",)) == 1

    def test_unmatched_synonym_not_counted(self):
        """Synonym absent from text → not counted."""
        from app.services.extraction_unit_gate import count_unit_tokens

        assert count_unit_tokens(nfc("range is 50 km"), ("mhz", "dbw")) == 0

    def test_partial_match_word_boundary(self):
        """'s' does NOT match 'sites' — token-bounded matching must apply.

        The canonical word-boundary test from the task spec:
          "50 sites" with ("s",) → 0
        """
        from app.services.extraction_unit_gate import count_unit_tokens

        # "s" is a suffix of "sites"; token boundary must prevent the match
        assert count_unit_tokens(nfc("50 sites"), ("s",)) == 0

    def test_no_word_boundary_false_positive_months(self):
        """'m' does NOT match 'months'."""
        from app.services.extraction_unit_gate import count_unit_tokens

        assert count_unit_tokens(nfc("9 months of operation"), ("m",)) == 0

    def test_empty_signature_returns_zero(self):
        """Empty signature → always 0."""
        from app.services.extraction_unit_gate import count_unit_tokens

        assert count_unit_tokens(nfc("50 km"), ()) == 0

    def test_empty_text_returns_zero(self):
        """Empty text → always 0."""
        from app.services.extraction_unit_gate import count_unit_tokens

        assert count_unit_tokens("", ("km",)) == 0

    def test_cap_at_twenty(self):
        """Signature with more than 20 matching synonyms → capped at 20."""
        from app.services.extraction_unit_gate import count_unit_tokens
        from app.services.field_value_grounding import SUFFIX_UNITS

        # Build a text that contains one token from every known unit group.
        # There are more than 20 groups in SUFFIX_UNITS; each synonyms list
        # has at least one entry. Build a signature of all known synonyms.
        all_syns: list[str] = []
        for syns in SUFFIX_UNITS.values():
            all_syns.extend(syns)
        # De-duplicate while preserving order (signature_for_fields yields sorted unique).
        seen: set[str] = set()
        sig: list[str] = []
        for s in all_syns:
            if s not in seen:
                seen.add(s)
                sig.append(s)
        # Build a text that actually matches every synonym.
        # Use "50 <syn>" patterns, one per synonym.
        text = " ".join(f"50 {s}" for s in sig)
        result = count_unit_tokens(nfc(text), tuple(sig))
        assert result == 20, f"expected 20 (cap), got {result}"

    def test_cyrillic_unit_synonym(self):
        """Cyrillic unit synonym км is counted."""
        from app.services.extraction_unit_gate import count_unit_tokens

        assert count_unit_tokens(nfc("дальность 50 км"), ("км",)) == 1

    def test_signature_with_unmatched_synonyms_counts_only_matched(self):
        """Only matched synonyms count; unmatched ones are ignored."""
        from app.services.extraction_unit_gate import count_unit_tokens

        # Text has km but not mhz or dbw.
        assert count_unit_tokens(nfc("range 50 km"), ("km", "mhz", "dbw")) == 1


# ---------------------------------------------------------------------------
# Helpers (local)
# ---------------------------------------------------------------------------

def _load_pass_cls(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)
