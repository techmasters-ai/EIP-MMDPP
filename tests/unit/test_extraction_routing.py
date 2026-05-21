"""C3 — Router unit tests.

Task 3.3: verify route_pass() returns the correct RouteDecision for all cases.

Tests are fully hermetic: no DB, no Celery, no MinIO.  All input is constructed
from SimpleNamespace pass_defs and plain dict doc_metadata.

Cases covered:
  (1) required=True → always RUN_FULL regardless of cues/hits.
  (2) phase=identity → always RUN_FULL regardless of cues/hits.
  (3) cues=[] (empty list) → always RUN_FULL (no cue config = no opinion).
  (4) cue hits in text_chunks → RUN_FULL + matched_cues + chunks_with_hits reported.
  (5) no cue hits → SKIP_NO_EVIDENCE.
  (6) case-insensitivity: uppercase cue matches lowercase chunk text.
  (7) case-insensitivity: lowercase cue matches UPPERCASE chunk text.
  (8) matched_cues contains the actual cues that matched (not duplicates).
  (9) chunks_with_hits = correct count of chunks with ≥1 hit.
  (10) table_captions and picture_captions are also searched.

Run standalone (no DB required):
    python3 -m pytest tests/unit/test_extraction_routing.py -v
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pass_def(
    name: str = "radar_power_rf",
    *,
    phase: str = "field_group",
    required: bool = False,
    cues: list[str] | None = None,
) -> SimpleNamespace:
    """Return a SimpleNamespace mimicking PassManifest with cues."""
    return SimpleNamespace(
        name=name,
        phase=phase,
        required=required,
        cues=list(cues or []),
    )


def _make_doc_metadata(
    text_chunks: list[str] | None = None,
    table_captions: list[str] | None = None,
    picture_captions: list[str] | None = None,
) -> dict:
    return {
        "text_chunks": list(text_chunks or []),
        "table_captions": list(table_captions or []),
        "picture_captions": list(picture_captions or []),
    }


def _route(pass_def, doc_metadata):
    from app.services.extraction_routing import route_pass
    return route_pass(pass_def, doc_metadata)


# ---------------------------------------------------------------------------
# 1. required=True → always RUN_FULL
# ---------------------------------------------------------------------------

class TestRequiredPassAlwaysRunFull:

    def test_required_with_no_cues_no_hits(self):
        """required=True, empty doc → RUN_FULL (system_links is always required)."""
        p = _make_pass_def("system_links", phase="relationship", required=True, cues=[])
        doc = _make_doc_metadata()
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert decision.reason == "required_pass"

    def test_required_with_cues_but_no_hits(self):
        """required=True, cues present but none hit → still RUN_FULL."""
        p = _make_pass_def("system_links", phase="relationship", required=True,
                           cues=["guidance", "radar"])
        doc = _make_doc_metadata(text_chunks=["irrelevant text about cats"])
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert decision.reason == "required_pass"

    def test_required_with_cues_and_hits(self):
        """required=True, cues hit → still RUN_FULL."""
        p = _make_pass_def("system_links", phase="relationship", required=True,
                           cues=["radar"])
        doc = _make_doc_metadata(text_chunks=["This document covers radar systems"])
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert decision.reason == "required_pass"


# ---------------------------------------------------------------------------
# 2. phase=identity → always RUN_FULL
# ---------------------------------------------------------------------------

class TestIdentityPassAlwaysRunFull:

    def test_identity_with_no_cues(self):
        """identity phase, empty cues → RUN_FULL."""
        p = _make_pass_def("radar_identity", phase="identity", required=False, cues=[])
        doc = _make_doc_metadata()
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert decision.reason == "identity_pass"

    def test_identity_with_cues_but_no_hits(self):
        """identity phase, cues present but none hit → still RUN_FULL."""
        p = _make_pass_def("radar_identity", phase="identity", required=False,
                           cues=["RF", "frequency"])
        doc = _make_doc_metadata(text_chunks=["banana bread recipe"])
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert decision.reason == "identity_pass"

    def test_missile_identity_no_cues(self):
        p = _make_pass_def("missile_identity", phase="identity", required=False, cues=[])
        doc = _make_doc_metadata(text_chunks=["irrelevant"])
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert decision.reason == "identity_pass"


# ---------------------------------------------------------------------------
# 3. Empty cues list → RUN_FULL (no cue config)
# ---------------------------------------------------------------------------

class TestEmptyCuesAlwaysRunFull:

    def test_field_group_with_empty_cues_and_content(self):
        """field_group, cues=[], content present → RUN_FULL (no opinion)."""
        p = _make_pass_def("radar_power_rf", phase="field_group", cues=[])
        doc = _make_doc_metadata(text_chunks=["lots of text"])
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert decision.reason == "no_cues_configured"

    def test_field_group_with_empty_cues_and_empty_doc(self):
        p = _make_pass_def("radar_power_rf", phase="field_group", cues=[])
        doc = _make_doc_metadata()
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert decision.reason == "no_cues_configured"


# ---------------------------------------------------------------------------
# 4. Cue hits → RUN_FULL with diagnostics
# ---------------------------------------------------------------------------

class TestCueHitsReturnRunFull:

    def test_single_cue_hit(self):
        """One cue matches one chunk → RUN_FULL."""
        p = _make_pass_def("radar_power_rf", phase="field_group",
                           cues=["frequency", "MHz", "kW"])
        doc = _make_doc_metadata(text_chunks=[
            "The operating frequency of the radar is 9.375 GHz.",
            "Completely unrelated text about administration.",
        ])
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert decision.chunks_with_hits >= 1
        assert len(decision.matched_cues) >= 1

    def test_multiple_cues_in_multiple_chunks(self):
        """Multiple cues hit multiple chunks → RUN_FULL; counts are right."""
        p = _make_pass_def("radar_power_rf", phase="field_group",
                           cues=["MHz", "kW", "dBW"])
        doc = _make_doc_metadata(text_chunks=[
            "Transmitter power 60 kW peak.",
            "Frequency: 9375 MHz.",
            "Unrelated.",
        ])
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert decision.chunks_with_hits == 2
        # Both "MHz" and "kW" should appear in matched_cues
        assert "MHz" in decision.matched_cues
        assert "kW" in decision.matched_cues

    def test_matched_cues_has_no_duplicates(self):
        """matched_cues contains each matching cue at most once (even if it hits many chunks)."""
        p = _make_pass_def("radar_power_rf", phase="field_group",
                           cues=["MHz"])
        doc = _make_doc_metadata(text_chunks=[
            "Frequency 9375 MHz.",
            "Secondary frequency 8500 MHz.",
            "Tertiary frequency 6800 MHz.",
        ])
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert decision.matched_cues.count("MHz") == 1, (
            f"matched_cues should deduplicate; got: {decision.matched_cues}"
        )
        assert decision.chunks_with_hits == 3


# ---------------------------------------------------------------------------
# 5. No cue hits → SKIP_NO_EVIDENCE
# ---------------------------------------------------------------------------

class TestNoCueHitsSkipNoEvidence:

    def test_skip_when_no_chunks_match(self):
        """None of the cues appear in any chunk → SKIP_NO_EVIDENCE."""
        p = _make_pass_def("radar_power_rf", phase="field_group",
                           cues=["MHz", "GHz", "kW", "RF"])
        doc = _make_doc_metadata(text_chunks=[
            "The logistic chain depended on supply delivery.",
            "Administrative processing proceeded as planned.",
        ])
        decision = _route(p, doc)
        assert decision.mode == "SKIP_NO_EVIDENCE"
        assert decision.chunks_with_hits == 0
        assert decision.matched_cues == []

    def test_skip_when_doc_has_no_chunks(self):
        """Empty text_chunks (and no captions) → SKIP_NO_EVIDENCE."""
        p = _make_pass_def("radar_power_rf", phase="field_group",
                           cues=["MHz", "GHz"])
        doc = _make_doc_metadata()
        decision = _route(p, doc)
        assert decision.mode == "SKIP_NO_EVIDENCE"
        assert decision.chunks_with_hits == 0

    def test_skip_reason_mentions_cue_count(self):
        """The SKIP_NO_EVIDENCE reason string mentions the cue count."""
        cues = ["MHz", "GHz", "kW"]
        p = _make_pass_def("radar_power_rf", phase="field_group", cues=cues)
        doc = _make_doc_metadata(text_chunks=["irrelevant"])
        decision = _route(p, doc)
        assert decision.mode == "SKIP_NO_EVIDENCE"
        assert str(len(cues)) in decision.reason, (
            f"reason should mention cue count {len(cues)}; got: {decision.reason!r}"
        )


# ---------------------------------------------------------------------------
# 6 + 7. Case-insensitivity
# ---------------------------------------------------------------------------

class TestCaseInsensitiveMatching:

    def test_uppercase_cue_matches_lowercase_text(self):
        """Cue 'RF' matches chunk containing 'rf'."""
        p = _make_pass_def("radar_power_rf", phase="field_group", cues=["RF", "ERP"])
        doc = _make_doc_metadata(text_chunks=["the rf transmitter output is 40 kw"])
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert "RF" in decision.matched_cues

    def test_lowercase_cue_matches_uppercase_text(self):
        """Cue 'mhz' matches chunk containing 'MHz' or 'MHZ'."""
        p = _make_pass_def("radar_power_rf", phase="field_group", cues=["mhz"])
        doc = _make_doc_metadata(text_chunks=["FREQUENCY: 9375 MHZ"])
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
        assert "mhz" in decision.matched_cues

    def test_mixed_case_cue_matches_mixed_case_text(self):
        """Cue 'pulse compression' matches 'Pulse Compression' in text."""
        p = _make_pass_def("radar_modulation", phase="field_group",
                           cues=["pulse compression"])
        doc = _make_doc_metadata(text_chunks=["Mode: Pulse Compression waveform"])
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"

    def test_no_hit_when_case_not_matching_and_would_otherwise_match(self):
        """Sanity: 'mHz' cue does NOT match when the chunk contains 'GHz' (different word)."""
        # Only false if case-insensitive comparison itself is wrong.
        # 'mhz' substring in 'ghz' → False: 'm', 'g' differ.
        p = _make_pass_def("radar_power_rf", phase="field_group", cues=["mhz"])
        doc = _make_doc_metadata(text_chunks=["Frequency: 9.375 GHz (not megahertz)"])
        # "mhz" is NOT a substring of "ghz" — this should SKIP
        decision = _route(p, doc)
        # "megahertz" doesn't contain "mhz" as a substring; "ghz" doesn't either.
        # This is SKIP territory. (If the test fails, the router has a bug.)
        assert decision.mode == "SKIP_NO_EVIDENCE", (
            f"'mhz' should NOT be a substring of 'ghz'; got mode={decision.mode}"
        )


# ---------------------------------------------------------------------------
# 10. Table captions + picture captions also searched
# ---------------------------------------------------------------------------

class TestAuxiliarySourcesSearched:

    def test_table_caption_hit_runs_pass(self):
        """Cue hit in table_captions → RUN_FULL even if text_chunks empty."""
        p = _make_pass_def("radar_power_rf", phase="field_group",
                           cues=["frequency", "MHz"])
        doc = _make_doc_metadata(
            text_chunks=["Only irrelevant text here."],
            table_captions=["Table 3 — Frequency and bandwidth parameters (MHz)"],
        )
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"

    def test_picture_caption_hit_runs_pass(self):
        """Cue hit in picture_captions → RUN_FULL."""
        p = _make_pass_def("missile_kinematics", phase="field_group",
                           cues=["range", "km"])
        doc = _make_doc_metadata(
            text_chunks=["Unrelated logistics text."],
            picture_captions=["Figure 2 — Maximum range envelope (km)"],
        )
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"

    def test_no_hit_in_any_source_skips(self):
        """No cue hit in text_chunks, table_captions, or picture_captions → SKIP."""
        p = _make_pass_def("radar_antenna", phase="field_group",
                           cues=["antenna", "beam", "dB"])
        doc = _make_doc_metadata(
            text_chunks=["The missile was launched at dawn."],
            table_captions=["Table 1 — Launch sequence timeline"],
            picture_captions=["Figure 1 — Missile silhouette"],
        )
        decision = _route(p, doc)
        assert decision.mode == "SKIP_NO_EVIDENCE"

    def test_only_picture_captions_provided_hit(self):
        """Only picture captions present, cue hits there → RUN_FULL."""
        p = _make_pass_def("missile_propulsion", phase="field_group",
                           cues=["thrust", "kN"])
        doc = _make_doc_metadata(
            picture_captions=["Figure 5 — Booster thrust vs altitude (kN)"],
        )
        decision = _route(p, doc)
        assert decision.mode == "RUN_FULL"
