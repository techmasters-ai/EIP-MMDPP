"""Unit tests for Tasks C2+C3 — lexical alias search + regex pattern search.

TDD: these tests are written BEFORE the implementation and are expected to
fail until app/services/extraction_lexical_search.py exists.

Coverage:
  C2 (lexical_hit_counts):
    - per-chunk alias hit counts on synthetic rows
    - negative terms counted SEPARATELY (not folded into alias_hits)
    - Cyrillic NFC normalisation (composed vs decomposed form → both match)
    - candidate_key: vertex_id when present, self_ref fallback
    - case-insensitive matching

  C3 (pattern_hit_counts):
    - literal-phrase match (no re: prefix)
    - re: prefix → compiled regex with re.IGNORECASE | re.MULTILINE
    - Cyrillic pattern case mirrors C2 (NFC normalised haystack)
    - invalid re: pattern raises at function entry (not swallowed at runtime)
    - supported_fields populated per matched field
"""
from __future__ import annotations

import re
import unicodedata

import pytest

from app.services.extraction_query_builder import FieldRetrievalQuery

# The module under test (will be created by the implementation step).
from app.services.extraction_lexical_search import (
    keyword_hit_counts,
    lexical_hit_counts,
    pattern_hit_counts,
    section_hit_counts,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic rows and field queries
# ---------------------------------------------------------------------------

def _row(
    self_ref: str,
    chunk_text: str,
    *,
    vertex_id: str | None = None,
) -> dict:
    """Minimal ExtractionChunk row dict."""
    row: dict = {
        "self_ref": self_ref,
        "chunk_text": chunk_text,
    }
    if vertex_id is not None:
        row["vertex_id"] = vertex_id
    return row


def _fq(
    field_name: str,
    aliases: tuple[str, ...] = (),
    negative_terms: tuple[str, ...] = (),
    evidence_patterns: tuple[str, ...] = (),
) -> FieldRetrievalQuery:
    """Build a minimal FieldRetrievalQuery for testing."""
    return FieldRetrievalQuery(
        field_name=field_name,
        query_text=f"query text for {field_name}",
        aliases=aliases,
        negative_terms=negative_terms,
        evidence_patterns=evidence_patterns,
        likely_sections=(),
        units=(),
    )


# ---------------------------------------------------------------------------
# C2 — lexical_hit_counts
# ---------------------------------------------------------------------------

class TestLexicalHitCounts:

    def test_alias_hit_count_basic(self):
        """Aliases present in chunk_text are counted."""
        rows = [_row("ref-1", "The radar system has a peak power output.")]
        fqs = [_fq("tx_power", aliases=("peak power", "radar"))]
        result = lexical_hit_counts(rows, fqs)
        assert "ref-1" in result
        assert result["ref-1"]["alias_hits"] == 2
        assert "tx_power" in result["ref-1"]["supported_fields"]

    def test_alias_miss_not_counted(self):
        """A chunk with no alias match has alias_hits == 0."""
        rows = [_row("ref-2", "Unrelated text about nothing specific.")]
        fqs = [_fq("range_km", aliases=("range", "distance", "km"))]
        result = lexical_hit_counts(rows, fqs)
        assert result["ref-2"]["alias_hits"] == 0
        assert result["ref-2"]["supported_fields"] == set()

    def test_multiple_rows_scored_independently(self):
        """Each row produces its own entry in the result dict."""
        rows = [
            _row("ref-A", "Range: 150 km maximum"),
            _row("ref-B", "No relevant content here"),
        ]
        fqs = [_fq("range_km", aliases=("range", "km"))]
        result = lexical_hit_counts(rows, fqs)
        assert result["ref-A"]["alias_hits"] == 2
        assert result["ref-B"]["alias_hits"] == 0

    def test_negative_terms_counted_separately_not_in_alias_hits(self):
        """negative_terms are tracked in negative_hits, NOT in alias_hits."""
        rows = [_row("ref-3", "The beacon frequency is 9.375 GHz radar altimeter.")]
        fqs = [_fq(
            "frequency_ghz",
            aliases=("frequency", "GHz"),
            negative_terms=("altimeter",),
        )]
        result = lexical_hit_counts(rows, fqs)
        entry = result["ref-3"]
        assert entry["alias_hits"] == 2          # "frequency" + "GHz"
        assert entry["negative_hits"] == 1       # "altimeter"
        # alias_hits must NOT count negative_terms
        assert entry["alias_hits"] != entry["alias_hits"] + entry["negative_hits"]

    def test_negative_terms_not_present_is_zero(self):
        """No negative terms present → negative_hits == 0."""
        rows = [_row("ref-4", "Transmitter peak power 500 kW.")]
        fqs = [_fq("tx_power", aliases=("peak power",), negative_terms=("altimeter",))]
        result = lexical_hit_counts(rows, fqs)
        assert result["ref-4"]["negative_hits"] == 0

    def test_case_insensitive_matching(self):
        """Alias matching is case-insensitive (casefold applied to both sides)."""
        rows = [_row("ref-5", "PEAK POWER output is 100 kW")]
        fqs = [_fq("tx_power", aliases=("peak power",))]
        result = lexical_hit_counts(rows, fqs)
        assert result["ref-5"]["alias_hits"] == 1

    def test_candidate_key_uses_vertex_id_when_present(self):
        """When vertex_id is present, it is used as the candidate_key."""
        rows = [_row("ref-6", "Some content", vertex_id="run-1:chunk_7")]
        fqs = [_fq("f", aliases=("content",))]
        result = lexical_hit_counts(rows, fqs)
        assert "run-1:chunk_7" in result
        assert "ref-6" not in result

    def test_candidate_key_falls_back_to_self_ref(self):
        """When vertex_id is absent (or None), self_ref is the key."""
        rows = [_row("ref-7", "Some content")]   # no vertex_id key in dict
        fqs = [_fq("f", aliases=("content",))]
        result = lexical_hit_counts(rows, fqs)
        assert "ref-7" in result

    def test_candidate_key_none_vertex_id_falls_back_to_self_ref(self):
        """vertex_id=None → self_ref is used as key."""
        rows = [_row("ref-8", "Some content", vertex_id=None)]
        fqs = [_fq("f", aliases=("content",))]
        result = lexical_hit_counts(rows, fqs)
        assert "ref-8" in result

    def test_cyrillic_nfc_normalisation_composed_matches_decomposed(self):
        """NFC normalisation ensures composed/decomposed Cyrillic both match.

        'й' (U+0439) is precomposed.  Decomposed form: и (U+0438) + кратка (U+0306).
        Both NFC-normalise to U+0439, so a search for the composed form MUST
        match text written in the decomposed form and vice versa.
        """
        # Compose а й (кратка as combining U+0306) by hand — decomposed form
        composed_alias = "й"          # й precomposed (U+0439)
        decomposed_text = "й"   # и + combining breve → й decomposed

        # Verify the test scaffolding is correct (these are NOT equal as raw bytes)
        assert composed_alias != decomposed_text
        # But after NFC they ARE equal
        assert unicodedata.normalize("NFC", composed_alias) == unicodedata.normalize("NFC", decomposed_text)

        chunk_text = f"Дальность обнаружения {decomposed_text}дет 150 км"
        rows = [_row("ref-cyr-1", chunk_text, vertex_id="run-cyr:chunk_1")]
        fqs = [_fq("range_km", aliases=(f"{composed_alias}дет",))]

        result = lexical_hit_counts(rows, fqs)
        assert result["run-cyr:chunk_1"]["alias_hits"] == 1

    def test_multiple_field_queries_supported_fields_union(self):
        """supported_fields accumulates field names from ALL matching field queries."""
        rows = [_row("ref-multi", "maximum range 200 km peak power 100 kW")]
        fqs = [
            _fq("range_km", aliases=("range", "km")),
            _fq("tx_power", aliases=("peak power", "kW")),
        ]
        result = lexical_hit_counts(rows, fqs)
        entry = result["ref-multi"]
        assert "range_km" in entry["supported_fields"]
        assert "tx_power" in entry["supported_fields"]
        assert entry["alias_hits"] == 4  # "range", "km", "peak power", "kW"

    def test_empty_rows_returns_empty_dict(self):
        rows = []
        fqs = [_fq("f", aliases=("x",))]
        assert lexical_hit_counts(rows, fqs) == {}

    def test_empty_field_queries_returns_zeroed_entries(self):
        rows = [_row("ref-9", "Some text")]
        result = lexical_hit_counts(rows, [])
        assert result["ref-9"]["alias_hits"] == 0
        assert result["ref-9"]["negative_hits"] == 0
        assert result["ref-9"]["supported_fields"] == set()


# ---------------------------------------------------------------------------
# C3 — pattern_hit_counts
# ---------------------------------------------------------------------------

class TestPatternHitCounts:

    def test_literal_phrase_match(self):
        """No re: prefix → treated as literal NFC+casefold substring."""
        rows = [_row("ref-p1", "Maximum range is 150 km at standard conditions.")]
        fqs = [_fq("range_km", evidence_patterns=("150 km",))]
        result = pattern_hit_counts(rows, fqs)
        assert result["ref-p1"]["pattern_hits"] == 1
        assert "range_km" in result["ref-p1"]["supported_fields"]

    def test_literal_phrase_no_match(self):
        rows = [_row("ref-p2", "Transmitter power 500 kW.")]
        fqs = [_fq("range_km", evidence_patterns=("150 km",))]
        result = pattern_hit_counts(rows, fqs)
        assert result["ref-p2"]["pattern_hits"] == 0

    def test_regex_pattern_match(self):
        """re: prefix strips the prefix and compiles as regex (IGNORECASE|MULTILINE)."""
        rows = [_row("ref-p3", "Range: 150 km\nMaximum detection 200 km")]
        fqs = [_fq("range_km", evidence_patterns=(r"re:\d+\s*km",))]
        result = pattern_hit_counts(rows, fqs)
        # Two matches: "150 km" and "200 km"
        assert result["ref-p3"]["pattern_hits"] == 2
        assert "range_km" in result["ref-p3"]["supported_fields"]

    def test_regex_case_insensitive(self):
        """re: pattern matches case-insensitively (IGNORECASE)."""
        rows = [_row("ref-p4", "Peak Power: 500 KW")]
        fqs = [_fq("tx_power", evidence_patterns=(r"re:peak power",))]
        result = pattern_hit_counts(rows, fqs)
        assert result["ref-p4"]["pattern_hits"] == 1

    def test_regex_multiline(self):
        """re: ^ and $ work as line anchors (MULTILINE)."""
        rows = [_row("ref-p5", "line one\nRange value\nline three")]
        fqs = [_fq("range_km", evidence_patterns=(r"re:^Range value$",))]
        result = pattern_hit_counts(rows, fqs)
        assert result["ref-p5"]["pattern_hits"] == 1

    def test_literal_case_insensitive(self):
        """Literal phrase matching is also case-insensitive via casefold."""
        rows = [_row("ref-p6", "MAXIMUM RANGE 150 KM")]
        fqs = [_fq("range_km", evidence_patterns=("maximum range",))]
        result = pattern_hit_counts(rows, fqs)
        assert result["ref-p6"]["pattern_hits"] == 1

    def test_cyrillic_nfc_normalisation_in_pattern(self):
        """NFC normalisation ensures Cyrillic composed/decomposed matches in patterns.

        Mirrors the C2 Cyrillic test: decomposed text in chunk, composed form
        in the evidence pattern.
        """
        composed_pattern = "йдет"    # "йдет" precomposed
        decomposed_text = "йдет"  # decomposed й + "дет"

        chunk_text = f"Цель {decomposed_text} в зоне поражения"
        rows = [_row("ref-cyr-p1", chunk_text)]
        fqs = [_fq("detection", evidence_patterns=(composed_pattern,))]

        result = pattern_hit_counts(rows, fqs)
        assert result["ref-cyr-p1"]["pattern_hits"] == 1

    def test_invalid_regex_raises_at_compile_time(self):
        """An invalid re: pattern raises re.error at function entry, not silently."""
        rows = [_row("ref-bad", "some text")]
        fqs = [_fq("f", evidence_patterns=("re:[unclosed",))]
        with pytest.raises(re.error):
            pattern_hit_counts(rows, fqs)

    def test_candidate_key_vertex_id_preferred(self):
        """pattern_hit_counts uses the same candidate_key logic as lexical."""
        rows = [_row("ref-pk", "target range 150 km", vertex_id="run-2:chunk_3")]
        fqs = [_fq("range_km", evidence_patterns=("150 km",))]
        result = pattern_hit_counts(rows, fqs)
        assert "run-2:chunk_3" in result
        assert "ref-pk" not in result

    def test_candidate_key_self_ref_fallback(self):
        rows = [_row("ref-pfb", "target range 150 km")]
        fqs = [_fq("range_km", evidence_patterns=("150 km",))]
        result = pattern_hit_counts(rows, fqs)
        assert "ref-pfb" in result

    def test_multiple_fields_supported_fields_union(self):
        """supported_fields accumulates all matching field names."""
        rows = [_row("ref-pmf", "range 150 km power 500 kW")]
        fqs = [
            _fq("range_km", evidence_patterns=("150 km",)),
            _fq("tx_power", evidence_patterns=("500 kW",)),
        ]
        result = pattern_hit_counts(rows, fqs)
        entry = result["ref-pmf"]
        assert "range_km" in entry["supported_fields"]
        assert "tx_power" in entry["supported_fields"]
        assert entry["pattern_hits"] == 2

    def test_empty_rows_returns_empty_dict(self):
        assert pattern_hit_counts([], [_fq("f", evidence_patterns=("x",))]) == {}

    def test_empty_field_queries_returns_zeroed_entries(self):
        rows = [_row("ref-pz", "some text")]
        result = pattern_hit_counts(rows, [])
        assert result["ref-pz"]["pattern_hits"] == 0
        assert result["ref-pz"]["supported_fields"] == set()

    def test_pattern_hit_limit_caps_diagnostic_samples(self):
        """pattern_hit_limit caps retained diagnostic samples (not hit count)."""
        # 100 rows, all matching; pattern_hit_limit=5 caps diagnostic samples only
        rows = [_row(f"ref-lim-{i}", "range 150 km") for i in range(100)]
        fqs = [_fq("range_km", evidence_patterns=("150 km",))]
        result = pattern_hit_counts(rows, fqs, pattern_hit_limit=5)
        # Every row still gets its own entry with its own hit count
        assert all(entry["pattern_hits"] == 1 for entry in result.values())
        # The function should not raise and should return all rows
        assert len(result) == 100


# ---------------------------------------------------------------------------
# SECTION — section_hit_counts (router-scoring section signal, v1)
#
# Mirrors lexical_hit_counts: NFC+casefold substring matching, but the haystack
# is the chunk's HEADING hierarchy (section_path / headings projected in Part 1)
# rather than chunk_text, and the needles are the type-matched committed entity
# names ("anchors"). Returns {candidate_key: {"section_hits": int}}.
# ---------------------------------------------------------------------------

def _srow(
    self_ref: str,
    *,
    headings: list[str] | None = None,
    section_path: str | None = None,
    vertex_id: str | None = None,
    chunk_text: str = "",
) -> dict:
    """Minimal ExtractionChunk row carrying heading projection fields."""
    row: dict = {"self_ref": self_ref, "chunk_text": chunk_text}
    if headings is not None:
        row["headings"] = headings
    if section_path is not None:
        row["section_path"] = section_path
    if vertex_id is not None:
        row["vertex_id"] = vertex_id
    return row


class TestSectionHitCounts:
    def test_anchor_present_in_headings_counts_one(self):
        rows = [_srow("r1", headings=["Chapter 2", "SNR-75 Fire Control"])]
        result = section_hit_counts(rows, ["SNR-75", "V-75"])
        assert result["r1"]["section_hits"] == 1

    def test_anchor_absent_from_headings_counts_zero(self):
        rows = [_srow("r2", headings=["Chapter 2", "General Description"])]
        result = section_hit_counts(rows, ["SNR-75", "V-75"])
        assert result["r2"]["section_hits"] == 0

    def test_case_insensitive_match(self):
        rows = [_srow("r3", headings=["snr-75 fire control radar"])]
        result = section_hit_counts(rows, ["SNR-75"])
        assert result["r3"]["section_hits"] == 1

    def test_cyrillic_nfc_decomposed_vs_composed_parity(self):
        # Heading stored decomposed (NFD); anchor supplied composed (NFC).
        # Both must match after the shared _nfc normalisation.
        composed = "Й"  # U+0419 (NFC)
        decomposed = unicodedata.normalize("NFD", composed)  # U+0418 U+0306
        assert composed != decomposed  # byte-level differ pre-normalisation
        rows = [_srow("r4", headings=[f"Section {decomposed}"])]
        result = section_hit_counts(rows, [f"Section {composed}"])
        assert result["r4"]["section_hits"] == 1

    def test_empty_headings_counts_zero(self):
        rows = [_srow("r5", headings=[])]
        result = section_hit_counts(rows, ["SNR-75"])
        assert result["r5"]["section_hits"] == 0

    def test_missing_heading_fields_counts_zero(self):
        # Legacy row: no headings AND no section_path columns at all.
        rows = [{"self_ref": "r5b", "chunk_text": "body text mentions SNR-75"}]
        result = section_hit_counts(rows, ["SNR-75"])
        # chunk_text is NOT the section haystack — section matching is heading-only.
        assert result["r5b"]["section_hits"] == 0

    def test_multiple_anchors_each_counted(self):
        rows = [_srow("r6", headings=["SNR-75 and V-75 deployment"])]
        result = section_hit_counts(rows, ["SNR-75", "V-75"])
        assert result["r6"]["section_hits"] == 2

    def test_same_anchor_listed_twice_counts_per_anchor(self):
        # Mirrors lexical_hit_counts: each anchor entry that matches contributes.
        rows = [_srow("r6b", headings=["SNR-75 fire control"])]
        result = section_hit_counts(rows, ["SNR-75", "SNR-75"])
        assert result["r6b"]["section_hits"] == 2

    def test_section_path_field_used_as_haystack(self):
        # A row with no `headings` list but a `section_path` breadcrumb string.
        rows = [_srow("r7", section_path="Chapter 2 > SNR-75 Fire Control")]
        result = section_hit_counts(rows, ["SNR-75"])
        assert result["r7"]["section_hits"] == 1

    def test_headings_and_section_path_both_searched(self):
        rows = [_srow(
            "r7b",
            headings=["Chapter 2"],
            section_path="Chapter 2 > SNR-75 Fire Control",
        )]
        result = section_hit_counts(rows, ["SNR-75"])
        assert result["r7b"]["section_hits"] >= 1

    def test_candidate_key_prefers_vertex_id(self):
        rows = [_srow("r8", headings=["SNR-75"], vertex_id="run1:chunk_8")]
        result = section_hit_counts(rows, ["SNR-75"])
        assert "run1:chunk_8" in result
        assert "r8" not in result
        assert result["run1:chunk_8"]["section_hits"] == 1

    def test_candidate_key_falls_back_to_self_ref(self):
        rows = [_srow("r9", headings=["SNR-75"])]
        result = section_hit_counts(rows, ["SNR-75"])
        assert "r9" in result

    def test_empty_rows_returns_empty_dict(self):
        assert section_hit_counts([], ["SNR-75"]) == {}

    def test_empty_anchors_zeroes_every_row(self):
        rows = [_srow("r10", headings=["SNR-75 fire control"])]
        result = section_hit_counts(rows, [])
        assert result["r10"]["section_hits"] == 0

    def test_blank_and_none_anchors_ignored(self):
        # Defensive: empty-string / None anchors must not match everything.
        rows = [_srow("r11", headings=["General Description"])]
        result = section_hit_counts(rows, ["", None, "  "])
        assert result["r11"]["section_hits"] == 0


class TestSectionHitCountsLikelySections:
    """Anchor-INDEPENDENT section signal — the ``likely_sections`` branch.

    These schema-typed section-name strings (instance-free) come from each
    field's ``json_schema_extra['retrieval']['likely_sections']`` and are
    available at field-pass routing time even when ``anchors`` is empty (the
    dispatch-ordering bug). They give the otherwise-dead ``section_norm`` /
    ``anchor_section_norm`` feature a non-constant source.
    """

    def test_likely_section_in_section_path_with_empty_anchors(self):
        # (a) EMPTY anchors but a likely_sections term in the section_path.
        #     Proves the signal is anchor-independent.
        rows = [_srow("ls1", section_path="Chapter 3 > Performance Characteristics")]
        result = section_hit_counts(
            rows, [], likely_sections=["Performance Characteristics"]
        )
        assert result["ls1"]["section_hits"] >= 1

    def test_likely_section_in_headings(self):
        # (b) likely_sections matching in the `headings` list.
        rows = [_srow("ls2", headings=["Chapter 3", "Technical Specifications"])]
        result = section_hit_counts(
            rows, [], likely_sections=["Technical Specifications"]
        )
        assert result["ls2"]["section_hits"] >= 1

    def test_anchors_and_likely_sections_combine(self):
        # (c) BOTH an anchor AND a likely_sections term present → counts add.
        rows = [_srow(
            "ls3",
            headings=["SNR-75 Fire Control", "Performance Characteristics"],
        )]
        result = section_hit_counts(
            rows, ["SNR-75"], likely_sections=["Performance Characteristics"]
        )
        # 1 anchor match + 1 likely_section match = 2.
        assert result["ls3"]["section_hits"] == 2

    def test_neither_anchor_nor_likely_section_matches(self):
        # (d) Neither matches → 0.
        rows = [_srow("ls4", headings=["General Description"])]
        result = section_hit_counts(
            rows, ["SNR-75"], likely_sections=["Performance Characteristics"]
        )
        assert result["ls4"]["section_hits"] == 0

    def test_likely_section_case_insensitive(self):
        rows = [_srow("ls5", headings=["technical specifications"])]
        result = section_hit_counts(
            rows, [], likely_sections=["Technical Specifications"]
        )
        assert result["ls5"]["section_hits"] >= 1

    def test_likely_section_cyrillic_nfc_parity(self):
        composed = "Й"  # U+0419 (NFC)
        decomposed = unicodedata.normalize("NFD", composed)  # U+0418 U+0306
        assert composed != decomposed
        rows = [_srow("ls6", headings=[f"Раздел {decomposed}"])]
        result = section_hit_counts(
            rows, [], likely_sections=[f"Раздел {composed}"]
        )
        assert result["ls6"]["section_hits"] >= 1

    def test_blank_and_none_likely_sections_ignored(self):
        # Defensive: empty-string / None likely_sections must not match all.
        rows = [_srow("ls7", headings=["General Description"])]
        result = section_hit_counts(rows, [], likely_sections=["", None, "  "])
        assert result["ls7"]["section_hits"] == 0

    def test_likely_section_absent_counts_zero(self):
        rows = [_srow("ls8", headings=["General Description"])]
        result = section_hit_counts(
            rows, [], likely_sections=["Performance Characteristics"]
        )
        assert result["ls8"]["section_hits"] == 0

    def test_default_likely_sections_preserves_anchor_only_behaviour(self):
        # Omitting likely_sections (defaults to ()) must equal anchor-only v1.
        rows = [_srow("ls9", headings=["SNR-75 fire control"])]
        with_default = section_hit_counts(rows, ["SNR-75"])
        explicit_empty = section_hit_counts(rows, ["SNR-75"], likely_sections=[])
        assert with_default == explicit_empty
        assert with_default["ls9"]["section_hits"] == 1

    def test_likely_sections_now_contribute_additively(self):
        # likely_sections is now an ACTIVE anchor-independent needle source.
        # The heading "SNR-75 fire control" carries BOTH the anchor "SNR-75"
        # (1) and the likely_section "fire control" (1) → additive 2.
        # (Supersedes the old "accepted-but-unused" v1 contract.)
        rows = [_srow("r12", headings=["SNR-75 fire control"])]
        result = section_hit_counts(
            rows, ["SNR-75"], likely_sections=("fire control",),
        )
        assert result["r12"]["section_hits"] == 2


# ---------------------------------------------------------------------------
# KEYWORD — keyword_hit_counts (per-pass lexical_keywords, decomposed-lexical)
#
# Mirrors lexical_hit_counts (NFC + casefold substring over chunk_text) but the
# needles are a per-pass configurable keyword list (RetrievalProfile.lexical_keywords)
# rather than schema field aliases. The COUNT is tracked SEPARATELY from the
# field-alias count so the decomposed-lexical C5 term can weight pass keywords
# independently of field labels. Returns {candidate_key: {"keyword_hits": int}}.
# ---------------------------------------------------------------------------


class TestKeywordHitCounts:
    def test_keyword_present_counts_one(self):
        rows = [_row("r1", "The missile reaches a high velocity at burnout.")]
        result = keyword_hit_counts(rows, ["velocity", "range"])
        assert result["r1"]["keyword_hits"] == 1

    def test_multiple_keywords_each_counted(self):
        rows = [_row("r2", "velocity and range are both specified here.")]
        result = keyword_hit_counts(rows, ["velocity", "range"])
        assert result["r2"]["keyword_hits"] == 2

    def test_keyword_absent_counts_zero(self):
        rows = [_row("r3", "unrelated prose about nothing in particular.")]
        result = keyword_hit_counts(rows, ["velocity", "range"])
        assert result["r3"]["keyword_hits"] == 0

    def test_case_insensitive_match(self):
        rows = [_row("r4", "Peak VELOCITY recorded during the test.")]
        result = keyword_hit_counts(rows, ["velocity"])
        assert result["r4"]["keyword_hits"] == 1

    def test_cyrillic_nfc_decomposed_vs_composed_parity(self):
        composed = "Й"  # U+0419 (NFC)
        decomposed = unicodedata.normalize("NFD", composed)  # U+0418 U+0306
        assert composed != decomposed
        rows = [_row("r5", f"keyword {decomposed} appears")]
        result = keyword_hit_counts(rows, [f"keyword {composed}"])
        assert result["r5"]["keyword_hits"] == 1

    def test_same_keyword_twice_counts_per_entry(self):
        # Mirrors lexical_hit_counts: a keyword listed twice contributes twice.
        rows = [_row("r5b", "velocity matters")]
        result = keyword_hit_counts(rows, ["velocity", "velocity"])
        assert result["r5b"]["keyword_hits"] == 2

    def test_empty_keywords_zeroes_every_row(self):
        rows = [_row("r6", "velocity and range here")]
        result = keyword_hit_counts(rows, [])
        assert result["r6"]["keyword_hits"] == 0

    def test_blank_and_none_keywords_ignored(self):
        # Defensive: empty-string / None keywords must not match every chunk.
        rows = [_row("r7", "general description")]
        result = keyword_hit_counts(rows, ["", None, "  "])
        assert result["r7"]["keyword_hits"] == 0

    def test_candidate_key_prefers_vertex_id(self):
        rows = [_row("r8", "velocity test", vertex_id="run1:chunk_8")]
        result = keyword_hit_counts(rows, ["velocity"])
        assert "run1:chunk_8" in result
        assert "r8" not in result

    def test_candidate_key_falls_back_to_self_ref(self):
        rows = [_row("r9", "velocity test")]
        result = keyword_hit_counts(rows, ["velocity"])
        assert "r9" in result

    def test_empty_rows_returns_empty_dict(self):
        assert keyword_hit_counts([], ["velocity"]) == {}

    def test_keyword_count_independent_of_field_aliases(self):
        """The SAME chunk: keyword_hit_counts only counts the keyword needles,
        NOT field aliases. Proves the two counts are SEPARATE features."""
        rows = [_row("r10", "the radar has high velocity")]
        # "radar" is a (hypothetical) field alias; only "velocity" is a keyword.
        kw = keyword_hit_counts(rows, ["velocity"])
        lex = lexical_hit_counts(rows, [_fq("speed", aliases=("radar",))])
        assert kw["r10"]["keyword_hits"] == 1       # only the keyword
        assert lex["r10"]["alias_hits"] == 1        # only the field alias
        # They are tracked on DIFFERENT result keys.
        assert "keyword_hits" in kw["r10"]
        assert "alias_hits" in lex["r10"]


# ---------------------------------------------------------------------------
# KEYWORD — word-boundary single-token matching (Task 12)
#
# Single-token needles (no whitespace after NFC+casefold) use Unicode-aware
# word boundaries via unit_token_regex — the same builder the unit-matcher
# uses. Multi-word phrases keep plain substring semantics unchanged.
# ---------------------------------------------------------------------------


class TestKeywordHitCountsBoundaryMatching:
    """Word-boundary matching for single-token keywords (Task 12).

    Single-token keywords must NOT fire inside a longer word; multi-word
    phrases keep the original substring semantics.
    """

    # --- "mach" single-token cases ---

    def test_mach_not_matched_in_machinery(self):
        """Single-token 'mach' must NOT fire inside 'machinery'."""
        rows = [_row("b1", "The machinery processes the material.")]
        result = keyword_hit_counts(rows, ["mach"])
        assert result["b1"]["keyword_hits"] == 0

    def test_mach_not_matched_in_supermach(self):
        """Single-token 'mach' must NOT fire inside 'supermach'."""
        rows = [_row("b2", "Supermach performance is claimed.")]
        result = keyword_hit_counts(rows, ["mach"])
        assert result["b2"]["keyword_hits"] == 0

    def test_mach_matched_standalone(self):
        """Single-token 'mach' MUST match when surrounded by non-word chars."""
        rows = [_row("b3", "Speed: mach 2")]
        result = keyword_hit_counts(rows, ["mach"])
        assert result["b3"]["keyword_hits"] == 1

    def test_mach_matched_as_phrase_component(self):
        """'mach 2' is a multi-word phrase; substring semantics apply — matches."""
        rows = [_row("b4", "at mach 2 the missile is effective")]
        result = keyword_hit_counts(rows, ["mach 2"])
        assert result["b4"]["keyword_hits"] == 1

    def test_mach_not_matched_with_trailing_digit(self):
        """Trailing digit is \\w, so trailing guard (?!\\w) rejects 'mach' in 'mach2'.

        'mach2' has no whitespace separating the digit, so it does not match
        the single-token 'mach' needle.  Contrast with 'mach 2' which IS a
        multi-word phrase and uses substring semantics (handled separately).
        This mirrors unit-matcher discipline (e.g. 'km' does not match 'km/h'
        without a whitespace separator).
        """
        rows = [_row("b5", "at mach2 the missile is effective")]
        result = keyword_hit_counts(rows, ["mach"])
        assert result["b5"]["keyword_hits"] == 0

    # --- "fins" single-token cases ---

    def test_fins_not_matched_in_muffins(self):
        """Single-token 'fins' must NOT fire inside 'muffins'."""
        rows = [_row("b6", "The muffins were popular at the event.")]
        result = keyword_hit_counts(rows, ["fins"])
        assert result["b6"]["keyword_hits"] == 0

    def test_fins_matched_as_standalone(self):
        """Single-token 'fins' MUST match when it is a standalone token."""
        rows = [_row("b7", "The missile fins folded before launch.")]
        result = keyword_hit_counts(rows, ["fins"])
        assert result["b7"]["keyword_hits"] == 1

    # --- multi-word phrase keeps substring semantics ---

    def test_phrase_pulse_repetition_interval_substring(self):
        """Multi-word phrase 'pulse repetition interval' uses substring semantics.

        No boundary wrapping is applied — it should fire anywhere inside the text.
        """
        rows = [_row("b8", "The nominal pulse repetition interval is 3 ms.")]
        result = keyword_hit_counts(rows, ["pulse repetition interval"])
        assert result["b8"]["keyword_hits"] == 1

    def test_phrase_substring_match_inside_larger_text(self):
        """Multi-word phrase matches even when flanked by additional words."""
        rows = [_row("b9", "listed: pulse repetition interval value 3 ms herein.")]
        result = keyword_hit_counts(rows, ["pulse repetition interval"])
        assert result["b9"]["keyword_hits"] == 1

    # --- Cyrillic single-token boundary ---

    def test_cyrillic_single_token_not_matched_inside_longer_word(self):
        """Cyrillic single-token needle must not fire inside a longer Cyrillic word.

        'князь' (prince) must not match inside 'великокняжеский' (grand-ducal).
        Both the leading and trailing guards are Unicode-aware (\\w matches
        Cyrillic letters), so the boundary is applied correctly.
        """
        rows = [_row("b10", "великокняжеский титул был присвоен.")]
        result = keyword_hit_counts(rows, ["князь"])
        assert result["b10"]["keyword_hits"] == 0

    def test_cyrillic_single_token_matched_standalone(self):
        """Cyrillic single-token needle matches when it is a standalone token."""
        rows = [_row("b11", "Титул: князь Василий был назначен.")]
        result = keyword_hit_counts(rows, ["князь"])
        assert result["b11"]["keyword_hits"] == 1

    # --- hyphenated single tokens ---

    def test_hyphenated_single_token_matched_correctly(self):
        """Hyphenated token 'x-band' is a single token (no whitespace).

        Boundary guards apply at the outer edges (before 'x', after 'd').
        'x-band radar' → match; 'x-band' flanked by non-word chars → match.
        """
        rows = [_row("b12", "operates in the x-band frequency range")]
        result = keyword_hit_counts(rows, ["x-band"])
        assert result["b12"]["keyword_hits"] == 1

    def test_hyphenated_token_not_matched_inside_longer_word(self):
        """Hyphenated token must not match as a leading fragment of a longer word.

        'x-band' must not fire if the trailing 'd' is directly followed by
        a letter (i.e. 'x-bander').
        """
        rows = [_row("b13", "the x-bander system is installed")]
        result = keyword_hit_counts(rows, ["x-band"])
        assert result["b13"]["keyword_hits"] == 0

    # --- alias-channel regression (scope guard) ---

    def test_alias_channel_still_substring_for_single_token(self):
        """lexical_hit_counts (alias channel) is BYTE-IDENTICAL — still substring.

        'mach' in haystack 'machinery' MUST still fire in lexical_hit_counts even
        though keyword_hit_counts (this function) suppresses it.  This test pins
        the intentional asymmetry: the production alias-channel scoring (lexical_weight
        0.20 × alias_hits ratio) uses substring semantics and must NOT be changed.
        """
        rows = [_row("alias-r1", "machinery processing plant")]
        # keyword_hit_counts: bounded — no match
        kw_result = keyword_hit_counts(rows, ["mach"])
        assert kw_result["alias-r1"]["keyword_hits"] == 0  # single-token boundary

        # lexical_hit_counts: plain substring — still matches
        fqs = [_fq("some_field", aliases=("mach",))]
        lex_result = lexical_hit_counts(rows, fqs)
        # 'mach' IS a substring of 'machinery' → alias_hits == 1 (unchanged)
        assert lex_result["alias-r1"]["alias_hits"] == 1  # substring semantics intact
