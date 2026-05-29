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
    lexical_hit_counts,
    pattern_hit_counts,
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
