"""Unit tests for Task C4 — merge_candidates / MergedCandidate.

TDD spec: write tests first, verify they FAIL on missing module, then implement.
"""
from __future__ import annotations

import pytest

from app.services.graph_store import GraphEntityResult


# ---------------------------------------------------------------------------
# Helper — build a synthetic GraphEntityResult with chunk fields in .properties
# ---------------------------------------------------------------------------

def _make_ger(
    *,
    vertex_id: str | None = None,
    self_ref: str,
    chunk_index: int = 0,
    chunk_text: str = "some text",
    source_refs: list[str] | None = None,
    token_count: int = 50,
    page_number: int | None = None,
    score: float | None = None,
    node_id: str = "node-1",
) -> GraphEntityResult:
    """Synthetic GraphEntityResult with chunk fields inside .properties."""
    props: dict = {
        "self_ref": self_ref,
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
        "source_refs": source_refs if source_refs is not None else [],
        "token_count": token_count,
        "page_number": page_number,
    }
    if vertex_id is not None:
        props["vertex_id"] = vertex_id
    return GraphEntityResult(
        node_id=node_id,
        name="TestChunk",
        entity_type="ExtractionChunk",
        score=score,
        properties=props,
    )


# ---------------------------------------------------------------------------
# Watched-fail guard — import must succeed after implementation is written
# ---------------------------------------------------------------------------

class TestImport:
    def test_module_importable(self):
        from app.services.extraction_candidate_scoring import (  # noqa: F401
            MergedCandidate,
            merge_candidates,
        )


# ---------------------------------------------------------------------------
# C4.1 — Full aggregation: entity_dense + field_dense + lexical + pattern
# ---------------------------------------------------------------------------

class TestFullAggregation:
    def test_single_chunk_in_all_sources(self):
        """A chunk present in every source merges into ONE MergedCandidate
        with all retrieval_sources present and field hints unioned."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(
            vertex_id="run1:chunk_0",
            self_ref="#s1",
            chunk_index=0,
            chunk_text="The SA-2 has a range of 45 km.",
            source_refs=["#s1"],
            token_count=10,
            page_number=3,
            score=0.91,
        )
        # Same vertex_id / candidate_key appearing in field_dense under "max_range_km"
        ger_field = _make_ger(
            vertex_id="run1:chunk_0",
            self_ref="#s1",
            chunk_index=0,
            chunk_text="The SA-2 has a range of 45 km.",
            source_refs=["#s1"],
            token_count=10,
            page_number=3,
            score=0.87,
        )

        entity_dense = [ger]
        field_dense = {"max_range_km": [ger_field]}
        lexical_hits = {
            "run1:chunk_0": {
                "alias_hits": 2,
                "negative_hits": 0,
                "supported_fields": {"max_range_km"},
            }
        }
        pattern_hits = {
            "run1:chunk_0": {
                "pattern_hits": 1,
                "supported_fields": {"max_range_km", "guidance"},
            }
        }

        results = merge_candidates(
            entity_dense=entity_dense,
            field_dense=field_dense,
            lexical_hits=lexical_hits,
            pattern_hits=pattern_hits,
            section_meta={},
            table_meta={},
        )

        assert len(results) == 1
        mc = results[0]
        assert mc.candidate_key == "run1:chunk_0"
        assert mc.retrieval_sources == {"dense", "field:max_range_km", "lexical", "pattern"}
        assert mc.alias_hits == 2
        assert mc.negative_hits == 0
        assert mc.pattern_hits == 1
        assert mc.section_hits == 0
        assert mc.content_type is None
        # field_scores: best score from max_range_km list
        assert "max_range_km" in mc.field_scores
        assert mc.field_scores["max_range_km"] == pytest.approx(0.87)
        # vector_score from entity_dense
        assert mc.vector_score == pytest.approx(0.91)
        # supported_field_hints: union from lexical + pattern + field_dense membership
        assert "max_range_km" in mc.supported_field_hints
        assert "guidance" in mc.supported_field_hints

    def test_page_number_preserved(self):
        """page_number must survive end-to-end through the merge."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(
            vertex_id="run1:chunk_5",
            self_ref="#s5",
            page_number=17,
            score=0.80,
        )
        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        assert len(results) == 1
        assert results[0].page_number == 17

    def test_page_number_none_preserved(self):
        """page_number=None (no lineage) must also survive, not be coerced."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(vertex_id="run1:chunk_6", self_ref="#s6", page_number=None)
        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        assert results[0].page_number is None


# ---------------------------------------------------------------------------
# C4.2 — Fallback key (no vertex_id → uses self_ref)
# ---------------------------------------------------------------------------

class TestFallbackKey:
    def test_no_vertex_id_uses_self_ref_as_key(self):
        """When vertex_id absent from properties, candidate_key == self_ref."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(
            vertex_id=None,   # <-- no vertex_id in properties
            self_ref="#legacy",
            chunk_index=0,
            score=0.75,
        )
        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        assert len(results) == 1
        assert results[0].candidate_key == "#legacy"

    def test_fallback_key_still_participates_in_lexical_lookup(self):
        """A fallback-keyed candidate can still receive lexical_hits."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(vertex_id=None, self_ref="#legacy2", score=0.70)
        lexical_hits = {
            "#legacy2": {"alias_hits": 3, "negative_hits": 1, "supported_fields": {"name"}}
        }
        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits=lexical_hits,
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        assert results[0].alias_hits == 3
        assert results[0].negative_hits == 1
        assert "lexical" in results[0].retrieval_sources


# ---------------------------------------------------------------------------
# C4.3 — Collision guard: same self_ref, distinct vertex_id → SEPARATE candidates
# ---------------------------------------------------------------------------

class TestCollisionGuard:
    def test_same_self_ref_distinct_vertex_id_yields_two_candidates(self):
        """merged-mode: self_ref repeats across chunks; vertex_id is the true key.
        Two GERs with the same self_ref but different vertex_id must NOT merge."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger_a = _make_ger(
            vertex_id="run1:chunk_0",
            self_ref="#s0",  # <-- same self_ref
            chunk_index=0,
            chunk_text="Text A",
            score=0.90,
        )
        ger_b = _make_ger(
            vertex_id="run1:chunk_1",
            self_ref="#s0",  # <-- same self_ref, DIFFERENT vertex_id
            chunk_index=1,
            chunk_text="Text B",
            score=0.85,
        )

        results = merge_candidates(
            entity_dense=[ger_a, ger_b],
            field_dense={},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )

        assert len(results) == 2, (
            "Two GERs with same self_ref but distinct vertex_id must produce "
            "two SEPARATE MergedCandidates"
        )
        keys = {mc.candidate_key for mc in results}
        assert keys == {"run1:chunk_0", "run1:chunk_1"}

    def test_same_vertex_id_deduplicates(self):
        """Two GERs with the SAME vertex_id (e.g. appeared in both entity_dense
        and field_dense) must merge into ONE candidate."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger_e = _make_ger(vertex_id="run1:chunk_2", self_ref="#s2", score=0.92)
        ger_f = _make_ger(vertex_id="run1:chunk_2", self_ref="#s2", score=0.88)

        results = merge_candidates(
            entity_dense=[ger_e],
            field_dense={"speed": [ger_f]},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        assert len(results) == 1
        assert results[0].candidate_key == "run1:chunk_2"


# ---------------------------------------------------------------------------
# C4.4 — No duplicate candidate_keys in output
# ---------------------------------------------------------------------------

class TestNoDuplicateKeys:
    def test_output_has_unique_candidate_keys(self):
        """The output list must never have duplicate candidate_key values."""
        from app.services.extraction_candidate_scoring import merge_candidates

        gx = _make_ger(vertex_id="run1:chunk_9", self_ref="#s9", score=0.80)
        gy = _make_ger(vertex_id="run1:chunk_10", self_ref="#s10", score=0.79)

        results = merge_candidates(
            entity_dense=[gx, gy],
            field_dense={"speed": [gx]},
            lexical_hits={"run1:chunk_9": {"alias_hits": 1, "negative_hits": 0, "supported_fields": set()}},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        keys = [mc.candidate_key for mc in results]
        assert len(keys) == len(set(keys)), f"Duplicate keys found: {keys}"


# ---------------------------------------------------------------------------
# C4.5 — field_scores: best (max) score per field across multiple hits
# ---------------------------------------------------------------------------

class TestFieldScoresBest:
    def test_best_score_wins_per_field(self):
        """When a field_dense list contains the same chunk twice (or with
        varying scores), field_scores[field] = max score."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger_low  = _make_ger(vertex_id="run1:chunk_3", self_ref="#s3", score=0.60)
        ger_high = _make_ger(vertex_id="run1:chunk_3", self_ref="#s3", score=0.95)

        results = merge_candidates(
            entity_dense=[],
            field_dense={"guidance": [ger_low, ger_high]},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        assert len(results) == 1
        assert results[0].field_scores["guidance"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# C4.6 — Empty inputs produce empty output
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_all_empty(self):
        from app.services.extraction_candidate_scoring import merge_candidates

        results = merge_candidates(
            entity_dense=[],
            field_dense={},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        assert results == []


# ---------------------------------------------------------------------------
# C4.7 — section_meta and table_meta plumbing
# ---------------------------------------------------------------------------

class TestSectionAndTableMeta:
    def test_section_hits_from_section_meta(self):
        """section_meta[key]["section_hits"] is read into MergedCandidate."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(vertex_id="run1:chunk_4", self_ref="#s4")
        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={},
            pattern_hits={},
            section_meta={"run1:chunk_4": {"section_hits": 5}},
            table_meta={},
        )
        assert results[0].section_hits == 5

    def test_section_hits_absent_defaults_to_zero(self):
        """If candidate_key not in section_meta, section_hits == 0."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(vertex_id="run1:chunk_4b", self_ref="#s4b")
        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        assert results[0].section_hits == 0

    def test_content_type_from_table_meta(self):
        """table_meta[key] is read into content_type."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(vertex_id="run1:chunk_7", self_ref="#s7")
        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={"run1:chunk_7": "table"},
        )
        assert results[0].content_type == "table"

    def test_content_type_absent_is_none(self):
        """When table_meta is empty (Phase D deferred), content_type=None."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(vertex_id="run1:chunk_8", self_ref="#s8")
        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        assert results[0].content_type is None


# ---------------------------------------------------------------------------
# C4.8 — supported_field_hints union from all three sources
# ---------------------------------------------------------------------------

class TestSupportedFieldHints:
    def test_hints_union_from_all_sources(self):
        """supported_field_hints = union(lexical.supported_fields,
        pattern.supported_fields, field_dense membership)."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(vertex_id="run1:chunk_11", self_ref="#s11", score=0.80)
        results = merge_candidates(
            entity_dense=[ger],
            field_dense={"speed": [ger]},
            lexical_hits={"run1:chunk_11": {"alias_hits": 1, "negative_hits": 0, "supported_fields": {"range"}}},
            pattern_hits={"run1:chunk_11": {"pattern_hits": 2, "supported_fields": {"guidance", "range"}}},
            section_meta={},
            table_meta={},
        )
        hints = results[0].supported_field_hints
        assert "range" in hints
        assert "guidance" in hints
        assert "speed" in hints  # from field_dense membership

    def test_hints_empty_when_no_signal(self):
        """If no lexical/pattern/field_dense signal, supported_field_hints is empty."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(vertex_id="run1:chunk_12", self_ref="#s12")
        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        assert results[0].supported_field_hints == set()


# ---------------------------------------------------------------------------
# C4.9 — Provenance completeness (chunk fields survive merge)
# ---------------------------------------------------------------------------

class TestProvenanceFields:
    def test_chunk_fields_populated_on_merged_candidate(self):
        """chunk_index, self_ref, chunk_text, source_refs, token_count all
        survive into MergedCandidate with correct values."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(
            vertex_id="run1:chunk_13",
            self_ref="#s13",
            chunk_index=13,
            chunk_text="Detailed specs follow.",
            source_refs=["#s13", "#s13b"],
            token_count=99,
            page_number=7,
            score=0.77,
        )
        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={},
            pattern_hits={},
            section_meta={},
            table_meta={},
        )
        mc = results[0]
        assert mc.chunk_index == 13
        assert mc.self_ref == "#s13"
        assert mc.chunk_text == "Detailed specs follow."
        assert mc.source_refs == ["#s13", "#s13b"]
        assert mc.token_count == 99
        assert mc.page_number == 7
        assert mc.vector_score == pytest.approx(0.77)
