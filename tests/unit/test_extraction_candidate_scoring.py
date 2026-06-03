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


# ---------------------------------------------------------------------------
# C4.10 — Keyword-only keys are DROPPED (keyword = precision not recall)
# ---------------------------------------------------------------------------

class TestKeywordOnlyDropped:
    """Plan-locked rule: lexical/pattern signals BOOST dense candidates only.
    Keys absent from entity_dense + field_dense must NOT produce MergedCandidates.
    The recall safety net is Phase E's lexical_table fallback, not this merge.
    """

    def test_lexical_only_key_absent_from_dense_is_dropped(self):
        """A candidate_key present ONLY in lexical_hits (not in entity_dense or
        field_dense) must NOT appear in the merge output."""
        from app.services.extraction_candidate_scoring import merge_candidates

        # One real dense candidate
        ger = _make_ger(vertex_id="run1:chunk_0", self_ref="#s0", score=0.85)

        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={
                "run1:chunk_0": {"alias_hits": 1, "negative_hits": 0, "supported_fields": set()},
                "run1:chunk_LEXONLY": {"alias_hits": 5, "negative_hits": 0, "supported_fields": {"range"}},
            },
            pattern_hits={},
            section_meta={},
            table_meta={},
        )

        keys = [mc.candidate_key for mc in results]
        assert "run1:chunk_LEXONLY" not in keys, (
            "Lexical-only candidate (absent from dense) must be dropped, not admitted"
        )
        assert "run1:chunk_0" in keys, "Dense candidate must still appear"
        assert len(results) == 1

    def test_pattern_only_key_absent_from_dense_is_dropped(self):
        """A candidate_key present ONLY in pattern_hits (not in dense) must be dropped."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(vertex_id="run1:chunk_0", self_ref="#s0", score=0.85)

        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={},
            pattern_hits={
                "run1:chunk_0": {"pattern_hits": 1, "supported_fields": set()},
                "run1:chunk_PATONLY": {"pattern_hits": 7, "supported_fields": {"guidance"}},
            },
            section_meta={},
            table_meta={},
        )

        keys = [mc.candidate_key for mc in results]
        assert "run1:chunk_PATONLY" not in keys, (
            "Pattern-only candidate (absent from dense) must be dropped, not admitted"
        )
        assert "run1:chunk_0" in keys
        assert len(results) == 1

    def test_lexical_and_pattern_only_key_absent_from_dense_is_dropped(self):
        """A key in BOTH lexical_hits AND pattern_hits but absent from dense is still dropped."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(vertex_id="run1:chunk_0", self_ref="#s0", score=0.80)

        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={"run1:chunk_BOTH_ONLY": {"alias_hits": 3, "negative_hits": 0, "supported_fields": set()}},
            pattern_hits={"run1:chunk_BOTH_ONLY": {"pattern_hits": 4, "supported_fields": set()}},
            section_meta={},
            table_meta={},
        )

        keys = [mc.candidate_key for mc in results]
        assert "run1:chunk_BOTH_ONLY" not in keys, (
            "Key in both lexical+pattern but absent from dense must still be dropped"
        )
        assert len(results) == 1

    def test_dense_candidate_with_lexical_and_pattern_gets_signals_attached(self):
        """A key present in dense that ALSO appears in lexical_hits and pattern_hits
        must correctly receive alias_hits, pattern_hits, and retrieval_sources tags."""
        from app.services.extraction_candidate_scoring import merge_candidates

        ger = _make_ger(vertex_id="run1:chunk_5", self_ref="#s5", score=0.88)

        results = merge_candidates(
            entity_dense=[ger],
            field_dense={},
            lexical_hits={
                "run1:chunk_5": {"alias_hits": 4, "negative_hits": 1, "supported_fields": {"max_range_km"}},
            },
            pattern_hits={
                "run1:chunk_5": {"pattern_hits": 2, "supported_fields": {"guidance"}},
            },
            section_meta={},
            table_meta={},
        )

        assert len(results) == 1
        mc = results[0]
        assert mc.candidate_key == "run1:chunk_5"
        assert mc.alias_hits == 4
        assert mc.negative_hits == 1
        assert mc.pattern_hits == 2
        assert "lexical" in mc.retrieval_sources
        assert "pattern" in mc.retrieval_sources
        assert "dense" in mc.retrieval_sources
        assert "max_range_km" in mc.supported_field_hints
        assert "guidance" in mc.supported_field_hints


# ===========================================================================
# C5 — score_candidates (post-rerank precision scoring)
# ===========================================================================
#
# Candidate representation fed to score_candidates:
#   A list of dicts, each carrying:
#     - "merged_candidate": MergedCandidate   (the C4 object)
#     - "content_text": str                   (chunk_text — for reranker compat)
#     - "reranker_score": float | absent       (written by rerank(); absent on unscorable)
#   Signals (alias_hits, pattern_hits, negative_hits, section_hits, content_type)
#   are read directly from the embedded MergedCandidate.
#
# Normalization: min-max for reranker_score (missing → 0.0); ratio-max for
# lexical/pattern/section/negative signals (alias_hits / max(1, pool_max), etc.)
# ===========================================================================


def _make_profile(**kwargs):
    """Build a RetrievalProfile with sane defaults, overridable by kwargs."""
    from app.services.ontology_bundles import RetrievalProfile

    defaults = dict(
        rerank_weight=1.0,
        lexical_weight=0.20,
        pattern_weight=0.15,
        section_weight=0.10,
        table_boost=0.08,
        negative_weight=0.20,
    )
    defaults.update(kwargs)
    return RetrievalProfile(**defaults)


def _make_scored_candidate(
    candidate_key: str,
    chunk_text: str = "some text",
    reranker_score: float | None = None,
    alias_hits: int = 0,
    pattern_hits: int = 0,
    negative_hits: int = 0,
    section_hits: int = 0,
    content_type: str | None = None,
) -> dict:
    """Build a dict in the representation score_candidates expects."""
    from app.services.extraction_candidate_scoring import MergedCandidate

    mc = MergedCandidate(
        candidate_key=candidate_key,
        chunk_index=0,
        self_ref=candidate_key,
        chunk_text=chunk_text,
        source_refs=[],
        token_count=50,
        page_number=None,
        vector_score=None,
        field_scores={},
        alias_hits=alias_hits,
        pattern_hits=pattern_hits,
        negative_hits=negative_hits,
        section_hits=section_hits,
        content_type=content_type,
        retrieval_sources=set(),
        supported_field_hints=set(),
    )
    d: dict = {
        "merged_candidate": mc,
        "content_text": chunk_text,
    }
    if reranker_score is not None:
        d["reranker_score"] = reranker_score
    return d


class TestScoreCandidatesImport:
    def test_score_candidates_importable(self):
        from app.services.extraction_candidate_scoring import score_candidates  # noqa: F401


class TestScoreCandidatesReankOnly:
    """C5.1 — With all keyword signals zero, final order EQUALS rerank order."""

    def test_rerank_order_preserved_when_no_keyword_signals(self):
        from app.services.extraction_candidate_scoring import score_candidates

        cfg = _make_profile()
        cands = [
            _make_scored_candidate("c1", reranker_score=0.9),
            _make_scored_candidate("c2", reranker_score=0.5),
            _make_scored_candidate("c3", reranker_score=0.1),
        ]
        result = score_candidates(cands, cfg)
        keys = [mc.candidate_key for mc, _ in result]
        assert keys == ["c1", "c2", "c3"], f"Expected rerank order, got {keys}"

    def test_returns_list_of_mc_float_tuples(self):
        from app.services.extraction_candidate_scoring import MergedCandidate, score_candidates

        cfg = _make_profile()
        cands = [_make_scored_candidate("c1", reranker_score=0.8)]
        result = score_candidates(cands, cfg)
        assert isinstance(result, list)
        assert len(result) == 1
        mc, score = result[0]
        assert isinstance(mc, MergedCandidate)
        assert isinstance(score, float)


class TestScoreCandidatesLexicalPromotion:
    """C5.2 — Strong alias hits promote a candidate ABOVE a higher-rerank rival."""

    def test_alias_hits_promote_above_higher_rerank(self):
        from app.services.extraction_candidate_scoring import score_candidates

        # c_low has a better reranker_score but zero keyword signals.
        # c_boost has a weaker reranker_score but strong alias hits.
        # With appropriate weights, c_boost should finish above c_low.
        cfg = _make_profile(
            rerank_weight=1.0,
            lexical_weight=2.0,   # strong lexical weight
        )
        c_high_rerank = _make_scored_candidate("c_high_rerank", reranker_score=0.9, alias_hits=0)
        c_boost = _make_scored_candidate("c_boost", reranker_score=0.1, alias_hits=5)

        result = score_candidates([c_high_rerank, c_boost], cfg)
        keys = [mc.candidate_key for mc, _ in result]
        assert keys[0] == "c_boost", (
            f"c_boost (strong alias hits) should lead after scoring; got {keys}"
        )

    def test_promotion_changes_selection_order_demonstrably(self):
        """The promotion is observable: scores for c_boost > c_high_rerank."""
        from app.services.extraction_candidate_scoring import score_candidates

        cfg = _make_profile(rerank_weight=1.0, lexical_weight=2.0)
        c_high_rerank = _make_scored_candidate("c_high_rerank", reranker_score=0.9, alias_hits=0)
        c_boost = _make_scored_candidate("c_boost", reranker_score=0.1, alias_hits=5)

        result = score_candidates([c_high_rerank, c_boost], cfg)
        scores = {mc.candidate_key: s for mc, s in result}
        assert scores["c_boost"] > scores["c_high_rerank"], (
            f"c_boost score ({scores['c_boost']:.4f}) must exceed "
            f"c_high_rerank ({scores['c_high_rerank']:.4f})"
        )


class TestScoreCandidatesNegativeDemote:
    """C5.3 — negative_hits demote but do NOT remove the candidate."""

    def test_only_negative_candidate_still_present(self):
        from app.services.extraction_candidate_scoring import score_candidates

        cfg = _make_profile(negative_weight=0.5)
        cands = [
            _make_scored_candidate("c_neg", reranker_score=0.7, negative_hits=10),
            _make_scored_candidate("c_pos", reranker_score=0.5),
        ]
        result = score_candidates(cands, cfg)
        keys = [mc.candidate_key for mc, _ in result]
        assert "c_neg" in keys, "Negative candidate must not be filtered out"

    def test_negative_candidate_demoted_below_clean_candidate(self):
        from app.services.extraction_candidate_scoring import score_candidates

        # Scenario: 3 candidates so that min-max can spread scores.
        # c_neg: best reranker (1.0 → rerank_norm=1.0) but heavy negatives.
        # c_clean: mid reranker (0.5 → rerank_norm=0.5), zero negatives.
        # c_floor: worst reranker (0.0 → rerank_norm=0.0), zero negatives.
        # negative_weight=0.9 so: c_neg final = 1.0*1.0 - 0.9*1.0 = 0.1
        # c_clean final = 1.0*0.5 = 0.5 → should finish above c_neg.
        cfg = _make_profile(
            rerank_weight=1.0,
            lexical_weight=0.0,
            pattern_weight=0.0,
            section_weight=0.0,
            table_boost=0.0,
            negative_weight=0.9,
        )
        c_neg = _make_scored_candidate("c_neg", reranker_score=1.0, negative_hits=3)
        c_clean = _make_scored_candidate("c_clean", reranker_score=0.5, negative_hits=0)
        c_floor = _make_scored_candidate("c_floor", reranker_score=0.0, negative_hits=0)

        result = score_candidates([c_neg, c_clean, c_floor], cfg)
        keys = [mc.candidate_key for mc, _ in result]
        assert keys[0] == "c_clean", (
            f"c_clean should lead after negative penalty demotes c_neg; got {keys}"
        )

    def test_final_score_clamped_to_zero(self):
        """Score >= 0 even when negative penalty exceeds all positive signals."""
        from app.services.extraction_candidate_scoring import score_candidates

        cfg = _make_profile(
            rerank_weight=0.0,
            lexical_weight=0.0,
            pattern_weight=0.0,
            section_weight=0.0,
            table_boost=0.0,
            negative_weight=99.0,   # absurd penalty
        )
        cands = [_make_scored_candidate("c", reranker_score=0.1, negative_hits=1)]
        result = score_candidates(cands, cfg)
        _, score = result[0]
        assert score >= 0.0, f"Score must be clamped >= 0, got {score}"


class TestScoreCandidatesAllZeroKeyword:
    """C5.4 — All keyword signals zero — order unchanged from rerank."""

    def test_all_zero_keyword_order_equals_rerank_order(self):
        from app.services.extraction_candidate_scoring import score_candidates

        cfg = _make_profile()
        cands = [
            _make_scored_candidate("a", reranker_score=0.8, alias_hits=0, pattern_hits=0),
            _make_scored_candidate("b", reranker_score=0.6, alias_hits=0, pattern_hits=0),
            _make_scored_candidate("c", reranker_score=0.2, alias_hits=0, pattern_hits=0),
        ]
        result = score_candidates(cands, cfg)
        keys = [mc.candidate_key for mc, _ in result]
        assert keys == ["a", "b", "c"]


class TestScoreCandidatesUnscorableHandling:
    """C5.5 — Candidates missing reranker_score (unscorable) — no KeyError, rerank_norm=0."""

    def test_missing_reranker_score_no_keyerror(self):
        from app.services.extraction_candidate_scoring import score_candidates

        cfg = _make_profile()
        cands = [
            _make_scored_candidate("c_scored", reranker_score=0.7),
            _make_scored_candidate("c_unscorable"),  # no reranker_score key
        ]
        # Must not raise
        result = score_candidates(cands, cfg)
        keys = [mc.candidate_key for mc, _ in result]
        assert "c_unscorable" in keys

    def test_unscorable_gets_zero_rerank_norm(self):
        """Unscorable candidate's contribution from the rerank term is 0,
        so it should rank below any scored candidate when rerank_weight > 0."""
        from app.services.extraction_candidate_scoring import score_candidates

        cfg = _make_profile(rerank_weight=1.0, lexical_weight=0.0, pattern_weight=0.0, section_weight=0.0)
        cands = [
            _make_scored_candidate("c_scored", reranker_score=0.7),
            _make_scored_candidate("c_unscorable"),   # no reranker_score
        ]
        result = score_candidates(cands, cfg)
        keys = [mc.candidate_key for mc, _ in result]
        assert keys[0] == "c_scored", f"Scored candidate should rank first; got {keys}"

    def test_all_unscorable_pool_handled(self):
        """All candidates missing reranker_score — no divide-by-zero, all returned."""
        from app.services.extraction_candidate_scoring import score_candidates

        cfg = _make_profile()
        cands = [
            _make_scored_candidate("u1"),
            _make_scored_candidate("u2"),
        ]
        result = score_candidates(cands, cfg)
        assert len(result) == 2


class TestScoreCandidatesSortStability:
    """C5.6 — Ties broken by reranker_score desc then candidate_key (stable)."""

    def test_tie_broken_by_reranker_score(self):
        from app.services.extraction_candidate_scoring import score_candidates

        # Same alias_hits → same lexical_norm; differ only in reranker_score
        cfg = _make_profile(rerank_weight=0.0, lexical_weight=1.0)
        cands = [
            _make_scored_candidate("z_lower", reranker_score=0.3, alias_hits=2),
            _make_scored_candidate("a_higher", reranker_score=0.8, alias_hits=2),
        ]
        result = score_candidates(cands, cfg)
        keys = [mc.candidate_key for mc, _ in result]
        # Both have identical final scores (rerank_weight=0 makes reranker_score
        # irrelevant to final, but tiebreaker uses it); a_higher should lead
        assert keys[0] == "a_higher"

    def test_tie_broken_by_candidate_key_lexicographic(self):
        """When final AND reranker_score are both equal (all unscorable), key order is stable."""
        from app.services.extraction_candidate_scoring import score_candidates

        cfg = _make_profile(rerank_weight=0.0, lexical_weight=0.0)
        cands = [
            _make_scored_candidate("z_key"),
            _make_scored_candidate("a_key"),
        ]
        result = score_candidates(cands, cfg)
        keys = [mc.candidate_key for mc, _ in result]
        assert keys == ["a_key", "z_key"]


class TestScoreCandidatesWeightsFromCfg:
    """C5.7 — All weights read from cfg; construct a RetrievalProfile with custom
    weights and assert they take effect."""

    def test_rerank_weight_zero_suppresses_rerank_contribution(self):
        """With rerank_weight=0, reranker_score has zero contribution to final."""
        from app.services.extraction_candidate_scoring import score_candidates

        # Give c_low a much better reranker but equal alias_hits.
        # With rerank_weight=0, they should be equal on final and tiebreak decides.
        cfg = _make_profile(rerank_weight=0.0, lexical_weight=1.0)
        cands = [
            _make_scored_candidate("c_low", reranker_score=0.1, alias_hits=3),
            _make_scored_candidate("c_high", reranker_score=0.9, alias_hits=3),
        ]
        result = score_candidates(cands, cfg)
        scores = {mc.candidate_key: s for mc, s in result}
        # Scores should be identical (both have equal alias_hits, rerank_weight=0)
        assert abs(scores["c_low"] - scores["c_high"]) < 1e-9

    def test_pattern_weight_takes_effect(self):
        """pattern_weight from cfg controls pattern_norm contribution."""
        from app.services.extraction_candidate_scoring import score_candidates

        cfg_low = _make_profile(rerank_weight=1.0, pattern_weight=0.0)
        cfg_high = _make_profile(rerank_weight=1.0, pattern_weight=5.0)

        c_with_pattern = _make_scored_candidate("c_pattern", reranker_score=0.5, pattern_hits=3)
        c_no_pattern = _make_scored_candidate("c_baseline", reranker_score=0.5, pattern_hits=0)

        # Low pattern_weight: c_pattern not distinguished
        res_low = score_candidates([c_with_pattern, c_no_pattern], cfg_low)
        # High pattern_weight: c_pattern clearly above c_baseline
        res_high = score_candidates([c_with_pattern, c_no_pattern], cfg_high)

        keys_high = [mc.candidate_key for mc, _ in res_high]
        assert keys_high[0] == "c_pattern", (
            f"High pattern_weight should put c_pattern first; got {keys_high}"
        )
        # With pattern_weight=0, c_pattern and c_baseline have same reranker_score
        # → equal final; order determined by tiebreak
        scores_low = {mc.candidate_key: s for mc, s in res_low}
        assert abs(scores_low["c_pattern"] - scores_low["c_baseline"]) < 1e-9

    def test_negative_weight_zero_ignores_negative_hits(self):
        """negative_weight=0 means negative_hits don't penalize the score."""
        from app.services.extraction_candidate_scoring import score_candidates

        cfg = _make_profile(negative_weight=0.0)
        c_neg = _make_scored_candidate("c_neg", reranker_score=0.8, negative_hits=100)
        c_pos = _make_scored_candidate("c_pos", reranker_score=0.4)

        result = score_candidates([c_neg, c_pos], cfg)
        keys = [mc.candidate_key for mc, _ in result]
        assert keys[0] == "c_neg", "With negative_weight=0, high reranker should still lead"

    def test_table_boost_applies_to_table_content_type(self):
        """table_boost from cfg is added when content_type == 'table'."""
        from app.services.extraction_candidate_scoring import score_candidates

        # Same reranker, no other signals. Only difference: one is a table.
        cfg = _make_profile(rerank_weight=1.0, table_boost=1.0)
        c_table = _make_scored_candidate("c_table", reranker_score=0.5, content_type="table")
        c_prose = _make_scored_candidate("c_prose", reranker_score=0.5)

        result = score_candidates([c_table, c_prose], cfg)
        scores = {mc.candidate_key: s for mc, s in result}
        assert scores["c_table"] > scores["c_prose"], (
            f"Table boost must increase score: c_table={scores['c_table']:.4f}, "
            f"c_prose={scores['c_prose']:.4f}"
        )

    def test_empty_pool_returns_empty(self):
        from app.services.extraction_candidate_scoring import score_candidates

        cfg = _make_profile()
        result = score_candidates([], cfg)
        assert result == []


# ===========================================================================
# E1 — fallback-decision helpers (pure functions, no DB/LLM/reranker)
# ===========================================================================
#
# field_coverage(candidates) -> dict[str, int]
#   {field_name: count of candidates whose supported_field_hints include it}
#
# enough_candidates(candidates, cfg) -> bool
#   count of candidates with REAL retrieval signal >= min(cfg.top_k, 10)
#   "real signal" == non-empty retrieval_sources (at least one genuine tag).
#   An all-noise pool (retrieval_sources == set() on every candidate) → False.
#
# enough_field_coverage(candidates, cfg) -> bool
#   len([f for f, n in field_coverage(candidates).items() if n > 0])
#     >= cfg.fallback_min_field_coverage
# ===========================================================================


def _make_mc(
    candidate_key: str = "k",
    retrieval_sources: set[str] | None = None,
    supported_field_hints: set[str] | None = None,
    alias_hits: int = 0,
    pattern_hits: int = 0,
) -> "MergedCandidate":
    """Build a minimal MergedCandidate for E1 tests."""
    from app.services.extraction_candidate_scoring import MergedCandidate

    return MergedCandidate(
        candidate_key=candidate_key,
        chunk_index=0,
        self_ref=candidate_key,
        chunk_text="text",
        source_refs=[],
        token_count=50,
        page_number=None,
        vector_score=None,
        field_scores={},
        alias_hits=alias_hits,
        pattern_hits=pattern_hits,
        negative_hits=0,
        section_hits=0,
        content_type=None,
        retrieval_sources=retrieval_sources if retrieval_sources is not None else set(),
        supported_field_hints=supported_field_hints if supported_field_hints is not None else set(),
    )


def _make_e1_profile(top_k: int = 10, fallback_min_field_coverage: int = 2, **kwargs):
    """RetrievalProfile for E1 tests — only top_k and fallback_min_field_coverage matter."""
    return _make_profile(top_k=top_k, fallback_min_field_coverage=fallback_min_field_coverage, **kwargs)


class TestE1Import:
    def test_helpers_importable(self):
        from app.services.extraction_candidate_scoring import (  # noqa: F401
            field_coverage,
            enough_candidates,
            enough_field_coverage,
        )


class TestE1EmptyPool:
    """Empty pool → all helpers reflect no signal."""

    def test_field_coverage_empty_pool_returns_empty_dict(self):
        from app.services.extraction_candidate_scoring import field_coverage

        assert field_coverage([]) == {}

    def test_enough_candidates_empty_pool_is_false(self):
        from app.services.extraction_candidate_scoring import enough_candidates

        cfg = _make_e1_profile(top_k=10)
        assert enough_candidates([], cfg) is False

    def test_enough_field_coverage_empty_pool_is_false(self):
        from app.services.extraction_candidate_scoring import enough_field_coverage

        cfg = _make_e1_profile(top_k=10, fallback_min_field_coverage=1)
        assert enough_field_coverage([], cfg) is False


class TestE1AllNoisePool:
    """KEY CASE: non-empty pool with NO real retrieval signal must trigger fallback.

    "Noise" candidates have retrieval_sources == set() — they were never tagged
    by any retrieval channel.  enough_candidates must return False so that the
    caller fires the fallback path rather than suppressing it.
    """

    def test_enough_candidates_all_noise_is_false(self):
        """[WATCHED-FAIL] All-noise pool must NOT satisfy enough_candidates."""
        from app.services.extraction_candidate_scoring import enough_candidates

        # 15 noise candidates — more than min(top_k=10, 10) but zero real signal
        noise_pool = [
            _make_mc(f"noise_{i}", retrieval_sources=set())
            for i in range(15)
        ]
        cfg = _make_e1_profile(top_k=10)
        # Must be False: all-noise means no real retrieval happened
        assert enough_candidates(noise_pool, cfg) is False

    def test_enough_candidates_single_noise_candidate_is_false(self):
        from app.services.extraction_candidate_scoring import enough_candidates

        cfg = _make_e1_profile(top_k=1)
        noise = [_make_mc("n", retrieval_sources=set())]
        assert enough_candidates(noise, cfg) is False

    def test_field_coverage_noise_candidates_with_field_hints_still_counts(self):
        """field_coverage is purely about supported_field_hints — it counts regardless
        of retrieval signal quality (it's a separate dimension from enough_candidates)."""
        from app.services.extraction_candidate_scoring import field_coverage

        # A noise candidate can still declare field hints (edge case)
        noise = [_make_mc("n", retrieval_sources=set(), supported_field_hints={"range"})]
        cov = field_coverage(noise)
        assert cov.get("range", 0) == 1


class TestE1SparsePool:
    """Sparse real pool (below threshold) → False; at/above threshold → True."""

    def test_sparse_pool_below_threshold_is_false(self):
        """3 real candidates when min(top_k=10, 10)=10 → not enough."""
        from app.services.extraction_candidate_scoring import enough_candidates

        pool = [
            _make_mc(f"c{i}", retrieval_sources={"dense"})
            for i in range(3)
        ]
        cfg = _make_e1_profile(top_k=10)
        assert enough_candidates(pool, cfg) is False

    def test_pool_exactly_at_threshold_is_true(self):
        """Exactly min(top_k=10, 10)=10 real candidates → True."""
        from app.services.extraction_candidate_scoring import enough_candidates

        pool = [
            _make_mc(f"c{i}", retrieval_sources={"dense"})
            for i in range(10)
        ]
        cfg = _make_e1_profile(top_k=10)
        assert enough_candidates(pool, cfg) is True

    def test_pool_above_threshold_is_true(self):
        """20 real candidates when threshold is 10 → True."""
        from app.services.extraction_candidate_scoring import enough_candidates

        pool = [
            _make_mc(f"c{i}", retrieval_sources={"field:max_range_km"})
            for i in range(20)
        ]
        cfg = _make_e1_profile(top_k=20)
        assert enough_candidates(pool, cfg) is True

    def test_top_k_above_10_caps_threshold_at_10(self):
        """min(top_k, 10) caps at 10: top_k=50 still requires only 10 real candidates."""
        from app.services.extraction_candidate_scoring import enough_candidates

        pool = [
            _make_mc(f"c{i}", retrieval_sources={"lexical"})
            for i in range(10)
        ]
        cfg = _make_e1_profile(top_k=50)
        assert enough_candidates(pool, cfg) is True

    def test_top_k_below_10_uses_top_k_as_threshold(self):
        """When top_k < 10, threshold = top_k (not 10). 3 real with top_k=3 → True."""
        from app.services.extraction_candidate_scoring import enough_candidates

        pool = [
            _make_mc(f"c{i}", retrieval_sources={"pattern"})
            for i in range(3)
        ]
        cfg = _make_e1_profile(top_k=3)
        assert enough_candidates(pool, cfg) is True

    def test_mixed_real_and_noise_counts_only_real(self):
        """A pool of 5 real + 10 noise candidates: only real ones count toward threshold."""
        from app.services.extraction_candidate_scoring import enough_candidates

        real_pool = [
            _make_mc(f"real_{i}", retrieval_sources={"dense"})
            for i in range(5)
        ]
        noise_pool = [
            _make_mc(f"noise_{i}", retrieval_sources=set())
            for i in range(10)
        ]
        cfg = _make_e1_profile(top_k=10)
        # 5 real < 10 threshold → False
        assert enough_candidates(real_pool + noise_pool, cfg) is False


class TestE1FieldCoverage:
    """field_coverage aggregates supported_field_hints across the pool."""

    def test_single_candidate_single_field(self):
        from app.services.extraction_candidate_scoring import field_coverage

        pool = [_make_mc("c", supported_field_hints={"range"})]
        assert field_coverage(pool) == {"range": 1}

    def test_multiple_candidates_same_field(self):
        from app.services.extraction_candidate_scoring import field_coverage

        pool = [
            _make_mc("c0", supported_field_hints={"range"}),
            _make_mc("c1", supported_field_hints={"range"}),
            _make_mc("c2", supported_field_hints={"range", "speed"}),
        ]
        cov = field_coverage(pool)
        assert cov["range"] == 3
        assert cov["speed"] == 1

    def test_no_overlap_across_candidates(self):
        from app.services.extraction_candidate_scoring import field_coverage

        pool = [
            _make_mc("c0", supported_field_hints={"range"}),
            _make_mc("c1", supported_field_hints={"guidance"}),
            _make_mc("c2", supported_field_hints={"speed"}),
        ]
        cov = field_coverage(pool)
        assert cov == {"range": 1, "guidance": 1, "speed": 1}

    def test_no_field_hints_returns_empty(self):
        from app.services.extraction_candidate_scoring import field_coverage

        pool = [_make_mc("c", supported_field_hints=set())]
        assert field_coverage(pool) == {}


class TestE1FieldCoverageEnough:
    """enough_field_coverage checks ≥ fallback_min_field_coverage fields have n > 0."""

    def test_enough_fields_covered(self):
        from app.services.extraction_candidate_scoring import enough_field_coverage

        pool = [
            _make_mc("c0", supported_field_hints={"range", "guidance"}),
            _make_mc("c1", supported_field_hints={"speed"}),
        ]
        # 3 distinct fields covered, threshold=2 → True
        cfg = _make_e1_profile(fallback_min_field_coverage=2)
        assert enough_field_coverage(pool, cfg) is True

    def test_exactly_at_field_threshold_is_true(self):
        from app.services.extraction_candidate_scoring import enough_field_coverage

        pool = [
            _make_mc("c0", supported_field_hints={"range"}),
            _make_mc("c1", supported_field_hints={"guidance"}),
        ]
        cfg = _make_e1_profile(fallback_min_field_coverage=2)
        assert enough_field_coverage(pool, cfg) is True

    def test_below_field_threshold_is_false(self):
        from app.services.extraction_candidate_scoring import enough_field_coverage

        pool = [_make_mc("c0", supported_field_hints={"range"})]
        cfg = _make_e1_profile(fallback_min_field_coverage=2)
        assert enough_field_coverage(pool, cfg) is False

    def test_empty_pool_field_coverage_false(self):
        from app.services.extraction_candidate_scoring import enough_field_coverage

        cfg = _make_e1_profile(fallback_min_field_coverage=1)
        assert enough_field_coverage([], cfg) is False

    def test_well_covered_pool_both_true(self):
        """Integration: a well-covered pool satisfies both enough_candidates
        and enough_field_coverage."""
        from app.services.extraction_candidate_scoring import (
            enough_candidates,
            enough_field_coverage,
        )

        fields = ["range", "guidance", "speed", "warhead", "propulsion"]
        pool = [
            _make_mc(
                f"c{i}",
                retrieval_sources={"dense", f"field:{fields[i % len(fields)]}"},
                supported_field_hints={fields[i % len(fields)]},
            )
            for i in range(12)  # 12 real candidates
        ]
        cfg = _make_e1_profile(top_k=10, fallback_min_field_coverage=3)
        assert enough_candidates(pool, cfg) is True
        assert enough_field_coverage(pool, cfg) is True


# ===========================================================================
# F1 — active_fields (§9 subset-schema extraction helper)
# ===========================================================================
#
# active_fields(candidates, template_cls, cfg) -> list[str]
#
# Locked rules:
#   - subset_schema_extraction=False → return ALL field names of template_cls
#     in schema order (no-op; default).
#   - When ON: active = union(supported_field_hints across candidates)
#       ∪ identity fields (model_config['graph_id_fields'])
#       ∪ required fields (model_fields[name].is_required())
#   - Only DROP fields with ZERO evidence AND not identity AND not required.
#   - Return in schema order (order of template_cls.model_fields).
#   - model_config may lack 'graph_id_fields' → handle gracefully (no crash).
#   - ZERO hardcoded field names — identity from model_config, required from
#     is_required().
# ===========================================================================


# ---------------------------------------------------------------------------
# Synthetic Pydantic Record for F1 tests (no import of real schemas needed)
# ---------------------------------------------------------------------------

def _make_f1_record_cls():
    """Build a synthetic pydantic BaseModel that mimics a Record class.

    Fields (in schema order):
      identity_field  : str = Field(...)               → identity + required
      required_field  : str = Field(...)               → required only
      optional_a      : str | None = Field(default=None)  → optional
      optional_b      : str | None = Field(default=None)  → optional
      optional_c      : str | None = Field(default=None)  → optional

    graph_id_fields = ['identity_field']
    """
    from pydantic import BaseModel, ConfigDict, Field
    from typing import Optional

    class SyntheticRecord(BaseModel):
        model_config = ConfigDict(
            graph_id_fields=["identity_field"],
        )
        identity_field: str = Field(...)
        required_field: str = Field(...)
        optional_a: Optional[str] = Field(default=None)
        optional_b: Optional[str] = Field(default=None)
        optional_c: Optional[str] = Field(default=None)

    return SyntheticRecord


def _make_f1_record_cls_no_graph_id():
    """Synthetic record class WITHOUT graph_id_fields in model_config."""
    from pydantic import BaseModel, ConfigDict, Field
    from typing import Optional

    class NoGraphIdRecord(BaseModel):
        model_config = ConfigDict()
        required_field: str = Field(...)
        optional_x: Optional[str] = Field(default=None)

    return NoGraphIdRecord


def _make_cfg_subset_on(**kwargs):
    """RetrievalProfile with subset_schema_extraction=True."""
    return _make_profile(subset_schema_extraction=True, **kwargs)


def _make_cfg_subset_off(**kwargs):
    """RetrievalProfile with subset_schema_extraction=False (default)."""
    return _make_profile(subset_schema_extraction=False, **kwargs)


class TestF1Import:
    """[WATCHED-FAIL] Import gate — must fail before implementation."""

    def test_active_fields_importable(self):
        from app.services.extraction_candidate_scoring import active_fields  # noqa: F401


class TestF1SubsetOff:
    """F1.1 — subset_schema_extraction=False → ALL fields, schema order, no-op."""

    def test_off_returns_all_fields(self):
        """With subset off, active_fields returns ALL model_fields keys regardless
        of evidence."""
        from app.services.extraction_candidate_scoring import active_fields

        template_cls = _make_f1_record_cls()
        cfg = _make_cfg_subset_off()

        # Zero candidates, zero evidence — still ALL fields
        result = active_fields([], template_cls, cfg)

        all_fields = list(template_cls.model_fields.keys())
        assert result == all_fields, (
            f"off=no-op must return ALL fields in schema order; "
            f"got {result}, expected {all_fields}"
        )

    def test_off_ignores_evidence_entirely(self):
        """Even with strong evidence for some fields, off still returns ALL."""
        from app.services.extraction_candidate_scoring import active_fields

        template_cls = _make_f1_record_cls()
        cfg = _make_cfg_subset_off()

        # Candidate with evidence only on optional_a
        mc = _make_mc("k", supported_field_hints={"optional_a"})
        result = active_fields([mc], template_cls, cfg)

        all_fields = list(template_cls.model_fields.keys())
        assert result == all_fields


class TestF1SubsetOn:
    """F1.2 — subset_schema_extraction=True — drop zero-evidence optional fields."""

    def test_evidenced_field_is_active(self):
        """A field in at least one candidate's supported_field_hints is active."""
        from app.services.extraction_candidate_scoring import active_fields

        template_cls = _make_f1_record_cls()
        cfg = _make_cfg_subset_on()

        mc = _make_mc("k", supported_field_hints={"optional_a"})
        result = active_fields([mc], template_cls, cfg)

        assert "optional_a" in result

    def test_zero_evidence_optional_field_is_dropped(self):
        """optional_b and optional_c with zero evidence are dropped when subset on."""
        from app.services.extraction_candidate_scoring import active_fields

        template_cls = _make_f1_record_cls()
        cfg = _make_cfg_subset_on()

        # Only optional_a has evidence
        mc = _make_mc("k", supported_field_hints={"optional_a"})
        result = active_fields([mc], template_cls, cfg)

        assert "optional_b" not in result, "optional_b has zero evidence, must be dropped"
        assert "optional_c" not in result, "optional_c has zero evidence, must be dropped"

    def test_zero_evidence_but_identity_field_stays(self):
        """[WATCHED-FAIL] Identity field with zero evidence must STILL be active.

        identity_field is in graph_id_fields — must never be dropped,
        even if no candidate mentions it.
        """
        from app.services.extraction_candidate_scoring import active_fields

        template_cls = _make_f1_record_cls()
        cfg = _make_cfg_subset_on()

        # No candidate mentions identity_field
        mc = _make_mc("k", supported_field_hints={"optional_a"})
        result = active_fields([mc], template_cls, cfg)

        assert "identity_field" in result, (
            "identity_field is in graph_id_fields and must be active even "
            "with zero evidence"
        )

    def test_zero_evidence_but_required_field_stays(self):
        """[WATCHED-FAIL] Required field (is_required()==True) with zero evidence
        must STILL be active, even if not in graph_id_fields."""
        from app.services.extraction_candidate_scoring import active_fields

        template_cls = _make_f1_record_cls()
        cfg = _make_cfg_subset_on()

        # No candidate mentions required_field (only optional_a has evidence)
        mc = _make_mc("k", supported_field_hints={"optional_a"})
        result = active_fields([mc], template_cls, cfg)

        assert "required_field" in result, (
            "required_field is_required()==True and must be active even "
            "with zero evidence"
        )

    def test_schema_order_preserved(self):
        """Result is in schema order (order of template_cls.model_fields), not
        insertion/evidence order."""
        from app.services.extraction_candidate_scoring import active_fields

        template_cls = _make_f1_record_cls()
        cfg = _make_cfg_subset_on()

        # Evidence on optional_c first, then optional_a — schema order must win
        mc = _make_mc(
            "k",
            supported_field_hints={"optional_c", "optional_a"},
        )
        result = active_fields([mc], template_cls, cfg)

        # The result must respect schema order: identity_field, required_field,
        # optional_a, optional_b (dropped, zero evidence), optional_c
        schema_order = list(template_cls.model_fields.keys())
        result_positions = {f: schema_order.index(f) for f in result if f in schema_order}
        sorted_by_schema = sorted(result, key=lambda f: result_positions[f])
        assert result == sorted_by_schema, (
            f"Result not in schema order: got {result}, "
            f"expected {sorted_by_schema}"
        )

    def test_evidence_union_across_candidates(self):
        """supported_field_hints is unioned across ALL candidates.
        A field evidenced by ANY candidate is active."""
        from app.services.extraction_candidate_scoring import active_fields

        template_cls = _make_f1_record_cls()
        cfg = _make_cfg_subset_on()

        mc1 = _make_mc("k1", supported_field_hints={"optional_a"})
        mc2 = _make_mc("k2", supported_field_hints={"optional_b"})
        result = active_fields([mc1, mc2], template_cls, cfg)

        assert "optional_a" in result
        assert "optional_b" in result
        # optional_c has no evidence → dropped
        assert "optional_c" not in result

    def test_no_candidates_zero_evidence_optional_dropped(self):
        """Empty candidate list → only identity + required fields survive."""
        from app.services.extraction_candidate_scoring import active_fields

        template_cls = _make_f1_record_cls()
        cfg = _make_cfg_subset_on()

        result = active_fields([], template_cls, cfg)

        # identity_field (graph_id_fields) → active
        assert "identity_field" in result
        # required_field (is_required) → active
        assert "required_field" in result
        # optional_a, optional_b, optional_c → dropped (zero evidence)
        assert "optional_a" not in result
        assert "optional_b" not in result
        assert "optional_c" not in result


class TestF1NoGraphIdFields:
    """F1.3 — model_config without graph_id_fields must not crash."""

    def test_no_graph_id_fields_does_not_crash(self):
        """If model_config has no 'graph_id_fields', active_fields must not raise."""
        from app.services.extraction_candidate_scoring import active_fields

        template_cls = _make_f1_record_cls_no_graph_id()
        cfg = _make_cfg_subset_on()

        mc = _make_mc("k", supported_field_hints={"optional_x"})
        # Must not raise KeyError or AttributeError
        result = active_fields([mc], template_cls, cfg)

        assert "optional_x" in result
        assert "required_field" in result  # still required even without graph_id_fields

    def test_no_graph_id_fields_off_returns_all(self):
        """Off mode with no graph_id_fields in model_config → still returns all."""
        from app.services.extraction_candidate_scoring import active_fields

        template_cls = _make_f1_record_cls_no_graph_id()
        cfg = _make_cfg_subset_off()

        result = active_fields([], template_cls, cfg)
        assert result == list(template_cls.model_fields.keys())


# ===========================================================================
# SAFETY — section_weight default 0.0 keeps the section term INERT.
#
# Router-scoring section-signal piece. section_hit_counts now wires REAL data,
# so section_norm becomes non-zero in the pool. The HARD requirement is that
# final_score and ordering stay byte-identical to "no section signal" because
# the DEFAULT section_weight is 0.0 (0.0 * section_norm == 0.0).
# ===========================================================================


class TestSectionWeightInertAtDefault:
    def test_default_section_weight_is_zero(self):
        """The model default itself must be 0.0 (the safety pin)."""
        from app.services.ontology_bundles import RetrievalProfile

        assert RetrievalProfile().section_weight == 0.0

    def test_section_term_byte_identical_with_default_weight(self):
        """final_score is byte-identical between a candidate set WITH non-zero
        section_hits and the SAME set with section_hits zeroed, when
        section_weight is the (new) default 0.0.

        This is the 0.0 * section_norm == 0.0 proof: the section term cannot
        move a single score even though section_norm is now non-zero.
        """
        from app.services.extraction_candidate_scoring import score_candidates
        from app.services.ontology_bundles import RetrievalProfile

        # Default profile: section_weight defaults to 0.0 now.
        cfg = RetrievalProfile()
        assert cfg.section_weight == 0.0

        # Candidates carry a spread of OTHER signals so the pool exercises every
        # term, plus a spread of section_hits (non-zero → section_norm > 0).
        def _pair(key, **kw):
            with_section = _make_scored_candidate(key, **kw)
            kw_no_sec = dict(kw)
            kw_no_sec["section_hits"] = 0
            without_section = _make_scored_candidate(key, **kw_no_sec)
            return with_section, without_section

        specs = [
            dict(reranker_score=0.9, alias_hits=2, pattern_hits=1, section_hits=5),
            dict(reranker_score=0.5, alias_hits=0, pattern_hits=3, section_hits=2),
            dict(reranker_score=0.1, alias_hits=4, negative_hits=1, section_hits=1),
            dict(reranker_score=0.3, section_hits=9),
            dict(alias_hits=1, section_hits=4),  # unscorable (no reranker_score)
        ]
        with_pool = []
        without_pool = []
        for i, kw in enumerate(specs):
            w, wo = _pair(f"c{i}", **kw)
            with_pool.append(w)
            without_pool.append(wo)

        res_with = score_candidates(with_pool, cfg)
        res_without = score_candidates(without_pool, cfg)

        keys_with = [mc.candidate_key for mc, _ in res_with]
        keys_without = [mc.candidate_key for mc, _ in res_without]
        scores_with = [s for _, s in res_with]
        scores_without = [s for _, s in res_without]

        # Ordering byte-identical.
        assert keys_with == keys_without
        # Scores byte-identical (exact equality, not approx) — section term is 0.
        assert scores_with == scores_without

    def test_nonzero_section_weight_DOES_move_score(self):
        """Counter-proof: with a non-zero section_weight (calibration), the same
        non-zero section_hits DO change the score — confirming the inertness in
        the test above is owed to the weight being 0.0, not to dead code.
        """
        from app.services.extraction_candidate_scoring import score_candidates
        from app.services.ontology_bundles import RetrievalProfile

        cfg_zero = RetrievalProfile(section_weight=0.0)
        cfg_hot = RetrievalProfile(section_weight=0.5)

        cands = [
            _make_scored_candidate("a", reranker_score=0.5, section_hits=4),
            _make_scored_candidate("b", reranker_score=0.5, section_hits=0),
        ]
        s_zero = {mc.candidate_key: sc for mc, sc in score_candidates(
            [dict(c) for c in cands], cfg_zero)}
        s_hot = {mc.candidate_key: sc for mc, sc in score_candidates(
            [dict(c) for c in cands], cfg_hot)}

        # With weight 0.0 the two candidates tie on the section term.
        assert s_zero["a"] == s_zero["b"]
        # With weight 0.5 the section-bearing candidate scores strictly higher.
        assert s_hot["a"] > s_hot["b"]
