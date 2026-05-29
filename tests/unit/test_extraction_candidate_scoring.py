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
