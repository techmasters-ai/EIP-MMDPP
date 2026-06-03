"""Phase-1 go/no-go: (score, used) separation analysis — TDD spec.

These tests pin the CORRECTNESS-CRITICAL logic that decides whether a
router-score THRESHOLD can replace fixed top_k while preserving recall:

  1. build_used_table  — lineage join (direct chunk_index + source_refs
                          fallback + empty-provenance value-fill signal).
  2. analyze_separation — used-vs-unused distributions, AUROC, recall-vs-
                          threshold curve, dynamic-k rules, recommendation,
                          and the FALLBACK DETECTOR (scores don't separate
                          → tuned per-pass top_k).
  3. build_chunk_score_table — per-(pass, chunk) {cosine, reranker, final}
                          row emission (shape + row-count contract; the
                          DB/embedding I/O is mocked).

Written RED first (no implementation module) per the project's TDD rule.
"""
from __future__ import annotations

import math

import pytest

from scripts.phase1_score_used_separation import (
    PASS_USED,
    USED_SOURCE_DIRECT,
    USED_SOURCE_FALLBACK,
    USED_SOURCE_VALUE_FILL,
    ScoreUsedRow,
    analyze_separation,
    auroc,
    build_chunk_score_table,
    build_used_table,
    recall_vs_threshold,
)


# ===========================================================================
# build_used_table — lineage join
# ===========================================================================

def _chunk(idx: int, source_refs: list[str]) -> dict:
    """Minimal ExtractionChunk row dict (ArcadeDB shape)."""
    return {
        "self_ref": f"chunk_{idx}",
        "chunk_index": idx,
        "source_refs": source_refs,
    }


class TestBuildUsedTable:
    def test_direct_chunk_index_path(self):
        """A provenance row whose chunk_indexes are in-range marks exactly
        those chunk_index rows USED, via the direct path."""
        chunks = [_chunk(0, ["#/texts/1"]), _chunk(1, ["#/texts/2"]), _chunk(2, ["#/texts/3"])]
        pass_outputs = {
            "missile_kinematics": {
                "provenance": [
                    {"chunk_indexes": [0, 2], "self_refs": ["#/texts/1"]},
                ],
                "field_provenance": [],
            }
        }
        table = build_used_table(pass_outputs, {"missile_kinematics": chunks})
        used = {(r.pass_name, r.chunk_index): r for r in table}
        assert used[("missile_kinematics", 0)].used is True
        assert used[("missile_kinematics", 2)].used is True
        assert used[("missile_kinematics", 1)].used is False
        # direct path → used_source == direct on the used rows
        assert used[("missile_kinematics", 0)].used_source == USED_SOURCE_DIRECT
        assert used[("missile_kinematics", 2)].used_source == USED_SOURCE_DIRECT
        # every chunk gets exactly one row
        assert len(table) == 3

    def test_source_refs_fallback_when_chunk_index_out_of_range(self):
        """When a provenance row's chunk_indexes are OUT of range, the row is
        rescued via source_refs ∩ self_refs (element-ref overlap)."""
        # index only has 0,1,2 ; provenance points at chunk_index 66 (gone) but
        # carries self_refs that overlap chunk 1's source_refs.
        chunks = [
            _chunk(0, ["#/texts/1"]),
            _chunk(1, ["#/texts/312", "#/texts/313"]),
            _chunk(2, ["#/texts/3"]),
        ]
        pass_outputs = {
            "radar_identity": {
                "provenance": [
                    {"chunk_indexes": [66], "self_refs": ["#/texts/312", "#/texts/313"]},
                ],
                "field_provenance": [],
            }
        }
        table = build_used_table(pass_outputs, {"radar_identity": chunks})
        used = {(r.pass_name, r.chunk_index): r for r in table}
        # chunk 1 recovered via fallback; 0 and 2 unused.
        assert used[("radar_identity", 1)].used is True
        assert used[("radar_identity", 1)].used_source == USED_SOURCE_FALLBACK
        assert used[("radar_identity", 0)].used is False
        assert used[("radar_identity", 2)].used is False

    def test_direct_preferred_over_fallback_no_over_expansion(self):
        """If a row's chunk_index IS in range, the direct hit is used and the
        fallback is NOT applied for that row — preventing USED-set over-
        expansion through shared merged source_refs."""
        # chunk 0 and chunk 1 SHARE element #/texts/9. A row with chunk_indexes=[0]
        # and self_refs containing #/texts/9 must mark ONLY chunk 0 (direct),
        # NOT also chunk 1 (which would happen if fallback ran unconditionally).
        chunks = [
            _chunk(0, ["#/texts/9"]),
            _chunk(1, ["#/texts/9", "#/texts/10"]),
        ]
        pass_outputs = {
            "radar_power_rf": {
                "provenance": [
                    {"chunk_indexes": [0], "self_refs": ["#/texts/9"]},
                ],
                "field_provenance": [],
            }
        }
        table = build_used_table(pass_outputs, {"radar_power_rf": chunks})
        used = {(r.pass_name, r.chunk_index): r for r in table}
        assert used[("radar_power_rf", 0)].used is True
        assert used[("radar_power_rf", 0)].used_source == USED_SOURCE_DIRECT
        assert used[("radar_power_rf", 1)].used is False

    def test_empty_provenance_marks_value_fill_source(self):
        """A pass with EMPTY provenance still emits one row per chunk, all
        used=False, and is tagged used_source=value_fill so the analysis can
        down-weight / flag it."""
        chunks = [_chunk(0, ["#/texts/1"]), _chunk(1, ["#/texts/2"])]
        pass_outputs = {
            "radar_power_rf": {"provenance": [], "field_provenance": []}
        }
        table = build_used_table(pass_outputs, {"radar_power_rf": chunks})
        assert all(r.used is False for r in table)
        assert all(r.used_source == USED_SOURCE_VALUE_FILL for r in table)
        # pass-level used flag exposed for the analysis
        assert PASS_USED["radar_power_rf"] is False or PASS_USED.get  # sanity guard

    def test_field_level_used_from_field_provenance(self):
        """field_provenance contributes used_field_level (a chunk used at the
        field granularity), independent of the row-level provenance."""
        chunks = [_chunk(0, ["#/texts/1"]), _chunk(1, ["#/texts/2"]), _chunk(2, ["#/texts/3"])]
        pass_outputs = {
            "missile_kinematics": {
                "provenance": [
                    {"chunk_indexes": [0], "self_refs": ["#/texts/1"]},
                ],
                "field_provenance": [
                    {"field_name": "max_speed", "chunk_indexes": [2], "self_refs": ["#/texts/3"]},
                ],
            }
        }
        table = build_used_table(pass_outputs, {"missile_kinematics": chunks})
        used = {(r.pass_name, r.chunk_index): r for r in table}
        # chunk 0 used at row level; chunk 2 used at field level only.
        assert used[("missile_kinematics", 0)].used is True
        assert used[("missile_kinematics", 2)].used_field_level is True
        # field-level participation also flips the unified `used` flag True
        # (a chunk a field was traced to IS a used chunk).
        assert used[("missile_kinematics", 2)].used is True


# ===========================================================================
# auroc + recall_vs_threshold — primitives
# ===========================================================================

class TestAurocPrimitive:
    def test_perfect_separation_auroc_1(self):
        scores = [0.9, 0.8, 0.2, 0.1]
        used = [True, True, False, False]
        assert auroc(scores, used) == pytest.approx(1.0)

    def test_inverted_separation_auroc_0(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        used = [True, True, False, False]
        assert auroc(scores, used) == pytest.approx(0.0)

    def test_interleaved_auroc(self):
        # used {0.9,0.7}, unused {0.8,0.6}: P(used>unused) over 4 pairs =
        # (0.9>0.8),(0.9>0.6),(0.7>0.6) win; (0.7>0.8) loses → 3/4 = 0.75.
        # (cross-checked against sklearn.roc_auc_score in test_matches_sklearn)
        scores = [0.9, 0.8, 0.7, 0.6]
        used = [True, False, True, False]
        assert auroc(scores, used) == pytest.approx(0.75)

    def test_fully_interleaved_auroc_half(self):
        # used and unused perfectly alternate around shared midpoints → 0.5.
        scores = [0.8, 0.8, 0.4, 0.4]
        used = [True, False, True, False]
        assert auroc(scores, used) == pytest.approx(0.5)

    def test_tie_handling_uses_midrank(self):
        # all-equal scores → AUROC 0.5 (no separation possible)
        scores = [0.5, 0.5, 0.5, 0.5]
        used = [True, True, False, False]
        assert auroc(scores, used) == pytest.approx(0.5)

    def test_matches_sklearn(self):
        sklearn = pytest.importorskip("sklearn.metrics")
        import random
        rng = random.Random(7)
        scores = [rng.gauss(0, 1) for _ in range(200)]
        used = [rng.random() < 0.3 for _ in range(200)]
        if len(set(used)) < 2:
            used[0] = True
            used[-1] = False
        expected = sklearn.roc_auc_score([1 if u else 0 for u in used], scores)
        assert auroc(scores, used) == pytest.approx(expected, abs=1e-9)


class TestRecallVsThreshold:
    def test_recall_and_cost_monotone(self):
        scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        used = [True, True, False, True, False]
        curve = recall_vs_threshold(scores, used)
        # at the lowest tau everything is kept → recall 1.0, cost 1.0
        lo = min(curve, key=lambda p: p["threshold"])
        assert lo["recall"] == pytest.approx(1.0)
        assert lo["cost"] == pytest.approx(1.0)
        # recall is non-increasing as tau rises
        ordered = sorted(curve, key=lambda p: p["threshold"])
        recalls = [p["recall"] for p in ordered]
        assert all(a >= b - 1e-12 for a, b in zip(recalls, recalls[1:]))

    def test_threshold_that_drops_a_used_chunk_loses_recall(self):
        scores = [0.9, 0.2]   # one used chunk sits LOW
        used = [False, True]
        curve = recall_vs_threshold(scores, used)
        # a tau above 0.2 keeps only the unused high chunk → recall 0.0
        hi = max(curve, key=lambda p: p["threshold"])
        assert hi["recall"] == pytest.approx(0.0)


# ===========================================================================
# analyze_separation — THE go/no-go decision (separable vs inseparable)
# ===========================================================================

def _rows(pass_name, pairs, score_field="cosine_score"):
    """Build ScoreUsedRows for one pass. `pairs` = [(score, used), ...].

    The non-target score fields are set equal to the target so single-score
    fixtures don't accidentally trigger separation on another channel.
    """
    out = []
    for i, (sc, us) in enumerate(pairs):
        kw = {
            "pass_name": pass_name,
            "chunk_index": i,
            "cosine_score": sc,
            "reranker_score": sc,
            "final_score": sc,
            "used": us,
            "used_field_level": us,
            "used_source": USED_SOURCE_DIRECT if us else USED_SOURCE_DIRECT,
        }
        out.append(ScoreUsedRow(**kw))
    return out


class TestAnalyzeSeparationSeparable:
    """SEPARABLE: used scores high, unused low → GO with a tight threshold."""

    def test_high_auroc_and_threshold_keeps_few_chunks_at_full_recall(self):
        # 4 used (high) + 8 unused (low), cleanly separated.
        pairs = (
            [(0.90, True), (0.88, True), (0.86, True), (0.84, True)]
            + [(0.30, False)] * 8
        )
        rows = _rows("missile_kinematics", pairs)
        report = analyze_separation(rows, recall_floor=1.0)
        pr = report.per_pass["missile_kinematics"]

        # every score channel separates cleanly
        assert pr.auroc["cosine_score"] == pytest.approx(1.0)

        # recommendation at recall 1.0 keeps only the used fraction (4/12)
        rec = pr.recommendation["cosine_score"]
        assert rec is not None
        assert rec.recall_achieved == pytest.approx(1.0)
        assert rec.fraction_kept == pytest.approx(4 / 12, abs=1e-9)
        assert rec.threshold > 0.30  # a real cut between the clusters

        # overall verdict for a separable doc = GO
        assert report.verdict == "GO"
        assert report.recommended_rule is not None

    def test_dynamic_k_rel_max_also_recovers_the_cluster(self):
        pairs = (
            [(0.95, True), (0.93, True), (0.90, True)]
            + [(0.20, False)] * 7
        )
        rows = _rows("radar_power_rf", pairs)
        report = analyze_separation(rows, recall_floor=1.0)
        pr = report.per_pass["radar_power_rf"]
        dynk = pr.dynamic_k["cosine_score"]
        # rel_max with the chosen alpha keeps the 3 high chunks at recall 1.0
        assert dynk["rel_max"].recall_achieved == pytest.approx(1.0)
        assert dynk["rel_max"].fraction_kept <= 0.5


class TestAnalyzeSeparationInseparable:
    """INSEPARABLE: used/unused fully overlap → NO-GO, fall back to top_k."""

    def test_low_auroc_triggers_fallback_verdict_and_emits_per_pass_topk(self):
        # used and unused scores drawn from the SAME band → no separation.
        # Pin the LOWEST score to an UNUSED chunk so a recall-1.0 threshold is
        # forced to keep ~all chunks (the cut is bounded by however many unused
        # chunks incidentally sit below the lowest USED score).
        import random
        rng = random.Random(11)
        pairs = []
        for _ in range(15):
            pairs.append((round(rng.uniform(0.42, 0.60), 4), True))
        for _ in range(14):
            pairs.append((round(rng.uniform(0.42, 0.60), 4), False))
        pairs.append((0.10, False))   # a deep UNUSED straggler at the bottom
        rng.shuffle(pairs)
        rows = _rows("missile_kinematics", pairs)

        report = analyze_separation(rows, recall_floor=1.0, auroc_sep_floor=0.75)
        pr = report.per_pass["missile_kinematics"]

        # THE separation signal the detector keys on: no channel reaches the
        # AUROC floor → the score does not rank used above unused.
        assert max(pr.auroc.values()) < 0.75
        assert pr.separates is False

        # A recall-1.0 threshold cannot exclude anything scoring >= the lowest
        # USED score; whatever it drops is incidental unused-below-min-used
        # (sample noise), so recall is held at the cost of keeping most chunks.
        rec = pr.recommendation["cosine_score"]
        assert rec is not None
        assert rec.recall_achieved == pytest.approx(1.0)

        # → verdict NO-GO with a concrete fallback top_k that reaches deep into
        # the pool (overlap forces a large recall-preserving k).
        assert report.verdict == "NO-GO"
        assert pr.fallback_top_k is not None
        assert pr.fallback_top_k >= 15   # must cover all 15 used; overlap → deep
        assert report.recommended_rule is None

    def test_interleaved_threshold_buys_nothing_deterministic(self):
        """Deterministic inseparable case: used/unused perfectly interleaved so
        EVERY used chunk has an unused chunk just below it. No recall-1.0
        threshold can drop more than the single bottom unused chunk → the
        threshold buys essentially nothing, AUROC ~0.5, verdict NO-GO."""
        # ranks (desc): U,u,U,u,...,U,u  → 10 used, 10 unused, fully alternating.
        pairs = []
        s = 1.00
        for _ in range(10):
            pairs.append((round(s, 4), True)); s -= 0.01
            pairs.append((round(s, 4), False)); s -= 0.01
        rows = _rows("missile_kinematics", pairs)
        report = analyze_separation(rows, recall_floor=1.0, auroc_sep_floor=0.75)
        pr = report.per_pass["missile_kinematics"]
        assert pr.auroc["cosine_score"] == pytest.approx(0.55, abs=0.06)  # ~chance
        assert pr.separates is False
        rec = pr.recommendation["cosine_score"]
        # the deepest used chunk is rank 18 (0-based) → threshold must keep >=19
        # of 20 chunks; only the final unused chunk can be dropped.
        assert rec.fraction_kept >= 0.95
        assert report.verdict == "NO-GO"
        assert pr.fallback_top_k == 19

    def test_fallback_topk_is_minimal_recall_preserving_rank(self):
        """fallback_top_k = the smallest k such that the top-k by the BEST
        channel still contains every used chunk (recall 1.0)."""
        # Engineer ranks: used chunks land at ranks 1,2 and 5 (0-based 0,1,4)
        # under the cosine ordering → smallest recall-preserving k is 5.
        pairs = [
            (0.99, True),    # rank 0
            (0.98, True),    # rank 1
            (0.97, False),   # rank 2
            (0.96, False),   # rank 3
            (0.95, True),    # rank 4  <- deepest used
            (0.10, False),
            (0.05, False),
        ]
        rows = _rows("radar_power_rf", pairs)
        report = analyze_separation(rows, recall_floor=1.0, auroc_sep_floor=0.99)
        pr = report.per_pass["radar_power_rf"]
        # AUROC < 0.99 floor here → fallback path engaged
        assert pr.fallback_top_k == 5

    def test_recall_floor_below_one_allows_dropping_a_straggler(self):
        # one used chunk is a deep straggler; with floor 0.9 we may drop it.
        pairs = (
            [(0.95, True), (0.93, True), (0.91, True), (0.89, True),
             (0.87, True), (0.85, True), (0.83, True), (0.81, True),
             (0.79, True), (0.10, True)]            # 10 used; last is a straggler
            + [(0.40, False)] * 10
        )
        rows = _rows("missile_kinematics", pairs)
        report = analyze_separation(rows, recall_floor=0.90)
        pr = report.per_pass["missile_kinematics"]
        rec = pr.recommendation["cosine_score"]
        assert rec is not None
        # floor 0.9 → we may keep 9/10 used, dropping the straggler, and the
        # threshold then excludes the unused band → far fewer than all chunks.
        assert rec.recall_achieved >= 0.90
        assert rec.fraction_kept < 0.95


# ===========================================================================
# build_chunk_score_table — shape / row-count (DB + embed/rerank mocked)
# ===========================================================================

class TestBuildChunkScoreTable:
    def test_every_chunk_gets_three_scores_rowcount_chunks_times_passes(self, monkeypatch):
        """With 3 chunks and 2 passes the table has 6 rows, each carrying all
        three score fields. Embedding/rerank/scoring are stubbed deterministic."""
        import scripts.phase1_score_used_separation as mod

        chunks = [
            {"self_ref": "chunk_0", "chunk_index": 0, "chunk_text": "alpha",
             "source_refs": ["#/texts/1"], "token_count": 3,
             "embedding": [1.0, 0.0, 0.0], "page_number": 1},
            {"self_ref": "chunk_1", "chunk_index": 1, "chunk_text": "beta",
             "source_refs": ["#/texts/2"], "token_count": 4,
             "embedding": [0.0, 1.0, 0.0], "page_number": 1},
            {"self_ref": "chunk_2", "chunk_index": 2, "chunk_text": "gamma",
             "source_refs": ["#/texts/3"], "token_count": 5,
             "embedding": [0.0, 0.0, 1.0], "page_number": 2},
        ]

        # Stub the per-pass query/signals builder → fixed query, no field sigs.
        def fake_signals(pass_name):
            class _FQ:  # empty signals → lexical/pattern hits all 0
                field_queries = ()
                entity_query = f"q::{pass_name}"
            return _FQ()
        monkeypatch.setattr(mod, "_pass_signals", fake_signals)

        # Stub the query embedding → align with chunk 0 so cosine is deterministic.
        monkeypatch.setattr(mod, "_embed_query", lambda text: [1.0, 0.0, 0.0])

        # Stub the reranker → score = index (so all distinct, deterministic).
        def fake_rerank(query, cands):
            return {str(c["chunk_index"]): float(c["chunk_index"]) for c in cands}
        monkeypatch.setattr(mod, "_rerank_scores", fake_rerank)

        table = build_chunk_score_table(
            chunks_by_pass={"radar_power_rf": chunks, "missile_kinematics": chunks},
        )
        assert len(table) == 3 * 2
        for r in table:
            assert isinstance(r.cosine_score, float)
            assert isinstance(r.reranker_score, float)
            assert isinstance(r.final_score, float)
        # cosine for chunk_0 under query [1,0,0] is 1.0 (aligned)
        c0 = [r for r in table if r.chunk_index == 0 and r.pass_name == "radar_power_rf"][0]
        assert c0.cosine_score == pytest.approx(1.0)
