"""Phase-1b RE-WEIGHTED component separator probe — TDD spec.

Phase-1 established that NO single router score (cosine / reranker / final)
separates USED chunks from UNUSED ones (best AUROC ~0.66-0.71 < 0.75 floor).

The hypothesis tested here: the production ``final_score`` is ONE fixed
weighting of components (rerank/lexical/pattern/section/table/negative) tuned
for RANKING.  A DIFFERENT weighting LEARNED from the lineage ``used`` labels
might separate.  This is supervised learning: component scores = features,
``used`` = label, fit the separation-optimal combination.

These tests pin the CORRECTNESS-CRITICAL pieces:

  1. ``component_breakdown`` / ``build_component_table`` — every (pass, chunk)
     row carries ALL the C5 components (cosine, rerank_norm, lexical_norm,
     pattern_norm, section_norm, is_table, negative_norm) AND the recombined
     components must reproduce the production ``final_score`` (lockstep with
     ``score_candidates``).
  2. ``per_component_auroc`` — a synthetic feature that perfectly predicts
     used → AUROC 1.0; a noise feature → ~0.5.
  3. ``fit_reweighted_separators`` (THE probe) — a SEPARABLE synthetic table
     (a known component combination separates) → high CV AUROC + the right
     dominant learned weight; an INSEPARABLE table → CV AUROC ~0.5 so we
     CANNOT be fooled by in-sample overfit.
  4. stable-label intersection (noise floor) — USED ∩ across two runs at the
     element / self_ref level.

Written RED first (no implementation module) per the project's TDD rule.
"""
from __future__ import annotations

import random

import pytest

from scripts.phase1b_reweight_separator import (
    COMPONENT_FIELDS,
    ComponentRow,
    build_component_table,
    build_stable_used_table,
    component_breakdown,
    fit_reweighted_separators,
    per_component_auroc,
)
from scripts.phase1_score_used_separation import ScoreUsedRow


# ===========================================================================
# component_breakdown — lockstep with production score_candidates
# ===========================================================================

class TestComponentBreakdownLockstep:
    """The recombined component breakdown MUST reproduce score_candidates'
    final_score exactly (same min-max + ratio-max normalisation, same weights).
    Otherwise the re-weighting probe is fitting on the wrong features."""

    def _profile(self):
        from app.services.ontology_bundles import RetrievalProfile
        return RetrievalProfile()

    def test_breakdown_recombines_to_production_final_score(self):
        from app.services.extraction_candidate_scoring import (
            MergedCandidate,
            score_candidates,
        )

        cfg = self._profile()

        # A pool with varied reranker + lexical + pattern + negative + table
        # signal so every component norm exercises a non-trivial value.
        def _mc(key, ci, alias, pat, neg, table):
            return MergedCandidate(
                candidate_key=key, chunk_index=ci, self_ref=f"#/c/{ci}",
                chunk_text=f"text {ci}", source_refs=[f"#/texts/{ci}"],
                token_count=10, page_number=1, vector_score=None,
                field_scores={}, alias_hits=alias, pattern_hits=pat,
                negative_hits=neg, section_hits=0,
                content_type=("table" if table else None),
                retrieval_sources={"dense"}, supported_field_hints=set(),
            )

        specs = [
            # (key, ci, rerank_raw, alias, pat, neg, table)
            ("a", 0, 5.0, 3, 2, 0, False),
            ("b", 1, 1.0, 0, 1, 1, True),
            ("c", 2, 3.0, 1, 0, 2, False),
            ("d", 3, 0.0, 2, 3, 0, True),
        ]
        c5_input = []
        for key, ci, rr, alias, pat, neg, table in specs:
            entry = {
                "merged_candidate": _mc(key, ci, alias, pat, neg, table),
                "content_text": f"text {ci}",
                "reranker_score": rr,
            }
            c5_input.append(entry)

        # Production final scores, keyed by chunk_index.
        prod = {mc.chunk_index: fs for mc, fs in score_candidates(c5_input, cfg)}

        # Our component breakdown, recombined with the SAME weights.
        breakdown = component_breakdown(c5_input, cfg)  # {ci: {component: value}}
        for ci, comps in breakdown.items():
            recombined = (
                cfg.rerank_weight * comps["rerank_norm"]
                + cfg.lexical_weight * comps["lexical_norm"]
                + cfg.pattern_weight * comps["pattern_norm"]
                + cfg.section_weight * comps["section_norm"]
                + cfg.table_boost * comps["is_table"]
                - cfg.negative_weight * comps["negative_norm"]
            )
            recombined = max(recombined, 0.0)
            assert recombined == pytest.approx(prod[ci], abs=1e-9), (
                f"chunk {ci}: recombined {recombined} != production {prod[ci]}"
            )

    def test_breakdown_carries_every_component_field(self):
        from app.services.extraction_candidate_scoring import MergedCandidate

        cfg = self._profile()
        c5_input = [{
            "merged_candidate": MergedCandidate(
                candidate_key="a", chunk_index=0, self_ref="#/c/0",
                chunk_text="t", source_refs=["#/texts/0"], token_count=1,
                page_number=1, vector_score=None, field_scores={},
                alias_hits=1, pattern_hits=1, negative_hits=0, section_hits=0,
                content_type=None, retrieval_sources={"dense"},
                supported_field_hints=set(),
            ),
            "content_text": "t",
            "reranker_score": 2.0,
        }]
        comps = component_breakdown(c5_input, cfg)[0]
        for f in ("rerank_norm", "lexical_norm", "pattern_norm",
                  "section_norm", "is_table", "negative_norm"):
            assert f in comps


# ===========================================================================
# build_component_table — shape / row-count (DB + embed/rerank mocked)
# ===========================================================================

class TestBuildComponentTable:
    def test_every_chunk_row_carries_all_component_columns(self, monkeypatch):
        """3 chunks x 2 passes = 6 rows, each carrying cosine + every C5
        component column. Embedding / rerank / signals are stubbed."""
        import scripts.phase1b_reweight_separator as mod

        chunks = [
            {"self_ref": "chunk_0", "chunk_index": 0, "chunk_text": "alpha",
             "source_refs": ["#/texts/1"], "token_count": 3,
             "embedding": [1.0, 0.0, 0.0], "page_number": 1},
            {"self_ref": "chunk_1", "chunk_index": 1, "chunk_text": "beta",
             "source_refs": ["#/texts/2"], "token_count": 4,
             "embedding": [0.0, 1.0, 0.0], "page_number": 1},
            {"self_ref": "chunk_2", "chunk_index": 2, "chunk_text": "gamma",
             "source_refs": ["#/tables/0"], "token_count": 5,
             "embedding": [0.0, 0.0, 1.0], "page_number": 2},
        ]

        def fake_signals(pass_name):
            class _FQ:
                field_queries = ()
                entity_query = f"q::{pass_name}"
            return _FQ()
        # _pass_signals lives in the Phase-1 module; build_component_table
        # reuses it via the phase1 import.
        import scripts.phase1_score_used_separation as p1
        monkeypatch.setattr(p1, "_pass_signals", fake_signals)
        monkeypatch.setattr(p1, "_embed_query", lambda text: [1.0, 0.0, 0.0])

        def fake_rerank(query, cands):
            return {str(c["chunk_index"]): float(c["chunk_index"]) for c in cands}
        monkeypatch.setattr(p1, "_rerank_scores", fake_rerank)

        table = build_component_table(
            chunks_by_pass={"radar_power_rf": chunks, "missile_kinematics": chunks},
        )
        assert len(table) == 3 * 2
        for r in table:
            assert isinstance(r, ComponentRow)
            for f in COMPONENT_FIELDS:
                assert isinstance(getattr(r, f), float), f"{f} not float"
        # chunk 2 is sourced from #/tables/0 → is_table == 1.0
        c2 = [r for r in table
              if r.chunk_index == 2 and r.pass_name == "radar_power_rf"][0]
        assert c2.is_table == pytest.approx(1.0)
        # chunk 0 cosine under aligned query [1,0,0] == 1.0
        c0 = [r for r in table
              if r.chunk_index == 0 and r.pass_name == "radar_power_rf"][0]
        assert c0.cosine == pytest.approx(1.0)


# ===========================================================================
# per_component_auroc — univariate signal detection
# ===========================================================================

def _comp_rows(pass_name, feats_used):
    """Build ComponentRows. `feats_used` = [(component_dict, used_bool), ...]."""
    out = []
    for i, (feats, used) in enumerate(feats_used):
        kw = {f: 0.0 for f in COMPONENT_FIELDS}
        kw.update(feats)
        out.append(ComponentRow(
            pass_name=pass_name, chunk_index=i, used=used, **kw,
        ))
    return out


class TestPerComponentAuroc:
    def test_perfect_predictor_component_auroc_1(self):
        """A component that is high iff used → AUROC 1.0 for that component."""
        rows = _comp_rows("missile_kinematics", [
            ({"rerank_norm": 0.9}, True),
            ({"rerank_norm": 0.8}, True),
            ({"rerank_norm": 0.2}, False),
            ({"rerank_norm": 0.1}, False),
        ])
        res = per_component_auroc(rows)
        # pooled AUROC for the perfectly-predictive component is 1.0
        assert res["pooled"]["rerank_norm"] == pytest.approx(1.0)

    def test_noise_component_auroc_about_half(self):
        """A component uncorrelated with used → AUROC ~0.5."""
        rng = random.Random(3)
        rows = []
        for i in range(200):
            used = rng.random() < 0.3
            kw = {f: 0.0 for f in COMPONENT_FIELDS}
            kw["lexical_norm"] = rng.random()  # pure noise
            rows.append(ComponentRow(
                pass_name="radar_identity", chunk_index=i, used=used, **kw))
        res = per_component_auroc(rows)
        assert res["pooled"]["lexical_norm"] == pytest.approx(0.5, abs=0.12)

    def test_reports_per_pass_and_pooled(self):
        rows = (
            _comp_rows("p1", [({"rerank_norm": 0.9}, True),
                              ({"rerank_norm": 0.1}, False)])
            + _comp_rows("p2", [({"rerank_norm": 0.9}, True),
                                ({"rerank_norm": 0.1}, False)])
        )
        res = per_component_auroc(rows)
        assert "pooled" in res and "per_pass" in res
        assert "p1" in res["per_pass"] and "p2" in res["per_pass"]


# ===========================================================================
# fit_reweighted_separators — THE probe (separable vs inseparable; CV honesty)
# ===========================================================================

class TestReweightedSeparatorSeparable:
    """A SEPARABLE table where a KNOWN component combination separates →
    high CV AUROC (not just in-sample) AND the right dominant weight."""

    def test_known_combination_separates_high_cv_and_right_weight(self):
        # Construct a table where `used` is a clean function of rerank_norm:
        # used chunks have rerank_norm drawn HIGH, unused LOW, with a margin;
        # all other components are noise. A correct re-weighting must put the
        # dominant weight on rerank_norm and achieve high CV AUROC.
        rng = random.Random(17)
        rows = []
        ci = 0
        for _pass in ("missile_kinematics", "radar_power_rf"):
            for _ in range(20):  # 20 used / pass
                kw = {f: rng.uniform(0.0, 0.3) for f in COMPONENT_FIELDS}
                kw["rerank_norm"] = rng.uniform(0.70, 1.0)
                rows.append(ComponentRow(_pass, ci, True, **kw)); ci += 1
            for _ in range(40):  # 40 unused / pass
                kw = {f: rng.uniform(0.0, 0.3) for f in COMPONENT_FIELDS}
                kw["rerank_norm"] = rng.uniform(0.0, 0.30)
                rows.append(ComponentRow(_pass, ci, False, **kw)); ci += 1

        fit = fit_reweighted_separators(rows, n_splits=5, seed=0)

        # The honest signal estimate (CV) must be high for a truly separable table.
        assert fit["logistic"]["cv_auroc"] > 0.90
        assert fit["gbm"]["cv_auroc"] > 0.85
        # In-sample is an even-higher ceiling.
        assert fit["logistic"]["in_sample_auroc"] >= fit["logistic"]["cv_auroc"] - 1e-9

        # Learned logistic weights: rerank_norm must dominate (largest |weight|).
        weights = fit["logistic"]["weights"]  # {component: coef}
        dominant = max(weights, key=lambda k: abs(weights[k]))
        assert dominant == "rerank_norm"


class TestReweightedSeparatorInseparable:
    """CRITICAL HONESTY: an INSEPARABLE table must yield CV AUROC ~0.5 even
    though in-sample (with ~7 features over ~120 rows) can OVERFIT high. This
    is the test that prevents us being fooled by in-sample numbers."""

    def test_inseparable_cv_about_half_even_if_in_sample_overfits(self):
        # used label is INDEPENDENT of every feature → no signal exists.
        rng = random.Random(101)
        rows = []
        ci = 0
        for _pass in ("missile_kinematics", "radar_power_rf"):
            for _ in range(60):
                kw = {f: rng.random() for f in COMPONENT_FIELDS}
                used = rng.random() < 0.25  # label independent of features
                rows.append(ComponentRow(_pass, ci, used, **kw)); ci += 1

        fit = fit_reweighted_separators(rows, n_splits=5, seed=0)

        # The honest CV estimate must sit near chance for BOTH models.
        assert fit["logistic"]["cv_auroc"] == pytest.approx(0.5, abs=0.18)
        assert fit["gbm"]["cv_auroc"] == pytest.approx(0.5, abs=0.20)

        # The OVERFITTING TRAP guard: the GBM (high capacity) in-sample AUROC
        # should be much higher than its CV AUROC — proving in-sample alone is
        # untrustworthy on this small/wide sample.
        assert fit["gbm"]["in_sample_auroc"] > fit["gbm"]["cv_auroc"] + 0.15

    def test_reports_both_in_sample_and_cv_for_both_models(self):
        rng = random.Random(5)
        rows = [ComponentRow("p", i, rng.random() < 0.3,
                             **{f: rng.random() for f in COMPONENT_FIELDS})
                for i in range(80)]
        fit = fit_reweighted_separators(rows, n_splits=4, seed=0)
        for model in ("logistic", "gbm"):
            assert "in_sample_auroc" in fit[model]
            assert "cv_auroc" in fit[model]
        assert "weights" in fit["logistic"]
        # the best re-weighted per-row score is returned for the recall sweep
        assert "logistic_scores" in fit and len(fit["logistic_scores"]) == len(rows)


# ===========================================================================
# build_stable_used_table — noise floor (used ∩ across two runs, element-level)
# ===========================================================================

def _su(pass_name, ci, used, self_ref):
    return ScoreUsedRow(
        pass_name=pass_name, chunk_index=ci,
        cosine_score=0.0, reranker_score=0.0, final_score=0.0,
        used=used, used_field_level=False, used_source="direct",
        self_ref=self_ref,
    )


class TestStableUsedTable:
    def test_intersects_used_at_self_ref_level(self):
        """A chunk is STABLE-used iff it is used in BOTH runs, matched by
        self_ref (the stable element anchor), not raw chunk_index."""
        # Run A: chunks 0,1,2 ; used = {0,1}. self_refs s0,s1,s2.
        run_a = [
            _su("missile_kinematics", 0, True, "s0"),
            _su("missile_kinematics", 1, True, "s1"),
            _su("missile_kinematics", 2, False, "s2"),
        ]
        # Run B: SAME self_refs but (hypothetically) re-indexed; used = {s1,s2}.
        run_b = [
            _su("missile_kinematics", 0, False, "s0"),
            _su("missile_kinematics", 1, True, "s1"),
            _su("missile_kinematics", 2, True, "s2"),
        ]
        # current-run index we want labels ON is run A.
        stable = build_stable_used_table(run_a, run_b)
        used = {(r.pass_name, r.chunk_index): r.used for r in stable}
        # only s1 is used in BOTH → stable-used. s0 (A-only) and s2 (B-only) drop.
        assert used[("missile_kinematics", 1)] is True
        assert used[("missile_kinematics", 0)] is False
        assert used[("missile_kinematics", 2)] is False
        # row count / ordering follows the current run (A): 3 rows.
        assert len(stable) == 3

    def test_maps_via_self_ref_when_indices_differ(self):
        """If run B uses a DIFFERENT chunk_index for the same self_ref, the
        intersection still matches on self_ref (stable anchor)."""
        run_a = [_su("p", 5, True, "elemX")]
        run_b = [_su("p", 99, True, "elemX")]  # same element, different index
        stable = build_stable_used_table(run_a, run_b)
        assert stable[0].used is True
        assert stable[0].chunk_index == 5  # current-run (A) index preserved
