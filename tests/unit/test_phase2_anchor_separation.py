"""Phase-2 C8 identity-anchor separation probe — TDD spec.

Phase-1 / 1b established that NO router score and NO learned re-weighting of the
C5 components separates USED chunks from UNUSED on SA-2 (best CV AUROC ~0.60),
because the lexical channel (``alias_hits``) barely fires: the lexical vocab is
spec-parameter labels, so entity NAMES ("SNR-75", "Fan Song") never enter the
matcher.

Commit c66094b wired the C8 identity-anchor channel: the worker harvests the
committed identity-entity NAMES from the persisted identity passes and hands
them to the router, which adds an anchor lexical sub-channel
(``extraction_chunk_search.py``: ``anchor_hit_count = sum(1 for a in
normalised_anchors if a in haystack)``; that count is folded into
``mc.alias_hits``).

This module asks, OFFLINE on the existing SA-2 data: if those anchor names are
fed into the SAME normalised-substring matcher, does the resulting
``anchor_alias_hits`` carry signal that SEPARATES USED from UNUSED per pass —
and does TYPE-MATCHING the anchors to each pass's target entity type
discriminate even when "contains ANY committed entity" (near-binary) does not?

These tests pin the CORRECTNESS-CRITICAL pure pieces (RED first — no
implementation module yet):

  1. ``anchor_alias_hits`` — counts anchor names whose NFC+casefold form is a
     substring of the NFC+casefold chunk text, using the SAME normalisation the
     production matcher uses (reuses ``extraction_lexical_search._nfc``). The
     brief's canonical case: "SNR-75 Fan Song" with anchors ["SNR-75","Fan
     Song","V-75"] → 2; absent anchor → 0.
  2. ``pass_typed_anchor_names`` — variant (b): for a pass, keep only the
     anchors whose ``entity_type`` is in the pass's ``primary_entity_types``
     (driven off bundle metadata — NO hardcoded type names).
  3. ``build_anchor_feature_table`` — per (pass, chunk) AnchorRow carrying both
     ``anchor_alias_hits`` (all anchors) and ``anchor_alias_hits_typed``
     (type-matched), joined onto the chunk index.
  4. ``attach_anchor_features`` + ``feature_matrix_with_anchors`` — the two
     anchor features extend the 7 baseline C5 component features; the WITHOUT
     matrix is a strict column-subset of the WITH matrix (apples-to-apples
     delta).
  5. ``fit_with_and_without_anchors`` — on a SEPARABLE synthetic table where the
     anchor feature is the ONLY signal, the WITH-anchors CV AUROC clears the
     floor and beats WITHOUT; on an INSEPARABLE table (anchors fire but are
     label-independent) the WITH CV AUROC stays ~0.5 so in-sample overfit
     cannot fool us.
"""
from __future__ import annotations

import random

import pytest

from scripts.phase2_anchor_alias_separation import (
    ANCHOR_FEATURE_FIELDS,
    AnchorRow,
    TypedAnchor,
    anchor_alias_hits,
    anchor_recall_curve,
    attach_anchor_features,
    build_anchor_feature_table,
    decide_verdict,
    feature_matrix_with_anchors,
    fit_with_and_without_anchors,
    pass_typed_anchor_names,
    per_feature_auroc,
)
from scripts.phase1b_reweight_separator import COMPONENT_FIELDS, ComponentRow


# ===========================================================================
# anchor_alias_hits — the matcher (mirrors the runtime C8 sub-channel)
# ===========================================================================

class TestAnchorAliasHits:
    def test_brief_canonical_case_two_present_one_absent(self):
        """"SNR-75 Fan Song" with anchors [SNR-75, Fan Song, V-75] → 2."""
        text = "The SNR-75 Fan Song guidance radar."
        anchors = ["SNR-75", "Fan Song", "V-75"]
        assert anchor_alias_hits(text, anchors) == 2

    def test_absent_anchor_zero(self):
        assert anchor_alias_hits("nothing relevant here", ["SNR-75", "Fan Song"]) == 0

    def test_case_insensitive_via_casefold(self):
        # production matcher casefolds both sides
        assert anchor_alias_hits("the snr-75 unit", ["SNR-75"]) == 1

    def test_counts_each_distinct_anchor_once_present(self):
        # count is over distinct anchor STRINGS that appear (sum of indicators),
        # matching the runtime ``sum(1 for a in anchors if a in haystack)``.
        text = "Fan Song Fan Song"  # same anchor twice in text
        assert anchor_alias_hits(text, ["Fan Song"]) == 1

    def test_empty_anchor_list_zero(self):
        assert anchor_alias_hits("SNR-75 Fan Song", []) == 0

    def test_empty_text_zero(self):
        assert anchor_alias_hits("", ["SNR-75"]) == 0

    def test_blank_anchors_skipped(self):
        # blank / whitespace anchors must not count (and must not match-all)
        assert anchor_alias_hits("anything", ["", "   "]) == 0

    def test_uses_production_nfc_normalisation(self):
        """Cyrillic in decomposed (NFD) vs precomposed (NFC) form must still
        match — the production matcher NFC-normalises BOTH sides. We assert the
        function uses the SAME normaliser (``extraction_lexical_search._nfc``)
        by feeding an NFD haystack and an NFC anchor."""
        import unicodedata

        nfc_anchor = unicodedata.normalize("NFC", "Алмаз")  # precomposed
        nfd_text = unicodedata.normalize("NFD", "система Алмаз")  # decomposed
        assert nfc_anchor != nfd_text  # byte forms differ
        assert anchor_alias_hits(nfd_text, [nfc_anchor]) == 1


# ===========================================================================
# pass_typed_anchor_names — variant (b) type matching (bundle-driven)
# ===========================================================================

def _pass_def(name, primary_entity_types):
    from types import SimpleNamespace
    return SimpleNamespace(name=name, primary_entity_types=list(primary_entity_types))


class TestPassTypedAnchorNames:
    def test_keeps_only_matching_entity_type(self):
        typed = [
            TypedAnchor("SNR-75", "RADAR_SYSTEM"),
            TypedAnchor("Fan Song", "RADAR_SYSTEM"),
            TypedAnchor("V-75", "MISSILE_SYSTEM"),
            TypedAnchor("Volkhov", "MISSILE_SYSTEM"),
        ]
        radar_pass = _pass_def("radar_power_rf", ["RADAR_SYSTEM"])
        names = pass_typed_anchor_names(radar_pass, typed)
        assert set(names) == {"SNR-75", "Fan Song"}

        missile_pass = _pass_def("missile_kinematics", ["MISSILE_SYSTEM"])
        assert set(pass_typed_anchor_names(missile_pass, typed)) == {"V-75", "Volkhov"}

    def test_multi_type_pass_unions(self):
        typed = [
            TypedAnchor("SNR-75", "RADAR_SYSTEM"),
            TypedAnchor("V-75", "MISSILE_SYSTEM"),
        ]
        both = _pass_def("combo", ["RADAR_SYSTEM", "MISSILE_SYSTEM"])
        assert set(pass_typed_anchor_names(both, typed)) == {"SNR-75", "V-75"}

    def test_no_matching_type_empty(self):
        typed = [TypedAnchor("SNR-75", "RADAR_SYSTEM")]
        p = _pass_def("missile_kinematics", ["MISSILE_SYSTEM"])
        assert pass_typed_anchor_names(p, typed) == []

    def test_dedups_names(self):
        typed = [
            TypedAnchor("SNR-75", "RADAR_SYSTEM"),
            TypedAnchor("SNR-75", "RADAR_SYSTEM"),
        ]
        p = _pass_def("radar_power_rf", ["RADAR_SYSTEM"])
        assert pass_typed_anchor_names(p, typed) == ["SNR-75"]


# ===========================================================================
# build_anchor_feature_table — per (pass, chunk) AnchorRow
# ===========================================================================

class TestBuildAnchorFeatureTable:
    def test_per_pass_all_vs_typed_counts(self):
        chunks = [
            {"self_ref": "c0", "chunk_index": 0,
             "chunk_text": "SNR-75 Fan Song radar"},
            {"self_ref": "c1", "chunk_index": 1,
             "chunk_text": "V-75 Volkhov missile"},
            {"self_ref": "c2", "chunk_index": 2,
             "chunk_text": "unrelated paragraph"},
        ]
        all_anchors = ["SNR-75", "Fan Song", "V-75", "Volkhov"]
        typed = [
            TypedAnchor("SNR-75", "RADAR_SYSTEM"),
            TypedAnchor("Fan Song", "RADAR_SYSTEM"),
            TypedAnchor("V-75", "MISSILE_SYSTEM"),
            TypedAnchor("Volkhov", "MISSILE_SYSTEM"),
        ]
        pass_defs = {
            "radar_power_rf": _pass_def("radar_power_rf", ["RADAR_SYSTEM"]),
            "missile_kinematics": _pass_def("missile_kinematics", ["MISSILE_SYSTEM"]),
        }
        rows = build_anchor_feature_table(
            chunks_by_pass={pn: chunks for pn in pass_defs},
            all_anchors=all_anchors,
            typed_anchors=typed,
            pass_defs=pass_defs,
        )
        by = {(r.pass_name, r.chunk_index): r for r in rows}
        # ALL anchors: chunk0 hits SNR-75+Fan Song (2); chunk1 hits V-75+Volkhov (2)
        assert by[("radar_power_rf", 0)].anchor_alias_hits == 2
        assert by[("radar_power_rf", 1)].anchor_alias_hits == 2
        assert by[("radar_power_rf", 2)].anchor_alias_hits == 0
        # TYPED for the RADAR pass: only radar anchors count → chunk0=2, chunk1=0
        assert by[("radar_power_rf", 0)].anchor_alias_hits_typed == 2
        assert by[("radar_power_rf", 1)].anchor_alias_hits_typed == 0
        # TYPED for the MISSILE pass: only missile anchors → chunk0=0, chunk1=2
        assert by[("missile_kinematics", 0)].anchor_alias_hits_typed == 0
        assert by[("missile_kinematics", 1)].anchor_alias_hits_typed == 2
        # ALL is pass-independent (same anchors every pass)
        assert by[("missile_kinematics", 1)].anchor_alias_hits == 2

    def test_row_count_matches_chunks_times_passes(self):
        chunks = [{"self_ref": f"c{i}", "chunk_index": i, "chunk_text": "x"}
                  for i in range(4)]
        pass_defs = {"p": _pass_def("p", ["RADAR_SYSTEM"])}
        rows = build_anchor_feature_table(
            chunks_by_pass={"p": chunks}, all_anchors=["x"],
            typed_anchors=[TypedAnchor("x", "RADAR_SYSTEM")], pass_defs=pass_defs)
        assert len(rows) == 4


# ===========================================================================
# attach_anchor_features + feature_matrix_with_anchors
# ===========================================================================

def _comp(pass_name, ci, used, **feats):
    kw = {f: 0.0 for f in COMPONENT_FIELDS}
    kw.update(feats)
    return ComponentRow(pass_name=pass_name, chunk_index=ci, used=used, **kw)


class TestAttachAnchorFeatures:
    def test_attaches_both_anchor_columns_by_pass_chunk(self):
        comp_rows = [
            _comp("radar_power_rf", 0, True, rerank_norm=0.9),
            _comp("radar_power_rf", 1, False, rerank_norm=0.1),
        ]
        anchor_rows = [
            AnchorRow("radar_power_rf", 0, anchor_alias_hits=3, anchor_alias_hits_typed=2),
            AnchorRow("radar_power_rf", 1, anchor_alias_hits=0, anchor_alias_hits_typed=0),
        ]
        merged = attach_anchor_features(comp_rows, anchor_rows)
        m0 = [r for r in merged if r["chunk_index"] == 0][0]
        assert m0["anchor_alias_hits"] == 3.0
        assert m0["anchor_alias_hits_typed"] == 2.0
        assert m0["rerank_norm"] == pytest.approx(0.9)
        assert m0["used"] is True

    def test_missing_anchor_row_defaults_zero(self):
        comp_rows = [_comp("p", 7, False)]
        merged = attach_anchor_features(comp_rows, [])
        assert merged[0]["anchor_alias_hits"] == 0.0
        assert merged[0]["anchor_alias_hits_typed"] == 0.0

    def test_without_matrix_is_strict_column_subset_of_with(self):
        rows = [
            {"pass_name": "p", "chunk_index": 0, "used": True,
             **{f: 0.5 for f in COMPONENT_FIELDS},
             "anchor_alias_hits": 2.0, "anchor_alias_hits_typed": 1.0},
            {"pass_name": "p", "chunk_index": 1, "used": False,
             **{f: 0.1 for f in COMPONENT_FIELDS},
             "anchor_alias_hits": 0.0, "anchor_alias_hits_typed": 0.0},
        ]
        X_with, y, names_with = feature_matrix_with_anchors(rows, include_anchors=True)
        X_wo, y2, names_wo = feature_matrix_with_anchors(rows, include_anchors=False)
        # WITHOUT names are exactly WITH names minus the anchor features
        assert list(names_wo) == [n for n in names_with if n not in ANCHOR_FEATURE_FIELDS]
        assert set(ANCHOR_FEATURE_FIELDS).issubset(set(names_with))
        assert set(ANCHOR_FEATURE_FIELDS).isdisjoint(set(names_wo))
        assert X_with.shape[0] == X_wo.shape[0] == 2
        assert X_with.shape[1] == X_wo.shape[1] + len(ANCHOR_FEATURE_FIELDS)
        assert list(y) == list(y2) == [1, 0]


# ===========================================================================
# per_feature_auroc — generic univariate AUROC over named features
# ===========================================================================

class TestPerFeatureAuroc:
    def test_perfect_anchor_feature_auroc_1(self):
        rows = [
            {"pass_name": "p", "chunk_index": 0, "used": True,
             "anchor_alias_hits": 3.0, "anchor_alias_hits_typed": 3.0},
            {"pass_name": "p", "chunk_index": 1, "used": True,
             "anchor_alias_hits": 2.0, "anchor_alias_hits_typed": 2.0},
            {"pass_name": "p", "chunk_index": 2, "used": False,
             "anchor_alias_hits": 0.0, "anchor_alias_hits_typed": 0.0},
            {"pass_name": "p", "chunk_index": 3, "used": False,
             "anchor_alias_hits": 0.0, "anchor_alias_hits_typed": 0.0},
        ]
        res = per_feature_auroc(rows, ANCHOR_FEATURE_FIELDS)
        assert res["pooled"]["anchor_alias_hits"] == pytest.approx(1.0)
        assert res["pooled"]["anchor_alias_hits_typed"] == pytest.approx(1.0)

    def test_constant_feature_auroc_half(self):
        rows = [
            {"pass_name": "p", "chunk_index": i, "used": (i < 2),
             "anchor_alias_hits": 1.0, "anchor_alias_hits_typed": 1.0}
            for i in range(4)
        ]
        res = per_feature_auroc(rows, ANCHOR_FEATURE_FIELDS)
        # a constant feature cannot separate → 0.5 by construction
        assert res["pooled"]["anchor_alias_hits"] == pytest.approx(0.5)

    def test_reports_pooled_per_pass_and_firing_rate(self):
        rows = [
            {"pass_name": "rp", "chunk_index": 0, "used": True,
             "anchor_alias_hits": 2.0, "anchor_alias_hits_typed": 1.0},
            {"pass_name": "rp", "chunk_index": 1, "used": False,
             "anchor_alias_hits": 0.0, "anchor_alias_hits_typed": 0.0},
            {"pass_name": "mk", "chunk_index": 0, "used": True,
             "anchor_alias_hits": 1.0, "anchor_alias_hits_typed": 0.0},
            {"pass_name": "mk", "chunk_index": 1, "used": False,
             "anchor_alias_hits": 3.0, "anchor_alias_hits_typed": 0.0},
        ]
        res = per_feature_auroc(rows, ANCHOR_FEATURE_FIELDS)
        assert "pooled" in res and "per_pass" in res and "firing_rate" in res
        assert "rp" in res["per_pass"] and "mk" in res["per_pass"]
        # firing rate = fraction of rows with feature > 0
        # anchor_alias_hits fires on 3/4 rows pooled
        assert res["firing_rate"]["pooled"]["anchor_alias_hits"] == pytest.approx(0.75)
        # typed fires on 1/4 pooled
        assert res["firing_rate"]["pooled"]["anchor_alias_hits_typed"] == pytest.approx(0.25)


# ===========================================================================
# fit_with_and_without_anchors — THE probe (does adding anchors help, honestly?)
# ===========================================================================

class TestFitWithAndWithoutAnchorsSeparable:
    """A SEPARABLE table where the TYPED anchor feature is the ONLY signal:
    WITH-anchors CV AUROC must clear the floor AND beat WITHOUT (which has no
    signal → ~0.5)."""

    def test_anchor_only_signal_with_beats_without_high_cv(self):
        rng = random.Random(11)
        rows = []
        ci = 0
        for _pass in ("missile_kinematics", "radar_power_rf"):
            for _ in range(20):  # used: typed anchor fires high
                row = {"pass_name": _pass, "chunk_index": ci, "used": True}
                for f in COMPONENT_FIELDS:
                    row[f] = rng.uniform(0.0, 0.4)  # base components = noise
                row["anchor_alias_hits"] = float(rng.randint(1, 4))
                row["anchor_alias_hits_typed"] = float(rng.randint(1, 4))
                rows.append(row); ci += 1
            for _ in range(40):  # unused: typed anchor never fires
                row = {"pass_name": _pass, "chunk_index": ci, "used": False}
                for f in COMPONENT_FIELDS:
                    row[f] = rng.uniform(0.0, 0.4)
                row["anchor_alias_hits"] = 0.0
                row["anchor_alias_hits_typed"] = 0.0
                rows.append(row); ci += 1

        out = fit_with_and_without_anchors(rows, n_splits=5, seed=0)
        with_cv = out["with_anchors"]["best_cv_auroc"]
        without_cv = out["without_anchors"]["best_cv_auroc"]
        assert with_cv > 0.90
        assert with_cv > without_cv + 0.10
        # the anchor delta is reported
        assert out["delta_best_cv_auroc"] == pytest.approx(with_cv - without_cv, abs=1e-9)

    def test_anchor_feature_dominates_learned_weights(self):
        rng = random.Random(13)
        rows = []
        ci = 0
        for _ in range(30):
            row = {"pass_name": "p", "chunk_index": ci, "used": True}
            for f in COMPONENT_FIELDS:
                row[f] = rng.uniform(0.0, 0.3)
            row["anchor_alias_hits"] = float(rng.randint(1, 5))
            row["anchor_alias_hits_typed"] = float(rng.randint(1, 5))
            rows.append(row); ci += 1
        for _ in range(60):
            row = {"pass_name": "p", "chunk_index": ci, "used": False}
            for f in COMPONENT_FIELDS:
                row[f] = rng.uniform(0.0, 0.3)
            row["anchor_alias_hits"] = 0.0
            row["anchor_alias_hits_typed"] = 0.0
            rows.append(row); ci += 1
        out = fit_with_and_without_anchors(rows, n_splits=5, seed=0)
        w = out["with_anchors"]["logistic_weights"]
        dominant = max(w, key=lambda k: abs(w[k]))
        assert dominant in ANCHOR_FEATURE_FIELDS


class TestFitWithAndWithoutAnchorsInseparable:
    """CRITICAL HONESTY: anchors that FIRE but are label-INDEPENDENT must NOT
    lift CV AUROC, even though in-sample (high-capacity GBM) can overfit. This
    is the SA-2 'near-binary single-system' failure mode in miniature."""

    def test_anchors_fire_but_independent_cv_stays_chance(self):
        rng = random.Random(101)
        rows = []
        ci = 0
        for _ in range(120):
            used = rng.random() < 0.25  # label independent of everything
            row = {"pass_name": "p", "chunk_index": ci, "used": used}
            for f in COMPONENT_FIELDS:
                row[f] = rng.random()
            # anchors FIRE (non-zero on most rows) but carry no signal
            row["anchor_alias_hits"] = float(rng.randint(0, 5))
            row["anchor_alias_hits_typed"] = float(rng.randint(0, 3))
            rows.append(row); ci += 1
        out = fit_with_and_without_anchors(rows, n_splits=5, seed=0)
        # honest CV near chance WITH anchors
        assert out["with_anchors"]["best_cv_auroc"] == pytest.approx(0.5, abs=0.18)
        # adding anchors does not materially move CV when they carry no signal
        assert out["delta_best_cv_auroc"] < 0.12
        # in-sample GBM overfits high → proves CV is the honest read
        assert (out["with_anchors"]["gbm_in_sample_auroc"]
                > out["with_anchors"]["gbm_cv_auroc"] + 0.15)

    def test_reports_with_without_and_delta_for_recall(self):
        rng = random.Random(5)
        rows = []
        for i in range(80):
            row = {"pass_name": "p", "chunk_index": i, "used": rng.random() < 0.3}
            for f in COMPONENT_FIELDS:
                row[f] = rng.random()
            row["anchor_alias_hits"] = float(rng.randint(0, 4))
            row["anchor_alias_hits_typed"] = float(rng.randint(0, 2))
            rows.append(row)
        out = fit_with_and_without_anchors(rows, n_splits=4, seed=0)
        for side in ("with_anchors", "without_anchors"):
            assert "best_cv_auroc" in out[side]
            assert "logistic_cv_auroc" in out[side]
            assert "gbm_cv_auroc" in out[side]
            assert "best_scores" in out[side]
            assert len(out[side]["best_scores"]) == len(rows)
        assert "delta_best_cv_auroc" in out


# ===========================================================================
# anchor_recall_curve — reuse of phase1 recall sweep over the WITH-anchors score
# ===========================================================================

class TestAnchorRecallCurve:
    def test_pooled_and_per_pass_recommendation(self):
        # a clean separable score: used rows score high, unused low.
        rows, scores = [], []
        for i in range(6):
            used = i < 3
            rows.append({"pass_name": "p", "chunk_index": i, "used": used})
            scores.append(0.9 if used else 0.1)
        rc = anchor_recall_curve(rows, scores, recall_floor=1.0)
        rec = rc["pooled_recommendation"]
        # full recall preserved while dropping the unused half → frac_kept 0.5
        assert rec.recall_achieved == pytest.approx(1.0)
        assert rec.fraction_kept == pytest.approx(0.5)
        assert "p" in rc["per_pass"]


# ===========================================================================
# decide_verdict — the (i)/(ii)/(iii) trichotomy, CV-driven and honest
# ===========================================================================

def _feat_auroc(fire_all, fire_typed):
    return {"firing_rate": {"pooled": {
        "anchor_alias_hits": fire_all,
        "anchor_alias_hits_typed": fire_typed,
    }}}


def _fit(with_best, without_best, *, with_logit_is=0.7, with_gbm_is=0.7,
         with_logit_cv=None, with_gbm_cv=None):
    wl = with_logit_cv if with_logit_cv is not None else with_best
    wg = with_gbm_cv if with_gbm_cv is not None else with_best
    return {
        "with_anchors": {
            "best_cv_auroc": with_best,
            "logistic_in_sample_auroc": with_logit_is,
            "gbm_in_sample_auroc": with_gbm_is,
            "logistic_cv_auroc": wl,
            "gbm_cv_auroc": wg,
        },
        "without_anchors": {"best_cv_auroc": without_best},
        "delta_best_cv_auroc": with_best - without_best,
    }


class _Rec:
    def __init__(self, fraction_kept):
        self.fraction_kept = fraction_kept
        self.recall_achieved = 1.0
        self.threshold = 0.5
        self.chunks_kept = 10


class TestDecideVerdict:
    def test_iii_nofire_when_firing_rate_below_floor(self):
        feat = _feat_auroc(fire_all=0.02, fire_typed=0.0)
        fit = _fit(0.55, 0.55)
        rc = {"pooled_recommendation": _Rec(0.5)}
        verdict, notes = decide_verdict(feat, fit, rc, fire_floor=0.05)
        assert verdict == "NOFIRE"
        assert any("(iii)" in n for n in notes)

    def test_i_improve_when_clears_floor_beats_baseline_and_reduces(self):
        feat = _feat_auroc(fire_all=0.7, fire_typed=0.6)
        # WITH best CV well above floor and far above WITHOUT
        fit = _fit(0.82, 0.60)
        rc = {"pooled_recommendation": _Rec(0.4)}  # reduces the pool
        verdict, notes = decide_verdict(feat, fit, rc, auroc_sep_floor=0.75)
        assert verdict == "IMPROVE"
        assert any("(i)" in n for n in notes)

    def test_ii_fire_nosep_when_fires_but_cv_below_floor(self):
        # the SA-2 case: anchors fire heavily but CV stays under the floor.
        feat = _feat_auroc(fire_all=0.98, fire_typed=0.70)
        fit = _fit(0.672, 0.607, with_gbm_is=0.97, with_gbm_cv=0.672,
                   with_logit_cv=0.629)
        rc = {"pooled_recommendation": _Rec(0.45)}
        verdict, notes = decide_verdict(feat, fit, rc, auroc_sep_floor=0.75)
        assert verdict == "FIRE_NOSEP"
        assert any("(ii)" in n for n in notes)
        # the overfitting trap must be surfaced when in-sample >> CV
        assert any("OVERFITTING TRAP" in n for n in notes)
        # and the heterogeneous-doc recommendation must be present
        assert any("HETEROGENEOUS" in n for n in notes)

    def test_ii_when_clears_floor_but_no_reduction(self):
        # even with a high CV, if no threshold reduces the pool it is not (i).
        feat = _feat_auroc(fire_all=0.7, fire_typed=0.6)
        fit = _fit(0.80, 0.60)
        rc = {"pooled_recommendation": _Rec(0.99)}  # keeps ~all → no reduction
        verdict, _ = decide_verdict(feat, fit, rc, auroc_sep_floor=0.75)
        assert verdict == "FIRE_NOSEP"
