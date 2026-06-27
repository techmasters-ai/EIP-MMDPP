"""Phase-1b: can we RE-WEIGHT the router's component metrics to separate
USED chunks from UNUSED ones?  (OFFLINE, deterministic, read-only.)

Phase-1 (``scripts/phase1_score_used_separation.py``) established that NO
single router score (cosine / reranker / final) separates the chunks a fact
was actually traced TO (USED, via lineage) from the rest — best AUROC was
~0.66-0.71, under the 0.75 floor.

The production ``final_score`` is ONE FIXED weighting of components, tuned for
RANKING::

    final = w_rerank   * rerank_norm
          + w_lexical  * lexical_norm
          + w_pattern  * pattern_norm
          + w_section  * section_norm
          + w_table    * is_table
          - w_negative * negative_norm

This module asks the supervised-learning question: with the COMPONENTS as
features and the lineage ``used`` flag as the label, does a DIFFERENT
(learned) weighting separate USED from UNUSED?  If even a high-capacity
nonlinear model can't (under honest cross-validation), the signal simply
isn't in these features and a re-weighted threshold cannot help → the
fallback is tuned per-pass top_k.

Method (everything REUSED from Phase-1 + the production C5 scorer):
  * COMPONENT capture — ``component_breakdown`` replays the EXACT C5
    normalisation (min-max for the reranker; ratio-max for the hit signals)
    over the full per-pass pool and RETURNS the per-candidate component
    vector.  A lockstep test pins recombine(components·weights) ==
    ``score_candidates`` final_score.
  * ``build_component_table`` — per (pass, chunk) ComponentRow with
    cosine + every C5 component (reuses Phase-1's embed / rerank / signal
    wrappers and ``_pass_signals``).
  * ``per_component_auroc`` — univariate AUROC(used, component) per pass +
    pooled (REUSES Phase-1's exact-tie ``auroc``).
  * ``fit_reweighted_separators`` — L2 logistic (the re-weighted aggregate;
    learned weights reported) + a gradient-boosted tree (nonlinear ceiling),
    each scored IN-SAMPLE and by stratified k-fold CV.  CV is the honest
    "does the signal exist within this doc" estimate; in-sample is only a
    ceiling and OVERFITS on ~60 chunks/pass × ~7 features.
  * recall-vs-threshold on the best re-weighted score (REUSES Phase-1's
    ``recall_vs_threshold`` + ``_threshold_recommendation``).
  * ``build_stable_used_table`` — the noise floor: USED ∩ across two runs of
    the SAME doc, matched at the stable ``self_ref`` element anchor (NOT raw
    chunk_index), labelled onto the current run's index.

Import-safe: no DB / model work at import time.  Pure functions are unit-
tested with synthetic fixtures (incl. a SEPARABLE table → high CV + right
dominant weight, and an INSEPARABLE table → CV ~0.5 so in-sample overfit
cannot fool us).  All DB / embedding / rerank I/O is the Phase-1 wrappers,
invoked only by ``main()``.

Usage (read-only on the live DBs)::

    python scripts/phase1b_reweight_separator.py \
        --run-id 5da1a210-0322-401e-a2f2-8a7cab68ca00 \
        --doc-id ddaa9e36-2854-47c3-bc94-ff38d531dafd \
        --noise-floor-run-id 625ad1bd-016d-46e4-8158-38720d0fe88d

Outputs (under reports/collection/):
    phase1b_components_<run8>.csv     — (pass, chunk, components, used)
    phase1b_reweight_<run8>.json      — full analysis report
    phase1b_reweight_<run8>.md        — the human-readable RESULT REPORT
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

# Reuse Phase-1's lineage join, AUROC, recall sweep, recommendation, the score
# table builder's machinery, and ALL the read-only I/O wrappers — nothing is
# reimplemented.
from scripts import phase1_score_used_separation as p1
from scripts.phase1_score_used_separation import (  # noqa: F401  (re-exported)
    ScoreUsedRow,
    USED_SOURCE_DIRECT,
    auroc,
    build_used_table,
    recall_vs_threshold,
)


# ---------------------------------------------------------------------------
# Component feature set
# ---------------------------------------------------------------------------
# The C5 components score_candidates computes per candidate, PLUS dense cosine
# (an absolute channel Phase-1 already recomputes).  These are the FEATURES the
# re-weighting fits on.  Order is the canonical feature-matrix column order.
COMPONENT_FIELDS: tuple[str, ...] = (
    "cosine",
    "rerank_norm",
    "lexical_norm",
    "pattern_norm",
    "section_norm",
    "is_table",
    "negative_norm",
)

# AUROC floor a (re-weighted) separator must clear to count as "the signal
# exists".  Same floor Phase-1 used for the single-score verdict.
DEFAULT_AUROC_SEP_FLOOR = p1.DEFAULT_AUROC_SEP_FLOOR  # 0.75


@dataclass
class ComponentRow:
    """One (pass, chunk) row: every C5 component feature + the used label."""
    pass_name: str
    chunk_index: int
    used: bool
    cosine: float = 0.0
    rerank_norm: float = 0.0
    lexical_norm: float = 0.0
    pattern_norm: float = 0.0
    section_norm: float = 0.0
    is_table: float = 0.0
    negative_norm: float = 0.0
    self_ref: str = ""


# ===========================================================================
# component_breakdown — replays the EXACT C5 normalisation, returns components
# ===========================================================================

def component_breakdown(
    candidates: list[dict[str, Any]],
    cfg: Any,
) -> dict[int, dict[str, float]]:
    """Return {chunk_index: {component_name: value}} for a C5 candidate pool.

    Replays the SAME pool-relative normalisation ``score_candidates`` uses so
    that ``sum(weight_i * component_i)`` (clamped at 0) reproduces the
    production ``final_score`` EXACTLY (pinned by the lockstep test):

        rerank_norm   : min-max over the pool's reranker_scores; missing → 0.0;
                        all-equal scorable pool → 1.0 (matches C5).
        lexical_norm  = alias_hits    / max(1, pool_max_alias_hits)
        pattern_norm  = pattern_hits  / max(1, pool_max_pattern_hits)
        section_norm  = section_hits  / max(1, pool_max_section_hits)
        negative_norm = negative_hits / max(1, pool_max_negative_hits)
        is_table      = 1.0 if content_type == "table" else 0.0

    NOTE: ``cosine`` is NOT a C5 component (final_score uses only the six
    above); it is added separately by ``build_component_table`` as an extra
    absolute feature.  This function returns the SIX C5 components only.
    """
    out: dict[int, dict[str, float]] = {}
    if not candidates:
        return out

    # rerank min-max — identical branch logic to score_candidates.
    raw_rr: list[float] = [c["reranker_score"] for c in candidates if "reranker_score" in c]
    if raw_rr and len(raw_rr) > 1:
        rr_min, rr_max = min(raw_rr), max(raw_rr)
        rr_span = rr_max - rr_min
    elif raw_rr:
        rr_min = rr_max = raw_rr[0]
        rr_span = 0.0
    else:
        rr_min = rr_max = rr_span = 0.0

    mcs = [c["merged_candidate"] for c in candidates]
    max_alias = max((mc.alias_hits for mc in mcs), default=0)
    max_pattern = max((mc.pattern_hits for mc in mcs), default=0)
    max_section = max((mc.section_hits for mc in mcs), default=0)
    max_negative = max((mc.negative_hits for mc in mcs), default=0)

    for c in candidates:
        mc = c["merged_candidate"]
        raw = c.get("reranker_score")
        if raw is None:
            rerank_norm = 0.0
        elif rr_span > 0.0:
            rerank_norm = (raw - rr_min) / rr_span
        else:
            rerank_norm = 1.0 if raw_rr else 0.0

        out[int(mc.chunk_index)] = {
            "rerank_norm": float(rerank_norm),
            "lexical_norm": float(mc.alias_hits / max(1, max_alias)),
            "pattern_norm": float(mc.pattern_hits / max(1, max_pattern)),
            "section_norm": float(mc.section_hits / max(1, max_section)),
            "negative_norm": float(mc.negative_hits / max(1, max_negative)),
            "is_table": 1.0 if mc.content_type == "table" else 0.0,
        }
    return out


# ===========================================================================
# build_component_table — per (pass, chunk) ComponentRow (reuses Phase-1 I/O)
# ===========================================================================

def build_component_table(
    *,
    chunks_by_pass: Mapping[str, Sequence[Mapping[str, Any]]],
    profile: Any = None,
) -> list[ComponentRow]:
    """Per (pass, chunk) ComponentRow with cosine + every C5 component.

    Mirrors ``phase1.build_chunk_score_table`` (same embed / rerank / signal
    machinery, same MergedCandidate construction) but emits the COMPONENT
    breakdown instead of the single final_score.  ``used`` is left False here
    (join the lineage labels in separately via ``label_component_table``).
    DB I/O is none — chunks are passed in; embed / rerank / signals are the
    Phase-1 wrappers (monkeypatched in tests on the phase1 module).
    """
    from app.services.extraction_candidate_scoring import MergedCandidate

    if profile is None:
        from app.services.ontology_bundles import RetrievalProfile
        profile = RetrievalProfile()
    pattern_limit = int(getattr(profile, "pattern_hit_limit", 50) or 50)

    from app.services.extraction_lexical_search import (
        lexical_hit_counts,
        pattern_hit_counts,
    )

    import numpy as np

    out: list[ComponentRow] = []
    for pass_name, chunks in chunks_by_pass.items():
        rows = [dict(c) for c in chunks if c.get("embedding")]
        if not rows:
            continue
        signals = p1._pass_signals(pass_name)
        query_text = getattr(signals, "entity_query", "") or ""
        field_queries = getattr(signals, "field_queries", ()) or ()

        # cosine (same normalise + dot as Phase-1)
        qv = np.asarray(p1._normalize_vec(p1._embed_query(query_text)), dtype=np.float32)
        emb = np.asarray([p1._normalize_vec(r["embedding"]) for r in rows], dtype=np.float32)
        cos = (emb @ qv).astype(float).tolist()
        cos_by_ci = {str(r["chunk_index"]): c for r, c in zip(rows, cos)}

        # reranker (full corpus, order-invariant)
        rer_cands = [
            {"content_text": r.get("chunk_text") or "", "chunk_index": r["chunk_index"]}
            for r in rows
        ]
        rer_map = p1._rerank_scores(query_text, rer_cands)

        # lexical / pattern hit signals (offline, pure)
        lex = lexical_hit_counts(rows, field_queries)
        pat = pattern_hit_counts(rows, field_queries, pattern_limit)

        c5_input: list[dict[str, Any]] = []
        for r in rows:
            key = str(r.get("vertex_id") or r.get("self_ref") or r["chunk_index"])
            ci = str(r["chunk_index"])
            lh = lex.get(key, {})
            ph = pat.get(key, {})
            mc = MergedCandidate(
                candidate_key=key,
                chunk_index=int(r["chunk_index"]),
                self_ref=str(r.get("self_ref", "")),
                chunk_text=r.get("chunk_text") or "",
                source_refs=list(r.get("source_refs") or []),
                token_count=int(r.get("token_count", 0) or 0),
                page_number=r.get("page_number"),
                vector_score=None,
                field_scores={},
                alias_hits=int(lh.get("alias_hits", 0)),
                pattern_hits=int(ph.get("pattern_hits", 0)),
                negative_hits=int(lh.get("negative_hits", 0)),
                section_hits=0,
                content_type=p1._table_content_type(r.get("source_refs") or []),
                retrieval_sources=set(),
                supported_field_hints=set(),
            )
            entry: dict[str, Any] = {"merged_candidate": mc, "content_text": mc.chunk_text}
            rr = rer_map.get(ci)
            if rr is not None:
                entry["reranker_score"] = rr
            c5_input.append(entry)

        comps = component_breakdown(c5_input, profile)

        for r in rows:
            ci_i = int(r["chunk_index"])
            ci = str(ci_i)
            cb = comps.get(ci_i, {})
            out.append(ComponentRow(
                pass_name=pass_name,
                chunk_index=ci_i,
                used=False,
                cosine=float(cos_by_ci.get(ci, 0.0)),
                rerank_norm=float(cb.get("rerank_norm", 0.0)),
                lexical_norm=float(cb.get("lexical_norm", 0.0)),
                pattern_norm=float(cb.get("pattern_norm", 0.0)),
                section_norm=float(cb.get("section_norm", 0.0)),
                is_table=float(cb.get("is_table", 0.0)),
                negative_norm=float(cb.get("negative_norm", 0.0)),
                self_ref=str(r.get("self_ref", "")),
            ))
    return out


def label_component_table(
    comp_rows: Sequence[ComponentRow],
    used_rows: Sequence[ScoreUsedRow],
) -> list[ComponentRow]:
    """Stamp the lineage ``used`` flag (from a ScoreUsedRow table) onto the
    ComponentRows, joining on (pass_name, chunk_index)."""
    used_idx = {(r.pass_name, r.chunk_index): r.used for r in used_rows}
    out: list[ComponentRow] = []
    for c in comp_rows:
        d = asdict(c)
        d["used"] = bool(used_idx.get((c.pass_name, c.chunk_index), c.used))
        out.append(ComponentRow(**d))
    return out


# ===========================================================================
# per_component_auroc — univariate signal detection (REUSES phase1.auroc)
# ===========================================================================

def per_component_auroc(
    rows: Sequence[ComponentRow],
) -> dict[str, Any]:
    """AUROC(used, component) for each component, pooled and per-pass.

    Returns::

        {
          "pooled":   {component: auroc, ...},
          "per_pass": {pass_name: {component: auroc, ...}, ...},
          "support":  {"pooled": {"n": .., "n_used": ..},
                       "per_pass": {pass_name: {"n": .., "n_used": ..}}},
        }

    Uses Phase-1's exact-tie ``auroc`` (matches sklearn.roc_auc_score). A
    constant component (no variance) yields 0.5 by construction.
    """
    used = [r.used for r in rows]
    pooled = {f: auroc([getattr(r, f) for r in rows], used) for f in COMPONENT_FIELDS}

    by_pass: dict[str, list[ComponentRow]] = {}
    for r in rows:
        by_pass.setdefault(r.pass_name, []).append(r)

    per_pass: dict[str, dict[str, float]] = {}
    support_pp: dict[str, dict[str, int]] = {}
    for pn, prs in by_pass.items():
        u = [r.used for r in prs]
        per_pass[pn] = {f: auroc([getattr(r, f) for r in prs], u) for f in COMPONENT_FIELDS}
        support_pp[pn] = {"n": len(prs), "n_used": sum(1 for x in u if x)}

    return {
        "pooled": pooled,
        "per_pass": per_pass,
        "support": {
            "pooled": {"n": len(rows), "n_used": sum(1 for x in used if x)},
            "per_pass": support_pp,
        },
    }


# ===========================================================================
# fit_reweighted_separators — THE probe (logistic + GBM; in-sample vs CV)
# ===========================================================================

def _feature_matrix(
    rows: Sequence[ComponentRow], *, add_pass_indicator: bool
) -> tuple[Any, Any, list[str]]:
    """Build (X, y, feature_names). When add_pass_indicator, one-hot the pass
    (drop-first) so a pooled fit can offset per-pass base rates."""
    import numpy as np

    base = list(COMPONENT_FIELDS)
    X_base = np.asarray(
        [[getattr(r, f) for f in base] for r in rows], dtype=np.float64
    )
    y = np.asarray([1 if r.used else 0 for r in rows], dtype=np.int64)

    names = list(base)
    if add_pass_indicator:
        passes = sorted({r.pass_name for r in rows})
        if len(passes) > 1:
            # drop-first one-hot
            keep = passes[1:]
            ind = np.asarray(
                [[1.0 if r.pass_name == p else 0.0 for p in keep] for r in rows],
                dtype=np.float64,
            )
            X = np.hstack([X_base, ind])
            names = list(base) + [f"pass={p}" for p in keep]
            return X, y, names
    return X_base, y, names


def _cv_auroc(estimator, X, y, *, n_splits: int, seed: int) -> float:
    """Stratified k-fold CV AUROC. Returns the mean across folds; NaN-safe.

    Falls back to a smaller fold count if a class is too small to stratify into
    n_splits, and returns 0.5 (chance) if CV cannot be formed at all (degenerate
    single-class label) — never raises.
    """
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 2 or n_neg < 2:
        return 0.5
    k = min(n_splits, n_pos, n_neg)
    if k < 2:
        return 0.5

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for tr, te in skf.split(X, y):
        if len(set(y[te].tolist())) < 2:
            continue
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        proba = est.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
    if not aucs:
        return 0.5
    return float(np.mean(aucs))


def _in_sample_auroc(estimator, X, y) -> tuple[float, Any]:
    """Fit on ALL rows, return (in-sample AUROC, per-row positive-class proba)."""
    from sklearn.metrics import roc_auc_score

    est = estimator
    est.fit(X, y)
    proba = est.predict_proba(X)[:, 1]
    if len(set(y.tolist())) < 2:
        return 0.5, proba
    return float(roc_auc_score(y, proba)), proba


def fit_reweighted_separators(
    rows: Sequence[ComponentRow],
    *,
    n_splits: int = 5,
    seed: int = 0,
    add_pass_indicator: bool = True,
) -> dict[str, Any]:
    """Fit the RE-WEIGHTED separators and report in-sample + CV AUROC.

    Two models, both with the components as features and ``used`` as label:

      * ``logistic`` — L2-regularised logistic regression. Its predicted
        positive-class probability IS the re-weighted aggregate; the learned
        coefficients ARE the re-weighting (reported per component). Features
        are standardised (zero-mean / unit-var) inside a Pipeline so the
        coefficients are comparable in magnitude.
      * ``gbm`` — a shallow gradient-boosted tree: an upper bound on ANY
        (nonlinear) separability. If even this cannot separate under CV, the
        signal is not in the features.

    Returns::

        {
          "logistic": {"in_sample_auroc": .., "cv_auroc": ..,
                       "weights": {component: coef, ...}, "intercept": ..},
          "gbm":      {"in_sample_auroc": .., "cv_auroc": ..,
                       "feature_importance": {component: imp, ...}},
          "logistic_scores": [per-row positive-class proba ...],
          "gbm_scores":      [per-row positive-class proba ...],
          "feature_names": [...],
          "n": .., "n_used": .., "n_splits_effective": ..,
        }

    CRITICAL: ``cv_auroc`` is the HONEST estimate of whether the signal exists
    within this doc; ``in_sample_auroc`` is only a ceiling and OVERFITS on a
    small/wide sample. Both are reported, clearly labelled, by the caller.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, names = _feature_matrix(rows, add_pass_indicator=add_pass_indicator)
    n = len(rows)
    n_used = int(y.sum())

    # ---- logistic (L2) ----
    # L2 is the default penalty in both sklearn 1.6 and 1.8 (the explicit
    # ``penalty="l2"`` keyword is deprecated in 1.8, so rely on the default and
    # keep C=1.0 — a moderate L2 strength — to be the regularisation knob).
    logit = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, solver="liblinear", max_iter=2000, random_state=seed,
        )),
    ])
    logit_cv = _cv_auroc(logit, X, y, n_splits=n_splits, seed=seed)
    # refit in-sample on a fresh clone so cv state doesn't leak
    from sklearn.base import clone as _clone
    logit_is_est = _clone(logit)
    logit_is, logit_scores = _in_sample_auroc(logit_is_est, X, y)

    # learned weights: coefficients in standardised space (magnitude-comparable),
    # mapped back to component names. Pass-indicator columns are reported too.
    clf = logit_is_est.named_steps["clf"]
    coefs = clf.coef_.ravel().tolist()
    intercept = float(clf.intercept_.ravel()[0])
    weights = {names[i]: float(coefs[i]) for i in range(len(names))}
    # the headline "which COMPONENT dominates" view excludes pass indicators
    comp_weights = {k: v for k, v in weights.items() if k in COMPONENT_FIELDS}

    # ---- gbm (nonlinear ceiling) ----
    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=1.0, random_state=seed,
    )
    gbm_cv = _cv_auroc(gbm, X, y, n_splits=n_splits, seed=seed)
    gbm_is_est = GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=1.0, random_state=seed,
    )
    gbm_is, gbm_scores = _in_sample_auroc(gbm_is_est, X, y)
    importances = getattr(gbm_is_est, "feature_importances_", None)
    feat_imp = (
        {names[i]: float(importances[i]) for i in range(len(names))
         if names[i] in COMPONENT_FIELDS}
        if importances is not None else {}
    )

    n_pos, n_neg = n_used, n - n_used
    eff_splits = max(2, min(n_splits, n_pos, n_neg)) if (n_pos >= 2 and n_neg >= 2) else 0

    return {
        "logistic": {
            "in_sample_auroc": logit_is,
            "cv_auroc": logit_cv,
            "weights": comp_weights,
            "all_weights": weights,
            "intercept": intercept,
        },
        "gbm": {
            "in_sample_auroc": gbm_is,
            "cv_auroc": gbm_cv,
            "feature_importance": feat_imp,
        },
        "logistic_scores": [float(s) for s in logit_scores],
        "gbm_scores": [float(s) for s in gbm_scores],
        "feature_names": names,
        "n": n,
        "n_used": n_used,
        "n_splits_effective": eff_splits,
    }


# ===========================================================================
# build_stable_used_table — noise floor (used ∩ across runs, element-anchored)
# ===========================================================================

def build_stable_used_table(
    current_run: Sequence[ScoreUsedRow],
    other_run: Sequence[ScoreUsedRow],
) -> list[ScoreUsedRow]:
    """STABLE-used label = used in BOTH runs, matched at the ``self_ref``
    element anchor (NOT raw chunk_index — a re-run may rebuild the merged
    index).  Labels are stamped onto the CURRENT run's rows (its chunk_index
    and ordering are preserved); a current-run chunk is stable-used iff it is
    used here AND some other-run chunk with the SAME self_ref is used there.

    Per pass:
        stable_used(chunk) = current.used(chunk)
                             AND self_ref(chunk) ∈ other_run_used_self_refs[pass]
    """
    # other-run USED self_refs, per pass.
    other_used_self: dict[str, set[str]] = {}
    for r in other_run:
        if r.used and r.self_ref:
            other_used_self.setdefault(r.pass_name, set()).add(r.self_ref)

    out: list[ScoreUsedRow] = []
    for r in current_run:
        stable = bool(r.used and r.self_ref in other_used_self.get(r.pass_name, set()))
        d = asdict(r)
        d["used"] = stable
        out.append(ScoreUsedRow(**d))
    return out


# ===========================================================================
# recall-vs-threshold on the best re-weighted score (REUSES phase1)
# ===========================================================================

def reweighted_recall_curve(
    rows: Sequence[ComponentRow],
    scores: Sequence[float],
    *,
    recall_floor: float = 1.0,
) -> dict[str, Any]:
    """Recall-vs-threshold for a re-weighted per-row score, pooled and per-pass.

    Mirrors Phase-1's ``recall_vs_threshold`` + ``_threshold_recommendation``
    (the same KEEP rule ``score >= tau``), so the re-weighted score's chunk-
    reduction-at-preserved-recall is directly comparable to the production
    final_score curve Phase-1 produced.
    """
    used = [r.used for r in rows]
    pooled_curve = recall_vs_threshold(list(scores), used)
    pooled_rec = p1._threshold_recommendation(pooled_curve, len(rows), recall_floor)

    by_pass: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_pass.setdefault(r.pass_name, []).append(i)

    per_pass: dict[str, Any] = {}
    for pn, idxs in by_pass.items():
        s = [scores[i] for i in idxs]
        u = [used[i] for i in idxs]
        curve = recall_vs_threshold(s, u)
        rec = p1._threshold_recommendation(curve, len(idxs), recall_floor)
        per_pass[pn] = {
            "n": len(idxs),
            "n_used": sum(1 for x in u if x),
            "recommendation": rec,
        }

    return {"pooled_recommendation": pooled_rec, "per_pass": per_pass}


# ===========================================================================
# Report dataclasses + serialization
# ===========================================================================

@dataclass
class ReweightReport:
    run_id: str
    doc_id: str
    n_rows: int
    n_used: int
    passes: list[str]
    per_component_auroc: dict[str, Any]
    fit: dict[str, Any]
    recall_curve: dict[str, Any]
    noise_floor: dict[str, Any] | None
    verdict: str
    auroc_sep_floor: float
    notes: list[str] = field(default_factory=list)


def _rec_to_jsonable(rec) -> dict[str, Any] | None:
    if rec is None:
        return None
    return asdict(rec)


def _curve_to_jsonable(curve: dict[str, Any]) -> dict[str, Any]:
    out = {
        "pooled_recommendation": _rec_to_jsonable(curve.get("pooled_recommendation")),
        "per_pass": {},
    }
    for pn, d in curve.get("per_pass", {}).items():
        out["per_pass"][pn] = {
            "n": d["n"], "n_used": d["n_used"],
            "recommendation": _rec_to_jsonable(d.get("recommendation")),
        }
    return out


def _write_component_csv(rows: Sequence[ComponentRow], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["pass_name", "chunk_index", "used", *COMPONENT_FIELDS, "self_ref"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            w.writerow({k: d[k] for k in fields})


# ===========================================================================
# Verdict
# ===========================================================================

def decide_verdict(
    fit: dict[str, Any],
    recall_curve: dict[str, Any],
    *,
    auroc_sep_floor: float = DEFAULT_AUROC_SEP_FLOOR,
    keeps_all_fraction: float = 0.95,
) -> tuple[str, list[str]]:
    """GO(Step B) / NO-GO(top_k) from the HONEST CV numbers.

    GO iff the BEST re-weighted CV AUROC clears the separation floor AND the
    best re-weighted score's recall-preserving threshold actually reduces the
    pool (keeps < keeps_all_fraction at recall_floor). Else NO-GO.

    In-sample numbers are explicitly NOT used for the verdict (overfitting).
    """
    notes: list[str] = []
    logit_cv = float(fit["logistic"]["cv_auroc"])
    gbm_cv = float(fit["gbm"]["cv_auroc"])
    best_cv = max(logit_cv, gbm_cv)

    rec = recall_curve.get("pooled_recommendation")
    threshold_reduces = rec is not None and rec.fraction_kept < keeps_all_fraction

    # overfitting-trap flag
    logit_is = float(fit["logistic"]["in_sample_auroc"])
    gbm_is = float(fit["gbm"]["in_sample_auroc"])
    if (logit_is - logit_cv) > 0.15 or (gbm_is - gbm_cv) > 0.15:
        notes.append(
            f"OVERFITTING TRAP: in-sample AUROC (logit {logit_is:.3f} / gbm "
            f"{gbm_is:.3f}) is much higher than CV (logit {logit_cv:.3f} / gbm "
            f"{gbm_cv:.3f}); the in-sample numbers are NOT trustworthy on this "
            f"small/wide sample — read the CV column."
        )

    if best_cv >= auroc_sep_floor and threshold_reduces:
        notes.append(
            f"Best re-weighted CV AUROC {best_cv:.3f} >= floor {auroc_sep_floor} "
            f"AND a recall-preserving threshold keeps only "
            f"{rec.fraction_kept*100:.0f}% of chunks → re-weighting is promising."
        )
        return "GO", notes

    if best_cv < auroc_sep_floor:
        notes.append(
            f"Best re-weighted CV AUROC {best_cv:.3f} < floor {auroc_sep_floor}: "
            f"even a learned (and nonlinear) recombination of the components does "
            f"not separate USED from UNUSED within this doc → the signal is not in "
            f"the router's component features."
        )
    if not threshold_reduces:
        kept = (rec.fraction_kept if rec is not None else 1.0)
        notes.append(
            f"No recall-preserving threshold on the best re-weighted score reduces "
            f"the pool (keeps {kept*100:.0f}% at the recall floor)."
        )
    notes.append("→ recommend tuned per-pass top_k (NO-GO for a re-weighted threshold).")
    return "NO-GO", notes


# ===========================================================================
# Markdown RESULT REPORT
# ===========================================================================

def _render_markdown(rep: ReweightReport, n_runs: int) -> str:
    L: list[str] = []
    L.append("# Phase-1b — re-weighted component separator probe\n")
    L.append(f"Run: `{rep.run_id}`  Doc: `{rep.doc_id}`  (samples: {n_runs})\n")
    L.append(f"Question: does a LEARNED re-weighting of the router's component "
             f"features separate USED chunks (lineage label) from UNUSED — where "
             f"Phase-1 showed NO single score does?\n")
    L.append(f"\nAUROC separation floor = {rep.auroc_sep_floor}; "
             f"N={rep.n_rows} rows, {rep.n_used} used, passes={rep.passes}\n")

    L.append(f"\n## BOTTOM LINE: **{rep.verdict}**"
             f"  {'(re-weighting promising → Step B)' if rep.verdict=='GO' else '(features lack the signal → tuned per-pass top_k)'}\n")
    for n in rep.notes:
        L.append(f"- {n}\n")

    # per-component AUROC
    L.append("\n## Per-component univariate AUROC (used vs unused)\n\n")
    pca = rep.per_component_auroc
    passes = sorted(pca["per_pass"].keys())
    L.append("| component | pooled | " + " | ".join(passes) + " |\n")
    L.append("|---|" + "---|" * (len(passes) + 1) + "\n")
    for comp in COMPONENT_FIELDS:
        cells = [f"{pca['pooled'][comp]:.3f}"]
        for pn in passes:
            cells.append(f"{pca['per_pass'][pn][comp]:.3f}")
        L.append(f"| {comp} | " + " | ".join(cells) + " |\n")
    sp = pca["support"]
    L.append(f"\n_support: pooled n={sp['pooled']['n']} used={sp['pooled']['n_used']}; "
             + "; ".join(f"{pn} n={sp['per_pass'][pn]['n']} used={sp['per_pass'][pn]['n_used']}"
                         for pn in passes) + "_\n")

    # re-weighted separator
    L.append("\n## Re-weighted separator: in-sample vs **CV** AUROC\n\n")
    L.append("| model | in-sample AUROC (ceiling, overfits) | **CV AUROC (honest)** |\n")
    L.append("|---|---|---|\n")
    fit = rep.fit
    L.append(f"| logistic (L2) | {fit['logistic']['in_sample_auroc']:.3f} | "
             f"**{fit['logistic']['cv_auroc']:.3f}** |\n")
    L.append(f"| GBM (nonlinear ceiling) | {fit['gbm']['in_sample_auroc']:.3f} | "
             f"**{fit['gbm']['cv_auroc']:.3f}** |\n")
    L.append(f"\n_CV = stratified {fit['n_splits_effective']}-fold within this doc; "
             f"{fit['n_used']}/{fit['n']} positives — small-sample, CV is the honest "
             f"estimate, in-sample is only a ceiling._\n")

    # learned logistic weights
    L.append("\n## Learned logistic weights (standardised; which components dominate)\n\n")
    L.append("| component | weight |\n|---|---|\n")
    w = fit["logistic"]["weights"]
    for comp in sorted(w, key=lambda k: -abs(w[k])):
        L.append(f"| {comp} | {w[comp]:+.3f} |\n")
    L.append(f"\n_intercept = {fit['logistic']['intercept']:+.3f}; pass-indicator "
             f"offsets omitted from this table (see JSON `all_weights`)._\n")
    if fit["gbm"]["feature_importance"]:
        L.append("\nGBM feature importance (nonlinear): " +
                 ", ".join(f"{k}={v:.2f}" for k, v in
                           sorted(fit["gbm"]["feature_importance"].items(),
                                  key=lambda kv: -kv[1])) + "\n")

    # recall vs threshold on re-weighted score
    L.append("\n## Recall-vs-threshold on the best re-weighted score "
             "(vs production final_score which kept ~all chunks)\n\n")
    rc = rep.recall_curve
    rec = rc.get("pooled_recommendation")
    if rec is not None:
        L.append("| scope | threshold | recall | frac_kept | chunks_kept |\n")
        L.append("|---|---|---|---|---|\n")
        L.append(f"| pooled | {rec.threshold:.4f} | {rec.recall_achieved:.2f} | "
                 f"{rec.fraction_kept:.2f} | {rec.chunks_kept} |\n")
        for pn in sorted(rc.get("per_pass", {})):
            d = rc["per_pass"][pn]
            r = d["recommendation"]
            if r is not None:
                L.append(f"| {pn} | {r.threshold:.4f} | {r.recall_achieved:.2f} | "
                         f"{r.fraction_kept:.2f} | {r.chunks_kept} |\n")
    else:
        L.append("_No recall-preserving threshold exists on the re-weighted score._\n")

    # noise floor
    L.append("\n## Noise floor (STABLE used label = used in BOTH runs)\n\n")
    if rep.noise_floor is None:
        L.append("_Not computed (second sample unavailable / unusable)._\n")
    elif rep.noise_floor.get("status") != "ok":
        L.append(f"_Not computed: {rep.noise_floor.get('reason','unknown')}_\n")
    else:
        nf = rep.noise_floor
        L.append(f"Second run: `{nf['other_run_id']}`. Stable-used "
                 f"{nf['n_stable_used']} of {nf['n_used_single']} single-run used "
                 f"(pooled).\n\n")
        L.append("| model | CV AUROC (single label) | CV AUROC (STABLE label) |\n")
        L.append("|---|---|---|\n")
        L.append(f"| logistic | {fit['logistic']['cv_auroc']:.3f} | "
                 f"{nf['fit']['logistic']['cv_auroc']:.3f} |\n")
        L.append(f"| GBM | {fit['gbm']['cv_auroc']:.3f} | "
                 f"{nf['fit']['gbm']['cv_auroc']:.3f} |\n")
        for n in nf.get("notes", []):
            L.append(f"- {n}\n")

    return "".join(L)


def _report_to_jsonable(rep: ReweightReport) -> dict[str, Any]:
    d = asdict(rep)
    # convert Recommendation dataclasses inside recall_curve
    d["recall_curve"] = _curve_to_jsonable(rep.recall_curve)
    if rep.noise_floor and rep.noise_floor.get("status") == "ok":
        d["noise_floor"]["recall_curve"] = _curve_to_jsonable(
            rep.noise_floor.get("recall_curve", {})
        )
    return d


# ===========================================================================
# main — the offline pipeline on a real run
# ===========================================================================

def _analyzable_passes(
    pass_outputs: Mapping[str, Mapping[str, Any]],
    *,
    routed_only: bool,
    routed: Sequence[str],
) -> list[str]:
    """Passes to analyze = those with a KNOWN (non-empty-provenance) USED set.

    Phase-1's published baseline analysed FOUR passes (the two field_group
    routed passes PLUS the two identity passes that carry provenance), and the
    brief's "~60 chunks/pass × 4 passes, ~12-16 positives/pass" refers to that
    same set.  A pass with empty provenance (e.g. ``system_links`` here) has an
    UNKNOWN USED set (value-fill) and is excluded so it cannot dilute the fit.

    ``routed_only`` restricts to the field_group passes (the strict routed
    scope) for an A/B against the wider 4-pass view.
    """
    def _has_known_used(po: Mapping[str, Any]) -> bool:
        return bool(po.get("provenance") or po.get("field_provenance"))

    if routed_only:
        return [p for p in routed if p in pass_outputs and _has_known_used(pass_outputs[p])]
    return sorted(pn for pn, po in pass_outputs.items() if _has_known_used(po))


def _build_labeled_components(
    run_id: str, analyze_passes: list[str]
) -> tuple[list[ComponentRow], list[ScoreUsedRow], dict[str, Any]]:
    """Fetch chunks + pass outputs for one run; return (labeled ComponentRows,
    the ScoreUsedRow lineage table, pass_outputs)."""
    chunks = p1._fetch_chunks(run_id)
    pass_outputs = p1._fetch_pass_outputs(run_id)
    chunks_by_pass = {pn: chunks for pn in analyze_passes if pn in pass_outputs}

    comp_rows = build_component_table(chunks_by_pass=chunks_by_pass)
    used_rows = build_used_table(pass_outputs, chunks_by_pass)
    labeled = label_component_table(comp_rows, used_rows)
    return labeled, used_rows, pass_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--doc-id", default="")
    ap.add_argument("--noise-floor-run-id", default="",
                    help="Second current-code full-doc run of the SAME doc; "
                         "USED ∩ across both (element-anchored) = STABLE label.")
    ap.add_argument("--bundle-key",
                    default=os.environ.get("PHASE1_BUNDLE_KEY", "air_defense_v3_merged_v1"))
    ap.add_argument("--recall-floor", type=float, default=1.0)
    ap.add_argument("--auroc-floor", type=float, default=DEFAULT_AUROC_SEP_FLOOR)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--routed-only", action="store_true",
                    help="Analyze only the field_group routed passes "
                         "(default: all passes with a known USED set — the "
                         "4-pass scope of the Phase-1 baseline).")
    ap.add_argument("--out-dir", default="reports/collection")
    args = ap.parse_args()

    os.environ.setdefault("PHASE1_BUNDLE_KEY", args.bundle_key)
    run8 = args.run_id.split("-")[0]

    routed = p1._default_routed_passes(args.bundle_key)
    print(f"[phase1b] routed (field_group) passes: {routed}")

    # determine analyzable passes from the run's provenance (known USED set).
    print(f"[phase1b] fetching pass outputs for run {args.run_id} ...")
    pass_outputs_probe = p1._fetch_pass_outputs(args.run_id)
    analyze = _analyzable_passes(
        pass_outputs_probe, routed_only=args.routed_only, routed=routed)
    print(f"[phase1b] analyzing passes (known USED set): {analyze}")

    print(f"[phase1b] building labeled component table for run {args.run_id} ...")
    labeled, used_rows, pass_outputs = _build_labeled_components(args.run_id, analyze)
    print(f"[phase1b] component rows={len(labeled)} "
          f"used={sum(1 for r in labeled if r.used)}")

    notes: list[str] = []

    print("[phase1b] per-component univariate AUROC ...")
    pca = per_component_auroc(labeled)

    print("[phase1b] fitting re-weighted separators (logistic + GBM; CV) ...")
    fit = fit_reweighted_separators(labeled, n_splits=args.cv_splits, seed=args.seed)
    print(f"   logistic  in-sample={fit['logistic']['in_sample_auroc']:.3f} "
          f"CV={fit['logistic']['cv_auroc']:.3f}")
    print(f"   gbm       in-sample={fit['gbm']['in_sample_auroc']:.3f} "
          f"CV={fit['gbm']['cv_auroc']:.3f}")

    # best re-weighted score for the recall sweep = the model with higher CV.
    if fit["gbm"]["cv_auroc"] > fit["logistic"]["cv_auroc"]:
        best_scores = fit["gbm_scores"]
        notes.append("Recall sweep uses the GBM re-weighted score (higher CV AUROC).")
    else:
        best_scores = fit["logistic_scores"]
        notes.append("Recall sweep uses the logistic re-weighted score.")
    recall_curve = reweighted_recall_curve(
        labeled, best_scores, recall_floor=args.recall_floor)

    # ---- noise floor ----
    noise_floor: dict[str, Any] | None = None
    if args.noise_floor_run_id:
        print(f"[phase1b] noise floor: second run {args.noise_floor_run_id} ...")
        try:
            other_labeled, other_used, other_po = _build_labeled_components(
                args.noise_floor_run_id, analyze)
            usable = any(
                (po.get("provenance") or po.get("field_provenance"))
                for po in other_po.values()
            )
            if not usable:
                noise_floor = {
                    "status": "unusable",
                    "other_run_id": args.noise_floor_run_id,
                    "reason": "second run has no usable provenance in any routed pass",
                }
            else:
                stable_used_rows = build_stable_used_table(used_rows, other_used)
                stable_labeled = label_component_table(labeled, stable_used_rows)
                n_stable = sum(1 for r in stable_labeled if r.used)
                n_single = sum(1 for r in labeled if r.used)
                stable_fit = fit_reweighted_separators(
                    stable_labeled, n_splits=args.cv_splits, seed=args.seed)
                stable_pca = per_component_auroc(stable_labeled)
                if stable_fit["gbm"]["cv_auroc"] > stable_fit["logistic"]["cv_auroc"]:
                    stable_scores = stable_fit["gbm_scores"]
                else:
                    stable_scores = stable_fit["logistic_scores"]
                stable_curve = reweighted_recall_curve(
                    stable_labeled, stable_scores, recall_floor=args.recall_floor)
                nf_notes: list[str] = []
                d_logit = stable_fit["logistic"]["cv_auroc"] - fit["logistic"]["cv_auroc"]
                d_gbm = stable_fit["gbm"]["cv_auroc"] - fit["gbm"]["cv_auroc"]
                nf_notes.append(
                    f"Denoising (stable label) shifts CV AUROC by "
                    f"{d_logit:+.3f} (logistic) / {d_gbm:+.3f} (GBM)."
                )
                noise_floor = {
                    "status": "ok",
                    "other_run_id": args.noise_floor_run_id,
                    "n_stable_used": n_stable,
                    "n_used_single": n_single,
                    "fit": stable_fit,
                    "per_component_auroc": stable_pca,
                    "recall_curve": stable_curve,
                    "notes": nf_notes,
                }
        except Exception as exc:  # pragma: no cover - infra failure path
            noise_floor = {
                "status": "error",
                "other_run_id": args.noise_floor_run_id,
                "reason": f"{type(exc).__name__}: {exc}",
            }
    else:
        noise_floor = None

    verdict, vnotes = decide_verdict(
        fit, recall_curve, auroc_sep_floor=args.auroc_floor)
    notes.extend(vnotes)

    rep = ReweightReport(
        run_id=args.run_id,
        doc_id=args.doc_id,
        n_rows=len(labeled),
        n_used=sum(1 for r in labeled if r.used),
        passes=sorted({r.pass_name for r in labeled}),
        per_component_auroc=pca,
        fit=fit,
        recall_curve=recall_curve,
        noise_floor=noise_floor,
        verdict=verdict,
        auroc_sep_floor=args.auroc_floor,
        notes=notes,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"phase1b_components_{run8}.csv")
    json_path = os.path.join(args.out_dir, f"phase1b_reweight_{run8}.json")
    md_path = os.path.join(args.out_dir, f"phase1b_reweight_{run8}.md")

    _write_component_csv(labeled, csv_path)
    with open(json_path, "w") as f:
        json.dump(_report_to_jsonable(rep), f, indent=2, default=str)
    n_runs = 2 if (noise_floor and noise_floor.get("status") == "ok") else 1
    with open(md_path, "w") as f:
        f.write(_render_markdown(rep, n_runs=n_runs))

    print(f"\n[phase1b] wrote {csv_path}")
    print(f"[phase1b] wrote {json_path}")
    print(f"[phase1b] wrote {md_path}")
    print(f"\n========== VERDICT: {rep.verdict} ==========")
    for n in rep.notes:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
