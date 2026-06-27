"""Phase-2: does the C8 identity-anchor signal SEPARATE USED from UNUSED chunks?

(OFFLINE, deterministic, read-only — no new run, no router/LLM changes.)

Background
----------
Phase-1 (``scripts/phase1_score_used_separation.py``) and Phase-1b
(``scripts/phase1b_reweight_separator.py``) established that NO router score and
NO learned re-weighting of the C5 components separates the chunks a fact was
traced TO (USED, via lineage) from the rest on SA-2 — best CV AUROC ~0.60. The
diagnosis: the lexical channel (``alias_hits``) barely fires (≈5% of chunks),
because the lexical vocabulary is spec-parameter LABELS; the entity NAMES
("SNR-75", "Fan Song", "Spoon Rest", …) never enter the matcher.

Commit c66094b wired the C8 identity-anchor channel: at field_group dispatch the
worker harvests the committed identity-entity NAMES from the persisted identity
passes (``_collect_committed_identity_anchors``) and hands them to the router,
which runs an anchor lexical sub-channel over the candidate pool
(``extraction_chunk_search``: ``anchor_hit_count = sum(1 for a in
normalised_anchors if a in haystack)``; folded into ``mc.alias_hits``).

The question this module answers, OFFLINE on the existing SA-2 chunks: if those
anchor names are fed into the SAME normalised-substring matcher, does the
resulting ``anchor_alias_hits`` now carry signal that SEPARATES USED from UNUSED
PER PASS — and does TYPE-MATCHING the anchors to each pass's target entity type
discriminate even when "contains ANY committed entity" (near-binary) does not?

Two anchor variants
-------------------
(a) ALL anchors — every committed identity NAME (radar + missile mixed). This is
    the CURRENT wiring (``_collect_committed_identity_anchors`` is exactly this).
(b) TYPE-MATCHED per pass — for each field_group pass, ONLY the anchors whose
    entity type is in the pass's ``primary_entity_types`` (radar_power_rf →
    RADAR_SYSTEM names; missile_kinematics → MISSILE_SYSTEM names). Driven off
    bundle metadata + the identity walk's per-ref ``entity_type`` — NO hardcoded
    type names.

Method (everything REUSED — nothing re-derived)
-----------------------------------------------
  * NORMALISATION + MATCHING — ``anchor_alias_hits`` reuses
    ``extraction_lexical_search._nfc`` (NFC) + casefold + substring, the EXACT
    steps the production C8 sub-channel uses.
  * ANCHOR HARVEST — variant (a) calls the production
    ``_collect_committed_identity_anchors`` verbatim; variant (b) reuses the
    SAME pipeline helpers (``load_pass_output`` → ``_parse_pass_response`` →
    ``_build_pre_merge_walk_summary`` → ``_extend_upstream_refs``) and captures
    each ref's ``entity_type`` alongside its name (the function itself returns
    only names, so the typed walk is the minimal sibling).
  * BASELINE FEATURES + USED LABELS — Phase-1b's ``build_component_table`` /
    ``label_component_table`` and Phase-1's ``build_used_table``.
  * AUROC / CV / recall — Phase-1's exact-tie ``auroc`` + ``recall_vs_threshold``
    + ``_threshold_recommendation``; Phase-1b's ``_cv_auroc`` / ``_in_sample_auroc``
    (estimator-generic) for the logistic + GBM re-weighted fit.

Import-safe: no DB / model work at import time. The pure functions
(``anchor_alias_hits``, ``pass_typed_anchor_names``, ``build_anchor_feature_table``,
``attach_anchor_features``, ``feature_matrix_with_anchors``, ``per_feature_auroc``,
``fit_with_and_without_anchors``) are unit-tested with synthetic fixtures; the DB
/ embedding / rerank I/O is the Phase-1 wrappers + the pipeline anchor harvest,
invoked only by ``main()``.

Usage (read-only on the live DBs)::

    PHASE1_DATABASE_URL=postgresql+psycopg2://eip:eip_secret@localhost:5437/eip \
    python scripts/phase2_anchor_alias_separation.py \
        --run-id 5da1a210-0322-401e-a2f2-8a7cab68ca00 \
        --doc-id ddaa9e36-2854-47c3-bc94-ff38d531dafd

Outputs (under reports/collection/):
    phase2_anchor_features_<run8>.csv  — (pass, chunk, components, anchor feats, used)
    phase2_anchor_<run8>.json          — full analysis report
    phase2_anchor_<run8>.md            — human-readable RESULT REPORT + verdict
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

# Reuse Phase-1 + Phase-1b machinery verbatim — nothing is reimplemented.
from scripts import phase1_score_used_separation as p1
from scripts import phase1b_reweight_separator as p1b
from scripts.phase1_score_used_separation import (
    ScoreUsedRow,
    auroc,
    build_used_table,
    recall_vs_threshold,
)
from scripts.phase1b_reweight_separator import (
    COMPONENT_FIELDS,
    ComponentRow,
    build_component_table,
    label_component_table,
)

# Production normalisation — the SAME function the C8 sub-channel uses.
from app.services.extraction_lexical_search import _nfc


# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------
# The two C8 anchor features, appended to the 7 baseline C5/cosine components.
# Order is the canonical anchor-column order.
ANCHOR_FEATURE_FIELDS: tuple[str, ...] = (
    "anchor_alias_hits",        # variant (a): ALL committed identity names
    "anchor_alias_hits_typed",  # variant (b): only this pass's entity-type names
)

# The full feature-matrix column order for the WITH-anchors fit.
ALL_FEATURE_FIELDS: tuple[str, ...] = tuple(COMPONENT_FIELDS) + ANCHOR_FEATURE_FIELDS

# AUROC floor a separator must clear to count as "the signal exists" — the same
# 0.75 floor Phase-1 / 1b used.
DEFAULT_AUROC_SEP_FLOOR = p1.DEFAULT_AUROC_SEP_FLOOR  # 0.75


@dataclass(frozen=True)
class TypedAnchor:
    """A committed identity NAME tagged with the entity TYPE that produced it."""
    name: str
    entity_type: str


@dataclass
class AnchorRow:
    """Per (pass, chunk) anchor-feature row (both variants)."""
    pass_name: str
    chunk_index: int
    anchor_alias_hits: int = 0          # variant (a) ALL anchors
    anchor_alias_hits_typed: int = 0    # variant (b) TYPE-MATCHED to the pass
    self_ref: str = ""


# ===========================================================================
# anchor_alias_hits — the matcher (mirrors the runtime C8 sub-channel exactly)
# ===========================================================================

def _normalise(text: str) -> str:
    """NFC + casefold — the EXACT normalisation the production matcher applies
    (``extraction_lexical_search._nfc`` then ``str.casefold``)."""
    return _nfc(text or "").casefold()


def anchor_alias_hits(chunk_text: str, anchors: Sequence[str]) -> int:
    """Number of DISTINCT anchor names whose normalised form is a substring of
    the normalised chunk text.

    Mirrors the runtime C8 sub-channel byte-for-byte::

        normalised_anchors = [NFC(a).casefold() for a in anchors if a]
        haystack           = NFC(chunk_text).casefold()
        anchor_hit_count   = sum(1 for a in normalised_anchors if a in haystack)

    So a text containing "SNR-75 Fan Song" with anchors
    ["SNR-75","Fan Song","V-75"] → 2; an absent anchor contributes 0; a repeated
    anchor still counts once (indicator per anchor string, not occurrences).
    Blank/whitespace anchors are skipped (they would otherwise match every text).
    """
    haystack = _normalise(chunk_text)
    if not haystack:
        return 0
    seen: set[str] = set()
    hits = 0
    for a in anchors or ():
        if not a or not a.strip():
            continue
        na = _normalise(a)
        if not na or na in seen:
            continue
        seen.add(na)
        if na in haystack:
            hits += 1
    return hits


# ===========================================================================
# pass_typed_anchor_names — variant (b) type matching (bundle-driven)
# ===========================================================================

def pass_typed_anchor_names(
    pass_def: Any, typed_anchors: Sequence[TypedAnchor]
) -> list[str]:
    """Names of the anchors whose ``entity_type`` is in this pass's
    ``primary_entity_types`` (the pass's target entity type[s]).

    Generalised: the relevant type set comes straight off ``pass_def`` (manifest
    metadata) — no literal type names. Deduplicates names, preserving first-seen
    order (deterministic).
    """
    target_types = set(getattr(pass_def, "primary_entity_types", None) or ())
    if not target_types:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for ta in typed_anchors:
        if ta.entity_type in target_types and ta.name and ta.name not in seen:
            seen.add(ta.name)
            out.append(ta.name)
    return out


# ===========================================================================
# build_anchor_feature_table — per (pass, chunk) AnchorRow
# ===========================================================================

def build_anchor_feature_table(
    *,
    chunks_by_pass: Mapping[str, Sequence[Mapping[str, Any]]],
    all_anchors: Sequence[str],
    typed_anchors: Sequence[TypedAnchor],
    pass_defs: Mapping[str, Any],
) -> list[AnchorRow]:
    """Per (pass, chunk) AnchorRow with both anchor variants.

    ``anchor_alias_hits``        = count of ALL committed identity names present
                                   (variant a; pass-INDEPENDENT — same anchor set
                                   for every pass, exactly the current wiring).
    ``anchor_alias_hits_typed``  = count of only this pass's entity-type names
                                   present (variant b; per-pass anchor set).

    Both reuse ``anchor_alias_hits`` (the production normalisation + matcher).
    DB I/O is none — chunks + anchors are passed in.
    """
    out: list[AnchorRow] = []
    # Pre-resolve the type-matched name list per pass once.
    typed_by_pass: dict[str, list[str]] = {}
    for pass_name in chunks_by_pass:
        pdf = pass_defs.get(pass_name)
        typed_by_pass[pass_name] = (
            pass_typed_anchor_names(pdf, typed_anchors) if pdf is not None else []
        )

    for pass_name, chunks in chunks_by_pass.items():
        typed_names = typed_by_pass.get(pass_name, [])
        for c in chunks:
            text = c.get("chunk_text") or ""
            out.append(AnchorRow(
                pass_name=pass_name,
                chunk_index=int(c["chunk_index"]),
                anchor_alias_hits=anchor_alias_hits(text, all_anchors),
                anchor_alias_hits_typed=anchor_alias_hits(text, typed_names),
                self_ref=str(c.get("self_ref", "")),
            ))
    return out


# ===========================================================================
# attach_anchor_features — merge anchor features onto the labeled components
# ===========================================================================

def attach_anchor_features(
    comp_rows: Sequence[ComponentRow],
    anchor_rows: Sequence[AnchorRow],
) -> list[dict[str, Any]]:
    """Return per (pass, chunk) feature dicts = every C5 component + cosine +
    the two anchor features + the ``used`` label, joined on (pass, chunk_index).

    A component row with no matching anchor row gets anchor features = 0.0 (the
    clean no-op — that pass/chunk simply had no anchor hit).
    """
    anchor_idx = {(a.pass_name, a.chunk_index): a for a in anchor_rows}
    out: list[dict[str, Any]] = []
    for cr in comp_rows:
        d: dict[str, Any] = {
            "pass_name": cr.pass_name,
            "chunk_index": cr.chunk_index,
            "used": bool(cr.used),
            "self_ref": cr.self_ref,
        }
        for f in COMPONENT_FIELDS:
            d[f] = float(getattr(cr, f))
        ar = anchor_idx.get((cr.pass_name, cr.chunk_index))
        d["anchor_alias_hits"] = float(ar.anchor_alias_hits if ar else 0.0)
        d["anchor_alias_hits_typed"] = float(ar.anchor_alias_hits_typed if ar else 0.0)
        out.append(d)
    return out


# ===========================================================================
# feature_matrix_with_anchors — (X, y, names); WITHOUT = strict subset of WITH
# ===========================================================================

def feature_matrix_with_anchors(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_anchors: bool,
    add_pass_indicator: bool = True,
) -> tuple[Any, Any, list[str]]:
    """Feature matrix from the attached feature dicts.

    Columns are the 7 baseline components, then (when ``include_anchors``) the 2
    anchor features, then (when ``add_pass_indicator`` and >1 pass) drop-first
    one-hot pass indicators. The WITHOUT-anchors column set is therefore a STRICT
    subset of the WITH-anchors set — making the CV-AUROC delta apples-to-apples
    (same base features + same pass offsets; the only difference is the anchors).
    """
    import numpy as np

    base = list(COMPONENT_FIELDS)
    if include_anchors:
        base = base + list(ANCHOR_FEATURE_FIELDS)

    X_base = np.asarray(
        [[float(r.get(f, 0.0)) for f in base] for r in rows], dtype=np.float64
    )
    y = np.asarray([1 if r.get("used") else 0 for r in rows], dtype=np.int64)
    names = list(base)

    if add_pass_indicator:
        passes = sorted({str(r["pass_name"]) for r in rows})
        if len(passes) > 1:
            keep = passes[1:]  # drop-first
            ind = np.asarray(
                [[1.0 if str(r["pass_name"]) == p else 0.0 for p in keep] for r in rows],
                dtype=np.float64,
            )
            X = np.hstack([X_base, ind])
            names = list(base) + [f"pass={p}" for p in keep]
            return X, y, names
    return X_base, y, names


# ===========================================================================
# per_feature_auroc — generic univariate AUROC + firing rate over named feats
# ===========================================================================

def per_feature_auroc(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Univariate AUROC(used, feature) per named feature, pooled + per-pass, plus
    the FIRING RATE (fraction of rows with feature > 0) pooled + per-pass.

    Reuses Phase-1's exact-tie ``auroc`` (matches sklearn.roc_auc_score). A
    constant feature yields AUROC 0.5 by construction.
    """
    feats = list(feature_names)
    used = [bool(r.get("used")) for r in rows]

    def _auroc_for(rs, f):
        return auroc([float(r.get(f, 0.0)) for r in rs], [bool(r.get("used")) for r in rs])

    def _fire_for(rs, f):
        n = len(rs)
        return (sum(1 for r in rs if float(r.get(f, 0.0)) > 0.0) / n) if n else 0.0

    pooled = {f: _auroc_for(rows, f) for f in feats}
    pooled_fire = {f: _fire_for(rows, f) for f in feats}

    by_pass: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_pass.setdefault(str(r["pass_name"]), []).append(r)

    per_pass: dict[str, dict[str, float]] = {}
    per_pass_fire: dict[str, dict[str, float]] = {}
    support_pp: dict[str, dict[str, int]] = {}
    for pn, rs in by_pass.items():
        per_pass[pn] = {f: _auroc_for(rs, f) for f in feats}
        per_pass_fire[pn] = {f: _fire_for(rs, f) for f in feats}
        u = [bool(r.get("used")) for r in rs]
        support_pp[pn] = {"n": len(rs), "n_used": sum(1 for x in u if x)}

    return {
        "pooled": pooled,
        "per_pass": per_pass,
        "firing_rate": {"pooled": pooled_fire, "per_pass": per_pass_fire},
        "support": {
            "pooled": {"n": len(rows), "n_used": sum(1 for x in used if x)},
            "per_pass": support_pp,
        },
    }


# ===========================================================================
# fit_with_and_without_anchors — THE probe (does adding anchors help, honestly?)
# ===========================================================================

def _fit_side(
    X, y, *, n_splits: int, seed: int, names: list[str]
) -> dict[str, Any]:
    """Fit logistic (L2) + GBM on (X, y); return in-sample + CV AUROC, learned
    logistic weights (mapped back to feature names), GBM importances, and the
    best-side per-row score (the model with higher CV AUROC). Reuses Phase-1b's
    estimator-generic ``_cv_auroc`` / ``_in_sample_auroc``."""
    from sklearn.base import clone as _clone
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # logistic (L2) — same spec as Phase-1b (standardised features, liblinear).
    logit = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, solver="liblinear", max_iter=2000, random_state=seed,
        )),
    ])
    logit_cv = p1b._cv_auroc(logit, X, y, n_splits=n_splits, seed=seed)
    logit_is_est = _clone(logit)
    logit_is, logit_scores = p1b._in_sample_auroc(logit_is_est, X, y)
    clf = logit_is_est.named_steps["clf"]
    coefs = clf.coef_.ravel().tolist()
    intercept = float(clf.intercept_.ravel()[0])
    all_weights = {names[i]: float(coefs[i]) for i in range(len(names))}
    # headline "which FEATURE dominates" view excludes pass indicators
    comp_weights = {k: v for k, v in all_weights.items() if not k.startswith("pass=")}

    # GBM (nonlinear ceiling) — same spec as Phase-1b.
    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=1.0, random_state=seed,
    )
    gbm_cv = p1b._cv_auroc(gbm, X, y, n_splits=n_splits, seed=seed)
    gbm_is_est = GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=1.0, random_state=seed,
    )
    gbm_is, gbm_scores = p1b._in_sample_auroc(gbm_is_est, X, y)
    importances = getattr(gbm_is_est, "feature_importances_", None)
    feat_imp = (
        {names[i]: float(importances[i]) for i in range(len(names))
         if not names[i].startswith("pass=")}
        if importances is not None else {}
    )

    best_cv = max(logit_cv, gbm_cv)
    best_scores = gbm_scores if gbm_cv > logit_cv else logit_scores

    return {
        "logistic_in_sample_auroc": logit_is,
        "logistic_cv_auroc": logit_cv,
        "logistic_weights": comp_weights,
        "logistic_all_weights": all_weights,
        "logistic_intercept": intercept,
        "gbm_in_sample_auroc": gbm_is,
        "gbm_cv_auroc": gbm_cv,
        "gbm_feature_importance": feat_imp,
        "best_cv_auroc": best_cv,
        "best_model": "gbm" if gbm_cv > logit_cv else "logistic",
        "best_scores": [float(s) for s in best_scores],
        "feature_names": names,
    }


def fit_with_and_without_anchors(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_splits: int = 5,
    seed: int = 0,
    add_pass_indicator: bool = True,
) -> dict[str, Any]:
    """Fit the re-weighted separators WITH and WITHOUT the C8 anchor features and
    report both sides + the CV-AUROC delta.

    WITHOUT = the 7 baseline components (+ pass offsets) — the Phase-1b
    configuration (~0.60 CV on SA-2). WITH = the same plus the two anchor
    features. The ``delta_best_cv_auroc`` is the HONEST measure of whether the C8
    anchors materially improve separation; in-sample numbers are a ceiling only.
    """
    X_with, y_with, names_with = feature_matrix_with_anchors(
        rows, include_anchors=True, add_pass_indicator=add_pass_indicator)
    X_wo, y_wo, names_wo = feature_matrix_with_anchors(
        rows, include_anchors=False, add_pass_indicator=add_pass_indicator)

    with_fit = _fit_side(X_with, y_with, n_splits=n_splits, seed=seed, names=names_with)
    without_fit = _fit_side(X_wo, y_wo, n_splits=n_splits, seed=seed, names=names_wo)

    n = len(rows)
    n_used = int(sum(1 for r in rows if r.get("used")))
    n_pos, n_neg = n_used, n - n_used
    eff_splits = max(2, min(n_splits, n_pos, n_neg)) if (n_pos >= 2 and n_neg >= 2) else 0

    return {
        "with_anchors": with_fit,
        "without_anchors": without_fit,
        "delta_best_cv_auroc": with_fit["best_cv_auroc"] - without_fit["best_cv_auroc"],
        "delta_logistic_cv_auroc": (
            with_fit["logistic_cv_auroc"] - without_fit["logistic_cv_auroc"]),
        "delta_gbm_cv_auroc": with_fit["gbm_cv_auroc"] - without_fit["gbm_cv_auroc"],
        "n": n,
        "n_used": n_used,
        "n_splits_effective": eff_splits,
    }


# ===========================================================================
# recall-vs-threshold on the best WITH-anchors score (REUSES phase1)
# ===========================================================================

def anchor_recall_curve(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    recall_floor: float = 1.0,
) -> dict[str, Any]:
    """Recall-vs-threshold for a per-row score, pooled + per-pass — the SAME KEEP
    rule (``score >= tau``) and recommendation as Phase-1, so the WITH-anchors
    chunk-reduction-at-preserved-recall is directly comparable to the Phase-1b
    re-weighted curve."""
    used = [bool(r.get("used")) for r in rows]
    pooled_curve = recall_vs_threshold(list(scores), used)
    pooled_rec = p1._threshold_recommendation(pooled_curve, len(rows), recall_floor)

    by_pass: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_pass.setdefault(str(r["pass_name"]), []).append(i)

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
# Verdict (i / ii / iii) — honest, CV-driven
# ===========================================================================

def decide_verdict(
    feat_auroc: dict[str, Any],
    fit: dict[str, Any],
    recall_curve: dict[str, Any],
    *,
    auroc_sep_floor: float = DEFAULT_AUROC_SEP_FLOOR,
    keeps_all_fraction: float = 0.95,
    fire_floor: float = 0.05,
) -> tuple[str, list[str]]:
    """Map the numbers to the brief's trichotomy:

      (i)   IMPROVE   — WITH-anchors best CV AUROC clears the floor AND beats
                        WITHOUT by a real margin AND a recall-preserving threshold
                        reduces the pool → the fix works; proceed to a threshold.
      (ii)  FIRE_NOSEP — anchors FIRE (pooled firing rate of variant a above the
                        floor) but CV does not separate (near-binary single-system;
                        even type-matched) → needs a HETEROGENEOUS doc.
      (iii) NOFIRE    — anchors do not fire as expected (firing rate below floor)
                        → investigate the wiring / harvest.

    In-sample numbers are NEVER used for the verdict (overfitting).
    """
    notes: list[str] = []
    fire_all = float(feat_auroc["firing_rate"]["pooled"].get("anchor_alias_hits", 0.0))
    fire_typed = float(feat_auroc["firing_rate"]["pooled"].get("anchor_alias_hits_typed", 0.0))

    with_best_cv = float(fit["with_anchors"]["best_cv_auroc"])
    without_best_cv = float(fit["without_anchors"]["best_cv_auroc"])
    delta = float(fit["delta_best_cv_auroc"])

    rec = recall_curve.get("pooled_recommendation")
    threshold_reduces = rec is not None and rec.fraction_kept < keeps_all_fraction

    # overfitting-trap flag (reported regardless of verdict)
    with_logit_is = float(fit["with_anchors"]["logistic_in_sample_auroc"])
    with_gbm_is = float(fit["with_anchors"]["gbm_in_sample_auroc"])
    with_logit_cv = float(fit["with_anchors"]["logistic_cv_auroc"])
    with_gbm_cv = float(fit["with_anchors"]["gbm_cv_auroc"])
    if (with_logit_is - with_logit_cv) > 0.15 or (with_gbm_is - with_gbm_cv) > 0.15:
        notes.append(
            f"OVERFITTING TRAP: WITH-anchors in-sample AUROC (logit "
            f"{with_logit_is:.3f} / gbm {with_gbm_is:.3f}) >> CV (logit "
            f"{with_logit_cv:.3f} / gbm {with_gbm_cv:.3f}); read the CV column."
        )

    # (iii) the anchors don't even fire as expected.
    if fire_all < fire_floor:
        notes.append(
            f"(iii) C8 anchors barely FIRE: variant-(a) pooled firing rate "
            f"{fire_all*100:.1f}% < floor {fire_floor*100:.0f}% — the harvest or "
            f"wiring is not delivering names into the matcher; investigate before "
            f"reading separation."
        )
        return "NOFIRE", notes

    notes.append(
        f"C8 anchors FIRE: variant-(a) pooled firing rate {fire_all*100:.1f}%, "
        f"variant-(b) type-matched {fire_typed*100:.1f}% "
        f"(Phase-1b baseline lexical_norm fired ~5%)."
    )

    # (i) materially improves AND a threshold buys a reduction.
    if (with_best_cv >= auroc_sep_floor and delta > 0.05 and threshold_reduces):
        notes.append(
            f"(i) WITH-anchors best CV AUROC {with_best_cv:.3f} >= floor "
            f"{auroc_sep_floor} and beats WITHOUT ({without_best_cv:.3f}) by "
            f"{delta:+.3f}, and a recall-preserving threshold keeps only "
            f"{rec.fraction_kept*100:.0f}% of chunks → the C8 fix materially "
            f"improves separation on SA-2; proceed to a threshold."
        )
        return "IMPROVE", notes

    # (ii) fires but does not separate.
    notes.append(
        f"(ii) C8 anchors FIRE but do NOT separate on SA-2: WITH-anchors best "
        f"CV AUROC {with_best_cv:.3f} (WITHOUT {without_best_cv:.3f}, delta "
        f"{delta:+.3f}) < floor {auroc_sep_floor}"
        + ("" if threshold_reduces else
           f"; no recall-preserving threshold reduces the pool "
           f"(keeps {(rec.fraction_kept if rec else 1.0)*100:.0f}%)")
        + ". SA-2 is a near-binary single-system doc — almost every chunk that "
        "names the (one) radar/missile system contains a committed anchor, so "
        "'contains an anchor' is nearly constant across USED and UNUSED, and "
        "even TYPE-MATCHING cannot create discrimination the doc does not have. "
        "Re-test on a HETEROGENEOUS multi-system doc (e.g. EWIRDB / a multi-"
        "radar source) where type-matched anchors can actually distinguish "
        "which pass each chunk belongs to."
    )
    return "FIRE_NOSEP", notes


# ===========================================================================
# Report dataclasses + serialization
# ===========================================================================

@dataclass
class AnchorReport:
    run_id: str
    doc_id: str
    bundle_key: str
    n_rows: int
    n_used: int
    passes: list[str]
    n_anchors_all: int
    n_anchors_typed_by_type: dict[str, int]
    per_feature_auroc: dict[str, Any]
    baseline_component_auroc: dict[str, Any]   # phase1b per-component AUROC (context)
    fit: dict[str, Any]
    recall_curve: dict[str, Any]
    verdict: str
    auroc_sep_floor: float
    notes: list[str] = field(default_factory=list)


def _rec_to_jsonable(rec) -> dict[str, Any] | None:
    return None if rec is None else asdict(rec)


def _curve_to_jsonable(curve: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pooled_recommendation": _rec_to_jsonable(curve.get("pooled_recommendation")),
        "per_pass": {},
    }
    for pn, d in curve.get("per_pass", {}).items():
        out["per_pass"][pn] = {
            "n": d["n"], "n_used": d["n_used"],
            "recommendation": _rec_to_jsonable(d.get("recommendation")),
        }
    return out


def _report_to_jsonable(rep: AnchorReport) -> dict[str, Any]:
    d = asdict(rep)
    d["recall_curve"] = _curve_to_jsonable(rep.recall_curve)
    return d


def _write_feature_csv(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = (
        ["pass_name", "chunk_index", "used"]
        + list(COMPONENT_FIELDS) + list(ANCHOR_FEATURE_FIELDS) + ["self_ref"]
    )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


# ===========================================================================
# Markdown RESULT REPORT
# ===========================================================================

def _render_markdown(rep: AnchorReport) -> str:
    L: list[str] = []
    L.append("# Phase-2 — C8 identity-anchor USED/UNUSED separation probe\n")
    L.append(f"Run: `{rep.run_id}`  Doc: `{rep.doc_id}`  Bundle: `{rep.bundle_key}`\n")
    L.append("Question: does the wired C8 identity-anchor channel — anchor NAMES "
             "fed into the production NFC+casefold substring matcher — make "
             "`anchor_alias_hits` separate USED from UNUSED chunks PER PASS, where "
             "Phase-1/1b showed the spec-label lexical channel does not "
             "(CV AUROC ~0.60)?\n")
    L.append(f"\nAUROC separation floor = {rep.auroc_sep_floor}; "
             f"N={rep.n_rows} rows, {rep.n_used} used, passes={rep.passes}\n")
    L.append(f"\nAnchors harvested: variant-(a) ALL = **{rep.n_anchors_all}** names; "
             f"variant-(b) by type = "
             + ", ".join(f"{t}={n}" for t, n in sorted(rep.n_anchors_typed_by_type.items()))
             + ".\n")

    verdict_label = {
        "IMPROVE": "(i) C8 anchors materially improve separation → proceed to a threshold",
        "FIRE_NOSEP": "(ii) anchors FIRE but do NOT separate on SA-2 → re-test on a heterogeneous doc",
        "NOFIRE": "(iii) anchors do not fire as expected → investigate",
    }.get(rep.verdict, rep.verdict)
    L.append(f"\n## BOTTOM LINE: **{rep.verdict}** — {verdict_label}\n")
    for n in rep.notes:
        L.append(f"- {n}\n")

    # firing rate + univariate AUROC of the anchor features, per pass.
    fa = rep.per_feature_auroc
    passes = sorted(fa["per_pass"].keys())
    L.append("\n## Anchor features — firing rate & univariate AUROC (used vs unused)\n\n")
    L.append("| feature | scope | firing_rate | AUROC |\n|---|---|---|---|\n")
    for feat in ANCHOR_FEATURE_FIELDS:
        L.append(f"| {feat} | pooled | "
                 f"{fa['firing_rate']['pooled'][feat]*100:.1f}% | "
                 f"{fa['pooled'][feat]:.3f} |\n")
        for pn in passes:
            L.append(f"| {feat} | {pn} | "
                     f"{fa['firing_rate']['per_pass'][pn][feat]*100:.1f}% | "
                     f"{fa['per_pass'][pn][feat]:.3f} |\n")
    sp = fa["support"]
    L.append(f"\n_support: pooled n={sp['pooled']['n']} used={sp['pooled']['n_used']}; "
             + "; ".join(f"{pn} n={sp['per_pass'][pn]['n']} used={sp['per_pass'][pn]['n_used']}"
                         for pn in passes) + "_\n")

    # baseline component AUROC (context: lexical_norm ~0.50 is the problem C8 targets)
    bca = rep.baseline_component_auroc
    L.append("\n## Baseline C5-component univariate AUROC (context — the gap C8 targets)\n\n")
    L.append("| component | pooled | " + " | ".join(passes) + " |\n")
    L.append("|---|" + "---|" * (len(passes) + 1) + "\n")
    for comp in COMPONENT_FIELDS:
        cells = [f"{bca['pooled'][comp]:.3f}"]
        for pn in passes:
            cells.append(f"{bca['per_pass'].get(pn, {}).get(comp, float('nan')):.3f}")
        L.append(f"| {comp} | " + " | ".join(cells) + " |\n")

    # the headline A/B: re-weighted CV AUROC WITH vs WITHOUT anchors.
    fit = rep.fit
    wa, wo = fit["with_anchors"], fit["without_anchors"]
    L.append("\n## Re-weighted CV AUROC — WITHOUT anchors (Phase-1b ~0.60) vs WITH anchors\n\n")
    L.append("| model | WITHOUT (baseline) CV | WITH-anchors CV | delta |\n")
    L.append("|---|---|---|---|\n")
    L.append(f"| logistic (L2) | {wo['logistic_cv_auroc']:.3f} | "
             f"{wa['logistic_cv_auroc']:.3f} | {fit['delta_logistic_cv_auroc']:+.3f} |\n")
    L.append(f"| GBM (nonlinear ceiling) | {wo['gbm_cv_auroc']:.3f} | "
             f"{wa['gbm_cv_auroc']:.3f} | {fit['delta_gbm_cv_auroc']:+.3f} |\n")
    L.append(f"| **best** | **{wo['best_cv_auroc']:.3f}** | "
             f"**{wa['best_cv_auroc']:.3f}** | **{fit['delta_best_cv_auroc']:+.3f}** |\n")
    L.append(f"\n_CV = stratified {fit['n_splits_effective']}-fold within this doc; "
             f"{fit['n_used']}/{fit['n']} positives — small-sample; CV is the honest "
             f"estimate, in-sample is only a ceiling._\n")
    L.append(f"\nIn-sample (ceiling, overfits): WITH logit "
             f"{wa['logistic_in_sample_auroc']:.3f} / gbm "
             f"{wa['gbm_in_sample_auroc']:.3f}.\n")

    # learned WITH-anchors logistic weights — do the anchor features carry weight?
    L.append("\n## Learned WITH-anchors logistic weights (standardised; which features dominate)\n\n")
    L.append("| feature | weight |\n|---|---|\n")
    w = wa["logistic_weights"]
    for fn in sorted(w, key=lambda k: -abs(w[k])):
        mark = " ◀ anchor" if fn in ANCHOR_FEATURE_FIELDS else ""
        L.append(f"| {fn}{mark} | {w[fn]:+.3f} |\n")
    if wa["gbm_feature_importance"]:
        L.append("\nGBM feature importance: " +
                 ", ".join(f"{k}={v:.2f}" for k, v in
                           sorted(wa["gbm_feature_importance"].items(),
                                  key=lambda kv: -kv[1])) + "\n")

    # recall vs threshold on the best WITH-anchors score.
    L.append("\n## Recall-vs-threshold on the best WITH-anchors re-weighted score\n\n")
    L.append("_CAVEAT: this curve is fit on the IN-SAMPLE best-side score (same "
             "method as Phase-1b), so the fraction-kept is an OPTIMISTIC ceiling — "
             "the threshold is chosen on the very rows it is scored against. It is "
             "NOT a CV-validated reduction; with the in-sample/CV gap here (GBM "
             "0.97 in-sample vs 0.67 CV) it would not survive out-of-fold. Read it "
             "as 'best case', not 'deployable'._\n\n")
    rc = rep.recall_curve
    rec = rc.get("pooled_recommendation")
    if rec is not None:
        L.append("| scope | threshold | recall | frac_kept | chunks_kept |\n")
        L.append("|---|---|---|---|---|\n")
        L.append(f"| pooled | {rec.threshold:.4f} | {rec.recall_achieved:.2f} | "
                 f"{rec.fraction_kept:.2f} | {rec.chunks_kept} |\n")
        for pn in sorted(rc.get("per_pass", {})):
            r = rc["per_pass"][pn]["recommendation"]
            if r is not None:
                L.append(f"| {pn} | {r.threshold:.4f} | {r.recall_achieved:.2f} | "
                         f"{r.fraction_kept:.2f} | {r.chunks_kept} |\n")
    else:
        L.append("_No recall-preserving threshold exists on the WITH-anchors score._\n")

    return "".join(L)


# ===========================================================================
# I/O wrappers (only invoked by main(); thin + monkeypatchable)
# ===========================================================================

def _harvest_anchors_all(
    *, db, run_id: str, manifest, ontology: dict, document_id: str
) -> list[str]:
    """Variant (a): call the PRODUCTION ``_collect_committed_identity_anchors``
    verbatim — this is exactly the current C8 wiring (every committed identity
    name)."""
    from app.workers.pipeline import _collect_committed_identity_anchors
    return _collect_committed_identity_anchors(
        db=db, run_id=run_id, manifest=manifest,
        ontology=ontology, document_id=document_id,
    )


def _harvest_anchors_typed(
    *, db, run_id: str, manifest, ontology: dict, document_id: str
) -> list[TypedAnchor]:
    """Variant (b): the SAME identity walk as the production harvester, but
    capturing each ref's ``entity_type`` alongside its name(s).

    Mirrors ``_collect_committed_identity_anchors`` step-for-step (identity-phase
    passes → ``load_pass_output`` → ``_parse_pass_response`` →
    ``_build_pre_merge_walk_summary`` → ``_extend_upstream_refs``), then reads
    ``ref.entity_type`` + ``ref.display_label`` + ``ref.aliases``. The function
    itself returns only names, so this typed sibling is the minimal reuse needed
    to tag names by type — NO hardcoded type names.
    """
    from app.workers.pipeline import (
        _build_pre_merge_walk_summary,
        _extend_upstream_refs,
        _parse_pass_response,
        load_pass_output,
    )

    try:
        identity_pass_defs = [
            p for p in manifest.passes if getattr(p, "phase", None) == "identity"
        ]
    except Exception:
        return []
    if not identity_pass_defs:
        return []

    out: list[TypedAnchor] = []
    seen: set[tuple[str, str]] = set()

    def _add(name, etype) -> None:
        if name and isinstance(name, str) and etype:
            key = (name.strip(), etype)
            if key[0] and key not in seen:
                seen.add(key)
                out.append(TypedAnchor(name=key[0], entity_type=etype))

    for id_pass in identity_pass_defs:
        try:
            row = load_pass_output(db, run_id, id_pass.name)
            if row is None or getattr(row, "execution_status", None) != "COMPLETE":
                continue
            pass_result = _parse_pass_response(
                row.extract_pass_response_json, id_pass, manifest)
            pass_result.pre_merge_walk = _build_pre_merge_walk_summary(
                pass_result, id_pass, ontology, document_id)
            refs: dict = {}
            _extend_upstream_refs(refs, pass_result, id_pass, ontology)
            for ref in refs.values():
                etype = getattr(ref, "entity_type", None)
                _add(getattr(ref, "display_label", None), etype)
                for alias in (getattr(ref, "aliases", None) or []):
                    _add(alias, etype)
        except Exception:
            continue
    return out


def _field_group_pass_defs(manifest) -> dict[str, Any]:
    """{pass_name: pass_def} for the field_group (routed) passes — the only
    passes the router narrows, and the scope for the type-matched anchor set."""
    return {
        p.name: p for p in manifest.passes
        if str(getattr(p, "phase", "") or "") == "field_group"
    }


def _analyzable_passes(pass_outputs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Passes with a KNOWN (non-empty-provenance) USED set — same 4-pass scope as
    the Phase-1b baseline (the two field_group passes + the two identity passes
    that carry provenance); empty-provenance passes (system_links) are excluded
    so they cannot dilute the fit."""
    def _has_known_used(po):
        return bool(po.get("provenance") or po.get("field_provenance"))
    return sorted(pn for pn, po in pass_outputs.items() if _has_known_used(po))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--bundle-key",
                    default=os.environ.get("PHASE1_BUNDLE_KEY", "air_defense_v3_merged_v1"))
    ap.add_argument("--recall-floor", type=float, default=1.0)
    ap.add_argument("--auroc-floor", type=float, default=DEFAULT_AUROC_SEP_FLOOR)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="reports/collection")
    args = ap.parse_args()

    os.environ.setdefault("PHASE1_BUNDLE_KEY", args.bundle_key)
    run8 = args.run_id.split("-")[0]

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.services.ontology_bundles import load_bundle_manifest
    from app.services.ontology_templates import load_ontology

    manifest = load_bundle_manifest(args.bundle_key)
    ontology = load_ontology(bundle_key=args.bundle_key)
    fg_pass_defs = _field_group_pass_defs(manifest)
    print(f"[phase2] field_group passes (type-matched scope): {sorted(fg_pass_defs)}")

    # ---- harvest anchors (read-only Postgres session via the app's DSN) ----
    print(f"[phase2] harvesting C8 anchors for run {args.run_id} ...")
    engine = create_engine(p1._sync_database_url())
    try:
        with Session(engine) as db:
            all_anchors = _harvest_anchors_all(
                db=db, run_id=args.run_id, manifest=manifest,
                ontology=ontology, document_id=args.doc_id)
            typed_anchors = _harvest_anchors_typed(
                db=db, run_id=args.run_id, manifest=manifest,
                ontology=ontology, document_id=args.doc_id)
    finally:
        engine.dispose()
    typed_by_type: dict[str, int] = {}
    for ta in typed_anchors:
        typed_by_type[ta.entity_type] = typed_by_type.get(ta.entity_type, 0) + 1
    print(f"[phase2] anchors: ALL={len(all_anchors)}  typed={len(typed_anchors)} "
          f"({typed_by_type})")

    # ---- fetch chunks + pass outputs, build labeled component table ----
    pass_outputs = p1._fetch_pass_outputs(args.run_id)
    analyze = _analyzable_passes(pass_outputs)
    print(f"[phase2] analyzing passes (known USED set): {analyze}")
    chunks = p1._fetch_chunks(args.run_id)
    print(f"[phase2] chunks={len(chunks)}")
    chunks_by_pass = {pn: chunks for pn in analyze if pn in pass_outputs}

    print("[phase2] building component table + USED labels (reused phase1b/phase1) ...")
    comp_rows = build_component_table(chunks_by_pass=chunks_by_pass)
    used_rows = build_used_table(pass_outputs, chunks_by_pass)
    labeled = label_component_table(comp_rows, used_rows)

    # type-matched anchor scope is per-PASS: for the two field_group passes use
    # their own entity type; for the identity passes (also analyzed) use their
    # primary_entity_types too (so radar_identity gets RADAR_SYSTEM names, etc.)
    # — all driven off the manifest pass_def.
    pass_defs_for_typed = {p.name: p for p in manifest.passes if p.name in analyze}

    print("[phase2] computing anchor features (variant a ALL + variant b TYPED) ...")
    anchor_rows = build_anchor_feature_table(
        chunks_by_pass=chunks_by_pass,
        all_anchors=all_anchors,
        typed_anchors=typed_anchors,
        pass_defs=pass_defs_for_typed,
    )
    feature_rows = attach_anchor_features(labeled, anchor_rows)

    # ---- analysis ----
    print("[phase2] per-feature univariate AUROC + firing rate ...")
    feat_auroc = per_feature_auroc(feature_rows, ANCHOR_FEATURE_FIELDS)
    baseline_component_auroc = p1b.per_component_auroc(labeled)

    print("[phase2] fitting re-weighted separators WITH vs WITHOUT anchors (CV) ...")
    fit = fit_with_and_without_anchors(
        feature_rows, n_splits=args.cv_splits, seed=args.seed)
    print(f"   WITHOUT best CV={fit['without_anchors']['best_cv_auroc']:.3f}  "
          f"WITH best CV={fit['with_anchors']['best_cv_auroc']:.3f}  "
          f"delta={fit['delta_best_cv_auroc']:+.3f}")

    best_scores = fit["with_anchors"]["best_scores"]
    recall_curve = anchor_recall_curve(
        feature_rows, best_scores, recall_floor=args.recall_floor)

    verdict, notes = decide_verdict(
        feat_auroc, fit, recall_curve, auroc_sep_floor=args.auroc_floor)

    rep = AnchorReport(
        run_id=args.run_id,
        doc_id=args.doc_id,
        bundle_key=args.bundle_key,
        n_rows=len(feature_rows),
        n_used=sum(1 for r in feature_rows if r.get("used")),
        passes=sorted({str(r["pass_name"]) for r in feature_rows}),
        n_anchors_all=len(all_anchors),
        n_anchors_typed_by_type=typed_by_type,
        per_feature_auroc=feat_auroc,
        baseline_component_auroc=baseline_component_auroc,
        fit=fit,
        recall_curve=recall_curve,
        verdict=verdict,
        auroc_sep_floor=args.auroc_floor,
        notes=notes,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"phase2_anchor_features_{run8}.csv")
    json_path = os.path.join(args.out_dir, f"phase2_anchor_{run8}.json")
    md_path = os.path.join(args.out_dir, f"phase2_anchor_{run8}.md")

    _write_feature_csv(feature_rows, csv_path)
    with open(json_path, "w") as f:
        json.dump(_report_to_jsonable(rep), f, indent=2, default=str)
    with open(md_path, "w") as f:
        f.write(_render_markdown(rep))

    print(f"\n[phase2] wrote {csv_path}")
    print(f"[phase2] wrote {json_path}")
    print(f"[phase2] wrote {md_path}")
    print(f"\n========== VERDICT: {rep.verdict} ==========")
    for n in rep.notes:
        print(f"  - {n}")


if __name__ == "__main__":
    main()
