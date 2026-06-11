#!/usr/bin/env python3
"""Guarded-ranker frontier evaluator (guarded-ranker spec §4).

Evaluates the production-candidate selection rule on a bake-off dataset CSV
(export_bakeoff_dataset output):

    keep = GATE UNION (unit_gate | table_gate)  OR  ranker keep
    ranker keep (per (run_id, pass_name) pool) =
        score >= quantile(q) of the pool's scores, count clipped to
        [k_min, k_max or pool size] by rank

where ``score`` is a leave-one-document-out (LODO) calibrated LogisticRegression
probability — GroupKFold by run_id, class_weight='balanced', StandardScaler —
so every row is scored by a model that never saw its document. A sign
constraint is enforced first (fit on ALL data): every feature must carry a
non-negative coefficient except ``negative_norm`` (non-positive); wrong-signed
features are dropped + logged and the model refit until clean.

Default feature list EXCLUDES the keyword trio (field_label_norm,
pass_keyword_norm, anchor_text_norm) per spec §4 — they are SA-2-overfit (see
memory: keyword features overfit; cosine generalizes).

Outputs, all in one run:
  * LODO AUROC under BOTH conventions, labeled exactly:
    "pooled-OOF" (one AUROC over the pooled out-of-fold probabilities) and
    "mean-per-fold" (per-held-out-doc AUROC averaged; single-class folds skipped)
  * the guarded frontier: for each q in the grid — recall (over used==1),
    kept fraction, gate-only kept fraction
  * baseline (a): production final_score top-k sweep (k=1..k-sweep-max)
  * baseline (b): calibrated-score-only quantile cut (NO gate union)

Usage::

    python3 -m scripts.eval_guarded_ranker --csv reports/dataset_v2/bakeoff_dataset.csv
    python3 -m scripts.eval_guarded_ranker --csv old.csv \
        --features cosine,rerank_norm,negative_norm --gate-cols ""
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Mapping, Sequence

DEFAULT_FEATURES = (
    "cosine",
    "max_field_cosine",
    "mean_top3_field_cosine",
    "rerank_norm",
    "is_table",
    "unit_token_count",
    "digit_density",
    "label_value_lines",
    "negative_norm",
)
DEFAULT_GATE_COLS = ("unit_gate", "table_gate")
# expected coefficient signs: >= 0 for everything except these (<= 0)
EXPECTED_NEGATIVE = frozenset({"negative_norm"})
SIGN_TOL = 1e-9
POOL_KEYS = ("run_id", "pass_name")


# ===========================================================================
# Pure helpers (unit-tested; no I/O, no model — scores come in precomputed)
# ===========================================================================
def parse_quantile_grid(spec: str) -> list[float]:
    """``"0.5:0.95:0.05"`` (start:stop:step, stop INCLUSIVE) -> [0.5, 0.55, ... 0.95]."""
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) != 3:
        raise ValueError(f"quantile grid must be start:stop:step, got {spec!r}")
    start, stop, step = (float(p) for p in parts)
    if step <= 0 or stop < start:
        raise ValueError(f"bad quantile grid {spec!r} (need step > 0, stop >= start)")
    out: list[float] = []
    q = start
    while q <= stop + step / 2.0:
        out.append(round(q, 10))
        q += step
    if not all(0.0 <= v <= 1.0 for v in out):
        raise ValueError(f"quantile grid {spec!r} leaves [0, 1]")
    return out


def guarded_keep_mask(df_pool, scores, q, k_min, k_max, gate_mask):
    """Keep mask for ONE (run_id, pass_name) pool.

    threshold   = quantile ``q`` of the pool's ``scores`` (linear interpolation)
    ranker keep = top-``k`` rows by score where ``k`` = count(score >= threshold)
                  clipped to [k_min, k_max or pool size]; the cap (k_max) is
                  hard — it wins over k_min when they conflict
    final keep  = ``gate_mask`` OR ranker keep  (the gate union is unconditional)
    """
    import numpy as np

    n = len(df_pool)
    s = np.asarray(scores, dtype=float)
    g = np.asarray(gate_mask, dtype=bool)
    if len(s) != n or len(g) != n:
        raise ValueError(f"scores/gate_mask length mismatch vs pool (n={n})")
    if n == 0:
        return np.zeros(0, dtype=bool)
    thr = float(np.quantile(s, q))
    n_at_thr = int((s >= thr).sum())
    cap = n if k_max is None or int(k_max) <= 0 else min(int(k_max), n)
    k = min(max(n_at_thr, int(k_min)), cap)
    order = np.argsort(-s, kind="stable")
    ranker = np.zeros(n, dtype=bool)
    ranker[order[:k]] = True
    return g | ranker


def pool_indices(df) -> dict[tuple, list[int]]:
    """POSITIONAL row indices per (run_id, pass_name) pool (index-agnostic)."""
    pools: dict[tuple, list[int]] = {}
    keys = list(zip(df["run_id"].tolist(), df["pass_name"].tolist()))
    for i, k in enumerate(keys):
        pools.setdefault(k, []).append(i)
    return pools


def guarded_keep_mask_all(df, scores, q, k_min, k_max, gate_mask):
    """guarded_keep_mask applied independently per (run_id, pass_name) pool."""
    import numpy as np

    s = np.asarray(scores, dtype=float)
    g = np.asarray(gate_mask, dtype=bool)
    keep = np.zeros(len(df), dtype=bool)
    for idx in pool_indices(df).values():
        ii = np.asarray(idx, dtype=int)
        keep[ii] = guarded_keep_mask(df.iloc[ii], s[ii], q, k_min, k_max, g[ii])
    return keep


def recall_and_kept(used, keep) -> tuple[float, float]:
    """(recall over used==1, kept fraction). recall=1.0 when no positives (vacuous)."""
    import numpy as np

    u = np.asarray(used).astype(int) == 1
    k = np.asarray(keep, dtype=bool)
    n_pos = int(u.sum())
    recall = float((u & k).sum() / n_pos) if n_pos else 1.0
    kept_frac = float(k.mean()) if len(k) else 0.0
    return recall, kept_frac


def topk_keep_mask_all(df, scores, k):
    """Baseline (a) helper: per-pool top-k by score (production-style cut)."""
    import numpy as np

    s = np.asarray(scores, dtype=float)
    keep = np.zeros(len(df), dtype=bool)
    for idx in pool_indices(df).values():
        ii = np.asarray(idx, dtype=int)
        order = np.argsort(-s[ii], kind="stable")
        keep[ii[order[: min(int(k), len(ii))]]] = True
    return keep


def frontier_table(df, scores, gate_mask, qs, k_min, k_max) -> list[dict[str, float]]:
    """(q, recall, kept_frac, gate_only_kept_frac) for each q in the grid."""
    import numpy as np

    used = df["used"]
    gate_only_kept = float(np.asarray(gate_mask, dtype=bool).mean()) if len(df) else 0.0
    rows: list[dict[str, float]] = []
    for q in qs:
        keep = guarded_keep_mask_all(df, scores, q, k_min, k_max, gate_mask)
        recall, kept_frac = recall_and_kept(used, keep)
        rows.append({
            "q": float(q),
            "recall": recall,
            "kept_frac": kept_frac,
            "gate_only_kept_frac": gate_only_kept,
        })
    return rows


# ===========================================================================
# Model: sign-constrained LODO LogReg
# ===========================================================================
def _build_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def fit_sign_constrained(X_all, y, features: Sequence[str]) -> tuple[list[str], list[dict]]:
    """Fit on ALL data; drop wrong-signed features, refit, iterate until clean.

    Expected signs: coef >= 0 for every feature except EXPECTED_NEGATIVE
    (coef <= 0). ``X_all`` is the full feature matrix column-aligned with
    ``features``. Returns (kept feature names, dropped log entries).
    """
    import numpy as np

    feats = list(features)
    cols = {f: i for i, f in enumerate(features)}
    dropped: list[dict] = []
    while feats:
        pipe = _build_pipeline()
        pipe.fit(np.asarray(X_all)[:, [cols[f] for f in feats]], y)
        coefs = pipe.named_steps["lr"].coef_[0]
        bad: list[tuple[str, float]] = []
        for f, c in zip(feats, coefs):
            if f in EXPECTED_NEGATIVE:
                if c > SIGN_TOL:
                    bad.append((f, float(c)))
            elif c < -SIGN_TOL:
                bad.append((f, float(c)))
        if not bad:
            return feats, dropped
        for f, c in bad:
            exp = "<= 0" if f in EXPECTED_NEGATIVE else ">= 0"
            print(f"[sign-check] DROP {f}: coef {c:+.4f} violates expected {exp} — refitting")
            feats.remove(f)
            dropped.append({"feature": f, "coef": c, "expected": exp})
    raise SystemExit("[sign-check] ALL features dropped by the sign constraint — nothing to fit")


def lodo_oof_scores(X, y, groups, *, label: str = "LODO LogReg") -> dict[str, Any]:
    """LODO scoring: GroupKFold by run_id; pooled-OOF probabilities via
    cross_val_predict + mean-per-fold AUROC via an explicit fold loop.
    Falls back to in-sample scoring (clearly labeled) when <2 docs."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    out: dict[str, Any] = {"label": label, "n_docs": len(set(groups))}
    pipe = _build_pipeline()
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        print(f"[{label}] WARNING: single doc — scores are IN-SAMPLE, not LODO")
        pipe.fit(X, y)
        out["scores"] = pipe.predict_proba(X)[:, 1]
        out["pooled_oof_auroc"] = None
        out["mean_per_fold_auroc"] = None
        out["convention"] = "in-sample (single doc — NOT LODO)"
        return out

    cv = GroupKFold(n_splits=len(uniq))
    oof = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    out["scores"] = oof
    out["pooled_oof_auroc"] = (
        float(roc_auc_score(y, oof)) if len(set(np.asarray(y).tolist())) > 1 else None
    )
    fold_aucs: list[float] = []
    skipped = 0
    for tr, te in cv.split(X, y, groups):
        if len(set(np.asarray(y)[tr].tolist())) < 2 or len(set(np.asarray(y)[te].tolist())) < 2:
            skipped += 1
            continue
        m = clone(pipe).fit(np.asarray(X)[tr], np.asarray(y)[tr])
        fold_aucs.append(float(roc_auc_score(np.asarray(y)[te], m.predict_proba(np.asarray(X)[te])[:, 1])))
    out["mean_per_fold_auroc"] = float(np.mean(fold_aucs)) if fold_aucs else None
    out["folds_scored"] = len(fold_aucs)
    out["folds_skipped_single_class"] = skipped
    out["convention"] = "GroupKFold by run_id (leave-one-document-out)"
    return out


# ===========================================================================
# Reporting
# ===========================================================================
def _fmt_frontier(rows: Sequence[Mapping[str, float]], title: str) -> str:
    L = [f"## {title}", f"{'q':>6} {'recall':>8} {'kept_frac':>10} {'gate_only_kept_frac':>20}"]
    for r in rows:
        L.append(f"{r['q']:>6.2f} {r['recall']:>8.3f} {r['kept_frac']:>10.3f} "
                 f"{r['gate_only_kept_frac']:>20.3f}")
    perfect = [r for r in rows if r["recall"] >= 1.0]
    if perfect:
        smallest = min(perfect, key=lambda r: r["q"])
        largest = max(perfect, key=lambda r: r["q"])
        L.append(f"smallest q with recall == 1.0: q={smallest['q']:.2f} "
                 f"(kept_frac {smallest['kept_frac']:.3f})")
        L.append(f"largest  q with recall == 1.0 (max savings): q={largest['q']:.2f} "
                 f"(kept_frac {largest['kept_frac']:.3f})")
    else:
        L.append("NO q in the grid holds recall == 1.0")
    return "\n".join(L)


def _fmt_topk(rows: Sequence[Mapping[str, float]]) -> str:
    L = ["## baseline (a): final_score per-pool top-k sweep",
         f"{'k':>4} {'recall':>8} {'kept_frac':>10}"]
    for r in rows:
        L.append(f"{int(r['k']):>4} {r['recall']:>8.3f} {r['kept_frac']:>10.3f}")
    perfect = [r for r in rows if r["recall"] >= 1.0]
    if perfect:
        k1 = min(perfect, key=lambda r: r["k"])
        L.append(f"smallest k with recall == 1.0: k={int(k1['k'])} (kept_frac {k1['kept_frac']:.3f})")
    else:
        L.append(f"NO k <= {int(rows[-1]['k'])} holds recall == 1.0")
    return "\n".join(L)


# ===========================================================================
# main
# ===========================================================================
def _split(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="bake-off dataset CSV")
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES),
                    help="comma list of model features (default EXCLUDES the "
                         "keyword trio per spec §4)")
    ap.add_argument("--gate-cols", default=",".join(DEFAULT_GATE_COLS),
                    help="comma list of 0/1 gate columns unioned into the keep "
                         "set; '' disables the gate union")
    ap.add_argument("--quantile-grid", default="0.5:0.95:0.05",
                    help="start:stop:step, stop inclusive")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=0, help="0 = uncapped")
    ap.add_argument("--k-sweep-max", type=int, default=50,
                    help="baseline (a) sweeps k=1..this")
    ap.add_argument("--out", default="", help="optional JSON output path")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd

    df = pd.read_csv(args.csv).reset_index(drop=True)
    for col in ("run_id", "pass_name", "used", "final_score"):
        if col not in df.columns:
            raise SystemExit(f"required column {col!r} missing from {args.csv}")
    qs = parse_quantile_grid(args.quantile_grid)

    # --- features (missing columns -> 0.0 with a warning; constant cols get
    # coef 0 under the sign-checked fit, so they are harmless) -------------
    features = _split(args.features)
    for f in features:
        if f not in df.columns:
            print(f"[features] WARNING: column {f!r} missing from CSV — filled 0.0")
            df[f] = 0.0

    # --- gate mask (absent/empty -> all-False with a warning, NOT a crash) -
    gate_cols = _split(args.gate_cols)
    gate_mask = np.zeros(len(df), dtype=bool)
    if not gate_cols:
        print("[gates] WARNING: no gate columns given — gate mask all-False "
              "(guarded frontier degenerates to calibrated-only)")
    for gc in gate_cols:
        if gc not in df.columns:
            print(f"[gates] WARNING: gate column {gc!r} missing from CSV — treated as all-0")
            continue
        gate_mask |= df[gc].astype(float).to_numpy() == 1.0

    y = (df["used"].astype(int) == 1).astype(int).to_numpy()
    groups = df["run_id"].astype(str).to_numpy()
    n_pos = int(y.sum())
    print(f"[data] {len(df)} rows, {n_pos} positives, {len(set(groups))} docs, "
          f"{len(pool_indices(df))} (run, pass) pools; gate covers "
          f"{int(gate_mask.sum())} rows ({gate_mask.mean()*100:.1f}%)")

    # --- sign-constrained feature selection (fit on ALL data) --------------
    X_all = df[features].astype(float).to_numpy()
    kept_features, dropped = fit_sign_constrained(X_all, y, features)
    print(f"[model] features after sign check ({len(kept_features)}/{len(features)}): "
          + ", ".join(kept_features))

    # --- LODO scores --------------------------------------------------------
    X = df[kept_features].astype(float).to_numpy()
    lodo = lodo_oof_scores(X, y, groups)
    scores = np.asarray(lodo["scores"], dtype=float)
    pa = lodo.get("pooled_oof_auroc")
    ma = lodo.get("mean_per_fold_auroc")
    print(f"[model] LODO AUROC — pooled-OOF: {pa:.3f}" if pa is not None
          else "[model] LODO AUROC — pooled-OOF: n/a")
    print(f"[model] LODO AUROC — mean-per-fold: {ma:.3f} "
          f"({lodo.get('folds_scored', 0)} folds scored, "
          f"{lodo.get('folds_skipped_single_class', 0)} skipped single-class)"
          if ma is not None else "[model] LODO AUROC — mean-per-fold: n/a")
    print("[model] frontier + baseline (b) use the pooled-OOF probabilities")

    # --- frontiers ----------------------------------------------------------
    guarded = frontier_table(df, scores, gate_mask, qs, args.k_min, args.k_max)
    no_gate = np.zeros(len(df), dtype=bool)
    calibrated_only = frontier_table(df, scores, no_gate, qs, args.k_min, args.k_max)
    topk_rows: list[dict[str, float]] = []
    fs = df["final_score"].astype(float).to_numpy()
    for k in range(1, args.k_sweep_max + 1):
        recall, kept_frac = recall_and_kept(df["used"], topk_keep_mask_all(df, fs, k))
        topk_rows.append({"k": float(k), "recall": recall, "kept_frac": kept_frac})

    print()
    print(_fmt_frontier(
        guarded, f"guarded frontier (gate union: {', '.join(gate_cols) or 'NONE'} "
                 f"| LODO score >= per-pool q, k_min={args.k_min}, "
                 f"k_max={args.k_max or 'uncapped'})"))
    print()
    print(_fmt_frontier(calibrated_only,
                        "baseline (b): calibrated-score-only (same quantile cut, NO gate union)"))
    print()
    print(_fmt_topk(topk_rows))

    if args.out:
        payload = {
            "csv": args.csv,
            "n_rows": len(df),
            "n_positives": n_pos,
            "n_docs": len(set(groups)),
            "features_requested": features,
            "features_kept": kept_features,
            "features_dropped_by_sign": dropped,
            "gate_cols": gate_cols,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "lodo": {
                "pooled_oof_auroc": pa,
                "mean_per_fold_auroc": ma,
                "folds_scored": lodo.get("folds_scored"),
                "folds_skipped_single_class": lodo.get("folds_skipped_single_class"),
                "convention": lodo.get("convention"),
            },
            "guarded_frontier": guarded,
            "calibrated_only_frontier": calibrated_only,
            "final_score_topk": topk_rows,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
