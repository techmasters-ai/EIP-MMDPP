#!/usr/bin/env python3
"""Guarded-ranker CALIBRATION (guarded-ranker spec Task 19).

One command turns a value-grounded bake-off dataset (``export_bakeoff_dataset``
output) into deployable ``ranker_weights`` + ``quantile_q`` — the numbers that
drop into a bundle manifest's ``retrieval`` block — with a finite-sample margin
and honest leave-one-document-out (LODO) reporting.

What it does, in order
----------------------
1. REFUSE-OR-PROCEED gate (recall-floor precondition). Runs
   ``scripts.check_gate_coverage`` programmatically on the input CSV: every
   positive (``used==1``) must be covered by ``unit_gate | table_gate`` (captured
   flag, or — with ``--bundle`` — the offline gate recompute). A positive the
   gate union cannot protect is a recall hole the guarded ranker could drop while
   it holds a true field value, so the script REFUSES to emit weights (non-zero
   exit) until coverage is clean.

2. SIGN-CONSTRAINED LogReg. Fits L2 ``LogisticRegression(class_weight=
   'balanced')`` with ``StandardScaler``; the regularization strength ``C`` is
   selected from ``--c-grid`` by NESTED GroupKFold (grouped by ``run_id`` so no
   document leaks across the inner split). Every coefficient must be non-negative
   EXCEPT ``negative_norm`` (<= 0). A wrong-signed coefficient → that feature is
   dropped, the model is refit, and the drop is logged — mirroring
   ``eval_guarded_ranker.fit_sign_constrained``.

3. QUANTILE + MARGIN. Scores every row out-of-fold (LODO GroupKFold by
   ``run_id``; pooled-OOF probabilities) and picks the SMALLEST ``quantile_q``
   from the grid whose guarded keep-set (gates ∪ per-pool quantile cut, clipped
   to ``[k_min, k_max]``) holds recall == 1.0 on EVERY held-out document.  THEN
   it applies the finite-sample margin: q is stepped DOWN by ``--margin-steps``
   grid steps (clamped to the grid floor).  This is a pad against small-sample
   noise — it is NOT a recall guarantee on unseen documents.  The honest LODO
   numbers (pooled-OOF + mean-per-fold AUROC, per-fold recall/kept_frac) are
   reported so the residual risk stays visible.

4. EMIT. Writes ``<out-dir>/guarded_ranker_fit.json`` (final weights, chosen q,
   per-fold recall/kept_frac, BOTH LODO conventions) AND prints a ready-to-paste
   manifest YAML snippet (``selection_mode: guarded_quantile``, ``quantile_q``,
   ``k_min``, ``k_max``, ``ranker_weights``). The ranker_weights keys are the
   dataset COLUMN names exactly (``max_field_cosine``, ``mean_top3_field_cosine``,
   ``rerank_norm``, ...) so the router's ``select_candidates`` helper reads them
   by the same names.

NO recall GUARANTEE: the margin is a finite-sample pad on a small, imbalanced,
SA-2-dominated corpus. recall==1.0 holds on the calibration folds BY
CONSTRUCTION (the gate union covers every positive); it is NOT proven on docs
the calibration never saw. Re-run on a broader corpus before trusting the q on
new document shapes.

Usage::

    python3 -m scripts.fit_guarded_ranker --csv reports/dataset_v2/bakeoff_dataset.csv
    python3 -m scripts.fit_guarded_ranker --csv data.csv --bundle air_defense_v3 \
        --margin-steps 1 --c-grid 0.01,0.1,1,10
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Mapping, Sequence

# Reuse the validated guarded-ranker primitives (sign-constrained fit, LODO
# scoring, per-pool guarded keep-set, recall) rather than reimplementing them.
from scripts.eval_guarded_ranker import (
    DEFAULT_FEATURES,
    DEFAULT_GATE_COLS,
    EXPECTED_NEGATIVE,
    SIGN_TOL,
    fit_sign_constrained,
    guarded_keep_mask_all,
    lodo_oof_scores,
    parse_quantile_grid,
    pool_indices,
    recall_and_kept,
)

DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)


# ===========================================================================
# Pure helpers
# ===========================================================================
def parse_c_grid(spec: str) -> list[float]:
    """``"0.01,0.1,1,10"`` -> [0.01, 0.1, 1.0, 10.0]. Every C must be > 0."""
    parts = [p.strip() for p in (spec or "").split(",") if p.strip()]
    if len(parts) < 1:
        raise ValueError(f"C grid must list >=1 value, got {spec!r}")
    out: list[float] = []
    for p in parts:
        try:
            v = float(p)
        except ValueError as e:
            raise ValueError(f"bad C value {p!r} in grid {spec!r}") from e
        if v <= 0:
            raise ValueError(f"C must be > 0, got {v} in grid {spec!r}")
        out.append(v)
    return out


def select_C(X, y, groups, c_grid: Sequence[float]) -> float:
    """Pick the C maximizing mean nested-GroupKFold AUROC (grouped by run_id so
    no document leaks across the split). Single C / <2 groups → that C.

    Ties and single-class inner folds are handled defensively: folds whose
    train/test split is single-class are skipped; a C with no scorable fold
    falls back to its mean of 0.5 (chance) so it never wins by accident.
    """
    import numpy as np
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cs = list(c_grid)
    uniq = sorted(set(groups))
    if len(cs) == 1 or len(uniq) < 2:
        return cs[0]

    Xn = np.asarray(X, dtype=float)
    yn = np.asarray(y, dtype=int)
    gn = np.asarray(groups)
    cv = GroupKFold(n_splits=len(uniq))
    best_c, best_score = cs[0], -1.0
    for c in cs:
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=c)),
        ])
        aucs: list[float] = []
        for tr, te in cv.split(Xn, yn, gn):
            if len(set(yn[tr].tolist())) < 2 or len(set(yn[te].tolist())) < 2:
                continue
            m = clone(pipe).fit(Xn[tr], yn[tr])
            aucs.append(roc_auc_score(yn[te], m.predict_proba(Xn[te])[:, 1]))
        score = float(np.mean(aucs)) if aucs else 0.5
        if score > best_score:
            best_c, best_score = c, score
    return best_c


def fit_final_weights(X, y, features: Sequence[str], C: float) -> dict[str, float]:
    """Refit on ALL data at the selected C and return the per-feature LogReg
    coefficient (in scaled space) keyed by feature name — these become
    ``ranker_weights``. The intercept is returned under ``__intercept``."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=C)),
    ])
    pipe.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    coefs = pipe.named_steps["lr"].coef_[0]
    weights = {f: float(c) for f, c in zip(features, coefs)}
    weights["__intercept"] = float(pipe.named_steps["lr"].intercept_[0])
    return weights


def per_fold_guarded_recall(df, scores, gate_mask, groups, q, k_min, k_max):
    """Per-held-out-document guarded recall/kept_frac using the pooled-OOF
    ``scores`` (the row's score came from a model that never saw its doc).

    The guarded keep-set is computed GLOBALLY (per (run_id, pass_name) pool)
    then sliced to each document — this is exactly the keep-set production would
    apply, restricted to the held-out doc's rows. Returns one dict per doc plus
    the worst (min) recall over docs.
    """
    import numpy as np

    keep = guarded_keep_mask_all(df, scores, q, k_min, k_max, gate_mask)
    g = np.asarray(groups)
    used = np.asarray(df["used"]).astype(int)
    rows: list[dict[str, Any]] = []
    for doc in sorted(set(g.tolist())):
        m = g == doc
        recall, kept_frac = recall_and_kept(used[m], keep[m])
        rows.append({
            "run_id": str(doc),
            "recall": recall,
            "kept_frac": kept_frac,
            "n_positives": int((used[m] == 1).sum()),
        })
    min_recall = min((r["recall"] for r in rows), default=1.0)
    return rows, min_recall


def choose_quantile(df, scores, gate_mask, groups, qs, k_min, k_max):
    """SMALLEST q whose per-fold guarded recall == 1.0 on EVERY held-out doc.

    Returns (chosen_q | None, per_q_summary). ``chosen_q`` is None when no q in
    the grid holds recall 1.0 everywhere.
    """
    summary: list[dict[str, Any]] = []
    chosen: float | None = None
    for q in qs:
        _, min_recall = per_fold_guarded_recall(df, scores, gate_mask, groups, q, k_min, k_max)
        keep = guarded_keep_mask_all(df, scores, q, k_min, k_max, gate_mask)
        _, kept_frac = recall_and_kept(df["used"], keep)
        ok = min_recall >= 1.0
        summary.append({"q": float(q), "min_fold_recall": float(min_recall),
                        "kept_frac": float(kept_frac), "recall_one_every_fold": ok})
        if ok and chosen is None:
            chosen = float(q)
    return chosen, summary


def apply_margin(qs: Sequence[float], q: float, margin_steps: int) -> float:
    """Step ``q`` DOWN by ``margin_steps`` grid steps (clamped to the grid
    floor). The finite-sample pad — see module docstring (NOT a guarantee)."""
    grid = sorted(set(float(x) for x in qs))
    # locate q in the grid (nearest, FP-safe)
    idx = min(range(len(grid)), key=lambda i: abs(grid[i] - q))
    idx = max(0, idx - max(int(margin_steps), 0))
    return grid[idx]


# ===========================================================================
# Orchestration (pure: DataFrame in, result dict out — unit-tested directly)
# ===========================================================================
def fit_and_select(
    df,
    *,
    features: Sequence[str],
    quantile_grid: Sequence[float] | None = None,
    gate_cols: Sequence[str] = DEFAULT_GATE_COLS,
    k_min: int = 3,
    k_max: int = 0,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    margin_steps: int = 1,
) -> dict[str, Any]:
    """End-to-end fit + selection on an already-loaded DataFrame.

    Does NOT run the gate-coverage refusal (that is ``main``'s precondition);
    callers that need it run ``scripts.check_gate_coverage`` first.
    """
    import numpy as np

    # Avoid mutating the caller's DataFrame (we fill missing feature columns
    # with 0.0 below — that must not leak back to the caller).
    df = df.copy()
    feats = list(features)
    for f in feats:
        if f not in df.columns:
            print(f"[features] WARNING: column {f!r} missing from CSV — filled 0.0")
            df[f] = 0.0
    qs = list(quantile_grid) if quantile_grid is not None else parse_quantile_grid("0.5:0.95:0.05")

    # gate mask
    gate_mask = np.zeros(len(df), dtype=bool)
    used_gate_cols: list[str] = []
    for gc in gate_cols:
        if gc in df.columns:
            gate_mask |= df[gc].astype(float).to_numpy() == 1.0
            used_gate_cols.append(gc)
        else:
            print(f"[gates] WARNING: gate column {gc!r} missing — treated as all-0")

    y = (df["used"].astype(int) == 1).astype(int).to_numpy()
    groups = df["run_id"].astype(str).to_numpy()

    # 1. sign-constrained feature selection (fit on ALL data)
    X_all = df[feats].astype(float).to_numpy()
    kept_features, dropped = fit_sign_constrained(X_all, y, feats)

    # 2. C selection by nested GroupKFold over the kept features
    Xk = df[kept_features].astype(float).to_numpy()
    C = select_C(Xk, y, groups, c_grid)

    # 3. final weights (refit on ALL data at C)
    weights_full = fit_final_weights(Xk, y, kept_features, C)
    ranker_weights = {f: weights_full[f] for f in kept_features}

    # 4. LODO pooled-OOF scores (model never sees a row's own doc). Use the
    #    SELECTED C so the reported LODO/AUROC and the q-selection characterize
    #    the SAME model whose weights are shipped (not sklearn's default C=1.0).
    lodo = lodo_oof_scores(Xk, y, groups, C=C)
    scores = np.asarray(lodo["scores"], dtype=float)

    # 5. smallest q with per-fold recall 1.0, then margin
    chosen_q, q_summary = choose_quantile(df, scores, gate_mask, groups, qs, k_min, k_max)
    if chosen_q is None:
        raise SystemExit(
            "[quantile] NO q in the grid holds guarded recall == 1.0 on every "
            "held-out document — cannot pick a recall-safe q. Inspect the q "
            "summary / widen k_min, or re-collect. (refusing to emit a q)"
        )
    q_after_margin = apply_margin(qs, chosen_q, margin_steps)

    # per-fold detail at the FINAL (post-margin) q
    per_fold, _ = per_fold_guarded_recall(df, scores, gate_mask, groups,
                                          q_after_margin, k_min, k_max)

    return {
        "features_requested": feats,
        "features_kept": kept_features,
        "features_dropped_by_sign": dropped,
        "C": float(C),
        "c_grid": [float(c) for c in c_grid],
        "ranker_weights": ranker_weights,
        "intercept": weights_full.get("__intercept"),
        "gate_cols": used_gate_cols,
        "k_min": int(k_min),
        "k_max": int(k_max),
        "quantile_grid": [float(q) for q in qs],
        "quantile_q_before_margin": float(chosen_q),
        "margin_steps": int(margin_steps),
        "quantile_q": float(q_after_margin),
        "quantile_summary": q_summary,
        "per_fold": per_fold,
        "lodo": {
            "pooled_oof_auroc": lodo.get("pooled_oof_auroc"),
            "mean_per_fold_auroc": lodo.get("mean_per_fold_auroc"),
            "folds_scored": lodo.get("folds_scored"),
            "folds_skipped_single_class": lodo.get("folds_skipped_single_class"),
            "convention": lodo.get("convention"),
        },
    }


def build_manifest_snippet(res: Mapping[str, Any], *, k_min: int, k_max: int) -> str:
    """Ready-to-paste manifest YAML. ranker_weights keys = dataset column names
    exactly (so the router reads them by the same names)."""
    import yaml

    block = {
        "selection_mode": "guarded_quantile",
        "quantile_q": round(float(res["quantile_q"]), 6),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "ranker_weights": {f: round(float(w), 6) for f, w in res["ranker_weights"].items()},
    }
    return yaml.safe_dump(block, sort_keys=False, default_flow_style=False)


# ===========================================================================
# main
# ===========================================================================
def _split(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", required=True, help="bake-off dataset CSV (export_bakeoff_dataset output)")
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES),
                    help="comma list of model features (default EXCLUDES the keyword trio)")
    ap.add_argument("--gate-cols", default=",".join(DEFAULT_GATE_COLS),
                    help="comma list of 0/1 gate columns unioned into the keep set")
    ap.add_argument("--quantile-grid", default="0.5:0.95:0.05",
                    help="start:stop:step (stop inclusive); the script picks the "
                         "SMALLEST q holding recall 1.0 on every fold, then steps "
                         "down by --margin-steps")
    ap.add_argument("--c-grid", default=",".join(str(c) for c in DEFAULT_C_GRID),
                    help="comma list of LogReg C values; C is chosen by nested "
                         "GroupKFold AUROC (grouped by run_id)")
    ap.add_argument("--margin-steps", type=int, default=1,
                    help="finite-sample pad: step the chosen q DOWN this many grid "
                         "steps (NOT a recall guarantee on unseen docs)")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=0, help="0 = uncapped")
    ap.add_argument("--bundle", default=None,
                    help="ontology bundle key for the offline gate recompute in "
                         "the recall-floor precondition (e.g. air_defense_v3); "
                         "omit → captured-flag-only coverage")
    ap.add_argument("--out", default="",
                    help="JSON output path (default <csv dir>/guarded_ranker_fit.json)")
    args = ap.parse_args(argv)

    import pandas as pd

    df = pd.read_csv(args.csv).reset_index(drop=True)
    for col in ("run_id", "pass_name", "used"):
        if col not in df.columns:
            raise SystemExit(f"required column {col!r} missing from {args.csv}")

    # ---- 1. recall-floor precondition (REFUSE-OR-PROCEED) -----------------
    from scripts.check_gate_coverage import (
        GateColumnsMissing,
        _resolve_signatures,
        check_gate_coverage,
    )

    signatures = _resolve_signatures(args.bundle) if args.bundle else None
    try:
        misses = check_gate_coverage(df, signatures=signatures)
    except GateColumnsMissing as e:
        print(f"REFUSING to emit weights — gate columns absent (recall-floor "
              f"UNCHECKABLE): {e}")
        return 2
    if misses:
        n_pos = int((df["used"].astype(int) == 1).sum())
        print(f"REFUSING to emit weights — GATE COVERAGE FAIL: {len(misses)}/"
              f"{n_pos} used==1 rows covered by NEITHER unit_gate/table_gate "
              "(captured or recomputed). The guarded ranker could drop a chunk "
              "holding a true value — fix coverage / re-collect before calibrating.")
        print("(doc_filename, pass_name, chunk_index, unit_gate, table_gate)")
        for m in misses[:20]:
            print(m)
        return 1
    n_pos = int((df["used"].astype(int) == 1).sum())
    print(f"[precondition] gate coverage OK: {n_pos}/{n_pos} positives covered "
          f"by unit_gate|table_gate"
          + ("" if signatures is not None else " (captured-flag only; pass --bundle for recompute)"))

    # ---- 2-5. fit + select -------------------------------------------------
    features = _split(args.features)
    qs = parse_quantile_grid(args.quantile_grid)
    c_grid = parse_c_grid(args.c_grid)
    print(f"[data] {len(df)} rows, {n_pos} positives, {df['run_id'].nunique()} docs, "
          f"{len(pool_indices(df))} (run, pass) pools")

    res = fit_and_select(
        df,
        features=features,
        quantile_grid=qs,
        gate_cols=_split(args.gate_cols),
        k_min=args.k_min,
        k_max=args.k_max,
        c_grid=c_grid,
        margin_steps=args.margin_steps,
    )

    # ---- report -----------------------------------------------------------
    print(f"[model] features after sign check ({len(res['features_kept'])}/"
          f"{len(features)}): {', '.join(res['features_kept'])}")
    if res["features_dropped_by_sign"]:
        for d in res["features_dropped_by_sign"]:
            print(f"[model]   dropped {d['feature']} (coef {d['coef']:+.4f}, "
                  f"expected {d['expected']})")
    print(f"[model] selected C = {res['C']} (nested GroupKFold by run_id)")
    pa, ma = res["lodo"]["pooled_oof_auroc"], res["lodo"]["mean_per_fold_auroc"]
    print(f"[model] LODO AUROC — pooled-OOF: {pa:.3f}" if pa is not None
          else "[model] LODO AUROC — pooled-OOF: n/a")
    print(f"[model] LODO AUROC — mean-per-fold: {ma:.3f} "
          f"({res['lodo'].get('folds_scored')} folds scored, "
          f"{res['lodo'].get('folds_skipped_single_class')} skipped single-class)"
          if ma is not None else "[model] LODO AUROC — mean-per-fold: n/a")
    print(f"[quantile] smallest q with recall 1.0 every fold = "
          f"{res['quantile_q_before_margin']:.2f}; after margin "
          f"(-{args.margin_steps} step) = {res['quantile_q']:.2f}")
    print("[quantile] per-fold recall/kept at the FINAL q:")
    for r in res["per_fold"]:
        print(f"           {r['run_id']}: recall {r['recall']:.3f} "
              f"kept_frac {r['kept_frac']:.3f} (pos={r['n_positives']})")
    print("[note] recall==1.0 holds on calibration folds BY CONSTRUCTION (gate "
          "union covers every positive). The margin is a finite-sample pad — NOT "
          "a recall guarantee on unseen document shapes.")

    snippet = build_manifest_snippet(res, k_min=args.k_min, k_max=args.k_max)
    print("\n# ---- ready-to-paste manifest retrieval snippet ----")
    print(snippet)

    # ---- emit JSON --------------------------------------------------------
    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                        "guarded_ranker_fit.json")
    payload = dict(res)
    payload["csv"] = args.csv
    payload["n_rows"] = len(df)
    payload["n_positives"] = n_pos
    payload["n_docs"] = int(df["run_id"].nunique())
    payload["manifest_snippet"] = snippet
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
