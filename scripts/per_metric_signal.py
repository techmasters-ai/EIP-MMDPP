#!/usr/bin/env python3
"""Univariate signal per metric — "is there enough signal in each feature ALONE?"

For every one of the 10 score components we combine, this fits a UNIVARIATE
logistic regression (StandardScaler + class-balanced LogReg) on the
value-grounded target and reports, per metric:

  * AUROC — honest cross-document discrimination via leave-one-document-out
    out-of-fold probabilities (GroupKFold by run). 0.50 = coin-flip (no signal),
    1.0 = perfect. This is THE "does signal exist alone" number.
  * in-sample AUROC — optimistic ceiling on the same single feature.
  * optimal threshold — the logistic-regression decision boundary (P=0.5),
    expressed back in the RAW feature units ("keep when feature >= t"), with the
    coefficient sign (+ = higher feature ⇒ more likely a true source chunk).
  * operating point AT that threshold (honest, from the LODO out-of-fold
    probabilities): recall, precision, specificity, F1, balanced-accuracy, and
    chunk savings (% of (pass,chunk) cells the threshold drops).
  * separation — nonzero rate + median value among POSITIVES vs NEGATIVES, so a
    sparse lexical feature's reach is visible (how many positives it can even
    fire on).

The threshold is derived with logistic regression ONLY (no Youden/F1 sweep),
per request. class_weight="balanced" matches the production frontier and the
recall-first objective (the P=0.5 boundary sits at the balanced operating point,
not the accuracy-optimal one).

    python3 -m scripts.per_metric_signal --runs <r1,...> [--target lineage_grounded] [--feature pass_keyword_norm]
"""
from __future__ import annotations
import argparse
import numpy as np

import scripts.a0_captured_separation as a0
from scripts.a0_captured_separation import FEATURES

# Group the 10 metrics so the report leads with the lexical/keyword family.
LEXICAL = ["pass_keyword_norm", "field_label_norm", "anchor_text_norm", "anchor_section_norm"]
SEMANTIC = ["cosine", "rerank_norm"]
STRUCTURAL = ["section_norm", "is_table", "pattern_norm", "negative_norm"]
ORDER = LEXICAL + SEMANTIC + STRUCTURAL


def _load(runs, target):
    rows_all = []
    for run in runs:
        rows, _ = a0.build_run_table(run, target_mode=target)
        rows_all += list(a0._labeled(rows))
    return rows_all


def _univariate(x, y, groups):
    """Return dict with LODO out-of-fold probs, in-sample probs, raw threshold,
    coef sign. None probs when CV can't be formed."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, GroupKFold

    X = x.reshape(-1, 1)
    n_groups = len(set(groups))
    mk = lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    # honest cross-document out-of-fold probabilities
    oof = None
    if n_groups >= 2 and 0 < int(y.sum()) < len(y):
        try:
            oof = cross_val_predict(
                mk(), X, y, groups=groups,
                cv=GroupKFold(n_splits=n_groups), method="predict_proba",
            )[:, 1]
        except Exception:
            oof = None
    # full-data fit → interpretable raw threshold at P=0.5
    full = mk().fit(X, y)
    lr = full.named_steps["logisticregression"]
    sc = full.named_steps["standardscaler"]
    coef = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    mean = float(sc.mean_[0])
    std = float(sc.scale_[0])
    raw_thr = mean + std * (-intercept / coef) if abs(coef) > 1e-12 else float("nan")
    insample = full.predict_proba(X)[:, 1]
    return {"oof": oof, "insample": insample, "raw_thr": raw_thr, "coef": coef}


def _ops(y, prob, thr=0.5):
    """Confusion-derived metrics at probability cut `thr`."""
    from sklearn.metrics import roc_auc_score, confusion_matrix
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    kept = pred.mean()
    auroc = roc_auc_score(y, prob) if len(set(y.tolist())) > 1 else float("nan")
    return {"recall": rec, "precision": prec, "specificity": spec, "f1": f1,
            "balacc": 0.5 * (rec + spec), "kept": kept, "savings": 1 - kept,
            "auroc": auroc, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--target", default="lineage_grounded",
                    choices=("used", "lineage", "lineage_grounded"))
    ap.add_argument("--feature", default=None, help="restrict to one feature")
    args = ap.parse_args(argv)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    rows = _load(runs, args.target)
    if not rows:
        print("no labeled rows"); return 1
    y = np.array([1 if r.used else 0 for r in rows], int)
    groups = [r.run_id for r in rows]
    N, n_pos = len(y), int(y.sum())
    feats = [args.feature] if args.feature else ORDER

    print(f"\n=== Univariate per-metric signal  (target={args.target}) ===")
    print(f"corpus: {N} (pass,chunk) candidates across {len(set(groups))} docs, "
          f"{n_pos} positives ({100*n_pos/N:.1f}%), {N-n_pos} negatives\n")

    results = []
    for f in feats:
        x = np.array([getattr(r, f) for r in rows], float)
        uni = _univariate(x, y, groups)
        prob = uni["oof"] if uni["oof"] is not None else uni["insample"]
        honest = uni["oof"] is not None
        op = _ops(y, prob, 0.5)
        ins = _ops(y, uni["insample"], 0.5)
        # separation: nonzero reach + medians by class
        xp, xn = x[y == 1], x[y == 0]
        nz_p = float((xp > 1e-9).mean()) if len(xp) else 0.0
        nz_n = float((xn > 1e-9).mean()) if len(xn) else 0.0
        results.append({
            "feature": f, "auroc": op["auroc"], "honest": honest,
            "insample_auroc": ins["auroc"], "raw_thr": uni["raw_thr"],
            "coef": uni["coef"], "op": op, "nz_p": nz_p, "nz_n": nz_n,
            "med_p": float(np.median(xp)) if len(xp) else 0.0,
            "med_n": float(np.median(xn)) if len(xn) else 0.0,
        })

    # ranked summary table
    order = sorted(results, key=lambda d: (-(d["auroc"] if d["auroc"] == d["auroc"] else -1)))
    print("RANKED by honest (LODO) AUROC:")
    print(f"{'metric':22s} {'AUROC':>6s} {'insmpl':>6s} {'recall':>6s} {'prec':>6s} "
          f"{'spec':>6s} {'F1':>6s} {'save%':>6s} {'thr(raw)':>9s} {'sgn':>3s} "
          f"{'nz+':>5s} {'nz-':>5s}")
    print("-" * 104)
    for d in order:
        op = d["op"]
        thr = d["raw_thr"]
        thr_s = f"{thr:9.3f}" if thr == thr else "      n/a"
        print(f"{d['feature']:22s} {d['auroc']:6.3f} {d['insample_auroc']:6.3f} "
              f"{op['recall']:6.2f} {op['precision']:6.2f} {op['specificity']:6.2f} "
              f"{op['f1']:6.2f} {100*op['savings']:6.1f} {thr_s} "
              f"{'+' if d['coef']>0 else '-':>3s} {100*d['nz_p']:5.0f} {100*d['nz_n']:5.0f}")

    # detailed verdict per metric (lexical first)
    print("\nPER-METRIC DETAIL (verdict):")
    for d in [r for r in results]:
        op = d["op"]
        verdict = _verdict(d, n_pos)
        fam = ("LEXICAL" if d["feature"] in LEXICAL else
               "SEMANTIC" if d["feature"] in SEMANTIC else "STRUCTURAL")
        print(f"\n[{fam}] {d['feature']}")
        print(f"  AUROC (LODO honest) = {d['auroc']:.3f}   in-sample = {d['insample_auroc']:.3f}"
              f"   ({'cross-doc' if d['honest'] else 'in-sample only — CV unformable'})")
        print(f"  LogReg threshold: keep when {d['feature']} "
              f"{'>=' if d['coef']>0 else '<='} {d['raw_thr']:.3f}  (coef sign {'+' if d['coef']>0 else '-'})")
        print(f"  at threshold: recall={op['recall']:.2f} precision={op['precision']:.2f} "
              f"specificity={op['specificity']:.2f} F1={op['f1']:.2f}  "
              f"savings={100*op['savings']:.0f}%  (TP={op['tp']} FP={op['fp']} FN={op['fn']} TN={op['tn']})")
        print(f"  reach: nonzero in {100*d['nz_p']:.0f}% of positives vs {100*d['nz_n']:.0f}% of negatives "
              f"(median +{d['med_p']:.3f} / -{d['med_n']:.3f})")
        print(f"  verdict: {verdict}")
    print()
    return 0


def _verdict(d, n_pos) -> str:
    a = d["auroc"]
    rec = d["op"]["recall"]
    nzp = d["nz_p"]
    if a != a:
        return "NO CV — too few positives/groups to judge cross-document."
    if a < 0.55:
        return "NO usable signal alone (AUROC≈coin-flip)."
    if nzp < 0.34:
        return (f"SPARSE — fires on only {100*nzp:.0f}% of positives, so it caps recall at ~{100*nzp:.0f}% "
                f"alone; useful as a high-precision booster, not a standalone gate.")
    if a < 0.65:
        return "WEAK standalone signal; contributes but cannot gate alone."
    if a < 0.75:
        return "MODERATE standalone signal."
    return "STRONG standalone signal — viable as a primary discriminator."


if __name__ == "__main__":
    raise SystemExit(main())
