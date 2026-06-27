#!/usr/bin/env python3
"""Plot recall vs % chunk savings for the curated bake-off corpus.

Honest cross-document curve: leave-one-document-out out-of-fold LogisticRegression
probabilities (GroupKFold by run/doc) over the clean lineage target, then sweep
the score threshold. At each threshold: keep_frac = chunks kept, savings = 1-keep,
recall = positives captured / total positives. Plots recall (x) vs savings (y).

    python3 -m scripts.plot_recall_vs_savings --runs <r1,r2,...>
"""
from __future__ import annotations
import argparse, os
import numpy as np

from scripts.a0_captured_separation import build_run_table, _labeled, FEATURES


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--target", default="lineage", choices=("lineage", "lineage_grounded"))
    ap.add_argument("--drop-features", default="", help="comma-sep feature names to exclude")
    ap.add_argument("--out", default="reports/collection/recall_vs_savings.png")
    args = ap.parse_args(argv)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    drop = {f.strip() for f in args.drop_features.split(",") if f.strip()}
    feats = [f for f in FEATURES if f not in drop]
    if drop:
        print(f"dropping features: {sorted(drop)} → using {feats}")

    X, y, groups = [], [], []
    for rid in runs:
        rows, _ = build_run_table(rid, target_mode=args.target)
        for r in _labeled(rows):
            X.append([getattr(r, f) for f in feats])
            y.append(1 if r.used else 0)
            groups.append(rid)
    X = np.array(X, float); y = np.array(y, int)
    n_pos = int(y.sum()); N = len(y)
    print(f"{N} chunks, {n_pos} positives, {len(set(groups))} docs")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, GroupKFold
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    n_splits = len(set(groups))
    oof = cross_val_predict(clf, X, y, groups=groups,
                            cv=GroupKFold(n_splits=n_splits), method="predict_proba")[:, 1]

    # sweep thresholds → (recall, savings)
    ths = np.unique(np.concatenate([[0.0], np.sort(oof), [1.0]]))
    recalls, savings = [], []
    for t in ths:
        kept = oof >= t
        keep_frac = kept.mean()
        rec = (y[kept].sum() / n_pos) if n_pos else 0.0
        recalls.append(rec); savings.append(1.0 - keep_frac)
    recalls = np.array(recalls); savings = np.array(savings) * 100.0

    # achievable savings at given recall: max savings among thresholds meeting recall
    def savings_at(rfloor):
        ok = recalls >= rfloor
        return savings[ok].max() if ok.any() else 0.0
    marks = [(1.00, savings_at(1.00)), (0.99, savings_at(0.99)),
             (0.95, savings_at(0.95)), (0.90, savings_at(0.90)), (0.80, savings_at(0.80))]
    print("recall → max % chunk savings:")
    for r, s in marks:
        print(f"  recall ≥ {r:.2f}: {s:.0f}% savings")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # frontier: for each recall, the best savings (monotone)
    order = np.argsort(recalls)
    rs, sv = recalls[order], savings[order]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(rs * 100, sv, "-", color="#1f77b4", lw=2, label="LODO out-of-fold (LogReg)")
    for r, s in marks:
        ax.plot(r * 100, s, "o", color="#d62728", ms=7)
        ax.annotate(f"R={r:.2f}\n{s:.0f}% saved", (r * 100, s),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Recall of extracted-from chunks (%)")
    ax.set_ylabel("Chunk savings — % of (pass,chunk) cells dropped")
    ax.set_title(f"Recall vs chunk savings — clean lineage target\n"
                 f"{len(set(groups))} docs, {N} chunks, {n_pos} positives (cross-document LODO)")
    ax.grid(True, alpha=0.3); ax.set_xlim(75, 101); ax.set_ylim(-2, 102)
    ax.invert_xaxis()  # full recall (100%) at left → savings grow as recall relaxes
    ax.legend(loc="upper left")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
