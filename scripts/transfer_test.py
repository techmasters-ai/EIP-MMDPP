#!/usr/bin/env python3
"""Transfer test: train the chunk-selection model on the OLD corpus (the 7
SA-2-family docs) and score a NEW held-out document with it — the honest
"does the model generalize to a doc it never saw?" measurement.

Reports, for each model (class-balanced LogReg = the frontier model; GBM = the
bake-off winner): AUROC on the new doc, the operating point at the model's own
P=0.5 boundary, and the recall-vs-savings the OLD model achieves on the NEW doc
(savings at recall 1.0 / .95 / .90). Also dumps per-feature learned weights so
you can see which signals the old model leaned on and whether they transferred.

    python3 -m scripts.transfer_test --train <old runs csv> --test <new run>
"""
from __future__ import annotations
import argparse
import numpy as np

import scripts.a0_captured_separation as a0
from scripts.a0_captured_separation import FEATURES, build_run_table, _labeled


def _load(runs, target):
    rows = []
    for r in runs:
        rr, _ = build_run_table(r, target_mode=target)
        rows += list(_labeled(rr))
    X = np.array([[getattr(r, f) for f in FEATURES] for r in rows], float)
    y = np.array([1 if r.used else 0 for r in rows], int)
    return X, y, rows


def _ops(y, proba, thr):
    from sklearn.metrics import confusion_matrix
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return dict(recall=rec, precision=prec, specificity=spec,
                kept=pred.mean(), savings=1 - pred.mean(),
                tp=tp, fp=fp, fn=fn, tn=tn)


def _savings_at(y, proba, rfloor):
    """Max chunk savings achievable on the test set while holding recall>=rfloor,
    sweeping the threshold (the OLD model's scores)."""
    order = np.argsort(-proba)
    ys = y[order]
    n_pos = int(y.sum()); N = len(y)
    best = 0.0
    # sweep keep-prefix sizes
    tp = 0
    for k in range(1, N + 1):
        tp += int(ys[k - 1] == 1)
        rec = tp / n_pos if n_pos else 1.0
        if rec >= rfloor:
            best = max(best, 1 - k / N)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="comma-sep OLD run ids")
    ap.add_argument("--test", required=True, help="comma-sep NEW run ids (held out)")
    ap.add_argument("--target", default="lineage_grounded")
    args = ap.parse_args(argv)
    train_runs = [r.strip() for r in args.train.split(",") if r.strip()]
    test_runs = [r.strip() for r in args.test.split(",") if r.strip()]

    Xtr, ytr, _ = _load(train_runs, args.target)
    Xte, yte, _ = _load(test_runs, args.target)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    print(f"\n=== TRANSFER TEST: train on OLD, score NEW (held out) ===")
    print(f"TRAIN: {len(ytr)} candidates, {int(ytr.sum())} positives, {len(train_runs)} docs")
    print(f"TEST : {len(yte)} candidates, {int(yte.sum())} positives, {len(test_runs)} doc(s)\n")

    models = {
        "LogReg (class-balanced, frontier model)":
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced")),
        "GradientBoosting (bake-off winner)":
            GradientBoostingClassifier(random_state=0, max_depth=2, n_estimators=100),
    }
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        auroc = roc_auc_score(yte, proba) if len(set(yte.tolist())) > 1 else float("nan")
        # in-sample (train) AUROC for reference
        auroc_tr = roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1])
        op = _ops(yte, proba, 0.5)
        print(f"--- {name} ---")
        print(f"  AUROC on NEW doc (transfer) = {auroc:.3f}   (train in-sample = {auroc_tr:.3f})")
        print(f"  at the model's P>=0.5 boundary: recall={op['recall']:.2f} "
              f"precision={op['precision']:.2f} savings={100*op['savings']:.0f}%  "
              f"(TP={op['tp']} FP={op['fp']} FN={op['fn']} TN={op['tn']})")
        print(f"  OLD-model recall→savings on NEW doc:")
        for rf in (1.00, 0.95, 0.90, 0.80):
            print(f"     recall ≥ {rf:.2f}: {100*_savings_at(yte, proba, rf):.0f}% savings")
        # learned weights (LogReg only)
        if hasattr(clf, "named_steps"):
            lr = clf.named_steps["logisticregression"]
            w = lr.coef_[0]
            order = np.argsort(-np.abs(w))
            print("  learned weights (|w| desc):")
            for i in order:
                print(f"     {FEATURES[i]:22s} {w[i]:+.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
