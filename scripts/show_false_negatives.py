#!/usr/bin/env python3
"""Show FALSE NEGATIVES: value-grounded positive (pass, chunk) cells the model
ranks LOW (would be dropped at a chunk-cut threshold), with the chunk text and
the field value it actually contains.

Scores every candidate with the same honest model the frontier uses
(class-balanced LogisticRegression, leave-one-document-out out-of-fold via
GroupKFold). Then lists the positives with the lowest scores — the unrankable
tail that pins the recall-1.0 frontier — each annotated with the grounded
field=value and a snippet of the chunk around it.

    python3 -m scripts.show_false_negatives --runs <r1,...> [--n 15]
"""
from __future__ import annotations
import argparse, re
import numpy as np

import scripts.a0_captured_separation as a0
from app.services.field_value_grounding import (
    nfc as _nfc, num_variants, units_for, value_in_chunk,
)


def _grounding_fields(pass_field_values, chunk_text):
    """Return [(field, value)] whose value grounds in chunk_text (numeric+unit)."""
    txt = _nfc(chunk_text)
    hits = []
    for f, vals in pass_field_values.items():
        units = units_for(f)
        if not units:
            continue
        for v in vals:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if value_in_chunk(num_variants(v), units, txt):
                hits.append((f, v))
    return hits


def _snippet(chunk_text, value, width=90):
    """Text window around the first occurrence of the value's digits."""
    digits = re.sub(r"\.0$", "", str(value))
    m = re.search(re.escape(digits.split(".")[0]), chunk_text)
    if not m:
        return chunk_text[:width].replace("\n", " ")
    a, b = max(0, m.start() - width // 2), min(len(chunk_text), m.start() + width // 2)
    return ("…" if a > 0 else "") + chunk_text[a:b].replace("\n", " ") + ("…" if b < len(chunk_text) else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args(argv)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    rows_all = []
    for run in runs:
        rows, _ = a0.build_run_table(run, target_mode="lineage_grounded")
        rows_all += [r for r in a0._labeled(rows)]
    X = np.array([[getattr(r, f) for f in a0.FEATURES] for r in rows_all], float)
    y = np.array([1 if r.used else 0 for r in rows_all], int)
    groups = [r.run_id for r in rows_all]
    n_pos = int(y.sum())

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, GroupKFold
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    oof = cross_val_predict(clf, X, y, groups=groups,
                            cv=GroupKFold(n_splits=len(set(groups))), method="predict_proba")[:, 1]

    # rank ALL candidates; a positive is a false negative if it scores below
    # negatives. Report the lowest-scoring positives + their percentile.
    order = np.argsort(oof)            # ascending
    rank = {i: k for k, i in enumerate(order)}   # 0 = lowest score
    N = len(oof)

    ct_cache = {run: a0._fetch_chunk_text(run) for run in runs}
    pv_cache = {}
    for run in runs:
        po = a0.fetch_pass_outputs(run)
        pv = {}
        for pn, blocks in po.items():
            fv = {}
            for v in (blocks.get("pass_output") or {}).values():
                if isinstance(v, list):
                    for rec in v:
                        if isinstance(rec, dict):
                            for f, val in rec.items():
                                if val is not None and val != "" and not isinstance(val, (list, dict)):
                                    fv.setdefault(f, []).append(val)
            pv[pn] = fv
        pv_cache[run] = pv

    pos = sorted(((oof[i], i) for i in range(N) if y[i] == 1), key=lambda t: t[0])
    print(f"=== {N} candidates, {n_pos} value-grounded positives, {len(set(groups))} docs ===")
    print(f"showing the {min(args.n, len(pos))} lowest-scoring positives (false negatives at a cut):\n")
    for score, i in pos[:args.n]:
        r = rows_all[i]
        pct = 100.0 * rank[i] / max(1, N - 1)   # % of candidates scoring BELOW this positive
        ct = ct_cache[r.run_id].get(r.chunk_index, "")
        hits = _grounding_fields(pv_cache[r.run_id].get(r.pass_name, {}), ct)
        hit_str = ", ".join(f"{f}={v}" for f, v in hits[:3]) or "(value not re-found)"
        snip = _snippet(ct, hits[0][1], 100) if hits else ct[:100].replace("\n", " ")
        print(f"[{r.run_id[:8]} | {r.pass_name} | chunk {r.chunk_index}] score={score:.3f} "
              f"(below {pct:.0f}% of all candidates)")
        print(f"    missed field value: {hit_str}")
        print(f"    chunk text: {snip}")
        # which features are ~zero for this positive (why it's unrankable)?
        feats = {f: getattr(r, f) for f in a0.FEATURES}
        nonzero = {f: round(v, 2) for f, v in feats.items() if abs(v) > 1e-6}
        print(f"    nonzero features: {nonzero or '(all ~0 — no signal fired)'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
