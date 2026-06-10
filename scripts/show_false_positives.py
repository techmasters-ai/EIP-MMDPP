#!/usr/bin/env python3
"""Show FALSE POSITIVES: (pass, chunk) cells the model ranks HIGH but that are
labeled NEGATIVE in the value-grounded target (no numeric value grounds there).

For each, classify WHY it's a high-scoring negative:
  * MISSED-NUMERIC  — a numeric+unit value of one of the pass's OWN fields IS in
    the chunk → the grounded target under-counted (grounding precision gap, not a
    model error). These should arguably be positives.
  * STRING-FIELD    — a string value of an own field is in the chunk (guidance_type,
    name…) → relevant but excluded by the numeric-only target by design.
  * MODEL-ERROR     — no own-field value present → the model likes an irrelevant
    chunk (the features that fired are the culprit; shown).

Same honest model as the frontier (class-balanced LogReg, LODO out-of-fold).

    python3 -m scripts.show_false_positives --runs <r1,...> [--n 15]
"""
from __future__ import annotations
import argparse, re, unicodedata
import numpy as np

import scripts.a0_captured_separation as a0
from app.services.field_value_grounding import (
    nfc as _nfc, num_variants, units_for, value_in_chunk,
)


def _strnorm(s) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", _nfc(str(s)).casefold())
                   if not unicodedata.combining(c))


def _classify(pass_own_values: dict, chunk_text: str):
    """Return (kind, detail). pass_own_values: field->[values] (pass's OWN fields)."""
    txt = _nfc(chunk_text)
    ntxt = _strnorm(chunk_text)
    num_hits, str_hits = [], []
    for f, vals in pass_own_values.items():
        units = units_for(f)
        for v in vals:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if units and value_in_chunk(num_variants(v), units, txt):
                    num_hits.append(f"{f}={v}")
            elif isinstance(v, str):
                sv = _strnorm(v)
                if len(sv) >= 4 and sv in ntxt:
                    str_hits.append(f"{f}={v!r}")
    if num_hits:
        return "MISSED-NUMERIC", ", ".join(num_hits[:3])
    if str_hits:
        return "STRING-FIELD", ", ".join(str_hits[:3])
    return "MODEL-ERROR", ""


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

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict, GroupKFold
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    oof = cross_val_predict(clf, X, y, groups=groups,
                            cv=GroupKFold(n_splits=len(set(groups))), method="predict_proba")[:, 1]

    # per-run, per-pass OWN field values (scoped via field_provenance, like the target)
    own_vals = {}
    for run in runs:
        po = a0.fetch_pass_outputs(run)
        d = {}
        for pn, blocks in po.items():
            own = {fp.get("field_name") for fp in (blocks.get("field_provenance") or [])
                   if isinstance(fp, dict) and fp.get("field_name")}
            fv = {}
            for v in (blocks.get("pass_output") or {}).values():
                if isinstance(v, list):
                    for rec in v:
                        if isinstance(rec, dict):
                            for f, val in rec.items():
                                if f in own and val not in (None, "") and not isinstance(val, (list, dict)):
                                    fv.setdefault(f, []).append(val)
            d[pn] = fv
        own_vals[run] = d
    ct_cache = {run: a0._fetch_chunk_text(run) for run in runs}

    negs = sorted(((oof[i], i) for i in range(len(oof)) if y[i] == 0), key=lambda t: -t[0])
    from collections import Counter
    kinds = Counter()
    print(f"=== {len(oof)} candidates, {int(y.sum())} positives; "
          f"top {args.n} HIGH-scoring NEGATIVES (false positives) ===\n")
    for score, i in negs[:args.n]:
        r = rows_all[i]
        ct = ct_cache[r.run_id].get(r.chunk_index, "")
        kind, detail = _classify(own_vals[r.run_id].get(r.pass_name, {}), ct)
        kinds[kind] += 1
        feats = {f: round(getattr(r, f), 2) for f in a0.FEATURES if abs(getattr(r, f)) > 1e-6}
        print(f"[{r.run_id[:8]} | {r.pass_name} | chunk {r.chunk_index}] score={score:.3f}  {kind}")
        if detail:
            print(f"    own-field value present: {detail}")
        print(f"    chunk text: {ct[:120].replace(chr(10),' ')}")
        print(f"    features: {feats}\n")
    print("class of top false-positives:", dict(kinds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
