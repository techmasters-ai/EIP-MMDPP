#!/usr/bin/env python3
"""Export the value-grounded chunk-selection dataset to a self-contained,
session-portable form (CSV + parquet + README) so models can be trained in a
fresh session without touching the live DB or re-running the pipeline.

One row per (run, pass, chunk) candidate. Columns:
  identifiers : run_id, document_id, doc_filename, pass_name, chunk_index,
                candidate_key, self_ref, page_number, token_count
  label       : used                (1 = value-grounded source chunk, target=lineage_grounded)
  production  : final_score          (current hand-weighted score being replaced)
  features    : the 10 model features (see FEATURES)
  text        : chunk_text           (for feature engineering / inspection)

    python3 -m scripts.export_bakeoff_dataset --runs <r1,...> --out-dir reports/dataset
"""
from __future__ import annotations
import argparse
import csv
import json
import os

import scripts.a0_captured_separation as a0
from scripts.a0_captured_separation import build_run_table, _labeled, FEATURES, NEW_FEATURES

ID_COLS = ["run_id", "document_id", "doc_filename", "pass_name", "chunk_index",
           "candidate_key", "self_ref", "page_number", "token_count"]
LABEL_COLS = ["used"]
PROD_COLS = ["final_score"]
TEXT_COLS = ["chunk_text"]
# NEW_FEATURES appended after FEATURES, before chunk_text. Re-exports of OLD
# runs (capture predates Tasks 6/7/8/10) carry zeros for them — expected.
COLUMNS = ID_COLS + LABEL_COLS + PROD_COLS + list(FEATURES) + list(NEW_FEATURES) + TEXT_COLS


def _run_doc_map(runs):
    """run_id -> (document_id, filename) from postgres."""
    from sqlalchemy import create_engine, text
    eng = create_engine(a0._pg_url())
    out = {}
    try:
        with eng.connect() as c:
            for r in runs:
                row = c.execute(text(
                    "SELECT r.document_id, d.filename FROM ingest.pipeline_runs r "
                    "JOIN ingest.documents d ON d.id=r.document_id WHERE r.id=:r"),
                    {"r": r}).fetchone()
                out[r] = (str(row[0]), row[1]) if row else ("", "")
    finally:
        eng.dispose()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--target", default="lineage_grounded")
    ap.add_argument("--out-dir", default="reports/dataset")
    args = ap.parse_args(argv)
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    docmap = _run_doc_map(runs)
    per_doc = {}
    records = []
    for run in runs:
        rows, _ = build_run_table(run, target_mode=args.target)
        lab = _labeled(rows)
        ct = a0._fetch_chunk_text(run)
        doc_id, fname = docmap.get(run, ("", ""))
        n_pos = 0
        for r in lab:
            rec = {
                "run_id": r.run_id, "document_id": doc_id, "doc_filename": fname,
                "pass_name": r.pass_name, "chunk_index": r.chunk_index,
                "candidate_key": r.candidate_key, "self_ref": r.self_ref,
                "page_number": r.page_number, "token_count": r.token_count,
                "used": 1 if r.used else 0, "final_score": r.final_score,
                "chunk_text": (ct.get(r.chunk_index, "") or "").replace("\n", " ").strip(),
            }
            for f in list(FEATURES) + list(NEW_FEATURES):
                rec[f] = getattr(r, f, 0.0)
            records.append(rec)
            n_pos += int(bool(r.used))
        per_doc[run] = {"document_id": doc_id, "doc_filename": fname,
                        "candidates": len(lab), "positives": n_pos}

    # CSV (always)
    csv_path = os.path.join(args.out_dir, "bakeoff_dataset.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(records)

    # parquet (best-effort)
    parquet_path = os.path.join(args.out_dir, "bakeoff_dataset.parquet")
    parquet_ok = False
    try:
        import pandas as pd
        pd.DataFrame.from_records(records, columns=COLUMNS).to_parquet(parquet_path, index=False)
        parquet_ok = True
    except Exception as e:
        parquet_path = f"(skipped: {e})"

    total = len(records)
    total_pos = sum(d["positives"] for d in per_doc.values())
    meta = {
        "target": args.target,
        "n_candidates": total,
        "n_positives": total_pos,
        "n_docs": len(runs),
        "features": list(FEATURES),
        "new_features": list(NEW_FEATURES),
        "label_column": "used",
        "production_score_column": "final_score",
        "group_column_for_LODO": "run_id (one per document)",
        "per_doc": per_doc,
    }
    with open(os.path.join(args.out_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    readme = f"""# Value-grounded chunk-selection dataset

Self-contained export of the per-(pass,chunk) candidate dataset used for the
chunk-narrowing calibration. One row = one retrieval candidate scored for one
extraction pass on one document.

## Files
- `bakeoff_dataset.csv` — {total} rows, all columns (universally loadable)
- `bakeoff_dataset.parquet` — same, typed (if pandas/pyarrow available)
- `dataset_meta.json` — machine-readable summary (per-doc counts, feature list)

## Label
- `used` (0/1): 1 = this chunk is a **value-grounded source** for a field the
  pass extracted (target = `{args.target}`: a numeric+unit field value of one of
  the pass's OWN fields is found in the chunk text). This is the honest
  ground-truth for "did this chunk need to be sent to the LLM for this pass".
- {total_pos} positives / {total} candidates ({100*total_pos/max(total,1):.1f}%) — heavily imbalanced.

## Features (model inputs) — {len(FEATURES)} legacy + {len(NEW_FEATURES)} new columns
{", ".join(FEATURES)}
- `cosine` is the only metric that generalizes across document shapes
  (see memory: keyword features were SA-2-overfit). `is_table` is currently a
  dead stub (always 0 — Phase D / table_meta not yet wired).

New (capture-only) features — zeros on runs captured before Tasks 6/7/8/10:
{", ".join(NEW_FEATURES)}

## Other columns
- `final_score`: the current hand-weighted production score this calibration aims
  to replace (NOT a feature — it's a baseline to beat).
- `run_id` / `document_id` / `doc_filename`: identifiers. **Use `run_id` as the
  group key for leave-one-document-out CV** — never let same-doc chunks land in
  both train and test, or AUROC is inflated.
- `chunk_text`: the candidate chunk text (for new feature engineering, e.g. a
  real table/prose classifier).

## Per-document breakdown
{chr(10).join(f"- {d['doc_filename'][:55]:55s} candidates={d['candidates']:4d}  positives={d['positives']}" for d in per_doc.values())}

## Two LODO conventions (read this — they differ on this small/imbalanced data)
- **mean-per-fold** (what the analysis reports): train on N-1 docs, score the
  held-out doc, AUROC per held-out doc, then average. LogReg = **0.852**, GBM = 0.885.
- **pooled-OOF**: collect held-out probabilities across all folds, one AUROC over
  the pool. LogReg ≈ 0.706. Lower because docs with few positives dominate the pool.
Neither is "wrong" — just report which you used. The frontier/recall-savings
sweep uses pooled-OOF probabilities.

## Load + train (example — mean-per-fold, matches the analysis)
```python
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

df = pd.read_parquet("bakeoff_dataset.parquet")   # or read_csv
FEATURES = {list(FEATURES)!r}
X, y, g = df[FEATURES].values, df["used"].values, df["run_id"].values
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
aucs = []
gkf = GroupKFold(n_splits=len(set(g)))
for tr, te in gkf.split(X, y, g):
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2: continue
    m = clone(clf).fit(X[tr], y[tr])
    aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:,1]))
print("mean-per-fold LODO AUROC:", np.mean(aucs))   # ~0.852, matches analysis
```

Regenerate: `python3 -m scripts.export_bakeoff_dataset --runs <ids> --target {args.target}`
"""
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"wrote {total} rows ({total_pos} positives, {len(runs)} docs) to {args.out_dir}/")
    print(f"  CSV     : {csv_path}")
    print(f"  parquet : {parquet_path if parquet_ok else parquet_path}")
    print(f"  meta    : {args.out_dir}/dataset_meta.json")
    print(f"  README  : {args.out_dir}/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
