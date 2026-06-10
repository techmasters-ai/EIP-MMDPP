# Value-grounded chunk-selection dataset

Self-contained export of the per-(pass,chunk) candidate dataset used for the
chunk-narrowing calibration. One row = one retrieval candidate scored for one
extraction pass on one document.

## Files
- `bakeoff_dataset.csv` — 1692 rows, all columns (universally loadable)
- `bakeoff_dataset.parquet` — same, typed (if pandas/pyarrow available)
- `dataset_meta.json` — machine-readable summary (per-doc counts, feature list)

## Label
- `used` (0/1): 1 = this chunk is a **value-grounded source** for a field the
  pass extracted (target = `lineage_grounded`: a numeric+unit field value of one of
  the pass's OWN fields is found in the chunk text). This is the honest
  ground-truth for "did this chunk need to be sent to the LLM for this pass".
- 116 positives / 1692 candidates (6.9%) — heavily imbalanced.

## Features (model inputs) — 10 columns
cosine, rerank_norm, field_label_norm, pass_keyword_norm, anchor_text_norm, anchor_section_norm, section_norm, is_table, pattern_norm, negative_norm
- `cosine` is the only metric that generalizes across document shapes
  (see memory: keyword features were SA-2-overfit). `is_table` is currently a
  dead stub (always 0 — Phase D / table_meta not yet wired).

## Other columns
- `final_score`: the current hand-weighted production score this calibration aims
  to replace (NOT a feature — it's a baseline to beat).
- `run_id` / `document_id` / `doc_filename`: identifiers. **Use `run_id` as the
  group key for leave-one-document-out CV** — never let same-doc chunks land in
  both train and test, or AUROC is inflated.
- `chunk_text`: the candidate chunk text (for new feature engineering, e.g. a
  real table/prose classifier).

## Per-document breakdown
- SA-2 Guideline _ Зенитный Ракетный Комплекс С-75 Двина_ candidates= 450  positives=5
- SA-2_and_SR-71_17_Apr_2020.pdf                          candidates= 378  positives=11
- SA-2 Surface-to-Air Missile _ National Museum of the Un candidates=  90  positives=4
- SNR-75 - Wikipedia.pdf                                  candidates=  63  positives=0
- S-75 Dvina.pdf                                          candidates=  63  positives=1
- V-75 SA-2 GUIDELINE.pdf                                 candidates=  54  positives=1
- Images_Demo_Doc.pdf                                     candidates= 144  positives=1
- Engagement and Fire Control Radars (S-Band, X-band).pdf candidates= 450  positives=93

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
FEATURES = ['cosine', 'rerank_norm', 'field_label_norm', 'pass_keyword_norm', 'anchor_text_norm', 'anchor_section_norm', 'section_norm', 'is_table', 'pattern_norm', 'negative_norm']
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

Regenerate: `python3 -m scripts.export_bakeoff_dataset --runs <ids> --target lineage_grounded`
