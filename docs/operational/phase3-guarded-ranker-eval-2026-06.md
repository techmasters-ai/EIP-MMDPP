# Phase 3 — Guarded-Ranker Evaluation (dataset_v2)

Date: 2026-06-17
Plan: `docs/superpowers/plans/2026-06-10-chunk-selection-guarded-ranker.md` (Task 17)
Dataset: `reports/dataset_v2/` (CSV, parquet, `dataset_meta.json`, `eval_guarded_ranker.json`)

## 1. Corpus

Fresh 8-doc re-collection with the full Phase 0–2 feature set live (post field-value-grounding fix; `score_components_all` captured per field-group pass). Each run carries 12 passes; 9 per doc emit `score_components_all`.

| run_id | document | candidates | positives |
|---|---|---|---|
| 02f2981a | Engagement and Fire Control Radars (S-/X-band) | 848 | 45 |
| 4f53397c | SA-2 Guideline С-75 Двина/Десна/Волхов | 475 | 5 |
| 39adb2f1 | SA-2_and_SR-71_17_Apr_2020 | 378 | 3 |
| 6b63d765 | Images_Demo_Doc | 144 | 1 |
| 003abaec | SA-2 Surface-to-Air Missile (NMUSAF) | 90 | 1 |
| ba94a76a | S-75 Dvina | 63 | 1 |
| ef8d54d2 | SNR-75 (Wikipedia) | 63 | 0 |
| 3c0424bf | V-75 SA-2 Guideline | 54 | 1 |
| **total** | **8 docs** | **2,115** | **57** |

Runs: 4 docs re-run post-deploy (Engagement 02f2981a, V-75 3c0424bf, SNR-75 ef8d54d2, Dvina ba94a76a); 4 from the 2026-06-11 collection, unaffected by the fallback fix. `score_components_all` coverage is uniform (9/9 field-group passes per doc).

**Corpus skew (caveat):** Engagement holds 45/57 positives (79%) and 40% of candidates; SNR-75 contributes a zero-positive fold. LODO is therefore Engagement-weighted, and the corpus is SA-2-family throughout. Treat absolute frontier numbers as in-corpus; the LODO AUROC is the generalization signal.

## 2. Method

- **Target / label:** `used` (lineage-grounded selection), `lineage_grounded` in meta.
- **Recall floor:** label-aligned OR-gates `unit_gate ∪ table_gate`. Coverage is a literal precondition: `check_gate_coverage` → **57/57 used==1 rows covered** (exit 0).
- **Ranker:** sign-constrained L2 logistic regression over the feature set; wrong-signed coefficients dropped and refit.
- **Selection under test:** `guarded` = (gate-flagged ∪ ranker score ≥ q-quantile of the pool). Compared against `calibrated_only` (ranker, no gates) and the production `final_score` top-k baseline.
- **Cross-validation:** GroupKFold by `run_id` (leave-one-document-out). Reported under **both conventions** — pooled out-of-fold and mean-per-fold.

## 3. Generalization (LODO)

| Convention | AUROC |
|---|---|
| Pooled out-of-fold | **0.916** |
| Mean per-fold | 0.909 |

7 folds scored; 1 skipped (SNR-75, single-class / 0 positives). Strong, document-transferable separation.

## 4. Frontier (all rows below at recall 1.0 unless noted)

### Guarded (gate ∪ ranker) — recall 1.0 at every quantile

| q | recall | kept_frac | gate-only kept | ranker-added (kept − gate) |
|---|---|---|---|---|
| 0.50 | 1.0 | 0.673 | 0.474 | 0.199 |
| 0.70 | 1.0 | 0.587 | 0.474 | 0.113 |
| 0.80 | 1.0 | 0.556 | 0.474 | 0.082 |
| 0.90 | 1.0 | 0.530 | 0.474 | 0.056 |
| **0.95** | **1.0** | **0.520** | 0.474 | 0.046 |

### Gates alone (the recall floor)

`gate_only_kept_frac = 0.474` at **recall 1.0** → **52.6% of candidates pruned with zero recall loss.**

### Calibrated-only (ranker, no gates) — never reaches recall 1.0

| q | recall | kept_frac |
|---|---|---|
| 0.50 | 0.895 | 0.506 |
| 0.70 | 0.877 | 0.318 |
| 0.85 | 0.632 | 0.191 |
| 0.95 | 0.368 | 0.112 |

### Production baseline (`final_score` top-k)

Tops out at **recall 0.474 @ k=50** (keeps 80%); recall 0.32 at k=15. The current top-k-by-final_score selection is far below the guarded approach at any operating point.

## 5. Cost table (interpretation)

- **Gates do the pruning:** 47.4% kept at recall 1.0 → a 52.6% reduction in chunks sent to the LLM, with every labeled positive retained.
- **The ranker is a generalization margin, not a pruner here:** since the gates already cover all labeled positives, the union only *adds* chunks (q=0.95: +4.6 pts → 52.0% kept). That margin is the defense for unseen docs where a positive may not trip a gate — justified by the 0.916 AUROC, not by in-corpus savings.
- **Gates are load-bearing:** the ranker alone never hits recall 1.0 (≤0.895). Removing the gates breaks the floor.

## 6. Feature model

Kept (sign-valid): `max_field_cosine`, `mean_top3_field_cosine`, `rerank_norm`, `unit_token_count`, `digit_density`.

Dropped by sign constraint: `cosine` (coef −0.06 — the field-cosine features absorb the dense signal, plain cosine goes redundant/negative), `is_table` (−0.29), `label_value_lines` (−0.27), `negative_norm` (wrong sign +0.08).

## 7. Verdict / Phase-4 input

The guarded-ranker design holds: **recall 1.0 at 47–53% chunk savings, LODO AUROC 0.916**, vastly better than the production top-k baseline (≤0.47 recall). Gate coverage is a clean 57/57. The design is ready to calibrate (Task 19: lock `ranker_weights` + `quantile_q` + finite-sample margin) and to wire behind `selection_mode` (Task 18), with bundle propagation defaulting to `topk` until the Phase-4 gate flip (Task 20).

**Caveats carried forward:** Engagement-weighted corpus; SNR-75 zero-positive fold; SA-2-family only; wall-clock from these runs is not an idle-pool baseline (the pool was contended) — but wall-time does not enter this recall/AUROC eval.
