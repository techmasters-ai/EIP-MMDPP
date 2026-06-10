# Chunk-Selection Calibration — Handoff

**Purpose of this doc:** let a fresh session continue the per-pass chunk-selection
work without this conversation's context. Self-contained: goal, current findings,
data, tools, and prioritized next steps.

Worktree: `/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry`
Date of handoff: 2026-06-10.

---

## 1. The objective (North Star)

For each extraction pass, send the LLM **only the chunks that actually contain
that pass's content** — maximize recall of true-source chunks while dropping as
many irrelevant (pass, chunk) candidates as possible. Sending fewer chunks is the
dominant wall-time lever (each pass currently scans ~all chunks: 60–174 LLM
batches/pass). The rule must **generalize across any document shape** (radar,
missile, spec-sheet, prose, tabular, any air-defense system — not just SA-2) and
must preserve **complete data lineage** (every kept chunk traceable).

Concretely: replace the hand-weighted `final_score` (in
`app/services/extraction_candidate_scoring.py`) with a learned per-pass selection
rule — a score threshold or dynamic-k — calibrated on ground-truth.

## 2. Mental model / how the data is built

- Unit = one **(pass, chunk) candidate**: a chunk the vector router retrieved &
  scored for one extraction pass on one document.
- Each candidate has **10 features** + the production `final_score` + a label.
- The vector router captures per-candidate `score_components` into
  `pipeline_pass_outputs.diagnostics_json->'router'->'score_components_all'`.
- **Label** (`used`, target = `lineage_grounded`): 1 iff a numeric+unit value of
  one of the pass's OWN fields is found in the chunk text (value-grounding). This
  is the honest "did this chunk need to go to the LLM for this pass" truth.
- **Honest metric = leave-one-document-out (LODO)**, grouped by `run_id` (one run
  per doc). Never let same-doc chunks land in both train and test — it inflates
  AUROC badly.

The 10 FEATURES (exact order):
```
cosine, rerank_norm, field_label_norm, pass_keyword_norm, anchor_text_norm,
anchor_section_norm, section_norm, is_table, pattern_norm, negative_norm
```

## 3. Current corpus

8 docs, **1692 candidates, 35 positives (2.2% — heavily imbalanced)**. The new
Engagement doc dominates the positive set.

| doc | candidates | positives |
|---|---:|---:|
| Engagement and Fire Control Radars (radar/missile spec) | 450 | **24** |
| SA-2 Guideline (RU) | 450 | 5 |
| SA-2_and_SR-71 | 378 | 2 |
| Images_Demo_Doc | 144 | 1 |
| SA-2 National Museum | 90 | 1 |
| S-75 Dvina | 63 | 1 |
| V-75 SA-2 GUIDELINE | 54 | 1 |
| SNR-75 Wikipedia | 63 | 0 |

8 run IDs (group keys), exact run→doc mapping in `reports/dataset/dataset_meta.json`:
```
e864ba84-6b3c-41d8-8da3-688cb3034524  28a58eb9-f0f3-4b6b-8b07-f55df6efd5ba
58767f3f-5112-4816-9427-e61d5d8c068b  e79a4866-750b-41e3-a4b4-521f3e31ff26
ff35b0e2-6a8a-4444-8a07-ee8e85a49011  295aea8e-bc3f-4003-81da-7713705c5daa
de6f44d9-69b8-4d3c-bcb3-f0eaa487af19  1329caf5-d57b-403d-96ea-1399a7d3d67f  (Engagement)
```

## 4. KEY FINDINGS (read before experimenting)

1. **Keyword features are OVERFIT to the prose-heavy SA-2 corpus.** Adding one
   diverse doc (Engagement) collapsed `pass_keyword_norm` LODO AUROC **0.769 →
   0.208** (coef sign flipped). It fires on **11/11 SA-2 positives but only 5/24
   Engagement positives**. `field_label_norm`, `anchor_text_norm` collapsed too.

2. **`cosine` (semantic) is the only generalizable standalone signal** — LODO
   AUROC 0.746 → **0.857**, fires on 100% of positives and negatives on both
   corpora. Lean on it. `negative_norm` also held (0.566 → 0.718).

3. **Ranking transfers; the threshold does NOT.** Train on the 7 SA-2 docs, score
   Engagement held-out (`scripts/transfer_test.py`): AUROC 0.827 (ranking OK) but
   at the model's native P≥0.5 cut it catches only **1/24** positives. Re-pick the
   threshold on the new doc and its scores still give recall 1.0 @ 49% savings.
   ⇒ **a fixed global threshold cannot generalize — you need adaptive / per-doc
   dynamic-k.**

4. **False-negatives are table-derived positives** ("Engagement Radar Peak Power
   [kW] 180.0", "1 = ±45°"). cosine fires on them (~0.5–0.6) but keywords don't.
   8 of Engagement's 24 positives are table-derived (keyword fires 1/8).

5. **`is_table` is a DEAD STUB** — always 0.0 (AUROC 0.500, zero variance).
   `table_meta={}` is passed at every call site in `extraction_chunk_search.py`
   ("Phase D deferred"). Drop it from models until task #70 wires it. Tables ARE
   extracted (via per-pass table normalization + deterministic overlay) — `is_table`
   is only the *scoring* flag that's unwired.

6. **The keyword VOCABULARY is already air-defense-general** (magnetron, phased
   array, rocket motor, seeker, …) — no SA-2-specific terms (grep-verified). The
   overfit is **structural** (general keywords appear in prose, not in table cells),
   NOT vocabulary-specificity. Making keywords "more general" won't fix it.

7. **Honest frontier: recall 1.0 @ ~24% savings** (LogReg LODO), down from the
   overfit ~68% on SA-2-only. Best-AUROC model (GBM, LODO 0.885) is poorly
   calibrated at recall 1.0 (needs keep-100%); LogReg is better at that operating
   point. (Earlier "keep 2%" was an in-sample reporting bug — now fixed to LODO.)

## 5. Data + tools to pick up

**Portable dataset** (no DB needed) — `reports/dataset/`:
- `bakeoff_dataset.csv` / `.parquet` — 1692 rows, label `used`, group `run_id`,
  10 features + `final_score` + `chunk_text`.
- `README.md` — columns, the two LODO conventions (mean-per-fold 0.852 vs
  pooled-OOF 0.706 — report which you use), load+train example.
- `dataset_meta.json` — per-doc counts, feature list.
- Regenerate: `python3 -m scripts.export_bakeoff_dataset --runs <ids> --target lineage_grounded`

**Analysis scripts** (`scripts/`, run with `A0_DATABASE_URL=postgresql+psycopg2://eip:eip_secret@localhost:5437/eip`):
- `a0_captured_separation.py` — core: `build_run_table`, `model_bakeoff`,
  `per_feature_auroc`, `recall_threshold_sweep`, `_lodo_auroc`, `FEATURES`.
  CLI: `--fit-runs <ids> --target lineage_grounded --out-dir <dir>`.
- `per_metric_signal.py` — univariate signal + LogReg threshold per metric.
- `transfer_test.py` — train OLD runs, score NEW held-out run.
- `plot_recall_vs_savings.py` — frontier (pooled-OOF).
- `show_false_negatives.py` / `show_false_positives.py` — error diagnostics.
- `export_bakeoff_dataset.py` — regenerate the portable dataset.

Production scoring code (what you're replacing): `app/services/extraction_candidate_scoring.py`
(`final_score`) and `extraction_chunk_search.py` (where `merge_candidates` is called).

## 6. Prioritized next experiments

- **(A) Wire `is_table` (task #70) — highest-value single fix.** Thread the table
  modality (`extraction_chunk_index.py:800`, `chunking.py:118`) into `table_meta`
  so `is_table` fires on table candidates. Targets the 8/24 positives keywords
  can't reach. Then re-run `per_metric_signal` to see if it becomes a live signal.
- **(B) Dynamic-k / per-doc adaptive threshold.** Finding #3 proves fixed cuts
  don't transfer. Try per-(doc,pass) top-k by score, or a threshold relative to
  each pass's score distribution. The harness already has a score-threshold-sweep
  arm (task #61).
- **(C) Down-weight / drop lexical keyword features; rebuild around cosine.**
  Compare a cosine-led model (cosine + negative_norm + is_table-once-wired) vs the
  full 10. Use `--drop-features` on the frontier script.
- **(D) Hyperparameter tuning (task #65), nested-LODO** to avoid the optimism the
  current single-split bake-off has.
- **(E) Add more diverse non-SA-2 spec docs.** Corpus is still SA-2 + 1 Engagement.
  Each diverse doc exposes more overfit. (See task #68.)
- **(F) Feature pruning:** `section_norm` ≡ `anchor_section_norm` (identical column
  — effectively 9 distinct features); `is_table` dead until #70.
- **(G) New candidate features** engineered from `chunk_text`: a real table/prose
  classifier, digit-density, unit-token presence, numeric-near-label patterns —
  these may recover the table-derived positives more generally than `is_table`.

## 7. Constraints & gotchas (don't relearn the hard way)

- **Generalization guardrail:** no equipment names / instance names in any rule or
  keyword; operate on entity type, schema, role, modality — never literal "SA-2",
  "S-75", etc. Keywords must be air-defense-general.
- **Complete data lineage** is a hard product requirement — any selection must keep
  the kept chunks traceable to source.
- **Precision is uninformative at 2.2% prevalence** — judge by AUROC + recall +
  savings, not precision (it pins near 0.02 for everything).
- **Small sample (35 positives):** AUROC gaps < ~0.05 are noise; Engagement
  dominates (24/35) so it drives the pooled signal.
- **`final_score` is the baseline to beat, not a feature.** Don't train on it.
- **Two LODO conventions diverge** (mean-per-fold vs pooled-OOF) — always state which.
- The live extraction pipeline currently uses the hand-weighted `final_score`;
  none of this calibration is wired into production yet — it's still analysis.

## 8. Memory pointers (auto-loaded context)

- `project_keyword_overfit_cosine_generalizes.md` — the core finding (this doc's §4).
- `project_bakeoff_corpus_and_findings.md` — corpus definition, positive = (pass,chunk) cell.
- `project_hnsw_postfilter_starvation.md`, `project_threshold_selection_vs_topk.md`,
  `project_schema_wide_retrieval_plan.md` — related retrieval/threshold context.
- Tasks: #61 (harness), #65 (hyperparams), #68 (more docs), #70 (wire is_table).
