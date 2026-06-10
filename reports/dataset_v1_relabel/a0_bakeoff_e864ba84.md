# A0 model bake-off — 1692 chunks, 116 USED, 8 docs

| model | pooled CV AUROC | leave-one-doc-out AUROC |
|---|---|---|
| LogisticRegression | 0.746 | 0.536 |
| RandomForestClassifier | 0.863 | 0.705 ⬅ |
| GradientBoostingClassifier | 0.849 | 0.697 |
| HistGradientBoostingClassifier | 0.830 | 0.666 |
| MLPClassifier | 0.772 | 0.551 |

**best (leave-one-doc-out): RandomForestClassifier** — LODO is the honest cross-document number (each doc held out once); pooled CV overstates because same-doc chunks can land in both train and test folds.
- best model recall≥1.00 (leave-one-doc-out): keep 100% of chunks (recall 1.00, threshold 0.000)
