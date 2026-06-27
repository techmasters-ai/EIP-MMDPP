# A0 model bake-off — 1692 chunks, 35 USED, 8 docs

| model | pooled CV AUROC | leave-one-doc-out AUROC |
|---|---|---|
| LogisticRegression | 0.860 | 0.869 |
| RandomForestClassifier | 0.852 | 0.879 |
| GradientBoostingClassifier | 0.867 | 0.858 |
| HistGradientBoostingClassifier | 0.807 | 0.881 ⬅ |
| MLPClassifier | 0.713 | 0.728 |

**best (leave-one-doc-out): HistGradientBoostingClassifier** — LODO is the honest cross-document number (each doc held out once); pooled CV overstates because same-doc chunks can land in both train and test folds.
- best model recall≥1.00 (leave-one-doc-out): keep 100% of chunks (recall 1.00, threshold 0.000)
