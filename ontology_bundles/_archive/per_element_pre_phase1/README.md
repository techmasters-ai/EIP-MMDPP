# Pre-Phase-1-promotion ontology bundle manifests

Archived on 2026-05-28 as part of Phase 1 Task 10.5 (bundle propagation +
merged-mode default).

## What's in here

Per-element-tuned `manifest.yaml` snapshots captured immediately before the
Task 10.5 propagation rewrote them with the uniform merged-mode retrieval
calibration `(min_similarity=0.35, top_n_candidates=50, top_k=15,
fallback_to_full=true)`.

| File | Source bundle | Pre-propagation retrieval calibration |
|---|---|---|
| `air_defense_v3.yaml`                  | `ontology_bundles/air_defense_v3`                  | `min_sim=0.35, top_n=50, top_k=30` (partial-prior-propagation state) |
| `air_defense_v3_baseline_subset.yaml`  | `ontology_bundles/air_defense_v3_baseline_subset`  | `min_sim=0.25, top_n=500, top_k=500` (C.7d threshold-only config) |
| `air_defense_v3_narrowing_v1.yaml`     | `ontology_bundles/air_defense_v3_narrowing_v1`     | `min_sim=0.25, top_n=300, top_k=50/60` (C.9b per-element knee) |

The `air_defense_v3_merged_v1` bundle was created in Task 9 already at the
target merged-mode calibration and is not archived here — it's the source of
truth that Task 10.5 propagated outward.

## Why these are kept

These snapshots are reference material for future A/B comparisons against
potential calibration updates. If a new sweep proposes different per-bundle
knees, restoring one of these manifests + a controlled reingest lets us
compare against the original per-element-tuned behavior without git-history
spelunking.

## How to revert a bundle

```bash
# Example: revert _narrowing_v1 to its pre-propagation manifest
cp ontology_bundles/_archive/per_element_pre_phase1/air_defense_v3_narrowing_v1.yaml \
   ontology_bundles/air_defense_v3_narrowing_v1/manifest.yaml

# Reverting the merged-mode default itself:
sed -i 's/^EXTRACTION_INDEX_MODE=merged$/EXTRACTION_INDEX_MODE=per_element/' .env.example
# (then update .env to match in your environment)
```

## Pointer to evidence

See `docs/handoffs/2026-05-28-merged-mode-bundle-propagation.md` for the
Phase 1 A/B evidence summary, the gate-override rationale (one of the two
recall gates partially failed at the time of propagation), and the
2-week retirement clock for per-element code paths.
