# Phase 1 Task 10.5 — merged-mode bundle propagation handoff

**Date:** 2026-05-28
**Plan:** `docs/superpowers/plans/2026-05-27-merged-chunk-routing.md`
**Branch:** `walltime/c0-telemetry`

## What changed

| Change | File(s) |
|---|---|
| Uniform merged-mode retrieval calibration `(min_sim=0.35, top_n=50, top_k=15, fallback_to_full=true)` propagated to all narrowed passes in 3 bundles | `ontology_bundles/air_defense_v3/manifest.yaml` (9 narrowed passes), `ontology_bundles/air_defense_v3_baseline_subset/manifest.yaml` (2), `ontology_bundles/air_defense_v3_narrowing_v1/manifest.yaml` (2) |
| Default `EXTRACTION_INDEX_MODE` flipped `per_element` → `merged` | `.env.example` |
| Pre-propagation manifests archived | `ontology_bundles/_archive/per_element_pre_phase1/` (3 yaml + README) |
| Initial LLM token cap raised `16384` → `32768` to reduce truncation-retry overhead on merged batches | `.env`, `.env.example` (DOCLING_GRAPH_LLM_MAX_TOKENS) |

## Phase 1 A/B evidence summary

The Task 10 A/B was contaminated by concurrent contention on the first attempt
(Dvina + SA-2 dispatched 2 seconds apart, sharing the 2-host Ollama pool).
Serial reruns were initiated for clean comparison. The serial runs gave:

### Dvina solo rerun (run_id `0a868956-ac4e-4a61-a64f-79fd9005f842`)

| Pass | Baseline `cfcc9539` | Phase 1 rerun #2 | Δ |
|---|---|---|---|
| radar_identity | 1 ent / 5.9m | 1 / 8.3m | flat |
| missile_identity | 1 ent / 12.1m | 1 / 15.5m | flat |
| missile_kinematics (narrowed) | 1 ent / 4.9m | **2 / 11.6m** | **+100% recall** ✓ |
| radar_power_rf (narrowed) | 1 ent / 7.4m | 0 / 6.8m (4 FAILED) | **-100% recall** ✗ |
| system_links | 1 rel / 2.5m | 2 / 2.4m | +1 rel |
| **TOTAL** | **4 ents, 1 rel / 14.6m** | **4 ents, 2 rels / 17.9m** | net flat ents, +23% wall |

### SA-2 solo rerun (run_id `6fc30668-cf27-481f-8f82-e44a06b07fca`, aborted at 142m elapsed)

| Pass | Baseline `7d46c487` | Phase 1 rerun | Δ |
|---|---|---|---|
| radar_power_rf (narrowed) | 22 ents / 14.3m | **24 / 19.8m** | **+9% recall** ✓ (but below +50% gate) |
| missile_kinematics (narrowed) | 16 ents / 24.5m | **28 / 96.6m** | **+75% recall** ✓✓ smashes gate |
| radar_identity | 25 ents / 111m | aborted in flight | n/a |
| missile_identity | 45 ents / 200m | aborted in flight | n/a |
| system_links | 57 rels / 28m | not run | n/a |

The SA-2 wall was inflated by 7 LLM-output-truncation events (each costing
~25-30 minutes on the retry ladder), prompting the `DOCLING_GRAPH_LLM_MAX_TOKENS`
bump to 32k. Even with that bump, gemma4:31b still hit the 32k cap on some
batches — Phase 2's structured/multi-channel retrieval is expected to shrink
per-batch input enough that this stops happening.

## Gate override rationale

The plan's Task 10.5 trigger required `≥+50% recall on both radar_power_rf
AND missile_kinematics`. Outcome:
- missile_kinematics: +75% / +100% on SA-2 / Dvina — passes ✓✓
- radar_power_rf: +9% on SA-2 / -100% on Dvina — **fails** ✗

User directed propagation despite the gate failure based on:
1. Strong kinematics signal on both docs (+75% / +100%) validates the
   merged-mode hypothesis for the higher-leverage extraction target.
2. Dvina has effectively no RF content (the SA-2 short summary doc) so its
   radar_power_rf signal is noise-bounded — a 1↔0 entity swing is not a
   meaningful regression.
3. SA-2 radar_power_rf is positive (+9%), even if below the +50% target.
4. Wall-time gate is not evaluable due to truncation overhead unrelated to
   merged-mode itself; serial-rerun pattern proved that contention not
   merged-mode was the original wall-blowup cause.

The override is documented here rather than in code so that a future revert
(via the archived manifests + `.env.example` flip) can be performed
deterministically if the Phase 1 hypothesis turns out wrong in production.

## How to use any bundle in merged mode now

```bash
# Any bundle works as merged-mode source; the EXTRACTION_INDEX_MODE env var
# is the global switch and is now 'merged' by default in .env.example.
# Per-bundle retrieval calibration is identical across all 4 bundles, so
# choice of bundle no longer affects retrieval knee — only which passes
# are dispatched.

# Example: reingest under merged mode with the production bundle
curl -X POST http://localhost:8000/v1/documents/<DOC_ID>/reingest \
  -H 'Content-Type: application/json' \
  -d '{"mode": "graph_only", "ontology_bundle_key": "air_defense_v3"}'
```

## How to revert if Phase 1 turns out wrong

```bash
# 1. Restore the 3 pre-propagation manifests
cp ontology_bundles/_archive/per_element_pre_phase1/air_defense_v3.yaml \
   ontology_bundles/air_defense_v3/manifest.yaml
cp ontology_bundles/_archive/per_element_pre_phase1/air_defense_v3_baseline_subset.yaml \
   ontology_bundles/air_defense_v3_baseline_subset/manifest.yaml
cp ontology_bundles/_archive/per_element_pre_phase1/air_defense_v3_narrowing_v1.yaml \
   ontology_bundles/air_defense_v3_narrowing_v1/manifest.yaml

# 2. Flip the env default back
sed -i 's/^EXTRACTION_INDEX_MODE=merged$/EXTRACTION_INDEX_MODE=per_element/' .env .env.example

# 3. Force-recreate workers to re-read .env
docker compose -p eip-mmdpp up -d --force-recreate worker-graph api docling-graph

# 4. Verify
docker exec eip-mmdpp-worker-graph-1 sh -c 'echo $EXTRACTION_INDEX_MODE'
# should print: per_element
```

The `air_defense_v3_merged_v1` bundle itself is not touched by reverts — it
remains the merged-mode reference bundle for future A/B work.

## 2-week retirement clock for per-element code paths

Per the plan's "Concrete backwards-compat removal trigger" section,
`extraction_index_mode` Settings field + `per_element` code paths
(`build_extraction_index`, `_walk_docling_elements`, `_render_text_chunk`,
per-element branches in Task 4/6 wires) become eligible for deletion when:

1. ~Phase 1 A/B passes both gates~ — overridden as documented above.
2. Phase 2 A/B passes both gates on Dvina + SA-2.
3. **2 consecutive weeks** of `EXTRACTION_INDEX_MODE=merged` in production
   with no open bug tagged `extraction-per-element`.

**Clock start: 2026-05-28.** Earliest deletion: **2026-06-11** (subject to
Phase 2 A/B completion).

Author of the deprecation PR runs a final pre-merge check:
```bash
grep -ri "per_element" app/services/ app/workers/ app/api/
```
Should return only the intended deletion sites, plus a 2-minute scan of
bugs filed in the prior 14 days.

## Verification

Bundle smoke test passed:
```
air_defense_v3                  → 9 narrowed passes, all (0.35, 50, 15, true)
air_defense_v3_baseline_subset  → 2 narrowed passes, all (0.35, 50, 15, true)
air_defense_v3_narrowing_v1     → 2 narrowed passes, all (0.35, 50, 15, true)
air_defense_v3_merged_v1        → 2 narrowed passes, all (0.35, 50, 15, true)
```

Sibling-allowlist sweep:
- `app/services/extraction_metrics.py:AIR_DEFENSE_BUNDLE_KEYS` — includes all 4
- `docker/docling-graph/app/evidence_gate.py:_AIR_DEFENSE_BUNDLE_KEYS` — includes all 4
- No other enumeration sites need additions

## Next steps

1. **Phase 2 (Tasks 11-12)**: Worker forwards `SelectedChunk.text` directly to
   docling-graph chunked path, eliminating docling-graph's re-chunking step
   (the architectural finding from this session: today the LLM sees 22/20
   chunks not the router's 15 selected chunks, because docling-graph
   re-chunks the scoped doc independently).
2. **Schema-wide retrieval upgrade**: see
   `docs/superpowers/plans/2026-05-28-schema-wide-retrieval-routing-upgrade.md`
   for structured per-field queries + multi-channel candidate generation +
   table-aware chunks. This supersedes parts of Phase 2 as originally specified.
3. **Run final Phase 1 SA-2 to completion** (only 3 of 5 passes terminated in
   the aborted rerun) if a clean wall number on identity-heavy passes is
   needed for the per-element retirement decision.
