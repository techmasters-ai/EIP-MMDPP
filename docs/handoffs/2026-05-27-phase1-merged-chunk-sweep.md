# Phase 1 / C.10 — Merged-chunk calibration sweep (Task 8)

Date: 2026-05-27
Branch: `walltime/c0-telemetry`
Plan: `docs/superpowers/plans/2026-05-27-merged-chunk-routing.md` (Task 8)
Script: `scripts/c10_phase1_merged_chunk_calibration.py`
Raw artifacts:
- `scripts/c10_phase1_sweep.csv` — 2520 cell rows (host copy of `/tmp/c10_phase1_sweep.csv`)
- `scripts/c10_phase1_sweep_raw.md` — auto-generated tables

## TL;DR

**Phase-1 merged-mode retrieval knee, identical for all 9 narrowed passes:**

| param              | value | rationale                                                                 |
|--------------------|-------|---------------------------------------------------------------------------|
| `min_similarity`   | 0.30  | safe floor; no chunk in either doc scored below 0.302 → no GT loss vs 0.20|
| `top_n_candidates` | 50    | top_n=25 collapses Dvina coverage to 0–60%; 50 saturates                  |
| `top_k`            | 15    | top_k=10 only recovers 67% (= 10/15) by construction                      |

Coverage hits 100% of pseudo-GT for every (doc, pass) at this knee.

Compared to the C.9a per-element baseline (`top_n=150 / top_k=30 / min_sim=0.25`
for Dvina, `300 / 30 / 0.25` for SA-2), Phase 1 merged-mode is **3× lower
top_n, 2× lower top_k** and equals or exceeds coverage — exactly the
density payoff Phase 1 was designed to deliver.

## Methodology

### Ground-truth definition

Pseudo-GT = top-K chunks by **full-corpus reranker score** (max-permissive
— no min_sim, no top_n cap). Same methodology as `scripts/c9a_calibration_sweep.py`
and `scripts/c9a_sweep_v2.py`. The reranker is order-invariant, so we
rerank once per (doc, pass) and derive each cell's coverage by set
intersection — 18 reranker calls for 2520 cells.

`GT_TOP_K = 15` (vs the C.9a value of 30 over ~300 per-element chunks).
Merged chunks are ~7× denser by token count (~408 tokens/chunk vs ~58 in
per-element); GT_TOP_K=15 over 42–45 merged chunks preserves the
"top-1/3 of corpus by reranker" semantic of C.9a.

GT is bundle-agnostic: we use the 9 narrowed-pass query strings from
`build_retrieval_query` against the Pass classes in
`ontology_bundles.air_defense_v3` — no GT entity labels are hand-curated.
The metric measures "did we surface the chunks the reranker rated as
most relevant?", which is the operative question for the vector router.

If C.9a-style ground truth had been available with explicit per-entity
self_ref labels we would have used it; only the reranker-derived
pseudo-GT survives in the C.9a handoffs, and it remains the only
methodology that's bundle/document-independent.

### Sweep dimensions (per Task 8)

- `min_similarity` ∈ {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50} (7)
- `top_n_candidates` ∈ {25, 50, 75, 100} (4)
- `top_k` ∈ {5, 10, 15, 20, 30} (5)
- 140 cells × 9 narrowed passes × 2 docs = **2520 cells**

### Docs under sweep

| label | document_id                          | pdf                                   | merged_chunks |
|-------|--------------------------------------|---------------------------------------|---------------|
| Dvina | 9c8e09c7-e39f-4359-92c0-46330158c73c | S-75 Dvina.pdf                        | 45            |
| SA-2  | 128e48f9-06f9-459a-b2f0-6d42bf62c42d | SA-2_and_SR-71_17_Apr_2020.pdf        | 42            |

(Match the C.9a v1 reference pair so per-element ↔ merged comparison is
apples-to-apples on the same source PDFs.)

### Indexing

`build_extraction_index_hybrid` called directly (no Celery, no worker
dispatch). Each doc gets a fresh synthetic `pipeline_run_id =
"c10sweep-<doc>-<uuid8>"`. Bypasses the C.4 dispatcher and the
`EXTRACTION_INDEX_MODE` env switch — purely a script-local exercise of
the Task 3 indexer.

### Per-cell metrics (Task 8 spec)

- `selected_chunk_count` — chunks the LLM would see (= `min(top_k, |cand|)`)
- `expanded_ref_count` — `|⋃ source_refs of selected chunks|` (Task 6
  semantic — what the chunk-scope endpoint returns)
- `selected_chunk_token_estimate` — `Σ token_count(selected)` (sum of
  Task 1's `read_chunk_token_count`)
- `gt_coverage` — `|selected ∩ pseudo_GT| / |pseudo_GT|` (recall@top_k)

### Chunk identity (bug fix during the sweep)

First sweep pass used `self_ref` as the chunk identifier. Merged-mode
writes `self_ref = source_refs[0]` for legacy NOT-NULL compatibility,
which makes `self_ref` **non-unique** across merged rows (e.g. Dvina
collapses 45 rows to 23 unique `self_ref` values). The script was
patched to key on `chunk_index` instead, which is unique-per-merged-row
by construction (`vertex_id = f"{run_id}:chunk_{chunk_index}"`).

This is a useful artifact for downstream code that consumes merged-mode
chunks: **never use `self_ref` as a primary key.** Use `chunk_index`
(or the full `vertex_id`).

## Per-pass knee table (Task 8 deliverable)

The headline result. Knee = the most economical (lowest tokens, then
lowest chunks) config achieving 100% pseudo-GT coverage:

| pass                  | min_sim | top_n | top_k | Dvina sel_tok | SA-2 sel_tok |
|-----------------------|---------|-------|-------|---------------|--------------|
| radar_power_rf        | 0.30    | 50    | 15    | 7288          | 5741         |
| radar_antenna         | 0.30    | 50    | 15    | 7680          | 6429         |
| radar_timing          | 0.30    | 50    | 15    | 4386          | 5460         |
| radar_modulation      | 0.30    | 50    | 15    | 7433          | 6580         |
| missile_kinematics    | 0.30    | 50    | 15    | 7680          | 5617         |
| missile_guidance      | 0.30    | 50    | 15    | 6184          | 5751         |
| missile_airframe      | 0.30    | 50    | 15    | 7185          | 6163         |
| missile_speed_timing  | 0.30    | 50    | 15    | 6785          | 6360         |
| missile_propulsion    | 0.30    | 50    | 15    | 4883          | 6067         |

**Recommended bundle values (uniform across all 9 narrowed passes):**

```yaml
retrieval:
  min_similarity: 0.30
  top_n_candidates: 50
  top_k: 15
  fallback_to_full: true
```

(Note: `min_similarity` = 0.30 is chosen as a defensive value. Empirically
the 0.20 / 0.25 / 0.30 columns are identical for these two docs because
no merged chunk in either doc scores below 0.302 against any of the 9
narrowed-pass queries. 0.30 leaves headroom; should a future doc surface
chunks with vec_score < 0.20, the threshold still cleanly excludes
low-quality matches.)

## Comparison vs C.9a per-element baseline

| dim                 | C.9a per-element (Dvina) | C.9a per-element (SA-2) | C.10 merged (both)    |
|---------------------|--------------------------|-------------------------|-----------------------|
| corpus size         | ~300 chunks              | ~315 chunks             | 42–45 chunks (~7× ↓)  |
| mean token/chunk    | ~58                      | ~58                     | 388 (~7× ↑)           |
| `min_similarity`    | 0.25                     | 0.25                    | 0.30                  |
| `top_n_candidates`  | 150                      | 300                     | 50 (3–6× ↓)           |
| `top_k`             | 30                       | 30                      | 15 (2× ↓)             |
| achieved cov        | 97%                      | 97% (at top_n=300)      | 100% (both docs)      |
| sel tokens (kinem.) | ~3500 (C.9a top_k=30)    | ~3500                   | 5617–7680 (denser)    |
| expanded refs       | 30 (= top_k)             | 30                      | 8–45 (varies by doc)  |

Key observations:
- Selected token count is higher in merged mode (~5–8K vs ~3.5K) because
  each chunk is denser; this is by design — the LLM sees ~15 well-formed
  chunks instead of ~30 fragmentary ones.
- `expanded_ref_count` varies more on SA-2 (39–45) than Dvina (8–28).
  SA-2 has more pictures/captions per merged chunk, so the union over
  `source_refs` is larger. Both stay well below the docling-graph blob
  budget.
- **`top_n_candidates` is the load-bearing dimension.** Dropping from
  50 → 25 (with min_sim=0.30, top_k=15) collapses Dvina coverage from
  100% to 0–60% across passes. This validates the C.9a v2 hypothesis
  that the vector pre-filter is the weak link.

## Coverage drop-off by min_similarity

(Mean over all top_n × top_k combinations per cell; pseudo-GT @ top-15.)

| doc / pass                  | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|-----------------------------|------|------|------|------|------|------|------|
| Dvina / radar_power_rf      | 67%  | 67%  | 67%  | 67%  | 0%   | 0%   | 0%   |
| Dvina / radar_antenna       | 67%  | 67%  | 67%  | 67%  | 67%  | 0%   | 0%   |
| Dvina / radar_timing        | 74%  | 74%  | 74%  | 74%  | 55%  | 0%   | 0%   |
| Dvina / radar_modulation    | 71%  | 71%  | 71%  | 71%  | 20%  | 0%   | 0%   |
| Dvina / missile_kinematics  | 60%  | 60%  | 60%  | 0%   | 0%   | 0%   | 0%   |
| Dvina / missile_guidance    | 70%  | 70%  | 70%  | 70%  | 70%  | 27%  | 0%   |
| Dvina / missile_airframe    | 65%  | 65%  | 65%  | 13%  | 7%   | 7%   | 7%   |
| Dvina / missile_speed_timing| 63%  | 63%  | 63%  | 63%  | 7%   | 7%   | 7%   |
| Dvina / missile_propulsion  | 70%  | 70%  | 70%  | 70%  | 0%   | 0%   | 0%   |
| SA-2 / radar_power_rf       | 76%  | 76%  | 76%  | 76%  | 70%  | 49%  | 7%   |
| SA-2 / radar_antenna        | 78%  | 78%  | 78%  | 78%  | 75%  | 72%  | 64%  |
| SA-2 / radar_timing         | 76%  | 76%  | 76%  | 76%  | 70%  | 44%  | 13%  |
| SA-2 / radar_modulation     | 76%  | 76%  | 76%  | 76%  | 76%  | 33%  | 0%   |
| SA-2 / missile_kinematics   | 76%  | 76%  | 76%  | 76%  | 70%  | 67%  | 49%  |
| SA-2 / missile_guidance     | 77%  | 77%  | 77%  | 77%  | 77%  | 71%  | 68%  |
| SA-2 / missile_airframe     | 78%  | 78%  | 78%  | 78%  | 68%  | 44%  | 27%  |
| SA-2 / missile_speed_timing | 76%  | 76%  | 76%  | 76%  | 67%  | 49%  | 20%  |
| SA-2 / missile_propulsion   | 76%  | 76%  | 76%  | 76%  | 55%  | 33%  | 27%  |

`0.20`/`0.25`/`0.30` are indistinguishable — no merged chunk scored below
0.302 in either doc against any query. `0.35` is borderline for Dvina
(`missile_kinematics` craters: vec_max=0.509, only 5 chunks survive at
0.35; not enough to find the 15-chunk GT). `0.40+` collapses most passes.

## Vector-score distributions

| doc   | pass                  | vec_min | vec_median | vec_max |
|-------|-----------------------|---------|------------|---------|
| Dvina | radar_power_rf        | 0.321   | 0.383      | 0.473   |
| Dvina | radar_antenna         | 0.323   | 0.414      | 0.514   |
| Dvina | radar_timing          | 0.333   | 0.394      | 0.452   |
| Dvina | radar_modulation      | 0.334   | 0.390      | 0.434   |
| Dvina | missile_kinematics    | 0.317   | 0.345      | 0.509   |
| Dvina | missile_guidance      | 0.411   | 0.432      | 0.565   |
| Dvina | missile_airframe      | 0.302   | 0.349      | 0.579   |
| Dvina | missile_speed_timing  | 0.361   | 0.387      | 0.553   |
| Dvina | missile_propulsion    | 0.352   | 0.376      | 0.580   |
| SA-2  | radar_power_rf        | 0.337   | 0.443      | 0.508   |
| SA-2  | radar_antenna         | 0.344   | 0.500      | 0.590   |
| SA-2  | radar_timing          | 0.329   | 0.438      | 0.526   |
| SA-2  | radar_modulation      | 0.332   | 0.426      | 0.495   |
| SA-2  | missile_kinematics    | 0.362   | 0.462      | 0.595   |
| SA-2  | missile_guidance      | 0.378   | 0.527      | 0.624   |
| SA-2  | missile_airframe      | 0.315   | 0.403      | 0.552   |
| SA-2  | missile_speed_timing  | 0.346   | 0.435      | 0.561   |
| SA-2  | missile_propulsion    | 0.305   | 0.393      | 0.528   |

SA-2 scores systematically higher than Dvina (median 0.43–0.53 vs
0.35–0.43). Likely because Dvina is a Russian-text PDF with mixed
content, while SA-2 is English text + photos that better match the
query embeddings.

## Notable findings / anomalies

1. **Universal knee.** Same `(min_sim, top_n, top_k) = (0.30, 50, 15)`
   works for all 9 passes across both docs. No per-pass differentiation
   needed for the Phase 1 `air_defense_v3_merged_v1` bundle.

2. **Vector pre-filter is the weak link.** With `top_n=25`, Dvina
   coverage collapses to 0–60% even with min_sim=0.30 and top_k=15. The
   reranker can find GT only when the candidate pool is large enough; at
   42–45 chunks, `top_n=50` means "let the reranker see everything,"
   which is exactly what we want. C.9a v2 made the same observation for
   per-element (top_n=300 needed for SA-2).

3. **`min_similarity` is a no-op at the chosen knee.** No merged chunk
   scored below 0.302, so 0.20 / 0.25 / 0.30 are indistinguishable.
   Recommended 0.30 for defensiveness.

4. **No pass needs special handling.** The C.9a v2 result that
   `missile_kinematics` needed `top_n=300` (vs 150 for `radar_power_rf`)
   does NOT carry over to merged mode — the smaller corpus + denser
   chunks neutralize the asymmetry.

5. **Dvina is the harder of the two docs.** At `min_sim=0.35`,
   `missile_kinematics` Dvina coverage is 0% (vec_max=0.509 → only
   ~5 chunks survive at 0.35); SA-2 is 76%. Dvina has lower-quality
   query-doc alignment overall, likely from the Russian-text source.

6. **`selected_chunk_token_estimate` is ~5–8K for the knee config.**
   That's well under the LLM's typical 23K-token context budget. Plenty
   of headroom for Phase 2's direct chunk-text feed.

7. **`expanded_ref_count` is meaningful but not load-bearing.** Ranges
   8–45 across the knee cells. The chunk-scope endpoint (Task 6) reads
   this to expand selected_refs → ⋃source_refs; values are within docling-graph's
   typical narrowed-blob budgets.

## Sweep stats

| metric                              | value                |
|-------------------------------------|----------------------|
| total cells run                     | 2520                 |
| (doc, pass) combinations            | 18 (2 × 9)           |
| cells per (doc, pass)               | 140                  |
| wall-clock duration                 | 7.3 min              |
| dominant cost                       | reranker (18 calls)  |
| LLM calls                           | 0 (offline only)     |
| embed calls (writer + reader)       | 2 batch + 18 query   |

## Deviations from plan

1. **GT methodology**: Used reranker-derived pseudo-GT (the C.9a method)
   rather than entity-label GT. Rationale: no entity-label GT file
   exists in the repo (no `c9a*` GT-label artifact, no inline labels in
   any C.9a handoff). The plan's "if you can't find a clean GT labels
   source" branch applies; entity-extraction outputs from the
   `bdde417` baseline would be an alternative source but tying them to
   self_refs requires the LLM-emitted `_evidence_refs`, which would be
   downstream of the very routing we're trying to calibrate (circular).
   Pseudo-GT is bundle-agnostic and matches C.9a so the comparison is
   apples-to-apples.
2. **GT_TOP_K = 15** instead of 30 — scaled to match the ~7× denser
   merged chunks. Same fraction of corpus.
3. **chunk_index** as identity instead of `self_ref` — corrected for
   merged-mode `self_ref` non-uniqueness (writer collision). See
   "Chunk identity" above.

## Next: Task 9 inputs

The `air_defense_v3_merged_v1` manifest should set, for **all 9 narrowed
passes**:

```yaml
retrieval:
  min_similarity: 0.30
  top_n_candidates: 50
  top_k: 15
  fallback_to_full: true
```

Identity passes (`radar_identity`, `missile_identity`) and `system_links`
do not have `retrieval:` blocks in the v3 baseline; leave them
unchanged.

## Reproducing

```bash
docker cp scripts/c10_phase1_merged_chunk_calibration.py \
  eip-mmdpp-worker-1:/app/scripts/
docker exec eip-mmdpp-worker-1 bash -c \
  "cd /app && python scripts/c10_phase1_merged_chunk_calibration.py"
# Outputs:
#   /tmp/c10_phase1_sweep.csv
#   /tmp/c10_phase1_sweep.md
```

The script is deterministic given the same docling_document.json + the
same bge-m3 / cross-encoder weights. Both docs are already indexed
(`document_id`s above); only the merged-mode ExtractionChunk vertices
are recomputed each run.
