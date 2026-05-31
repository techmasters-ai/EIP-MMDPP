# Batch-Anchored Lineage for Truncated / Salvaged Extraction — Design

- **Date:** 2026-05-31
- **Status:** Design (awaiting user review → implementation plan)
- **Worktree:** `walltime/c0-telemetry`
- **Related:** [[project_extracted_from_root_cause]], [[project_catchall_worker_stale_code_trap]],
  prior `2026-05-30-element-uid-lineage-fix-design.md` (Tasks 0–5: worker gate + entity commit).

## 1. Problem

When an extraction pass's LLM output truncates (gemma4 `TRUNCATION_AT_NUM_PREDICT`,
`len(content)=0` under pool contention), docling-graph recovers via a retry/salvage path.
The recovery preserves the **entities** (recall is fine) but produces provenance rows with
`element_uid=""` and `page=null` — i.e. **no lineage**. Downstream:

- the worker's strict lineage gate (`_partition_entities_by_lineage`, pipeline.py:481) rejects
  the lineage-less entities, so they never commit; or (pre-gate) they commit unreachable.
- `EXTRACTED_FROM` edges are never built → **both** the graph-query endpoint
  (`api/v1/graph_store.py:91`) and the RAG retrieval endpoint (`api/v1/retrieval.py:696-795`)
  return empty entity→chunk lineage. (`MENTIONED_IN` has no query consumer.)

This violates the project's hard requirement: **every extracted value must trace to its exact
source chunk + document + page — including retry-recovered and salvaged content.** Dropping
salvaged entities (a coverage hit) is **not** an acceptable resolution.

### Verified evidence (run `e6c27008`, SA-2, `air_defense_v3_merged_v1`)

- **0 of 141** provenance rows across all 5 passes had a non-empty `element_uid` or non-null `page`.
- Truncation recurred under self-induced pool contention (3 concurrent passes × 2 workers on a
  2-host pool); every truncated call **retried and recovered** content (`final=recovered_content`,
  0 empty-after-retry), so this is purely a *lineage* loss, not a recall loss.
- `app.main` logged `no chunk metadata available from doc_processor or trace — provenance will be
  empty` at exactly the two affected passes (13:47, 14:58).
- The doc *was* chunked: `chunks=22 batches=19` (radar_power_rf), `chunks=20 batches=20`
  (missile_kinematics) — batches map **~1:1 to chunks**.
- `missile_kinematics` confirmed **freshly extracted** (new `instance_id`s vs the prior run),
  not rehydrated — so the empty lineage is a property of fresh salvaged extraction.

## 2. Root cause (precise)

The **clean** path already stamps batch-anchored provenance.
`normalize_delta_ir_batch_results` (ir_normalizer.py:551) builds, per successful batch, from that
batch's `chunk_metadata`:

```python
provenance = {
    "batch_index": batch_index,
    "chunk_indexes": chunk_indexes,   # from batch_plan[batch_index]  (orchestrator.py:188 tuples carry chunk_index)
    "page_numbers": page_numbers,     # from chunk_metadata[ci].page_numbers
    "self_refs": batch_self_refs,     # union of chunk_metadata[ci].self_refs
    "evidence_ids": batch_evidence_ids,
}
```
`_attach_evidence_to_prov` (ir_normalizer.py:56) even **falls back to batch evidence IDs** when a
node cited none. So successful-batch nodes get real `self_refs` + `page`, and the app-layer
`_resolve_element_uid` Strategy 3 (`provenance.self_refs[0]`, provenance.py:232) + `_resolve_page`
(provenance.py:314) resolve them.

**The gap:** truncated/recovered content does **not** flow through this stamping. It is not in
`successful_results` (so the normalizer never sees it), and/or the salvage degrades to a raw-text
extraction path (`_extract_direct_mode_from_text`, many_to_one.py:238 → `extract_delta_from_text`)
whose chunks carry **no DoclingDocument `self_refs`**. With `self_refs`/`chunk_indexes` absent on
the node, every `_resolve_element_uid` strategy fails, and the fallback synthesizer
`synthesize_provenance_from_pass_output` (provenance.py:90) emits `element_uid=""` because its
`chunk_to_self_refs` map (sourced from `doc_processor.last_chunk_metadata`, main.py:1106) is also
empty on that path.

So lineage is lost at **both** levels: the model dropped per-node citations, **and** the
batch-level chunk map that would substitute never reaches the salvaged nodes — even though the
chunks, their `self_refs`, and their pages all exist.

## 3. Decision

**Approach A — batch-anchored fallback (chosen).** Attribute each salvaged entity to the chunk(s)
of the **batch it was extracted from**. This is *guaranteed correct* (the entity provably came
from within that batch — a fact, not a heuristic) and carries the batch's real page(s). On this
doc (≈1 chunk/batch) it is exact; on docs that pack many chunks per batch it is a tight,
provably-correct superset.

Rejected: **text-matching** entity→chunk (fuzzy; can mis-attribute, corrupting the trust metadata
lineage is meant to guarantee) and **pass-level fan-out** (attributes every entity to all ~20 of
the pass's chunks — non-empty but uselessly coarse). Text-match refinement *within* the batch set
remains a future option if wide-batch docs need it, but is out of scope here.

## 4. Design

### Component 1 — Stamp batch-anchored provenance for ALL emitted nodes (patch 0006)

At the orchestrator's batch-results assembly seam, guarantee that **every** node — from a
validated batch *or* a recovered/salvaged one — receives `provenance.{self_refs, page_numbers,
chunk_indexes}` derived from its batch's `chunk_metadata`, mirroring what
`normalize_delta_ir_batch_results` already does for successful batches.

- **Fill-when-missing:** if the node already carries valid per-node citations (clean path), keep
  them; only fall back to batch-level anchors when per-node lineage is absent. Clean extractions
  are never degraded.
- **Preserve real chunk metadata through salvage:** the salvage path must retain the
  DoclingDocument-derived `chunk_metadata` (with `self_refs` + `page_numbers`) rather than
  degrading to raw-text chunks that have none — otherwise `batch_self_refs` is empty even after
  stamping. (Implementation pins whether the bypass is "recovered content not in
  `successful_results`", text-mode degradation, or both — see §6.)

### Component 2 — Resolution (no change needed)

Stamped `self_refs` resolve via the existing `_resolve_element_uid` Strategy 3; stamped
`page_numbers` via `_resolve_page`. This is independent of the app-layer `chunk_to_self_refs` map
that was observed empty — the node is self-describing for lineage.

### Component 3 — Worker gate stays as the safety net

`_partition_entities_by_lineage` (pipeline.py:481) is unchanged. After the fix it should **pass**
(entities have lineage) instead of rejecting. It remains the strict guarantee that nothing
lineage-less ever commits silently.

## 5. Semantics

- **Multi-chunk batch:** node gets the union of its batch's chunk `self_refs` → one
  `EXTRACTED_FROM` edge per chunk (correct superset). 1-chunk batch → exactly one (precise).
- **Multi-document:** a batch's chunks are always single-document; the same entity across documents
  accrues `EXTRACTED_FROM` edges to each document's chunks at the shared identity-keyed vertex.
- **Page-less sources:** `page` stays `null` only when the source genuinely lacks pagination —
  never fabricated. Guarantee is **chunk + document always; page when the source carries it.**

## 6. Verification (TDD-first)

1. **Failing unit test** (docling-graph, `tests/unit`): a batch whose nodes have no
   `evidence_ids`/`self_refs` (simulating salvage) → assert emitted nodes carry their batch's
   `self_refs` + `page`. Writing this test pins the exact bypass sub-path.
2. **E2E on SA-2** (truncating run): **0** provenance rows with empty `element_uid` across passes →
   **0** `LINEAGE_GATE` rejections at merge → `EXTRACTED_FROM > 0` → trace SNR-75 → chunk → page on
   **both** the graph-query and RAG endpoints.
3. **Regression:** a clean (non-truncated) pass still yields precise per-node lineage; no
   over-attribution introduced.
4. **Discriminator:** a genuinely empty doc still yields 0 entities (no fabricated lineage).

## 7. Scope / non-goals

- **In:** one library patch (`docker/docling-graph/patches/0006-*.patch`) at the orchestrator /
  `ir_normalizer` salvage-assembly seam; one docling-graph unit test; rebuild docling-graph
  (COPY image) + redeploy.
- **Out:** text-matching; worker gate logic changes; clean-path extraction changes; the gemma4
  contention/death-spiral itself (separate concern — truncation is treated as a production reality
  the system must survive with lineage intact).

## 8. Deploy notes

- docling-graph ships via COPY + patches → **rebuild required** (`-p eip-mmdpp`).
- After redeploy, restart **both** `worker-1` (catch-all consuming `graph`) **and** `worker-graph-1`
  — restarting only one leaves a stale graph-queue consumer (the trap that invalidated the earlier
  Task-5 gate). Confirm `StartedAt` advances past the code mtime on both.
- Schema fields touched (if any) must land in `air_defense_v3` first, then narrowed siblings.

## 9. Open implementation question (resolved during TDD)

Exactly which salvage sub-path bypasses `normalize_delta_ir_batch_results` — recovered content not
reaching `successful_results`, raw-text-mode degradation, or both. The failing unit test (§6.1)
reproduces the empty-provenance-on-salvage and localizes the seam where the single stamping hook
belongs.
