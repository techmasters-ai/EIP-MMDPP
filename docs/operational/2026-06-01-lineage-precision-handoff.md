# Handoff — Precise entity→chunk lineage (EXTRACTED_FROM)

**Date:** 2026-06-01 · **Branch:** `walltime/c0-telemetry` · **Status:** EXTRACTED_FROM=0 bug FIXED + shipped-quality; PRECISION not yet met (lineage is coarse). Task 6 gate ran 6/7 PASS, NOT marked complete.

---

## 1. Goal

**Overarching project goal:** maximize recall, minimize walltime, increase the precision of chunks sent to the LLM — generalizing across any document shape. A **hard requirement** runs through all of it: **complete data lineage** — every extracted field value (entities AND relationships) must trace back to its **exact source text chunk + document + page**, plus trust/validity metadata. Lineage is not optional polish; it's a stated invariant of the system.

**Immediate goal (this work):** restore **precise** entity→chunk lineage — i.e. `EXTRACTED_FROM` edges where each entity links to the **specific chunk(s) it was actually extracted from**, so the graph-query and RAG endpoints can answer "where did this fact come from?" with an exact chunk + page. This must hold on real production runs, not just clean lab conditions.

---

## 2. The issue

Two layers, one fixed and one open.

**(FIXED) `EXTRACTED_FROM` = 0 — no lineage at all.** Root cause: docling-graph patch `0002` rewrote `_extract_from_docling_document` to run the delta CHUNKED-BATCHES path for DoclingDocument input, but never stored `doc_processor.last_chunk_metadata` — so `app/main.py` built empty `chunk_to_self_refs`/`chunk_to_page_numbers` maps → every provenance row had `element_uid=""`/`page=null` → the worker lineage gate rejected all entities → `EXTRACTED_FROM` never built (0 / 2711 across all runs, universal, pre-existing). **Fixed** via Parts A/B/C (8 commits, all two-stage reviewed). Verified on run `b9bc23a5` (2026-06-01): `EXTRACTED_FROM` 0 → **6844**; 139/139 provenance rows now carry `element_uid`+`page`; entities commit (26); a value traces **SNR-75 → chunk c1c6123d → page 2 → SA-2 doc**; 0 provenance-drop warnings; 0 lineage-gate rejections.

**(OPEN) The lineage is COARSE, not precise.** Every entity links to ~**all 102** chunks of the document (SNR-75 has **736** `EXTRACTED_FROM` edges; 66 "coarse fan-out" WARNs in the run). The fan-out-width precision gate correctly **FAILED**: worst entity = 103/102 chunks (~101%). So a lineage query returns "this fact is supported by the whole document" — which does **not** meet the exact-source-chunk requirement. (Coarse vs precise: both make entity→chunk edges; coarse = entity→all chunks, useless for citation; precise = entity→its actual source chunk(s)+page, auditable.)

**(SECONDARY) `/v1/graph/query` returns `sources=null` for SNR-75** despite its 736 `EXTRACTED_FROM` edges. Lineage exists in the graph but isn't surfacing to the consumer endpoint — a separate traversal/match bug, lower priority.

---

## 3. My thoughts on the cause (verified, not assumed)

**The coarseness is STRUCTURAL and UNIVERSAL — NOT truncation-driven.** (I initially blamed the gemma4 truncation death-spiral; that was wrong and is retracted. ~7 of ~173 batches truncated last run (~4%); every pass extracted its entities fine. Truncation is a red herring for the coarseness.)

The real chain, traced link-by-link in code:

1. There are **two** provenance producers in docling-graph `app/`:
   - **Primary:** `build_provenance_from_context` (provenance.py:342) — walks `context.knowledge_graph` nodes and reads each node's `provenance` dict (incl. per-node `evidence_ids`). This is where **Parts B/C engage** (Part B prefers per-node `evidence_ids[0]`; Part C maps that `#/` self_ref → concrete chunk).
   - **Fallback:** `synthesize_provenance_from_pass_output` (provenance.py:84) — runs when the primary returns `[]`. It hardcodes **chunk-0's** self_ref for *every* entity and emits **no `evidence_ids`**. Inherently coarse; bypasses Parts B/C entirely.

2. **The primary path ALWAYS returns `[]`, on every run.** `context.knowledge_graph` is built by `pydantic_list_to_graph` → `_create_nodes_pass` (graph_converter.py:184-254), which sets node attributes **only from the entity Pydantic model's schema fields** (`for field_name, field_value in model`). The entity schemas (`RADAR_SYSTEM`, etc.) declare **no `provenance`/`evidence_ids` field** — so knowledge_graph nodes structurally never carry provenance → `build_provenance_from_context` finds nothing on any node → returns `[]`. Confirmed by all-history log counts: synth fired 4/4 entity passes; `build_provenance: dropping node` = **0** (the primary builder never even produced a droppable node).

3. **So `synthesize_provenance_from_pass_output` is effectively the ONLY path that ever runs** — and it's coarse by construction (chunk-0 + no evidence_ids). Part A's `last_chunk_metadata` store is what makes the synth path now produce a *non-empty* element_uid (its maps are populated) — that's the EXTRACTED_FROM 0→6844 win — but synth can never be precise as written.

4. **The precise per-node `evidence_ids` DO exist** — on the **delta merged graph**, which docling-graph already persists to `delta_merged_graph.json` and loads back onto `context._delta_merged_graph` (main.py:1172-1191), explicitly described as carrying per-node "provenance dicts (evidence_ids, self_refs)." There is also already a function `build_relationship_provenance_from_delta_trace` (imported main.py:183) that mines the delta layer for **relationship** provenance — i.e. the pattern of "get precise provenance from the delta graph, not from knowledge_graph" already exists in the codebase, just only for edges.

**Why option 2 (make `pass_output` synth precise) was ruled out:** `pass_output` is the projected Pydantic template root; its entity items carry no per-node citation (`item_has_citation=f` across all 8 historical runs). Evidence lives on the delta graph, not on `pass_output`. **Why option 1 (concurrency-capped re-run) is futile:** eliminating truncation won't make `build_provenance_from_context` work — the knowledge_graph still won't carry provenance, so synth still fires, still coarse.

**Proposed fix direction:** source **entity** provenance from `context._delta_merged_graph` (which has per-node evidence_ids → precise self_ref) at provenance-build time, instead of (or before) the chunk-0 synth fallback — mirroring the existing `build_relationship_provenance_from_delta_trace` pattern. This is a docling-graph `app/` change (main.py + provenance.py), tractable, and does **not** depend on eliminating truncation. Needs to be confirmed by checking the delta_merged_graph node shape (in progress).

---

## 4. State / deliverables

- **8 commits on `walltime/c0-telemetry`** (all two-stage reviewed): `027fb79` Part A (patch 0002 store + Dockerfile `--fuzz=0||exit1`); `c77d174`+`acf24cd` Part B (prefer per-node evidence_ids, numeric-smallest deterministic key, inverted contradicting test); `d97e92e`+`bc5f4db` Part C (worker `_resolve_mention_chunks`+`_load_identity_map`); `4912267` wire 2 docling-graph tests into run_tests.sh; `ed7f5e3`+`dd2febd` hardened verifier (run-scoped EXTRACTED_FROM, fan-out-width precision, edge-anchored trace, fail-closed log checks).
- **Plan:** `docs/superpowers/plans/2026-05-31-last-chunk-metadata-store-fix.md` (+`.tasks.json`). Tasks 1-5 complete; **Task 6 (user-ordered gate) in_progress** — gate ran 6/7 PASS, precision FAILED, NOT marked complete.
- **Relationship/edge lineage = SCOPED OUT** (Task 5): edges store provenance as a JSON property on the domain edge, no edge→chunk lineage mechanism; bug #59 territory, separate.
- **Deployed build is live** (06-01T05:16Z): docling-graph rebuilt (all 5 patches clean), both workers restarted, all 3 StartedAt advanced; Parts A/B/C confirmed in the running containers.
- Full detail + retracted theories: memory `project_extracted_from_root_cause.md`.

## 4b. Refined diagnosis + remedy (user analysis, VERIFIED 2026-06-01)

User-supplied analysis, key claims re-verified in code:
- **Primary path is dead for TWO independent reasons (verified):** (1) `build_provenance_from_context` skips nodes via `_is_entity_label` which requires `label == label.upper()` (provenance.py:202), but the graph converter sets `label = model.__class__.__name__` (mixed-case class name, graph_converter.py:214) — `ontology_name` lives in `model_config` (entities.py:148), NOT on the node. So every node is filtered out before provenance is even checked. (2) Even if labels matched, nodes carry no provenance. Existing tests mask this by mocking uppercase labels + provenance-bearing nodes.
- **Batch-fallback coarseness (verified, CRITICAL):** `_attach_evidence_to_prov` (ir_normalizer.py:56) sets `out["evidence_ids"] = valid or list(batch_evidence_ids)` — when the LLM cites nothing valid, the node gets the WHOLE batch evidence pool. So a naive "read delta node.evidence_ids" fix is STILL coarse for uncited nodes. The fix MUST distinguish explicit citations from batch fallback.
- **Node-provenance union gap (plausible, confirm at design):** `merge_delta_graphs` (helpers.py:317) preserves per-property `__property_provenance` but does not union top-level node provenance across deduped duplicate entities — so full value lineage needs BOTH node provenance and property provenance.

**Agreed remedy (6 parts) — replaces my earlier one-line "read node.evidence_ids":**
1. New **service** builder `build_entity_provenance_from_delta_graph(context, template_cls, ExtractionProvenance, ...)` — runs BEFORE `build_provenance_from_context` and BEFORE synth. Walks `context._delta_merged_graph["nodes"]`, maps delta `path`/`node_type` → template `ontology_name`/`graph_id_fields`, emits one row per precise evidence self-ref (NOT one per entity-batch).
2. Resolve each row evidence_id → exact page/chunk via `chunk_to_evidence_units`/chunk metadata (element_uid="#/texts/N", page=that unit's page, chunk_index=its chunk, evidence_ids=[that self_ref]).
3. Use node.provenance + `property_evidence` + merged `__property_provenance` so deduped entities retain all cited chunks across batches.
4. Normalizer change: tag `evidence_source="explicit"|"batch_fallback"`. Batch fallback = usable degraded lineage but MUST NOT pass the precision gate.
5. **Worker resolver fix (`_resolve_mention_chunks`, pipeline.py:1771):** build chunk map from `TextChunk.chunk_metadata/self_refs/evidence_ids` as aliases to chunk IDs; make unresolved PRECISE self_refs **fail-closed** (no edge) instead of all-document fan-out.
6. Keep `synthesize_provenance_from_pass_output` only as an explicitly-degraded fallback; if it fires, precision FAILS by design.
- `/v1/graph/query sources=null`: secondary; matched RID likely isn't the RID carrying the edges, or needs entity_id/canonical fallback. Test after precision.
**Do NOT ship current state as satisfying lineage.** Options 1 (re-run capped) and 2 (precise pass_output synth) remain ruled out.

## 5. Open decision for the user
**(A)** Build the delta-sourced precise-entity-provenance fix (looks tractable; mirrors existing relationship code), or **(B)** ship the EXTRACTED_FROM=0 win as-is and treat precision + the `/v1/graph/query` null bug as documented follow-ups. (Options 1 "re-run capped" and 2 "precise pass_output synth" are both ruled out by the verification above.)

## 6. Where I was (resuming now)
Verifying two facts that confirm fix-direction (A) is viable: (a) does `delta_merged_graph.json` carry per-**node** `evidence_ids` (entities), not just per-relationship? and (b) the exact shape of `build_relationship_provenance_from_delta_trace` as the pattern to mirror for entities.
