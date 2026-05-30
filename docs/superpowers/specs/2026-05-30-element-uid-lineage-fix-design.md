# Design: Restore `element_uid` provenance so entities commit with full lineage

**Date:** 2026-05-30
**Branch:** walltime/c0-telemetry
**Status:** DESIGN — awaiting user review before writing-plans
**Hard requirement addressed:** every extracted field value must be traceable to its source text chunk, document, and page (data-lineage requirement).

## Problem

Extracted entities are **not committed to ArcadeDB** (graph has 0 entity vertices despite the pipeline reporting 128 entities "extracted" on the SA-2 doc), and there are **no entity→chunk lineage edges**. A field value therefore cannot be traced to its chunk/document/page. This violates the hard data-lineage requirement and means the graph contains only document-structure, not domain knowledge.

This generalizes the previously-known bug #59 (system_links 0 committed edges) — it is not just relationships; **no entity type commits.**

## Verified root cause (log-timestamp confirmed, not hypothesis)

1. docling-graph's `doc_processor.last_chunk_metadata` is **EMPTY** when `main.py` builds provenance. The warning *"no chunk metadata available from doc_processor or trace — provenance will be empty. Check that strategy_ops set doc_processor.last_chunk_metadata after chunking"* fires (26 occurrences; timestamps match each SA-2 pass exactly: 13:05, 14:45, 14:59, 17:16).
2. → `chunk_to_self_refs` map (built in `main.py` from `last_chunk_metadata`) is empty.
3. → the primary provenance path (`build_provenance_from_context`) yields nothing ("library knowledge_graph path yielded none"); the fallback `synthesize_provenance_from_pass_output` fires but, with an empty `chunk_to_self_refs`, emits `element_uid="", page=null, chunk_index=null, evidence_ids=[]` for **every** entity (verified in the stored response: 24 radar entities, all empty provenance).
4. → the worker's `_parse_pass_response` (pipeline.py ~3676) **drops every provenance row lacking a non-empty `element_uid`** → entities count: 0 → merge commits 0 → empty graph → no lineage edges (`derive_structure_links` logged `entity_links=0`).

**One upstream defect (empty `last_chunk_metadata` → empty `element_uid`) causes BOTH symptoms** the user cares about: the missing lineage AND the empty graph.

### What is verified vs. open
- ✅ The set-code exists and the wire-OFF + delta path reaches it: `many_to_one._extract_contract_driven` (line ~349) → `extract_delta_from_document` (strategy_ops.py:60) → `doc_processor.extract_chunks_with_metadata` (builds `self_refs` at document_processor.py:317) → `doc_processor.last_chunk_metadata = chunk_metadata` (strategy_ops.py:73).
- ✅ The `doc_processor` instance is **shared** (`many_to_one` constructs one `self.doc_processor` at line 112 and passes that same instance to the delta strategy at 276/361) — so a read/write instance mismatch is ruled out as the likely cause.
- ❓ **OPEN (becomes plan step 1, a diagnostic):** why `main.py` reads `last_chunk_metadata` empty despite the set on a shared instance. Leading candidates: (a) `extract_chunks_with_metadata` produces `chunk_metadata` with empty `self_refs` because `_evidence_units_for_chunk` returns `[]` for these chunks; (b) a re-chunk/reset between set and read clears it; (c) a path variant (e.g. markdown-first) runs that doesn't set it. The plan **proves which before fixing.**

## Strategy (user-approved)

**Fix the root cause in docling-graph so `element_uid`/`page`/`chunk_index` always populate; KEEP the worker's strict `_parse_pass_response` drop** as the hard-lineage guarantee (no entity without lineage ever commits). The worker is **not** changed. The entire fix lives in docling-graph (the gitignored vendored clone) and ships as a tracked patch.

## Components

### Component 1 — Diagnostic (plan step 1, before any code change)
**Goal:** pinpoint exactly why `last_chunk_metadata` is empty at the `main.py` read.
**How:** one instrumented `graph_only` run on the SA-2 doc (already ingested) with targeted logging at: the `many_to_one` route selection, `extract_chunks_with_metadata` return (len + whether `self_refs` populated), the `strategy_ops` set, and the `main.py:1105` read. Confirms candidate (a)/(b)/(c) before touching code.
**Depends on:** idle LLM pool; SA-2 doc.

### Component 2 — The fix (location determined by Component 1)
**Goal:** ensure the chunking route the field-group passes take sets `doc_processor.last_chunk_metadata` with real `self_refs` (chunk_id → self_refs) that survive to the `main.py` read, so `chunk_to_self_refs` is non-empty and the provenance resolver/synthesizer emits real `element_uid`/`page`/`chunk_index`.
**Interface:** unchanged — `main.py` already reads `context.extractor.doc_processor.last_chunk_metadata`; the fix makes that read non-empty.
**Constraint:** docling-graph change lands as a tracked patch `docker/docling-graph/patches/0005-*.patch` (the clone is gitignored; patches are applied at build per the 0003/0004 pattern). Must apply cleanly against a clean upstream clone.

### Component 3 — Verification gate
**Goal:** prove entities + lineage commit end-to-end.
**Checks (on a post-fix graph_only SA-2 run, idle pool):**
- docling-graph response: provenance rows have **non-empty `element_uid` / `page` / `self_refs`**.
- worker: **zero** "dropping provenance row missing required fields" warnings.
- ArcadeDB: entity vertices > 0 (RADAR_SYSTEM / MISSILE_SYSTEM / …); `EXTRACTED_FROM` and/or `MENTIONED_IN` edges > 0.
- **Trace one field value → its text chunk + document + page** (the hard requirement, demonstrated).
- **Discriminator preserved:** a genuinely-empty doc (image/waveform) still yields 0 entities legitimately — confirm via docling-graph `raw_node_count` (=1 wrapper = genuinely empty; >1 then dropped = the bug). Some documents genuinely have no entities; the fix must not conflate "none to extract" with "dropped for empty provenance."

## Data flow

**Broken (current):**
```
scoped DoclingDocument → many_to_one._extract_contract_driven
  → extract_delta_from_document → doc_processor.extract_chunks_with_metadata (builds self_refs)
  → doc_processor.last_chunk_metadata = chunk_metadata        [strategy_ops.py:73]
  → main.py reads last_chunk_metadata                          [main.py:1105]  ⚠️ EMPTY
  → chunk_to_self_refs = {} → synthesize_provenance(...) → element_uid=""
  → worker _parse_pass_response DROPS all entities → 0 committed → no lineage
```
The break is **between the set (strategy_ops:73) and the read (main.py:1105)** — Component 1 pins which.

**Fixed:** identical path, `last_chunk_metadata` non-empty at the read → real `element_uid`/`page`/`self_refs` → entities kept → committed → `EXTRACTED_FROM`/`MENTIONED_IN` lineage edges → field value traces to chunk+doc+page.

## Testing

1. **Component-1 diagnostic** is itself a test: instrumented run proves the exact empty-source before any change.
2. **docling-graph unit test** (in the clone's suite AND mirrored to `tests/unit/` so `scripts/run_tests.sh` collects it): a DoclingDocument with known elements → assert `extract_chunks_with_metadata` yields `chunk_metadata` with non-empty `self_refs`, and that `last_chunk_metadata` is non-empty at the read site.
3. **Provenance unit test:** `synthesize_provenance_from_pass_output` with a populated `chunk_to_self_refs` emits non-empty `element_uid` for every entity (regression guard).
4. **End-to-end gate (Component 3)** on the SA-2 doc — the five checks above, including the field→chunk→page trace and the empty-doc discriminator.
5. **Patch durability:** `0005-*.patch` applies cleanly against a clean clone in the Docker build.

## Out of scope
- The worker's strict-drop policy stays (no soft-fail). 
- Chunk-minimization / coverage work (deferred — depends on this fix landing first so recall/lineage data is real).
- The paused notebooks-collection ingest restarts AFTER this fix lands (re-run all 21 docs fresh on the fixed code).

## Scope / production-impact note
All evidence is from the walltime branch + merged/narrow path. Whether production `air_defense_v3` (non-narrowed) is affected is **unconfirmed**; the 82 existing CommunityReports imply entities committed at some earlier point. Component 1's diagnostic + Component 3's gate will clarify whether this is branch/path-specific or systemic. If systemic, the patch fixes it everywhere (the provenance path is shared).
