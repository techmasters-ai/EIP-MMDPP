# Design: Fix entity commit + restore field-value lineage (chunk/document/page)

**Date:** 2026-05-30
**Branch:** walltime/c0-telemetry
**Status:** DESIGN rev 2 — revised per code review (split entity-commit from lineage); awaiting user review before writing-plans
**Hard requirement addressed:** every extracted field value must be traceable to its source text chunk, document, and page (data-lineage requirement).

## Open questions (must resolve before writing-plans)
1. **Lineage-commit policy (drives Component 2C):** if an entity's lineage cannot be resolved, should the pipeline (a) **refuse to commit** it (strict — enforces "no entity without lineage"), or (b) **commit it + log a soft-fail TODO** (recall-safe, lineage best-effort)? My earlier "keep worker unchanged = guarantee" was wrong; either choice needs explicit code.
2. **Scope:** confirm production `air_defense_v3` (non-narrowed) shares these failures, or is this branch/merged-path specific? Component 1's diagnostic answers this.

## Problem

Two symptoms observed on the SA-2 run (`9d48fc1e`), which a code review confirmed are **TWO SEPARATE failures** that must be diagnosed independently (my first draft wrongly fused them into one causal chain):

1. **Entity-commit failure:** ArcadeDB has **0 entity vertices** despite the pipeline reporting 128 entities extracted. (Generalizes bug #59 beyond just system_links.)
2. **Lineage failure:** even where entities would commit, there are **no entity→chunk lineage edges** and no per-field `element_uid`/`page` — so a field value can't be traced to chunk/document/page. Violates the hard data-lineage requirement.

### Correction from review (what I verified, what I got wrong)
- ❌ **WRONG (my first draft):** "empty `element_uid` → `_parse_pass_response` drops entities → merge commits 0 → empty graph." 
- ✅ **VERIFIED truth:** `_parse_pass_response` (pipeline.py:3668-3683) only drops malformed **provenance rows**, NOT entities. Entities are built independently from `pass_output` into `template_instance` (3659-3664). Live replay on SA-2 radar_power_rf → `_cached_entities() = 24` (not 0); full `merge_and_resolve` over all passes → **MERGED entities = 22, relationships = 0**. So merge produces 22 entities, NOT zero. (My earlier "entities=0" was a test artifact — I read a non-existent `.entities` attr; `PassResult` exposes `template_instance` / `_cached_entities()` / `iter_entities_of_type`.)
- **Therefore the empty graph is a failure DOWNSTREAM of merge** (in `_import_graph_phase_nodes` / `upsert_nodes_batch_sync`, pipeline.py:1217/1295), OR the real-run merge never committed (no SA-2 merge log line found; the only merge-success log at 21:09 is the Fan_Song image doc with a legit `entities:0`). **Undiagnosed — the plan must diagnose it directly, NOT assume element_uid is the cause.**

## Verified facts (lineage half of the problem)

The lineage failure IS rooted in empty `element_uid` (this part of the original root cause holds, log-timestamp confirmed):
1. docling-graph's `doc_processor.last_chunk_metadata` reads **EMPTY** in `main.py` → warning *"no chunk metadata available... verify strategy_ops set doc_processor.last_chunk_metadata after chunking"* (26×; timestamps match each SA-2 pass: 13:05/14:45/14:59/17:16).
2. → `chunk_to_self_refs` empty → primary provenance path yields none → `synthesize_provenance_from_pass_output` emits `element_uid="", page=null, chunk_index=null, evidence_ids=[]` for every entity (verified in stored response).
3. → provenance rows dropped by `_parse_pass_response`; even if kept, an entity can commit with empty `record.provenance` (just lacking lineage). So missing lineage is explained; **0 committed entities is NOT.**

### Active-route caveat (review finding, Medium)
This branch's `many_to_one` has TWO delta routes. When `_pre_built_chunks` is set (the wire), it BYPASSES `extract_chunks_with_metadata` (many_to_one.py:349) and builds metadata from `entry["source_refs"]` instead (many_to_one.py:561), with `page_numbers: []` (568). The wire is currently OFF (no pre-built chunks → `extract_delta_from_document` route), but the diagnostic must instrument BOTH routes (selected_chunks payload count, `source_refs` count, `_extract_delta_from_pre_built_chunks`, and `last_chunk_metadata`) since the fix must hold whether or not the wire is later enabled.

### Page requirement (review finding, Medium)
Making `self_refs` non-empty does NOT by itself satisfy the page requirement. Pre-built metadata sets `page_numbers: []` (many_to_one.py:568); fallback provenance sets `page=None` (provenance.py:155); `_resolve_page` only reads `page_numbers` (provenance.py:284). The fix must explicitly resolve `page` — by carrying `page_numbers`/evidence units from the worker's selected chunks, or resolving `source_refs` against the DoclingDocument.

## Strategy (revised per review)

Two independent workstreams, because the two failures have different causes:

- **A. Entity-commit:** diagnose why 22 merged entities don't reach ArcadeDB (service post-filter / postprocess / merge dispatch-vs-commit / `_import_graph_phase_nodes` / `upsert_nodes_batch_sync`), then fix the actual break. **Do not assume element_uid is involved.**
- **B. Lineage:** fix docling-graph so `element_uid` AND `page` populate (both delta routes), so committed entities carry real lineage edges + properties.

**Hard-lineage guarantee — explicit decision needed (review finding, High):** "leave the worker unchanged" does NOT guarantee "no entity without lineage commits" — current code can commit an entity with empty `record.provenance`. If the hard rule is to be *enforced*, the design needs an explicit lineage-required commit gate (worker or service side). **Open question for the user** (see below) — the answer shapes workstream A's commit policy.

## Components

### Component 1 — Diagnostic: why don't 22 merged entities reach ArcadeDB? (workstream A; plan step 1, before any code change)
**Goal:** localize the entity-commit break by instrumenting each stage *separately*, NOT assuming element_uid.
**Instrument, in order:** (1) docling-graph service post-filter / IDENTITY_FILTER / postprocess (does the response still carry the entities?); (2) `load_completed_pass_outputs` + rehydrate (we confirmed 24/21/36/47 offline — verify in the real run); (3) `merge_and_resolve` output count in the *real* run (offline replay = 22); (4) merge **dispatch vs. commit** — was `derive_ontology_graph_merge` actually invoked for `9d48fc1e`, and did `_import_graph_phase_nodes` → `upsert_nodes_batch_sync` execute and return RIDs?; (5) ArcadeDB write result.
**Also (workstream B diagnostic):** instrument the provenance path on BOTH delta routes — `extract_chunks_with_metadata` return AND `_extract_delta_from_pre_built_chunks` (`source_refs` count, `page_numbers`), the `strategy_ops` set, and the `main.py` read of `last_chunk_metadata`.
**How:** one instrumented `graph_only` run on the SA-2 doc (already ingested), idle pool.
**Output:** definitive localization of (A) the commit break and (B) the provenance-empty source, before any fix.

### Component 2A — Fix entity commit (location from Component 1)
**Goal:** ensure merged entities actually upsert to ArcadeDB. Fix the specific break Component 1 finds (e.g. merge not dispatched, upsert error swallowed, post-filter dropping all). Likely worker/service side.

### Component 2B — Fix lineage population (docling-graph)
**Goal:** populate `element_uid` AND `page` for every entity on whichever delta route runs, so committed entities carry real lineage. Must cover the pre-built-chunk route (carry `page_numbers`/evidence units or resolve `source_refs` against the DoclingDocument — `page_numbers:[]` today) AND the from-document route (empty `last_chunk_metadata`).
**Constraint:** docling-graph changes land as a tracked patch `docker/docling-graph/patches/0005-*.patch` (gitignored clone; applied at build per the 0003/0004 pattern; must apply cleanly against a clean clone). Worker/service changes (2A, and any lineage-gate) are normal commits on the branch.

### Component 2C — Lineage-required commit gate (CONDITIONAL — pending user decision)
**Only if** the user confirms the hard rule "no entity without lineage commits." Add an explicit gate (worker or service side) that refuses to commit an entity lacking resolvable lineage, rather than relying on docling-graph always succeeding. If the user prefers "commit + soft-fail TODO when lineage can't resolve," this component is replaced by a soft-fail log per [[feedback_softfails_to_todo]]. **This is an open question (below), not yet decided.**

### Component 3 — Verification gate
**Goal:** prove entities + lineage commit end-to-end.
**Checks (post-fix graph_only SA-2 run, idle pool):**
- **Entity commit:** ArcadeDB entity vertices > 0 (RADAR_SYSTEM / MISSILE_SYSTEM / …), count consistent with merged-entity count.
- **Lineage:** docling-graph response provenance rows carry non-empty `element_uid` + `page_numbers` (note: the response field is **`evidence_ids`**, not `self_refs` — verify `evidence_ids` populated, OR add a `self_refs` field intentionally); worker emits **zero** "dropping provenance row" warnings; `EXTRACTED_FROM`/`MENTIONED_IN` edges > 0.
- **The hard requirement, demonstrated:** trace one field value → its text chunk + document + **page**.
- **Discriminator preserved:** a genuinely-empty doc (image/waveform) still yields 0 entities legitimately — confirm via docling-graph `raw_node_count` (=1 wrapper = genuinely empty; >1 then absent from graph = a bug). Some documents genuinely have no entities; the fix must not conflate "none to extract" with "dropped/lost."

## Testing

1. **Component-1 diagnostic** is itself the first test: instrumented run localizes BOTH the commit break and the provenance-empty source before any change.
2. **docling-graph unit tests** (clone's suite AND mirrored to `tests/unit/` so `scripts/run_tests.sh` collects them): for BOTH delta routes, assert the produced metadata carries non-empty `self_refs` AND `page_numbers`, and that `last_chunk_metadata` is non-empty at the read site.
3. **Provenance unit test:** `synthesize_provenance_from_pass_output` with a populated `chunk_to_self_refs` emits non-empty `element_uid` AND resolved `page` for every entity (regression guard).
4. **Entity-commit unit/integration test:** the specific break Component 1 finds gets a focused regression test (e.g. merge dispatched + upsert returns RIDs for N entities).
5. **End-to-end gate (Component 3)** on the SA-2 doc — all checks, including the field→chunk→**page** trace and the empty-doc discriminator.
6. **Patch durability:** `0005-*.patch` applies cleanly against a clean clone in the Docker build.

## Out of scope
- The worker's strict-drop policy stays (no soft-fail). 
- Chunk-minimization / coverage work (deferred — depends on this fix landing first so recall/lineage data is real).
- The paused notebooks-collection ingest restarts AFTER this fix lands (re-run all 21 docs fresh on the fixed code).

## Scope / production-impact note
All evidence is from the walltime branch + merged/narrow path. Whether production `air_defense_v3` (non-narrowed) is affected is **unconfirmed**; the 82 existing CommunityReports imply entities committed at some earlier point. Component 1's diagnostic + Component 3's gate will clarify whether this is branch/path-specific or systemic. If systemic, the patch fixes it everywhere (the provenance path is shared).
