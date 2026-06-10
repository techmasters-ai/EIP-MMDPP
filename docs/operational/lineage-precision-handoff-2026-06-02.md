# Lineage Precision — Work Handoff (2026-06-02)

Branch: `walltime/c0-telemetry` · HEAD at writing: `4ce5a97` · Author of cycle: Claude Code session 926165cc

---

## 1. Overall goal (why any of this exists)

The extraction pipeline has three standing objectives, to hold for **any** document shape (not just SA-2):

1. **Maximize recall** — extract every entity/field/relationship the document actually supports.
2. **Minimize chunks/tokens sent to the LLM per extraction pass** — this is the dominant driver of extraction wall-time.
3. **Maximize precision** — no spurious entities/edges.

Overarching these is a **hard requirement: complete data lineage.** Every extracted **entity**, every **field** value, and every **relationship** must trace back to its *exact source chunk + document + page*, with trust/validity metadata. Lineage is not a nice-to-have — it is the measurement instrument for objective #2: you cannot safely narrow the chunks sent to the LLM unless you can prove, per extracted fact, which chunk it came from. If lineage is coarse or missing, "narrowing without losing recall" is unmeasurable.

## 2. Where this sits in the bigger plan

The immediate work is a **precise-lineage verification gate** that must read **ALL CHECKS PASS** (entity + field + relationship precision, plus recall) on a real SA-2 `graph_only` run. Only once the gate is trustworthy *and* green do we proceed to the **P-series 21-document collection** (`#13`), which ingests ~21 varied docs to **calibrate chunk-selection** — finding the minimum chunk set per pass that preserves recall, using the now-precise lineage as ground truth. The user has explicitly required: **gate green first, then PAUSE for explicit go-ahead before launching the 21-doc collection.**

Related context: the P-series is also motivated by the finding that `missile_identity`/`radar_identity` send ~70 chunks/67 batches to the LLM (vs ~20 for the smaller passes) — narrowing those is the biggest wall-time lever, and precise lineage is what makes it safe.

## 3. Pipeline architecture (1-minute refresher for whoever picks this up)

- **worker** (Celery, `app/workers/pipeline.py`) orchestrates ingest. `graph_only` re-runs only the extraction+graph phases on an already-converted document (same `document_id`, new `pipeline_run_id` each run).
- **docling-graph** (FastAPI service, port 8002, `docker/docling-graph/app/`, COPY image → **rebuild on change**) runs the 5 extraction passes against gemma4:31b and returns per-pass results incl. provenance.
- **gemma4:31b Ollama pool** — two hosts `10.0.1.121` + `10.0.1.109`.
- **ArcadeDB** — the graph store; entities are **global** vertices (shared across documents), relationships are **global** edges; both carry `document_ids` (a LIST) and edges now carry `pipeline_run_id`. `EXTRACTED_FROM` edges (entity→chunk) carry `pipeline_run_id` and are the run-scoped lineage discriminator.
- Worker code under `app/**` is **bind-mounted** into the worker containers (restart to load, no rebuild). docling-graph code is **COPY'd** (rebuild required). The compose project `eip-mmdpp` is anchored to **this worktree**. Restart **both** `worker-1` (catch-all, also consumes the graph queue) and `worker-graph-1`.

The 5 passes: `radar_identity`, `radar_power_rf`, `missile_identity`, `missile_kinematics`, `system_links`. The first four extract entities+fields (+ intra-pass typed edges). `system_links` extracts cross-entity **domain relationships** (ASSOCIATED_WITH, CUES, VARIANT_OF) as DTO-node relationships referencing entities from other passes via pass-local ref-ids (`E001`, `E002`, …).

## 4. The three lineage targets and how each is built

- **Entity lineage** = `EXTRACTED_FROM` edges (entity vertex → TextChunk vertex), carrying `document_id`, `page`, `pipeline_run_id`. Built in `derive_structure_links`.
- **Field lineage** = `_field_evidence` rows on the entity vertex, each resolving a field value → `chunk_id` + page. Built in the merge phase.
- **Relationship lineage** = `source_chunk_ids` (+ `source_pages`, `source_self_refs`) properties on the committed domain edge. Built in `_build_relationship_records` (merge) and written by `upsert_relationships_batch_sync`.

All three depend on resolving a provenance **self_ref** (`#/texts/N`, `#/tables/N`, `#/pictures/N`) to a concrete `chunk_id`. The resolver bridges self_ref → `element_uid` (via `identity_map`, persisted at ingest) → `chunk_id` (via `element_uid_chunk_map`), with a direct self_ref→chunk lookup short-circuit.

---

## 5. Changes this cycle — each with reasoning

Commits are on `walltime/c0-telemetry`, oldest→newest. (Fix A/B/C predate this session; summarized for continuity.)

### Fix A/B/C (prior session — context)
- **Fix A** — `_parse_pass_response` was dropping field-provenance rows whose `supporting_snippet` was empty. Relaxed: keep a row that has a resolvable `element_uid`/`self_refs` even without prose snippet (chunk lineage doesn't require human-readable text).
- **Fix B** — the gate's entity-recall metric compared committed distinct identities against the *per-pass instance sum* (double-counting entities emitted by multiple passes) → false "20% recall". Fixed to compare against the **distinct merged-node count**.
- **Fix C** — the docling-graph provenance builder ignored `system_links` DTO-node relationships; extended it to emit `relationship_provenance` for them.

### Fix D1 — `pipeline_run_id` on domain edges — `26c0a15`
**Why:** committed ASSOCIATED_WITH/CUES edges had `pipeline_run_id = NULL`. The upsert SQL builder (`_build_upsert_relationship_script`) set `document_ids` but never the run id, even though every other edge writer did and the schema defines the column. Without it, run-scoped reasoning about edges is impossible.
**Change:** added a `pipeline_run_id = :run_id` SET to the CREATE and UPDATE branches, bound from `provenance.pipeline_run_id`, guarded for absent provenance.

### Fix G — gate query bugs — `d9ffcfc`
**Why:** the gate itself was lying. (1) FIELD precision read **0** `_field_evidence` rows because it filtered `WHERE document_id = :doc`, but entity vertices are `identity_scope=global` and carry **no** `document_id` property at all → matched nothing. (2) RELATIONSHIP precision counted **170** edges because the script's structural-edge set omitted the document-anchor edges `HAS_IMAGE` (34) + `NEAR_TEXT` (136), so it counted structural edges as domain relationships.
**Change:** FIELD check reads the global entity set (matching how the entities are actually stored); RELATIONSHIP check imports `_STRUCTURAL_EDGE_TYPES` from the schema and adds the anchor edges, restricting to true domain edge types. These exposed the *real* residual signals (field nulls; stale rel edges) rather than masking them.

### D2 diagnostic — theory overturned (no commit; recorded in memory)
**Why it mattered:** the working theory was "relationship provenance isn't delivered to `_build_relationship_records`." A read-only reproduction against live data **refuted** that — the worker code *did* attach chunks (66/8 in repro) when fed the real rows. The committed NULL edges turned out to be a **stale 2026-05-31 build** (created_at + NULL pipeline_run_id + pre-Fix-C provenance shape all predated the fixes). Lesson baked into the deploy runbook below: **verify code is live in-container before trusting a run** — a stale build was the actual cause of an earlier "failure." The diagnostic surfaced three genuine residuals → the next three fixes.

### Fix R — per-edge precise relationship lineage — `6fb2691` + `1c93fd3` + `9e2c2d6`
**Why:** even when chunks attached, lineage was **coarse** — because `system_links` DTO rows had `source/target_instance_id = None`, every row collapsed into one `(_FALLBACK, rel_type)` bucket, so **every** ASSOCIATED_WITH edge received the **same** ~66-chunk union (~65% of the document). That violates the "precise per-relationship lineage" hard requirement. The fix: the docling-graph DTO node already carries `from_ref_id`/`to_ref_id` (e.g. `E002`/`E041`) in its properties but was throwing them away.
**Changes:**
- docling-graph `provenance.py` DTO branch now reads `from_ref_id`/`to_ref_id` and emits them on the row (and prefers the per-node `evidence_ids` as the precise anchor over the coarse self_refs batch union). New fields mirrored on both `ExtractionRelationshipProvenance` dataclasses (wire contract).
- worker `_build_relationship_records` resolves a row's `from_ref_id`/`to_ref_id` through `upstream_refs` to `(from_identity, to_identity)` and keys the **precise triple** `(from_identity, rel_type, to_identity)` — matching the committed edge's key, so each edge gets only its own chunks. Fallback retained only for genuinely unmatchable rows (now logged).
- `1c93fd3` — **alias gap (caught in spec review):** the edge's `from_identity`/`to_identity` are alias-*canonicalized*, but the precise key was built from raw pre-alias `upstream_refs` identities, so when canonicalization fires (Step 6 designation aliases) the key would miss and the edge would get NULL lineage silently. Fixed by canonicalizing the resolved identities with the same `identity_aliases` map before keying.
- Added an **identity_map-empty guard**: a loud WARN when relationship rows exist but `identity_map` is empty (the silent MinIO-miss failure mode that would turn lineage NULL).

### Fix F — precise field/entity chunk bridge — `cbacd11` + `f3fe0fa`
**Why:** the merge-phase `element_uid_chunk_map` had only ~34 keys, **all images** — because it joined `DocumentElement.artifact_id ↔ TextChunk.artifact_id`, but native HybridChunker leaves prose/table chunks with `artifact_id = None`. Effect: 6 field rows were NULL chunk_id, **and** ~60 "resolved" field rows had actually resolved to *image_description* chunks via entity fallback — broadly imprecise, not just 6 nulls.
**Change:** seed a `self_ref → chunk_id` map from the **ArcadeDB `TextChunk.self_refs` property** (the authoritative prose/table bridge, e.g. `#/texts/99 → chunk@page5`, `#/tables/1 → 2 chunks@page7`) into the resolver, so the direct self_ref lookup resolves prose/table refs. Mirrored in `derive_structure_links` for entity `EXTRACTED_FROM` parity. Multi-chunk table self_refs pick the lowest chunk_index deterministically for the scalar. `f3fe0fa` refactored the two seed call sites into one `_augment_element_uid_chunk_map_from_arcadedb` helper.

### Fix H — un-gate lineage SETs on relationship UPDATE — `9bc144a`
**Why (the real root cause of the relationship FAIL on run 625ad1bd):** `_build_upsert_relationship_script`'s UPDATE branch gated the **entire** statement on `AND NOT (document_ids CONTAINS :doc_id)`. On any re-run of the same doc, the edge's `document_ids` already contains the doc → UPDATE matches **zero rows** → `pipeline_run_id` **and** `source_chunk_ids` are never written onto pre-existing edges. Only newly-*created* edges got lineage (the 3 of 32 that passed). The membership gate was correct in 2026-04 when the UPDATE only appended `document_ids`; Fix D1 and the source_chunk_ids work later piled must-always-apply SETs into the same gated statement.
**Change:** removed the membership predicate from the WHERE; made the `document_ids` append idempotent **inside the SET** via `document_ids = (CASE WHEN document_ids CONTAINS :doc THEN document_ids ELSE document_ids || [:doc] END)` (ArcadeDB dialect confirmed in the manual + a live re-upsert integration test). Lineage SETs now always apply on UPDATE; doc-ids list stays dup-free. Sibling audit confirmed `_build_upsert_node_script` and structural-edge builders don't share the bug.

### Fix G2 — run-scope the relationship-precision gate check — `4ce5a97`
**Why:** because relationship edges are global and accumulate across 30+ prior runs, the gate counted 32 domain edges when only ~3–20 belong to the current run, scoring stale pre-fix edges (NULL `pipeline_run_id`) → false FAIL even after Fix H. The run-scoped EXTRACTED_FROM check already filters by `pipeline_run_id`; the relationship check must too.
**Change:** `relationship_edge_rows` now scopes by `pipeline_run_id = :run` instead of `document_ids CONTAINS :doc`, preserving the domain-edge-type exclusion and the per-edge `source_chunk_ids` PASS condition. This measures the lineage of the edges *this run* committed — the correct semantics.

---

## 6. The re-gate runs

- **d6638d8e** (Fix A+B+C build): entity ✅, recall ✅, field ✅-in-data-but-gate-bug, relationship ❌. Drove Fix D1/G + the D2 diagnostic.
- **625ad1bd** (Fix D1+G+R+F build, deploy verified live): **3 of 4 PASS** —
  - ✅ ENTITY precision (run-scoped EXTRACTED_FROM=238, fan-out median 2 / max 25 of 102 chunks).
  - ✅ **FIELD precision: 69 rows, 0 null chunk_id** (was 6 nulls + ~60 imprecise) — **Fix F proven end-to-end.**
  - ✅ RECALL (28 committed ≥ 22 merged = 127%); no_chunk_metadata=0, drops=0, gate-rejections=0.
  - ❌ RELATIONSHIP precision (3/32 edges) — root-caused to the upsert membership gate → **Fix H + Fix G2**.
  - 3 truncations, all recovered cleanly (the missile passes hit the 32K cap and recovered via 65K retry).

---

## 7. Current state / what we're doing now

All seven fixes (D1, G, R×3, F×2, H, G2) are **committed and TDD-verified** on `walltime/c0-telemetry`. Fix R and Fix F each went through implementer → spec review → code-quality review. Fix H carries a passing **live-ArcadeDB integration test**. Fix G2 has a run-scoping unit test.

**Immediate next steps (in order):**
1. (Optional diligence) spec/quality review of Fix H + Fix G2 — both are small, well-rooted, and verified, so this is light.
2. **Deploy Fix H** — worker-only (`app/services/arcadedb_graph.py` is bind-mounted): restart **both** `worker-1` and `worker-graph-1`, verify `StartedAt` advanced and `CASE WHEN document_ids` is present in-container. **No docling-graph rebuild needed** (Fix H + G2 don't touch the COPY'd service). Fix G2 is host-script only.
3. **Verify** — see §8.
4. If gate is **ALL PASS**: update `VERIFICATION_CHECKLIST.md` + README with the new lineage-precision behaviors, then **PAUSE for explicit user go-ahead** before the 21-doc P-series collection (`#13`).

## 8. How to verify cheaply (avoid another 5h run)

Fix H is purely in the upsert SQL; `_build_relationship_records` was proven correct (19/20 edges in repro). So a full 5h `graph_only` re-extraction is likely unnecessary. Preferred path:
- **Re-run only the merge/commit phase for the existing run 625ad1bd** — its 5 pass outputs are already persisted. Re-dispatching `derive_ontology_graph_merge` (with the fixed code live) would re-upsert the ~20 edges via the now-ungated UPDATE, attaching `source_chunk_ids` + `pipeline_run_id` to all of them in ~1s. Then run `scripts/verify_lineage_e2e.py --run 625ad1bd-016d-46e4-8158-38720d0fe88d` (now run-scoped via Fix G2) → relationship precision should pass. **Caveat to check first:** whether re-dispatching the merge on a COMPLETE run is blocked by the run-state machine / reconcile guard, and whether it re-triggers the downstream chain. If unsafe, fall back to a fresh `graph_only` run.
- Fallback: a fresh `graph_only` run (~5h) via `POST /v1/documents/ddaa9e36-2854-47c3-bc94-ff38d531dafd/reingest {"mode":"graph_only"}` on api `localhost:8005`, then the detached gate driver `scripts/run_precise_lineage_gate.sh <run_id> 0`.

## 9. Known residuals (NOT blockers for this gate)

- **VARIANT_OF emits 0 relationship_provenance rows** from docling-graph — a separate coverage gap (`project_variant_of_coverage_gap`). After run-scoping, VARIANT_OF edges either won't be counted or need their own provenance path; track separately.
- **Coarse-smear fully eliminated only for ref-bearing rows.** Rows with neither resolvable refs nor instance ids still fall back (now logged). On this corpus that's rare.
- **Chunks genuinely lacking `self_refs` in ArcadeDB** can't be bridged by Fix F; those would surface as the guard/fallback, not silently.

## 10. Key files & commits

| Fix | Commit | Files |
|-----|--------|-------|
| D1 pipeline_run_id | `26c0a15` | `app/services/arcadedb_graph.py` |
| G gate queries | `d9ffcfc` | `scripts/verify_lineage_e2e.py` |
| R rel per-edge | `6fb2691`,`1c93fd3`,`9e2c2d6` | `docker/docling-graph/app/{provenance,schemas}.py`, `app/workers/pipeline.py`, `app/services/extraction_merge.py` |
| F chunk bridge | `cbacd11`,`f3fe0fa` | `app/workers/pipeline.py` |
| H un-gate UPDATE | `9bc144a` | `app/services/arcadedb_graph.py` |
| G2 run-scope gate | `4ce5a97` | `scripts/verify_lineage_e2e.py` |

Gate script: `scripts/verify_lineage_e2e.py` (run with `--run <id>`). Detached driver: `scripts/run_precise_lineage_gate.sh <run_id> <pre>` → writes `reports/collection/precise_lineage_gate_result.txt`.

## 11. Deploy / verify runbook (the gotchas that have bitten us)

- **Verify code is live in-container before any run.** Stale builds caused a false "failure" once. `docker exec <container> grep -c <marker> <file>`.
- docling-graph = COPY → `docker compose -p eip-mmdpp build docling-graph && up -d docling-graph`; confirm new image SHA + health.
- worker = bind-mount → `docker restart eip-mmdpp-worker-1 eip-mmdpp-worker-graph-1`; confirm `StartedAt` advanced.
- Restart **both** workers (worker-1 is a catch-all that also consumes the graph queue).
- Confirm the **Ollama pool is idle** before timing-sensitive runs (active-inference probe should return <~1s); truncation under contention corrupts results.
- The gate verdict file is **overwritten per run** — confirm `head -2` names the run you expect.
- Never use `POST /v1/documents/{id}/cancel` — it hard-deletes the document + all derived data.
