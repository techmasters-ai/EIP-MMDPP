# Deterministic Batch-Positional Lineage — Design

- **Date:** 2026-06-01
- **Status:** Design (awaiting user review → implementation plan)
- **Worktree:** `walltime/c0-telemetry`
- **Supersedes the precision approach in:** `2026-05-31-last-chunk-metadata-store-fix.md` (Parts A/B/C shipped; the
  LLM-citation precision mechanism in Part B is demoted to optional by this design).
- **Related:** [[project_extracted_from_root_cause]], [[project_precise_lineage_validation]],
  handoff `docs/operational/2026-06-01-lineage-precision-handoff.md`.

## 1. Problem & principle

Entity→chunk lineage edges (`EXTRACTED_FROM`) now exist (a prior fix took them 0→6844) but are **coarse**:
every entity links to ~all 102 chunks of the document (SNR-75 = 736 edges). This fails the hard requirement:
**maintain recall AND give every entity, field, and relationship precise lineage to its exact source
chunk + document + page.**

Root of the coarseness (validated by three independent code reviews, [[project_precise_lineage_validation]]):
the system tried to use the **LLM's per-node `evidence_ids` citations** as the precision signal. When the model
cites nothing valid, `_attach_evidence_to_prov` (ir_normalizer.py:56) falls back to the whole batch evidence
pool (`evidence_ids = valid or list(batch_evidence_ids)`), and the worker then fans out to *all* document
chunks. On the last run, 66 entities collapsed to a single shared self_ref. **Relying on the LLM to cite its own
source is fragile and unnecessary.**

**Principle: lineage is positional, not model-asserted.** At extraction time the runtime deterministically knows
which chunk(s) fed each batch — `batch_plan[batch_index] → chunk_indexes → chunk_metadata[ci].{self_refs,
page_numbers}`. The normalizer already computes this exact `provenance` dict (ir_normalizer.py:607-613) but then
lets the LLM's `valid` list overwrite it. **The fix inverts the priority: the batch's positional chunk lineage
is always stamped and authoritative; the LLM citation is at most an optional within-batch narrowing hint, never
the source of truth.** This dissolves the recall/precision conflict — nothing depends on the model citing, so no
entity is ever dropped for being "uncited."

## 2. One mechanism, three targets (entity / field / relationship)

The normalizer loop (`ir_normalizer.py:586`, `for batch_index, graph in enumerate(batch_results)`) already visits
**nodes** (entities, line 783), their **`property_evidence`** (fields, line 796), and **relationships**
(line ~895) — all with `batch_index` + `batch_plan` + `chunk_metadata` in scope, and `__delta_node_uid =
f"b{batch_index}:n{node_idx}"` already records the batch. One positional-stamp change covers all three:

- **Entity** → `node.provenance` stamped with the batch's `self_refs` + `chunk_indexes` + `page_numbers`
  (authoritative; replaces the `valid or batch_pool` overwrite).
- **Field** → each `node.provenance.property_evidence` entry stamped with the same batch lineage (a property
  extracted in batch *b* came from batch *b*'s chunk(s)).
- **Relationship** → `rel.provenance` stamped identically at line ~895 (same positional source). Closes the
  previously "scoped-out" relationship gap with the same one-line priority inversion — not separate machinery.

**LLM citations are demoted to diagnostics, not precision.** Per the review recommendation, Part B
("prefer per-node `evidence_ids`", commits `c77d174`/`acf24cd`) is **simplified out of the load-bearing path**:
`_resolve_element_uid` no longer prefers the single cited self_ref. The LLM's cited refs are preserved ONLY as an
advisory `cited_refs` field on provenance (diagnostic — lets us later measure citation quality / build optional
narrowing), and never drive chunk resolution.

## 2.5 Wire contract (end-to-end provenance payloads)

The positional lineage is a **list** of chunk anchors per item (the batch's chunk-set), so every hop must carry
lists, not the single scalar `element_uid`/`chunk_index` the current path collapses to. This section is the
authoritative contract the plan implements; each hop is verified against current code.

**Entity provenance** (service `ExtractionProvenance`, schemas.py:304 — already has `evidence_ids`/`page_numbers`
lists; ADD `self_refs`/`chunk_indexes` lists + `cited_refs` diagnostic):
| field | meaning | source |
|---|---|---|
| `self_refs: list[str]` | batch's chunk self_refs (`#/texts/N`) — AUTHORITATIVE | `batch_self_refs` (ir_normalizer.py:611) |
| `chunk_indexes: list[int]` | batch's chunk indices | `chunk_indexes` (ir_normalizer.py:587) |
| `page_numbers: list[int]` | batch's chunk pages | `page_numbers` (ir_normalizer.py:610) |
| `cited_refs: list[str]` | LLM-cited self_refs — DIAGNOSTIC only | `valid` (ir_normalizer.py:55) |
| `element_uid: str` | kept for back-comp = `self_refs[0]` | derived |

- Worker `ExtractionProvenance` (extraction_merge.py:~175) gains the same `self_refs`/`chunk_indexes`/`cited_refs`
  lists; `_parse_pass_response` populates them from the response.
- `mentions[]` (pipeline.py:362-371) — currently forwards scalar `element_uid`/`chunk_index`/`page`; ADD
  `self_refs`/`chunk_indexes`/`page_numbers` lists so the batch-set survives the audit-blob boundary (this is the
  exact point the list is dropped today).
- `_resolve_mention_chunks` (pipeline.py:1771) — gains a `batch_chunk_self_refs`/`batch_chunk_indexes` parameter;
  resolves each self_ref→chunk; on miss, attributes to the batch chunk-set (Section 3a); NEVER all-document.

**Field provenance** (service `ExtractionFieldProvenance`, provenance.py — currently built by
`build_auto_field_evidence` via value-text-match over chunks, provenance.py:481): REPLACE the text-match source
with **positional** batch lineage from `node.provenance.property_evidence`. Carry through worker `FieldEvidenceRow`
(extraction_merge.py / pipeline.py:3855), which gains `self_refs: list[str]` + `chunk_indexes: list[int]` and gets
`chunk_id` RESOLVED (no longer hardcoded `None`, pipeline.py:3856). Per-field anchor = the batch chunk-set of the
batch that emitted that property.

**Relationship provenance** (service rel `provenance`, ir_normalizer.py:~895; worker `RelationshipRecord.provenance`,
pipeline.py:1510): same positional `self_refs`/`chunk_indexes`/`page_numbers` lists. **Representation decision
(chosen): resolved chunk refs as edge PROPERTIES** — not a new edge/vertex type. `upsert_relationships_batch_sync`
already returns edge RIDs (arcadedb_graph.py:2452) but the caller discards them (pipeline.py:1521); CAPTURE them and
write `source_chunk_ids: list`, `source_pages: list`, `source_self_refs: list` as edge properties (resolved in the
worker via the same self_ref→chunk map). Mirrors how entity provenance already rides as properties; lineage dies
with the edge on delete (no extra cascade). Query surface: read the edge's `source_chunk_ids`/`source_pages` (a
property filter, not a graph traversal — accepted tradeoff). Register the new edge properties in arcadedb_schema.py
(Phase 2 rel-edge property DDL, ~line 335).

## 3. Worker resolution + a home for field/relationship lineage

### 3a. Resolve to the batch's chunk(s); delete all-document fan-out
Once the service stamps authoritative batch self_refs, the worker's `derive_structure_links` resolves
`#/texts/N → chunk_id` via `identity_map`, and on a miss **attributes to the stamped batch `chunk_indexes` —
never fans out to all document chunks.** Two fan-out sites get this rule and the all-document branch is DELETED:
`_resolve_mention_chunks` (~pipeline.py:1771) and the legacy `content_metadata` block (~pipeline.py:9416-9436;
confirm-or-fix — reviewer found it likely inert for the delta path but it must be verified on a real run).

**Namespace gap (must close, or resolution always misses):** `identity_map` is keyed on ingest's
`elements[].metadata["self_ref"]` (pipeline.py:4877-4881) while provenance uses `#/texts/N`. These must be
reconciled so resolution actually hits. Because lineage is positional, a resolver miss never drops an
entity — it lands on the batch's chunk(s), which always exist (recall safe by construction).

### 3b. A home for field & relationship lineage (per §2.5 wire contract)
- **Field:** resolve each `property_evidence` self_ref → chunk via the same resolver; store resolved
  `chunk_id`+`page`+`self_refs`+`chunk_indexes` on the entity's `_field_evidence` (today `FieldEvidenceRow.chunk_id`
  is hardcoded `None`, pipeline.py:3856, never resolved). Surface via the entity's existing provenance API.
- **Relationship:** resolve `rel.provenance` self_refs → chunk and write `source_chunk_ids`/`source_pages`/
  `source_self_refs` as **properties on the domain edge** (capture the upsert RID discarded at pipeline.py:1521;
  register props in arcadedb_schema.py). NOT a new edge/vertex type — see §2.5 representation decision. A
  relationship then traces to its source chunk(s) via its edge properties, replacing the opaque JSON-string
  provenance property with no chunk link (arcadedb_graph.py:280-283).

This is mechanical (reuse the one resolver across all three) and is what makes "every entity, field, AND
relationship" literally true rather than entity-only.

## 4. Precision guarantee & the multi-chunk-batch boundary

Precision is bounded by **batch width**:
- **Batch = 1 chunk** (configured norm: `batch_token_size == chunk_max_tokens == 512`; observed 20 chunks/20
  batches, 22/19) → every entity/field/rel resolves to **exactly one chunk**: single-chunk precise,
  deterministic, no model involvement. Common case.
- **Batch > 1 chunk** (rare; e.g. 70 chunks/67 batches when small chunks merge) → a **provably-correct K-chunk
  set** ("from one of these K adjacent chunks of this batch"), never wrong, never all-document. K is small (2-3).

**Guarantee:** exact chunk when batch=1 chunk; a tight, correct, adjacent K-chunk set otherwise; always the real
page(s). Meets "exact source chunk+document+page" in the common case; degrades gracefully (not catastrophically)
in the rare multi-chunk batch.

**Single-chunk-everywhere** is an optional, measured follow-on (NOT a blocker for this design):
1. Config lever — force batch=1 chunk (`batch_token_size = chunk_max_tokens`) → K=1 always. Cost: more batches =
   more LLM calls = walltime (counts against the walltime goal).
2. Deterministic text-match narrowing within the batch's K chunks (model-independent string match over a tiny
   candidate set; mismatch falls back to the K-set, never wrong).
The Section 5 gate **measures the K-distribution** so we can see whether refinement is even needed before doing it.

## 5. Verification & precision gate

Harden `scripts/verify_lineage_e2e.py` to prove the real requirement across all three targets, run-attributed:

- **Entity precision (positional):** per committed entity, `EXTRACTED_FROM` target-chunk count equals the batch
  width of its source batch (≈1 common case), NOT ~all chunks. Report the K-distribution. FAIL if any entity
  fans out to all-document.
- **Field precision (new):** `_field_evidence` rows carry a RESOLVED `chunk_id`+`page` (not `None`), and the chunk
  is within the entity's batch chunk-set.
- **Relationship precision (new):** committed relationships have a relationship→chunk lineage edge resolving to
  chunk+page within the source nodes' batch chunk-set.
- **Recall (both-constraints), per-target baseline:** define the baseline per target so valid rejection isn't
  misread as lineage loss —
  - *Entity:* committed entity vertices (run-scoped) == merged-entity count from the run's audit blob (the
    LLM-extracted set that survived the lineage gate). NOT the global vertex count (idempotent upsert makes a
    global delta meaningless — see [[project_precise_lineage_validation]]).
  - *Field:* `_field_evidence` rows with a resolved `chunk_id` == count of extracted property-evidence entries
    for committed entities (every extracted field keeps its lineage; none silently drop to `chunk_id=None`).
  - *Relationship:* committed edges == relationships that PASSED validation (`_enforce_relationship_triple`,
    arcadedb_graph.py:2489) — i.e. baseline is post-validation accepted rels, so a triple correctly rejected by
    schema validation does NOT count as lineage recall loss. Report rejected-rel count separately.
  Positional lineage makes per-target recall hold by construction; the gate proves it against these baselines.
- **Run-scoping & honesty:** all counts filtered by `pipeline_run_id` (not global); trace uses `EXTRACTED_FROM`
  specifically (NOT `MENTIONED_IN`, which pre-satisfies it); warnings scoped to the run window; fail-closed on
  no-evidence (missing log / empty result FAILS, never silently passes).
- **End-to-end:** fresh SA-2 `graph_only` run on the rebuilt+redeployed build; then the `/v1/graph/query` SNR-75
  check (secondary `sources=null` bug — verified AFTER precision; likely a separate RID-match/entity_id-fallback
  issue, graph_store.py:64 → arcadedb_graph.py:1488).

"ALL CHECKS PASS" then means: **recall held AND precise lineage for entity + field + relationship** — not
entity-only.

## 6. Components / file structure

- `docker/docling-graph/repo/.../delta/ir_normalizer.py` (~586-613, 783-806, ~895) — **library patch**: stamp
  authoritative positional batch `self_refs`/`chunk_indexes`/`page_numbers` on node.provenance, property_evidence,
  and rel.provenance; stop the `valid or batch_pool` overwrite; add `cited_refs` diagnostic. Ships as a new
  docling-graph patch (gitignored repo; patch in the build stack, Dockerfile loop already hardened `--fuzz=0 || exit 1`).
- `docker/docling-graph/app/schemas.py` (~304) — `ExtractionProvenance` += `self_refs`/`chunk_indexes`/`cited_refs`;
  `ExtractionFieldProvenance` += positional lists. `docker/docling-graph/app/provenance.py` — `_resolve_element_uid`
  no longer prefers single cited self_ref (Part B demotion, ~239); `build_auto_field_evidence` (~481) sourced from
  positional batch lineage, not value text-match.
- `app/services/extraction_merge.py` — worker `ExtractionProvenance` + `FieldEvidenceRow` += positional lists.
- `app/workers/pipeline.py` — `_parse_pass_response` populate lists; `mentions[]` (362) forward lists;
  `_resolve_mention_chunks` (1771) gains batch chunk-set param, fail-to-batch-set not all-document (BOTH fan-out
  sites incl. legacy 9416-9436); close `identity_map`↔`#/texts/N` namespace gap; resolve+store field
  `chunk_id`/`page`; capture rel-upsert RIDs (1521) + write `source_chunk_ids`/`source_pages`/`source_self_refs`
  edge props.
- `app/services/arcadedb_graph.py` (~2448 upsert path) + `app/services/arcadedb_schema.py` (~335 rel-edge property
  DDL) — register + write the relationship source-chunk edge properties (NO new edge type).
- `scripts/verify_lineage_e2e.py` — the all-three-targets + per-target-recall gate (Section 5).
- Tests: docling-graph normalizer unit test (positional stamp, all three, batch>1 K-set, cited_refs separate);
  worker resolver unit test (resolve→chunk, miss→batch-set not all-doc, namespace bridge); field + relationship
  lineage unit tests (resolved chunk_id; edge source-chunk props).

## 7. Scope / non-goals

- **In:** deterministic positional lineage for entity + field + relationship; worker fail-to-batch resolution +
  namespace-gap close; field/rel chunk-resolution + storage; all-three precision gate.
- **Out (measured follow-ons, not blockers):** single-chunk-everywhere narrowing (Section 4 options); the
  `/v1/graph/query sources=null` bug (verified after precision); the gemma4 truncation/contention behavior
  (positional lineage is truncation-independent — a truncated batch still has a known chunk).

## 8. Deploy notes

- docling-graph is COPY-based → rebuild + `up -d --force-recreate -p eip-mmdpp docling-graph`; restart BOTH
  `worker-1` (catch-all, also consumes `graph`) AND `worker-graph-1`; confirm all `StartedAt` advanced; idle pool
  before the verification run. Never `POST /v1/documents/{id}/cancel` (hard-deletes); to stop a run, revoke+reset,
  ask first.
