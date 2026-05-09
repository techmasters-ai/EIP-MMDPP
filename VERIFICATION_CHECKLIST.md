# EIP-MMDPP Feature & Behavior Verification Checklist

> **Purpose:** After every code change, bug fix, or new feature, this checklist must be reviewed to ensure no existing features have been removed or broken. Run the full test suite first (`./scripts/run_tests.sh`), then verify the items below that are relevant to the changed code.
>
> **Protocol:** Every code modification requires: (1) full unit test suite passes, (2) relevant sections of this checklist verified, (3) any new features added to this checklist before merging.

---

## Architecture Note (2026-04 ArcadeDB Migration)

This checklist reflects the post-migration architecture:
- **Graph + Vectors:** ArcadeDB (replaces Neo4j + Qdrant)
- **Document conversion:** Docling (unchanged)
- **Entity extraction:** Docling-Graph service with bundle-based five-pass extraction via `/extract-pass` endpoint (hand-authored fixed Pydantic schemas under `ontology_bundles/`)
- **Global query:** Community detection (Louvain/Leiden) + LLM report synthesis (replaces Microsoft GraphRAG)
- **Chunk storage:** PostgreSQL authoritative for content; ArcadeDB carries embeddings + filter metadata via TextChunk/ImageChunk vertices
- **Provenance:** EXTRACTED_FROM edges with document_id and page_numbers (entity vertices are shared across documents)

## ArcadeDB Authoritative Reference

**For ANY verification, feature check, or debugging related to ArcadeDB, consult `ArcadeDB Manual.pdf` (in the repository root) as the authoritative source.**

The manual is the ground truth for:
- **SQL syntax** (DDL, DML, graph traversal with `out()`/`in()`/`both()` and `{min,max}` depth, `LUCENE` fulltext queries)
- **Vector search** — Section 4.14. LSMVectorIndex with HNSW/Vamana, COSINE/DOT_PRODUCT/EUCLIDEAN similarity, INT8/BINARY/PRODUCT quantization, `vectorNeighbors()` function, `efSearch` parameter, multi-modal search
- **Schema management** — Section 4.8. Vertex types, edge types, properties, `CREATE TYPE IF NOT EXISTS`, inheritance, schema-full vs schema-less modes
- **Indexes** — Section 4.9. LSM_TREE, Hash, LSM_VECTOR, FULL_TEXT, case-insensitive collation
- **Graph algorithms** — Appendix 8.1. `algo.louvain`, `algo.leiden`, PageRank, centrality measures, node embeddings (Node2Vec, FastRP, GraphSAGE, HashGNN), path finding, similarity, structural
- **Vector functions** — Section 6.6. Full reference of `vectorNeighbors`, `vectorCosineSimilarity`, etc.
- **HTTP/JSON API** — Section 6.4. Endpoint paths, `{database}` path parameter, authentication, transactions
- **Cross-model queries** — Section 4.13.6. Combining graph traversal + vector similarity in one SQL
- **Multi-model architecture** — Section 4.13. How Graph, Document, Vector, Time-series models share the same storage
- **Transaction model** — Section 4.6. MVCC, isolation levels, optimistic locking
- **Docker deployment** — Section 5.5.20. Volume mounts, environment variables, health checks
- **Performance tuning** — Section 5.5 (Operations). Bucket selection, parallel writes, memory settings

When adding new ArcadeDB-related checklist items or debugging a failing test, verify behavior against the manual before asserting that the code is wrong. The manual reflects ArcadeDB's actual capabilities and limitations.

**Common debugging workflow:**
1. Test fails or behavior is unexpected → check `ArcadeDB Manual.pdf` for the relevant section
2. Confirm the SQL/API usage matches the manual
3. If the manual contradicts the implementation → fix the implementation
4. If the manual confirms the implementation → investigate test assumptions or environment issues

## Native-First Principle

**Always prefer native Docling, Docling-Graph, and ArcadeDB functionality over custom code.** This applies to every gap analysis, bug fix, feature addition, and refactor.

Before writing custom logic, check whether the library already provides the capability:

| Domain | Check first | Authoritative reference |
|--------|------------|------------------------|
| **Vector search** (ANN, scoring, fusion) | `vectorNeighbors()`, `vectorCosineSimilarity()`, `efSearch` parameter, distance metrics | ArcadeDB Manual §4.14, §6.6 |
| **Graph traversal** (neighborhood, path, pattern) | `MATCH` syntax, `out()`/`in()`/`both()` with depth, Cypher | ArcadeDB Manual §4.13.6, §6.3.1 |
| **Schema & indexes** | `CREATE TYPE IF NOT EXISTS`, LSM_VECTOR, FULL_TEXT, UNIQUE, BucketSelectionStrategy | ArcadeDB Manual §4.8, §4.9, §5.5.24 |
| **Graph algorithms** (community, centrality, similarity) | `algo.louvain`, `algo.leiden`, PageRank, Node2Vec | ArcadeDB Manual Appendix 8.1 |
| **Text search** (fulltext, fuzzy) | FULL_TEXT index with CONTAINSTEXT keyword, `text.levenshteinDistance()` | ArcadeDB Manual §4.9, §6.3.1 |
| **Entity/relationship extraction** | `run_pipeline()`, `PipelineConfig`, delta/staged extraction, entity merge | Docling-Graph library API |
| **Document conversion** (PDF, images, tables) | Docling service `/convert` endpoint, `DoclingDocument` JSON | Docling library API |

**Red flags for custom code that should use a library feature:**
- Custom Python-side score fusion when ArcadeDB has `vectorCosineSimilarity()` or `$distance`
- Custom graph traversal loops when a single MATCH query with depth control would suffice
- Custom entity dedup logic when Docling-Graph's delta normalizer handles entity merge
- Custom fulltext scoring when ArcadeDB's CONTAINSTEXT and `extraction_confidence` field are available
- Custom batch insert loops when ArcadeDB's `sqlscript` or batch endpoint handles multiple statements
- Custom concurrency gating when ArcadeDB's MVCC and BucketSelectionStrategy handle parallel writes

**During code review, verify:**
- [ ] No custom SQL builder that reimplements what MATCH or a native ArcadeDB function provides
- [ ] No custom vector scoring that ignores `$distance` or `vectorCosineSimilarity()`
- [ ] No custom traversal that could be a single MATCH with directed edges and depth bounds
- [ ] No custom extraction pipeline that bypasses `run_pipeline()` / `PipelineConfig`
- [ ] Graph fulltext results carry `extraction_confidence` from vertex properties
- [ ] Filters are pushed into ArcadeDB/Postgres WHERE clauses, not post-filtered in Python

---

## 1. INGEST PIPELINE

### 1.1 Document Preparation & Validation

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Docling document conversion (VLM-based PDF parsing) | PDFs cannot be parsed; documents stuck in PROCESSING | Upload PDF, check status progresses through PREPARE stage | 1, 2.21 |
| DoclingDocument JSON persisted to MinIO | `derive_ontology_graph` cannot retrieve structured document for delta extraction | `eip-derived/{doc}/docling_document.json` exists after prepare_document | 3.0 |
| Standalone image file synthesis (JPEG, PNG, TIFF, BMP, GIF, WEBP) | Standalone images ingest with 0 elements; blank viewer | Upload `.jpg`, check `document_elements` has 1 image element with `storage_key` | 2.30 |
| Unicode normalization (em-dashes, non-breaking spaces) | NaN embeddings from bge-m3; elements silently drop from vectors | Ingest doc with em-dashes; verify ArcadeDB TextChunk vertices have valid float embeddings | 2 |
| Stale run cleanup on worker startup | Crashed workers leave documents in PROCESSING permanently | Kill worker mid-ingest, restart; document reverts to PENDING | 2.18 |
| Re-upload on failure | Re-uploading failed doc returns 409 indefinitely | Upload doc, let it fail, re-upload same file without error | 2 |

### 1.2 Pipeline Stage Ordering & Resilience

The ingest pipeline is a sequential 12-stage Celery chain. Each stage receives `(document_id, run_id)` and writes a `StageRun` row. The stages execute in this exact order:

```
1.  prepare_document                — Docling conversion, element extraction, MinIO persistence
2.  detect_and_translate            — Per-element language detection + LLM translation
3.  derive_document_metadata        — LLM metadata extraction (summary, date, classification)
4.  purge_document_derivations      — Clean up prior run data (for reingest)
5.  derive_picture_descriptions     — LLM multimodal image descriptions
6.  derive_text_chunks_and_embeddings — Chunking + BGE text embedding → ArcadeDB TextChunk vertices
7.  derive_image_embeddings         — CLIP image embedding → ArcadeDB ImageChunk vertices
8.  derive_ontology_graph           — Bundle-passes extraction (see §2.0 for full workflow)
9.  collect_derivations             — Post-derivation checkpoint (marks PROCESSING)
10. derive_structure_links          — Chunk links (NEXT_CHUNK, SAME_SECTION, SAME_ARTIFACT, SAME_PAGE)
11. derive_canonicalization         — Entity alias resolution (exact → alias → fuzzy → new)
12. finalize_document               — Terminal status (COMPLETE / PARTIAL_COMPLETE)
```

`start_ingest_pipeline(document_id, *, ontology_bundle_key=None, use_case_key=None)` resolves the bundle via three-tier precedence (explicit → source default → system default), snapshots bundle metadata onto the `PipelineRun` row, and dispatches the 12-stage chain. Returns `IngestDispatchResult(pipeline_run_id, celery_task_id)`.

`reingest_graph_only(doc_id, request)` resolves the bundle via four-tier graph_only precedence (explicit → inherited from latest run → source default → system default), creates a new `PipelineRun` with `mode='graph_only'`, and dispatches a 3-stage chain: `derive_ontology_graph → derive_structure_links → finalize_document`.

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Sequential 12-stage chain (no parallel chord) | Stage ordering violations; race conditions | Upload doc; verify stages execute in order via `GET /documents/{id}/stages` | 3.2 |
| Stage tasks return error dicts instead of raising | Single task failure kills entire document pipeline | Ingest doc with corrupted image; pipeline still reaches PARTIAL_COMPLETE | 2.22 |
| `ensure_ready_sync()` called on every graph-writing task | Worker starts before ArcadeDB schema sync; writes fail | Each derive_* task calls `graph_store.ensure_ready_sync()` before first write; retries on missing types | 3.0 |
| Bundle snapshotted on PipelineRun at dispatch time | Downstream stages use wrong ontology version | `PipelineRun.ontology_bundle_key` matches the resolved bundle; does not change if system default changes mid-run | 3.2 |
| Duplicate dispatch prevention (FOR UPDATE) | Concurrent uploads create multiple PipelineRuns for same document | `start_ingest_pipeline` uses `SELECT ... FOR UPDATE` on active runs; returns existing run_id if active | 2.23 |
| `graph_only` reingest inherits bundle from latest run | Re-extraction uses wrong ontology after source default changes | `reingest_graph_only` reads `latest_run.ontology_bundle_key` as inheritance tier; falls back to source/system default if NULL (legacy run) | 3.2 |

### 1.3 Element Deduplication

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Ingest-time element dedup (modality+page+section+text+bbox) | Duplicate elements bloat vector store; redundant search results | Parse doc with duplicate captions; check `document_elements` has only unique entries | 2.8, 2.20 |
| Image dedup includes raw bytes hash | Distinct images on same page with empty captions silently dropped | Ingest doc with 2+ distinct images on same page; both appear in `document_elements` | 2.30 |
| Text chunk dedup before embedding | Duplicate text vectors waste ArcadeDB space | Doc with repeated sections appears only once per unique text in TextChunk vertices | 2.20 |

### 1.4 Picture Description & Image Analysis

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| LLM-generated image descriptions (multimodal VLM) | Images not text-searchable; CLIP embedding insufficient for specific queries | Ingest doc with images; call `/v1/documents/{id}/image-descriptions`; receive descriptions | 2.9 |
| Image descriptions split into sections and embedded as BGE text vectors | Image descriptions not discoverable via text search | Query concept visible in image; find `image_description` modality results | 2.26 |
| SAME_ARTIFACT chunk_links between description sections | Description sections orphaned from source image; graph expansion fails | Text query matches description section; expansion retrieves siblings + original image | 2.26 |
| Image description hover tooltips in DoclingViewer | No context when hovering over embedded images in viewer | Open PDF in viewer; hover over image; see "AI Image Analysis" tooltip | 2.30 |
| Standalone image display with description panel | Standalone images appear with no visual or description | Open standalone image in viewer; see image rendered + AI Image Analysis panel | 2.30 |
| `image-descriptions` endpoint returns `artifact_id` | Frontend cannot render standalone image in fallback panel | Call `/v1/documents/{id}/image-descriptions`; response includes `artifact_id` | 2.30 |
| SoftTimeLimitExceeded handler for picture descriptions | Timeout kills task without cleanup; document stuck PROCESSING | Set short limit; task returns error dict; document reaches PARTIAL_COMPLETE | 2.25 |

### 1.5 Embedding & Vector Storage

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Text embedding creates TextChunk vertex + `text_embedding` property | PostgreSQL chunk row exists but no vector for search | Ingest doc; verify ArcadeDB `SELECT FROM TextChunk WHERE document_id = :id` returns vertices with embeddings | 3.0 |
| Image embedding creates ImageChunk vertex + `image_embedding` property | Image search returns no results | Ingest doc with images; verify ArcadeDB ImageChunk vertices have 512-dim CLIP embeddings | 3.0 |
| PostgreSQL authoritative for chunk content | Chunk text/content lost when vectors move to ArcadeDB | `retrieval.text_chunks` still has `chunk_text`, `page_number`, `classification`; only `embedding` and `qdrant_point_id` columns removed | 3.0 |
| `chunk_id` bridge between stores | Cross-store lookups fail | PostgreSQL `text_chunks.id` matches ArcadeDB `TextChunk.chunk_id` | 3.0 |
| CLIP image embedding (OpenCLIP ViT-B-32, 512-dim) | Cannot match image queries; visual content invisible to retrieval | Query with `query_image` + `strategy=hybrid`; receive image matches | 2 |
| BGE asymmetric query/passage prefixes | Query-document semantic matching degrades | Query "S-75 Dvina" matches document mentioning system in top-5 | 2.24 |

### 1.6 Translation & Language Handling

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Per-element language detection + LLM translation | Non-English text not translated; search fails on foreign text | Ingest Russian doc; `translated_text` populated; downstream uses English | 2.27 |
| Classification marking detection on original text | Translated text loses classification context; wrong marking | Ingest doc marked "SECRET" in Russian; classification correctly identified | 2.27 |

### 1.7 Document Analysis & Metadata

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| LLM-based metadata extraction (summary, date, classification, source) | Documents lack context for retrieval ranking | Ingest report; `/v1/documents/{id}/metadata` returns all fields | 1, 2.12 |
| Per-document summary used as image description context | Image descriptions lack document context; generic output | Ingest doc with images; descriptions reference document summary content | 2.9 |

---

## 2. GRAPH EXTRACTION & ONTOLOGY

> **ArcadeDB reference:** Consult `ArcadeDB Manual.pdf` sections 4.8 (Schema), 4.10 (Graph Database), 4.9 (Indexes), and 6.10 (SQL Syntax) when verifying ArcadeDB-backed features below.

### 2.0 Extraction Workflow — End-to-End Technical Detail

This section documents the full logical flow inside `derive_ontology_graph` when `graph_extraction_engine` is the bundle-passes path (the only path after PR 3). Each numbered step corresponds to a code path that can be individually verified.

**Architecture (per-pass fan-in, 2026-05-06):** `derive_ontology_graph` now operates as a thin **dispatcher** rather than a monolithic inline executor. On entry it writes a summary `StageRun` (RUNNING), then claims each pass from the `dispatched_phases` JSONB state machine and publishes a per-pass Celery task (`run_extract_pass`) to the broker. Each `run_extract_pass` task executes one ontology pass — HTTP call, StageRun write, `pipeline_pass_outputs` row — and terminates. A separate **fan-in** task (`finalize_ontology_graph_passes`) fires after all per-pass tasks resolve; it collects pass outputs, runs the required-pass gate, merge/resolve, three-phase graph import, and writes the summary StageRun to COMPLETE (or FAILED). A beat-scheduled **reconciler** (`reconcile_ontology_graph_runs`) re-dispatches passes that became stranded (stale `claimed` > 30 s; stale `dispatched` > 2 × `pass_soft_time_limit` without a matching `pipeline_pass_outputs` row).

Key behavioral differences from the legacy monolithic path: (1) a single bad pass is isolated — the remaining passes continue; (2) already-resolved passes are skipped on retry (idempotent re-dispatch); (3) the overall graph-stage time limit is no longer a single 8 h budget — each pass has its own 1 h `pass_soft_time_limit`; (4) optional passes that fail degrade gracefully, leaving the run in PARTIAL_COMPLETE with the successful passes' graph data present; (5) cancel-mid-extraction is safe via a targeted FK swallow that does not obscure unrelated integrity errors.

**Entry:** The Celery task `derive_ontology_graph(self, document_id, run_id)` is called as stage 8 of the 12-stage ingest pipeline. It is preceded by `derive_image_embeddings` and followed by `collect_derivations`.

**Step 1 — Stage-summary StageRun creation.** A summary row is inserted into `ingest.stage_runs` with `pass_name=NULL`, `stage_name='derive_ontology_graph'`, `status='RUNNING'`, `attempt=self.request.retries+1`. This row tracks the overall stage outcome (separate from per-pass rows). The `stage_summary_id` is captured for later terminalization.

**Step 2 — GraphWriteTracker initialization.** A `GraphWriteTracker()` instance is created with `any_mutation_attempted=False`. This tracker is passed to every phase helper and flipped to `True` immediately before the first graph_store mutation. If the orchestrator catches an exception, it consults `tracker.any_mutation_attempted` to decide whether rollback is needed.

**Step 3 — Bundle + ontology + document loading.** The `PipelineRun` row is read to get `ontology_bundle_key`. The bundle manifest is loaded via `load_bundle_manifest(bundle_key)`. The ontology is loaded via `load_ontology(bundle_key=bundle_key)`. The enriched DoclingDocument JSON is downloaded from MinIO via `_build_docling_document_json(document_id)`.

**Step 4 — Per-pass loop.** For each pass in `manifest.passes` (in declared order):

- **4a — Skip check** (`_should_skip`). Only fires for `kind=relationships_only` passes with `skip_if_no_upstream_endpoints=True`. Walks `depends_on`, filters `upstream_refs` by `pass_origin`, and checks whether any `(source_type, rel_type, target_type)` triple in `validation_matrix` can be satisfied. If not, writes a StageRun with `execution_status=SKIPPED`, `skip_reason=NO_UPSTREAM_ENDPOINTS` and returns.

- **4b — HTTP call** (`_call_extract_pass`). POSTs `{bundle_key, pass_name, docling_document_json, upstream_entities?}` to the docling-graph service's `/extract-pass` endpoint. Transport errors and HTTP 5xx raise `PassRetryable`; HTTP 4xx raises `PassTerminal`.

- **4c — Response parsing** (`_parse_pass_response`). The service response's `pass_output` dict is validated into the pass's Pydantic template class via `model_validate()`. Pydantic validation failures raise `PassTerminal` (not retryable — the LLM output shape won't heal on retry).

- **4d — Retry / terminal handling.** `PassRetryable` triggers backoff (`30s × 2^(attempt-1)`, capped 300s) and a new StageRun row with `attempt+1`. After `pass_max_retries` (default 3) exhausted, a required pass raises `IngestFailed`; an optional pass returns silently. `PassTerminal` immediately raises `IngestFailed` for required passes.

- **4e — Yield classification** (`classify_yield`). Computes `YieldStatus` from entity/relationship counts: `DEGRADED` (≥4 rels, ≥75% rejected), `EMPTY` (0 primary + 0 bridge + 0 rels), `BRIDGES_ONLY` (0 primary, >0 bridge), or `HIT` (everything else).

- **4f — StageRun write** (`_write_stage_run`). Upserts a per-pass StageRun row with `execution_status`, `yield_status`, `skip_reason`, entity/relationship counts, schema_size_chars, structured_output_mode. Uses the partial unique index `uq_stage_runs_run_pass_attempt`.

- **4g — Upstream ref extension.** If any downstream pass `depends_on` this pass, the extracted entities are added to `upstream_refs` keyed by compact ref_ids (`E001`, `E002`, ...) so the downstream pass can reference them via `from_ref_id` / `to_ref_id`.

**Step 5 — Required-pass gate** (`check_required_pass_gate`). Queries the latest StageRun per required pass. `COMPLETE` or `SKIPPED` with authorized reason (`NO_UPSTREAM_ENDPOINTS`) passes. `FAILED` or unauthorized skip fails. Missing StageRun raises `WorkerInvariantError`. Gate failure raises `IngestFailed`.

**Step 6 — Merge and resolve** (`merge_and_resolve`). Two-pass algorithm:
- **Entity merge.** Walks every entity type in the ontology; for each, iterates the pass template's list field (e.g., `radar_systems`). Builds a `LogicalIdentity` from `identity_fields` + `identity_scope`. Bridge entities with identical identity across passes collapse into one `MergedEntityRecord` with `pass_origins = {pass_a, pass_b}`. Confidence = max across merges.
- **Relationship resolve.** For each relationship, checks in order: `MISSING_REL_TYPE` → `INVALID_IDENTITY_PAYLOAD` → `UNKNOWN_REF_ID` (system_links ref_id lookup) → `FROM_ENDPOINT_NOT_FOUND` → `TO_ENDPOINT_NOT_FOUND` → `INVALID_TRIPLE` (validation_matrix check). Accepted edges get `confidence = 0.8 if None else raw` (explicit 0.0 preserved). Rejected edges are counted per-pass and sampled in metrics.

**Step 7 — Post-merge yield updates** (`_apply_post_merge_yield_updates`). Recomputes yield_status for each COMPLETE pass using post-merge rejection counts. Only `HIT → DEGRADED` transitions are applied (the only direction spec §6.2 allows post-merge).

**Step 8 — Run metrics write** (`_write_pipeline_run_metrics`). Populates `PipelineRun.metrics` JSONB with: `pass_outcomes` (rolled up from `v_latest_pass_attempts`), `document_extraction_anomaly` (true iff all 3 core passes ended EMPTY/BRIDGES_ONLY), `pass_degraded_count`, `overall_relationship_rejection_ratio`, `rejected_relationships_sample` (up to 20 per pass per reason), `bundle_legacy` (always False for new runs), `bundle_key_display`.

**Step 9 — Three-phase graph import.**
- **Phase 2 — Node upsert** (`_import_graph_phase_nodes`). Builds `NodeRecord` list in pure Python. Calls `tracker.mark()`. Calls `graph_store.upsert_nodes_batch_sync(records, provenance)`. `HAS_PROVENANCE` edges are auto-created by the graph store. Returns `identity_to_rid` mapping.
- **Phase 3 — Domain edge upsert** (`_import_graph_phase_domain_edges`). Builds `RelationshipRecord` list. Calls `tracker.mark()` (idempotent). Calls `graph_store.upsert_relationships_batch_sync(records, provenance)`. Domain edges carry `document_ids` as a LIST, not a single string.
- **Phase 4 — Structural edges** (`_import_graph_phase_structural_edges`). Loads TextChunk vertices from ArcadeDB. Calls `derive_rules.derive_structural_edges()` to produce `MENTIONED_IN` edges (entity display label substring-matched against chunk text). Iterates derived edges, calling `tracker.mark()` + `graph_store.create_structural_edge_sync()` for each. Empty derived list = true no-op.

**Step 10 — DocumentGraphExtraction snapshot write** (`_upsert_document_graph_extraction`). Upserts the `DocumentGraphExtraction` row with audit-blob `graph_json` (entity/edge counts by type, primary vs bridge totals, rejection reasons), `pipeline_run_id`, `ontology_bundle_key`, `ontology_name`, `ontology_version`.

**Step 11 — Success terminalization.** Stage-summary StageRun → `COMPLETE`, `rollback_executed=False`. For `graph_only` mode: `PipelineRun.status=COMPLETE`, `Document.pipeline_status=COMPLETE`. For `full` mode: PipelineRun stays `PROCESSING` (downstream `finalize_document` terminalizes).

**Failure paths:**
- **Gate failure** (`IngestFailed`): `rollback_executed=False`, `PipelineRun.status=FAILED`, `Document.pipeline_status=PARTIAL_COMPLETE`. No rollback (no graph writes occurred).
- **Unexpected exception** (any other): checks `tracker.any_mutation_attempted`. If True: calls `_attempt_rollback(document_id)` → `_delete_extraction_layer_graph(document_id)` → `graph_store.delete_extraction_layer_graph_sync()`, sets `rollback_executed=True`. If False: skips rollback, sets `rollback_executed=False`. In both cases: `PipelineRun.status=FAILED`, `Document.pipeline_status=PARTIAL_COMPLETE`.
- **Bookkeeping failure** (exception during terminalization itself): logged at ERROR, session rolled back, original exception re-raised.

**Rollback contract** (`delete_extraction_layer_graph_sync`):
- DELETES: document-scoped entity vertices, domain edges (remove doc from `document_ids` list + prune empty), `HAS_PROVENANCE` edges targeting this Document vertex (via `@in`), `MENTIONED_IN` edges with `source='derive_rules'`.
- PRESERVES: TextChunk/ImageChunk vertices, the structural Document vertex, global entity vertices, cross-document `HAS_PROVENANCE` edges.

### 2.1 Bundle-Based Extraction Architecture

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Bundle-based extraction (`ontology_bundles/air_defense_v3/`) | No extraction schemas; pipeline cannot extract | Bundle directory contains ontology.yaml, manifest.yaml, coverage.yaml, validators.py, 5 extraction schema modules, derive_rules.py. Run `python tools/check_extraction_coverage.py` — 0 errors | 3.2 |
| Five-pass extraction (reference, radar_domain, missile_domain, other_systems, system_links) | Monolithic extraction misses domain-specific entities | All 5 passes declared in manifest.yaml; each pass has a corresponding module in `extraction_schemas/` with a top-level Pydantic class | 3.2 |
| Per-pass StageRun tracking | Cannot diagnose which pass failed or produced low yield | Each pass writes its own StageRun row with execution_status, yield_status, skip_reason, entity count, relationship count | 3.2 |
| Required-pass gate | Merge proceeds with incomplete data; graph missing critical entities | All 5 required passes must complete or be authorized-skipped before merge proceeds; IngestFailed if any required pass exhausts retries | 3.2 |
| Transport-error retries don't burn business pass budget | Service restarts kill in-flight extractions and exhaust the 3-attempt business limit; docs marked degraded for infra reasons | `httpx.TransportError`/`TimeoutException` raise `PassTransportError` (subclass of `PassRetryable`); `_run_single_pass` retries up to `pass_max_transport_retries` (default 3) on a separate counter without incrementing business `attempt` | 3.2 |
| Service stub-on-error → retryable | `docling-graph` returns 200 with empty stub when `run_pipeline()` raised internally; worker silently treated as success | `_call_extract_pass` raises `PassRetryable` when `diagnostics.pipeline_error` is set; clean zero-yield without that marker is logged but not retried (legitimate empty results possible for off-domain passes) | 3.2 |
| Merge and resolve | Duplicate entities across passes; broken relationships | Entities keyed by LogicalIdentity; bridge entities (PlatformEntity, SpecificationEntity) collapse across passes; relationships resolved by identity-dict or ref_id lookup; 6 rejection reasons tracked in merge metrics | 3.2 |
| Three-phase graph import (nodes → domain edges → structural edges) | Missing edges or broken provenance | Nodes imported with `tracker.mark()`, then domain edges, then structural edges (MENTIONED_IN from derive_rules). HAS_PROVENANCE auto-created by `upsert_nodes_batch_sync` | 3.2 |
| Tracker-gated rollback | Failures before first mutation trigger unnecessary graph deletion; failures after first mutation leave stale data | GraphWriteTracker gates `_delete_extraction_layer_graph` — failures before first mutation skip rollback; failures after first mutation trigger rollback | 3.2 |
| `/extract-pass` replaces `/extract-all` | Old endpoint returns 404; extraction fails | New wire contract: `{bundle_key, pass_name, docling_document_json, upstream_entities?}`. Verify Docling-Graph service responds to POST `/extract-pass` | 3.2 |
| Bundle selection threading | Wrong bundle used for extraction | `Source.default_ontology_bundle_key` persisted at source level; `PipelineRun.ontology_bundle_key` snapshotted at dispatch time; three-tier precedence: explicit override → source default → system default (`DEFAULT_ONTOLOGY_BUNDLE_KEY`) | 3.2 |
| Per-pass Celery task isolation | One bad pass blew the entire 8h ontology-graph stage | Each pass runs in its own Celery task with `pass_soft_time_limit=3600s`; per-pass `StageRun` + `pipeline_pass_outputs` row | 4.x |
| `dispatched_phases` state machine (claimed/dispatched/completed) | Dispatcher crash between DB claim and broker accept silently strands the run | Phase entries record state + timestamps + task_id; reconciler repairs stale `claimed` (>30s) and stale `dispatched` (>2× pass_soft_time_limit + no pass-output row) | 4.x |
| `derive_ontology_graph` summary StageRun lifecycle | finalize_document gate fails because the stage row is missing or stuck RUNNING | Dispatcher writes RUNNING; merge writes COMPLETE; required-pass terminalization writes FAILED. Same stage_name as before so finalize_document's REQUIRED_STAGES check is satisfied | 4.x |
| Authorized SKIPPED preserved through fan-in | system_links with no upstream endpoints recorded as FAILED — semantic regression | Pass task writes execution_status=SKIPPED with skip_reason; fan-in counts SKIPPED as terminal but excludes from merge inputs | 4.x |
| COMPLETE-without-output detection | Crash between StageRun COMPLETE write and pass-output save lets merge run with missing inputs | `_assert_stage_run_pass_output_consistency` cross-checks before merge; raises WorkerInvariantError on drift | 4.x |
| Targeted FK swallow on cancel | Broad IntegrityError swallow hides unique-constraint or JSON errors | save_pass_output only swallows the specific `pipeline_pass_outputs_pipeline_run_id_fkey` violation; all other integrity errors raise | 4.x |
| Per-document pass concurrency cap | 11 entity passes × 4 LLM connections = 44 concurrent calls per doc swamps Ollama | `pass_concurrency_per_document` (default 2) caps in-flight entity passes via shared claim/dispatch flow for initial + follow-up | 4.x |

### 2.2 Extraction Status & Monitoring

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Extraction status endpoint (`GET /documents/{id}/extraction-status`) | Cannot inspect per-pass results or graph state | Returns three-concept split: `document_status`, `latest_run` (with per-pass details including execution_status, yield_status, entity/rel counts), `graph_snapshot` (nullable), `graph_queryable` (cross-run rollback-aware) | 3.2 |
| `graph_queryable` cross-run computation (`compute_status_signals`) | False-positive "graph queryable" after rollback | Uses composite `(started_at, id)` row-constructor comparison to find `rollback_executed=True` summary rows from runs strictly newer than the snapshot's run. Handles: no-snapshot → False; snapshot + no rollback → True; snapshot + newer rollback → False; orphan snapshot (deleted PipelineRun) → conservative False | 3.2 |
| `is_stale` computation | Stale graph not flagged to users | True when `latest_run.id != snapshot.pipeline_run_id` OR `latest_run.status != COMPLETE`. Nested inside `graph_snapshot` only when snapshot exists; omitted when null | 3.2 |
| Coverage checker in CI (`tools/check_extraction_coverage.py`) | Schema drift silently breaks extraction | Runs 13 rules + manifest self-consistency; all entity/relationship types in manifest appear in coverage.yaml; schema fields subset of ontology properties | 3.2 |
| CI lints (`tools/ci_lints.sh`) | Legacy references to deleted code creep back | Prevents 8 classes of legacy references: template_builder, layered_extraction, ontology_layers, /extract-all, graph_layered_*, layer_map.yaml, ontology symlink, graph_extraction_engine flag | 3.2 |

### 2.3 Entity Resolution & Provenance (unchanged)

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Entity merge on `(entity_type, identity_fields)` NOT universal name | Cross-type name collisions merge unrelated entities | Doc with "Patriot" PLATFORM and "Patriot" MISSILE_SYSTEM creates two distinct vertices | 3.0 |
| Provenance via EXTRACTED_FROM edges (not vertex property) | Shared entities lose track of source documents | Entity mentioned in 3 docs has 3 EXTRACTED_FROM edges; deleting 1 doc leaves 2 | 3.0 |
| Relationship edges carry `document_ids` list | Relationships established by multiple docs lose provenance | Same relationship from 2 docs has `document_ids=[doc1, doc2]` | 3.0 |
| Entity alias resolution (exact → alias → fuzzy match → new) | "S-75" and "SA-2 Dvina" are separate entities; expansion incomplete | Ingest 2 docs with alternate names; query returns unified entity | 2.9 |
| Classification preserved on conflict | Reingest overwrites human-curated classification | Set classification to SECRET, reingest; verify still SECRET | 2.23 |
| Reserved word handling (TABLE → TABLE_REF) | ArcadeDB schema creation fails on reserved SQL keyword | TABLE ontology type maps to TABLE_REF vertex type; downstream code uses original name | 3.0 |
| ArcadeDB schema sync from active ontology | Schema drifts from ontology definition | API startup, registry activation, active registry PUT — schema sync runs with correct ontology | 3.0 |
| Schema sync is additive only | Schema sync removes types that still have data | Remove entity type from ontology, re-sync; type remains in ArcadeDB (data preserved) | 3.0 |

### Document-structure anchors (2026-04-21)

- [ ] New entity types: IMAGE, TEXT_BLOCK (in addition to existing DOCUMENT/SECTION/FIGURE/TABLE)
- [ ] New relationships: HAS_IMAGE (DOCUMENT→IMAGE, SECTION→IMAGE), NEAR_TEXT (FIGURE/IMAGE → TEXT_BLOCK)
- [ ] SECTION-level attribution: SECTION→FIGURE, SECTION→TABLE, SECTION→IMAGE HAS_* edges present
- [ ] DocumentEntity.storage_key populated from Document SQL row
- [ ] FigureEntity.storage_key + ImageEntity.storage_key are schema fields but emission is null until Artifact.self_ref migration lands (tracked separately)
- [ ] Re-ingestion (or one-shot migration) required for legacy documents whose uncaptioned pictures are currently FIGUREs — they re-classify to IMAGE on re-ingest

---

## 3. RETRIEVAL & SEARCH

> **ArcadeDB reference:** Consult `ArcadeDB Manual.pdf` sections 4.14 (Vector Search), 6.6 (Vector Functions), and 4.13.6 (Cross-Model Queries) when verifying retrieval features below.

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| BGE text search (strategy=basic) | Cannot retrieve by semantic content | Query with `strategy=basic`; receive ranked text chunks via ArcadeDB `vectorNeighbors` | 1 |
| CLIP image search (strategy=hybrid, modality_filter=image) | Cannot search images by text | Query image concept; receive image results via ArcadeDB ImageChunk vector search | 2 |
| Hybrid search (text + image merge, dedupe, rescore) | Cannot leverage both embeddings together | Query `strategy=hybrid`; receive both text and image results | 2 |
| Global query (community reports + LLM synthesis) | Cannot answer broad "what systems are in this corpus" questions | Query `strategy=global`; receive synthesized answer from community reports with source citations | 3.0 |
| Cross-model queries (vector + graph traversal in one SQL) | Multiple round-trips per retrieval; slower hybrid | ArcadeDB `vectorNeighbors() LET entity = chunk.in('EXTRACTED_FROM')` returns chunk + graph context | 3.0 |
| Weighted fusion scoring (0.65 semantic + 0.20 structure + 0.15 ontology) | Naive averaging ignores structural context | Verify formula applied in results scoring | 2.8, 2.24 |
| Document-structure expansion (chunk_links traversal) | Related chunks not retrieved together | Query matches chunk; neighbor chunks in same section also retrieved | 2.8 |
| Ontology expansion via graph traversal | Query "S-75" misses docs about "SA-2 Guideline" | Query "S-75"; receive results mentioning "SA-2" via relationships | 2 |
| Cross-encoder reranker (bge-reranker-v2-m3) | Top results less relevant without reranking | Enable reranker; top results re-ordered by cross-encoder | 2.24 |
| Content-level deduplication (oversample 8x, filter) | Duplicate text appears multiple times | Top-k results have no duplicate content | 2.20 |
| Min cosine similarity threshold (default 0.25) | Irrelevant noise in results | All returned results score >= threshold | 1, 2.24 |
| Military ID bonus (0.03 for AN/, NSN, MIL-STD matches) | Exact military system mentions not prioritized | Query with military ID; receives score bonus | 2.20 |

---

## 3b. ONTOLOGY REGISTRY & QUERY PROFILES

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Active registry ontology as shared backend loader | Backend ignores custom ontology; uses only repo YAML | Create registry with custom ontology, activate; `load_ontology()` returns registry ontology | 2.34 |
| Schema sync triggered on active registry PUT | ArcadeDB schema drifts after in-place ontology edits | Edit active registry ontology via PUT; schema sync runs automatically; new types appear in ArcadeDB | 3.0 |
| Ontology cache TTL (5s) + explicit invalidation | Stale ontology served after registry update | Update active registry ontology; within 1s new ontology is returned by `load_ontology()` | 2.34 |
| Scoring weights cache invalidation on ontology change | Retrieval scoring uses stale weights after ontology switch | Activate registry with different `scoring_weights`; next retrieval uses new weights | 2.34 |
| Docling-Graph receives ontology_definition per request | Extraction uses startup YAML instead of active registry | Activate registry; ingest doc; Docling-Graph logs show registry ontology version | 2.34 |
| Docling-Graph RuntimeExtractionContext caching | Template rebuilding on every request (performance) | Same ontology sent repeatedly; second call uses cached templates (LRU by hash) | 2.34 |
| Query profile registry CRUD (create, list, get, update, activate) | Cannot manage ontology registries | Create registry via UI; list shows it; activate toggles `is_active` | 2.34 |
| Per-profile CRUD (append, update, delete) | Cannot manage individual query profiles | Add section profile; update its traversals; delete it | 2.34 |
| Dossier profile references validated | Dossier references non-existent section; search fails | Create dossier with missing `section_profile_ids`; API returns 400 | 2.34 |
| Section profile deletion blocked if dossier references it | Deleting section breaks referencing dossier silently | Delete section referenced by dossier; API returns 400 | 2.34 |
| Section search (POST /v1/query-profiles/search/section) | Cannot execute single-profile graph traversal | Search with section profile_id + entity name; receive traversal results via GraphStore | 2.34 |
| Dossier search (POST /v1/query-profiles/search/dossier) | Cannot execute multi-section compound query | Search with dossier profile_id; receive sections with items | 2.34 |
| Root entity resolution (alias + fulltext + tie-break) | Search for entity alias returns 404 | Search "SA-2" when canonical name is "S-75 Dvina"; entity resolves via `resolve_root_entity()` | 2.34 |
| Query profile modes in Search page dropdown | Custom query profiles not accessible from search UI | Activate registry with exposed profiles; Search page dropdown shows them | 2.34 |
| Profiles tab disabled until active registry exists | User creates profiles without ontology context | No active registry; Profiles tab greyed out or empty state shown | 2.34 |
| Starter profiles seeded from repository ontology | User must build profiles from scratch | Click seed button; pre-built section/dossier profiles populate | 2.34 |

---

## 4. DOCLING DOCUMENT VIEWER

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| DoclingViewer with `<docling-img>` web component | Documents unreadable; raw JSON only | Open PDF in viewer; see rendered pages with bounding boxes | 2.11 |
| Docling-raw endpoint enriched with description annotations | Images in viewer have no hover context | `/v1/documents/{id}/docling-raw` returns pictures with `annotations: [{kind: "description", ...}]` | 2.30 |
| Standalone image 404 guard in docling-raw | Standalone images render empty Docling page instead of fallback | Standalone image docling-raw returns 404; viewer shows image + description panel | 2.30 |
| Translation toggle with language banner | Non-English documents unreadable in viewer | Non-English doc has "Translate" toggle; switches between original and translated | 2.27 |
| Per-element translations overlay | No inline translation context | Hover over non-English text; see English translation | 2.27 |
| Image proxy (GET /v1/images/{chunk_id}) | Image results broken due to Docker-internal URLs | Image results display correctly; no CORS errors | 2.11 |

---

## 5. COMMUNITY DETECTION & GLOBAL QUERY

> **ArcadeDB reference:** Consult `ArcadeDB Manual.pdf` Appendix 8.1 (Graph Algorithms — Louvain, Leiden) and section 6.5 (Graph Algorithms function reference) when verifying community detection features below.

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Scheduled community detection (Celery Beat, configurable interval) | Community reports stale; global query returns outdated synthesis | Beat schedule runs `run_community_detection_task` at `COMMUNITY_DETECTION_INTERVAL_MINUTES` | 3.0 |
| Manual trigger (POST /v1/community/detect with mode=incremental|full) | Cannot force recomputation after ingest | Trigger detection; task starts; status endpoint returns run_id | 3.0 |
| Post-ingest hook (counter threshold) | Community reports lag behind ingest | `COMMUNITY_DETECTION_POST_INGEST_ENABLED=true`; after N documents ingested, detection triggers | 3.0 |
| Redis lock prevents overlapping runs | Concurrent community detection corrupts reports | Trigger during active run; second trigger returns "skipped" | 3.0 |
| Domain-entity projection (excludes structural types) | Community clusters dominated by Document/TextChunk structural nodes | Louvain runs only on ontology entity types; Document, TextChunk, ImageChunk, Alias, CommunityReport excluded | 3.0 |
| Incremental reports (membership hash diff) | Regenerates all reports on every run (expensive) | Second run with unchanged graph regenerates 0 reports; `reports_reused` counter populated | 3.0 |
| Membership hash uses `(entity_type, name)` tuples | Hash collisions across types treat changed communities as unchanged | Hash sort key includes type; cross-type entities with same name don't collide | 3.0 |
| LLM report generation with configurable model | Cannot swap between Ollama models | `COMMUNITY_REPORT_LLM_MODEL=llama3.2` (or any) controls which model generates reports | 3.0 |
| Configurable prompt via `COMMUNITY_REPORT_LLM_PROMPT` | Cannot customize domain-specific report prompts | Env var override replaces default military domain prompt | 3.0 |
| CommunityReport vertex with report_embedding | Global query vector search has nothing to match | ArcadeDB `SELECT FROM CommunityReport WHERE report_embedding IS NOT NULL` returns rows after detection | 3.0 |
| Global query strategy returns synthesized answer | Cannot answer corpus-wide questions | Query `strategy=global`; receives text response with community sources cited | 3.0 |
| community_runs table tracks run history | Cannot audit community detection history | PostgreSQL `retrieval.community_runs` has rows for each run with status, counts, timings | 3.0 |

---

## 6. TRUSTED DATA & GOVERNANCE

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Trusted data proposal workflow (PROPOSED -> APPROVED -> INDEXED) | Untrusted data indexed without oversight | Submit via `/v1/trusted-data/ingest`; status is PROPOSED until curator approves | 2.6 |
| Trusted data search via ArcadeDB TrustedTextChunk vertices | Cannot distinguish human-reviewed from extracted knowledge | `/v1/trusted-data/query` returns only approved data via `vector_search("TrustedTextChunk", ...)` | 3.0 |
| Feedback endpoint (auto-creates patch) | No mechanism to report extraction errors | `/v1/feedback` with correction creates patch | 2 |
| Governance re-embed path uses GraphStore | Chunk embeddings updated via legacy Qdrant/pgvector writes | Patch apply flow calls `graph_store.set_vertex_embedding()` instead of `chunk.embedding = ...` | 3.0 |
| Dual-approval for graph mutations | Single curator can corrupt graph | Graph patch with one approval rejected | Planned |

---

## 7. INFRASTRUCTURE & RESILIENCE

> **ArcadeDB reference:** Consult `ArcadeDB Manual.pdf` sections 5.5.20 (Docker), 4.15 (High Availability), 6.9 (Settings), and 4.6 (Transactions) when verifying infrastructure features below.

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| ArcadeDB service (built from source via manage.sh) | Graph + vector database unavailable | `docker compose ps` shows arcadedb healthy; `GET http://localhost:2480/api/v1/ready` returns 204 | 3.0 |
| manage.sh `ensure_all_repos()` pulls ArcadeDB repo for source build | ArcadeDB container uses stale code | `./manage.sh --start` logs "Pulling latest ArcadeDB..." before `dc build`. **Note:** Docling and Docling-Graph install from PyPI packages (not from cloned repos) — their versions update via `pip install` in their Dockerfiles, not `git pull`. | 3.0 |
| ArcadeDB token-based auth with re-auth on 401 | Long-running workers lose auth mid-task | Client retries once on 401 with fresh login; no manual token refresh | 3.0 |
| ArcadeDB schema sync at API startup | Ingest writes fail due to missing vertex/edge types | API startup logs "ArcadeDB schema synced: N types, M properties, K indexes" | 3.0 |
| Worker `ensure_ready_sync()` before first graph write | Race condition when worker starts before API schema sync | Graph-writing tasks retry with backoff if vertex types don't exist | 3.0 |
| LSM_VECTOR indexes with HNSW (hierarchical) | Slow vector search at scale (100K+ vectors) | TextChunk and CommunityReport indexes have `addHierarchy: true` | 3.0 |
| INT8 quantization on vector indexes | 4x memory usage for vectors | LSM_VECTOR metadata includes `quantization: 'INT8'` | 3.0 |
| Configurable retries per pipeline stage | Hard-coded retries impossible to tune | Set `PREPARE_MAX_RETRIES=3`; doc fails after 3 attempts | 2.18 |
| Configurable task time limits (soft + hard) | Fixed limits fail for varying doc sizes | Set `PREPARE_SOFT_TIME_LIMIT=7200`; large PDFs complete | 2.18 |
| Docling 503 retry loop (no budget consumed) | Busy errors consume retry budget | 503 triggers in-task retry with backoff; doesn't decrement max retries | 2.14 |
| Redis semaphore for Docling concurrency | GPU OOM or CPU saturation from unbounded conversions | `DOCLING_CONCURRENCY=1`; uploads queue instead of timeout | 2.11 |
| Docling lock held through retry (no gap) | Another task steals lock during retry window | Semaphore held during `self.retry()` | 2.23 |
| Docling health probe (advisory, non-blocking) | Health check failure blocks conversion | Docling unresponsive; probe logs warning but conversion proceeds | 2.11 |
| Atomic PipelineRun check-and-set (FOR UPDATE) | Concurrent ingest creates multiple runs | Submit same doc twice simultaneously; only one run created | 2.23 |
| Document-scoped singleflight Redis lock | prepare_document executes twice concurrently | Lock acquired at start, released at end; no concurrent prepares | 2.23 |
| Celery visibility timeout (10800s) | Long tasks redelivered prematurely; duplicate execution | 3-hour task completes without redelivery | 2.23 |
| Worker split profile (--profile split) | Single worker pool bottleneck | `docker compose --profile split up`; separate containers for ingest/embed/graph | 2.16 |
| Worker topology isolation (manage.sh) | Single and split workers both running; conflicts | manage.sh auto-stops opposite profile | 2.23 |

---

## 8. WEB UI

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Query mode dropdown (basic, hybrid, global + dynamic query profiles) | User stuck with default strategy | Query page dropdown shows all modes in grouped optgroups; each runs correct query | 3.0 |
| Global query detail view | Synthesized answer with source citations not visible | Select `global` mode, submit query; GlobalQueryDetail shows text + expandable community sources | 3.0 |
| Modality sub-filter (text, image, all) for hybrid | Cannot isolate result types | Filter visible when hybrid selected; affects results | 2.11 |
| Live status polling with terminal state detection | UI polls forever; wastes resources | Polling stops at COMPLETE/FAILED/PARTIAL_COMPLETE/ERROR | 2.11 |
| Terminal status badges (green/red/amber) | Cannot distinguish success from failure | Color-coded badges for all terminal states | 2.11 |
| Directory Monitor (register/remove watch dirs) | Cannot set up auto-ingest | Directory Monitor page lists active dirs; add/remove works | 2.5 |
| Graph Explorer (entity/relationship search + creation) | Cannot explore or curate knowledge graph | Search and creation forms with full ontology | 2.5, 2.11 |
| Graph Explorer subgraph view (neighborhood visualization) | Clicking graph circle shows only 1 node for orphan entities | Search entity; click graph circle; see multi-node subgraph | 2.31 |
| Trusted Data panel (submit/approve/reject/search) | Cannot interact with trusted data layer | Submission form, approval queue, search interface | 2.6 |

---

## 9. HEALTH & CONFIGURATION

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Liveness probe (GET /v1/health) | Crashed API not detected | Returns `{"status": "ok"}` | 1 |
| Readiness probe with dependency checks | API starts before dependencies; early requests fail | `/v1/health/ready` returns `"degraded"` when deps down, `"ready"` when up | 1 |
| Community detection API endpoints | Cannot trigger or monitor community runs | `/v1/community/detect`, `/v1/community/status`, `/v1/community/reports` all respond | 3.0 |

---

## 10. OLLAMA CONFIGURATION

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Per-role Ollama URLs (LLM/VLM/Embedding) | Cannot point different model types at different Ollama instances | Set `OLLAMA_LLM_BASE_URL` to different host; doc analysis uses that host; embedding uses `OLLAMA_EMBEDDING_BASE_URL` | 3.1 |
| URL fallback to OLLAMA_BASE_URL | Specialized URL left blank breaks all LLM calls | Leave `OLLAMA_LLM_BASE_URL` blank; `get_ollama_llm_url()` returns `OLLAMA_BASE_URL` | 3.1 |
| Docling-Graph inherits LLM URL via docker-compose cascade | Docling-Graph uses wrong Ollama instance | `OLLAMA_LLM_BASE_URL` set; docker-compose passes it to docling-graph container as `OLLAMA_BASE_URL` | 3.1 |
| All LLM consumers use correct getter | Some consumers hardcode `ollama_base_url` instead of role-specific getter | `embedding.py` uses `get_ollama_embedding_url()`, `document_analysis.py:58` uses `get_ollama_llm_url()`, `document_analysis.py:202` uses `get_ollama_vlm_url()` | 3.1 |

---

## 11. NOTEBOOKS

Both notebooks use **dynamic introspection** — they iterate `load_ontology()` / `ontology_bundles.air_defense_v3.entities.ALL_ENTITIES` / `RelationshipType` enum members rather than hardcoded type lists. No cell modifications are required when entity types or relationship types are added to the ontology.

| Notebook | Status | Notes |
|---|---|---|
| `notebooks/ingest_walkthrough.ipynb` | Dynamic — no hardcoded type lists | Reload from disk in JupyterLab + restart kernel after pulling new entity/rel types; stale executed outputs are cleared on cells that changed |
| `notebooks/raw_libraries_walkthrough.ipynb` | Dynamic — no hardcoded type lists; no inline walker replica | Re-run to pick up new walker output; §10 documents what's out-of-scope vs. the full pipeline |

### Task 6 additions (2026-04-21)

- [x] `ingest_walkthrough.ipynb` — stale outputs from cell `edadcd2a` (ontology summary) cleared; outputs showed 44 entity types / 50 relationship types from a pre-IMAGE/TEXT_BLOCK run.
- [x] `ingest_walkthrough.ipynb` — §8 "Anchor walker" markdown cell + `derive_document_anchors.apply()` cell added at the end; shows SECTION / FIGURE / TABLE / IMAGE / TEXT_BLOCK counts and HAS_IMAGE / NEAR_TEXT edge counts for the current document.
- [x] `raw_libraries_walkthrough.ipynb` — no code changes required (dynamic introspection, no inline walker replica).

### After ontology schema changes

1. Reload both notebooks from disk in JupyterLab (`File → Reload Notebook from Disk`).
2. Restart the kernel.
3. Run all cells top-to-bottom; the ontology summary cell in `ingest_walkthrough` will reflect the current counts automatically.

---

## KNOWN FRAGILE FEATURES (Historically Broken)

These features have broken before and should be tested carefully after any change:

1. **Image dedup** (2.30) — Distinct images on same page collapsed. Test with multiple images, empty captions, same page.
2. **Image description tooltips** (2.25, 2.30) — Annotations must inject into docling-raw JSON. Hover over image in viewer.
3. **Standalone image synthesis** (2.30) — Must not trigger for non-image files. Test with .txt, .pdf, .docx.
4. **Docling timeouts** (2.14, 2.21) — Large PDFs timeout. Test with >100 page PDFs.
5. **Docling 503 handling** (2.14) — Must not consume retry budget. Test with mock 503s.
6. **Chord resilience** (2.22) — Single task failure must not kill pipeline. Verify error dicts returned.
7. **Concurrent dispatch** (2.23) — Multiple uploads of same doc must not create duplicate runs.
8. **Stale run cleanup** (2.18, 2.23) — Crashed workers must reset PROCESSING docs. Test worker crash.
9. **Image description text search** (2.26) — Descriptions must appear in text search results.
10. **Chunk_links traversal** (2.8, 2.26) — Structure expansion must include all link types.
11. **Translation tooltips** (2.27) — Must not break image description annotations.
12. **Ontology subgraph orphans** (2.31) — Specification entities must connect to parent systems via SPECIFIED_BY. Test with "missile" search; click graph circle; verify multi-node subgraph.
13. **Ontology cache invalidation chain** (2.34) — `invalidate_ontology_cache()` must clear both the ontology TTL cache and the scoring weights LRU cache. Test by switching active registry and verifying new weights take effect.
14. **Query profile dossier search** (2.34) — Dossier search compiles multi-section traversals via GraphStore. Test with profiles that have multiple traversals with different hop ranges.
15. **ArcadeDB schema sync on PUT** (3.0) — In-place ontology edits on active registry must trigger schema sync. Test by adding a new entity type via PUT and verifying it appears in ArcadeDB schema.
16. **Entity merge key correctness** (3.0) — Dedup uses `(entity_type, identity_fields)`, NOT universal name. Test with cross-type name collisions (e.g., "Patriot" PLATFORM vs MISSILE_SYSTEM).
17. **Provenance via EXTRACTED_FROM** (3.0) — Entity vertices shared across documents do NOT carry `source_document_id`. Provenance flows through EXTRACTED_FROM edges. Test document delete: shared entity survives with remaining edges; orphaned entities (zero edges) are cleaned up.
18. **Worker ensure_ready_sync race** (3.0) — Worker can start before API schema sync. Graph-writing tasks must retry until types exist. Test by starting worker before API and verifying retry/backoff behavior.
19. **Community detection projection** (3.0) — Louvain must run only on domain entity types, excluding structural types (Document, TextChunk, ImageChunk, Alias, CommunityReport). Verify communities are not dominated by chunk clusters.
20. **Membership hash stability** (3.0) — Hash must use `(entity_type, name)` tuples to prevent cross-type collisions and remain stable through canonical_name changes.
21. **Per-role Ollama URL fallback** (3.1) — Blank specialized URL must fall back to `OLLAMA_BASE_URL`. Test by setting only `OLLAMA_BASE_URL` and verifying all LLM/VLM/embedding calls work.
22. **ArcadeDB UPSERT RETURN AFTER** (3.1) — All UPSERT statements must use `RETURN AFTER @rid` to get the RID back. Without this, upserts return `{count: 1}` instead of the RID, breaking downstream operations.
23. **ArcadeDB CONTAINSTEXT** (3.1) — Fulltext search must use `CONTAINSTEXT` keyword, NOT `LUCENE`. Per-type iteration required since ArcadeDB has no abstract `V` base type.
24. **upsert_relationship param collision** (3.1) — from_identity and to_identity params must use `f_`/`t_` prefixes to avoid key collision when both contain `name`.
25. **Bundle coverage checker rule 8** (3.2) — extraction schema field names must be a subset of ontology properties (SYSTEM_FIELDS exempt). Drift between hand-authored schemas and ontology.yaml silently produces empty extraction results.
26. **Bridge entity consistency across passes** (3.2) — PlatformEntity and SpecificationEntity must be structurally identical in radar_domain, missile_domain, and other_systems (checker rule 13).
27. **Graph import tracker ordering** (3.2) — `tracker.mark()` must be called AFTER pure-Python record construction and BEFORE the first graph_store mutation. A misplaced mark causes either missed rollback (mark too late) or unnecessary rollback (mark too early).
28. **`document_ids` list on domain edges** (3.2) — domain relationship edges carry `document_ids` as a LIST, not `document_id` as a string. The rollback primitive uses "remove from list + prune empty" — not simple `WHERE document_id = :id`.
29. **ArcadeDB `@in` vs `in`** (3.2) — HAS_PROVENANCE edge target traversal requires `@in` (not plain `in`) in WHERE clauses. Plain `in` silently returns 0 rows.
30. **Partial unique index upsert** (3.2) — `_write_stage_run` uses `ON CONFLICT` with `index_elements` + `index_where` for the partial unique index. SQLAlchemy versions < 2.0 may not support `index_where` on `on_conflict_do_update`.

---

## TESTING PROTOCOL

### After Every Code Change
1. Run full unit test suite: `uv run pytest tests/unit/ --tb=short`
2. Review this checklist for sections relevant to changed files
3. Add new features/fixes to this checklist before committing

### Quick Smoke Test (5 min)
1. Upload PDF document → verify COMPLETE status
2. Query text search (strategy=basic) → verify results with preview
3. Query hybrid search → verify cross-model results
4. Check `/v1/health/ready` → all systems ready

### Integration Test (30 min)
1. Upload multi-page PDF with images and tables
2. Verify text search, image search, hybrid search
3. Trigger community detection; verify global query returns synthesized answer
4. Verify DoclingViewer image hover tooltips
5. Verify standalone image upload + viewer
6. Verify document delete removes chunk vertices and cleans orphan entities

### Regression Test (60 min)
Run against all 30 Known Fragile Features listed above.

---

## CRITICAL CONFIGURATION PARAMETERS

| Config | Default | What breaks if wrong |
|---|---|---|
| `CHUNK_MAX_TOKENS` | 512 | Embedding quality degrades |
| `EMBED_TEXT_BATCH_SIZE` | 128 | Large docs timeout |
| `DOCLING_CONCURRENCY` | 1 | GPU OOM from parallel conversions |
| `DOCLING_TIMEOUT_SECONDS` | 3600 | Large PDFs timeout |
| `CELERY_VISIBILITY_TIMEOUT` | 10800 | Long tasks redelivered |
| `RETRIEVAL_SEMANTIC_WEIGHT` | 0.65 | Fusion scoring breaks |
| `RERANKER_ENABLED` | true | Ranking quality drops |
| `GRAPH_NODE_MIN_CONFIDENCE` | 0.60 | Low-confidence entities pollute graph |
| `ARCADEDB_URL` | `http://arcadedb:2480` | Cannot reach graph database |
| `ARCADEDB_DATABASE` | `eip_knowledge_graph` | Wrong database targeted |
| `COMMUNITY_DETECTION_ENABLED` | true | Global query returns empty |
| `COMMUNITY_DETECTION_ALGORITHM` | `leiden` | Wrong algorithm may fail on small graphs |
| `COMMUNITY_DETECTION_INTERVAL_MINUTES` | 60 | Too frequent = LLM cost; too rare = stale reports |
| `COMMUNITY_DETECTION_POST_INGEST_THRESHOLD` | 5 | Hook triggers too often or too rarely |
| `DEFAULT_ONTOLOGY_BUNDLE_KEY` | `air_defense_v3` | Bundle resolution falls back to system default; wrong value = unknown bundle error |
| `PASS_MAX_RETRIES` | `3` | Per-pass retry budget; exhausted = IngestFailed for required passes |
| `STRUCTURED_OUTPUT_THRESHOLD_CHARS` | `8000` | Schema size ceiling for structured LLM output; exceeded = fallback to JSON mode |
