# TODO — Remaining Work

**Last updated:** 2026-04-18

---

## Open Items

### Infrastructure / Worker Queue Isolation (Deferred)

**#28. Isolate `scan_watch_directories` from pipeline worker queue**
**Status:** Deferred. Post-migration engineering work, separate PR.
**Files:** `app/workers/celery_app.py`, `docker-compose.yml`, `scripts/full_purge_and_reingest.py`

**Observed failure mode (2026-04-17 corpus re-ingest):**
During the docs-alignment migration run, the beat-driven
`scan_watch_directories` poller (every 30s → 120 tasks/hour) piled up
77+ unprocessed tasks in the `ingest` queue while pipeline workers
were blocked for hours on long-running `derive_ontology_graph` tasks
(LLM passes at 20–60 min each on llama3.3:70b thinking=high). With
`worker_prefetch_multiplier=1` and FIFO queue consumption, chain
continuations (`collect_derivations`, `derive_structure_links`,
`derive_canonicalization`, `finalize_document`) sat behind the
watcher backlog and appeared — incorrectly — as a chain break. The
chain WAS firing; continuations were just deep in the queue.

**What needs to be done:**
1. Add a dedicated `watcher` queue. Route
   `app.workers.watcher.scan_watch_directories` to it in
   `celery_app.py::task_routes`.
2. Update `docker-compose.yml` worker subscriptions so the primary
   pipeline worker does NOT consume from `watcher`. Either start a
   separate, low-priority watcher worker OR accept that watch-dir
   polling pauses during heavy ingest runs.
3. Update `scripts/full_purge_and_reingest.py` so the migration
   driver `docker compose stop beat` after step 2 (containers stop)
   and restarts beat only at the end of step 8 — during the ingest
   run, no new scan_watch_directories tasks should be generated at
   all.
4. Optional: add a ceiling on beat-generated tasks (drop-if-queue-
   depth-already-contains-same-task pattern) so intermittent worker
   restarts can't accumulate unbounded backlog.

**Why this matters:**
Without isolation, any long-running pipeline task starves every
downstream chain continuation by queue position, even when the
chain callback fires correctly. The symptom looks identical to a
chain break, which sent debugging off the rails for an hour during
the 2026-04-17 run. Proper queue isolation eliminates this failure
mode entirely.

**Acceptance:**
- `scan_watch_directories` tasks do not appear in the `ingest` queue.
- Migration script stops beat for the duration of a re-ingest.
- A load test with both stages running doesn't starve pipeline
  continuations regardless of queue depth.

---

### Feature Additions (Deferred)

**#27. LLM-based entity mention resolution**
**Status:** Deferred. Now unblocked but intentionally deferred due to high cost/latency tradeoff.
**Files:** `app/workers/pipeline.py` (`_build_entity_mentions`), new module TBD

**Current state:**
- `_build_entity_mentions` uses regex/substring matching to link extracted entities to document chunks.
- This catches exact text matches but misses: paraphrases ("the fire control radar" -> APG-77), abbreviations ("FCR" -> APG-77), misspellings, coreferences ("the system" -> S-400).

**What needs to be done:**
1. Add an LLM-based mention resolution mode, configurable via `ENTITY_MENTION_RESOLUTION_MODE` env var (values: `regex`, `llm`, default `regex`).
2. When `llm` mode is active, send each chunk text along with the entity list to an LLM to determine which entities are mentioned (even implicitly).
3. Batch chunks to reduce LLM call count (e.g., 10 chunks per call with boundary markers).
4. Fall back to regex mode on LLM failure.
5. Benchmark accuracy improvement vs. latency/cost tradeoff.

**Tradeoffs:**
| | Regex | LLM |
|---|---|---|
| Exact mentions | Yes | Yes |
| Paraphrases | No | Yes |
| Abbreviations/coreferences | No | Yes |
| Speed | ~50ms per doc | Minutes (thousands of LLM calls) |
| Cost | Free | Significant |
| Determinism | 100% | Non-deterministic |

**Acceptance:**
- Configurable via env var (regex remains the default)
- LLM mode produces more complete EXTRACTED_FROM edges
- Latency is bounded via batching

---

**#29. Route image-heavy / handwritten docs through VLM backend + one-to-one**
**Status:** Deferred. Future optimization; not a fix for the current migration. Revisit after the `llama3.1:8b` + `batch=1024` + `gleaning=2` LLM baseline stabilizes.
**Files:** `app/workers/pipeline.py` (`_derive_ontology_graph_bundle_passes`), `docker/docling-graph/app/main.py` (per-request backend dispatch), `app/models/ingest.py` (optional `backend_hint` column)

**Observation (2026-04-18):**
The docs-sanctioned pattern for image-heavy / handwritten / form-style documents is `backend="vlm"` + `processing_mode="one-to-one"` + `numind/NuExtract-2.0-8B` — see `docling-graph-docs.md` §Extraction Backends ("VLM Backend Criteria") and §The Extraction Process ("Choose the Right Backend"). Four docs in the current corpus fit that profile:
- `cw_radar.jpg`
- `Fan_Song_Radar.jpeg`
- `chinese_handwritten_notes.pdf`
- `chinese_handwritten_notes_2.pdf`

On the current LLM-only path these docs either run the ontology pass on very sparse markdown (low recall) or fail the quality gate entirely (empty LLM output).

**What needs to be done:**
1. Add a per-document `backend_hint` at ingest (heuristic: `mime_type in image/*` OR `page_count==1 and ocr_text_ratio<X` → `"vlm"`, else `"llm"`).
2. Extend `/extract-pass` (or add `/extract-pass-vlm`) so the worker can request the VLM backend per call without changing the global service config.
3. Pull `numind/NuExtract-2.0-8B` on the Ollama host (or run via vLLM/local GPU runtime supported by docling-graph's `VlmBackend`).
4. Make the worker send the raw source bytes (not DoclingDocument JSON) for VLM passes, since VLM reads images directly.
5. Add `backend.cleanup()` call in the docling-graph service handler so VLM GPU memory is freed between requests (per docs best-practices note).

**Why this matters:**
Keeps prose extraction on the fast LLM+delta path while giving the image/handwritten minority of the corpus the docs-aligned VLM treatment. Without this, these docs consistently produce sparse or empty ontology output regardless of how well the LLM side is tuned.

**Acceptance:**
- `cw_radar.jpg` and `Fan_Song_Radar.jpeg` produce non-empty ontology entities end-to-end.
- The two `chinese_handwritten_notes*.pdf` docs either extract real entities or are explicitly marked "no-extract" (e.g. by a hint that says "image-only, no target domain content").
- No regression on the LLM/delta path for prose PDFs.

---

## Completed Items (Reference)

### Gaps/Bugs Fixed

- **#35.** Fixed upstream repo clone/update contract for Docling and Docling-Graph (verification checklist/design spec updated to reflect actual PyPI package contract)
- **#36.** Fixed vector_search() dropping chunk metadata (`chunk_id`, `document_id`, `artifact_id`, `modality`, `text`) that retrieval depends on
- **#37.** Fixed ontology/evidence traversal edge direction and identifier mismatch (superseded by #49)
- **#38.** Fixed alias resolution property name inconsistency (standardized on `alias_name`; `entity_type` filter moved to linked entity)
- **#39.** Aligned docling-graph wrapper with canonical template/pipeline API (`is_entity=True`/`edge()` pattern)
- **#40.** Wired retrieval `filters` (classification, document_id, modality constraints) that were silently ignored
- **#41.** Fixed document canonicalization scope (superseded by #50)
- **#42.** Fixed orphan cleanup system field (`@cat` changed to `@class`)
- **#43.** Added `$distance` projection to community-report vector search (scores were collapsing to `0.0`)
- **#44.** Fixed stale test harness for docling-graph integration tests (patched `app.state.templates` instead of deleted global)
- **#45.** Reconciled Stage 1 worker/service request+response contract (worker/service request and response shape aligned)
- **#46.** Made derive_ontology_graph() consume persisted DoclingDocument JSON from MinIO instead of reconstructed plain text
- **#47.** Normalized extraction output shape before graph import (adapter maps Docling-Graph output to `{nodes, edges}` format)
- **#48.** Fixed entity-to-chunk mention wiring to be element-complete (image chunks included; partial fallback for zero-mention entities)
- **#49.** Fixed EXTRACTED_FROM traversal direction and identifier type across all callers (UUID-to-RID lookup added)
- **#50.** Fixed canonicalization to use graph traversal for document-entity discovery (replaced LUCENE search with Document->chunk->entity edges)
- **#5.** Implemented validation_matrix enforcement in upsert_relationship (reject/warn on invalid triples)
- **#51.** Fixed query profiles to honor their own directed traversal model (native MATCH patterns from each step's direction/hops)
- **#52.** Fixed dossier/query-profile evidence attachment (separate entity-centric vs chunk-centric helpers)
- **#53.** Fixed hybrid cross-modal fallback UUID/RID mismatch (UUID-to-RID resolution added)
- **#54.** Fixed ontology expansion to preserve graph context from returned chunks (enriched traversal results)
- **#55.** Implemented retrieval filter push-down into native queries (document_id as WHERE clause on outer SELECT)
- **#56.** Aligned API defaults with advertised retrieval settings (top_k=20, min_confidence=0.1)
- **#57.** Fixed graph fulltext search to carry Lucene $score through result model (score priority: $distance > $score > extraction_confidence)
- **#58.** Wired co-extracted entity discovery into the query stack (fallback in entity resolution)
- **#59.** Updated stale integration tests to assert current response shape (strategy/modality_filter)

### Features Implemented

- **#1.** Implemented real LLM community report generation (`_call_llm_for_report` with Ollama, JSON parsing, thinking-model handling)
- **#2.** Implemented community report embedding (`_embed_report` wraps `embed_texts` via `asyncio.to_thread`)
- **#3.** Wired LLM synthesis into global query response (single synthesized answer with raw reports in context)
- **#4.** Added batch HTTP operations to replace N+1 patterns (sqlscript batches for nodes, edges, chunks with embeddings)
- **#6.** Implemented post-ingest hook counter + threshold trigger for community detection (Redis counter with configurable threshold)
- **#7.** Added batch_create_entity_chunk_edges to GraphStore Protocol (single sqlscript call replaces per-edge loop)
- **#21.** Added VLM extraction backend option to docling-graph service (`DOCLING_GRAPH_BACKEND` env var with `llm`/`vlm` options)
- **#22.** Implemented dual-approval workflow for graph mutations (`approvals` list, duplicate-curator prevention, config flag)
- **#25.** Added benchmark suite for retrieval strategies (`tests/benchmarks/` with latency assertions)
- **#28.** Native ArcadeDB vector functions for cross-model queries (efSearch parameter, graph_vector_search with vectorCosineSimilarity, Python-side fusion scoring)
- **#29.** MATCH syntax for graph traversal queries (get_neighborhood and get_neighborhood_graph rewritten to MATCH pattern syntax)
- **#72.** Restored Global Search indexing UI on the Ontology page (GlobalSearchIndexingPanel with schedule display, manual trigger, status, reports browser)

### Code Quality / Refactoring

- **#8.** Optimized schema sync to batch DDL statements (~200 calls reduced to 7 phase-scoped sqlscript batches)
- **#9.** Implemented time-based ensure_ready caching (configurable TTL via `ARCADEDB_READY_CACHE_SECONDS`)
- **#10.** Extracted shared SQL building to reduce async/sync duplication (`_build_*_sql()` helpers return `(sql, params)` tuples)
- **#11.** Consolidated Redis client factory (singleton `get_redis()` in `redis_utils.py` with `close_redis()` shutdown hook)
- **#12.** Standardized Redis locking idiom (`r.lock()` pattern; `redis_lock()` helper; bare `SET NX` patterns replaced)
- **#13.** Parameterized vector_search instead of duplicating for image (unified method; `image_vector_search` removed)
- **#14.** Unified RESERVED_WORD_MAP definitions (shared `ontology/arcadedb_reserved_words.json` contract file)
- **#15.** Replaced manual env var helpers with pydantic_settings in docling-graph (`DoclingGraphSettings(BaseSettings)`)
- **#16.** Moved module-level globals to FastAPI `app.state` in docling-graph service
- **#17.** Derived `_STRUCTURAL_TYPES` from `arcadedb_schema.STRUCTURAL_TYPES` export (single source of truth)
- **#18.** Scoped resolve_root_entity queries to specific types (optional `entity_type` parameter with `V` fallback)
- **#19.** Pre-compiled regex patterns in `_build_entity_mentions` (one compilation per entity instead of per element)
- **#20.** Collapsed pipeline stage section comments (comments now explain WHY, not WHAT)
- **#23.** Added ArcadeDB connection pooling (singleton GraphStore; httpx `Limits` tuned)
- **#24.** Added observability for ArcadeDB operations (latency histograms, error counters, slow-query logging, `/metrics` endpoint)
- **#26.** Migrated docling-graph `templates.py` to use `template_builder.py` logic (removed duplication; single source of truth)
- **#30.** Added BucketSelectionStrategy `'thread'` for write-heavy types (TextChunk, ImageChunk, high-write entity types)
- **#31.** Enabled ArcadeDB Prometheus metrics plugin (`PrometheusMetricsPlugin` in docker-compose JAVA_OPTS)
- **#32.** Configured automatic backup scheduler (`backup.json` with cron schedule and retention)
- **#33.** Used `text.levenshteinDistance()` for fuzzy entity matching (server-side fuzzy matching in canonicalization)
- **#34.** Added EXPLAIN/PROFILE tooling for query plan validation (health-check asserts no full scans on critical paths)

### Architecture (Native-First) Implemented

- **#60.** Made DoclingDocument the authoritative mutable artifact through the pipeline
- **#61.** Replaced custom chunker with native Docling chunkers
- **#62.** Used Docling's native enrichment path for translation and picture descriptions
- **#66.** Compiled query-profile traversals into native MATCH instead of generic undirected walk
- **#67.** Enriched GraphStore result model to carry native ArcadeDB semantics
- **#68.** Pushed retrieval filters into native queries
- **#69.** Returned enriched graph context from ontology expansion traversal
- **#71.** Decided: hybrid approach for Docling-Graph templates (auto-generate from YAML, conform to canonical patterns). Decision feeds into #63.

### Completed During Migration (Pre-merge Fixes)

- Fixed `set_vertex_embedding` signature mismatch between Protocol and implementation
- Fixed Protocol encapsulation violations (direct `_client`/`_database` access) -- added proper methods
- Fixed `upsert_relationships_batch_sync` dropped `record.properties` (copy-paste bug)
- Fixed Redis client leak in `community_tasks.py` (no `r.close()`)
- Fixed direct `os.environ.get` bypassing settings in community module
- Parallelized `cross_model_search` and `get_graph_stats` sequential queries with `asyncio.gather`
- Removed stale references (GraphRAG, Neo4j, Qdrant, ChunkRef, AGE) across 8 files
- Removed narrating step-number comments in `delete_document_graph` and `sync_schema_from_ontology`
- Fixed `delete_document_graph` to remove document_id from relationship edge document_ids lists and delete empty-list edges

---

## Verbatim Reviews (Reference)

### Verbatim Native-First Review (2026-04-06)

> No code changes made. The branch does call native `DocumentConverter`, Docling-Graph `run_pipeline()`, and ArcadeDB `MATCH`/`vectorNeighbors()`, but the dominant pattern is to immediately flatten or wrap those native objects into app-specific contracts and then reimplement upstream behavior in Python.
>
> **Findings**
>
> 1. Critical: `DoclingDocument` is not kept as the authoritative working object after conversion. The Docling wrapper converts natively, then immediately flattens the result into custom `ConvertedElement`/`ExtractedChunk` DTOs and later stages mutate `DocumentElement` rows plus hand-built markdown instead of mutating and reserializing the `DoclingDocument` itself. This diverges from Docling's documented post-conversion mutation/enrichment flow and is the root cause of the stale JSON vs current-text split.
>
> 2. High: native Docling chunkers are bypassed in favor of a custom heuristic chunker over flattened rows. Upstream Docling documents two native approaches and explicitly recommends chunking directly from `DoclingDocument` via `BaseChunker`/`HybridChunker`; this branch instead uses `structure_aware_chunk()` with approximate token counting and manual overlap logic.
>
> 3. High: Docling's own enrichment path is replaced with custom translation and picture-description stages that do not update the canonical Docling JSON. The converter explicitly disables picture description, then later stages write translated text to `DocumentElement.translated_text`, append image descriptions to markdown, and inject `_enriched_text` into the extraction payload. Native Docling examples instead mutate the `DoclingDocument` and regenerate output from that object.
>
> 4. High: Docling-Graph template authoring is replaced by a custom ontology-to-Pydantic generator. The upstream docs show explicit Pydantic models using `graph_id_fields`, `is_entity=False` for components, and `edge()` for relationships, while this branch auto-derives identity fields, never emits component semantics, and encodes edges with `json_schema_extra={"edge_label": ...}`.
>
> 5. High: the wrapper introduces a custom "unified template" container instead of using canonical authored template classes. This is a custom accommodation of Docling-Graph's singular `PipelineConfig.template` surface, but it moves semantics out of explicit domain models into a generated wrapper model with list fields for every entity type.
>
> 6. High: native Docling-Graph graph output, provenance, and metadata are normalized into an app-specific `{entities, relationships}` contract and then partially discarded. The client flattens the node-link graph, the pipeline keeps only `properties`, ignores most response `metadata`, and rebuilds mention grounding with regex matching instead of consuming upstream provenance/resolver output.
>
> 7. High: ArcadeDB is not the primary structure graph for retrieval even though the pipeline writes structural edges there. The main hybrid expansion path reads Postgres `retrieval.chunk_links` first and only falls back to ArcadeDB neighborhood traversal for legacy cases.
>
> 8. High: query profiles and dossiers do not compile their own traversal model into native ArcadeDB `MATCH` traversals. The profile schema supports directed multi-step paths with `direction`, `min_hops`, and `max_hops`, but execution collapses that into one generic undirected neighborhood walk using `.both(...)`.
>
> 9. Medium: the backend-agnostic `GraphStore` abstraction strips native ArcadeDB result semantics and encourages lowest-common-denominator query usage.
>
> 10. Medium: retrieval filters are handled in Python after search/expansion instead of being pushed into native Postgres/ArcadeDB queries, even though SQL-side filter builders already exist.
>
> 11. Medium: evidence and ontology expansion are routed through custom helper contracts instead of native ArcadeDB graph semantics.
>
> 12. Low: native integration coverage is weak, so many of these custom-vs-native drifts are not protected by tests.
>
> **Handoff Priorities**
> 1. Make `DoclingDocument` the authoritative mutable artifact through translation and picture enrichment, then regenerate markdown/JSON from it.
> 2. Replace the custom Docling chunker where feasible with native Docling chunkers, or document why retrieval chunking must intentionally differ.
> 3. Decide whether Docling-Graph templates should be explicit canonical Pydantic models instead of generated wrappers.
> 4. Preserve Docling-Graph provenance and metadata instead of rebuilding mention grounding in Python.
> 5. Move retrieval, query-profile traversal, and dossier evidence onto native ArcadeDB graph traversal instead of Postgres-first/generic-wrapper execution.
>
> **Sources**
> - ArcadeDB Manual.pdf
> - https://docling-project.github.io/docling/concepts/chunking/
> - https://docling-project.github.io/docling/examples/translate/
> - https://docling-project.github.io/docling/examples/enrich_doclingdocument/
> - https://ibm.github.io/docling-graph/usage/examples/docling-document-input/
> - https://ibm.github.io/docling-graph/fundamentals/schema-definition/entities-vs-components/

---

### Verbatim Retrieval Query Mechanisms Review (2026-04-06)

> ArcadeDB usage itself is mostly canonical here; the main problems are in the custom wrappers around it. I checked this against ArcadeDB Manual.pdf. Docling and Docling-Graph are upstream to graph quality, but they are not directly in the active query path, so the retrieval review is mostly an ArcadeDB and application-layer review.
>
> **Findings**
>
> 1. High: query profiles do not honor their own traversal model. The schema supports per-step `direction`, `min_hops`, and `max_hops` at query_profiles.py (line 13), but execution collapses a profile to `rel_types + max_depth` and runs one undirected neighborhood traversal at query_profiles.py (lines 556, 569). The backend traversal is `MATCH ... .both(...)` at arcadedb_graph.py (line 792). So profiles are effectively "any of these rel types within N hops," not the configured directed path model.
>
> 2. High: dossier and query-profile evidence attachment is wired through a chunk-centric helper using entity IDs. Both services call `get_ontology_linked_chunks(item.node_id)` for entities at dossier_service.py (line 338) and query_profiles.py (line 645). But `get_ontology_linked_chunks()` is explicitly implemented as `chunk <- EXTRACTED_FROM - entities -> other chunks` at arcadedb_graph.py (line 856). That is correct for a seed chunk, not for an entity RID, so dossier/query-profile evidence is likely empty or wrong.
>
> 3. High: the hybrid cross-modal fallback for legacy documents is probably broken by UUID/RID mismatch. `_expand_via_cross_modal()` passes `str(seed.chunk_id)` into `get_neighborhood()` at retrieval.py (lines 339, 646). `get_neighborhood()` then interpolates that value directly into `@rid = {node_id}` at arcadedb_graph.py (line 792). Seed chunk IDs are UUIDs, not ArcadeDB RIDs, and this path does not resolve them first.
>
> 4. High: hybrid "ontology expansion" is not preserving enough native graph information. `_expand_via_ontology()` expects `target_chunk_type`, `rel_type`, and entity context at retrieval.py (line 710), but `get_ontology_linked_chunks()` returns raw chunk rows from `expand(out('EXTRACTED_FROM'))` at arcadedb_graph.py (line 889). That means relation typing collapses to the fallback `RELATED_TO`, entity names are blank, and image chunks default to `text_chunk` lookup and can be dropped.
>
> 5. Medium: retrieval filters are only partially implemented, and only after search/expansion. The request schema still exposes `classification`, `modalities`, `source_ids`, and `document_ids` at retrieval.py (line 37). But `unified_query()` applies filters only after strategy execution at retrieval.py (line 78), and `_apply_query_filters()` explicitly skips `source_ids` at retrieval.py (line 56). SQL-side filter builders exist at _retrieval_helpers.py (line 120), but they are not used by the live retrieval path.
>
> 6. Medium: the API defaults do not match the advertised retrieval settings. `UnifiedQueryRequest.top_k` defaults to `10` and `min_confidence` defaults to `None` at retrieval.py (line 58), while `/v1/settings/retrieval` exposes `20` and `0.1` from config at retrieval.py (line 1124) and config.py (line 152). Raw API clients that omit these fields do not get the documented defaults.
>
> 7. Medium: graph query responses surface the wrong score semantics and strip too much node data. Fulltext search is correctly ordered by Lucene `$score` at arcadedb_graph.py (line 328), but `_to_entity()` ignores `$score` and only maps vector distance or `extraction_confidence` at arcadedb_graph.py (line 53). `/graph/query` then returns `match.extraction_confidence` as the score at graph_store.py (line 66). `/graph/neighborhood` also returns minimal node/edge payloads rather than fuller native properties at arcadedb_graph.py (line 833).
>
> 8. Low: co-extracted discovery exists in the backend, but it is not actually wired into the query stack. The backend method exists at arcadedb_graph.py (line 921), and `resolve_root_entity()` in query profiles even comments about a "co-extracted fallback" at query_profiles.py (line 536). But the implementation falls back directly to `resolve_root_entity()` exact lookup and never calls `get_co_extracted_entities()`.
>
> 9. Low: verification is weaker than the branch suggests. The concrete ArcadeDB retrieval path is not what most passing tests validate, because the shared fixture replaces GraphStore with mocks at conftest.py (line 206). There are also stale tests that still assert `response["mode"]` at test_retrieval_api.py (line 15) and test_full_pipeline.py (line 66), while the live response shape uses `strategy` and `modality_filter` at retrieval.py (line 114).
>
> **Native-First Direction**
> - Keep the current native ArcadeDB primitives. `vectorNeighbors(...)->expand(...)` and `MATCH` traversal are the right foundation.
> - Generate native `MATCH` from query-profile traversal steps instead of collapsing them into a custom undirected neighborhood wrapper.
> - Split chunk-centric expansion from entity-centric evidence lookup instead of forcing both through `get_ontology_linked_chunks()`.
> - Push filters into native Postgres/ArcadeDB queries where possible instead of post-filtering in Python.
> - Surface native relevance where it exists. For graph fulltext, that means carrying Lucene `$score` through instead of substituting extraction confidence.

---

### Verbatim Graph Extraction Pipeline Review (2026-04-06)

The following is the complete graph extraction review for reference when addressing items #45-#50.

> **Graph Extraction Review**
>
> The stage ordering is sound: extract entities/relationships and chunks in parallel, wire only after both exist, then canonicalize. The implementation problem is not the DAG shape. It is that the graph extraction contract has drifted across the worker, the Docling-Graph service, and the downstream consumers.
>
> **Stage 1 is built against a stale Docling-Graph API.**
> derive_ontology_graph() reconstructs full_text from DocumentElement rows and calls extract_graph_all(full_text, document_id) at pipeline.py (line 2226) and pipeline.py (line 2257). The client sends text in docling_graph_service.py (line 172), but the service now requires docling_document_json in schemas.py (line 10) and consumes that in main.py (line 141). The service also returns graph and metadata, not entities and relationships, but the worker still reads result.get("entities") and result.get("relationships") at pipeline.py (line 2258). If this code path is live, successful extraction is likely being interpreted as zero nodes and zero edges.
>
> **The pipeline already has the canonical structured artifact, but the graph stage ignores it.**
> prepare_document persists docling_document.json to object storage at pipeline.py (line 915). Another stage later downloads that same JSON at pipeline.py (line 1401). derive_ontology_graph() does neither. It rebuilds plain text from normalized elements instead. That throws away layout, structure, and native provenance before the extraction service even runs.
>
> **The current "primary" mention path is not LLM grounding; it is lexical matching.**
> _build_entity_mentions() uses word-boundary regex for short names and substring matching for longer names at pipeline.py (line 2116). It does not resolve paraphrase, coreference, metonymy, abbreviation expansion, or implicit references. So your #27 concern is valid, but more precisely: the implementation never attempts semantic mention grounding in the first place.
>
> **Partial mention miss cases are never repaired.**
> derive_structure_links() only falls back to artifact-wide entity->chunk linking when mentions is empty, not when mentions are incomplete, at pipeline.py (line 2696) and pipeline.py (line 2708). That means if lexical matching finds 2 real mentions and misses 3 implicit ones, the stage keeps the 2 and silently drops the 3. The fallback does not help recall unless the primary path fails completely.
>
> **The entity->chunk wiring is text-centric, not fully element-centric.**
> The element_uid -> chunk_id map is built only from text_chunks via artifact_id at pipeline.py (line 2672). image_chunks are wired to the document and same-page neighbors, but they are not included in the mention map used for EXTRACTED_FROM. So even if extraction identifies entities grounded in image/schematic elements, the primary wiring path does not appear to attach them to image chunks.
>
> **Even perfect EXTRACTED_FROM edges would not currently pay off fully, because traversal is broken downstream.**
> The graph writer creates EXTRACTED_FROM as entity -> chunk at arcadedb_graph.py (line 1453), but lookup traverses in('EXTRACTED_FROM') in arcadedb_graph.py (line 724). Retrieval also passes chunk UUIDs into that helper at retrieval.py (line 314) and retrieval.py (line 696), while the helper interpolates directly into ArcadeDB SQL as if it were a RID. So the downstream consumers you listed are currently degraded both by missing edges and by broken traversal semantics.
>
> **Canonicalization is in the right place in the DAG, but the implementation is not giving the graph a reliable dedup pass.**
> derive_canonicalization runs after wiring, which is the correct point architecturally. But canonicalize_document_entities() discovers "document entities" by calling fulltext_search_sync(document_id) at canonicalization.py (line 64), and that search is just WHERE name LUCENE :query at arcadedb_graph.py (line 1568). Alias lookup is also broken on alias vs alias_name. So the dedup stage is not operating on solid document-local provenance.
>
> **What I'd Tell Another Agent**
>
> Reconcile the Stage 1 contract first.
> The worker, HTTP client, and service do not agree on request or response shape. Until that is fixed, all discussion about mention precision is second-order.
>
> Make derive_ontology_graph() consume persisted docling_document.json.
> The canonical structured document already exists. Using reconstructed full_text is both lossy and out of sync with the actual service API.
>
> Normalize the extraction output shape before import.
> The worker needs a stable adapter from Docling-Graph output to nodes/edges/provenance records. Right now it assumes an old shape.
>
> Reassess mention grounding only after the contract is stable.
> At that point the real question becomes whether to keep lexical mention building, enrich it with LLM/entity-resolution logic, or consume provenance directly from Docling-Graph output if available.
>
> Fix traversal before measuring graph-extraction quality.
> Otherwise better entity->chunk edges will still not show up properly in retrieval, dossier evidence, or ontology expansion.
>
> **Test Gaps**
>
> The client tests still assert the old text-based interface in test_docling_graph_client.py (line 119).
> The docling-graph integration tests are stale and patch a removed _templates global at test_pipeline_integration.py (line 43).
> There is no strong end-to-end test that proves: DoclingDocument JSON -> Docling-Graph extraction -> graph_json mentions -> EXTRACTED_FROM edges -> retrieval traversal all agree on the same identifiers and schema.
>
> My bottom-line assessment is: the architecture is defensible, but the current graph extraction process is not trustworthy until the worker/service contract is repaired. After that, #27 becomes a meaningful optimization target; before that, it is not the primary blocker.

---

### Verbatim Code Analysis and Review (2026-04-05)

The following is the complete standalone review for reference when addressing the above items.

> Standalone review of the current implementation only; no base-branch comparison.
>
> **Findings**
>
> 1. Critical: the "clone/update upstream repos on every build" contract is not actually satisfied for Docling or Docling-Graph, and it is not part of `docker compose build` itself. [manage.sh](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/manage.sh#L123), [manage.sh](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/manage.sh#L174), [docker-compose.yml](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker-compose.yml#L111), [docker-compose.yml](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker-compose.yml#L143), [docker/arcadedb/Dockerfile](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/arcadedb/Dockerfile#L6), [docker/docling/Dockerfile](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling/Dockerfile#L102), [docker/docling-graph/Dockerfile](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/Dockerfile#L83), [docker/docling/requirements.txt](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling/requirements.txt#L1), [docker/docling-graph/requirements.txt](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/requirements.txt#L5). Only ArcadeDB consumes a cloned `repo/`; Docling and Docling-Graph install released packages. A plain `docker compose build` clones nothing, and even `manage.sh` continues on `git pull` failure with only a warning at [manage.sh](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/manage.sh#L115). This also conflicts with the repo's stated verification/design contract in [VERIFICATION_CHECKLIST.md](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/VERIFICATION_CHECKLIST.md#L236) and [2026-04-04-arcadedb-migration-design.md](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docs/superpowers/specs/2026-04-04-arcadedb-migration-design.md#L395).
>
> 2. Critical: `vector_search()` drops the chunk metadata that retrieval depends on. [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L858), [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L69), [app/api/v1/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/api/v1/retrieval.py#L449), [app/api/v1/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/api/v1/retrieval.py#L523). The API expects `chunk_id`, `document_id`, `artifact_id`, `modality`, and chunk text in `hit.properties`, but the ArcadeDB query only projects entity fields plus distance/RID. The ArcadeDB manual says `expand(vectorNeighbors(...))` returns all document properties; the current query throws those away.
>
> 3. Critical: ontology/evidence traversal is broken by both edge direction and identifier mismatch. EXTRACTED_FROM edges are written entity -> chunk in [app/workers/pipeline.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/workers/pipeline.py#L2742) and [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1453), but lookup traverses `in('EXTRACTED_FROM')` in [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L730). Retrieval also passes `str(seed.chunk_id)` into that helper in [app/api/v1/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/api/v1/retrieval.py#L314) and [app/api/v1/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/api/v1/retrieval.py#L326), but the helper interpolates directly into `FROM {node_id}`, which is RID-oriented SQL. Impact: ontology expansion and evidence attachment are likely empty or reversed in [app/services/query_profiles.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/query_profiles.py#L659) and [app/services/dossier_service.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/dossier_service.py#L352).
>
> 4. High: alias resolution is internally inconsistent and likely nonfunctional against ArcadeDB. Alias schema/creation use `alias_name` in [app/services/arcadedb_schema.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_schema.py#L60) and [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L786), but lookup queries `WHERE alias = :alias` in [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L813) and [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1603). The optional `entity_type` filter is also applied on the `Alias` vertex query itself. That directly affects root resolution in [app/services/query_profiles.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/query_profiles.py#L519), [app/services/dossier_service.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/dossier_service.py#L166), and canonicalization in [app/services/canonicalization.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/canonicalization.py#L129).
>
> 5. High: the docling-graph wrapper is not following canonical docling-graph template usage. [docker/docling-graph/app/main.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/app/main.py#L56), [docker/docling-graph/app/main.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/app/main.py#L93), [docker/docling-graph/app/template_builder.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/app/template_builder.py#L101), [docker/docling-graph/app/template_builder.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/app/template_builder.py#L218), [config.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/.venv/lib/python3.12/site-packages/docling_graph/config.py#L113). The library expects a singular `PipelineConfig.template`; this service builds many templates and passes only the first one. The generated models also use `graph_id_fields` but not the documented `is_entity=True`/`edge()` pattern. That is a real misalignment with docling-graph's canonical API surface, not just a style difference.
>
> 6. High: retrieval `filters` are still public API but are silently ignored. [app/schemas/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/schemas/retrieval.py#L64), [app/schemas/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/schemas/retrieval.py#L37). `app/api/v1/retrieval.py` does not read `body.filters` anywhere, so callers can request classification/document/modality constraints and get unfiltered results with no warning.
>
> 7. Medium: document canonicalization is not actually document-scoped. [app/services/canonicalization.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/canonicalization.py#L68), [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1582). The code claims it finds entities linked to a document, but it really does `WHERE name LUCENE :query` using the document ID string. That makes the canonicalization pass logically disconnected from document/chunk/entity linkage.
>
> 8. Medium: orphan cleanup likely uses the wrong ArcadeDB system field. [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1107), [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1527). The manual defines `@class` as the type name and `@cat` as the type category, so filtering `@cat NOT IN ['Document', 'TextChunk', 'ImageChunk', 'Alias']` is almost certainly targeting the wrong field.
>
> 9. Medium: community-report vector search loses similarity scores. [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1035), [app/api/v1/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/api/v1/retrieval.py#L993). The search query does not project distance/score, but global retrieval assumes `score` exists, so synthesis metadata collapses to `0.0`.
>
> 10. Medium: the highest-risk integration points are not currently verified. GraphStore is globally stubbed in [tests/conftest.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/tests/conftest.py#L212), so passing backend/query-profile tests do not validate the concrete ArcadeDB alias/chunk/vector behavior. The docling-graph integration suite currently fails because it still patches `_templates` at [docker/docling-graph/tests/test_pipeline_integration.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/tests/test_pipeline_integration.py#L43), which no longer exists.
>
> **Verification**
> - Backend suites passed: `test_arcadedb_graph`, `test_query_coverage`, `test_community_tasks`, `test_startup_bootstrap`, plus `test_query_profiles` and `test_query_profiles_api`.
> - Those passing suites are not strong evidence for the concrete backend behavior because the shared fixture replaces GraphStore methods with mocks.
> - `docker/docling-graph` tests partly pass, but `tests/test_pipeline_integration.py` currently errors at setup because the test harness is stale relative to `app.main`.
>
> **Sources**
> - [ArcadeDB Manual.pdf](/home/josh/development/EIP-MMDPP/ArcadeDB%20Manual.pdf)
> - [IBM/docling-graph README](https://raw.githubusercontent.com/IBM/docling-graph/main/README.md)
> - [docling-project/docling README](https://raw.githubusercontent.com/docling-project/docling/main/README.md)
> - [ArcadeData/arcadedb repo](https://github.com/ArcadeData/arcadedb)
