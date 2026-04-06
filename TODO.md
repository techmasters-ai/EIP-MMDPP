# TODO — Future Work and Optimizations

This document tracks work identified during the ArcadeDB migration and Docling-Graph refactor that was intentionally deferred. Each item has enough context to be executed independently.

**Last updated:** 2026-04-05
**Related branch:** `feature/arcadedb`
**Related specs:**
- `docs/superpowers/specs/2026-04-04-arcadedb-migration-design.md`
- `docs/superpowers/specs/2026-04-04-docling-graph-pipeline-refactor-design.md`

---

## Priority Key

- **P0** — Blocks production use. Do first.
- **P1** — Significant functional gap or performance issue. Schedule soon.
- **P2** — Code quality, maintainability, or optimization. Do when touching related code.
- **P3** — Nice-to-have, low risk, low impact.

---

## P0 — Required for Production

### 1. Implement real LLM community report generation ✅ DONE (2026-04-05)

**Status:** ~~Stub implementation exists.~~ Implemented: `_call_llm_for_report()` calls Ollama `/v1/chat/completions` with JSON-output prompt, parses via `parse_llm_json_loose`, line-fallback on non-JSON, handles `reasoning_content` for thinking models. `_generate_community_report()` fetches real relationships via the new `GraphStore.get_relationships_between_entities()` method and interpolates them into the prompt.
**Files:** `app/services/arcadedb_community.py` (functions `_generate_community_report`, `_call_llm_for_report`)

**Current state:**
- `_call_llm_for_report()` returns a hardcoded stub
- `_generate_community_report()` has a placeholder `"(relationships would be fetched from graph)"` string
- Without this, the global query strategy cannot produce meaningful synthesized answers

**What needs to be done:**
1. Implement real Ollama LLM call in `_call_llm_for_report()`. Use the same pattern as `app/services/document_analysis.py` or `app/services/translation.py` — both make LLM calls via httpx to the Ollama chat API endpoint.
2. Use `settings.community_report_llm_model` (already configured) as the model name.
3. Use `settings.community_report_llm_prompt` (env var override) as the prompt template. Default template is hardcoded in the module (`_DEFAULT_PROMPT`).
4. Fetch actual relationships between community members: query ArcadeDB for edges where both endpoints are in the community member list. Add a new GraphStore method if needed (e.g., `get_edges_between_entities(entity_names: list[str]) -> list[dict]`).
5. Format the prompt with entities, relationships, and optionally evidence chunks (query TextChunk vertices via EXTRACTED_FROM).
6. Parse the LLM response into `{title, summary}`. Expect JSON output; fall back to using the first line as title.
7. Write tests in `tests/unit/test_arcadedb_community.py` covering the LLM call path with a mocked httpx client.

**Acceptance:**
- Community detection run generates non-stub reports
- Reports include domain-relevant titles and summaries
- Global query returns meaningful synthesized answers citing community reports

---

### 2. Implement community report embedding ✅ DONE (2026-04-05)

**Status:** ~~Stub implementation exists.~~ Implemented: `_embed_report()` wraps `embed_texts` in `asyncio.to_thread`, embeds title+summary together, returns `None` on empty input or failure. `run_community_detection()` writes the embedding via `graph_store.set_vertex_embedding()` after upserting the CommunityReport.
**Files:** `app/services/arcadedb_community.py` (function `_embed_report`)

**Current state:**
- `_embed_report()` returns `None`
- Without real embeddings, `CommunityReport.report_embedding` is never populated
- Global query `vectorNeighbors('CommunityReport[report_embedding]', ...)` returns empty results

**What needs to be done:**
1. Implement real embedding call. Use the existing embedding pattern from `app/services/embedding.py` (which calls Ollama `/v1/embeddings` with BGE-M3).
2. The function should take the report text (title + summary) and return a 1024-dim float list.
3. Update `run_community_detection()` to write the embedding via `graph_store.set_vertex_embedding()` after creating the CommunityReport vertex.
4. Add a test that verifies the embedding is a list of 1024 floats.

**Acceptance:**
- After community detection runs, `SELECT FROM CommunityReport WHERE report_embedding IS NOT NULL` returns all reports
- Global query finds reports by semantic similarity to query text

---

### 3. Wire LLM synthesis into global query response ✅ DONE (2026-04-05)

**Status:** ~~Global query returns raw community reports, not synthesized answer.~~ Implemented: `_global_query()` calls `_synthesize_global_answer()` which issues a single Ollama chat call with a configurable prompt (`COMMUNITY_GLOBAL_SYNTHESIS_PROMPT`), returns the synthesized answer as the primary `content_text` in a single `QueryResultItem`, and preserves raw reports + scores in `context["reports"]`. Falls back to concatenated reports if the LLM call fails.
**Files:** `app/api/v1/retrieval.py` (function `_global_query`)

**Current state:**
- `_global_query()` calls `search_community_reports()` and returns the raw reports
- The spec calls for "LLM synthesis: combine reports into comprehensive answer, citing source documents and page numbers"
- Current output is a list of community summaries, not a unified answer

**What needs to be done:**
1. After fetching top-k community reports, build a synthesis prompt: "Given these community reports, synthesize a comprehensive answer to the user's question: {query}. Community reports: {reports}. Cite the community_id and any referenced source documents."
2. Make an LLM call (reuse pattern from `_call_llm_for_report`).
3. Return the synthesized answer as the primary `content_text` in the `UnifiedQueryResponse`, with individual community reports available in the `context` field.
4. Preserve source document citations so the frontend can display them.

**Acceptance:**
- Global query returns a single synthesized answer, not a list
- Frontend `GlobalQueryDetail` component shows the answer with expandable community sources

---

### 4. Add batch HTTP operations to replace N+1 patterns ✅ DONE (2026-04-05)

**Status:** ~~Functional but slow.~~ Implemented: `upsert_nodes_batch` / `_sync` and `upsert_relationships_batch` / `_sync` now build a single `sqlscript` with row-suffixed params (`:name_0`, `:f_name_1`, `:t_name_1`, etc.) and issue one HTTP call per batch. Provenance edges also batched. `create_text_chunk_vertex` / `create_image_chunk_vertex` accept an `embedding` param that is folded into the CREATE SQL. New batch methods `create_text_chunks_batch_sync` / `create_image_chunks_batch_sync` create all chunks (with embeddings) in one sqlscript call. Pipeline stages updated to collect records and call the batch methods once instead of looping.
**Files:** `app/services/arcadedb_graph.py`, `app/workers/pipeline.py`

**Current state:**
- `upsert_nodes_batch` and `upsert_nodes_batch_sync` are fake batches (loops calling individual upserts)
- `upsert_relationships_batch` / `_sync` same issue
- Pipeline embedding stages call `create_text_chunk_vertex_sync` + `set_vertex_embedding_sync` per chunk (2 HTTP calls × N chunks)
- `derive_structure_links` calls `get_chunk_rid_sync` + `create_structural_edge_sync` per chunk (2 × N)
- ArcadeDB batch endpoint (`POST /api/v1/batch/{database}`) exists in the client but is never used for node/edge operations

**What needs to be done:**
1. Update `upsert_nodes_batch` to use ArcadeDB's NDJSON batch endpoint:
   - Build NDJSON body: one line per vertex with `{"@type": "vertex", "@class": "RADAR_SYSTEM", "name": "...", ...}`
   - Call `self._client.batch(database, ndjson_body)` once
   - Parse response `idMapping` if needed
2. Update `upsert_relationships_batch` similarly with edge records: `{"@type": "edge", "@class": "HAS_COMPONENT", "@from": "rid1", "@to": "rid2", ...}`
3. Update `create_text_chunk_vertex` to include the embedding in the initial CREATE SQL instead of making two calls: `CREATE VERTEX TextChunk SET chunk_id = ..., text_embedding = [...], ...`
4. Update `derive_structure_links` in `pipeline.py` to batch chunk RID lookups and edge creations via batch endpoint.
5. Benchmark: document with 50 entities, 100 relationships, 200 chunks should drop from 400+ HTTP calls to 5-10.

**Acceptance:**
- Ingest time for a typical 20-page document drops significantly
- Integration tests confirm entities/relationships still created correctly

---

## P1 — Significant Functional Gaps

### 5. Implement validation_matrix enforcement in upsert_relationship ✅ DONE (2026-04-05)

**Status:** ~~Spec requires it; implementation does not check.~~ Implemented: `ArcadeDBGraphStore` caches a validation-matrix set loaded from `load_validation_matrix()`, refreshed on `sync_schema()`. `upsert_relationship`, `upsert_relationships_batch`, and the sync variant all check `(from_type, rel_type, to_type)` against the matrix. New `GRAPH_REJECT_INVALID_RELATIONSHIPS` env var controls hard reject vs warn-and-skip (default: warn). Empty matrix = permissive (no matrix defined = no enforcement).
**Files:** `app/services/arcadedb_graph.py` (function `upsert_relationship`)

**Current state:**
- Spec (Addendum: Validation matrix enforcement): "The GraphStore `upsert_relationship()` checks `(source_type, rel_type, target_type)` against the loaded matrix and rejects invalid triples."
- Current implementation writes any triple to ArcadeDB without checking.
- Invalid triples (e.g., `FREQUENCY_BAND -> INSTALLED_ON -> PLATFORM`) silently accepted.

**What needs to be done:**
1. Load the validation_matrix from the active ontology (use `app/services/ontology_templates.py:load_validation_matrix()`).
2. Cache the matrix in-memory on the GraphStore instance (refresh on schema sync).
3. In `upsert_relationship`, check if `(from_type, rel_type, to_type)` is in the matrix. If not, either raise `ValidationError` or log a warning and skip.
4. Add a setting: `GRAPH_REJECT_INVALID_RELATIONSHIPS=true` to control hard reject vs warn.
5. Add unit tests for valid and invalid triples.

**Acceptance:**
- Attempting to write an invalid relationship triple raises/logs per config
- Tests cover the matrix lookup and fallback cases

---

### 6. Implement post-ingest hook counter + threshold trigger ✅ DONE (2026-04-05)

**Status:** ~~Spec requires it; only scheduled and manual triggers exist.~~ Implemented: `finalize_document` now calls `_maybe_trigger_post_ingest_community_detection()` after marking status COMPLETE. The helper INCRs `community:pending_ingest_count` in Redis, logs the count/threshold, and when the threshold is reached resets the counter to 0 and dispatches `run_community_detection_task.delay(mode="incremental")`. Errors are swallowed so ingestion never fails. Controlled by `COMMUNITY_DETECTION_POST_INGEST_ENABLED` / `COMMUNITY_DETECTION_POST_INGEST_THRESHOLD` env vars.
**Files:** `app/workers/pipeline.py` (in `finalize_document` task), `app/workers/community_tasks.py`

**Current state:**
- Spec (Section 4): "After finalize_document completes, if COMMUNITY_DETECTION_POST_INGEST_ENABLED is true, increment a Redis counter. When counter reaches COMMUNITY_DETECTION_POST_INGEST_THRESHOLD, trigger community detection task."
- Scheduled trigger (Celery Beat) and manual trigger (POST /v1/community/detect) work
- Post-ingest automatic trigger is missing

**What needs to be done:**
1. In `finalize_document()` in pipeline.py, after marking document COMPLETE, check `settings.community_detection_post_ingest_enabled`.
2. If enabled, `INCR` a Redis counter at key `community:pending_ingest_count`.
3. If the counter reaches `settings.community_detection_post_ingest_threshold`, reset the counter (`SET` to 0) and call `run_community_detection_task.delay(mode="incremental")`.
4. Add a test that verifies the counter increments and triggers at the threshold.

**Acceptance:**
- After ingesting N documents (where N = threshold), community detection runs automatically
- Counter resets after each trigger

---

### 7. Add batch_create_entity_chunk_edges to GraphStore Protocol ✅ DONE (2026-04-05)

**Status:** ~~Pipeline loops through individual edges instead.~~ Implemented: new `EntityChunkEdge` dataclass, new Protocol method `batch_create_entity_chunk_edges_sync(edges)` implemented in `ArcadeDBGraphStore` via a single sqlscript call with one `CREATE EDGE EXTRACTED_FROM FROM (SELECT FROM {type} WHERE name=...) TO :rid` per row. `derive_structure_links` now collects all `(ent_name, ent_type, chunk_rid)` tuples and issues one batch call instead of looping per-edge. The legacy singular `create_entity_chunk_edge_sync` remains for single-edge callers.
**Files:** `app/services/graph_store.py`, `app/services/arcadedb_graph.py`, `app/workers/pipeline.py`

**Current state:**
- Spec (Section 2) lists `batch_create_entity_chunk_edges` as a Protocol method
- Implementation only has singular `create_entity_chunk_edge_sync`
- Pipeline `derive_structure_links` calls it in a loop (lines ~2728-2739 of pipeline.py)
- For a doc with 50 entities × 5 chunk mentions each, that's 250 sequential HTTP calls

**What needs to be done:**
1. Add `batch_create_entity_chunk_edges(edges: list[EntityChunkEdge]) -> int` to GraphStore Protocol, plus sync variant.
2. Define `EntityChunkEdge` dataclass with `entity_name`, `entity_type`, `chunk_id`, `page_number`, `document_id`.
3. Implement in `arcadedb_graph.py` using the ArcadeDB batch endpoint or a single multi-INSERT SQL statement.
4. Update `derive_structure_links` to collect all edge descriptors, then call the batch method once.
5. Benchmark: should drop per-document edge creation from N sequential calls to 1.

**Acceptance:**
- Structure link creation time drops significantly on docs with many entities
- Graph still has correct EXTRACTED_FROM edges

---

### 8. Optimize schema sync to batch DDL statements ✅ DONE (2026-04-05)

**Status:** ~~Schema sync makes 150+ sequential HTTP calls at API startup.~~ Implemented: `sync_schema_from_ontology` now groups DDL into 7 phase-scoped sqlscript batches (entity types + props, edge types + props, structural vertex types + props, structural edge types + props, vector indexes, fulltext indexes, unique indexes). Each phase is one HTTP call. New `_run_ddl_batch` helper falls back to per-statement execution on batch failure so one bad statement doesn't drop the whole phase. Expected reduction: ~200 calls → 7 calls per startup.
**Files:** `app/services/arcadedb_schema.py`

**Current state:**
- `sync_schema_from_ontology()` issues one HTTP call per `CREATE VERTEX TYPE`, `CREATE PROPERTY`, `CREATE INDEX`
- For an ontology with 46 entity types × 7 common props + specific props + structural types + indexes, this is ~200 sequential calls
- API startup takes measurably longer than necessary

**What needs to be done:**
1. Concatenate DDL statements into batches separated by semicolons.
2. Use ArcadeDB's `sqlscript` language (passes multiple statements in a single transaction): `client.command(database, "sqlscript", "CREATE ...; CREATE ...; CREATE ...")`.
3. Alternatively, group related statements (all vertex types, all properties for one type, all indexes for one type) into separate batch calls.
4. Verify idempotency is preserved — `IF NOT EXISTS` clauses must still work within sqlscript blocks.
5. Benchmark: API startup schema sync should drop from ~5-10s to <2s on typical ontologies.

**Acceptance:**
- API startup completes faster
- Schema sync output is unchanged (same types, properties, indexes created)
- Idempotent runs still produce no errors

---

### 9. Implement time-based ensure_ready caching

**Status:** Every graph-writing task makes an extra HTTP call at startup.
**Files:** `app/services/arcadedb_graph.py` (functions `ensure_ready`, `ensure_ready_sync`)

**Current state:**
- Every Celery task calling `ensure_ready_sync()` queries `schema:database` to verify readiness
- For a pipeline with 5 graph-writing stages per document, that's 5 extra HTTP round-trips per document
- ArcadeDB readiness rarely changes between stages

**What needs to be done:**
1. Add a class-level timestamp cache: `_last_ready_check: float = 0`
2. In `ensure_ready()` / `_sync`, skip the check if `time.monotonic() - self._last_ready_check < 60.0` (configurable via `ARCADEDB_READY_CACHE_SECONDS`).
3. Update the timestamp only on successful check.
4. On exception, clear the timestamp so the next call re-checks.
5. Add a test that verifies caching behavior (first call checks, second call within TTL skips, after TTL re-checks).

**Acceptance:**
- Pipeline stages after the first skip the readiness check
- Worker logs show "ArcadeDB ready (cached)" for subsequent calls
- Cache invalidates on failure

---

## P2 — Code Quality and Maintainability

### 10. Extract shared SQL building to reduce async/sync duplication

**Status:** `arcadedb_graph.py` has ~500 lines of near-duplicate async/sync code.
**Files:** `app/services/arcadedb_graph.py`

**Current state:**
- Every async method has a `_sync` twin with identical SQL logic and differing only in `await` + `command_sync` vs `command`
- Examples: `_upsert_node_impl` / `_upsert_node_impl_sync`, `create_text_chunk_vertex` / `_sync`, `delete_document_graph` / `_sync`, `fulltext_search` / `_sync`, etc.
- Risk: logic drift between async and sync (already happened with `upsert_relationships_batch_sync` dropping `record.properties`, which was caught in code review and fixed)

**What needs to be done:**
1. For each duplicated method pair, extract a helper that returns `(sql_string, params_dict)`.
2. Async method: `sql, params = self._build_X_sql(...); return await self._client.command(db, "sql", sql, params)`.
3. Sync method: `sql, params = self._build_X_sql(...); return self._client.command_sync(db, "sql", sql, params)`.
4. For methods with multiple SQL calls (e.g., `delete_document_graph` has 5 statements), extract a list of `(sql, params)` tuples and loop over them.
5. This should roughly halve the file size.
6. Run all unit tests to ensure no regression.

**Acceptance:**
- `arcadedb_graph.py` is significantly shorter
- All existing tests still pass
- No divergence between async/sync behavior

---

### 11. Consolidate Redis client factory

**Status:** `Redis.from_url(settings.celery_broker_url)` called in 5+ files.
**Files:** `app/workers/community_tasks.py`, `app/api/v1/sources.py`, `app/workers/pipeline.py`, `app/services/docling_graph_service.py`, and others

**Current state:**
- Each caller creates its own Redis client
- No shared connection pool
- `docling_graph_service.py` has a private `_get_redis()` but it's module-local
- Risk of connection leaks (e.g., the `community_tasks.py` leak that was fixed in review)

**What needs to be done:**
1. Create `app/services/redis_utils.py` with a singleton `get_redis()` function that caches the client.
2. Add a `close_redis()` function for shutdown hooks.
3. Replace all `Redis.from_url(...)` calls across the codebase with `get_redis()`.
4. For Celery workers, register `close_redis()` as a shutdown handler.
5. Add tests verifying the singleton behavior.

**Acceptance:**
- Single Redis client shared across modules
- No connection leaks on long-running workers
- Shutdown cleanly closes the connection

---

### 12. Standardize Redis locking idiom

**Status:** Two different patterns in the codebase.
**Files:** `app/workers/community_tasks.py`, `app/services/docling_graph_service.py`

**Current state:**
- `community_tasks.py` uses bare `r.set(key, value, nx=True, ex=TTL)` + `r.delete()` in try/finally
- `docling_graph_service.py` uses `r.lock()` with `acquire(blocking=False)` and context manager
- Two idioms make the codebase harder to reason about

**What needs to be done:**
1. Pick one pattern (recommend `r.lock()` for its auto-release and re-entry protection).
2. Add a helper `redis_lock(key, ttl, blocking=False)` in `app/services/redis_utils.py` (from TODO #11).
3. Replace bare `SET NX` patterns with the helper.
4. Verify locks still release on task failure.

**Acceptance:**
- Single locking pattern across the codebase
- All existing lock-protected operations still work

---

### 13. Parameterize vector_search instead of duplicating for image

**Status:** `vector_search` and `image_vector_search` are near-identical.
**Files:** `app/services/arcadedb_graph.py`

**Current state:**
- `vector_search()` queries `TextChunk[text_embedding]`
- `image_vector_search()` queries `ImageChunk[image_embedding]`
- Structurally identical; only the index name differs

**What needs to be done:**
1. Unify into `vector_search(vertex_type, embedding_property, query_vector, top_k, filters)` (which is already the Protocol signature).
2. Update callers: `_text_vector_search` passes `("TextChunk", "text_embedding", ...)`, `_image_vector_search` passes `("ImageChunk", "image_embedding", ...)`.
3. Remove `image_vector_search` from Protocol and implementation.
4. Add convenience wrapper methods if the call sites are verbose.
5. Run retrieval tests to ensure both paths still work.

**Acceptance:**
- Only one vector search method exists
- Text and image search both work via the unified method

---

### 14. Unify RESERVED_WORD_MAP definitions

**Status:** Duplicated between main app and docling-graph service.
**Files:** `app/services/arcadedb_schema.py` (line 12), `docker/docling-graph/app/template_builder.py` (line 12)

**Current state:**
- Both files define `RESERVED_WORD_MAP = {"TABLE": "TABLE_REF"}`
- Both files define `_safe_type_name()` with identical logic
- Risk: one changes, the other doesn't, causing schema/template drift

**What needs to be done:**
1. These are in separate deployables (main app container vs docling-graph container), so direct import isn't possible.
2. Option A: Create a shared ontology contract file (e.g., `ontology/arcadedb_reserved_words.json`) that both services read at startup.
3. Option B: Accept the duplication but add a CI check that verifies the two definitions match (grep both files, compare values).
4. Option C: Pass the reserved word mapping in the extraction request from main app to docling-graph service.
5. Recommend Option A as the cleanest solution.

**Acceptance:**
- No risk of drift between the two definitions
- Tests verify consistency

---

### 15. Replace manual env var helpers with pydantic_settings

**Status:** `docker/docling-graph/app/config_builder.py` hand-rolls env var parsing.
**Files:** `docker/docling-graph/app/config_builder.py`

**Current state:**
- Helpers `_env_str`, `_env_int`, `_env_float`, `_env_bool`, `_env_int_or_none` wrap `os.environ.get`
- The main app (`app/config.py`) uses `pydantic_settings.BaseSettings` for the same purpose
- Duplication of pattern

**What needs to be done:**
1. Add `pydantic_settings>=2.6.0` to `docker/docling-graph/requirements.txt`.
2. Create a `DoclingGraphSettings(BaseSettings)` class with all `DOCLING_GRAPH_*` fields and correct types.
3. Use `model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)`.
4. Replace the `_env_*` helpers with `settings.field_name` access.
5. Add a unit test verifying env var overrides work.

**Acceptance:**
- `config_builder.py` uses pydantic_settings
- All existing env var overrides still work
- Tests pass

---

### 16. Move module-level globals to FastAPI app.state

**Status:** `docker/docling-graph/app/main.py` uses `global` for state.
**Files:** `docker/docling-graph/app/main.py`

**Current state:**
- `_templates`, `_ontology_version`, `_pipeline_version`, `_extraction_semaphore`, `_ontology_cache` are module-level globals
- The `lifespan` context manager uses `global` declarations
- FastAPI's canonical pattern is `app.state.templates`, `app.state.ontology_version`, etc.

**What needs to be done:**
1. Replace each `global _foo` with `app.state.foo`.
2. Update the lifespan function signature to accept `app` and set `app.state.*`.
3. Update the endpoint handlers to read from `request.app.state.*` or capture `app` via closure.
4. Run tests to verify state is still accessible.

**Acceptance:**
- No `global` declarations in main.py
- Tests still pass
- State is accessible via `app.state`

---

### 17. Derive _STRUCTURAL_TYPES from arcadedb_schema

**Status:** Two separate lists of structural types can drift.
**Files:** `app/services/arcadedb_community.py` (line 14), `app/services/arcadedb_schema.py` (line 22)

**Current state:**
- `arcadedb_community.py` hardcodes `_STRUCTURAL_TYPES = {"Document", "TextChunk", ...}`
- `arcadedb_schema.py` has `_STRUCTURAL_VERTEX_TYPES` dict with the same type names as keys
- If schema adds a new structural type, community projection filter won't know about it

**What needs to be done:**
1. In `arcadedb_community.py`, replace the hardcoded set with:
   ```python
   from app.services.arcadedb_schema import _STRUCTURAL_VERTEX_TYPES
   _STRUCTURAL_TYPES = set(_STRUCTURAL_VERTEX_TYPES.keys())
   ```
2. Or better, export a public `STRUCTURAL_TYPES` constant from `arcadedb_schema.py` and import it.
3. Verify the community detection still filters correctly.

**Acceptance:**
- Single source of truth for structural type names
- Adding a new structural type to schema automatically affects community projection

---

### 18. Scope resolve_root_entity queries to specific types

**Status:** Queries all of `V` instead of specific entity type.
**Files:** `app/services/arcadedb_graph.py` (functions `resolve_root_entity`, `resolve_root_entity_sync`)

**Current state:**
- `SELECT *, @rid AS node_id FROM V WHERE name = :name` scans the entire base vertex class
- Bypasses type-specific fulltext indexes
- When the caller knows the entity type (like in `create_entity_chunk_edge_sync`), it should use the specific type

**What needs to be done:**
1. Update signature to accept an optional `entity_type` parameter (already in spec).
2. If `entity_type` is provided, query `SELECT ... FROM {entity_type} WHERE ...` instead of `FROM V`.
3. Update callers to pass the type when known.
4. Keep `V` fallback for generic search across all types.

**Acceptance:**
- Type-scoped queries use type-specific indexes
- Generic queries still work when type is unknown

---

### 19. Pre-compile regex patterns in _build_entity_mentions

**Status:** Re-compiling regex inside a nested loop.
**Files:** `app/workers/pipeline.py` (function `_build_entity_mentions`)

**Current state:**
- For each entity, `re.compile()` is called fresh inside a loop over document elements
- For 50 entities × 200 elements = 10,000 iterations, 10,000 regex compilations

**What needs to be done:**
1. Pre-compile all entity patterns once before the element loop.
2. Store as `dict[entity_name, compiled_pattern]`.
3. Inside the loop, look up the pre-compiled pattern.
4. Benchmark on a typical document.

**Acceptance:**
- Single compilation per entity
- Matching logic unchanged

---

### 20. Collapse pipeline stage section comments

**Status:** Minor — narrating comments remain in some places.
**Files:** `app/services/arcadedb_graph.py`, `app/services/arcadedb_schema.py`

**Current state:**
- Most numbered comments were removed during simplification
- Some explanatory comments (e.g., orphan cleanup explanation in delete_document_graph) remain, which is fine
- Review any new code added to these files to ensure comments explain WHY, not WHAT

**What needs to be done:**
- Pass during the next touch of these files. Not urgent.

---

## P3 — Nice-to-Have

### 21. Add VLM extraction backend option to docling-graph service

**Status:** LLM-only extraction.
**Files:** `docker/docling-graph/app/main.py`, `docker/docling-graph/app/config_builder.py`

**Current state:**
- Spec decision: "VLM support: LLM only, VLM as future enhancement"
- docling-graph library supports VLM backend (NuExtract-2.0) for vision-based extraction
- For visually complex documents (block diagrams, complex tables, spectrum plots), VLM extraction may be more accurate than text-based LLM extraction

**What needs to be done:**
1. Add `DOCLING_GRAPH_BACKEND` env var (values: `llm`, `vlm`).
2. In `config_builder.py`, set `PipelineConfig.backend` accordingly.
3. VLM requires GPU access on the docling-graph container — update `docker-compose.yml` to include GPU reservation for docling-graph service when VLM is enabled.
4. Add a request-level override so specific documents can use VLM while others use LLM.
5. Document the trade-offs in the spec.

**Acceptance:**
- Can switch extraction backend via env var
- VLM extraction works on visually complex documents

---

### 22. Dual-approval for graph mutations

**Status:** Planned in spec, not implemented.
**Files:** `app/api/v1/governance.py`, `app/models/governance.py`

**Current state:**
- Trusted data proposal workflow exists (PROPOSED → APPROVED → INDEXED) but uses single-curator approval
- Spec mentions dual-approval for graph mutations as "Planned"
- No enforcement of requiring two curator sign-offs before applying graph patches

**What needs to be done:**
1. Add `approvals: list[uuid]` column to patch model.
2. Update approval endpoint to append the curator ID instead of setting status.
3. Only transition to APPROVED when `len(approvals) >= 2`.
4. Add a configuration flag: `GOVERNANCE_DUAL_APPROVAL_REQUIRED=true`.
5. Update frontend to show approval progress (1/2 curators).
6. Prevent the same curator from approving twice.

**Acceptance:**
- Patches require two distinct curator approvals before applying
- UI shows approval progress
- Single-curator approval still supported when config disabled

---

### 23. Add ArcadeDB connection pooling

**Status:** Single httpx client per GraphStore instance.
**Files:** `app/services/arcadedb_client.py`

**Current state:**
- `ArcadeDBClient` uses a single `httpx.AsyncClient` and `httpx.Client`
- If `get_graph_store()` creates a new instance per Celery task, each task gets a new TCP connection
- httpx has built-in connection pooling per client, but only if the client is reused

**What needs to be done:**
1. Verify whether `get_graph_store()` returns a singleton or a new instance per call.
2. If new instance: convert to singleton pattern (similar to Redis factory from TODO #11).
3. If singleton: verify the underlying httpx client is reused and has sensible pool settings.
4. Tune `httpx.Limits(max_keepalive_connections, max_connections)` for expected load.
5. Add connection pool metrics to logging.

**Acceptance:**
- TCP connections are reused across tasks
- Pool settings tuned for workload

---

### 24. Add observability for ArcadeDB operations

**Status:** No metrics or tracing on GraphStore calls.
**Files:** `app/services/arcadedb_client.py`, `app/services/arcadedb_graph.py`

**Current state:**
- No Prometheus metrics for query latency, errors, or throughput
- No OpenTelemetry spans for distributed tracing
- Debugging slow queries requires manual log inspection

**What needs to be done:**
1. Add per-method latency histograms.
2. Add error counters by error type (401, 5xx, timeout, etc.).
3. Add a slow query log threshold (configurable, default 1s).
4. Consider OpenTelemetry spans wrapping each ArcadeDB call for distributed tracing.
5. Expose metrics via `/metrics` endpoint.

**Acceptance:**
- Dashboard shows ArcadeDB operation health
- Slow queries are logged for investigation

---

### 25. Add a benchmark suite for retrieval strategies

**Status:** No automated performance regression testing.
**Files:** `tests/benchmarks/` (new directory)

**Current state:**
- No way to detect if a code change slows down retrieval
- Manual testing required to verify performance

**What needs to be done:**
1. Create `tests/benchmarks/test_retrieval_performance.py`.
2. Seed ArcadeDB with a known corpus (1000+ documents).
3. Benchmark each strategy: basic, hybrid, global.
4. Assert latency is within acceptable bounds.
5. Run on CI as a nightly job (not on every PR — too expensive).

**Acceptance:**
- CI catches performance regressions
- Baseline numbers documented

---

### 26. Migrate docling-graph templates.py to use template_builder.py logic

**Status:** Two different template-building implementations in docling-graph service.
**Files:** `docker/docling-graph/app/templates.py`, `docker/docling-graph/app/template_builder.py`

**Current state:**
- `templates.py` exists from before the refactor; uses naive first-field heuristic for graph_id_fields
- `template_builder.py` exists from the refactor; uses the proper priority-based derivation
- `templates.py` is still referenced in tests but the service now uses `template_builder.py`
- Duplication of template-building concept

**What needs to be done:**
1. Verify `templates.py` is still needed (check if any code imports it).
2. If not needed, delete it and its test file.
3. If needed, update it to import from or delegate to `template_builder.py`.
4. Consolidate the test files.

**Acceptance:**
- Single source of truth for template building in docling-graph service
- All tests still pass

---

### 27. LLM-based entity mention resolution

**Status:** Not started. **Blocked by #45-#49** — not actionable until the worker/service contract is repaired and traversal is fixed.
**Files:** `app/workers/pipeline.py` (`_build_entity_mentions`), new module TBD

**Current state:**
- `_build_entity_mentions` uses regex/substring matching to link extracted entities to document chunks.
- This catches exact text matches but misses: paraphrases ("the fire control radar" → APG-77), abbreviations ("FCR" → APG-77), misspellings, coreferences ("the system" → S-400).

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

### 28. Use native vectorRRFScore / vectorHybridScore for multi-vector fusion

**Status:** Not started.
**Files:** `app/api/v1/retrieval.py` (`_multi_modal_pipeline`, `compute_fusion_score`), `app/services/arcadedb_graph.py` (`cross_model_search`)

**Current state:** Multi-vector fusion (text + image) is done Python-side via custom `compute_fusion_score()`. ArcadeDB provides built-in `vectorRRFScore()`, `vectorHybridScore()`, `vectorMultiScore()`, and `vectorNormalizeScores()` SQL functions that do this server-side.

**What needs to be done:** Replace Python-side fusion with a single ArcadeDB SQL query using `vectorRRFScore()` or `vectorHybridScore()`. Eliminates transferring two separate result sets and moves computation to the DB engine.

**Reference:** ArcadeDB Manual Section 6.3.1 (Overview Functions table)

---

### 29. Use MATCH syntax for graph traversal queries

**Status:** Not started.
**Files:** `app/services/arcadedb_graph.py` (`get_neighborhood`, `get_neighborhood_graph`)

**Current state:** Neighborhood queries use `SELECT expand(both(){1,N}) FROM #rid`. ArcadeDB manual recommends MATCH for graph traversals: "If you are looking for the most efficient way to traverse a graph, we suggest using MATCH instead."

**What needs to be done:** Rewrite to `MATCH {class: V, as: root, where: (@rid = :rid)}.both(){1,:depth} RETURN ...` for cleaner pattern matching and better query plan optimization.

**Reference:** ArcadeDB Manual Section 6.3.1

---

### 30. BucketSelectionStrategy 'thread' for write-heavy types

**Status:** Not started.
**Files:** `app/services/arcadedb_schema.py`

**Current state:** Default bucket selection causes write contention on parallel pipeline ingestion for TextChunk, ImageChunk, and entity types.

**What needs to be done:** Add `ALTER TYPE TextChunk BucketSelectionStrategy 'thread'` (and similarly for ImageChunk and high-write entity types) as a post-schema-sync step to eliminate contention and ConcurrentModificationException retries.

**Reference:** ArcadeDB Manual Section 5.5.24 (Troubleshooting: "Performance: insertion is slow")

---

### 31. Enable ArcadeDB Prometheus metrics plugin

**Status:** Not started.
**Files:** `docker-compose.yml`

**Current state:** No database-level metrics are exposed despite having observability infrastructure. ArcadeDB has a built-in `PrometheusMetricsPlugin` exposing cache hits, transaction stats, and query throughput at `/metrics`.

**What needs to be done:** Add `arcadedb.server.plugins=Prometheus:com.arcadedb.metrics.prometheus.PrometheusMetricsPlugin` to JAVA_OPTS in docker-compose.yml.

**Reference:** ArcadeDB Manual Section 5.5.23 (Monitoring)

---

### 32. Configure automatic backup scheduler

**Status:** Not started.
**Files:** `docker/arcadedb/backup.json` (new), `docker-compose.yml`

**Current state:** No automated backups configured. ArcadeDB has a built-in automatic backup scheduler with tiered retention, cron scheduling, and time windows.

**What needs to be done:** Create a `backup.json` configuration and mount it into the ArcadeDB container. Configure cron schedule, retention, and backup directory.

**Reference:** ArcadeDB Manual Section 5.5.10

---

### 33. Use text.levenshteinDistance() for fuzzy entity matching

**Status:** Not started.
**Files:** `app/services/arcadedb_graph.py`, canonicalization service

**Current state:** Entity canonicalization pulls entity names to Python for fuzzy comparison. ArcadeDB provides a built-in `text.levenshteinDistance()` SQL function for server-side fuzzy matching.

**What needs to be done:** Use `text.levenshteinDistance()` in canonicalization queries to do fuzzy matching server-side instead of pulling data to Python.

**Reference:** ArcadeDB Manual Section 6.3.1 (Extended Functions)

---

### 34. Add EXPLAIN/PROFILE tooling for query plan validation

**Status:** Not started.
**Files:** tests or CI tooling

**Current state:** No tooling verifies that critical queries hit indexes rather than scanning. ArcadeDB provides `EXPLAIN` and `PROFILE` commands for query plan inspection.

**What needs to be done:** Add a health-check or CI step that runs `EXPLAIN` on critical query paths (vector search, fulltext search, RID lookups) and asserts no full scans.

**Reference:** ArcadeDB Manual Section 5.5.24 (Query Optimization)

---

## P0 — Code Review Findings (2026-04-05)

The following items were identified by a standalone code analysis and review.
They are ordered by severity (Critical, High, Medium). Each must be addressed
before the feature/arcadedb branch can be considered production-ready.

### 35. Fix upstream repo clone/update contract for Docling and Docling-Graph

**Status:** Not started. **Severity:** Critical.
**Files:** `manage.sh`, `docker-compose.yml`, `docker/docling/Dockerfile`, `docker/docling-graph/Dockerfile`, `docker/docling/requirements.txt`, `docker/docling-graph/requirements.txt`, `VERIFICATION_CHECKLIST.md`

Only ArcadeDB consumes a cloned `repo/`; Docling and Docling-Graph install released PyPI packages. A plain `docker compose build` clones nothing, and `manage.sh` continues on `git pull` failure with only a warning. This conflicts with the stated verification/design contract.

**What needs to be done:** Either (a) make the Docling/Docling-Graph Dockerfiles actually consume the cloned repos via `pip install -e /repo` or `COPY repo/ ...`, or (b) update the verification checklist and design spec to reflect the real contract (PyPI packages, not git-main tracking).

---

### 36. Fix vector_search() dropping chunk metadata that retrieval depends on

**Status:** Not started. **Severity:** Critical.
**Files:** `app/services/arcadedb_graph.py` (line ~858), `app/api/v1/retrieval.py` (lines ~449, ~523)

The API expects `chunk_id`, `document_id`, `artifact_id`, `modality`, and chunk text in `hit.properties`, but the ArcadeDB query only projects entity fields plus distance/RID. The ArcadeDB manual says `expand(vectorNeighbors(...))` returns all document properties; the current SELECT projection throws those away.

**What needs to be done:** Add `chunk_id`, `document_id`, `artifact_id`, `modality`, `text`, `chunk_text` to the vector_search SELECT projection, or revert to `SELECT *` for TextChunk/ImageChunk vector searches specifically.

---

### 37. Fix ontology/evidence traversal edge direction and identifier mismatch

**Status:** Not started. **Severity:** Critical. **Superseded by #49 which has full scope.**
**Files:** `app/services/arcadedb_graph.py` (line ~730, ~1453), `app/workers/pipeline.py` (line ~2742), `app/api/v1/retrieval.py` (lines ~314, ~326), `app/services/query_profiles.py` (line ~659), `app/services/dossier_service.py` (line ~352)

EXTRACTED_FROM edges are written entity->chunk, but `get_ontology_linked_chunks` traverses `in('EXTRACTED_FROM')` which goes the wrong direction. Retrieval also passes `str(seed.chunk_id)` into `get_ontology_linked_chunks`, but the helper uses `FROM {node_id}` which expects an ArcadeDB RID, not a UUID string.

**What needs to be done:** See #49 for full scope and execution context. Fix traversal direction and identifier type across all callers.

---

### 38. Fix alias resolution property name inconsistency

**Status:** Not started. **Severity:** High.
**Files:** `app/services/arcadedb_schema.py` (line ~60), `app/services/arcadedb_graph.py` (lines ~786, ~813, ~1603), `app/services/query_profiles.py` (line ~519), `app/services/dossier_service.py` (line ~166), `app/services/canonicalization.py` (line ~129)

Alias schema/creation uses `alias_name`, but lookup queries use `WHERE alias = :alias`. The optional `entity_type` filter is also applied on the Alias vertex query itself (not the linked entity). This breaks root resolution and canonicalization.

**What needs to be done:** Standardize on one property name (`alias_name` per schema) in all queries. Move `entity_type` filter to the linked entity (via the HAS_ALIAS edge traversal), not the Alias vertex.

---

### 39. Align docling-graph wrapper with canonical template/pipeline API

**Status:** Not started. **Severity:** High.
**Files:** `docker/docling-graph/app/main.py` (lines ~56, ~93), `docker/docling-graph/app/template_builder.py` (lines ~101, ~218)

The library expects a singular `PipelineConfig.template`; this service builds many templates and passes only the first one. The generated models also use `graph_id_fields` but not the documented `is_entity=True`/`edge()` pattern. This is a real misalignment with docling-graph's canonical API surface.

**What needs to be done:** Study the canonical docling-graph template API, restructure template_builder to produce a single template conforming to the `is_entity=True`/`edge()` pattern, and pass it correctly to `PipelineConfig.template`.

---

### 40. Wire retrieval `filters` (classification, document_id, modality constraints)

**Status:** Not started. **Severity:** High.
**Files:** `app/schemas/retrieval.py` (lines ~37, ~64), `app/api/v1/retrieval.py`

`body.filters` is part of the public API schema but is silently ignored — callers can request classification/document/modality constraints and get unfiltered results with no warning.

**What needs to be done:** Read `body.filters` in the retrieval handlers and apply classification, document_ids, and modalities constraints to the ArcadeDB queries (WHERE clause additions or post-filter).

---

### 41. Fix document canonicalization scope

**Status:** Not started. **Severity:** Medium. **Superseded by #50 which has full scope.**
**Files:** `app/services/canonicalization.py` (line ~68), `app/services/arcadedb_graph.py` (line ~1582)

The code claims to find entities linked to a document, but does `WHERE name LUCENE :query` using the document ID string instead of traversing Document->chunk->entity edges. This makes the canonicalization pass logically disconnected from actual document/entity linkage.

**What needs to be done:** See #50 for full scope and execution context. Replace LUCENE search with graph traversal after #49 fixes traversal direction.

---

### 42. Fix orphan cleanup system field (@cat vs @class)

**Status:** Not started. **Severity:** Medium.
**Files:** `app/services/arcadedb_graph.py` (lines ~1107, ~1527)

The ArcadeDB manual defines `@class` as the type name and `@cat` as the type category. The orphan cleanup query filters `@cat NOT IN ['Document', 'TextChunk', ...]` which targets the wrong field.

**What needs to be done:** Change `@cat` to `@class` in the orphan cleanup WHERE clause.

---

### 43. Add $distance projection to community-report vector search

**Status:** Not started. **Severity:** Medium.
**Files:** `app/services/arcadedb_graph.py` (line ~1035), `app/api/v1/retrieval.py` (line ~993)

The community-report vector search query does not project `$distance`, but global retrieval assumes `score` exists. Synthesis metadata collapses to `0.0`.

**What needs to be done:** Add `$distance` to the SELECT projection in `search_community_reports_by_vector`, and map it to a `score` key in the returned dict.

---

### 44. Fix stale test harness for docling-graph integration tests

**Status:** Not started. **Severity:** Medium.
**Files:** `docker/docling-graph/tests/test_pipeline_integration.py` (line ~43), `tests/conftest.py` (line ~212)

The test patches `_templates` which no longer exists (moved to `app.state`). The shared conftest globally stubs GraphStore, so passing backend/query-profile tests do not validate concrete ArcadeDB behavior.

**What needs to be done:** (a) Update `test_pipeline_integration.py` to patch `app.state.templates` instead of the deleted module-level global. (b) Consider adding a separate integration test marker that uses the real ArcadeDBGraphStore (not the mock) for critical path validation.

---

## P0 — Graph Extraction Pipeline Repair (2026-04-06)

The following items were identified by a deep review of the graph extraction
pipeline against the current Docling-Graph service contract. They must be
addressed in order — later items depend on earlier ones being fixed first.
The architecture (parallel extract+chunk → wire → canonicalize) is sound;
the issue is contract drift between the worker, the service, and downstream
consumers.

**Execution order: #45 → #46 → #47 → #48 → #49 → #50 → then #27 becomes meaningful.**

### 45. Reconcile Stage 1 worker/service request+response contract

**Status:** Not started. **Severity:** Critical. **Do first.**
**Files:** `app/workers/pipeline.py` (lines ~2226, ~2257-2258), `app/services/docling_graph_service.py` (line ~172), `docker/docling-graph/app/main.py` (line ~141), `docker/docling-graph/app/schemas.py` (line ~10)

**Problem:** The worker sends `text` via `extract_graph_all(full_text, document_id)`, but the Docling-Graph service now expects `docling_document_json` and returns `{graph, metadata}`, not `{entities, relationships}`. The worker reads `result.get("entities")` and `result.get("relationships")` which will be empty/missing from the new response shape. If this code path is live, successful extraction is being interpreted as zero nodes and zero edges.

**What needs to be done:**
1. Read the current Docling-Graph service `/extract-all` endpoint schema to determine what it actually accepts and returns.
2. Update `docling_graph_service.py` client to send the correct request shape.
3. Update `derive_ontology_graph()` to parse the actual response shape.
4. Add an adapter/normalizer if the response structure differs from the `{nodes: [...], edges: [...]}` format the rest of the pipeline expects.
5. Update `tests/unit/test_docling_graph_client.py` which still asserts the old text-based interface.

**Why this is first:** Nothing downstream works if the extraction contract is broken. All discussion about mention precision, traversal direction, and canonicalization quality is second-order until this is fixed.

---

### 46. Make derive_ontology_graph() consume persisted DoclingDocument JSON

**Status:** Not started. **Severity:** Critical. **Do second.**
**Files:** `app/workers/pipeline.py` (lines ~2218-2226, ~915, ~1401)

**Problem:** `prepare_document` already persists `docling_document.json` to MinIO at line ~915. Another stage downloads it at line ~1401. But `derive_ontology_graph()` ignores the persisted artifact and reconstructs plain text from normalized elements. This throws away layout, structure, table boundaries, image context, and native provenance information before the extraction service even runs.

**What needs to be done:**
1. Download the persisted `docling_document.json` from MinIO (same pattern as the stage at line ~1401).
2. Pass it to the Docling-Graph service as `docling_document_json` instead of reconstructed `full_text`.
3. Keep the `full_text` fallback for documents that don't have a persisted DoclingDocument (backward compat).
4. The document summary/classification prefix can still be passed as metadata context.

**Why this matters:** The structured DoclingDocument preserves element boundaries, page numbers, table structure, and image positions that the LLM can use for more accurate extraction. Plain text concatenation loses all of this.

---

### 47. Normalize extraction output shape before graph import

**Status:** Not started. **Severity:** High. **Do third.**
**Files:** `app/workers/pipeline.py` (lines ~2275-2343)

**Problem:** The worker assumes the extraction result has `{entities: [...], relationships: [...]}` shape with specific field names (`name`, `entity_type`, `confidence`, `from_name`, `to_name`, etc.). If the Docling-Graph service returns a different shape (e.g., `{graph: {nodes: [...], edges: [...]}, metadata: {...}}`), the import loop produces zero records silently.

**What needs to be done:**
1. After fixing #45, define the actual response shape from Docling-Graph.
2. Write a thin adapter function that normalizes the response into the `{nodes: [...], edges: [...]}` format the rest of the pipeline expects.
3. Map any field-name differences (e.g., `confidence` vs `extraction_confidence`, `id` vs `name`).
4. Preserve any provenance metadata (element_uid, page_number) that the service includes in its output.
5. Unit test the adapter with sample Docling-Graph output.

---

### 48. Fix entity→chunk mention wiring to be element-complete

**Status:** Not started. **Severity:** High. **Do fourth.**
**Files:** `app/workers/pipeline.py` (lines ~2672-2705, ~2116-2179)

**Problems (three related issues):**

**(a) Image chunks excluded from mention map.** The `element_uid → chunk_id` map (line ~2672) is built only from `text_chunks` via `artifact_id`. Image chunks are wired to the document and same-page neighbors, but are not in the mention map used for EXTRACTED_FROM. So entities grounded in image/schematic elements are never linked to their image chunks.

**(b) Partial mention miss is never repaired.** The fallback path (line ~2708) only fires when `mentions` is completely empty, not when it's incomplete. If lexical matching finds 2 mentions and misses 3 implicit ones, the stage keeps the 2 and drops the 3. The fallback doesn't help recall unless the primary path fails entirely.

**(c) Lexical matching is the only mention grounding.** `_build_entity_mentions()` uses regex/substring only. No semantic resolution of paraphrases, abbreviations, coreferences, or metonymic references. (#27 addresses this but is not actionable until #45-#47 are stable.)

**What needs to be done:**
1. Include `image_chunks` in the `element_uid → chunk_id` map alongside text_chunks.
2. After the primary mentions path, run the fallback for any entities that have zero mentions (not just when the whole mentions list is empty).
3. If the Docling-Graph extraction output includes its own provenance/element mapping (determined after fixing #45), consume that directly instead of running lexical matching.
4. Defer #27 (LLM mention resolution) until the contract is stable.

---

### 49. Fix EXTRACTED_FROM traversal direction and identifier type

**Status:** Not started. **Severity:** Critical. **Do fifth (or alongside #37).**
**Files:** `app/services/arcadedb_graph.py` (lines ~724-730, ~1453), `app/api/v1/retrieval.py` (lines ~314, ~326, ~696), `app/services/query_profiles.py` (line ~659), `app/services/dossier_service.py` (line ~352)

**Note:** This overlaps with existing #37. Consolidating the full scope here.

**Problem (two related bugs):**
1. **Edge direction:** EXTRACTED_FROM edges are created as entity→chunk, but `get_ontology_linked_chunks` traverses `in('EXTRACTED_FROM')` from a node. If the node is a chunk, `in()` gives you entities that point TO this chunk — that's correct for "find entities extracted from this chunk." But if the node is an entity, `in()` gives you nothing (entities are the source, not the target). The callers in retrieval.py pass chunk-level seeds, so the direction may be situationally correct, but needs verification against each call site.
2. **Identifier type:** Retrieval passes `str(seed.chunk_id)` (a UUID string like `"a1b2c3d4-..."`) into `get_ontology_linked_chunks`, but the helper does `FROM {node_id}` which expects an ArcadeDB RID like `#10:5`. A UUID string in the FROM clause will cause a SQL error or return nothing.

**What needs to be done:**
1. Audit every caller of `get_ontology_linked_chunks` to determine whether they pass RIDs or UUIDs.
2. If callers pass UUIDs: add a RID lookup step (`get_chunk_rid_sync`) before calling the traversal helper.
3. Verify the traversal direction is correct for each call site's intent (chunk→entities vs entity→chunks).
4. Add a test that exercises the real traversal path (not just mocked calls).

---

### 50. Fix canonicalization to use graph traversal for document-entity discovery

**Status:** Not started. **Severity:** Medium. **Do after traversal is fixed.**
**Files:** `app/services/canonicalization.py` (line ~64-68), `app/services/arcadedb_graph.py` (line ~1568-1582)

**Note:** This overlaps with existing #41. Consolidating the full scope here.

**Problem:** `canonicalize_document_entities()` discovers "document entities" by calling `fulltext_search_sync(document_id)`, which does `WHERE name LUCENE :query` using the document ID string. This searches for entities whose *name* matches the document ID — which is nonsensical. The intended behavior is to find all entities linked to a specific document via the graph.

**What needs to be done:**
1. Replace the LUCENE search with a graph traversal: `SELECT DISTINCT in('EXTRACTED_FROM').@class, in('EXTRACTED_FROM').name FROM (SELECT expand(out('CONTAINS_TEXT')) FROM Document WHERE document_id = :doc_id)` (or equivalent).
2. This depends on #49 being fixed first (traversal direction must be correct).
3. Also fix the alias property name mismatch (existing #38) since canonicalization calls `search_by_alias`.

---

## Verbatim Graph Extraction Pipeline Review (2026-04-06)

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
> derive_structure_links() only falls back to artifact-wide entity→chunk linking when mentions is empty, not when mentions are incomplete, at pipeline.py (line 2696) and pipeline.py (line 2708). That means if lexical matching finds 2 real mentions and misses 3 implicit ones, the stage keeps the 2 and silently drops the 3. The fallback does not help recall unless the primary path fails completely.
>
> **The entity→chunk wiring is text-centric, not fully element-centric.**
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
> Otherwise better entity→chunk edges will still not show up properly in retrieval, dossier evidence, or ontology expansion.
>
> **Test Gaps**
>
> The client tests still assert the old text-based interface in test_docling_graph_client.py (line 119).
> The docling-graph integration tests are stale and patch a removed _templates global at test_pipeline_integration.py (line 43).
> There is no strong end-to-end test that proves: DoclingDocument JSON -> Docling-Graph extraction -> graph_json mentions -> EXTRACTED_FROM edges -> retrieval traversal all agree on the same identifiers and schema.
>
> My bottom-line assessment is: the architecture is defensible, but the current graph extraction process is not trustworthy until the worker/service contract is repaired. After that, #27 becomes a meaningful optimization target; before that, it is not the primary blocker.

---

## Verbatim Code Analysis and Review (2026-04-05)

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

---

## Completed During Migration (For Reference)

These items were identified during review but fixed before merging:

- ✅ `set_vertex_embedding` signature mismatch between Protocol and implementation
- ✅ Protocol encapsulation violations (direct `_client`/`_database` access) — added proper methods
- ✅ `upsert_relationships_batch_sync` dropped `record.properties` (copy-paste bug)
- ✅ Redis client leak in `community_tasks.py` (no `r.close()`)
- ✅ Direct `os.environ.get` bypassing settings in community module
- ✅ `cross_model_search` and `get_graph_stats` sequential queries → parallelized with `asyncio.gather`
- ✅ Stale references (GraphRAG, Neo4j, Qdrant, ChunkRef, AGE) across 8 files
- ✅ Narrating step-number comments in `delete_document_graph` and `sync_schema_from_ontology`
- ✅ `delete_document_graph` now removes document_id from relationship edge document_ids lists and deletes empty-list edges
