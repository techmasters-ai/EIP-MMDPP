# Docling-Graph Pipeline Refactor Design

**Date:** 2026-04-04
**Status:** Approved
**Scope:** Refactor docling-graph service to use library pipeline, delta extraction, proper templates, and NetworkX graph output. Eliminate Qdrant by consolidating vector search into ArcadeDB.

## Overview

The current docling-graph FastAPI service (`docker/docling-graph/`) bypasses the docling-graph library's extraction pipeline entirely. It makes raw LiteLLM calls, does character-based chunking, and returns flat JSON. This refactor replaces the hand-rolled extraction with proper `run_pipeline()` / `PipelineConfig` integration, using delta extraction with direct fallback, edge-annotated Pydantic templates generated from the ontology, and structure-aware chunking from DoclingDocument input.

Additionally, Qdrant is eliminated. ArcadeDB's native LSMVectorIndex (HNSW/Vamana with COSINE similarity, INT8 quantization) replaces Qdrant for all vector search. Text and image embeddings become properties on ArcadeDB vertices with vector indexes.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Return format | NetworkX graph JSON | Native pipeline output, stable node IDs, built-in dedup |
| Docling integration | Pass DoclingDocument JSON (skip re-conversion) | Structure-aware chunking requires DoclingDocument, not flat text |
| Extraction contract | Delta with direct fallback | Highest accuracy for long docs + graph-centric ontology. Quality gate auto-falls back to direct |
| Relationship extraction | Edge fields on templates + validation pass | Single-pass entity+relationship extraction via delta. Validation pass catches missed edges |
| VLM support | LLM only, VLM as future enhancement | Current VLM coverage (Docling conversion + image descriptions) is sufficient |
| Vector search | ArcadeDB native (LSMVectorIndex) | Eliminates Qdrant, removes data duplication, enables cross-model graph+vector queries |
| Cross-model queries | Enhance hybrid strategy internally | Same API, same response contract, better internals via native ArcadeDB cross-model SQL |

## Section 1: Service Architecture

The refactored service becomes a thin FastAPI wrapper around `run_pipeline()`.

### File structure

```
docker/docling-graph/app/
  main.py             -- FastAPI wrapper (~200 lines, down from 922)
  template_builder.py  -- Ontology YAML -> Pydantic templates with edge() fields
  config_builder.py    -- Env vars -> PipelineConfig construction
  schemas.py           -- Request/response models (revised)
```

### Flow

1. Startup: Load ontology YAML -> `template_builder.build_templates()` -> Pydantic templates with `graph_id_fields`, `edge()`, typed properties
2. Request: Receive DoclingDocument JSON + ontology definition
3. Build config: `PipelineConfig` with delta extraction, Ollama backend, structure-aware chunking
4. Run pipeline: `context = run_pipeline(config)` -> PipelineContext with NetworkX graph
5. Validation pass: Lighter LLM call on full text with extracted entities as context (catches missed edges)
6. Return: Serialized NetworkX graph JSON + metadata

### What gets removed from main.py

- `_llm_call()` -- replaced by pipeline's internal LLM client
- `_chunk_text()` -- replaced by DocumentChunker with real tokenizer
- `_parse_json_from_llm()` -- replaced by pipeline's structured output + validation
- `_extract_entities_for_group()` -- replaced by delta extraction
- `_extract_relationships()` -- replaced by edge() fields + validation pass
- `_dedup_entities()` -- replaced by delta's graph merge + resolvers
- `_run_full_extraction()` -- replaced by `run_pipeline()`
- All GROUP_PROMPTS, GROUP_FEW_SHOT_EXAMPLES -- domain knowledge moves into template field descriptions

### Scalability

Within a single instance:
- `DOCLING_GRAPH_PARALLEL_WORKERS` controls concurrent delta batch LLM calls (ThreadPoolExecutor)
- `DOCLING_GRAPH_MAX_CONCURRENT_EXTRACTIONS` limits per-instance concurrency via semaphore

Across instances:
- Service is stateless. Scale horizontally via Docker Compose replicas or Kubernetes.
- Upstream Redis concurrency gate (`docling_graph_concurrency`) controls how many parallel extractions the pipeline sends.

## Section 2: Template Builder

`template_builder.py` reads the ontology YAML and generates proper docling-graph Pydantic templates.

### Mapping rules

| Ontology concept | Docling-graph template concept |
|---|---|
| Entity type | Pydantic model with `graph_id_fields` derived per type (see rules below) |
| Entity properties | Typed fields with `Field(description=..., examples=[...])` |
| Relationship (from validation_matrix) | `edge()` field on source entity template |
| Property enums | `Literal` types or string with description listing valid values |
| Cardinality one_to_one | `Optional[TargetType] = edge(label=...)` |
| Cardinality one_to_many / many_to_many | `List[TargetType] = edge(label=..., default_factory=list)` |

### graph_id_fields derivation rules

Each entity type gets `graph_id_fields` derived from its ontology properties, NOT a universal default:

1. If properties include `name` or `system_name` -> use that (e.g., RADAR_SYSTEM, PLATFORM, COMPONENT)
2. If properties include a field with `_id` suffix -> use that (e.g., DOCUMENT -> `document_id`, FIGURE -> `figure_id`, TABLE -> `table_id`)
3. If properties include `title` or `heading` -> use that (e.g., SECTION -> `heading`)
4. For types with composite identity -> use multiple fields (e.g., SPECIFICATION -> `["parameter", "value"]`)
5. Fallback -> first required property, or first property if none required

Examples:
- RADAR_SYSTEM: `graph_id_fields=["name"]` (has `name` in properties, mapped from system_name)
- DOCUMENT: `graph_id_fields=["document_id"]`
- FIGURE: `graph_id_fields=["figure_id"]`
- TABLE: `graph_id_fields=["table_id"]`
- SECTION: `graph_id_fields=["heading"]`
- SPECIFICATION: `graph_id_fields=["parameter", "value"]`
- ASSERTION: `graph_id_fields=["assertion_text"]`

### Generation rules

1. All fields Optional except the identity field(s) -- extraction may not find every property
2. `graph_id_fields` derived per entity type (see rules above) -- stable dedup by identity
3. Edge fields generated from validation_matrix: for each row where source=THIS_TYPE, create an edge field pointing to the target type
4. Field descriptions from ontology property descriptions -- guide the LLM during extraction
5. Field examples from ontology property examples -- help the LLM produce correct formats
6. Reserved word handling: TABLE -> TABLE_REF mapping applied transparently
7. Templates rebuilt at startup and when per-request ontology_definition provided (cached by ontology hash)
8. Domain-specific extraction heuristics from current prompts.py preserved via: (a) field descriptions with extraction hints (e.g., "Look in Methods section"), (b) edge() fields encoding valid relationships from validation_matrix, (c) validation pass for heuristics that cannot be expressed as template structure (e.g., SPECIFICATION->SPECIFIED_BY enforcement)

### Library version requirements

The service does not pin a specific docling-graph version (manage.sh --start pulls latest from the GitHub repo). Instead, it validates the required API surface at startup and fails fast with a clear error if missing. Required surface: run_pipeline(), PipelineConfig with extraction_contract="delta", delta resolvers, delta normalizer, gleaning, structured output, edge() helper, graph_id_fields support.

## Section 3: Pipeline Configuration Builder

`config_builder.py` reads all environment variables and constructs a `PipelineConfig` for each request.

### Environment variables -- complete list

**Scaling and concurrency:**
- `DOCLING_GRAPH_PARALLEL_WORKERS=2` -- delta batch LLM call parallelism
- `DOCLING_GRAPH_MAX_CONCURRENT_EXTRACTIONS=2` -- max concurrent extractions per instance

**Delta extraction:**
- `DOCLING_GRAPH_EXTRACTION_CONTRACT=delta` -- direct | staged | delta
- `DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE=2048` -- tokens per delta batch
- `DOCLING_GRAPH_BATCH_SPLIT_MAX_RETRIES=1` -- retries on failed batches

**Chunking:**
- `DOCLING_GRAPH_USE_CHUNKING=true` -- enable structure-aware chunking
- `DOCLING_GRAPH_CHUNK_MAX_TOKENS=512` -- max tokens per chunk

**Gleaning:**
- `DOCLING_GRAPH_GLEANING_ENABLED=true` -- second-pass extraction for recall
- `DOCLING_GRAPH_GLEANING_MAX_PASSES=1` -- number of gleaning passes

**Delta resolvers:**
- `DOCLING_GRAPH_RESOLVERS_ENABLED=true` -- fuzzy/semantic entity dedup
- `DOCLING_GRAPH_RESOLVERS_MODE=semantic` -- off | fuzzy | semantic | chain
- `DOCLING_GRAPH_RESOLVER_FUZZY_THRESHOLD=0.8`
- `DOCLING_GRAPH_RESOLVER_SEMANTIC_THRESHOLD=0.8`

**Delta quality gate:**
- `DOCLING_GRAPH_QUALITY_REQUIRE_ROOT=true`
- `DOCLING_GRAPH_QUALITY_MIN_INSTANCES=20` -- min attached nodes before fallback
- `DOCLING_GRAPH_QUALITY_MAX_PARENT_MISS=4`
- `DOCLING_GRAPH_QUALITY_ADAPTIVE_PARENT=true`

**Delta normalizer:**
- `DOCLING_GRAPH_NORMALIZER_VALIDATE_PATHS=true`
- `DOCLING_GRAPH_NORMALIZER_CANONICALIZE_IDS=true`
- `DOCLING_GRAPH_NORMALIZER_STRIP_NESTED=true`
- `DOCLING_GRAPH_NORMALIZER_ATTACH_PROVENANCE=true`

**Delta identity filter:**
- `DOCLING_GRAPH_IDENTITY_FILTER_ENABLED=true`
- `DOCLING_GRAPH_IDENTITY_FILTER_STRICT=false`

**Structured output:**
- `DOCLING_GRAPH_STRUCTURED_OUTPUT=true`
- `DOCLING_GRAPH_STRUCTURED_SPARSE_CHECK=true`

**Validation pass:**
- `DOCLING_GRAPH_VALIDATION_PASS_ENABLED=true`
- `DOCLING_GRAPH_VALIDATION_LLM_MODEL=gpt-oss:120b`

**Processing mode:**
- `DOCLING_GRAPH_PROCESSING_MODE=many-to-one`
- `DOCLING_GRAPH_DOCLING_CONFIG=ocr`

**LLM configuration:**
- `DOCLING_GRAPH_LLM_PROVIDER=ollama`
- `DOCLING_GRAPH_LLM_MODEL=granite3-dense:8b`
- `DOCLING_GRAPH_LLM_TIMEOUT=10800`
- `DOCLING_GRAPH_LLM_MAX_TOKENS=64000`
- `DOCLING_GRAPH_LLM_TEMPERATURE=0.1`
- `DOCLING_GRAPH_LLM_CONTEXT_LIMIT=` (optional override)
- `DOCLING_GRAPH_LLM_MAX_OUTPUT_TOKENS=` (optional override)

**Ollama server endpoints (3 separate servers):**
- `OLLAMA_LLM_BASE_URL=http://ollama:11434` -- LLM inference
- `OLLAMA_VLM_BASE_URL=http://ollama:11434` -- VLM inference
- `OLLAMA_EMBEDDING_BASE_URL=http://ollama:11434` -- Embedding inference
- `OLLAMA_NUM_CTX=16384`
- `OLLAMA_THINK=` -- thinking effort (low|medium|high)

**Model names per functionality:**
- `DOCLING_VLM_MODEL=granite-docling-258M` -- Docling document conversion
- `DOCLING_GRAPH_LLM_MODEL=granite3-dense:8b` -- Graph entity/relationship extraction
- `DOCLING_GRAPH_VALIDATION_LLM_MODEL=gpt-oss:120b` -- Relationship validation pass
- `IMAGE_DESCRIPTION_VLM_MODEL=llama3.2` -- Image description generation
- `DOCUMENT_METADATA_LLM_MODEL=llama3.2` -- Document metadata extraction
- `TRANSLATION_LLM_MODEL=llama3.2` -- Language detection and translation
- `COMMUNITY_REPORT_LLM_MODEL=llama3.2` -- Community report generation
- `TEXT_EMBEDDING_MODEL=bge-m3:latest` -- Text embeddings
- `IMAGE_EMBEDDING_MODEL=ViT-B-32` -- Image embeddings
- `IMAGE_EMBEDDING_PRETRAINED=openai` -- Image embedding weights
- `RERANKER_MODEL=BAAI/bge-reranker-v2-m3` -- Cross-encoder reranking

## Section 4: Request/Response Contract

### New contract

```
POST /extract-all
Request: {
    document_id: str,
    docling_document_json: dict,       -- Full DoclingDocument (replaces flat text)
    ontology_definition: dict | None,
    ontology_version: str | None
}
Response: {
    graph: dict,                        -- Serialized NetworkX graph (node-link JSON)
    metadata: {
        node_count, edge_count, node_types, edge_types,
        extraction_contract, gleaning_passes, resolvers_applied,
        quality_gate_passed, validation_pass_applied, validation_pass_edges_added
    },
    model: str,
    provider: str,
    ontology_version: str | None
}
```

### Graph serialization

NetworkX graph serialized via `node_link_data()`. Each node carries type, identity fields, properties, and `_provenance` (batch_id, chunk_index, page_numbers). Each edge carries source, target, label, confidence, and `_provenance`.

### Provenance granularity

Delta provenance provides `batch_id` and `chunk_index` per node. The pipeline maps `chunk_index` to DocumentElement `element_uid` values (via chunker boundary metadata from the DoclingDocument), then to `chunk_id` values (via the existing derive_structure_links element-to-chunk mapping). This preserves element-level EXTRACTED_FROM edge granularity -- entities link to specific chunks, not every chunk on a page. Page numbers are a secondary attribute on the provenance, not the primary linking key.

### Adapter layer for graph_json persistence

The pipeline task `derive_ontology_graph` transforms the NetworkX node-link response into the existing `DocumentGraphExtraction.graph_json` contract (`{nodes, edges, mentions, _ingest_filter}`) before PostgreSQL storage. Downstream consumers of graph_json see the same structure as today. No rewrite of audit/test consumers needed.

### Key differences from current contract

- Input: DoclingDocument JSON instead of flat text
- Output: NetworkX graph instead of flat entity/relationship lists
- Dedup: Built-in via graph_id_fields + delta resolvers
- Provenance: Per-node chunk_index mapped to element_uid then chunk_id (element-level granularity)
- Relationships: Embedded in templates via edge() + validation pass
- Node IDs: Stable from docling-graph NodeIDRegistry

### Backward compatibility

The `/extract` single-group endpoint is removed. All extraction goes through `/extract-all`.

## Section 5: Pipeline Task Changes

### derive_ontology_graph -- new flow

1. Retrieve DoclingDocument JSON from MinIO (already persisted by prepare_document in eip-derived bucket)
2. POST to docling-graph `/extract-all` with DoclingDocument JSON
3. Receive NetworkX graph JSON with provenance
4. Apply confidence quality gates (final filter)
5. Import nodes and edges into ArcadeDB via GraphStore
6. Build EXTRACTED_FROM edges from provenance chunk_index -> element_uid -> chunk_id mapping
7. Transform NetworkX response to graph_json contract via adapter layer
8. Store graph_json in PostgreSQL audit trail (DocumentGraphExtraction)

### prepare_document -- no change needed

`prepare_document` already uploads `docling_document.json` to MinIO (`eip-derived` bucket). No new PostgreSQL column needed. `derive_ontology_graph` downloads it from MinIO when needed.

### derive_text_chunks_and_embeddings change

Instead of upserting embeddings to Qdrant, store as vertex property in ArcadeDB:
```python
graph_store.set_vertex_embedding_sync(
    vertex_type="TextChunk",
    vertex_id=chunk_id,
    embedding_property="text_embedding",
    embedding=embedding_vector,
)
```

### derive_image_embeddings change

Same pattern -- store CLIP embeddings as `image_embedding` property on ImageChunk vertices in ArcadeDB.

## Section 6: Qdrant Elimination

### ArcadeDB vector indexes

```sql
CREATE INDEX ON TextChunk (text_embedding) LSM_VECTOR METADATA {
    dimensions: 1024, similarity: 'COSINE', quantization: 'INT8', addHierarchy: true
}
CREATE INDEX ON ImageChunk (image_embedding) LSM_VECTOR METADATA {
    dimensions: 512, similarity: 'COSINE', quantization: 'INT8'
}
CREATE INDEX ON CommunityReport (report_embedding) LSM_VECTOR METADATA {
    dimensions: 1024, similarity: 'COSINE', quantization: 'INT8', addHierarchy: true
}
CREATE INDEX ON TrustedTextChunk (text_embedding) LSM_VECTOR METADATA {
    dimensions: 1024, similarity: 'COSINE', quantization: 'INT8', addHierarchy: true
}
```

### New vertex types

TextChunk: chunk_id (FK to PostgreSQL), document_id, page_number, modality, classification, text_embedding (LIST, 1024-dim)
ImageChunk: chunk_id (FK to PostgreSQL), document_id, artifact_id, page_number, image_embedding (LIST, 512-dim) -- description stays in PostgreSQL
TrustedTextChunk: chunk_id, document_id, content_text, text_embedding (LIST, 1024-dim), source, classification
CommunityReport: community_id, membership_hash, title, summary, member_count, key_entities, key_relationships, report_embedding (LIST, 1024-dim), model_name, generated_at

### Chunk data ownership

PostgreSQL `retrieval.text_chunks` and `retrieval.image_chunks` tables remain authoritative for chunk content, section hierarchy, translated text, and relational metadata. ArcadeDB TextChunk/ImageChunk vertices carry chunk_id (FK to PostgreSQL), embedding, and minimal filter metadata (document_id, page_number, modality, classification). The embedding moves from Qdrant to ArcadeDB. The chunk content stays in PostgreSQL. `chunk_id` is the stable bridge between stores. `qdrant_point_id` column on PostgreSQL chunk models is dropped.

### Community report ownership

ArcadeDB is authoritative for community reports (CommunityReport vertices). The `community_reports` PostgreSQL table is removed. `community_runs` (pipeline state) stays in PostgreSQL.

### Trusted-data migration

Trusted-data semantic search moves entirely from Qdrant to ArcadeDB:
- `eip_trusted_text` Qdrant collection -> TrustedTextChunk vertex type with LSM_VECTOR index
- `qdrant_point_id` on trusted-data models -> removed
- `app/api/v1/trusted_data.py` -> uses GraphStore vector_search() instead of Qdrant
- `app/workers/trusted_data_tasks.py` -> writes to ArcadeDB instead of Qdrant
- `app/models/trusted_data.py` -> qdrant_point_id column removed

### Hybrid retrieval -- cross-model queries

The `hybrid` strategy uses ArcadeDB cross-model queries instead of separate Qdrant + Neo4j calls:

```sql
-- Semantic search + graph expansion in one query
SELECT chunk.*, entity.name, entity.entity_type
FROM (
    SELECT expand(vectorNeighbors('TextChunk[text_embedding]', :query_vector, :top_k))
) AS chunk
LET entity = chunk.in('EXTRACTED_FROM')
```

Same API endpoints, same UnifiedQueryResponse contract, better performance via single-database queries.

### Docker changes

Remove: qdrant service, qdrant_data volume, qdrant_test_data volume
Remove dependency: qdrant-client from pyproject.toml
Remove file: app/services/qdrant_store.py

### GraphStore Protocol additions

```python
async def vector_search(self, vertex_type: str, embedding_property: str,
                         query_vector: list[float], top_k: int,
                         filters: dict | None = None) -> list[dict]: ...

async def set_vertex_embedding(self, vertex_type: str, vertex_id: str,
                                embedding_property: str,
                                embedding: list[float]) -> None: ...

async def cross_model_search(self, query_vector: list[float], top_k: int,
                              expand_edges: list[str] | None = None,
                              filters: dict | None = None) -> list[dict]: ...
```

## Section 7: Architecture After Refactor

```
PostgreSQL
  - ingest schema (documents, elements, document_graph_extractions)
  - retrieval schema (chunk_links)
  - query_profiles schema
  - community_runs table (pipeline state)

ArcadeDB (graph + vector + search)
  - 46 entity vertex types (from ontology, schema-full)
  - 50+ relationship edge types (from ontology)
  - TextChunk vertices + text_embedding (1024-dim, HNSW)
  - ImageChunk vertices + image_embedding (512-dim, HNSW)
  - Document vertices (structural)
  - CommunityReport vertices + report_embedding (1024-dim, HNSW)
  - Structural edges (CONTAINS_TEXT, CONTAINS_IMAGE, SAME_PAGE, EXTRACTED_FROM, HAS_ALIAS)
  - Full-text indexes on entity names
  - Community detection algorithms (Louvain/Leiden)
  - Cross-model graph+vector queries

MinIO (object storage)
  - eip-raw (uploaded documents)
  - eip-derived (extracted images)
```

Two data stores instead of four (PostgreSQL + ArcadeDB + MinIO).

## Section 8: Files Changed

### Docling-graph service

Rewritten: main.py (922 -> ~200 lines), schemas.py
Created: template_builder.py, config_builder.py
Removed: prompts.py

### Docling-graph tests

Rewritten: test_extraction.py
Created: test_template_builder.py, test_config_builder.py, test_pipeline_integration.py, test_validation_pass.py
Removed: test_direct_extraction.py, test_prompts.py

### Main application

Modified: pipeline.py (derive_ontology_graph, derive_text_chunks_and_embeddings, derive_image_embeddings, derive_structure_links), docling_graph_service.py, config.py, retrieval.py (hybrid strategy), trusted_data.py, trusted_data_tasks.py, models/trusted_data.py (remove qdrant_point_id), models/retrieval.py (remove qdrant_point_id)
Removed: app/services/qdrant_store.py

Note: derive_text_chunks_and_embeddings and derive_image_embeddings become graph writers (creating ArcadeDB TextChunk/ImageChunk vertices with embeddings). derive_structure_links changes shape (creates Document vertex + structural edges to existing TextChunk/ImageChunk vertices, no longer creates separate chunk pointer nodes).

### Docker

Modified: docker-compose.yml (remove qdrant), docker-compose.test.yml (remove qdrant test config), env.example

### Dependencies

Removed from docling-graph service: litellm, json-repair
Removed from main app: qdrant-client
Note: docling-graph is NOT version-pinned. manage.sh --start pulls latest from GitHub repo. Service validates required API surface at startup.
