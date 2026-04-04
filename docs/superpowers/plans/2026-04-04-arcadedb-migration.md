# ArcadeDB Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Microsoft GraphRAG, replace Neo4j with ArcadeDB, eliminate Qdrant (vectors into ArcadeDB), add community-based global query. Two data stores instead of four.

**Architecture:** ArcadeDB replaces both Neo4j (graph) and Qdrant (vectors) using native LSMVectorIndex (HNSW). GraphStore Protocol abstracts all graph/vector operations. Entity merge on (entity_type, identity_fields). Provenance via EXTRACTED_FROM edges. Community detection via ArcadeDB native Louvain/Leiden algorithms.

**Tech Stack:** Python 3.12, ArcadeDB (built from source), httpx (async/sync), FastAPI, Celery, PostgreSQL, React/TypeScript

**Specs:**
- `docs/superpowers/specs/2026-04-04-arcadedb-migration-design.md`
- `docs/superpowers/specs/2026-04-04-docling-graph-pipeline-refactor-design.md`

---

## File Map

### Phase 1: Foundation — ArcadeDB Client + GraphStore

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `app/services/graph_store.py` | GraphStore Protocol + data classes (ProvenanceMetadata, NodeRecord, etc.) |
| Create | `app/services/arcadedb_client.py` | httpx async/sync wrapper for ArcadeDB HTTP API |
| Create | `app/services/arcadedb_graph.py` | GraphStore implementation using ArcadeDB client |
| Create | `app/services/arcadedb_schema.py` | Ontology-driven schema sync |
| Create | `tests/unit/test_graph_store_protocol.py` | Protocol conformance tests |
| Create | `tests/unit/test_arcadedb_client.py` | Client unit tests (mocked HTTP) |
| Create | `tests/unit/test_arcadedb_graph.py` | GraphStore operation tests |
| Create | `tests/unit/test_arcadedb_schema.py` | Schema sync tests |

### Phase 2: GraphRAG Removal

| Action | Path | Responsibility |
|--------|------|---------------|
| Delete | `app/services/graphrag_service.py` | 647 lines |
| Delete | `app/services/graphrag_config.py` | 144 lines |
| Delete | `app/services/graphrag_bridge.py` | 106 lines |
| Delete | `app/services/graphrag_prompts.py` | 464 lines |
| Delete | `app/services/graphrag_runtime_patches.py` | 165 lines |
| Delete | `app/services/graphrag_provenance.py` | 456 lines |
| Delete | `app/workers/graphrag_tasks.py` | 212 lines |
| Delete | `tests/unit/test_graphrag_*.py` | 7 test files |
| Modify | `app/workers/celery_app.py` | Remove GraphRAG beat schedules, shutdown handler |
| Modify | `app/schemas/retrieval.py` | Remove graphrag_* strategies, add global, remove async job models |
| Modify | `app/api/v1/retrieval.py` | Remove GraphRAG query functions |
| Modify | `app/api/v1/agent.py` | Remove GraphRAG strategy references |
| Modify | `pyproject.toml` | Remove graphrag, pyarrow deps |

### Phase 3: Neo4j Removal + GraphStore Wiring

| Action | Path | Responsibility |
|--------|------|---------------|
| Delete | `app/services/neo4j_graph.py` | 913 lines |
| Delete | `app/services/neo4j_dossier_service.py` | 554 lines |
| Delete | `scripts/migrate_age_to_neo4j.py` | Migration script |
| Delete | `docker/neo4j/` | Init scripts |
| Delete | `tests/unit/test_neo4j_graph_operations.py` | 454 lines |
| Delete | `tests/unit/test_neighborhood_graph.py` | 91 lines |
| Delete | `tests/unit/test_extracted_from_edges.py` | 361 lines |
| Delete | `tests/unit/test_upsert_relationships_batch.py` | 74 lines |
| Delete | `tests/unit/test_graph_service.py` | 63 lines |
| Delete | `tests/unit/test_canonicalization.py` | 252 lines |
| Delete | `tests/integration/test_graph_store_api.py` | 76 lines |
| Delete | `tests/integration/test_pipeline_graph.py` | 523 lines |
| Create | `app/services/dossier_service.py` | Rewrite of neo4j_dossier_service.py using GraphStore |
| Modify | `app/db/session.py` | Remove Neo4j drivers, add get_graph_store() |
| Modify | `app/main.py` | Remove Neo4j bootstrap, add ArcadeDB schema sync |
| Modify | `app/config.py` | Remove neo4j_* settings, add arcadedb_* settings |
| Modify | `app/services/canonicalization.py` | Use GraphStore instead of neo4j_graph |
| Modify | `app/services/query_profiles.py` | Rewrite Cypher → ArcadeDB SQL via GraphStore |
| Modify | `app/api/v1/graph_store.py` | Use GraphStore instead of neo4j drivers |
| Modify | `app/api/v1/sources.py` | Document delete via GraphStore |
| Modify | `app/api/v1/governance.py` | Remove Neo4j references |
| Modify | `app/api/v1/query_profiles.py` | Use GraphStore |

### Phase 4: Qdrant Removal + Vector Migration

| Action | Path | Responsibility |
|--------|------|---------------|
| Delete | `app/services/qdrant_store.py` | 301 lines |
| Delete | `tests/unit/test_qdrant_store.py` | 346 lines |
| Modify | `app/models/retrieval.py` | Remove pgvector Vector import, drop embedding + qdrant_point_id columns |
| Modify | `app/models/trusted_data.py` | Remove qdrant_point_id |
| Modify | `app/workers/pipeline.py` | Embedding stages → ArcadeDB vertices |
| Modify | `app/workers/trusted_data_tasks.py` | Use GraphStore vector ops |
| Modify | `app/api/v1/trusted_data.py` | Use GraphStore vector search |
| Modify | `app/api/v1/retrieval.py` | Replace Qdrant calls with ArcadeDB vectorNeighbors |
| Modify | `app/schemas/trusted_data.py` | Remove qdrant_point_id from response |
| Modify | `app/db/session.py` | Remove Qdrant client init |
| Modify | `pyproject.toml` | Remove qdrant-client, pgvector |
| Modify | `docker/postgres/init/01_extensions.sql` | Remove pgvector extension |
| Create | `alembic/versions/XXXX_drop_vector_columns.py` | Drop embedding + qdrant_point_id columns |

### Phase 5: Pipeline Migration

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `app/workers/pipeline.py` | derive_ontology_graph, derive_structure_links, derive_canonicalization, purge — all via GraphStore |
| Create | `tests/unit/test_pipeline_graph_operations.py` | Pipeline graph write tests |
| Create | `tests/unit/test_pipeline_structure_links.py` | Structure link tests |
| Create | `tests/unit/test_pipeline_canonicalization.py` | Canonicalization tests |
| Create | `tests/unit/test_provenance_metadata.py` | Provenance on every node/edge |

### Phase 6: Community Detection + Global Query

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `app/services/arcadedb_community.py` | Community detection + LLM report generation |
| Create | `app/workers/community_tasks.py` | Celery tasks (scheduled, manual, post-ingest) |
| Create | `app/api/v1/community.py` | Community API endpoints |
| Create | `tests/unit/test_arcadedb_community.py` | Community detection tests |
| Create | `tests/unit/test_community_tasks.py` | Celery task tests |
| Create | `tests/unit/test_community_api.py` | API endpoint tests |
| Create | `tests/unit/test_retrieval_global_query.py` | Global query strategy tests |
| Create | `alembic/versions/XXXX_add_community_runs.py` | community_runs table |
| Modify | `app/workers/celery_app.py` | Add community-detection beat schedule |

### Phase 7: Docker + Infrastructure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `docker/arcadedb/Dockerfile` | Multi-stage build from source |
| Modify | `docker-compose.yml` | Remove neo4j + qdrant, add arcadedb |
| Modify | `docker-compose.test.yml` | Remove neo4j/qdrant test config, add arcadedb |
| Modify | `manage.sh` | Add ensure_all_repos() for ArcadeDB + Docling pulls |
| Modify | `env.example` | Remove NEO4J/QDRANT/GRAPHRAG vars, add ARCADEDB/COMMUNITY vars |
| Modify | `.gitignore` | Add docker/arcadedb/repo/, docker/docling/repo/, docker/docling-graph/repo/ |

### Phase 8: Frontend

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `frontend/src/api/client.ts` | Remove graphrag types/functions, add global strategy |
| Modify | `frontend/src/components/QueryPage.tsx` | Remove GraphRAG components, add GlobalQueryDetail |
| Modify | `frontend/src/components/QueryProfileRegistryPage.tsx` | Remove graphrag references |
| Modify | `frontend/src/components/GraphExplorer.tsx` | Verify no neo4j-specific references |
| Modify | `frontend/src/App.tsx` | Remove Neo4j comment |

### Phase 9: Test Cleanup + New Tests

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `tests/conftest.py` | Remove neo4j/qdrant fixtures, add GraphStore mock |
| Modify | `tests/unit/test_config.py` | Update for new settings |
| Modify | `tests/unit/test_startup_bootstrap.py` | Update for ArcadeDB schema sync |
| Modify | `tests/unit/test_query_coverage.py` | Update for new strategies |
| Modify | `tests/unit/test_retrieval_schemas.py` | Update QueryStrategy enum |
| Create | `tests/unit/test_dossier_service.py` | Dossier service tests |
| Create | `tests/unit/test_query_profiles_arcadedb.py` | Query profile ArcadeDB SQL tests |
| Create | `tests/unit/test_retrieval_strategies.py` | All strategy dispatch tests |
| Modify | `example_queries.py` | Update for new strategies |
| Modify | `README.md` | Update architecture |

---

## Chunk 1: Foundation

### Task 1: GraphStore Protocol + data classes

**Files:**
- Create: `app/services/graph_store.py`
- Create: `tests/unit/test_graph_store_protocol.py`

- [ ] **Step 1: Write the Protocol and data classes**

Create `app/services/graph_store.py` with:

```python
"""GraphStore Protocol — backend-agnostic interface for graph + vector operations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

@dataclass
class ProvenanceMetadata:
    document_id: str
    page_numbers: list[int] = field(default_factory=list)
    upload_datetime: str | None = None
    document_datetime: str | None = None

@dataclass
class NodeRecord:
    entity_type: str
    identity_fields: dict[str, Any]
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    extraction_confidence: float = 1.0

@dataclass
class RelationshipRecord:
    from_type: str
    from_identity: dict[str, Any]
    to_type: str
    to_identity: dict[str, Any]
    rel_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    extraction_confidence: float = 1.0

@dataclass
class GraphEntityResult:
    node_id: str
    name: str
    entity_type: str
    canonical_name: str | None = None
    extraction_confidence: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)

@dataclass
class SchemaSyncReport:
    types_created: int = 0
    properties_added: int = 0
    indexes_created: int = 0
    errors: list[str] = field(default_factory=list)

@runtime_checkable
class GraphStore(Protocol):
    # --- Node operations ---
    async def upsert_node(self, entity_type: str, identity_fields: dict[str, Any],
                          name: str, properties: dict[str, Any],
                          provenance: ProvenanceMetadata,
                          extraction_confidence: float = 1.0) -> str: ...
    async def upsert_nodes_batch(self, nodes: list[NodeRecord],
                                  provenance: ProvenanceMetadata) -> int: ...
    async def upsert_document_node(self, document_id: str, title: str,
                                    provenance: ProvenanceMetadata) -> str: ...
    async def create_text_chunk_vertex(self, chunk_id: str, document_id: str,
                                        page_number: int | None, modality: str,
                                        classification: str) -> str: ...
    async def create_image_chunk_vertex(self, chunk_id: str, document_id: str,
                                         artifact_id: str | None,
                                         page_number: int | None) -> str: ...

    # --- Edge operations ---
    async def upsert_relationship(self, from_type: str, from_identity: dict[str, Any],
                                   to_type: str, to_identity: dict[str, Any],
                                   rel_type: str, properties: dict[str, Any],
                                   provenance: ProvenanceMetadata,
                                   extraction_confidence: float = 1.0) -> None: ...
    async def upsert_relationships_batch(self, relationships: list[RelationshipRecord],
                                          provenance: ProvenanceMetadata) -> int: ...
    async def create_structural_edge(self, from_type: str, from_id: str,
                                      to_type: str, to_id: str,
                                      edge_type: str,
                                      provenance: ProvenanceMetadata) -> None: ...

    # --- Query operations ---
    async def search_nodes(self, query_text: str, entity_types: list[str] | None,
                           top_k: int) -> list[GraphEntityResult]: ...
    async def resolve_root_entity(self, query_text: str, root_types: list[str],
                                   top_k: int) -> list[GraphEntityResult]: ...
    async def fulltext_search(self, query_text: str, entity_types: list[str] | None,
                               top_k: int) -> list[tuple[GraphEntityResult, float]]: ...
    async def get_neighborhood(self, entity_type: str, name: str,
                                hops: int) -> dict[str, Any]: ...
    async def get_neighborhood_graph(self, entity_type: str, name: str,
                                      hops: int) -> dict[str, Any]: ...
    async def get_ontology_linked_chunks(self, entity_names: list[str],
                                          rel_types: list[str]) -> list[str]: ...
    async def get_graph_stats(self) -> dict[str, Any]: ...
    async def get_relationship_count(self, entity_type: str, entity_name: str) -> int: ...
    async def get_co_extracted_entities(self, document_id: str,
                                         entity_types: list[str]) -> list[GraphEntityResult]: ...

    # --- Alias operations ---
    async def create_alias(self, entity_type: str, entity_name: str,
                           alias_name: str) -> None: ...
    async def search_by_alias(self, alias_name: str,
                               entity_types: list[str]) -> list[GraphEntityResult]: ...
    async def set_canonical_name(self, entity_type: str, entity_name: str,
                                  canonical_name: str) -> None: ...

    # --- Vector operations ---
    async def vector_search(self, vertex_type: str, embedding_property: str,
                             query_vector: list[float], top_k: int,
                             filters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
    async def set_vertex_embedding(self, vertex_type: str, vertex_id: str,
                                    embedding_property: str,
                                    embedding: list[float]) -> None: ...
    async def cross_model_search(self, query_vector: list[float], top_k: int,
                                  expand_edges: list[str] | None = None,
                                  filters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...

    # --- Document lifecycle ---
    async def delete_document_graph(self, document_id: str) -> int: ...

    # --- Schema management ---
    async def sync_schema(self, ontology: dict[str, Any]) -> SchemaSyncReport: ...
    async def ensure_indexes(self) -> None: ...
    async def ensure_ready(self) -> None: ...

    # --- Sync variants (for Celery workers) ---
    def upsert_node_sync(self, entity_type: str, identity_fields: dict[str, Any],
                          name: str, properties: dict[str, Any],
                          provenance: ProvenanceMetadata,
                          extraction_confidence: float = 1.0) -> str: ...
    def upsert_nodes_batch_sync(self, nodes: list[NodeRecord],
                                 provenance: ProvenanceMetadata) -> int: ...
    def upsert_relationships_batch_sync(self, relationships: list[RelationshipRecord],
                                         provenance: ProvenanceMetadata) -> int: ...
    def create_structural_edge_sync(self, from_type: str, from_id: str,
                                     to_type: str, to_id: str,
                                     edge_type: str,
                                     provenance: ProvenanceMetadata) -> None: ...
    def set_vertex_embedding_sync(self, vertex_type: str, vertex_id: str,
                                   embedding_property: str,
                                   embedding: list[float]) -> None: ...
    def create_text_chunk_vertex_sync(self, chunk_id: str, document_id: str,
                                       page_number: int | None, modality: str,
                                       classification: str) -> str: ...
    def create_image_chunk_vertex_sync(self, chunk_id: str, document_id: str,
                                        artifact_id: str | None,
                                        page_number: int | None) -> str: ...
    def delete_document_graph_sync(self, document_id: str) -> int: ...
    def ensure_ready_sync(self) -> None: ...
    def close_sync(self) -> None: ...
```

- [ ] **Step 2: Write protocol conformance test**

```python
# tests/unit/test_graph_store_protocol.py
"""Verify GraphStore Protocol definition and data classes."""
import pytest
from app.services.graph_store import (
    GraphStore, ProvenanceMetadata, NodeRecord, RelationshipRecord,
    GraphEntityResult, SchemaSyncReport,
)

class TestDataClasses:
    def test_provenance_metadata_defaults(self):
        p = ProvenanceMetadata(document_id="doc-1")
        assert p.document_id == "doc-1"
        assert p.page_numbers == []

    def test_node_record(self):
        n = NodeRecord(entity_type="RADAR_SYSTEM", identity_fields={"name": "Test"},
                       name="Test", properties={"freq": "9.4 GHz"})
        assert n.entity_type == "RADAR_SYSTEM"
        assert n.extraction_confidence == 1.0

    def test_graph_entity_result(self):
        r = GraphEntityResult(node_id="rid", name="Test", entity_type="PLATFORM")
        assert r.canonical_name is None

    def test_protocol_is_runtime_checkable(self):
        assert hasattr(GraphStore, '__protocol_attrs__') or isinstance(GraphStore, type)
```

- [ ] **Step 3: Run tests, commit**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/arcadedb
uv run pytest tests/unit/test_graph_store_protocol.py -v
git add app/services/graph_store.py tests/unit/test_graph_store_protocol.py
git commit -m "feat: add GraphStore Protocol and data classes"
```

---

### Task 2: ArcadeDB HTTP client

**Files:**
- Create: `app/services/arcadedb_client.py`
- Create: `tests/unit/test_arcadedb_client.py`

- [ ] **Step 1: Implement ArcadeDBClient**

Thin httpx wrapper with:
- Token-based auth (POST /api/v1/login → bearer token, re-auth on 401)
- `query(database, language, command, params)` → GET/POST /api/v1/query/{database}
- `command(database, language, command, params)` → POST /api/v1/command/{database}
- `batch(database, data, light_edges)` → POST /api/v1/batch/{database}
- `begin/commit/rollback` for transactions
- Both async (httpx.AsyncClient) and sync (httpx.Client) variants
- Retry on 401 (re-authenticate once)

See spec section "ArcadeDBClient (transport layer)" for full API.

- [ ] **Step 2: Write tests with mocked HTTP responses**

Test cases:
- `test_query_sends_correct_request` — verifies URL, auth header, body format
- `test_command_sends_post` — verifies write operations use POST
- `test_auth_token_refresh_on_401` — first call gets 401, re-auths, retries
- `test_batch_sends_ndjson` — verifies batch endpoint format
- `test_sync_client_works` — sync variants function correctly
- `test_begin_commit_transaction` — session ID management

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/unit/test_arcadedb_client.py -v
git add app/services/arcadedb_client.py tests/unit/test_arcadedb_client.py
git commit -m "feat: add ArcadeDB HTTP client with token auth and sync/async support"
```

---

### Task 3: ArcadeDB GraphStore implementation

**Files:**
- Create: `app/services/arcadedb_graph.py`
- Create: `tests/unit/test_arcadedb_graph.py`

- [ ] **Step 1: Implement ArcadeDBGraphStore**

Implements GraphStore Protocol using ArcadeDBClient. Key methods:
- `upsert_node` → ArcadeDB SQL UPDATE...UPSERT WHERE
- `upsert_nodes_batch` → batch endpoint with JSONL
- `upsert_relationship` → SQL with MERGE-like semantics
- `search_nodes` → fulltext index query
- `resolve_root_entity` → alias match → fulltext → relationship-count tie-break
- `vector_search` → `vectorNeighbors()` SQL function
- `set_vertex_embedding` → UPDATE vertex SET embedding = :vector
- `cross_model_search` → combined vectorNeighbors + graph traversal
- `delete_document_graph` → delete Document vertex, chunks, structural edges, orphan cleanup
- `ensure_ready` → verify database exists, required types exist (with retry/backoff)

See spec for ArcadeDB SQL equivalents of all Neo4j Cypher queries.

- [ ] **Step 2: Write tests with mocked ArcadeDBClient**

Test all GraphStore methods via mock client. Verify correct SQL generated for:
- Entity upsert (identity fields, not universal name)
- Relationship upsert (document_ids list)
- Fulltext search
- Vector search (vectorNeighbors)
- Cross-model search
- Document graph deletion (cascade + orphan cleanup)
- ensure_ready with retry

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/unit/test_arcadedb_graph.py -v
git add app/services/arcadedb_graph.py tests/unit/test_arcadedb_graph.py
git commit -m "feat: add ArcadeDB GraphStore implementation"
```

---

### Task 4: Schema sync from ontology

**Files:**
- Create: `app/services/arcadedb_schema.py`
- Create: `tests/unit/test_arcadedb_schema.py`

- [ ] **Step 1: Implement schema sync**

`sync_schema_from_ontology(client, ontology)`:
1. Fetch current ArcadeDB schema: `SELECT FROM schema:types`
2. For each entity_type → `CREATE VERTEX TYPE IF NOT EXISTS {name}`
3. For each property → `CREATE PROPERTY IF NOT EXISTS {type}.{prop} {arcade_type}`
4. For each relationship_type → `CREATE EDGE TYPE IF NOT EXISTS {name}`
5. Create structural types (TextChunk, ImageChunk, TrustedTextChunk, Document, Alias, CommunityReport)
6. Create LSM_VECTOR indexes on embedding properties
7. Create fulltext indexes on entity name properties
8. Reserved word handling (TABLE → TABLE_REF)
9. Return SchemaSyncReport

- [ ] **Step 2: Write tests**

- `test_creates_vertex_types_from_ontology` — verify CREATE VERTEX TYPE commands
- `test_creates_edge_types_from_ontology` — verify CREATE EDGE TYPE commands
- `test_creates_structural_types` — TextChunk, ImageChunk, Document, Alias
- `test_creates_vector_indexes` — LSM_VECTOR on text_embedding, image_embedding
- `test_idempotent` — running twice produces no errors
- `test_reserved_word_handling` — TABLE → TABLE_REF

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/unit/test_arcadedb_schema.py -v
git add app/services/arcadedb_schema.py tests/unit/test_arcadedb_schema.py
git commit -m "feat: add ontology-driven ArcadeDB schema sync"
```

---

### Task 5: Session + config wiring

**Files:**
- Modify: `app/db/session.py`
- Modify: `app/config.py`
- Modify: `app/main.py`

- [ ] **Step 1: Add ArcadeDB settings to config.py**

Add to Settings class:
```python
# ArcadeDB
arcadedb_url: str = "http://arcadedb:2480"
arcadedb_user: str = "root"
arcadedb_password: str = "eip_arcadedb_secret"
arcadedb_database: str = "eip_knowledge_graph"

# Community detection
community_detection_enabled: bool = True
community_detection_interval_minutes: int = 60
community_detection_post_ingest_enabled: bool = True
community_detection_post_ingest_threshold: int = 5
community_detection_algorithm: str = "leiden"
community_detection_resolution: float = 1.0
community_detection_max_iterations: int = 20
community_report_llm_model: str = "llama3.2"
community_report_llm_prompt: str = ""  # default prompt in community module
```

Remove: all `neo4j_*`, `graphrag_*`, `qdrant_*` settings.

- [ ] **Step 2: Add get_graph_store() to session.py**

Remove Neo4j driver singletons and Qdrant client. Add:
```python
_graph_store: ArcadeDBGraphStore | None = None

def get_graph_store() -> GraphStore:
    global _graph_store
    if _graph_store is None:
        from app.services.arcadedb_client import ArcadeDBClient
        from app.services.arcadedb_graph import ArcadeDBGraphStore
        settings = get_settings()
        client = ArcadeDBClient(settings.arcadedb_url, settings.arcadedb_user, settings.arcadedb_password)
        _graph_store = ArcadeDBGraphStore(client, settings.arcadedb_database)
    return _graph_store
```

- [ ] **Step 3: Update main.py startup**

Replace Neo4j bootstrap with ArcadeDB schema sync:
```python
graph_store = get_graph_store()
await graph_store.sync_schema(active_ontology)
await graph_store.ensure_indexes()
```

- [ ] **Step 4: Update tests**

Modify `tests/unit/test_config.py` and `tests/unit/test_startup_bootstrap.py` for new settings.

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/db/session.py app/main.py tests/unit/test_config.py tests/unit/test_startup_bootstrap.py
git commit -m "feat: wire ArcadeDB client, GraphStore, and schema sync into app startup"
```

---

## Chunk 2: Removal + Migration

### Task 6: Remove GraphRAG completely

**Files:** See Phase 2 file map above.

- [ ] **Step 1: Delete all GraphRAG service files**

```bash
rm app/services/graphrag_service.py app/services/graphrag_config.py \
   app/services/graphrag_bridge.py app/services/graphrag_prompts.py \
   app/services/graphrag_runtime_patches.py app/services/graphrag_provenance.py \
   app/workers/graphrag_tasks.py
```

- [ ] **Step 2: Delete all GraphRAG test files**

```bash
rm tests/unit/test_graphrag_service.py tests/unit/test_graphrag_config.py \
   tests/unit/test_graphrag_bridge.py tests/unit/test_graphrag_provenance.py \
   tests/unit/test_graphrag_prompts.py tests/unit/test_graphrag_runtime_patches.py \
   tests/unit/test_graphrag_query_task.py
```

- [ ] **Step 3: Delete GraphRAG docs**

```bash
rm docs/superpowers/plans/2026-03-17-microsoft-graphrag-integration.md \
   docs/superpowers/plans/2026-03-18-graphrag-fixes-and-drift-basic.md \
   docs/superpowers/plans/2026-03-31-async-graphrag-queries.md \
   docs/superpowers/plans/2026-04-02-graphrag-context-provenance.md \
   docs/superpowers/plans/2026-04-02-graphrag-citation-provenance.md \
   docs/superpowers/specs/2026-03-31-async-graphrag-queries-design.md \
   docs/superpowers/specs/2026-04-02-graphrag-context-provenance-design.md \
   docs/superpowers/specs/2026-04-02-graphrag-citation-provenance-design.md
```

- [ ] **Step 4: Update celery_app.py**

Remove GraphRAG beat schedules (`graphrag-indexing`, `graphrag-auto-tune`), remove `close_graphrag_loop()` shutdown handler, remove graphrag queue routing.

- [ ] **Step 5: Update retrieval schemas**

In `app/schemas/retrieval.py`:
- Remove `graphrag_local`, `graphrag_global`, `graphrag_drift` from QueryStrategy enum
- Add `global_` = "global" to QueryStrategy
- Remove `GraphRAGJobSubmitResponse`, `GraphRAGJobStatusResponse`
- Remove `_MODE_MAP` entries for graphrag_*, add "global"

- [ ] **Step 6: Update retrieval.py**

Remove `_graphrag_local_query()`, `_graphrag_global_query()`, `_graphrag_drift_query()`, async job submission/polling logic. Remove graphrag imports.

- [ ] **Step 7: Update agent.py**

Remove GraphRAG strategy references from `get_agent_context`.

- [ ] **Step 8: Update pyproject.toml**

Remove: `graphrag>=3.0.0`, `pyarrow>=14.0.0`

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: remove Microsoft GraphRAG completely"
```

---

### Task 7: Remove Neo4j + wire GraphStore

**Files:** See Phase 3 file map above.

- [ ] **Step 1: Delete Neo4j service files**

```bash
rm app/services/neo4j_graph.py app/services/neo4j_dossier_service.py \
   scripts/migrate_age_to_neo4j.py
rm -rf docker/neo4j/
```

- [ ] **Step 2: Delete Neo4j test files**

```bash
rm tests/unit/test_neo4j_graph_operations.py tests/unit/test_neighborhood_graph.py \
   tests/unit/test_extracted_from_edges.py tests/unit/test_upsert_relationships_batch.py \
   tests/unit/test_graph_service.py tests/unit/test_canonicalization.py \
   tests/integration/test_graph_store_api.py tests/integration/test_pipeline_graph.py
```

- [ ] **Step 3: Create dossier_service.py**

Rewrite `neo4j_dossier_service.py` using GraphStore. Replace all Cypher templates with GraphStore method calls. See spec "Query Translation Examples" for ArcadeDB SQL equivalents.

- [ ] **Step 4: Update canonicalization.py**

Replace `neo4j_graph` imports with `get_graph_store()`. Replace `fulltext_search_entity` → `graph_store.fulltext_search`. Replace `create_alias_edge` → `graph_store.create_alias`.

- [ ] **Step 5: Update query_profiles.py**

Rewrite Cypher query templates to use GraphStore methods. This is the most substantial rewrite — see spec for worked translation examples (system_components traversal, multi-step traversal).

- [ ] **Step 6: Update graph_store.py API**

Replace `get_neo4j_async_driver()` with `get_graph_store()` in all four endpoints.

- [ ] **Step 7: Update sources.py**

Document delete: replace Neo4j Cypher deletion with `graph_store.delete_document_graph()` as step 7 of 8 in the delete orchestration.

- [ ] **Step 8: Update governance.py**

Remove Neo4j/AGE comments. Update re-embed path to use `graph_store.set_vertex_embedding()`.

- [ ] **Step 9: Remove neo4j from pyproject.toml**

Remove: `neo4j>=5.25.0`

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat: remove Neo4j, wire all graph operations through GraphStore"
```

---

### Task 8: Remove Qdrant + vector migration

**Files:** See Phase 4 file map above.

- [ ] **Step 1: Delete Qdrant files**

```bash
rm app/services/qdrant_store.py tests/unit/test_qdrant_store.py
```

- [ ] **Step 2: Update retrieval models**

In `app/models/retrieval.py`:
- Remove `from pgvector.sqlalchemy import Vector`
- Remove `embedding` columns from TextChunk and ImageChunk
- Remove `qdrant_point_id` columns

- [ ] **Step 3: Update trusted_data models**

Remove `qdrant_point_id` from `app/models/trusted_data.py`.

- [ ] **Step 4: Create Alembic migration**

Drop `embedding` (Vector) and `qdrant_point_id` columns from text_chunks and image_chunks. Drop pgvector extension.

- [ ] **Step 5: Update pipeline.py embedding stages**

`derive_text_chunks_and_embeddings`: After creating PostgreSQL chunk row, also create ArcadeDB TextChunk vertex with embedding via `graph_store.create_text_chunk_vertex_sync()` + `graph_store.set_vertex_embedding_sync()`.

`derive_image_embeddings`: Same pattern for ImageChunk vertices.

- [ ] **Step 6: Update trusted_data_tasks.py**

Replace `upsert_trusted_vector()` (Qdrant) with `graph_store.set_vertex_embedding_sync()` on TrustedTextChunk vertices.

- [ ] **Step 7: Update retrieval.py**

Replace all Qdrant search calls with `graph_store.vector_search()` and `graph_store.cross_model_search()`. Map Qdrant filter objects to GraphStore filter dicts.

- [ ] **Step 8: Update session.py and config.py**

Remove Qdrant client initialization and settings.

- [ ] **Step 9: Update Docker postgres init**

Remove pgvector extension from `docker/postgres/init/01_extensions.sql`.

- [ ] **Step 10: Remove dependencies**

Remove from pyproject.toml: `qdrant-client>=1.13.0`, `pgvector>=0.3.6`

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat: remove Qdrant, migrate vectors to ArcadeDB LSMVectorIndex"
```

---

### Task 9: Pipeline migration

**Files:** See Phase 5 file map above.

- [ ] **Step 1: Update derive_ontology_graph**

- Retrieve DoclingDocument JSON from MinIO
- POST to docling-graph `/extract-all` with new contract
- Receive NetworkX graph JSON
- Apply confidence quality gates
- Import via GraphStore (upsert_nodes_batch_sync, upsert_relationships_batch_sync)
- Build EXTRACTED_FROM edges from provenance chunk_index mapping
- Adapter layer: transform to graph_json contract for PostgreSQL storage

- [ ] **Step 2: Update derive_structure_links**

- Create Document vertex via GraphStore
- Create structural edges (CONTAINS_TEXT, CONTAINS_IMAGE, SAME_PAGE) to existing TextChunk/ImageChunk vertices
- No more ChunkRef creation — chunks already exist from embedding stages

- [ ] **Step 3: Update derive_canonicalization**

Use `graph_store.fulltext_search_sync()` and `graph_store.create_alias_sync()`.

- [ ] **Step 4: Update purge_document**

Use `graph_store.delete_document_graph_sync()`.

- [ ] **Step 5: Add ensure_ready_sync to graph-writing tasks**

Each graph-writing task calls `graph_store.ensure_ready_sync()` before first write.

- [ ] **Step 6: Write pipeline tests**

Create test files for pipeline graph operations, structure links, canonicalization, provenance.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: migrate all pipeline stages to GraphStore"
```

---

### Task 10: Community detection + global query

**Files:** See Phase 6 file map above.

- [ ] **Step 1: Implement arcadedb_community.py**

- `run_community_detection()` — Louvain/Leiden on domain-entity projection
- `_compute_membership_hash()` — SHA-256 of sorted (entity_type, name) tuples
- `_diff_communities()` — compare new vs stored, determine regenerate/skip/delete
- `_generate_community_report()` — LLM call with entities + relationships + evidence
- `_embed_community_report()` — BGE embedding for vector search

- [ ] **Step 2: Implement community_tasks.py**

- `run_community_detection_task(mode)` — Celery task with Redis lock
- Post-ingest hook: counter + threshold trigger

- [ ] **Step 3: Implement community.py API**

Endpoints: POST /detect, GET /status, GET /status/{run_id}, GET /reports, GET /reports/{id}

- [ ] **Step 4: Add global query to retrieval.py**

`_global_query()`: embed query → vectorNeighbors on CommunityReport → LLM synthesis → UnifiedQueryResponse

- [ ] **Step 5: Add community-detection to Celery Beat**

- [ ] **Step 6: Create Alembic migration for community_runs table**

- [ ] **Step 7: Write tests**

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: add community detection and global query strategy"
```

---

## Chunk 3: Infrastructure + Frontend + Cleanup

### Task 11: Docker infrastructure

**Files:** See Phase 7 file map above.

- [ ] **Step 1: Create docker/arcadedb/Dockerfile**

Multi-stage build: JDK 21 builder → JRE 21 runtime. Gradle build with -x test.

- [ ] **Step 2: Update docker-compose.yml**

Remove neo4j + qdrant services and volumes. Add arcadedb service with healthcheck (GET /api/v1/ready). Update depends_on for api, worker, worker-graph.

- [ ] **Step 3: Update docker-compose.test.yml**

Remove neo4j/qdrant test config. Add arcadedb test config.

- [ ] **Step 4: Update manage.sh**

Add `ensure_all_repos()` calling `ensure_repo()` for ArcadeDB, Docling, Docling-Graph repos. Called before `dc build` in cmd_start.

- [ ] **Step 5: Update env.example**

Remove all NEO4J_*, QDRANT_*, GRAPHRAG_* vars. Add ARCADEDB_* and COMMUNITY_* vars.

- [ ] **Step 6: Update .gitignore**

Add docker/arcadedb/repo/, docker/docling/repo/, docker/docling-graph/repo/

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: add ArcadeDB Docker service, remove Neo4j/Qdrant infrastructure"
```

---

### Task 12: Frontend updates

**Files:** See Phase 8 file map above.

- [ ] **Step 1: Update client.ts**

Remove: `QueryStrategy` graphrag values, `GraphRAGJobSubmitResponse`/`GraphRAGJobStatusResponse` interfaces, `submitGraphRAGQuery()`/`getGraphRAGQueryStatus()`/`getGraphRAGQueryResult()` functions.
Add: `global` to QueryStrategy enum.

- [ ] **Step 2: Update QueryPage.tsx**

Remove: `GraphRAGLocalDetail`, `GraphRAGGlobalDetail` components, three GraphRAG modes from MODES array.
Add: `global` mode to MODES, `GlobalQueryDetail` component (shows synthesized answer with expandable community sources).

- [ ] **Step 3: Update remaining frontend files**

QueryProfileRegistryPage.tsx, GraphExplorer.tsx, App.tsx — remove graphrag/neo4j references.

- [ ] **Step 4: Commit**

```bash
git add -A frontend/
git commit -m "feat: update frontend — remove GraphRAG UI, add global query mode"
```

---

### Task 13: Test cleanup + conftest

**Files:** See Phase 9 file map above.

- [ ] **Step 1: Update conftest.py**

Remove all neo4j mock fixtures and qdrant mock fixtures. Add:
```python
@pytest.fixture
def mock_graph_store():
    """GraphStore mock for unit tests."""
    return MagicMock(spec=GraphStore)
```

- [ ] **Step 2: Update test_config.py**

Test all new arcadedb_*, community_* settings with defaults. Verify no neo4j_*/graphrag_*/qdrant_* settings.

- [ ] **Step 3: Update remaining test files**

- test_startup_bootstrap.py — ArcadeDB schema sync at startup
- test_query_coverage.py — basic, hybrid, global strategies
- test_retrieval_schemas.py — updated QueryStrategy enum
- test_retrieval_helpers.py — remove Qdrant-specific tests

- [ ] **Step 4: Create new test files**

- test_dossier_service.py
- test_query_profiles_arcadedb.py
- test_retrieval_strategies.py
- test_retrieval_global_query.py

- [ ] **Step 5: Update example_queries.py and README.md**

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "test: update test suite for ArcadeDB migration"
```

---

### Task 14: Final verification

- [ ] **Step 1: Verify no stale imports**

```bash
grep -r "neo4j" app/ --include="*.py" -l
grep -r "graphrag" app/ --include="*.py" -l
grep -r "qdrant" app/ --include="*.py" -l
grep -r "pgvector" app/ --include="*.py" -l
```

All should return empty (except comments in migration files).

- [ ] **Step 2: Verify all acceptance criteria**

Walk through all 22 acceptance criteria from the spec.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final verification — no stale references"
```

---

## Summary

| Task | Phase | Description | Est. complexity |
|------|-------|-------------|----------------|
| 1 | Foundation | GraphStore Protocol + data classes | Low |
| 2 | Foundation | ArcadeDB HTTP client | Medium |
| 3 | Foundation | ArcadeDB GraphStore implementation | High |
| 4 | Foundation | Schema sync from ontology | Medium |
| 5 | Foundation | Session + config wiring | Medium |
| 6 | Removal | Remove GraphRAG completely | Low (deletion) |
| 7 | Removal | Remove Neo4j + wire GraphStore | High |
| 8 | Removal | Remove Qdrant + vector migration | High |
| 9 | Migration | Pipeline migration | High |
| 10 | New feature | Community detection + global query | High |
| 11 | Infrastructure | Docker infrastructure | Medium |
| 12 | Frontend | Frontend updates | Medium |
| 13 | Cleanup | Test cleanup + new tests | Medium |
| 14 | Verification | Final verification | Low |
