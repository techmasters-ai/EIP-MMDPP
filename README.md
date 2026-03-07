# EIP-MMDPP

Multi-modal document processing and retrieval platform for defense/military use cases.

Ingests PDFs, DOCX, images, and technical drawings → converts documents via Docling (granite-docling-258M VLM) → embeds text (BGE) and images (CLIP) into Qdrant vector collections, builds a military equipment knowledge graph (Neo4j), runs GraphRAG community detection and reporting, and maintains governed trusted data (Cognee). Supports 7 retrieval modes: text basic, text only, images only, multi-modal, trusted data, GraphRAG local, and GraphRAG global. Includes a user feedback → curator patch approval workflow and a React web UI.

## Architecture

### Knowledge Layers

```
                    ┌──────────────────────────────────────────┐
                    │           Neo4j Knowledge Graph           │
                    │   Document ←→ ChunkRef nodes              │
                    │   Entity nodes (LLM + regex extracted)    │
                    │   Ontology relations (44 predicates)      │
                    │   Alias nodes (entity canonicalization)   │
                    │   Fulltext index (fuzzy entity search)    │
                    └──────────┬───────────────┬────────────────┘
                               │               │
                    ┌──────────▼──────┐ ┌──────▼──────────┐
                    │ Qdrant:         │ │ Qdrant:          │
                    │ eip_text_chunks │ │ eip_image_chunks │
                    │ BGE 1024-dim    │ │ CLIP 512-dim     │
                    │ Cosine distance │ │ Cosine distance  │
                    └─────────────────┘ └─────────────────┘

                    ┌──────────────────────────────────────────┐
                    │       GraphRAG Community Layer             │
                    │   Leiden/Louvain community detection       │
                    │   LLM-generated community reports          │
                    │   Local + Global search modes              │
                    └──────────────────────────────────────────┘

                    ┌──────────────────────────────────────────┐
                    │       Cognee Trusted Data Layer            │
                    │   NetworkX + LanceDB (separate store)     │
                    │   Governed: PROPOSED → APPROVED/REJECTED  │
                    └──────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology |
|---|---|
| API | FastAPI (Python 3.11) |
| Processing | Celery + Redis |
| Database | PostgreSQL 16 (metadata, chunk_links, governance) |
| Graph Database | Neo4j Community Edition (knowledge graph, ontology, canonicalization) |
| Vector Database | Qdrant OSS (text + image embeddings) |
| Object Storage | MinIO |
| Text Embeddings | `BAAI/bge-large-en-v1.5` (1024-dim, fully local) |
| Image Embeddings | OpenCLIP ViT-B/32 (512-dim, cross-modal) |
| Document Conversion | Docling + `ibm-granite/granite-docling-258M` VLM |
| Graph Extraction | docling-graph + LLM (ontology-driven entity/relationship extraction) |
| GraphRAG | Microsoft graphrag (community detection, reports, local/global search) |
| Trusted Data | Cognee (NetworkX graph + LanceDB vector, governed approval workflow) |
| Frontend | React 18 + TypeScript + Vite (TecMasters design system) |

All ML inference runs **fully locally** — no cloud API calls required (air-gapped deployment).

### Docker Services (9 containers)

| Service | Purpose |
|---|---|
| `api` | FastAPI application server |
| `worker` | Celery worker (ingest pipeline) |
| `beat` | Celery Beat (periodic tasks, GraphRAG indexing) |
| `postgres` | PostgreSQL 16 (metadata, chunk_links, governance) |
| `redis` | Celery broker + result backend |
| `minio` | S3-compatible object storage |
| `docling` | Document conversion service (granite-docling-258M VLM) |
| `neo4j` | Neo4j Community Edition (knowledge graph) |
| `qdrant` | Qdrant OSS (vector search) |

## Quickstart

```bash
# 1. Copy environment config and set required values
cp env.example .env
# Edit .env — at minimum set LLM_PROVIDER and (if openai) OPENAI_API_KEY

# 2. Start all services (builds images, runs migrations, waits for health)
./manage.sh --start

# 3. API + web UI
#    Web UI:  http://localhost:8000/
#    API docs: http://localhost:8000/docs
#    Neo4j Browser: http://localhost:7474/
#    Qdrant Dashboard: http://localhost:6333/dashboard
```

## manage.sh — Project Management CLI

All service lifecycle, database, worker, and test operations are available through `./manage.sh`:

```bash
# Service lifecycle
./manage.sh --start              # Build and start all services; wait for health
./manage.sh --stop               # Stop all services (preserves data)
./manage.sh --restart            # Restart without rebuilding images
./manage.sh --status             # Show service status and health checks
./manage.sh --logs [service]     # Stream logs (api, worker, beat, postgres, redis, minio, docling, neo4j, qdrant)
./manage.sh --blow-away          # Destroy everything: containers, volumes, data

# Database
./manage.sh --migrate            # Run alembic upgrade head
./manage.sh --seed               # Run ontology seeder

# Testing (delegates to scripts/run_tests.sh)
./manage.sh --test               # Full suite
./manage.sh --test unit          # Unit tests only
./manage.sh --test integration   # Integration tests
./manage.sh --test e2e           # End-to-end tests
```

## LLM Provider Configuration

A single `LLM_PROVIDER` env var controls the LLM backend for **all** LLM-dependent features (graph extraction, GraphRAG reports, Cognee trusted data). Each feature specifies its own model via a dedicated env var.

| Value | Description |
|---|---|
| `ollama` | Uses local Ollama server. Fully air-gapped. Requires `OLLAMA_BASE_URL`. |
| `openai` | Uses OpenAI API. Requires `OPENAI_API_KEY`. |
| `mock` | Disables all LLM calls. Used in tests and environments without an LLM. |

```bash
# Air-gapped (Ollama) setup
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434

# Per-feature model selection
DOCLING_GRAPH_MODEL=llama3.2       # Model for graph entity/relationship extraction
GRAPHRAG_MODEL=llama3.2            # Model for GraphRAG community report generation
COGNEE_MODEL=llama3.2              # Model for Cognee trusted data operations
```

## Running Tests

```bash
# Full suite (unit → integration → E2E)
./scripts/run_tests.sh

# Individual layers
./scripts/run_tests.sh unit
./scripts/run_tests.sh integration
./scripts/run_tests.sh e2e

# Skip coverage instrumentation (faster, lower RAM)
SKIP_COV=1 ./scripts/run_tests.sh unit

# Keep stack running after tests
KEEP_STACK=1 ./scripts/run_tests.sh
```

## API Endpoints (v1)

### Sources & Document Upload
- `POST /v1/sources` — create a document collection
- `POST /v1/sources/{id}/documents` — upload a document (streams to MinIO, triggers pipeline; 409 on duplicate unless previous upload FAILED)
- `GET /v1/documents/{id}/status` — poll pipeline status (includes stage summary)
- `GET /v1/documents/{id}/stages` — detailed pipeline stage diagnostics
- `POST /v1/documents/{id}/reingest` — re-run pipeline (`{"mode": "full|embeddings_only|graph_only"}`); resets status to PENDING; returns 409 if already PROCESSING

### Directory Watcher
- `POST /v1/watch-dirs` — register a directory for auto-ingest
- `DELETE /v1/watch-dirs/{id}` — remove watch directory
- Per-directory `poll_interval_seconds` respected (directories only scanned when enough time has elapsed since last scan)

### Graph Store (Neo4j)
- `POST /v1/graph/ingest/entity` — create an entity node
- `POST /v1/graph/ingest/relationship` — create a relationship edge
- `POST /v1/graph/query` — Cypher traversal query

### Trusted Data (Cognee)
- `POST /v1/memory/ingest` — propose knowledge (status: PROPOSED)
- `GET /v1/memory/proposals` — list proposals (filterable by status)
- `POST /v1/memory/proposals/{id}/approve` — curator approves → writes to Cognee
- `POST /v1/memory/proposals/{id}/reject` — curator rejects
- `POST /v1/memory/query` — search approved trusted data

### Unified Retrieval

```json
POST /v1/retrieval/query
{
  "query_text": "Patriot PAC-3 guidance computer specifications",
  "strategy": "hybrid",
  "modality_filter": "all",
  "top_k": 10,
  "include_context": true
}
```

Response returns a flat ranked results list:

```json
{
  "strategy": "hybrid",
  "modality_filter": "all",
  "results": [
    { "chunk_id": "...", "score": 0.92, "modality": "text", "content_text": "..." },
    { "chunk_id": "...", "score": 0.78, "modality": "image", "content_text": "..." }
  ],
  "total": 2
}
```

Query strategies:

| Strategy | Modality Filter | Input | Pipeline | Output |
|---|---|---|---|---|
| `basic` | `all` | Text only | BGE vector search (Qdrant) | Text chunks |
| `hybrid` | `text` | Text or image | Full multi-modal pipeline | Filtered to text |
| `hybrid` | `image` | Text or image | Full multi-modal pipeline | Filtered to images |
| `hybrid` | `all` | Text or image | Full multi-modal pipeline | All results |
| `memory` | `all` | Text | Cognee search | Approved trusted data |
| `graphrag_local` | `all` | Text | Entity-centric + community reports | Entity matches with community context |
| `graphrag_global` | `all` | Text | Cross-community summarization | Community reports ranked by relevance |

> **Backward compatibility**: The legacy `mode` field (e.g. `"mode": "text_only"`) is still accepted and maps to the corresponding `strategy` + `modality_filter` combination.

The hybrid pipeline runs: parallel vector search (BGE + CLIP via Qdrant `asyncio.gather`) → document-structure expansion (chunk_links table) → ontology traversal (Neo4j entity relationships) → weighted fusion scoring → deduplicate → rank → filter by modality.

Image-modality results include a presigned `image_url` for inline display in the UI.

**Weighted Fusion Scoring**: `final = 0.65*semantic + 0.20*doc_structure + 0.15*ontology + MIL-ID bonus`. MIL-ID bonus matches NSN, MIL-STD, ELNOT, DIEQP, and AN/ designators. All weights are configurable via environment variables (see `env.example`).

### Agent / LangGraph Context

```
GET /v1/agent/context
  ?query=Patriot+PAC-3+guidance+computer
  &strategy=basic
  &top_k=10
  &include_sources=true
```

Returns a pre-formatted markdown context string for direct injection into an LLM prompt. Supports all query strategies. Accepts `strategy` + `modality_filter` params (and the deprecated `mode` param for backward compatibility).

```python
# LangGraph usage example
resp = requests.get("http://localhost:8000/v1/agent/context",
                    params={"query": query, "strategy": "basic"})
system_msg = f"Use this context:\n\n{resp.json()['context']}"
```

### Governance
- `POST /v1/feedback` — submit a correction on a retrieved result
- `POST /v1/patches/{id}/approve` — curator approves a patch
- `POST /v1/patches/{id}/apply` — apply an approved patch

All Neo4j graph mutations (node/edge create, update, delete) require **dual-curator approval**. Text and classification corrections require a single curator.

## Ontology

The knowledge graph uses a 5-layer ontology grounded in DoDAF DM2 concepts:

1. **Reference & Provenance** — Documents, sections, figures, tables, assertions
2. **Military Equipment** — Platforms, systems, subsystems, components
3. **EM/RF Signal & Radar** — Emissions, waveforms, modulation, antennas, receivers, processing
4. **Weapon / Missile / AAA** — Missiles, seekers, guidance, propulsion, artillery
5. **Operational / Capability** — Capabilities, engagement timelines, performance measures

~35 entity types, 44 relationship predicates, enforced via validation matrix at graph write time.

See `ontology/base.yaml` for the full schema.

## GraphRAG

Community detection and cross-community search powered by Microsoft's `graphrag` library:

- **Indexing**: Celery Beat runs Leiden/Louvain community detection on the Neo4j graph, then generates LLM community reports (configurable interval via `GRAPHRAG_INDEXING_INTERVAL_MINUTES`)
- **Local search**: Entity-centric retrieval with community report context — finds relevant entities and enriches results with their community summaries
- **Global search**: Cross-community summarization — retrieves and ranks community reports for broad analytical questions

## Ingest Pipeline

Manifest-first architecture with parallel derivation stages and idempotent writes:

```
prepare_document  (validate + detect + Docling convert + persist document_elements)
    ↓
┌── derive_text_chunks_and_embeddings ──┐
│── derive_image_embeddings             │  (parallel Celery chord)
└── derive_ontology_graph ──────────────┘
    ↓
collect_derivations  (chord callback)
    ↓
derive_structure_links  (needs embedding output committed)
    ↓
derive_canonicalization  (entity resolution pass)
    ↓
finalize_document
```

Key features:
- **Canonical element store** (`document_elements` table) — parse once, derive many
- **Parallel derivations** — embedding and graph extraction run concurrently via Celery chord
- **Sequential structure links** — runs after embeddings are committed (avoids race condition)
- **Entity canonicalization** — post-extraction alias resolution (exact → alias → fuzzy match → new)
- **Idempotent writes** — deterministic chunk keys with `ON CONFLICT DO UPDATE`
- **Dual vector store** — embeddings upserted to Qdrant with `qdrant_point_id` cross-reference in Postgres
- **Run/stage tracking** — `pipeline_runs` and `stage_runs` tables for diagnostics
- **Worker split** — optional queue isolation: `docker compose --profile split up`
- **Docling concurrency gate** — Redis semaphore with `DOCLING_CONCURRENCY` permits (default 1) controls parallel Docling conversions; queued tasks wait and retry instead of timing out; health check runs only after permit acquisition to avoid false "unavailable" during busy conversions; health probe timeout configurable via `DOCLING_HEALTH_TIMEOUT` (default 10s)
- **Configurable retries** — retry counts and delays for all pipeline stages configurable via env vars (`PREPARE_MAX_RETRIES`, `EMBED_MAX_RETRIES`, etc.); documents stay in PROCESSING status during retries and only show FAILED after all retries are exhausted
- **Stage run attempt tracking** — each retry creates a separate `stage_runs` row with incrementing `attempt` number, preserving full retry history per stage
- **Task time limits** — `soft_time_limit` / `time_limit` on all tasks prevent indefinite blocking
- **Re-upload on failure** — re-uploading a file that previously FAILED removes the old record and re-ingests (no 409)
- **Reingest safety** — reingest endpoint rejects requests when pipeline is already PROCESSING (409); failure handlers use the task's own `run_id` to avoid cross-run contamination
- **Terminal status handling** — UI polling stops for all terminal states (COMPLETE, ERROR, FAILED, PARTIAL_COMPLETE); FAILED shows red badge, PARTIAL_COMPLETE shows amber warning badge with error context

The `prepare_document` task calls the dedicated Docling service which extracts text, tables, images, equations, and schematics in a single VLM pass. If the Docling service is unavailable and `DOCLING_FALLBACK_ENABLED=true`, the pipeline falls back to legacy extraction.

Graph extraction uses LLM (via `LLM_PROVIDER`) for ontology-driven entity/relationship extraction with triple validation, with regex NER as fallback when LLM is unavailable. Extracted entities and relationships are imported directly to Neo4j. Graph data is stored once per document (`document_graph_extractions`), not per artifact.

## Data Migration (from AGE)

For existing installations migrating from Apache AGE:

```bash
# 1. Deploy Neo4j + Qdrant (empty)
docker compose up -d neo4j qdrant

# 2. Initialize Qdrant collections
python scripts/init_qdrant_collections.py

# 3. Migrate graph data: AGE → Neo4j
python scripts/migrate_age_to_neo4j.py

# 4. Run Alembic migration (adds qdrant_point_id columns)
./manage.sh --migrate
```

All migration scripts are idempotent (MERGE/upsert).

## Performance Tuning

### Single-Node Tuning Matrix

Use `docker compose --profile split up -d --build` for all tiers. This runs separate worker processes for ingest, embed, and graph queues.

| Tier | Hardware | Docling settings | Worker settings (split profile) | Queue/scheduling | UI polling |
|---|---|---|---|---|---|
| **S** (dev) | 8 vCPU, 32 GB RAM, no GPU | `DOCLING_DEVICE=cpu`, `DOCLING_DTYPE=float32`, `DOCLING_CONCURRENCY=1` | `WORKER_INGEST_CONCURRENCY=1`, `WORKER_EMBED_CONCURRENCY=1`, `WORKER_GRAPH_CONCURRENCY=1` | `WATCH_DIR_POLL_INTERVAL_SECONDS=60` | 5s poll interval |
| **M** (workstation) | 16 vCPU, 64 GB RAM, 1 GPU (24 GB+) | `DOCLING_DEVICE=cuda`, `DOCLING_DTYPE=bfloat16`, `DOCLING_CONCURRENCY=2` | `WORKER_INGEST_CONCURRENCY=2`, `WORKER_EMBED_CONCURRENCY=2`, `WORKER_GRAPH_CONCURRENCY=2` | Watcher 30s | 3–5s with backoff |
| **L** (server) | 32 vCPU, 128 GB RAM, 1 strong GPU (40–80 GB) | `DOCLING_DEVICE=cuda`, `DOCLING_DTYPE=bfloat16`, `DOCLING_CONCURRENCY=3` | `WORKER_INGEST_CONCURRENCY=3`, `WORKER_EMBED_CONCURRENCY=4`, `WORKER_GRAPH_CONCURRENCY=3` | Watcher 20–30s | 3s + backoff |
| **XL** (big server) | 48+ vCPU, 256 GB RAM, 2 GPUs | `DOCLING_DEVICE=cuda`, `DOCLING_DTYPE=bfloat16`, `DOCLING_CONCURRENCY=4` | `WORKER_INGEST_CONCURRENCY=4`, `WORKER_EMBED_CONCURRENCY=6`, `WORKER_GRAPH_CONCURRENCY=4` | Watcher 15–20s | 3s + backoff |

Start command: `docker compose --profile split up -d --build`

### Guardrails

1. **Keep `WORKER_INGEST_CONCURRENCY <= DOCLING_CONCURRENCY`** to avoid Docling-capacity retry storms. When ingest workers outnumber Docling permits, excess tasks retry-loop and can exhaust their retry budget.
2. **Use split workers** (`docker compose --profile split up -d --build`). The default single-worker mode shares concurrency across all queues.
3. **For CPU use `DOCLING_DTYPE=float32`; for GPU use `DOCLING_DTYPE=bfloat16`.** Do NOT use `bfloat32` — it is not a valid PyTorch dtype.
4. If capacity retries still fail, raise `PREPARE_MAX_RETRIES` as a temporary mitigation — the real fix is concurrency alignment.

### Multi-Node Scaling Matrix

| Tier | Cluster shape | Docling pool | Ingest pool (prepare/finalize) | Embed pool | Graph pool | Broker/DB notes |
|---|---|---|---|---|---|---|
| **MN-1** | 3 worker + 1 API node | 1 GPU replica, `DOCLING_CONCURRENCY=2` | 2 workers (`concurrency=1` each) | 1 worker (`concurrency=2`) | 1 worker (`concurrency=1`) | Single Redis/Postgres acceptable |
| **MN-2** | 6 worker + 2 API nodes | 2 GPU replicas, total permits=4 | 4 workers (`concurrency=1`) | 2 workers (`concurrency=2` each) | 1–2 workers (`concurrency=2`) | Redis HA (sentinel/managed), Postgres primary+replica |
| **MN-3** | 10 worker + 3 API nodes | 3 GPU replicas, total permits=6 | 6 workers (`concurrency=1`) | 3 workers (`concurrency=3`) | 2 workers (`concurrency=2`) | Managed Redis, Postgres tuned pools, Qdrant/Neo4j on dedicated hosts |
| **MN-4** | 16+ worker + 4 API nodes | 4 GPU replicas, total permits=8 | 8 workers (`concurrency=1`) | 4 workers (`concurrency=4`) | 3 workers (`concurrency=2`) | Separate stateful cluster tier (Redis/Postgres/Qdrant/Neo4j) |

### Multi-Node Rules

1. Keep **total ingest concurrency <= total Docling permits** across all nodes (prevents `Docling at capacity` retry storms).
2. Run **exactly one Beat scheduler** (the `beat` service) — never scale the Beat container.
3. Put the directory watcher on a **dedicated queue/worker** so scans never block ingest.
4. Use **split workers only** (ingest/embed/graph separated) — mixed-queue workers cause head-of-line blocking.
5. Add queue tuning: `worker_prefetch_multiplier=1`, worker recycle/memory caps.

### Autoscaling Triggers

1. Scale **ingest workers** when `ingest` queue depth stays > 2× available permits for 2+ minutes.
2. Scale **embed workers** when `embed` queue age > 60s.
3. Scale **graph workers** when `graph` queue age > 120s.
4. Scale **Docling replicas first** if the prepare stage dominates wall time.

## Implementation Phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Core data pipeline: upload → text extract → embed → semantic query | Complete |
| 2 | Multi-modal pipeline, graph extraction, all query modes, directory watcher | Complete |
| 2.5 | React web UI (upload, directory monitor, query), LangGraph agent endpoint | Complete |
| 2.6 | Cognee integration: trusted data query mode, dual-ingest pipeline step | Complete |
| 2.7 | Knowledge restructure: split vector tables, per-layer endpoints, unified query, docling-graph, trusted data governance, UI overhaul | Complete |
| 2.8 | Pipeline consolidation (manifest-first, parallel derivations, idempotent) + Retrieval upgrades (weighted fusion, chunk_links, image display) | Complete |
| 2.9 | Architecture upgrade: Neo4j + Qdrant + GraphRAG + expanded ontology + entity canonicalization | Complete |
| 3 | Auth (JWT + ABAC), governance workflow | Planned |
| 4 | Hardening, full test coverage, observability | Planned |
| 5 | Ontology versioning, CI/CD, advanced features | Planned |

## Project Structure

```
app/
├── api/v1/               # FastAPI routers
│   ├── retrieval.py      #   Unified query endpoint (strategy + modality_filter)
│   ├── _retrieval_helpers.py #   Retrieval pipeline helpers
│   ├── agent.py          #   LangGraph agent context endpoint
│   ├── _agent_helpers.py #   Agent response formatting
│   ├── graph_store.py    #   Graph entity/relationship ingest + query (Neo4j)
│   ├── memory.py         #   Trusted data proposals + approval + search
│   ├── governance.py     #   Feedback + patch state machine
│   ├── sources.py        #   Sources CRUD, document upload, watch dirs
│   └── health.py         #   Health check endpoint
├── services/
│   ├── cognee_service.py       # Cognee async wrapper
│   ├── docling_client.py       # HTTP client for Docling conversion service
│   ├── docling_graph_service.py # LLM-powered graph extraction → Neo4j import
│   ├── graphrag_service.py     # GraphRAG community detection, reports, search
│   ├── neo4j_graph.py          # Neo4j Cypher operations (sync + async)
│   ├── qdrant_store.py         # Qdrant vector upsert/search
│   ├── canonicalization.py     # Entity alias resolution + fuzzy match
│   ├── chunking.py             # Structure-aware document chunking
│   ├── ontology_templates.py   # YAML → Pydantic extraction templates + validation
│   └── ner.py                  # Military NER (offline regex, EM/RF patterns, fallback)
├── workers/
│   ├── pipeline.py       # Celery ingest pipeline (parallel text/image embed)
│   └── watcher.py        # Celery Beat directory watcher
├── models/               # SQLAlchemy ORM (ingest, retrieval, governance, auth, trusted_data)
└── schemas/              # Pydantic request/response schemas
docker/
├── docling/              # Docling VLM conversion service (granite-docling-258M)
├── neo4j/                # Neo4j init scripts (constraints, indexes)
└── postgres/             # Custom Postgres (pgvector)
ontology/
└── base.yaml             # Military equipment ontology (5 layers, 35+ types, 44 predicates)
scripts/
├── init_qdrant_collections.py    # Create Qdrant collections with indexes
├── migrate_age_to_neo4j.py       # One-time AGE → Neo4j migration
└── seed_ontology.py              # Seed ontology types from YAML
frontend/
├── src/components/       # React components
│   ├── QueryPage.tsx     #   Multi-strategy search with flat ranked results
│   ├── FileUpload.tsx    #   Document upload
│   ├── IngestPage.tsx    #   Unified ingest page
│   ├── GraphExplorer.tsx #   Graph search + entity/relationship creation (full ontology)
│   ├── MemoryPanel.tsx   #   Trusted data proposals + approval + search
│   ├── DirectoryMonitor.tsx # Watch directory management
│   └── Nav.tsx           #   Navigation
└── src/api/client.ts     # Typed API client (all endpoints)
tests/
├── unit/                 # Pure-logic tests (no DB required)
├── integration/          # API tests against real Postgres/Redis/MinIO/Neo4j/Qdrant stack
├── pipeline/             # Pipeline task tests
├── e2e/                  # End-to-end workflow tests
└── fixtures/             # Sample documents for test pipelines
```
