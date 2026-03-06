# EIP-MMDPP

Multi-modal document processing and retrieval platform for defense/military use cases.

Ingests PDFs, DOCX, images, and technical drawings → converts documents via Docling (granite-docling-258M VLM) → embeds text (BGE) and images (CLIP) into separate vector stores, builds a military equipment knowledge graph (Apache AGE), and maintains governed trusted data (Cognee). Supports 5 retrieval modes: text basic, text only, images only, multi-modal, and trusted data search. Includes a user feedback → curator patch approval workflow and a React web UI.

## Architecture

### Knowledge Layers

```
                    ┌──────────────────────────────────────────┐
                    │          Apache AGE Graph (eip_kg)        │
                    │   DOCUMENT ←→ CHUNK_REF nodes             │
                    │   Ontology entities (LLM-extracted)       │
                    │   CONTAINS_TEXT / CONTAINS_IMAGE /         │
                    │   SAME_PAGE / EXTRACTED_FROM / ontology    │
                    └──────────┬───────────────┬────────────────┘
                               │               │
                    ┌──────────▼──────┐ ┌──────▼──────────┐
                    │ retrieval.      │ │ retrieval.       │
                    │ text_chunks     │ │ image_chunks     │
                    │ BGE 1024-dim    │ │ CLIP 512-dim     │
                    │ HNSW index      │ │ HNSW index       │
                    └─────────────────┘ └─────────────────┘

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
| Database | PostgreSQL 16 with pgvector (HNSW) + Apache AGE (openCypher graph) |
| Object Storage | MinIO |
| Text Embeddings | `BAAI/bge-large-en-v1.5` (1024-dim, fully local) |
| Image Embeddings | OpenCLIP ViT-B/32 (512-dim, cross-modal) |
| Document Conversion | Docling + `ibm-granite/granite-docling-258M` VLM |
| Graph Extraction | docling-graph + LLM (ontology-driven entity/relationship extraction) |
| Trusted Data | Cognee (NetworkX graph + LanceDB vector, governed approval workflow) |
| Frontend | React 18 + TypeScript + Vite (TecMasters design system) |

All ML inference runs **fully locally** — no cloud API calls required (air-gapped deployment).

## Quickstart

```bash
# 1. Copy environment config and set required values
cp .env.example .env
# Edit .env — at minimum set LLM_PROVIDER and (if openai) OPENAI_API_KEY

# 2. Start all services (builds images, runs migrations, waits for health)
./manage.sh --start

# 3. API + web UI
#    Web UI:  http://localhost:8000/
#    API docs: http://localhost:8000/docs
```

## manage.sh — Project Management CLI

All service lifecycle, database, worker, and test operations are available through `./manage.sh`:

```bash
# Service lifecycle
./manage.sh --start              # Build and start all services; wait for health
./manage.sh --stop               # Stop all services (preserves data)
./manage.sh --restart            # Restart without rebuilding images
./manage.sh --status             # Show service status and health checks
./manage.sh --logs [service]     # Stream logs (api, worker, beat, postgres, redis, minio, docling)
./manage.sh --blow-away          # Destroy everything: containers, volumes, data

# Database
./manage.sh --migrate            # Run alembic upgrade head
./manage.sh --seed               # Run ontology seeder
./manage.sh --db-shell           # Open interactive psql shell

# Workers
./manage.sh --worker-status      # Show Celery worker/beat task info

# Testing (delegates to scripts/run_tests.sh)
./manage.sh --test               # Full suite
./manage.sh --test unit          # Unit tests only
./manage.sh --test integration   # Integration tests
./manage.sh --test e2e           # End-to-end tests
```

## LLM Provider Configuration

A single `LLM_PROVIDER` env var controls the LLM backend for **all** LLM-dependent features (graph extraction, Cognee trusted data). Each feature specifies its own model via a dedicated env var.

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
COGNEE_MODEL=llama3.2              # Model for Cognee trusted data operations
```

## Running Tests

```bash
# Full suite (unit → integration → contract → E2E)
./scripts/run_tests.sh

# Individual layers
./scripts/run_tests.sh unit
./scripts/run_tests.sh integration
./scripts/run_tests.sh e2e

# Keep stack running after tests
KEEP_STACK=1 ./scripts/run_tests.sh
```

## API Endpoints (v1)

### Sources & Document Upload
- `POST /v1/sources` — create a document collection
- `POST /v1/sources/{id}/documents` — upload a document (streams to MinIO, triggers pipeline; 409 on duplicate file within same source)
- `GET /v1/documents/{id}/status` — poll pipeline status

### Directory Watcher
- `POST /v1/watch-dirs` — register a directory for auto-ingest
- `DELETE /v1/watch-dirs/{id}` — remove watch directory

### Text Vector Store
- `POST /v1/text/ingest` — embed and store a text chunk directly (BGE)
- `POST /v1/text/query` — semantic search on text_chunks

### Image Vector Store
- `POST /v1/images/ingest` — embed and store an image directly (CLIP)
- `POST /v1/images/query` — semantic search on image_chunks (text-to-image or image-to-image)

### Graph Store (Apache AGE)
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
  "mode": "multi_modal",
  "top_k": 10,
  "include_context": true
}
```

Response returns a flat ranked results list:

```json
{
  "mode": "multi_modal",
  "results": [
    { "chunk_id": "...", "score": 0.92, "modality": "text", "content_text": "..." },
    { "chunk_id": "...", "score": 0.78, "modality": "image", "content_text": "..." }
  ],
  "total": 2
}
```

Query modes:

| Mode | Input | Pipeline | Output |
|---|---|---|---|
| `text_basic` | Text only | BGE vector search | Text chunks |
| `text_only` | Text or image | Full multi-modal pipeline | Filtered to text |
| `images_only` | Text or image | Full multi-modal pipeline | Filtered to images |
| `multi_modal` | Text or image | Full multi-modal pipeline | All results |
| `memory` | Text | Cognee search | Approved trusted data |

The multi-modal pipeline (modes 2-4) runs: vector search (BGE + CLIP) → cross-modal graph bridging (structural edges) → ontology traversal (entity relationships, 4 hops) → deduplicate → rank by score → filter by mode.

### Agent / LangGraph Context

```
GET /v1/agent/context
  ?query=Patriot+PAC-3+guidance+computer
  &mode=text_basic
  &top_k=10
  &include_sources=true
```

Returns a pre-formatted markdown context string for direct injection into an LLM prompt. Supports all 5 query modes.

```python
# LangGraph usage example
resp = requests.get("http://localhost:8000/v1/agent/context",
                    params={"query": query, "mode": "text_basic"})
system_msg = f"Use this context:\n\n{resp.json()['context']}"
```

### Governance
- `POST /v1/feedback` — submit a correction on a retrieved result
- `POST /v1/patches/{id}/approve` — curator approves a patch
- `POST /v1/patches/{id}/apply` — apply an approved patch

All Apache AGE graph mutations (node/edge create, update, delete) require **dual-curator approval**. Text and classification corrections require a single curator.

## Ingest Pipeline

```
validate_and_store
  → detect_modalities
  → convert_document          (Docling + granite-docling-258M — unified extraction)
  → embed_text_chunks         ┐
  → embed_image_chunks        ┘ (parallel via Celery chord)
  → extract_graph             (docling-graph + LLM entity/relationship extraction)
  → import_graph              (NetworkX → Apache AGE)
  → connect_document_elements (DOCUMENT/CHUNK_REF/SAME_PAGE + EXTRACTED_FROM edges)
  → finalize_artifact
```

The `convert_document` task calls the dedicated Docling service which extracts text, tables, images, equations, and schematics in a single VLM pass. If the Docling service is unavailable and `DOCLING_FALLBACK_ENABLED=true`, the pipeline falls back to legacy extraction.

Text and image embedding run in parallel. Graph extraction uses LLM (via `LLM_PROVIDER`) for ontology-driven entity/relationship extraction, with regex NER as fallback when LLM is unavailable.

## Implementation Phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Core data pipeline: upload → text extract → embed → semantic query | Complete |
| 2 | Multi-modal pipeline, graph extraction, all query modes, directory watcher | Complete |
| 2.5 | React web UI (upload, directory monitor, query), LangGraph agent endpoint | Complete |
| 2.6 | Cognee integration: trusted data query mode, dual-ingest pipeline step | Complete |
| 2.7 | Knowledge restructure: split vector tables, per-layer endpoints, unified query, docling-graph, trusted data governance, UI overhaul | Complete |
| 3 | Auth (JWT + ABAC), governance workflow | Planned |
| 4 | Hardening, full test coverage, observability | Planned |
| 5 | Ontology versioning, CI/CD, advanced features | Planned |

## Project Structure

```
app/
├── api/v1/               # FastAPI routers
│   ├── retrieval.py      #   Unified multi-mode query endpoint
│   ├── text_store.py     #   Text vector store ingest + query
│   ├── image_store.py    #   Image vector store ingest + query
│   ├── graph_store.py    #   Graph entity/relationship ingest + query
│   ├── memory.py         #   Trusted data proposals + approval + search
│   ├── agent.py          #   LangGraph agent context endpoint
│   ├── governance.py     #   Feedback + patch state machine
│   └── sources.py        #   Sources CRUD, document upload, watch dirs
├── services/
│   ├── cognee_service.py       # Cognee async wrapper
│   ├── docling_client.py       # HTTP client for Docling conversion service
│   ├── docling_graph_service.py # LLM-powered graph extraction (docling-graph)
│   ├── ontology_templates.py   # YAML → Pydantic extraction templates
│   ├── ner.py                  # Military NER (offline regex, fallback)
│   └── graph.py                # Apache AGE Cypher helpers
├── workers/
│   ├── pipeline.py       # Celery ingest pipeline (parallel text/image embed)
│   └── watcher.py        # Celery Beat directory watcher
├── models/               # SQLAlchemy ORM (ingest, retrieval, governance, auth, trusted_data)
└── schemas/              # Pydantic request/response schemas
docker/
└── docling/              # Docling VLM conversion service (granite-docling-258M)
frontend/
├── src/components/       # React components
│   ├── QueryPage.tsx     #   Multi-mode search with flat ranked results
│   ├── FileUpload.tsx    #   Document upload
│   ├── TextIngest.tsx    #   Direct text chunk ingest
│   ├── ImageIngest.tsx   #   Direct image ingest with drag-drop
│   ├── GraphExplorer.tsx #   Graph search + entity/relationship creation
│   ├── MemoryPanel.tsx   #   Trusted data proposals + approval + search
│   ├── DirectoryMonitor.tsx # Watch directory management
│   └── Nav.tsx           #   Navigation (6 pages)
└── src/api/client.ts     # Typed API client (all endpoints)
tests/
├── unit/                 # Pure-logic tests (no DB required)
├── integration/          # API tests against real Postgres/Redis/MinIO stack
├── pipeline/             # Pipeline task tests
├── e2e/                  # End-to-end workflow tests
└── fixtures/             # Sample documents for test pipelines
```
