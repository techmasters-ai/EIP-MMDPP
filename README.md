# EIP-MMDPP

Multi-modal document processing and retrieval platform for defense/military use cases.

Ingests PDFs, DOCX, images, and technical drawings → extracts text, performs OCR, embeds into a vector DB, and builds a military equipment knowledge graph. Supports semantic, graph, hybrid, cross-modal, and Cognee-powered retrieval. Includes a user feedback → curator patch approval workflow and a React web UI.

## Architecture

| Component | Technology |
|---|---|
| API | FastAPI (Python 3.11) |
| Processing | Celery + Redis |
| Database | PostgreSQL 16 with pgvector (HNSW) + Apache AGE (openCypher graph) |
| Object Storage | MinIO |
| Text Embeddings | `BAAI/bge-large-en-v1.5` (1024-dim, fully local) |
| Image Embeddings | OpenCLIP ViT-B/32 (512-dim, cross-modal) |
| OCR | Tesseract 5 → EasyOCR fallback |
| Schematics | llava via Ollama (local VLM) |
| Knowledge Graph AI | Cognee (NetworkX graph + LanceDB vector, no extra services) |
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
./manage.sh --logs [service]     # Stream logs (api, worker, beat, postgres, redis, minio, ollama)
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

### LLM Provider Configuration

The `LLM_PROVIDER` setting controls the backend for Cognee's knowledge graph AI:

| Value | Description |
|---|---|
| `openai` | Uses OpenAI API. Requires `OPENAI_API_KEY`. |
| `ollama` | Uses local Ollama server. Requires `OLLAMA_BASE_URL` and `OLLAMA_LLM_MODEL`. Fully air-gapped. |
| `mock` | Disables all LLM/Cognee calls. Used in tests and environments without an LLM. |

```bash
# Air-gapped (Ollama) setup
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_LLM_MODEL=llama3.2
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

### Sources & Ingest
- `POST /v1/sources` — create a document collection
- `POST /v1/sources/{id}/documents` — upload a document (streams to MinIO)
- `GET /v1/documents/{id}/status` — poll pipeline status

### Directory Watcher
- `POST /v1/watch-dirs` — register a directory for auto-ingest
- `DELETE /v1/watch-dirs/{id}` — remove watch directory

### Retrieval
```json
POST /v1/retrieval/query
{
  "query": "Patriot PAC-3 guidance computer specifications",
  "mode": "hybrid",
  "top_k": 10
}
```
Modes: `semantic` | `graph` | `hybrid` | `cross_modal`

### Agent / LangGraph Context

```
GET /v1/agent/context
  ?query=Patriot+PAC-3+guidance+computer
  &mode=hybrid
  &top_k=10
  &include_sources=true
```

Returns a pre-formatted markdown context string for direct injection into an LLM prompt. Modes: `semantic` | `graph` | `hybrid` | `cross_modal` | `cognee_graph`.

```python
# LangGraph usage example
resp = requests.get("http://localhost:8000/v1/agent/context",
                    params={"query": query, "mode": "hybrid"})
system_msg = f"Use this context:\n\n{resp.json()['context']}"
```

The `cognee_graph` mode runs LLM-enhanced knowledge graph reasoning via Cognee across four search strategies (GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, SUMMARIES) in parallel. Cognee is also fed documents automatically during ingest.

### Governance
- `POST /v1/feedback` — submit a correction on a retrieved result
- `POST /v1/patches/{id}/approve` — curator approves a patch
- `POST /v1/patches/{id}/apply` — apply an approved patch

All Apache AGE graph mutations (node/edge create, update, delete) require **dual-curator approval**. Text and classification corrections require a single curator.

## Ingest Pipeline

```
validate_and_store
  → detect_modalities
  → [chord: extract_text | extract_images | run_ocr | process_schematics]
  → collect_metadata
  → chunk_and_embed
  → extract_graph_entities
  → import_graph
  → finalize_artifact
  → ingest_to_cognee   (non-fatal: failure logs a warning, does not affect pipeline status)
```

## Implementation Phases

| Phase | Scope | Status |
|---|---|---|
| 1 | Core data pipeline: upload → text extract → embed → semantic query | Complete |
| 2 | Multi-modal pipeline, graph extraction, all query modes, directory watcher | Complete |
| 2.5 | React web UI (upload, directory monitor, query), LangGraph agent endpoint | Complete |
| 2.6 | Cognee integration: `cognee_graph` query mode, dual-ingest pipeline step | Complete |
| 3 | Auth (JWT + ABAC), governance workflow | Planned |
| 4 | Hardening, full test coverage, observability | Planned |
| 5 | Ontology versioning, CI/CD, advanced features | Planned |

## Project Structure

```
app/
├── api/v1/           # FastAPI routers (retrieval, agent, governance, sources, watch-dirs)
│   └── _agent_helpers.py  # Pure helpers (no DB imports — unit-testable)
├── services/
│   ├── cognee_service.py  # Cognee async wrapper
│   ├── ner.py             # Military NER (offline regex)
│   └── graph.py           # Apache AGE Cypher helpers
├── workers/
│   ├── pipeline.py        # Celery multi-modal ingest pipeline
│   └── watcher.py         # Celery Beat directory watcher
├── models/            # SQLAlchemy ORM (ingest, retrieval, governance, auth)
└── schemas/           # Pydantic request/response schemas
frontend/
├── src/components/    # React components (FileUpload, QueryPage, DirectoryMonitor, Nav)
└── src/api/client.ts  # Typed API client
tests/
├── unit/              # Pure-logic tests (no DB required)
├── integration/       # API tests against real Postgres/Redis/MinIO stack
└── fixtures/          # Sample documents for test pipelines
```
