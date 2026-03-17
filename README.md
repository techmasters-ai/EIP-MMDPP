# EIP-MMDPP

Multi-modal document processing and retrieval platform for defense/military use cases.

Ingests PDFs, DOCX, images, and technical drawings → converts documents via Docling (granite-docling-258M VLM) → embeds text (BGE) and images (CLIP) into Qdrant vector collections, builds a military equipment knowledge graph (Neo4j), runs GraphRAG community detection and reporting, and maintains governed trusted data (dedicated Qdrant collection with human-review gate). Supports 6 retrieval modes: text basic, text only, images only, multi-modal, GraphRAG local, and GraphRAG global. Trusted data has its own query endpoint. Includes a user feedback → curator patch approval workflow and a React web UI.

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
                    │       Qdrant: eip_trusted_text             │
                    │   Trusted Data (human-reviewed, indexed)   │
                    │   BGE 1024-dim, cosine distance            │
                    │   PROPOSED → APPROVED_PENDING_INDEX →      │
                    │     APPROVED_INDEXED | INDEX_FAILED        │
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
| Text Embeddings | `BAAI/bge-large-en-v1.5` (1024-dim, asymmetric query/passage prefixes) |
| Image Embeddings | OpenCLIP ViT-B/32 (512-dim, cross-modal) |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder (CPU default, GPU optional) |
| Document Conversion | Docling + `ibm-granite/granite-docling-258M` VLM |
| Graph Extraction | Docling-Graph service (ontology-driven entity/relationship extraction via LLM structured outputs, port 8002) |
| GraphRAG | Microsoft graphrag (community detection, reports, local/global search) |
| Trusted Data | Dedicated Qdrant collection + Celery indexing (human-reviewed, vector-indexed) |
| Frontend | React 18 + TypeScript + Vite (TecMasters design system) |

All ML inference runs **fully locally** — no cloud API calls required (air-gapped deployment).

### Docker Services (10 containers)

| Service | Purpose |
|---|---|
| `api` | FastAPI application server |
| `worker` | Celery worker (ingest pipeline) |
| `beat` | Celery Beat (periodic tasks, GraphRAG indexing) |
| `postgres` | PostgreSQL 16 (metadata, chunk_links, governance) |
| `redis` | Celery broker + result backend |
| `minio` | S3-compatible object storage |
| `docling` | Document conversion service (granite-docling-258M VLM) |
| `docling-graph` | Ontology-driven entity/relationship extraction service (port 8002) |
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
./manage.sh --logs [service]     # Stream logs (api, worker, beat, postgres, redis, minio, docling, docling-graph, neo4j, qdrant)
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

A single `LLM_PROVIDER` env var controls the LLM backend for **all** LLM-dependent features (graph extraction, GraphRAG reports). Each feature specifies its own model via a dedicated env var.

| Value | Description |
|---|---|
| `ollama` | Uses local Ollama server. Fully air-gapped. Requires `OLLAMA_BASE_URL`. |
| `openai` | Uses OpenAI API. Requires `OPENAI_API_KEY`. |
| `mock` | Disables all LLM calls. Used in tests and environments without an LLM. |

```bash
# Air-gapped (Ollama) setup
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_NUM_CTX=16384                  # Context window for Ollama (must fit prompt + response)

# Per-feature model selection
GRAPHRAG_MODEL=llama3.2           # Model for GraphRAG community report generation

# Docling-Graph service (ontology-driven graph extraction)
DOCLING_GRAPH_BASE_URL=http://docling-graph:8002  # Docling-Graph service URL
DOCLING_GRAPH_TIMEOUT=300                         # HTTP timeout for extraction calls (seconds)
DOCLING_GRAPH_CONCURRENCY=2                       # Max concurrent extraction requests
GRAPH_NODE_MIN_CONFIDENCE=0.60                    # Min entity confidence for Neo4j import
GRAPH_REL_MIN_CONFIDENCE=0.55                     # Min relationship confidence for Neo4j import

DOCLING_FALLBACK_ENABLED=false            # Fall back to legacy extraction on Docling 5xx (default false)
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

### Trusted Data
- `POST /v1/trusted-data/ingest` — propose knowledge (status: PROPOSED)
- `GET /v1/trusted-data/proposals` — list submissions (filterable by status)
- `POST /v1/trusted-data/proposals/{id}/approve` — curator approves → enqueues Celery task to embed + index in Qdrant
- `POST /v1/trusted-data/proposals/{id}/reject` — curator rejects
- `POST /v1/trusted-data/proposals/{id}/reindex` — re-enqueue failed/pending indexing
- `POST /v1/trusted-data/query` — search approved trusted data (direct Qdrant vector search)

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
| `graphrag_local` | `all` | Text | Entity-centric + community reports | Entity matches with community context |
| `graphrag_global` | `all` | Text | Cross-community summarization | Community reports ranked by relevance |

> **Backward compatibility**: The legacy `mode` field (e.g. `"mode": "text_only"`) is still accepted and maps to the corresponding `strategy` + `modality_filter` combination.

The hybrid pipeline runs: parallel vector search (BGE + CLIP via Qdrant `asyncio.gather`) → document-structure expansion (chunk_links table) → ontology traversal (Neo4j entity relationships) → independent re-scoring of expanded chunks → weighted fusion scoring → deduplicate → cross-encoder reranking (bge-reranker-v2-m3) → min score threshold filter → rank → filter by modality.

Image-modality results include an `image_url` served via the API proxy (`GET /v1/images/{chunk_id}`), which streams from MinIO with 1-hour cache headers. This avoids exposing Docker-internal MinIO hostnames in presigned URLs and works in air-gapped environments without hostname configuration.

**Weighted Fusion Scoring**: `final = 0.65*semantic + 0.20*doc_structure + 0.15*ontology + MIL-ID bonus`. MIL-ID bonus matches NSN, MIL-STD, ELNOT, DIEQP, and AN/ designators. All weights are configurable via environment variables (see `env.example`). Results below `RETRIEVAL_MIN_SCORE_THRESHOLD` (default 0.25) are dropped. Top candidates are re-scored by a cross-encoder reranker (`RERANKER_MODEL`, default `BAAI/bge-reranker-v2-m3`, configurable via `RERANKER_DEVICE`, `RERANKER_ENABLED`, `RERANKER_TOP_N`).

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

See `ontology/ontology.yaml` for the full schema.

## GraphRAG

Community detection and cross-community search powered by Microsoft's `graphrag` library:

- **Indexing**: Celery Beat runs Leiden/Louvain community detection on the Neo4j graph, then generates LLM community reports (configurable interval via `GRAPHRAG_INDEXING_INTERVAL_MINUTES`). Manual trigger: `POST /v1/graphrag/index` dispatches the task immediately (Redis lock prevents overlapping runs)
- **Local search**: Entity-centric retrieval with community report context — finds relevant entities via Neo4j fulltext index and enriches results with their community summaries and full node properties
- **Global search**: Cross-community summarization — retrieves and ranks community reports for broad analytical questions

**Prerequisites**: GraphRAG queries require data in the knowledge graph:
- **Local search** needs entities in Neo4j — ensure documents have been ingested with successful graph extraction (`derive_ontology_graph` task must complete without errors)
- **Global search** needs community reports — ensure `GRAPHRAG_INDEXING_ENABLED=true` (default) and that at least one indexing cycle has run after entities exist in Neo4j
- If prerequisites are not met, the API returns explicit errors (404 for no entity matches, 409 for missing community reports) instead of silent empty results

## Web UI

React 18 + TypeScript + Vite single-page application served by the API container. The frontend builds as Stage 1 of the main Dockerfile (`node:22-alpine`).

### Search Documents (`QueryPage`)

Four query modes with a mode selector bar:

| Mode | Strategy | Description |
|---|---|---|
| **Text Basic** | `basic` | BGE vector search over text chunks |
| **Multi-Modal** | `hybrid` | Full multi-modal pipeline (text + image). Shows a modality sub-filter: All / Text Only / Images Only |
| **GraphRAG Local** | `graphrag_local` | Entity-centric search with community report context |
| **GraphRAG Global** | `graphrag_global` | Cross-community summarization for broad analytical questions |

**Result cards** show:
- Always-visible text preview (first ~300 chars of `content_text`)
- Inline image thumbnails for image-modality results (click for lightbox)
- Expandable "Show details" section with full text and all metadata (`chunk_id`, `artifact_id`, `document_id`, `score`, `modality`, `page_number`, `classification`, full `context` object)

**GraphRAG-specific exploration:**
- **Local results**: Entity properties table (name, type, confidence, artifact) + community reports list (title, summary) rendered inline
- **Global results**: Community title + level badge, expandable full report text

Images are served via the API proxy (`GET /v1/images/{chunk_id}`) which streams from MinIO — no Docker-internal hostnames exposed to the browser.

### Document Upload (`FileUpload`)

Drag-and-drop or click-to-upload with real-time pipeline status polling. Supports PDF, DOCX, PNG, JPG, TIFF. Adaptive polling intervals (2s → 5s → 10s) based on elapsed time. Retry button for FAILED/ERROR documents. When a source is selected, shows all historical documents for that source with live status updates and retry support.

### Other Pages

- **Ingest** — unified ingest page with upload + status overview
- **Directory Monitor** — register/remove watch directories for auto-ingest
- **Graph Explorer** — Neo4j entity/relationship search + manual creation (full ontology support)
- **Trusted Data** — submit, approve/reject, reindex, and search human-reviewed knowledge

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
- **Ingest dedup** — duplicate extracted elements (same modality+page+section+text+bbox) suppressed before persistence; text chunks deduplicated by content before embedding to prevent redundant Qdrant vectors
- **Retrieval diversity** — content-level deduplication across all search modes: `_text_vector_search` over-fetches candidates (`RETRIEVAL_DIVERSITY_OVERSAMPLE_FACTOR`, default 8×) then deduplicates by `(document_id, page_number, normalized_text)` keeping highest score; hybrid pipeline applies same diversity pass after chunk-id dedup
- **Dual vector store** — embeddings batch-upserted to Qdrant (single RPC per document) with `qdrant_point_id` cross-reference in Postgres
- **Batch Neo4j writes** — entities and relationships grouped by label and upserted via UNWIND (one Cypher call per label group instead of per-node)
- **Run/stage tracking** — `pipeline_runs` and `stage_runs` tables for diagnostics
- **Worker split** — optional queue isolation: `docker compose --profile split up`
- **Docling concurrency gate** — Redis semaphore with `DOCLING_CONCURRENCY` permits (default 1) controls parallel Docling conversions; queued tasks wait and retry instead of timing out; health check is advisory (logs warning but proceeds with conversion) to avoid starvation when the Docling service runs CPU-bound VLM conversion; health probe timeout configurable via `DOCLING_HEALTH_TIMEOUT` (default 5s)
- **Docling threadpool isolation** — The Docling service runs conversion in a threadpool (`run_in_threadpool`) so the `/health` endpoint remains responsive during CPU-bound VLM processing; an `asyncio.Semaphore` (capacity from `DOCLING_MAX_CONCURRENT`, default 1 on CPU) gates concurrent conversions and returns 503 when saturated
- **Configurable retries** — retry counts and delays for all pipeline stages configurable via env vars (`PREPARE_MAX_RETRIES`, `EMBED_MAX_RETRIES`, etc.); documents stay in PROCESSING status during retries and only show FAILED after all retries are exhausted; Docling 503 (busy) and `SoftTimeLimitExceeded` retries do NOT consume the retry budget
- **Chord resilience** — derivation tasks (text/image embeddings, graph extraction) return error dicts instead of raising on terminal failure, ensuring the chord callback and `finalize_document` always execute; `SoftTimeLimitExceeded` caught explicitly to return gracefully; chord `on_error` errback marks document FAILED if a hard time limit kills a chord member
- **Truncated JSON repair** — LLM graph extraction output truncated by token limits is automatically repaired via `json-repair` before falling back to `DeterministicExtractionError`
- **Recursive chunk splitting** — when a graph extraction chunk fails with deterministic LLM error, the chunk is recursively halved (2500→1250→625, floor 600 chars) and each sub-chunk retried; partial graph from successful chunks/sub-chunks still allows COMPLETE status; only total failure of all chunks triggers PARTIAL_COMPLETE
- **Batched text embedding + Qdrant upserts** — large documents (thousands of text elements) no longer send all vectors in a single Qdrant RPC; embedding and upserts are batched via `EMBED_TEXT_BATCH_SIZE` and `QDRANT_UPSERT_BATCH_SIZE` (default 128 each); Qdrant client timeout configurable via `QDRANT_TIMEOUT_SECONDS` (default 60s)
- **Stage run attempt tracking** — each retry creates a separate `stage_runs` row with incrementing `attempt` number, preserving full retry history per stage
- **Task time limits** — `soft_time_limit` / `time_limit` on all tasks read from env-var settings at registration time (not hardcoded), ensuring `.env` tuning takes effect without code changes
- **Stale run cleanup** — on worker startup, documents stuck in PROCESSING (from prior crashes) are reset to PENDING and their PipelineRuns marked FAILED via Celery `worker_ready` signal
- **Re-upload on failure** — re-uploading a file that previously FAILED removes the old record and re-ingests (no 409)
- **Reingest safety** — reingest endpoint rejects requests when pipeline is already PROCESSING (409); failure handlers use the task's own `run_id` to avoid cross-run contamination
- **Concurrent dispatch prevention** — atomic `FOR UPDATE` check in `start_ingest_pipeline()` prevents duplicate PipelineRun creation; document-scoped Redis singleflight lock in `prepare_document` prevents concurrent execution; supersession guard aborts stale tasks from prior cleanup cycles
- **Celery visibility timeout** — `CELERY_VISIBILITY_TIMEOUT` (default 10800s / 3h) prevents Redis from redelivering long-running tasks that appear stuck
- **Worker topology isolation** — `manage.sh` auto-stops opposite worker set when switching between single and split modes
- **Idempotent artifact persistence** — `_persist_extraction_results` uses `ON CONFLICT DO UPDATE` with deterministic artifact IDs so reingest/retry never fails with PK collision; image storage keys are deterministic (`artifacts/{doc_id}/images/{artifact_id}.{ext}`) to prevent MinIO object churn; `classification` is preserved on conflict (never overwritten by extraction)
- **Terminal status handling** — UI polling stops for all terminal states (COMPLETE, ERROR, FAILED, PARTIAL_COMPLETE); FAILED shows red badge, PARTIAL_COMPLETE shows amber warning badge with error context

The `prepare_document` task calls the dedicated Docling service which extracts text, tables, images, equations, and schematics in a single VLM pass. If the Docling service is unavailable and `DOCLING_FALLBACK_ENABLED=true`, the pipeline falls back to legacy extraction.

Graph extraction is performed by the **Docling-Graph service** (port 8002), which handles ontology-driven entity/relationship extraction via LLM structured outputs, text chunking, deduplication, and confidence scoring. The pipeline's `derive_ontology_graph` task sends document text to Docling-Graph via HTTP and imports the returned entities/relationships into Neo4j. Entities below `GRAPH_NODE_MIN_CONFIDENCE` (default 0.60) and relationships below `GRAPH_REL_MIN_CONFIDENCE` (default 0.55) are filtered at import time. Graph data is stored once per document (`document_graph_extractions`), not per artifact. The extraction task runs on a dedicated `graph_extract` queue, separate from downstream graph tasks.

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
| 2.6 | Trusted data layer: governed knowledge with human-review gate | Complete |
| 2.7 | Knowledge restructure: split vector tables, per-layer endpoints, unified query, docling-graph, trusted data governance, UI overhaul | Complete |
| 2.8 | Pipeline consolidation (manifest-first, parallel derivations, idempotent) + Retrieval upgrades (weighted fusion, chunk_links, image display) | Complete |
| 2.9 | Architecture upgrade: Neo4j + Qdrant + GraphRAG + expanded ontology + entity canonicalization | Complete |
| 2.10 | Docling-graph fixes (chunked extraction, property persistence, word-boundary mentions, queue isolation) + Trusted Data simplification (Cognee → Qdrant-backed, Celery indexing) | Complete |
| 2.11 | Graph extraction hardening (fail-closed, retry/backoff, concurrency gate) + Docling health-check fix (threadpool, advisory probe) + Search UI overhaul (4-mode selector, modality sub-filter, GraphRAG entity/report exploration, image proxy, result card improvements) + Polling fix | Complete |
| 2.12 | LLM extraction reliability: Ollama structured outputs (full JSON schema via `format`), direct httpx (removed LiteLLM), deterministic error classification (skip retries for empty/non-JSON), Docling 5xx fallback gate | Complete |
| 2.13 | Retrieval fixes: text preview hydration (chunk_text in Qdrant payload + Postgres backfill), image URL prefix fix, GraphRAG precondition checks (404/409 instead of silent empty) | Complete |
| 2.14 | Docling 503 storm fix: increased timeouts (30 min for large PDFs), fixed concurrency=1 to match Docling capacity, SoftTimeLimitExceeded no longer consumes retry budget, 503 uses 5-min backoff | Complete |
| 2.15 | GraphRAG report generation: LiteLLM → direct Ollama httpx (matching extraction path), manual indexing trigger (`POST /v1/graphrag/index`), removed litellm dependency. Ingest page: historical document listing per source with live status polling and retry. | Complete |
| 2.16 | Pipeline performance: batch Qdrant upserts, batch Neo4j UNWIND writes, duplicate image upload elimination, Celery prefetch=1, graph chunk size 5000→2.5× fewer LLM calls, split worker profile in manage.sh, 16GB GPU .env optimization | Complete |
| 2.17 | Pipeline stabilization: Docling timeout 300→1500s, in-task 503 retry loop (no budget consumed), truncated LLM JSON repair with entity filtering (json-repair), chord tasks return error dicts (finalize always runs), deterministic VlmPipeline failures fail-fast, GraphRAG routed off ingest worker, GraphRAG local search uses fulltext index, graph chunk size 2500 / max tokens 1200 | Complete |
| 2.18 | Runtime config alignment: Celery task time limits read from settings (not hardcoded in decorators), GraphRAG local search passes sync DB session for community report enrichment, fulltext entity results preserve all node fields, stale PROCESSING/RUNNING cleanup on worker startup | Complete |
| 2.19 | Large document resilience: batched text embedding + Qdrant upserts (EMBED_TEXT_BATCH_SIZE, QDRANT_UPSERT_BATCH_SIZE, QDRANT_TIMEOUT_SECONDS), recursive chunk splitting on graph extraction failure (2500→1250→625 chars), stage_run marked FAILED before retry (no stale RUNNING rows), config defaults aligned with .env | Complete |
| 2.20 | Search result diversity: content-level dedup in all search modes (over-fetch + diversify by doc/page/text), ingest-time element dedup (conservative modality+page+section+text+bbox key), text chunk dedup before embedding | Complete |
| 2.21 | Large-document timeout fix: `DOCLING_TIMEOUT_SECONDS` 300→3600, `PREPARE_SOFT_TIME_LIMIT` 1800→4200, `DOCLING_LOCK_TIMEOUT` 1800→4200 (90-page PDFs take ~30 min on CPU), pinned docling==2.76.0/docling-core==2.67.1, full traceback logging on Docling errors | Complete |
| 2.22 | Chord resilience fix: `SoftTimeLimitExceeded` handlers on all chord member tasks (return error dict instead of dying), chord `on_error` errback marks document FAILED on hard kills, `GRAPH_SOFT_TIME_LIMIT` 600→1800 / `GRAPH_TIME_LIMIT` 660→1860 for large-document LLM extraction | Complete |
| 2.23 | Concurrent pipeline dispatch fix: atomic PipelineRun check-and-set (`FOR UPDATE`), document-scoped singleflight Redis lock in `prepare_document`, supersession guard aborts stale tasks, Docling lock held through `self.retry()` (no gap for lock theft), configurable 503 retry limit (`DOCLING_503_MAX_RETRIES=20`), Celery Redis visibility timeout (`CELERY_VISIBILITY_TIMEOUT=10800`), stale cleanup marks PipelineRuns FAILED, worker topology overlap prevention in `manage.sh`, Docling `MAX_CONCURRENT` aligned with pipeline `DOCLING_CONCURRENCY` | Complete |
| 2.24 | Comprehensive quality pass: enriched ontology properties (descriptions/examples/patterns), validation matrix (all relationship types), fixed extraction prompt (valid few-shot, property descriptions, type restrictions), post-extraction validation (_validate_entity_types, _validate_properties), BGE asymmetric query/passage prefixes, cross-encoder reranker (bge-reranker-v2-m3), structure-aware chunking in pipeline, min score threshold + image oversample, GraphRAG global fulltext filtering + local BM25 scoring, fuzzy match score normalization, independent re-scoring of expanded chunks, model pre-download in manage.sh, upgraded to llama3.1:8b | Complete |
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
│   ├── trusted_data.py   #   Trusted data proposals + approval + indexing + search
│   ├── governance.py     #   Feedback + patch state machine
│   ├── sources.py        #   Sources CRUD, document upload, watch dirs
│   └── health.py         #   Health check endpoint
├── services/
│   ├── docling_client.py       # HTTP client for Docling conversion service
│   ├── docling_graph_service.py # HTTP client for Docling-Graph extraction service
│   ├── graphrag_service.py     # GraphRAG community detection, reports, search
│   ├── neo4j_graph.py          # Neo4j Cypher operations (sync + async)
│   ├── qdrant_store.py         # Qdrant vector upsert/search
│   ├── canonicalization.py     # Entity alias resolution + fuzzy match
│   ├── chunking.py             # Structure-aware document chunking
│   ├── reranker.py             # Cross-encoder reranker (bge-reranker-v2-m3)
│   ├── ontology_templates.py   # YAML → Pydantic extraction templates + validation
│   └── storage.py              # MinIO storage operations
├── workers/
│   ├── pipeline.py             # Celery ingest pipeline (parallel text/image embed)
│   ├── trusted_data_tasks.py   # Celery task for trusted data embedding + Qdrant indexing
│   └── watcher.py              # Celery Beat directory watcher
├── models/               # SQLAlchemy ORM (ingest, retrieval, governance, auth, trusted_data)
└── schemas/              # Pydantic request/response schemas
docker/
├── docling/              # Docling VLM conversion service (granite-docling-258M)
├── docling-graph/        # Docling-Graph extraction service (ontology-driven, port 8002)
├── neo4j/                # Neo4j init scripts (constraints, indexes)
└── postgres/             # Custom Postgres (pgvector)
ontology/
└── ontology.yaml         # Military equipment ontology (5 layers, 35+ types, 44 predicates)
scripts/
├── init_qdrant_collections.py    # Create Qdrant collections with indexes
├── migrate_age_to_neo4j.py       # One-time AGE → Neo4j migration
└── seed_ontology.py              # Seed ontology types from YAML
frontend/
├── src/components/       # React components
│   ├── QueryPage.tsx     #   Multi-strategy search (4 modes, modality sub-filter, GraphRAG exploration)
│   ├── FileUpload.tsx    #   Document upload
│   ├── IngestPage.tsx    #   Unified ingest page
│   ├── GraphExplorer.tsx #   Graph search + entity/relationship creation (full ontology)
│   ├── TrustedDataPanel.tsx #   Trusted data submissions + approval + indexing + search
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
