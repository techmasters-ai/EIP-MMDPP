# EIP-MMDPP

Multi-modal document processing and retrieval platform for defense/military use cases.

Ingests PDFs, DOCX, PPTX, XLSX, HTML, Markdown, CSV, images, and technical drawings → converts documents via Docling (PdfPipeline + dlparse_v4 + EasyOCR) → extracts LLM-generated document metadata (summary, date, classification, source characterization) and picture descriptions via Ollama → embeds text (BGE-M3 via Ollama) and images (CLIP) into Qdrant vector collections → builds a military equipment knowledge graph (Neo4j) via parallel ontology-driven entity/relationship extraction → runs GraphRAG community detection and reporting → maintains governed trusted data (dedicated Qdrant collection with human-review gate). Supports 8 retrieval modes: text basic, hybrid (text only / images only / multi-modal), GraphRAG local, GraphRAG global, GraphRAG drift, and GraphRAG basic. Includes a user feedback → curator patch approval workflow, document cancel/delete lifecycle, and a React web UI.

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
| Text Embeddings | `bge-m3:latest` via Ollama `/v1/embeddings` API (1024-dim) |
| Image Embeddings | OpenCLIP EVA02-E-14-plus (1024-dim, cross-modal) |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder (GPU-accelerated) |
| Document Conversion | Docling PdfPipeline (dlparse_v4 + EasyOCR + TableFormer), SimplePipeline for Office/HTML/MD |
| Document Analysis | LLM-based metadata extraction (summary, date, classification, source) + multimodal picture descriptions via Ollama |
| Graph Extraction | Docling-Graph service (chunked entity extraction across 5 ontology groups in parallel + global relationship pass on full text, with few-shot examples and ontology-derived validation, port 8002) |
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
GRAPHRAG_LLM_MODEL=llama3.2      # Model for GraphRAG community report generation
GRAPHRAG_LLM_PROVIDER=ollama     # ollama | openai (defaults to LLM_PROVIDER if not set)

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

### Health & Operations

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/health` | Liveness probe — returns `{"status": "ok"}` |
| `GET` | `/v1/health/ready` | Readiness probe — checks Postgres, Redis, MinIO; returns `{"status": "ready"|"degraded", "checks": {...}}` |

### Sources & Document Upload

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/sources` | Create a document source/collection |
| `GET` | `/v1/sources` | List all sources |
| `POST` | `/v1/sources/{id}/documents` | Upload document (multipart file → MinIO, triggers pipeline; 409 on duplicate unless previous FAILED) |
| `GET` | `/v1/sources/{id}/documents` | List documents in a source |
| `GET` | `/v1/documents/{id}/status` | Poll pipeline status (includes stage summary) |
| `POST` | `/v1/documents/batch-status` | Batch status check for multiple document IDs |
| `GET` | `/v1/documents/{id}/stages` | Detailed pipeline stage diagnostics (per-stage status, attempt, metrics, error) |
| `POST` | `/v1/documents/{id}/reingest` | Re-run pipeline — `{"mode": "full|embeddings_only|graph_only"}`; 409 if already PROCESSING |
| `POST` | `/v1/documents/{id}/cancel` | Cancel PROCESSING document — revokes Celery tasks, cleans up all data stores |
| `DELETE` | `/v1/documents/{id}` | Hard-delete a non-processing document and all derived data |
| `DELETE` | `/v1/sources/{id}/documents` | Delete all documents in a source (409 if any are PROCESSING) |
| `GET` | `/v1/documents/{id}/metadata` | LLM-extracted document metadata (summary, date, classification, source) |
| `GET` | `/v1/documents/{id}/docling` | DoclingDocument viewer (markdown + JSON + image injection) |
| `GET` | `/v1/documents/{id}/docling-raw` | Raw DoclingDocument JSON stream with base64 images |
| `GET` | `/v1/documents/{id}/translation` | Translated markdown for non-English documents |
| `GET` | `/v1/documents/{id}/element-translations` | Per-element translations for DoclingViewer overlay |
| `GET` | `/v1/documents/{id}/image-descriptions` | LLM-generated descriptions for image elements |
| `GET` | `/v1/documents/{id}/artifacts` | List extracted artifacts (images, tables, schematics) |
| `GET` | `/v1/artifacts/{id}` | Get single artifact by ID |
| `GET` | `/v1/documents/{id}/artifacts/{artifact_id}/image` | Stream artifact image from MinIO |

**Create a source and upload a document:**

```python
import requests

BASE = "http://localhost:8000/v1"

# Create source
source = requests.post(f"{BASE}/sources", json={"name": "intel-reports", "description": "Field reports"}).json()

# Upload document
with open("report.pdf", "rb") as f:
    resp = requests.post(
        f"{BASE}/sources/{source['id']}/documents",
        files={"file": ("report.pdf", f, "application/pdf")},
    )
doc = resp.json()

# Poll until complete
import time
while True:
    status = requests.get(f"{BASE}/documents/{doc['id']}/status").json()
    print(status["pipeline_status"])
    if status["pipeline_status"] in ("COMPLETE", "FAILED", "ERROR", "PARTIAL_COMPLETE"):
        break
    time.sleep(5)
```

### Directory Watcher

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/watch-dirs` | Register a directory for auto-ingest |
| `GET` | `/v1/watch-dirs` | List all registered watch directories |
| `DELETE` | `/v1/watch-dirs/{id}` | Remove watch directory |

Per-directory `poll_interval_seconds` respected (directories only scanned when enough time has elapsed since last scan).

### Graph Store (Neo4j)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/graph/ingest/entity` | Create/update an entity node |
| `POST` | `/v1/graph/ingest/relationship` | Create/update a relationship edge |
| `POST` | `/v1/graph/query` | Search knowledge graph by entity name with hop traversal |
| `POST` | `/v1/graph/neighborhood` | Get entity's full neighborhood graph for visualization |

```python
# Search the knowledge graph
resp = requests.post(f"{BASE}/graph/query", json={
    "query": "S-300",
    "top_k": 10,
    "hop_count": 2,
})
for node in resp.json():
    print(node["modality"], node["content_text"], node["score"])
```

### Ontology Registry & Query Profiles

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/query-profiles/registries` | List all registries |
| `POST` | `/v1/query-profiles/registries` | Create a new registry |
| `GET` | `/v1/query-profiles/registries/{id}` | Get a single registry |
| `PUT` | `/v1/query-profiles/registries/{id}` | Update a registry |
| `POST` | `/v1/query-profiles/registries/{id}/activate` | Set as the active registry |
| `POST` | `/v1/query-profiles/registries/{id}/profiles` | Add a query profile |
| `PUT` | `/v1/query-profiles/registries/{id}/profiles/{pid}` | Update a query profile |
| `DELETE` | `/v1/query-profiles/registries/{id}/profiles/{pid}` | Delete a query profile |
| `GET` | `/v1/query-profiles` | Get active registry's exposed profiles |
| `GET` | `/v1/query-profiles/default-template` | Get starter template with pre-built profiles |
| `POST` | `/v1/query-profiles/search/section` | Execute a section profile graph traversal |
| `POST` | `/v1/query-profiles/search/dossier` | Execute a dossier (multi-section) search |

### Trusted Data

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/trusted-data/ingest` | Propose knowledge for trusted data layer (status: PROPOSED) |
| `GET` | `/v1/trusted-data/proposals` | List submissions (filterable by `?status=`) |
| `POST` | `/v1/trusted-data/proposals/{id}/approve` | Curator approves → enqueues Celery task to embed + index in Qdrant |
| `POST` | `/v1/trusted-data/proposals/{id}/reject` | Curator rejects submission |
| `POST` | `/v1/trusted-data/proposals/{id}/reindex` | Re-enqueue failed/pending indexing |
| `POST` | `/v1/trusted-data/query` | Search approved trusted data (Qdrant vector search) |

```python
# Submit trusted knowledge
submission = requests.post(f"{BASE}/trusted-data/ingest", json={
    "content": "The SA-2 Guideline uses a Fan Song fire control radar.",
    "source_context": "Field manual FM-2024-001",
    "confidence": 0.95,
}).json()

# Approve it
requests.post(f"{BASE}/trusted-data/proposals/{submission['id']}/approve", json={"notes": "Verified"})

# Search trusted data
results = requests.post(f"{BASE}/trusted-data/query", json={
    "query": "SA-2 fire control radar",
    "top_k": 5,
}).json()
```

### Feedback & Governance

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/feedback` | Submit a correction on a retrieved result (auto-creates patch) |
| `GET` | `/v1/patches` | List patches (filterable by `?state=`) |
| `GET` | `/v1/patches/{id}` | Get a patch by ID |
| `POST` | `/v1/patches/{id}/approve` | Curator approves a patch (dual-approval required for graph mutations) |
| `POST` | `/v1/patches/{id}/reject` | Curator rejects a patch |
| `POST` | `/v1/patches/{id}/apply` | Apply an approved patch to the data platform |

All Neo4j graph mutations (node/edge create, update, delete) require **dual-curator approval**. Text and classification corrections require a single curator.

### Unified Retrieval

**Endpoint:** `POST /v1/retrieval/query`

**Request schema (`UnifiedQueryRequest`):**

| Field | Type | Default | Description |
|---|---|---|---|
| `query_text` | `string` | — | Text query (max 4096 chars). Required unless `query_image` is provided. |
| `query_image` | `string` | `null` | Base64-encoded PNG/JPG image or artifact UUID reference. Only used by `hybrid` strategy. |
| `strategy` | `string` | `"basic"` | One of: `basic`, `hybrid`, `graphrag_local`, `graphrag_global`, `graphrag_drift`, `graphrag_basic` |
| `modality_filter` | `string` | `"all"` | `all`, `text`, or `image`. Only affects `hybrid` strategy. |
| `top_k` | `int` | `10` | Number of results to return (1-100) |
| `include_context` | `bool` | `true` | Include full chunk text in results |
| `min_confidence` | `float` | `null` | Minimum score threshold (0.0-1.0); defaults to server config |
| `reranker_top_n` | `int` | `null` | Candidates to rerank (1-200); defaults to server config |
| `filters` | `object` | `null` | Filter by `classification`, `modalities`, `source_ids`, `document_ids` |

**Response schema (`UnifiedQueryResponse`):**

| Field | Type | Description |
|---|---|---|
| `query_text` | `string` | Echo of the query text |
| `query_image` | `string` | Truncated echo of query image (first 100 chars) |
| `strategy` | `string` | Strategy used |
| `modality_filter` | `string` | Modality filter applied |
| `results` | `array` | Ranked list of `QueryResultItem` objects (see below) |
| `total` | `int` | Number of results returned |

**Result item schema (`QueryResultItem`):**

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `uuid` | Chunk identifier (null for GraphRAG responses) |
| `artifact_id` | `uuid` | Source artifact identifier |
| `document_id` | `uuid` | Source document identifier |
| `score` | `float` | Relevance score |
| `modality` | `string` | `text`, `image`, `table`, `schematic`, `image_description`, or `graphrag_response` |
| `content_text` | `string` | Chunk text or LLM-generated response |
| `page_number` | `int` | Source page number |
| `classification` | `string` | Security classification (default: `UNCLASSIFIED`) |
| `context` | `object` | Graph neighbors, GraphRAG context (`{source, graphrag_context: {entities, community_reports}}`) |
| `image_url` | `string` | API proxy URL for image results (`/v1/images/{chunk_id}`) |

**Query strategies:**

| Strategy | Modality Filter | Input | Pipeline | Output | Speed |
|---|---|---|---|---|---|
| `basic` | `all` | Text only | BGE vector search (Qdrant) | Ranked text/table chunks | Fast (1-3s) |
| `hybrid` | `all`/`text`/`image` | Text and/or image | Full multi-modal pipeline | Mixed text, image, table, schematic chunks | Medium (5-15s) |
| `graphrag_local` | `all` | Text only | Entity-centric + community reports | Single LLM-generated response with entity context | Slow (30-90s) |
| `graphrag_global` | `all` | Text only | Cross-community summarization | Single LLM-generated multi-paragraph summary | Slow (30-90s) |
| `graphrag_drift` | `all` | Text only | Community-informed expansion (DRIFT) | Single LLM-generated in-depth analysis | Slowest (30-120s) |
| `graphrag_basic` | `all` | Text only | Vector search over text units | Single LLM-generated concise answer | Medium (5-15s) |

> **Backward compatibility**: The legacy `mode` field (e.g. `"mode": "text_only"`) is still accepted and maps to the corresponding `strategy` + `modality_filter` combination.

#### 1. Basic Text Query

BGE vector search over text chunks in Qdrant. No LLM calls, no graph expansion.

```python
import requests

resp = requests.post("http://localhost:8000/v1/retrieval/query", json={
    "query_text": "radar signal processing specifications",
    "strategy": "basic",
    "top_k": 5,
    "include_context": True,
})
data = resp.json()
for item in data["results"]:
    print(f"[{item['score']:.3f}] {item['modality']}: {item['content_text'][:100]}")
```

#### 2. Multi-Modal Query (Hybrid)

Full pipeline: BGE text + CLIP image search → Neo4j graph expansion → ontology traversal → weighted fusion scoring → cross-encoder reranking. Accepts text, base64 image, or both.

```python
import base64, requests

# Text-only hybrid (searches text AND images)
resp = requests.post("http://localhost:8000/v1/retrieval/query", json={
    "query_text": "VHF radar internal components",
    "strategy": "hybrid",
    "modality_filter": "all",  # "all", "text", or "image"
    "top_k": 5,
})
data = resp.json()
for item in data["results"]:
    print(f"[{item['score']:.3f}] {item['modality']}: {item['content_text'][:100]}")
    if item.get("image_url"):
        print(f"  Image: http://localhost:8000{item['image_url']}")

# Image-based search (base64-encoded)
with open("photo.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()
resp = requests.post("http://localhost:8000/v1/retrieval/query", json={
    "query_image": image_b64,
    "strategy": "hybrid",
    "modality_filter": "image",
    "top_k": 5,
})

# Combined text + image search
resp = requests.post("http://localhost:8000/v1/retrieval/query", json={
    "query_text": "missile launcher diagram",
    "query_image": image_b64,
    "strategy": "hybrid",
    "modality_filter": "all",
    "top_k": 5,
})
```

#### 3. GraphRAG Local Query

Entity-centric search with community context reports. Returns a detailed LLM-generated response. Requires at least one successful GraphRAG indexing cycle.

```python
import requests

resp = requests.post("http://localhost:8000/v1/retrieval/query", json={
    "query_text": "What are the key components of the S-300 air defense system?",
    "strategy": "graphrag_local",
    "top_k": 10,
}, timeout=120)

data = resp.json()
# GraphRAG returns a single result with modality="graphrag_response"
result = data["results"][0]
print(result["content_text"])  # LLM-generated detailed explanation
ctx = result.get("context", {}).get("graphrag_context", {})
print(f"Entities: {len(ctx.get('entities', []))}")
print(f"Community reports: {len(ctx.get('community_reports', []))}")
```

#### 4. GraphRAG Global Query

Cross-community summarization for broad, holistic questions. Synthesizes answers across all community boundaries.

```python
import requests

resp = requests.post("http://localhost:8000/v1/retrieval/query", json={
    "query_text": "What are the major categories of air defense systems and how do they compare?",
    "strategy": "graphrag_global",
    "top_k": 10,
}, timeout=120)

data = resp.json()
print(data["results"][0]["content_text"])  # Multi-paragraph summary
```

#### 5. GraphRAG Drift Query

Community-informed expansion search (Microsoft DRIFT algorithm). Iteratively expands across communities for nuanced queries.

```python
import requests

resp = requests.post("http://localhost:8000/v1/retrieval/query", json={
    "query_text": "How do radar warning receivers interact with electronic countermeasure systems?",
    "strategy": "graphrag_drift",
    "top_k": 10,
}, timeout=180)

data = resp.json()
print(data["results"][0]["content_text"])  # In-depth analysis
```

#### 6. GraphRAG Basic Query

Vector search over GraphRAG-extracted text units. Fastest GraphRAG method, works with partial indexing.

```python
import requests

resp = requests.post("http://localhost:8000/v1/retrieval/query", json={
    "query_text": "SA-2 Guideline missile specifications",
    "strategy": "graphrag_basic",
    "top_k": 10,
}, timeout=60)

data = resp.json()
print(data["results"][0]["content_text"])  # Concise answer
```

#### Using Filters

All strategies support optional filters to narrow results:

```python
import requests

resp = requests.post("http://localhost:8000/v1/retrieval/query", json={
    "query_text": "radar specifications",
    "strategy": "basic",
    "top_k": 10,
    "min_confidence": 0.5,
    "reranker_top_n": 30,
    "filters": {
        "classification": "UNCLASSIFIED",
        "modalities": ["text", "table"],
        "document_ids": ["550e8400-e29b-41d4-a716-446655440000"],
    },
})
```

### Image Proxy

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/images/{chunk_id}` | Stream image from MinIO for an image chunk (1-hour cache) |
| `GET` | `/v1/images/artifact/{artifact_id}` | Stream image from MinIO by artifact ID (direct lookup) |

Image-modality results include an `image_url` served via the API proxy, which streams from MinIO with 1-hour cache headers. This avoids exposing Docker-internal MinIO hostnames in presigned URLs and works in air-gapped environments.

### Retrieval Settings

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/settings/retrieval` | Query defaults: `{top_k, reranker_top_n, min_confidence}` |
| `GET` | `/v1/settings/graphrag` | GraphRAG config: `{indexing_enabled, indexing_interval_minutes, last_indexing_at}` |

The hybrid pipeline runs: parallel vector search (BGE + CLIP via Qdrant `asyncio.gather`) → document-structure expansion (chunk_links table) → ontology traversal (Neo4j entity relationships) → independent re-scoring of expanded chunks → weighted fusion scoring → deduplicate → cross-encoder reranking (bge-reranker-v2-m3) → min score threshold filter → rank → filter by modality.

Image-modality results include an `image_url` served via the API proxy (`GET /v1/images/{chunk_id}`), which streams from MinIO with 1-hour cache headers. This avoids exposing Docker-internal MinIO hostnames in presigned URLs and works in air-gapped environments without hostname configuration.

**Weighted Fusion Scoring**: `final = 0.65*semantic + 0.20*doc_structure + 0.15*ontology + MIL-ID bonus`. MIL-ID bonus matches NSN, MIL-STD, ELNOT, DIEQP, and AN/ designators. All weights are configurable via environment variables (see `env.example`). Results below `RETRIEVAL_MIN_SCORE_THRESHOLD` (default 0.25) are dropped. Top candidates are re-scored by a cross-encoder reranker (`RERANKER_MODEL`, default `BAAI/bge-reranker-v2-m3`, configurable via `RERANKER_DEVICE`, `RERANKER_ENABLED`, `RERANKER_TOP_N`).

### Multi-Modal Query Walkthrough

**Example query:** `"VHF radar internal components"` with `strategy=hybrid`, `modality_filter=all`

**Step 1 — Parallel vector search (seeds)**

Two searches run concurrently via `asyncio.gather`:

- **BGE text search** — query embedded with `BAAI/bge-large-en-v1.5`, searched against `eip_text_chunks` (1024-dim cosine). Matches text chunks, table chunks, and **image description sections** (all stored as BGE vectors in the same collection). Over-fetches by 8× for diversity, filters below 0.25, content-deduplicates, reranks, returns top-k.
- **CLIP image search** — query embedded with OpenCLIP ViT-B/32 text encoder, searched against `eip_image_chunks` (512-dim cosine). Matches images by pixel similarity to the text concept. Scores are typically lower (0.1–0.3) because CLIP cross-modal alignment is loose.

Seeds from both searches are merged (highest score per chunk_id kept).

**Step 2 — Per-seed graph expansion**

For each seed, three strategies run (bounded to 16 concurrent):

- **Document-structure expansion** (Postgres `chunk_links` table) — follows pre-computed structural links: `NEXT_CHUNK` (reading order), `SAME_SECTION` (under same heading), `SAME_ARTIFACT` (from same image/table), `SAME_PAGE` (text ↔ image on same page). If an image description section is a seed, `SAME_ARTIFACT` surfaces sibling sections and `SAME_PAGE` surfaces the original CLIP image chunk.
- **Cross-modal bridging** (Neo4j, fallback only) — for legacy documents without chunk_links. Traverses structural edges up to 3 hops to bridge text ↔ image.
- **Ontology traversal** (Neo4j knowledge graph) — follows entity relationships. If a chunk mentions "S-75 Dvina" and the graph has `S-75 Dvina –[VARIANT_OF]→ SA-2 Guideline`, chunks about "SA-2 Guideline" are surfaced with per-relation weights from `ontology.yaml`.

**Step 3 — Re-score ontology-expanded chunks**

Ontology-expanded chunks get their actual BGE cosine similarity to the query computed (replacing the inherited parent score), then re-run through the fusion formula to preserve ontology relation weights.

**Step 4 — Fusion scoring**

All results (seeds + expanded) scored with:
```
final = 0.65 × semantic + 0.20 × doc_structure + 0.15 × ontology + MIL-ID bonus
```
Hop penalty decays scores for multi-hop expansions. Military identifier bonus (+0.03) fires when query and content share AN/ designators, NSNs, or MIL-STD numbers.

**Step 5 — Deduplicate → filter → sort → rerank**

- Deduplicate by chunk_id (keep highest score)
- Content-level diversification (same text on same page deduplicated)
- Filter by `modality_filter` (text includes `text`, `table`, `image_description`; image includes `image`, `schematic`)
- Sort by score descending, cap at `top_k`
- Cross-encoder reranking (`bge-reranker-v2-m3`) on top candidates

**Step 6 — Image URL resolution**

- `modality="image"` or `"schematic"` → `image_url = /v1/images/{chunk_id}` (the chunk IS the image)
- `modality="image_description"` (hybrid strategy only) → batch-lookup `ImageChunk` by `artifact_id` → `image_url = /v1/images/{image_chunk_id}` (links to the original image the description was generated from)
- `modality="image_description"` in basic strategy → no `image_url` (text-only result)

**Example result set:**

| Rank | Score | Modality | Content | Image |
|---|---|---|---|---|
| 1 | 0.999 | `image_description` | "Executive Summary: The image depicts the internal components of a VHF radar system..." | Original radar photo via `image_url` |
| 2 | 0.952 | `image_description` | "Intelligence Value: This image is of moderate intelligence value..." | Same radar photo |
| 3 | 0.920 | `image_description` | "Radar / Sensor Analysis: The image reveals components consistent with a VHF radar..." | Same radar photo |
| 4 | 0.253 | `image` | CLIP image description of SA-2 deployment map | Deployment map via `image_url` |
| 5 | 0.001 | `text` | "The P-12 Spoon Rest radar operates in the VHF band..." | None |

Image description sections (ranks 1–3) share the same `artifact_id` and `image_url` — they're different analytical sections of the same image's LLM-generated description. The CLIP image result (rank 4) is a different image found by pixel similarity. The text result (rank 5) is a regular text chunk.

### Agent / LangGraph Context

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/agent/context` | Pre-formatted markdown context string for LLM prompt injection |

Query params: `query` (required), `strategy`, `modality_filter`, `top_k`, `include_sources`.

Returns `{query, strategy, modality_filter, total_results, context, sources}`. Supports all query strategies.

```python
# LangGraph usage example
resp = requests.get("http://localhost:8000/v1/agent/context",
                    params={"query": "Patriot PAC-3 guidance", "strategy": "basic", "top_k": 10})
data = resp.json()
system_msg = f"Use this context:\n\n{data['context']}"
```

## Ontology

The knowledge graph uses a 5-layer ontology grounded in DoDAF DM2 concepts:

1. **Reference & Provenance** — Documents, sections, figures, tables, assertions
2. **Military Equipment** — Platforms, systems, subsystems, components
3. **EM/RF Signal & Radar** — Emissions, waveforms, modulation, antennas, receivers, processing
4. **Weapon / Missile / AAA** — Missiles, seekers, guidance, propulsion, artillery
5. **Operational / Capability** — Capabilities, engagement timelines, performance measures

46 entity types, 50 relationship predicates, enforced via validation matrix at graph write time.

See `ontology/ontology.yaml` for the full schema.

## Creating Custom Queries

Custom queries let you define deterministic graph traversal patterns that surface specific entity relationships from Neo4j. Once created, they appear as options in the Search Documents dropdown alongside the built-in retrieval modes.

### Overview

The system works in three layers:

1. **Ontology Registry** — Holds your ontology definition (entity types, relationship types, validation rules).
2. **Section Profiles** — Define a single traversal pattern (e.g., "find all components of a system").
3. **Dossier Profiles** — Bundle multiple section profiles into a compound report (e.g., "full system dossier = components + RF parameters + performance").

### Step-by-Step: Creating Your First Custom Query

#### Step 1: Create an Ontology Registry

1. Navigate to the **Ontology & Query Profiles** page.
2. Click **Create New Registry**.
3. Give it a name (e.g., "My Custom Ontology").
4. In the **Ontology Definition** JSON editor, paste or edit your ontology. You can start with the repository default by clicking **Load Default Ontology**.
5. Click **Save** and then **Activate** the registry.

The ontology definition must include `entity_types` and `relationship_types`. Here is a minimal example:

```json
{
  "version": "1.0",
  "entity_types": [
    {
      "name": "PLATFORM",
      "description": "Military platform or vehicle",
      "properties": {
        "properties": {
          "designation": { "type": "string" },
          "country_of_origin": { "type": "string" }
        }
      }
    },
    {
      "name": "RADAR_SYSTEM",
      "description": "Radar or sensor system",
      "properties": {
        "properties": {
          "designation": { "type": "string" },
          "frequency_band": { "type": "string" }
        }
      }
    }
  ],
  "relationship_types": [
    {
      "name": "HAS_COMPONENT",
      "description": "System contains this component",
      "source_type": "PLATFORM",
      "target_type": "RADAR_SYSTEM"
    }
  ]
}
```

#### Step 2: Create a Section Profile

A section profile defines one traversal pattern from a root entity through the graph.

1. Switch to the **Query Profiles** tab (enabled once an active registry exists).
2. Click **Add Section Profile**.
3. Fill in the form:
   - **ID**: `platform_radars` (unique identifier, no spaces)
   - **Label**: `Platform Radars` (shown in the Search dropdown)
   - **Root Entity Types**: Select `PLATFORM` (the types a user can search for)
   - **Target Entity Types**: Select `RADAR_SYSTEM` (the types to return as results)
   - **Traversal**: Add one step:
     - **Direction**: Outward
     - **Relationship Types**: `HAS_COMPONENT`
     - **Min/Max Hops**: 1 / 2
   - **Exposed**: Check this box so it appears in the Search dropdown
4. Click **Save**.

**What this does**: When a user types "Patriot" in the Search page with this profile selected, the system:
1. Resolves "Patriot" to a `PLATFORM` entity in Neo4j (via alias + fulltext matching).
2. Traverses outward along `HAS_COMPONENT` edges (1-2 hops).
3. Filters results to only `RADAR_SYSTEM` entity types.
4. Returns matching radar systems with their properties, aliases, and supporting evidence.

#### Step 3: Create a Dossier Profile (Optional)

A dossier bundles multiple section profiles into one compound query.

1. Click **Add Dossier Profile**.
2. Fill in:
   - **ID**: `system_dossier`
   - **Label**: `Full System Dossier`
   - **Root Entity Types**: Select all system types (`PLATFORM`, `RADAR_SYSTEM`, etc.)
   - **Section Profiles**: Select the section profiles to include (e.g., `platform_radars`, `system_components`, `rf_parameters`)
   - **Exposed**: Check this box
3. Click **Save**.

When a user runs this dossier, each section executes independently and results are grouped by section in the response.

#### Step 4: Use in Search

1. Go to the **Search Documents** page.
2. Open the **Query mode** dropdown — your custom profiles appear under a "Query Profiles" group.
3. Select your profile (e.g., "Platform Radars").
4. Type an entity name (e.g., "S-300" or "Patriot").
5. Click **Search**.

Results show entity properties, relationship paths, aliases, and evidence from source documents.

### Using Starter Profiles

Instead of building profiles from scratch, you can seed pre-built profiles:

1. On the **Query Profiles** tab, click **Load Starter Profiles**.
2. The system generates section and dossier profiles based on the active ontology's relationship types.
3. Review the generated profiles and edit them as needed.
4. The default starter profiles include:
   - **System Components** — finds subsystems, components, and assemblies
   - **RF Parameters** — finds frequency bands, emissions, waveforms, and antennas
   - **System Performance** — finds capabilities, performance specs, and engagement timelines
   - **System Dossier** — combines all three sections above

### Key Concepts

**Root Entity Types**: The entity types a user can search for. If a user types "AN/MPQ-65", the system looks for entities matching that name among the root types.

**Target Entity Types**: The entity types returned as results. After finding the root entity, the traversal follows relationship edges and returns only entities matching the target types.

**Traversals**: Each traversal is a sequence of steps through the graph. A step specifies:
- **Direction**: `out` (follow edges forward) or `in` (follow edges backward)
- **Relationship Types**: Which edge types to follow (e.g., `HAS_COMPONENT`, `SPECIFIED_BY`)
- **Min/Max Hops**: How far to traverse (1-4 hops)

Multiple traversals create parallel paths. For example, to find both components and specifications of a system, define two traversals: one following `HAS_COMPONENT` and another following `SPECIFIED_BY`.

**Exposed**: Only profiles marked as "exposed" appear in the Search Documents dropdown. Use this to keep work-in-progress profiles hidden while developing them.

### API Usage (curl / Python)

All query profile operations are available via REST API. Below are examples for the most common workflows.

**Create a registry with an ontology:**

```bash
# Create and activate a registry
curl -X POST http://localhost:8000/v1/query-profiles/registries \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Custom Ontology",
    "ontology_version": "1.0",
    "ontology_definition": {
      "version": "1.0",
      "entity_types": [
        {"name": "PLATFORM", "description": "Military platform", "properties": {"properties": {"designation": {"type": "string"}}}},
        {"name": "RADAR_SYSTEM", "description": "Radar system", "properties": {"properties": {"designation": {"type": "string"}}}}
      ],
      "relationship_types": [
        {"name": "HAS_COMPONENT", "source_type": "PLATFORM", "target_type": "RADAR_SYSTEM"}
      ]
    },
    "is_active": true
  }'
```

**Add a section profile:**

```bash
# Add a section profile to the active registry (replace REGISTRY_ID)
curl -X POST http://localhost:8000/v1/query-profiles/registries/REGISTRY_ID/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "id": "platform_radars",
    "label": "Platform Radars",
    "kind": "section",
    "exposed": true,
    "root_entity_types": ["PLATFORM"],
    "target_entity_types": ["RADAR_SYSTEM"],
    "traversals": [
      {
        "steps": [
          {"direction": "out", "rel_types": ["HAS_COMPONENT"], "min_hops": 1, "max_hops": 2}
        ]
      }
    ]
  }'
```

**Search with a section profile:**

```bash
# Search for radars associated with "Patriot"
curl -X POST http://localhost:8000/v1/query-profiles/search/section \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": "platform_radars",
    "query_text": "Patriot",
    "top_k": 25,
    "include_aliases": true,
    "include_evidence": true,
    "evidence_top_k": 3
  }'
```

**Search with a dossier profile:**

```bash
# Run a full system dossier (returns multiple sections)
curl -X POST http://localhost:8000/v1/query-profiles/search/dossier \
  -H "Content-Type: application/json" \
  -d '{
    "profile_id": "system_dossier",
    "query_text": "S-300",
    "top_k": 25
  }'
```

**Python example (full workflow):**

```python
import requests

BASE = "http://localhost:8000/v1"

# 1. Create and activate a registry
registry = requests.post(f"{BASE}/query-profiles/registries", json={
    "name": "Defense Ontology",
    "ontology_definition": {
        "version": "1.0",
        "entity_types": [
            {"name": "PLATFORM", "description": "Military platform",
             "properties": {"properties": {"designation": {"type": "string"}}}},
            {"name": "RADAR_SYSTEM", "description": "Radar system",
             "properties": {"properties": {"frequency_band": {"type": "string"}}}},
        ],
        "relationship_types": [
            {"name": "HAS_COMPONENT", "source_type": "PLATFORM", "target_type": "RADAR_SYSTEM"},
        ],
    },
    "is_active": True,
}).json()
registry_id = registry["id"]

# 2. Add a section profile
requests.post(f"{BASE}/query-profiles/registries/{registry_id}/profiles", json={
    "id": "platform_radars",
    "label": "Platform Radars",
    "kind": "section",
    "exposed": True,
    "root_entity_types": ["PLATFORM"],
    "target_entity_types": ["RADAR_SYSTEM"],
    "traversals": [{"steps": [
        {"direction": "out", "rel_types": ["HAS_COMPONENT"], "min_hops": 1, "max_hops": 2},
    ]}],
})

# 3. Search for radars on the S-300 platform
result = requests.post(f"{BASE}/query-profiles/search/section", json={
    "profile_id": "platform_radars",
    "query_text": "S-300",
    "include_aliases": True,
    "include_evidence": True,
}).json()

print(f"Root entity: {result['resolved_root']['name']}")
for item in result["items"]:
    print(f"  {item['name']} ({item['entity_type']}) — "
          f"via {' -> '.join(item['relationship_types'])}")
```

**Get the starter template (pre-built profiles for the repository ontology):**

```bash
curl http://localhost:8000/v1/query-profiles/default-template | python -m json.tool
```

**List exposed profiles for the active registry:**

```bash
curl http://localhost:8000/v1/query-profiles
```

### Limitations

- Custom queries only search the Neo4j knowledge graph. They do not perform vector/semantic search.
- Entities must exist in Neo4j (via document ingestion + graph extraction) before they can be found.
- Changing the active ontology does not retroactively re-extract already-ingested documents. Re-ingest documents to apply a new ontology.

## GraphRAG

Community detection and cross-community search powered by Microsoft's `graphrag` library. GraphRAG owns the full extraction pipeline — it reads documents from Postgres, chunks them, extracts entities/relationships via LLM, detects communities (Leiden clustering), generates community reports, and produces text embeddings for search.

### Architecture

```
Postgres documents ──→ Bridge (export_all) ──→ input/documents.parquet
                                                    │
                                               GraphRAG Pipeline
                                                    │
                                        ┌───────────┴────────────┐
                                        ▼                        ▼
                                   First run:              Subsequent runs:
                                 IndexingMethod.Standard  is_update_run=True
                                   (full build)           (incremental delta)
                                        │                        │
                                        ▼                        ▼
                              output/entities.parquet    update_output/{ts}/delta/
                              output/relationships.parquet   → merged into output/
                              output/communities.parquet
                              output/community_reports.parquet
                              output/text_units.parquet
```

**Neo4j and GraphRAG serve different purposes**: Neo4j stores the document structure graph (chunk links, same-page edges) and ontology entities for real-time traversal during multi-modal search expansion. GraphRAG owns community-based search with its own LLM-extracted entity graph stored in Parquet files.

### Indexing

- **Scheduled**: Celery Beat runs indexing on configurable interval (`GRAPHRAG_INDEXING_INTERVAL_MINUTES`, default 1440)
- **Manual**: `POST /v1/graphrag/index` dispatches immediately (Redis lock prevents overlapping runs)
- **Incremental**: First run does full extraction; subsequent runs use `is_update_run=True` for delta-only processing. Previous output is backed up before updates; restored on failure
- **Stale lock cleanup**: Redis locks are automatically cleared on worker startup (prevents stuck locks from killed containers)
- **Dry-run**: Set `GRAPHRAG_DRY_RUN=true` to validate config without running indexing
- **Extraction prompt**: Configurable via `GRAPHRAG_EXTRACTION_PROMPT` env var (includes military ontology by default)
- **Method**: Standard (LLM-based, default) or Fast (NLP-based, `GRAPHRAG_USE_FAST_METHOD=true`)
- **Cache**: LLM response caching enabled by default (`GRAPHRAG_CACHE_ENABLED`); unchanged text units skip LLM calls on re-runs

### Search (4 modes)

| Mode | API Strategy | Description | Key Settings |
|---|---|---|---|
| **Local** | `graphrag_local` | Entity-centric retrieval with community context | `GRAPHRAG_LOCAL_RESPONSE_TYPE`, `GRAPHRAG_DYNAMIC_COMMUNITY_SELECTION` |
| **Global** | `graphrag_global` | Cross-community summarization for broad questions | `GRAPHRAG_GLOBAL_RESPONSE_TYPE`, `GRAPHRAG_DYNAMIC_COMMUNITY_SELECTION` |
| **DRIFT** | `graphrag_drift` | Community-informed expansion search | `GRAPHRAG_DRIFT_RESPONSE_TYPE` |
| **Basic** | `graphrag_basic` | Vector search over text units | `GRAPHRAG_BASIC_RESPONSE_TYPE` |

All response types, community level, and dynamic community selection are configurable via env vars.

### Async Query Execution

GraphRAG queries involve LLM calls that can take 1-3+ minutes. To prevent browser timeout errors, all 4 search modes use an async job pattern:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/retrieval/graphrag/submit` | Submit query as Celery task, returns `{job_id, status: "pending"}` (HTTP 202) |
| `GET` | `/v1/retrieval/graphrag/status/{job_id}` | Poll status: `pending`, `running`, `completed`, or `failed` |
| `GET` | `/v1/retrieval/graphrag/result/{job_id}` | Fetch full `UnifiedQueryResponse` when complete (409 if still running) |

The frontend polls with exponential backoff (1s → 10s cap). Results are stored in Redis for 24h. The synchronous `POST /v1/retrieval/query` endpoint still works for all strategies (backward compatible).

```python
import time, requests

BASE = "http://localhost:8000/v1"

# Step 1: Submit
job = requests.post(f"{BASE}/retrieval/graphrag/submit", json={
    "query_text": "What are the key components of the S-300 system?",
    "strategy": "graphrag_local",
}).json()
job_id = job["job_id"]

# Step 2: Poll with exponential backoff
delay = 1.0
while True:
    time.sleep(delay)
    status = requests.get(f"{BASE}/retrieval/graphrag/status/{job_id}").json()
    print(f"Status: {status['status']}")
    if status["status"] == "completed":
        break
    if status["status"] == "failed":
        print(f"Error: {status.get('error')}")
        break
    delay = min(delay * 1.5, 10.0)

# Step 3: Fetch result
result = requests.get(f"{BASE}/retrieval/graphrag/result/{job_id}").json()
print(result["results"][0]["content_text"])
```

### GraphRAG Indexing & Tuning

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/graphrag/index` | Dispatch GraphRAG indexing as Celery task (idempotent via Redis lock) |
| `POST` | `/v1/graphrag/tune` | Dispatch GraphRAG prompt auto-tuning as Celery task |

### Prerequisites
- GraphRAG queries require at least one successful indexing cycle (`GRAPHRAG_INDEXING_ENABLED=true`)
- API returns 409 for missing community reports instead of silent empty results

## Web UI

React 18 + TypeScript + Vite single-page application served by the API container. The frontend builds as Stage 1 of the main Dockerfile (`node:22-alpine`).

### Search Documents (`QueryPage`)

A dropdown selector groups query modes into two categories:

**Retrieval modes** (always available):

| Mode | Strategy | Description |
|---|---|---|
| **Text Basic** | `basic` | BGE vector search over text chunks |
| **Multi-Modal** | `hybrid` | Full multi-modal pipeline (text + image). Shows a modality sub-filter: All / Text Only / Images Only |
| **GraphRAG Local** | `graphrag_local` | Entity-centric search with community report context |
| **GraphRAG Global** | `graphrag_global` | Cross-community summarization for broad analytical questions |
| **GraphRAG Drift** | `graphrag_drift` | Community-informed expansion search |
| **GraphRAG Basic** | `graphrag_basic` | Vector search over text units |

**Query Profile modes** (appear when an active registry has exposed profiles):

Custom ontology-driven graph traversal queries. These are defined via the Ontology & Query Profiles page and appear automatically in the dropdown once exposed. See [Creating Custom Queries](#creating-custom-queries) below.

**Result cards** show:
- Always-visible text preview (first ~300 chars of `content_text`)
- Inline image thumbnails for image-modality results (click for lightbox)
- Expandable "Show details" section with full text and all metadata (`chunk_id`, `artifact_id`, `document_id`, `score`, `modality`, `page_number`, `classification`, full `context` object)

**GraphRAG-specific exploration:**
- **Local results**: Entity properties table (name, type, confidence, artifact) + community reports list (title, summary) rendered inline
- **Global results**: Community title + level badge, expandable full report text

Images are served via the API proxy (`GET /v1/images/{chunk_id}`) which streams from MinIO — no Docker-internal hostnames exposed to the browser.

### Document Upload (`FileUpload`)

Drag-and-drop or click-to-upload with real-time pipeline status polling. Supports PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, PNG, JPG, TIFF. Adaptive polling intervals (2s → 5s → 10s) based on elapsed time. Cancel button for PROCESSING documents (revokes Celery tasks, cleans up all data stores, removes the document). Retry button for FAILED/ERROR documents. Delete button for completed/failed documents. When a source is selected, shows all historical documents for that source with live status updates, cancel, retry, and delete support. DoclingDocument viewer for COMPLETE documents with bounding-box overlay, metadata panel (summary, date, classification), and plaintext fallback for .txt files. **Image description tooltips**: hovering over embedded images (PDF, DOCX, PPTX) shows LLM-generated descriptions via the Docling web component's built-in tooltip system; standalone image documents (.png, .jpeg) display the description in a persistent panel.

### Other Pages

- **Ingest** — unified ingest page with upload + status overview
- **Directory Monitor** — register/remove watch directories for auto-ingest
- **Graph Explorer** — Neo4j entity/relationship search + manual creation (full ontology support)
- **Ontology & Query Profiles** — create/manage ontology registries and build custom graph traversal queries (see [Creating Custom Queries](#creating-custom-queries))
- **Trusted Data** — submit, approve/reject, reindex, and search human-reviewed knowledge

## Ingest Pipeline

Manifest-first architecture with parallel derivation stages and idempotent writes:

```
prepare_document  (validate + detect + Docling convert + persist document_elements)
    ↓
detect_and_translate  (per-element language detection + LLM translation if non-English)
    ↓
derive_document_metadata  (LLM: summary, date, classification, source characterization)
    ↓
derive_picture_descriptions  (LLM: multimodal image descriptions with summary context)
    ↓
purge_document_derivations  (clean up prior run data for reingest)
    ↓
┌── derive_text_chunks_and_embeddings ──┐
│      (includes image description       │
│       section embedding pass)          │
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
- **Foreign language translation** — per-element language detection (`langdetect`) triggers LLM translation of non-English content; translated text replaces `DocumentElement.content_text` for downstream processing; original preserved in MinIO; classification marking detection runs against original text; DoclingViewer offers a "Translate" toggle
- **Image description text search** — LLM-generated image descriptions split into sections, embedded as BGE text vectors in `eip_text_chunks`, searchable via standard text queries; SAME_ARTIFACT chunk_links tie sections together for graph expansion; `image_url` resolved via artifact_id batch lookup so matching sections display with original image
- **Standalone image support** — uploading standalone image files (JPEG, PNG, TIFF, BMP, GIF, WEBP) synthesizes an image element when Docling returns 0 elements, enabling CLIP embedding, LLM description, and text search; DoclingViewer shows the image with AI Image Analysis description panel
- **Image description hover tooltips** — the `GET /v1/documents/{id}/docling-raw` endpoint injects LLM-generated descriptions as `{kind: "description"}` annotations into Docling JSON picture items; the `<docling-tooltip>` web component renders "AI Image Analysis" tooltips on hover over embedded images in the DoclingViewer
- **Image dedup by content** — element deduplication includes a hash of raw image bytes for image/schematic elements, preventing distinct images on the same page from being silently dropped (previously images with empty captions and null bounding boxes on the same page shared identical dedup keys)
- **Stage skip completeness** — all pipeline stages mark their stage_run COMPLETE even when skipping (e.g. `derive_picture_descriptions` on .txt files, `derive_document_metadata` when disabled), preventing false PARTIAL_COMPLETE from `finalize_document`
- **Stale run cleanup** — on worker startup, documents stuck in PROCESSING (from prior crashes) are reset to PENDING and their PipelineRuns marked FAILED via Celery `worker_ready` signal
- **Re-upload on failure** — re-uploading a file that previously FAILED removes the old record and re-ingests (no 409)
- **Reingest safety** — reingest endpoint rejects requests when pipeline is already PROCESSING (409); failure handlers use the task's own `run_id` to avoid cross-run contamination
- **Concurrent dispatch prevention** — atomic `FOR UPDATE` check in `start_ingest_pipeline()` prevents duplicate PipelineRun creation; document-scoped Redis singleflight lock in `prepare_document` prevents concurrent execution; supersession guard aborts stale tasks from prior cleanup cycles
- **Celery visibility timeout** — `CELERY_VISIBILITY_TIMEOUT` (default 10800s / 3h) prevents Redis from redelivering long-running tasks that appear stuck
- **Worker topology isolation** — `manage.sh` auto-stops opposite worker set when switching between single and split modes
- **Idempotent artifact persistence** — `_persist_extraction_results` uses `ON CONFLICT DO UPDATE` with deterministic artifact IDs so reingest/retry never fails with PK collision; image storage keys are deterministic (`artifacts/{doc_id}/images/{artifact_id}.{ext}`) to prevent MinIO object churn; `classification` is preserved on conflict (never overwritten by extraction)
- **Terminal status handling** — UI polling stops for all terminal states (COMPLETE, ERROR, FAILED, PARTIAL_COMPLETE); FAILED shows red badge, PARTIAL_COMPLETE shows amber warning badge with error context

The `prepare_document` task calls the dedicated Docling service which extracts text, tables, images, equations, and schematics in a single VLM pass. If the Docling service is unavailable and `DOCLING_FALLBACK_ENABLED=true`, the pipeline falls back to legacy extraction.

Graph extraction is performed by the **Docling-Graph service** (port 8002) via the `/extract-all` endpoint, which runs all 5 ontology groups in parallel (reference, equipment, rf_signal, weapon, operational) plus a relationship extraction pass — 6 LLM calls total instead of the previous ~45 sequential calls. Each group extracts ALL entity instances (multiple per type) via direct LLM calls with structured JSON output, bypassing the slow per-entity-type `run_pipeline` approach. The pipeline's `derive_ontology_graph` task makes a single HTTP call and imports the returned entities/relationships into Neo4j. Entities below `GRAPH_NODE_MIN_CONFIDENCE` (default 0.60) and relationships below `GRAPH_REL_MIN_CONFIDENCE` (default 0.55) are filtered at import time. Graph data is stored once per document (`document_graph_extractions`). Extraction runs on a dedicated `graph_extract` queue.

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
| 2.25 | Image description tooltips: hover tooltips on embedded images via Docling web component (`kind: "description"` annotation fix), persistent description panel for standalone image documents, image-descriptions API endpoint, `SoftTimeLimitExceeded` handler for `derive_picture_descriptions`, configurable picture description timeouts, stage_run status fix for skipped stages (prevents false PARTIAL_COMPLETE on .txt and other non-image documents) | Complete |
| 2.26 | Image description text search: LLM-generated image descriptions split into sections and embedded as BGE text vectors in `eip_text_chunks` (searchable via standard text queries), `image_description` modality with SAME_ARTIFACT chunk_links between sections, image URL resolution via artifact_id batch lookup, modality filter update. Graph expansion fixes: pass `query_text` to doc-structure fusion for military ID bonus, ontology re-scoring preserves relation weights, cross-modal expansion uses fusion formula, deprecated asyncio API fixes, configurable cross-modal LIMIT, dead code cleanup. Picture description timeout tripled (3h). GraphRAG LLM timeout configurable (default 3h). DoclingViewer page centering. | Complete |
| 2.27 | Foreign language translation: per-element language detection (`langdetect`) + LLM translation in ingest pipeline, translated text stored alongside original in MinIO, all downstream stages (metadata, chunking, embeddings, ontology) operate on English translation, classification marking detection uses original text, translation API endpoint, DoclingViewer "Translate" toggle with language banner | Complete |
| 2.28 | GraphRAG architecture overhaul: separated bridge input/ from GraphRAG output/ (fixed overwrite bug), fixed `_get_method()` double-suffix bug (`standard-update-update`), removed dead Neo4j entity/relationship bridge exports (Path A — GraphRAG owns extraction), fixed double-chunking (pass full documents not pre-chunked), fixed auto-tune prompt filename mismatch, incremental indexing with pre-update backup/restore, stale Redis lock cleanup on worker startup, configurable response types per search method, cache toggle, fast method option, dry-run mode, dynamic community selection, covariates support, increased default cluster size (10→50) | Complete |
| 2.29 | Async GraphRAG queries: submit/poll/fetch pattern via Celery tasks to eliminate browser timeout on long-running LLM queries (1-3+ min), exponential backoff polling in frontend, Redis singleton + job tracking for bogus ID detection, Literal-typed status schema, monkey-patch for GraphRAG incremental indexing `update_final_documents` NaN bug, 29 unit tests covering task + endpoints + helpers | Complete |
| 2.30 | Image display fixes: standalone image files (JPEG/PNG/TIFF/etc.) synthesize an image element when Docling returns 0 elements (CLIP embed + LLM describe + text search), DoclingViewer shows image + description panel; server-side annotation injection in docling-raw endpoint wires `{kind: "description"}` annotations to Docling JSON pictures for hover tooltips; image dedup includes raw image bytes hash to prevent distinct same-page images from being dropped; `image-descriptions` endpoint returns `artifact_id` for image display | Complete |
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
│   ├── graphrag_service.py     # GraphRAG indexing (extraction, communities, reports) + search
│   ├── graphrag_bridge.py      # Postgres → GraphRAG input bridge (document export)
│   ├── graphrag_config.py      # GraphRAG configuration builder
│   ├── graphrag_prompts.py     # Military ontology prompts for extraction + search
│   ├── neo4j_graph.py          # Neo4j Cypher operations (sync + async)
│   ├── qdrant_store.py         # Qdrant vector upsert/search
│   ├── canonicalization.py     # Entity alias resolution + fuzzy match
│   ├── chunking.py             # Structure-aware document chunking
│   ├── reranker.py             # Cross-encoder reranker (bge-reranker-v2-m3)
│   ├── translation.py          # Foreign language detection + LLM translation
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
