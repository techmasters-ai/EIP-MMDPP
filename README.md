# EIP-MMDPP

Multi-modal document processing and retrieval platform for defense/military use cases.

Ingests PDFs, DOCX, PPTX, XLSX, HTML, Markdown, CSV, images, and technical drawings → converts documents via Docling (PdfPipeline + dlparse_v4 + EasyOCR) → extracts LLM-generated document metadata (summary, date, classification, source characterization) and picture descriptions via Ollama → embeds text (BGE-M3 via Ollama) and images (SigLIP2 via OpenCLIP) into ArcadeDB vector collections → builds a military equipment knowledge graph (ArcadeDB) via bundle-based multi-pass entity/relationship extraction (hand-authored fixed schemas; identity → field-group → relationship passes; schema-derived `absolute_union` chunk selection; per-pass dispatch + merge-and-resolve + lineage-gated graph import) → runs Louvain community detection and LLM-generated community reports → maintains governed trusted data (dedicated vector collection with human-review gate). Supports 3 retrieval strategies: basic (text vector search), hybrid (text + image multi-modal), and global (community-aware LLM synthesis). Includes a user feedback → curator patch approval workflow, document cancel/delete lifecycle, and a React web UI.

> 📖 **For a complete, end-to-end description of every workflow and feature, see [`docs/SYSTEM-DESCRIPTION.md`](docs/SYSTEM-DESCRIPTION.md).** This README covers setup, operations, the API surface, and ontology/chunk-selector authoring.

## Architecture

### Knowledge Layers

```
                    ┌──────────────────────────────────────────┐
                    │         ArcadeDB Knowledge Graph          │
                    │   Document ←→ TextChunk / ImageChunk      │
                    │   Entity nodes (LLM + regex extracted)    │
                    │   Ontology relations (32 predicates)      │
                    │   Alias nodes (entity canonicalization)   │
                    │   Fulltext index (fuzzy entity search)    │
                    │   BGE 1024-dim text vectors (native)      │
                    │   SigLIP2 1024-dim image vectors (native) │
                    └──────────┬───────────────┬────────────────┘
                               │               │
                    ┌──────────▼──────────────────────────────┐
                    │       Community Detection Layer          │
                    │   Louvain community detection            │
                    │   LLM-generated community summaries      │
                    │   Global strategy: community synthesis   │
                    └─────────────────────────────────────────┘

                    ┌──────────────────────────────────────────┐
                    │       Trusted Data Layer                   │
                    │   ArcadeDB trusted_text collection         │
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
| Graph + Vector Database | ArcadeDB (knowledge graph, ontology, canonicalization, native vector search) |
| Object Storage | MinIO |
| Text Embeddings | `bge-m3` (1024-dim via Ollama `/v1/embeddings` API) |
| Image Embeddings | OpenCLIP `ViT-L-16-SigLIP2-256` (`webli`, 1024-dim, local CPU/GPU) |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder (GPU-accelerated) |
| Document Conversion | Docling PdfPipeline (dlparse_v4 + EasyOCR + TableFormer), SimplePipeline for Office/HTML/MD |
| Document Analysis | LLM-based metadata extraction (summary, date, classification, source) + multimodal picture descriptions via Ollama |
| Graph Extraction | Docling-Graph service (bundle-based `/extract-pass` endpoint; multi-pass identity/field-group/relationship extraction with hand-authored Pydantic schemas; schema-derived `absolute_union` chunk selection; per-pass dispatch + retry + skip; merge-and-resolve; lineage-gated graph import; port 8002) |
| Community Detection | Louvain community detection over ArcadeDB graph + LLM-generated community summaries |
| Trusted Data | ArcadeDB trusted_text collection + Celery indexing (human-reviewed, vector-indexed) |
| Frontend | React 18 + TypeScript + Vite (TecMasters design system) |

All ML inference runs **fully locally** — no cloud API calls required (air-gapped deployment).

### Docker Services (13 containers with the default split-worker profile)

| Service | Purpose |
|---|---|
| `api` | FastAPI application server |
| `worker-ingest` | Celery worker — `celery`/`ingest`/`extract` queues (split profile, **default**) |
| `worker-embed` | Celery worker — `embed` queue (split profile) |
| `worker-graph` | Celery worker — `graph`/`graph_extract` queues (split profile) |
| `worker` | Single Celery worker, all queues (legacy `--start-mixed` mode only) |
| `beat` | Celery Beat (dispatcher tick, reconciliation watchdog, directory watcher, community detection) |
| `postgres` | PostgreSQL 16 (metadata, pipeline runs, pass outputs, chunk_links, governance) |
| `redis` | Celery broker + result backend |
| `minio` (+ `minio-init`) | S3-compatible object storage (raw + derived buckets) |
| `docling` | Document conversion service (granite-docling VLM) |
| `docling-graph` | Ontology-driven entity/relationship extraction service (port 8002) |
| `arcadedb` | ArcadeDB (knowledge graph + native vector search, replaces Neo4j + Qdrant) |
| `jupyter` | Analysis notebooks (dev) |

> The **split-worker profile is the default** (`./manage.sh --start`): `worker-ingest`, `worker-embed`, and `worker-graph` replace the single `worker`. Use `--start-mixed` for the legacy single-worker layout.

## Quickstart

```bash
# 1. Copy environment config and set required values
cp env.example .env
# Edit .env — at minimum set LLM_PROVIDER and (if openai) OPENAI_API_KEY

# 2. Start all services (builds images, runs migrations, waits for health)
./manage.sh --start

# 3. API + web UI
#    Web UI:      http://localhost:8000/
#    API docs:    http://localhost:8000/docs
#    ArcadeDB UI: http://localhost:2480/
```

## manage.sh — Project Management CLI

All service lifecycle, database, worker, and test operations are available through `./manage.sh`:

```bash
# Service lifecycle
./manage.sh --start              # Build and start all services; wait for health
./manage.sh --stop               # Stop all services (preserves data)
./manage.sh --restart            # Restart without rebuilding images
./manage.sh --status             # Show service status and health checks
./manage.sh --logs [service]     # Stream logs (api, worker, beat, postgres, redis, minio, docling, docling-graph, arcadedb)
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

A single `LLM_PROVIDER` env var controls the LLM backend for **all** LLM-dependent features (graph extraction, community report generation). Each feature specifies its own model via a dedicated env var.

| Value | Description |
|---|---|
| `ollama` | Uses local Ollama server. Fully air-gapped. Requires `OLLAMA_BASE_URL`. |
| `openai` | Uses OpenAI API. Requires `OPENAI_API_KEY`. |
| `mock` | Disables all LLM calls. Used in tests and environments without an LLM. |

```bash
# Air-gapped (Ollama) setup
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_LLM_BASE_URL=                              # Chat/reasoning: doc analysis, translation, community reports, global query
OLLAMA_VLM_BASE_URL=                              # Vision/multimodal: picture description
OLLAMA_EMBEDDING_BASE_URL=                        # Embedding: BGE text embeddings
OLLAMA_NUM_CTX=16384                              # Context window for Ollama (must fit prompt + response)
LLM_MAX_TOKENS=64000

# Per-feature model selection
DOC_ANALYSIS_LLM_MODEL=gpt-oss:120b              # Model for document metadata extraction
DOC_ANALYSIS_LLM_THINK=high                      # true|false for most models; low|medium|high only for gpt-oss
PICTURE_DESCRIPTION_MODEL=gemma3:27b              # Model for multimodal image descriptions
PICTURE_DESCRIPTION_THINK=false                  # true|false for most models; low|medium|high only for gpt-oss
TRANSLATION_MODEL=gpt-oss:120b                    # Model for foreign language translation
TRANSLATION_THINK=medium                         # true|false for most models; low|medium|high only for gpt-oss
COMMUNITY_REPORT_LLM_MODEL=gpt-oss:120b          # Model for community report generation
COMMUNITY_REPORT_LLM_THINK=low                   # true|false for most models; low|medium|high only for gpt-oss
DOCLING_GRAPH_LLM_MODEL=gpt-oss:120b             # Model for ontology-driven graph extraction
DOCLING_GRAPH_LLM_THINK=false                    # Extraction default: false
DOCLING_GRAPH_LLM_PROVIDER=ollama                 # ollama | openai (defaults to LLM_PROVIDER if not set)

# Docling-Graph service (ontology-driven graph extraction)
DOCLING_GRAPH_BASE_URL=http://docling-graph:8002  # Docling-Graph service URL
DOCLING_GRAPH_TIMEOUT=64800                       # R2 backstop: worker→DG HTTP timeout (18h); above the longest legit pass (~11.4h observed) with margin
DOCLING_GRAPH_LLM_TIMEOUT=64800                  # R2 backstop: per-LLM-call HTTP timeout (18h); 1800s stream-timeout is the real per-call wall
DOCLING_GRAPH_CONCURRENCY=2                       # Max concurrent extraction requests
DOCLING_GRAPH_BATCH_HARD_TIMEOUT_SECONDS=3600    # R1 no-progress (inter-arrival) batch watchdog window (1h); NOT a total-elapsed ceiling
GRAPH_NODE_MIN_CONFIDENCE=0.60                    # Min entity confidence for ArcadeDB import
GRAPH_REL_MIN_CONFIDENCE=0.55                     # Min relationship confidence for ArcadeDB import

DEFAULT_ONTOLOGY_BUNDLE_KEY=air_defense_v3         # Bundle resolution system default
PASS_MAX_RETRIES=3                                 # Per-pass retry budget for logic failures (5xx, malformed JSON, validation)
PASS_MAX_TRANSPORT_RETRIES=3                       # Separate budget for transport failures (disconnect, timeout, DNS); does not burn PASS_MAX_RETRIES
PASS_SOFT_TIME_LIMIT=72000                         # R2 backstop: per-pass Celery soft time limit (20h); above the 18h HTTP backstop so HTTP fails the pass cleanly first
PASS_CONCURRENCY_PER_DOCUMENT=2                    # Max in-flight entity-extraction passes per document
RECONCILER_PERIOD_SECONDS=60                       # How often reconcile_ontology_graph_runs runs (beat schedule)
PHASE_CLAIM_STALE_SECONDS=30                       # How long a `claimed` phase entry must be old before reconciler repairs it

# Extraction reliability ladder (R2 progress-aware watchdog — see "Extraction reliability (R1/R2)" below)
DG_PROGRESS_POLLER_ENABLED=true                   # Beat poller writes docling-graph /progress → stage_runs.metrics.progress
EXTRACTION_PROGRESS_POLL_SECONDS=30               # Poll interval for the progress poller
RECONCILER_PROGRESS_AWARE=true                    # A dispatched pass still advancing is NOT reclaimed regardless of age
RECONCILER_NO_PROGRESS_THRESHOLD_S=7200           # A dispatched pass with no progress advance for this long is reclaimable
RECONCILER_STALE_DISPATCHED_S=86400               # Absolute stale-dispatched backstop (24h), decoupled from 2×PASS_SOFT_TIME_LIMIT
INTERNAL_API_BASE_URL=http://api:8000             # worker → /v1/extraction/chunk-scope endpoint base URL
VECTOR_ROUTER_CHUNK_SCOPE_TIMEOUT_S=1200.0        # Timeout for the chunk-scope HTTP call (R6 fix: 60→1200 so large-doc capture survives)
DOCLING_FALLBACK_ENABLED=false                    # Fall back to legacy extraction on Docling 5xx (default false)
```

**Extraction reliability (R1/R2).** Two layered guards make the reduced timeout
ladder above safe for both slow-but-healthy large docs and genuinely stuck passes:

- **R1 — no-progress batch watchdog.** `DOCLING_GRAPH_BATCH_HARD_TIMEOUT_SECONDS`
  (3600s) is an *inter-arrival* window, not a total-elapsed ceiling: the watchdog
  trips only when NO batch completes within the window, so a large doc that
  legitimately needs many hours of aggregate batch time never mis-degrades. It sits
  above the 1800s OllamaPool stream wall-timeout so a single slow generation can't
  trip it.
- **R2 — progress heartbeat + progress-aware reconciler.** docling-graph publishes
  per-pass batch progress (`GET /progress`, port 8002); the beat poller
  (`DG_PROGRESS_POLLER_ENABLED`) writes it into `stage_runs.metrics.progress`; the
  reconciler (`RECONCILER_PROGRESS_AWARE`) never reclaims a pass whose progress is
  still advancing, regardless of absolute age. Only a no-progress pass past
  `RECONCILER_STALE_DISPATCHED_S` (24h) is reclaimed. This is what makes the reduced
  18h/20h backstops safe — they are last-resort backstops, not the primary guard.

See `docs/operational/production-reliability-2026-06.md` for the full root-cause
and deploy detail.

#### Absolute-union chunk selection (extraction chunk-scope)

Each field-group extraction pass selects which document chunks to send to the LLM. The production selector is **absolute_union**: an *absolute, per-chunk* keep test (not a relative top-k or quantile cut), so a pass selects **0 to all** chunks based purely on whether each chunk carries content that pass can use.

A chunk is kept for a pass iff **any** of four content signals fires:

`keep ⇔ measurement(pass) OR categorical(pass) OR image_presence(pass) OR max_field_cosine ≥ cosine_tau`

All four signals are **derived from the bundle's ontology schema** — there is **no per-corpus training or weight fitting** (contrast the legacy `guarded_quantile`, which fit `ranker_weights` per corpus).

| Signal | Derived from | Fires when the chunk… |
|---|---|---|
| **measurement** | the pass's numeric field **unit suffixes** (e.g. `range_km`→length, `mass_kg`→mass) | contains a number+unit token in any of the pass's *dimensions*. Dimension-grouped: a length field makes the pass fire on **any** length unit (km, m, mi, feet, …). |
| **categorical** | the pass's **enum fields** (e.g. `scan_type`, `guidance_type`) | contains an enum value or a mapped prose phrase. |
| **image_presence** | the pass's **`_photo`/image fields** | has a `#/pictures/` ref in its `source_refs`. |
| **cosine** | per-field dense retrieval cosine | `max_field_cosine ≥ cosine_tau` — the catch-all for relevant chunks with no explicit signal. |

Because measurement/categorical are **pass-specific**, units distinguish usefulness per pass: a propulsion pass (`*_kg`/`*_sec` fields) fires on mass/time chunks; an antenna pass (`*_deg`/`*_dbi`) fires on angle/gain chunks; neither fires on the other's chunks. This is the core design lever — see "Adapting the Chunk Selector to a New Corpus or Ontology".

**0-to-all + empty-selection contract.** If no chunk fires for a pass, the chunk-scope endpoint returns `mode=empty_selection`; the worker maps it to a zero-chunk scope and the pass finalizes **ZERO_YIELD / COMPLETE** — it does **not** fall open to full-doc and does **not** FAIL. (Off-domain passes — e.g. a radar-modulation pass on a pure missile doc — correctly yield 0.) If every chunk fires, all are selected. There is no `k_min` floor or `k_max` cap.

| Manifest field (per-pass `retrieval:` block) | Default | Meaning |
|---|---|---|
| `selection_mode` | `topk` (code default) | `absolute_union` = the signal-union above; `topk` = byte-identical legacy `c5_scored[:top_k]`; `guarded_quantile` = legacy gate∪quantile cut. |
| `cosine_tau` | `0.55` | absolute_union only — the single tunable knob (raise for precision, lower for recall). |

**Live status (production default).** All four shipped air_defense bundles (`air_defense_v3`, `air_defense_v3_baseline_subset`, `air_defense_v3_narrowing_v1`, `air_defense_v3_merged_v1`) run their field-group passes on `selection_mode: absolute_union`, `cosine_tau: 0.55`, under `VECTOR_ROUTER_MODE=narrow_only` with `WORKER_FORWARD_SELECTED_CHUNKS=false` (the selector's `self_refs` narrow scope; docling-graph still re-chunks/sanitizes the scoped doc — recall-safe). Identity passes are not routable and run full-doc. Selection diagnostics emitted per pass: `selection_mode`, `selection_k`, `measurement_keeps`, `categorical_keeps`, `image_keeps`, `cosine_keeps`.

Validated against bake-off ground truth: absolute_union ≈ **95.7% recall at ≈24% of chunks selected**, vs guarded_quantile ≈ 60.9% at ≈51%. The `topk` and `guarded_quantile` modes remain supported for legacy/uncalibrated passes.

#### Per-role Ollama URLs

The three `OLLAMA_*_BASE_URL` variables allow pointing different model types at different Ollama instances (e.g., LLM on GPU-A100, embeddings on CPU, VLM on GPU-3090). Each falls back to `OLLAMA_BASE_URL` when left empty.

| URL Variable | Falls back to | Used by |
|---|---|---|
| `OLLAMA_LLM_BASE_URL` | `OLLAMA_BASE_URL` | Doc analysis, translation, community reports, global query synthesis |
| `OLLAMA_VLM_BASE_URL` | `OLLAMA_BASE_URL` | Picture description (gemma3, llava, etc.) |
| `OLLAMA_EMBEDDING_BASE_URL` | `OLLAMA_BASE_URL` | BGE text embeddings |

#### Pool URLs (NEW)

To scale across multiple Ollama instances (e.g., a bank of 8 gemma4:31b servers), each role accepts a JSON-array env var. When set, the corresponding pool load-balances using least-in-flight routing with round-robin tie-break (one retry on a different instance for connection/timeout errors).

| Pool Variable | Used by |
|---|---|
| `OLLAMA_LLM_BASE_URLS` | All chat/reasoning calls (doc analysis, translation, community reports, global synthesis, docling-graph extraction) |
| `OLLAMA_VLM_BASE_URLS` | Picture description |
| `OLLAMA_EMBEDDING_BASE_URLS` | BGE text embeddings |

Example:

```bash
OLLAMA_LLM_BASE_URLS=["http://10.0.1.121:11434","http://10.0.1.122:11434","http://10.0.1.123:11434","http://10.0.1.124:11434"]
```

Priority order per role: plural pool > singular `OLLAMA_*_BASE_URL` > `OLLAMA_BASE_URL`. Existing `.env` files keep working unchanged — pools are opt-in.

#### Per-function pool URLs (NEW in Chunk 6)

For finer-grained control, each LLM-using function can specify its own pool. When set, the function-specific pool overrides the role-level pool above.

| Function Variable                 | Falls back to (in order)                                                                       |
|-----------------------------------|------------------------------------------------------------------------------------------------|
| `DOCLING_GRAPH_LLM_BASE_URLS`     | `OLLAMA_LLM_BASE_URLS` → `OLLAMA_LLM_BASE_URL` → `OLLAMA_BASE_URL`                             |
| `DOC_ANALYSIS_LLM_BASE_URLS`      | `OLLAMA_LLM_BASE_URLS` → `OLLAMA_LLM_BASE_URL` → `OLLAMA_BASE_URL`                             |
| `TRANSLATION_LLM_BASE_URLS`       | `OLLAMA_LLM_BASE_URLS` → `OLLAMA_LLM_BASE_URL` → `OLLAMA_BASE_URL`                             |
| `COMMUNITY_REPORT_LLM_BASE_URLS`  | `OLLAMA_LLM_BASE_URLS` → `OLLAMA_LLM_BASE_URL` → `OLLAMA_BASE_URL` (also used by global-query) |
| `PICTURE_DESCRIPTION_BASE_URLS`   | `OLLAMA_VLM_BASE_URLS` → `OLLAMA_VLM_BASE_URL` → `OLLAMA_BASE_URL`                             |
| `TEXT_EMBEDDING_BASE_URLS`        | `OLLAMA_EMBEDDING_BASE_URLS` → `OLLAMA_EMBEDDING_BASE_URL` → `OLLAMA_BASE_URL`                 |

Common patterns:

```bash
# Pattern 1: single bank for everything (default — leave the per-function vars empty)
OLLAMA_LLM_BASE_URLS=["http://10.0.1.121:11434","http://10.0.1.122:11434"]

# Pattern 2: graph extraction on bank A (gemma4:31b), other chat functions on bank B (gpt-oss:120b)
DOCLING_GRAPH_LLM_BASE_URLS=["http://gemma-host-1:11434","http://gemma-host-2:11434"]
OLLAMA_LLM_BASE_URLS=["http://gpt-oss-host-1:11434","http://gpt-oss-host-2:11434"]

# Pattern 3: every function pinned to its own host
DOCLING_GRAPH_LLM_BASE_URLS=["http://gemma-bank:11434"]
DOC_ANALYSIS_LLM_BASE_URLS=["http://gpt-oss:11434"]
TRANSLATION_LLM_BASE_URLS=["http://llama:11434"]
COMMUNITY_REPORT_LLM_BASE_URLS=["http://gpt-oss:11434"]
PICTURE_DESCRIPTION_BASE_URLS=["http://gemma-vlm:11434"]
TEXT_EMBEDDING_BASE_URLS=["http://bge-host:11434"]
```

For diagnostics on docling-graph extraction fan-out, set `DOCLING_GRAPH_DEBUG_ENDPOINTS=true` and query `GET /debug/routing-metrics` on port 8002. Returns per-URL request counts. Default-off to avoid leaking backend URLs on the published port; disable again after diagnosis.

The docling-graph service also exposes a read-only **`GET /progress`** heartbeat on port 8002 (R2 progress-aware watchdog — see "Extraction reliability (R1/R2)" above). It returns in-flight per-pass batch progress, `{"passes": [{run_id, pass_name, done, total, phase, started_at, updated_at, age_s}, ...]}`, optionally filtered by a `pipeline_run_id` query param. The beat poller (`DG_PROGRESS_POLLER_ENABLED`) consumes it and mirrors it into `stage_runs.metrics.progress`; the progress-aware reconciler reads that to avoid reclaiming a still-advancing pass.

Per-request thinking is configured separately per application:

| Think Variable | Used by |
|---|---|
| `DOC_ANALYSIS_LLM_THINK` | Document metadata extraction |
| `PICTURE_DESCRIPTION_THINK` | Picture description |
| `TRANSLATION_THINK` | Translation |
| `COMMUNITY_REPORT_LLM_THINK` | Community reports and global query synthesis |
| `DOCLING_GRAPH_LLM_THINK` | Docling-graph extraction only |

For most Ollama models, `think` should be `true` or `false`. `low`, `medium`, and `high` are gpt-oss-specific thinking levels and should only be used with `gpt-oss` models.

The docling-graph service inherits `OLLAMA_LLM_BASE_URL` and its own `DOCLING_GRAPH_LLM_THINK` override via the docker-compose cascade.

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
| `POST` | `/v1/documents/{id}/reingest` | Re-run pipeline — `{"mode": "full|embeddings_only|graph_only", "ontology_bundle_key"?: str, "use_case_key"?: str}`; 409 if already PROCESSING |
| `GET` | `/v1/documents/{id}/extraction-status` | Three-concept extraction status: document_status, latest_run (per-pass details), graph_snapshot, graph_queryable (cross-run rollback-aware) |
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

### Graph Store (ArcadeDB)

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
| `POST` | `/v1/trusted-data/proposals/{id}/approve` | Curator approves → enqueues Celery task to embed + index in ArcadeDB |
| `POST` | `/v1/trusted-data/proposals/{id}/reject` | Curator rejects submission |
| `POST` | `/v1/trusted-data/proposals/{id}/reindex` | Re-enqueue failed/pending indexing |
| `POST` | `/v1/trusted-data/query` | Search approved trusted data (ArcadeDB vector search) |

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

All ArcadeDB graph mutations (node/edge create, update, delete) require **dual-curator approval**. Text and classification corrections require a single curator.

### Unified Retrieval

**Endpoint:** `POST /v1/retrieval/query`

**Request schema (`UnifiedQueryRequest`):**

| Field | Type | Default | Description |
|---|---|---|---|
| `query_text` | `string` | — | Text query (max 4096 chars). Required unless `query_image` is provided. |
| `query_image` | `string` | `null` | Base64-encoded PNG/JPG image or artifact UUID reference. Only used by `hybrid` strategy. |
| `strategy` | `string` | `"basic"` | One of: `basic`, `hybrid`, `global` |
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
| `chunk_id` | `uuid` | Chunk identifier (null for global community responses) |
| `artifact_id` | `uuid` | Source artifact identifier |
| `document_id` | `uuid` | Source document identifier |
| `score` | `float` | Relevance score |
| `modality` | `string` | `text`, `image`, `table`, `schematic`, `image_description`, or `community_response` |
| `content_text` | `string` | Chunk text or LLM-generated response |
| `page_number` | `int` | Source page number |
| `classification` | `string` | Security classification (default: `UNCLASSIFIED`) |
| `context` | `object` | Graph neighbors, community context (`{source, community_context: {community_id, summary}}`) |
| `image_url` | `string` | API proxy URL for image results (`/v1/images/{chunk_id}`) |
| `source_characterization` | `string` (nullable) | Source characterization from document metadata |
| `date_of_information` | `string` (nullable) | Date of information from document metadata |
| `extraction_confidence` | `float` (nullable) | Confidence score from graph extraction |
| `sources` | `array` (nullable) | Array of `{document_id, page_number, classification, chunk_text_preview}` source references |

**Query strategies:**

| Strategy | Modality Filter | Input | Pipeline | Output | Speed |
|---|---|---|---|---|---|
| `basic` | `all` | Text only | BGE vector search (ArcadeDB) | Ranked text/table chunks | Fast (1-3s) |
| `hybrid` | `all`/`text`/`image` | Text and/or image | Full multi-modal pipeline | Mixed text, image, table, schematic chunks | Medium (5-15s) |
| `global` | `all` | Text only | Louvain community detection + LLM synthesis | LLM-generated response grounded in community summaries | Medium-slow (15-60s) |

> **Backward compatibility**: The legacy `mode` field (e.g. `"mode": "text_only"`) is still accepted and maps to the corresponding `strategy` + `modality_filter` combination.

#### 1. Basic Text Query

BGE vector search over text chunks in ArcadeDB. No LLM calls, no graph expansion.

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

Full pipeline: BGE text + CLIP image search → ArcadeDB graph expansion → ontology traversal → weighted fusion scoring → cross-encoder reranking. Accepts text, base64 image, or both.

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

#### 3. Global Query (Community-Aware)

Louvain community detection groups related entities in the ArcadeDB knowledge graph. LLM-generated community summaries are used to synthesize a broad, holistic answer. Requires at least one successful community detection run.

```python
import requests

resp = requests.post("http://localhost:8000/v1/retrieval/query", json={
    "query_text": "What are the major categories of air defense systems and how do they compare?",
    "strategy": "global",
    "top_k": 10,
}, timeout=120)

data = resp.json()
# Global returns a single result with community context
result = data["results"][0]
print(result["content_text"])  # LLM-generated synthesis
ctx = result.get("context", {}).get("community_context", {})
if ctx:
    print(f"Community: {ctx.get('community_id')}")
    print(f"Summary: {ctx.get('summary', '')[:200]}")
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
| `GET` | `/v1/settings/community` | Community detection config: `{detection_enabled, detection_interval_minutes, last_detection_at}` |

The hybrid pipeline runs: parallel vector search (BGE + CLIP via ArcadeDB `asyncio.gather`) → document-structure expansion (ArcadeDB structural edges with Postgres fallback) → ontology traversal (ArcadeDB entity relationships) → independent re-scoring of expanded chunks → weighted fusion scoring → deduplicate → cross-encoder reranking (bge-reranker-v2-m3) → min score threshold filter → rank → filter by modality.

Image-modality results include an `image_url` served via the API proxy (`GET /v1/images/{chunk_id}`), which streams from MinIO with 1-hour cache headers. This avoids exposing Docker-internal MinIO hostnames in presigned URLs and works in air-gapped environments without hostname configuration.

**Weighted Fusion Scoring**: `final = 0.65*semantic + 0.20*doc_structure + 0.15*ontology + MIL-ID bonus`. MIL-ID bonus matches NSN, MIL-STD, ELNOT, DIEQP, and AN/ designators. All weights are configurable via environment variables (see `env.example`). Results below `RETRIEVAL_MIN_SCORE_THRESHOLD` (default 0.25) are dropped. Top candidates are re-scored by a cross-encoder reranker (`RERANKER_MODEL`, default `BAAI/bge-reranker-v2-m3`, configurable via `RERANKER_DEVICE`, `RERANKER_ENABLED`, `RERANKER_TOP_N`).

### Multi-Modal Query Walkthrough

**Example query:** `"VHF radar internal components"` with `strategy=hybrid`, `modality_filter=all`

**Step 1 — Parallel vector search (seeds)**

Two searches run concurrently via `asyncio.gather`:

- **BGE text search** — query embedded with `bge-m3` (via Ollama), searched against `TextChunk` vertices in ArcadeDB (1024-dim cosine). Matches text chunks, table chunks, and **image description sections**. Over-fetches by 8× for diversity, filters below 0.25, content-deduplicates, reranks, returns top-k.
- **Image search** — query embedded with the OpenCLIP `ViT-L-16-SigLIP2-256` text encoder, searched against `ImageChunk` vertices in ArcadeDB (1024-dim cosine). Matches images by similarity to the text concept. Scores are typically lower (0.1–0.3) because cross-modal alignment is loose.

Seeds from both searches are merged (highest score per chunk_id kept).

**Step 2 — Per-seed graph expansion**

For each seed, three strategies run (bounded to 16 concurrent):

- **Document-structure expansion** (ArcadeDB structural edges — NEXT_CHUNK, SAME_SECTION, SAME_ARTIFACT, SAME_PAGE, with Postgres fallback) — follows pre-computed structural links: `NEXT_CHUNK` (reading order), `SAME_SECTION` (under same heading), `SAME_ARTIFACT` (from same image/table), `SAME_PAGE` (text ↔ image on same page). If an image description section is a seed, `SAME_ARTIFACT` surfaces sibling sections and `SAME_PAGE` surfaces the original CLIP image chunk.
- **Cross-modal bridging** (ArcadeDB, fallback only) — for legacy documents without chunk_links. Traverses structural edges up to 3 hops to bridge text ↔ image.
- **Ontology traversal** (ArcadeDB knowledge graph) — follows entity relationships. If a chunk mentions "S-75 Dvina" and the graph has `S-75 Dvina –[VARIANT_OF]→ SA-2 Guideline`, chunks about "SA-2 Guideline" are surfaced with per-relation weights from `ontology.yaml`.

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

19 entity types, 32 relationship predicates, enforced via validation matrix at graph write time. Entity resolution uses exact → alias → fuzzy match canonicalization.

See `ontology_bundles/air_defense_v3/ontology.yaml` for the full schema.

## Adding a Custom Ontology Bundle

The extraction system is domain-agnostic. While the shipped `air_defense_v3` bundle targets military equipment, you can create bundles for any domain: medical records, legal contracts, financial reports, supply chain, etc.

### Bundle Architecture

A bundle is a self-contained directory under `ontology_bundles/<bundle_key>/` containing everything the extraction pipeline needs:

```
ontology_bundles/
└── <your_bundle_key>/
    ├── __init__.py                     # Empty (makes it a Python package)
    ├── ontology.yaml                   # Domain ontology (entity types, relationships, properties)
    ├── manifest.yaml                   # Pass registry (which passes exist, in what order)
    ├── coverage.yaml                   # What's extracted vs derived vs validate-only
    ├── validators.py                   # Shared Pydantic field validators for messy LLM output
    ├── derive_rules.py                 # Deterministic post-merge edge derivation
    └── extraction_schemas/
        ├── __init__.py
        └── <pass_name>.py             # One module per extraction pass
```

### Step-by-Step: Creating a Medical Records Bundle

This walkthrough creates a `medical_records_v1` bundle that extracts patients, diagnoses, medications, procedures, and their relationships from clinical documents.

#### Step 1: Define the ontology (`ontology.yaml`)

The ontology declares your domain's entity types, their properties, identity fields, and a validation matrix of allowed relationships.

```yaml
# ontology_bundles/medical_records_v1/ontology.yaml

entity_types:
  - name: PATIENT
    label: "Patient"
    identity_fields: [patient_id]
    identity_scope: document          # Different patients per document
    description: "A patient record"
    properties:
      type: object
      properties:
        patient_id: { type: string, description: "Medical record number" }
        name: { type: string, description: "Patient full name" }
        date_of_birth: { type: string, description: "DOB in ISO format" }
        gender: { type: string, description: "Patient gender" }

  - name: DIAGNOSIS
    label: "Diagnosis"
    identity_fields: [icd_code]
    identity_scope: global            # Same ICD code = same diagnosis everywhere
    description: "A medical diagnosis"
    properties:
      type: object
      properties:
        icd_code: { type: string, description: "ICD-10 code" }
        description: { type: string, description: "Diagnosis description" }
        severity: { type: string, description: "mild / moderate / severe / critical" }

  - name: MEDICATION
    label: "Medication"
    identity_fields: [drug_name]
    identity_scope: global
    description: "A prescribed medication"
    properties:
      type: object
      properties:
        drug_name: { type: string, description: "Generic drug name" }
        dosage: { type: string, description: "Dosage with units" }
        route: { type: string, description: "Route of administration" }
        frequency: { type: string, description: "Dosing frequency" }

  - name: PROCEDURE
    label: "Medical Procedure"
    identity_fields: [procedure_name]
    identity_scope: document
    description: "A medical procedure performed"
    properties:
      type: object
      properties:
        procedure_name: { type: string, description: "Procedure name" }
        cpt_code: { type: string, description: "CPT code" }
        date_performed: { type: string, description: "Date in ISO format" }
        outcome: { type: string, description: "Procedure outcome" }

  - name: LAB_RESULT
    label: "Lab Result"
    identity_fields: [test_name, date_collected]
    identity_scope: document
    description: "A laboratory test result"
    properties:
      type: object
      properties:
        test_name: { type: string, description: "Lab test name" }
        value: { type: string, description: "Result value with units" }
        reference_range: { type: string, description: "Normal range" }
        date_collected: { type: string, description: "Collection date" }
        abnormal: { type: string, description: "normal / high / low / critical" }

  # Document-structure entities (same as air_defense_v3)
  - name: SECTION
    label: "Document Section"
    identity_fields: [heading, page_start]
    identity_scope: document
    properties:
      type: object
      properties:
        heading: { type: string }
        page_start: { type: integer }
        page_end: { type: integer }

relationship_types:
  - name: DIAGNOSED_WITH
    description: "Patient has this diagnosis"
    source_type: PATIENT
    target_type: DIAGNOSIS
  - name: PRESCRIBED
    description: "Patient is prescribed this medication"
    source_type: PATIENT
    target_type: MEDICATION
  - name: UNDERWENT
    description: "Patient underwent this procedure"
    source_type: PATIENT
    target_type: PROCEDURE
  - name: HAS_LAB_RESULT
    description: "Patient has this lab result"
    source_type: PATIENT
    target_type: LAB_RESULT
  - name: TREATS
    description: "Medication treats this diagnosis"
    source_type: MEDICATION
    target_type: DIAGNOSIS
  - name: INDICATED_BY
    description: "Procedure indicated by lab finding"
    source_type: PROCEDURE
    target_type: LAB_RESULT

validation_matrix:
  - { source: PATIENT, relationship: DIAGNOSED_WITH, target: DIAGNOSIS }
  - { source: PATIENT, relationship: PRESCRIBED, target: MEDICATION }
  - { source: PATIENT, relationship: UNDERWENT, target: PROCEDURE }
  - { source: PATIENT, relationship: HAS_LAB_RESULT, target: LAB_RESULT }
  - { source: MEDICATION, relationship: TREATS, target: DIAGNOSIS }
  - { source: PROCEDURE, relationship: INDICATED_BY, target: LAB_RESULT }
```

**Key design decisions:**
- `identity_scope: global` for things that are the same concept across all documents (e.g., a specific drug or ICD code)
- `identity_scope: document` for things scoped to a specific patient encounter (e.g., a patient, a specific lab result)
- `identity_fields` must be properties the LLM can reliably extract -- choose the most stable, unique identifiers

#### Step 2: Define the extraction passes (`manifest.yaml`)

Passes control how many LLM calls happen per document and what each call extracts.

```yaml
# ontology_bundles/medical_records_v1/manifest.yaml

bundle_key: medical_records_v1
manifest_schema_version: "1.0.0"
ontology_name: "Clinical Document Ontology"
ontology_version: "1.0.0"
extraction_profile_version: "1.0.0"

passes:
  - name: document_structure
    required: true
    kind: entities
    input_mode: document_only
    module: extraction_schemas.document_structure
    template_class: DocumentStructurePass
    primary_entity_types: [SECTION]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: clinical_entities
    required: true
    kind: entities_and_relationships
    input_mode: document_only
    module: extraction_schemas.clinical_entities
    template_class: ClinicalEntitiesPass
    primary_entity_types: [PATIENT, DIAGNOSIS, MEDICATION, PROCEDURE, LAB_RESULT]
    bridge_entity_types: []
    extracted_relationship_types:
      [DIAGNOSED_WITH, PRESCRIBED, UNDERWENT, HAS_LAB_RESULT]
    depends_on: []

  - name: clinical_links
    required: true
    kind: relationships_only
    input_mode: document_plus_entity_refs
    module: extraction_schemas.clinical_links
    template_class: ClinicalLinksPass
    primary_entity_types: []
    bridge_entity_types: []
    extracted_relationship_types: [TREATS, INDICATED_BY]
    depends_on: [clinical_entities]
    skip_if_no_upstream_endpoints: true
    skip_justification: >
      When clinical_entities finds no medications, procedures, or lab
      results, there is nothing for clinical_links to connect.
```

**Design guidance:**
- Start with 2-3 passes. More passes = more LLM calls = slower but potentially more accurate per-call.
- Put all "primary" entity extraction in one pass. Put cross-entity relationships in a second.
- Use `depends_on` + `input_mode: document_plus_entity_refs` for relationship-only passes that need to reference entities from earlier passes.

#### Step 3: Declare coverage (`coverage.yaml`)

```yaml
# ontology_bundles/medical_records_v1/coverage.yaml

bundle_key: medical_records_v1
version: "1.0.0"

entity_types:
  extract:
    - SECTION
    - PATIENT
    - DIAGNOSIS
    - MEDICATION
    - PROCEDURE
    - LAB_RESULT
  derive: []

relationship_types:
  extract:
    - DIAGNOSED_WITH
    - PRESCRIBED
    - UNDERWENT
    - HAS_LAB_RESULT
    - TREATS
    - INDICATED_BY
  derive:
    - HAS_PROVENANCE
    - MENTIONED_IN
```

#### Step 4: Write the extraction schemas

Each pass gets a Python module with Pydantic models. **Every field must be `Optional` with a default** so partial LLM output does not crash.

```python
# ontology_bundles/medical_records_v1/extraction_schemas/clinical_entities.py

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ..validators import coerce_optional_confidence


class PatientEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    patient_id: Optional[str] = None
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    confidence: Optional[float] = None
    _v_conf = field_validator("confidence", mode="before")(coerce_optional_confidence)


class DiagnosisEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    icd_code: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    _v_conf = field_validator("confidence", mode="before")(coerce_optional_confidence)


class MedicationEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    drug_name: Optional[str] = None
    dosage: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    confidence: Optional[float] = None
    _v_conf = field_validator("confidence", mode="before")(coerce_optional_confidence)


class ProcedureEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    procedure_name: Optional[str] = None
    cpt_code: Optional[str] = None
    date_performed: Optional[str] = None
    outcome: Optional[str] = None
    confidence: Optional[float] = None
    _v_conf = field_validator("confidence", mode="before")(coerce_optional_confidence)


class LabResultEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    test_name: Optional[str] = None
    value: Optional[str] = None
    reference_range: Optional[str] = None
    date_collected: Optional[str] = None
    abnormal: Optional[str] = None
    confidence: Optional[float] = None
    _v_conf = field_validator("confidence", mode="before")(coerce_optional_confidence)


class ClinicalRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rel_type: Optional[str] = None
    from_type: Optional[str] = None
    from_identity: Optional[dict[str, Any]] = None
    to_type: Optional[str] = None
    to_identity: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None
    _v_conf = field_validator("confidence", mode="before")(coerce_optional_confidence)


class ClinicalEntitiesPass(BaseModel):
    """Top-level template for the clinical_entities pass."""
    model_config = ConfigDict(extra="ignore")
    patients: list[PatientEntity] = Field(default_factory=list)
    diagnoses: list[DiagnosisEntity] = Field(default_factory=list)
    medications: list[MedicationEntity] = Field(default_factory=list)
    procedures: list[ProcedureEntity] = Field(default_factory=list)
    lab_results: list[LabResultEntity] = Field(default_factory=list)
    relationships: list[ClinicalRelationship] = Field(default_factory=list)
```

**Naming convention:** The top-level pass class field names MUST be the lowercase plural of the entity type name (e.g., `PATIENT` -> `patients`, `LAB_RESULT` -> `lab_results`). The merge layer uses this convention to discover entity lists.

> **Chunk-selector design:** name numeric fields with unit suffixes (`*_km`, `*_kg`, `*_sec`), use enum fields for closed vocabularies, and `*_photo` fields for figures — these drive the `absolute_union` chunk selector's per-pass signals with no training. See "Adapting the Chunk Selector to a New Corpus or Ontology".

#### Step 5: Write shared validators (`validators.py`)

Copy from `ontology_bundles/air_defense_v3/validators.py` as a starting point. Add domain-specific validators if needed (e.g., ICD-10 code normalization).

#### Step 6: Write derive_rules (`derive_rules.py`)

Copy the structure from `ontology_bundles/air_defense_v3/derive_rules.py`. The `derive_structural_edges` function produces deterministic edges (like `MENTIONED_IN` from entities to text chunks). Customize or keep as-is.

#### Step 7: Activate your bundle

Set the system default in `.env`:

```bash
DEFAULT_ONTOLOGY_BUNDLE_KEY=medical_records_v1
```

Or set it per-source via the API:

```python
requests.post(f"{BASE}/sources", json={
    "name": "patient-records",
    "default_ontology_bundle_key": "medical_records_v1",
})
```

Or override per-reingest:

```python
requests.post(f"{BASE}/documents/{doc_id}/reingest", json={
    "mode": "graph_only",
    "ontology_bundle_key": "medical_records_v1",
})
```

#### Step 8: Validate with the coverage checker

```bash
python tools/check_extraction_coverage.py
```

This runs 13 rules against your bundle and reports any issues:
- Every entity in manifest.yaml must be in coverage.yaml
- Every relationship must be in the validation_matrix
- Schema size must be under the structured-output threshold
- All fields must be Optional (partial-safety)
- Identity fields must exist as properties
- Bridge entities must be consistent across passes

Fix any reported errors before ingesting documents.

#### Step 9: Rebuild and test

```bash
# Rebuild images (the Dockerfiles COPY ontology_bundles/)
docker compose build worker docling-graph

# Verify importability
./scripts/smoke_test_bundle_import.sh

# Ingest a test document
curl -X POST http://localhost:8000/v1/sources/{source_id}/documents \
  -F "file=@patient_record.pdf"
```

### Multiple Bundles

You can have multiple bundles side by side:

```
ontology_bundles/
├── air_defense_v3/     # Military equipment
├── medical_records_v1/ # Clinical documents
└── legal_contracts_v1/ # Legal agreements
```

Different sources can use different bundles:

```python
# Military source
requests.post(f"{BASE}/sources", json={
    "name": "intel-reports",
    "default_ontology_bundle_key": "air_defense_v3",
})

# Medical source
requests.post(f"{BASE}/sources", json={
    "name": "patient-records",
    "default_ontology_bundle_key": "medical_records_v1",
})
```

Documents ingested from each source automatically use the source's default bundle. The system default (`DEFAULT_ONTOLOGY_BUNDLE_KEY`) is the fallback when a source does not specify one.

### Design Tips

1. **Start small.** Begin with 3-5 entity types and 3-5 relationships. Add more after validating extraction quality.
2. **Choose identity_fields carefully.** These determine entity merge behavior. Pick fields the LLM can reliably extract and that uniquely identify the entity.
3. **Use `identity_scope: global` sparingly.** Global entities merge across documents -- great for well-known things (drug names, ICD codes) but risky for ambiguous names.
4. **Keep schemas under 8000 chars.** The JSON schema is sent to the LLM for structured output. Huge schemas degrade extraction quality. Split into more passes if needed.
5. **Test with `graph_only` reingest.** After tuning schemas, use `mode: "graph_only"` to re-extract without re-processing the document through Docling.
6. **Check the coverage checker.** Run `python tools/check_extraction_coverage.py` after every schema change. It catches drift before it reaches production.

## Adapting the Chunk Selector to a New Corpus or Ontology

The **absolute_union** selector needs **no per-corpus training** — no labeled data, no weight fitting, no quantile calibration. Its per-pass discrimination is **derived entirely from your ontology's field schema**. Adapting it to a new domain (medical, legal, finance, …) means **designing the schema so each pass's content signals are meaningful**, not fitting a model.

### Mental model

For each field-group pass the selector keeps a chunk iff `measurement OR categorical OR image OR cosine≥τ` (see "Absolute-union chunk selection"). Three of the four signals come straight from how you NAME and TYPE the pass's fields; the fourth (cosine) is a fixed threshold. So the design question per pass is: **what surface signal marks a chunk as relevant — a unit, an enum phrase, an image, or just semantic similarity?**

### Design principle 1 — Units make passes selective (the main lever)

Name every numeric field with a **unit suffix**. The suffix maps to a physical **dimension**, and a pass's dimension set is the union over its fields. Because the measurement matcher is **dimension-grouped**, declaring one `*_km` field makes the pass fire on *any* length-bearing chunk (km, m, mi, feet, …) — and NOT on a chunk that only has, say, frequency units. This is how units distinguish per-pass usefulness:

- a **propulsion** pass with `burn_time_sec`, `mass_kg` → dimensions {time, mass} → fires on seconds/kilograms chunks, not antenna-gain chunks.
- an **antenna** pass with `beamwidth_deg`, `gain_dbi` → {angle, gain} → fires on degree/dBi chunks, not burn-time chunks.

The suffix→dimension map lives in `app/services/extraction_pass_signal_config.py` (`SUFFIX_DIMENSION`); the unit surface forms in `app/services/extraction_signal_detectors.py` (`DIMENSION_UNITS`). Shipped: `_km/_m/_mm/_cm`→length, `_deg/_rad`→angle, `_sec/_usec/_ms/_ns`→time, `_mhz/_ghz/_khz/_hz`→frequency, `_mps`→velocity, `_kg/_g`→mass, `_dbi`→gain, `_kw/_w`→power. **For a new domain add your units** — e.g. medical `_mgdl`/`_mmhg`, finance `_usd` — by adding the suffix→dimension to `SUFFIX_DIMENSION` and the unit surface forms (abbreviations + spelled-out + plurals) to `DIMENSION_UNITS`. **Avoid bare single-character units** (m, s, g) in `DIMENSION_UNITS`: they false-match inside identifiers (e.g. the `m` in `S-75M`). Rely on 2+ char abbreviations and spelled-out forms.

### Design principle 2 — Enum fields enable categorical matching

For attributes with a closed vocabulary (status, type, mode), declare an **enum field** and register it: add the field name to `CATEGORICAL_PHRASE_FIELDS` (`extraction_pass_signal_config.py`) and its values / prose phrases to `CATEGORICAL_PHRASES` (`extraction_signal_detectors.py`). A chunk then fires the categorical signal if it contains one of those phrases. **Keep phrases ≥4 chars and prefer multi-word** to avoid substring false-positives (a bare `arh` matches "warhead"; `clos` matches "closure"). This lets passes whose content is qualitative (e.g. a guidance pass: "semi-active radar homing") be selected without numeric units.

### Design principle 3 — Image fields enable image-presence

For entities documented by figures/photos, declare a `*_photo` (or `*_image`) field. The pass then fires on any chunk whose `source_refs` include a `#/pictures/` ref — catching figure-bearing chunks that may have little text.

### Design principle 4 — cosine_tau catches the rest

`cosine_tau` (default 0.55) keeps chunks semantically relevant to the pass's fields but carrying no explicit unit/enum/image signal. It is the one tunable knob: raise for precision, lower for recall. Leave it at 0.55 unless a validation run shows a systematic gap.

### Procedure

1. **Build the bundle + schema** (see "Adding a Custom Ontology Bundle"), applying design principles 1–3: unit-suffixed numeric fields, enum fields for closed vocabularies, `*_photo` fields for figures.
2. **Extend the signal lexicons** for your domain's units/enums: `SUFFIX_DIMENSION` + `DIMENSION_UNITS` (units); `CATEGORICAL_PHRASE_FIELDS` + `CATEGORICAL_PHRASES` (enums). This is the only code change, and it is schema-driven, not data-fit. Unit-test with `tests/unit/test_extraction_signal_detectors.py` and `tests/unit/test_extraction_pass_signal_config.py`.
3. **Configure each field-group pass** in the manifest's `retrieval:` block: `selection_mode: absolute_union`, `cosine_tau: 0.55`. Leave **identity passes** without a `retrieval:` block — they run full-doc and are not routable.
4. **Verify the config is schema-derived as intended:**
   ```bash
   python3 -c "from app.services.extraction_pass_signal_config import derive_pass_signal_config as d; import json; print(json.dumps({k:(sorted(v.dimensions),sorted(v.categorical_fields),v.has_image_field) for k,v in d('<your_bundle>').items()}, indent=2))"
   ```
   Confirm each pass's dimensions/categorical/image match your intent (e.g. a propulsion pass shows `[["mass","time"], [], false]`).
5. **Deploy:** manifest edit, then `docker compose -p eip-mmdpp up -d --force-recreate api worker worker-graph` (bind-mounted code reloads on restart), and set `VECTOR_ROUTER_MODE=narrow_only`.
6. **Validate (optional, recommended) against ground truth — a check, not training.** If you have lineage-grounded labels, compare per-pass recall/precision (bake-off harness). Expect off-domain passes to yield 0 (empty_selection→ZERO_YIELD) — that is correct, not a failure.

### Notes

- **No calibration loop.** Unlike the legacy guarded-ranker (which fit `ranker_weights`/`quantile_q` per corpus — the `scripts/` calibration tools `export_bakeoff_dataset` / `check_gate_coverage` / `eval_guarded_ranker` / `fit_guarded_ranker` still exist for that deprecated `guarded_quantile` path), absolute_union has zero learned parameters. Re-targeting a new corpus needs only schema design + lexicon extension.
- **Empty is expected.** A pass with no matching content on a doc returns 0 chunks → ZERO_YIELD/COMPLETE. Design one pass per attribute-group; not every pass fires on every doc.
- **Units are the highest-leverage choice.** A pass whose numeric fields all share a dimension common across the corpus over-selects (everything has that unit); a pass with a distinctive dimension self-selects cleanly. Choose field units that discriminate the pass's content.

## Creating Custom Queries

Custom queries let you define deterministic graph traversal patterns that surface specific entity relationships from ArcadeDB. Once created, they appear as options in the Search Documents dropdown alongside the built-in retrieval modes.

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
1. Resolves "Patriot" to a `PLATFORM` entity in ArcadeDB (via alias + fulltext matching).
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

- Custom queries only search the ArcadeDB knowledge graph. They do not perform vector/semantic search.
- Entities must exist in ArcadeDB (via document ingestion + graph extraction) before they can be found.
- Changing the active ontology does not retroactively re-extract already-ingested documents. Re-ingest documents to apply a new ontology.

## Community Detection

Louvain community detection over the ArcadeDB knowledge graph generates community groupings and LLM-synthesized summaries that power the `global` retrieval strategy.

### Architecture

```
ArcadeDB entity graph ──→ Louvain detection ──→ Community assignments (Postgres)
                                                        │
                                            LLM community report generation
                                                        │
                                            community_reports (Postgres)
                                                        │
                                            Global query: select top-k
                                            communities by relevance ──→
                                            LLM synthesis → single response
```

### Detection & Scheduling

- **Scheduled**: Celery Beat runs detection on configurable interval (`COMMUNITY_DETECTION_INTERVAL_MINUTES`, default 1440)
- **Manual**: `POST /v1/community/detect` dispatches immediately (Redis lock prevents overlapping runs)
- **Stale lock cleanup**: Redis locks are automatically cleared on worker startup

### Prerequisites

- `global` strategy queries require at least one successful community detection run
- API returns 409 for missing community reports instead of silent empty results

## Web UI

React 18 + TypeScript + Vite single-page application served by the API container. The frontend builds as Stage 1 of the main Dockerfile (`node:22-alpine`).

### Search Documents (`QueryPage`)

A dropdown selector groups query modes into two categories:

**Retrieval modes** (always available):

| Mode | Strategy | Description |
|---|---|---|
| **Text Basic** | `basic` | BGE vector search over text chunks in ArcadeDB |
| **Multi-Modal** | `hybrid` | Full multi-modal pipeline (text + image). Shows a modality sub-filter: All / Text Only / Images Only |
| **Global** | `global` | Louvain community-aware synthesis for broad analytical questions |

**Query Profile modes** (appear when an active registry has exposed profiles):

Custom ontology-driven graph traversal queries. These are defined via the Ontology & Query Profiles page and appear automatically in the dropdown once exposed. See [Creating Custom Queries](#creating-custom-queries) below.

**Result cards** show:
- Always-visible text preview (first ~300 chars of `content_text`)
- Inline image thumbnails for image-modality results (click for lightbox)
- Expandable "Show details" section with full text and all metadata (`chunk_id`, `artifact_id`, `document_id`, `score`, `modality`, `page_number`, `classification`, full `context` object)

**Global query results**: Community title + level badge, expandable community summary text rendered inline.

Images are served via the API proxy (`GET /v1/images/{chunk_id}`) which streams from MinIO — no Docker-internal hostnames exposed to the browser.

### Document Upload (`FileUpload`)

Drag-and-drop or click-to-upload with real-time pipeline status polling. Supports PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, PNG, JPG, TIFF. Adaptive polling intervals (2s → 5s → 10s) based on elapsed time. Cancel button for PROCESSING documents (revokes Celery tasks, cleans up all data stores, removes the document). Retry button for FAILED/ERROR documents. Delete button for completed/failed documents. When a source is selected, shows all historical documents for that source with live status updates, cancel, retry, and delete support. DoclingDocument viewer for COMPLETE documents with bounding-box overlay, metadata panel (summary, date, classification), and plaintext fallback for .txt files. **Image description tooltips**: hovering over embedded images (PDF, DOCX, PPTX) shows LLM-generated descriptions via the Docling web component's built-in tooltip system; standalone image documents (.png, .jpeg) display the description in a persistent panel.

### Other Pages

- **Ingest** — unified ingest page with upload + status overview
- **Directory Monitor** — register/remove watch directories for auto-ingest
- **Graph Explorer** — ArcadeDB entity/relationship search + manual creation (full ontology support)
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
purge_document_derivations  (clean up prior run data for reingest)
    ↓
derive_picture_descriptions  (LLM: multimodal image descriptions with summary context)
    ↓
derive_text_chunks_and_embeddings  (includes image description section embedding pass)
    ↓
derive_image_embeddings
    ↓
derive_ontology_graph  (5-pass extraction → merge-and-resolve → three-phase graph import)
    ↓
collect_derivations
    ↓
derive_structure_links  (needs embedding output committed)
    ↓
derive_canonicalization  (entity resolution pass)
    ↓
finalize_document
```

Key features:
- **Canonical element store** (`document_elements` table) — parse once, derive many
- **Sequential 12-stage chain** — all stages run sequentially (the parallel chord was removed)
- **Bundle-based graph extraction** — `derive_ontology_graph` dispatches the bundle's passes via `/extract-pass` in three phases (identity → field-group → relationship), narrows each pass with the schema-derived `absolute_union` chunk selector, merges results, resolves entities, applies a strict lineage gate, and imports (nodes → domain edges → structural edges). `air_defense_v3` ships 12 passes (2 identity + 9 field-group + 1 relationship).
- **Sequential structure links** — runs after embeddings are committed (avoids race condition)
- **Entity canonicalization** — post-extraction alias resolution (exact → alias → fuzzy match → new)
- **Idempotent writes** — deterministic chunk keys with `ON CONFLICT DO UPDATE`
- **Ingest dedup** — duplicate extracted elements (same modality+page+section+text+bbox) suppressed before persistence; text chunks deduplicated by content before embedding to prevent redundant ArcadeDB vectors
- **Retrieval diversity** — content-level deduplication across all search modes: `_text_vector_search` over-fetches candidates (`RETRIEVAL_DIVERSITY_OVERSAMPLE_FACTOR`, default 8×) then deduplicates by `(document_id, page_number, normalized_text)` keeping highest score; hybrid pipeline applies same diversity pass after chunk-id dedup
- **Batch ArcadeDB writes** — entities and relationships grouped and upserted via ArcadeDB HTTP batch API (one request per batch instead of per-node)
- **Run/stage tracking** — `pipeline_runs` and `stage_runs` tables for diagnostics
- **Worker split** — optional queue isolation: `docker compose --profile split up`
- **Docling concurrency gate** — Redis semaphore with `DOCLING_CONCURRENCY` permits (default 1) controls parallel Docling conversions; queued tasks wait and retry instead of timing out; health check is advisory (logs warning but proceeds with conversion) to avoid starvation when the Docling service runs CPU-bound VLM conversion; health probe timeout configurable via `DOCLING_HEALTH_TIMEOUT` (default 5s)
- **Docling threadpool isolation** — The Docling service runs conversion in a threadpool (`run_in_threadpool`) so the `/health` endpoint remains responsive during CPU-bound VLM processing; an `asyncio.Semaphore` (capacity from `DOCLING_MAX_CONCURRENT`, default 1 on CPU) gates concurrent conversions and returns 503 when saturated
- **Configurable retries** — retry counts and delays for all pipeline stages configurable via env vars (`PREPARE_MAX_RETRIES`, `EMBED_MAX_RETRIES`, etc.); documents stay in PROCESSING status during retries and only show FAILED after all retries are exhausted; Docling 503 (busy) and `SoftTimeLimitExceeded` retries do NOT consume the retry budget
- **Stage resilience** — derivation tasks return error dicts instead of raising on terminal failure, ensuring `finalize_document` always executes; `SoftTimeLimitExceeded` caught explicitly to return gracefully
- **Truncated JSON repair** — LLM graph extraction output truncated by token limits is automatically repaired via `json-repair` before falling back to `DeterministicExtractionError`
- **Recursive chunk splitting** — when a graph extraction chunk fails with deterministic LLM error, the chunk is recursively halved (2500→1250→625, floor 600 chars) and each sub-chunk retried; partial graph from successful chunks/sub-chunks still allows COMPLETE status; only total failure of all chunks triggers PARTIAL_COMPLETE
- **Batched text embedding + ArcadeDB upserts** — large documents (thousands of text elements) are batched via `EMBED_TEXT_BATCH_SIZE` (default 128) before writing to ArcadeDB
- **Stage run attempt tracking** — each retry creates a separate `stage_runs` row with incrementing `attempt` number, preserving full retry history per stage
- **Task time limits** — `soft_time_limit` / `time_limit` on all tasks read from env-var settings at registration time (not hardcoded), ensuring `.env` tuning takes effect without code changes
- **Foreign language translation** — per-element language detection (`langdetect`) triggers LLM translation of non-English content; translated text replaces `DocumentElement.content_text` for downstream processing; original preserved in MinIO; classification marking detection runs against original text; DoclingViewer offers a "Translate" toggle
- **Image description text search** — LLM-generated image descriptions split into sections, embedded as BGE text vectors in ArcadeDB `TextChunk` vertices, searchable via standard text queries; SAME_ARTIFACT chunk_links tie sections together for graph expansion; `image_url` resolved via artifact_id batch lookup so matching sections display with original image
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

Graph extraction is performed by the **Docling-Graph service** (port 8002) via the `/extract-pass` endpoint using a bundle-based architecture. The orchestrator in `derive_ontology_graph` dispatches the bundle's passes in three time-ordered phases — **identity** (e.g. `radar_identity`, `missile_identity`; emit primary entities), **field-group** (e.g. `radar_power_rf`, `radar_antenna`, `missile_guidance`, `missile_propulsion`; emit property subsets onto those entities), and **relationship** (`system_links`; `input_mode=document_plus_entity_refs`, emits cross-entity edges via ref-ids) — each with its own hand-authored Pydantic schema from the active ontology bundle (`ontology_bundles/air_defense_v3/`, which ships 12 passes). Before dispatch, each pass is narrowed to relevant chunks by the schema-derived `absolute_union` selector (see "Adapting the Chunk Selector" below); a pass that legitimately matches no chunks finalizes as ZERO_YIELD/COMPLETE via the `empty_selection` contract. Each pass sends `{bundle_key, pass_name, docling_document_json, selected_chunks?, upstream_entities?, field_subset?}` to the Docling-Graph service, which returns extracted entities, relationships, and per-entity/field/relationship provenance. After all required passes complete (with per-pass retry + skip logic and a required-pass gate), the merge-and-resolve layer merges entities by `LogicalIdentity`, canonicalizes cross-pass aliases (table overlay / Mechanism A1), resolves relationships by identity-dict or ref_id lookup against `upstream_refs`, validates every edge against the bundle `VALIDATION_MATRIX`, and tracks rejection reasons. A strict **lineage gate** then drops any entity lacking an `element_uid`+page provenance row before import. The three-phase graph import writes nodes (with `tracker.mark()`), then domain edges, then structural edges (MENTIONED_IN from derive_rules). HAS_PROVENANCE edges are auto-created during node upsert. Entities below `GRAPH_NODE_MIN_CONFIDENCE` (default 0.60) and relationships below `GRAPH_REL_MIN_CONFIDENCE` (default 0.55) are filtered at import time. The `GraphWriteTracker` gates rollback on failure — failures before the first graph mutation skip rollback to avoid deleting data from a prior successful run. Per-field lineage is precise: each extracted field's source chunk is taken from the merge-time `__property_provenance` map (the chunk of the batch that emitted that field's value), so a spec value (e.g. "max range 50 km") is attributed to the chunk it physically appears in, not the chunk where the entity name first appears.

## Data Migration (from Neo4j + Qdrant to ArcadeDB)

For installations migrating from the previous Neo4j + Qdrant architecture:

```bash
# 1. Start ArcadeDB
docker compose up -d arcadedb

# 2. Run Alembic migration
./manage.sh --migrate

# 3. Re-ingest documents to rebuild graph + vector data in ArcadeDB
#    (existing Postgres metadata is preserved; only graph/vector data is rebuilt)
```

ArcadeDB replaces both Neo4j (knowledge graph) and Qdrant (vector search) in a single service. No separate collection initialization script is needed — ArcadeDB schemas are created automatically on first use.

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
| **MN-3** | 10 worker + 3 API nodes | 3 GPU replicas, total permits=6 | 6 workers (`concurrency=1`) | 3 workers (`concurrency=3`) | 2 workers (`concurrency=2`) | Managed Redis, Postgres tuned pools, ArcadeDB on dedicated host |
| **MN-4** | 16+ worker + 4 API nodes | 4 GPU replicas, total permits=8 | 8 workers (`concurrency=1`) | 4 workers (`concurrency=4`) | 3 workers (`concurrency=2`) | Separate stateful cluster tier (Redis/Postgres/ArcadeDB) |

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
| 3.0 | ArcadeDB migration: replace Neo4j + Qdrant with ArcadeDB (unified graph + vector store), replace GraphRAG with Louvain community detection + global query strategy, GraphStore Protocol abstraction | Complete |
| 3.1 | ArcadeDB SQL compatibility: UPSERT RETURN AFTER @rid, CONTAINSTEXT replacing LUCENE, SELECT expand() replacing MATCH on V, nested expand for depth, UNIQUE indexes for UPSERT, param collision fix, per-role Ollama URLs (LLM/VLM/embedding), TypeScript build fixes | Complete |
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
| 3.2 | Extraction refactor: remove runtime template generation, bundle-based fixed schemas, 5-pass extraction, per-pass StageRun tracking, merge-and-resolve, tracker-gated rollback, status API with cross-run graph_queryable, CI lints, baseline harness | Complete |
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
│   ├── graph_store.py    #   ArcadeDB graph entity/relationship ingest + query
│   ├── community.py      #   Community detection endpoints
│   ├── trusted_data.py   #   Trusted data proposals + approval + indexing + search
│   ├── governance.py     #   Feedback + patch state machine
│   ├── sources.py        #   Sources CRUD, document upload, watch dirs
│   └── health.py         #   Health check endpoint
├── services/
│   ├── docling_client.py       # HTTP client for Docling conversion service
│   ├── docling_graph_service.py # HTTP client for Docling-Graph extraction service
│   ├── arcadedb_client.py      # ArcadeDB HTTP client (REST/JSON API, token auth)
│   ├── arcadedb_graph.py       # ArcadeDB GraphStore implementation (graph + vector + fulltext)
│   ├── arcadedb_schema.py      # Ontology-driven ArcadeDB schema sync (DDL batching)
│   ├── arcadedb_community.py   # Community detection pipeline (Louvain + LLM reports)
│   ├── graph_store.py          # GraphStore Protocol (async/sync interface)
│   ├── docling_enrichment.py   # DoclingDocument enrichment helpers (translations, context)
│   ├── canonicalization.py     # Entity alias resolution + fuzzy match
│   ├── chunking.py             # Native HybridChunker integration (Docling)
│   ├── reranker.py             # Cross-encoder reranker (bge-reranker-v2-m3)
│   ├── translation.py          # Foreign language detection + LLM translation
│   ├── extraction_merge.py        # Merge + resolve logic for bundle-passes path
│   ├── status_signals.py          # Cross-run graph_queryable computation
│   ├── ontology_bundles.py        # Worker-side bundle loader (manifest, coverage, resolve)
│   ├── ontology_templates.py      # YAML → Pydantic extraction templates + validation
│   └── storage.py                 # MinIO storage operations
├── workers/
│   ├── pipeline.py             # Celery ingest pipeline (sequential 12-stage chain)
│   ├── dispatch_types.py       # IngestDispatchResult frozen dataclass
│   ├── trusted_data_tasks.py   # Celery task for trusted data embedding + ArcadeDB indexing
│   └── watcher.py              # Celery Beat directory watcher
├── models/               # SQLAlchemy ORM (ingest, retrieval, governance, auth, trusted_data)
└── schemas/              # Pydantic request/response schemas
docker/
├── docling/              # Docling VLM conversion service (granite-docling-258M)
├── docling-graph/        # Docling-Graph extraction service (ontology-driven, port 8002)
├── arcadedb/             # ArcadeDB built from source (Dockerfile + JDK 21)
└── postgres/             # Custom Postgres (metadata, governance)
ontology_bundles/
├── __init__.py
├── _shared/
│   └── limits.py               # Shared schema-size threshold constant
└── air_defense_v3/
    ├── __init__.py
    ├── ontology.yaml           # Canonical ontology (identity_fields + identity_scope)
    ├── manifest.yaml           # 5-pass registry (names, kinds, input modes, entity/rel types)
    ├── coverage.yaml           # Extract/derive bucket declarations
    ├── validators.py           # Shared Pydantic field validators
    ├── derive_rules.py         # Deterministic structural edge derivation (MENTIONED_IN)
    └── extraction_schemas/
        ├── __init__.py
        ├── reference.py        # SECTION, FIGURE, TABLE, ASSERTION
        ├── radar_domain.py     # 7 primary + 2 bridge entities + relationships
        ├── missile_domain.py   # 5 primary + 2 bridge entities + relationships
        ├── other_systems.py    # 5 primary + 2 bridge entities + relationships
        └── system_links.py     # Cross-pass relationships only (ASSOCIATED_WITH, CUES)
tools/
├── check_extraction_coverage.py   # Bundle coverage checker (13 rules + manifest consistency)
├── ci_lints.sh                    # 8 legacy-reference prevention lints
├── extraction_baseline_harness.py # Soak comparison CLI tool
└── extraction_coverage/
    ├── rules.py                   # Rule implementations
    └── manifest_consistency.py    # Manifest self-consistency sub-checks
scripts/
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
├── integration/          # API tests against real Postgres/Redis/MinIO/ArcadeDB stack
├── native/               # Native ArcadeDB integration tests (real DB)
├── pipeline/             # Pipeline task tests
├── e2e/                  # End-to-end workflow tests
└── fixtures/             # Sample documents for test pipelines
```
