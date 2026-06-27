# EIP-MMDPP — Complete System Description

> **Status:** current as of the `main` merge (absolute_union chunk selection + 354-commit branch integration).
> **Scope:** every workflow and feature, end to end. For setup/usage commands see [`README.md`](../README.md); for ontology authoring see the README's "Adding a Custom Ontology Bundle" and "Adapting the Chunk Selector" sections.

---

## 1. What EIP-MMDPP Is

EIP-MMDPP (Enterprise Intelligence Platform — Multi-Modal Document Processing Pipeline) is an **air-gappable, multi-modal document understanding and retrieval platform** for defense/military intelligence use cases. It ingests heterogeneous documents (PDF, DOCX, PPTX, XLSX, HTML, Markdown, CSV, images, technical drawings), converts them with Docling, and builds three interlocking knowledge layers:

1. **A vector-searchable chunk store** — text (BGE-M3, 1024-d) and image (SigLIP2, 1024-d) embeddings in ArcadeDB.
2. **An ontology-driven knowledge graph** — entities and relationships extracted by a bundle-defined multi-pass LLM pipeline, with strict chunk-level lineage.
3. **A community/synthesis layer** — Louvain community detection over the graph plus LLM-generated community reports for global question answering.

On top of these it provides **hybrid multi-modal retrieval** (text + image + graph-structure fusion with cross-encoder reranking), a **governed trusted-data layer** (human-reviewed knowledge), a **feedback → curator-patch governance loop**, and a **React web UI**. All ML inference can run fully locally via Ollama — no cloud calls required.

**Design principles that recur throughout the system:**
- **Schema-driven, not name-driven.** Extraction and chunk selection operate on entity types, field units, and enum fields — never hardcoded equipment names — so a new corpus is onboarded by authoring an ontology bundle, not by editing engine code.
- **Complete lineage.** Every committed entity, field, and relationship carries source chunk IDs and page numbers; every retrieval result carries exact source text, document, page, and trust metadata.
- **Reliability by construction.** A ledger-based orchestrator with a progress-aware watchdog distinguishes "slow but progressing" from "stalled," so long extractions on dense documents complete instead of being killed.

---

## 2. Architecture

### 2.1 Knowledge Layers

```
┌───────────────────────────────────────────────────────────┐
│                 ArcadeDB (multi-model store)               │
│  Documents ←→ TextChunk / ImageChunk / ExtractionChunk     │
│  Domain entity vertices (RADAR_SYSTEM, MISSILE_SYSTEM, …)   │
│  Domain + structural edges (EXTRACTED_FROM, BELONGS_TO, …)  │
│  Collection (source) vertices, Alias vertices              │
│  Native HNSW vectors: BGE-M3 1024-d text, SigLIP2 1024-d   │
└──────────────┬───────────────────────────┬────────────────┘
               │                           │
    ┌──────────▼─────────────┐   ┌─────────▼──────────────┐
    │  Community Layer        │   │  Trusted Data Layer    │
    │  Louvain detection      │   │  TrustedTextChunk      │
    │  LLM community reports   │   │  human-review gate     │
    │  Global synthesis        │   │  PROPOSED→APPROVED…    │
    └─────────────────────────┘   └────────────────────────┘
```

### 2.2 Technology Stack

| Component | Technology |
|---|---|
| API | FastAPI (Python 3.11, async SQLAlchemy) |
| Task processing | Celery + Redis (broker + result backend) |
| Relational store | PostgreSQL 16 (documents, pipeline runs, pass outputs, governance) |
| Graph + vector store | ArcadeDB (knowledge graph + native HNSW vector search) |
| Object storage | MinIO (raw documents + derived blobs) |
| Document conversion | Docling (PdfPipeline: dlparse_v4 + EasyOCR + TableFormer; SimplePipeline for Office/HTML/MD) |
| Graph extraction | `docling-graph` service (vendored + patched), `/extract-pass` endpoint, port 8002 |
| Text embeddings | `bge-m3` (1024-d) via Ollama `/v1/embeddings` |
| Image embeddings | OpenCLIP `ViT-L-16-SigLIP2-256` (pretrained `webli`, 1024-d), local |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder |
| LLM (all functions) | Ollama, `gemma4:31b` (graph extraction, doc analysis, picture description, translation, community reports) |
| Frontend | React 18 + TypeScript + Vite |

### 2.3 Service Topology (13 containers; split-worker default)

| Service | Role | Image vs bind-mount |
|---|---|---|
| `api` | FastAPI app server | COPY image + bind-mount `./app`, `./ontology_bundles` |
| `worker` | Celery worker, all queues (mixed/legacy mode) | COPY image + bind-mount `./app` |
| `worker-ingest` | Celery worker, `celery,ingest,extract` queues (split) | bind-mount `./app` |
| `worker-embed` | Celery worker, `embed` queue (split) | bind-mount `./app` |
| `worker-graph` | Celery worker, `graph,graph_extract` queues (split) | bind-mount `./app` |
| `beat` | Celery Beat scheduler (dispatcher tick, watchdog, watcher, community) | bind-mount `./app` |
| `postgres` | PostgreSQL 16 | COPY image |
| `redis` | Celery broker + result backend | stock image |
| `minio` (+ `minio-init`) | S3-compatible object storage | stock image |
| `arcadedb` | Knowledge graph + native vector search | COPY image |
| `docling` | Document OCR/conversion (granite-docling VLM) | COPY image |
| `docling-graph` | Ontology-driven extraction service | COPY image (vendored `repo/` + `patches/`) |
| `jupyter` | Analysis notebooks | bind-mount |

**Deployment note.** `worker*`/`api`/`beat` load code from the host `./app` bind-mount (restart picks up changes — no rebuild). `docling-graph`, `docling`, `arcadedb`, `postgres` are baked COPY images (changes require `docker compose build`). The compose project (`-p eip-mmdpp`) must be anchored to the checkout you intend to run from — its `./app`, `./ontology_bundles`, and `.env` resolve relative to the project directory.

### 2.4 Named Volumes (persistent data)

| Volume | Holds | Essential? |
|---|---|---|
| `postgres_data` | all run history, pass outputs, governance, metrics | **yes** |
| `arcadedb_data` | the knowledge graph + vector indexes | **yes** |
| `arcadedb_backups` | ArcadeDB backup snapshots | yes |
| `minio_data` | raw documents (`eip-raw`) + derived blobs (`eip-derived`) | **yes** |
| `redis_data` | broker/result state | rebuildable |
| `model_cache` | HuggingFace/OpenCLIP model cache | re-downloadable |
| `watch_dirs`, `celerybeat_data` | directory-watcher inputs, beat schedule | non-essential |

`./manage.sh --blow-away` runs `docker compose down -v` (deletes **all** of the above) plus `rm -rf reports/` — irreversible. Back up `postgres_data`, `arcadedb_data`, and `minio_data` first.

---

## 3. Core Workflows

### 3.1 Document Ingestion Pipeline

A document is uploaded (`POST /v1/sources/{source_id}/documents`) streaming to MinIO, deduped by file hash, recorded in Postgres (`pipeline_status=PENDING`), and `start_ingest_pipeline()` creates a `PipelineRun` and seeds the first ledger stage. A **ledger-based dispatcher** (`dispatch_pending_pipeline_stages()`, ~5 s beat tick) atomically advances each stage `PENDING → DISPATCHED → RUNNING → COMPLETE/FAILED` and publishes the corresponding Celery task.

**Stages (sequential):**

| # | Stage | Queue | Purpose |
|---|---|---|---|
| 1 | `prepare_document` | ingest | Docling conversion, element extraction, Unicode normalization |
| 2 | `detect_and_translate` | ingest | Language detection + LLM translation |
| 3 | `derive_document_metadata` | ingest | LLM: summary, date, classification, source characterization |
| 4 | `derive_picture_descriptions` | ingest | VLM image descriptions (with document context) |
| 5 | `purge_document_derivations` | ingest | Idempotent cleanup of prior derived data (re-ingest safety) |
| 6 | `derive_text_chunks_and_embeddings` | embed | Structure-aware chunking + BGE-M3 embeddings |
| 7 | `derive_image_embeddings` | embed | SigLIP2 image embeddings |
| 8 | `derive_document_anchors` | graph | Docling-anchor walker → SECTION vertices + structure |
| 9 | `derive_ontology_graph` | graph_extract | **Fan-out** to per-pass extraction, then **fan-in** merge |
| 10 | `collect_derivations` | ingest | Post-extraction checkpoint |
| 11 | `derive_structure_links` | graph | EXTRACTED_FROM + chunk-link edges |
| 12 | `derive_canonicalization` | graph | Entity alias resolution |
| 13 | `finalize_document` | ingest | Terminal status assignment |

**Status semantics.** `PipelineRun.status` and `Document.pipeline_status`: `PENDING → PROCESSING →` one of `COMPLETE` (all required stages succeeded, no review flag), `PARTIAL_COMPLETE` (failed/missing stages or degraded extraction), `PENDING_HUMAN_REVIEW`, or `FAILED` (retry cap hit). A document's extraction quality is separately classified `ok | degraded | anomaly` (`_classify_extraction_quality`): `ok` = ≥1 domain pass produced entities; `degraded` = no domain hits but chunks/sections exist; `anomaly` = upstream processing broke.

**Chunking.** `structure_aware_chunk` / Docling `HybridChunker` (tokenizer-aware, default 512 tokens / 64 overlap) never splits a table or equation mid-chunk and keeps headings with following content. Chunk IDs are deterministic (`uuid5(document_id:element_uid)`) for idempotent re-ingest. Optional table normalization (`EMBEDDING_TABLE_NORMALIZATION_ENABLED`) renders matrix/list tables to normalized rows + a summary. With `EXTRACTION_INDEX_MODE=merged` (current), the index stores one `ExtractionChunk` row per HybridChunker output chunk (vs `per_element`).

**Re-ingest** (`POST /documents/{id}/reingest`, modes `full | embeddings_only | graph_only`) re-runs the relevant stages. **Cancel** (`POST .../cancel`) and **DELETE** hard-delete the document and all derived data across Postgres, ArcadeDB, and MinIO — they are *not* graceful stops.

### 3.2 Knowledge Extraction

Extraction is governed by an **ontology bundle** (`ontology_bundles/<key>/`) — for the reference corpus, `air_defense_v3`. Stage 9 (`derive_ontology_graph`) reads the bundle manifest and runs its **passes** in three time-ordered phases:

- **Identity passes** (`phase: identity`, `input_mode: document_only`) — emit primary entity instances keyed by identity fields (`graph_id_fields`, e.g. `system_name`). Examples: `radar_identity`, `missile_identity`.
- **Field-group passes** (`phase: field_group`, `document_only`) — emit property subsets for the same entity types (e.g. `radar_power_rf`, `radar_antenna`, `missile_guidance`, `missile_propulsion`). Each pass extracts only its focus fields; the merge step accumulates them onto one entity by identity.
- **Relationship pass** (`phase: relationship`, `input_mode: document_plus_entity_refs`) — `system_links` receives the upstream entity catalog as a prompt preamble and emits cross-entity relationships (`ASSOCIATED_WITH`, `CUES`, `VARIANT_OF`) using ref-ids resolved at merge time. `depends_on` lists all upstream passes; `skip_if_no_upstream_endpoints` skips it when no linkable entities exist.

`air_defense_v3` ships **12 passes** (2 identity + 9 field-group + 1 relationship).

**The docling-graph service.** Each pass is an HTTP `POST /extract-pass` to the `docling-graph` service with the (optionally chunk-scoped) DoclingDocument, bundle/pass keys, optional `upstream_entities`, optional `selected_chunks`, and optional `field_subset`. The service loads the pass's Pydantic template class, runs a **DeltaOrchestrator** that chunks the document and issues parallel LLM batches (default 2 concurrent workers per request), applies an **evidence gate** (rejects extractions whose values aren't supported by the batch text), and returns the populated template plus three provenance streams (entity, field, relationship). A **table overlay** (Mechanism A1) deterministically parses table facts and cross-entity aliases before the LLM call.

The service runs a **vendored, patched** copy of the docling-graph library (`docker/docling-graph/repo/`, gitignored) with tracked patches:

| Patch | Effect |
|---|---|
| `0001` orchestrator batch hard timeout | Inter-arrival (no-progress) watchdog — aborts only if *no* batch completes within `DOCLING_GRAPH_BATCH_HARD_TIMEOUT_SECONDS`; slow-but-progressing passes never trip |
| `0002` chunked batches | Accept pre-built `selected_chunks` (merged-chunk routing) |
| `0003` delta-prompt death-spiral fix | Drops contradictory "context only" batch header that caused gemma4 0-relationship loops |
| `0004` f3 subset schema | Field-subset extraction via thread-local override |
| `0005` lineage chunk metadata | chunk_index/page_number in entity provenance |
| `0006` positional node/rel provenance | Positional `self_refs[]` / `chunk_indexes[]` lists for multi-element provenance |

In-process monkey-patches add a per-pass DOCUMENT-CONTEXT cap (so `document_plus_entity_refs` passes see the full upstream catalog, not a 600-char truncation) and prompt/resolver/gleaning tweaks.

**Merge and resolution** (`merge_and_resolve`, worker-side):
1. **Table-overlay canonicalization** — cross-pass duplicates (e.g. an industry vs NATO designation of the same system) are aliased to one canonical identity *before* identity computation; an `identity_aliases` map is exposed downstream.
2. **Entity merge** — every instance gets a `LogicalIdentity` (`entity_type` + identity field tuple + scope `global|document`); collisions union non-null properties and take max confidence.
3. **Provenance aggregation** — per-entity and per-field evidence rows are grouped onto the merged record.
4. **Relationship resolution** — DTO edges (system_links) resolve `from_ref_id`/`to_ref_id` against `upstream_refs`; typed-edge templates emit edges via a graph walk. Every edge is triple-checked against the bundle's `VALIDATION_MATRIX[(from_type, rel_type, to_type)]`; invalid triples and unresolved refs are rejected with explicit reasons.

**Lineage (strict gate).** Before anything is committed, `_partition_entities_by_lineage()` requires every entity to have ≥1 provenance row with a non-empty `element_uid` **and** a non-null `page`. Entities failing the gate (and edges referencing them) are dropped with an ERROR log. Committed lineage:
- **Entity→chunk**: `EXTRACTED_FROM` edges carrying `source_chunk_ids` + `source_pages`, resolved from positional `self_refs`.
- **Field→chunk**: per-field evidence (`__property_provenance`) so each field value points at the exact chunk it came from (not the entity's first-seen chunk).
- **Relationship→chunk**: per-edge `source_chunk_ids` / `source_pages`.

### 3.3 Chunk Selection (`absolute_union`)

To avoid extracting every chunk with every pass (slow, and a hallucination risk), a **vector router** narrows each pass to the chunks likely to contain its fields. The production selector is **`absolute_union`** — a schema-derived, training-free rule. The endpoint `POST /v1/extraction/chunk-scope` scores candidate chunks and keeps a chunk for a pass iff **any** signal fires:

```
keep ⟺ measurement(pass) ∨ categorical(pass) ∨ image_presence(pass) ∨ max_field_cosine ≥ cosine_tau
```

All signals derive at request time from the bundle's extraction schema — no per-corpus training:
- **measurement** — the pass has fields with unit suffixes (`_km`, `_mhz`, `_kg`, `_sec`, `_deg`, `_dbi`, `_kw`, `_mps`, …); the chunk text contains a matching unit token (bounded matcher, ~80 surface forms).
- **categorical** — the pass has enum fields (`scan_type`, `guidance_type`, `seeker_type`, …); the chunk contains a known enum phrase ("phased array", "active radar homing", …).
- **image** — the pass has `*_photo`/`*_image` fields; the chunk has a `#/pictures/` source ref.
- **cosine** — best per-field dense similarity ≥ `cosine_tau` (default **0.55**, per-pass tunable in the manifest).

`select_candidates()` returns 0-to-all chunks (no `k_min`/`k_max`/quantile) and emits diagnostics (`selection_k`, `measurement_keeps`, `categorical_keeps`, `image_keeps`, `cosine_keeps`).

**The empty_selection contract.** Because `absolute_union` is a *true* selection (zero is a valid answer, not a failure), when it selects nothing the endpoint returns `mode="empty_selection"` (it does **not** walk the legacy relaxed→lexical→identity→full fallback ladder). The worker maps that to a zero-chunk scope, makes **no** docling-graph call, and finalizes the pass as **COMPLETE / ZERO_YIELD**. This is distinct from a genuine failure and from a fall-open to full-doc.

**Selector history.** `absolute_union` replaced the earlier `guarded_quantile` selector (gate ∪ quantile ranker, which required per-corpus fitting). On the bake-off ground truth, `absolute_union` reaches ~95.7% recall at ~24% of chunks selected vs `guarded_quantile`'s ~60.9% at ~51%. `selection_mode` ∈ `{topk, guarded_quantile, absolute_union}`; all four shipped `air_defense_v3` bundles use `absolute_union` / `cosine_tau=0.55`.

**Vector router modes** (`VECTOR_ROUTER_MODE`): `disabled` (always full-doc), `shadow` (compute + log but always dispatch full-doc — safe rollout), `narrow_only` (current — narrow on `selected_refs`/`empty_selection`; fall open to full-doc on degraded fallback or when the document is below `NARROW_MIN_DOC_TOKENS`, default 0, which guards small docs from recall loss). `apply_chunk_scope()` rewrites the DoclingDocument's `body.children` to the selected refs plus their section headings, preserving the `texts[]`/`tables[]`/`pictures[]`/`groups[]` arrays by reference (including correct re-parenting of nested list groups so DoclingDocument validation passes).

### 3.4 Graph Construction & Lineage

Merged entities become ArcadeDB vertices (one class per ontology entity type, plus `Document`, `Collection`, `TextChunk`, `ImageChunk`, `ExtractionChunk`, `CommunityReport`, `Alias`). Edges include domain relationships (ontology-defined) and structural edges: `EXTRACTED_FROM` (entity→chunk lineage), `BELONGS_TO` (document→collection), `NEXT_CHUNK`/`SAME_PAGE`/`SAME_SECTION`/`SAME_ARTIFACT` (chunk graph for retrieval expansion), `HAS_SECTION`/`HAS_FIGURE`/`HAS_TABLE`/`CHILD_OF` (document structure), `HAS_ALIAS`. Edge properties carry `document_ids` (idempotent union on re-upsert), `pipeline_run_id`, `source_chunk_ids`, `source_pages`, and per-edge `source_self_refs`.

### 3.5 Community Detection & Reports

`POST /v1/community/detect` (or a periodic beat task / post-ingest trigger above a document threshold) runs ArcadeDB-native **Louvain** community detection over domain entities. Each community's members + relationships + evidence are summarized by an LLM (`COMMUNITY_REPORT_LLM_PROMPT`) into a `{title, summary}` `CommunityReport` vertex with its own embedding. A membership hash makes regeneration incremental (unchanged communities are skipped). Source documents per community are collected by traversing `EXTRACTED_FROM`.

### 3.6 Retrieval / RAG

`POST /v1/retrieval/query` is the unified entry point, with strategies:

- **basic** — BGE text vector search over `TextChunk` (excludes long `image_description` chunks), diversify, rerank.
- **hybrid (multi-modal)** — the full pipeline:
  1. Parallel text (BGE) + image (CLIP/SigLIP, or text-projected-to-image) vector searches (over-fetched by `diversity_oversample_factor`, capped at `diversity_max_candidates`, floored at `min_score_threshold`).
  2. Parallel per-seed expansion (≤16 concurrent): document-structure traversal (`NEXT_CHUNK`/`SAME_SECTION`/`SAME_ARTIFACT`/`SAME_PAGE` with weights and hop penalty) and ontology traversal (entity→related entity→chunk); ontology chunks are re-scored against the query embedding.
  3. Merge + dedupe by chunk + content-diversify, filter by modality, sort, cap.
  4. Cross-encoder rerank (`bge-reranker-v2-m3`) of the top-N.
- **global** — embed the query, vector-search `CommunityReport`, and LLM-synthesize an answer across the top reports (`COMMUNITY_GLOBAL_SYNTHESIS_PROMPT`), returning one result with cited reports and deduplicated source documents.

**Score fusion:** `semantic_weight·sem + doc_structure_weight·max(doc,cross) + ontology_weight·onto + mil_id_bonus` (defaults 0.65 / 0.20 / 0.15, plus a small bonus when military designators match). **Every result carries lineage**: `document_name`, `source_characterization`, `date_of_information`, `classification`, `page_numbers`, `self_refs`/`evidence_ids`, optional `table_chunk` block, and a `context` describing how the chunk was reached. Image results expose proxy URLs (`/v1/images/{chunk_id}`, `/v1/images/artifact/{artifact_id}`). `GET /v1/agent/context` returns the same retrieval as ready-to-inject markdown for LangGraph agents.

### 3.7 Governance

- **Trusted data** — `POST /v1/trusted-data/ingest` creates a `PROPOSED` submission; a curator approves (`→ APPROVED_PENDING_INDEX`, enqueues a Celery indexing task that embeds it into a `TrustedTextChunk` → `APPROVED_INDEXED`) or rejects. `POST /v1/trusted-data/query` vector-searches the approved trusted layer.
- **Feedback → patches** — `POST /v1/feedback` on a retrieval result auto-generates an RFC-6902 patch (types: wrong_text, wrong/missing entity/relationship, merge/delete entity, …). Patches move `under_review → approved → (dual_approved if required) → applied`. `governance_dual_approval_required` (default false) gates whether two distinct curators are needed before a patch applies to the platform.

---

## 4. Reliability & Orchestration

- **Ledger dispatcher** — stages are rows with explicit states; a beat tick advances them atomically, so a crashed worker leaves a recoverable row rather than a lost chain.
- **Reconciliation watchdog** (`reconcile_ontology_graph_runs`, ~60 s) repairs: stale-claimed phases (dispatcher crashed before `.delay`), completed-but-not-terminal phases (task wrote output then crashed), and stuck-without-advance runs. With `RECONCILER_PROGRESS_AWARE=true` it consults a per-pass progress map (fed by the `DG_PROGRESS_POLLER_ENABLED` poller hitting docling-graph `/progress`) and **skips reclaiming passes still making progress** within `RECONCILER_NO_PROGRESS_THRESHOLD_S` (2 h) — this is what lets a 2 h+ identity pass on a dense document finish instead of being killed (the R1→R2 "progress-aware watchdog").
- **Retry ladder** — per-pass: transport errors retry without bumping attempts; retryable (5xx/partial) retry up to `pass_max_retries` with backoff; terminal (4xx/validation) do not retry. A periodic stale-run sweeper restarts whole runs up to `MAX_DOC_RETRY_COUNT` before marking a document FAILED. A `GraphWriteTracker` only rolls back the graph if a mutation actually occurred.
- **Idempotency** — deterministic chunk IDs, `purge_document_derivations`, and idempotent edge `document_ids` unions make re-ingest safe.

---

## 5. Ontology Bundles

A bundle is a self-contained Python package under `ontology_bundles/<key>/`:

| File | Defines |
|---|---|
| `manifest.yaml` | pass list (phase, kind, input_mode, template_class, depends_on, retrieval/selection config) |
| `ontology.yaml` | entity/relationship type definitions |
| `entities.py` | generated Pydantic entity classes (identity fields, typed-edge fields) |
| `relationships.py` | `RelationshipType` enum |
| `validators.py` | field coercion/normalization |
| `extraction_schemas/<pass>.py` | per-pass Pydantic template + records (focus fields only) |
| `validation_matrix.py` | `VALIDATION_MATRIX[(from,rel,to)] → bool` |
| `coverage.yaml` | per-pass field coverage metrics |
| `derive_rules.py` | custom structural-link derivation |

**Selectivity by design.** Because chunk selection is schema-derived, the bundle author controls routing through schema choices: unit-suffixed fields make a pass measurement-selective, enum fields make it categorical-selective, `*_photo` fields make it image-selective, and cosine catches the rest. There are four `air_defense_v3` bundles (production + three narrowed siblings); **schema changes land in `air_defense_v3` first, then propagate** to the siblings. The README's "Adding a Custom Ontology Bundle" and "Adapting the Chunk Selector to a New Corpus or Ontology" sections are the authoring guides.

---

## 6. Configuration

- **LLM provider** — `LLM_PROVIDER` ∈ `ollama | openai | mock`. Production is `ollama` (air-gapped). A cascading multi-bank Ollama pool routes each function to its own host(s): role-level pools (`OLLAMA_LLM_BASE_URLS`, `OLLAMA_VLM_BASE_URLS`, `OLLAMA_EMBEDDING_BASE_URLS`) with per-function overrides (`DOCLING_GRAPH_LLM_BASE_URLS`, `COMMUNITY_REPORT_LLM_BASE_URLS`, …), load-balanced least-in-flight.
- **Models (deployed)** — text `bge-m3` (1024-d); image `ViT-L-16-SigLIP2-256`/`webli` (1024-d); reranker `bge-reranker-v2-m3` (CPU); all generative functions `gemma4:31b`.
- **Selection/router** — `VECTOR_ROUTER_MODE=narrow_only`, `EXTRACTION_INDEX_MODE=merged`, `NARROW_MIN_DOC_TOKENS`, `WORKER_FORWARD_SELECTED_CHUNKS`; per-pass `selection_mode`/`cosine_tau` in manifests.
- **Reliability** — `RECONCILER_PROGRESS_AWARE`, `RECONCILER_NO_PROGRESS_THRESHOLD_S`, `DG_PROGRESS_POLLER_ENABLED`, `DOCLING_GRAPH_BATCH_HARD_TIMEOUT_SECONDS`, `PASS_SOFT_TIME_LIMIT`, `MAX_DOC_RETRY_COUNT`, plus generous `DOCLING_GRAPH_*_TIMEOUT` values for dense-doc extraction.
- **Retrieval** — `RETRIEVAL_*_WEIGHT`, `RERANKER_*`, `RETRIEVAL_DIVERSITY_*`, `QUERY_DEFAULT_TOP_K`, `RETRIEVAL_MIN_SCORE_THRESHOLD`.

`.env.example` mirrors the live `.env` (config values synced; secrets kept as `REPLACE_ME` placeholders). Every new env var must appear in **both** `.env` and `.env.example`.

---

## 7. API Surface (v1)

All routes are under `/v1` (`app/api/v1/router.py`).

- **Health/ops** — `GET /health`, `GET /health/ready`.
- **Sources & documents** — create/list sources; upload/list documents; status, extraction-status, batch-status, stages; reingest; cancel; delete; artifacts; docling/docling-raw/metadata/translation/element-translations/image-descriptions; artifact image stream.
- **Directory watcher** — register/list/remove watch directories.
- **Graph store** — ingest entity/relationship; query/neighborhood.
- **Query profiles & ontology registry** — registry CRUD + activate; profile append/update/delete; active profiles; default template; section/dossier search.
- **Trusted data** — ingest; list proposals; approve/reject/reindex; query.
- **Unified retrieval** — `POST /retrieval/query`; image proxies; `GET /settings/retrieval`.
- **Governance** — feedback; patch list/get/approve/reject/apply.
- **Community** — detect; status; settings; reports list/get.
- **Extraction routing** — `POST /extraction/chunk-scope` (per-pass selection).
- **Agent** — `GET /agent/context` (markdown for LangGraph).

Full interactive docs at `/docs`.

---

## 8. Operations

`./manage.sh` is the operations CLI:

| Command | Action |
|---|---|
| `--start` / `--start-split` | build + start (split workers, default); pulls vendored repos, downloads models, waits for health |
| `--start-mixed` | single mixed worker (legacy/low-load) |
| `--stop` | stop containers, **preserve data** (`down --remove-orphans`) |
| `--restart` | restart without rebuild |
| `--status` / `--worker-status` | service + Celery health |
| `--logs [service]` | stream logs |
| `--migrate` / `--seed` / `--db-shell` | alembic upgrade / ontology seed / psql |
| `--test [unit\|integration\|e2e]` | run the test suite (`scripts/run_tests.sh`) |
| `--blow-away` | **destroy everything** (`down -v` + `rm -rf reports/`) — irreversible, confirms first |

**Backup before reset.** `postgres_data` → `pg_dump`; `arcadedb_data`/`minio_data` → volume tar (or ArcadeDB backup). `reports/` is deleted by `--blow-away`, so back it up too. Testing is layered (unit → integration → e2e) with a Docker stack overlay (`docker-compose.test.yml`).

---

## 9. Frontend & Tooling

- **`frontend/`** — React 18 + TypeScript + Vite web UI for upload, retrieval/search, the Docling viewer (with translation overlays and image descriptions), query-profile editing, governance/patch approval, community browsing, and status monitoring.
- **`notebooks/`** — Jupyter notebooks for analysis/experimentation.
- **`tools/`, `scripts/`** — admin/eval utilities (ontology seeding, ranker fitting, test runner, corpus surveys).

---

## 10. Data Model Reference (selected)

**PostgreSQL** (`ingest` schema): `documents`, `pipeline_runs`, `stage_runs` (ledger + `metrics.progress`), `pipeline_pass_outputs` (per-pass `execution_status`, `yield_status`, entity counts, diagnostics), `artifacts`, `document_elements`; `retrieval` schema: `text_chunks`, `image_chunks`, `chunk_links`; governance/trusted-data tables; query-profile registries.

**ArcadeDB** vertices: `Document`, `Collection`, `TextChunk`, `ImageChunk`, `ExtractionChunk` (per-run, HNSW vector), `TrustedTextChunk`, `CommunityReport`, `Alias`, and one class per ontology entity type. Edges: `EXTRACTED_FROM`, `BELONGS_TO`, `CONTAINS_TEXT`/`CONTAINS_IMAGE`, `NEXT_CHUNK`/`SAME_PAGE`/`SAME_SECTION`/`SAME_ARTIFACT`, `HAS_SECTION`/`HAS_FIGURE`/`HAS_TABLE`/`CHILD_OF`, `HAS_ALIAS`, plus ontology domain edges.

---

## 11. Glossary

- **Bundle** — a self-contained ontology + extraction-schema package defining what is extracted and how passes are routed.
- **Pass** — one LLM extraction unit (identity / field-group / relationship).
- **LogicalIdentity** — the merge key for an entity (type + identity field values + scope).
- **absolute_union** — the schema-derived chunk selector (union of measurement/categorical/image/cosine signals).
- **empty_selection** — a legitimate zero-chunk selection → pass finalizes COMPLETE/ZERO_YIELD.
- **Lineage gate** — the invariant that every committed entity has an `element_uid` + page provenance.
- **Vector router** — the component that narrows each pass to selected chunks (`narrow_only` in production).
- **R1/R2 watchdog** — the reconciler; R2 adds progress-awareness so slow-but-progressing passes are not reclaimed.
