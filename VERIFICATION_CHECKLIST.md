# EIP-MMDPP Feature & Behavior Verification Checklist

> **Purpose:** After every code change, bug fix, or new feature, this checklist must be reviewed to ensure no existing features have been removed or broken. Run the full test suite first (`./scripts/run_tests.sh`), then verify the items below that are relevant to the changed code.
>
> **Protocol:** Every code modification requires: (1) full unit test suite passes, (2) relevant sections of this checklist verified, (3) any new features added to this checklist before merging.

---

## 1. INGEST PIPELINE

### 1.1 Document Preparation & Validation

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Docling document conversion (VLM-based PDF parsing) | PDFs cannot be parsed; documents stuck in PROCESSING | Upload PDF, check status progresses through PREPARE stage | 1, 2.21 |
| Standalone image file synthesis (JPEG, PNG, TIFF, BMP, GIF, WEBP) | Standalone images ingest with 0 elements; blank viewer | Upload `.jpg`, check `document_elements` has 1 image element with `storage_key` | 2.30 |
| Unicode normalization (em-dashes, non-breaking spaces) | NaN embeddings from bge-m3; elements silently drop from vectors | Ingest doc with em-dashes; verify Qdrant vectors are valid floats | 2 |
| Stale run cleanup on worker startup | Crashed workers leave documents in PROCESSING permanently | Kill worker mid-ingest, restart; document reverts to PENDING | 2.18 |
| Re-upload on failure | Re-uploading failed doc returns 409 indefinitely | Upload doc, let it fail, re-upload same file without error | 2 |

### 1.2 Parallel Derivations & Chord Resilience

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Parallel chord (text/image/graph run concurrently) | 3x ingest latency per document | Upload doc; verify text, image, graph tasks overlap in time | 1, 2.22 |
| Chord tasks return error dicts instead of raising | Single task failure kills entire document pipeline | Ingest doc with corrupted image; pipeline still reaches PARTIAL_COMPLETE | 2.22 |
| Chord `on_error` errback marks doc FAILED | Hard time limit kill leaves document in PROCESSING forever | Set very short time limit, ingest large doc; verify FAILED not PROCESSING | 2.22 |

### 1.3 Element Deduplication

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Ingest-time element dedup (modality+page+section+text+bbox) | Duplicate elements bloat vector store; redundant search results | Parse doc with duplicate captions; check `document_elements` has only unique entries | 2.8, 2.20 |
| Image dedup includes raw bytes hash | Distinct images on same page with empty captions silently dropped | Ingest doc with 2+ distinct images on same page; both appear in `document_elements` | 2.30 |
| Text chunk dedup before embedding | Duplicate text vectors waste Qdrant space | Doc with repeated sections appears only once per unique text in Qdrant | 2.20 |

### 1.4 Picture Description & Image Analysis

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| LLM-generated image descriptions (multimodal VLM) | Images not text-searchable; CLIP embedding insufficient for specific queries | Ingest doc with images; call `/v1/documents/{id}/image-descriptions`; receive descriptions | 2.9 |
| Image descriptions split into sections and embedded as BGE text vectors | Image descriptions not discoverable via text search | Query concept visible in image; find `image_description` modality results | 2.26 |
| SAME_ARTIFACT chunk_links between description sections | Description sections orphaned from source image; graph expansion fails | Text query matches description section; expansion retrieves siblings + original image | 2.26 |
| Image description hover tooltips in DoclingViewer | No context when hovering over embedded images in viewer | Open PDF in viewer; hover over image; see "AI Image Analysis" tooltip | 2.30 |
| Standalone image display with description panel | Standalone images appear with no visual or description | Open standalone image in viewer; see image rendered + AI Image Analysis panel | 2.30 |
| `image-descriptions` endpoint returns `artifact_id` | Frontend cannot render standalone image in fallback panel | Call `/v1/documents/{id}/image-descriptions`; response includes `artifact_id` | 2.30 |
| SoftTimeLimitExceeded handler for picture descriptions | Timeout kills task without cleanup; document stuck PROCESSING | Set short limit; task returns error dict; document reaches PARTIAL_COMPLETE | 2.25 |

### 1.5 Embedding & Vector Storage

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Batched text embedding + Qdrant upserts | Large documents timeout on single Qdrant RPC | Ingest large doc; verify completion within timeout | 2.16, 2.19 |
| Dual vector store (Postgres + Qdrant cross-reference) | Qdrant failure causes complete data loss | Every Qdrant point has corresponding `text_chunks` row with `qdrant_point_id` | 1, 2.8 |
| CLIP image embedding (OpenCLIP ViT-B-32, 512-dim) | Cannot match image queries; visual content invisible to retrieval | Query with `query_image` + `strategy=hybrid`; receive image matches | 2 |
| Text preview hydration (chunk_text in Qdrant payload) | Search results have no text preview; N+1 DB fetches | Query text search; `content_text` populated without additional DB calls | 2.13 |
| BGE asymmetric query/passage prefixes | Query-document semantic matching degrades | Query "S-75 Dvina" matches document mentioning system in top-5 | 2.24 |

### 1.6 Translation & Language Handling

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Per-element language detection + LLM translation | Non-English text not translated; search fails on foreign text | Ingest Russian doc; `translated_text` populated; downstream uses English | 2.27 |
| Classification marking detection on original text | Translated text loses classification context; wrong marking | Ingest doc marked "SECRET" in Russian; classification correctly identified | 2.27 |

### 1.7 Document Analysis & Metadata

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| LLM-based metadata extraction (summary, date, classification, source) | Documents lack context for retrieval ranking | Ingest report; `/v1/documents/{id}/metadata` returns all fields | 1, 2.12 |
| Per-document summary used as image description context | Image descriptions lack document context; generic output | Ingest doc with images; descriptions reference document summary content | 2.9 |

---

## 2. GRAPH EXTRACTION & ONTOLOGY

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Docling-Graph service (chunked extraction with ontology validation) | No knowledge graph; purely keyword-based retrieval | Ingest doc with entities; `/v1/graph/query` finds extracted entities | 2.7 |
| Recursive chunk splitting on failure (2500 -> 1250 -> 625 chars) | Single failing chunk kills entire graph extraction | Induce LLM error on one chunk; partial graph still indexed | 2.19 |
| Truncated JSON repair (json-repair library) | LLM token limits cause truncated JSON; chunk dropped | Simulate truncated extraction JSON; partial entities recovered | 2.17 |
| Post-extraction validation (_validate_entity_types, _validate_properties) | Invalid types/properties pollute graph | Extraction result validated before persistence; invalid properties dropped | 2.24 |
| Entity alias resolution (exact -> alias -> fuzzy match -> new) | "S-75" and "SA-2 Dvina" are separate entities; expansion incomplete | Ingest 2 docs with alternate names; query returns unified entity | 2.9 |
| Batch Neo4j writes via UNWIND | Per-node writes cause 100s of round-trips; ingest hangs | Large doc with 1000+ entities ingested in <30s | 2.16 |
| Relationship upsert matches by name only (not entity type) | Entity-type mismatches silently drop relationships | Ingest doc; verify SPECIFIED_BY edges exist between systems and specifications | 2.31 |
| SPECIFIED_BY prompt instructions in relationship extraction | Specification entities orphaned from parent systems | Ingest doc with specs; Neo4j has SPECIFIED_BY edges from system → spec | 2.31 |
| Idempotent Neo4j writes (MERGE) | Re-ingest creates duplicate entities | Reingest same doc; entity count unchanged | 2.8 |
| Classification preserved on conflict | Reingest overwrites human-curated classification | Set classification to SECRET, reingest; verify still SECRET | 2.23 |

---

## 3. RETRIEVAL & SEARCH

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| BGE text search (strategy=basic) | Cannot retrieve by semantic content | Query with `strategy=basic`; receive ranked text chunks | 1 |
| CLIP image search (strategy=hybrid, modality_filter=image) | Cannot search images by text | Query image concept; receive image results | 2 |
| Hybrid search (text + image merge, dedupe, rescore) | Cannot leverage both embeddings together | Query `strategy=hybrid`; receive both text and image results | 2 |
| Weighted fusion scoring (0.65 semantic + 0.20 structure + 0.15 ontology) | Naive averaging ignores structural context | Verify formula applied in results scoring | 2.8, 2.24 |
| Document-structure expansion (chunk_links traversal) | Related chunks not retrieved together | Query matches chunk; neighbor chunks in same section also retrieved | 2.8 |
| Ontology expansion (Neo4j traversal up to 2 hops) | Query "S-75" misses docs about "SA-2 Guideline" | Query "S-75"; receive results mentioning "SA-2" via relationships | 2 |
| Cross-encoder reranker (bge-reranker-v2-m3) | Top results less relevant without reranking | Enable reranker; top results re-ordered by cross-encoder | 2.24 |
| Content-level deduplication (oversample 8x, filter) | Duplicate text appears multiple times | Top-k results have no duplicate content | 2.20 |
| Min cosine similarity threshold (default 0.25) | Irrelevant noise in results | All returned results score >= threshold | 1, 2.24 |
| Military ID bonus (0.03 for AN/, NSN, MIL-STD matches) | Exact military system mentions not prioritized | Query with military ID; receives score bonus | 2.20 |
| GraphRAG local search (entity + community reports) | Cannot drill into specific entities | `strategy=graphrag_local`; receive entity + community context | 2.9 |
| GraphRAG global search (cross-community summaries) | Cannot generate high-level summaries | `strategy=graphrag_global`; receive multi-community synthesis | 2.9 |
| GraphRAG drift search (community-informed expansion) | Cannot detect novel concepts | `strategy=graphrag_drift`; receive in-depth analysis | 2.9 |
| GraphRAG basic search (vector over text units) | No simple fallback when community detection fails | `strategy=graphrag_basic`; receive concise answer | 2.9 |
| GraphRAG precondition checks (409 if indexing not complete) | Silent empty results confuse user | Query before indexing returns 409; after indexing returns results | 2.13 |
| Async GraphRAG queries (submit/poll/fetch via Celery) | Long queries timeout in browser | Submit query; poll status; fetch results asynchronously | 2.29 |

---

## 4. DOCLING DOCUMENT VIEWER

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| DoclingViewer with `<docling-img>` web component | Documents unreadable; raw JSON only | Open PDF in viewer; see rendered pages with bounding boxes | 2.11 |
| Docling-raw endpoint enriched with description annotations | Images in viewer have no hover context | `/v1/documents/{id}/docling-raw` returns pictures with `annotations: [{kind: "description", ...}]` | 2.30 |
| Standalone image 404 guard in docling-raw | Standalone images render empty Docling page instead of fallback | Standalone image docling-raw returns 404; viewer shows image + description panel | 2.30 |
| Translation toggle with language banner | Non-English documents unreadable in viewer | Non-English doc has "Translate" toggle; switches between original and translated | 2.27 |
| Per-element translations overlay | No inline translation context | Hover over non-English text; see English translation | 2.27 |
| Image proxy (GET /v1/images/{chunk_id}) | Image results broken due to Docker-internal URLs | Image results display correctly; no CORS errors | 2.11 |

---

## 5. GRAPHRAG SERVICE

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Scheduled indexing (Celery Beat, configurable interval) | Community reports stale | Indexing runs on configured interval | 2.9 |
| Manual index trigger (POST /v1/graphrag/index with confirm=true) | Cannot force re-indexing | Trigger index; full rebuild starts | 2.15 |
| Manual update trigger (POST /v1/graphrag/update) | Cannot incrementally update | Trigger update; only new docs processed | 2.28 |
| Update with no new docs halts immediately | Wastes LLM calls re-processing unchanged data | Update with no new files; halts at `load_update_documents` in <5s | 2.28 |
| Redis lock prevents overlapping runs | Multiple indexing runs corrupt reports | Trigger during active run; skips with "locked" | 2.15 |
| Incremental indexing (delta-only on subsequent runs) | Each run reprocesses entire graph | Second run only processes new entities | 2.28 |
| Backup/restore on update failure | Failed update leaves corrupted state | Induce failure; previous state restored from backup | 2.28 |
| Stale Redis lock cleanup on worker startup | Worker crash leaves lock held indefinitely | Kill worker mid-index; restart; lock auto-cleaned | 2.28 |

---

## 6. TRUSTED DATA & GOVERNANCE

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Trusted data proposal workflow (PROPOSED -> APPROVED -> INDEXED) | Untrusted data indexed without oversight | Submit via `/v1/trusted-data/ingest`; status is PROPOSED until curator approves | 2.6 |
| Trusted data search (separate Qdrant collection) | Cannot distinguish human-reviewed from extracted knowledge | `/v1/trusted-data/query` returns only approved data | 2.6 |
| Feedback endpoint (auto-creates patch) | No mechanism to report extraction errors | `/v1/feedback` with correction creates patch | 2 |
| Dual-approval for graph mutations | Single curator can corrupt graph | Graph patch with one approval rejected | Planned (3) |

---

## 7. INFRASTRUCTURE & RESILIENCE

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Configurable retries per pipeline stage | Hard-coded retries impossible to tune | Set `PREPARE_MAX_RETRIES=3`; doc fails after 3 attempts | 2.18 |
| Configurable task time limits (soft + hard) | Fixed limits fail for varying doc sizes | Set `PREPARE_SOFT_TIME_LIMIT=7200`; large PDFs complete | 2.18 |
| Docling 503 retry loop (no budget consumed) | Busy errors consume retry budget | 503 triggers in-task retry with backoff; doesn't decrement max retries | 2.14 |
| Redis semaphore for Docling concurrency | GPU OOM or CPU saturation from unbounded conversions | `DOCLING_CONCURRENCY=1`; uploads queue instead of timeout | 2.11 |
| Docling lock held through retry (no gap) | Another task steals lock during retry window | Semaphore held during `self.retry()` | 2.23 |
| Docling health probe (advisory, non-blocking) | Health check failure blocks conversion | Docling unresponsive; probe logs warning but conversion proceeds | 2.11 |
| Atomic PipelineRun check-and-set (FOR UPDATE) | Concurrent ingest creates multiple runs | Submit same doc twice simultaneously; only one run created | 2.23 |
| Document-scoped singleflight Redis lock | prepare_document executes twice concurrently | Lock acquired at start, released at end; no concurrent prepares | 2.23 |
| Celery visibility timeout (10800s) | Long tasks redelivered prematurely; duplicate execution | 3-hour task completes without redelivery | 2.23 |
| Worker split profile (--profile split) | Single worker pool bottleneck | `docker compose --profile split up`; separate containers for ingest/embed/graph | 2.16 |
| Worker topology isolation (manage.sh) | Single and split workers both running; conflicts | manage.sh auto-stops opposite profile | 2.23 |

---

## 8. WEB UI

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| 6-mode query selector (basic, hybrid, 4x graphrag) | User stuck with default strategy | Query page has mode selector; each runs correct query | 2.11 |
| Modality sub-filter (text, image, all) for hybrid | Cannot isolate result types | Filter visible when hybrid selected; affects results | 2.11 |
| Live status polling with terminal state detection | UI polls forever; wastes resources | Polling stops at COMPLETE/FAILED/PARTIAL_COMPLETE/ERROR | 2.11 |
| Terminal status badges (green/red/amber) | Cannot distinguish success from failure | Color-coded badges for all terminal states | 2.11 |
| Directory Monitor (register/remove watch dirs) | Cannot set up auto-ingest | Directory Monitor page lists active dirs; add/remove works | 2.5 |
| Graph Explorer (entity/relationship search + creation) | Cannot explore or curate knowledge graph | Search and creation forms with full ontology | 2.5, 2.11 |
| Graph Explorer subgraph view (neighborhood visualization) | Clicking graph circle shows only 1 node for orphan entities | Search entity; click graph circle; see multi-node subgraph (direct edges or CO_OCCURS_WITH) | 2.31 |
| Trusted Data panel (submit/approve/reject/search) | Cannot interact with trusted data layer | Submission form, approval queue, search interface | 2.6 |

---

## 9. HEALTH & CONFIGURATION

| Feature | What breaks without it | Verify | Phase |
|---|---|---|---|
| Liveness probe (GET /v1/health) | Crashed API not detected | Returns `{"status": "ok"}` | 1 |
| Readiness probe with dependency checks | API starts before dependencies; early requests fail | `/v1/health/ready` returns `"degraded"` when deps down, `"ready"` when up | 1 |

---

## KNOWN FRAGILE FEATURES (Historically Broken)

These features have broken before and should be tested carefully after any change:

1. **Image dedup** (2.30) — Distinct images on same page collapsed. Test with multiple images, empty captions, same page.
2. **Image description tooltips** (2.25, 2.30) — Annotations must inject into docling-raw JSON. Hover over image in viewer.
3. **Standalone image synthesis** (2.30) — Must not trigger for non-image files. Test with .txt, .pdf, .docx.
4. **Docling timeouts** (2.14, 2.21) — Large PDFs timeout. Test with >100 page PDFs.
5. **Docling 503 handling** (2.14) — Must not consume retry budget. Test with mock 503s.
6. **Chord resilience** (2.22) — Single task failure must not kill pipeline. Verify error dicts returned.
7. **Concurrent dispatch** (2.23) — Multiple uploads of same doc must not create duplicate runs.
8. **GraphRAG incremental update** (2.28, 2.29) — NaN bugs in update process. Verify reindex completes.
9. **Stale run cleanup** (2.18, 2.23) — Crashed workers must reset PROCESSING docs. Test worker crash.
10. **Image description text search** (2.26) — Descriptions must appear in text search results.
11. **Chunk_links traversal** (2.8, 2.26) — Structure expansion must include all link types.
12. **Translation tooltips** (2.27) — Must not break image description annotations.
13. **Ontology subgraph orphans** (2.31) — Specification entities must connect to parent systems via SPECIFIED_BY. Test with "missile" search; click graph circle; verify multi-node subgraph.

---

## TESTING PROTOCOL

### After Every Code Change
1. Run full unit test suite: `python -m pytest tests/unit/ --tb=short`
2. Review this checklist for sections relevant to changed files
3. Add new features/fixes to this checklist before committing

### Quick Smoke Test (5 min)
1. Upload PDF document -> verify COMPLETE status
2. Query text search -> verify results with preview
3. Check `/v1/health/ready` -> all systems ready

### Integration Test (30 min)
1. Upload multi-page PDF with images and tables
2. Verify text search, image search, hybrid search
3. Verify GraphRAG local/global search (after indexing)
4. Verify DoclingViewer image hover tooltips
5. Verify standalone image upload + viewer

### Regression Test (60 min)
Run against all 12 Known Fragile Features listed above.

---

## CRITICAL CONFIGURATION PARAMETERS

| Config | Default | What breaks if wrong |
|---|---|---|
| `CHUNK_MAX_TOKENS` | 512 | Embedding quality degrades |
| `EMBED_TEXT_BATCH_SIZE` | 128 | Large docs timeout |
| `QDRANT_TIMEOUT_SECONDS` | 60 | Large upserts fail |
| `DOCLING_CONCURRENCY` | 1 | GPU OOM from parallel conversions |
| `DOCLING_TIMEOUT_SECONDS` | 3600 | Large PDFs timeout |
| `CELERY_VISIBILITY_TIMEOUT` | 10800 | Long tasks redelivered |
| `RETRIEVAL_SEMANTIC_WEIGHT` | 0.65 | Fusion scoring breaks |
| `RERANKER_ENABLED` | true | Ranking quality drops |
| `GRAPHRAG_INDEXING_ENABLED` | true | GraphRAG queries return 409 |
| `GRAPH_NODE_MIN_CONFIDENCE` | 0.60 | Low-confidence entities pollute graph |
