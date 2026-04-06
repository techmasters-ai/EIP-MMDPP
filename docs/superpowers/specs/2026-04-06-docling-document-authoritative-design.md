# Design: DoclingDocument as Authoritative Mutable Artifact

**Date:** 2026-04-06
**Status:** Approved
**TODO items:** #60, #61, #62

---

## Problem

The DoclingDocument JSON is persisted once in `prepare_document` and never updated. Translation (Stage 2) and picture descriptions (Stage 4) mutate DocumentElement rows but not the DoclingDocument. By the time `derive_ontology_graph` (Stage 5c) and `derive_text_chunks_and_embeddings` (Stage 5a) run, the JSON in MinIO is stale — it has no translations and no picture descriptions.

The existing `/v1/documents/{id}/docling-raw` endpoint returns this stale JSON.

## Solution

Make the DoclingDocument JSON in MinIO the **single source of truth** that accumulates enrichments as each pipeline stage completes. Each enrichment stage reloads the JSON from MinIO, applies its mutations, and re-persists the updated version. After the pipeline completes, the DoclingDocument contains the full enriched document.

## Design

### Data flow

```
prepare_document:
  Docling service → DoclingDocument dict
  → Persist to MinIO as docling_document.json (v1: original)
  → Flatten to DocumentElement rows (relational index)

detect_and_translate:
  → Load docling_document.json from MinIO
  → For each translated element: find matching item by element_uid,
    set a "translated_text" field on it
  → Re-persist to MinIO as docling_document.json (v2: + translations)
  → Still update DocumentElement.translated_text (backward compat)

derive_picture_descriptions:
  → Load docling_document.json from MinIO
  → For each described image: find matching item by element_uid,
    set/update the description text on it
  → Re-persist to MinIO as docling_document.json (v3: + translations + descriptions)
  → Still update DocumentElement.content_text (backward compat)

derive_text_chunks_and_embeddings:
  → Load docling_document.json from MinIO
  → Reconstruct DoclingDocument object via DoclingDocument.load_from_json()
    or model_validate()
  → Use native HybridChunker(tokenizer, max_tokens) to chunk
  → Each chunk carries page_number, section context, element provenance
  → Embed chunks via BGE, create TextChunk rows + ArcadeDB vertices
  → Fall back to current structure_aware_chunk() if DoclingDocument
    reconstruction fails (backward compat for pre-Docling documents)

derive_ontology_graph:
  → Load docling_document.json from MinIO (now enriched)
  → Pass directly to Docling-Graph service (has translations + descriptions)
  → No more _enriched_text reconstruction needed

finalize_document:
  → docling_document.json in MinIO is the final enriched version
  → /v1/documents/{id}/docling-raw returns this enriched document
```

### API endpoint

The existing `GET /v1/documents/{id}/docling-raw` endpoint already returns the DoclingDocument JSON from MinIO. After this refactor, that JSON will be the fully enriched version (with translations and picture descriptions). No new endpoint needed — the existing one returns the right data once the pipeline keeps it updated.

### DoclingDocument mutation pattern

Each enrichment stage follows this pattern:

```python
# Load current DoclingDocument from MinIO
raw = download_bytes_sync(bucket, f"artifacts/{document_id}/docling_document.json")
doc_dict = json.loads(raw)

# Apply enrichments to the dict
# (mutate items in doc_dict matching by element_uid)

# Re-persist
upload_bytes_sync(
    json.dumps(doc_dict, ensure_ascii=False, default=str).encode(),
    bucket,
    f"artifacts/{document_id}/docling_document.json",
    content_type="application/json; charset=utf-8",
)
```

Mutations operate on the dict representation, not a reconstructed DoclingDocument object, because:
- The dict is what we persist and what the API returns
- Reconstruction to a full DoclingDocument object can fail on older documents
- Dict mutation is simpler and more resilient

### Native chunking (#61)

Replace `structure_aware_chunk()` with Docling's `HybridChunker`:

```python
from docling.chunking import HybridChunker
from docling.datamodel.document import DoclingDocument

doc = DoclingDocument.model_validate(doc_dict)
chunker = HybridChunker(
    tokenizer=settings.text_embedding_model,
    max_tokens=settings.chunk_max_tokens,
)
for chunk in chunker.chunk(doc):
    # chunk.text, chunk.meta.doc_items, chunk.meta.headings, etc.
```

Falls back to `structure_aware_chunk()` on reconstruction failure.

### Backward compatibility

- DocumentElement rows continue to be created and updated (relational queries depend on them)
- DocumentElement.translated_text and content_text still updated (backfill functions use them)
- Markdown files still generated from the DoclingDocument (viewer uses them)
- Pre-refactor documents without a valid DoclingDocument JSON use the existing DocumentElement-based code path

### Files changed

| File | Change |
|------|--------|
| `app/workers/pipeline.py` | detect_and_translate: reload+mutate+re-persist DoclingDocument JSON |
| `app/workers/pipeline.py` | derive_picture_descriptions: reload+mutate+re-persist DoclingDocument JSON |
| `app/workers/pipeline.py` | derive_text_chunks_and_embeddings: load DoclingDocument, use HybridChunker |
| `app/workers/pipeline.py` | derive_ontology_graph: load enriched DoclingDocument directly (remove _enriched_text hack) |
| `app/services/chunking.py` | Keep as fallback; add wrapper for HybridChunker |

### Not changed

- `prepare_document` — already persists DoclingDocument JSON correctly
- `app/services/docling_client.py` — no change needed
- `docker/docling/app/converter.py` — no change needed
- `/v1/documents/{id}/docling-raw` endpoint — already returns MinIO JSON
- DocumentElement table — still populated, still used for relational queries
