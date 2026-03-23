# Image Description Text Search — Design Spec

**Date:** 2026-03-23
**Status:** Approved

## Overview

Embed LLM-generated image descriptions as BGE text vectors in the existing `eip_text_chunks` Qdrant collection, making them searchable via the standard text search path. Each description is split into sections (by markdown/numbered headers), and each section becomes its own chunk. Matching sections return with the original image. Sections of the same description are linked via SAME_ARTIFACT chunk_links for graph expansion.

This adds a new query path to multi-modal search: text queries can now match against image description content, surfacing relevant images based on semantic understanding of what's depicted — not just CLIP pixel similarity.

## Data Model

### Postgres: `retrieval.text_chunks`

Image description sections are stored as regular TextChunk rows with a distinguishing modality:

| Field | Value |
|---|---|
| `modality` | `"image_description"` |
| `chunk_text` | One section of the description (header included) |
| `chunk_index` | Offset at 100000 + image_element_order * 100 + section_index (avoids collision with text chunk indices) |
| `document_id` | Inherited from the source image's DocumentElement |
| `page_number` | Inherited from the source image's DocumentElement |
| `artifact_id` | The artifact_id of the parent image (ties sections together and enables SAME_ARTIFACT expansion) |

**Precondition:** Image elements with NULL `artifact_id` are skipped — `artifact_id` is NOT NULL on TextChunk so the FK constraint requires it.

**Deterministic chunk ID:** Generated from `uuid.UUID(hashlib.md5(f"{document_id}:{image_element_uid}:{section_index}:{model_version}".encode()).hexdigest())` for idempotent retries, following the same sha256+md5 approach used by existing text chunk ID generation.

### Qdrant: `eip_text_chunks`

Standard 1024-dim BGE embedding points with payload:

| Payload field | Value |
|---|---|
| `modality` | `"image_description"` |
| `chunk_text` | Section text |
| `document_id` | UUID |
| `page_number` | Integer |
| `artifact_id` | UUID of the parent image artifact |
| `chunk_id` | UUID of the TextChunk row |
| `classification` | Inherited from document metadata |

### Chunk Links: `retrieval.chunk_links`

Two link types created for image description chunks:

1. **SAME_ARTIFACT** — bidirectional neighbor-only (prev/next by section order) links between sections of the same image description. Uses existing `retrieval_weight_same_artifact` config (default 0.82). This follows the same neighbor-only pattern used by the existing `derive_structure_links` task for SAME_ARTIFACT to avoid O(n^2). With `retrieval_doc_max_hops=2`, all sibling sections remain reachable through multi-hop expansion.

2. **SAME_PAGE** — bidirectional links between image description chunks and other chunks on the same page (including the original CLIP image chunk). Weight from existing `retrieval_weight_same_page` config (default 0.78). Note: `derive_structure_links` already runs after the chord and creates SAME_PAGE links based on page_number, so these links are created naturally by the existing pipeline — no additional link creation needed in `derive_text_chunks_and_embeddings`.

## Section Splitting

A generic splitter that works regardless of prompt structure:

1. **Primary split**: Markdown-style headers (`# ...`, `## ...`), numbered headers (`1)`, `1.`, `**Section:**`)
2. **Fallback**: If no headers detected, split on double-newlines (paragraph breaks)
3. **Skip** empty or trivially short sections (< 20 characters) after splitting
4. **Prepend** the header text to each chunk body so the chunk is self-contained
5. **Single-section descriptions**: If only one section results, emit it as a single chunk
6. **Preamble text** before the first header is included as its own section

This approach is prompt-agnostic — if the user changes the picture description prompt to produce different sections, the splitter adapts automatically.

## Pipeline Changes

### Task: `derive_text_chunks_and_embeddings`

**File:** `app/workers/pipeline.py`

After the existing pass that processes text/table/heading/equation/schematic elements, add a second pass for image descriptions:

1. Query `DocumentElement` rows where `element_type = 'image'` AND `content_text IS NOT NULL` (non-empty) AND `artifact_id IS NOT NULL` for the current document
2. For each image element with a description:
   a. Split `content_text` into sections using the generic splitter
   b. Generate deterministic chunk IDs from `document_id + element_uid + section_index + model_version`
   c. Create `TextChunk` rows with `modality = "image_description"`, `chunk_index` offset at 100000+, referencing the image element's `artifact_id`, `page_number`, and `document_id`
   d. Embed each section with BGE (`embed_texts` with passage prefix)
   e. Upsert to Qdrant `eip_text_chunks`
3. Create **SAME_ARTIFACT** chunk_links between consecutive sections of the same image description (neighbor-only, bidirectional)

SAME_PAGE links are handled by the existing `derive_structure_links` task which runs after the chord.

### Section Splitter

**New function in:** `app/services/chunking.py` (or inline in the pipeline task if a chunking module doesn't exist)

```python
def split_description_sections(description: str) -> list[str]:
    """Split an image description into sections by headers.

    Handles markdown headers (# / ## / ###), numbered headers (1) / 1.),
    and bold headers (**Title:**). Falls back to paragraph splitting.
    Returns list of section strings with headers prepended.
    Skips sections shorter than 20 characters.
    """
```

## Query Path

### Search

Text queries already search `eip_text_chunks` with BGE embeddings, so image description chunks are searched automatically. The existing expansion pipeline follows SAME_ARTIFACT and SAME_PAGE links from these seeds. Reranking works via `chunk_text`.

### Modality Filter Fix

**File:** `app/api/v1/retrieval.py`

The `modality_filter` logic must be updated to include `"image_description"` in the text filter set:

```python
# Current:
if body.modality_filter == ModalityFilter.text:
    deduped = [r for r in deduped if r.modality in ("text", "table")]

# Updated:
if body.modality_filter == ModalityFilter.text:
    deduped = [r for r in deduped if r.modality in ("text", "table", "image_description")]
```

This ensures image description results appear when users filter to text-only results.

## Result Rendering

### API Response

When a result has `modality = "image_description"`, the response must include `image_url` pointing to the original image so the frontend can display it alongside the matching section text.

**Resolution path:** The TextChunk stores `artifact_id` → look up the corresponding `ImageChunk` by `artifact_id` and `document_id` → use its chunk_id to construct the image URL (`/v1/images/{image_chunk_id}`).

**Implementation:** In the existing `_populate_image_urls()` function in `retrieval.py`:

1. Collect all results with `modality == "image_description"` and extract their `(artifact_id, document_id)` pairs
2. Batch-query `retrieval.image_chunks` for matching rows (single query, not N+1)
3. Build a lookup map from `(artifact_id, document_id)` → `image_chunk.id`
4. Set `image_url = f"/v1/images/{image_chunk_id}"` on each matching result

This requires `artifact_id` to be available on the result object. It is already stored in the Qdrant payload and backfilled via `_backfill_content_text()`.

### Frontend

The existing QueryPage already displays `image_url` when present on a result. Image description results will automatically show the image alongside the matching section text. No frontend changes expected — the `modality` value `"image_description"` is new but the rendering logic already handles the `image_url` + `chunk_text` combination.

## What Doesn't Change

- Qdrant collection schema (reusing `eip_text_chunks`, same 1024-dim BGE vectors)
- Search endpoint API contract (`UnifiedQueryRequest` / `UnifiedQueryResponse` — new modality value but same shape)
- Fusion scoring, expansion, reranking logic
- GraphRAG query paths
- CLIP image embedding pipeline (`derive_image_embeddings` unchanged)
- `derive_picture_descriptions` task (unchanged — it still generates descriptions, this feature consumes them downstream)
- `derive_structure_links` task (already creates SAME_PAGE links by page_number — image description chunks benefit automatically)
- Trusted data pipeline

## Configuration

No new configuration parameters needed. Reuses existing:
- `retrieval_weight_same_artifact` (default 0.82)
- `retrieval_weight_same_page` (default 0.78)
- BGE embedding model and dimensions
- All existing fusion, expansion, and reranking settings
