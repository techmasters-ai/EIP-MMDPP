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
| `document_id` | Inherited from the source image's DocumentElement |
| `page_number` | Inherited from the source image's DocumentElement |
| `section_path` | `"image_description/{image_element_order}/section_{n}"` |
| `artifact_id` | The artifact_id of the parent image (ties sections together and enables SAME_ARTIFACT expansion) |

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

1. **SAME_ARTIFACT** — bidirectional links between all sections of the same image description (weight from existing `retrieval_weight_same_artifact` config, default 0.82). Enables expansion: if section 3 matches, sections 1, 2, 4, ... surface through document-structure expansion.

2. **SAME_PAGE** — bidirectional links between image description chunks and other chunks on the same page (including the original CLIP image chunk). Weight from existing `retrieval_weight_same_page` config (default 0.78).

## Section Splitting

A generic splitter that works regardless of prompt structure:

1. **Primary split**: Markdown-style headers (`# ...`, `## ...`), numbered headers (`1)`, `1.`, `**Section:**`)
2. **Fallback**: If no headers detected, split on double-newlines (paragraph breaks)
3. **Skip** empty sections after splitting
4. **Prepend** the header text to each chunk body so the chunk is self-contained

This approach is prompt-agnostic — if the user changes the picture description prompt to produce different sections, the splitter adapts automatically.

## Pipeline Changes

### Task: `derive_text_chunks_and_embeddings`

**File:** `app/workers/pipeline.py`

After the existing pass that processes text/table/heading/equation/schematic elements, add a second pass for image descriptions:

1. Query `DocumentElement` rows where `element_type = 'image'` AND `content_text IS NOT NULL` (non-empty) for the current document
2. For each image element with a description:
   a. Split `content_text` into sections using the generic splitter
   b. Create `TextChunk` rows with `modality = "image_description"`, referencing the image element's `artifact_id`, `page_number`, and `document_id`
   c. Embed each section with BGE (`embed_texts` with passage prefix)
   d. Upsert to Qdrant `eip_text_chunks`
3. Create **SAME_ARTIFACT** chunk_links between all sections of the same image description (bidirectional)
4. Create **SAME_PAGE** chunk_links between description chunks and other chunks sharing the same `page_number` and `document_id`

### Section Splitter

**New function in:** `app/services/chunking.py` (or inline in the pipeline task if a chunking module doesn't exist)

```python
def split_description_sections(description: str) -> list[str]:
    """Split an image description into sections by headers.

    Handles markdown headers (# / ## / ###), numbered headers (1) / 1.),
    and bold headers (**Title:**). Falls back to paragraph splitting.
    Returns list of section strings with headers prepended.
    """
```

## Query Path

**No changes needed to the search endpoint or fusion logic.** This works automatically because:

- Text queries already search `eip_text_chunks` with BGE embeddings
- Image description chunks appear as results with `modality = "image_description"`
- The existing document-structure expansion follows SAME_ARTIFACT and SAME_PAGE links from these seeds
- Cross-modal bridging (Neo4j fallback) and ontology traversal work on any seed chunk
- Reranking works via `chunk_text`

## Result Rendering

### API Response

When a result has `modality = "image_description"`, the response must include `image_url` pointing to the original image so the frontend can display it alongside the matching section text.

**Resolution path:** The TextChunk stores `artifact_id` → look up the corresponding `ImageChunk` by `artifact_id` and `document_id` → use its chunk_id to construct the image URL (`/v1/images/{image_chunk_id}`).

**Implementation:** In the existing `_populate_image_urls()` function in `retrieval.py`, add a branch for `modality == "image_description"`: query the `ImageChunk` table by `artifact_id` + `document_id` to find the image chunk, then set `image_url = f"/v1/images/{image_chunk.id}"`.

### Frontend

The existing QueryPage already displays `image_url` when present on a result. Image description results will automatically show the image alongside the matching section text. No frontend changes expected — the `modality` value `"image_description"` is new but the rendering logic already handles the `image_url` + `chunk_text` combination.

## What Doesn't Change

- Qdrant collection schema (reusing `eip_text_chunks`, same 1024-dim BGE vectors)
- Search endpoint API contract (`UnifiedQueryRequest` / `UnifiedQueryResponse` — new modality value but same shape)
- Fusion scoring, expansion, reranking logic
- GraphRAG query paths
- CLIP image embedding pipeline (`derive_image_embeddings` unchanged)
- `derive_picture_descriptions` task (unchanged — it still generates descriptions, this feature consumes them downstream)
- Trusted data pipeline

## Configuration

No new configuration parameters needed. Reuses existing:
- `retrieval_weight_same_artifact` (default 0.82)
- `retrieval_weight_same_page` (default 0.78)
- BGE embedding model and dimensions
- All existing fusion, expansion, and reranking settings
