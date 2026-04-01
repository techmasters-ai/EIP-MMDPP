# Image Display & Tooltip Fixes

**Date:** 2026-04-01
**Status:** Approved

## Problem

Two bugs prevent images from being fully usable in the platform:

1. **Standalone image files (JPEG, PNG, TIFF, BMP, GIF, WEBP) show blank in DoclingViewer.** Docling processes these but returns 0 elements. No `DocumentElement` or `Artifact` is created, so the entire downstream pipeline (CLIP embedding, LLM description, text embedding) is a no-op. The document shows COMPLETE with 0 elements.

2. **PDF image description tooltips don't work.** LLM-generated descriptions exist in the DB for all image elements, but the Docling `<docling-tooltip>` web component requires `annotations: [{kind: "description", text: "..."}]` on each picture item in the JSON. All pictures have `annotations: []`. The annotation injection was never implemented despite commit `0d68b2c` claiming it was (the commit only added the API endpoint and standalone image fallback panel, not the actual injection).

### History (Bug 2)

- `7712504` (Mar 17): Rewrote DoclingViewer with `<docling-img>` web component. Included `<docling-tooltip>` but no annotations populated.
- `0d68b2c` (Mar 20): Commit message claimed "Add kind:description to picture annotations" but the diff only added the image-descriptions API fetch and standalone panel. Annotation injection was never implemented.
- `f2cc23c` (Mar 24): Added translation tooltips using `{kind: "description"}` annotations on text/table items. This was the only working use of annotations.
- `83548c9` (Mar 24): Replaced annotation-based translations with direct `item.text` overwrite, removing the only working annotation injection code.

## Design

### Fix 1: Standalone Image Element Synthesis

**File:** `app/workers/pipeline.py` — `prepare_document()`

After Docling returns 0 elements and deduplication runs, detect the case where:
- `len(result.elements) == 0`
- `mime_type.startswith("image/")`

Synthesize a single `ExtractedChunk`:
- `modality = "image"`
- `chunk_text = None`
- `raw_image_bytes = file_bytes` (the original uploaded file)
- `page_number = 1`
- `metadata = {"label": "picture", "ext": <extension from mime>}`
- `bounding_box = None`

Set `result.num_pages = 1`.

The existing pipeline then:
1. Generates `element_uid` (deterministic from page + order + modality + content hash)
2. Creates `Artifact` row and uploads image to MinIO
3. Creates `DocumentElement` with `element_type="image"` and `storage_key`
4. `derive_picture_descriptions` finds the image element, generates LLM description
5. `derive_image_embeddings` creates CLIP vector
6. `derive_text_chunks_and_embeddings` Pass 2 embeds the description sections as BGE text vectors

**Scope:** ~12 lines, single `if` block.

### Fix 2: Server-Side Annotation Injection

**File:** `app/api/v1/sources.py` — `get_docling_raw_json()`

Instead of returning the raw Docling JSON directly, enrich it with image description annotations before returning.

**Steps:**
1. Parse the JSON bytes from MinIO into a dict
2. Query DB for image descriptions: all `document_elements` where `document_id` matches, `element_type = 'image'`, and `content_text IS NOT NULL`, ordered by `element_order`
3. Group descriptions by `page_number`
4. Group `docJson["pictures"]` by `prov[0].page_no`
5. For each page, zip pictures and descriptions in positional order
6. For each matched pair, append `{"kind": "description", "text": content_text}` to `picture["annotations"]`
7. Re-serialize to JSON and return

**Matching rationale:** Both the pictures array and the document_elements table are produced from the same Docling conversion in document reading order. Grouping by page and matching positionally within each page is authoritative because both data sources originate from the same `prepare_document` run.

**Safety:** If picture count != description count on a page, match as many as possible (zip behavior) and skip extras. This handles edge cases like failed description generation for some images.

**Scope:** ~30 lines added to the endpoint.

### No Frontend Changes Required

The `<docling-tooltip>` component with `AnnotationPictureDescription` already:
- Checks `canDrawAnnotation(i)` via `Ot.PictureDescription(i)` which tests `i.kind === "description"`
- Renders `<p class="label"><span>AI Image Analysis:</span></p>` + scrolling `${i.text}` content

The existing `imageDescriptions` state and fetch in `DoclingViewer.tsx` remain for the standalone image fallback panel (`!docJson && imageDescriptions.length > 0`).

## Files Changed

| File | Change |
|---|---|
| `app/workers/pipeline.py` | Add ~12 lines: synthesize image element when Docling returns 0 for image MIME |
| `app/api/v1/sources.py` | Add ~30 lines: inject description annotations into docling-raw JSON response |

## Testing

- **Standalone image:** Upload a JPEG/PNG. Verify pipeline creates 1 element, generates description, CLIP embeds. Verify DoclingViewer shows the image and description panel.
- **PDF tooltips:** Open a PDF with images in DoclingViewer. Hover over an image. Verify the "AI Image Analysis" tooltip appears with the LLM description scrolling.
- **Edge cases:** Document with no images (no change). Document with images but no descriptions yet (empty annotations, no tooltip). Standalone image with Docling returning >0 elements (shouldn't trigger synthesis).
