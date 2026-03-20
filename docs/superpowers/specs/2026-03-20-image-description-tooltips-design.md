# Image Description Tooltips — Design Spec

**Date:** 2026-03-20
**Status:** Approved

## Overview

Show LLM-generated image descriptions when users interact with images in the DoclingViewer. Two mechanisms depending on document type:

1. **Embedded images** (PDF, DOCX, PPTX, etc.) — hover tooltips via the existing Docling web component
2. **Standalone image documents** (.png, .jpeg, .tiff, etc.) — persistent description panel in DoclingViewer

## Design

### 1. Backend: Annotation Kind Fix

**File:** `app/services/document_analysis.py`

Add `"kind": "description"` to the annotation dict written by `describe_pictures()`:

```python
pic["annotations"].append({
    "kind": "description",
    "text": desc,
    "source": "llm",
    "model": model,
})
```

The Docling web component (`docling-components.js`) already includes a `docling-picture-description` annotation renderer that:
- Checks `annotation.kind === "description"` via `canDrawAnnotation()`
- Checks item label is `"chart"` or `"picture"` via `canDrawItem()`
- Renders tooltip with "AI Image Analysis:" label and auto-scrolling text

The `<docling-tooltip>` element is already present in the iframe HTML template (`buildDoclingHtml` in DoclingViewer.tsx). No frontend changes needed for this part.

**Scope:** All Docling-processed documents with embedded images (PDF, DOCX, PPTX, XLSX, HTML, MD).

### 2. Backend: Image Descriptions Endpoint

**New endpoint:** `GET /v1/documents/{documentId}/image-descriptions`

**File:** `app/api/v1/sources.py`

Query `DocumentElement` rows where `element_type = 'image'` and `content_text IS NOT NULL` for the given document. Return:

```json
[
  {
    "element_uid": "...",
    "content_text": "LLM-generated description...",
    "page_number": 1
  }
]
```

Returns empty array when no image descriptions exist.

### 3. Frontend: API Client

**File:** `frontend/src/api/client.ts`

Add `getDocumentImageDescriptions(documentId: string)` function calling the new endpoint.

### 4. Frontend: Standalone Image Description Panel

**File:** `frontend/src/components/DoclingViewer.tsx`

- Fetch image descriptions via `getDocumentImageDescriptions()` alongside existing metadata fetch
- When `docJson` is null (standalone/legacy-extracted image files), render a description panel if descriptions exist
- Panel styled like the existing "AI-Extracted Document Metadata" box: bordered card, `var(--color-surface-2)` background, "AI Image Analysis" label, description text
- `max-height` with overflow scroll for long descriptions
- Panel does NOT render for documents with Docling JSON (those use hover tooltips)

### 5. Error Handling & Edge Cases

- **No description available:** Panel/tooltip simply don't appear. No error state.
- **Previously ingested documents:** Won't have `kind` field in annotations. Require re-ingestion for hover tooltips. No migration needed — re-ingest is one-click in the UI.
- **Large descriptions:** Docling web component handles via auto-scroll (250px max viewport). Standalone panel uses max-height with overflow.

### 6. Testing

- Unit test: `describe_pictures()` annotations include `kind: "description"`
- Unit test: image-descriptions endpoint returns correct data
- Unit test: image-descriptions endpoint returns empty array for no-image documents
- Frontend: standalone image panel renders when descriptions exist
- Frontend: panel hidden when docJson present (PDF etc.)
- Regression: existing test suite passes
