# DoclingDocument Viewer — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Add a full-page modal viewer that displays the DoclingDocument for completed uploads, with a toggle between rendered markdown (with inline images) and raw JSON tree.

**Architecture:** Persist the DoclingDocument markdown and JSON to MinIO during pipeline processing. Serve via a new API endpoint that injects presigned image URLs. Frontend renders in a full-page modal with markdown/JSON toggle.

**Tech Stack:** react-markdown + remark-gfm (frontend rendering), MinIO (storage)

---

## 1. Pipeline & Storage Changes

**During `prepare_document`** (after Docling conversion succeeds, in `app/workers/pipeline.py`):

1. Call `doc.export_to_dict()` → serialize as JSON → upload to MinIO:
   `eip-derived/artifacts/{document_id}/docling_document.json`
2. Use the already-generated markdown (`doc.export_to_markdown()`) → upload to MinIO:
   `eip-derived/artifacts/{document_id}/docling_document.md`
3. Both uploads happen right after element extraction, before the function returns
4. If either upload fails, log a warning but do NOT fail the pipeline — the viewer is non-critical

**Docling converter change** (`docker/docling/app/converter.py`):
- Add `document_json` field to `ConvertResponse` schema
- Populate via `doc.export_to_dict()` alongside existing `elements` and `markdown` exports

**On reingest:** Existing `purge_document_derivations` already cleans `eip-derived/artifacts/{document_id}/`, so old files are removed and fresh ones written on re-processing.

**Legacy documents:** Documents processed before this change won't have the files. The API returns 404, and the frontend hides the "View" button.

## 2. Backend API Endpoint

**New endpoint:** `GET /v1/documents/{document_id}/docling`

**Route location:** `app/api/v1/sources.py`

**Response schema** (`DoclingDocumentResponse`):
```json
{
  "document_id": "uuid",
  "filename": "report.pdf",
  "markdown": "# Title\n\nParagraph text...\n\n![image](presigned_url)...",
  "document_json": { "...full DoclingDocument dict..." },
  "images": [
    { "element_uid": "1-3-image-abc123", "url": "https://minio.../presigned..." }
  ]
}
```

**Logic:**
1. Fetch `docling_document.md` and `docling_document.json` from MinIO
2. If either file doesn't exist → return 404 with message "DoclingDocument not available for this document"
3. Query the `artifacts` table for image artifacts belonging to this document → generate MinIO presigned URLs
4. In the markdown, replace image references with the presigned URLs so images render inline
5. Return markdown (with live image URLs), raw JSON, and image URL list

## 3. Frontend Components

### New Component: `DoclingViewer.tsx`

Full-page modal for viewing the DoclingDocument.

**Props:** `documentId`, `filename`, `onClose`

**Behavior:**
- On mount, calls `GET /v1/documents/{document_id}/docling`
- Shows a loading spinner while fetching
- Toggle bar at top: "Document" (markdown view) | "JSON" (raw tree)
- **Markdown view:** Rendered via `react-markdown` with `remark-gfm` plugin. Images render inline via presigned URLs already embedded in the markdown string.
- **JSON view:** Pretty-printed JSON in a scrollable `<pre>` block with monospace font
- Close button (X) in the top-right corner
- Modal fills most of the viewport (~90% width, ~90% height, centered) with internal scroll

### Modified: `FileUpload.tsx`

- Add eye icon / "View" button to each document row
- Only visible when `pipeline_status === "COMPLETE"`
- Clicking opens `DoclingViewer` modal with that document's ID and filename
- State tracks which document (if any) has the viewer open

### Modified: `api/client.ts`

- Add `getDoclingDocument(documentId: string): Promise<DoclingDocumentResponse>`
- Calls `GET /v1/documents/{document_id}/docling`

### Modified: `styles.css`

- Modal overlay (backdrop, centered panel, scrollable body)
- Toggle bar for markdown/JSON switch
- Markdown content styling (headings, tables, images, paragraphs within the viewer)
- Reuse existing design system variables (`--color-surface`, `--color-border`, etc.)

## 4. Dependencies & Files Summary

**New npm dependencies:**
- `react-markdown` — lightweight markdown renderer
- `remark-gfm` — GitHub-flavored markdown plugin (tables, strikethrough)

**Files to create:**
- `frontend/src/components/DoclingViewer.tsx`

**Files to modify:**
- `frontend/src/components/FileUpload.tsx` — add "View" button, modal state
- `frontend/src/api/client.ts` — add `getDoclingDocument()` function
- `frontend/src/styles.css` — modal, toggle, markdown content styles
- `app/workers/pipeline.py` — persist docling_document.json and docling_document.md to MinIO in `prepare_document`
- `app/api/v1/sources.py` — add `GET /v1/documents/{document_id}/docling` endpoint
- `app/schemas/` — add `DoclingDocumentResponse` schema
- `docker/docling/app/converter.py` — add `document_json` to `ConvertResponse`
- `docker/docling/app/schemas.py` — add `document_json` field to `ConvertResponse`

**No changes to:**
- Database schema / migrations
- Existing pipeline stages beyond `prepare_document`
- Graph Explorer or Query page
