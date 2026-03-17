# DoclingDocument Viewer with Page Images + Scrolling Text

**Date**: 2026-03-17
**Status**: Approved

## Problem

The current DoclingViewer renders markdown via `react-markdown`, losing the visual layout of the original document. The Streamlit reference app (`docling_streamlit_app`) uses a `<docling-img>` Lit web component that renders actual PDF page images with bounding box overlays, tooltips, and auto-scrolling AI picture descriptions.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Display mode | Replace markdown view with `<docling-img>` page images | User requested full replacement |
| Picture descriptions | granite-vision-3.3-2b via Docling's built-in preset | Air-gapped, same GPU as docling service |
| JS component source | Copy from Streamlit repo as-is | Not published on npm, custom build with scrolling |
| JS hosting | Static file via FastAPI + Vite public dir | No new service, works in dev and prod |
| API for JSON | New streaming endpoint `/docling-raw` | Avoids parsing overhead for 1MB+ JSON with base64 images |
| Existing /docling endpoint | Keep as-is | Used by agent endpoint and external consumers |

## Components

### 1. Docling Pipeline — Picture Descriptions

**Files:** `docker/docling/Dockerfile`, `docker/docling/app/converter.py`

- Pre-download `ibm-granite/granite-vision-3.3-2b` in Dockerfile (alongside granite-docling-258M)
- Set `do_picture_description=True` and `picture_description_options=granite_picture_description` in `VlmPipelineOptions`
- Descriptions auto-populate `pictures[].annotations[{kind, text, provenance}]` in `export_to_dict()` output
- Existing documents get descriptions on reingest

### 2. Static JS Serving

**Files:** `frontend/public/static/docling-components.js`

- Copy `docling_streamlit_app/static/docling-components.js` (~1984 lines, minified Lit build with scrolling)
- Vite copies `public/` to `dist/` at build — served at `/static/docling-components.js`
- In dev, Vite serves `public/` directly from dev server
- FastAPI production mount already serves `frontend/dist/` with SPA fallback

### 3. New API Endpoint

**Files:** `app/api/v1/sources.py`

`GET /v1/documents/{document_id}/docling-raw`

- Streams `docling_document.json` bytes directly from MinIO (bucket: `minio_bucket_derived`, key: `artifacts/{document_id}/docling_document.json`)
- Returns `Content-Type: application/json` with raw bytes — no Python deserialization
- Returns 404 if JSON not found
- No auth required (same as existing endpoints)

### 4. Frontend DoclingViewer Rewrite

**Files:** `frontend/src/components/DoclingViewer.tsx`, `frontend/src/api/client.ts`

**API client:**
- Add `getDoclingRawJson(documentId): Promise<object>` — fetches from `/v1/documents/{id}/docling-raw`

**DoclingViewer component:**
- Two view modes: "Document" (default) | "JSON"
- Document mode:
  - Fetches DoclingDocument JSON via `getDoclingRawJson()`
  - Builds HTML string with `<docling-img>` web component (same pattern as Streamlit's `to_json_rendered_content`)
  - Renders via iframe `srcdoc` attribute
  - `<docling-img pagenumbers>` with `<docling-tooltip>` child
  - CSS: page shadow, responsive sizing within modal
- JSON mode:
  - Shows raw JSON with 2-space indentation (existing behavior)
- Modal: click-outside-to-close, Escape key, responsive sizing

**HTML template (embedded in iframe srcdoc):**
```html
<!doctype html>
<html>
  <head>
    <script src="/static/docling-components.js" type="module"></script>
    <style>
      docling-img { gap: 1rem; }
      docling-img::part(page) { box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.2); }
    </style>
  </head>
  <body>
    <docling-img id="dclimg" pagenumbers>
      <docling-tooltip></docling-tooltip>
    </docling-img>
    <script id="dcljson" type="application/json">{JSON}</script>
    <script>
      function applySrc() {
        document.getElementById('dclimg').src =
          JSON.parse(document.getElementById('dcljson').textContent);
      }
      if (!customElements.get('docling-img')) {
        customElements.whenDefined('docling-img').then(applySrc);
      } else { applySrc(); }
    </script>
  </body>
</html>
```

## Data Flow

```
[Upload PDF] → [Docling service: convert + picture descriptions via granite-vision-3.3-2b]
                    ↓
            export_to_dict() → JSON with pages[].image.uri (base64) + pictures[].annotations[]
                    ↓
            Stored in MinIO: artifacts/{doc_id}/docling_document.json
                    ↓
[Frontend: View button] → GET /v1/documents/{id}/docling-raw → raw JSON stream
                    ↓
            Build HTML with <docling-img> + embedded JSON → iframe srcdoc
                    ↓
            Web component renders page images, bounding boxes, tooltips, scrolling text
```

## Files Changed

| File | Action |
|------|--------|
| `docker/docling/Dockerfile` | Modify — add granite-vision-3.3-2b download |
| `docker/docling/app/converter.py` | Modify — enable picture descriptions |
| `frontend/public/static/docling-components.js` | Create — copy from Streamlit repo |
| `app/api/v1/sources.py` | Modify — add `/docling-raw` endpoint |
| `frontend/src/api/client.ts` | Modify — add `getDoclingRawJson()` |
| `frontend/src/components/DoclingViewer.tsx` | Rewrite — iframe + web component |
