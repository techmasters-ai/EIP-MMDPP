# DoclingDocument Viewer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Replace the markdown-based DoclingViewer with the `<docling-img>` web component that renders PDF page images with bounding box overlays, tooltips, and auto-scrolling AI picture descriptions powered by granite-vision-3.3-2b.

**Architecture:** The Docling Docker service gets granite-vision-3.3-2b for picture descriptions. A new streaming endpoint serves raw DoclingDocument JSON from MinIO. The React frontend embeds the `<docling-img>` Lit web component in an iframe via `srcdoc`, loading the JS from a static file served by Vite/FastAPI.

**Tech Stack:** Docling + granite-vision-3.3-2b, FastAPI streaming, React + iframe srcdoc, Lit web components

**Design doc:** `docs/plans/2026-03-17-docling-viewer-design.md`

---

### Task 0: Copy docling-components.js to frontend

**Files:**
- Create: `frontend/public/static/docling-components.js`

**Step 1: Copy the file**

```bash
mkdir -p frontend/public/static
cp ~/development/docling_streamlit_app/static/docling-components.js frontend/public/static/docling-components.js
```

**Step 2: Verify Vite serves it in dev**

```bash
cd frontend && npx vite --host 0.0.0.0 &
curl -s http://localhost:5173/static/docling-components.js | head -5
kill %1
```

Expected: First 5 lines of the JS file (the `@license` header).

**Step 3: Commit**

```bash
git add frontend/public/static/docling-components.js
git commit -m "feat: add docling-components.js web component for document viewer"
```

---

### Task 1: Add /docling-raw streaming endpoint

**Files:**
- Modify: `app/api/v1/sources.py`

**Step 1: Add the endpoint**

Add after the existing `/documents/{document_id}/docling` endpoint (around line 650):

```python
@router.get("/documents/{document_id}/docling-raw")
async def get_docling_raw_json(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    """Stream the raw DoclingDocument JSON from MinIO.

    Returns the full DoclingDocument including base64 page images,
    intended for the <docling-img> web component viewer.
    """
    from fastapi.responses import Response
    from app.services.storage import download_bytes_async

    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    base_key = f"artifacts/{str(document_id)}"
    bucket = settings.minio_bucket_derived

    try:
        json_bytes = await download_bytes_async(bucket, f"{base_key}/docling_document.json")
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="DoclingDocument JSON not available. Re-ingest to generate.",
        )

    return Response(content=json_bytes, media_type="application/json")
```

**Step 2: Verify the endpoint responds**

```bash
# Requires running API + MinIO + a processed document
curl -s http://localhost:8000/v1/documents/{some_doc_id}/docling-raw | python3 -c "import sys,json; d=json.load(sys.stdin); print('keys:', list(d.keys())); print('pages:', len(d.get('pages',{})))"
```

Expected: `keys: ['schema_name', 'version', 'name', ...]` and `pages: N`

**Step 3: Commit**

```bash
git add app/api/v1/sources.py
git commit -m "feat: add /docling-raw streaming endpoint for DoclingDocument JSON"
```

---

### Task 2: Add frontend API client function

**Files:**
- Modify: `frontend/src/api/client.ts`

**Step 1: Add the function**

Add after the existing `getDoclingDocument` function:

```typescript
export async function getDoclingRawJson(documentId: string): Promise<Record<string, unknown>> {
  const res = await fetch(`/v1/documents/${documentId}/docling-raw`);
  return handleResponse<Record<string, unknown>>(res);
}
```

**Step 2: Commit**

```bash
git add frontend/src/api/client.ts
git commit -m "feat: add getDoclingRawJson API client function"
```

---

### Task 3: Rewrite DoclingViewer component

**Files:**
- Modify: `frontend/src/components/DoclingViewer.tsx`

**Step 1: Rewrite the component**

Replace the entire contents of `DoclingViewer.tsx`:

```tsx
import { useEffect, useState, useMemo } from "react";
import { getDoclingRawJson } from "../api/client";

interface DoclingViewerProps {
  documentId: string;
  filename: string;
  onClose: () => void;
}

type ViewMode = "document" | "json";

function buildDoclingHtml(docJson: Record<string, unknown>): string {
  const jsonStr = JSON.stringify(docJson);
  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <script src="/static/docling-components.js" type="module"></script>
    <style>
      body { margin: 0; background: #f5f5f5; }
      docling-img { gap: 1rem; }
      docling-img::part(page) {
        box-shadow: 0 0.5rem 1rem 0 rgba(0, 0, 0, 0.2);
      }
    </style>
  </head>
  <body>
    <docling-img id="dclimg" pagenumbers>
      <docling-tooltip></docling-tooltip>
    </docling-img>

    <script id="dcljson" type="application/json">${jsonStr}</script>

    <script>
      (function() {
        function applySrc() {
          try {
            var data = JSON.parse(document.getElementById('dcljson').textContent);
            var el = document.getElementById('dclimg');
            if (el) el.src = data;
          } catch (e) {
            console.error('Failed to set docling-img src:', e);
          }
        }
        if (!customElements.get('docling-img')) {
          customElements.whenDefined('docling-img').then(applySrc);
        } else {
          applySrc();
        }
      })();
    </script>
  </body>
</html>`;
}

export function DoclingViewer({ documentId, filename, onClose }: DoclingViewerProps) {
  const [docJson, setDocJson] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("document");

  useEffect(() => {
    setLoading(true);
    setError(null);
    getDoclingRawJson(documentId)
      .then(setDocJson)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Failed to load document"),
      )
      .finally(() => setLoading(false));
  }, [documentId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const srcdoc = useMemo(() => {
    if (!docJson) return "";
    return buildDoclingHtml(docJson);
  }, [docJson]);

  return (
    <div className="docling-overlay" onClick={onClose}>
      <div className="docling-modal" onClick={(e) => e.stopPropagation()}>
        <div className="docling-modal-header">
          <h3 className="docling-modal-title" title={filename}>
            {filename}
          </h3>
          <div className="docling-mode-toggle">
            <button
              className={`mode-btn${mode === "document" ? " active" : ""}`}
              onClick={() => setMode("document")}
            >
              Document
            </button>
            {docJson && (
              <button
                className={`mode-btn${mode === "json" ? " active" : ""}`}
                onClick={() => setMode("json")}
              >
                JSON
              </button>
            )}
          </div>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="docling-modal-body">
          {loading && (
            <div className="empty-state">
              <span className="spinner" />
              <p className="mt-sm">Loading document...</p>
            </div>
          )}

          {error && <div className="alert alert-error">{error}</div>}

          {docJson && mode === "document" && (
            <iframe
              srcDoc={srcdoc}
              title={`DoclingViewer: ${filename}`}
              style={{
                width: "100%",
                height: "100%",
                border: "none",
                minHeight: "600px",
              }}
              sandbox="allow-scripts allow-same-origin"
            />
          )}

          {docJson && mode === "json" && (
            <pre className="docling-json-content">
              {JSON.stringify(docJson, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Remove unused imports**

The old component imported `react-markdown` and `remarkGfm`. These are no longer needed by DoclingViewer. However, if other components use them, leave the packages installed. Just ensure the import is removed from this file.

**Step 3: Verify the viewer renders**

```bash
cd frontend && npm run dev
# Open browser, navigate to Ingest page, upload a document (or use existing COMPLETE document)
# Click "View" button — should open modal with page images
```

Expected: Modal shows PDF page images with bounding box overlays. Page numbers visible.

**Step 4: Commit**

```bash
git add frontend/src/components/DoclingViewer.tsx
git commit -m "feat: rewrite DoclingViewer with docling-img web component"
```

---

### Task 4: Enable granite-vision-3.3-2b picture descriptions in Docling

**Files:**
- Modify: `docker/docling/Dockerfile`
- Modify: `docker/docling/app/converter.py`

**Step 1: Add model download to Dockerfile**

Add a second `snapshot_download` line after the existing granite-docling-258M download (line 101):

```dockerfile
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-docling-258M')"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ibm-granite/granite-vision-3.3-2b')"
```

**Step 2: Enable picture descriptions in converter.py**

In `init_converter()`, modify the `VlmPipelineOptions` construction:

```python
from docling.datamodel.pipeline_options import granite_picture_description

pipeline_options = VlmPipelineOptions(
    accelerator_options=accel,
    force_backend_text=True,
    generate_picture_images=True,
    images_scale=2.0,
    do_picture_description=True,
    picture_description_options=granite_picture_description,
)
```

Also import at top of the function and update the fallback `_convert_without_picture_images` to also set `do_picture_description=False` explicitly (it already doesn't set it, so it defaults to False — but be explicit):

```python
opts = VlmPipelineOptions(
    accelerator_options=accel,
    force_backend_text=True,
    generate_picture_images=False,
    do_picture_description=False,
    images_scale=2.0,
)
```

**Step 3: Build the Docker image (this will take a while — large model download)**

```bash
docker compose build docling
```

Expected: Image builds successfully with both models pre-downloaded.

**Step 4: Verify picture descriptions in output**

```bash
docker compose up -d docling
# Wait for model loading, then send a test document:
curl -s -X POST http://localhost:8001/convert \
  -F "file=@test_document.pdf" | python3 -c "
import sys, json
d = json.load(sys.stdin)
dj = d.get('document_json', {})
pics = dj.get('pictures', [])
print(f'Pictures: {len(pics)}')
for p in pics[:2]:
    anns = p.get('annotations', [])
    print(f'  Annotations: {len(anns)}')
    for a in anns:
        print(f'    text preview: {a.get(\"text\",\"\")[:100]}')
"
```

Expected: Pictures have `annotations` with `text` content from granite-vision-3.3-2b.

**Step 5: Commit**

```bash
git add docker/docling/Dockerfile docker/docling/app/converter.py
git commit -m "feat: enable granite-vision-3.3-2b picture descriptions in Docling pipeline"
```

---

### Task 5: Run tests and update README

**Files:**
- Run: `./scripts/run_tests.sh` or `python3 -m pytest tests/`
- Modify: `README.md` (if DoclingViewer or picture descriptions are mentioned)

**Step 1: Run test suite**

```bash
./scripts/run_tests.sh
```

Expected: All tests pass. No tests directly test the new endpoint (it's a MinIO read), but existing tests should not regress.

**Step 2: Verify end-to-end**

1. Start all services: `./manage.sh --start`
2. Upload a PDF with images via the frontend
3. Wait for pipeline completion
4. Click "View" — should show page images with bounding boxes
5. If the document has pictures, annotations should show with scrolling text

**Step 3: Update README if needed**

If the README mentions the DoclingViewer, update to reflect the new page-image display.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README for DoclingDocument page-image viewer"
```
