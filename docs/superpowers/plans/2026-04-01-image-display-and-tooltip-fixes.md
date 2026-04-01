# Image Display & Tooltip Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix standalone image files producing 0 elements (blank viewer) and wire up PDF image description hover tooltips.

**Architecture:** Two surgical fixes — synthesize an image element in the ingest pipeline when Docling returns nothing for standalone images, and inject description annotations server-side in the docling-raw API endpoint so the existing `<docling-tooltip>` web component renders them on hover.

**Tech Stack:** Python (FastAPI, Celery, SQLAlchemy), Docling web components

**Spec:** `docs/superpowers/specs/2026-04-01-image-display-and-tooltip-fixes-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `app/workers/pipeline.py` | Modify (line ~762) | Add standalone image element synthesis |
| `app/api/v1/sources.py` | Modify (line ~768) | Add annotation injection to docling-raw endpoint |
| `tests/unit/test_pipeline.py` | Modify | Add test for standalone image synthesis |
| `tests/unit/test_docling_raw_annotations.py` | Create | Test annotation injection logic |

---

## Chunk 1: Fix 1 — Standalone Image Element Synthesis

### Task 1: Write test for standalone image synthesis

**Files:**
- Modify: `tests/unit/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_pipeline.py`:

```python
class TestStandaloneImageSynthesis:
    """Verify that standalone image files get a synthesized element
    when Docling returns 0 elements."""

    def test_synthesis_produces_one_image_element(self):
        """When Docling returns 0 elements for an image MIME type,
        _synthesize_standalone_image should produce a single image ExtractedChunk."""
        from app.workers.pipeline import _synthesize_standalone_image
        from app.services.extraction import ExtractedChunk

        fake_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = _synthesize_standalone_image(fake_bytes, "image/png")

        assert result is not None
        assert len(result) == 1
        chunk = result[0]
        assert isinstance(chunk, ExtractedChunk)
        assert chunk.modality == "image"
        assert chunk.chunk_text == ""
        assert chunk.page_number == 1
        assert chunk.raw_image_bytes == fake_bytes
        assert chunk.metadata["label"] == "picture"
        assert chunk.metadata["ext"] == "png"

    def test_synthesis_returns_none_for_non_image(self):
        """Non-image MIME types should return None (no synthesis)."""
        from app.workers.pipeline import _synthesize_standalone_image

        result = _synthesize_standalone_image(b"hello", "application/pdf")
        assert result is None

    def test_synthesis_handles_jpeg_extension(self):
        """JPEG MIME should produce ext='jpeg'."""
        from app.workers.pipeline import _synthesize_standalone_image

        result = _synthesize_standalone_image(b"\xff\xd8\xff", "image/jpeg")
        assert result is not None
        assert result[0].metadata["ext"] == "jpeg"

    def test_synthesis_always_produces_for_image_mime(self):
        """The function always produces for image MIME — the caller
        guards on len(result.elements) == 0."""
        from app.workers.pipeline import _synthesize_standalone_image

        result = _synthesize_standalone_image(b"\x89PNG", "image/png")
        assert result is not None  # synthesis always produces if image mime
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/josh/development/EIP-MMDPP && python -m pytest tests/unit/test_pipeline.py::TestStandaloneImageSynthesis -v 2>&1 | tail -10`

Expected: FAIL — `ImportError: cannot import name '_synthesize_standalone_image'`

### Task 2: Implement standalone image synthesis

**Files:**
- Modify: `app/workers/pipeline.py:762` (after dedup block, before element_uids loop)

- [ ] **Step 3: Add the helper function**

Add this function in `app/workers/pipeline.py`, after `_dedupe_extracted_elements` (around line 141), before `_update_document_status`:

```python
def _synthesize_standalone_image(file_bytes: bytes, mime_type: str) -> list | None:
    """Synthesize a single image element for standalone image files.

    When Docling returns 0 elements for an image MIME type, create an
    ExtractedChunk so the pipeline can CLIP-embed and LLM-describe it.

    Returns a list with one ExtractedChunk, or None if mime_type is not an image.
    """
    if not mime_type.startswith("image/"):
        return None

    from app.services.extraction import ExtractedChunk

    ext = mime_type.split("/")[-1]  # e.g. "png", "jpeg", "tiff"
    return [ExtractedChunk(
        chunk_text="",
        modality="image",
        page_number=1,
        raw_image_bytes=file_bytes,
        metadata={"label": "picture", "ext": ext},
        bounding_box=None,
    )]
```

- [ ] **Step 4: Call the helper in prepare_document**

In `app/workers/pipeline.py`, after the dedup block (line ~761), before the `# 5. Build element_uids` comment (line ~763), insert:

```python
        # 4b. Standalone image fallback — synthesize element when Docling
        #     returns 0 elements for an image file (JPEG, PNG, TIFF, etc.)
        if len(result.elements) == 0 and mime_type.startswith("image/"):
            synthesized = _synthesize_standalone_image(file_bytes, mime_type)
            if synthesized:
                result.elements = synthesized
                result.num_pages = max(result.num_pages, 1)
                logger.info(
                    "prepare_document: synthesized standalone image element for %s (mime=%s)",
                    document_id, mime_type,
                )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/josh/development/EIP-MMDPP && python -m pytest tests/unit/test_pipeline.py::TestStandaloneImageSynthesis -v 2>&1 | tail -10`

Expected: 4 passed

- [ ] **Step 6: Run full unit test suite to check for regressions**

Run: `cd /home/josh/development/EIP-MMDPP && python -m pytest tests/unit/ -x -q 2>&1 | tail -10`

Expected: All pass, no regressions

- [ ] **Step 7: Commit**

```bash
git add app/workers/pipeline.py tests/unit/test_pipeline.py
git commit -m "fix: synthesize image element for standalone image files

When Docling returns 0 elements for image MIME types (JPEG, PNG, TIFF,
etc.), synthesize a single ExtractedChunk so the pipeline can CLIP-embed
and LLM-describe the image. Previously standalone images showed blank
in the viewer with COMPLETE status and 0 elements.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 2: Fix 2 — Server-Side Annotation Injection

### Task 3: Write test for annotation injection logic

**Files:**
- Create: `tests/unit/test_docling_raw_annotations.py`

- [ ] **Step 8: Write the failing tests**

Create `tests/unit/test_docling_raw_annotations.py`:

```python
"""Unit tests for docling-raw annotation injection.

Tests the _inject_image_description_annotations helper that enriches
Docling JSON with image description tooltips from the database.
"""

import pytest

pytestmark = pytest.mark.unit


class TestInjectImageDescriptionAnnotations:
    """Test the annotation injection helper used by get_docling_raw_json."""

    def test_injects_description_into_picture_annotations(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [{"page_no": 1}], "annotations": []},
            ],
            "texts": [{"text": "hello"}],
        }
        descriptions = [
            {"page_number": 1, "content_text": "A radar system photo."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)

        annos = doc_json["pictures"][0]["annotations"]
        assert len(annos) == 1
        assert annos[0]["kind"] == "description"
        assert annos[0]["text"] == "A radar system photo."
        assert annos[0]["provenance"] == "llm-generated"

    def test_matches_by_page_and_order(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [{"page_no": 1}], "annotations": []},
                {"prov": [{"page_no": 1}], "annotations": []},
                {"prov": [{"page_no": 2}], "annotations": []},
            ],
            "texts": [],
        }
        descriptions = [
            {"page_number": 1, "content_text": "First image on page 1."},
            {"page_number": 1, "content_text": "Second image on page 1."},
            {"page_number": 2, "content_text": "Image on page 2."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)

        assert doc_json["pictures"][0]["annotations"][0]["text"] == "First image on page 1."
        assert doc_json["pictures"][1]["annotations"][0]["text"] == "Second image on page 1."
        assert doc_json["pictures"][2]["annotations"][0]["text"] == "Image on page 2."

    def test_skips_pictures_with_empty_prov(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [], "annotations": []},
                {"prov": [{"page_no": 1}], "annotations": []},
            ],
            "texts": [],
        }
        descriptions = [
            {"page_number": 1, "content_text": "Page 1 image."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)

        # First picture (no prov) should have no annotations
        assert len(doc_json["pictures"][0]["annotations"]) == 0
        # Second picture should get the description
        assert len(doc_json["pictures"][1]["annotations"]) == 1

    def test_handles_mismatched_counts_gracefully(self):
        """More pictures than descriptions on a page — extra pictures get nothing."""
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [{"page_no": 1}], "annotations": []},
                {"prov": [{"page_no": 1}], "annotations": []},
                {"prov": [{"page_no": 1}], "annotations": []},
            ],
            "texts": [],
        }
        descriptions = [
            {"page_number": 1, "content_text": "Only one description."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)

        assert len(doc_json["pictures"][0]["annotations"]) == 1
        assert len(doc_json["pictures"][1]["annotations"]) == 0
        assert len(doc_json["pictures"][2]["annotations"]) == 0

    def test_no_descriptions_leaves_annotations_empty(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [{"page_no": 1}], "annotations": []},
            ],
            "texts": [],
        }

        _inject_image_description_annotations(doc_json, [])

        assert len(doc_json["pictures"][0]["annotations"]) == 0

    def test_no_pictures_is_noop(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {"pictures": [], "texts": [{"text": "hi"}]}
        descriptions = [
            {"page_number": 1, "content_text": "Orphan description."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)
        assert doc_json["pictures"] == []

    def test_preserves_existing_annotations(self):
        from app.api.v1.sources import _inject_image_description_annotations

        doc_json = {
            "pictures": [
                {"prov": [{"page_no": 1}], "annotations": [
                    {"kind": "classification", "predicted_classes": []}
                ]},
            ],
            "texts": [],
        }
        descriptions = [
            {"page_number": 1, "content_text": "New description."},
        ]

        _inject_image_description_annotations(doc_json, descriptions)

        annos = doc_json["pictures"][0]["annotations"]
        assert len(annos) == 2
        assert annos[0]["kind"] == "classification"
        assert annos[1]["kind"] == "description"
```

- [ ] **Step 9: Run tests to verify they fail**

Run: `cd /home/josh/development/EIP-MMDPP && python -m pytest tests/unit/test_docling_raw_annotations.py -v 2>&1 | tail -10`

Expected: FAIL — `ImportError: cannot import name '_inject_image_description_annotations'`

### Task 4: Implement annotation injection

**Files:**
- Modify: `app/api/v1/sources.py:768-796`

- [ ] **Step 10: Add the helper function**

Add this function in `app/api/v1/sources.py`, just before the `get_docling_raw_json` endpoint (before line 768):

```python
def _inject_image_description_annotations(
    doc_json: dict,
    descriptions: list[dict],
) -> None:
    """Inject image description annotations into Docling JSON pictures.

    Matches pictures to descriptions by page number + positional order
    within each page. Both originate from the same Docling conversion,
    so ordering is authoritative.

    Mutates doc_json in place.

    Note: The ``annotations`` field on Docling items is deprecated in newer
    docling-core versions. This uses it because the <docling-tooltip> web
    component reads annotations with kind="description" to render tooltips.
    """
    from collections import defaultdict

    pictures = doc_json.get("pictures", [])
    if not pictures or not descriptions:
        return

    # Group descriptions by page
    descs_by_page: dict[int, list[dict]] = defaultdict(list)
    for desc in descriptions:
        pg = desc.get("page_number")
        if pg is not None:
            descs_by_page[pg].append(desc)

    # Group pictures by page (from prov)
    pics_by_page: dict[int, list[dict]] = defaultdict(list)
    for pic in pictures:
        prov = pic.get("prov")
        if prov and len(prov) > 0:
            pg = prov[0].get("page_no")
            if pg is not None:
                pics_by_page[pg].append(pic)

    # Zip-match within each page
    for page_no, page_pics in pics_by_page.items():
        page_descs = descs_by_page.get(page_no, [])
        for pic, desc in zip(page_pics, page_descs):
            if not isinstance(pic.get("annotations"), list):
                pic["annotations"] = []
            pic["annotations"].append({
                "kind": "description",
                "text": desc["content_text"],
                "provenance": "llm-generated",
            })
```

- [ ] **Step 11: Update the endpoint to use the helper**

Replace the `get_docling_raw_json` endpoint (lines 768-796) with:

```python
@router.get("/documents/{document_id}/docling-raw")
async def get_docling_raw_json(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    """Return the DoclingDocument JSON enriched with image description annotations.

    Returns the full DoclingDocument including base64 page images,
    intended for the <docling-img> web component viewer. Image descriptions
    from the pipeline are injected as annotations so <docling-tooltip>
    renders them on hover.
    """
    import json as _json
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

    doc_json = _json.loads(json_bytes)

    # Standalone image guard: if Docling produced no usable content,
    # return 404 so the viewer falls through to the standalone fallback
    # panel which shows the raw image + description text.
    if not doc_json.get("pictures") and not doc_json.get("texts"):
        raise HTTPException(
            status_code=404,
            detail="DoclingDocument has no content elements. Use standalone viewer.",
        )

    # Inject image description annotations for <docling-tooltip>
    from sqlalchemy import text

    rows = (await db.execute(
        text("""
            SELECT page_number, content_text
            FROM ingest.document_elements
            WHERE document_id = cast(:doc_id AS uuid)
              AND element_type = 'image'
              AND content_text IS NOT NULL
              AND length(content_text) > 10
            ORDER BY element_order
        """),
        {"doc_id": str(document_id)},
    )).fetchall()

    if rows:
        descriptions = [
            {"page_number": row[0], "content_text": row[1]}
            for row in rows
        ]
        _inject_image_description_annotations(doc_json, descriptions)

    enriched_bytes = _json.dumps(doc_json).encode("utf-8")
    return Response(content=enriched_bytes, media_type="application/json")
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `cd /home/josh/development/EIP-MMDPP && python -m pytest tests/unit/test_docling_raw_annotations.py -v 2>&1 | tail -10`

Expected: 7 passed

- [ ] **Step 13: Run full unit test suite**

Run: `cd /home/josh/development/EIP-MMDPP && python -m pytest tests/unit/ -x -q 2>&1 | tail -10`

Expected: All pass

- [ ] **Step 14: Commit**

```bash
git add app/api/v1/sources.py tests/unit/test_docling_raw_annotations.py
git commit -m "fix: inject image description annotations into docling-raw JSON

Enrich the Docling JSON returned by GET /v1/documents/{id}/docling-raw
with {kind: 'description', text: ..., provenance: 'llm-generated'}
annotations on picture items. The <docling-tooltip> web component reads
these to render hover tooltips with AI image analysis.

Matching is by page number + positional order within each page.
Returns 404 for empty Docling JSON (standalone images) so the viewer
falls through to the description panel.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 3: Rebuild, Test End-to-End, Update Docs

### Task 5: Rebuild and verify

- [ ] **Step 15: Rebuild the API container**

```bash
docker compose up -d --build api
```

Wait for healthy status.

- [ ] **Step 16: Verify Fix 2 — PDF image description tooltips**

Test with an existing PDF that has images:

```bash
# Check that annotations are now present in the docling-raw response
curl -s http://localhost:8005/v1/documents/d74ca6c9-9753-40aa-afe3-e6849390e092/docling-raw | \
  python3 -c "
import sys, json
d = json.load(sys.stdin)
pics = d.get('pictures', [])
with_annos = [p for p in pics if p.get('annotations')]
print(f'{len(with_annos)}/{len(pics)} pictures have description annotations')
if with_annos:
    a = with_annos[0]['annotations'][0]
    print(f'First annotation: kind={a[\"kind\"]}, provenance={a.get(\"provenance\")}, text={a[\"text\"][:80]}...')
"
```

Expected: `81/89 pictures have description annotations` (or similar — matches the count of image elements with descriptions in the DB).

- [ ] **Step 17: Verify Fix 2 — standalone image returns 404 for docling-raw**

```bash
# The standalone JPEG should return 404 so viewer falls through to description panel
curl -s -o /dev/null -w "%{http_code}" http://localhost:8005/v1/documents/3149dc57-0419-4f60-8c7c-9c856a8c9774/docling-raw
```

Expected: `404`

- [ ] **Step 18: Rebuild worker containers for Fix 1**

```bash
docker compose up -d --build worker-ingest worker-embed worker-graph
```

- [ ] **Step 19: Verify Fix 1 — re-ingest the standalone JPEG**

```bash
# Reingest the Fan_Song_Radar.jpeg to test element synthesis
curl -s -X POST http://localhost:8005/v1/documents/3149dc57-0419-4f60-8c7c-9c856a8c9774/reingest | python3 -m json.tool
```

Then poll status until COMPLETE. Check elements were created:

```bash
# After pipeline completes, verify 1 image element exists
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -c "
SELECT count(*), element_type
FROM ingest.document_elements
WHERE document_id = '3149dc57-0419-4f60-8c7c-9c856a8c9774'
GROUP BY element_type;"
```

Expected: `1 | image`

- [ ] **Step 20: Commit any doc updates if needed and final verification**

Open the web UI:
1. Navigate to Uploads, find Fan_Song_Radar.jpeg, click View → should show the image + AI Image Analysis description panel
2. Navigate to a PDF with images, click View → hover over an image → should see the "AI Image Analysis" tooltip with scrolling description

If both work, the fixes are complete.

```bash
git add -A && git status
# Only commit if there are changes (e.g. README updates)
```
