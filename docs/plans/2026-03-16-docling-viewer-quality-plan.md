# DoclingDocument Viewer Quality Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Fix DoclingDocument viewer output quality — optimize VLM config, post-process markdown, map images inline, support text files.

**Architecture:** (1) Update Docling converter to use modern VlmPipelineOptions with force_backend_text and generate_picture_images, (2) Add markdown cleanup module to strip repetition/noise, (3) Map `<!-- image -->` placeholders to real artifact URLs in the API endpoint, (4) Fall back to raw text for .txt files.

**Tech Stack:** No new dependencies. Docling converter config, Python string processing, backend endpoint logic, minor frontend tweak.

---

### Task 1: Create markdown cleanup module

**Files:**
- Create: `docker/docling/app/cleanup.py`

**Step 1: Create the cleanup module**

```python
"""Post-processing cleanup for Docling-generated markdown.

Removes VLM hallucination artifacts: repeated lines, watermark spam,
bare image placeholders, and excessive whitespace.
"""

from __future__ import annotations

import re
from collections import Counter


def clean_markdown(text: str) -> str:
    """Apply all cleanup rules to a markdown string."""
    text = _collapse_consecutive_duplicates(text)
    text = _strip_spam_lines(text)
    text = _remove_bare_image_comments(text)
    text = _collapse_blank_lines(text)
    text = _strip_trailing_whitespace(text)
    return text


def _collapse_consecutive_duplicates(text: str) -> str:
    """Collapse 3+ consecutive identical lines into one."""
    lines = text.split("\n")
    result: list[str] = []
    prev = None
    count = 0
    for line in lines:
        stripped = line.strip()
        if stripped == prev:
            count += 1
            if count < 3:
                result.append(line)
        else:
            prev = stripped
            count = 1
            result.append(line)
    return "\n".join(result)


def _strip_spam_lines(text: str) -> str:
    """Remove lines that appear 5+ times in the document (keep first occurrence)."""
    lines = text.split("\n")
    stripped_counts = Counter(line.strip() for line in lines if line.strip())

    spam_lines = {line for line, count in stripped_counts.items() if count >= 5}
    # Don't strip common markdown patterns
    spam_lines -= {"", "---", "***", "___", "|", "<!-- image -->"}

    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in spam_lines:
            if stripped not in seen:
                seen.add(stripped)
                result.append(line)
            # else: skip duplicate spam line
        else:
            result.append(line)
    return "\n".join(result)


def _remove_bare_image_comments(text: str) -> str:
    """Remove standalone <!-- image --> comment lines.

    These are Docling placeholders that will be replaced by actual
    image references in the API endpoint.
    """
    return re.sub(r"^\s*<!-- image -->\s*$", "", text, flags=re.MULTILINE)


def _collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive blank lines into 2."""
    return re.sub(r"\n{4,}", "\n\n\n", text)


def _strip_trailing_whitespace(text: str) -> str:
    """Strip trailing whitespace from each line."""
    return "\n".join(line.rstrip() for line in text.split("\n"))
```

**Step 2: Commit**

```bash
git add docker/docling/app/cleanup.py
git commit -m "feat: add markdown cleanup module for Docling output post-processing"
```

---

### Task 2: Update Docling converter configuration

**Files:**
- Modify: `docker/docling/app/converter.py`

**Step 1: Update `init_converter()` to use modern VlmPipelineOptions**

Replace lines 30-64 (the entire `init_converter` function) with:

```python
def init_converter() -> None:
    """Initialize the Docling DocumentConverter with VLM pipeline.

    Called once at service startup. Loads the granite-docling-258M model
    with optimized settings for document conversion quality.
    """
    global _converter, _model_loaded

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    logger.info("Loading Docling converter with model=%s device=%s dtype=%s", MODEL_PATH, DEVICE, DTYPE)

    accel = AcceleratorOptions(
        device=DEVICE,
        cuda_use_flash_attention2=(DEVICE == "cuda"),
    )

    pipeline_options = VlmPipelineOptions(
        accelerator_options=accel,
        force_backend_text=True,
        generate_picture_images=True,
        images_scale=2.0,
    )

    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        },
    )
    _model_loaded = True
    logger.info("Docling converter loaded successfully.")
```

Key changes:
- Uses default `VlmConvertOptions` which already targets granite-docling-258M with `max_new_tokens=8192` and `scale=2.0`
- `force_backend_text=True` — uses PDF text layer instead of VLM OCR (huge quality win)
- `generate_picture_images=True` — extracts actual image crops
- `images_scale=2.0` — 2x page resolution
- `cuda_use_flash_attention2=True` — faster inference on GPU
- Removes deprecated `vlm_model_name`/`vlm_model_device`/`vlm_model_dtype` params

**Step 2: Apply markdown cleanup in `convert_document()`**

After line 108 (`markdown = doc.export_to_markdown()`), add the cleanup import and call:

```python
        from app.cleanup import clean_markdown
        markdown = clean_markdown(markdown)
```

So lines 107-109 become:

```python
        elements = _extract_elements(doc)
        markdown = doc.export_to_markdown()
        from app.cleanup import clean_markdown
        markdown = clean_markdown(markdown)
        document_json = doc.export_to_dict()
```

**Step 3: Verify import**

Run: `docker compose exec docling python3 -c "from app.converter import init_converter; print('ok')"`
Expected: PASS

**Step 4: Commit**

```bash
git add docker/docling/app/converter.py
git commit -m "feat: optimize Docling VLM config and apply markdown cleanup"
```

---

### Task 3: Map image placeholders to artifact URLs in API endpoint

**Files:**
- Modify: `app/api/v1/sources.py`

**Step 1: Add image placeholder replacement**

In the `get_docling_document` endpoint, after the image list is built (after line 512), add logic to replace `<!-- image -->` placeholders in the markdown with actual image URLs. Insert before the `return DoclingDocumentResponse(...)` at line 514:

```python
    # Replace <!-- image --> placeholders with actual image markdown tags.
    # Map Nth placeholder to Nth image artifact (ordered by element_order).
    if images:
        # Sort images by element_uid to maintain document order
        # (element_uid format: "page-order-type-hash")
        sorted_images = sorted(images, key=lambda img: img.element_uid)
        placeholder = "<!-- image -->"
        for img_ref in sorted_images:
            # Replace one placeholder at a time, in order
            markdown_text = markdown_text.replace(
                placeholder,
                f"![image]({img_ref.url})",
                1,  # replace only the first occurrence
            )
```

Also add ordering to the image artifact query. Change the artifact query (lines 495-501) to order by element_order:

```python
    stmt = (
        sa_select(Artifact)
        .where(
            Artifact.document_id == document_id,
            Artifact.artifact_type.in_(["image", "schematic"]),
            Artifact.storage_key.isnot(None),
        )
        .order_by(Artifact.id)  # deterministic order
    )
```

**Step 2: Commit**

```bash
git add app/api/v1/sources.py
git commit -m "feat: replace image placeholders with artifact URLs in DoclingDocument endpoint"
```

---

### Task 4: Add text file fallback to DoclingDocument endpoint

**Files:**
- Modify: `app/api/v1/sources.py`

**Step 1: Add text file fallback**

In the `get_docling_document` endpoint, modify the except block (lines 476-481) to fall back to raw text for text files. Replace the except block with:

```python
    except Exception as exc:
        # Fall back to raw file content for text files
        if doc.mime_type and doc.mime_type.startswith("text/"):
            try:
                raw_bytes = await download_bytes_async(
                    doc.storage_bucket, doc.storage_key
                )
                return DoclingDocumentResponse(
                    document_id=str(document_id),
                    filename=doc.filename or "",
                    markdown=raw_bytes.decode("utf-8", errors="replace"),
                    document_json={},
                    images=[],
                )
            except Exception:
                pass  # fall through to 404

        logger.info("get_docling_document: DoclingDocument not found for %s: %s", document_id, exc)
        raise HTTPException(
            status_code=404,
            detail="DoclingDocument not available for this document. Re-ingest to generate.",
        )
```

**Step 2: Commit**

```bash
git add app/api/v1/sources.py
git commit -m "feat: fall back to raw text content for text files in DoclingDocument endpoint"
```

---

### Task 5: Hide JSON toggle for empty document_json in DoclingViewer

**Files:**
- Modify: `frontend/src/components/DoclingViewer.tsx`

**Step 1: Conditionally render the JSON toggle**

Replace the mode toggle div (lines 46-58) with:

```typescript
          <div className="docling-mode-toggle">
            <button
              className={`mode-btn${mode === "markdown" ? " active" : ""}`}
              onClick={() => setMode("markdown")}
            >
              Document
            </button>
            {data && Object.keys(data.document_json).length > 0 && (
              <button
                className={`mode-btn${mode === "json" ? " active" : ""}`}
                onClick={() => setMode("json")}
              >
                JSON
              </button>
            )}
          </div>
```

**Step 2: Commit**

```bash
git add frontend/src/components/DoclingViewer.tsx
git commit -m "feat: hide JSON toggle when document_json is empty"
```

---

### Task 6: Rebuild Docling service and verify

**Step 1: Rebuild the Docling Docker image**

Run: `docker compose -f docker-compose.yml up -d --build docling`

**Step 2: Verify health**

Run: `curl http://localhost:8001/health`
Expected: `{"status":"ok","model_loaded":true,...}`

**Step 3: Reingest a test document to verify quality**

Upload or reingest a document and check that:
- The "View" button shows clean markdown with inline images
- Text files show raw content
- JSON toggle is hidden for text files
- Repetitive lines are deduplicated

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "test: verify DoclingDocument viewer quality improvements"
```
