# Document Analysis & Enhanced Viewer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LLM-based document metadata extraction and picture description enrichment to the ingestion pipeline, switch Docling to PdfPipeline with dlparse_v4, and display metadata above the document viewer.

**Architecture:** Two new sequential Celery tasks (`derive_document_metadata` → `derive_picture_descriptions`) insert between `prepare_document` and the existing parallel chord. Docling switches from VlmPipeline to PdfPipelineOptions. A new `document_analysis.py` service handles LLM calls to Ollama. Frontend DoclingViewer gains a metadata panel.

**Tech Stack:** Python/FastAPI/Celery, Ollama (gpt-oss:120b + gemma3:27b), Docling PdfPipeline, Alembic, React/TypeScript

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `.env` | Modify | Add doc analysis + picture description env vars |
| `app/config.py` | Modify | Add corresponding settings fields |
| `alembic/versions/0009_add_document_metadata.py` | Create | Migration: add `document_metadata` JSONB column |
| `app/models/ingest.py` | Modify | Add `document_metadata` column to Document model |
| `app/services/document_analysis.py` | Create | LLM metadata extraction + picture description service |
| `app/workers/pipeline.py` | Modify | Add 2 new tasks, rechain pipeline |
| `app/api/v1/sources.py` | Modify | Add metadata endpoint |
| `docker/docling/app/converter.py` | Modify | Switch to PdfPipeline + dlparse_v4 |
| `frontend/src/api/client.ts` | Modify | Add `getDocumentMetadata` function |
| `frontend/src/components/DoclingViewer.tsx` | Modify | Add metadata panel above document |

---

## Task 1: Environment Variables and Config

**Files:**
- Modify: `.env`
- Modify: `app/config.py:124-139`

- [ ] **Step 1: Add env vars to `.env`**

Append after the GraphRAG section:

```env
# Document Analysis (LLM metadata extraction after Docling conversion)
DOC_ANALYSIS_ENABLED=true
DOC_ANALYSIS_LLM_MODEL=gpt-oss:120b
DOC_ANALYSIS_TIMEOUT=300
DOC_ANALYSIS_SUMMARY_PROMPT=Summarize this document in 3-5 sentences for a technical reader. Focus on the main subject, scope, and notable findings. Do not include source endnote markings such as [1].
DOC_ANALYSIS_DATE_PROMPT=Extract the most relevant date of information (publication date, report date, or coverage window). If only month/year appears, return that. If there is a range, return it exactly. If unsure, return Unknown. Provide ONLY the date or range.
DOC_ANALYSIS_SOURCE_PROMPT=Characterize the source: 1) Organization or author (if unknown return UNKNOWN) 2) Type of information (website, journal, etc; if unknown return UNKNOWN) 3) Reliability score 1-10. Format: Organization: <name>\nType: <type>\nReliability: <score>/10
DOC_ANALYSIS_CLASSIFICATION_PROMPT=Identify the document classification marking if present (UNCLASSIFIED, CUI, FOUO, SECRET, TOP SECRET). If none, reply UNCLASSIFIED. Provide ONLY the marking.

# Picture Description (post-conversion via Ollama, uses document summary as context)
PICTURE_DESCRIPTION_MODEL=gemma3:27b
PICTURE_DESCRIPTION_TIMEOUT=120
PICTURE_DESCRIPTION_PROMPT=Analyze this image from a multi-modal PDF using the required narrative sections and the missile/radar/S&T emphasis. Return sections 1-8 exactly as specified. Use the PDF Summary for context but rely on visual evidence.\n\n- PDF Summary: {document_summary}\n\n- Image:

# Docling OCR
DOCLING_OCR_LANG=en
```

- [ ] **Step 2: Add settings fields to `app/config.py`**

Add after the `graphrag_tune_interval_minutes` field (around line 138):

```python
    # Document Analysis (LLM metadata extraction)
    doc_analysis_enabled: bool = True
    doc_analysis_llm_model: str = "gpt-oss:120b"
    doc_analysis_timeout: int = 300
    doc_analysis_summary_prompt: str = "Summarize this document in 3-5 sentences for a technical reader. Focus on the main subject, scope, and notable findings. Do not include source endnote markings such as [1]."
    doc_analysis_date_prompt: str = "Extract the most relevant date of information (publication date, report date, or coverage window). If only month/year appears, return that. If there is a range, return it exactly. If unsure, return Unknown. Provide ONLY the date or range."
    doc_analysis_source_prompt: str = "Characterize the source: 1) Organization or author (if unknown return UNKNOWN) 2) Type of information (website, journal, etc; if unknown return UNKNOWN) 3) Reliability score 1-10. Format: Organization: <name>\\nType: <type>\\nReliability: <score>/10"
    doc_analysis_classification_prompt: str = "Identify the document classification marking if present (UNCLASSIFIED, CUI, FOUO, SECRET, TOP SECRET). If none, reply UNCLASSIFIED. Provide ONLY the marking."

    # Picture Description (post-conversion enrichment via Ollama)
    picture_description_model: str = "gemma3:27b"
    picture_description_timeout: int = 120
    picture_description_prompt: str = "Analyze this image from a multi-modal PDF using the required narrative sections and the missile/radar/S&T emphasis. Return sections 1-8 exactly as specified. Use the PDF Summary for context but rely on visual evidence.\\n\\n- PDF Summary: {document_summary}\\n\\n- Image:"

    # Docling OCR language
    docling_ocr_lang: str = "en"
```

- [ ] **Step 3: Commit**

```bash
git add .env app/config.py
git commit -m "feat: add document analysis and picture description config settings"
```

---

## Task 2: Database Migration

**Files:**
- Create: `alembic/versions/0009_add_document_metadata.py`
- Modify: `app/models/ingest.py:38-82`

- [ ] **Step 1: Create Alembic migration**

Create `alembic/versions/0009_add_document_metadata.py`:

```python
"""Add document_metadata JSONB column to documents table.

Revision ID: 0009
Revises: 0008
Create Date: 2026-03-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("document_metadata", JSONB, nullable=True),
        schema="ingest",
    )


def downgrade() -> None:
    op.drop_column("documents", "document_metadata", schema="ingest")
```

- [ ] **Step 2: Add column to Document model**

In `app/models/ingest.py`, add after the `uploaded_by` field (around line 70):

```python
    # LLM-extracted document metadata (summary, date, classification, source)
    document_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True, default=None
    )
```

Add `JSONB` to the SQLAlchemy imports at the top of the file:

```python
from sqlalchemy.dialects.postgresql import JSONB
```

- [ ] **Step 3: Commit**

```bash
git add alembic/versions/0009_add_document_metadata.py app/models/ingest.py
git commit -m "feat: add document_metadata JSONB column to documents table"
```

---

## Task 3: Document Analysis Service

**Files:**
- Create: `app/services/document_analysis.py`

- [ ] **Step 1: Create the service**

Create `app/services/document_analysis.py`:

```python
"""LLM-based document metadata extraction and picture description enrichment.

Calls Ollama for:
1. Document metadata (summary, date, classification, source) via configurable model
2. Picture descriptions via configurable multimodal model with summary context
"""

import base64
import logging
from datetime import datetime, timezone

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


def extract_document_metadata(markdown: str) -> dict:
    """Extract metadata from document markdown via LLM.

    Calls the configured LLM (DOC_ANALYSIS_LLM_MODEL) with four sequential
    prompts: summary, date, source characterization, classification.

    Returns dict with keys: document_summary, date_of_information,
    classification, source_characterization, generated_at.
    """
    settings = get_settings()
    model = settings.doc_analysis_llm_model
    timeout = settings.doc_analysis_timeout

    def _llm_call(system_prompt: str, user_text: str) -> str:
        """Call Ollama chat completions and return the assistant message."""
        resp = httpx.post(
            f"{settings.ollama_base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # Truncate markdown to avoid exceeding context window
    max_chars = settings.ollama_num_ctx * 3  # rough char-to-token ratio
    doc_text = markdown[:max_chars] if len(markdown) > max_chars else markdown

    document_summary = _llm_call(settings.doc_analysis_summary_prompt, doc_text)
    logger.info("Document summary extracted (%d chars)", len(document_summary))

    date_of_information = _llm_call(settings.doc_analysis_date_prompt, doc_text)
    logger.info("Date of information: %s", date_of_information)

    source_characterization = _llm_call(settings.doc_analysis_source_prompt, doc_text)
    logger.info("Source characterization extracted")

    classification = _llm_call(settings.doc_analysis_classification_prompt, doc_text)
    # Normalize classification
    valid_classes = {"UNCLASSIFIED", "CUI", "FOUO", "SECRET", "TOP SECRET"}
    classification = classification.upper().strip()
    if classification not in valid_classes:
        classification = "UNCLASSIFIED"

    return {
        "document_summary": document_summary,
        "date_of_information": date_of_information,
        "classification": classification,
        "source_characterization": source_characterization,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def describe_pictures(docling_json: dict, document_summary: str) -> dict:
    """Enrich picture items in Docling JSON with LLM-generated descriptions.

    Iterates items in the Docling JSON body, finds PictureItems with embedded
    images, sends each to the configured multimodal model (PICTURE_DESCRIPTION_MODEL)
    with the document summary as context.

    Returns the modified docling_json dict.
    """
    settings = get_settings()
    model = settings.picture_description_model
    timeout = settings.picture_description_timeout
    prompt_template = settings.picture_description_prompt.replace("\\n", "\n")
    prompt = prompt_template.replace("{document_summary}", document_summary)

    # Navigate Docling JSON structure to find picture items
    body = docling_json.get("body") or docling_json.get("main-text") or []
    if isinstance(body, dict):
        body = body.get("children", [])

    pictures_found = 0
    pictures_described = 0

    def _find_pictures(items):
        """Recursively find picture items in the document tree."""
        nonlocal pictures_found, pictures_described
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue

            label = item.get("label", "")
            if label == "picture" or item.get("type") == "picture":
                pictures_found += 1
                # Find the image data
                image_data = _extract_image_data(item, docling_json)
                if image_data:
                    description = _describe_single_image(image_data, prompt, model, timeout, settings)
                    if description:
                        # Store description in the item
                        if "captions" not in item:
                            item["captions"] = []
                        item["captions"].append({
                            "text": description,
                            "source": "llm",
                            "model": model,
                        })
                        # Also set text/description fields
                        item["description"] = description
                        pictures_described += 1

            # Recurse into children
            children = item.get("children", [])
            if children:
                _find_pictures(children)

    # Also check the top-level pictures collection
    pictures = docling_json.get("pictures", [])
    if isinstance(pictures, list):
        for pic in pictures:
            if isinstance(pic, dict):
                pictures_found += 1
                image_ref = pic.get("image", {})
                uri = image_ref.get("uri", "") if isinstance(image_ref, dict) else ""
                if uri and uri.startswith("data:"):
                    # Extract base64 from data URI
                    b64 = uri.split(",", 1)[1] if "," in uri else ""
                    if b64:
                        description = _describe_single_image(b64, prompt, model, timeout, settings)
                        if description:
                            pic["description"] = description
                            if "annotations" not in pic:
                                pic["annotations"] = []
                            pic["annotations"].append({
                                "text": description,
                                "source": "llm",
                                "model": model,
                            })
                            pictures_described += 1

    _find_pictures(body if isinstance(body, list) else [])

    logger.info(
        "Picture descriptions: found=%d, described=%d, model=%s",
        pictures_found, pictures_described, model,
    )
    return docling_json


def _extract_image_data(item: dict, docling_json: dict) -> str | None:
    """Extract base64 image data from a picture item."""
    # Check for inline image data
    image = item.get("image", {})
    if isinstance(image, dict):
        uri = image.get("uri", "")
        if uri.startswith("data:"):
            return uri.split(",", 1)[1] if "," in uri else None

    # Check for image reference in the pictures collection
    ref = item.get("$ref") or item.get("self_ref", "")
    if ref and isinstance(ref, str):
        pictures = docling_json.get("pictures", [])
        if isinstance(pictures, list):
            for pic in pictures:
                if isinstance(pic, dict) and pic.get("self_ref") == ref:
                    img = pic.get("image", {})
                    if isinstance(img, dict):
                        uri = img.get("uri", "")
                        if uri.startswith("data:"):
                            return uri.split(",", 1)[1] if "," in uri else None

    return None


def _describe_single_image(
    image_b64: str, prompt: str, model: str, timeout: int, settings
) -> str | None:
    """Send a single image to the multimodal LLM for description."""
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 2048,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        logger.debug("Picture description (%d chars): %.100s...", len(content), content)
        return content
    except Exception as e:
        logger.warning("Picture description failed: %s", e)
        return None
```

- [ ] **Step 2: Commit**

```bash
git add app/services/document_analysis.py
git commit -m "feat: add document_analysis service for LLM metadata and picture descriptions"
```

---

## Task 4: Pipeline Tasks and Rechaining

**Files:**
- Modify: `app/workers/pipeline.py:27,321-370`

- [ ] **Step 1: Add two new Celery tasks**

Add the following task definitions in `app/workers/pipeline.py` after the `prepare_document` task (before the `derive_text_chunks_and_embeddings` task). Follow the existing patterns for error handling, status updates, and stage run tracking.

```python
@celery_app.task(
    bind=True,
    name="app.workers.pipeline.derive_document_metadata",
    max_retries=2,
    default_retry_delay=30,
    soft_time_limit=settings.doc_analysis_timeout + 60,
    time_limit=settings.doc_analysis_timeout + 120,
    queue="ingest",
)
def derive_document_metadata(self, document_id: str, run_id: str | None = None) -> dict:
    """Extract document metadata (summary, date, classification, source) via LLM."""
    logger.info("derive_document_metadata: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_document_metadata")

    settings = get_settings()
    if not settings.doc_analysis_enabled:
        logger.info("derive_document_metadata: disabled, skipping for %s", document_id)
        return {"stage": "derive_document_metadata", "status": "skipped"}

    db = _get_db()
    try:
        if run_id:
            _update_stage_run(db, run_id, "derive_document_metadata", "RUNNING", attempt=self.request.retries + 1)

        # Load markdown from MinIO
        from app.services.storage import download_bytes_sync
        base_key = f"artifacts/{document_id}"
        bucket = settings.minio_bucket_derived
        md_bytes = download_bytes_sync(bucket, f"{base_key}/docling_document.md")
        markdown = md_bytes.decode("utf-8")

        # Extract metadata via LLM
        from app.services.document_analysis import extract_document_metadata
        metadata = extract_document_metadata(markdown)

        # Store in documents.document_metadata
        from sqlalchemy import text
        db.execute(
            text("UPDATE ingest.documents SET document_metadata = :meta WHERE id = :doc_id"),
            {"meta": json.dumps(metadata), "doc_id": document_id},
        )
        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_document_metadata", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={"summary_length": len(metadata.get("document_summary", ""))},
            )

        logger.info(
            "derive_document_metadata: document_id=%s classification=%s",
            document_id, metadata.get("classification"),
        )
        return {"stage": "derive_document_metadata", "status": "ok", "metadata": metadata}

    except Exception as exc:
        logger.error("derive_document_metadata failed for %s: %s", document_id, exc)
        if run_id:
            _update_stage_run(db, run_id, "derive_document_metadata", "FAILED", attempt=self.request.retries + 1, error=str(exc))
        _record_failed_stage(document_id, "derive_document_metadata", str(exc))
        return {"stage": "derive_document_metadata", "status": "failed", "error": str(exc)}
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.workers.pipeline.derive_picture_descriptions",
    max_retries=1,
    default_retry_delay=30,
    soft_time_limit=1800,
    time_limit=1860,
    queue="ingest",
)
def derive_picture_descriptions(self, document_id: str, run_id: str | None = None) -> dict:
    """Enrich picture items with LLM-generated descriptions using document summary context."""
    logger.info("derive_picture_descriptions: document_id=%s run_id=%s", document_id, run_id)
    _update_document_status(document_id, STATUS_PROCESSING, stage="derive_picture_descriptions")

    settings = get_settings()
    db = _get_db()
    try:
        if run_id:
            _update_stage_run(db, run_id, "derive_picture_descriptions", "RUNNING", attempt=self.request.retries + 1)

        # Load document metadata for summary
        from sqlalchemy import text as sa_text
        row = db.execute(
            sa_text("SELECT document_metadata FROM ingest.documents WHERE id = :doc_id"),
            {"doc_id": document_id},
        ).first()
        document_summary = ""
        if row and row[0]:
            document_summary = row[0].get("document_summary", "")

        # Load Docling JSON from MinIO
        from app.services.storage import download_bytes_sync, upload_bytes_sync
        base_key = f"artifacts/{document_id}"
        bucket = settings.minio_bucket_derived
        json_bytes = download_bytes_sync(bucket, f"{base_key}/docling_document.json")
        import json as json_mod
        docling_json = json_mod.loads(json_bytes)

        # Enrich pictures with descriptions
        from app.services.document_analysis import describe_pictures
        updated_json = describe_pictures(docling_json, document_summary)

        # Write updated JSON back to MinIO
        upload_bytes_sync(
            json_mod.dumps(updated_json, ensure_ascii=False, default=str).encode("utf-8"),
            bucket,
            f"{base_key}/docling_document.json",
            content_type="application/json; charset=utf-8",
        )

        # Update DocumentElement rows for picture elements
        from app.models.ingest import DocumentElement
        from sqlalchemy import select
        pic_elements = db.execute(
            select(DocumentElement).where(
                DocumentElement.document_id == uuid.UUID(document_id),
                DocumentElement.element_type == "image",
            )
        ).scalars().all()

        pictures_updated = 0
        pictures = updated_json.get("pictures", [])
        for elem in pic_elements:
            # Try to find matching picture with description
            for pic in pictures:
                if isinstance(pic, dict) and pic.get("description"):
                    desc = pic["description"]
                    if elem.content_text != desc:
                        elem.content_text = desc
                        pictures_updated += 1
                        break
        db.commit()

        if run_id:
            _update_stage_run(
                db, run_id, "derive_picture_descriptions", "COMPLETE",
                attempt=self.request.retries + 1,
                metrics={"pictures_updated": pictures_updated},
            )

        logger.info("derive_picture_descriptions: document_id=%s updated=%d", document_id, pictures_updated)
        return {"stage": "derive_picture_descriptions", "status": "ok", "pictures_updated": pictures_updated}

    except Exception as exc:
        logger.error("derive_picture_descriptions failed for %s: %s", document_id, exc)
        if run_id:
            _update_stage_run(db, run_id, "derive_picture_descriptions", "FAILED", attempt=self.request.retries + 1, error=str(exc))
        _record_failed_stage(document_id, "derive_picture_descriptions", str(exc))
        return {"stage": "derive_picture_descriptions", "status": "failed", "error": str(exc)}
    finally:
        db.close()
```

- [ ] **Step 2: Rechain the pipeline in `start_ingest_pipeline`**

Replace the pipeline chain in `start_ingest_pipeline` (lines 354-368):

```python
    pipeline = chain(
        prepare_document.si(document_id, run_id),
        derive_document_metadata.si(document_id, run_id),
        derive_picture_descriptions.si(document_id, run_id),
        purge_document_derivations.si(document_id, run_id),
        chord(
            group(
                derive_text_chunks_and_embeddings.si(document_id, run_id),
                derive_image_embeddings.si(document_id, run_id),
                derive_ontology_graph.si(document_id, run_id),
            ),
            collect_derivations.s(document_id, run_id),
        ).on_error(errback),
        derive_structure_links.si(document_id, run_id),
        derive_canonicalization.si(document_id, run_id),
        finalize_document.si(document_id, run_id),
    )
```

- [ ] **Step 3: Add `import json` at top of pipeline.py if not already present**

- [ ] **Step 4: Commit**

```bash
git add app/workers/pipeline.py
git commit -m "feat: add derive_document_metadata and derive_picture_descriptions pipeline tasks"
```

---

## Task 5: Switch Docling to PdfPipeline

**Files:**
- Modify: `docker/docling/app/converter.py:78-166`

- [ ] **Step 1: Rewrite `init_converter` function**

Replace `init_converter` (lines 78-123):

```python
def init_converter() -> None:
    """Initialize the Docling DocumentConverter with PdfPipeline.

    Uses dlparse_v4 backend with EasyOCR, TableFormer FAST, and
    formula/code enrichment. Picture descriptions are handled
    post-conversion by the pipeline via Ollama.
    """
    global _converter, _model_loaded

    _patch_pil_crop()

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        EasyOcrOptions,
        TableStructureOptions,
        TableFormerMode,
    )
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    ocr_lang_str = os.environ.get("DOCLING_OCR_LANG", "en")
    ocr_languages = [lang.strip() for lang in ocr_lang_str.split(",")]

    logger.info(
        "Loading Docling converter: PdfPipeline, dlparse_v4, EasyOCR(%s), device=%s",
        ocr_languages, DEVICE,
    )

    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=DEVICE),
        do_ocr=True,
        ocr_options=EasyOcrOptions(lang=ocr_languages, use_gpu=(DEVICE == "cuda")),
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.FAST,
        ),
        do_formula_enrichment=True,
        do_code_enrichment=True,
        generate_picture_images=True,
        generate_page_images=True,
        images_scale=1.0,
        do_picture_description=False,
    )

    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        },
    )
    _model_loaded = True
    logger.info("Docling converter loaded successfully (PdfPipeline + dlparse_v4).")
```

- [ ] **Step 2: Rewrite `_convert_without_picture_images` fallback**

Replace `_convert_without_picture_images` (lines 131-166):

```python
def _convert_without_picture_images(tmp_path: str):
    """Retry conversion with generate_picture_images=False.

    Used as a fallback when image cropping crashes.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        EasyOcrOptions,
        TableStructureOptions,
        TableFormerMode,
    )
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    ocr_lang_str = os.environ.get("DOCLING_OCR_LANG", "en")
    ocr_languages = [lang.strip() for lang in ocr_lang_str.split(",")]

    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=DEVICE),
        do_ocr=True,
        ocr_options=EasyOcrOptions(lang=ocr_languages, use_gpu=(DEVICE == "cuda")),
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.FAST,
        ),
        do_formula_enrichment=True,
        do_code_enrichment=True,
        generate_picture_images=False,
        generate_page_images=True,
        images_scale=1.0,
        do_picture_description=False,
    )

    fallback = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        },
    )
    return fallback.convert(source=tmp_path)
```

- [ ] **Step 3: Commit**

```bash
git add docker/docling/app/converter.py
git commit -m "feat: switch Docling from VlmPipeline to PdfPipeline with dlparse_v4"
```

---

## Task 6: Metadata API Endpoint

**Files:**
- Modify: `app/api/v1/sources.py`

- [ ] **Step 1: Add metadata endpoint**

Add after the `get_docling_raw_json` endpoint:

```python
@router.get("/documents/{document_id}/metadata")
async def get_document_metadata(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    """Return the LLM-extracted document metadata (summary, date, classification, source)."""
    from sqlalchemy import text

    row = (await db.execute(
        text("SELECT document_metadata FROM ingest.documents WHERE id = :doc_id"),
        {"doc_id": str(document_id)},
    )).first()

    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Document metadata not yet extracted")

    return row[0]
```

- [ ] **Step 2: Commit**

```bash
git add app/api/v1/sources.py
git commit -m "feat: add GET /v1/documents/{id}/metadata endpoint"
```

---

## Task 7: Frontend — Metadata Panel in DoclingViewer

**Files:**
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/components/DoclingViewer.tsx`

- [ ] **Step 1: Add `getDocumentMetadata` to API client**

Add in `frontend/src/api/client.ts` after the `getDoclingRawJson` function:

```typescript
export async function getDocumentMetadata(
  documentId: string,
): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`/v1/documents/${documentId}/metadata`);
    if (res.status === 404) return null;
    return handleResponse<Record<string, unknown>>(res);
  } catch {
    return null;
  }
}
```

- [ ] **Step 2: Update DoclingViewer to fetch and display metadata**

In `frontend/src/components/DoclingViewer.tsx`:

Add import:
```typescript
import { getDoclingRawJson, getDoclingDocument, getDocumentMetadata } from "../api/client";
```

Add metadata state alongside existing state:
```typescript
const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null);
```

Fetch metadata in the existing `useEffect` (alongside Docling JSON fetch):
```typescript
getDocumentMetadata(documentId).then(setMetadata);
```

Add metadata panel rendering above the iframe in the modal body (before the document viewer):
```tsx
{metadata && (
  <div style={{
    border: "1px solid var(--color-border)",
    borderRadius: "var(--radius)",
    padding: "0.75rem 1rem",
    marginBottom: "0.5rem",
    background: "var(--color-surface-2)",
    fontSize: "0.85rem",
  }}>
    <div style={{ fontWeight: 600, marginBottom: "0.5rem", color: "var(--color-text-muted)" }}>
      AI-Extracted Document Metadata
    </div>
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.25rem 1.5rem" }}>
      <div><strong>Classification:</strong> {String(metadata.classification || "UNCLASSIFIED")}</div>
      <div><strong>Date of Information:</strong> {String(metadata.date_of_information || "Unknown")}</div>
      <div style={{ gridColumn: "1 / -1" }}>
        <strong>Source:</strong> {String(metadata.source_characterization || "Unknown")}
      </div>
      <div style={{ gridColumn: "1 / -1" }}>
        <strong>Summary:</strong> {String(metadata.document_summary || "")}
      </div>
    </div>
  </div>
)}
```

- [ ] **Step 3: TypeScript build check**

```bash
cd frontend && npx tsc --noEmit
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api/client.ts frontend/src/components/DoclingViewer.tsx
git commit -m "feat: add metadata panel above document in DoclingViewer"
```
