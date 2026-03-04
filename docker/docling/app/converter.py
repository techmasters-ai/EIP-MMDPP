"""Docling document converter wrapper.

Loads granite-docling-258M at startup and provides a convert_document function
that processes files through the Docling VLM pipeline.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import time
from pathlib import Path

from app.schemas import ConvertedElement, ConvertResponse

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("DOCLING_MODEL_PATH", "/models/granite-docling-258M")
DEVICE = os.environ.get("DOCLING_DEVICE", "cuda")
DTYPE = os.environ.get("DOCLING_DTYPE", "bfloat16")

# Module-level converter singleton — initialized via init_converter()
_converter = None
_model_loaded = False


def init_converter() -> None:
    """Initialize the Docling DocumentConverter with VLM pipeline.

    Called once at service startup. Loads the granite-docling-258M model.
    """
    global _converter, _model_loaded

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        VlmPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    logger.info("Loading Docling converter with model=%s device=%s", MODEL_PATH, DEVICE)

    pipeline_options = VlmPipelineOptions(
        vlm_model_name=MODEL_PATH,
        vlm_model_device=DEVICE,
        vlm_model_dtype=DTYPE,
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


def is_model_loaded() -> bool:
    """Check if the model has been loaded."""
    return _model_loaded


def convert_document(file_bytes: bytes, filename: str) -> ConvertResponse:
    """Convert a document through the Docling VLM pipeline.

    Args:
        file_bytes: Raw document bytes.
        filename: Original filename (used for format detection).

    Returns:
        ConvertResponse with structured elements and full markdown.
    """
    if not _model_loaded or _converter is None:
        return ConvertResponse(
            status="error",
            filename=filename,
            num_pages=0,
            elements=[],
            markdown="",
            processing_time_ms=0,
            error="Model not loaded",
        )

    start = time.monotonic()

    # Docling requires a file path — write to temp file
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = _converter.convert(source=tmp_path)
        doc = result.document

        elements = _extract_elements(doc)
        markdown = doc.export_to_markdown()
        num_pages = _count_pages(doc)

        elapsed_ms = (time.monotonic() - start) * 1000

        return ConvertResponse(
            status="ok",
            filename=filename,
            num_pages=num_pages,
            elements=elements,
            markdown=markdown,
            processing_time_ms=round(elapsed_ms, 1),
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.error("Docling conversion failed for %s: %s", filename, exc)
        return ConvertResponse(
            status="error",
            filename=filename,
            num_pages=0,
            elements=[],
            markdown="",
            processing_time_ms=round(elapsed_ms, 1),
            error=str(exc),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _extract_elements(doc) -> list[ConvertedElement]:
    """Iterate Docling document items and map to ConvertedElement list."""
    from docling_core.types.doc import (
        DocItemLabel,
        ImageRefMode,
    )

    elements: list[ConvertedElement] = []

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        page_no = _get_page_number(item)

        if label in (
            DocItemLabel.SECTION_HEADER,
            DocItemLabel.TITLE,
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
        ):
            text = item.export_to_markdown()
            if text.strip():
                elements.append(
                    ConvertedElement(
                        element_type="heading",
                        content_text=text.strip(),
                        page_number=page_no,
                    )
                )

        elif label == DocItemLabel.TABLE:
            md_table = item.export_to_markdown()
            elements.append(
                ConvertedElement(
                    element_type="table",
                    content_text=md_table,
                    page_number=page_no,
                    metadata={"label": "table"},
                )
            )

        elif label == DocItemLabel.PICTURE:
            # Try to get image bytes
            image_b64 = None
            caption_text = ""
            try:
                image = item.get_image(doc)
                if image is not None:
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    image_b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                pass

            try:
                caption_text = item.export_to_markdown().strip()
            except Exception:
                pass

            elements.append(
                ConvertedElement(
                    element_type="image",
                    content_text=caption_text,
                    page_number=page_no,
                    image_base64=image_b64,
                    metadata={"label": "picture", "ext": "png"},
                )
            )

        elif label == DocItemLabel.FORMULA:
            text = item.export_to_markdown()
            if text.strip():
                elements.append(
                    ConvertedElement(
                        element_type="equation",
                        content_text=text.strip(),
                        page_number=page_no,
                        metadata={"label": "formula"},
                    )
                )

        else:
            # Default: treat as text (paragraphs, lists, captions, etc.)
            text = item.export_to_markdown()
            if text and text.strip():
                elements.append(
                    ConvertedElement(
                        element_type="text",
                        content_text=text.strip(),
                        page_number=page_no,
                    )
                )

    return elements


def _get_page_number(item) -> int | None:
    """Extract page number from a Docling document item."""
    try:
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            return prov[0].page_no
    except Exception:
        pass
    return None


def _count_pages(doc) -> int:
    """Count pages in a Docling document."""
    try:
        return len(doc.pages)
    except Exception:
        return 0
