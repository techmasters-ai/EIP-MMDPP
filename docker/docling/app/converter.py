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

MODEL_PATH = os.environ.get("DOCLING_MODEL_PATH", "ibm-granite/granite-docling-258M")
DEVICE = os.environ.get("DOCLING_DEVICE", "cuda")
DTYPE = os.environ.get("DOCLING_DTYPE", "bfloat16")

# Module-level converter singleton — initialized via init_converter()
_converter = None
_model_loaded = False


def _patch_pil_crop() -> None:
    """Monkey-patch PIL Image.crop to handle malformed bounding boxes.

    Workaround for docling-core bug where VLM predicts bounding boxes
    with inverted/zero-area coordinates, causing either:
    - ValueError: Coordinate 'lower' is less than 'upper'
    - SystemError: tile cannot extend outside image
    See: https://github.com/docling-project/docling/issues/2763
    """
    try:
        from PIL import Image as PILImage

        _original_crop = PILImage.Image.crop

        def _safe_crop(self, box=None):
            if box is not None:
                left, upper, right, lower = box
                # Fix inverted coordinates
                if right < left:
                    left, right = right, left
                if lower < upper:
                    upper, lower = lower, upper
                # Ensure minimum 1px dimensions
                if right <= left:
                    right = left + 1
                if lower <= upper:
                    lower = upper + 1
                # Clamp to image bounds
                right = min(right, self.width)
                lower = min(lower, self.height)
                left = max(0, left)
                upper = max(0, upper)
                if right <= left or lower <= upper:
                    logger.warning("Malformed crop box (%s) on image %s, returning 1x1 placeholder", box, self.size)
                    return PILImage.new(self.mode, (1, 1))
                box = (left, upper, right, lower)
            return _original_crop(self, box)

        PILImage.Image.crop = _safe_crop
        logger.info("Patched PIL Image.crop for malformed bounding box workaround")
    except Exception as e:
        logger.warning("Failed to patch PIL Image.crop: %s", e)


def init_converter() -> None:
    """Initialize the Docling DocumentConverter with VLM pipeline.

    Called once at service startup. Loads the granite-docling-258M model
    with optimized settings for document conversion quality.
    """
    global _converter, _model_loaded

    _patch_pil_crop()

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions, granite_picture_description
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    logger.info("Loading Docling converter with model=%s device=%s dtype=%s", MODEL_PATH, DEVICE, DTYPE)

    accel = AcceleratorOptions(
        device=DEVICE,
        cuda_use_flash_attention2=False,  # SDPA used instead — flash_attn requires bf16 but docling loads model in fp32
    )

    pipeline_options = VlmPipelineOptions(
        accelerator_options=accel,
        force_backend_text=True,
        generate_picture_images=True,
        images_scale=2.0,
        do_picture_description=True,
        picture_description_options=granite_picture_description,
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


def _convert_without_picture_images(tmp_path: str):
    """Retry conversion with generate_picture_images=False.

    Used as a fallback when the VLM produces malformed bounding boxes
    that crash PIL during image cropping.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    accel = AcceleratorOptions(
        device=DEVICE,
        cuda_use_flash_attention2=False,
    )
    opts = VlmPipelineOptions(
        accelerator_options=accel,
        force_backend_text=True,
        generate_picture_images=False,
        do_picture_description=False,
        images_scale=2.0,
    )
    fallback = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=opts,
            ),
            InputFormat.IMAGE: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=opts,
            ),
        },
    )
    return fallback.convert(source=tmp_path)


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
        try:
            result = _converter.convert(source=tmp_path)
        except (SystemError, RuntimeError) as crop_exc:
            # Workaround for docling-core bug: VLM predicts zero-area bounding
            # boxes → PIL crashes with "tile cannot extend outside image" when
            # generate_picture_images=True.  Retry without picture extraction.
            # See: https://github.com/docling-project/docling/issues/2763
            if "tile cannot extend outside image" in str(crop_exc) or "VlmPipeline failed" in str(crop_exc):
                logger.warning(
                    "Docling image crop failed for %s, retrying without generate_picture_images: %s",
                    filename, crop_exc,
                )
                result = _convert_without_picture_images(tmp_path)
            else:
                raise

        doc = result.document

        elements = _extract_elements(doc)
        markdown = doc.export_to_markdown()
        from app.cleanup import clean_markdown
        markdown = clean_markdown(markdown)
        document_json = doc.export_to_dict()
        num_pages = _count_pages(doc)

        elapsed_ms = (time.monotonic() - start) * 1000

        return ConvertResponse(
            status="ok",
            filename=filename,
            num_pages=num_pages,
            elements=elements,
            markdown=markdown,
            processing_time_ms=round(elapsed_ms, 1),
            document_json=document_json,
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.exception("Docling conversion failed for %s: %s", filename, exc)
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


def _get_item_text(item) -> str:
    """Get text from a Docling item, handling API differences across versions.

    docling-core >=2.60 removed export_to_markdown() from TextItem-based types
    (TextItem, SectionHeaderItem, TitleItem, FormulaItem) in favor of a .text field.
    TableItem and PictureItem still have export_to_markdown().
    """
    # Prefer .text attribute (new API)
    text = getattr(item, "text", None)
    if text is not None:
        return text
    # Fallback to export_to_markdown (TableItem, PictureItem, older versions)
    if hasattr(item, "export_to_markdown"):
        return item.export_to_markdown()
    return ""


def _extract_elements(doc) -> list[ConvertedElement]:
    """Iterate Docling document items and map to ConvertedElement list."""
    from docling_core.types.doc import (
        DocItemLabel,
        ImageRefMode,
    )

    elements: list[ConvertedElement] = []
    order_counter = 0
    # Track current section path for heading hierarchy
    section_stack: list[tuple[int, str]] = []  # (level, heading_text)

    for item, level in doc.iterate_items():
        label = getattr(item, "label", None)
        page_no = _get_page_number(item)

        if label in (
            DocItemLabel.SECTION_HEADER,
            DocItemLabel.TITLE,
            DocItemLabel.PAGE_HEADER,
            DocItemLabel.PAGE_FOOTER,
        ):
            text = _get_item_text(item)
            if text.strip():
                heading_level = level if level and level > 0 else 1
                if label == DocItemLabel.TITLE:
                    heading_level = 1

                # Update section stack
                while section_stack and section_stack[-1][0] >= heading_level:
                    section_stack.pop()
                section_stack.append((heading_level, text.strip()))
                section_path = " > ".join(s[1] for s in section_stack)

                uid = _make_element_uid(page_no, order_counter, "heading", text.strip())
                elements.append(
                    ConvertedElement(
                        element_type="heading",
                        content_text=text.strip(),
                        page_number=page_no,
                        element_uid=uid,
                        element_order=order_counter,
                        heading_level=heading_level,
                        section_path=section_path,
                    )
                )
                order_counter += 1

        elif label == DocItemLabel.TABLE:
            md_table = _get_item_text(item)
            section_path = " > ".join(s[1] for s in section_stack) if section_stack else None
            uid = _make_element_uid(page_no, order_counter, "table", md_table)
            elements.append(
                ConvertedElement(
                    element_type="table",
                    content_text=md_table,
                    page_number=page_no,
                    metadata={"label": "table"},
                    element_uid=uid,
                    element_order=order_counter,
                    section_path=section_path,
                )
            )
            order_counter += 1

        elif label == DocItemLabel.PICTURE:
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
                caption_text = (getattr(item, "caption_text", None) or _get_item_text(item)).strip()
            except Exception:
                pass

            section_path = " > ".join(s[1] for s in section_stack) if section_stack else None
            uid = _make_element_uid(page_no, order_counter, "image", caption_text)
            elements.append(
                ConvertedElement(
                    element_type="image",
                    content_text=caption_text,
                    page_number=page_no,
                    image_base64=image_b64,
                    metadata={"label": "picture", "ext": "png"},
                    element_uid=uid,
                    element_order=order_counter,
                    section_path=section_path,
                )
            )
            order_counter += 1

        elif label == DocItemLabel.FORMULA:
            text = _get_item_text(item)
            if text.strip():
                section_path = " > ".join(s[1] for s in section_stack) if section_stack else None
                uid = _make_element_uid(page_no, order_counter, "equation", text.strip())
                elements.append(
                    ConvertedElement(
                        element_type="equation",
                        content_text=text.strip(),
                        page_number=page_no,
                        metadata={"label": "formula"},
                        element_uid=uid,
                        element_order=order_counter,
                        section_path=section_path,
                    )
                )
                order_counter += 1

        else:
            text = _get_item_text(item)
            if text and text.strip():
                section_path = " > ".join(s[1] for s in section_stack) if section_stack else None
                uid = _make_element_uid(page_no, order_counter, "text", text.strip())
                elements.append(
                    ConvertedElement(
                        element_type="text",
                        content_text=text.strip(),
                        page_number=page_no,
                        element_uid=uid,
                        element_order=order_counter,
                        section_path=section_path,
                    )
                )
                order_counter += 1

    return elements


def _make_element_uid(page_number: int | None, order: int, element_type: str, content: str) -> str:
    """Generate a deterministic element UID from position and content."""
    import hashlib
    content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{page_number or 0}-{order}-{element_type}-{content_hash}"


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
