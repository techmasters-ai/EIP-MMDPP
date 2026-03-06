"""Docling document conversion service client.

Calls the dedicated Docling Docker service for document conversion.
Maps responses to ExtractedChunk for compatibility with the existing pipeline.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

import httpx

from app.config import get_settings
from app.services.extraction import ExtractedChunk

logger = logging.getLogger(__name__)

# Docling element_type -> ExtractedChunk modality
_MODALITY_MAP = {
    "text": "text",
    "heading": "text",
    "table": "table",
    "image": "image",
    "equation": "text",
    "schematic": "schematic",
}


@dataclass
class DoclingConversionResult:
    """Result from the Docling service."""

    elements: list[ExtractedChunk]
    markdown: str
    num_pages: int
    processing_time_ms: float


def convert_document_sync(
    file_bytes: bytes,
    filename: str,
    timeout: float | None = None,
) -> DoclingConversionResult:
    """Send a document to the Docling service for conversion.

    Maps the Docling response elements to ExtractedChunk instances for
    compatibility with the existing pipeline (chunk_and_embed, NER, graph).
    """
    settings = get_settings()
    url = f"{settings.docling_service_url}/convert"
    if timeout is None:
        timeout = settings.docling_timeout_seconds

    response = httpx.post(
        url,
        files={"file": (filename, file_bytes)},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    if data.get("status") == "error":
        raise RuntimeError(f"Docling conversion failed: {data.get('error', 'unknown')}")

    chunks = _map_elements_to_chunks(data.get("elements", []))
    return DoclingConversionResult(
        elements=chunks,
        markdown=data.get("markdown", ""),
        num_pages=data.get("num_pages", 0),
        processing_time_ms=data.get("processing_time_ms", 0),
    )


def check_health_sync() -> bool:
    """Check if the Docling service is healthy and model is loaded."""
    settings = get_settings()
    try:
        resp = httpx.get(
            f"{settings.docling_service_url}/health",
            timeout=5.0,
        )
        return resp.status_code == 200 and resp.json().get("model_loaded", False)
    except Exception:
        return False


def _map_elements_to_chunks(elements: list[dict]) -> list[ExtractedChunk]:
    """Map Docling ConvertedElement dicts to ExtractedChunk instances."""
    chunks = []
    for elem in elements:
        element_type = elem.get("element_type", "text")
        modality = _MODALITY_MAP.get(element_type, "text")

        raw_image_bytes = None
        if elem.get("image_base64"):
            raw_image_bytes = base64.b64decode(elem["image_base64"])

        # Preserve structural metadata from v2 Docling output
        meta = dict(elem.get("metadata", {}))
        if elem.get("element_uid"):
            meta["element_uid"] = elem["element_uid"]
        if elem.get("element_order") is not None:
            meta["element_order"] = elem["element_order"]
        if elem.get("heading_level") is not None:
            meta["heading_level"] = elem["heading_level"]
        if elem.get("section_path"):
            meta["section_path"] = elem["section_path"]

        chunk = ExtractedChunk(
            chunk_text=elem.get("content_text", ""),
            modality=modality,
            page_number=elem.get("page_number"),
            bounding_box=elem.get("bounding_box"),
            raw_image_bytes=raw_image_bytes,
            ocr_confidence=elem.get("confidence"),
            ocr_engine="docling-granite" if elem.get("confidence") else None,
            requires_human_review=False,
            metadata=meta,
        )
        chunks.append(chunk)

    return chunks
