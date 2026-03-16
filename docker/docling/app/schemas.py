"""Pydantic request/response schemas for the Docling conversion service."""

from __future__ import annotations

from pydantic import BaseModel


class ConvertedElement(BaseModel):
    """A single extracted element from the document."""

    element_type: str  # text | table | image | equation | heading
    content_text: str
    page_number: int | None = None
    bounding_box: dict | None = None
    confidence: float | None = None
    metadata: dict = {}
    image_base64: str | None = None
    # Structural metadata (v2)
    element_uid: str | None = None  # deterministic ID
    element_order: int | None = None  # sequential position in document
    heading_level: int | None = None  # 1-6 for headings
    section_path: str | None = None  # accumulated heading path


class ConvertResponse(BaseModel):
    """Full document conversion result."""

    status: str  # ok | error
    filename: str
    num_pages: int
    elements: list[ConvertedElement]
    markdown: str
    processing_time_ms: float
    document_json: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str
    device: str
