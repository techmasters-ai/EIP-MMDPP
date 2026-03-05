"""Pydantic schemas for image vector store ingest/query endpoints."""

import uuid
from typing import Optional

from pydantic import Field

from app.schemas.common import APIModel
from app.schemas.retrieval import QueryFilters


class ImageChunkIngest(APIModel):
    image: str  # base64-encoded image data
    content_type: str = Field(default="image/png", pattern="^image/")
    source_id: Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    alt_text: Optional[str] = None
    page_number: Optional[int] = None
    classification: str = "UNCLASSIFIED"


class ImageChunkIngestResponse(APIModel):
    chunk_id: uuid.UUID
    chunks_created: int = 1


class ImageQueryRequest(APIModel):
    query_text: Optional[str] = Field(None, max_length=4096)
    query_image: Optional[str] = None  # base64-encoded image
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[QueryFilters] = None
