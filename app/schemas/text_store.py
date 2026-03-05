"""Pydantic schemas for text vector store ingest/query endpoints."""

import uuid
from typing import Optional

from pydantic import Field

from app.schemas.common import APIModel
from app.schemas.retrieval import QueryFilters


class TextChunkIngest(APIModel):
    text: str = Field(..., min_length=1, max_length=50000)
    source_id: Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    modality: str = Field(default="text", pattern="^(text|table|schematic)$")
    page_number: Optional[int] = None
    classification: str = "UNCLASSIFIED"
    metadata: Optional[dict] = None


class TextChunkIngestResponse(APIModel):
    chunk_ids: list[uuid.UUID]
    chunks_created: int


class TextQueryRequest(APIModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[QueryFilters] = None
    include_context: bool = True
