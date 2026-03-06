"""Pydantic schemas for retrieval (query) endpoints."""

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import Field, model_validator

from app.schemas.common import APIModel


class QueryMode(str, Enum):
    text_basic = "text_basic"
    text_only = "text_only"
    images_only = "images_only"
    multi_modal = "multi_modal"
    memory = "memory"


class QueryFilters(APIModel):
    classification: Optional[str] = None
    modalities: Optional[list[str]] = None  # text, image, table, schematic
    source_ids: Optional[list[uuid.UUID]] = None
    document_ids: Optional[list[uuid.UUID]] = None


class QueryResultItem(APIModel):
    chunk_id: Optional[uuid.UUID] = None
    artifact_id: Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    score: float
    modality: str  # text | image | table | schematic | graph_node
    content_text: Optional[str] = None
    page_number: Optional[int] = None
    classification: str = "UNCLASSIFIED"
    context: Optional[dict[str, Any]] = None  # graph neighbors, source info, etc.
    image_url: Optional[str] = None  # presigned MinIO URL for image-modality results


class UnifiedQueryRequest(APIModel):
    query_text: Optional[str] = Field(None, max_length=4096)
    query_image: Optional[str] = None  # base64-encoded image or artifact reference
    mode: QueryMode = Field(default=QueryMode.text_basic)
    filters: Optional[QueryFilters] = None
    top_k: int = Field(default=10, ge=1, le=100)
    include_context: bool = True

    @model_validator(mode="after")
    def require_at_least_one_query(self):
        if not self.query_text and not self.query_image:
            raise ValueError("At least one of query_text or query_image is required")
        return self


class UnifiedQueryResponse(APIModel):
    query_text: Optional[str] = None
    query_image: Optional[str] = None
    mode: str
    results: list[QueryResultItem]
    total: int


# Legacy aliases for backwards compatibility
class QueryRequest(APIModel):
    query: str = Field(..., min_length=1, max_length=4096)
    mode: QueryMode = QueryMode.text_basic
    filters: Optional[QueryFilters] = None
    top_k: int = Field(default=10, ge=1, le=100)
    include_context: bool = True


class QueryResponse(APIModel):
    query: str
    mode: QueryMode
    results: list[QueryResultItem]
    total_results: int
