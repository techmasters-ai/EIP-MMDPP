"""Pydantic schemas for retrieval (query) endpoints."""

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import Field

from app.schemas.common import APIModel


class QueryMode(str, Enum):
    semantic = "semantic"
    graph = "graph"
    hybrid = "hybrid"
    cross_modal = "cross_modal"
    cognee_graph = "cognee_graph"


class QueryFilters(APIModel):
    classification: Optional[str] = None
    modalities: Optional[list[str]] = None  # text, image, table, schematic
    source_ids: Optional[list[uuid.UUID]] = None
    document_ids: Optional[list[uuid.UUID]] = None


class QueryRequest(APIModel):
    query: str = Field(..., min_length=1, max_length=4096)
    mode: QueryMode = QueryMode.semantic
    filters: Optional[QueryFilters] = None
    top_k: int = Field(default=10, ge=1, le=100)
    include_context: bool = True


class QueryResultItem(APIModel):
    chunk_id: Optional[uuid.UUID] = None
    artifact_id: Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    score: float
    modality: str  # text | image | table | schematic | graph_node
    content_text: Optional[str]
    page_number: Optional[int]
    classification: str
    context: Optional[dict[str, Any]] = None  # graph neighbors, source info, etc.


class QueryResponse(APIModel):
    query: str
    mode: QueryMode
    results: list[QueryResultItem]
    total_results: int
