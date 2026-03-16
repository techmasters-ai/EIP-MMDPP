"""Pydantic schemas for retrieval (query) endpoints."""

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import Field, model_validator

from app.schemas.common import APIModel


class QueryStrategy(str, Enum):
    basic = "basic"
    hybrid = "hybrid"
    graphrag_local = "graphrag_local"
    graphrag_global = "graphrag_global"


class ModalityFilter(str, Enum):
    all = "all"
    text = "text"
    image = "image"


# Backward-compat mapping from legacy mode strings to (strategy, modality_filter)
_MODE_MAP: dict[str, tuple[QueryStrategy, ModalityFilter]] = {
    "text_basic": (QueryStrategy.basic, ModalityFilter.all),
    "text_only": (QueryStrategy.hybrid, ModalityFilter.text),
    "images_only": (QueryStrategy.hybrid, ModalityFilter.image),
    "multi_modal": (QueryStrategy.hybrid, ModalityFilter.all),
    "graphrag_local": (QueryStrategy.graphrag_local, ModalityFilter.all),
    "graphrag_global": (QueryStrategy.graphrag_global, ModalityFilter.all),
    # Also accept new strategy names directly as mode
    "basic": (QueryStrategy.basic, ModalityFilter.all),
    "hybrid": (QueryStrategy.hybrid, ModalityFilter.all),
}


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
    strategy: QueryStrategy = Field(default=QueryStrategy.basic)
    modality_filter: ModalityFilter = Field(default=ModalityFilter.all)
    mode: Optional[str] = Field(None, description="Deprecated: use strategy + modality_filter")
    filters: Optional[QueryFilters] = None
    top_k: int = Field(default=10, ge=1, le=100)
    include_context: bool = True

    @model_validator(mode="after")
    def resolve_legacy_mode(self):
        """Map legacy 'mode' field to strategy + modality_filter."""
        if self.mode:
            if self.mode not in _MODE_MAP:
                raise ValueError(
                    f"Unknown mode: '{self.mode}'. "
                    f"Valid modes: {', '.join(sorted(_MODE_MAP.keys()))}"
                )
            mapped_strategy, mapped_modality = _MODE_MAP[self.mode]
            # Only apply mapping if strategy/modality_filter weren't explicitly set
            # (i.e. still at defaults). This lets new clients override.
            if self.strategy == QueryStrategy.basic and self.modality_filter == ModalityFilter.all:
                self.strategy = mapped_strategy
                self.modality_filter = mapped_modality
        return self

    @model_validator(mode="after")
    def require_at_least_one_query(self):
        if not self.query_text and not self.query_image:
            raise ValueError("At least one of query_text or query_image is required")
        return self


class UnifiedQueryResponse(APIModel):
    query_text: Optional[str] = None
    query_image: Optional[str] = None
    strategy: str
    modality_filter: str
    results: list[QueryResultItem]
    total: int


class DoclingImageRef(APIModel):
    element_uid: str
    url: str


class DoclingDocumentResponse(APIModel):
    document_id: str
    filename: str
    markdown: str
    document_json: dict[str, Any]
    images: list[DoclingImageRef] = []
