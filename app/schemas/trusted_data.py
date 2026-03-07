"""Pydantic schemas for the trusted data endpoints."""

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from app.schemas.common import APIModel


class TrustedDataCreate(APIModel):
    content: str = Field(..., min_length=1, max_length=50000)
    source_context: Optional[dict[str, Any]] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TrustedDataResponse(APIModel):
    id: uuid.UUID
    content: str
    source_context: Optional[dict[str, Any]] = None
    proposed_by: uuid.UUID
    confidence: float
    status: str
    reviewed_by: Optional[uuid.UUID] = None
    review_notes: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    # Indexing lifecycle
    index_status: Optional[str] = None
    index_error: Optional[str] = None
    qdrant_point_id: Optional[uuid.UUID] = None
    embedding_model: Optional[str] = None
    embedded_at: Optional[datetime] = None


class TrustedDataReview(APIModel):
    notes: Optional[str] = None


class TrustedDataQueryRequest(APIModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=10, ge=1, le=50)


class TrustedDataQueryResult(APIModel):
    content_text: str
    score: float
    submission_id: Optional[str] = None
    confidence: Optional[float] = None
    classification: Optional[str] = None


class TrustedDataQueryResponse(APIModel):
    query: str
    results: list[TrustedDataQueryResult]
    total: int
