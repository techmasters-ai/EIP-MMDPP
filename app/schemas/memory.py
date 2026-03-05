"""Pydantic schemas for the Cognee memory layer endpoints."""

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from app.schemas.common import APIModel


class MemoryProposalCreate(APIModel):
    content: str = Field(..., min_length=1, max_length=50000)
    source_context: Optional[dict[str, Any]] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryProposalResponse(APIModel):
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


class MemoryProposalReview(APIModel):
    notes: Optional[str] = None


class MemoryQueryRequest(APIModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=10, ge=1, le=50)


class MemoryQueryResponse(APIModel):
    query: str
    results: list[dict[str, Any]]
    total: int
