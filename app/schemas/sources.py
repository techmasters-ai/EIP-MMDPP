"""Pydantic schemas for sources, documents, artifacts, and watch directories."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import Field

from app.schemas.common import APIModel


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class SourceCreate(APIModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class SourceResponse(APIModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

class DocumentResponse(APIModel):
    id: uuid.UUID
    source_id: uuid.UUID
    filename: str
    mime_type: Optional[str]
    file_size_bytes: Optional[int]
    pipeline_status: str
    pipeline_stage: Optional[str]
    failed_stages: Optional[list[str]]
    uploaded_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


class DocumentStatusResponse(APIModel):
    id: uuid.UUID
    filename: str
    pipeline_status: str
    pipeline_stage: Optional[str]
    failed_stages: Optional[list[str]]
    error_message: Optional[str]
    celery_task_id: Optional[str]
    updated_at: datetime
    # V2 optional fields
    pipeline_version: Optional[str] = None
    current_run_id: Optional[uuid.UUID] = None
    stage_summary: Optional[list[dict]] = None


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

class ArtifactResponse(APIModel):
    id: uuid.UUID
    document_id: uuid.UUID
    artifact_type: str
    content_text: Optional[str]
    content_metadata: Optional[dict]
    page_number: Optional[int]
    ocr_confidence: Optional[float]
    ocr_engine: Optional[str]
    requires_human_review: bool
    classification: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Watch Directories
# ---------------------------------------------------------------------------

class WatchDirCreate(APIModel):
    source_id: uuid.UUID
    path: str = Field(..., min_length=1)
    poll_interval_seconds: int = Field(default=30, ge=5, le=3600)
    file_patterns: list[str] = Field(
        default=["*.pdf", "*.docx", "*.txt", "*.png", "*.jpg", "*.tiff"]
    )


class WatchDirResponse(APIModel):
    id: uuid.UUID
    source_id: uuid.UUID
    path: str
    enabled: bool
    poll_interval_seconds: int
    file_patterns: list[str]
    created_at: datetime
    updated_at: datetime
