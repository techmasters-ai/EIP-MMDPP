"""Pydantic schemas for sources, documents, artifacts, and watch directories."""

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from app.schemas.common import APIModel


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class SourceCreate(APIModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    # Task 3.7 + spec §7.4 bundle threading. Purely additive. Both default
    # to None so existing callers that don't know about bundles keep
    # working unchanged. Chunk 4 Task 4.2 wires these into
    # start_ingest_pipeline's resolve_bundle_key precedence as the
    # source-default tier.
    default_ontology_bundle_key: Optional[str] = None
    default_use_case_key: Optional[str] = None


class SourceResponse(APIModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    # Task 3.7: exposed on the response so UIs can display which bundle
    # a source defaults to.
    default_ontology_bundle_key: Optional[str] = None
    default_use_case_key: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Reingest request (Task 3.7)
# ---------------------------------------------------------------------------

class ReingestRequest(APIModel):
    """Body for ``POST /documents/{document_id}/reingest``.

    All fields optional. Introduced in PR 2 (Task 3.7) to thread bundle
    overrides through the reingest path. Previously the route accepted
    ``body: dict = None`` which made the shape opaque. Existing clients
    that send ``{"mode": "full"}`` keep working because every field has
    a sensible default.

    The ``ontology_bundle_key`` and ``use_case_key`` fields are accepted
    by the route but not yet forwarded into ``start_ingest_pipeline`` —
    that wiring lands in Chunk 4 Task 4.2. Accepting the fields now
    (without acting on them) keeps the API surface forward-compatible.
    """

    mode: Literal["full", "embeddings_only", "graph_only"] = "full"
    ontology_bundle_key: Optional[str] = None
    use_case_key: Optional[str] = None


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


class BatchStatusRequest(APIModel):
    document_ids: list[uuid.UUID] = Field(..., max_length=500)


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
