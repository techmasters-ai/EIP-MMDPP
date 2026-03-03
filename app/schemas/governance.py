"""Pydantic schemas for feedback and patch governance endpoints."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import Field

from app.schemas.common import APIModel


class FeedbackType(str, Enum):
    wrong_text = "WRONG_TEXT"
    wrong_classification = "WRONG_CLASSIFICATION"
    incorrect_entity = "INCORRECT_ENTITY"
    missing_relationship = "MISSING_RELATIONSHIP"
    missing_entity = "MISSING_ENTITY"
    delete_entity = "DELETE_ENTITY"
    merge_entity = "MERGE_ENTITY"


# Graph mutations requiring dual-curator approval
GRAPH_MUTATION_TYPES = {
    FeedbackType.incorrect_entity,
    FeedbackType.missing_relationship,
    FeedbackType.missing_entity,
    FeedbackType.delete_entity,
    FeedbackType.merge_entity,
}


class FeedbackCreate(APIModel):
    query_text: Optional[str] = None
    chunk_id: Optional[uuid.UUID] = None
    artifact_id: Optional[uuid.UUID] = None
    feedback_type: FeedbackType
    proposed_value: Optional[dict[str, Any]] = None
    notes: Optional[str] = None


class FeedbackResponse(APIModel):
    id: uuid.UUID
    feedback_type: str
    query_text: Optional[str]
    chunk_id: Optional[uuid.UUID]
    artifact_id: Optional[uuid.UUID]
    proposed_value: Optional[dict]
    notes: Optional[str]
    submitted_by: uuid.UUID
    created_at: datetime


class PatchState(str, Enum):
    draft = "DRAFT"
    under_review = "UNDER_REVIEW"
    approved = "APPROVED"
    dual_approved = "DUAL_APPROVED"
    rejected = "REJECTED"
    applied = "APPLIED"
    reverted = "REVERTED"


class PatchResponse(APIModel):
    id: uuid.UUID
    source_feedback_id: uuid.UUID
    patch_type: str
    state: str
    requires_dual_approval: bool
    target_table: str
    target_id: Optional[uuid.UUID]
    patch_payload: dict
    previous_snapshot: Optional[dict]
    created_by: uuid.UUID
    created_at: datetime
    updated_at: datetime


class PatchApprovalCreate(APIModel):
    notes: Optional[str] = None


class PatchApprovalResponse(APIModel):
    id: uuid.UUID
    patch_id: uuid.UUID
    curator_id: uuid.UUID
    decision: str
    notes: Optional[str]
    decided_at: datetime
