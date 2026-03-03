"""Feedback submission and patch governance endpoints."""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.models.governance import Feedback, Patch, PatchApproval, PatchEvent
from app.schemas.governance import (
    GRAPH_MUTATION_TYPES,
    FeedbackCreate,
    FeedbackResponse,
    FeedbackType,
    PatchApprovalCreate,
    PatchApprovalResponse,
    PatchResponse,
    PatchState,
)

router = APIRouter(tags=["governance"])


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def _is_graph_mutation(feedback_type: FeedbackType) -> bool:
    return feedback_type in GRAPH_MUTATION_TYPES


def _feedback_to_patch_payload(feedback: Feedback) -> tuple[str, dict]:
    """Translate feedback into a (patch_type, RFC 6902 payload) tuple."""
    ft = feedback.feedback_type
    proposed = feedback.proposed_value or {}

    if ft == FeedbackType.wrong_text.value:
        return "chunk_text_correction", {
            "target_table": "retrieval.chunks",
            "operations": [
                {"op": "replace", "path": "/chunk_text", "value": proposed.get("text", "")}
            ],
        }
    elif ft == FeedbackType.wrong_classification.value:
        return "classification_correction", {
            "target_table": "retrieval.chunks",
            "operations": [
                {
                    "op": "replace",
                    "path": "/classification",
                    "value": proposed.get("classification", "UNCLASSIFIED"),
                }
            ],
        }
    elif ft == FeedbackType.incorrect_entity.value:
        return "entity_update", {
            "target_table": "age_graph.nodes",
            "operations": [
                {"op": "replace", "path": "/properties", "value": proposed}
            ],
        }
    elif ft == FeedbackType.missing_relationship.value:
        return "relationship_add", {
            "target_table": "age_graph.edges",
            "operations": [{"op": "add", "path": "/", "value": proposed}],
        }
    elif ft == FeedbackType.missing_entity.value:
        return "entity_add", {
            "target_table": "age_graph.nodes",
            "operations": [{"op": "add", "path": "/", "value": proposed}],
        }
    elif ft == FeedbackType.delete_entity.value:
        return "entity_delete", {
            "target_table": "age_graph.nodes",
            "operations": [{"op": "remove", "path": "/"}],
        }
    elif ft == FeedbackType.merge_entity.value:
        return "entity_merge", {
            "target_table": "age_graph.nodes",
            "operations": [{"op": "merge", "path": "/", "value": proposed}],
        }
    else:
        return "unknown", {"operations": []}


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_feedback(
    body: FeedbackCreate,
    db: AsyncSession = Depends(get_async_session),
) -> FeedbackResponse:
    """Submit feedback on a retrieval result. Automatically creates a patch."""
    system_user = uuid.UUID("00000000-0000-0000-0000-000000000001")

    feedback = Feedback(
        query_text=body.query_text,
        chunk_id=body.chunk_id,
        artifact_id=body.artifact_id,
        feedback_type=body.feedback_type.value,
        proposed_value=body.proposed_value,
        notes=body.notes,
        submitted_by=system_user,
    )
    db.add(feedback)
    await db.flush()

    # Auto-generate patch
    patch_type, patch_payload = _feedback_to_patch_payload(feedback)
    requires_dual = _is_graph_mutation(body.feedback_type)

    patch = Patch(
        source_feedback_id=feedback.id,
        patch_type=patch_type,
        state=PatchState.under_review.value,
        requires_dual_approval=requires_dual,
        target_table=patch_payload.get("target_table", ""),
        target_id=body.chunk_id or body.artifact_id,
        patch_payload=patch_payload,
        created_by=system_user,
    )
    db.add(patch)
    await db.flush()

    # Record creation event
    event = PatchEvent(
        patch_id=patch.id,
        event_type="CREATED",
        actor_id=system_user,
        metadata={"feedback_type": feedback.feedback_type},
    )
    db.add(event)

    return FeedbackResponse.model_validate(feedback)


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


@router.get("/patches", response_model=list[PatchResponse])
async def list_patches(
    state: Optional[str] = None,
    db: AsyncSession = Depends(get_async_session),
) -> list[PatchResponse]:
    """List patches, optionally filtered by state."""
    query = select(Patch).order_by(Patch.created_at.desc())
    if state:
        query = query.where(Patch.state == state.upper())

    result = await db.execute(query)
    patches = result.scalars().all()
    return [PatchResponse.model_validate(p) for p in patches]


@router.get("/patches/{patch_id}", response_model=PatchResponse)
async def get_patch(
    patch_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> PatchResponse:
    """Get a patch by ID."""
    patch = await db.get(Patch, patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail="Patch not found.")
    return PatchResponse.model_validate(patch)


@router.post("/patches/{patch_id}/approve", response_model=PatchResponse)
async def approve_patch(
    patch_id: uuid.UUID,
    body: PatchApprovalCreate,
    db: AsyncSession = Depends(get_async_session),
) -> PatchResponse:
    """Curator approves a patch. For graph mutations, a second curator must approve."""
    # TODO(Phase 3): use actual authenticated curator ID
    curator_id = uuid.UUID("00000000-0000-0000-0000-000000000002")

    # Acquire advisory lock to prevent race conditions
    lock_key = patch_id.int & 0x7FFFFFFFFFFFFFFF
    await db.execute(text(f"SELECT pg_advisory_xact_lock({lock_key})"))

    patch = await db.get(Patch, patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail="Patch not found.")

    if patch.state not in (PatchState.under_review.value, PatchState.approved.value):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patch in state '{patch.state}' cannot be approved.",
        )

    # Check for self-approval (same curator approving twice)
    existing_approvals = await db.execute(
        select(PatchApproval).where(PatchApproval.patch_id == patch_id)
    )
    approvals = existing_approvals.scalars().all()
    if any(a.curator_id == curator_id for a in approvals):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Same curator cannot provide both approvals.",
        )

    approval = PatchApproval(
        patch_id=patch_id,
        curator_id=curator_id,
        decision="approved",
        notes=body.notes,
    )
    db.add(approval)

    # Transition state
    if patch.requires_dual_approval and patch.state == PatchState.approved.value:
        patch.state = PatchState.dual_approved.value
    else:
        patch.state = PatchState.approved.value

    event = PatchEvent(
        patch_id=patch_id,
        event_type="APPROVED",
        actor_id=curator_id,
        metadata={"new_state": patch.state},
    )
    db.add(event)
    await db.flush()

    return PatchResponse.model_validate(patch)


@router.post("/patches/{patch_id}/reject", response_model=PatchResponse)
async def reject_patch(
    patch_id: uuid.UUID,
    body: PatchApprovalCreate,
    db: AsyncSession = Depends(get_async_session),
) -> PatchResponse:
    """Curator rejects a patch."""
    curator_id = uuid.UUID("00000000-0000-0000-0000-000000000002")

    patch = await db.get(Patch, patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail="Patch not found.")

    if patch.state not in (PatchState.under_review.value, PatchState.approved.value):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patch in state '{patch.state}' cannot be rejected.",
        )

    approval = PatchApproval(
        patch_id=patch_id,
        curator_id=curator_id,
        decision="rejected",
        notes=body.notes,
    )
    db.add(approval)
    patch.state = PatchState.rejected.value

    event = PatchEvent(
        patch_id=patch_id,
        event_type="REJECTED",
        actor_id=curator_id,
    )
    db.add(event)
    await db.flush()

    return PatchResponse.model_validate(patch)


@router.post("/patches/{patch_id}/apply", response_model=PatchResponse)
async def apply_patch(
    patch_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> PatchResponse:
    """Apply an approved patch to the data platform."""
    system_user = uuid.UUID("00000000-0000-0000-0000-000000000001")

    lock_key = patch_id.int & 0x7FFFFFFFFFFFFFFF
    await db.execute(text(f"SELECT pg_advisory_xact_lock({lock_key})"))

    patch = await db.get(Patch, patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail="Patch not found.")

    required_state = (
        PatchState.dual_approved.value
        if patch.requires_dual_approval
        else PatchState.approved.value
    )
    if patch.state != required_state:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patch must be in state '{required_state}' to apply. Current: '{patch.state}'.",
        )

    # Apply the patch payload
    try:
        await _apply_patch_payload(db, patch)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Patch application failed: {e}",
        )

    patch.state = PatchState.applied.value
    event = PatchEvent(
        patch_id=patch_id,
        event_type="APPLIED",
        actor_id=system_user,
    )
    db.add(event)
    await db.flush()

    return PatchResponse.model_validate(patch)


async def _apply_patch_payload(db: AsyncSession, patch: Patch) -> None:
    """Apply RFC 6902 JSON Patch operations to the target record."""
    import jsonpatch

    operations = patch.patch_payload.get("operations", [])
    target_table = patch.patch_table if hasattr(patch, "patch_table") else patch.target_table

    if target_table == "retrieval.chunks" and patch.target_id:
        from app.models.retrieval import Chunk

        chunk = await db.get(Chunk, patch.target_id)
        if not chunk:
            raise ValueError(f"Chunk {patch.target_id} not found.")

        # Snapshot current state
        patch.previous_snapshot = {
            "chunk_text": chunk.chunk_text,
            "classification": chunk.classification,
        }

        # Apply JSON Patch to snapshot
        patched = jsonpatch.apply_patch(patch.previous_snapshot, operations)
        if "chunk_text" in patched:
            chunk.chunk_text = patched["chunk_text"]
            # Re-embed on text change
            from app.services.embedding import embed_query
            chunk.embedding = embed_query(chunk.chunk_text)
        if "classification" in patched:
            chunk.classification = patched["classification"]

    elif "age_graph" in target_table:
        # Phase 2: implement AGE graph mutation via Cypher
        pass
