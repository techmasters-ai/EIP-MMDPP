"""Cognee memory layer — governed knowledge proposal, approval, and query endpoints."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.models.memory import MemoryProposal
from app.schemas.memory import (
    MemoryProposalCreate,
    MemoryProposalResponse,
    MemoryProposalReview,
    MemoryQueryRequest,
    MemoryQueryResponse,
)

router = APIRouter(tags=["memory"])
logger = logging.getLogger(__name__)

# Placeholder user ID until JWT auth is implemented
_PLACEHOLDER_USER = uuid.UUID("00000000-0000-0000-0000-000000000001")


@router.post("/memory/ingest", response_model=MemoryProposalResponse, status_code=201)
async def propose_memory(
    body: MemoryProposalCreate,
    db: AsyncSession = Depends(get_async_session),
) -> MemoryProposalResponse:
    """Agent proposes knowledge for the governed memory layer.

    Creates a proposal with status PROPOSED. A curator must approve before
    the knowledge is written to Cognee.
    """
    proposal = MemoryProposal(
        content=body.content,
        source_context=body.source_context,
        proposed_by=_PLACEHOLDER_USER,
        confidence=body.confidence,
        status="PROPOSED",
    )
    db.add(proposal)
    await db.commit()
    await db.refresh(proposal)

    return _to_response(proposal)


@router.get("/memory/proposals", response_model=list[MemoryProposalResponse])
async def list_proposals(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_async_session),
) -> list[MemoryProposalResponse]:
    """List memory proposals, optionally filtered by status."""
    query = select(MemoryProposal).order_by(MemoryProposal.created_at.desc()).limit(limit)
    if status:
        query = query.where(MemoryProposal.status == status.upper())

    result = await db.execute(query)
    proposals = result.scalars().all()
    return [_to_response(p) for p in proposals]


@router.post(
    "/memory/proposals/{proposal_id}/approve",
    response_model=MemoryProposalResponse,
)
async def approve_proposal(
    proposal_id: uuid.UUID,
    body: MemoryProposalReview,
    db: AsyncSession = Depends(get_async_session),
) -> MemoryProposalResponse:
    """Curator approves a memory proposal → writes content to Cognee."""
    proposal = await db.get(MemoryProposal, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    if proposal.status != "PROPOSED":
        raise HTTPException(
            status_code=409,
            detail=f"Proposal is already {proposal.status}",
        )

    # Update proposal status
    proposal.status = "APPROVED"
    proposal.reviewed_by = _PLACEHOLDER_USER
    proposal.review_notes = body.notes
    proposal.reviewed_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(proposal)

    # Write to Cognee (non-blocking — failures don't revert approval)
    try:
        from app.services.cognee_service import cognee_add, cognee_cognify

        await cognee_add(proposal.content, "approved_memory")
        await cognee_cognify("approved_memory")
        logger.info("Memory proposal %s approved and written to Cognee", proposal_id)
    except Exception as exc:
        logger.warning(
            "Cognee write failed for approved proposal %s: %s", proposal_id, exc
        )

    return _to_response(proposal)


@router.post(
    "/memory/proposals/{proposal_id}/reject",
    response_model=MemoryProposalResponse,
)
async def reject_proposal(
    proposal_id: uuid.UUID,
    body: MemoryProposalReview,
    db: AsyncSession = Depends(get_async_session),
) -> MemoryProposalResponse:
    """Curator rejects a memory proposal — nothing is written to Cognee."""
    proposal = await db.get(MemoryProposal, proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    if proposal.status != "PROPOSED":
        raise HTTPException(
            status_code=409,
            detail=f"Proposal is already {proposal.status}",
        )

    proposal.status = "REJECTED"
    proposal.reviewed_by = _PLACEHOLDER_USER
    proposal.review_notes = body.notes
    proposal.reviewed_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(proposal)

    return _to_response(proposal)


@router.post("/memory/query", response_model=MemoryQueryResponse)
async def query_memory(
    body: MemoryQueryRequest,
    db: AsyncSession = Depends(get_async_session),
) -> MemoryQueryResponse:
    """Search Cognee approved memory."""
    from app.services.cognee_service import cognee_search

    results = await cognee_search(body.query, body.top_k)

    return MemoryQueryResponse(
        query=body.query,
        results=[
            {
                "content_text": r.content_text,
                "score": r.score,
                "modality": r.modality,
            }
            for r in results
        ],
        total=len(results),
    )


def _to_response(proposal: MemoryProposal) -> MemoryProposalResponse:
    return MemoryProposalResponse(
        id=proposal.id,
        content=proposal.content,
        source_context=proposal.source_context,
        proposed_by=proposal.proposed_by,
        confidence=proposal.confidence,
        status=proposal.status,
        reviewed_by=proposal.reviewed_by,
        review_notes=proposal.review_notes,
        reviewed_at=proposal.reviewed_at,
        created_at=proposal.created_at,
        updated_at=proposal.updated_at,
    )
