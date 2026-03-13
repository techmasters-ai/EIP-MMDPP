"""Trusted data — governed knowledge proposal, approval, indexing, and query endpoints."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.models.trusted_data import TrustedDataSubmission
from app.schemas.trusted_data import (
    TrustedDataCreate,
    TrustedDataResponse,
    TrustedDataReview,
    TrustedDataQueryRequest,
    TrustedDataQueryResponse,
    TrustedDataQueryResult,
)

router = APIRouter(tags=["trusted-data"])
logger = logging.getLogger(__name__)

# Placeholder user ID until JWT auth is implemented
_PLACEHOLDER_USER = uuid.UUID("00000000-0000-0000-0000-000000000001")


@router.post("/trusted-data/ingest", response_model=TrustedDataResponse, status_code=201)
async def propose_trusted_data(
    body: TrustedDataCreate,
    db: AsyncSession = Depends(get_async_session),
) -> TrustedDataResponse:
    """Submit knowledge for the trusted data layer.

    Creates a submission with status PROPOSED. A curator must approve before
    the content is embedded and indexed in the trusted Qdrant collection.
    """
    submission = TrustedDataSubmission(
        content=body.content,
        source_context=body.source_context,
        proposed_by=_PLACEHOLDER_USER,
        confidence=body.confidence,
        status="PROPOSED",
    )
    db.add(submission)
    await db.commit()
    await db.refresh(submission)

    return _to_response(submission)


@router.get("/trusted-data/proposals", response_model=list[TrustedDataResponse])
async def list_proposals(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_async_session),
) -> list[TrustedDataResponse]:
    """List trusted data submissions, optionally filtered by status."""
    query = (
        select(TrustedDataSubmission)
        .order_by(TrustedDataSubmission.created_at.desc())
        .limit(limit)
    )
    if status:
        query = query.where(TrustedDataSubmission.status == status.upper())

    result = await db.execute(query)
    submissions = result.scalars().all()
    return [_to_response(s) for s in submissions]


@router.post(
    "/trusted-data/proposals/{proposal_id}/approve",
    response_model=TrustedDataResponse,
)
async def approve_proposal(
    proposal_id: uuid.UUID,
    body: TrustedDataReview,
    db: AsyncSession = Depends(get_async_session),
) -> TrustedDataResponse:
    """Curator approves a submission → enqueues Celery task for embedding + Qdrant indexing."""
    submission = await db.get(TrustedDataSubmission, proposal_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    if submission.status != "PROPOSED":
        raise HTTPException(
            status_code=409,
            detail=f"Submission is already {submission.status}",
        )

    submission.status = "APPROVED_PENDING_INDEX"
    submission.reviewed_by = _PLACEHOLDER_USER
    submission.review_notes = body.notes
    submission.reviewed_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(submission)

    # Enqueue Celery indexing task
    try:
        from app.workers.trusted_data_tasks import index_trusted_submission

        index_trusted_submission.delay(str(submission.id))
        logger.info("Submission %s approved — indexing task enqueued", proposal_id)
    except Exception as exc:
        logger.warning(
            "Failed to enqueue indexing task for %s: %s", proposal_id, exc
        )
        submission.index_status = "INDEX_FAILED"
        submission.index_error = f"Failed to enqueue: {exc}"
        await db.commit()
        await db.refresh(submission)

    return _to_response(submission)


@router.post(
    "/trusted-data/proposals/{proposal_id}/reject",
    response_model=TrustedDataResponse,
)
async def reject_proposal(
    proposal_id: uuid.UUID,
    body: TrustedDataReview,
    db: AsyncSession = Depends(get_async_session),
) -> TrustedDataResponse:
    """Curator rejects a submission — nothing is indexed."""
    submission = await db.get(TrustedDataSubmission, proposal_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    if submission.status != "PROPOSED":
        raise HTTPException(
            status_code=409,
            detail=f"Submission is already {submission.status}",
        )

    submission.status = "REJECTED"
    submission.reviewed_by = _PLACEHOLDER_USER
    submission.review_notes = body.notes
    submission.reviewed_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(submission)

    return _to_response(submission)


@router.post(
    "/trusted-data/proposals/{proposal_id}/reindex",
    response_model=TrustedDataResponse,
)
async def reindex_proposal(
    proposal_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> TrustedDataResponse:
    """Re-enqueue indexing for a failed or pending submission."""
    submission = await db.get(TrustedDataSubmission, proposal_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    if submission.status not in ("INDEX_FAILED", "APPROVED_PENDING_INDEX"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot reindex submission with status {submission.status}",
        )

    submission.status = "APPROVED_PENDING_INDEX"
    submission.index_status = None
    submission.index_error = None

    await db.commit()
    await db.refresh(submission)

    try:
        from app.workers.trusted_data_tasks import index_trusted_submission

        index_trusted_submission.delay(str(submission.id))
        logger.info("Submission %s reindex enqueued", proposal_id)
    except Exception as exc:
        logger.warning("Failed to enqueue reindex for %s: %s", proposal_id, exc)
        submission.index_status = "INDEX_FAILED"
        submission.index_error = f"Failed to enqueue: {exc}"
        await db.commit()
        await db.refresh(submission)

    return _to_response(submission)


@router.post("/trusted-data/query", response_model=TrustedDataQueryResponse)
async def query_trusted_data(
    body: TrustedDataQueryRequest,
    db: AsyncSession = Depends(get_async_session),
) -> TrustedDataQueryResponse:
    """Search the trusted data Qdrant collection."""
    from app.services.embedding import embed_texts
    from app.services.qdrant_store import search_trusted_vectors

    vectors = embed_texts([body.query], query=True)
    results = await search_trusted_vectors(vectors[0], top_k=body.top_k)

    return TrustedDataQueryResponse(
        query=body.query,
        results=[
            TrustedDataQueryResult(
                content_text=r.get("content_text", ""),
                score=r.get("score", 0.0),
                submission_id=r.get("submission_id"),
                confidence=r.get("confidence"),
                classification=r.get("classification"),
            )
            for r in results
        ],
        total=len(results),
    )


def _to_response(submission: TrustedDataSubmission) -> TrustedDataResponse:
    return TrustedDataResponse(
        id=submission.id,
        content=submission.content,
        source_context=submission.source_context,
        proposed_by=submission.proposed_by,
        confidence=submission.confidence,
        status=submission.status,
        reviewed_by=submission.reviewed_by,
        review_notes=submission.review_notes,
        reviewed_at=submission.reviewed_at,
        created_at=submission.created_at,
        updated_at=submission.updated_at,
        index_status=submission.index_status,
        index_error=submission.index_error,
        qdrant_point_id=submission.qdrant_point_id,
        embedding_model=submission.embedding_model,
        embedded_at=submission.embedded_at,
    )
