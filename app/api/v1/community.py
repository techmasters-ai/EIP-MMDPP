"""Community detection API endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/community", tags=["community"])


class DetectRequest(BaseModel):
    mode: str = "incremental"  # "incremental" or "full"


class DetectResponse(BaseModel):
    run_id: str
    status: str = "submitted"


@router.post("/detect", response_model=DetectResponse)
async def trigger_detection(request: DetectRequest):
    """Trigger community detection as a background Celery task."""
    from app.workers.community_tasks import run_community_detection_task

    run_id = str(uuid.uuid4())
    run_community_detection_task.delay(mode=request.mode, run_id=run_id)
    return DetectResponse(run_id=run_id)


@router.get("/status")
async def get_status():
    """Return the most recent community detection run record."""
    from app.db.session import get_async_session
    from sqlalchemy import text

    async with get_async_session() as session:
        result = await session.execute(
            text(
                "SELECT * FROM retrieval.community_runs "
                "ORDER BY created_at DESC LIMIT 1"
            )
        )
        row = result.mappings().first()
        if not row:
            return {"status": "no_runs"}
        return dict(row)


@router.get("/status/{run_id}")
async def get_run_status(run_id: str):
    """Return the run record for a specific detection run."""
    from app.db.session import get_async_session
    from sqlalchemy import text

    async with get_async_session() as session:
        result = await session.execute(
            text("SELECT * FROM retrieval.community_runs WHERE id = :id"),
            {"id": run_id},
        )
        row = result.mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        return dict(row)


@router.get("/reports")
async def list_reports():
    """List all community reports (up to 100)."""
    from app.db.session import get_graph_store

    graph_store = get_graph_store()
    reports = await graph_store.list_community_reports(limit=100)
    return {"reports": reports, "total": len(reports)}


@router.get("/reports/{community_id}")
async def get_report(community_id: int):
    """Return a single community report by community ID."""
    from app.db.session import get_graph_store

    graph_store = get_graph_store()
    report = await graph_store.get_community_report(community_id)
    if not report:
        raise HTTPException(status_code=404, detail="Community not found")
    return report
