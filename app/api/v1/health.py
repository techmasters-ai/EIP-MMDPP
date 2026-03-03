"""Health and readiness endpoints."""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["ops"])
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, str]


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Returns 200 if the API process is alive."""
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadyResponse, summary="Readiness probe")
async def ready() -> ReadyResponse:
    """Returns 200 if all dependencies are reachable."""
    checks: dict[str, str] = {}
    all_ok = True

    # Check PostgreSQL
    try:
        from app.db.session import async_engine
        async with async_engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
        all_ok = False

    # Check Redis
    try:
        import redis.asyncio as aioredis
        from app.config import get_settings
        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        all_ok = False

    # Check MinIO
    try:
        from app.services.storage import get_async_s3_client
        from app.config import get_settings
        settings = get_settings()
        async with get_async_s3_client() as client:
            await client.list_buckets()
        checks["minio"] = "ok"
    except Exception as e:
        checks["minio"] = f"error: {e}"
        all_ok = False

    status = "ready" if all_ok else "degraded"
    return ReadyResponse(status=status, checks=checks)
