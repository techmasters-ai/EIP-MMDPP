"""Celery tasks for community detection."""

from __future__ import annotations

import logging
import uuid

import redis as redis_lib

from app.workers.celery_app import celery_app
from app.config import get_settings

logger = logging.getLogger(__name__)

COMMUNITY_LOCK_KEY = "community:detection:lock"
COMMUNITY_LOCK_TTL = 3600  # seconds — matches the longest plausible detection run


@celery_app.task(bind=True, queue="graph", max_retries=1)
def run_community_detection_task(
    self,
    mode: str = "incremental",
    run_id: str | None = None,
):
    """Run community detection.

    Args:
        mode: "incremental" (skip unchanged communities) or "full" (regenerate all).
        run_id: Optional externally-supplied UUID for tracking; generated if absent.
    """
    import asyncio
    from app.db.session import get_graph_store
    from app.services.arcadedb_community import run_community_detection

    settings = get_settings()
    if not settings.community_detection_enabled:
        logger.info("Community detection disabled; skipping run")
        return {"status": "skipped", "reason": "detection disabled"}

    run_id = run_id or str(uuid.uuid4())

    # Distributed lock: prevent concurrent detection runs
    r = redis_lib.Redis.from_url(settings.celery_broker_url)
    if not r.set(COMMUNITY_LOCK_KEY, run_id, nx=True, ex=COMMUNITY_LOCK_TTL):
        logger.info("Community detection already running; skipping run %s", run_id)
        return {"status": "skipped", "reason": "detection already running"}

    try:
        _record_run_start(run_id, mode)

        graph_store = get_graph_store()
        result = asyncio.run(run_community_detection(graph_store, mode=mode))

        _record_run_complete(run_id, result)
        return result
    except Exception as exc:
        _record_run_failed(run_id, str(exc))
        raise
    finally:
        r.delete(COMMUNITY_LOCK_KEY)


# ---------------------------------------------------------------------------
# PostgreSQL run-record helpers
# ---------------------------------------------------------------------------

def _record_run_start(run_id: str, mode: str) -> None:
    """Insert a RUNNING community_runs row."""
    from app.db.session import get_sync_session
    from sqlalchemy import text

    with get_sync_session() as session:
        session.execute(
            text(
                "INSERT INTO retrieval.community_runs "
                "(id, status, trigger, started_at) "
                "VALUES (:id, 'RUNNING', :trigger, NOW())"
            ),
            {"id": run_id, "trigger": mode.upper()},
        )
        session.commit()


def _record_run_complete(run_id: str, result: dict) -> None:
    """Update the run record with success stats."""
    from app.db.session import get_sync_session
    from sqlalchemy import text

    with get_sync_session() as session:
        session.execute(
            text(
                "UPDATE retrieval.community_runs SET "
                "status = 'COMPLETE', "
                "total_communities = :total, "
                "reports_generated = :gen, "
                "reports_reused = :reused, "
                "completed_at = NOW() "
                "WHERE id = :id"
            ),
            {
                "id": run_id,
                "total": result.get("total_communities", 0),
                "gen": result.get("reports_generated", 0),
                "reused": result.get("reports_reused", 0),
            },
        )
        session.commit()


def _record_run_failed(run_id: str, error: str) -> None:
    """Update the run record with failure info."""
    from app.db.session import get_sync_session
    from sqlalchemy import text

    with get_sync_session() as session:
        session.execute(
            text(
                "UPDATE retrieval.community_runs SET "
                "status = 'FAILED', "
                "error_message = :err, "
                "completed_at = NOW() "
                "WHERE id = :id"
            ),
            {"id": run_id, "err": error[:1000]},
        )
        session.commit()
