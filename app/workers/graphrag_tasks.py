"""GraphRAG indexing — Celery Beat periodic task.

Runs community detection and report generation on the Neo4j knowledge graph.
Gated by GRAPHRAG_INDEXING_ENABLED. Uses a Redis lock to prevent overlapping runs.
"""

import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, soft_time_limit=1800, time_limit=1860)
def run_graphrag_indexing_task(self) -> dict:
    """Celery Beat task wrapper for GraphRAG community detection + report generation."""
    from app.config import get_settings
    from app.db.session import get_neo4j_driver, get_sync_session
    from app.services.graphrag_service import run_graphrag_indexing

    settings = get_settings()
    if not settings.graphrag_indexing_enabled:
        return {"skipped": True}

    # Redis lock to prevent overlapping runs
    redis_url = settings.celery_broker_url
    import redis

    r = redis.from_url(redis_url)
    lock = r.lock("graphrag:indexing:lock", timeout=1800, blocking=False)

    if not lock.acquire(blocking=False):
        logger.info("GraphRAG indexing already in progress — skipping")
        return {"skipped": True, "reason": "locked"}

    try:
        neo4j_driver = get_neo4j_driver()
        db = get_sync_session()
        try:
            stats = run_graphrag_indexing(neo4j_driver, db)
            logger.info("GraphRAG indexing complete: %s", stats)
            return stats
        finally:
            db.close()
    finally:
        try:
            lock.release()
        except Exception:
            pass
