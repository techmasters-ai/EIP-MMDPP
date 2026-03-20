"""GraphRAG Celery tasks -- indexing and prompt auto-tuning.

Indexing: scheduled hourly (configurable), exports Neo4j graph to Parquet,
runs Microsoft GraphRAG community detection + report generation.

Auto-tuning: scheduled daily (configurable), refines prompts based on corpus.

Both tasks use Redis locks to prevent overlapping runs.
"""

import logging

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, soft_time_limit=3600, time_limit=3660)
def run_graphrag_indexing_task(self) -> dict:
    """Run GraphRAG indexing pipeline as a Celery task."""
    from app.config import get_settings
    from app.db.session import get_neo4j_driver, get_sync_session
    from app.services.graphrag_service import run_graphrag_indexing

    settings = get_settings()
    if not settings.graphrag_indexing_enabled:
        return {"skipped": True}

    import redis

    r = redis.from_url(settings.celery_broker_url)
    lock = r.lock("graphrag:indexing:lock", timeout=3600, blocking=False)

    if not lock.acquire(blocking=False):
        logger.info("GraphRAG indexing already in progress -- skipping")
        return {"skipped": True, "reason": "locked"}

    try:
        neo4j_driver = get_neo4j_driver()
        db = get_sync_session()
        try:
            stats = run_graphrag_indexing(neo4j_driver, db)
            logger.info("GraphRAG indexing complete: %s", stats)

            # Record completion timestamp for the UI countdown
            import datetime
            try:
                r.set("graphrag:last_indexed_at",
                      datetime.datetime.now(datetime.timezone.utc).isoformat())
            except Exception:
                pass

            return stats
        finally:
            db.close()
    finally:
        try:
            lock.release()
        except Exception:
            pass


@celery_app.task(bind=True, soft_time_limit=3600, time_limit=3660)
def run_graphrag_auto_tune_task(self) -> dict:
    """Run GraphRAG prompt auto-tuning as a Celery task."""
    from app.config import get_settings
    from app.services.graphrag_service import run_auto_tune

    settings = get_settings()
    if not settings.graphrag_indexing_enabled:
        return {"skipped": True}

    import redis

    r = redis.from_url(settings.celery_broker_url)
    lock = r.lock("graphrag:tuning:lock", timeout=3600, blocking=False)

    if not lock.acquire(blocking=False):
        logger.info("GraphRAG auto-tuning already in progress -- skipping")
        return {"skipped": True, "reason": "locked"}

    try:
        result = run_auto_tune()
        logger.info("GraphRAG auto-tuning complete: %s", result)
        return result
    finally:
        try:
            lock.release()
        except Exception:
            pass
