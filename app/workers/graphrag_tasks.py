"""GraphRAG Celery tasks -- indexing and prompt auto-tuning.

Indexing: scheduled hourly (configurable), exports documents to Parquet,
runs Microsoft GraphRAG extraction + community detection + report generation.

Auto-tuning: scheduled daily (configurable), refines prompts based on corpus.

Both tasks use Redis locks to prevent overlapping runs.
Lock cleanup runs on worker startup to clear stale locks from killed containers.
"""

import logging

from app.config import get_settings as _get_settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_settings = _get_settings()


def cleanup_stale_locks():
    """Clear stale GraphRAG Redis locks on worker startup.

    When containers are killed mid-indexing, Redis locks persist until TTL
    expires (up to 16+ hours). This clears them so the next scheduled run
    can proceed.
    """
    import redis

    try:
        r = redis.from_url(_settings.celery_broker_url)
        for key in ("graphrag:indexing:lock", "graphrag:tuning:lock"):
            if r.exists(key):
                r.delete(key)
                logger.info("Cleared stale Redis lock: %s", key)
    except Exception:
        logger.warning("Failed to cleanup stale GraphRAG locks", exc_info=True)


# Run cleanup on module import (worker startup)
cleanup_stale_locks()


@celery_app.task(bind=True, soft_time_limit=_settings.graphrag_llm_timeout,
                 time_limit=_settings.graphrag_llm_timeout + 60)
def run_graphrag_indexing_task(self) -> dict:
    """Run GraphRAG indexing pipeline as a Celery task."""
    from app.config import get_settings
    from app.db.session import get_neo4j_driver, get_sync_session
    from app.services.graphrag_service import run_graphrag_indexing

    settings = get_settings()
    if not settings.graphrag_indexing_enabled:
        return {"skipped": True}

    # Dry-run mode: validate config without running indexing
    if settings.graphrag_dry_run:
        from app.services.graphrag_config import build_graphrag_config
        config = build_graphrag_config(settings)
        logger.info("GraphRAG dry-run: config validated successfully. model=%s, provider=%s",
                     settings.graphrag_llm_model, settings.graphrag_llm_provider)
        return {"dry_run": True, "config_valid": True}

    import redis

    r = redis.from_url(settings.celery_broker_url)
    lock = r.lock("graphrag:indexing:lock", timeout=settings.graphrag_llm_timeout, blocking=False)

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


_GRAPHRAG_SEARCH_FN = {
    "graphrag_local": "local_search",
    "graphrag_global": "global_search",
    "graphrag_drift": "drift_search",
    "graphrag_basic": "basic_search",
}


@celery_app.task(soft_time_limit=300, time_limit=360)
def run_graphrag_query_task(request_dict: dict) -> dict:
    """Run a GraphRAG query as an async Celery task.

    Returns a serialized UnifiedQueryResponse dict on success,
    or a dict with an 'error' key on failure.
    """
    from app.services import graphrag_service

    strategy = request_dict.get("strategy", "")
    query_text = request_dict.get("query_text", "")
    min_confidence = request_dict.get("min_confidence")

    fn_name = _GRAPHRAG_SEARCH_FN.get(strategy)
    if not fn_name:
        return {"error": f"Invalid GraphRAG strategy: {strategy}"}

    if not query_text:
        return {"error": "query_text is required for GraphRAG queries"}

    search_fn = getattr(graphrag_service, fn_name)
    try:
        graphrag_result = search_fn(query_text)
    except Exception as exc:
        logger.exception("GraphRAG %s query failed", strategy)
        return {"error": str(exc)}

    response = graphrag_result.get("response", "")
    if not response:
        error_key = graphrag_result.get("error", "")
        if error_key == "communities_not_indexed":
            return {
                "error": "GraphRAG indexing has not completed yet. "
                "Run indexing and wait for community detection to finish.",
            }
        if strategy in ("graphrag_local", "graphrag_global"):
            return {"error": f"GraphRAG {strategy}: no results found."}
        return _build_response(strategy, query_text, [])

    result_item = {
        "score": 1.0,
        "modality": "graphrag_response",
        "content_text": response,
        "classification": "UNCLASSIFIED",
        "context": {
            "source": strategy,
            "graphrag_context": graphrag_result.get("context", {}),
        },
    }

    results = [result_item]

    if min_confidence is not None:
        results = [r for r in results if r["score"] >= min_confidence]

    return _build_response(strategy, query_text, results)


def _build_response(strategy: str, query_text: str, results: list[dict]) -> dict:
    return {
        "query_text": query_text,
        "query_image": None,
        "strategy": strategy,
        "modality_filter": "all",
        "results": results,
        "total": len(results),
    }
