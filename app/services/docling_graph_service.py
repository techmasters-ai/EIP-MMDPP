"""Docling-Graph HTTP client.

Calls the standalone Docling-Graph service for ontology-driven
entity/relationship extraction. Replaces the previous in-process
LLM extraction pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import redis as redis_lib

from app.config import get_settings

logger = logging.getLogger(__name__)

# Redis client for concurrency gating (initialised lazily)
_redis_client: redis_lib.Redis | None = None


def _get_redis() -> redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis_lib.Redis.from_url(settings.celery_broker_url)
    return _redis_client


class DeterministicExtractionError(ValueError):
    """Extraction failure that will not resolve on retry."""


class DoclingGraphCapacityError(RuntimeError):
    """All Docling-Graph concurrency permits are in use.

    Raised so the calling pipeline task can catch and retry.
    """


def extract_graph(
    text: str,
    document_id: str,
    *,
    ontology_version: str | None = None,
    template_group: str | None = None,
    mode: str = "entities",
    entities_context: list[dict] | None = None,
) -> dict[str, Any]:
    """Extract entities and relationships via the Docling-Graph service.

    Returns a dict with keys: entities, relationships, ontology_version, model, provider.
    Raises httpx.HTTPStatusError on service errors (caller should retry).
    Raises DoclingGraphCapacityError when all concurrency permits are in use.
    """
    settings = get_settings()
    url = f"{settings.docling_graph_base_url}/extract"
    timeout = settings.docling_graph_timeout

    payload: dict[str, Any] = {
        "document_id": document_id,
        "text": text,
        "mode": mode,
    }
    if ontology_version:
        payload["ontology_version"] = ontology_version
    if template_group:
        payload["template_group"] = template_group
    if entities_context is not None:
        payload["entities_context"] = entities_context

    # --- Redis concurrency gate (mirrors Docling permit pattern in pipeline.py) ---
    r = _get_redis()
    concurrency = settings.docling_graph_concurrency
    lock_timeout = timeout + 60  # auto-release safety margin beyond HTTP timeout
    permit_lock = None

    for permit_i in range(concurrency):
        candidate = r.lock(
            f"docling-graph:permit:{permit_i}",
            timeout=lock_timeout,
            blocking=False,
        )
        if candidate.acquire(blocking=False):
            permit_lock = candidate
            break

    if permit_lock is None:
        logger.warning(
            "Docling-Graph at capacity (%d/%d) for document %s — raising for retry",
            concurrency,
            concurrency,
            document_id,
        )
        raise DoclingGraphCapacityError(
            f"All {concurrency} Docling-Graph permits in use"
        )

    logger.info(
        "Calling Docling-Graph service for document %s (%d chars, group=%s, mode=%s, permit acquired)",
        document_id, len(text), template_group or "legacy", mode,
    )

    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    finally:
        try:
            permit_lock.release()
        except redis_lib.exceptions.LockNotOwnedError:
            logger.warning(
                "Docling-Graph permit lock expired before release for document %s",
                document_id,
            )

    result = response.json()

    entity_count = len(result.get("entities", []))
    rel_count = len(result.get("relationships", []))
    logger.info(
        "Docling-Graph returned %d entities, %d relationships for document %s (group=%s, mode=%s, model=%s)",
        entity_count, rel_count, document_id, template_group or "legacy", mode, result.get("model", "unknown"),
    )

    return result
