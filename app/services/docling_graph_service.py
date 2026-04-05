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
from app.services.ontology_templates import load_ontology
from app.services.redis_utils import get_redis as _get_redis

logger = logging.getLogger(__name__)


class DeterministicExtractionError(ValueError):
    """Extraction failure that will not resolve on retry."""


class DoclingGraphCapacityError(RuntimeError):
    """All Docling-Graph concurrency permits are in use.

    Raised so the calling pipeline task can catch and retry.
    """


def _resolve_ontology(
    ontology_definition: dict[str, Any] | None,
    ontology_version: str | None,
) -> tuple[dict[str, Any], str]:
    """Return (effective_ontology, effective_version) from explicit args or active registry."""
    ontology = ontology_definition or load_ontology()
    version = ontology_version or str(ontology.get("version") or "")
    return ontology, version


def extract_graph(
    text: str,
    document_id: str,
    *,
    ontology_version: str | None = None,
    ontology_definition: dict[str, Any] | None = None,
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
    effective_ontology, effective_ontology_version = _resolve_ontology(
        ontology_definition, ontology_version,
    )

    payload: dict[str, Any] = {
        "document_id": document_id,
        "text": text,
        "mode": mode,
        "ontology_definition": effective_ontology,
    }
    if effective_ontology_version:
        payload["ontology_version"] = effective_ontology_version
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


def extract_graph_all(
    text: str,
    document_id: str,
    *,
    ontology_definition: dict[str, Any] | None = None,
    ontology_version: str | None = None,
) -> dict[str, Any]:
    """Extract all entities (5 groups in parallel) + relationships in one call.

    Uses the /extract-all endpoint which runs 5 parallel entity extraction
    LLM calls + 1 relationship extraction call internally.

    Returns a dict with keys: entities, relationships, ontology_version, model, provider.
    """
    settings = get_settings()
    url = f"{settings.docling_graph_base_url}/extract-all"
    timeout = settings.docling_graph_timeout
    effective_ontology, effective_ontology_version = _resolve_ontology(
        ontology_definition, ontology_version,
    )

    # --- Redis concurrency gate ---
    r = _get_redis()
    concurrency = settings.docling_graph_concurrency
    lock_timeout = timeout + 60
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
        raise DoclingGraphCapacityError(
            f"All {concurrency} Docling-Graph permits in use"
        )

    logger.info(
        "Calling Docling-Graph /extract-all for document %s (%d chars, permit acquired)",
        document_id, len(text),
    )

    try:
        response = httpx.post(
            url,
            json={
                "document_id": document_id,
                "text": text,
                "ontology_definition": effective_ontology,
                "ontology_version": effective_ontology_version,
            },
            timeout=timeout,
        )
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
    logger.info(
        "Docling-Graph /extract-all returned %d entities, %d relationships for document %s (model=%s)",
        len(result.get("entities", [])),
        len(result.get("relationships", [])),
        document_id,
        result.get("model", "unknown"),
    )

    return result
