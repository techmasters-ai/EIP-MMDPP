"""Docling-Graph HTTP client.

Calls the standalone Docling-Graph service for ontology-driven
entity/relationship extraction. Replaces the previous in-process
LLM extraction pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class DeterministicExtractionError(ValueError):
    """Extraction failure that will not resolve on retry."""


def extract_graph(
    text: str,
    document_id: str,
    *,
    ontology_version: str | None = None,
) -> dict[str, Any]:
    """Extract entities and relationships via the Docling-Graph service.

    Returns a dict with keys: entities, relationships, ontology_version, model, provider.
    Raises httpx.HTTPStatusError on service errors (caller should retry).
    """
    settings = get_settings()
    url = f"{settings.docling_graph_base_url}/extract"
    timeout = settings.docling_graph_timeout

    payload: dict[str, Any] = {
        "document_id": document_id,
        "text": text,
    }
    if ontology_version:
        payload["ontology_version"] = ontology_version

    logger.info(
        "Calling Docling-Graph service for document %s (%d chars)",
        document_id,
        len(text),
    )

    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    result = response.json()

    entity_count = len(result.get("entities", []))
    rel_count = len(result.get("relationships", []))
    logger.info(
        "Docling-Graph returned %d entities, %d relationships for document %s (model=%s)",
        entity_count,
        rel_count,
        document_id,
        result.get("model", "unknown"),
    )

    return result
