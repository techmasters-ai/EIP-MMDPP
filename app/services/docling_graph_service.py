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


def _acquire_permit(document_id: str, timeout: float):
    """Acquire a Docling-Graph concurrency permit. Returns the lock or raises."""
    settings = get_settings()
    r = _get_redis()
    concurrency = settings.docling_graph_concurrency
    lock_timeout = timeout + 60  # auto-release safety margin beyond HTTP timeout

    for permit_i in range(concurrency):
        candidate = r.lock(
            f"docling-graph:permit:{permit_i}",
            timeout=lock_timeout,
            blocking=False,
        )
        if candidate.acquire(blocking=False):
            return candidate

    logger.warning(
        "Docling-Graph at capacity (%d/%d) for document %s — raising for retry",
        concurrency, concurrency, document_id,
    )
    raise DoclingGraphCapacityError(
        f"All {concurrency} Docling-Graph permits in use"
    )


def _release_permit(permit_lock, document_id: str) -> None:
    """Release a concurrency permit, tolerating expiry."""
    try:
        permit_lock.release()
    except redis_lib.exceptions.LockNotOwnedError:
        logger.warning(
            "Docling-Graph permit lock expired before release for document %s",
            document_id,
        )


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

    # --- Redis concurrency gate ---
    permit_lock = _acquire_permit(document_id, timeout)

    logger.info(
        "Calling Docling-Graph service for document %s (%d chars, group=%s, mode=%s, permit acquired)",
        document_id, len(text), template_group or "legacy", mode,
    )

    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    finally:
        _release_permit(permit_lock, document_id)

    raw = response.json()
    result = _normalize_extraction_result(raw)

    entity_count = len(result.get("entities", []))
    rel_count = len(result.get("relationships", []))
    logger.info(
        "Docling-Graph returned %d entities, %d relationships for document %s (group=%s, mode=%s, model=%s)",
        entity_count, rel_count, document_id, template_group or "legacy", mode, result.get("model", "unknown"),
    )

    return result


def extract_graph_all(
    docling_document_json: dict[str, Any],
    document_id: str,
    *,
    ontology_definition: dict[str, Any] | None = None,
    ontology_version: str | None = None,
) -> dict[str, Any]:
    """Extract all entities + relationships via the /extract-all endpoint.

    Sends the structured DoclingDocument JSON (preserving layout, tables,
    and element provenance) and returns a normalized dict with keys:
    ``entities``, ``relationships``, ``ontology_version``, ``model``,
    ``provider``.

    The Docling-Graph service returns a NetworkX node-link graph; this
    function normalizes it into the ``{entities, relationships}`` shape
    that the rest of the pipeline expects.
    """
    settings = get_settings()
    url = f"{settings.docling_graph_base_url}/extract-all"
    timeout = settings.docling_graph_timeout
    effective_ontology, effective_ontology_version = _resolve_ontology(
        ontology_definition, ontology_version,
    )

    # --- Redis concurrency gate ---
    permit_lock = _acquire_permit(document_id, timeout)

    logger.info(
        "Calling Docling-Graph /extract-all for document %s (permit acquired)",
        document_id,
    )

    try:
        response = httpx.post(
            url,
            json={
                "document_id": document_id,
                "docling_document_json": docling_document_json,
                "ontology_definition": effective_ontology,
                "ontology_version": effective_ontology_version,
            },
            timeout=timeout,
        )
        response.raise_for_status()
    finally:
        _release_permit(permit_lock, document_id)

    raw = response.json()
    result = _normalize_extraction_result(raw)
    logger.info(
        "Docling-Graph /extract-all returned %d entities, %d relationships for document %s (model=%s)",
        len(result.get("entities", [])),
        len(result.get("relationships", [])),
        document_id,
        result.get("model", "unknown"),
    )

    return result


def _normalize_extraction_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize Docling-Graph's NetworkX node-link response into the
    ``{entities, relationships}`` shape the pipeline expects.

    Docling-Graph returns::

        {
            "graph": {"nodes": [...], "links": [...]},  # NetworkX node-link
            "metadata": {...},
            "model": "...",
            "provider": "docling-graph",
        }

    The pipeline expects::

        {
            "entities": [{name, entity_type, confidence, properties}, ...],
            "relationships": [{from_name, to_name, from_type, to_type, rel_type, confidence}, ...],
            "model": "...",
            "provider": "...",
        }
    """
    graph = raw.get("graph", {})
    nodes = graph.get("nodes", [])
    links = graph.get("links", [])

    # -- If the response already has the old shape, pass through unchanged --
    if "entities" in raw and "graph" not in raw:
        return raw

    entities: list[dict[str, Any]] = []
    for node in nodes:
        name = node.get("name", node.get("id", ""))
        entity_type = node.get("type", node.get("entity_type", "UNKNOWN"))
        confidence = node.get("confidence", 0.8)
        # Collect all non-system keys as properties
        skip_keys = {"id", "name", "type", "entity_type", "confidence",
                      "provenance", "_provenance", "__property_provenance"}
        props = {k: v for k, v in node.items() if k not in skip_keys}
        # Preserve provenance from the library's delta normalizer
        # (batch_index, chunk_indexes, page_numbers) — enables element-level
        # source tracking without rebuilding it from regex mention matching.
        prov = node.get("provenance") or node.get("_provenance")
        entities.append({
            "name": name,
            "entity_type": entity_type,
            "confidence": confidence,
            "properties": props,
            "provenance": prov,
        })

    relationships: list[dict[str, Any]] = []
    # Build a quick node-id → (name, type) lookup for edge resolution
    node_lookup: dict[str, tuple[str, str]] = {}
    for node in nodes:
        nid = str(node.get("id", ""))
        nname = node.get("name", nid)
        ntype = node.get("type", node.get("entity_type", "UNKNOWN"))
        node_lookup[nid] = (nname, ntype)

    for link in links:
        src = str(link.get("source", ""))
        tgt = str(link.get("target", ""))
        from_name, from_type = node_lookup.get(src, (src, "UNKNOWN"))
        to_name, to_type = node_lookup.get(tgt, (tgt, "UNKNOWN"))
        rel_type = link.get("label", link.get("type", link.get("rel_type", "RELATED_TO")))
        confidence = link.get("confidence", 0.8)
        skip_keys = {"source", "target", "label", "type", "rel_type", "confidence",
                      "provenance", "_provenance", "edge_label"}
        props = {k: v for k, v in link.items() if k not in skip_keys}
        prov = link.get("provenance") or link.get("_provenance")
        relationships.append({
            "from_name": from_name,
            "from_type": from_type,
            "to_name": to_name,
            "to_type": to_type,
            "rel_type": rel_type,
            "confidence": confidence,
            "properties": props,
            "provenance": prov,
        })

    return {
        "entities": entities,
        "relationships": relationships,
        "model": raw.get("model", "unknown"),
        "provider": raw.get("provider", "docling-graph"),
        "ontology_version": raw.get("ontology_version"),
        "metadata": raw.get("metadata"),
    }
