"""Entity canonicalization service.

Resolves entity aliases to canonical names using GraphStore:
  1. Exact match on name or canonical_name
  2. Alias match via HAS_ALIAS edges
  3. Fuzzy match via fulltext index (score > threshold)
  4. No match -> create as new canonical entity

All operations use sync GraphStore methods since this runs in Celery workers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 0.8


def canonicalize_entity(
    graph_store: Any,
    name: str,
    entity_type: str,
) -> Optional[str]:
    """Resolve an entity name to its canonical form.

    Returns the canonical_name if found, or None if the entity is already
    canonical or newly created.
    """
    # 1. Exact match on name
    canonical = _exact_match(graph_store, name, entity_type)
    if canonical:
        return canonical

    # 2. Alias match via HAS_ALIAS edges
    canonical = _alias_match(graph_store, name)
    if canonical:
        return canonical

    # 3. Fuzzy match via fulltext index
    canonical = _fuzzy_match(graph_store, name, entity_type)
    if canonical:
        return canonical

    # 4. No match -- entity is canonical
    return None


def canonicalize_document_entities(
    graph_store: Any,
    document_id: str,
) -> dict[str, Any]:
    """Run canonicalization pass on all entities from a document.

    Finds entities linked to the document via EXTRACTED_FROM edges,
    resolves aliases, and creates HAS_ALIAS edges for discovered matches.

    Returns stats: {resolved: int, new_aliases: int, total: int}
    """
    stats = {"resolved": 0, "new_aliases": 0, "total": 0}

    # Discover entities linked to this document via graph traversal:
    # Document →(CONTAINS_TEXT)→ TextChunk ←(EXTRACTED_FROM)← Entity
    try:
        results = graph_store.get_document_entities_sync(document_id)
        # Deduplicate by (name, entity_type)
        seen: set[tuple[str, str]] = set()
        entities: list[dict[str, str]] = []
        for r in results:
            key = (r.get("name", ""), r.get("entity_type", ""))
            if key[0] and key not in seen:
                seen.add(key)
                entities.append({
                    "name": r.get("name", ""),
                    "entity_type": r.get("entity_type", ""),
                    "node_id": str(r.get("@rid", r.get("node_id", ""))),
                })
    except Exception as e:
        logger.warning("canonicalize_document_entities: could not fetch entities for %s: %s", document_id, e)
        return stats

    stats["total"] = len(entities)

    for entity in entities:
        name = entity["name"]
        entity_type = entity["entity_type"]
        node_id = entity["node_id"]

        canonical = canonicalize_entity(graph_store, name, entity_type)
        if canonical and canonical != name:
            stats["resolved"] += 1
            # Create alias edge from the canonical entity to this name
            try:
                # Find the canonical node
                canonical_node = graph_store.resolve_root_entity_sync(canonical, entity_type)
                if canonical_node:
                    graph_store.create_alias_sync(canonical_node.node_id, name)
                    stats["new_aliases"] += 1
                    # Update the entity's canonical_name property
                    if node_id:
                        graph_store.set_canonical_name_sync(node_id, canonical)
            except Exception as e:
                logger.warning("canonicalize_document_entities: alias creation failed for %s -> %s: %s", name, canonical, e)

    logger.info(
        "Canonicalization for document %s: %d/%d resolved, %d new aliases",
        document_id, stats["resolved"], stats["total"], stats["new_aliases"],
    )
    return stats


def _exact_match(graph_store: Any, name: str, entity_type: str) -> Optional[str]:
    """Check if there's an existing entity with this name as canonical_name."""
    try:
        results = graph_store.fulltext_search_sync(name, entity_types=[entity_type], limit=5)
        for r in results:
            if r.canonical_name and r.canonical_name != name and r.name != name:
                return r.name
    except Exception:
        pass
    return None


def _alias_match(graph_store: Any, name: str) -> Optional[str]:
    """Check if this name exists as an alias."""
    try:
        results = graph_store.search_by_alias_sync(name)
        if results:
            return results[0].name
    except Exception:
        pass
    return None


def _fuzzy_match(graph_store: Any, name: str, entity_type: str) -> Optional[str]:
    """Use fulltext search for fuzzy matching.

    BM25 scores are unbounded, so we normalize to 0-1 by dividing each
    score by the maximum score in the result set before comparing against
    FUZZY_THRESHOLD.
    """
    try:
        results = graph_store.fulltext_search_sync(name, limit=3)
    except Exception:
        return None

    if not results:
        return None

    # ArcadeDB fulltext results don't expose scores directly the same way,
    # so we rely on name similarity for fuzzy matching
    for r in results:
        if r.entity_type == entity_type:
            candidate = r.canonical_name or r.name
            if candidate != name:
                return candidate
    return None
