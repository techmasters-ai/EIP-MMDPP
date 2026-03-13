"""Entity canonicalization service.

Resolves entity aliases to canonical names using:
  1. Exact match on name or canonical_name in Neo4j
  2. Alias match via HAS_ALIAS edges
  3. Fuzzy match via Neo4j fulltext index (score > threshold)
  4. No match → create as new canonical entity
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

FUZZY_THRESHOLD = 0.8


def canonicalize_entity(
    driver,
    name: str,
    entity_type: str,
) -> Optional[str]:
    """Resolve an entity name to its canonical form.

    Returns the canonical_name if found, or None if the entity is already
    canonical or newly created.
    """
    # 1. Exact match on name
    canonical = _exact_match(driver, name, entity_type)
    if canonical:
        return canonical

    # 2. Alias match via HAS_ALIAS edges
    canonical = _alias_match(driver, name)
    if canonical:
        return canonical

    # 3. Fuzzy match via fulltext index
    canonical = _fuzzy_match(driver, name, entity_type)
    if canonical:
        return canonical

    # 4. No match — entity is canonical
    return None


def canonicalize_document_entities(
    driver,
    document_id: str,
) -> dict[str, Any]:
    """Run canonicalization pass on all entities from a document.

    Finds entities linked to the document via EXTRACTED_FROM edges,
    resolves aliases, and creates HAS_ALIAS edges for discovered matches.

    Returns stats: {resolved: int, new_aliases: int, total: int}
    """
    from app.services.neo4j_graph import create_alias_edge

    stats = {"resolved": 0, "new_aliases": 0, "total": 0}

    # Get all entities extracted from this document's chunks
    query = """
        MATCH (d:Document {document_id: $document_id})
              -[:CONTAINS_TEXT|CONTAINS_IMAGE]->(c:ChunkRef)
              <-[:EXTRACTED_FROM]-(e:Entity)
        RETURN DISTINCT e.name AS name, e.entity_type AS entity_type
    """

    try:
        with driver.session() as session:
            result = session.run(query, document_id=document_id)
            entities = [dict(record) for record in result]
    except Exception as e:
        logger.warning("canonicalize_document_entities failed for %s: %s", document_id, e)
        return stats

    stats["total"] = len(entities)

    for entity in entities:
        name = entity["name"]
        entity_type = entity["entity_type"]

        canonical = canonicalize_entity(driver, name, entity_type)
        if canonical and canonical != name:
            stats["resolved"] += 1
            # Create alias edge
            success = create_alias_edge(driver, canonical, name, entity_type)
            if success:
                stats["new_aliases"] += 1
                # Update the entity's canonical_name property
                _set_canonical_name(driver, name, entity_type, canonical)

    logger.info(
        "Canonicalization for document %s: %d/%d resolved, %d new aliases",
        document_id, stats["resolved"], stats["total"], stats["new_aliases"],
    )
    return stats


def _exact_match(driver, name: str, entity_type: str) -> Optional[str]:
    """Check if there's an existing entity with this name as canonical_name."""
    query = """
        MATCH (n:Entity {entity_type: $entity_type})
        WHERE n.canonical_name = $name AND n.name <> $name
        RETURN n.name AS canonical_name
        LIMIT 1
    """

    try:
        with driver.session() as session:
            result = session.run(query, name=name, entity_type=entity_type)
            record = result.single()
            return record["canonical_name"] if record else None
    except Exception:
        return None


def _alias_match(driver, name: str) -> Optional[str]:
    """Check if this name exists as an alias."""
    query = """
        MATCH (e:Entity)-[:HAS_ALIAS]->(a:Alias {alias_name: $name})
        RETURN e.name AS canonical_name
        LIMIT 1
    """

    try:
        with driver.session() as session:
            result = session.run(query, name=name)
            record = result.single()
            return record["canonical_name"] if record else None
    except Exception:
        return None


def _fuzzy_match(driver, name: str, entity_type: str) -> Optional[str]:
    """Use fulltext search for fuzzy matching.

    BM25 scores are unbounded, so we normalize to 0-1 by dividing each
    score by the maximum score in the result set before comparing against
    FUZZY_THRESHOLD.
    """
    from app.services.neo4j_graph import fulltext_search_entity

    results = fulltext_search_entity(driver, name, limit=3)
    if not results:
        return None

    max_score = max(r["score"] for r in results) if results else 1.0
    for r in results:
        if r["entity_type"] == entity_type:
            normalized_score = r["score"] / max_score if max_score > 0 else 0
            if normalized_score > FUZZY_THRESHOLD:
                candidate = r.get("canonical_name") or r["name"]
                if candidate != name:
                    return candidate
    return None


def _set_canonical_name(driver, name: str, entity_type: str, canonical: str) -> None:
    """Set the canonical_name property on an entity node."""
    query = """
        MATCH (n:Entity {name: $name, entity_type: $entity_type})
        SET n.canonical_name = $canonical
    """

    try:
        with driver.session() as session:
            session.run(query, name=name, entity_type=entity_type, canonical=canonical)
    except Exception as e:
        logger.warning("_set_canonical_name failed for %s: %s", name, e)
