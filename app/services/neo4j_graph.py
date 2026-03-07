"""Neo4j knowledge-graph operations.

Replaces app/services/graph.py (Apache AGE).  Provides sync helpers for
Celery workers and async helpers for FastAPI routes.  All operations target
the default Neo4j database.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cache for EXTRACTED_FROM edge existence check (avoids per-query Neo4j lookups)
_extracted_from_cache: dict[str, Any] = {"exists": None, "checked_at": 0.0}
_EXTRACTED_FROM_CACHE_TTL = 60  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_label(label: str) -> str:
    """Sanitize a string for use as a Neo4j node/edge label."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", label)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "UNKNOWN"


# ---------------------------------------------------------------------------
# Sync operations (Celery workers)
# ---------------------------------------------------------------------------

def upsert_node(
    driver,
    entity_type: str,
    name: str,
    artifact_id: str,
    confidence: float,
    properties: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Create or update an entity node.  Returns the node UUID."""
    node_id = str(uuid.uuid4())
    props = dict(properties or {})
    props.update({
        "id": node_id,
        "name": name,
        "entity_type": entity_type,
        "artifact_id": artifact_id,
        "confidence": confidence,
    })

    label = _sanitize_label(entity_type)
    query = f"""
        MERGE (n:Entity:{label} {{name: $name, entity_type: $entity_type}})
        ON CREATE SET n += $props
        ON MATCH SET
            n.last_artifact_id = $artifact_id,
            n.confidence = CASE
                WHEN n.confidence < $confidence THEN $confidence
                ELSE n.confidence
            END
        RETURN n.id AS node_id
    """

    try:
        with driver.session() as session:
            result = session.run(
                query,
                name=name,
                entity_type=entity_type,
                artifact_id=artifact_id,
                confidence=confidence,
                props=props,
            )
            record = result.single()
            return record["node_id"] if record else node_id
    except Exception as e:
        logger.warning("upsert_node failed for %s/%s: %s", entity_type, name, e)
        return None


def upsert_relationship(
    driver,
    from_name: str,
    from_type: str,
    to_name: str,
    to_type: str,
    rel_type: str,
    artifact_id: str,
    confidence: float,
) -> bool:
    """Create or update a directed relationship between two entity nodes."""
    from_label = _sanitize_label(from_type)
    to_label = _sanitize_label(to_type)
    rel_label = _sanitize_label(rel_type)

    query = f"""
        MATCH (a:Entity:{from_label} {{name: $from_name}})
        MATCH (b:Entity:{to_label} {{name: $to_name}})
        MERGE (a)-[r:{rel_label} {{artifact_id: $artifact_id}}]->(b)
        ON MATCH SET r.confidence = CASE
            WHEN r.confidence < $confidence THEN $confidence
            ELSE r.confidence
        END
        RETURN count(r) AS cnt
    """

    try:
        with driver.session() as session:
            session.run(
                query,
                from_name=from_name,
                to_name=to_name,
                artifact_id=artifact_id,
                confidence=confidence,
            )
        return True
    except Exception as e:
        logger.warning(
            "upsert_relationship failed %s→%s [%s]: %s",
            from_name, to_name, rel_type, e,
        )
        return False


# ---------------------------------------------------------------------------
# Document / Chunk structural operations (sync)
# ---------------------------------------------------------------------------

def upsert_document_node(
    driver,
    document_id: str,
    title: str,
    properties: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Create or update a Document node."""
    props = dict(properties or {})
    props.update({"document_id": document_id, "title": title})

    query = """
        MERGE (d:Document {document_id: $document_id})
        ON CREATE SET d += $props
        ON MATCH SET d.title = $title
        RETURN d.document_id AS doc_id
    """

    try:
        with driver.session() as session:
            session.run(query, document_id=document_id, title=title, props=props)
        return document_id
    except Exception as e:
        logger.warning("upsert_document_node failed for %s: %s", document_id, e)
        return None


def upsert_chunk_ref_node(
    driver,
    chunk_id: str,
    chunk_type: str,
) -> Optional[str]:
    """Create or update a ChunkRef node."""
    query = """
        MERGE (c:ChunkRef {chunk_id: $chunk_id})
        ON CREATE SET c.chunk_type = $chunk_type
        RETURN c.chunk_id AS cid
    """

    try:
        with driver.session() as session:
            session.run(query, chunk_id=chunk_id, chunk_type=chunk_type)
        return chunk_id
    except Exception as e:
        logger.warning("upsert_chunk_ref_node failed for %s: %s", chunk_id, e)
        return None


def create_structural_edge(
    driver,
    from_id: str,
    to_id: str,
    edge_type: str,
) -> bool:
    """Create a structural edge (CONTAINS_TEXT, CONTAINS_IMAGE, SAME_PAGE).

    These are deterministic edges and do NOT require dual-curator approval.
    """
    label = _sanitize_label(edge_type)

    if edge_type in ("CONTAINS_TEXT", "CONTAINS_IMAGE"):
        query = f"""
            MATCH (d:Document {{document_id: $from_id}})
            MATCH (c:ChunkRef {{chunk_id: $to_id}})
            MERGE (d)-[r:{label}]->(c)
            RETURN count(r) AS cnt
        """
    elif edge_type == "SAME_PAGE":
        query = f"""
            MATCH (a:ChunkRef {{chunk_id: $from_id}})
            MATCH (b:ChunkRef {{chunk_id: $to_id}})
            MERGE (a)-[r:{label}]->(b)
            RETURN count(r) AS cnt
        """
    else:
        query = f"""
            MATCH (a:ChunkRef {{chunk_id: $from_id}})
            MATCH (b:ChunkRef {{chunk_id: $to_id}})
            MERGE (a)-[r:{label}]->(b)
            RETURN count(r) AS cnt
        """

    try:
        with driver.session() as session:
            session.run(query, from_id=from_id, to_id=to_id)
        return True
    except Exception as e:
        logger.warning(
            "create_structural_edge failed %s→%s [%s]: %s",
            from_id, to_id, edge_type, e,
        )
        return False


def create_entity_chunk_edge(
    driver,
    entity_name: str,
    entity_type: str,
    chunk_id: str,
) -> bool:
    """Create an EXTRACTED_FROM edge from an entity to a ChunkRef node."""
    label = _sanitize_label(entity_type)
    query = f"""
        MATCH (e:Entity:{label} {{name: $entity_name}})
        MATCH (c:ChunkRef {{chunk_id: $chunk_id}})
        MERGE (e)-[r:EXTRACTED_FROM]->(c)
        RETURN count(r) AS cnt
    """

    try:
        with driver.session() as session:
            session.run(query, entity_name=entity_name, chunk_id=chunk_id)
        return True
    except Exception as e:
        logger.warning(
            "create_entity_chunk_edge failed %s[%s]→%s: %s",
            entity_name, entity_type, chunk_id, e,
        )
        return False


# ---------------------------------------------------------------------------
# Query helpers (sync)
# ---------------------------------------------------------------------------

def search_nodes_by_name(
    driver,
    search_term: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search entity nodes whose name contains the search term (case-insensitive)."""
    query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($search)
        RETURN n, n.entity_type AS entity_type
        LIMIT $limit
    """

    try:
        with driver.session() as session:
            result = session.run(query, search=search_term, limit=limit)
            return [
                {"node": dict(record["n"]), "entity_type": record["entity_type"]}
                for record in result
            ]
    except Exception as e:
        logger.warning("search_nodes_by_name failed for '%s': %s", search_term, e)
        return []


def get_node_neighborhood(
    driver,
    node_name: str,
    hop_count: int = 2,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get a node and its neighborhood up to hop_count hops."""
    hop_count = max(1, min(hop_count, 4))

    query = f"""
        MATCH (start:Entity {{name: $name}})
        OPTIONAL MATCH (start)-[r*1..{hop_count}]-(neighbor:Entity)
        WITH start, r, neighbor
        LIMIT $limit
        RETURN start, type(head(r)) AS rel_type, neighbor
    """

    try:
        with driver.session() as session:
            result = session.run(query, name=node_name, limit=limit)
            return [
                {
                    "node": dict(record["start"]) if record["start"] else None,
                    "rel_type": record["rel_type"],
                    "neighbor": dict(record["neighbor"]) if record["neighbor"] else None,
                }
                for record in result
            ]
    except Exception as e:
        logger.warning("get_node_neighborhood failed for '%s': %s", node_name, e)
        return []


def get_graph_stats(driver) -> dict[str, int]:
    """Return total node and edge counts."""
    stats: dict[str, int] = {"nodes": 0, "edges": 0}
    try:
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS cnt")
            record = result.single()
            if record:
                stats["nodes"] = record["cnt"]

            result = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            record = result.single()
            if record:
                stats["edges"] = record["cnt"]
    except Exception as e:
        logger.warning("get_graph_stats failed: %s", e)
    return stats


# ---------------------------------------------------------------------------
# Async operations (FastAPI)
# ---------------------------------------------------------------------------

async def search_nodes_async(
    driver,
    search_term: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Async search for entity nodes."""
    query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($search)
        RETURN n, n.entity_type AS entity_type
        LIMIT $limit
    """

    try:
        async with driver.session() as session:
            result = await session.run(query, search=search_term, limit=limit)
            records = await result.data()
            return [
                {"node": dict(r["n"]), "entity_type": r["entity_type"]}
                for r in records
            ]
    except Exception as e:
        logger.warning("search_nodes_async failed for '%s': %s", search_term, e)
        return []


async def get_neighborhood_async(
    driver,
    node_name: str,
    hop_count: int = 2,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Async version of get_node_neighborhood."""
    hop_count = max(1, min(hop_count, 4))

    query = f"""
        MATCH (start:Entity {{name: $name}})
        OPTIONAL MATCH (start)-[r*1..{hop_count}]-(neighbor:Entity)
        WITH start, r, neighbor
        LIMIT $limit
        RETURN start, type(head(r)) AS rel_type, neighbor
    """

    try:
        async with driver.session() as session:
            result = await session.run(query, name=node_name, limit=limit)
            records = await result.data()
            return [
                {
                    "node": dict(r["start"]) if r["start"] else None,
                    "rel_type": r["rel_type"],
                    "neighbor": dict(r["neighbor"]) if r["neighbor"] else None,
                }
                for r in records
            ]
    except Exception as e:
        logger.warning("get_neighborhood_async failed for '%s': %s", node_name, e)
        return []


async def _has_extracted_from_edges(driver) -> bool:
    """Check if any EXTRACTED_FROM edges exist (cached for 60s)."""
    now = time.monotonic()
    if (
        _extracted_from_cache["exists"] is not None
        and now - _extracted_from_cache["checked_at"] < _EXTRACTED_FROM_CACHE_TTL
    ):
        return _extracted_from_cache["exists"]

    try:
        async with driver.session() as session:
            result = await session.run(
                "MATCH ()-[r:EXTRACTED_FROM]->() RETURN count(r) > 0 AS has_edges LIMIT 1"
            )
            record = await result.single()
            exists = bool(record and record["has_edges"])
    except Exception:
        exists = False

    _extracted_from_cache["exists"] = exists
    _extracted_from_cache["checked_at"] = now
    return exists


async def get_ontology_linked_chunks_async(
    driver,
    chunk_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find chunks linked via ontology relationships.

    Path: ChunkRef <-[EXTRACTED_FROM]- Entity -[ontology_rel]- Related
          -[EXTRACTED_FROM]-> ChunkRef
    """
    # Short-circuit if no EXTRACTED_FROM edges exist (prevents warning spam)
    if not await _has_extracted_from_edges(driver):
        return []

    query = """
        MATCH (src:ChunkRef {chunk_id: $chunk_id})
              <-[:EXTRACTED_FROM]-(entity:Entity)
              -[r]-(related:Entity)
              -[:EXTRACTED_FROM]->(target:ChunkRef)
        WHERE target.chunk_id <> $chunk_id
        RETURN target.chunk_id AS target_chunk_id,
               target.chunk_type AS target_chunk_type,
               type(r) AS rel_type,
               entity.name AS entity_name,
               related.name AS related_name
        LIMIT $limit
    """

    try:
        async with driver.session() as session:
            result = await session.run(query, chunk_id=chunk_id, limit=limit)
            records = await result.data()
            return [
                {
                    "target_chunk_id": r["target_chunk_id"],
                    "target_chunk_type": r["target_chunk_type"],
                    "rel_type": r["rel_type"],
                    "entity_name": r["entity_name"],
                    "related_name": r["related_name"],
                }
                for r in records
            ]
    except Exception as e:
        logger.warning(
            "get_ontology_linked_chunks_async failed for chunk %s: %s",
            chunk_id, e,
        )
        return []


async def get_graph_stats_async(driver) -> dict[str, int]:
    """Async version of get_graph_stats."""
    stats: dict[str, int] = {"nodes": 0, "edges": 0}
    try:
        async with driver.session() as session:
            result = await session.run("MATCH (n) RETURN count(n) AS cnt")
            record = await result.single()
            if record:
                stats["nodes"] = record["cnt"]

            result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            record = await result.single()
            if record:
                stats["edges"] = record["cnt"]
    except Exception as e:
        logger.warning("get_graph_stats_async failed: %s", e)
    return stats


# ---------------------------------------------------------------------------
# Fulltext search (used by canonicalization)
# ---------------------------------------------------------------------------

def fulltext_search_entity(
    driver,
    query_text: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search entities using the fulltext index on name + canonical_name."""
    query = """
        CALL db.index.fulltext.queryNodes('entity_name_fulltext', $query_text)
        YIELD node, score
        RETURN node.name AS name,
               node.canonical_name AS canonical_name,
               node.entity_type AS entity_type,
               score
        LIMIT $limit
    """

    try:
        with driver.session() as session:
            result = session.run(query, query_text=query_text, limit=limit)
            return [dict(record) for record in result]
    except Exception as e:
        logger.warning("fulltext_search_entity failed for '%s': %s", query_text, e)
        return []


def create_alias_edge(
    driver,
    canonical_name: str,
    alias_name: str,
    entity_type: str,
) -> bool:
    """Create an Alias node and HAS_ALIAS edge from the canonical entity."""
    label = _sanitize_label(entity_type)
    query = f"""
        MATCH (e:Entity:{label} {{name: $canonical_name}})
        MERGE (a:Alias {{alias_name: $alias_name}})
        MERGE (e)-[r:HAS_ALIAS]->(a)
        RETURN count(r) AS cnt
    """

    try:
        with driver.session() as session:
            session.run(
                query,
                canonical_name=canonical_name,
                alias_name=alias_name,
            )
        return True
    except Exception as e:
        logger.warning(
            "create_alias_edge failed %s→%s: %s",
            canonical_name, alias_name, e,
        )
        return False


# ---------------------------------------------------------------------------
# Bootstrap — indexes and constraints
# ---------------------------------------------------------------------------

def ensure_indexes(driver) -> None:
    """Create required Neo4j indexes and constraints if they don't exist.

    Called at API startup to ensure the graph is ready for queries.
    Raises on failure so the startup health check can catch it.
    """
    statements = [
        # Fulltext index for entity name search (canonicalization + retrieval)
        """
        CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
        FOR (n:Entity) ON EACH [n.name]
        """,
        # Uniqueness constraint on Document node
        """
        CREATE CONSTRAINT document_id_unique IF NOT EXISTS
        FOR (d:Document) REQUIRE d.document_id IS UNIQUE
        """,
        # Index on ChunkRef.chunk_id for fast lookups
        """
        CREATE INDEX chunk_ref_chunk_id IF NOT EXISTS
        FOR (c:ChunkRef) ON (c.chunk_id)
        """,
    ]

    with driver.session() as session:
        for stmt in statements:
            session.run(stmt.strip())

    logger.info("Neo4j indexes and constraints ensured")
