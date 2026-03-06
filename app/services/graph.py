"""Apache AGE graph operations.

Provides synchronous helpers for Celery workers and async helpers for
FastAPI routes. All Cypher queries target the 'eip_kg' graph created during
database initialization.

AGE requires:
  LOAD 'age';
  SET search_path = ag_catalog, "$user", public;
before any Cypher function can be called in a session.

Parameter binding: AGE accepts a third agtype argument to cypher() for
named parameters accessed via $param_name in Cypher. We use this to avoid
SQL injection from entity names extracted by the NER pipeline.

Driver-specific syntax for the agtype parameter:
  - psycopg2 (sync/Celery):  :params::agtype  → sends $1::agtype server-side
  - asyncpg  (async/FastAPI): CAST(:params AS agtype) → asyncpg can't parse ::

Colon escaping: Cypher uses colons for label syntax (n:LABEL) and map
keys ({key: $val}). SQLAlchemy's text() treats :word as a bind parameter.
We escape colons inside the Cypher $$...$$ block with \\: so SQLAlchemy
passes them through as literal colons to PostgreSQL.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

GRAPH_NAME = "eip_kg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape_cypher(cypher: str) -> str:
    r"""Escape colons in Cypher text so SQLAlchemy text() doesn't mis-parse them.

    Cypher uses colons for label syntax (n:LABEL) and map literals
    ({key: $val}). SQLAlchemy's text() interprets :word as a named bind
    parameter. Replacing : with \: tells SQLAlchemy to emit a literal colon.
    The real bind parameter (:params outside $$...$$) is unaffected.
    """
    return cypher.replace(":", r"\:")


def _sanitize_label(label: str) -> str:
    """Sanitize a string for use as an AGE node/edge label.

    Allows only alphanumeric characters and underscores. AGE labels must
    match [A-Za-z_][A-Za-z0-9_]*.
    """
    import re
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", label)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "UNKNOWN"


def _parse_agtype(value: Any) -> Any:
    """Parse an agtype value returned from PostgreSQL.

    AGE returns agtype values as strings. This helper handles the common cases:
    - JSON objects/arrays → parsed dict/list
    - Quoted strings → stripped string
    - Null → None
    - Numbers → int/float
    """
    if value is None:
        return None

    s = str(value).strip()

    if s == "null":
        return None

    # Try JSON parse first (handles objects, arrays, numbers, booleans)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip surrounding quotes from plain strings
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]

    return s


# ---------------------------------------------------------------------------
# Session setup — must be called once per DB session before any graph ops
# ---------------------------------------------------------------------------

def setup_age_session(session: Session) -> None:
    """Enable AGE and set the required search_path for the session."""
    session.execute(text("LOAD 'age'"))
    session.execute(text("SET search_path = ag_catalog, \"$user\", public"))


# ---------------------------------------------------------------------------
# Node upsert
# ---------------------------------------------------------------------------

def upsert_node(
    session: Session,
    entity_type: str,
    name: str,
    artifact_id: str,
    confidence: float,
    properties: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Create or update a graph node for an extracted entity.

    Uses MERGE on (entity_type {name, artifact_id}) to be idempotent.
    Returns the node's id property (UUID string), or None on failure.
    """
    node_id = str(uuid.uuid4())
    props = properties or {}
    props.update({
        "id": node_id,
        "name": name,
        "artifact_id": artifact_id,
        "confidence": confidence,
    })

    params_json = json.dumps({"props": props, "name": name})

    cypher = f"""
        MERGE (n:{_sanitize_label(entity_type)} {{name: $props.name}})
        ON CREATE SET
            n.id = $props.id,
            n.artifact_id = $props.artifact_id,
            n.confidence = $props.confidence,
            n.properties = $props
        ON MATCH SET
            n.last_artifact_id = $props.artifact_id,
            n.confidence = CASE
                WHEN n.confidence < $props.confidence THEN $props.confidence
                ELSE n.confidence
            END
        RETURN n.id AS node_id
    """

    try:
        setup_age_session(session)
        with session.begin_nested():
            result = session.execute(
                text(f"""
                    SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                        :params::agtype) AS (node_id agtype)
                """),
                {"params": params_json},
            )
            row = result.fetchone()
        if row:
            returned_id = str(row[0]).strip('"')
            return returned_id if returned_id != "null" else node_id
        return node_id
    except Exception as e:
        logger.warning("upsert_node failed for %s/%s: %s", entity_type, name, e)
        return None


def upsert_relationship(
    session: Session,
    from_name: str,
    from_type: str,
    to_name: str,
    to_type: str,
    rel_type: str,
    artifact_id: str,
    confidence: float,
) -> bool:
    """Create or update a directed relationship between two nodes.

    Matches source and target by name+label. Returns True on success.
    """
    params_json = json.dumps({
        "from_name": from_name,
        "to_name": to_name,
        "artifact_id": artifact_id,
        "confidence": confidence,
    })

    from_label = _sanitize_label(from_type)
    to_label = _sanitize_label(to_type)
    rel_label = _sanitize_label(rel_type)

    cypher = f"""
        MATCH (a:{from_label} {{name: $from_name}})
        MATCH (b:{to_label} {{name: $to_name}})
        MERGE (a)-[r:{rel_label} {{artifact_id: $artifact_id}}]->(b)
        ON MATCH SET r.confidence = CASE
            WHEN r.confidence < $confidence THEN $confidence
            ELSE r.confidence
        END
        RETURN count(r) AS count
    """

    try:
        setup_age_session(session)
        with session.begin_nested():
            session.execute(
                text(f"""
                    SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                        :params::agtype) AS (count agtype)
                """),
                {"params": params_json},
            )
        return True
    except Exception as e:
        logger.warning(
            "upsert_relationship failed %s→%s [%s]: %s", from_name, to_name, rel_type, e
        )
        return False


# ---------------------------------------------------------------------------
# Node queries
# ---------------------------------------------------------------------------

def search_nodes_by_name(
    session: Session,
    search_term: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search graph nodes whose name contains the search term (case-insensitive)."""
    setup_age_session(session)

    params_json = json.dumps({"search": search_term.lower(), "limit": limit})

    cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS $search
        RETURN n, label(n) AS entity_type
        LIMIT $limit
    """

    try:
        result = session.execute(
            text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                    :params::agtype) AS (node agtype, entity_type agtype)
            """),
            {"params": params_json},
        )
        return [
            {
                "node": _parse_agtype(row[0]),
                "entity_type": _parse_agtype(row[1]),
            }
            for row in result.fetchall()
        ]
    except Exception as e:
        logger.warning("search_nodes_by_name failed for '%s': %s", search_term, e)
        return []


def get_node_neighborhood(
    session: Session,
    node_name: str,
    hop_count: int = 2,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get a node and its neighborhood up to hop_count hops.

    Returns a flat list of {node, rel_type, neighbor} dicts.
    """
    setup_age_session(session)

    if hop_count < 1:
        hop_count = 1
    if hop_count > 4:
        hop_count = 4  # AGE can struggle with deep traversals

    params_json = json.dumps({"name": node_name, "limit": limit})

    # Build variable-length path pattern (AGE supports *1..N syntax)
    path_pattern = f"*1..{hop_count}"

    cypher = f"""
        MATCH (start {{name: $name}})
        OPTIONAL MATCH path = (start)-[r{path_pattern}]-(neighbor)
        RETURN start, type(r) AS rel_type, neighbor
        LIMIT $limit
    """

    try:
        result = session.execute(
            text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                    :params::agtype) AS (node agtype, rel_type agtype, neighbor agtype)
            """),
            {"params": params_json},
        )
        return [
            {
                "node": _parse_agtype(row[0]),
                "rel_type": _parse_agtype(row[1]),
                "neighbor": _parse_agtype(row[2]),
            }
            for row in result.fetchall()
        ]
    except Exception as e:
        logger.warning("get_node_neighborhood failed for '%s': %s", node_name, e)
        return []


def get_graph_stats(session: Session) -> dict[str, int]:
    """Return total node and edge counts for the knowledge graph."""
    setup_age_session(session)

    cypher_nodes = "MATCH (n) RETURN count(n) AS cnt"
    cypher_edges = "MATCH ()-[r]->() RETURN count(r) AS cnt"

    stats: dict[str, int] = {"nodes": 0, "edges": 0}
    try:
        result = session.execute(
            text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $${cypher_nodes}$$)
                AS (cnt agtype)
            """)
        )
        row = result.fetchone()
        if row:
            stats["nodes"] = int(_parse_agtype(row[0]) or 0)

        result = session.execute(
            text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $${cypher_edges}$$)
                AS (cnt agtype)
            """)
        )
        row = result.fetchone()
        if row:
            stats["edges"] = int(_parse_agtype(row[0]) or 0)
    except Exception as e:
        logger.warning("get_graph_stats failed: %s", e)

    return stats


# ---------------------------------------------------------------------------
# Document/Chunk structural graph operations
# ---------------------------------------------------------------------------

def upsert_document_node(
    session: Session,
    document_id: str,
    title: str,
    properties: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Create or update a DOCUMENT node in the knowledge graph."""
    props = properties or {}
    props.update({"document_id": document_id, "title": title})
    params_json = json.dumps({"props": props, "doc_id": document_id})

    cypher = """
        MERGE (d:DOCUMENT {document_id: $doc_id})
        ON CREATE SET d = $props
        ON MATCH SET d.title = $props.title
        RETURN d.document_id AS doc_id
    """

    try:
        setup_age_session(session)
        with session.begin_nested():
            session.execute(
                text(f"""
                    SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                        :params::agtype) AS (doc_id agtype)
                """),
                {"params": params_json},
            )
        return document_id
    except Exception as e:
        logger.warning("upsert_document_node failed for %s: %s", document_id, e)
        return None


def upsert_chunk_ref_node(
    session: Session,
    chunk_id: str,
    chunk_type: str,
) -> Optional[str]:
    """Create or update a CHUNK_REF node linking to a text_chunk or image_chunk."""
    params_json = json.dumps({"chunk_id": chunk_id, "chunk_type": chunk_type})

    cypher = """
        MERGE (c:CHUNK_REF {chunk_id: $chunk_id})
        ON CREATE SET c.chunk_type = $chunk_type
        RETURN c.chunk_id AS cid
    """

    try:
        setup_age_session(session)
        with session.begin_nested():
            session.execute(
                text(f"""
                    SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                        :params::agtype) AS (cid agtype)
                """),
                {"params": params_json},
            )
        return chunk_id
    except Exception as e:
        logger.warning("upsert_chunk_ref_node failed for %s: %s", chunk_id, e)
        return None


def create_structural_edge(
    session: Session,
    from_id: str,
    to_id: str,
    edge_type: str,
) -> bool:
    """Create a structural edge between DOCUMENT/CHUNK_REF nodes.

    Supports: CONTAINS_TEXT, CONTAINS_IMAGE, SAME_PAGE.
    These are deterministic edges and do NOT require dual-curator approval.
    """
    label = _sanitize_label(edge_type)
    params_json = json.dumps({"from_id": from_id, "to_id": to_id})

    # Determine which node types to match based on edge type
    if edge_type == "CONTAINS_TEXT":
        cypher = f"""
            MATCH (d:DOCUMENT {{document_id: $from_id}})
            MATCH (c:CHUNK_REF {{chunk_id: $to_id}})
            MERGE (d)-[r:{label}]->(c)
            RETURN count(r) AS cnt
        """
    elif edge_type == "CONTAINS_IMAGE":
        cypher = f"""
            MATCH (d:DOCUMENT {{document_id: $from_id}})
            MATCH (c:CHUNK_REF {{chunk_id: $to_id}})
            MERGE (d)-[r:{label}]->(c)
            RETURN count(r) AS cnt
        """
    elif edge_type == "SAME_PAGE":
        cypher = f"""
            MATCH (a:CHUNK_REF {{chunk_id: $from_id}})
            MATCH (b:CHUNK_REF {{chunk_id: $to_id}})
            MERGE (a)-[r:{label}]->(b)
            RETURN count(r) AS cnt
        """
    else:
        # Generic: match any node with id properties
        cypher = f"""
            MATCH (a {{chunk_id: $from_id}})
            MATCH (b {{chunk_id: $to_id}})
            MERGE (a)-[r:{label}]->(b)
            RETURN count(r) AS cnt
        """

    try:
        setup_age_session(session)
        with session.begin_nested():
            session.execute(
                text(f"""
                    SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                        :params::agtype) AS (cnt agtype)
                """),
                {"params": params_json},
            )
        return True
    except Exception as e:
        logger.warning(
            "create_structural_edge failed %s→%s [%s]: %s", from_id, to_id, edge_type, e
        )
        return False


def create_entity_chunk_edge(
    session: Session,
    entity_name: str,
    entity_type: str,
    chunk_id: str,
) -> bool:
    """Create an EXTRACTED_FROM edge from an ontology entity to a CHUNK_REF node.

    Links entities discovered during graph extraction back to the text chunks
    they were extracted from, enabling chunk-level provenance traversal.
    """
    label = _sanitize_label(entity_type)
    params_json = json.dumps({
        "entity_name": entity_name,
        "chunk_id": chunk_id,
    })

    cypher = f"""
        MATCH (e:{label} {{name: $entity_name}})
        MATCH (c:CHUNK_REF {{chunk_id: $chunk_id}})
        MERGE (e)-[r:EXTRACTED_FROM]->(c)
        RETURN count(r) AS cnt
    """

    try:
        setup_age_session(session)
        with session.begin_nested():
            session.execute(
                text(f"""
                    SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                        :params::agtype) AS (cnt agtype)
                """),
                {"params": params_json},
            )
        return True
    except Exception as e:
        logger.warning(
            "create_entity_chunk_edge failed %s[%s]→%s: %s",
            entity_name, entity_type, chunk_id, e,
        )
        return False


# ---------------------------------------------------------------------------
# Async versions (FastAPI context)
# ---------------------------------------------------------------------------

async def setup_age_session_async(session) -> None:
    """Enable AGE for an async SQLAlchemy session."""
    await session.execute(text("LOAD 'age'"))
    await session.execute(text("SET search_path = ag_catalog, \"$user\", public"))


async def search_nodes_async(
    session,
    search_term: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Async version of search_nodes_by_name for FastAPI routes."""
    await setup_age_session_async(session)

    params_json = json.dumps({"search": search_term.lower(), "limit": limit})

    cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS $search
        RETURN n, label(n) AS entity_type
        LIMIT $limit
    """

    try:
        result = await session.execute(
            text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                    CAST(:params AS agtype)) AS (node agtype, entity_type agtype)
            """),
            {"params": params_json},
        )
        return [
            {
                "node": _parse_agtype(row[0]),
                "entity_type": _parse_agtype(row[1]),
            }
            for row in result.fetchall()
        ]
    except Exception as e:
        logger.warning("search_nodes_async failed for '%s': %s", search_term, e)
        return []


async def get_neighborhood_async(
    session,
    node_name: str,
    hop_count: int = 2,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Async version of get_node_neighborhood for FastAPI routes."""
    await setup_age_session_async(session)

    hop_count = max(1, min(hop_count, 4))
    params_json = json.dumps({"name": node_name, "limit": limit})
    path_pattern = f"*1..{hop_count}"

    cypher = f"""
        MATCH (start {{name: $name}})
        OPTIONAL MATCH path = (start)-[r{path_pattern}]-(neighbor)
        RETURN start, type(r) AS rel_type, neighbor
        LIMIT $limit
    """

    try:
        result = await session.execute(
            text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                    CAST(:params AS agtype)) AS (node agtype, rel_type agtype, neighbor agtype)
            """),
            {"params": params_json},
        )
        return [
            {
                "node": _parse_agtype(row[0]),
                "rel_type": _parse_agtype(row[1]),
                "neighbor": _parse_agtype(row[2]),
            }
            for row in result.fetchall()
        ]
    except Exception as e:
        logger.warning("get_neighborhood_async failed for '%s': %s", node_name, e)
        return []


async def get_ontology_linked_chunks_async(
    session,
    chunk_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find chunks linked via ontology relationships.

    Path: CHUNK_REF <-[EXTRACTED_FROM]- Entity -[ontology_rel]- RelatedEntity
          -[EXTRACTED_FROM]-> CHUNK_REF

    Returns list of dicts with keys:
        target_chunk_id, target_chunk_type, rel_type, entity_name, related_name
    """
    await setup_age_session_async(session)

    params_json = json.dumps({"chunk_id": chunk_id, "limit": limit})

    cypher = """
        MATCH (src:CHUNK_REF {chunk_id: $chunk_id})<-[:EXTRACTED_FROM]-(entity)-[r]-(related)-[:EXTRACTED_FROM]->(target:CHUNK_REF)
        WHERE target.chunk_id <> $chunk_id
        RETURN target.chunk_id, target.chunk_type, type(r), entity.name, related.name
        LIMIT $limit
    """

    try:
        result = await session.execute(
            text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                    CAST(:params AS agtype)) AS (
                        target_chunk_id agtype,
                        target_chunk_type agtype,
                        rel_type agtype,
                        entity_name agtype,
                        related_name agtype
                    )
            """),
            {"params": params_json},
        )
        return [
            {
                "target_chunk_id": str(_parse_agtype(row[0])).strip('"'),
                "target_chunk_type": str(_parse_agtype(row[1])).strip('"'),
                "rel_type": str(_parse_agtype(row[2])).strip('"'),
                "entity_name": str(_parse_agtype(row[3])).strip('"'),
                "related_name": str(_parse_agtype(row[4])).strip('"'),
            }
            for row in result.fetchall()
        ]
    except Exception as e:
        logger.warning(
            "get_ontology_linked_chunks_async failed for chunk %s: %s",
            chunk_id, e,
        )
        return []
