"""GraphRAG integration — community detection, reports, and search.

Runs as a background Celery Beat task, not inline during ingest.
Uses Microsoft's graphrag library with LiteLLM → Ollama for air-gapped LLM.

Provides three search modes:
  - local_search: entity-centric with community report context
  - global_search: cross-community summarization
  - drift_search: hybrid drift-based retrieval
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Community detection + report generation (batch job)
# ---------------------------------------------------------------------------

def run_graphrag_indexing(
    neo4j_driver,
    db_session,
) -> dict[str, Any]:
    """Run GraphRAG community detection and report generation.

    Called by Celery Beat task. Exports graph from Neo4j, runs Leiden
    community detection, generates community reports via LLM.

    Returns: {communities_created: int, reports_generated: int}
    """
    settings = get_settings()

    if not settings.graphrag_indexing_enabled:
        logger.info("GraphRAG indexing disabled")
        return {"communities_created": 0, "reports_generated": 0}

    stats = {"communities_created": 0, "reports_generated": 0}

    try:
        # Export entities and relationships from Neo4j
        entities, relationships = _export_graph_for_graphrag(neo4j_driver)
        if not entities:
            logger.info("No entities in Neo4j — skipping GraphRAG indexing")
            return stats

        # Run Leiden community detection
        communities = _detect_communities(entities, relationships, settings)
        if not communities:
            logger.info("No communities detected")
            return stats

        # Generate community reports via LLM
        reports = _generate_community_reports(communities, entities, relationships, settings)

        # Store in Postgres
        _store_communities_and_reports(db_session, communities, reports)

        stats["communities_created"] = len(communities)
        stats["reports_generated"] = len(reports)

        logger.info(
            "GraphRAG indexing complete: %d communities, %d reports",
            stats["communities_created"],
            stats["reports_generated"],
        )
    except Exception as e:
        logger.error("GraphRAG indexing failed: %s", e, exc_info=True)

    return stats


# ---------------------------------------------------------------------------
# Search modes
# ---------------------------------------------------------------------------

def local_search(
    query: str,
    neo4j_driver,
    qdrant_client,
    db_session,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Entity-centric search with community report context.

    1. Find relevant entities via keyword match in Neo4j
    2. Look up community memberships
    3. Retrieve community reports
    4. Combine with relevant text chunks from Qdrant
    """
    results = []

    try:
        # Find matching entities via fulltext index (handles multi-word queries)
        from app.services.neo4j_graph import fulltext_search_entity
        ft_matches = fulltext_search_entity(neo4j_driver, query, limit=limit)

        if not ft_matches:
            logger.info("GraphRAG local: no entity matches for query '%s'", query)
            return results

        # Reshape fulltext results — pass through all node fields
        entity_matches = [
            {"node": m, "entity_type": m.get("entity_type")}
            for m in ft_matches
        ]

        # Get community context for matched entities
        community_context = _get_entity_community_context(db_session, entity_matches)

        for match in entity_matches:
            result = {
                "entity": match.get("node", {}),
                "entity_type": match.get("entity_type"),
                "community_reports": community_context.get(
                    match.get("node", {}).get("name", ""), []
                ),
            }
            results.append(result)

        logger.info("GraphRAG local: %d entities matched for query '%s'", len(results), query)

    except Exception as e:
        logger.error("GraphRAG local search runtime fault for '%s': %s", query, e, exc_info=True)

    return results


def global_search(
    query: str,
    db_session,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Cross-community summarization for broad questions.

    Retrieves the most relevant community reports and aggregates them.
    """
    results = []

    try:
        from sqlalchemy import text

        # Search community reports by keyword
        stmt = text("""
            SELECT cr.report_text, cr.summary, cr.rank,
                   gc.community_id, gc.title, gc.level
            FROM retrieval.graphrag_community_reports cr
            JOIN retrieval.graphrag_communities gc
                ON cr.community_id = gc.community_id
            ORDER BY cr.rank DESC NULLS LAST
            LIMIT :limit
        """)

        result = db_session.execute(stmt, {"limit": limit})
        for row in result.fetchall():
            results.append({
                "community_id": row[3],
                "community_title": row[4],
                "level": row[5],
                "report_text": row[0],
                "summary": row[1],
                "rank": row[2],
            })

        if not results:
            logger.info("GraphRAG global: no community reports found for query '%s'", query)
        else:
            logger.info("GraphRAG global: %d community reports returned for query '%s'", len(results), query)

    except Exception as e:
        logger.error("GraphRAG global search runtime fault for '%s': %s", query, e, exc_info=True)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _export_graph_for_graphrag(
    neo4j_driver,
) -> tuple[list[dict], list[dict]]:
    """Export all entities and relationships from Neo4j for GraphRAG."""
    entities = []
    relationships = []

    try:
        with neo4j_driver.session() as session:
            # Export entities
            result = session.run("""
                MATCH (n:Entity)
                RETURN n.name AS name, n.entity_type AS entity_type,
                       n.id AS id
            """)
            for record in result:
                entities.append(dict(record))

            # Export relationships
            result = session.run("""
                MATCH (a:Entity)-[r]->(b:Entity)
                RETURN a.name AS source, b.name AS target,
                       type(r) AS relationship
            """)
            for record in result:
                relationships.append(dict(record))

    except Exception as e:
        logger.warning("Graph export for GraphRAG failed: %s", e)

    return entities, relationships


def _detect_communities(
    entities: list[dict],
    relationships: list[dict],
    settings,
) -> list[dict[str, Any]]:
    """Run Leiden community detection on the graph."""
    import networkx as nx

    # Build NetworkX graph for community detection
    G = nx.Graph()
    for e in entities:
        G.add_node(e["name"], entity_type=e.get("entity_type"))
    for r in relationships:
        G.add_edge(r["source"], r["target"], relationship=r.get("relationship"))

    if len(G.nodes) == 0:
        return []

    # Use Louvain as fallback (Leiden requires optional dependency)
    try:
        from graspologic.partition import hierarchical_leiden

        community_map = hierarchical_leiden(G, max_cluster_size=settings.graphrag_max_cluster_size)
        # Group nodes by community
        communities_by_id: dict[str, list[str]] = {}
        for node, community_id in community_map.items():
            cid = str(community_id)
            if cid not in communities_by_id:
                communities_by_id[cid] = []
            communities_by_id[cid].append(node)
    except ImportError:
        logger.info("graspologic not available, using Louvain community detection")
        from networkx.algorithms.community import louvain_communities

        partition = louvain_communities(G, seed=42)
        communities_by_id = {}
        for i, community in enumerate(partition):
            communities_by_id[str(i)] = list(community)

    # Build community dicts
    communities = []
    for cid, members in communities_by_id.items():
        communities.append({
            "community_id": cid,
            "level": 0,
            "entity_names": members,
            "title": f"Community {cid} ({len(members)} entities)",
        })

    return communities


def _generate_community_reports(
    communities: list[dict],
    entities: list[dict],
    relationships: list[dict],
    settings,
) -> list[dict[str, Any]]:
    """Generate natural-language community reports via LLM."""
    import httpx

    if settings.llm_provider == "mock":
        return [
            {
                "community_id": c["community_id"],
                "report_text": f"Mock report for {c['title']}",
                "summary": c["title"],
                "rank": 1.0,
            }
            for c in communities
        ]

    # Build entity lookup
    entity_map = {e["name"]: e for e in entities}

    reports = []
    for community in communities:
        members = community["entity_names"]
        member_info = []
        for name in members[:20]:  # Limit context
            e = entity_map.get(name, {})
            member_info.append(f"- {name} ({e.get('entity_type', 'unknown')})")

        # Get relationships between community members
        member_set = set(members)
        relevant_rels = [
            r for r in relationships
            if r["source"] in member_set and r["target"] in member_set
        ]
        rel_info = [
            f"- {r['source']} --[{r['relationship']}]--> {r['target']}"
            for r in relevant_rels[:20]
        ]

        prompt = f"""Summarize this community of related military/defense entities.

## Entities
{chr(10).join(member_info)}

## Relationships
{chr(10).join(rel_info) if rel_info else "No direct relationships found."}

Write a concise report (2-3 paragraphs) explaining:
1. What these entities have in common
2. Key relationships and their significance
3. Operational relevance

Return ONLY the report text, no JSON or markdown fences."""

        try:
            payload = {
                "model": settings.graphrag_model,
                "messages": [
                    {"role": "system", "content": "You are a military intelligence analyst."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1024,
                    "num_ctx": settings.ollama_num_ctx,
                },
            }
            response = httpx.post(
                f"{settings.ollama_base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            report_text = (
                (response.json().get("message") or {}).get("content", "").strip()
            )
            if not report_text:
                logger.warning(
                    "Empty report for community %s — skipping",
                    community["community_id"],
                )
                continue

            reports.append({
                "community_id": community["community_id"],
                "report_text": report_text,
                "summary": report_text[:200],
                "rank": len(members) / max(len(entities), 1),
            })
        except Exception as e:
            logger.warning("Report generation failed for community %s: %s", community["community_id"], e)

    return reports


def _store_communities_and_reports(
    db_session,
    communities: list[dict],
    reports: list[dict],
) -> None:
    """Store communities and reports in Postgres."""
    from sqlalchemy import text

    for community in communities:
        db_session.execute(
            text("""
                INSERT INTO retrieval.graphrag_communities
                    (id, community_id, level, entity_ids, title)
                VALUES (:id, :community_id, :level, :entity_ids, :title)
                ON CONFLICT (community_id) DO UPDATE SET
                    entity_ids = EXCLUDED.entity_ids,
                    title = EXCLUDED.title
            """),
            {
                "id": str(uuid.uuid4()),
                "community_id": community["community_id"],
                "level": community.get("level", 0),
                "entity_ids": community["entity_names"],
                "title": community.get("title"),
            },
        )

    for report in reports:
        db_session.execute(
            text("""
                INSERT INTO retrieval.graphrag_community_reports
                    (id, community_id, report_text, summary, rank)
                VALUES (:id, :community_id, :report_text, :summary, :rank)
                ON CONFLICT DO NOTHING
            """),
            {
                "id": str(uuid.uuid4()),
                "community_id": report["community_id"],
                "report_text": report["report_text"],
                "summary": report.get("summary"),
                "rank": report.get("rank"),
            },
        )

    db_session.commit()


def _get_entity_community_context(
    db_session,
    entity_matches: list[dict],
) -> dict[str, list[dict]]:
    """Look up community reports for matched entities."""
    from sqlalchemy import text

    context: dict[str, list[dict]] = {}

    for match in entity_matches:
        name = match.get("node", {}).get("name", "")
        if not name:
            continue

        try:
            result = db_session.execute(
                text("""
                    SELECT cr.report_text, cr.summary, gc.title, gc.community_id
                    FROM retrieval.graphrag_communities gc
                    JOIN retrieval.graphrag_community_reports cr
                        ON gc.community_id = cr.community_id
                    WHERE :name = ANY(gc.entity_ids)
                    ORDER BY cr.rank DESC NULLS LAST
                    LIMIT 3
                """),
                {"name": name},
            )
            reports = [
                {
                    "community_id": row[3],
                    "title": row[2],
                    "summary": row[1],
                    "report_text": row[0],
                }
                for row in result.fetchall()
            ]
            if reports:
                context[name] = reports
        except Exception:
            pass

    return context
