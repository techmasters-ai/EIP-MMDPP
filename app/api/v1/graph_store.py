"""Graph (AGE) — direct entity/relationship ingest and Cypher query endpoints."""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.schemas.graph_store import (
    GraphEntityIngest,
    GraphIngestResponse,
    GraphQueryRequest,
    GraphRelationshipIngest,
)
from app.schemas.retrieval import QueryResultItem

router = APIRouter(tags=["graph"])
logger = logging.getLogger(__name__)


@router.post("/graph/ingest/entity", response_model=GraphIngestResponse)
async def ingest_entity(
    body: GraphEntityIngest,
    db: AsyncSession = Depends(get_async_session),
) -> GraphIngestResponse:
    """Create or update an entity node in the AGE knowledge graph.

    Note: Graph entity mutations require dual-curator approval via governance.
    This endpoint creates the node directly for authorized users.
    """
    from app.services.graph import upsert_node

    # Run sync graph operation via async session's sync connection
    def _upsert(sync_session):
        return upsert_node(
            session=sync_session,
            entity_type=body.entity_type,
            name=body.name,
            artifact_id="direct_ingest",
            confidence=1.0,
            properties=body.properties,
        )

    node_id = await db.run_sync(lambda s: _upsert(s))

    if node_id:
        return GraphIngestResponse(status="created", node_id=node_id)
    return GraphIngestResponse(status="failed", message="Could not create node")


@router.post("/graph/ingest/relationship", response_model=GraphIngestResponse)
async def ingest_relationship(
    body: GraphRelationshipIngest,
    db: AsyncSession = Depends(get_async_session),
) -> GraphIngestResponse:
    """Create or update a relationship edge in the AGE knowledge graph."""
    from app.services.graph import upsert_relationship

    def _upsert(sync_session):
        return upsert_relationship(
            session=sync_session,
            from_name=body.from_entity,
            from_type=body.from_type,
            to_name=body.to_entity,
            to_type=body.to_type,
            rel_type=body.relationship_type,
            artifact_id="direct_ingest",
            confidence=1.0,
        )

    ok = await db.run_sync(lambda s: _upsert(s))

    if ok:
        return GraphIngestResponse(status="created")
    return GraphIngestResponse(status="failed", message="Could not create relationship")


@router.post("/graph/query", response_model=list[QueryResultItem])
async def query_graph(
    body: GraphQueryRequest,
    db: AsyncSession = Depends(get_async_session),
) -> list[QueryResultItem]:
    """Search the AGE knowledge graph by entity name and return neighborhood."""
    from app.services.graph import search_nodes_async, get_neighborhood_async

    # Search for matching nodes
    matches = await search_nodes_async(db, body.query, limit=body.top_k)

    results: list[QueryResultItem] = []

    for match in matches:
        node = match.get("node", {})
        if not isinstance(node, dict):
            continue

        entity_type = match.get("entity_type", "UNKNOWN")
        name = node.get("name", "")

        # Get neighborhood for each matched entity
        neighbors = await get_neighborhood_async(
            db, name, hop_count=body.hop_count, limit=10
        )

        results.append(
            QueryResultItem(
                score=node.get("confidence", 0.5),
                modality="graph_node",
                content_text=name,
                page_number=None,
                classification="UNCLASSIFIED",
                context={
                    "entity_type": entity_type,
                    "entity": node,
                    "neighbors": neighbors[:5],
                },
            )
        )

    return results[:body.top_k]
