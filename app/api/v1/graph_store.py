"""Graph (Neo4j) — direct entity/relationship ingest and Cypher query endpoints."""

import logging

from fastapi import APIRouter

from app.db.session import get_neo4j_async_driver
from app.schemas.graph_store import (
    GraphEntityIngest,
    GraphIngestResponse,
    GraphNeighborhoodRequest,
    GraphNeighborhoodResponse,
    GraphQueryRequest,
    GraphRelationshipIngest,
)
from app.schemas.retrieval import QueryResultItem

router = APIRouter(tags=["graph"])
logger = logging.getLogger(__name__)


@router.post("/graph/ingest/entity", response_model=GraphIngestResponse)
async def ingest_entity(
    body: GraphEntityIngest,
) -> GraphIngestResponse:
    """Create or update an entity node in the Neo4j knowledge graph.

    Note: Graph entity mutations require dual-curator approval via governance.
    This endpoint creates the node directly for authorized users.
    """
    from app.services.neo4j_graph import upsert_node

    driver = get_neo4j_async_driver()

    # Neo4j async driver uses its own sessions — run sync upsert in executor
    import asyncio
    from app.db.session import get_neo4j_driver

    sync_driver = get_neo4j_driver()
    loop = asyncio.get_event_loop()
    node_id = await loop.run_in_executor(
        None,
        lambda: upsert_node(
            sync_driver,
            entity_type=body.entity_type,
            name=body.name,
            artifact_id="direct_ingest",
            confidence=1.0,
            properties=body.properties,
        ),
    )

    if node_id:
        return GraphIngestResponse(status="created", node_id=node_id)
    return GraphIngestResponse(status="failed", message="Could not create node")


@router.post("/graph/ingest/relationship", response_model=GraphIngestResponse)
async def ingest_relationship(
    body: GraphRelationshipIngest,
) -> GraphIngestResponse:
    """Create or update a relationship edge in the Neo4j knowledge graph."""
    from app.services.neo4j_graph import upsert_relationship

    import asyncio
    from app.db.session import get_neo4j_driver

    sync_driver = get_neo4j_driver()
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(
        None,
        lambda: upsert_relationship(
            sync_driver,
            from_name=body.from_entity,
            from_type=body.from_type,
            to_name=body.to_entity,
            to_type=body.to_type,
            rel_type=body.relationship_type,
            artifact_id="direct_ingest",
            confidence=1.0,
        ),
    )

    if ok:
        return GraphIngestResponse(status="created")
    return GraphIngestResponse(status="failed", message="Could not create relationship")


@router.post("/graph/query", response_model=list[QueryResultItem])
async def query_graph(
    body: GraphQueryRequest,
) -> list[QueryResultItem]:
    """Search the Neo4j knowledge graph by entity name and return neighborhood."""
    from app.services.neo4j_graph import search_nodes_async, get_neighborhood_async

    driver = get_neo4j_async_driver()

    # Search for matching nodes
    matches = await search_nodes_async(driver, body.query, limit=body.top_k)

    results: list[QueryResultItem] = []

    for match in matches:
        node = match.get("node", {})
        if not isinstance(node, dict):
            continue

        entity_type = match.get("entity_type", "UNKNOWN")
        name = node.get("name", "")

        # Get neighborhood for each matched entity
        neighbors = await get_neighborhood_async(
            driver, name, hop_count=body.hop_count, limit=10
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


@router.post("/graph/neighborhood", response_model=GraphNeighborhoodResponse)
async def get_neighborhood(
    body: GraphNeighborhoodRequest,
) -> GraphNeighborhoodResponse:
    """Get an entity's full neighborhood graph for visualization."""
    from app.services.neo4j_graph import get_neighborhood_graph_async

    driver = get_neo4j_async_driver()
    result = await get_neighborhood_graph_async(
        driver, body.entity_name, hop_count=body.hop_count
    )

    return GraphNeighborhoodResponse(
        center=result["center"],
        nodes=result["nodes"],
        edges=result["edges"],
    )
