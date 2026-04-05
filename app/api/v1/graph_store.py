"""Graph store -- entity/relationship ingest and deterministic query endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session, get_graph_store
from app.schemas.graph_store import (
    GraphEntityIngest,
    GraphIngestResponse,
    GraphEntityResult,
    GraphNeighborhoodRequest,
    GraphNeighborhoodResponse,
    GraphQueryRequest,
    GraphRelationshipIngest,
    SystemDossierResponse,
    SystemQueryRequest,
    SystemSectionResponse,
)
from app.schemas.retrieval import QueryResultItem
from app.services.graph_store import NodeRecord, RelationshipRecord

router = APIRouter(tags=["graph"])
logger = logging.getLogger(__name__)


@router.post("/graph/ingest/entity", response_model=GraphIngestResponse)
async def ingest_entity(
    body: GraphEntityIngest,
) -> GraphIngestResponse:
    """Create or update an entity node in the knowledge graph."""
    graph_store = get_graph_store()
    record = NodeRecord(
        entity_type=body.entity_type,
        identity_fields={"name": body.name, "entity_type": body.entity_type},
        name=body.name,
        properties=body.properties or {},
        extraction_confidence=1.0,
    )
    node_id = await graph_store.upsert_node(record)
    if node_id:
        return GraphIngestResponse(status="created", node_id=node_id)
    return GraphIngestResponse(status="failed", message="Could not create node")


@router.post("/graph/ingest/relationship", response_model=GraphIngestResponse)
async def ingest_relationship(
    body: GraphRelationshipIngest,
) -> GraphIngestResponse:
    """Create or update a relationship edge in the knowledge graph."""
    graph_store = get_graph_store()
    record = RelationshipRecord(
        from_type=body.from_type,
        from_identity={"name": body.from_entity, "entity_type": body.from_type},
        to_type=body.to_type,
        to_identity={"name": body.to_entity, "entity_type": body.to_type},
        rel_type=body.relationship_type,
        extraction_confidence=1.0,
    )
    edge_id = await graph_store.upsert_relationship(record)
    if edge_id:
        return GraphIngestResponse(status="created")
    return GraphIngestResponse(status="failed", message="Could not create relationship")


@router.post("/graph/query", response_model=list[QueryResultItem])
async def query_graph(
    body: GraphQueryRequest,
) -> list[QueryResultItem]:
    """Search the knowledge graph by entity name and return neighborhood."""
    graph_store = get_graph_store()

    # Search for matching nodes
    matches = await graph_store.fulltext_search(
        body.query, limit=body.top_k,
    )

    results: list[QueryResultItem] = []

    for match in matches:
        name = match.name
        entity_type = match.entity_type

        # Get neighborhood for each matched entity
        neighbors = await graph_store.get_neighborhood(
            match.node_id, depth=body.hop_count,
        )
        neighbor_dicts = [
            {"name": n.name, "entity_type": n.entity_type, "node_id": n.node_id}
            for n in neighbors[:5]
        ]

        results.append(
            QueryResultItem(
                score=match.extraction_confidence or 0.5,
                modality="graph_node",
                content_text=name,
                page_number=None,
                classification="UNCLASSIFIED",
                context={
                    "entity_type": entity_type,
                    "entity": match.properties,
                    "neighbors": neighbor_dicts,
                },
            )
        )

    return results[:body.top_k]


@router.post("/graph/neighborhood", response_model=GraphNeighborhoodResponse)
async def get_neighborhood(
    body: GraphNeighborhoodRequest,
) -> GraphNeighborhoodResponse:
    """Get an entity's full neighborhood graph for visualization."""
    graph_store = get_graph_store()

    # Resolve the entity by name first
    entity = await graph_store.resolve_root_entity(body.entity_name)
    if entity is None:
        return GraphNeighborhoodResponse(center=None, nodes=[], edges=[])

    result = await graph_store.get_neighborhood_graph(
        entity.node_id, depth=body.hop_count,
    )

    center = {
        "name": entity.name,
        "entity_type": entity.entity_type,
        "id": entity.node_id,
        **entity.properties,
    }

    return GraphNeighborhoodResponse(
        center=center,
        nodes=result.get("nodes", []),
        edges=result.get("edges", []),
    )


# Legacy deterministic dossier endpoints -- kept for backward compatibility.
# New ontology-driven dossier queries should use the query profiles system
# at /query-profiles/search/section and /query-profiles/search/dossier.
@router.post("/graph/system-dossier", response_model=SystemDossierResponse)
async def get_system_dossier(
    body: SystemQueryRequest,
    db: AsyncSession = Depends(get_async_session),
) -> SystemDossierResponse:
    """Return a deterministic, provenance-backed dossier for one system."""
    from app.services.dossier_service import (
        SystemNotFoundError,
        build_system_dossier,
    )

    graph_store = get_graph_store()
    try:
        return await build_system_dossier(graph_store, db, body)
    except SystemNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _system_section(body: SystemQueryRequest, db: AsyncSession, section: str) -> SystemSectionResponse:
    from app.services.dossier_service import SystemNotFoundError, build_section_response
    graph_store = get_graph_store()
    try:
        return await build_section_response(graph_store, db, body, section)
    except SystemNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/graph/system-components", response_model=SystemSectionResponse)
async def get_system_components(body: SystemQueryRequest, db: AsyncSession = Depends(get_async_session)):
    return await _system_section(body, db, "components")


@router.post("/graph/system-rf-parameters", response_model=SystemSectionResponse)
async def get_system_rf_parameters(body: SystemQueryRequest, db: AsyncSession = Depends(get_async_session)):
    return await _system_section(body, db, "rf_parameters")


@router.post("/graph/system-performance", response_model=SystemSectionResponse)
async def get_system_performance(body: SystemQueryRequest, db: AsyncSession = Depends(get_async_session)):
    return await _system_section(body, db, "performance")
