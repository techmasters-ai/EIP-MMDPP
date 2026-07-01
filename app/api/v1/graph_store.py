"""Graph store -- entity/relationship ingest and deterministic query endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session, get_graph_store
from app.services.arcadedb_graph import (
    GRAPH_VIEW_EDGE_TYPES,
    GRAPH_VIEW_NODE_TYPES,
)
from app.schemas.graph_store import (
    GraphEntityIngest,
    GraphIngestResponse,
    GraphNeighborhoodRequest,
    GraphNeighborhoodResponse,
    GraphQueryRequest,
    GraphRelationshipIngest,
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

        # Attach source document evidence for data lineage — traverse
        # entity → EXTRACTED_FROM → chunk to find source documents + pages.
        evidence_chunks = await graph_store.get_entity_evidence_chunks(
            match.node_id, limit=3,
        )
        source_docs: list[dict] = []
        first_doc_id = None
        first_doc_name = None
        first_page = None
        first_text = None
        highest_class = "UNCLASSIFIED"
        class_rank = {"UNCLASSIFIED": 0, "CUI": 1, "FOUO": 2, "SECRET": 3, "TOP SECRET": 4}
        for chunk in evidence_chunks:
            doc_id = chunk.get("document_id", "")
            page = chunk.get("page_number")
            chunk_class = chunk.get("classification", "UNCLASSIFIED")
            chunk_text = chunk.get("text") or chunk.get("chunk_text") or ""
            if first_doc_id is None and doc_id:
                first_doc_id = doc_id
                first_page = page
                first_text = chunk_text[:500] if chunk_text else None
            if class_rank.get(chunk_class, 0) > class_rank.get(highest_class, 0):
                highest_class = chunk_class
            source_docs.append({
                "document_id": doc_id,
                "page_number": page,
                "classification": chunk_class,
                "chunk_text_preview": chunk_text[:200] if chunk_text else "",
            })

        # Surface flat entity provenance stored by _build_node_record (C.1).
        # Uniform wire shape: entity hits carry evidence_ids/page_numbers
        # just like text-chunk hits, so UI consumers need no branching.
        entity_evidence_ids: list[str] = match.properties.get("_evidence_ids") or []
        entity_page_numbers: list[int] = match.properties.get("_page_numbers") or []

        results.append(
            QueryResultItem(
                document_id=first_doc_id,
                score=match.extraction_confidence or 0.5,
                modality="graph_node",
                content_text=first_text or name,
                page_number=first_page,
                classification=highest_class,
                sources=source_docs if source_docs else None,
                evidence_ids=entity_evidence_ids,
                page_numbers=entity_page_numbers,
                # self_refs intentionally empty for entity hits — entities
                # span multiple chunks, not a single DoclingDocument element.
                context={
                    "entity_type": entity_type,
                    "entity_name": name,
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

    # Apply the curated Graph Explorer view filter: keep the ontology +
    # document-structure node/edge types, drop raw document chunks and their
    # layout-adjacency edges (SAME_PAGE/NEXT_CHUNK/...).
    result = await graph_store.get_neighborhood_graph(
        entity.node_id,
        depth=body.hop_count,
        rel_types=list(GRAPH_VIEW_EDGE_TYPES),
        node_types=GRAPH_VIEW_NODE_TYPES,
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


# Legacy /graph/system-* endpoints removed in the flat-schema profile
# refactor (spec §4.13). Replaced by:
#   POST /v1/query-profiles/search/section   (kind=section_properties)
#   POST /v1/query-profiles/search/dossier   (kind=dossier)
# See alembic/versions/0018_starter_profiles_to_section_properties.py
# for the registry-data migration.
