"""Unified retrieval endpoint supporting semantic, graph, hybrid, and cross-modal queries."""

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.schemas.retrieval import QueryMode, QueryRequest, QueryResponse, QueryResultItem

router = APIRouter(tags=["retrieval"])
logger = logging.getLogger(__name__)


@router.post("/retrieval/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    db: AsyncSession = Depends(get_async_session),
) -> QueryResponse:
    """Unified retrieval query supporting all modes.

    - **semantic**: vector similarity search over text chunks (BGE-large HNSW)
    - **graph**: entity name search + neighborhood traversal via Apache AGE
    - **hybrid**: semantic results score-boosted by graph entity co-occurrence
    - **cross_modal**: CLIP-based text → image retrieval
    """
    if body.mode == QueryMode.semantic:
        results = await _semantic_query(db, body)
    elif body.mode == QueryMode.graph:
        results = await _graph_query(db, body)
    elif body.mode == QueryMode.hybrid:
        results = await _hybrid_query(db, body)
    elif body.mode == QueryMode.cross_modal:
        results = await _cross_modal_query(db, body)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown query mode: {body.mode}")

    return QueryResponse(
        query=body.query,
        mode=body.mode,
        results=results,
        total_results=len(results),
    )


# ---------------------------------------------------------------------------
# Semantic query — pgvector HNSW cosine similarity
# ---------------------------------------------------------------------------

async def _semantic_query(
    db: AsyncSession, body: QueryRequest
) -> list[QueryResultItem]:
    from app.services.embedding import embed_query

    query_embedding = embed_query(body.query)
    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    filters = body.filters
    classification_filter = ""
    modality_filter = ""
    if filters and filters.classification:
        # classification values come from an enum / controlled set, not raw user input
        classification_filter = f"AND c.classification = '{filters.classification}'"
    if filters and filters.modalities:
        modalities_str = ",".join(f"'{m}'" for m in filters.modalities)
        modality_filter = f"AND c.modality IN ({modalities_str})"

    sql = text(f"""
        SELECT
            c.id          AS chunk_id,
            c.artifact_id,
            c.chunk_text,
            c.modality,
            c.page_number,
            c.classification,
            1 - (c.embedding <=> '{embedding_literal}'::vector) AS score
        FROM retrieval.chunks c
        WHERE c.embedding IS NOT NULL
        {classification_filter}
        {modality_filter}
        ORDER BY c.embedding <=> '{embedding_literal}'::vector
        LIMIT :top_k
    """)

    result = await db.execute(sql, {"top_k": body.top_k})
    rows = result.mappings().all()

    return [
        QueryResultItem(
            chunk_id=row["chunk_id"],
            artifact_id=row["artifact_id"],
            score=float(row["score"]),
            modality=row["modality"],
            content_text=row["chunk_text"] if body.include_context else None,
            page_number=row["page_number"],
            classification=row["classification"],
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Graph query — Apache AGE openCypher neighborhood traversal
# ---------------------------------------------------------------------------

async def _graph_query(
    db: AsyncSession, body: QueryRequest
) -> list[QueryResultItem]:
    """Search entity names in the knowledge graph and return their neighbors.

    Uses AGE's named-parameter mechanism (:params::agtype) to avoid injection.
    Falls back to empty list if AGE has no data yet.
    """
    from app.services.graph import setup_age_session_async, _parse_agtype

    await setup_age_session_async(db)

    search_term = body.query.strip()
    params_json = json.dumps({"search": search_term.lower(), "limit": body.top_k})

    cypher = """
        MATCH (e)
        WHERE toLower(e.name) CONTAINS $search
        OPTIONAL MATCH (e)-[r*1..2]-(neighbor)
        RETURN e, type(r[0]) AS rel_type, neighbor
        LIMIT $limit
    """

    try:
        result = await db.execute(
            text(f"""
                SELECT * FROM cypher('kg', $${cypher}$$, :params::agtype)
                AS (entity agtype, rel_type agtype, neighbor agtype)
            """),
            {"params": params_json},
        )
        rows = result.fetchall()
    except Exception as e:
        logger.warning("Graph query failed (AGE may not have data yet): %s", e)
        return []

    results: list[QueryResultItem] = []
    seen: set[str] = set()

    for row in rows:
        entity_raw = row[0]
        if not entity_raw:
            continue
        entity_key = str(entity_raw)
        if entity_key in seen:
            continue
        seen.add(entity_key)

        entity = _parse_agtype(entity_raw)
        neighbor = _parse_agtype(row[2])
        rel_type = _parse_agtype(row[1])

        # Extract name from the agtype node dict
        entity_name = ""
        if isinstance(entity, dict):
            entity_name = entity.get("properties", {}).get("name", "") or entity.get("name", "")

        results.append(
            QueryResultItem(
                score=1.0,
                modality="graph_node",
                content_text=entity_name if body.include_context else None,
                classification="UNCLASSIFIED",
                context={
                    "entity": entity,
                    "rel_type": rel_type,
                    "neighbor": neighbor,
                },
            )
        )

    return results


# ---------------------------------------------------------------------------
# Hybrid query — vector + graph score-boosting
# ---------------------------------------------------------------------------

async def _hybrid_query(
    db: AsyncSession, body: QueryRequest
) -> list[QueryResultItem]:
    """Semantic results boosted by graph entity co-occurrence.

    Strategy:
    1. Retrieve 2×top_k semantic candidates
    2. Search graph for entities matching the query
    3. Boost the score of semantic chunks whose text mentions a graph entity
    4. Return top_k by final score
    """
    # Fetch more candidates so boosting can promote relevant ones
    wider_body = body.model_copy(update={"top_k": body.top_k * 2})
    semantic_results = await _semantic_query(db, wider_body)
    graph_results = await _graph_query(db, body)

    # Collect entity names found in graph results
    graph_entity_names: set[str] = set()
    for gr in graph_results:
        if gr.content_text:
            graph_entity_names.add(gr.content_text.lower())
        if gr.context and isinstance(gr.context.get("entity"), dict):
            name = (
                gr.context["entity"].get("properties", {}).get("name", "")
                or gr.context["entity"].get("name", "")
            )
            if name:
                graph_entity_names.add(name.lower())

    # Boost semantic results whose text contains a graph entity name
    GRAPH_BOOST = 0.15
    boosted: list[QueryResultItem] = []
    for item in semantic_results:
        boost = 0.0
        if graph_entity_names and item.content_text:
            text_lower = item.content_text.lower()
            if any(name in text_lower for name in graph_entity_names):
                boost = GRAPH_BOOST
        boosted.append(item.model_copy(update={"score": min(1.0, item.score + boost)}))

    # Append graph-only results (tagged with graph modality) after semantic
    for gr in graph_results:
        boosted.append(gr)

    # Sort by score descending, deduplicate by chunk_id
    seen_ids: set[str] = set()
    final: list[QueryResultItem] = []
    for item in sorted(boosted, key=lambda x: x.score, reverse=True):
        item_key = str(item.chunk_id) if item.chunk_id else str(item.content_text)
        if item_key not in seen_ids:
            seen_ids.add(item_key)
            final.append(item)
        if len(final) >= body.top_k:
            break

    return final


# ---------------------------------------------------------------------------
# Cross-modal query — CLIP text → image retrieval
# ---------------------------------------------------------------------------

async def _cross_modal_query(
    db: AsyncSession, body: QueryRequest
) -> list[QueryResultItem]:
    """Find images semantically related to a text query using CLIP embeddings."""
    from app.services.embedding import embed_text_for_image_search

    query_embedding = embed_text_for_image_search(body.query)
    embedding_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    sql = text(f"""
        SELECT
            c.id          AS chunk_id,
            c.artifact_id,
            c.chunk_text,
            c.modality,
            c.page_number,
            c.classification,
            1 - (c.image_embedding <=> '{embedding_literal}'::vector) AS score
        FROM retrieval.chunks c
        WHERE c.image_embedding IS NOT NULL
        ORDER BY c.image_embedding <=> '{embedding_literal}'::vector
        LIMIT :top_k
    """)

    result = await db.execute(sql, {"top_k": body.top_k})
    rows = result.mappings().all()

    return [
        QueryResultItem(
            chunk_id=row["chunk_id"],
            artifact_id=row["artifact_id"],
            score=float(row["score"]),
            modality=row["modality"],
            content_text=row["chunk_text"] if body.include_context else None,
            page_number=row["page_number"],
            classification=row["classification"],
        )
        for row in rows
    ]
