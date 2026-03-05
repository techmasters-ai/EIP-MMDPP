"""Unified retrieval endpoint — queries any combination of knowledge layers."""

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.schemas.retrieval import (
    QueryMode,
    QueryResultItem,
    SectionResults,
    UnifiedQueryRequest,
    UnifiedQueryResponse,
)

router = APIRouter(tags=["retrieval"])
logger = logging.getLogger(__name__)


@router.post("/retrieval/query", response_model=UnifiedQueryResponse)
async def unified_query(
    body: UnifiedQueryRequest,
    db: AsyncSession = Depends(get_async_session),
) -> UnifiedQueryResponse:
    """Unified multi-mode retrieval query.

    Accepts text and/or image input, queries any combination of knowledge
    layers, returns structured sections per mode.

    Modes:
    - **text_semantic**: BGE vector search on text_chunks
    - **image_semantic**: CLIP vector search on image_chunks
    - **graph**: AGE Cypher traversal with ontology relationships
    - **cross_modal**: Text→graph→images or Image→graph→text
    - **memory**: Search Cognee approved memory
    """
    sections: dict[str, SectionResults] = {}

    for mode in body.modes:
        try:
            if mode == QueryMode.text_semantic:
                results = await _text_semantic_query(db, body)
            elif mode == QueryMode.image_semantic:
                results = await _image_semantic_query(db, body)
            elif mode == QueryMode.graph:
                results = await _graph_query(db, body)
            elif mode == QueryMode.cross_modal:
                results = await _cross_modal_query(db, body)
            elif mode == QueryMode.memory:
                results = await _memory_query(body)
            else:
                results = []
            sections[mode.value] = SectionResults(results=results, total=len(results))
        except Exception as e:
            logger.warning("Mode %s failed: %s", mode, e)
            sections[mode.value] = SectionResults(results=[], total=0)

    return UnifiedQueryResponse(
        query_text=body.query_text,
        query_image=body.query_image[:100] if body.query_image else None,
        modes=[m.value for m in body.modes],
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Text semantic query — pgvector HNSW on text_chunks
# ---------------------------------------------------------------------------

async def _text_semantic_query(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    if not body.query_text:
        return []

    from app.services.embedding import embed_texts

    query_embedding = embed_texts([body.query_text])[0]
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    filter_clauses = _build_text_filters(body)

    sql = text(f"""
        SELECT tc.id, tc.artifact_id, tc.document_id, tc.chunk_text,
               tc.modality, tc.page_number, tc.classification,
               1 - (tc.embedding <=> :embedding::vector) AS score
        FROM retrieval.text_chunks tc
        WHERE tc.embedding IS NOT NULL
        {filter_clauses}
        ORDER BY tc.embedding <=> :embedding::vector
        LIMIT :top_k
    """)

    result = await db.execute(sql, {"embedding": embedding_str, "top_k": body.top_k})
    rows = result.fetchall()

    return [
        QueryResultItem(
            chunk_id=row[0],
            artifact_id=row[1],
            document_id=row[2],
            score=float(row[7]),
            modality=row[4],
            content_text=row[3] if body.include_context else None,
            page_number=row[5],
            classification=row[6],
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Image semantic query — pgvector HNSW on image_chunks
# ---------------------------------------------------------------------------

async def _image_semantic_query(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    from app.services.embedding import embed_text_for_clip, embed_images

    if body.query_image:
        import base64
        import io
        from PIL import Image

        image_bytes = base64.b64decode(body.query_image)
        pil_image = Image.open(io.BytesIO(image_bytes))
        query_embedding = embed_images([pil_image])[0]
    elif body.query_text:
        query_embedding = embed_text_for_clip(body.query_text)
    else:
        return []

    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    filter_clauses = _build_image_filters(body)

    sql = text(f"""
        SELECT ic.id, ic.artifact_id, ic.document_id, ic.chunk_text,
               ic.modality, ic.page_number, ic.classification,
               1 - (ic.embedding <=> :embedding::vector) AS score
        FROM retrieval.image_chunks ic
        WHERE ic.embedding IS NOT NULL
        {filter_clauses}
        ORDER BY ic.embedding <=> :embedding::vector
        LIMIT :top_k
    """)

    result = await db.execute(sql, {"embedding": embedding_str, "top_k": body.top_k})
    rows = result.fetchall()

    return [
        QueryResultItem(
            chunk_id=row[0],
            artifact_id=row[1],
            document_id=row[2],
            score=float(row[7]),
            modality=row[4],
            content_text=row[3] if body.include_context else None,
            page_number=row[5],
            classification=row[6],
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Graph query — AGE Cypher traversal
# ---------------------------------------------------------------------------

async def _graph_query(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    query = body.query_text or ""
    if not query.strip():
        return []

    from app.services.graph import search_nodes_async, get_neighborhood_async

    matches = await search_nodes_async(db, query, limit=body.top_k)

    results: list[QueryResultItem] = []
    for match in matches:
        node = match.get("node", {})
        if not isinstance(node, dict):
            continue

        entity_type = match.get("entity_type", "UNKNOWN")
        name = node.get("name", "") or node.get("properties", {}).get("name", "")

        neighbors = await get_neighborhood_async(db, name, hop_count=2, limit=10)

        results.append(
            QueryResultItem(
                score=node.get("confidence", 0.5) if isinstance(node.get("confidence"), (int, float)) else 0.5,
                modality="graph_node",
                content_text=name if body.include_context else None,
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


# ---------------------------------------------------------------------------
# Cross-modal query — graph-bridged text↔image traversal
# ---------------------------------------------------------------------------

async def _cross_modal_query(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    """Cross-modal search via graph bridging.

    Text → text_chunks → graph edges → connected image_chunks
    Image → image_chunks → graph edges → connected text_chunks
    """
    from app.services.graph import setup_age_session_async, _parse_agtype

    results: list[QueryResultItem] = []

    # Text → Image path
    if body.query_text:
        text_results = await _text_semantic_query(
            db, body.model_copy(update={"top_k": 5})
        )
        for tr in text_results:
            if tr.chunk_id:
                connected = await _get_connected_chunks(
                    db, str(tr.chunk_id), "image_chunk"
                )
                results.extend(connected)

    # Image → Text path
    if body.query_image:
        image_results = await _image_semantic_query(
            db, body.model_copy(update={"top_k": 5})
        )
        for ir in image_results:
            if ir.chunk_id:
                connected = await _get_connected_chunks(
                    db, str(ir.chunk_id), "text_chunk"
                )
                results.extend(connected)

    # Deduplicate and sort by score
    seen: set[str] = set()
    unique: list[QueryResultItem] = []
    for r in sorted(results, key=lambda x: x.score, reverse=True):
        key = str(r.chunk_id)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:body.top_k]


async def _get_connected_chunks(
    db: AsyncSession,
    chunk_id: str,
    target_chunk_type: str,
) -> list[QueryResultItem]:
    """Follow graph edges from a CHUNK_REF to find connected chunks of the target type."""
    from app.services.graph import setup_age_session_async, _parse_agtype, GRAPH_NAME

    await setup_age_session_async(db)

    params_json = json.dumps({"chunk_id": chunk_id, "target_type": target_chunk_type})

    cypher = """
        MATCH (src:CHUNK_REF {chunk_id: $chunk_id})-[r*1..3]-(target:CHUNK_REF {chunk_type: $target_type})
        RETURN target.chunk_id AS target_chunk_id, type(r[0]) AS edge_type
        LIMIT 10
    """

    try:
        result = await db.execute(
            text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $${cypher}$$,
                    :params::agtype) AS (target_id agtype, edge_type agtype)
            """),
            {"params": params_json},
        )
        rows = result.fetchall()
    except Exception as e:
        logger.debug("Cross-modal graph traversal failed: %s", e)
        return []

    items: list[QueryResultItem] = []
    for row in rows:
        target_id_str = str(_parse_agtype(row[0])).strip('"')
        edge_type = str(_parse_agtype(row[1])).strip('"')

        # Look up the actual chunk data
        if target_chunk_type == "image_chunk":
            chunk_data = await _lookup_image_chunk(db, target_id_str)
        else:
            chunk_data = await _lookup_text_chunk(db, target_id_str)

        if chunk_data:
            chunk_data.context = {"cross_modal_edge": edge_type, "source_chunk_id": chunk_id}
            items.append(chunk_data)

    return items


async def _lookup_text_chunk(db: AsyncSession, chunk_id: str) -> QueryResultItem | None:
    sql = text("""
        SELECT id, artifact_id, document_id, chunk_text, modality,
               page_number, classification
        FROM retrieval.text_chunks WHERE id = :cid
    """)
    result = await db.execute(sql, {"cid": chunk_id})
    row = result.fetchone()
    if not row:
        return None
    return QueryResultItem(
        chunk_id=row[0], artifact_id=row[1], document_id=row[2],
        score=0.8, modality=row[4], content_text=row[3],
        page_number=row[5], classification=row[6],
    )


async def _lookup_image_chunk(db: AsyncSession, chunk_id: str) -> QueryResultItem | None:
    sql = text("""
        SELECT id, artifact_id, document_id, chunk_text, modality,
               page_number, classification
        FROM retrieval.image_chunks WHERE id = :cid
    """)
    result = await db.execute(sql, {"cid": chunk_id})
    row = result.fetchone()
    if not row:
        return None
    return QueryResultItem(
        chunk_id=row[0], artifact_id=row[1], document_id=row[2],
        score=0.8, modality=row[4], content_text=row[3],
        page_number=row[5], classification=row[6],
    )


# ---------------------------------------------------------------------------
# Memory query — Cognee search
# ---------------------------------------------------------------------------

async def _memory_query(body: UnifiedQueryRequest) -> list[QueryResultItem]:
    if not body.query_text:
        return []

    from app.services.cognee_service import cognee_search

    return await cognee_search(body.query_text, body.top_k)


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _build_text_filters(body: UnifiedQueryRequest) -> str:
    clauses = ""
    if body.filters:
        if body.filters.classification:
            clauses += f" AND tc.classification = '{body.filters.classification}'"
        if body.filters.document_ids:
            doc_ids = ",".join(f"'{d}'" for d in body.filters.document_ids)
            clauses += f" AND tc.document_id IN ({doc_ids})"
    return clauses


def _build_image_filters(body: UnifiedQueryRequest) -> str:
    clauses = ""
    if body.filters:
        if body.filters.classification:
            clauses += f" AND ic.classification = '{body.filters.classification}'"
        if body.filters.document_ids:
            doc_ids = ",".join(f"'{d}'" for d in body.filters.document_ids)
            clauses += f" AND ic.document_id IN ({doc_ids})"
    return clauses
