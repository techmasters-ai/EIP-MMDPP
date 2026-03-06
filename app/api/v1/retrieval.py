"""Unified retrieval endpoint — single multi-modal pipeline with mode-based filtering."""

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1._retrieval_helpers import (
    CROSS_MODAL_DECAY,
    ONTOLOGY_DECAY,
    build_image_filters as _build_image_filters,
    build_text_filters as _build_text_filters,
    deduplicate_results as _deduplicate_results,
)
from app.db.session import get_async_session
from app.schemas.retrieval import (
    QueryMode,
    QueryResultItem,
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
    """Unified retrieval query.

    Modes:
    - **text_basic**: BGE vector search on text_chunks only
    - **text_only**: Multi-modal pipeline, filtered to text results
    - **images_only**: Multi-modal pipeline, filtered to image results
    - **multi_modal**: Multi-modal pipeline, all results
    - **memory**: Cognee approved memory search
    """
    try:
        if body.mode == QueryMode.text_basic:
            results = await _text_vector_search(db, body)
        elif body.mode == QueryMode.memory:
            results = await _memory_query(body)
        elif body.mode in (QueryMode.text_only, QueryMode.images_only, QueryMode.multi_modal):
            results = await _multi_modal_pipeline(db, body)
        else:
            results = []
    except Exception as e:
        logger.warning("Query mode %s failed: %s", body.mode, e)
        results = []

    return UnifiedQueryResponse(
        query_text=body.query_text,
        query_image=body.query_image[:100] if body.query_image else None,
        mode=body.mode.value,
        results=results,
        total=len(results),
    )


# ---------------------------------------------------------------------------
# Multi-modal pipeline (shared by text_only, images_only, multi_modal)
# ---------------------------------------------------------------------------

async def _multi_modal_pipeline(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    """Shared pipeline: vector search + cross-modal bridging + ontology traversal.

    1. Vector search (text BGE + image CLIP as appropriate)
    2. For each seed chunk: cross-modal graph bridging (structural edges)
    3. For each seed chunk: ontology traversal (entity relationships)
    4. Combine, deduplicate, rank, filter by mode
    """
    seed_results: list[QueryResultItem] = []

    # Step 1: Vector searches
    if body.query_text:
        text_results = await _text_vector_search(db, body)
        seed_results.extend(text_results)

        image_results = await _image_vector_search(db, body)
        seed_results.extend(image_results)

    if body.query_image:
        image_results = await _image_vector_search(db, body)
        # Avoid duplicating if both query_text and query_image provided
        existing_ids = {str(r.chunk_id) for r in seed_results if r.chunk_id}
        for r in image_results:
            if str(r.chunk_id) not in existing_ids:
                seed_results.append(r)

    # Steps 2+3: Expand each seed chunk via graph
    expanded: list[QueryResultItem] = []
    for seed in seed_results:
        if not seed.chunk_id:
            continue
        chunk_id_str = str(seed.chunk_id)

        cross_modal = await _expand_via_cross_modal(db, chunk_id_str, seed.score)
        expanded.extend(cross_modal)

        ontology = await _expand_via_ontology(db, chunk_id_str, seed.score)
        expanded.extend(ontology)

    all_results = seed_results + expanded

    # Step 4: Deduplicate by chunk_id, keep highest score
    deduped = _deduplicate_results(all_results)

    # Step 5: Filter by mode
    if body.mode == QueryMode.text_only:
        deduped = [r for r in deduped if r.modality in ("text", "table")]
    elif body.mode == QueryMode.images_only:
        deduped = [r for r in deduped if r.modality in ("image", "schematic")]

    # Sort by score descending, cap at top_k
    deduped.sort(key=lambda x: x.score, reverse=True)
    return deduped[:body.top_k]


# ---------------------------------------------------------------------------
# Text vector search — pgvector HNSW on text_chunks (BGE)
# ---------------------------------------------------------------------------

async def _text_vector_search(
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
               1 - (tc.embedding <=> CAST(:embedding AS vector)) AS score
        FROM retrieval.text_chunks tc
        WHERE tc.embedding IS NOT NULL
        {filter_clauses}
        ORDER BY tc.embedding <=> CAST(:embedding AS vector)
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
# Image vector search — pgvector HNSW on image_chunks (CLIP)
# ---------------------------------------------------------------------------

async def _image_vector_search(
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
               1 - (ic.embedding <=> CAST(:embedding AS vector)) AS score
        FROM retrieval.image_chunks ic
        WHERE ic.embedding IS NOT NULL
        {filter_clauses}
        ORDER BY ic.embedding <=> CAST(:embedding AS vector)
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
# Graph expansion — cross-modal bridging (structural edges)
# ---------------------------------------------------------------------------

async def _expand_via_cross_modal(
    db: AsyncSession,
    chunk_id: str,
    source_score: float,
) -> list[QueryResultItem]:
    """Follow structural graph edges (SAME_PAGE, CONTAINS_TEXT/IMAGE, EXTRACTED_FROM)
    to find connected chunks. Score decays from source."""
    from app.services.graph import setup_age_session_async, _parse_agtype, _escape_cypher, GRAPH_NAME

    await setup_age_session_async(db)
    params_json = json.dumps({"chunk_id": chunk_id})

    cypher = """
        MATCH (src:CHUNK_REF {chunk_id: $chunk_id})-[r*1..3]-(target:CHUNK_REF)
        WHERE target.chunk_id <> $chunk_id
        RETURN target.chunk_id AS target_chunk_id,
               target.chunk_type AS target_chunk_type,
               type(r[0]) AS edge_type
        LIMIT 5
    """

    try:
        result = await db.execute(
            text(f"""
                SELECT * FROM cypher('{GRAPH_NAME}', $${_escape_cypher(cypher)}$$,
                    CAST(:params AS agtype)) AS (
                        target_id agtype, target_type agtype, edge_type agtype
                    )
            """),
            {"params": params_json},
        )
        rows = result.fetchall()
    except Exception as e:
        logger.debug("Cross-modal expansion failed for %s: %s", chunk_id, e)
        return []

    items: list[QueryResultItem] = []
    for row in rows:
        target_id_str = str(_parse_agtype(row[0])).strip('"')
        target_type = str(_parse_agtype(row[1])).strip('"')
        edge_type = str(_parse_agtype(row[2])).strip('"')

        chunk_data = await _lookup_chunk_by_type(db, target_id_str, target_type)
        if chunk_data:
            chunk_data.score = source_score * CROSS_MODAL_DECAY
            chunk_data.context = {
                "source": "cross_modal",
                "edge_type": edge_type,
                "source_chunk_id": chunk_id,
            }
            items.append(chunk_data)

    return items


# ---------------------------------------------------------------------------
# Graph expansion — ontology traversal (entity relationships)
# ---------------------------------------------------------------------------

async def _expand_via_ontology(
    db: AsyncSession,
    chunk_id: str,
    source_score: float,
) -> list[QueryResultItem]:
    """Follow ontology relationships (entity->related_entity->chunk) to find
    semantically related chunks via the knowledge graph."""
    from app.services.graph import get_ontology_linked_chunks_async

    linked = await get_ontology_linked_chunks_async(db, chunk_id, limit=5)

    items: list[QueryResultItem] = []
    for link in linked:
        target_id = link["target_chunk_id"]
        target_type = link["target_chunk_type"]

        chunk_data = await _lookup_chunk_by_type(db, target_id, target_type)
        if chunk_data:
            chunk_data.score = source_score * ONTOLOGY_DECAY
            chunk_data.context = {
                "source": "ontology",
                "rel_type": link["rel_type"],
                "entity_name": link["entity_name"],
                "related_name": link["related_name"],
                "source_chunk_id": chunk_id,
            }
            items.append(chunk_data)

    return items


# ---------------------------------------------------------------------------
# Chunk lookups
# ---------------------------------------------------------------------------

async def _lookup_chunk_by_type(
    db: AsyncSession, chunk_id: str, chunk_type: str
) -> QueryResultItem | None:
    """Route to the correct lookup based on chunk_type."""
    if chunk_type == "image_chunk":
        return await _lookup_image_chunk(db, chunk_id)
    return await _lookup_text_chunk(db, chunk_id)


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
        score=0.0, modality=row[4], content_text=row[3],
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
        score=0.0, modality=row[4], content_text=row[3],
        page_number=row[5], classification=row[6],
    )


# ---------------------------------------------------------------------------
# Memory query — Cognee search (unchanged)
# ---------------------------------------------------------------------------

async def _memory_query(body: UnifiedQueryRequest) -> list[QueryResultItem]:
    if not body.query_text:
        return []

    from app.services.cognee_service import cognee_search

    return await cognee_search(body.query_text, body.top_k)


# Helpers (_deduplicate_results, _build_text_filters, _build_image_filters)
# are imported from _retrieval_helpers.py at the top of this file.
