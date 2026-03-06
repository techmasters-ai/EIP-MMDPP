"""Unified retrieval endpoint — single multi-modal pipeline with mode-based filtering."""

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1._retrieval_helpers import (
    build_image_filters as _build_image_filters,
    build_text_filters as _build_text_filters,
    compute_fusion_score,
    deduplicate_results as _deduplicate_results,
    get_cross_modal_decay,
    get_ontology_decay,
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

# Max concurrent seed expansions
_EXPAND_CONCURRENCY = 16


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

    # Populate presigned image URLs for image-modality results
    await _populate_image_urls(db, results)

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
    """Shared pipeline: parallel vector search + parallel graph expansion + fusion scoring.

    1. Parallel vector search (text BGE + image CLIP)
    2. Parallel per-seed expansion (doc-structure + ontology)
    3. Batch chunk lookups
    4. Score fusion + deduplicate + mode filter + sort + cap
    """
    t0 = time.monotonic()

    # Set up AGE once for the entire request (not per-seed)
    try:
        from app.services.graph import setup_age_session_async
        await setup_age_session_async(db)
    except Exception as e:
        logger.debug("AGE session setup failed (graph expansion will be skipped): %s", e)

    # Step 1: Parallel vector searches
    search_tasks: list = []
    if body.query_text:
        search_tasks.append(_text_vector_search(db, body))
        search_tasks.append(_image_vector_search(db, body))
    if body.query_image:
        search_tasks.append(_image_vector_search(db, body))

    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
    seed_results = _merge_seed_results(search_results)
    t_search = time.monotonic()

    # Step 2: Parallel per-seed expansion (bounded concurrency)
    expanded = await _expand_seeds(db, seed_results, body.include_context)
    t_expand = time.monotonic()

    all_results = seed_results + expanded

    # Step 3: Deduplicate by chunk_id, keep highest score
    deduped = _deduplicate_results(all_results)

    # Step 4: Filter by mode
    if body.mode == QueryMode.text_only:
        deduped = [r for r in deduped if r.modality in ("text", "table")]
    elif body.mode == QueryMode.images_only:
        deduped = [r for r in deduped if r.modality in ("image", "schematic")]

    # Sort by score descending, cap at top_k
    deduped.sort(key=lambda x: x.score, reverse=True)
    final = deduped[:body.top_k]

    t_total = time.monotonic()
    logger.info(
        "retrieval: mode=%s search=%.0fms expand=%.0fms total=%.0fms seeds=%d expanded=%d returned=%d",
        body.mode.value,
        (t_search - t0) * 1000,
        (t_expand - t_search) * 1000,
        (t_total - t0) * 1000,
        len(seed_results),
        len(expanded),
        len(final),
    )
    return final


def _merge_seed_results(results_lists: list) -> list[QueryResultItem]:
    """Merge multiple search result lists, deduplicating by chunk_id."""
    seen: set[str] = set()
    merged: list[QueryResultItem] = []
    for result in results_lists:
        if isinstance(result, Exception):
            logger.debug("Search task failed: %s", result)
            continue
        for r in result:
            key = str(r.chunk_id) if r.chunk_id else str(id(r))
            if key not in seen:
                seen.add(key)
                merged.append(r)
    return merged


async def _expand_seeds(
    db: AsyncSession,
    seeds: list[QueryResultItem],
    include_context: bool,
) -> list[QueryResultItem]:
    """Parallel per-seed expansion with bounded concurrency."""
    sem = asyncio.Semaphore(_EXPAND_CONCURRENCY)

    async def _expand_one(seed: QueryResultItem) -> list[QueryResultItem]:
        async with sem:
            chunk_id_str = str(seed.chunk_id)
            items: list[QueryResultItem] = []

            # Try doc-structure expansion (chunk_links table) first
            doc_items = await _expand_via_doc_structure(db, chunk_id_str, seed.score, include_context)
            if doc_items:
                items.extend(doc_items)
            else:
                # Fallback to AGE cross-modal for pre-v2 documents
                cross_items = await _expand_via_cross_modal(db, chunk_id_str, seed.score, include_context)
                items.extend(cross_items)

            onto_items = await _expand_via_ontology(db, chunk_id_str, seed.score, include_context)
            items.extend(onto_items)
            return items

    expansion_lists = await asyncio.gather(
        *[_expand_one(s) for s in seeds if s.chunk_id],
        return_exceptions=True,
    )
    return [
        item
        for sublist in expansion_lists
        if not isinstance(sublist, Exception)
        for item in sublist
    ]


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

    filter_clauses, filter_params = _build_text_filters(body)

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

    params = {"embedding": embedding_str, "top_k": body.top_k, **filter_params}
    result = await db.execute(sql, params)
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

    filter_clauses, filter_params = _build_image_filters(body)

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

    params = {"embedding": embedding_str, "top_k": body.top_k, **filter_params}
    result = await db.execute(sql, params)
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
# Document-structure expansion (chunk_links table — v2 ingest)
# ---------------------------------------------------------------------------

async def _expand_via_doc_structure(
    db: AsyncSession,
    chunk_id: str,
    source_score: float,
    include_context: bool,
) -> list[QueryResultItem]:
    """Expand via document-structure links (chunk_links table).

    Returns empty list if no chunk_links exist (pre-v2 documents).
    """
    from app.config import get_settings
    s = get_settings()

    sql = text("""
        SELECT cl.target_chunk_id, cl.link_type, cl.weight, cl.hop
        FROM retrieval.chunk_links cl
        WHERE cl.source_chunk_id = :chunk_id
          AND cl.hop <= :max_hops
        ORDER BY cl.weight DESC
        LIMIT :limit
    """)
    try:
        result = await db.execute(sql, {
            "chunk_id": chunk_id,
            "max_hops": s.retrieval_doc_max_hops,
            "limit": s.retrieval_doc_expand_k,
        })
        rows = result.fetchall()
    except Exception as e:
        logger.debug("Doc-structure expansion failed for %s: %s", chunk_id, e)
        return []

    if not rows:
        return []

    # Batch lookup all target chunks
    chunk_map = await _batch_lookup_chunks(
        db,
        [(str(row[0]),) for row in rows],
        include_context,
    )

    items: list[QueryResultItem] = []
    for row in rows:
        target_id = str(row[0])
        link_type = row[1]
        weight = float(row[2])
        hops = int(row[3])

        chunk = chunk_map.get(target_id)
        if not chunk:
            continue

        chunk.score = compute_fusion_score(
            semantic_score=source_score,
            doc_structure_weight=weight,
            doc_structure_hops=hops,
            content_text=chunk.content_text,
        )
        chunk.context = {
            "source": "doc_structure",
            "link_type": link_type,
            "hops": hops,
            "source_chunk_id": chunk_id,
        }
        items.append(chunk)

    return items


# ---------------------------------------------------------------------------
# Graph expansion — cross-modal bridging (AGE structural edges, fallback)
# ---------------------------------------------------------------------------

async def _expand_via_cross_modal(
    db: AsyncSession,
    chunk_id: str,
    source_score: float,
    include_context: bool = True,
) -> list[QueryResultItem]:
    """Follow structural graph edges (SAME_PAGE, CONTAINS_TEXT/IMAGE, EXTRACTED_FROM)
    to find connected chunks. Score decays from source.

    This is the fallback for documents ingested before v2 (no chunk_links).
    """
    from app.services.graph import _parse_agtype, _escape_cypher, GRAPH_NAME

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
    decay = get_cross_modal_decay()
    for row in rows:
        target_id_str = str(_parse_agtype(row[0])).strip('"')
        target_type = str(_parse_agtype(row[1])).strip('"')
        edge_type = str(_parse_agtype(row[2])).strip('"')

        chunk_data = await _lookup_chunk_by_type(db, target_id_str, target_type, include_context)
        if chunk_data:
            chunk_data.score = source_score * decay
            chunk_data.context = {
                "source": "cross_modal",
                "edge_type": edge_type,
                "source_chunk_id": chunk_id,
                "degraded": True,
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
    include_context: bool = True,
) -> list[QueryResultItem]:
    """Follow ontology relationships (entity->related_entity->chunk) to find
    semantically related chunks via the knowledge graph."""
    from app.services.graph import get_ontology_linked_chunks_async
    from app.config import get_settings

    s = get_settings()
    linked = await get_ontology_linked_chunks_async(db, chunk_id, limit=s.retrieval_ontology_expand_k)

    items: list[QueryResultItem] = []
    decay = get_ontology_decay()
    for link in linked:
        target_id = link["target_chunk_id"]
        target_type = link["target_chunk_type"]

        chunk_data = await _lookup_chunk_by_type(db, target_id, target_type, include_context)
        if chunk_data:
            chunk_data.score = source_score * decay
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
# Batch chunk lookups (fixes N+1 query pattern)
# ---------------------------------------------------------------------------

async def _batch_lookup_chunks(
    db: AsyncSession,
    chunk_refs: list[tuple[str, ...]],
    include_context: bool,
) -> dict[str, QueryResultItem]:
    """Batch lookup chunks by ID across both tables. Returns {chunk_id: QueryResultItem}.

    chunk_refs is a list of tuples; first element is always the chunk_id string.
    Tries text_chunks first, then image_chunks for any not found.
    """
    if not chunk_refs:
        return {}

    chunk_ids = [ref[0] for ref in chunk_refs]
    results: dict[str, QueryResultItem] = {}

    # Try text_chunks
    sql = text("""
        SELECT id, artifact_id, document_id, chunk_text, modality,
               page_number, classification
        FROM retrieval.text_chunks WHERE id = ANY(:ids)
    """)
    rows = (await db.execute(sql, {"ids": chunk_ids})).fetchall()
    for row in rows:
        results[str(row[0])] = QueryResultItem(
            chunk_id=row[0], artifact_id=row[1], document_id=row[2],
            score=0.0, modality=row[4],
            content_text=row[3] if include_context else None,
            page_number=row[5], classification=row[6],
        )

    # Try image_chunks for any not found
    missing = [cid for cid in chunk_ids if cid not in results]
    if missing:
        sql = text("""
            SELECT id, artifact_id, document_id, chunk_text, modality,
                   page_number, classification
            FROM retrieval.image_chunks WHERE id = ANY(:ids)
        """)
        rows = (await db.execute(sql, {"ids": missing})).fetchall()
        for row in rows:
            results[str(row[0])] = QueryResultItem(
                chunk_id=row[0], artifact_id=row[1], document_id=row[2],
                score=0.0, modality=row[4],
                content_text=row[3] if include_context else None,
                page_number=row[5], classification=row[6],
            )

    return results


# ---------------------------------------------------------------------------
# Single chunk lookups (used by cross-modal and ontology fallback)
# ---------------------------------------------------------------------------

async def _lookup_chunk_by_type(
    db: AsyncSession, chunk_id: str, chunk_type: str, include_context: bool = True
) -> QueryResultItem | None:
    """Route to the correct lookup based on chunk_type."""
    if chunk_type == "image_chunk":
        return await _lookup_image_chunk(db, chunk_id, include_context)
    return await _lookup_text_chunk(db, chunk_id, include_context)


async def _lookup_text_chunk(
    db: AsyncSession, chunk_id: str, include_context: bool = True
) -> QueryResultItem | None:
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
        score=0.0, modality=row[4],
        content_text=row[3] if include_context else None,
        page_number=row[5], classification=row[6],
    )


async def _lookup_image_chunk(
    db: AsyncSession, chunk_id: str, include_context: bool = True
) -> QueryResultItem | None:
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
        score=0.0, modality=row[4],
        content_text=row[3] if include_context else None,
        page_number=row[5], classification=row[6],
    )


# ---------------------------------------------------------------------------
# Image URL population (presigned MinIO URLs for image results)
# ---------------------------------------------------------------------------

async def _populate_image_urls(
    db: AsyncSession, results: list[QueryResultItem]
) -> None:
    """Batch-fetch presigned URLs for image-modality results."""
    image_chunk_ids = [
        str(r.chunk_id) for r in results
        if r.modality in ("image", "schematic") and r.chunk_id
    ]
    if not image_chunk_ids:
        return

    try:
        from app.services.storage import generate_presigned_url_async
    except ImportError:
        logger.debug("storage service not available for image URL generation")
        return

    sql = text("""
        SELECT ic.id, a.storage_bucket, a.storage_key
        FROM retrieval.image_chunks ic
        JOIN ingest.artifacts a ON a.id = ic.artifact_id
        WHERE ic.id = ANY(:ids)
          AND a.storage_bucket IS NOT NULL AND a.storage_key IS NOT NULL
    """)
    try:
        rows = (await db.execute(sql, {"ids": image_chunk_ids})).fetchall()
    except Exception as e:
        logger.debug("Image URL lookup failed: %s", e)
        return

    if not rows:
        return

    # Generate presigned URLs in parallel
    url_map: dict[str, str] = {}

    async def _gen_url(chunk_id: str, bucket: str, key: str) -> None:
        try:
            url_map[chunk_id] = await generate_presigned_url_async(bucket, key)
        except Exception as e:
            logger.debug("Presigned URL generation failed for %s: %s", chunk_id, e)

    await asyncio.gather(*[_gen_url(str(r[0]), r[1], r[2]) for r in rows])

    for result in results:
        if result.chunk_id and str(result.chunk_id) in url_map:
            result.image_url = url_map[str(result.chunk_id)]


# ---------------------------------------------------------------------------
# Memory query — Cognee search (unchanged)
# ---------------------------------------------------------------------------

async def _memory_query(body: UnifiedQueryRequest) -> list[QueryResultItem]:
    if not body.query_text:
        return []

    from app.services.cognee_service import cognee_search

    return await cognee_search(body.query_text, body.top_k)
