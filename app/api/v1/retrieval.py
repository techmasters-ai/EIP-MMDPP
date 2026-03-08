"""Unified retrieval endpoint — single multi-modal pipeline with mode-based filtering."""

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
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
from app.db.session import get_async_session, get_neo4j_async_driver, get_qdrant_async_client
from app.schemas.retrieval import (
    ModalityFilter,
    QueryResultItem,
    QueryStrategy,
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
    - **text_basic**: BGE vector search on text_chunks only (Qdrant)
    - **text_only**: Multi-modal pipeline, filtered to text results
    - **images_only**: Multi-modal pipeline, filtered to image results
    - **multi_modal**: Multi-modal pipeline, all results
    - **memory**: Cognee approved memory search
    - **graphrag_local**: Entity-centric retrieval with community context reports
    - **graphrag_global**: Cross-community summarization for broad questions
    """
    try:
        if body.strategy == QueryStrategy.basic:
            results = await _text_vector_search(db, body)
        elif body.strategy == QueryStrategy.graphrag_local:
            results = await _graphrag_local_query(db, body)
        elif body.strategy == QueryStrategy.graphrag_global:
            results = await _graphrag_global_query(db, body)
        elif body.strategy == QueryStrategy.hybrid:
            results = await _multi_modal_pipeline(db, body)
        else:
            results = []
    except HTTPException:
        raise  # Let GraphRAG precondition errors propagate as-is
    except Exception as e:
        logger.warning("Query strategy %s failed: %s", body.strategy, e)
        results = []

    # Backfill content_text from Postgres for results missing it (pre-existing data)
    if body.include_context:
        await _backfill_content_text(db, results)

    # Populate presigned image URLs for image-modality results
    await _populate_image_urls(db, results)

    return UnifiedQueryResponse(
        query_text=body.query_text,
        query_image=body.query_image[:100] if body.query_image else None,
        strategy=body.strategy.value,
        modality_filter=body.modality_filter.value,
        results=results,
        total=len(results),
    )


@router.get("/images/{chunk_id}")
async def get_image(
    chunk_id: str,
    db: AsyncSession = Depends(get_async_session),
):
    """Stream an image from MinIO for a given image chunk ID.

    Used by the frontend to display image results without exposing
    Docker-internal MinIO presigned URLs.
    """
    sql = text("""
        SELECT a.storage_bucket, a.storage_key
        FROM retrieval.image_chunks ic
        JOIN ingest.artifacts a ON a.id = ic.artifact_id
        WHERE ic.id = :chunk_id
          AND a.storage_bucket IS NOT NULL AND a.storage_key IS NOT NULL
    """)
    row = (await db.execute(sql, {"chunk_id": chunk_id})).first()
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    bucket, key = row[0], row[1]
    try:
        from app.services.storage import download_bytes_async
        image_bytes = await download_bytes_async(bucket, key)
    except Exception as e:
        logger.warning("Image download failed for chunk %s: %s", chunk_id, e)
        raise HTTPException(status_code=502, detail="Failed to fetch image from storage")

    # Guess content type from key extension
    ext = key.rsplit(".", 1)[-1].lower() if "." in key else "png"
    content_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/png")

    return StreamingResponse(
        iter([image_bytes]),
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ---------------------------------------------------------------------------
# Multi-modal pipeline (shared by text_only, images_only, multi_modal)
# ---------------------------------------------------------------------------

async def _multi_modal_pipeline(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    """Shared pipeline: parallel vector search + parallel graph expansion + fusion scoring.

    1. Parallel vector search (text BGE via Qdrant + image CLIP via Qdrant)
    2. Parallel per-seed expansion (doc-structure + ontology via Neo4j)
    3. Batch chunk lookups
    4. Score fusion + deduplicate + mode filter + sort + cap
    """
    t0 = time.monotonic()

    # Step 1: Parallel vector searches (Qdrant)
    search_tasks: list = []
    if body.query_text:
        search_tasks.append(_text_vector_search(db, body))
    # Image search: prefer image-to-image when query_image is present, text-to-CLIP otherwise
    if body.query_image or body.query_text:
        search_tasks.append(_image_vector_search(db, body))

    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
    seed_results = _merge_seed_results(search_results)
    t_search = time.monotonic()

    # Step 2: Parallel per-seed expansion (bounded concurrency, Neo4j)
    expanded = await _expand_seeds(db, seed_results, body.include_context, body.query_text)
    t_expand = time.monotonic()

    all_results = seed_results + expanded

    # Step 3: Deduplicate by chunk_id, keep highest score
    deduped = _deduplicate_results(all_results)

    # Step 4: Filter by modality
    if body.modality_filter == ModalityFilter.text:
        deduped = [r for r in deduped if r.modality in ("text", "table")]
    elif body.modality_filter == ModalityFilter.image:
        deduped = [r for r in deduped if r.modality in ("image", "schematic")]

    # Sort by score descending, cap at top_k
    deduped.sort(key=lambda x: x.score, reverse=True)
    final = deduped[:body.top_k]

    t_total = time.monotonic()
    logger.info(
        "retrieval: strategy=%s modality=%s search=%.0fms expand=%.0fms total=%.0fms seeds=%d expanded=%d returned=%d",
        body.strategy.value,
        body.modality_filter.value,
        (t_search - t0) * 1000,
        (t_expand - t_search) * 1000,
        (t_total - t0) * 1000,
        len(seed_results),
        len(expanded),
        len(final),
    )
    return final


def _merge_seed_results(results_lists: list) -> list[QueryResultItem]:
    """Merge multiple search result lists, keeping highest score per chunk_id."""
    best: dict[str, QueryResultItem] = {}
    for result in results_lists:
        if isinstance(result, Exception):
            logger.debug("Search task failed: %s", result)
            continue
        for r in result:
            key = str(r.chunk_id) if r.chunk_id else str(id(r))
            if key not in best or r.score > best[key].score:
                best[key] = r
    return list(best.values())


async def _expand_seeds(
    db: AsyncSession,
    seeds: list[QueryResultItem],
    include_context: bool,
    query_text: str | None = None,
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
                # Fallback to Neo4j cross-modal for legacy documents
                cross_items = await _expand_via_cross_modal(chunk_id_str, seed.score, include_context)
                items.extend(cross_items)

            onto_items = await _expand_via_ontology(chunk_id_str, seed.score, include_context, query_text)
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
# Text vector search — Qdrant on eip_text_chunks (BGE)
# ---------------------------------------------------------------------------

async def _text_vector_search(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    if not body.query_text:
        return []

    from app.services.embedding import embed_texts
    from app.services.qdrant_store import search_text_vectors_async

    query_embedding = embed_texts([body.query_text])[0]

    # Build Qdrant filter from request filters
    qdrant_filters = _build_qdrant_filters(body)

    qdrant_client = get_qdrant_async_client()
    hits = await search_text_vectors_async(
        qdrant_client,
        query_vector=query_embedding,
        limit=body.top_k,
        filters=qdrant_filters,
    )

    # Map Qdrant results back to QueryResultItem using payload metadata
    results = []
    for hit in hits:
        payload = hit.get("payload", {})
        results.append(
            QueryResultItem(
                chunk_id=payload.get("chunk_id"),
                artifact_id=payload.get("artifact_id"),
                document_id=payload.get("document_id"),
                score=float(hit.get("score", 0.0)),
                modality=payload.get("modality", "text"),
                content_text=payload.get("chunk_text") if body.include_context else None,
                page_number=payload.get("page_number"),
                classification=payload.get("classification", "UNCLASSIFIED"),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Image vector search — Qdrant on eip_image_chunks (CLIP)
# ---------------------------------------------------------------------------

async def _image_vector_search(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    from app.services.embedding import embed_text_for_clip, embed_images
    from app.services.qdrant_store import search_image_vectors_async

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

    qdrant_filters = _build_qdrant_filters(body)

    qdrant_client = get_qdrant_async_client()
    hits = await search_image_vectors_async(
        qdrant_client,
        query_vector=query_embedding,
        limit=body.top_k,
        filters=qdrant_filters,
    )

    results = []
    for hit in hits:
        payload = hit.get("payload", {})
        results.append(
            QueryResultItem(
                chunk_id=payload.get("chunk_id"),
                artifact_id=payload.get("artifact_id"),
                document_id=payload.get("document_id"),
                score=float(hit.get("score", 0.0)),
                modality=payload.get("modality", "image"),
                content_text=payload.get("chunk_text") if body.include_context else None,
                page_number=payload.get("page_number"),
                classification=payload.get("classification", "UNCLASSIFIED"),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Qdrant filter builder
# ---------------------------------------------------------------------------

def _build_qdrant_filters(body: UnifiedQueryRequest) -> dict[str, Any] | None:
    """Convert UnifiedQueryRequest filters to Qdrant filter dict."""
    if not body.filters:
        return None

    filters: dict[str, Any] = {}
    if body.filters.classification:
        filters["classification"] = body.filters.classification
    if body.filters.document_ids:
        filters["document_id"] = [str(d) for d in body.filters.document_ids]
    if body.filters.modalities:
        if len(body.filters.modalities) == 1:
            filters["modality"] = body.filters.modalities[0]
        else:
            filters["modality"] = list(body.filters.modalities)

    return filters if filters else None


# ---------------------------------------------------------------------------
# Document-structure expansion (chunk_links table — stays in Postgres)
# ---------------------------------------------------------------------------

async def _expand_via_doc_structure(
    db: AsyncSession,
    chunk_id: str,
    source_score: float,
    include_context: bool,
) -> list[QueryResultItem]:
    """Expand via document-structure links (chunk_links table).

    Returns empty list if no chunk_links exist (legacy documents).
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
# Graph expansion — cross-modal bridging (Neo4j structural edges, fallback)
# ---------------------------------------------------------------------------

async def _expand_via_cross_modal(
    chunk_id: str,
    source_score: float,
    include_context: bool = True,
) -> list[QueryResultItem]:
    """Follow structural graph edges (SAME_PAGE, CONTAINS_TEXT/IMAGE, EXTRACTED_FROM)
    to find connected chunks via Neo4j. Score decays from source.

    This is the fallback for documents ingested before chunk_links existed.
    """
    driver = get_neo4j_async_driver()

    query = """
        MATCH (src:ChunkRef {chunk_id: $chunk_id})-[*1..3]-(target:ChunkRef)
        WHERE target.chunk_id <> $chunk_id
        RETURN target.chunk_id AS target_chunk_id,
               target.chunk_type AS target_chunk_type
        LIMIT 5
    """

    try:
        async with driver.session() as session:
            result = await session.run(query, chunk_id=chunk_id)
            records = await result.data()
    except Exception as e:
        logger.debug("Cross-modal expansion failed for %s: %s", chunk_id, e)
        return []

    items: list[QueryResultItem] = []
    decay = get_cross_modal_decay()

    from app.db.session import AsyncSessionFactory
    async with AsyncSessionFactory() as db_session:
        for record in records:
            target_id = record["target_chunk_id"]
            target_type = record.get("target_chunk_type", "text_chunk")
            chunk_data = await _lookup_chunk_by_type(db_session, target_id, target_type, include_context)

            if chunk_data:
                chunk_data.score = source_score * decay
                chunk_data.context = {
                    "source": "cross_modal",
                    "source_chunk_id": chunk_id,
                    "degraded": True,
                }
                items.append(chunk_data)

    return items


# ---------------------------------------------------------------------------
# Graph expansion — ontology traversal (entity relationships via Neo4j)
# ---------------------------------------------------------------------------

async def _expand_via_ontology(
    chunk_id: str,
    source_score: float,
    include_context: bool = True,
    query_text: str | None = None,
) -> list[QueryResultItem]:
    """Follow ontology relationships (entity->related_entity->chunk) to find
    semantically related chunks via the Neo4j knowledge graph."""
    from app.services.neo4j_graph import get_ontology_linked_chunks_async
    from app.config import get_settings

    driver = get_neo4j_async_driver()
    s = get_settings()
    linked = await get_ontology_linked_chunks_async(driver, chunk_id, limit=s.retrieval_ontology_expand_k)

    items: list[QueryResultItem] = []

    # Look up chunk metadata from Postgres
    from app.db.session import AsyncSessionFactory
    async with AsyncSessionFactory() as db_session:
        for link in linked:
            target_id = link["target_chunk_id"]
            target_type = link["target_chunk_type"]

            chunk_data = await _lookup_chunk_by_type(db_session, target_id, target_type, include_context)
            if chunk_data:
                chunk_data.score = compute_fusion_score(
                    semantic_score=source_score,
                    ontology_rel_type=link.get("rel_type", "RELATED_TO"),
                    ontology_hops=1,
                    content_text=chunk_data.content_text,
                    query_text=query_text,
                )
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
# GraphRAG local search — entity-centric + community reports
# ---------------------------------------------------------------------------

async def _graphrag_local_query(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    """Entity-centric search with community report context."""
    if not body.query_text:
        return []

    from app.services.graphrag_service import local_search
    from app.db.session import get_neo4j_driver

    # GraphRAG local_search uses sync Neo4j driver internally
    import asyncio
    loop = asyncio.get_event_loop()
    neo4j_driver = get_neo4j_driver()
    qdrant_client = get_qdrant_async_client()

    graphrag_results = await loop.run_in_executor(
        None,
        lambda: local_search(
            query=body.query_text,
            neo4j_driver=neo4j_driver,
            qdrant_client=None,  # Not used in current implementation
            db_session=None,  # Will create own session for community lookup
            limit=body.top_k,
        ),
    )

    if not graphrag_results:
        raise HTTPException(
            status_code=404,
            detail="GraphRAG local: no matching entities found in the knowledge graph. "
            "Ensure documents have been ingested with successful graph extraction.",
        )

    # Convert GraphRAG results to QueryResultItem format
    results = []
    for i, gr in enumerate(graphrag_results):
        entity = gr.get("entity", {})
        community_reports = gr.get("community_reports", [])

        results.append(
            QueryResultItem(
                score=max(0.5, 1.0 - (i * 0.05)),  # Rank-based scoring
                modality="graph_node",
                content_text=entity.get("name", ""),
                page_number=None,
                classification="UNCLASSIFIED",
                context={
                    "source": "graphrag_local",
                    "entity_type": gr.get("entity_type"),
                    "entity": entity,
                    "community_reports": [
                        {"title": r.get("title"), "summary": r.get("summary")}
                        for r in community_reports[:3]
                    ],
                },
            )
        )

    return results[:body.top_k]


# ---------------------------------------------------------------------------
# GraphRAG global search — cross-community summarization
# ---------------------------------------------------------------------------

async def _graphrag_global_query(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    """Cross-community summarization for broad questions."""
    if not body.query_text:
        return []

    from app.config import get_settings
    settings = get_settings()
    if not settings.graphrag_indexing_enabled:
        raise HTTPException(
            status_code=409,
            detail="GraphRAG global search is unavailable: GRAPHRAG_INDEXING_ENABLED is false.",
        )

    from app.services.graphrag_service import global_search
    from app.db.session import SyncSessionFactory

    # global_search uses sync SQLAlchemy session
    import asyncio
    loop = asyncio.get_event_loop()

    def _run_global():
        session = SyncSessionFactory()
        try:
            return global_search(
                query=body.query_text,
                db_session=session,
                limit=body.top_k,
            )
        finally:
            session.close()

    graphrag_results = await loop.run_in_executor(None, _run_global)

    if not graphrag_results:
        raise HTTPException(
            status_code=409,
            detail="GraphRAG global: no community reports available. "
            "Run the GraphRAG indexing pipeline to generate community reports.",
        )

    results = []
    for gr in graphrag_results:
        results.append(
            QueryResultItem(
                score=gr.get("rank", 0.5) or 0.5,
                modality="community_report",
                content_text=gr.get("summary") or gr.get("report_text", "")[:500],
                page_number=None,
                classification="UNCLASSIFIED",
                context={
                    "source": "graphrag_global",
                    "community_id": gr.get("community_id"),
                    "community_title": gr.get("community_title"),
                    "level": gr.get("level"),
                    "report_text": gr.get("report_text"),
                },
            )
        )

    return results[:body.top_k]


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

async def _backfill_content_text(
    db: AsyncSession, results: list[QueryResultItem]
) -> None:
    """Batch-fill content_text from Postgres for results missing it.

    Handles pre-existing Qdrant points that were indexed without chunk_text
    in their payload.  Queries both text_chunks and image_chunks tables.
    """
    missing = [
        r for r in results
        if r.content_text is None and r.chunk_id is not None
    ]
    if not missing:
        return

    chunk_ids = [str(r.chunk_id) for r in missing]

    # Batch lookup from both chunk tables
    sql = text("""
        SELECT id::text, chunk_text FROM retrieval.text_chunks
        WHERE id = ANY(:ids)
        UNION ALL
        SELECT id::text, chunk_text FROM retrieval.image_chunks
        WHERE id = ANY(:ids)
    """)
    rows = (await db.execute(sql, {"ids": chunk_ids})).fetchall()
    text_map = {row[0]: row[1] for row in rows if row[1]}

    for r in missing:
        cid = str(r.chunk_id)
        if cid in text_map:
            r.content_text = text_map[cid]


async def _populate_image_urls(
    db: AsyncSession, results: list[QueryResultItem]
) -> None:
    """Set image_url to the API proxy path for image-modality results.

    Uses /v1/images/{chunk_id} which streams from MinIO, avoiding
    presigned URLs that contain the Docker-internal MinIO hostname.
    """
    for result in results:
        if result.modality in ("image", "schematic") and result.chunk_id:
            result.image_url = f"/v1/images/{result.chunk_id}"


# ---------------------------------------------------------------------------
# GraphRAG manual indexing trigger
# ---------------------------------------------------------------------------


@router.post("/graphrag/index")
async def trigger_graphrag_indexing():
    """Dispatch GraphRAG community detection + report generation as a Celery task.

    The task is idempotent — a Redis lock prevents overlapping runs.
    """
    from app.workers.graphrag_tasks import run_graphrag_indexing_task

    task = run_graphrag_indexing_task.delay()
    return {"status": "indexing_started", "task_id": str(task.id)}


# ---------------------------------------------------------------------------
# Memory query — Cognee search (unchanged)
# ---------------------------------------------------------------------------

