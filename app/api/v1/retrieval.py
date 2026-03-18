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
    diversify_results as _diversify_results,
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
        elif body.strategy == QueryStrategy.graphrag_drift:
            results = await _graphrag_drift_query(db, body)
        elif body.strategy == QueryStrategy.graphrag_basic:
            results = await _graphrag_basic_query(db, body)
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
# Cross-encoder reranker helper
# ---------------------------------------------------------------------------

def _apply_reranker(
    results: list[QueryResultItem], body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    """Re-rank results using cross-encoder if enabled and query_text is present."""
    from app.config import get_settings
    from app.services.reranker import rerank as cross_encoder_rerank

    _s = get_settings()
    if not _s.reranker_enabled or not body.query_text:
        return results

    rerank_input = [
        {
            "chunk_id": str(r.chunk_id or ""),
            "content_text": r.content_text or "",
            "score": r.score,
            "artifact_id": r.artifact_id,
            "document_id": r.document_id,
            "modality": r.modality,
            "page_number": r.page_number,
            "classification": r.classification,
        }
        for r in results[:_s.reranker_top_n]
    ]
    reranked = cross_encoder_rerank(body.query_text, rerank_input, top_k=body.top_k)

    # Rebuild result items from reranked dicts
    return [
        QueryResultItem(
            chunk_id=r["chunk_id"],
            artifact_id=r.get("artifact_id"),
            document_id=r.get("document_id"),
            score=r.get("reranker_score", r.get("score", 0.0)),
            modality=r.get("modality", "text"),
            content_text=r.get("content_text"),
            page_number=r.get("page_number"),
            classification=r.get("classification", "UNCLASSIFIED"),
        )
        for r in reranked
    ]


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

    # Step 2b: Re-score ontology-expanded chunks independently against the query
    expanded = await _rescore_expanded_chunks(expanded, body.query_text)

    all_results = seed_results + expanded

    # Step 3: Deduplicate by chunk_id, then by content
    deduped = _deduplicate_results(all_results)
    deduped = _diversify_results(deduped)

    # Step 4: Filter by modality
    if body.modality_filter == ModalityFilter.text:
        deduped = [r for r in deduped if r.modality in ("text", "table")]
    elif body.modality_filter == ModalityFilter.image:
        deduped = [r for r in deduped if r.modality in ("image", "schematic")]

    # Sort by score descending, cap at top_k
    deduped.sort(key=lambda x: x.score, reverse=True)
    final = deduped[:body.top_k]

    # Re-rank top candidates using cross-encoder
    final = _apply_reranker(final, body)

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
# Re-score expanded chunks independently against the query
# ---------------------------------------------------------------------------

async def _rescore_expanded_chunks(
    expanded: list[QueryResultItem],
    query_text: str | None,
) -> list[QueryResultItem]:
    """Re-score ontology-expanded chunks using embedding similarity to the query.

    Ontology-expanded chunks initially inherit a decayed fusion score from their
    parent seed.  This replaces that score with the chunk's actual cosine
    similarity to the query embedding, preventing low-relevance expansions from
    ranking artificially high.
    """
    if not expanded or not query_text:
        return expanded

    # Only re-score ontology-sourced text chunks (they have content_text)
    ontology_chunks = [
        c for c in expanded
        if (c.context or {}).get("source") == "ontology"
        and c.content_text
    ]

    if not ontology_chunks:
        return expanded

    import asyncio
    import numpy as np
    from app.services.embedding import embed_texts

    loop = asyncio.get_event_loop()

    # Run embedding in executor to avoid blocking the event loop
    chunk_texts = [c.content_text for c in ontology_chunks]

    def _embed():
        query_emb = np.array(embed_texts([query_text], query=True)[0])
        chunk_embs = np.array(embed_texts(chunk_texts))
        # Cosine similarity (embeddings are already L2-normalized by BGE)
        similarities = chunk_embs @ query_emb
        return similarities

    similarities = await loop.run_in_executor(None, _embed)

    for chunk, sim in zip(ontology_chunks, similarities):
        chunk.score = max(float(sim), 0.0)

    return expanded


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

    query_embedding = embed_texts([body.query_text], query=True)[0]

    # Build Qdrant filter from request filters
    qdrant_filters = _build_qdrant_filters(body)

    from app.config import get_settings as _get_settings
    _settings = _get_settings()

    # Over-fetch to compensate for content-level dedup
    oversample = min(
        body.top_k * _settings.retrieval_diversity_oversample_factor,
        _settings.retrieval_diversity_max_candidates,
    )

    qdrant_client = get_qdrant_async_client()
    hits = await search_text_vectors_async(
        qdrant_client,
        query_vector=query_embedding,
        limit=oversample,
        filters=qdrant_filters,
    )

    # Filter by minimum score threshold
    min_score = _settings.retrieval_min_score_threshold
    hits = [h for h in hits if h.get("score", 0.0) >= min_score]

    # Map Qdrant results — always fetch chunk_text for content dedup
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
                content_text=payload.get("chunk_text"),
                page_number=payload.get("page_number"),
                classification=payload.get("classification", "UNCLASSIFIED"),
            )
        )

    # Content-level diversification, then trim to requested top_k
    results = _diversify_results(results)
    results.sort(key=lambda r: r.score, reverse=True)
    results = results[:body.top_k]

    # Re-rank top candidates using cross-encoder
    results = _apply_reranker(results, body)

    # Strip content_text if not requested
    if not body.include_context:
        for r in results:
            r.content_text = None

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

    from app.config import get_settings as _get_settings
    settings = _get_settings()

    # Over-fetch to allow score filtering while still returning enough results
    oversample = min(
        body.top_k * settings.retrieval_diversity_oversample_factor,
        settings.retrieval_diversity_max_candidates,
    )

    qdrant_client = get_qdrant_async_client()
    hits = await search_image_vectors_async(
        qdrant_client,
        query_vector=query_embedding,
        limit=oversample,
        filters=qdrant_filters,
    )

    # Filter by minimum score threshold
    min_score = settings.retrieval_min_score_threshold
    hits = [h for h in hits if h.get("score", 0.0) >= min_score]

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

    # Sort by score and trim to requested top_k
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:body.top_k]


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
    """Entity-centric search with community report context (Microsoft GraphRAG)."""
    if not body.query_text:
        return []

    from app.services.graphrag_service import local_search

    loop = asyncio.get_event_loop()
    graphrag_result = await loop.run_in_executor(
        None, local_search, body.query_text,
    )

    response = graphrag_result.get("response", "")
    if not response:
        raise HTTPException(
            status_code=404,
            detail="GraphRAG local: no matching entities found in the knowledge graph.",
        )

    return [QueryResultItem(
        score=1.0,
        modality="graphrag_response",
        content_text=response,
        classification="UNCLASSIFIED",
        context={
            "source": "graphrag_local",
            "graphrag_context": graphrag_result.get("context", {}),
        },
    )]


# ---------------------------------------------------------------------------
# GraphRAG global search -- cross-community summarization
# ---------------------------------------------------------------------------

async def _graphrag_global_query(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    """Cross-community summarization for broad questions (Microsoft GraphRAG)."""
    if not body.query_text:
        return []

    from app.services.graphrag_service import global_search

    loop = asyncio.get_event_loop()
    graphrag_result = await loop.run_in_executor(
        None, global_search, body.query_text,
    )

    response = graphrag_result.get("response", "")
    if not response:
        raise HTTPException(
            status_code=409,
            detail="GraphRAG global: no community reports available. "
            "Run the GraphRAG indexing pipeline first.",
        )

    return [QueryResultItem(
        score=1.0,
        modality="graphrag_response",
        content_text=response,
        classification="UNCLASSIFIED",
        context={
            "source": "graphrag_global",
            "graphrag_context": graphrag_result.get("context", {}),
        },
    )]


# ---------------------------------------------------------------------------
# GraphRAG DRIFT search -- community-informed expansion
# ---------------------------------------------------------------------------

async def _graphrag_drift_query(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    """Community-informed expansion search (Microsoft GraphRAG DRIFT)."""
    if not body.query_text:
        return []

    from app.services.graphrag_service import drift_search

    loop = asyncio.get_event_loop()
    graphrag_result = await loop.run_in_executor(
        None, drift_search, body.query_text,
    )

    response = graphrag_result.get("response", "")
    if not response:
        return []

    return [QueryResultItem(
        score=1.0,
        modality="graphrag_response",
        content_text=response,
        classification="UNCLASSIFIED",
        context={
            "source": "graphrag_drift",
            "graphrag_context": graphrag_result.get("context", {}),
        },
    )]


# ---------------------------------------------------------------------------
# GraphRAG basic search -- vector search over text units
# ---------------------------------------------------------------------------

async def _graphrag_basic_query(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    """Vector search over text units (Microsoft GraphRAG basic)."""
    if not body.query_text:
        return []

    from app.services.graphrag_service import basic_search

    loop = asyncio.get_event_loop()
    graphrag_result = await loop.run_in_executor(
        None, basic_search, body.query_text,
    )

    response = graphrag_result.get("response", "")
    if not response:
        return []

    return [QueryResultItem(
        score=1.0,
        modality="graphrag_response",
        content_text=response,
        classification="UNCLASSIFIED",
        context={
            "source": "graphrag_basic",
            "graphrag_context": graphrag_result.get("context", {}),
        },
    )]


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

async def _lookup_chunk(
    db: AsyncSession, chunk_id: str, table: str = "text_chunks",
    include_context: bool = True,
) -> QueryResultItem | None:
    """Lookup a single chunk by ID from the given retrieval table."""
    sql = text(f"""
        SELECT id, artifact_id, document_id, chunk_text, modality,
               page_number, classification
        FROM retrieval.{table} WHERE id = :cid
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


async def _lookup_chunk_by_type(
    db: AsyncSession, chunk_id: str, chunk_type: str, include_context: bool = True
) -> QueryResultItem | None:
    """Route to the correct lookup based on chunk_type."""
    table = "image_chunks" if chunk_type == "image_chunk" else "text_chunks"
    return await _lookup_chunk(db, chunk_id, table, include_context)


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
    """Dispatch GraphRAG indexing as a Celery task.

    The task is idempotent -- a Redis lock prevents overlapping runs.
    """
    from app.workers.graphrag_tasks import run_graphrag_indexing_task

    task = run_graphrag_indexing_task.delay()
    return {"status": "indexing_started", "task_id": str(task.id)}


@router.post("/graphrag/tune")
async def trigger_graphrag_tuning():
    """Dispatch GraphRAG prompt auto-tuning as a Celery task."""
    from app.workers.graphrag_tasks import run_graphrag_auto_tune_task

    task = run_graphrag_auto_tune_task.delay()
    return {"status": "tuning_started", "task_id": str(task.id)}


# ---------------------------------------------------------------------------
# Memory query — Cognee search (unchanged)
# ---------------------------------------------------------------------------

