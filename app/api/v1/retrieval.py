"""Unified retrieval endpoint — single multi-modal pipeline with mode-based filtering."""

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1._retrieval_helpers import (
    compute_fusion_score,
    deduplicate_results as _deduplicate_results,
    diversify_results as _diversify_results,
    get_cross_modal_decay,
)
from app.db.session import get_async_session, get_graph_store
from app.schemas.retrieval import (
    ModalityFilter,
    QueryFilters,
    QueryResultItem,
    QueryStrategy,
    UnifiedQueryRequest,
    UnifiedQueryResponse,
)

router = APIRouter(tags=["retrieval"])
logger = logging.getLogger(__name__)

# Max concurrent seed expansions
_EXPAND_CONCURRENCY = 16

# Shared image extension -> MIME type mapping
_IMAGE_CONTENT_TYPES = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "tiff": "image/tiff", "tif": "image/tiff", "gif": "image/gif",
    "webp": "image/webp",
}


def _apply_query_filters(
    results: list[QueryResultItem],
    filters: QueryFilters,
) -> list[QueryResultItem]:
    """Post-filter results by classification, document_ids, and modalities."""
    filtered = results
    if filters.classification:
        filtered = [r for r in filtered if r.classification == filters.classification]
    if filters.document_ids:
        doc_id_strs = {str(d) for d in filters.document_ids}
        filtered = [r for r in filtered if str(r.document_id) in doc_id_strs]
    if filters.modalities:
        mod_set = set(filters.modalities)
        filtered = [r for r in filtered if r.modality in mod_set]
    if filters.source_ids:
        # source_ids require a document→source lookup; for now filter by
        # document_name prefix or context. This is a best-effort filter
        # since source_id is not on the chunk directly.
        logger.debug("source_ids filter requested but chunk-level source_id unavailable; skipping")
    return filtered


@router.post("/retrieval/query", response_model=UnifiedQueryResponse)
async def unified_query(
    body: UnifiedQueryRequest,
    db: AsyncSession = Depends(get_async_session),
) -> UnifiedQueryResponse:
    """Unified retrieval query.

    Modes:
    - **text_basic**: BGE vector search on text_chunks only (ArcadeDB)
    - **text_only**: Multi-modal pipeline, filtered to text results
    - **images_only**: Multi-modal pipeline, filtered to image results
    - **multi_modal**: Multi-modal pipeline, all results
    - **global**: Cross-community summarization (community detection — Task 10)
    """
    try:
        if body.strategy == QueryStrategy.basic:
            results = await _text_vector_search(db, body)
        elif body.strategy == QueryStrategy.hybrid:
            results = await _multi_modal_pipeline(db, body)
        elif body.strategy == QueryStrategy.global_:
            return await _global_query(body)
        else:
            results = []
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Query strategy %s failed: %s", body.strategy, e)
        results = []

    # Apply user-requested filters (classification, document_ids, modalities)
    if body.filters:
        results = _apply_query_filters(results, body.filters)

    # Filter by minimum confidence
    if body.min_confidence is not None:
        results = [r for r in results if r.score >= body.min_confidence]

    # Backfill content_text from Postgres for results missing it (pre-existing data)
    if body.include_context:
        await _backfill_content_text(db, results)

    # Backfill page_number from Postgres for results missing it
    await _backfill_page_numbers(db, results)

    # Spec 2026-05-11-table-aware-chunking §11.5 — populate table_chunk
    # block for results whose underlying TextChunk has chunk_metadata.
    await _backfill_table_chunk_metadata(db, results)

    # Populate document names from ingest.documents
    await _backfill_document_names(db, results)

    # Populate data lineage: source characterization, date, classification
    await _backfill_document_lineage(db, results)

    # Populate presigned image URLs for image-modality results
    await _populate_image_urls(db, results, strategy=body.strategy)

    return UnifiedQueryResponse(
        query_text=body.query_text,
        query_image=body.query_image[:100] if body.query_image else None,
        strategy=body.strategy.value,
        modality_filter=body.modality_filter.value,
        results=results,
        total=len(results),
    )


@router.get("/images/artifact/{artifact_id}")
async def get_image_by_artifact(
    artifact_id: str,
    db: AsyncSession = Depends(get_async_session),
):
    """Stream an image from MinIO by artifact ID (direct lookup, no ImageChunk intermediary)."""
    sql = text("""
        SELECT storage_bucket, storage_key
        FROM ingest.artifacts
        WHERE id = CAST(:artifact_id AS uuid)
          AND storage_bucket IS NOT NULL AND storage_key IS NOT NULL
    """)
    row = (await db.execute(sql, {"artifact_id": artifact_id})).first()
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    bucket, key = row[0], row[1]
    try:
        from app.services.storage import download_bytes_async
        image_bytes = await download_bytes_async(bucket, key)
    except Exception as e:
        logger.warning("Image download failed for artifact %s: %s", artifact_id, e)
        raise HTTPException(status_code=502, detail="Failed to fetch image from storage")

    ext = key.rsplit(".", 1)[-1].lower() if "." in key else "png"
    content_type = _IMAGE_CONTENT_TYPES.get(ext, "image/png")

    return StreamingResponse(
        iter([image_bytes]),
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=3600"},
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

    ext = key.rsplit(".", 1)[-1].lower() if "." in key else "png"
    content_type = _IMAGE_CONTENT_TYPES.get(ext, "image/png")

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
    """Re-rank results using cross-encoder if enabled and query_text is present.

    Preserves ALL fields on the original QueryResultItem (context, sources,
    lineage metadata). Only the score is updated with the reranker's score.
    Content text is truncated for the reranker input only, not on the result.
    """
    from app.config import get_settings
    from app.services.reranker import rerank as cross_encoder_rerank

    _s = get_settings()
    if not _s.reranker_enabled or not body.query_text:
        return results

    _RERANK_MAX_CHARS = 512
    top_n = body.reranker_top_n or _s.reranker_top_n
    candidates = results[:top_n]
    remainder = results[top_n:]

    # Build a lookup from chunk_id to original result for score update
    by_key: dict[str, QueryResultItem] = {}
    rerank_input = []
    for r in candidates:
        key = str(r.chunk_id or id(r))
        by_key[key] = r
        rerank_input.append({
            "chunk_id": key,
            "content_text": (r.content_text or "")[:_RERANK_MAX_CHARS],
            "score": r.score,
        })

    reranked = cross_encoder_rerank(body.query_text, rerank_input, top_k=body.top_k)

    # Update scores on the ORIGINAL items, preserving context/sources/lineage
    output: list[QueryResultItem] = []
    for r in reranked:
        key = r["chunk_id"]
        original = by_key.get(key)
        if original:
            original.score = r.get("reranker_score", r.get("score", original.score))
            output.append(original)

    # Append unscorable items (no content_text) from remainder
    for r in remainder:
        if r not in output:
            output.append(r)

    return output[:body.top_k]


# ---------------------------------------------------------------------------
# Multi-modal pipeline (shared by text_only, images_only, multi_modal)
# ---------------------------------------------------------------------------

async def _multi_modal_pipeline(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    """Shared pipeline: parallel vector search + parallel graph expansion + fusion scoring.

    1. Parallel vector search (text BGE + image CLIP via ArcadeDB)
    2. Parallel per-seed expansion (doc-structure + ontology via GraphStore)
    3. Batch chunk lookups
    4. Score fusion + deduplicate + mode filter + sort + cap
    """
    t0 = time.monotonic()

    # Step 1: Parallel vector searches (ArcadeDB)
    search_tasks: list = []
    if body.query_text:
        search_tasks.append(_text_vector_search(db, body))
    # Image search: prefer image-to-image when query_image is present, text-to-CLIP otherwise
    if body.query_image or body.query_text:
        search_tasks.append(_image_vector_search(db, body))

    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
    seed_results = _merge_seed_results(search_results)
    t_search = time.monotonic()

    # Step 2: Parallel per-seed expansion (bounded concurrency, GraphStore)
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
        deduped = [r for r in deduped if r.modality in ("image", "schematic", "image_description")]

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
        from app.config import get_settings
        async with sem:
            chunk_id_str = str(seed.chunk_id)
            items: list[QueryResultItem] = []

            # Try doc-structure expansion (chunk_links table) first
            doc_items = await _expand_via_doc_structure(db, chunk_id_str, seed.score, include_context, query_text)
            if doc_items:
                items.extend(doc_items)
            else:
                # Fallback to GraphStore cross-modal for legacy documents
                cross_items = await _expand_via_cross_modal(chunk_id_str, seed.score, include_context, query_text)
                items.extend(cross_items)

            onto_items = await _expand_via_ontology(chunk_id_str, seed.score, include_context, query_text)
            items.extend(onto_items)

            if get_settings().retrieval_domain_expansion_enabled:
                domain_items = await _expand_via_domain_relations(chunk_id_str, seed.score, include_context, query_text)
                items.extend(domain_items)
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

    import numpy as np
    from app.services.embedding import embed_texts

    loop = asyncio.get_running_loop()

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
        cosine_sim = max(float(sim), 0.0)
        rel_type = (chunk.context or {}).get("rel_type", "RELATED_TO")
        chunk.score = compute_fusion_score(
            semantic_score=cosine_sim,
            ontology_rel_type=rel_type,
            ontology_hops=1,
            content_text=chunk.content_text,
            query_text=query_text,
        )

    return expanded


# ---------------------------------------------------------------------------
# Text vector search — ArcadeDB TextChunk[text_embedding] (BGE)
# ---------------------------------------------------------------------------

async def _text_vector_search(
    db: AsyncSession, body: UnifiedQueryRequest
) -> list[QueryResultItem]:
    if not body.query_text:
        return []

    from app.services.embedding import embed_texts

    query_embedding = embed_texts([body.query_text], query=True)[0]

    from app.config import get_settings as _get_settings
    _settings = _get_settings()

    # Over-fetch to compensate for content-level dedup
    oversample = min(
        body.top_k * _settings.retrieval_diversity_oversample_factor,
        _settings.retrieval_diversity_max_candidates,
    )

    graph_store = get_graph_store()
    # Push document_id filter into native ArcadeDB query when present
    doc_ids = [str(d) for d in body.filters.document_ids] if body.filters and body.filters.document_ids else None
    hits = await graph_store.vector_search(
        "TextChunk", "text_embedding", query_embedding, oversample,
        document_ids=doc_ids,
    )

    # Filter by minimum score threshold
    min_score = _settings.retrieval_min_score_threshold

    # Exclude image_description modality for basic text search — those are
    # long VLM-generated analyses that dominate results and belong in multi-modal.
    exclude_img_desc = body.strategy == QueryStrategy.basic
    results = []
    for hit in hits:
        props = hit.properties or {}
        score = hit.extraction_confidence or 0.0
        if score < min_score:
            continue
        modality = props.get("modality", "text")
        if exclude_img_desc and modality == "image_description":
            continue
        # Skip trusted_text entries from regular retrieval
        if modality == "trusted_text":
            continue
        results.append(
            QueryResultItem(
                chunk_id=props.get("chunk_id", hit.node_id),
                artifact_id=props.get("artifact_id"),
                document_id=props.get("document_id"),
                score=float(score),
                modality=modality,
                content_text=props.get("text") or props.get("chunk_text"),
                page_number=props.get("page_number"),
                classification=props.get("classification", "UNCLASSIFIED"),
                self_refs=props.get("self_refs") or [],
                evidence_ids=props.get("evidence_ids") or [],
                page_numbers=props.get("page_numbers") or [],
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
# Image vector search — ArcadeDB ImageChunk[image_embedding] (CLIP)
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

    from app.config import get_settings as _get_settings
    settings = _get_settings()

    # Over-fetch to allow score filtering while still returning enough results
    oversample = min(
        body.top_k * settings.retrieval_diversity_oversample_factor,
        settings.retrieval_diversity_max_candidates,
    )

    graph_store = get_graph_store()
    hits = await graph_store.vector_search(
        "ImageChunk", "image_embedding", query_embedding, oversample,
    )

    # Filter by minimum score threshold
    min_score = settings.retrieval_min_score_threshold

    results = []
    for hit in hits:
        props = hit.properties or {}
        score = hit.extraction_confidence or 0.0
        if score < min_score:
            continue
        results.append(
            QueryResultItem(
                chunk_id=props.get("chunk_id", hit.node_id),
                artifact_id=props.get("artifact_id"),
                document_id=props.get("document_id"),
                score=float(score),
                modality=props.get("modality", "image"),
                content_text=(props.get("text") or props.get("chunk_text")) if body.include_context else None,
                page_number=props.get("page_number"),
                classification=props.get("classification", "UNCLASSIFIED"),
                self_refs=props.get("self_refs") or [],
                evidence_ids=props.get("evidence_ids") or [],
                page_numbers=props.get("page_numbers") or [],
            )
        )

    # Sort by score and trim to requested top_k
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:body.top_k]


# ---------------------------------------------------------------------------
# Document-structure expansion (chunk_links table — stays in Postgres)
# ---------------------------------------------------------------------------

async def _expand_via_doc_structure(
    db: AsyncSession,
    chunk_id: str,
    source_score: float,
    include_context: bool,
    query_text: str | None = None,
) -> list[QueryResultItem]:
    """Expand via document-structure links.

    Primary path: ArcadeDB native graph traversal (NEXT_CHUNK, SAME_PAGE,
    SAME_SECTION, SAME_ARTIFACT edges with weight properties).
    Fallback: Postgres chunk_links table for pre-migration documents.
    """
    from app.config import get_settings
    s = get_settings()

    # Primary: ArcadeDB structural neighbor query
    graph_store = get_graph_store()
    rows = []
    try:
        arcadedb_results = await graph_store.get_structural_neighbors(
            chunk_id,
            max_hops=s.retrieval_doc_max_hops,
            limit=s.retrieval_doc_expand_k,
        )
        if arcadedb_results:
            rows = [
                (r.get("chunk_id", ""), r.get("link_type", ""), float(r.get("weight", 0.8)), 1)
                for r in arcadedb_results
                if r.get("chunk_id")
            ]
    except Exception as e:
        logger.debug("ArcadeDB doc-structure expansion failed for %s: %s", chunk_id, e)

    # Fallback: Postgres chunk_links for pre-migration documents
    if not rows:
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
            rows = [(str(r[0]), r[1], float(r[2]), int(r[3])) for r in result.fetchall()]
        except Exception as e:
            logger.debug("Postgres doc-structure expansion failed for %s: %s", chunk_id, e)
            return []

    if not rows:
        return []

    # Batch lookup all target chunks
    chunk_map = await _batch_lookup_chunks(
        db,
        [(row[0],) for row in rows],
        include_context,
    )

    items: list[QueryResultItem] = []
    for row in rows:
        target_id = row[0]
        link_type = row[1]
        weight = row[2]
        hops = row[3]

        chunk = chunk_map.get(target_id)
        if not chunk:
            continue

        chunk.score = compute_fusion_score(
            semantic_score=source_score,
            doc_structure_weight=weight,
            doc_structure_hops=hops,
            content_text=chunk.content_text,
            query_text=query_text,
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
# Graph expansion — cross-modal bridging (GraphStore structural edges, fallback)
# ---------------------------------------------------------------------------

async def _expand_via_cross_modal(
    chunk_id: str,
    source_score: float,
    include_context: bool = True,
    query_text: str | None = None,
) -> list[QueryResultItem]:
    """Follow structural graph edges (SAME_PAGE, CONTAINS_TEXT/IMAGE, EXTRACTED_FROM)
    to find connected chunks via GraphStore. Score decays from source.

    This is the fallback for documents ingested before chunk_links existed.
    """
    from app.config import get_settings
    s = get_settings()
    graph_store = get_graph_store()

    try:
        # Use ontology-linked-chunks traversal which returns actual chunk data
        # (chunk_id, chunk_type, document_id, etc.) via MATCH query.
        linked = await graph_store.get_ontology_linked_chunks(chunk_id)
        records = []
        for row in linked[:s.retrieval_doc_expand_k]:
            cid = row.get("chunk_id", "")
            ctype = row.get("chunk_type", "TextChunk")
            if cid:
                records.append({
                    "target_chunk_id": cid,
                    "target_chunk_type": "image_chunk" if ctype == "ImageChunk" else "text_chunk",
                    "entity_name": row.get("entity_name"),
                    "rel_type": "EXTRACTED_FROM",
                })
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
                chunk_data.score = compute_fusion_score(
                    semantic_score=source_score,
                    cross_modal_decay=decay,
                    content_text=chunk_data.content_text,
                    query_text=query_text,
                )
                chunk_data.context = {
                    "source": "cross_modal",
                    "source_chunk_id": chunk_id,
                    "degraded": True,
                }
                items.append(chunk_data)

    return items


# ---------------------------------------------------------------------------
# Graph expansion — ontology traversal (entity relationships via GraphStore)
# ---------------------------------------------------------------------------

async def _expand_via_ontology(
    chunk_id: str,
    source_score: float,
    include_context: bool = True,
    query_text: str | None = None,
) -> list[QueryResultItem]:
    """Follow ontology relationships (entity->related_entity->chunk) to find
    semantically related chunks via the knowledge graph."""
    from app.config import get_settings

    graph_store = get_graph_store()
    s = get_settings()
    try:
        linked = await graph_store.get_ontology_linked_chunks(chunk_id)
        linked = linked[:s.retrieval_ontology_expand_k]
    except Exception as e:
        logger.debug("Ontology expansion failed for %s: %s", chunk_id, e)
        return []

    items: list[QueryResultItem] = []

    # Look up chunk metadata from Postgres
    from app.db.session import AsyncSessionFactory
    async with AsyncSessionFactory() as db_session:
        for link in linked:
            target_id = link.get("target_chunk_id", link.get("chunk_id", ""))
            target_type = link.get("target_chunk_type", "text_chunk")

            if not target_id:
                continue

            chunk_data = await _lookup_chunk_by_type(db_session, target_id, target_type, include_context)
            if chunk_data:
                # The link's entity_name/entity_type is the intermediate entity
                # that connects the source chunk to this target chunk via
                # EXTRACTED_FROM edges. Use it for provenance display.
                entity_name = link.get("entity_name", "")
                entity_type = link.get("entity_type", "")
                chunk_data.score = compute_fusion_score(
                    semantic_score=source_score,
                    ontology_rel_type="EXTRACTED_FROM",
                    ontology_hops=1,
                    content_text=chunk_data.content_text,
                    query_text=query_text,
                )
                chunk_data.context = {
                    "source": "ontology",
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "source_chunk_id": chunk_id,
                }
                items.append(chunk_data)

    return items


async def _expand_via_domain_relations(
    chunk_id: str,
    source_score: float,
    include_context: bool = True,
    query_text: str | None = None,
) -> list[QueryResultItem]:
    """Follow CURATED domain relations (entity -[rel]- related_entity -> chunk)
    one hop to retrieve ontologically-related chunks. Augments co-mention."""
    from app.config import get_settings
    from app.api.v1._retrieval_helpers import get_curated_domain_relations, get_retrieval_relation_weights

    s = get_settings()
    graph_store = get_graph_store()
    try:
        linked = await graph_store.get_related_entity_chunks(
            chunk_id, get_curated_domain_relations(), s.retrieval_domain_expand_k,
        )
    except Exception as e:  # pragma: no cover
        logger.debug("Domain-relation expansion failed for %s: %s", chunk_id, e)
        return []

    weights = get_retrieval_relation_weights()
    items: list[QueryResultItem] = []
    from app.db.session import AsyncSessionFactory
    async with AsyncSessionFactory() as db_session:
        for link in linked:
            target_id = link.get("target_chunk_id") or link.get("chunk_id", "")
            target_type = link.get("target_chunk_type", "text_chunk")
            if not target_id:
                continue
            chunk_data = await _lookup_chunk_by_type(db_session, target_id, target_type, include_context)
            if not chunk_data:
                continue
            rel_type = str(link.get("rel_type", "RELATED_TO"))
            chunk_data.score = compute_fusion_score(
                semantic_score=source_score,
                ontology_rel_type=rel_type,
                ontology_hops=1,
                content_text=chunk_data.content_text,
                query_text=query_text,
                relation_weights=weights,
            )
            chunk_data.context = {
                "source": "ontology_relation",
                "rel_type": rel_type,
                "related_entity": link.get("related_entity", ""),
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

    Handles results where ArcadeDB vertex lacks text content.
    Queries both text_chunks and image_chunks tables.
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


async def _backfill_table_chunk_metadata(
    db: AsyncSession, results: list[QueryResultItem]
) -> None:
    """Batch-fill table_chunk block from text_chunks.chunk_metadata.

    Spec 2026-05-11-table-aware-chunking §11.5. Only fires for results
    whose underlying TextChunk has chunk_metadata IS NOT NULL — the
    partial expression index ix_text_chunks_chunk_kind keeps this fast.
    """
    from app.schemas.retrieval import TableChunkBlock

    candidates = [r for r in results if r.chunk_id is not None and r.table_chunk is None]
    if not candidates:
        return

    chunk_ids = [str(r.chunk_id) for r in candidates]
    sql = text("""
        SELECT id::text, chunk_metadata
        FROM retrieval.text_chunks
        WHERE id = ANY(:ids) AND chunk_metadata IS NOT NULL
    """)
    rows = (await db.execute(sql, {"ids": chunk_ids})).fetchall()
    by_id: dict[str, dict] = {row[0]: row[1] for row in rows if row[1]}
    for r in candidates:
        cm = by_id.get(str(r.chunk_id))
        if not cm:
            continue
        r.table_chunk = TableChunkBlock(
            kind=cm.get("chunk_kind", "unknown"),
            table_ref=cm.get("table_ref", ""),
            table_caption=cm.get("table_caption"),
            entity_display_name=cm.get("entity_display_name"),
            section=cm.get("section"),
            column_index=cm.get("column_index"),
            cell_refs=cm.get("cell_refs") or [],
            row_labels=cm.get("row_labels") or [],
        )


async def _backfill_page_numbers(
    db: AsyncSession, results: list[QueryResultItem]
) -> None:
    """Batch-fill page_number from Postgres for results missing it.

    Queries both text_chunks and image_chunks tables.
    """
    missing = [
        r for r in results
        if r.page_number is None and r.chunk_id is not None
    ]
    if not missing:
        return

    chunk_ids = [str(r.chunk_id) for r in missing]

    sql = text("""
        SELECT id::text, page_number FROM retrieval.text_chunks
        WHERE id = ANY(:ids) AND page_number IS NOT NULL
        UNION ALL
        SELECT id::text, page_number FROM retrieval.image_chunks
        WHERE id = ANY(:ids) AND page_number IS NOT NULL
    """)
    rows = (await db.execute(sql, {"ids": chunk_ids})).fetchall()
    page_map = {row[0]: row[1] for row in rows}

    for r in missing:
        cid = str(r.chunk_id)
        if cid in page_map:
            r.page_number = page_map[cid]


async def _backfill_document_names(
    db: AsyncSession, results: list[QueryResultItem]
) -> None:
    """Batch-fill document_name from ingest.documents for results that have a document_id."""
    needs_name = [
        r for r in results
        if r.document_name is None and r.document_id is not None
    ]
    if not needs_name:
        return

    doc_ids = list({str(r.document_id) for r in needs_name})

    sql = text("""
        SELECT id::text, filename FROM ingest.documents
        WHERE id = ANY(:ids)
    """)
    rows = (await db.execute(sql, {"ids": doc_ids})).fetchall()
    name_map = {row[0]: row[1] for row in rows if row[1]}

    for r in needs_name:
        did = str(r.document_id)
        if did in name_map:
            r.document_name = name_map[did]


async def _backfill_document_lineage(
    db: AsyncSession, results: list[QueryResultItem]
) -> None:
    """Batch-fill source_characterization, date_of_information, and classification
    from ingest.documents.document_metadata for results that have a document_id.

    This ensures every retrieval result carries full data lineage for
    trust, validity, and source characterization.
    """
    needs_lineage = [r for r in results if r.document_id is not None]
    if not needs_lineage:
        return

    doc_ids = list({str(r.document_id) for r in needs_lineage})

    sql = text("""
        SELECT id::text, document_metadata FROM ingest.documents
        WHERE id = ANY(:ids) AND document_metadata IS NOT NULL
    """)
    meta_map: dict[str, dict] = {}
    try:
        result = await db.execute(sql, {"ids": doc_ids})
        rows = result.fetchall()
        for row in rows:
            if row[1]:
                meta = row[1] if isinstance(row[1], dict) else __import__("json").loads(row[1])
                meta_map[row[0]] = meta
    except Exception:
        return  # Graceful degradation — lineage is best-effort

    for r in needs_lineage:
        meta = meta_map.get(str(r.document_id))
        if not meta:
            continue
        if r.source_characterization is None:
            r.source_characterization = meta.get("source_characterization")
        if r.date_of_information is None:
            r.date_of_information = meta.get("date_of_information")
        # Upgrade classification from document metadata if currently default
        if r.classification == "UNCLASSIFIED":
            doc_class = meta.get("classification")
            if doc_class and doc_class != "UNCLASSIFIED":
                r.classification = doc_class


async def _populate_image_urls(
    db: AsyncSession, results: list[QueryResultItem],
    strategy: QueryStrategy = QueryStrategy.hybrid,
) -> None:
    """Set image_url to the API proxy path for image-modality results.

    For image/schematic: uses chunk_id directly (chunk IS the image).
    For image_description: looks up the ImageChunk by artifact_id to find the image.
        Only in hybrid (multi-modal) strategy — basic text search omits images
        for image_description results.
    """
    # Direct image/schematic results
    for result in results:
        if result.modality in ("image", "schematic") and result.chunk_id:
            result.image_url = f"/v1/images/{result.chunk_id}"

    # Image description results — only attach images for hybrid (multi-modal) search
    if strategy != QueryStrategy.hybrid:
        return

    # Image description results — link directly via artifact_id (no ImageChunk intermediary)
    for r in results:
        if r.modality == "image_description" and r.artifact_id:
            r.image_url = f"/v1/images/artifact/{r.artifact_id}"


# ---------------------------------------------------------------------------
# Global query — community report search
# ---------------------------------------------------------------------------

async def _global_query(body: UnifiedQueryRequest) -> UnifiedQueryResponse:
    """Query via community reports (global/cross-community strategy).

    Embeds the query text, retrieves the most similar CommunityReport
    vertices, then calls an LLM to synthesize a single comprehensive answer
    citing the source communities. The raw reports are preserved in the
    first result's ``context`` field so the frontend can display them.
    """
    from app.db.session import get_graph_store
    from app.services.arcadedb_community import search_community_reports
    from app.services.embedding import embed_texts

    if not body.query_text:
        return UnifiedQueryResponse(
            query_text=body.query_text or "",
            strategy="global",
            modality_filter=body.modality_filter.value,
            results=[],
            total=0,
        )

    graph_store = get_graph_store()
    query_vector = embed_texts([body.query_text], query=True)[0]

    reports = await search_community_reports(graph_store, query_vector, top_k=body.top_k)

    if not reports:
        return UnifiedQueryResponse(
            query_text=body.query_text,
            strategy="global",
            modality_filter=body.modality_filter.value,
            results=[],
            total=0,
        )

    synthesis = await _synthesize_global_answer(body.query_text, reports)

    raw_reports = [
        {
            "community_id": r.get("community_id"),
            "title": r.get("title"),
            "summary": r.get("summary"),
            "member_count": r.get("member_count"),
            "score": float(r.get("score", 0.0)),
            "source_documents": r.get("source_documents", []),
        }
        for r in reports
    ]
    top_score = max((float(r.get("score", 0.0)) for r in reports), default=0.0)

    # Build deduplicated sources list from all community reports' source_documents.
    # This makes every global answer fully traceable to specific documents + pages.
    all_sources: list[dict] = []
    seen_doc_ids: set[str] = set()
    highest_classification = "UNCLASSIFIED"
    classification_rank = {"UNCLASSIFIED": 0, "CUI": 1, "FOUO": 2, "SECRET": 3, "TOP SECRET": 4}
    for r in reports:
        for src in r.get("source_documents", []):
            doc_id = src.get("document_id", "")
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                all_sources.append(src)
                src_class = src.get("classification", "UNCLASSIFIED")
                if classification_rank.get(src_class, 0) > classification_rank.get(highest_classification, 0):
                    highest_classification = src_class

    return UnifiedQueryResponse(
        query_text=body.query_text,
        strategy="global",
        modality_filter=body.modality_filter.value,
        results=[
            QueryResultItem(
                chunk_id=None,
                document_id=None,
                score=top_score,
                modality="community_report",
                content_text=synthesis,
                page_number=None,
                # Derive classification from the highest-classified source document
                classification=highest_classification,
                sources=all_sources if all_sources else None,
                context={
                    "source": "global",
                    "synthesis": True,
                    "reports": raw_reports,
                },
            )
        ],
        total=1,
    )


_DEFAULT_GLOBAL_SYNTHESIS_PROMPT = """You are answering a user question using summaries of communities of related entities from a knowledge graph.

User question:
{query}

Community reports (ranked by relevance):
{reports}

Write a single comprehensive answer that:
1. Directly addresses the user's question.
2. Draws on the community reports above.
3. Where possible, note which community report(s) a claim comes from, e.g. "[community 3]".
4. Is concise but complete. Do not repeat the reports verbatim.

Note: A complete list of source documents with page numbers is provided separately in the response metadata. Your answer does not need to duplicate that list, but referencing community IDs helps the reader trace claims.
"""


async def _synthesize_global_answer(
    query: str,
    reports: list[dict],
) -> str:
    """Call LLM to synthesize a single answer from community reports."""
    from app.config import get_settings
    from app.services.ollama_clients import get_community_report_client

    settings = get_settings()
    client = get_community_report_client()

    template = settings.community_global_synthesis_prompt or _DEFAULT_GLOBAL_SYNTHESIS_PROMPT

    reports_text = "\n\n".join(
        f"[community {r.get('community_id', '?')}] "
        f"{r.get('title', '')}\n{r.get('summary', '')}"
        for r in reports
    )
    prompt = template.replace("{query}", query).replace("{reports}", reports_text)

    timeout = settings.doc_analysis_timeout  # reused from doc_analysis_timeout pending a dedicated community_global_synthesis_timeout setting

    def _sync_call() -> str:
        return client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=settings.community_report_llm_model,
            temperature=0.2,
            max_tokens=settings.llm_max_tokens,
            think=settings.get_community_report_llm_think(),
            timeout_s=float(timeout),
        )

    try:
        content = await asyncio.to_thread(_sync_call)
        return content or _fallback_concatenated_reports(reports)
    except Exception as exc:
        logger.warning("Global synthesis LLM call failed: %s; returning concatenated reports", exc)
        return _fallback_concatenated_reports(reports)


def _fallback_concatenated_reports(reports: list[dict]) -> str:
    """Fallback answer when the synthesis LLM call fails."""
    return "\n\n".join(
        f"[community {r.get('community_id', '?')}] "
        f"{r.get('title', '')}\n{r.get('summary', '')}"
        for r in reports
    )


# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------


@router.get("/settings/retrieval")
async def get_retrieval_settings():
    """Return query defaults for the frontend."""
    from app.config import get_settings
    settings = get_settings()
    return {
        "top_k": settings.query_default_top_k,
        "reranker_top_n": settings.reranker_top_n,
        "min_confidence": settings.query_default_min_confidence,
        "ontology_reserved_slots": settings.retrieval_ontology_reserved_slots,
    }
