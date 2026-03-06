"""Text vector store — direct ingest and semantic query endpoints."""

import logging
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.schemas.text_store import TextChunkIngest, TextChunkIngestResponse, TextQueryRequest
from app.schemas.retrieval import QueryResultItem

router = APIRouter(tags=["text"])
logger = logging.getLogger(__name__)


@router.post("/text/ingest", response_model=TextChunkIngestResponse)
async def ingest_text(
    body: TextChunkIngest,
    db: AsyncSession = Depends(get_async_session),
) -> TextChunkIngestResponse:
    """Embed text and store directly in text_chunks (bypasses the document pipeline)."""
    from app.models.retrieval import TextChunk
    from app.services.extraction import chunk_text
    from app.services.embedding import embed_texts

    # Chunk the input text
    text_chunks = chunk_text(body.text)
    if not text_chunks:
        text_chunks = [body.text]

    # Embed all chunks
    embeddings = embed_texts(text_chunks)

    chunk_ids = []
    for idx, (chunk_str, embedding) in enumerate(zip(text_chunks, embeddings)):
        chunk = TextChunk(
            artifact_id=body.source_id or uuid.uuid4(),  # placeholder if no artifact
            document_id=body.document_id or uuid.uuid4(),  # placeholder if no document
            chunk_index=idx,
            chunk_text=chunk_str,
            embedding=embedding,
            modality=body.modality,
            page_number=body.page_number,
            classification=body.classification,
        )
        db.add(chunk)
        chunk_ids.append(chunk.id)

    await db.commit()
    return TextChunkIngestResponse(chunk_ids=chunk_ids, chunks_created=len(chunk_ids))


@router.post("/text/query", response_model=list[QueryResultItem])
async def query_text(
    body: TextQueryRequest,
    db: AsyncSession = Depends(get_async_session),
) -> list[QueryResultItem]:
    """Semantic search on text_chunks using BGE embeddings."""
    from app.services.embedding import embed_texts

    query_embedding = embed_texts([body.query])[0]
    results = await _text_semantic_search(db, query_embedding, body.top_k, body.filters)
    return results


async def _text_semantic_search(
    db: AsyncSession,
    query_embedding: list[float],
    top_k: int = 10,
    filters=None,
) -> list[QueryResultItem]:
    """Run pgvector HNSW cosine search on text_chunks."""
    from sqlalchemy import text

    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    filter_clauses = ""
    if filters:
        if filters.classification:
            filter_clauses += f" AND tc.classification = '{filters.classification}'"
        if filters.document_ids:
            doc_ids = ",".join(f"'{d}'" for d in filters.document_ids)
            filter_clauses += f" AND tc.document_id IN ({doc_ids})"

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

    result = await db.execute(sql, {"embedding": embedding_str, "top_k": top_k})
    rows = result.fetchall()

    return [
        QueryResultItem(
            chunk_id=row[0],
            artifact_id=row[1],
            document_id=row[2],
            score=float(row[7]),
            modality=row[4],
            content_text=row[3],
            page_number=row[5],
            classification=row[6],
        )
        for row in rows
    ]
