"""Image vector store — direct ingest and semantic query endpoints."""

import base64
import io
import logging
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.schemas.image_store import (
    ImageChunkIngest,
    ImageChunkIngestResponse,
    ImageQueryRequest,
)
from app.schemas.retrieval import QueryResultItem

router = APIRouter(tags=["images"])
logger = logging.getLogger(__name__)


@router.post("/images/ingest", response_model=ImageChunkIngestResponse)
async def ingest_image(
    body: ImageChunkIngest,
    db: AsyncSession = Depends(get_async_session),
) -> ImageChunkIngestResponse:
    """Embed image and store directly in image_chunks (bypasses the document pipeline)."""
    from PIL import Image
    from app.models.retrieval import ImageChunk
    from app.services.embedding import embed_images

    # Decode base64 image
    image_bytes = base64.b64decode(body.image)
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Embed with CLIP
    embeddings = embed_images([pil_image])

    chunk = ImageChunk(
        artifact_id=body.source_id or uuid.uuid4(),
        document_id=body.document_id or uuid.uuid4(),
        chunk_index=0,
        chunk_text=body.alt_text,
        embedding=embeddings[0],
        modality="image",
        page_number=body.page_number,
        classification=body.classification,
    )
    db.add(chunk)
    await db.commit()

    return ImageChunkIngestResponse(chunk_id=chunk.id)


@router.post("/images/query", response_model=list[QueryResultItem])
async def query_images(
    body: ImageQueryRequest,
    db: AsyncSession = Depends(get_async_session),
) -> list[QueryResultItem]:
    """Semantic search on image_chunks using CLIP embeddings."""
    from app.services.embedding import embed_images, embed_text_for_clip

    if body.query_image:
        # Image-to-image search: embed the query image with CLIP
        from PIL import Image

        image_bytes = base64.b64decode(body.query_image)
        pil_image = Image.open(io.BytesIO(image_bytes))
        query_embedding = embed_images([pil_image])[0]
    elif body.query_text:
        # Text-to-image search: embed the query text with CLIP text encoder
        query_embedding = embed_text_for_clip(body.query_text)
    else:
        return []

    results = await _image_semantic_search(db, query_embedding, body.top_k, body.filters)
    return results


async def _image_semantic_search(
    db: AsyncSession,
    query_embedding: list[float],
    top_k: int = 10,
    filters=None,
) -> list[QueryResultItem]:
    """Run pgvector HNSW cosine search on image_chunks."""
    from sqlalchemy import text

    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    filter_clauses = ""
    if filters:
        if filters.classification:
            filter_clauses += f" AND ic.classification = '{filters.classification}'"
        if filters.document_ids:
            doc_ids = ",".join(f"'{d}'" for d in filters.document_ids)
            filter_clauses += f" AND ic.document_id IN ({doc_ids})"

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
