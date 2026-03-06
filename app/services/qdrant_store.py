"""Qdrant vector store operations.

Provides sync helpers for Celery workers and async helpers for FastAPI routes.
Manages two collections:
  - eip_text_chunks: 1024-dim cosine (BGE-large-en-v1.5)
  - eip_image_chunks: 512-dim cosine (OpenCLIP ViT-B/32)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collection initialization
# ---------------------------------------------------------------------------

def ensure_collections(client) -> None:
    """Create Qdrant collections if they don't already exist."""
    settings = get_settings()
    _ensure_collection(
        client,
        settings.qdrant_text_collection,
        dim=settings.text_embedding_dim,
    )
    _ensure_collection(
        client,
        settings.qdrant_image_collection,
        dim=settings.image_embedding_dim,
    )


def _ensure_collection(client, name: str, dim: int) -> None:
    """Create a single collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        logger.info("Qdrant collection '%s' already exists", name)
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Create payload indexes for filtering
    for field in ("document_id", "artifact_id", "modality", "classification", "page_number"):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema="keyword" if field != "page_number" else "integer",
            )
        except Exception:
            pass  # index may already exist

    logger.info("Created Qdrant collection '%s' (dim=%d, cosine)", name, dim)


# ---------------------------------------------------------------------------
# Sync upsert (Celery workers)
# ---------------------------------------------------------------------------

def upsert_text_vector(
    client,
    point_id: uuid.UUID,
    vector: list[float],
    payload: dict[str, Any],
) -> None:
    """Upsert a single text vector into the text collection."""
    settings = get_settings()
    client.upsert(
        collection_name=settings.qdrant_text_collection,
        points=[
            PointStruct(
                id=str(point_id),
                vector=vector,
                payload=payload,
            )
        ],
    )


def upsert_image_vector(
    client,
    point_id: uuid.UUID,
    vector: list[float],
    payload: dict[str, Any],
) -> None:
    """Upsert a single image vector into the image collection."""
    settings = get_settings()
    client.upsert(
        collection_name=settings.qdrant_image_collection,
        points=[
            PointStruct(
                id=str(point_id),
                vector=vector,
                payload=payload,
            )
        ],
    )


def upsert_text_vectors_batch(
    client,
    points: list[PointStruct],
) -> None:
    """Batch upsert text vectors."""
    settings = get_settings()
    client.upsert(
        collection_name=settings.qdrant_text_collection,
        points=points,
    )


def upsert_image_vectors_batch(
    client,
    points: list[PointStruct],
) -> None:
    """Batch upsert image vectors."""
    settings = get_settings()
    client.upsert(
        collection_name=settings.qdrant_image_collection,
        points=points,
    )


# ---------------------------------------------------------------------------
# Sync search (Celery workers)
# ---------------------------------------------------------------------------

def search_text_vectors(
    client,
    query_vector: list[float],
    limit: int = 20,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Search the text collection. Returns list of {id, score, payload}."""
    settings = get_settings()
    qdrant_filter = _build_filter(filters) if filters else None

    results = client.query_points(
        collection_name=settings.qdrant_text_collection,
        query=query_vector,
        limit=limit,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [
        {
            "id": str(point.id),
            "score": point.score,
            "payload": point.payload,
        }
        for point in results.points
    ]


def search_image_vectors(
    client,
    query_vector: list[float],
    limit: int = 20,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Search the image collection. Returns list of {id, score, payload}."""
    settings = get_settings()
    qdrant_filter = _build_filter(filters) if filters else None

    results = client.query_points(
        collection_name=settings.qdrant_image_collection,
        query=query_vector,
        limit=limit,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [
        {
            "id": str(point.id),
            "score": point.score,
            "payload": point.payload,
        }
        for point in results.points
    ]


# ---------------------------------------------------------------------------
# Async search (FastAPI)
# ---------------------------------------------------------------------------

async def search_text_vectors_async(
    client,
    query_vector: list[float],
    limit: int = 20,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Async search the text collection."""
    settings = get_settings()
    qdrant_filter = _build_filter(filters) if filters else None

    results = await client.query_points(
        collection_name=settings.qdrant_text_collection,
        query=query_vector,
        limit=limit,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [
        {
            "id": str(point.id),
            "score": point.score,
            "payload": point.payload,
        }
        for point in results.points
    ]


async def search_image_vectors_async(
    client,
    query_vector: list[float],
    limit: int = 20,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Async search the image collection."""
    settings = get_settings()
    qdrant_filter = _build_filter(filters) if filters else None

    results = await client.query_points(
        collection_name=settings.qdrant_image_collection,
        query=query_vector,
        limit=limit,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [
        {
            "id": str(point.id),
            "score": point.score,
            "payload": point.payload,
        }
        for point in results.points
    ]


# ---------------------------------------------------------------------------
# Delete helpers
# ---------------------------------------------------------------------------

def delete_by_document_id(client, document_id: str) -> None:
    """Delete all vectors for a given document from both collections."""
    settings = get_settings()
    doc_filter = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    )
    for collection in (settings.qdrant_text_collection, settings.qdrant_image_collection):
        try:
            client.delete(collection_name=collection, points_selector=doc_filter)
        except Exception as e:
            logger.warning("delete_by_document_id failed for %s/%s: %s", collection, document_id, e)


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------

def _build_filter(filters: dict[str, Any]) -> Filter:
    """Build a Qdrant Filter from a simple {field: value} dict.

    Scalar values use MatchValue; list values use MatchAny (OR semantics).
    """
    conditions = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, list):
            conditions.append(
                FieldCondition(key=key, match=MatchAny(any=value))
            )
        else:
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=str(value)))
            )
    return Filter(must=conditions) if conditions else Filter()
