"""Qdrant vector store operations.

Provides sync helpers for Celery workers and async helpers for FastAPI routes.
Manages three collections:
  - eip_text_chunks: 1024-dim cosine (BGE-large-en-v1.5)
  - eip_image_chunks: 512-dim cosine (OpenCLIP ViT-B/32)
  - eip_trusted_text: 1024-dim cosine (trusted data submissions)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

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
    _ensure_trusted_collection(client)


def _ensure_trusted_collection(client) -> None:
    """Create the trusted data collection if it doesn't exist."""
    settings = get_settings()
    name = settings.qdrant_trusted_text_collection
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        logger.info("Qdrant trusted collection '%s' already exists", name)
        return

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=settings.text_embedding_dim, distance=Distance.COSINE
        ),
    )

    for field in ("submission_id", "status", "classification"):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema="keyword",
            )
        except Exception:
            pass
    try:
        client.create_payload_index(
            collection_name=name,
            field_name="confidence",
            field_schema="float",
        )
    except Exception:
        pass

    logger.info("Created Qdrant trusted collection '%s' (dim=%d)", name, settings.text_embedding_dim)


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
# Generic upsert / search (collection-agnostic)
# ---------------------------------------------------------------------------

def upsert_vectors(client, collection: str, points: list[PointStruct]) -> None:
    """Upsert points to any collection."""
    client.upsert(collection_name=collection, points=points)


def upsert_vector(
    client,
    collection: str,
    point_id: uuid.UUID | str,
    vector: list[float],
    payload: dict[str, Any],
) -> None:
    """Upsert a single point to any collection."""
    upsert_vectors(client, collection, [
        PointStruct(id=str(point_id), vector=vector, payload=payload)
    ])


def search_vectors(
    client,
    collection: str,
    query_vector: list[float],
    limit: int = 20,
    filters: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Search any collection (sync)."""
    qdrant_filter = _build_filter(filters) if filters else None
    kwargs: dict[str, Any] = dict(
        collection_name=collection, query=query_vector,
        limit=limit, query_filter=qdrant_filter, with_payload=True,
    )
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    results = client.query_points(**kwargs)
    return [{"id": str(p.id), "score": p.score, "payload": p.payload} for p in results.points]


async def search_vectors_async(
    client,
    collection: str,
    query_vector: list[float],
    limit: int = 20,
    filters: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Search any collection (async)."""
    qdrant_filter = _build_filter(filters) if filters else None
    kwargs: dict[str, Any] = dict(
        collection_name=collection, query=query_vector,
        limit=limit, query_filter=qdrant_filter, with_payload=True,
    )
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    results = await client.query_points(**kwargs)
    return [{"id": str(p.id), "score": p.score, "payload": p.payload} for p in results.points]


# ---------------------------------------------------------------------------
# Backward-compat aliases (text / image collections)
# ---------------------------------------------------------------------------

def upsert_text_vector(client, point_id, vector, payload):
    upsert_vector(client, get_settings().qdrant_text_collection, point_id, vector, payload)


def upsert_image_vector(client, point_id, vector, payload):
    upsert_vector(client, get_settings().qdrant_image_collection, point_id, vector, payload)


def upsert_text_vectors_batch(client, points):
    upsert_vectors(client, get_settings().qdrant_text_collection, points)


def upsert_image_vectors_batch(client, points):
    upsert_vectors(client, get_settings().qdrant_image_collection, points)


def search_text_vectors(client, query_vector, limit=20, filters=None, score_threshold=None):
    return search_vectors(client, get_settings().qdrant_text_collection, query_vector, limit, filters, score_threshold)


def search_image_vectors(client, query_vector, limit=20, filters=None, score_threshold=None):
    return search_vectors(client, get_settings().qdrant_image_collection, query_vector, limit, filters, score_threshold)


async def search_text_vectors_async(client, query_vector, limit=20, filters=None, score_threshold=None):
    return await search_vectors_async(client, get_settings().qdrant_text_collection, query_vector, limit, filters, score_threshold)


async def search_image_vectors_async(client, query_vector, limit=20, filters=None, score_threshold=None):
    return await search_vectors_async(client, get_settings().qdrant_image_collection, query_vector, limit, filters, score_threshold)


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

# ---------------------------------------------------------------------------
# Trusted data operations
# ---------------------------------------------------------------------------

def upsert_trusted_vector(
    client,
    point_id: str,
    vector: list[float],
    payload: dict[str, Any],
) -> None:
    """Upsert a single point to the trusted text collection."""
    settings = get_settings()
    client.upsert(
        collection_name=settings.qdrant_trusted_text_collection,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        ],
    )


async def search_trusted_vectors(
    query_vector: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search the trusted text collection (async)."""
    from app.db.session import get_qdrant_async_client

    settings = get_settings()
    client = get_qdrant_async_client()

    results = await client.query_points(
        collection_name=settings.qdrant_trusted_text_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        {"score": point.score, **point.payload}
        for point in results.points
    ]


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
