#!/usr/bin/env python3
"""Migrate pgvector embeddings to Qdrant.

Reads text_chunks and image_chunks from PostgreSQL (pgvector embeddings)
and upserts them into Qdrant collections. Idempotent — uses chunk IDs as
Qdrant point IDs.

Usage:
    python scripts/migrate_pgvector_to_qdrant.py

Requires both PostgreSQL (with pgvector data) and Qdrant to be running.
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings


def main():
    settings = get_settings()

    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
    from sqlalchemy import create_engine, text

    engine = create_engine(settings.sync_database_url)
    qdrant = QdrantClient(url=settings.qdrant_url)

    # Ensure collections exist
    from app.services.qdrant_store import ensure_collections

    ensure_collections(qdrant)

    # Migrate text chunks
    print("Migrating text chunk embeddings...")
    text_count = 0
    batch: list[PointStruct] = []
    batch_size = 100

    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT id, artifact_id, document_id, modality, classification, "
            "page_number, embedding "
            "FROM retrieval.text_chunks "
            "WHERE embedding IS NOT NULL"
        ))
        for row in result.fetchall():
            chunk_id = str(row[0])
            embedding = list(row[6]) if row[6] else None
            if not embedding:
                continue

            batch.append(PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,
                    "artifact_id": str(row[1]),
                    "document_id": str(row[2]),
                    "modality": row[3],
                    "classification": row[4],
                    "page_number": row[5],
                },
            ))
            text_count += 1

            if len(batch) >= batch_size:
                qdrant.upsert(
                    collection_name=settings.qdrant_text_collection,
                    points=batch,
                )
                batch = []

        if batch:
            qdrant.upsert(
                collection_name=settings.qdrant_text_collection,
                points=batch,
            )
            batch = []

    print(f"Migrated {text_count} text chunk embeddings")

    # Update qdrant_point_id in Postgres
    print("Updating qdrant_point_id in text_chunks...")
    with engine.connect() as conn:
        conn.execute(text(
            "UPDATE retrieval.text_chunks "
            "SET qdrant_point_id = id "
            "WHERE embedding IS NOT NULL AND qdrant_point_id IS NULL"
        ))
        conn.commit()

    # Migrate image chunks
    print("Migrating image chunk embeddings...")
    image_count = 0

    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT id, artifact_id, document_id, modality, classification, "
            "page_number, embedding "
            "FROM retrieval.image_chunks "
            "WHERE embedding IS NOT NULL"
        ))
        for row in result.fetchall():
            chunk_id = str(row[0])
            embedding = list(row[6]) if row[6] else None
            if not embedding:
                continue

            batch.append(PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,
                    "artifact_id": str(row[1]),
                    "document_id": str(row[2]),
                    "modality": row[3],
                    "classification": row[4],
                    "page_number": row[5],
                },
            ))
            image_count += 1

            if len(batch) >= batch_size:
                qdrant.upsert(
                    collection_name=settings.qdrant_image_collection,
                    points=batch,
                )
                batch = []

        if batch:
            qdrant.upsert(
                collection_name=settings.qdrant_image_collection,
                points=batch,
            )

    print(f"Migrated {image_count} image chunk embeddings")

    # Update qdrant_point_id in Postgres
    print("Updating qdrant_point_id in image_chunks...")
    with engine.connect() as conn:
        conn.execute(text(
            "UPDATE retrieval.image_chunks "
            "SET qdrant_point_id = id "
            "WHERE embedding IS NOT NULL AND qdrant_point_id IS NULL"
        ))
        conn.commit()

    print(f"Migration complete: {text_count} text + {image_count} image vectors")


if __name__ == "__main__":
    main()
