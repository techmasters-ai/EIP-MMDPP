#!/usr/bin/env python3
"""Initialize Qdrant collections for EIP-MMDPP.

Creates the text and image vector collections with proper dimensions
and payload indexes. Idempotent — safe to run multiple times.

Usage:
    python scripts/init_qdrant_collections.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.services.qdrant_store import ensure_collections


def main():
    settings = get_settings()

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url)
    print(f"Connected to Qdrant at {settings.qdrant_url}")

    ensure_collections(client)

    # Verify
    collections = [c.name for c in client.get_collections().collections]
    print(f"Qdrant collections: {collections}")

    for name in (settings.qdrant_text_collection, settings.qdrant_image_collection):
        if name in collections:
            info = client.get_collection(name)
            print(f"  {name}: {info.vectors_count} vectors, dim={info.config.params.vectors.size}")
        else:
            print(f"  WARNING: {name} not found!")
            sys.exit(1)

    print("Qdrant initialization complete.")


if __name__ == "__main__":
    main()
