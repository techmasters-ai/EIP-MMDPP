"""Embedding services.

Text: Ollama API with bge-large (1024-dim) — shared by both text ingest and GraphRAG
Image: OpenCLIP ViT-B/32 (512-dim) for cross-modal search

Text embeddings are served by Ollama's OpenAI-compatible /v1/embeddings endpoint.
Image embeddings run locally via OpenCLIP.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import httpx
import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

# Reusable HTTP client for Ollama embedding calls
_http_client: httpx.Client | None = None


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=120.0)
    return _http_client


def embed_texts(texts: list[str], batch_size: int = 64, *, query: bool = False) -> list[list[float]]:
    """Embed a list of text strings via Ollama's OpenAI-compatible API.

    Args:
        texts: List of input strings.
        batch_size: Max texts per API call (Ollama handles batches internally).
        query: If True, use the BGE query prefix (for search queries).
               If False, use the passage prefix (for indexing documents).

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    if not texts:
        return []

    settings = get_settings()

    # BGE models use asymmetric prefixes: different for queries vs passages
    if "bge" in settings.text_embedding_model.lower():
        if query:
            texts = [f"Represent this query for searching relevant passages: {t}" for t in texts]
        else:
            texts = [f"Represent this sentence: {t}" for t in texts]

    client = _get_http_client()
    api_url = f"{settings.ollama_base_url}/v1/embeddings"
    all_embeddings: list[list[float]] = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.post(
            api_url,
            json={"model": settings.text_embedding_model, "input": batch},
        )
        resp.raise_for_status()
        data = resp.json()
        # Sort by index to maintain order
        items = sorted(data["data"], key=lambda x: x["index"])
        all_embeddings.extend(item["embedding"] for item in items)

    # L2-normalize
    arr = np.array(all_embeddings)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    arr = arr / norms
    return arr.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single search query."""
    return embed_texts([query], query=True)[0]


# ---------------------------------------------------------------------------
# Image embedding (OpenCLIP ViT-B/32 — shared text/image space)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_clip_model():
    """Load and cache the OpenCLIP model and preprocessing transforms."""
    import open_clip

    s = get_settings()
    model_name = s.image_embedding_model
    pretrained = s.image_embedding_pretrained
    logger.info("Loading CLIP model: %s / %s", model_name, pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    logger.info("CLIP model loaded (dim=%d)", s.image_embedding_dim)
    return model, preprocess, tokenizer


def embed_images(pil_images: list) -> list[list[float]]:
    """Embed a list of PIL Images using CLIP. Returns float vectors.

    Args:
        pil_images: List of PIL.Image.Image objects.

    Returns:
        List of 512-dim float vectors.
    """
    if not pil_images:
        return []

    import torch

    model, preprocess, _ = _get_clip_model()

    images_tensor = torch.stack([preprocess(img) for img in pil_images])
    with torch.no_grad():
        features = model.encode_image(images_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().tolist()


def embed_image(pil_image) -> list[float]:
    """Embed a single PIL Image."""
    return embed_images([pil_image])[0]


def embed_text_for_clip(text: str) -> list[float]:
    """Embed a text string in the CLIP image embedding space for cross-modal queries."""
    import torch
    import open_clip

    model, _, tokenizer = _get_clip_model()

    tokens = tokenizer([text])
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().tolist()[0]
