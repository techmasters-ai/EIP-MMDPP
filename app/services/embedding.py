"""Local embedding services.

Text: sentence-transformers with BAAI/bge-large-en-v1.5 (1024-dim)
Image: OpenCLIP ViT-B/32 (512-dim) for cross-modal search

Both run fully locally — no cloud API calls (air-gapped deployment).
Models are loaded lazily and cached in-process.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Text embedding (BGE-large or paraphrase-MiniLM in test)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_text_model():
    """Load and cache the sentence-transformer text embedding model."""
    from sentence_transformers import SentenceTransformer

    model_name = settings.text_embedding_model
    logger.info("Loading text embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Text embedding model loaded (dim=%d)", settings.text_embedding_dim)
    return model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a list of text strings. Returns list of float vectors.

    Args:
        texts: List of input strings.
        batch_size: Batch size for GPU inference (64 maximises GPU utilisation).

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    if not texts:
        return []

    model = _get_text_model()

    # BGE models perform better with a query prefix for retrieval tasks
    if "bge" in settings.text_embedding_model.lower():
        texts = [f"Represent this sentence: {t}" for t in texts]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single search query."""
    return embed_texts([query])[0]


# ---------------------------------------------------------------------------
# Image embedding (OpenCLIP ViT-B/32 — shared text/image space)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_clip_model():
    """Load and cache the OpenCLIP model and preprocessing transforms."""
    import open_clip

    model_name = settings.image_embedding_model
    pretrained = settings.image_embedding_pretrained
    logger.info("Loading CLIP model: %s / %s", model_name, pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    logger.info("CLIP model loaded (dim=%d)", settings.image_embedding_dim)
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


def embed_text_for_image_search(text: str) -> list[float]:
    """Embed a text string in the CLIP image embedding space for cross-modal queries."""
    import torch
    import open_clip

    model, _, tokenizer = _get_clip_model()

    tokens = tokenizer([text])
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().tolist()[0]
