"""Cross-encoder reranker for retrieval result re-scoring.

Uses BAAI/bge-reranker-v2-m3 (or configurable model) to re-score
the top-N retrieval candidates against the actual query text.

Runs on CPU by default (RERANKER_DEVICE=cpu). Set to 'cuda' for GPU.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reranker_model():
    """Load and cache the cross-encoder reranker model."""
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    model_name = settings.reranker_model
    device = settings.reranker_device
    logger.info("Loading reranker model: %s (device=%s)", model_name, device)
    model = CrossEncoder(model_name, device=device)
    logger.info("Reranker model loaded")
    return model


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Re-score candidates using cross-encoder and return top_k sorted by relevance.

    Args:
        query: The search query text.
        candidates: List of result dicts, each must have 'content_text'.
        top_k: Number of top results to return.

    Returns:
        Re-scored and sorted candidates (top_k).
    """
    settings = get_settings()
    if not settings.reranker_enabled:
        return candidates

    if not candidates or not query:
        return candidates[:top_k]

    # Filter candidates that have text to score
    scoreable = [(i, c) for i, c in enumerate(candidates) if c.get("content_text")]
    unscorable = [c for c in candidates if not c.get("content_text")]

    if not scoreable:
        return candidates[:top_k]

    model = _get_reranker_model()

    # Build query-document pairs
    pairs = [(query, c["content_text"]) for _, c in scoreable]
    scores = model.predict(pairs)

    # Attach scores and sort
    scored = []
    for (orig_idx, candidate), score in zip(scoreable, scores):
        candidate = dict(candidate)  # copy to avoid mutation
        candidate["reranker_score"] = float(score)
        scored.append(candidate)

    scored.sort(key=lambda x: x["reranker_score"], reverse=True)

    # Append unscorable items at the end
    result = scored + unscorable
    return result[:top_k]
