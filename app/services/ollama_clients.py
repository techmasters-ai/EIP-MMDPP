"""Module-level cached factories for the role-scoped Ollama clients.

One pool per role (LLM / VLM / embedding). Cached via `lru_cache` so each
process has at most one client per role; tests can `clear_cache()` to
rebuild against patched env.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.ollama_pool_client import (
    OllamaChatClient, OllamaEmbeddingClient, OllamaPool,
)


@lru_cache(maxsize=1)
def get_llm_client() -> OllamaChatClient:
    s = get_settings()
    return OllamaChatClient(
        pool=OllamaPool(urls=s.get_ollama_llm_urls()),
        model=s.doc_analysis_llm_model,  # default; per-call overrides via .chat(model=...)
        timeout_s=float(s.doc_analysis_timeout),
        max_tokens=s.llm_max_tokens,
    )


@lru_cache(maxsize=1)
def get_vlm_client() -> OllamaChatClient:
    s = get_settings()
    return OllamaChatClient(
        pool=OllamaPool(urls=s.get_ollama_vlm_urls()),
        model=s.picture_description_model,
        timeout_s=float(s.picture_description_timeout),
        max_tokens=s.llm_max_tokens,
    )


@lru_cache(maxsize=1)
def get_embedding_client() -> OllamaEmbeddingClient:
    s = get_settings()
    return OllamaEmbeddingClient(
        pool=OllamaPool(urls=s.get_ollama_embedding_urls()),
        model=s.text_embedding_model,
        timeout_s=120.0,
    )
