"""Process-cached per-function Ollama client factories.

Each LLM-using function in the system gets its own factory that builds
a dedicated `OllamaChatClient` (chat/VLM) or `OllamaEmbeddingClient`
(embedding) backed by a function-specific URL pool from `Settings`.
This lets the operator route different LLM functions to different banks
of Ollama instances — e.g. doc analysis on a gpt-oss:120b host, graph
extraction on a gemma4:31b bank, embeddings on a CPU node.

Each factory is @lru_cache(maxsize=1); the first call constructs the
client, subsequent calls return it. Tests must call `<factory>.cache_clear()`
to rebuild against patched env. Env values frozen at first call:
function-specific URLs, role-level URLs, model name, timeout, and any
per-function `*_THINK` setting.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.ollama_pool_client import (
    OllamaChatClient, OllamaEmbeddingClient, OllamaPool,
)


@lru_cache(maxsize=1)
def get_doc_analysis_client() -> OllamaChatClient:
    s = get_settings()
    return OllamaChatClient(
        pool=OllamaPool(urls=s.get_doc_analysis_llm_urls()),
        model=s.doc_analysis_llm_model,
        timeout_s=float(s.doc_analysis_timeout),
        max_tokens=s.llm_max_tokens,
    )


@lru_cache(maxsize=1)
def get_translation_client() -> OllamaChatClient:
    s = get_settings()
    return OllamaChatClient(
        pool=OllamaPool(urls=s.get_translation_llm_urls()),
        model=s.translation_model,
        timeout_s=float(s.translation_timeout),
        max_tokens=s.llm_max_tokens,
    )


@lru_cache(maxsize=1)
def get_community_report_client() -> OllamaChatClient:
    """Used by both community-report generation and global-query synthesis
    (the latter is part of the global-query strategy, which uses the
    same model + timeout settings).
    """
    s = get_settings()
    return OllamaChatClient(
        pool=OllamaPool(urls=s.get_community_report_llm_urls()),
        model=s.community_report_llm_model,
        timeout_s=float(s.doc_analysis_timeout),  # historical reuse
        max_tokens=s.llm_max_tokens,
    )


@lru_cache(maxsize=1)
def get_picture_description_client() -> OllamaChatClient:
    s = get_settings()
    return OllamaChatClient(
        pool=OllamaPool(urls=s.get_picture_description_urls()),
        model=s.picture_description_model,
        timeout_s=float(s.picture_description_timeout),
        max_tokens=s.llm_max_tokens,
    )


@lru_cache(maxsize=1)
def get_text_embedding_client() -> OllamaEmbeddingClient:
    s = get_settings()
    # Embeddings are fast; 120s covers worst-case batch.
    return OllamaEmbeddingClient(
        pool=OllamaPool(urls=s.get_text_embedding_urls()),
        model=s.text_embedding_model,
        timeout_s=120.0,
    )
