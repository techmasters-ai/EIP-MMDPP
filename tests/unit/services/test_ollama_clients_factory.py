"""Tests for the module-level pool/client factory cache."""
import pytest

from app.config import get_settings
from app.services import ollama_clients
from app.services.ollama_clients import (
    get_embedding_client,
    get_llm_client,
    get_vlm_client,
)


@pytest.fixture(autouse=True)
def _clear_factory_caches():
    """Hermetic isolation for every test in this file.

    The factories are module-level lru_caches, so a prior test (or another
    test file run earlier in the session) can leave a singleton in place
    that captures the wrong env. Clear before AND after every test so
    other suites don't inherit our mock URLs.
    """
    get_settings.cache_clear()
    ollama_clients.get_llm_client.cache_clear()
    ollama_clients.get_vlm_client.cache_clear()
    ollama_clients.get_embedding_client.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()
        ollama_clients.get_llm_client.cache_clear()
        ollama_clients.get_vlm_client.cache_clear()
        ollama_clients.get_embedding_client.cache_clear()


def test_llm_client_is_cached_singleton():
    c1 = get_llm_client()
    c2 = get_llm_client()
    assert c1 is c2


def test_factories_use_role_specific_pools(monkeypatch):
    monkeypatch.setenv("OLLAMA_LLM_BASE_URLS", '["http://llm-1"]')
    monkeypatch.setenv("OLLAMA_VLM_BASE_URLS", '["http://vlm-1"]')
    monkeypatch.setenv("OLLAMA_EMBEDDING_BASE_URLS", '["http://emb-1"]')

    assert get_llm_client().pool.urls == ["http://llm-1"]
    assert get_vlm_client().pool.urls == ["http://vlm-1"]
    assert get_embedding_client().pool.urls == ["http://emb-1"]
