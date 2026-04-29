"""Tests for the module-level pool/client factory cache."""
from app.services.ollama_clients import (
    get_llm_client, get_vlm_client, get_embedding_client,
)


def test_llm_client_is_cached_singleton():
    c1 = get_llm_client()
    c2 = get_llm_client()
    assert c1 is c2


def test_factories_use_role_specific_pools(monkeypatch):
    # Order matters: clear caches BEFORE patching env, otherwise a previous
    # test's cached singleton wins and the new env values are ignored.
    from app.config import get_settings
    from app.services import ollama_clients
    get_settings.cache_clear()
    ollama_clients.get_llm_client.cache_clear()
    ollama_clients.get_vlm_client.cache_clear()
    ollama_clients.get_embedding_client.cache_clear()
    monkeypatch.setenv("OLLAMA_LLM_BASE_URLS", '["http://llm-1"]')
    monkeypatch.setenv("OLLAMA_VLM_BASE_URLS", '["http://vlm-1"]')
    monkeypatch.setenv("OLLAMA_EMBEDDING_BASE_URLS", '["http://emb-1"]')

    assert get_llm_client().pool.urls == ["http://llm-1"]
    assert get_vlm_client().pool.urls == ["http://vlm-1"]
    assert get_embedding_client().pool.urls == ["http://emb-1"]
