"""Tests for the per-function cached factory module.

Each factory is @lru_cache(maxsize=1) so each process has at most one
client per function. Tests clear caches around env mutation to stay
hermetic.
"""
import pytest

from app.config import get_settings
from app.services import ollama_clients


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear all factory caches before AND after each test so polluted
    singletons don't leak across tests in the file."""
    def _clear():
        get_settings.cache_clear()
        ollama_clients.get_doc_analysis_client.cache_clear()
        ollama_clients.get_translation_client.cache_clear()
        ollama_clients.get_community_report_client.cache_clear()
        ollama_clients.get_picture_description_client.cache_clear()
        ollama_clients.get_text_embedding_client.cache_clear()
    _clear()
    yield
    _clear()


def test_each_factory_is_cached_singleton():
    c1 = ollama_clients.get_doc_analysis_client()
    c2 = ollama_clients.get_doc_analysis_client()
    assert c1 is c2


def test_each_factory_uses_function_specific_pool(monkeypatch):
    monkeypatch.setenv("DOC_ANALYSIS_LLM_BASE_URLS", '["http://da:11434"]')
    monkeypatch.setenv("TRANSLATION_LLM_BASE_URLS", '["http://tr:11434"]')
    monkeypatch.setenv("COMMUNITY_REPORT_LLM_BASE_URLS", '["http://cr:11434"]')
    monkeypatch.setenv("PICTURE_DESCRIPTION_BASE_URLS", '["http://pd:11434"]')
    monkeypatch.setenv("TEXT_EMBEDDING_BASE_URLS", '["http://te:11434"]')

    assert ollama_clients.get_doc_analysis_client().pool.urls == ["http://da:11434"]
    assert ollama_clients.get_translation_client().pool.urls == ["http://tr:11434"]
    assert ollama_clients.get_community_report_client().pool.urls == ["http://cr:11434"]
    assert ollama_clients.get_picture_description_client().pool.urls == ["http://pd:11434"]
    assert ollama_clients.get_text_embedding_client().pool.urls == ["http://te:11434"]


def test_factories_pin_to_role_specific_models(monkeypatch):
    """Per-function factories use the role's model setting at construction."""
    monkeypatch.setenv("DOC_ANALYSIS_LLM_MODEL", "gpt-oss:120b")
    monkeypatch.setenv("TRANSLATION_MODEL", "llama3.3:70b")
    monkeypatch.setenv("COMMUNITY_REPORT_LLM_MODEL", "llama3.2")
    monkeypatch.setenv("PICTURE_DESCRIPTION_MODEL", "gemma3:27b")
    monkeypatch.setenv("TEXT_EMBEDDING_MODEL", "bge-m3:latest")
    monkeypatch.setenv("OLLAMA_LLM_BASE_URLS", '["http://h:11434"]')
    monkeypatch.setenv("OLLAMA_VLM_BASE_URLS", '["http://h:11434"]')
    monkeypatch.setenv("OLLAMA_EMBEDDING_BASE_URLS", '["http://h:11434"]')

    assert ollama_clients.get_doc_analysis_client().model == "gpt-oss:120b"
    assert ollama_clients.get_translation_client().model == "llama3.3:70b"
    assert ollama_clients.get_community_report_client().model == "llama3.2"
    assert ollama_clients.get_picture_description_client().model == "gemma3:27b"
    assert ollama_clients.get_text_embedding_client().model == "bge-m3:latest"
