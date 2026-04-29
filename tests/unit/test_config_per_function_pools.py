"""Test the per-function plural URL env vars and their 4-tier cascade.

Cascade priority (function-specific > role-level > singular > base):
  DOCLING_GRAPH_LLM_BASE_URLS    > OLLAMA_LLM_BASE_URLS       > OLLAMA_LLM_BASE_URL       > OLLAMA_BASE_URL
  DOC_ANALYSIS_LLM_BASE_URLS     > OLLAMA_LLM_BASE_URLS       > OLLAMA_LLM_BASE_URL       > OLLAMA_BASE_URL
  TRANSLATION_LLM_BASE_URLS      > OLLAMA_LLM_BASE_URLS       > OLLAMA_LLM_BASE_URL       > OLLAMA_BASE_URL
  COMMUNITY_REPORT_LLM_BASE_URLS > OLLAMA_LLM_BASE_URLS       > OLLAMA_LLM_BASE_URL       > OLLAMA_BASE_URL
  PICTURE_DESCRIPTION_BASE_URLS  > OLLAMA_VLM_BASE_URLS       > OLLAMA_VLM_BASE_URL       > OLLAMA_BASE_URL
  TEXT_EMBEDDING_BASE_URLS       > OLLAMA_EMBEDDING_BASE_URLS > OLLAMA_EMBEDDING_BASE_URL > OLLAMA_BASE_URL
"""
import pytest

from app.config import Settings


def _build_settings(monkeypatch, **env: str) -> Settings:
    """Construct a fresh Settings without polluting the LRU-cached singleton."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return Settings(_env_file=None)


# ----- Function-specific overrides win over role-level -----

def test_doc_analysis_function_pool_overrides_role(monkeypatch):
    s = _build_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://base:11434",
        OLLAMA_LLM_BASE_URLS='["http://role-1:11434"]',
        DOC_ANALYSIS_LLM_BASE_URLS='["http://gpt-oss-host-1:11434","http://gpt-oss-host-2:11434"]',
    )
    assert s.get_doc_analysis_llm_urls() == [
        "http://gpt-oss-host-1:11434",
        "http://gpt-oss-host-2:11434",
    ]


def test_translation_function_pool_overrides_role(monkeypatch):
    s = _build_settings(
        monkeypatch,
        OLLAMA_LLM_BASE_URLS='["http://role-host:11434"]',
        TRANSLATION_LLM_BASE_URLS='["http://llama-host:11434"]',
    )
    assert s.get_translation_llm_urls() == ["http://llama-host:11434"]


def test_community_report_function_pool_overrides_role(monkeypatch):
    s = _build_settings(
        monkeypatch,
        OLLAMA_LLM_BASE_URLS='["http://role-host:11434"]',
        COMMUNITY_REPORT_LLM_BASE_URLS='["http://gpt-oss-host:11434"]',
    )
    assert s.get_community_report_llm_urls() == ["http://gpt-oss-host:11434"]


def test_picture_description_function_pool_overrides_role(monkeypatch):
    s = _build_settings(
        monkeypatch,
        OLLAMA_VLM_BASE_URLS='["http://role-host:11434"]',
        PICTURE_DESCRIPTION_BASE_URLS='["http://gemma-vlm:11434"]',
    )
    assert s.get_picture_description_urls() == ["http://gemma-vlm:11434"]


def test_text_embedding_function_pool_overrides_role(monkeypatch):
    s = _build_settings(
        monkeypatch,
        OLLAMA_EMBEDDING_BASE_URLS='["http://role-host:11434"]',
        TEXT_EMBEDDING_BASE_URLS='["http://bge-host:11434"]',
    )
    assert s.get_text_embedding_urls() == ["http://bge-host:11434"]


# ----- Role-level pools serve as fallback when function-specific is empty -----

def test_function_falls_back_to_role_level(monkeypatch):
    s = _build_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://base:11434",
        OLLAMA_LLM_BASE_URLS='["http://role-1:11434","http://role-2:11434"]',
        DOC_ANALYSIS_LLM_BASE_URLS="",  # blank → fall through to role
    )
    assert s.get_doc_analysis_llm_urls() == [
        "http://role-1:11434",
        "http://role-2:11434",
    ]


# ----- Singular fallback when both plurals are empty -----

def test_function_falls_back_through_singular(monkeypatch):
    s = _build_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://base:11434",
        OLLAMA_LLM_BASE_URL="http://singular:11434",
        OLLAMA_LLM_BASE_URLS="",
        TRANSLATION_LLM_BASE_URLS="",
    )
    assert s.get_translation_llm_urls() == ["http://singular:11434"]


# ----- Base URL when nothing else is set -----

def test_function_falls_back_to_base(monkeypatch):
    s = _build_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://base:11434",
        OLLAMA_LLM_BASE_URL="",
        OLLAMA_LLM_BASE_URLS="",
        COMMUNITY_REPORT_LLM_BASE_URLS="",
    )
    assert s.get_community_report_llm_urls() == ["http://base:11434"]


# ----- Role isolation: chat-role function vs vlm-role function vs embedding-role function -----

def test_picture_description_uses_vlm_cascade_not_llm(monkeypatch):
    """PICTURE_DESCRIPTION must cascade through OLLAMA_VLM_*, NOT OLLAMA_LLM_*."""
    s = _build_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://base:11434",
        OLLAMA_LLM_BASE_URLS='["http://llm-pool:11434"]',
        OLLAMA_VLM_BASE_URLS='["http://vlm-pool:11434"]',
        PICTURE_DESCRIPTION_BASE_URLS="",
    )
    assert s.get_picture_description_urls() == ["http://vlm-pool:11434"]


def test_text_embedding_uses_embedding_cascade_not_llm(monkeypatch):
    s = _build_settings(
        monkeypatch,
        OLLAMA_LLM_BASE_URLS='["http://llm-pool:11434"]',
        OLLAMA_EMBEDDING_BASE_URLS='["http://embed-pool:11434"]',
        TEXT_EMBEDDING_BASE_URLS="",
    )
    assert s.get_text_embedding_urls() == ["http://embed-pool:11434"]


# ----- Malformed / blank-entry rejection (parity with role-level) -----

def test_malformed_function_pool_raises_at_read(monkeypatch):
    s = _build_settings(
        monkeypatch,
        DOC_ANALYSIS_LLM_BASE_URLS='["http://h1',  # unclosed
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        s.get_doc_analysis_llm_urls()


def test_blank_entry_in_function_pool_raises(monkeypatch):
    s = _build_settings(
        monkeypatch,
        TEXT_EMBEDDING_BASE_URLS='["http://h1:11434",""]',
    )
    with pytest.raises(ValueError, match="contains blank entries"):
        s.get_text_embedding_urls()
