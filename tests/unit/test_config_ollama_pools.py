"""Test the new plural OLLAMA_*_BASE_URLS env vars and getters."""
import os
import pytest


def _reload_settings(monkeypatch, **env: str):
    """Build a fresh Settings instance reading the patched env.

    Avoids importlib.reload() — that replaces the module object globally and
    leaves stale references in any other module that already imported names
    from app.config, breaking unrelated tests in the same session.
    """
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from app.config import Settings, get_settings
    get_settings.cache_clear()
    # Explicitly bypass the .env file (we want the test's env to win) by
    # passing _env_file=None. Settings will read os.environ otherwise.
    return Settings(_env_file=None)


def test_plural_takes_precedence_over_singular(monkeypatch):
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URL="http://singular:11434",
        OLLAMA_LLM_BASE_URLS='["http://h1:11434","http://h2:11434"]',
    )
    assert s.get_ollama_llm_urls() == [
        "http://h1:11434", "http://h2:11434",
    ]


def test_singular_used_when_plural_empty(monkeypatch):
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URL="http://singular:11434",
        OLLAMA_LLM_BASE_URLS="",
    )
    assert s.get_ollama_llm_urls() == ["http://singular:11434"]


def test_base_url_used_when_both_empty(monkeypatch):
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URL="",
        OLLAMA_LLM_BASE_URLS="",
    )
    assert s.get_ollama_llm_urls() == ["http://fallback:11434"]


def test_singular_getter_returns_first_url(monkeypatch):
    """Back-compat: existing call sites that only know about singular keep
    working — get_ollama_llm_url() returns urls[0]."""
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URLS='["http://h1:11434","http://h2:11434"]',
    )
    assert s.get_ollama_llm_url() == "http://h1:11434"


def test_vlm_and_embedding_pools_independent(monkeypatch):
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URLS='["http://llm-1"]',
        OLLAMA_VLM_BASE_URLS='["http://vlm-1","http://vlm-2"]',
        OLLAMA_EMBEDDING_BASE_URLS='["http://emb-1"]',
    )
    assert s.get_ollama_llm_urls() == ["http://llm-1"]
    assert s.get_ollama_vlm_urls() == ["http://vlm-1", "http://vlm-2"]
    assert s.get_ollama_embedding_urls() == ["http://emb-1"]


def test_blank_plural_env_var_does_not_crash_startup(monkeypatch):
    """Production .env files commonly leave OLLAMA_LLM_BASE_URLS= blank when
    the operator uses the singular var instead. Storing the plural as a raw
    str (not list[str]) sidesteps pydantic-settings' SettingsError trap."""
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URL="http://singular:11434",
        OLLAMA_LLM_BASE_URLS="",  # the trap case — must NOT raise
        OLLAMA_VLM_BASE_URLS="",
        OLLAMA_EMBEDDING_BASE_URLS="",
    )
    # Stored as raw string, not list.
    assert s.ollama_llm_base_urls == ""
    # Helper falls through to the singular var.
    assert s.get_ollama_llm_urls() == ["http://singular:11434"]


def test_malformed_plural_url_var_raises_at_read_time(monkeypatch):
    """Misconfigured JSON in plural env var must fail loudly when first
    consumed (rather than silently falling through to singular)."""
    import pytest
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URLS='["http://h1', # malformed (unclosed)
    )
    # Construction succeeds; failure is at first call to the helper.
    with pytest.raises(ValueError, match="not valid JSON"):
        s.get_ollama_llm_urls()


def test_non_array_plural_url_var_raises_at_read_time(monkeypatch):
    import pytest
    s = _reload_settings(
        monkeypatch,
        OLLAMA_LLM_BASE_URLS='"http://just-a-string"',
    )
    with pytest.raises(ValueError, match="JSON array of strings"):
        s.get_ollama_llm_urls()
