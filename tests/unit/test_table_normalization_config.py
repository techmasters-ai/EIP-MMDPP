import os
import pytest
from app.services.table_normalization.config import (
    is_table_normalization_enabled_graph,
    is_table_normalization_enabled_embedding,
    is_experimental_table_facts_enabled,
    is_suppress_raw_table_markdown_enabled,
    table_whole_limit,
    table_column_limit,
    min_table_normalization_tokens,
    embedding_chunk_max_tokens,
    embedding_table_summary_max_tokens,
    validate_token_invariants,
)


def test_defaults_are_off_for_master_switches(monkeypatch):
    for v in (
        "DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED",
        "DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS",
        "EMBEDDING_TABLE_NORMALIZATION_ENABLED",
    ):
        monkeypatch.delenv(v, raising=False)
    assert is_table_normalization_enabled_graph() is False
    assert is_experimental_table_facts_enabled() is False
    assert is_table_normalization_enabled_embedding() is False


def test_suppress_default_true(monkeypatch):
    monkeypatch.delenv("DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN", raising=False)
    assert is_suppress_raw_table_markdown_enabled() is True


def test_threshold_defaults(monkeypatch):
    for v in (
        "DOCLING_GRAPH_TABLE_WHOLE_LIMIT",
        "DOCLING_GRAPH_TABLE_COLUMN_LIMIT",
        "MIN_TABLE_NORMALIZATION_TOKENS",
        "EMBEDDING_TABLE_SUMMARY_MAX_TOKENS",
    ):
        monkeypatch.delenv(v, raising=False)
    assert table_whole_limit() == 1500
    assert table_column_limit() == 1200
    assert min_table_normalization_tokens() == 256
    assert embedding_table_summary_max_tokens() == 300


def test_flags_respond_to_env(monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", "true")
    assert is_table_normalization_enabled_graph() is True
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", "false")
    assert is_table_normalization_enabled_graph() is False


def test_embedding_chunk_max_tokens_returns_int():
    n = embedding_chunk_max_tokens()
    assert isinstance(n, int)
    assert n > 0


def _set_token_env(monkeypatch, *, whole, column, chunk_max, batch):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_WHOLE_LIMIT", str(whole))
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_COLUMN_LIMIT", str(column))
    monkeypatch.setenv("DOCLING_GRAPH_CHUNK_MAX_TOKENS", str(chunk_max))
    monkeypatch.setenv("DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE", str(batch))


def test_validate_invariants_ok(monkeypatch):
    _set_token_env(monkeypatch, whole=1024, column=1024, chunk_max=1024, batch=1024)
    validate_token_invariants()  # does not raise


def test_validate_rejects_whole_limit_over_chunk_max(monkeypatch):
    _set_token_env(monkeypatch, whole=1500, column=1024, chunk_max=1024, batch=1024)
    with pytest.raises(ValueError, match=r"\[global\] TABLE_WHOLE_LIMIT=1500"):
        validate_token_invariants()


def test_validate_rejects_column_limit_over_chunk_max(monkeypatch):
    _set_token_env(monkeypatch, whole=1024, column=1500, chunk_max=1024, batch=1024)
    with pytest.raises(ValueError, match=r"\[global\] TABLE_COLUMN_LIMIT=1500"):
        validate_token_invariants()


def test_validate_rejects_chunk_max_over_batch(monkeypatch):
    _set_token_env(monkeypatch, whole=1024, column=1024, chunk_max=2048, batch=1024)
    with pytest.raises(ValueError, match=r"\[global\] CHUNK_MAX_TOKENS=2048"):
        validate_token_invariants()


def test_validate_reports_all_violations(monkeypatch):
    _set_token_env(monkeypatch, whole=1500, column=1500, chunk_max=1024, batch=512)
    with pytest.raises(ValueError) as excinfo:
        validate_token_invariants()
    msg = str(excinfo.value)
    assert "TABLE_WHOLE_LIMIT=1500" in msg
    assert "TABLE_COLUMN_LIMIT=1500" in msg
    assert "CHUNK_MAX_TOKENS=1024" in msg


# Per-pass override tests --------------------------------------------------

def test_per_pass_whole_limit_returns_override_when_set(monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_WHOLE_LIMIT", "512")
    monkeypatch.setenv("DOCLING_GRAPH_SYSTEM_LINKS_TABLE_WHOLE_LIMIT", "4096")
    assert table_whole_limit() == 512
    assert table_whole_limit("system_links") == 4096


def test_per_pass_column_limit_returns_override_when_set(monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_COLUMN_LIMIT", "512")
    monkeypatch.setenv("DOCLING_GRAPH_SYSTEM_LINKS_TABLE_COLUMN_LIMIT", "4096")
    assert table_column_limit() == 512
    assert table_column_limit("system_links") == 4096


def test_per_pass_falls_back_to_global_when_no_override(monkeypatch):
    """When DOCLING_GRAPH_SYSTEM_LINKS_<X> is unset, the global value applies."""
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_WHOLE_LIMIT", "512")
    monkeypatch.delenv("DOCLING_GRAPH_SYSTEM_LINKS_TABLE_WHOLE_LIMIT", raising=False)
    assert table_whole_limit("system_links") == 512


def test_per_pass_unknown_pass_falls_back_to_global(monkeypatch):
    """Passes with no overrides set fall back transparently."""
    monkeypatch.setenv("DOCLING_GRAPH_TABLE_COLUMN_LIMIT", "512")
    assert table_column_limit("some_random_pass") == 512


def test_validate_passes_when_global_and_per_pass_both_ok(monkeypatch):
    _set_token_env(monkeypatch, whole=512, column=512, chunk_max=512, batch=512)
    monkeypatch.setenv("DOCLING_GRAPH_SYSTEM_LINKS_TABLE_WHOLE_LIMIT", "4096")
    monkeypatch.setenv("DOCLING_GRAPH_SYSTEM_LINKS_TABLE_COLUMN_LIMIT", "4096")
    monkeypatch.setenv("DOCLING_GRAPH_SYSTEM_LINKS_CHUNK_MAX_TOKENS", "4096")
    monkeypatch.setenv("DOCLING_GRAPH_SYSTEM_LINKS_LLM_BATCH_TOKEN_SIZE", "4096")
    validate_token_invariants()  # does not raise


def test_validate_rejects_per_pass_invariant_violation(monkeypatch):
    """Per-pass overrides must satisfy the same invariant ladder."""
    _set_token_env(monkeypatch, whole=512, column=512, chunk_max=512, batch=512)
    # Misconfigured: WHOLE > CHUNK_MAX for system_links.
    monkeypatch.setenv("DOCLING_GRAPH_SYSTEM_LINKS_TABLE_WHOLE_LIMIT", "8000")
    monkeypatch.setenv("DOCLING_GRAPH_SYSTEM_LINKS_CHUNK_MAX_TOKENS", "4096")
    with pytest.raises(ValueError, match=r"\[system_links\] TABLE_WHOLE_LIMIT=8000"):
        validate_token_invariants()


def test_validate_skips_per_pass_when_no_overrides_present(monkeypatch):
    """If no SYSTEM_LINKS_* env vars are set, we don't re-validate that set."""
    _set_token_env(monkeypatch, whole=512, column=512, chunk_max=512, batch=512)
    for tail in (
        "TABLE_WHOLE_LIMIT", "TABLE_COLUMN_LIMIT",
        "CHUNK_MAX_TOKENS", "LLM_BATCH_TOKEN_SIZE",
    ):
        monkeypatch.delenv(f"DOCLING_GRAPH_SYSTEM_LINKS_{tail}", raising=False)
    validate_token_invariants()  # does not raise; only global set is validated


