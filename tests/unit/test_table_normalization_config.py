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
