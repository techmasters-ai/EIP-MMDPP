"""Env-var reading for the table normalization layer.

Single source of truth for flag names and defaults. All readers check
the env var on every call — no module-level caching of values — so
runtime flag flips (per the §13 rollout) take effect immediately."""
from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("table_normalization.config: %s=%r is not an int; using default %d", name, raw, default)
        return default


def is_table_normalization_enabled_graph() -> bool:
    return _env_bool("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", False)


def is_table_normalization_enabled_embedding() -> bool:
    return _env_bool("EMBEDDING_TABLE_NORMALIZATION_ENABLED", False)


def is_experimental_table_facts_enabled() -> bool:
    return _env_bool("DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS", False)


def is_suppress_raw_table_markdown_enabled() -> bool:
    return _env_bool("DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN", True)


def table_whole_limit() -> int:
    return _env_int("DOCLING_GRAPH_TABLE_WHOLE_LIMIT", 1500)


def table_column_limit() -> int:
    return _env_int("DOCLING_GRAPH_TABLE_COLUMN_LIMIT", 1200)


def min_table_normalization_tokens() -> int:
    return _env_int("MIN_TABLE_NORMALIZATION_TOKENS", 256)


def embedding_chunk_max_tokens() -> int:
    return _env_int("EMBEDDING_CHUNK_MAX_TOKENS", 512)


def embedding_table_summary_max_tokens() -> int:
    return _env_int("EMBEDDING_TABLE_SUMMARY_MAX_TOKENS", 300)
