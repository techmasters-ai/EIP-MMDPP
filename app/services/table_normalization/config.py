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


def _env_int_per_pass(global_var: str, pass_name: str | None, default: int) -> int:
    """Return the per-pass override if present, else the global value.

    The override env var is constructed as DOCLING_GRAPH_<PASS_UPPER>_<TAIL>
    where TAIL is the portion of global_var after the DOCLING_GRAPH_ prefix.
    Falls back silently when the override is unset — callers don't need to
    branch on whether per-pass tuning is configured."""
    if pass_name:
        if not global_var.startswith("DOCLING_GRAPH_"):
            raise ValueError(
                f"_env_int_per_pass expected DOCLING_GRAPH_ prefix, got {global_var!r}"
            )
        tail = global_var[len("DOCLING_GRAPH_"):]
        override_key = f"DOCLING_GRAPH_{pass_name.upper()}_{tail}"
        if override_key in os.environ:
            return _env_int(override_key, _env_int(global_var, default))
    return _env_int(global_var, default)


def is_table_normalization_enabled_graph() -> bool:
    return _env_bool("DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED", False)


def is_table_normalization_enabled_embedding() -> bool:
    return _env_bool("EMBEDDING_TABLE_NORMALIZATION_ENABLED", False)


def is_experimental_table_facts_enabled() -> bool:
    return _env_bool("DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS", False)


def is_suppress_raw_table_markdown_enabled() -> bool:
    return _env_bool("DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN", True)


def table_whole_limit(pass_name: str | None = None) -> int:
    return _env_int_per_pass("DOCLING_GRAPH_TABLE_WHOLE_LIMIT", pass_name, 1500)


def table_column_limit(pass_name: str | None = None) -> int:
    return _env_int_per_pass("DOCLING_GRAPH_TABLE_COLUMN_LIMIT", pass_name, 1200)


def min_table_normalization_tokens() -> int:
    return _env_int("MIN_TABLE_NORMALIZATION_TOKENS", 256)


def embedding_chunk_max_tokens() -> int:
    return _env_int("EMBEDDING_CHUNK_MAX_TOKENS", 512)


def embedding_table_summary_max_tokens() -> int:
    return _env_int("EMBEDDING_TABLE_SUMMARY_MAX_TOKENS", 300)


_PER_PASS_OVERRIDE_PASSES: tuple[str, ...] = ("system_links",)


def _validate_one_invariant_set(
    pass_name: str | None,
    whole: int,
    column: int,
    chunk_max: int,
    batch: int,
) -> list[str]:
    label = f"[{pass_name}]" if pass_name else "[global]"
    problems: list[str] = []
    if whole > chunk_max:
        problems.append(
            f"{label} TABLE_WHOLE_LIMIT={whole} > CHUNK_MAX_TOKENS={chunk_max}: "
            f"whole-table chunks will be re-split by HybridChunker."
        )
    if column > chunk_max:
        problems.append(
            f"{label} TABLE_COLUMN_LIMIT={column} > CHUNK_MAX_TOKENS={chunk_max}: "
            f"column chunks will be re-split by HybridChunker."
        )
    if chunk_max > batch:
        problems.append(
            f"{label} CHUNK_MAX_TOKENS={chunk_max} > LLM_BATCH_TOKEN_SIZE={batch}: "
            f"a single max-size chunk won't fit in one LLM batch."
        )
    return problems


def validate_token_invariants() -> None:
    """Enforce: synthesized table chunks fit inside chunker + batcher budgets.

    Invariant ladder (per the §spec rollout discussion 2026-05-13):

        WHOLE_LIMIT     <=  CHUNK_MAX_TOKENS      <=  BATCH_TOKEN_SIZE
        COLUMN_LIMIT    <=  CHUNK_MAX_TOKENS

    Validates the global set and each per-pass override set declared in
    _PER_PASS_OVERRIDE_PASSES. Violations of the first two cause
    HybridChunker to re-split synthesized table chunks at arbitrary
    boundaries (defeating normalization). Violation of the third causes
    the LLM batcher to drop or partial-fit chunks.

    Raises ValueError on violation so misconfiguration surfaces at
    startup, not silently mid-extraction."""
    problems: list[str] = []

    # Global
    g_whole = table_whole_limit()
    g_column = table_column_limit()
    g_chunk_max = _env_int("DOCLING_GRAPH_CHUNK_MAX_TOKENS", 512)
    g_batch = _env_int("DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE", 1024)
    problems.extend(_validate_one_invariant_set(
        None, g_whole, g_column, g_chunk_max, g_batch,
    ))

    # Each per-pass override set (only validated if at least one of the
    # four pass-specific env vars is actually set — otherwise we'd be
    # re-validating the global set).
    for pn in _PER_PASS_OVERRIDE_PASSES:
        any_override = any(
            f"DOCLING_GRAPH_{pn.upper()}_{tail}" in os.environ
            for tail in (
                "TABLE_WHOLE_LIMIT",
                "TABLE_COLUMN_LIMIT",
                "CHUNK_MAX_TOKENS",
                "LLM_BATCH_TOKEN_SIZE",
            )
        )
        if not any_override:
            continue
        whole = table_whole_limit(pn)
        column = table_column_limit(pn)
        chunk_max = _env_int_per_pass("DOCLING_GRAPH_CHUNK_MAX_TOKENS", pn, 512)
        batch = _env_int_per_pass("DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE", pn, 1024)
        problems.extend(_validate_one_invariant_set(
            pn, whole, column, chunk_max, batch,
        ))

    if problems:
        raise ValueError(
            "Table normalization token-budget invariants violated:\n  - "
            + "\n  - ".join(problems)
        )
    logger.info(
        "table_normalization invariants OK: global whole=%d column=%d chunk_max=%d batch=%d",
        g_whole, g_column, g_chunk_max, g_batch,
    )
