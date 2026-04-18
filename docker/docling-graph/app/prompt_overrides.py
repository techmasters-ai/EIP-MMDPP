"""Prompt-text overrides for the docling-graph delta extractor.

STATUS: INSTALLED via ``main.py`` after evidence of empty-output failures
on prose docs. Library has a documented post-extraction safety net
(``delta_identity_filter_enabled`` — docling-graph-docs.md line 2757) that
drops section/chapter-title entities, so the belt-and-suspenders prompt
gate ("Emit only when this batch contains the table/structure...") is
redundant. The gate wording is library-internal, not documented contract,
so we prefer to rewrite a handful of exact phrases at import time rather
than fork the library. Upstream should be notified so a softer default
can be considered; if/when upstream adopts prose-friendly phrasing, remove
this patch.

The upstream library builds its catalog prompt block with anti-
hallucination clauses that assume extractable content is tabular:

- "Emit only when this batch contains the table/structure that defines
  these identities; otherwise omit."
- "Use only identity values from the document (e.g. from tables or
  named items); do not use section or chapter titles."
- "Valid <id>: only values from the document structure (...). Do not
  use section or chapter titles."

For prose-heavy technical docs (radar / missile / ADA manuals with
narrative descriptions of systems), llama3.3:70b interprets "emit only
when this batch contains the table/structure" literally and returns
``{"nodes": [], "relationships": []}``. The downstream quality gate
then fails with ``path_counts={}`` and ``/extract-pass`` returns 500.

Rather than forking the library, this module wraps
``build_catalog_prompt_block`` at import time and rewrites those
specific clauses to softer, prose-friendly equivalents. The wrapper
only touches exact-string matches — it does not reinterpret catalog
shape or semantics — so upstream upgrades that re-author surrounding
text will fail closed (leaving the upstream text intact) rather than
silently mis-applying.

Apply via :func:`install` from the FastAPI startup path; idempotent.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Ordered replacements — applied to the assembled prompt block. Each
# tuple is (before, after). Whitespace and leading-space preserved so
# the original structure (single trailing space separator between
# clauses) still round-trips.
_PROMPT_REWRITES: tuple[tuple[str, str], ...] = (
    (
        " Use only identity values from the document (e.g. from tables or named items); "
        "do not use section or chapter titles.",
        " Use identity values drawn from the document; avoid using section "
        "or chapter titles as identities.",
    ),
    (
        " Emit only when this batch contains the table/structure that defines "
        "these identities; otherwise omit.",
        "",
    ),
    (
        "only values from the document (e.g. ",
        "example values include ",
    ),
    (
        "only values from the document structure (e.g. ",
        "example values include ",
    ),
    (
        "Do not use section or chapter titles.",
        "Avoid using section or chapter titles as identities.",
    ),
)


def _rewrite_prompt_block(text: str) -> tuple[str, int]:
    """Return (rewritten_text, replacement_count)."""
    count = 0
    for before, after in _PROMPT_REWRITES:
        if before in text:
            count += text.count(before)
            text = text.replace(before, after)
    return text, count


def _wrap(original: Callable[..., str]) -> Callable[..., str]:
    def wrapped(catalog: Any) -> str:
        block = original(catalog)
        rewritten, count = _rewrite_prompt_block(block)
        # Warning level so we are guaranteed to see it regardless of how the
        # host process has configured root-logger handlers.
        logger.warning(
            "prompt_overrides.wrapped: catalog_block=%d chars, %d clauses rewritten",
            len(rewritten), count,
        )
        return rewritten

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    wrapped.__name__ = getattr(original, "__name__", "build_catalog_prompt_block")
    return wrapped


_installed = False


def install() -> None:
    """Install the wrapper on every known consumer of build_catalog_prompt_block.

    Must also patch the orchestrator and runtime modules because each
    imports the function by name (``from .schema_mapper import
    build_catalog_prompt_block``), which creates a local binding that
    is not updated by patching the source module alone.
    """
    global _installed
    if _installed:
        return

    from docling_graph.core.extractors.contracts.delta import (
        schema_mapper as _sm,
    )
    from docling_graph.core.extractors.contracts.delta import (
        orchestrator as _orch,
    )
    from docling_graph.core.extractors.contracts.delta import (
        runtime as _rt,
    )

    original = _sm.build_catalog_prompt_block
    wrapped = _wrap(original)

    _sm.build_catalog_prompt_block = wrapped
    _orch.build_catalog_prompt_block = wrapped
    _rt.build_catalog_prompt_block = wrapped

    _installed = True
    # WARNING level so it lands in uvicorn's default stderr handler even
    # before FastAPI finishes its own logging config.
    logger.warning(
        "prompt_overrides: INSTALLED prose-friendly rewrites on "
        "build_catalog_prompt_block (schema_mapper, orchestrator, runtime)"
    )
