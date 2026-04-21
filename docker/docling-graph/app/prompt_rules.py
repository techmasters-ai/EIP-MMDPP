"""Service-side installer for the delta extraction system-prompt rewrite.

Wraps ``docling_graph.core.extractors.contracts.delta.prompts.get_delta_batch_prompt``
so the ``system`` string returned on every call is the rewritten version
from ``ontology_bundles._shared.prompt_rules.DELTA_SYSTEM_PROMPT``.

See ``ontology_bundles/_shared/prompt_rules.py`` for the full rationale.
Short version: the library's Rules 2-4 forbid mention-level extraction
("identity comes from tables / section titles / captions") while *allowing*
section-title identities, which is the inverse of what our corpus needs.
We want structure-backed extraction when structure exists AND prose-
mention extraction when it's the only evidence; the library's own
post-extraction ``delta_identity_filter`` handles section-title slippage.

Install from the FastAPI startup path (``main.py``) — idempotent.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from ontology_bundles._shared.prompt_rules import DELTA_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_INSTALLED = False


def install() -> None:
    """Wrap ``get_delta_batch_prompt`` to return our system prompt. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return

    from docling_graph.core.extractors.contracts.delta import prompts

    original: Callable[..., dict[str, str]] = prompts.get_delta_batch_prompt

    def wrapped(*args: Any, **kwargs: Any) -> dict[str, str]:
        result = original(*args, **kwargs)
        # Replace ONLY the system string. Library user prompt structure is
        # unchanged.
        if isinstance(result, dict) and "system" in result:
            result["system"] = DELTA_SYSTEM_PROMPT
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    prompts.get_delta_batch_prompt = wrapped

    _INSTALLED = True
    logger.info(
        "prompt_rules: installed delta system-prompt rewrite "
        "(%d chars) — Rules 2/3/4 allow mention-level extraction.",
        len(DELTA_SYSTEM_PROMPT),
    )
