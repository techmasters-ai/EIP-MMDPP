"""Service-side installer for two prompt-related patches:

1. The delta extraction system-prompt rewrite — replaces the library's
   stock ``get_delta_batch_prompt`` system string with
   ``ontology_bundles._shared.prompt_rules.DELTA_SYSTEM_PROMPT``.

2. The semantic-guide budget expansion — the library's
   ``build_compact_semantic_guide`` defaults to 250 chars per field +
   4000 chars total, which **truncates carefully-crafted FORBIDDEN-values
   blocks and unit-conversion instructions** mid-sentence and **drops
   entire flat-checklist fields off the end** of the guide once the
   total cap is hit. For RadarSystemEntity (~30 fields with rich
   descriptions) this means fields like ``nominal_rf_mhz``,
   ``scan_type``, ``antenna_dim_az_m``, ``beamwidth_az_deg`` etc. don't
   appear in the prompt at all — the LLM has no guidance to populate
   them. We override the defaults to give the full descriptions room
   to breathe (1500/30000).

Both wrappers patch every downstream module that holds a local binding
to the original function (``from X import Y`` captures at import time;
replacing the source binding doesn't rebind the imported names).

Install from the FastAPI startup path (``main.py``) — idempotent.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from ontology_bundles._shared.prompt_rules import select_delta_system_prompt

logger = logging.getLogger(__name__)

_INSTALLED = False

# Large enough to fit RadarSystemEntity's 30+ fields with their full
# FORBIDDEN-values blocks, unit-conversion instructions, and enum
# explanations. The library's defaults (250 / 4000) cause silent
# truncation that drops half the flat-checklist fields off the prompt.
_SEMANTIC_GUIDE_MAX_CHARS_PER_FIELD = 1500
_SEMANTIC_GUIDE_MAX_TOTAL_CHARS = 30000


def install() -> None:
    """Wrap library prompt builders. Idempotent.

    ``get_delta_batch_prompt`` and ``build_compact_semantic_guide`` are
    both imported by other modules via ``from X import Y``, which
    captures a local reference at import time. Replacing the source
    binding alone does not rebind those locals, so we patch every
    module that holds one.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    # ------------------------------------------------------------------
    # Patch 1: system prompt rewrite.
    # ------------------------------------------------------------------
    from docling_graph.core.extractors.contracts.delta import prompts

    original: Callable[..., dict[str, str]] = prompts.get_delta_batch_prompt

    def wrapped(*args: Any, **kwargs: Any) -> dict[str, str]:
        result = original(*args, **kwargs)
        if isinstance(result, dict) and "system" in result:
            result["system"] = select_delta_system_prompt(*args, **kwargs)
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]

    prompts.get_delta_batch_prompt = wrapped

    from docling_graph.core.extractors.contracts.delta import orchestrator
    orchestrator.get_delta_batch_prompt = wrapped
    from docling_graph.core.extractors.contracts.delta import runtime
    if hasattr(runtime, "get_delta_batch_prompt"):
        runtime.get_delta_batch_prompt = wrapped
    from docling_graph.core.extractors.contracts.delta import __init__ as delta_init  # type: ignore[attr-defined]
    if hasattr(delta_init, "get_delta_batch_prompt"):
        delta_init.get_delta_batch_prompt = wrapped

    # ------------------------------------------------------------------
    # Patch 2: semantic-guide budget expansion.
    # ------------------------------------------------------------------
    from docling_graph.llm_clients import schema_utils

    original_guide: Callable[..., str] = schema_utils.build_compact_semantic_guide

    def wrapped_guide(schema: dict, *, max_chars_per_field: int = _SEMANTIC_GUIDE_MAX_CHARS_PER_FIELD,
                      max_total_chars: int = _SEMANTIC_GUIDE_MAX_TOTAL_CHARS,
                      max_depth: int = 3) -> str:
        return original_guide(
            schema,
            max_chars_per_field=max_chars_per_field,
            max_total_chars=max_total_chars,
            max_depth=max_depth,
        )

    wrapped_guide.__wrapped__ = original_guide  # type: ignore[attr-defined]

    schema_utils.build_compact_semantic_guide = wrapped_guide

    # Rebind in every module that did
    # `from .schema_utils import build_compact_semantic_guide` (or the
    # transitive `from ..... import build_compact_semantic_guide` in
    # schema_mapper).
    from docling_graph.core.extractors.contracts.delta import schema_mapper
    if hasattr(schema_mapper, "build_compact_semantic_guide"):
        schema_mapper.build_compact_semantic_guide = wrapped_guide

    _INSTALLED = True
    logger.warning(
        "prompt_rules: installed delta system-prompt rewrite + semantic-guide "
        "budget expansion (max_chars_per_field=%d, max_total_chars=%d).",
        _SEMANTIC_GUIDE_MAX_CHARS_PER_FIELD, _SEMANTIC_GUIDE_MAX_TOTAL_CHARS,
    )
