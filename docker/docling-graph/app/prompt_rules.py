"""Service-side installer for two prompt-related patches:

1. The delta extraction system-prompt rewrite — replaces the library's
   stock ``get_delta_batch_prompt`` system string with
   ``ontology_bundles._shared.prompt_rules.DELTA_SYSTEM_PROMPT``.

2. The semantic-guide rewrite — the library's default guide truncates
   fields too aggressively and also renders schema examples inline
   (``e.g. 38.0``), which is harmful for numeric extraction. We expand
   the budget but sanitize the LLM-facing schema first: numeric examples
   are removed, typical-value ranges are stripped from numeric
   descriptions, and long forbidden-identity lists are compressed.

Both wrappers patch every downstream module that holds a local binding
to the original function (``from X import Y`` captures at import time;
replacing the source binding doesn't rebind the imported names).

Install from the FastAPI startup path (``main.py``) — idempotent.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Any, Callable

from ontology_bundles._shared.prompt_rules import select_delta_system_prompt

from app._numeric_candidates import (
    extract_numeric_candidates,
    render_candidate_block,
)

logger = logging.getLogger(__name__)

_INSTALLED = False

# Large enough to fit RadarSystemEntity's 30+ fields after LLM-facing
# sanitization. The library's defaults (250 / 4000) cause silent truncation
# that drops half the flat-checklist fields off the prompt.
_SEMANTIC_GUIDE_MAX_CHARS_PER_FIELD = 1500
_SEMANTIC_GUIDE_MAX_TOTAL_CHARS = 30000

_TYPICAL_NUMERIC_PHRASES = (
    "typical",
    "common radar bands",
    "common barker",
    "phased arrays",
    "classical high-altitude",
    "exo-atmospheric",
)


def _schema_allows_numeric(schema: Any) -> bool:
    """Return True when a JSON-schema fragment permits number/integer."""
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    if schema_type in {"number", "integer"}:
        return True
    if isinstance(schema_type, list) and any(t in {"number", "integer"} for t in schema_type):
        return True
    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list) and any(_schema_allows_numeric(item) for item in variants):
            return True
    return False


def _split_description_sentences(description: str) -> list[str]:
    """Split on sentence boundaries while preserving decimal numbers."""
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", description.strip())
        if part.strip()
    ]


def _sanitize_numeric_description(description: str) -> str:
    sentences = _split_description_sentences(description)
    kept: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(phrase in lowered for phrase in _TYPICAL_NUMERIC_PHRASES):
            continue
        if lowered.startswith("higher gain ="):
            continue
        kept.append(sentence)

    result = " ".join(kept).strip() or description.strip()
    reminder = (
        "Emit only a value directly stated by the source for this exact "
        "field, or a mechanically converted value from an explicit source "
        "value; otherwise emit null."
    )
    if reminder not in result:
        result = f"{result} {reminder}"
    return result


def _compress_forbidden_identity_description(description: str) -> str:
    if "FORBIDDEN values" not in description or len(description) <= 700:
        return description

    first_sentence = _split_description_sentences(description)[0]
    lowered = description.lower()
    if "weapon/missile systems, not radars" in lowered:
        forbidden_summary = (
            "Never emit weapon or missile systems, aircraft, platforms, or "
            "target names as radar system_name. Examples of forbidden "
            "associated names include SA-2, SA-3, Patriot, S-300/S-400, "
            "U-2, SR-71, F-series aircraft, MiG aircraft, and Su aircraft. "
            "If the text gives only an associated weapon/platform/target, "
            "omit the radar entity; if it also states the radar's own name, "
            "emit that radar name."
        )
    elif "radars, not missile/weapon systems" in lowered:
        forbidden_summary = (
            "Never emit radar names, aircraft, platforms, or target names "
            "as missile system_name. Examples of forbidden associated names "
            "include Fan Song, Spoon Rest, Flap Lid, Tombstone, AN/MPQ-65, "
            "AN/SPY-1, U-2, SR-71, F-series aircraft, MiG aircraft, and Su "
            "aircraft. If the text gives only an associated radar/platform/"
            "target, omit the missile entity; if it also states the missile's "
            "own name, emit that missile name."
        )
    else:
        forbidden_summary = (
            "Never emit associated systems, platforms, or targets as this "
            "entity identity. Emit only the entity's own canonical name when "
            "the current batch states it directly."
        )
    return f"{first_sentence} {forbidden_summary} Reject descriptive phrases."


def sanitize_schema_for_llm(schema: dict[str, Any]) -> dict[str, Any]:
    """Return an LLM-facing schema copy with low-signal prompt noise removed.

    This does not change the Pydantic model, validators, persisted output
    schema, or downstream postprocessing. It only changes what the model sees
    in the compact semantic guide and structured-output request.
    """
    sanitized = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if _schema_allows_numeric(node):
                node.pop("examples", None)
                description = node.get("description")
                if isinstance(description, str):
                    node["description"] = _sanitize_numeric_description(description)
            else:
                description = node.get("description")
                if isinstance(description, str):
                    node["description"] = _compress_forbidden_identity_description(description)

            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(sanitized)
    return sanitized


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
        # Phase A item 5: inject numeric-candidate hint block after the
        # batch-document section. Pre-extracted regex spans tell the LLM
        # which numeric values appear verbatim, turning numeric extraction
        # from "copy a number" into "classify a span" — much easier for
        # general-purpose models.
        if isinstance(result, dict) and "user" in result:
            batch_md = kwargs.get("batch_markdown", "")
            if isinstance(batch_md, str) and batch_md:
                candidates = extract_numeric_candidates(batch_md)
                block = render_candidate_block(candidates)
                if block:
                    user_text = result["user"]
                    marker = "=== END BATCH DOCUMENT ==="
                    if marker in user_text:
                        before, after = user_text.split(marker, 1)
                        result["user"] = before + marker + "\n\n" + block + after.lstrip("\n")
                    else:
                        result["user"] = user_text + "\n\n" + block
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
            sanitize_schema_for_llm(schema),
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
