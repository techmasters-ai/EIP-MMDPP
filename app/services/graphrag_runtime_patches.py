"""Runtime monkey-patches for Microsoft GraphRAG.

Patches are applied at import time via install_graphrag_runtime_patches().
All patches are idempotent and guarded by version/signature checks.

Current patches:
  - DRIFTPrimer.decompose_query: replace strict json.loads parsing with
    tolerant multi-strategy parser (parse_llm_json_loose) so DRIFT search
    survives malformed/truncated JSON from Ollama thinking models.
"""

import inspect
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _extract_content(model_response: Any) -> str:
    """Extract usable text from a graphrag_llm completion response.

    Thinking models (Ollama with think=medium/high) may leave ``content``
    empty and place the answer in a reasoning field instead.  Mirrors the
    fallback chain in docling-graph's ``_extract_llm_content``.
    """
    raw = model_response.content or ""
    if raw.strip():
        return raw

    msg = model_response.choices[0].message
    for attr in ("reasoning_content", "reasoning", "thinking"):
        val = getattr(msg, attr, None) or ""
        if isinstance(val, str) and val.strip():
            logger.info(
                "DRIFT primer: content empty — falling back to %s (%d chars)",
                attr, len(val),
            )
            return val

    blocks = getattr(msg, "thinking_blocks", None)
    if blocks:
        logger.info("DRIFT primer: content empty — falling back to thinking_blocks")
        return str(blocks)

    return ""


_PATCHES_INSTALLED = False


class DriftPrimerParseError(Exception):
    """DRIFT primer returned JSON that could not be parsed or validated."""


def install_graphrag_runtime_patches() -> None:
    """Install all GraphRAG runtime patches. Idempotent."""
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return
    _PATCHES_INSTALLED = True

    _patch_drift_primer()


def _assert_patch_target_compatible() -> bool:
    """Check that DRIFTPrimer.decompose_query has the expected signature.

    Returns True if compatible, False otherwise.
    """
    try:
        from graphrag.query.structured_search.drift_search.primer import (
            DRIFTPrimer,
        )

        sig = inspect.signature(DRIFTPrimer.decompose_query)
        params = list(sig.parameters.keys())
        # Expected: (self, query, reports)
        if params != ["self", "query", "reports"]:
            logger.warning(
                "DRIFTPrimer.decompose_query signature changed: %s — skipping patch",
                params,
            )
            return False
        return True
    except Exception:
        logger.warning(
            "Could not inspect DRIFTPrimer.decompose_query — skipping patch",
            exc_info=True,
        )
        return False


def _patch_drift_primer() -> None:
    """Monkey-patch DRIFTPrimer.decompose_query with tolerant JSON parsing."""
    if not _assert_patch_target_compatible():
        return

    try:
        from graphrag.query.structured_search.drift_search.primer import (
            DRIFTPrimer,
            PrimerResponse,
        )
        from graphrag.prompts.query.drift_search_system_prompt import (
            DRIFT_PRIMER_PROMPT,
        )
    except ImportError:
        logger.warning(
            "GraphRAG DRIFT primer symbols not found — skipping patch",
            exc_info=True,
        )
        return

    async def _patched_decompose_query(
        self, query: str, reports: pd.DataFrame
    ) -> tuple[dict, dict[str, int]]:
        """Tolerant replacement for DRIFTPrimer.decompose_query.

        Changes from the original:
          - Calls completion_async without response_format (avoids the strict
            json.loads path in graphrag_llm's structure_response.py)
          - Parses the raw content with parse_llm_json_loose
          - Validates with PrimerResponse(**parsed)
          - Passes primer_llm_max_tokens as max_completion_tokens
        """
        from app.services.llm_json import parse_llm_json_loose

        community_reports = "\n\n".join(reports["full_content"].tolist())
        prompt = DRIFT_PRIMER_PROMPT.format(
            query=query, community_reports=community_reports
        )

        # Call without response_format to skip graphrag_llm's strict json.loads
        model_response = await self.chat_model.completion_async(
            messages=prompt,
            max_completion_tokens=self.config.primer_llm_max_tokens,
        )

        raw = _extract_content(model_response)

        parsed = parse_llm_json_loose(raw)
        if not isinstance(parsed, dict):
            raise DriftPrimerParseError(
                f"DRIFT primer returned non-object JSON (got {type(parsed).__name__})"
            )

        try:
            primer = PrimerResponse(**parsed)
        except Exception as exc:
            raise DriftPrimerParseError(
                f"DRIFT primer JSON failed validation: {exc}"
            ) from exc

        token_ct = {
            "llm_calls": 1,
            "prompt_tokens": len(self.tokenizer.encode(prompt)),
            "output_tokens": len(self.tokenizer.encode(raw)),
        }
        return primer.model_dump(), token_ct

    DRIFTPrimer.decompose_query = _patched_decompose_query
    logger.info(
        "Patched DRIFTPrimer.decompose_query with tolerant JSON parsing"
    )
