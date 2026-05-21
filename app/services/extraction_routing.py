"""C3 — Pre-pass skip router.

Deterministic, pure-function router that examines per-pass cue lists declared
in the manifest and decides whether to RUN_FULL or SKIP_NO_EVIDENCE for an
optional pass.

Design decisions (all user-locked):
  - Cue match algorithm: case-insensitive substring.  No regex, no word boundaries.
  - Cues live in manifest as ``cues: list[str]`` per pass (locality of reference).
  - Required passes and identity passes are NEVER skipped.
  - Empty cue list → no opinion → always RUN_FULL.
  - ``RUN_SLICED`` is reserved for C4c (chunk-scope selection); never produced here.

Env var ``DOCLING_GRAPH_SKIP_ROUTER_MODE`` is read by the CALLER
(``_claim_and_dispatch_pass``) not by this module — the router is a pure
function that always returns a decision.  The caller decides whether to act
on it based on the mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class RouteDecision:
    """Decision returned by ``route_pass``.

    Attributes:
        mode:             ``"RUN_FULL"`` (default), ``"RUN_SLICED"`` (reserved for
                          C4c — never produced here), or ``"SKIP_NO_EVIDENCE"``.
        reason:           Human-readable explanation for logging / diagnostics.
        matched_cues:     Deduplicated list of cues that had ≥1 substring hit
                          across all searched content.  Empty on SKIP or when
                          no cues are configured.
        chunks_with_hits: Number of content items (text chunks, table captions,
                          picture captions) that contained ≥1 cue hit.  Zero
                          on SKIP or when no content exists.
    """

    mode: Literal["RUN_FULL", "RUN_SLICED", "SKIP_NO_EVIDENCE"]
    reason: str
    matched_cues: list[str] = field(default_factory=list)
    chunks_with_hits: int = 0


def route_pass(pass_def, doc_metadata: dict) -> RouteDecision:
    """Examine doc content for cue matches per the pass's manifest-declared ``cues``.

    Parameters
    ----------
    pass_def:
        A ``PassManifest`` instance (or any object with ``phase: str``,
        ``required: bool``, ``cues: list[str]``).
    doc_metadata:
        A dict with at least:
          - ``"text_chunks"``    : list[str] of chunk text bodies
          - ``"table_captions"`` : list[str] (optional, may be absent or empty)
          - ``"picture_captions"``: list[str] (optional, may be absent or empty)

    Returns
    -------
    RouteDecision with mode ``"RUN_FULL"`` (default) or ``"SKIP_NO_EVIDENCE"``.

    Short-circuit rules (in priority order):
      1. ``pass_def.required == True``   → always ``RUN_FULL``.
      2. ``pass_def.phase == "identity"`` → always ``RUN_FULL``.
      3. ``pass_def.cues == []``         → always ``RUN_FULL`` (no opinion).
      4. Case-insensitive substring scan against text_chunks + table_captions +
         picture_captions. Any hit → ``RUN_FULL``. No hits → ``SKIP_NO_EVIDENCE``.
    """
    # 1. Required passes are always run.
    if getattr(pass_def, "required", False):
        return RouteDecision(
            mode="RUN_FULL",
            reason="required_pass",
            matched_cues=[],
            chunks_with_hits=0,
        )

    # 2. Identity passes are always run (they seed the roster for C4).
    if getattr(pass_def, "phase", None) == "identity":
        return RouteDecision(
            mode="RUN_FULL",
            reason="identity_pass",
            matched_cues=[],
            chunks_with_hits=0,
        )

    # 3. No cues configured → no opinion.
    cues: list[str] = list(getattr(pass_def, "cues", []) or [])
    if not cues:
        return RouteDecision(
            mode="RUN_FULL",
            reason="no_cues_configured",
            matched_cues=[],
            chunks_with_hits=0,
        )

    # 4. Case-insensitive substring scan.
    cues_lower = [c.lower() for c in cues]

    # Gather all content items to search: text chunks + table captions + picture captions.
    text_chunks: list[str] = list(doc_metadata.get("text_chunks") or [])
    table_captions: list[str] = list(doc_metadata.get("table_captions") or [])
    picture_captions: list[str] = list(doc_metadata.get("picture_captions") or [])
    all_items: list[str] = text_chunks + table_captions + picture_captions

    matched_cues_set: set[str] = set()
    chunks_with_hits: int = 0

    for item in all_items:
        item_lower = item.lower()
        item_hit = False
        for cue, cue_lower in zip(cues, cues_lower):
            if cue_lower in item_lower:
                matched_cues_set.add(cue)
                item_hit = True
        if item_hit:
            chunks_with_hits += 1

    if chunks_with_hits == 0:
        return RouteDecision(
            mode="SKIP_NO_EVIDENCE",
            reason=f"no cue hits among {len(cues)} cues",
            matched_cues=[],
            chunks_with_hits=0,
        )

    return RouteDecision(
        mode="RUN_FULL",
        reason=f"{chunks_with_hits} chunks with cue hits",
        matched_cues=sorted(matched_cues_set),  # sorted for determinism
        chunks_with_hits=chunks_with_hits,
    )
