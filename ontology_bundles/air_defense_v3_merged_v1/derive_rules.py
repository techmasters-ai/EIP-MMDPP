"""Deterministic post-merge structural edge derivation.

Produces edges that are NOT extracted by the LLM — only rules whose
outputs are 100% deterministic given the inputs.

IMPORTANT: HAS_PROVENANCE edges are NOT produced here. They are
created automatically by graph_store.upsert_nodes_batch_sync via
its internal _create_provenance_edges_batch_sync helper whenever a
non-None ProvenanceMetadata is passed in phase 2 of the worker's
three-phase import. Duplicating them here would produce two
HAS_PROVENANCE edges per entity. See spec §3.8 + §5.6 Phase 2.
"""
from __future__ import annotations

import logging
import re
from typing import Any

# ChunkForDerivation and DerivedEdge are canonically defined in
# app.services.extraction_merge (Task 3.4). Import from there to avoid
# duplication. Re-export so existing callers that import from this module
# continue to work without changes.
from app.services.extraction_merge import ChunkForDerivation, DerivedEdge

logger = logging.getLogger(__name__)

__all__ = ["ChunkForDerivation", "DerivedEdge", "normalize_name", "derive_structural_edges"]

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_name(name: str | None) -> str:
    """Canonicalize a display label for substring matching against chunk text.
    Returns '' for None or empty input so callers can skip falsy results."""
    if not name:
        return ""
    return _WHITESPACE_RE.sub(" ", name.strip().lower())


def derive_structural_edges(
    merged: Any,
    identity_to_rid: dict,
    chunks: list[ChunkForDerivation],
    document_rid: str,
) -> list[DerivedEdge]:
    """Deterministic edges that are NOT extracted by the LLM.

    IMPORTANT: HAS_PROVENANCE edges are NOT produced here — they are
    auto-created in phase 2 by graph_store.upsert_nodes_batch_sync
    via its internal _create_provenance_edges_batch_sync helper.

    Current rules:
    - MENTIONED_IN: from each extracted entity to every chunk whose
      normalized text contains the entity's normalized display label.
    """
    edges: list[DerivedEdge] = []

    for entity in merged.entities:
        from_rid = identity_to_rid.get(entity.identity)
        # Truthy-check: _extract_rids pads missing RIDs with "" (empty string),
        # not None — so `is None` lets empty strings through and we'd emit
        # `CREATE EDGE MENTIONED_IN FROM  TO #...` which ArcadeDB rejects as
        # a SQL parse error. Skip both None and empty.
        if not from_rid:
            logger.warning(
                "derive_structural_edges: skipping entity %s (identity=%s) — "
                "no upsert RID available (upsert likely returned an empty row)",
                getattr(entity, "display_label", None), entity.identity,
            )
            continue
        canonical = normalize_name(entity.display_label)
        if not canonical:
            continue
        for chunk in chunks:
            if canonical in chunk.text_normalized:
                if not chunk.rid:
                    continue
                edges.append(
                    DerivedEdge(
                        from_id=from_rid,
                        to_id=chunk.rid,
                        rel_type="MENTIONED_IN",
                        confidence=entity.confidence,
                    )
                )

    # CONTAINS_TEXT / CONTAINS_IMAGE / NEXT_CHUNK are handled by the
    # existing derive_structure_links stage — not duplicated here.
    # HAS_PROVENANCE is handled by upsert_nodes_batch_sync — see above.

    return edges
