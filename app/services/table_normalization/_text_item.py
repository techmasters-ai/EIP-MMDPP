"""Convert a GraphTableChunk into a docling TextItem dict for texts[].

Mirrors the pattern at _table_facts.py:818-826: self_ref is hand-rolled
as f"#/texts/{next_text_idx}"; the caller threads next_text_idx and
bumps it after each call.

Records (text_idx, cell_refs) in _provenance_bridge for downstream
field-provenance enrichment (spec §11.6 channel A).
"""
from __future__ import annotations

from typing import Tuple

from app.services.table_normalization.models import GraphTableChunk
from app.services.table_normalization._provenance_bridge import (
    record_text_idx_cell_refs,
)


def _text_item_from_chunk(
    gtc: GraphTableChunk,
    *,
    next_text_idx: int,
) -> Tuple[dict, int]:
    """Build a docling TextItem dict for a GraphTableChunk.

    Returns (text_item, next_text_idx + 1). Caller must thread the
    returned next_text_idx into subsequent calls to avoid collisions
    with existing #/texts/N entries.

    Side effect: records (next_text_idx, gtc.cell_refs) in the
    process-local provenance bridge for downstream
    _enrich_field_provenance_with_cell_refs lookup.
    """
    record_text_idx_cell_refs(next_text_idx, list(gtc.cell_refs))

    return ({
        "self_ref": f"#/texts/{next_text_idx}",
        # parent / children / content_layer are REQUIRED by DoclingDocument's
        # TextItem Pydantic model; without them the library raises a
        # ValidationError during "Input Normalization". Matches the shape
        # _table_facts.py:818-826 uses for its synthesized TextItems.
        "parent": {"$ref": "#/body"},
        "children": [],
        "content_layer": "body",
        "label": "text",
        "prov": [],  # empty — cell refs flow through the bridge, not prov[].$ref
        "orig": gtc.text,
        "text": gtc.text,
    }, next_text_idx + 1)
