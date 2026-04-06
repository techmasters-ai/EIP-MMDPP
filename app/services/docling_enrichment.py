"""Helpers for building enriched DoclingDocument copies for native tools.

The persisted DoclingDocument JSON in MinIO stays in the original language.
These functions build temporary enriched copies with translations applied
for consumption by HybridChunker and Docling-Graph.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _parse_self_ref(self_ref: str) -> tuple[str, int]:
    """Parse a Docling self_ref like '#/texts/0' into ('texts', 0).

    Returns ('', -1) if the format is unrecognized.
    """
    if not self_ref or not self_ref.startswith("#/"):
        return ("", -1)
    parts = self_ref[2:].split("/")
    if len(parts) != 2:
        return ("", -1)
    try:
        return (parts[0], int(parts[1]))
    except (ValueError, IndexError):
        return ("", -1)


def _build_enriched_copy_for_chunking(doc_dict: dict) -> dict:
    """Build a temporary copy with translations applied for HybridChunker.

    - Applies translations to canonical .text fields on text items
    - For tables: replaces TableData with single-cell table containing translated text
    - Does NOT inject context nodes (chunking should not see synthetic nodes)
    - meta.description on pictures is already in the persisted JSON
    """
    enriched = copy.deepcopy(doc_dict)
    enrichments = enriched.pop("_enrichments", {})

    for self_ref, trans in enrichments.get("translations", {}).items():
        collection, idx = _parse_self_ref(self_ref)
        if not collection or collection not in enriched or idx >= len(enriched.get(collection, [])):
            continue
        item = enriched[collection][idx]

        if collection == "tables":
            # Tables are structured: content lives in data.table_cells[].text.
            # Replace with single-cell table containing the full translated text.
            item["data"] = {
                "table_cells": [{
                    "text": trans["translated_text"],
                    "row_span": 1, "col_span": 1,
                    "start_row_offset_idx": 0, "end_row_offset_idx": 1,
                    "start_col_offset_idx": 0, "end_col_offset_idx": 1,
                    "column_header": False, "row_header": False,
                    "row_section": False, "fillable": False,
                }],
                "num_rows": 1, "num_cols": 1,
            }
        else:
            # TextItem, FormulaItem, SectionHeaderItem — all have .text
            item["text"] = trans["translated_text"]

    return enriched


def _build_enriched_copy_for_graph(doc_dict: dict) -> dict:
    """Build a temporary copy with translations + context node for Docling-Graph.

    Same as chunking copy, plus injects summary/classification context via
    native DoclingDocument.add_text() for safe tree insertion.
    """
    enriched = _build_enriched_copy_for_chunking(doc_dict)

    context = doc_dict.get("_enrichments", {}).get("context", {})
    if not (context.get("summary") or context.get("classification")):
        return enriched

    context_text = ""
    if context.get("summary"):
        context_text += f"[Document Summary]: {context['summary']}\n"
    if context.get("classification"):
        context_text += f"[Classification]: {context['classification']}\n"

    if context_text:
        try:
            from docling.datamodel.document import DoclingDocument
            doc = DoclingDocument.model_validate(enriched)
            doc.add_text(text=context_text, label="paragraph")
            enriched = doc.export_to_dict()
        except Exception as exc:
            logger.warning("Failed to inject context via native API: %s", exc)
            # Graceful degradation — enriched copy still has translations

    return enriched


def _regenerate_translated_markdown(doc_dict: dict) -> str:
    """Generate translated markdown from enriched copy + picture description appendix.

    Native export_to_markdown() does not serialize PictureMeta.description,
    so we append picture descriptions as a markdown appendix.
    """
    enriched = _build_enriched_copy_for_chunking(doc_dict)

    try:
        from docling.datamodel.document import DoclingDocument
        doc = DoclingDocument.model_validate(enriched)
        md = doc.export_to_markdown()
    except Exception as exc:
        logger.warning("Failed to generate markdown from DoclingDocument: %s", exc)
        # Fallback: concatenate translated text items
        md = "\n\n".join(
            item.get("text", "")
            for item in enriched.get("texts", [])
            if item.get("text")
        )

    # Append picture descriptions
    pic_descs = []
    for pic in doc_dict.get("pictures", []):
        meta = pic.get("meta") or {}
        desc = meta.get("description", {})
        if isinstance(desc, dict) and desc.get("text"):
            pic_descs.append(f"**Image:** {desc['text']}")
    if pic_descs:
        md += "\n\n---\n## Image Descriptions\n\n" + "\n\n".join(pic_descs)

    return md
