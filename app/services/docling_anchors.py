"""Docling anchor walker — deterministic document structure emission.

Replaces the LLM reference pass (deleted in C-1) for SECTION / FIGURE /
TABLE / DOCUMENT entities. Structure is derived from the DoclingDocument
tree, not the LLM, per docs R-rules and spec §3.3.

This module exposes:
  * ``_extract_document_number_from_front_matter`` — scans the first
    N titles/section-headers for a MIL-STD / TM / ISO-style designator
    (implemented in D-2).
  * ``walk`` — traverses a DoclingDocument and emits MergedEntityRecord
    + MergedEdgeRecord instances for the anchor ontology (D-3).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from docling_core.types.doc import DocItemLabel

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


# Matches MIL-STD/TM/MIL-DTL/MIL-HDBK/MIL-PRF/ANSI-IEEE/ISO/DoD style
# designators anywhere in a title or heading. Requires the designator
# token to be followed by at least one alphanumeric identifier char so
# plain text like "See TM" alone doesn't match.
_DOC_NUMBER_RE = re.compile(
    r"\b(?:TM|MIL-STD|MIL-DTL|MIL-HDBK|MIL-PRF|ANSI/IEEE|ISO|DoD)"
    r"\s*[-\s]?[A-Z0-9][\w.-]+",
    re.IGNORECASE,
)

_FRONT_MATTER_LIMIT = 30


def _extract_document_number_from_front_matter(
    docling_doc: "DoclingDocument",
) -> str | None:
    """Return the first MIL-STD/TM/ISO-style designator found in the
    first ``_FRONT_MATTER_LIMIT`` title or section-header items, or
    None if none is found.

    Case-insensitive; the matched substring is returned with its
    original casing. Non-title / non-section-header items advance the
    30-item limit counter but are not scanned for designators — docs
    R-rule: document number lives in front matter, not body.
    """
    count = 0
    for item, _level in docling_doc.iterate_items():
        if count >= _FRONT_MATTER_LIMIT:
            return None
        count += 1
        label = getattr(item, "label", None)
        if label not in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
            continue
        text = getattr(item, "text", None) or ""
        match = _DOC_NUMBER_RE.search(text)
        if match:
            return match.group(0).strip()
    return None
