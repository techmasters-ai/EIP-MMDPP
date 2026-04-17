"""D-2: Document-number extraction helper tests (partial).

The main walker battery (D-3) extends this file. For D-2 we only cover
``_extract_document_number_from_front_matter`` which scans the first 30
items of a DoclingDocument for MIL-STD / TM / ISO / ANSI-style
designators in title or section-header text.
"""
from __future__ import annotations

import pytest

from docling_core.types.doc import DoclingDocument

from app.services.docling_anchors import _extract_document_number_from_front_matter


def _build_doc(titles: list[str], headings: list[tuple[str, int]] | None = None) -> DoclingDocument:
    """Build a minimal DoclingDocument fixture for front-matter tests.

    ``titles`` are added via ``add_title`` in order; ``headings`` is a
    list of ``(text, level)`` tuples added via ``add_heading`` in order
    AFTER all titles. A synthetic origin is provided so DoclingDocument
    can construct itself.
    """
    doc = DoclingDocument(
        name="Test",
        origin={
            "mimetype": "application/pdf",
            "filename": "test.pdf",
            "binary_hash": "0" * 32,
        },
    )
    for text in titles:
        doc.add_title(text=text)
    for text, level in (headings or []):
        doc.add_heading(text=text, level=level)
    return doc


def test_extracts_tm_designator_from_title():
    doc = _build_doc(titles=["TM 9-1425-386-12 Operator Manual"])
    assert _extract_document_number_from_front_matter(doc) == "TM 9-1425-386-12"


def test_returns_none_when_title_is_plain_text():
    doc = _build_doc(titles=["Introduction"])
    assert _extract_document_number_from_front_matter(doc) is None


def test_extracts_mil_std_designator():
    doc = _build_doc(titles=["MIL-STD-1553B Interface Standard"])
    assert _extract_document_number_from_front_matter(doc) == "MIL-STD-1553B"


def test_extracts_designator_from_section_header():
    """Document number may appear in a section header instead of the title."""
    doc = _build_doc(
        titles=["Operator Manual"],
        headings=[("Document: MIL-DTL-31000G Revision 1", 1)],
    )
    assert _extract_document_number_from_front_matter(doc) == "MIL-DTL-31000G"


def test_returns_none_when_no_designator_anywhere():
    doc = _build_doc(
        titles=["Some Manual"],
        headings=[("Chapter 1", 1), ("Overview", 2)],
    )
    assert _extract_document_number_from_front_matter(doc) is None


def test_returns_first_match_when_multiple_present():
    doc = _build_doc(
        titles=["TM 9-1425-386-12 Operator Manual"],
        headings=[("Refers to MIL-STD-1553B", 1)],
    )
    # First 30 items scanned, first match wins.
    assert _extract_document_number_from_front_matter(doc) == "TM 9-1425-386-12"


def test_case_insensitive_match():
    doc = _build_doc(titles=["tm 9-1425-386-12 operator manual"])
    assert _extract_document_number_from_front_matter(doc) == "tm 9-1425-386-12"


def test_bails_after_first_30_items():
    # Flood with 35 non-matching headings, then a matching one at position 36.
    headings = [(f"Chapter {i}", 1) for i in range(35)]
    headings.append(("TM 9-1425-386-12 Annex", 2))
    doc = _build_doc(titles=["No Designator Here"], headings=headings)
    assert _extract_document_number_from_front_matter(doc) is None
