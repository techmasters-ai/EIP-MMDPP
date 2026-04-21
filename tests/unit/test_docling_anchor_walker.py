"""D-2 + D-3: Docling anchor walker tests.

D-2 covers ``_extract_document_number_from_front_matter`` — MIL-STD /
TM / ISO-style designator extraction from front matter.

D-3 covers ``walk()`` — deterministic SECTION / FIGURE / TABLE /
DOCUMENT emission with section-stack mirroring converter.py:296–318.
Seven scenarios:
  1. Empty-structure doc → fallback SECTION(section_number="0")
  2. 3-level hierarchy → sections numbered "1", "1.1", "1.1.1"
  3. Siblings at same depth → "1.1", "1.1.2" per document order
  4. Figure count matches docling_document.pictures; figure_ref=self_ref
  5. Table count matches docling_document.tables
  6. CHILD_OF edges: one per non-root section
  7. HAS_SECTION/HAS_FIGURE/HAS_TABLE skipped when no document_number
"""
from __future__ import annotations

import pytest

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import TableCell, TableData

from app.services.docling_anchors import (
    _extract_document_number_from_front_matter,
    walk,
)


def _build_doc(
    titles: list[str] | None = None,
    headings: list[tuple[str, int]] | None = None,
    picture_count: int = 0,
    table_count: int = 0,
) -> DoclingDocument:
    """Build a minimal DoclingDocument fixture.

    Items are added in this order: titles → headings → pictures → tables.
    ``picture_count`` / ``table_count`` produce that many auto-ref'd
    entries via ``add_picture`` / ``add_table``. A synthetic origin is
    provided so DoclingDocument can construct itself.
    """
    doc = DoclingDocument(
        name="Test",
        origin={
            "mimetype": "application/pdf",
            "filename": "test.pdf",
            "binary_hash": "0" * 32,
        },
    )
    for text in titles or []:
        doc.add_title(text=text)
    for text, level in headings or []:
        doc.add_heading(text=text, level=level)
    for _ in range(picture_count):
        doc.add_picture()
    for _ in range(table_count):
        td = TableData(
            num_rows=1,
            num_cols=1,
            table_cells=[
                TableCell(
                    row_span=1, col_span=1,
                    start_row_offset_idx=0, end_row_offset_idx=1,
                    start_col_offset_idx=0, end_col_offset_idx=1,
                    text="a",
                )
            ],
        )
        doc.add_table(data=td)
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


# ---------------------------------------------------------------------------
# D-3 walker tests (7 scenarios from plan)
# ---------------------------------------------------------------------------

DOC_UUID = "doc-uuid-1"
RUN_ID = "run-id-1"


def _collect_section_numbers(merged, *, sort: bool = False) -> list[str]:
    nums = [
        e.identity.identity_tuple[0]
        for e in merged.entities
        if e.identity.entity_type == "SECTION"
    ]
    return sorted(nums) if sort else nums


def _sections_by_number(merged) -> dict[str, object]:
    return {
        e.identity.identity_tuple[0]: e
        for e in merged.entities
        if e.identity.entity_type == "SECTION"
    }


def test_walker_empty_structure_emits_fallback_section():
    """Scenario 1: doc with no headings/titles → fallback SECTION(section_number='0')."""
    doc = _build_doc()  # no titles, no headings
    doc_json = doc.model_dump()
    merged = walk(doc_json, DOC_UUID, RUN_ID, ontology={})
    nums = _collect_section_numbers(merged)
    assert nums == ["0"], f"expected fallback SECTION '0', got {nums}"
    # No DOCUMENT (no document_number extractable).
    doc_entities = [e for e in merged.entities if e.identity.entity_type == "DOCUMENT"]
    assert doc_entities == []
    # No HAS_* edges either.
    has_edges = [e for e in merged.edges if e.rel_type.startswith("HAS_")]
    assert has_edges == []


def test_walker_three_level_hierarchy():
    """Scenario 2: Chapter 1 → Section 1.1 → Subsection 1.1.2 → numbers '1','1.1','1.1.1'."""
    doc = _build_doc(
        headings=[
            ("Chapter 1", 1),
            ("Section 1.1", 2),
            ("Subsection 1.1.2", 3),
        ]
    )
    merged = walk(doc.model_dump(), DOC_UUID, RUN_ID, ontology={})
    nums = _collect_section_numbers(merged)
    # Document order: Chapter 1 registers path=("Chapter 1",), number "1".
    # Then Section 1.1 registers path=("Chapter 1","Section 1.1"), number "1.1".
    # Then Subsection 1.1.2 registers path=(...,"Subsection 1.1.2"), number "1.1.1".
    assert nums == ["1", "1.1", "1.1.1"], f"got {nums}"


def test_walker_sibling_sections_get_sequential_numbers():
    """Scenario 3: Parent with two children → "1.1" and "1.2" by document order."""
    doc = _build_doc(
        headings=[
            ("Alpha", 1),
            ("Bravo", 2),
            ("Charlie", 2),
        ]
    )
    merged = walk(doc.model_dump(), DOC_UUID, RUN_ID, ontology={})
    by_num = _sections_by_number(merged)
    # Alpha is "1"; Bravo first sibling = "1.1"; Charlie second sibling = "1.2".
    assert set(by_num.keys()) == {"1", "1.1", "1.2"}
    assert by_num["1"].properties["heading"] == "Alpha"
    assert by_num["1.1"].properties["heading"] == "Bravo"
    assert by_num["1.2"].properties["heading"] == "Charlie"


def test_walker_figure_count_and_ref():
    """Scenario 4: Figure count matches docling_document.pictures; figure_ref=self_ref."""
    doc = _build_doc(titles=["Manual"], picture_count=3)
    merged = walk(doc.model_dump(), DOC_UUID, RUN_ID, ontology={})
    figures = [e for e in merged.entities if e.identity.entity_type == "FIGURE"]
    assert len(figures) == 3
    refs = {e.identity.identity_tuple[0] for e in figures}
    assert refs == {p.self_ref for p in doc.pictures}


def test_walker_table_count():
    """Scenario 5: Table count matches docling_document.tables."""
    doc = _build_doc(titles=["Manual"], table_count=2)
    merged = walk(doc.model_dump(), DOC_UUID, RUN_ID, ontology={})
    tables = [e for e in merged.entities if e.identity.entity_type == "TABLE"]
    assert len(tables) == 2
    refs = {e.identity.identity_tuple[0] for e in tables}
    assert refs == {t.self_ref for t in doc.tables}


def test_walker_child_of_edges():
    """Scenario 6: CHILD_OF edges — one per non-root section, pointing to parent SECTION."""
    doc = _build_doc(
        headings=[
            ("Chapter 1", 1),
            ("Section 1.1", 2),
            ("Chapter 2", 1),
            ("Section 2.1", 2),
        ]
    )
    merged = walk(doc.model_dump(), DOC_UUID, RUN_ID, ontology={})
    child_of = [e for e in merged.edges if e.rel_type == "CHILD_OF"]
    # Two non-root sections: "1.1" → "1" and "2.1" → "2". (No siblings at root.)
    assert len(child_of) == 2
    pairs = {
        (e.from_identity.identity_tuple[0], e.to_identity.identity_tuple[0])
        for e in child_of
    }
    assert pairs == {("1.1", "1"), ("2.1", "2")}


def test_walker_skips_has_edges_when_no_document_number():
    """Scenario 7: No document_number extractable → no DOCUMENT entity → no HAS_* edges."""
    doc = _build_doc(
        titles=["Some Manual"],  # no designator
        headings=[("Chapter 1", 1)],
        picture_count=1,
        table_count=1,
    )
    merged = walk(doc.model_dump(), DOC_UUID, RUN_ID, ontology={})
    # DOCUMENT entity absent
    doc_entities = [e for e in merged.entities if e.identity.entity_type == "DOCUMENT"]
    assert doc_entities == []
    # HAS_SECTION / HAS_FIGURE / HAS_TABLE edges absent
    has_edges = [
        e for e in merged.edges
        if e.rel_type in ("HAS_SECTION", "HAS_FIGURE", "HAS_TABLE")
    ]
    assert has_edges == []


def test_walker_emits_document_and_has_edges_when_designator_present():
    """Positive case for scenario 7: designator present → DOCUMENT + HAS_* edges emitted."""
    doc = _build_doc(
        titles=["TM 9-1425-386-12 Operator Manual"],
        headings=[("Chapter 1", 1)],
        picture_count=2,
        table_count=1,
    )
    merged = walk(doc.model_dump(), DOC_UUID, RUN_ID, ontology={})
    doc_entities = [e for e in merged.entities if e.identity.entity_type == "DOCUMENT"]
    assert len(doc_entities) == 1
    assert doc_entities[0].identity.identity_tuple == ("TM 9-1425-386-12",)
    # TITLE and Chapter 1 are both effective-level-1 headings, so the TITLE
    # is popped when Chapter 1 is pushed. That yields 2 top-level SECTION
    # vertices — one HAS_SECTION edge per section, plus one HAS_FIGURE per
    # picture and one HAS_TABLE per table.
    rel_counts = {}
    for e in merged.edges:
        rel_counts[e.rel_type] = rel_counts.get(e.rel_type, 0) + 1
    section_count = sum(
        1 for e in merged.entities if e.identity.entity_type == "SECTION"
    )
    assert rel_counts.get("HAS_SECTION") == section_count
    assert rel_counts.get("HAS_FIGURE") == 2
    assert rel_counts.get("HAS_TABLE") == 1


def test_walker_returns_merged_extraction_shape():
    """Smoke: returned MergedExtraction carries the expected bookkeeping fields."""
    doc = _build_doc(titles=["Manual"], headings=[("Chapter 1", 1)])
    merged = walk(doc.model_dump(), DOC_UUID, RUN_ID, ontology={})
    assert merged.document_id == DOC_UUID
    assert merged.pipeline_run_id == RUN_ID
    assert merged.rejected_edges == []
    assert merged.rejections_by_pass == {}
    # Every emitted entity carries pass_origins={'document_anchors'}
    for e in merged.entities:
        assert e.pass_origins == {"document_anchors"}


# ---------------------------------------------------------------------------
# D-5 — fixture JSON files in tests/fixtures/docling_anchors/
# ---------------------------------------------------------------------------

import json as _json
from pathlib import Path as _Path

_FIXTURE_DIR = _Path(__file__).resolve().parent.parent / "fixtures" / "docling_anchors"


def _load(name: str) -> dict:
    return _json.loads((_FIXTURE_DIR / name).read_text())


def test_fixture_sa2_minimal_produces_4_sections():
    """Fixture: title + 3 headings at mixed levels. Walker emits 4 SECTIONs
    (Title → Chapter 1 [sibling], Chapter 2 [sibling], Section 2.1 [child])."""
    merged = walk(_load("sa2_minimal.json"), DOC_UUID, RUN_ID, ontology={})
    nums = _collect_section_numbers(merged)
    # TITLE is level 1; Chapter 1 is level 1 (pops title); Chapter 2 level 1
    # (pops Chapter 1); Section 2.1 level 2 (child of Chapter 2).
    # Expected: "1", "2", "3", "3.1".
    assert set(nums) == {"1", "2", "3", "3.1"}, f"got {sorted(nums)}"


def test_fixture_empty_structure_emits_fallback():
    merged = walk(_load("empty_structure.json"), DOC_UUID, RUN_ID, ontology={})
    nums = _collect_section_numbers(merged)
    assert nums == ["0"]


def test_fixture_with_figures_tables_counts_match():
    fixture = _load("with_figures_tables.json")
    merged = walk(fixture, DOC_UUID, RUN_ID, ontology={})
    figures = [e for e in merged.entities if e.identity.entity_type == "FIGURE"]
    tables = [e for e in merged.entities if e.identity.entity_type == "TABLE"]
    assert len(figures) == len(fixture.get("pictures", []))
    assert len(tables) == len(fixture.get("tables", []))


def test_fixture_with_document_number_emits_document_entity():
    merged = walk(_load("with_document_number.json"), DOC_UUID, RUN_ID, ontology={})
    docs = [e for e in merged.entities if e.identity.entity_type == "DOCUMENT"]
    assert len(docs) == 1
    assert docs[0].identity.identity_tuple == ("MIL-STD-1553B",)
    # HAS_SECTION edges emitted (one per SECTION).
    section_count = sum(
        1 for e in merged.entities if e.identity.entity_type == "SECTION"
    )
    has_sections = [e for e in merged.edges if e.rel_type == "HAS_SECTION"]
    assert len(has_sections) == section_count


# ---------------------------------------------------------------------------
# 3a — _ensure_root_section helper + end-of-traversal fallback
# ---------------------------------------------------------------------------

def test_walker_zero_headings_zero_pictures_still_emits_root_section():
    """End-of-traversal fallback — covers headingless docs with nothing
    anchored either. Must still produce SECTION(section_number="0").
    Adaption: _build_doc has no 'texts' kwarg; a no-heading/no-picture doc
    is equivalent — _build_doc() with no args produces that fixture."""
    doc = _build_doc()  # no titles, no headings, no pictures, no tables
    merged = walk(doc.model_dump(mode="json"), "doc-1", "run-1", ontology={})
    numbers = _collect_section_numbers(merged)
    assert numbers == ["0"], f"expected ['0'], got {numbers}"


# ---------------------------------------------------------------------------
# 3b — per-picture/table section tracking + pre-heading lazy seed
# ---------------------------------------------------------------------------

def test_walker_pre_heading_picture_lazy_seeds_root_section():
    """A picture appearing before the first heading (in a doc that HAS
    headings later) must attach to the synthetic section_number='0'.
    Without lazy-seeding, section_by_path.get(()) returns None and the
    SECTION→IMAGE edge is silently dropped. Design §4.1a."""
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import ProvenanceItem, BoundingBox
    doc = DoclingDocument(name="pre_heading_pic")
    doc.add_picture(
        image=None,
        prov=ProvenanceItem(page_no=1, bbox=BoundingBox(l=0, t=0, r=100, b=100), charspan=(0, 0)),
    )
    doc.add_heading("Section 1", level=1)
    merged = walk(doc.model_dump(mode="json"), "doc-1", "run-1", ontology={})

    sections = _sections_by_number(merged)
    assert "0" in sections, "root fallback SECTION(section_number='0') missing"
