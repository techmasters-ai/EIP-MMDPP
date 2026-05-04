"""Tests for column-major table pivoting (Phase B / B1+B2)."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_PIVOT_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_pivot.py"


def _load_pivot():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_table_pivot", _PIVOT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cell(text, row, col, *, row_header=False, row_span=1, col_span=1):
    return {
        "text": text,
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + row_span,
        "start_col_offset_idx": col,
        "end_col_offset_idx": col + col_span,
        "row_span": row_span,
        "col_span": col_span,
        "row_header": row_header,
        "column_header": False,
    }


def _variants_table_doc():
    """Mini variants-style column-major table: Industry Designation row label
    spans cols 0-1; data starts at col 2; subsequent rows include
    Missile Type (key) and Max Range / Max Alt (specs)."""
    cells = [
        # Row 0: Industry Designation row, label spans cols 0-1
        _cell("Industry Designation", 0, 0, row_header=True, col_span=2),
        _cell("SA-75", 0, 2),
        _cell("S-75", 0, 3),
        _cell("S-75M", 0, 4),
        # Row 1: Missile Type — key identifier
        _cell("Missile Type", 1, 0, row_header=True, col_span=2),
        _cell("1D", 1, 2),
        _cell("13D", 1, 3),
        _cell("13DM", 1, 4),
        # Row 2: Max Range — spec
        _cell("Max Range m", 2, 0, row_header=True, col_span=2),
        _cell("29000", 2, 2),
        _cell("34000", 2, 3),
        _cell("43000", 2, 4),
        # Row 3: Max Alt — spec
        _cell("Max Alt m", 3, 0, row_header=True, col_span=2),
        _cell("22000", 3, 2),
        _cell("27000", 3, 3),
        _cell("30000", 3, 4),
    ]
    return {
        "tables": [
            {
                "self_ref": "#/tables/0",
                "data": {
                    "table_cells": cells,
                    "num_rows": 4,
                    "num_cols": 5,
                },
                "prov": [{"page_no": 6}],
            }
        ],
        "texts": [],
        "body": {"children": []},
    }


def test_synthesize_pivots_column_major_variants_table():
    pivot = _load_pivot()
    doc = _variants_table_doc()

    out, n = pivot.synthesize_pivoted_table_texts(doc)

    # Three data columns -> three new text items.
    assert n == 3
    new_texts = [t["text"] for t in out["texts"]]
    assert len(new_texts) == 3

    # Each summary names BOTH the Industry Designation and the Missile Type
    # for its column, then lists the spec rows.
    assert any(
        "Industry Designation=SA-75" in s
        and "Missile Type=1D" in s
        and "Max Range m=29000" in s
        and "Max Alt m=22000" in s
        for s in new_texts
    )
    assert any(
        "Industry Designation=S-75" in s and "Missile Type=13D" in s for s in new_texts
    )
    assert any(
        "Industry Designation=S-75M" in s
        and "Missile Type=13DM" in s
        and "Max Range m=43000" in s
        for s in new_texts
    )


def test_synthesize_appends_to_body_children_so_chunker_walks_them():
    pivot = _load_pivot()
    doc = _variants_table_doc()

    out, n = pivot.synthesize_pivoted_table_texts(doc)

    assert n == 3
    children = out["body"]["children"]
    assert len(children) == 3
    # Each new child is a $ref to the corresponding new texts[] entry.
    refs = sorted(c["$ref"] for c in children)
    assert refs == [f"#/texts/{i}" for i in range(3)]


def test_synthesize_skips_row_major_tables():
    pivot = _load_pivot()
    # A row-major financial-style table: column headers in row 0 (column_header
    # not row_header), data rows 1+. No row_header flags set on col-0 cells.
    cells = [
        # Row 0 — column headers
        {**_cell("Quarter", 0, 0), "column_header": True},
        {**_cell("Revenue", 0, 1), "column_header": True},
        {**_cell("Cost", 0, 2), "column_header": True},
        # Row 1 — data
        _cell("Q1", 1, 0),
        _cell("100", 1, 1),
        _cell("80", 1, 2),
        _cell("Q2", 2, 0),
        _cell("120", 2, 1),
        _cell("85", 2, 2),
        _cell("Q3", 3, 0),
        _cell("130", 3, 1),
        _cell("90", 3, 2),
    ]
    doc = {
        "tables": [{"data": {"table_cells": cells, "num_rows": 4, "num_cols": 3}}],
        "texts": [],
        "body": {"children": []},
    }

    out, n = pivot.synthesize_pivoted_table_texts(doc)

    assert n == 0
    assert out["texts"] == []


def test_synthesize_skips_when_no_key_label_present():
    """Column-major table with row labels but none matching identity patterns
    should be skipped — without an identifier we have nothing to attribute."""
    pivot = _load_pivot()
    cells = [
        _cell("Frequency", 0, 0, row_header=True),
        _cell("100", 0, 1),
        _cell("200", 0, 2),
        _cell("300", 0, 3),
        _cell("Power", 1, 0, row_header=True),
        _cell("10", 1, 1),
        _cell("20", 1, 2),
        _cell("30", 1, 3),
        _cell("Range", 2, 0, row_header=True),
        _cell("5", 2, 1),
        _cell("10", 2, 2),
        _cell("15", 2, 3),
        _cell("Bandwidth", 3, 0, row_header=True),
        _cell("1", 3, 1),
        _cell("2", 3, 2),
        _cell("3", 3, 3),
    ]
    doc = {
        "tables": [{"data": {"table_cells": cells, "num_rows": 4, "num_cols": 4}}],
        "texts": [],
        "body": {"children": []},
    }

    out, n = pivot.synthesize_pivoted_table_texts(doc)
    assert n == 0


def test_synthesize_preserves_original_tables_array():
    pivot = _load_pivot()
    doc = _variants_table_doc()
    original_table_cells = doc["tables"][0]["data"]["table_cells"][:]

    out, n = pivot.synthesize_pivoted_table_texts(doc)

    assert n == 3
    # Original tables[] is untouched.
    assert out["tables"][0]["data"]["table_cells"] == original_table_cells


def test_synthesize_handles_doc_with_no_tables():
    pivot = _load_pivot()
    doc = {"tables": [], "texts": [], "body": {"children": []}}
    out, n = pivot.synthesize_pivoted_table_texts(doc)
    assert n == 0
    assert out["texts"] == []


def test_synthesize_caps_at_max_synthesized():
    pivot = _load_pivot()
    # 20 data columns; cap at 10.
    cells = [_cell("Missile Type", 0, 0, row_header=True)]
    cells += [_cell(f"M{i}", 0, i) for i in range(1, 21)]
    cells += [_cell("Max Range m", 1, 0, row_header=True)]
    cells += [_cell(f"{1000 + i*10}", 1, i) for i in range(1, 21)]
    cells += [_cell("Max Alt m", 2, 0, row_header=True)]
    cells += [_cell(f"{2000 + i*10}", 2, i) for i in range(1, 21)]
    cells += [_cell("Length mm", 3, 0, row_header=True)]
    cells += [_cell(f"{10000 + i}", 3, i) for i in range(1, 21)]
    doc = {
        "tables": [{"data": {"table_cells": cells, "num_rows": 4, "num_cols": 21}}],
        "texts": [],
        "body": {"children": []},
    }

    out, n = pivot.synthesize_pivoted_table_texts(doc, max_synthesized=10)
    assert n == 10
    assert len(out["texts"]) == 10
