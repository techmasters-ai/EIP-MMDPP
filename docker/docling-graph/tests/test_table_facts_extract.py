"""Tests for extract_label_rows (spec §5.3)."""
import importlib.util
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    import sys
    spec = importlib.util.spec_from_file_location("dg_tf", _FACTS_PATH)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["dg_tf"] = m
    spec.loader.exec_module(m)
    return m


def _cell(text, row, col, *, row_header=False, col_span=1):
    return {
        "text": text,
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + 1,
        "start_col_offset_idx": col,
        "end_col_offset_idx": col + col_span,
        "row_span": 1, "col_span": col_span,
        "row_header": row_header, "column_header": False,
    }


def _column_major_table_3rows_3entities():
    """3 rows × 4 cols (1 label + 3 entity columns)."""
    cells = [
        _cell("Length mm", 0, 0, row_header=True),
        _cell("10726", 0, 1), _cell("10841", 0, 2), _cell("10778", 0, 3),
        _cell("Diameter mm", 1, 0, row_header=True),
        _cell("654", 1, 1), _cell("654", 1, 2), _cell("654", 1, 3),
        _cell("Weight kg", 2, 0, row_header=True),
        _cell("2163", 2, 1), _cell("2283", 2, 2), _cell("2391", 2, 3),
    ]
    return {"data": {"table_cells": cells, "num_rows": 3, "num_cols": 4}}


def test_column_major_extraction_basic():
    tf = _load()
    table = _column_major_table_3rows_3entities()
    rows = tf.extract_label_rows(table, tf.Shape.COLUMN_MAJOR)
    assert len(rows) == 3
    assert rows[0]["label_text"] == "Length mm"
    assert rows[0]["data_cells"] == {1: "10726", 2: "10841", 3: "10778"}
    assert rows[1]["label_text"] == "Diameter mm"
    assert rows[2]["label_text"] == "Weight kg"


def test_column_major_label_col_span_carried_through():
    tf = _load()
    cells = [
        _cell("Industry Designation", 0, 0, row_header=True, col_span=2),
        _cell("SA-75", 0, 2), _cell("S-75", 0, 3), _cell("S-75M", 0, 4),
        _cell("Length mm", 1, 0, row_header=True, col_span=2),
        _cell("10726", 1, 2), _cell("10841", 1, 3), _cell("10778", 1, 4),
    ]
    table = {"data": {"table_cells": cells, "num_rows": 2, "num_cols": 5}}
    rows = tf.extract_label_rows(table, tf.Shape.COLUMN_MAJOR)
    assert rows[0]["label_text"] == "Industry Designation"
    assert rows[0]["label_col_span"] == 2
    assert rows[0]["data_cells"] == {2: "SA-75", 3: "S-75", 4: "S-75M"}


def test_row_major_transposition():
    """Top row holds labels; each subsequent row is one entity."""
    tf = _load()
    cells = [
        {**_cell("System", 0, 0), "column_header": True},
        {**_cell("Length mm", 0, 1), "column_header": True},
        {**_cell("Weight kg", 0, 2), "column_header": True},
        _cell("1D",  1, 0), _cell("10726", 1, 1), _cell("2163", 1, 2),
        _cell("13D", 2, 0), _cell("10841", 2, 1), _cell("2283", 2, 2),
        _cell("20D", 3, 0), _cell("10778", 3, 1), _cell("2391", 3, 2),
    ]
    table = {"data": {"table_cells": cells, "num_rows": 4, "num_cols": 3}}
    rows = tf.extract_label_rows(table, tf.Shape.ROW_MAJOR)
    assert len(rows) == 2
    by_label = {r["label_text"]: r for r in rows}
    assert by_label["Length mm"]["data_cells"] == {1: "10726", 2: "10841", 3: "10778"}
    assert by_label["Weight kg"]["data_cells"] == {1: "2163", 2: "2283", 3: "2391"}


def test_skips_empty_label_rows():
    tf = _load()
    cells = [
        _cell("Length mm", 0, 0, row_header=True),
        _cell("10726", 0, 1), _cell("10841", 0, 2), _cell("10778", 0, 3),
        _cell("", 1, 0, row_header=True),
        _cell("654", 1, 1), _cell("654", 1, 2), _cell("654", 1, 3),
        _cell("Weight kg", 2, 0, row_header=True),
        _cell("2163", 2, 1), _cell("2283", 2, 2), _cell("2391", 2, 3),
    ]
    table = {"data": {"table_cells": cells, "num_rows": 3, "num_cols": 4}}
    rows = tf.extract_label_rows(table, tf.Shape.COLUMN_MAJOR)
    assert len(rows) == 2
    assert {r["label_text"] for r in rows} == {"Length mm", "Weight kg"}


def test_section_header_row_with_row_header_false_is_included():
    """SA-2-style section header: col-0 cell with row_header=False but col_span
    bridging into data cols (here 6 > label_width 1) — must be in the LabelRow
    stream so detect_section_context's header-row strategy can match it."""
    tf = _load()
    cells = [
        _cell("Missile Type",        0, 0, row_header=True),
        _cell("1D",  0, 1), _cell("13D", 0, 2), _cell("13DM", 0, 3), _cell("13DA", 0, 4), _cell("20D", 0, 5),
        _cell("Total Weight kg",     1, 0, row_header=True),
        _cell("2163", 1, 1), _cell("2283", 1, 2), _cell("2283", 1, 3), _cell("2289", 1, 4), _cell("2391", 1, 5),
        # Section header — row_header=False, spans all 6 cols (label_width=1, span=6 > 1)
        _cell("1st Stage",           2, 0, col_span=6),
        _cell("Weight kg",           3, 0, row_header=True),
        _cell("1135", 3, 1), _cell("1032", 3, 2), _cell("1032", 3, 3), _cell("1032", 3, 4), _cell("1011", 3, 5),
        _cell("2nd Stage",           4, 0, col_span=6),
        _cell("Diameter mm",         5, 0, row_header=True),
        _cell("500", 5, 1), _cell("500", 5, 2), _cell("500", 5, 3), _cell("500", 5, 4), _cell("500", 5, 5),
    ]
    table = {"data": {"table_cells": cells, "num_rows": 6, "num_cols": 6}}
    rows = tf.extract_label_rows(table, tf.Shape.COLUMN_MAJOR)

    # Should have 6 LabelRow entries: 4 row_header=True + 2 section_header
    labels = [r["label_text"] for r in rows]
    assert "1st Stage" in labels
    assert "2nd Stage" in labels
    assert "Total Weight kg" in labels
    assert "Weight kg" in labels
    assert "Diameter mm" in labels

    # Section-header rows must have empty data_cells (the col_span=6 cell
    # occupies all data cols at that row, so no separate data cells exist).
    by_label = {r["label_text"]: r for r in rows}
    assert by_label["1st Stage"]["data_cells"] == {}
    assert by_label["2nd Stage"]["data_cells"] == {}

    # Spec rows still have their data cells correctly populated.
    assert by_label["Weight kg"]["data_cells"] == {1: "1135", 2: "1032", 3: "1032", 4: "1032", 5: "1011"}


def test_section_header_does_NOT_capture_ordinary_label_cells_with_col_span():
    """An identity row like 'Industry Designation' with col_span=2 == label_width
    must NOT be confused with a section header (col_span must be STRICTLY GREATER
    than label_width to qualify as a section header). Otherwise rows 0-3 of the
    SA-2 variants table (Industry/Military/NATO/Fan Song designation) — which all
    have row_header=True AND col_span=2 — would be captured by both branches."""
    tf = _load()
    cells = [
        _cell("Industry Designation", 0, 0, row_header=True, col_span=2),
        _cell("SA-75", 0, 2), _cell("S-75", 0, 3), _cell("S-75M", 0, 4),
        _cell("Length mm",            1, 0, row_header=True, col_span=2),
        _cell("10726", 1, 2), _cell("10841", 1, 3), _cell("10778", 1, 4),
    ]
    table = {"data": {"table_cells": cells, "num_rows": 2, "num_cols": 5}}
    rows = tf.extract_label_rows(table, tf.Shape.COLUMN_MAJOR)
    # Both rows kept (via row_header=True branch). Neither flagged as section.
    labels = [r["label_text"] for r in rows]
    assert "Industry Designation" in labels
    assert "Length mm" in labels
    # Industry Designation row has its identity data_cells populated normally.
    by_label = {r["label_text"]: r for r in rows}
    assert by_label["Industry Designation"]["data_cells"] == {2: "SA-75", 3: "S-75", 4: "S-75M"}
