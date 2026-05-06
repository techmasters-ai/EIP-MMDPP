"""Tests for detect_table_shape (spec §5.2)."""
import importlib.util
import sys
from pathlib import Path

_FACTS_PATH = Path(__file__).resolve().parent.parent / "app" / "_table_facts.py"


def _load():
    spec = importlib.util.spec_from_file_location("dg_tf_shape", _FACTS_PATH)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["dg_tf_shape"] = m
    spec.loader.exec_module(m)
    return m


def _cell(text, row, col, *, row_header=False, col_header=False, row_span=1, col_span=1):
    return {
        "text": text,
        "start_row_offset_idx": row,
        "end_row_offset_idx": row + row_span,
        "start_col_offset_idx": col,
        "end_col_offset_idx": col + col_span,
        "row_span": row_span,
        "col_span": col_span,
        "row_header": row_header,
        "column_header": col_header,
    }


def test_below_4x4_floor_returns_other():
    """Tables smaller than 4 rows × 4 cols are skipped."""
    tf = _load()
    table = {"data": {"table_cells": [_cell("a", 0, 0)], "num_rows": 1, "num_cols": 1}}
    assert tf.detect_table_shape(table) == tf.Shape.OTHER


def test_column_major_detection():
    """Leftmost col has row_header=True majority."""
    tf = _load()
    cells = []
    for r in range(4):
        cells.append(_cell(f"label{r}", r, 0, row_header=True))
    for r in range(4):
        for c in range(1, 4):
            cells.append(_cell(f"v{r}{c}", r, c))
    table = {"data": {"table_cells": cells, "num_rows": 4, "num_cols": 4}}
    assert tf.detect_table_shape(table) == tf.Shape.COLUMN_MAJOR


def test_row_major_detection():
    """Top row has column_header=True majority."""
    tf = _load()
    cells = []
    for c in range(4):
        cells.append(_cell(f"hdr{c}", 0, c, col_header=True))
    for r in range(1, 4):
        for c in range(4):
            cells.append(_cell(f"v{r}{c}", r, c))
    table = {"data": {"table_cells": cells, "num_rows": 4, "num_cols": 4}}
    assert tf.detect_table_shape(table) == tf.Shape.ROW_MAJOR


def test_hybrid_multi_row_left_labels():
    """Multiple rows have row_header=True at col 0 with no data — composite identity."""
    tf = _load()
    cells = []
    cells.append(_cell("Industry Designation", 0, 0, row_header=True))
    cells.append(_cell("Missile Type", 1, 0, row_header=True))
    cells.append(_cell("Length mm", 2, 0, row_header=True))
    cells.append(_cell("Weight kg", 3, 0, row_header=True))
    for r in range(4):
        for c in range(1, 4):
            cells.append(_cell(f"v{r}{c}", r, c))
    table = {"data": {"table_cells": cells, "num_rows": 4, "num_cols": 4}}
    shape = tf.detect_table_shape(table)
    assert shape in (tf.Shape.HYBRID, tf.Shape.COLUMN_MAJOR)


def test_other_shape_when_neither_pattern_matches():
    tf = _load()
    cells = []
    for r in range(4):
        for c in range(4):
            cells.append(_cell(f"v{r}{c}", r, c))
    table = {"data": {"table_cells": cells, "num_rows": 4, "num_cols": 4}}
    assert tf.detect_table_shape(table) == tf.Shape.OTHER
