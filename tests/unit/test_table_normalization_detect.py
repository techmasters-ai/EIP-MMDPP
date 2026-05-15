import json
import pytest
from pathlib import Path
from app.services.table_normalization.detect import (
    detect_shape, SPEC_ROW_KEYWORDS, SECTION_KEYWORDS, IDENTITY_LABEL_KEYWORDS,
)
from app.services.table_normalization.models import Shape


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def test_sa2_fixture_detected_as_hybrid():
    shape = detect_shape(SA2_FIXTURE["table_cells"], SA2_FIXTURE)
    assert shape == Shape.HYBRID


def test_undersized_table_returns_other():
    tiny = {"table_cells": [
        {"text": "a", "row_header": True, "start_row_offset_idx": 0, "end_row_offset_idx": 0, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
    ]}
    assert detect_shape(tiny["table_cells"], tiny) == Shape.OTHER


def test_plain_column_major_table():
    cells = [
        {"text": "Max Range", "row_header": True, "start_row_offset_idx": 0, "end_row_offset_idx": 0, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
        {"text": "Weight", "row_header": True, "start_row_offset_idx": 1, "end_row_offset_idx": 1, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
        {"text": "Length", "row_header": True, "start_row_offset_idx": 2, "end_row_offset_idx": 2, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
        {"text": "Diameter", "row_header": True, "start_row_offset_idx": 3, "end_row_offset_idx": 3, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
        {"text": "Speed", "row_header": True, "start_row_offset_idx": 4, "end_row_offset_idx": 4, "start_col_offset_idx": 0, "end_col_offset_idx": 0},
    ]
    for col in range(1, 5):
        for row in range(5):
            cells.append({
                "text": f"v{row}{col}", "row_header": False,
                "start_row_offset_idx": row, "end_row_offset_idx": row,
                "start_col_offset_idx": col, "end_col_offset_idx": col,
            })
    table_data = {"num_rows": 5, "num_cols": 5, "table_cells": cells}
    assert detect_shape(cells, table_data) == Shape.COLUMN_MAJOR


def test_row_major_table():
    cells = []
    for col_idx, label in enumerate(["Variant", "Max Range", "Weight", "Length", "Speed"]):
        cells.append({
            "text": label, "column_header": True,
            "start_row_offset_idx": 0, "end_row_offset_idx": 0,
            "start_col_offset_idx": col_idx, "end_col_offset_idx": col_idx,
        })
    for row in range(1, 5):
        for col in range(5):
            cells.append({
                "text": f"v{row}{col}", "column_header": False, "row_header": False,
                "start_row_offset_idx": row, "end_row_offset_idx": row,
                "start_col_offset_idx": col, "end_col_offset_idx": col,
            })
    table_data = {"num_rows": 5, "num_cols": 5, "table_cells": cells}
    assert detect_shape(cells, table_data) == Shape.ROW_MAJOR


def test_keyword_lists_are_frozensets():
    assert isinstance(SPEC_ROW_KEYWORDS, frozenset)
    assert isinstance(SECTION_KEYWORDS, frozenset)
    assert isinstance(IDENTITY_LABEL_KEYWORDS, frozenset)
    assert "max range" in SPEC_ROW_KEYWORDS
    assert "1st stage" in SECTION_KEYWORDS
    assert "nato designation" in IDENTITY_LABEL_KEYWORDS


def test_malformed_cells_return_other_not_crash():
    bad = {"table_cells": [{"text": "x"}]}  # missing all offset fields
    assert detect_shape(bad["table_cells"], bad) == Shape.OTHER
