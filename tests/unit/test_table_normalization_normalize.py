import json
import pytest
from pathlib import Path
from app.services.table_normalization import normalize_tables
from app.services.table_normalization.models import Shape


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def _wrap_in_doc_json(table_fixture: dict, table_idx: int = 0) -> dict:
    return {
        "tables": [table_fixture],
        "texts": [],
    }


def test_normalize_sa2_returns_hybrid():
    doc = _wrap_in_doc_json(SA2_FIXTURE)
    result = normalize_tables(doc)
    assert len(result) == 1
    nt = result[0]
    assert nt.shape == Shape.HYBRID
    assert nt.table_index == 0
    assert nt.self_ref == "#/tables/0"


def test_normalize_other_table_carries_empty_cells():
    doc = _wrap_in_doc_json({"table_cells": [{"text": "x"}], "text": "raw"})
    result = normalize_tables(doc)
    assert len(result) == 1
    nt = result[0]
    assert nt.shape == Shape.OTHER
    assert nt.cells == ()
    assert nt.raw_markdown == "raw"


def test_normalize_does_not_mutate_doc_json():
    doc = _wrap_in_doc_json(SA2_FIXTURE)
    snapshot = json.dumps(doc, sort_keys=True)
    normalize_tables(doc)
    after = json.dumps(doc, sort_keys=True)
    assert snapshot == after


def test_normalize_skips_empty_cells():
    """Per §8 step 5 — empty cell values are dropped from NormalizedCells."""
    doc = _wrap_in_doc_json(SA2_FIXTURE)
    result = normalize_tables(doc)
    nt = result[0]
    assert all(c.value.strip() for c in nt.cells)


def test_normalize_extracts_units_from_row_labels():
    """A row labeled 'Max Range (m)' should yield NormalizedRow.unit == 'm'."""
    cells = []
    cells.append({
        "text": "Max Range (m)", "row_header": True,
        "start_row_offset_idx": 0, "end_row_offset_idx": 0,
        "start_col_offset_idx": 0, "end_col_offset_idx": 0,
    })
    for r in range(1, 4):
        cells.append({
            "text": f"Row{r}", "row_header": True,
            "start_row_offset_idx": r, "end_row_offset_idx": r,
            "start_col_offset_idx": 0, "end_col_offset_idx": 0,
        })
    for r in range(4):
        for c in range(1, 5):
            cells.append({
                "text": f"v{r}{c}", "row_header": False,
                "start_row_offset_idx": r, "end_row_offset_idx": r,
                "start_col_offset_idx": c, "end_col_offset_idx": c,
            })
    table = {"table_cells": cells, "num_rows": 4, "num_cols": 5}
    doc = _wrap_in_doc_json(table)
    result = normalize_tables(doc)
    nt = result[0]
    matches = [r for r in nt.rows if "Max Range" in (r.label or "")]
    assert matches, f"no 'Max Range' row found; rows={[r.label for r in nt.rows]}"
    assert matches[0].unit == "m"


def test_normalize_continues_on_per_table_failure():
    """One bad table doesn't stop other tables from being normalized."""
    doc = {
        "tables": [
            SA2_FIXTURE,                  # good
            None,                          # corrupt — will raise inside per-table loop
            SA2_FIXTURE,                  # good again
        ],
        "texts": [],
    }
    result = normalize_tables(doc)
    assert len(result) == 3
    assert result[0].shape == Shape.HYBRID
    assert result[1].shape == Shape.OTHER
    assert result[2].shape == Shape.HYBRID
