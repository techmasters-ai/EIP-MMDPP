"""Strict-qualification gate tests for spec §3 / §5.2.

Guards against the user-flagged failure mode: a small earlier column-
major-shaped table starving the real variants table at row 6+.
"""
# Loader pattern: the docling-graph `app/` package is shadowed by the
# repo-root `app/` in normal test runs (pytest's rootdir wins index 0
# in sys.path; conftest only APPENDS the service root). To make
# `from app._alias_map import …` lazy imports inside _table_facts.py
# resolve to the docling-graph copy, prepend `docker/docling-graph/`
# AND `docker/docling-graph/app/` to sys.path before importing.
# Same pattern as test_table_facts_resolve.py and test_table_overlay_extract.py
# (Task 2). Pure importlib.spec_from_file_location does NOT work here.
import sys
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _SERVICE_ROOT / "app"
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_SERVICE_ROOT))
import _table_facts as _tf  # noqa: E402  (sys.path setup must precede)


def _load_table_facts():
    return _tf


def _make_qualifying_missile_table(num_cols: int = 5):
    """num_cols entity columns + label col 0. 2 identity rows
    (Missile Type, NATO Designation) + 2 spec rows (Length mm,
    Diameter mm). num_rows >= 4 satisfies _is_column_major_or_hybrid's
    rows-and-cols >= 4 gate. All entity columns have non-empty cells
    in BOTH identity rows (so it qualifies under all 4 gates)."""
    cells = []
    label_rows = (
        ("Missile Type", True),
        ("NATO Designation", True),
        ("Length mm", True),
        ("Diameter mm", True),
    )
    for r, (label, is_header) in enumerate(label_rows):
        cells.append({
            "start_row_offset_idx": r, "start_col_offset_idx": 0,
            "end_col_offset_idx": 1, "row_header": is_header,
            "text": label,
        })
    for col_idx in range(1, num_cols + 1):
        col_vals = (f"M{col_idx}", f"NATO{col_idx}", "10726", "654")
        for r, val in enumerate(col_vals):
            cells.append({
                "start_row_offset_idx": r,
                "start_col_offset_idx": col_idx,
                "end_col_offset_idx": col_idx + 1,
                "row_header": False, "text": val,
            })
    return {"data": {"table_cells": cells, "num_rows": 4,
                     "num_cols": num_cols + 1}}


def _make_unqualified_3_col_table():
    """3 entity columns (< 4) -> fails entity_columns gate."""
    return _make_qualifying_missile_table(num_cols=3)


def _make_unqualified_sparse_identity_table():
    """5 entity columns, identity rows exist, but only column 1 has
    non-empty cells in the identity rows -> fails sparse-identity gate."""
    table = _make_qualifying_missile_table(num_cols=5)
    cells = table["data"]["table_cells"]
    # Blank identity-row cells in cols 2..5 (rows 0 and 1).
    for c in cells:
        if (c.get("start_row_offset_idx") in (0, 1)
                and c.get("start_col_offset_idx", 0) >= 2):
            c["text"] = ""
    return table


def _make_qualifying_radar_table(num_cols: int = 5):
    """Radar version. 2 identity rows (Radar Variant, Radar Type) + 2
    spec rows (Frequency MHz, Range km) -> num_rows = 4."""
    cells = []
    label_rows = (
        ("Radar Variant", True),
        ("Radar Type", True),
        ("Frequency MHz", True),
        ("Range km", True),
    )
    for r, (label, is_header) in enumerate(label_rows):
        cells.append({
            "start_row_offset_idx": r, "start_col_offset_idx": 0,
            "end_col_offset_idx": 1, "row_header": is_header,
            "text": label,
        })
    for col_idx in range(1, num_cols + 1):
        col_vals = (f"R{col_idx}", f"Type{col_idx}", "3000", "75")
        for r, val in enumerate(col_vals):
            cells.append({
                "start_row_offset_idx": r,
                "start_col_offset_idx": col_idx,
                "end_col_offset_idx": col_idx + 1,
                "row_header": False, "text": val,
            })
    return {"data": {"table_cells": cells, "num_rows": 4,
                     "num_cols": num_cols + 1}}


def test_unqualified_earlier_table_does_not_starve_real_variants_table():
    """Doc has [unqualified_3col, qualifying_5col]. extract_table_overlay
    must skip the first (tables_skipped_unqualified++) and pick the
    second."""
    tf = _load_table_facts()
    doc = {"tables": [
        _make_unqualified_3_col_table(),
        _make_qualifying_missile_table(num_cols=5),
    ]}
    overlay, stats = tf.extract_table_overlay(doc)
    assert "MISSILE_SYSTEM" in overlay.alias_map_by_entity_type
    assert len(overlay.alias_map_by_entity_type["MISSILE_SYSTEM"]) > 0
    assert stats["tables_skipped_unqualified"] == 1
    assert stats["tables_skipped_multi"] == 0


def test_entity_columns_gate_under_4_rejects():
    tf = _load_table_facts()
    doc = {"tables": [_make_unqualified_3_col_table()]}
    overlay, stats = tf.extract_table_overlay(doc)
    assert overlay.alias_map_by_entity_type == {}
    assert stats["tables_skipped_unqualified"] == 1


def test_sparse_identity_cells_rejects():
    tf = _load_table_facts()
    doc = {"tables": [_make_unqualified_sparse_identity_table()]}
    overlay, stats = tf.extract_table_overlay(doc)
    assert overlay.alias_map_by_entity_type == {}
    assert stats["tables_skipped_unqualified"] == 1


def test_radar_qualifying_table_before_missile_v1_picks_first():
    """v1 picker is entity-type-agnostic. Radar table comes first -> it
    wins; missile table goes to tables_skipped_multi."""
    tf = _load_table_facts()
    doc = {"tables": [
        _make_qualifying_radar_table(num_cols=5),
        _make_qualifying_missile_table(num_cols=5),
    ]}
    overlay, stats = tf.extract_table_overlay(doc)
    assert "RADAR_SYSTEM" in overlay.alias_map_by_entity_type
    assert "MISSILE_SYSTEM" not in overlay.alias_map_by_entity_type
    assert stats["tables_skipped_multi"] == 1


def test_two_qualifying_missile_tables_first_wins():
    tf = _load_table_facts()
    doc = {"tables": [
        _make_qualifying_missile_table(num_cols=5),
        _make_qualifying_missile_table(num_cols=4),
    ]}
    overlay, stats = tf.extract_table_overlay(doc)
    assert "MISSILE_SYSTEM" in overlay.alias_map_by_entity_type
    assert stats["tables_skipped_multi"] == 1
