"""Tests for deterministic prompt hint extraction."""

import importlib.util
from pathlib import Path


_CANDIDATES_PATH = Path(__file__).resolve().parent.parent / "app" / "_numeric_candidates.py"


def _load_numeric_candidates():
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_numeric_candidates",
        _CANDIDATES_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flattened_table_rows_group_cells_by_row_number():
    numeric_candidates = _load_numeric_candidates()
    text = (
        "SA-2 Guideline Variant Specifications "
        "Industry Designation, 1 = Industry Designation. "
        "Military Designation, 1 = Military Designation. "
        "Missile Type, 1 = Missile Type. "
        "Max Range, 1 = m. "
        "Industry Designation, 2 = SA-75. "
        "Military Designation, 2 = SA-75. "
        "Missile Type, 2 = 1D. "
        "Max Range, 2 = 29000. "
        "Industry Designation, 3 = S-75. "
        "Military Designation, 3 = S-75. "
        "Missile Type, 3 = 13D. "
        "Max Range, 3 = 34000."
    )

    rows = numeric_candidates.extract_flattened_table_rows(text)

    assert any("row 2:" in row and "Missile Type=1D" in row for row in rows)
    assert any("row 2:" in row and "Max Range=29000" in row for row in rows)
    assert any("row 3:" in row and "Missile Type=13D" in row for row in rows)


def test_flattened_table_block_explains_row_local_attribution():
    numeric_candidates = _load_numeric_candidates()

    block = numeric_candidates.render_flattened_table_block([
        "row 2: Missile Type=1D | Max Range=29000",
    ])

    assert "cells sharing the same row number belong together" in block
    assert "active schema" in block
    assert "not to an adjacent system" in block


def test_keyed_table_rows_emit_per_identifier_lines():
    numeric_candidates = _load_numeric_candidates()
    text = (
        "Industry Designation, 1 = Industry Designation. "
        "Missile Type, 1 = Missile Type. "
        "Max Range, 1 = m. "
        "Industry Designation, 2 = SA-75. "
        "Missile Type, 2 = 1D. "
        "Max Range, 2 = 29000. "
        "Max Alt, 2 = 22000. "
        "Industry Designation, 3 = S-75. "
        "Missile Type, 3 = 13D. "
        "Max Range, 3 = 34000."
    )

    rows = numeric_candidates.extract_keyed_table_rows(text)

    # Each data row produces one line per identifier cell. Spec values are
    # listed inline, already attributed to that identifier.
    assert any(
        "Missile Type=1D" in r and "Max Range=29000" in r and "Max Alt=22000" in r
        for r in rows
    )
    assert any(
        "Missile Type=13D" in r and "Max Range=34000" in r for r in rows
    )
    assert any(
        "Industry Designation=SA-75" in r and "Max Range=29000" in r for r in rows
    )


def test_keyed_table_rows_skip_header_and_unit_rows():
    numeric_candidates = _load_numeric_candidates()
    # Row 1 is the header row — values mirror labels or are unit tokens.
    text = (
        "Missile Type, 1 = Missile Type. "
        "Max Range, 1 = m. "
        "Missile Type, 2 = 1D. "
        "Max Range, 2 = 29000."
    )

    rows = numeric_candidates.extract_keyed_table_rows(text)

    # Only one line, for row 2 keyed by the single identifier.
    assert len(rows) == 1
    assert "Missile Type=1D" in rows[0]
    assert "Max Range=29000" in rows[0]


def test_keyed_table_rows_skip_rows_without_identifier():
    numeric_candidates = _load_numeric_candidates()
    # Row 2 has only spec cells, no identifier — should not produce a hint.
    text = (
        "Max Range, 1 = m. "
        "Max Alt, 1 = m. "
        "Max Range, 2 = 29000. "
        "Max Alt, 2 = 22000."
    )

    rows = numeric_candidates.extract_keyed_table_rows(text)
    assert rows == []


def test_keyed_table_block_explains_attribution():
    numeric_candidates = _load_numeric_candidates()

    block = numeric_candidates.render_keyed_table_block([
        "Missile Type=1D: Max Range=29000, Max Alt=22000",
    ])

    assert "already attributed to a per-row identifier" in block
    assert "Do NOT shuffle a spec to a different identifier" in block
    assert "Missile Type=1D" in block
