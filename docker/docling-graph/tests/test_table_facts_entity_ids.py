"""Tests for derive_entity_ids (spec §5.3.5)."""
import sys
from pathlib import Path

# Add app directory to path for direct import.
app_path = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(app_path))

import _table_facts as tf


def _row(idx, label, data):
    return {"row_idx": idx, "label_text": label, "label_col_span": 1, "data_cells": data}


def test_single_key_label_row():
    rows = [
        _row(0, "Missile Type", {1: "1D", 2: "13D", 3: "13DM"}),
        _row(1, "Length mm",    {1: "10726", 2: "10841", 3: "10841"}),
    ]
    ids = tf.derive_entity_ids(rows, tf.Shape.COLUMN_MAJOR)
    assert ids == {1: "1D", 2: "13D", 3: "13DM"}


def test_hybrid_composite_identity():
    rows = [
        _row(0, "Industry Designation", {1: "SA-75", 2: "S-75", 3: "S-75M"}),
        _row(1, "Missile Type",         {1: "1D",    2: "13D",  3: "13DM"}),
        _row(2, "Length mm",            {1: "10726", 2: "10841", 3: "10841"}),
    ]
    ids = tf.derive_entity_ids(rows, tf.Shape.HYBRID)
    assert ids == {1: "SA-75 1D", 2: "S-75 13D", 3: "S-75M 13DM"}


def test_no_key_label_row_returns_empty():
    """If no row matches _KEY_LABEL_PATTERNS, derive_entity_ids returns {}."""
    rows = [
        _row(0, "Length mm", {1: "10726", 2: "10841"}),
        _row(1, "Weight kg", {1: "2163", 2: "2283"}),
    ]
    ids = tf.derive_entity_ids(rows, tf.Shape.COLUMN_MAJOR)
    assert ids == {}


def test_empty_data_cell_excluded_from_composite():
    """Hybrid composite with one column missing the upper id — only complete columns appear."""
    rows = [
        _row(0, "Industry Designation", {1: "SA-75", 2: "", 3: "S-75M"}),
        _row(1, "Missile Type",         {1: "1D",    2: "13D",  3: "13DM"}),
    ]
    ids = tf.derive_entity_ids(rows, tf.Shape.HYBRID)
    assert ids[1] == "SA-75 1D"
    assert ids[2] == "13D"
    assert ids[3] == "S-75M 13DM"


def test_collision_last_write_wins():
    """Two columns producing the same composite — last one wins.
    derive_entity_ids deduplicates so only one entry per unique composite
    appears in the result."""
    rows = [
        _row(0, "Missile Type", {1: "1D", 2: "1D", 3: "13DM"}),
    ]
    ids = tf.derive_entity_ids(rows, tf.Shape.COLUMN_MAJOR)
    assert ids == {2: "1D", 3: "13DM"}
