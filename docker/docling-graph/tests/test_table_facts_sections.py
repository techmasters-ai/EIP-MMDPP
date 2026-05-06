"""Tests for detect_section_context (spec §5.4)."""
import importlib.util
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
_FACTS_PATH = _APP_DIR / "_table_facts.py"
_ALIAS_MAP_PATH = _APP_DIR / "_alias_map.py"


def _load():
    # First load _alias_map
    spec_alias = importlib.util.spec_from_file_location("app._alias_map", _ALIAS_MAP_PATH)
    m_alias = importlib.util.module_from_spec(spec_alias)
    m_alias.__package__ = "app"
    sys.modules["app._alias_map"] = m_alias
    assert spec_alias.loader is not None
    spec_alias.loader.exec_module(m_alias)

    # Then load _table_facts which imports from _alias_map
    spec = importlib.util.spec_from_file_location("app._table_facts", _FACTS_PATH)
    m = importlib.util.module_from_spec(spec)
    m.__package__ = "app"
    sys.modules["app._table_facts"] = m
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def _row(idx, label, data):
    return {"row_idx": idx, "label_text": label, "label_col_span": 1, "data_cells": data}


def test_embedded_section_keyword_in_label():
    """SA-2 PDF style: '1st Stage Weight kg' has section embedded in label."""
    tf = _load()
    rows = [
        _row(0, "Length mm",            {1: "10726"}),
        _row(1, "1st Stage Weight kg",  {1: "1135"}),
        _row(2, "1st Stage Time sec",   {1: "4.0"}),
        _row(3, "2nd Stage Weight kg",  {1: "1028"}),
    ]
    result = tf.detect_section_context(rows)
    contexts = {r["row_idx"]: ctx for r, ctx in result}
    assert contexts[0] is None
    assert contexts[1] == "1st Stage"
    assert contexts[2] == "1st Stage"
    assert contexts[3] == "2nd Stage"


def test_embedded_strips_section_from_label_text():
    """When a section keyword is embedded, the returned LabelRow's label_text
    has the section keyword removed so resolve_alias can look up the bare
    label without the section prefix."""
    tf = _load()
    rows = [
        _row(1, "1st Stage Weight kg",  {1: "1135"}),
    ]
    result = tf.detect_section_context(rows)
    new_row, ctx = result[0]
    assert ctx == "1st Stage"
    assert new_row["label_text"] == "Weight kg"  # section prefix stripped


def test_header_row_strategy():
    """Header-row marker rows propagate context to subsequent rows."""
    tf = _load()
    rows = [
        _row(0, "Missile Type", {1: "1D", 2: "13D"}),
        _row(1, "Total Weight kg", {1: "2163", 2: "2283"}),
        _row(2, "1st Stage", {}),  # header-row marker (no data cells)
        _row(3, "Weight kg",  {1: "1135", 2: "1032"}),
        _row(4, "Time sec",   {1: "4.0", 2: "4.0"}),
        _row(5, "2nd Stage", {}),
        _row(6, "Weight kg",  {1: "1028", 2: "1251"}),
    ]
    result = tf.detect_section_context(rows)
    contexts = {r["row_idx"]: ctx for r, ctx in result if r["row_idx"] not in (2, 5)}
    assert contexts[0] is None
    assert contexts[1] is None
    assert contexts[3] == "1st Stage"
    assert contexts[4] == "1st Stage"
    assert contexts[6] == "2nd Stage"
    out_rows = [r for r, _ in result]
    out_idxs = {r["row_idx"] for r in out_rows}
    assert 2 not in out_idxs
    assert 5 not in out_idxs


def test_embedded_wins_over_header_row():
    """If a row has both an embedded keyword and a propagating header-row
    context that would assign a different keyword, embedded wins."""
    tf = _load()
    rows = [
        _row(0, "1st Stage", {}),  # header-row marker
        _row(1, "2nd Stage Weight kg", {1: "1028"}),  # embedded "2nd Stage"
    ]
    result = tf.detect_section_context(rows)
    contexts = {r["row_idx"]: ctx for r, ctx in result if r["row_idx"] != 0}
    assert contexts[1] == "2nd Stage"
