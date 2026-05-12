"""Task 17: verify _text_item_from_chunk produces a docling TextItem dict
and records cell_refs in the provenance bridge."""
from __future__ import annotations

import json
from pathlib import Path

from app.services.table_normalization import normalize_tables, render_for_graph
from app.services.table_normalization._text_item import _text_item_from_chunk
from app.services.table_normalization import _provenance_bridge as bridge


def test_text_item_assigns_self_ref_and_records_bridge():
    sa2 = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())
    nt = normalize_tables({"tables": [sa2], "texts": []})[0]
    chunks = render_for_graph(nt, token_limit_whole=100, token_limit_column=100)  # force per-column

    bridge.reset()
    next_text_idx = 100
    items: list[dict] = []
    for c in chunks:
        item, next_text_idx = _text_item_from_chunk(c, next_text_idx=next_text_idx)
        items.append(item)

    # Each item has a #/texts/N self_ref bumped from the starting value
    for i, item in enumerate(items):
        assert item["self_ref"] == f"#/texts/{100 + i}"

    # Bridge contains entries for each chunk that had non-empty cell_refs
    for i, c in enumerate(chunks):
        if c.cell_refs:
            assert bridge.cell_refs_for_text_idx(100 + i) == list(c.cell_refs)
        else:
            assert bridge.cell_refs_for_text_idx(100 + i) == []


def test_text_item_empty_prov_for_other_table():
    """Shape.OTHER chunks have empty cell_refs; bridge not populated for those entries."""
    nt = normalize_tables({"tables": [{"table_cells": [], "text": "raw"}], "texts": []})[0]
    chunks = render_for_graph(nt, token_limit_whole=1500, token_limit_column=1200)

    bridge.reset()
    item, _ = _text_item_from_chunk(chunks[0], next_text_idx=42)
    assert bridge.cell_refs_for_text_idx(42) == []


def test_text_item_dict_shape_matches_docling_convention():
    """Returned dict has self_ref, label, prov (empty list), orig, text."""
    sa2 = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())
    nt = normalize_tables({"tables": [sa2], "texts": []})[0]
    chunks = render_for_graph(nt, token_limit_whole=100, token_limit_column=100)

    bridge.reset()
    item, _ = _text_item_from_chunk(chunks[0], next_text_idx=5)
    assert set(item.keys()) == {"self_ref", "label", "prov", "orig", "text"}
    assert item["label"] == "text"
    assert item["prov"] == []
    assert item["orig"] == item["text"]
    assert item["text"] == chunks[0].text


def test_caller_threads_next_text_idx_correctly():
    """Returned next_text_idx is the input + 1; caller threads through."""
    sa2 = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())
    nt = normalize_tables({"tables": [sa2], "texts": []})[0]
    chunks = render_for_graph(nt, token_limit_whole=100, token_limit_column=100)

    bridge.reset()
    next_text_idx = 200
    for c in chunks:
        item, returned_idx = _text_item_from_chunk(c, next_text_idx=next_text_idx)
        assert returned_idx == next_text_idx + 1
        next_text_idx = returned_idx
