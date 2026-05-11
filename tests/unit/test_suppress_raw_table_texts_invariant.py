import json
from pathlib import Path
from app.services.table_normalization import normalize_tables
from app.services.table_normalization._pipeline_hooks import _suppress_raw_table_texts


def test_blanks_in_place_does_not_remove_entries():
    doc = {
        "tables": [{"table_cells": [], "text": "raw"}],  # OTHER table — not suppressed
        "texts": [
            {"self_ref": "#/texts/0", "text": "prose 1"},
            {"self_ref": "#/texts/1", "text": "prose 2", "orig": "prose 2 orig"},
        ],
    }
    normalized = normalize_tables(doc)
    initial_len = len(doc["texts"])
    _suppress_raw_table_texts(doc, normalized)
    assert len(doc["texts"]) == initial_len


def test_blanks_only_target_self_refs():
    sa2 = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())
    doc = {
        "tables": [sa2],
        "texts": [
            {"self_ref": "#/texts/0", "text": "prose", "orig": "prose"},
            {"self_ref": "#/tables/0", "text": "the flattened table text", "orig": "the flattened table text"},
            {"self_ref": "#/texts/1", "text": "more prose", "orig": "more prose"},
        ],
    }
    normalized = normalize_tables(doc)
    _suppress_raw_table_texts(doc, normalized)

    # texts[0] and texts[2] (prose) unchanged
    assert doc["texts"][0]["text"] == "prose"
    assert doc["texts"][2]["text"] == "more prose"

    # texts[1] (the table mirror) blanked
    assert doc["texts"][1]["text"] == ""
    assert doc["texts"][1]["orig"] == ""

    # All entries still present (no reindexing)
    assert len(doc["texts"]) == 3


def test_tables_array_not_mutated():
    """CRITICAL: doc_json['tables'] is byte-identical pre/post."""
    sa2 = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())
    doc = {
        "tables": [sa2],
        "texts": [{"self_ref": "#/tables/0", "text": "x", "orig": "x"}],
    }
    before = json.dumps(doc["tables"], sort_keys=True)
    normalized = normalize_tables(doc)
    _suppress_raw_table_texts(doc, normalized)
    after = json.dumps(doc["tables"], sort_keys=True)
    assert before == after, "tables[] was mutated — overlay path will break"


def test_other_shape_tables_preserved():
    """Tables with Shape.OTHER keep their flat-text mirror."""
    doc = {
        "tables": [{"table_cells": [], "text": "raw"}],
        "texts": [{"self_ref": "#/tables/0", "text": "raw flat", "orig": "raw flat"}],
    }
    normalized = normalize_tables(doc)
    _suppress_raw_table_texts(doc, normalized)
    # OTHER table → not suppressed; flat text preserved
    assert doc["texts"][0]["text"] == "raw flat"
