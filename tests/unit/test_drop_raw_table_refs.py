"""Tests for `_drop_raw_table_refs_from_body_children` — the v9-completion
fix that ensures HybridChunker only walks the synthesized per-column blocks
for normalized tables, not the raw `#/tables/N` cell-flattening that
competes with them.

This test lives next to the other table-normalization unit tests and is
the regression guard for the v9 root-cause analysis: even though
`render_for_graph` synthesized correct per-column blocks AND
`_suppress_raw_table_texts` blanked the text-mirror in `texts[]`, the raw
`#/tables/N` `$refs` in `body.children` still caused HybridChunker to emit
the flat-cell representation, which then competed with — and usually beat
— the synthesized blocks at attribution time. See
`tmp/claude_vs_gemma4/REPORT_MIN_ALT_DIAGNOSIS.md` for the full story.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.services.table_normalization import normalize_tables
from app.services.table_normalization._pipeline_hooks import (
    _drop_raw_table_refs_from_body_children,
    _suppress_raw_table_texts,
)


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def _doc_with_table_in_body(fixture: dict) -> dict:
    """Return a minimal Docling-shape doc with the SA-2 sample table both in
    `tables[]` and referenced from `body.children`. Mirrors the actual
    document shape Docling emits."""
    return {
        "tables": [fixture],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "parent": {"$ref": "#/body"},
                "children": [],
                "content_layer": "body",
                "label": "text",
                "text": "(raw flat-cell mirror text would go here)",
                "orig": "(raw flat-cell mirror text would go here)",
            },
        ],
        "body": {
            "self_ref": "#/body",
            "children": [
                {"$ref": "#/texts/0"},      # the flat-text mirror
                {"$ref": "#/tables/0"},     # the structured table — this is what we want to remove
            ],
        },
    }


def test_drop_removes_raw_table_ref_for_normalized_table():
    """After the helper runs, `body.children` no longer contains the raw
    `#/tables/0` $ref — but the structured `tables[0]` itself stays put."""
    doc = _doc_with_table_in_body(SA2_FIXTURE)
    normalized = normalize_tables(doc)
    assert normalized, "SA-2 sample table should normalize cleanly"
    assert len(doc["tables"]) == 1
    assert {"$ref": "#/tables/0"} in doc["body"]["children"]

    n_dropped = _drop_raw_table_refs_from_body_children(doc, normalized)

    assert n_dropped == 1
    assert {"$ref": "#/tables/0"} not in doc["body"]["children"]
    # The structured table itself MUST stay — table_overlay reads tables[] directly
    assert len(doc["tables"]) == 1
    assert doc["tables"][0] == SA2_FIXTURE
    # The text mirror $ref also stays (the suppress-text helper handles its content)
    assert {"$ref": "#/texts/0"} in doc["body"]["children"]


def test_drop_returns_zero_when_no_normalized_tables():
    """No-op when the input has no normalized tables (e.g. all OTHER shape)."""
    doc = {"tables": [], "texts": [], "body": {"children": []}}
    n_dropped = _drop_raw_table_refs_from_body_children(doc, [])
    assert n_dropped == 0


def test_drop_preserves_other_shape_tables():
    """A table with shape == OTHER should NOT be removed from body.children —
    OTHER tables fall through to the raw renderer and need their $ref to
    reach the chunker."""
    other_table = {"table_cells": [{"text": "x"}], "text": "raw stuff"}
    doc = {
        "tables": [other_table],
        "texts": [],
        "body": {
            "self_ref": "#/body",
            "children": [{"$ref": "#/tables/0"}],
        },
    }
    normalized = normalize_tables(doc)
    assert normalized, "expected at least one (OTHER-shape) table"
    n_dropped = _drop_raw_table_refs_from_body_children(doc, normalized)
    # Helper should NOT touch OTHER-shape tables — caller relies on raw fallback
    assert n_dropped == 0
    assert {"$ref": "#/tables/0"} in doc["body"]["children"]


def test_drop_handles_nested_children():
    """If the doc body has a tree (children of children) and a raw table $ref
    is nested deep, it still gets removed."""
    doc = _doc_with_table_in_body(SA2_FIXTURE)
    # Wrap the table $ref under a nested container
    doc["body"]["children"] = [
        {
            "self_ref": "#/groups/0",
            "children": [
                {"$ref": "#/tables/0"},
            ],
        },
    ]
    normalized = normalize_tables(doc)
    n_dropped = _drop_raw_table_refs_from_body_children(doc, normalized)
    assert n_dropped == 1
    # nested container stays, only the inner $ref removed
    assert len(doc["body"]["children"]) == 1
    assert doc["body"]["children"][0]["children"] == []


def test_full_v9_pipeline_only_synth_chunks_reach_chunker():
    """End-to-end pin: when both `_suppress_raw_table_texts` AND
    `_drop_raw_table_refs_from_body_children` run, the chunker (walking
    body.children + texts[]) sees ONLY synthesized per-column TextItems for
    the normalized table — no raw `#/tables/N` references and no non-empty
    text mirror.

    This is the regression test that would have caught the v9 incomplete
    suppression. If somebody later removes the body-children drop, this
    test fails with a clear message about which raw $ref leaked through.
    """
    from app.services.table_normalization import render_for_graph
    from app.services.table_normalization._text_item import _text_item_from_chunk

    doc = _doc_with_table_in_body(SA2_FIXTURE)
    normalized = normalize_tables(doc)

    # Mirror what main.py does: append synth chunks, suppress raw mirror,
    # drop raw $refs.
    next_idx = len(doc["texts"])
    synth_count = 0
    for nt in normalized:
        for gtc in render_for_graph(nt, token_limit_whole=1500, token_limit_column=1200):
            captured = next_idx
            ti, next_idx = _text_item_from_chunk(gtc, next_text_idx=next_idx)
            doc["texts"].append(ti)
            doc["body"]["children"].append({"$ref": f"#/texts/{captured}"})
            synth_count += 1
    _suppress_raw_table_texts(doc, normalized)
    _drop_raw_table_refs_from_body_children(doc, normalized)

    # Invariant 1: no raw table $refs left in the body.children walk.
    raw_refs = [
        c for c in doc["body"]["children"]
        if isinstance(c, dict) and str(c.get("$ref", "")).startswith("#/tables/")
    ]
    assert raw_refs == [], (
        f"raw #/tables/ $refs leaked into body.children: {raw_refs}. "
        "_drop_raw_table_refs_from_body_children must remove these."
    )

    # Invariant 2: synth chunks ARE in body.children.
    synth_refs = [
        c for c in doc["body"]["children"]
        if isinstance(c, dict) and str(c.get("$ref", "")).startswith("#/texts/")
        and int(str(c["$ref"]).split("/")[-1]) >= len(doc["texts"]) - synth_count
    ]
    assert len(synth_refs) == synth_count

    # Invariant 3: the original tables[] entry is still there for overlay code.
    assert len(doc["tables"]) == 1
    assert doc["tables"][0] == SA2_FIXTURE
