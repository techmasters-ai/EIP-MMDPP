"""Tests for render_for_graph — the graph-side table renderer.

See §9 of docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md.
"""
import json
from pathlib import Path

from app.services.table_normalization import normalize_tables, render_for_graph
from app.services.table_normalization.models import ChunkKind


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def _doc(fixture):
    return {"tables": [fixture], "texts": []}


def test_other_table_emits_one_table_whole_chunk():
    nt_list = normalize_tables(_doc({"table_cells": [{"text": "x"}], "text": "raw stuff"}))
    chunks = render_for_graph(nt_list[0], token_limit_whole=1500, token_limit_column=1200)
    assert len(chunks) == 1
    assert chunks[0].chunk_kind == ChunkKind.TABLE_WHOLE
    # text should be raw_markdown (just "raw stuff") or fallback containing it
    assert "raw stuff" in chunks[0].text or chunks[0].text == "raw stuff"


def test_sa2_emits_one_chunk_per_column():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    # token_limit_whole=100 forces below the SA-2 whole-table render (~270 tok),
    # so we hit the per-column path. token_limit_column=1200 keeps each column intact.
    chunks = render_for_graph(nt, token_limit_whole=100, token_limit_column=1200)
    column_chunks = [c for c in chunks if c.chunk_kind == ChunkKind.TABLE_ENTITY_COLUMN]
    section_chunks = [c for c in chunks if c.chunk_kind == ChunkKind.TABLE_ENTITY_SECTION]
    # Either pattern; total chunks >= #columns when columns split, == #columns when not
    assert len(column_chunks) + len(section_chunks) >= len(nt.columns)


def test_chunk_cell_refs_point_into_source_table():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    # Force per-column path so we have TABLE_ENTITY_COLUMN chunks to check.
    chunks = render_for_graph(nt, token_limit_whole=100, token_limit_column=1200)
    for c in chunks:
        if c.chunk_kind in (ChunkKind.TABLE_ENTITY_COLUMN, ChunkKind.TABLE_ENTITY_SECTION):
            assert all(
                ref.startswith(f"#/tables/{nt.table_index}/data/table_cells/")
                for ref in c.cell_refs
            )


def test_small_table_below_whole_limit_emits_one_chunk():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_graph(nt, token_limit_whole=100000, token_limit_column=1200)
    table_whole = [c for c in chunks if c.chunk_kind == ChunkKind.TABLE_WHOLE]
    assert len(table_whole) == 1


def test_chunk_format_contains_entity_block():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    # Force per-column path; per-entity ENTITY: block is only in column/section chunks.
    chunks = render_for_graph(nt, token_limit_whole=100, token_limit_column=1200)
    for c in chunks:
        if c.chunk_kind == ChunkKind.TABLE_ENTITY_COLUMN:
            assert "ENTITY:" in c.text
            assert "TABLE:" in c.text


def test_sa2_graph_chunks_match_snapshot():
    expected = json.loads(Path("tests/fixtures/sa2_graph_chunks_expected.json").read_text())
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_graph(nt, token_limit_whole=1500, token_limit_column=1200)
    actual = [
        {
            **c.__dict__,
            "chunk_kind": c.chunk_kind.value,
            "page_numbers": list(c.page_numbers),
            "cell_refs": list(c.cell_refs),
            "row_labels": list(c.row_labels),
        }
        for c in chunks
    ]
    assert actual == expected
