"""Tests for render_for_embedding — the embedding-side table renderer.

See §10 of docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md.
"""
import json
from pathlib import Path

import pytest

from app.services.table_normalization import normalize_tables, render_for_embedding
from app.services.table_normalization.models import ChunkKind


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def _doc(fixture):
    return {"tables": [fixture], "texts": []}


def test_always_emits_summary():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_embedding(nt, token_limit=512, summary_limit=300)
    summaries = [c for c in chunks if c.chunk_kind == ChunkKind.TABLE_SUMMARY]
    assert len(summaries) == 1


def test_small_table_emits_summary_plus_whole():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    # Generous limit ensures the table fits
    chunks = render_for_embedding(nt, token_limit=100000, summary_limit=300)
    kinds = [c.chunk_kind for c in chunks]
    assert ChunkKind.TABLE_SUMMARY in kinds
    assert ChunkKind.TABLE_WHOLE in kinds


def test_large_table_emits_summary_plus_columns():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    # Force splitting with a tiny limit
    chunks = render_for_embedding(nt, token_limit=10, summary_limit=300)
    kinds = {c.chunk_kind for c in chunks}
    assert ChunkKind.TABLE_SUMMARY in kinds
    assert ChunkKind.TABLE_WHOLE not in kinds


def test_summary_chunk_capped_at_summary_limit():
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_embedding(nt, token_limit=512, summary_limit=50)
    summary = next(c for c in chunks if c.chunk_kind == ChunkKind.TABLE_SUMMARY)
    from app.services.table_normalization.tokens import count_bge_m3_tokens
    # Slack for tokenizer variance; truncation should respect ~50
    assert count_bge_m3_tokens(summary.text) <= 70


def test_sa2_embedding_chunks_match_snapshot():
    expected = json.loads(Path("tests/fixtures/sa2_embedding_chunks_expected.json").read_text())
    nt = normalize_tables(_doc(SA2_FIXTURE))[0]
    chunks = render_for_embedding(nt, token_limit=512, summary_limit=300)
    actual = [
        {**c.__dict__, "chunk_kind": c.chunk_kind.value,
         "page_numbers": list(c.page_numbers), "cell_refs": list(c.cell_refs),
         "row_labels": list(c.row_labels)}
        for c in chunks
    ]
    assert actual == expected
