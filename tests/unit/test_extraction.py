"""Unit tests for document extraction utilities."""

import pytest

from app.services.extraction import chunk_text, _table_to_text


pytestmark = pytest.mark.unit


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "This is a short sentence."
        chunks = chunk_text(text, max_tokens=512)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_long_text_is_split(self):
        # 600 words — should split at max_words ≈ 384 (512 * 0.75)
        words = ["word"] * 600
        text = " ".join(words)
        chunks = chunk_text(text, max_tokens=512, overlap_tokens=64)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        words = [f"w{i}" for i in range(500)]
        text = " ".join(words)
        chunks = chunk_text(text, max_tokens=200, overlap_tokens=40)
        # Each chunk should start before the end of the previous chunk
        assert len(chunks) >= 2
        # With overlap_tokens=40 (≈30 words at 0.75 ratio), the last 30 words
        # of chunk[0] and the first 30 words of chunk[1] must intersect.
        overlap_words = int(40 * 0.75) + 5  # ≈30 + safety margin
        first_end_words = set(chunks[0].split()[-overlap_words:])
        second_start_words = set(chunks[1].split()[:overlap_words])
        assert first_end_words & second_start_words, (
            f"Expected overlap between end of chunk[0] and start of chunk[1]. "
            f"End words: {list(first_end_words)[:5]}..., "
            f"Start words: {list(second_start_words)[:5]}..."
        )

    def test_single_word_text(self):
        chunks = chunk_text("hello")
        assert chunks == ["hello"]


class TestTableToText:
    def test_simple_table(self):
        table = [["Part Number", "Description", "Qty"], ["PN-001", "Screw M4", "10"]]
        text = _table_to_text(table)
        assert "Part Number" in text
        assert "PN-001" in text
        assert "|" in text

    def test_empty_table(self):
        assert _table_to_text([]) == ""
        assert _table_to_text([[]]) == ""

    def test_none_cells_handled(self):
        table = [["A", None, "C"]]
        text = _table_to_text(table)
        assert "A" in text
        assert "C" in text
