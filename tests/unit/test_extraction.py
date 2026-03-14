"""Unit tests for document extraction utilities."""

import pytest

from app.services.extraction import _table_to_text


pytestmark = pytest.mark.unit


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
