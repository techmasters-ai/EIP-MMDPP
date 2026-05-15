import json
from pathlib import Path
from app.services.table_normalization import normalize_tables


def test_render_column_as_text_returns_expected_format():
    """Snapshot test: SA-2 column 1 renders to a string with required format markers."""
    from app.services.table_normalization.render_graph import _render_column_as_text

    fixture = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())
    doc = {"tables": [fixture], "texts": []}
    nt = normalize_tables(doc)[0]
    assert len(nt.columns) >= 1

    text = _render_column_as_text(nt.columns[0], nt, nt.sections)

    assert "TABLE:" in text
    assert "ENTITY:" in text
    for s in nt.sections:
        assert s.name.upper() in text


def test_render_column_byte_identical_across_renderers():
    """Both renderers must produce byte-identical output for the same column."""
    from app.services.table_normalization.render_graph import _render_column_as_text as render_g
    from app.services.table_normalization.render_embedding import _render_column_as_text as render_e

    assert render_g is render_e  # same function object — single source of truth
