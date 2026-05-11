import json
from pathlib import Path
from app.services.table_normalization import normalize_tables
from app.services.table_normalization._pipeline_hooks import _normalized_table_size_tokens


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def test_size_function_returns_positive_for_real_table():
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    n = _normalized_table_size_tokens(nt)
    assert n > 0


def test_size_function_sums_column_renderings():
    """Canonical contract per spec rev. 7 §10.1: sum across columns."""
    from app.services.table_normalization.tokens import count_bge_m3_tokens
    from app.services.table_normalization.render_graph import _render_column_as_text

    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    expected = sum(
        count_bge_m3_tokens(_render_column_as_text(col, nt, nt.sections))
        for col in nt.columns
    )
    assert _normalized_table_size_tokens(nt) == expected
