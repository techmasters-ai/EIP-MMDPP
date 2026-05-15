import json
from pathlib import Path
from app.services.table_normalization import normalize_tables, render_for_embedding
from app.services.table_normalization._pipeline_hooks import _substitute_table_chunks


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def _fake_native_chunk(text, self_refs, headings=()):
    """Build a duck-typed object matching what HybridChunker emits."""
    class _Prov:
        def __init__(self, page_no): self.page_no = page_no

    class _Item:
        def __init__(self, self_ref):
            self.self_ref = self_ref
            self.prov = [_Prov(1)]

    class _Meta:
        def __init__(self, items, headings):
            self.doc_items = items
            self.headings = list(headings)

    class _Chunk:
        def __init__(self, text, items, headings):
            self.text = text
            self.meta = _Meta(items, headings)

    items = [_Item(ref) for ref in self_refs]
    return _Chunk(text, items, headings)


def test_table_dominant_substitution():
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    normalized_by_table_idx = {nt.table_index: nt}
    native = _fake_native_chunk("raw table text", ["#/tables/0", "#/tables/0", "#/tables/0"])
    out = _substitute_table_chunks(
        [native], normalized_by_table_idx, render_for_embedding,
        token_limit=512, summary_limit=300, min_table_tokens=0,
    )
    assert all(getattr(c, "text", "") != "raw table text" for c in out)
    assert len(out) > 0


def test_non_table_passthrough():
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    normalized_by_table_idx = {nt.table_index: nt}
    native = _fake_native_chunk("prose", ["#/texts/100"])
    out = _substitute_table_chunks(
        [native], normalized_by_table_idx, render_for_embedding,
        token_limit=512, summary_limit=300, min_table_tokens=0,
    )
    assert out == [native]


def test_small_table_passes_through():
    """Tables below MIN_TABLE_NORMALIZATION_TOKENS pass through unchanged."""
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    normalized_by_table_idx = {nt.table_index: nt}
    native = _fake_native_chunk("small table text", ["#/tables/0"])
    out = _substitute_table_chunks(
        [native], normalized_by_table_idx, render_for_embedding,
        token_limit=512, summary_limit=300, min_table_tokens=999999,
    )
    assert out == [native]


def test_subsequent_native_chunks_for_same_table_dropped():
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    normalized_by_table_idx = {nt.table_index: nt}
    n1 = _fake_native_chunk("first", ["#/tables/0", "#/tables/0"])
    n2 = _fake_native_chunk("second", ["#/tables/0"])
    out = _substitute_table_chunks(
        [n1, n2], normalized_by_table_idx, render_for_embedding,
        token_limit=512, summary_limit=300, min_table_tokens=0,
    )
    assert all(getattr(c, "text", "") not in ("first", "second") for c in out)


def test_content_coverage_assertion():
    """Union of substituted chunks' cell_refs covers all cells of the normalized table."""
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    normalized_by_table_idx = {nt.table_index: nt}
    native = _fake_native_chunk("dominant", ["#/tables/0", "#/tables/0", "#/tables/0"])
    out = _substitute_table_chunks(
        [native], normalized_by_table_idx, render_for_embedding,
        token_limit=512, summary_limit=300, min_table_tokens=0,
    )
    collected_refs: set[str] = set()
    for c in out:
        em = getattr(c, "extra_metadata", None) or {}
        for ref in em.get("cell_refs", []):
            collected_refs.add(ref)
    expected_refs = {c.cell_ref.self_ref for c in nt.cells}
    assert expected_refs.issubset(collected_refs), \
        f"missing cells: {expected_refs - collected_refs}"
