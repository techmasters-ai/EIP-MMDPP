import json
from pathlib import Path
from app.services.table_normalization import normalize_tables, render_for_embedding
from app.services.table_normalization._pipeline_hooks import _NormalizedTableChunkAdapter


SA2_FIXTURE = json.loads(Path("tests/fixtures/sa2_sample_table.json").read_text())


def test_adapter_ducktypes_native_chunk_interface():
    """Adapter exposes .text, .meta.doc_items[].self_ref, .meta.doc_items[].prov[].page_no, .meta.headings."""
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    chunks = render_for_embedding(nt, token_limit=512, summary_limit=300)
    etc = chunks[0]
    adapter = _NormalizedTableChunkAdapter(
        etc=etc, parent_headings=("Chapter 4", "Variants"),
        parent_table_ref="#/tables/0",
    )

    assert adapter.text == etc.text
    assert hasattr(adapter, "meta")
    items = adapter.meta.doc_items
    assert len(items) == 1
    assert items[0].self_ref == "#/tables/0"  # NOT a cell ref — today-shape only


def test_adapter_preserves_parent_headings():
    """Section_path regression guard: native chunk's parent_headings preserved."""
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    chunks = render_for_embedding(nt, token_limit=512, summary_limit=300)
    etc = chunks[0]
    adapter = _NormalizedTableChunkAdapter(
        etc=etc, parent_headings=("Section A",),
        parent_table_ref="#/tables/0",
    )
    assert adapter.meta.headings == ("Section A",)


def test_adapter_exposes_extra_metadata():
    """extra_metadata carries chunk_kind + cell_refs for the metadata column."""
    nt = normalize_tables({"tables": [SA2_FIXTURE], "texts": []})[0]
    chunks = render_for_embedding(nt, token_limit=512, summary_limit=300)
    etc = chunks[0]
    adapter = _NormalizedTableChunkAdapter(
        etc=etc, parent_headings=(),
        parent_table_ref="#/tables/0",
    )
    em = adapter.extra_metadata
    assert em["chunk_kind"] == etc.chunk_kind.value
    assert "cell_refs" in em
    assert "row_labels" in em
