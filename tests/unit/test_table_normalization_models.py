import pytest
from app.services.table_normalization.models import (
    Shape, ChunkKind, CellRef, NormalizedCell, NormalizedRow,
    NormalizedColumn, TableSection, NormalizedTable,
    GraphTableChunk, EmbeddingTableChunk,
)


def test_shape_enum_values():
    assert Shape.COLUMN_MAJOR.value == "column_major"
    assert Shape.ROW_MAJOR.value == "row_major"
    assert Shape.HYBRID.value == "hybrid"
    assert Shape.OTHER.value == "other"


def test_chunk_kind_enum_values():
    assert ChunkKind.TABLE_SUMMARY.value == "table_summary"
    assert ChunkKind.TABLE_WHOLE.value == "table_whole"
    assert ChunkKind.TABLE_ENTITY_COLUMN.value == "table_entity_column"
    assert ChunkKind.TABLE_ENTITY_SECTION.value == "table_entity_section"


def test_normalized_table_is_frozen():
    nt = NormalizedTable(
        table_index=0, self_ref="#/tables/0", caption=None,
        page_numbers=(1,), shape=Shape.OTHER, rows=(), columns=(),
        sections=(), cells=(), raw_markdown="",
    )
    with pytest.raises((AttributeError, Exception)):  # frozen dataclass raises FrozenInstanceError
        nt.caption = "mutated"


def test_cell_ref_self_ref_format():
    cr = CellRef(table_index=3, row_idx=5, col_idx=2, self_ref="#/tables/3/data/table_cells/17")
    assert cr.self_ref.startswith("#/tables/")
    assert cr.self_ref.endswith("/17")


def test_graph_table_chunk_carries_chunk_kind():
    gtc = GraphTableChunk(
        text="...", table_ref="#/tables/3", page_numbers=(6,),
        chunk_kind=ChunkKind.TABLE_ENTITY_COLUMN,
        entity_display_name="S-75M2", section=None, column_index=7,
        cell_refs=("#/tables/3/data/table_cells/42",),
        row_labels=("Max Range",),
    )
    assert gtc.chunk_kind == ChunkKind.TABLE_ENTITY_COLUMN
    assert "S-75M2" in gtc.entity_display_name
