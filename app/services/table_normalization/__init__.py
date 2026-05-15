"""Table normalization layer — see docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md."""
from app.services.table_normalization.models import (
    Shape, ChunkKind, CellRef, NormalizedCell, NormalizedRow,
    NormalizedColumn, TableSection, NormalizedTable,
    GraphTableChunk, EmbeddingTableChunk,
)
from app.services.table_normalization.normalize import normalize_tables
from app.services.table_normalization.render_graph import render_for_graph
from app.services.table_normalization.render_embedding import render_for_embedding

__all__ = [
    "Shape", "ChunkKind", "CellRef", "NormalizedCell", "NormalizedRow",
    "NormalizedColumn", "TableSection", "NormalizedTable",
    "GraphTableChunk", "EmbeddingTableChunk",
    "normalize_tables",
    "render_for_graph",
    "render_for_embedding",
]
