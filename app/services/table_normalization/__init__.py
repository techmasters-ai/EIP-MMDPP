"""Table normalization layer — see docs/superpowers/specs/2026-05-11-table-aware-chunking-design.md."""
from app.services.table_normalization.models import (
    Shape, ChunkKind, CellRef, NormalizedCell, NormalizedRow,
    NormalizedColumn, TableSection, NormalizedTable,
    GraphTableChunk, EmbeddingTableChunk,
)

__all__ = [
    "Shape", "ChunkKind", "CellRef", "NormalizedCell", "NormalizedRow",
    "NormalizedColumn", "TableSection", "NormalizedTable",
    "GraphTableChunk", "EmbeddingTableChunk",
]
