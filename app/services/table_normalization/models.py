"""Frozen dataclasses and enums for the table normalization layer.

The NormalizedTable model is the only contract between normalization
(normalize.py) and the renderers (render_graph.py, render_embedding.py).
All types are frozen — immutable post-construction — so renderers can't
mutate state across calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Shape(str, Enum):
    COLUMN_MAJOR = "column_major"
    ROW_MAJOR = "row_major"
    HYBRID = "hybrid"
    OTHER = "other"


class ChunkKind(str, Enum):
    TABLE_SUMMARY = "table_summary"
    TABLE_WHOLE = "table_whole"
    TABLE_ENTITY_COLUMN = "table_entity_column"
    TABLE_ENTITY_SECTION = "table_entity_section"


@dataclass(frozen=True)
class CellRef:
    table_index: int
    row_idx: int
    col_idx: int
    self_ref: str


@dataclass(frozen=True)
class NormalizedCell:
    row_idx: int
    col_idx: int
    row_label: str | None
    column_identity: dict[str, str]
    section: str | None
    value: str
    unit: str | None
    cell_ref: CellRef


@dataclass(frozen=True)
class NormalizedRow:
    row_idx: int
    label: str
    is_identity_row: bool
    is_section_header: bool
    section: str | None
    unit: str | None


@dataclass(frozen=True)
class NormalizedColumn:
    col_idx: int
    identity: dict[str, str]
    display_name: str


@dataclass(frozen=True)
class TableSection:
    name: str
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class NormalizedTable:
    table_index: int
    self_ref: str
    caption: str | None
    page_numbers: tuple[int, ...]
    shape: Shape
    rows: tuple[NormalizedRow, ...]
    columns: tuple[NormalizedColumn, ...]
    sections: tuple[TableSection, ...]
    cells: tuple[NormalizedCell, ...]
    raw_markdown: str


@dataclass(frozen=True)
class GraphTableChunk:
    text: str
    table_ref: str
    page_numbers: tuple[int, ...]
    chunk_kind: ChunkKind
    entity_display_name: str | None
    section: str | None
    column_index: int | None
    cell_refs: tuple[str, ...]
    row_labels: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingTableChunk:
    text: str
    table_ref: str
    page_numbers: tuple[int, ...]
    chunk_kind: ChunkKind
    entity_display_name: str | None
    section: str | None
    column_index: int | None
    cell_refs: tuple[str, ...]
    row_labels: tuple[str, ...]
