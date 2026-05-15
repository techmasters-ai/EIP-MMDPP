"""Embedding-side renderer.

Always emits a TABLE_SUMMARY chunk. Emits a TABLE_WHOLE when the table
fits within token_limit; otherwise emits per-entity-column chunks
(with section-split for oversized columns).
"""
from __future__ import annotations

from app.services.table_normalization.models import (
    NormalizedTable, NormalizedColumn, TableSection,
    EmbeddingTableChunk, ChunkKind, Shape,
)
from app.services.table_normalization.tokens import count_bge_m3_tokens
from app.services.table_normalization.render_graph import (
    _render_column_as_text,
    _render_whole_table,
    _render_column_section,
    _render_row_line,
)

__all__ = ["_render_column_as_text", "render_for_embedding"]


def render_for_embedding(
    table: NormalizedTable,
    token_limit: int,
    summary_limit: int,
) -> list[EmbeddingTableChunk]:
    """Per §10 of the spec."""
    out: list[EmbeddingTableChunk] = []

    # Shape.OTHER → one TABLE_WHOLE with raw_markdown
    if table.shape == Shape.OTHER:
        return [EmbeddingTableChunk(
            text=table.raw_markdown,
            table_ref=table.self_ref,
            page_numbers=table.page_numbers,
            chunk_kind=ChunkKind.TABLE_WHOLE,
            entity_display_name=None,
            section=None,
            column_index=None,
            cell_refs=(),
            row_labels=(),
        )]

    # Always emit summary
    summary_text = _render_summary(table, summary_limit)
    out.append(EmbeddingTableChunk(
        text=summary_text,
        table_ref=table.self_ref,
        page_numbers=table.page_numbers,
        chunk_kind=ChunkKind.TABLE_SUMMARY,
        entity_display_name=None,
        section=None,
        column_index=None,
        cell_refs=tuple(c.cell_ref.self_ref for c in table.cells),
        row_labels=tuple(sorted({c.row_label or "" for c in table.cells})),
    ))

    # Whole-table check
    whole_text = _render_whole_table(table)
    if count_bge_m3_tokens(whole_text) <= token_limit:
        out.append(EmbeddingTableChunk(
            text=whole_text,
            table_ref=table.self_ref,
            page_numbers=table.page_numbers,
            chunk_kind=ChunkKind.TABLE_WHOLE,
            entity_display_name=None,
            section=None,
            column_index=None,
            cell_refs=tuple(c.cell_ref.self_ref for c in table.cells),
            row_labels=tuple(sorted({c.row_label or "" for c in table.cells})),
        ))
        return out

    # Per-column emission (with section-split for oversized columns)
    for col in table.columns:
        col_text = _render_column_as_text(col, table, table.sections)
        col_cells = [c for c in table.cells if c.col_idx == col.col_idx]
        col_refs = tuple(c.cell_ref.self_ref for c in col_cells)
        col_row_labels = tuple(sorted({c.row_label or "" for c in col_cells}))

        if count_bge_m3_tokens(col_text) <= token_limit:
            out.append(EmbeddingTableChunk(
                text=col_text,
                table_ref=table.self_ref,
                page_numbers=table.page_numbers,
                chunk_kind=ChunkKind.TABLE_ENTITY_COLUMN,
                entity_display_name=col.display_name,
                section=None,
                column_index=col.col_idx,
                cell_refs=col_refs,
                row_labels=col_row_labels,
            ))
        else:
            for section in table.sections:
                sec_text = _render_column_section(col, table, section)
                sec_cells = [c for c in col_cells if c.section == section.name]
                if not sec_cells:
                    continue
                out.append(EmbeddingTableChunk(
                    text=sec_text,
                    table_ref=table.self_ref,
                    page_numbers=table.page_numbers,
                    chunk_kind=ChunkKind.TABLE_ENTITY_SECTION,
                    entity_display_name=col.display_name,
                    section=section.name,
                    column_index=col.col_idx,
                    cell_refs=tuple(c.cell_ref.self_ref for c in sec_cells),
                    row_labels=tuple(sorted({c.row_label or "" for c in sec_cells})),
                ))
    return out


def _render_summary(table: NormalizedTable, summary_limit: int) -> str:
    """Per §10 emission rule 2."""
    caption = table.caption or table.self_ref
    pages = " ".join(str(p) for p in table.page_numbers) if table.page_numbers else ""
    variants = ", ".join(c.display_name for c in table.columns)
    spec_labels = sorted({c.row_label for c in table.cells if c.row_label})
    props = ", ".join(spec_labels)

    def _build_text(v: str, p: str) -> str:
        parts = [f"TABLE: {caption}"]
        if pages:
            parts.append(f"SOURCE: page {pages}; ref {table.self_ref}")
        else:
            parts.append(f"SOURCE: ref {table.self_ref}")
        parts.append(f"VARIANTS: {v}")
        parts.append(f"PROPERTIES: {p}")
        return "\n".join(parts)

    text = _build_text(variants, props)

    # Truncate VARIANTS, then PROPERTIES, until under summary_limit
    safety = 0
    while count_bge_m3_tokens(text) > summary_limit and safety < 100:
        safety += 1
        # Try truncating variants first
        if ", " in variants and not variants.endswith("..."):
            variants = variants.rsplit(", ", 1)[0] + ", ..."
            text = _build_text(variants, props)
            continue
        # Then properties
        if ", " in props and not props.endswith("..."):
            props = props.rsplit(", ", 1)[0] + ", ..."
            text = _build_text(variants, props)
            continue
        break
    return text
