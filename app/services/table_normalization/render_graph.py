"""Graph-side renderer. Also exports _render_column_as_text as the
shared column-rendering helper used by both renderers (single source
of truth for chunk text format).

`render_for_graph` is the public entry point — emits one
GraphTableChunk per entity column (or per (column, section) when a
column overflows its budget), or a single TABLE_WHOLE chunk for small
tables and Shape.OTHER passthroughs. See §9 of the design spec.
"""
from __future__ import annotations

from app.services.table_normalization.models import (
    NormalizedTable, NormalizedColumn, TableSection,
    GraphTableChunk, ChunkKind, Shape,
)
from app.services.table_normalization.tokens import count_bge_m3_tokens


# Hint emitted at the top of every synthesized table block. The Unit Policy in
# the system prompt explicitly honors preambles declaring "the applicable unit"
# for unitless numerics — this line is that preamble for spec-sheet tables
# that omit a unit row (common in SAM/radar reference documents).
UNIT_HINT = (
    "UNITS: Numeric values in this block are in SI base units "
    "(metres for length/range, kilograms for mass, m/s for speed, "
    "seconds for time, degrees for angle, MHz for frequency) "
    "unless a value is explicitly labeled with another unit."
)


def _render_column_as_text(
    column: NormalizedColumn,
    table: NormalizedTable,
    sections: tuple[TableSection, ...],
    *,
    emit_unit_hint: bool = False,
) -> str:
    """Produce the identity+sections+rows block for one entity column.

    Both the graph and embedding renderers call this helper; their outputs
    differ only by what they wrap around this block. See §9 of the spec
    for the exact text format. The graph-side caller passes
    emit_unit_hint=True so the LLM extraction prompt sees the SI-base unit
    preamble for tables whose source rows omit explicit units.
    """
    parts: list[str] = []

    # TABLE header
    caption = table.caption or table.self_ref
    parts.append(f"TABLE: {caption}")
    if table.page_numbers:
        parts.append(f"SOURCE: page {' '.join(str(p) for p in table.page_numbers)}")
    if emit_unit_hint:
        parts.append(UNIT_HINT)
    parts.append("")

    # ENTITY block — full identity dict
    parts.append("ENTITY:")
    for k, v in column.identity.items():
        parts.append(f"- {k}: {v}")
    parts.append("")

    # Section blocks
    spec_cells_by_section: dict[str | None, list[tuple[str, str, str | None]]] = {}
    for cell in table.cells:
        if cell.col_idx != column.col_idx:
            continue
        bucket = cell.section
        spec_cells_by_section.setdefault(bucket, []).append(
            (cell.row_label or "", cell.value, cell.unit)
        )

    # Render GENERAL first, then named sections in document order
    if None in spec_cells_by_section:
        parts.append("GENERAL:")
        for label, value, unit in spec_cells_by_section[None]:
            parts.append(_render_row_line(label, value, unit))
        parts.append("")

    for section in sections:
        rows_for_section = spec_cells_by_section.get(section.name)
        if not rows_for_section:
            continue
        parts.append(f"{section.name.upper()}:")
        for label, value, unit in rows_for_section:
            parts.append(_render_row_line(label, value, unit))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _render_row_line(label: str, value: str, unit: str | None) -> str:
    if unit:
        return f"- {label}: {value} {unit}"
    return f"- {label}: {value}"


def render_for_graph(
    table: NormalizedTable,
    token_limit_whole: int,
    token_limit_column: int,
) -> list[GraphTableChunk]:
    """Render a NormalizedTable into graph-side chunks per §9 of the spec.

    Decision tree:
      1. Shape.OTHER → one TABLE_WHOLE chunk carrying raw_markdown.
      2. Whole-table render fits within `token_limit_whole` → one TABLE_WHOLE.
      3. Otherwise emit one TABLE_ENTITY_COLUMN per column, splitting any
         column that exceeds `token_limit_column` into per-section
         TABLE_ENTITY_SECTION chunks (identity header repeated).
    """
    # 1. Shape.OTHER → raw_markdown passthrough
    if table.shape == Shape.OTHER:
        return [GraphTableChunk(
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

    # 2. Whole-table rendering check
    whole_text = _render_whole_table(table, emit_unit_hint=True)
    whole_tokens = count_bge_m3_tokens(whole_text)
    if whole_tokens <= token_limit_whole:
        return [GraphTableChunk(
            text=whole_text,
            table_ref=table.self_ref,
            page_numbers=table.page_numbers,
            chunk_kind=ChunkKind.TABLE_WHOLE,
            entity_display_name=None,
            section=None,
            column_index=None,
            cell_refs=tuple(c.cell_ref.self_ref for c in table.cells),
            row_labels=tuple(sorted({c.row_label or "" for c in table.cells})),
        )]

    # 3. Per-column emission
    out: list[GraphTableChunk] = []
    for col in table.columns:
        col_text = _render_column_as_text(col, table, table.sections, emit_unit_hint=True)
        col_tokens = count_bge_m3_tokens(col_text)
        col_cells = [c for c in table.cells if c.col_idx == col.col_idx]
        col_refs = tuple(c.cell_ref.self_ref for c in col_cells)
        col_row_labels = tuple(sorted({c.row_label or "" for c in col_cells}))

        if col_tokens <= token_limit_column:
            out.append(GraphTableChunk(
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
            # Column overflows budget → split by section; identity header repeats
            for section in table.sections:
                sec_cells = [c for c in col_cells if c.section == section.name]
                if not sec_cells:
                    continue
                sec_text = _render_column_section(col, table, section, emit_unit_hint=True)
                out.append(GraphTableChunk(
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


def _render_whole_table(table: NormalizedTable, *, emit_unit_hint: bool = False) -> str:
    """Whole-table rendering: identity header + each column block stacked."""
    parts: list[str] = []
    caption = table.caption or table.self_ref
    parts.append(f"TABLE: {caption}")
    if table.page_numbers:
        parts.append(f"SOURCE: page {' '.join(str(p) for p in table.page_numbers)}")
    if emit_unit_hint:
        parts.append(UNIT_HINT)
    parts.append("")
    for col in table.columns:
        parts.append(_render_column_as_text(col, table, table.sections, emit_unit_hint=False).rstrip())
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_column_section(
    column: NormalizedColumn,
    table: NormalizedTable,
    section: TableSection,
    *,
    emit_unit_hint: bool = False,
) -> str:
    """Single section of one column with identity header repeated.

    Used when a per-column render exceeds the column token budget — we
    split per section so each chunk still carries the full entity
    identity needed to interpret the values."""
    parts: list[str] = []
    caption = table.caption or table.self_ref
    parts.append(f"TABLE: {caption}")
    if table.page_numbers:
        parts.append(f"SOURCE: page {' '.join(str(p) for p in table.page_numbers)}")
    if emit_unit_hint:
        parts.append(UNIT_HINT)
    parts.append("")
    parts.append("ENTITY:")
    for k, v in column.identity.items():
        parts.append(f"- {k}: {v}")
    parts.append("")
    parts.append(f"{section.name.upper()}:")
    for c in table.cells:
        if c.col_idx != column.col_idx or c.section != section.name:
            continue
        parts.append(_render_row_line(c.row_label or "", c.value, c.unit))
    return "\n".join(parts).rstrip() + "\n"
