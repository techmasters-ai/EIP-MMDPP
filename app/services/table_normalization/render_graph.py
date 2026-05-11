"""Graph-side renderer. Also exports _render_column_as_text as the
shared column-rendering helper used by both renderers (single source
of truth for chunk text format).

Task 6 implements only _render_column_as_text + the helpers used by
Tasks 7/8. Task 7 will add render_for_graph and _render_whole_table /
_render_column_section.
"""
from __future__ import annotations

from app.services.table_normalization.models import (
    NormalizedTable, NormalizedColumn, TableSection,
)


def _render_column_as_text(
    column: NormalizedColumn,
    table: NormalizedTable,
    sections: tuple[TableSection, ...],
) -> str:
    """Produce the identity+sections+rows block for one entity column.

    Both the graph and embedding renderers call this helper; their outputs
    differ only by what they wrap around this block. See §9 of the spec
    for the exact text format.
    """
    parts: list[str] = []

    # TABLE header
    caption = table.caption or table.self_ref
    parts.append(f"TABLE: {caption}")
    if table.page_numbers:
        parts.append(f"SOURCE: page {' '.join(str(p) for p in table.page_numbers)}")
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
