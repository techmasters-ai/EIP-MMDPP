"""Normalize Docling tables into the shared NormalizedTable model.

Pure function: reads doc_json['tables']; never writes doc_json.
Per-table exceptions are caught; one bad table doesn't break others.
See §8 of the design spec."""
from __future__ import annotations

import logging
import re
from typing import Any
from app.services.table_normalization.detect import (
    detect_shape, IDENTITY_LABEL_KEYWORDS, SECTION_KEYWORDS, SPEC_ROW_KEYWORDS,
)
from app.services.table_normalization.models import (
    Shape, CellRef, NormalizedCell, NormalizedRow, NormalizedColumn,
    TableSection, NormalizedTable,
)

logger = logging.getLogger(__name__)


_UNIT_RE = re.compile(r"\(\s*([a-zA-Z/°²³]+)\s*\)\s*$")
_DISPLAY_NAME_PREFERENCE = (
    "industry designation", "military designation",
    "nato designation", "missile type",
)


# 2026-05-16 (Option C): per-row unit detection from cell values.
# Some tables encode units as a dedicated cell per row (typically the
# column adjacent to the row label) rather than in the label itself or
# inferred from context. Detecting this here lets the synth renderer
# emit per-row units even when `_extract_unit(label)` returns None.
#
# Vocabulary is intentionally narrow — we only match well-known SI and
# imperial units. Free-text values like "metres" are deliberately excluded
# to avoid false positives on words that share spellings with unit names.
_KNOWN_UNIT_TOKENS: frozenset[str] = frozenset({
    # distance — SI
    "m", "km", "mm", "cm", "µm", "um", "nm",
    # distance — imperial
    "ft", "in", "inch", "inches", "yd", "yard", "yards",
    "nmi", "nautical", "mi", "mile", "miles",
    # mass — SI
    "kg", "g", "mg", "t", "tonne", "tonnes",
    # mass — imperial
    "lb", "lbs", "oz", "ton", "tons",
    # time
    "s", "sec", "ms", "min", "h", "hr", "hrs",
    "μs", "us",
    # speed
    "m/s", "km/h", "kph", "kt", "kts", "knot", "knots", "mph",
    # angle
    "deg", "rad", "mil", "°",
    # frequency
    "hz", "khz", "mhz", "ghz",
    # power
    "w", "kw", "mw", "dbw", "dbm",
    # gain / loss / ratio
    "db", "dbi", "dbd",
    # thrust / force
    "n", "kn",
    # pressure (occasionally in propellant specs)
    "pa", "kpa", "mpa", "psi", "bar",
})


def _detect_row_unit_from_cells(row_idx: int, cells: list[dict]) -> str | None:
    """Scan cells in row `row_idx` (skipping the label column) for a value
    matching a known unit token. Returns the original-cased token, or None.

    This catches SAM-spec table conventions where col-1 of each numeric
    row contains the unit ('m' for distances, 'm/s' for speeds, 'kg' for
    masses, etc.) rather than embedding the unit in the row label.
    """
    for c in cells:
        if c.get("start_row_offset_idx") != row_idx:
            continue
        if c.get("start_col_offset_idx") == 0:
            continue  # the label column is not a unit cell
        text = (c.get("text") or "").strip()
        if not text:
            continue
        # short tokens only — multi-word cells are values, not unit markers
        if len(text) > 8:
            continue
        if text.lower() in _KNOWN_UNIT_TOKENS:
            return text
    return None


def normalize_tables(doc_json: dict) -> list[NormalizedTable]:
    """Public entry. Returns one NormalizedTable per doc_json['tables'] entry."""
    tables = (doc_json or {}).get("tables") or []
    return [_per_table_safe(doc_json, i) for i in range(len(tables))]


def _per_table_safe(doc_json: dict, table_index: int) -> NormalizedTable:
    try:
        return _per_table(doc_json, table_index)
    except Exception as exc:
        logger.warning(
            "table_normalization.normalize: table %d failed (%s); returning OTHER",
            table_index, exc,
        )
        return _empty_normalized(doc_json, table_index)


def _per_table(doc_json: dict, table_index: int) -> NormalizedTable:
    table = doc_json["tables"][table_index]
    if not isinstance(table, dict):
        raise ValueError(f"table {table_index} is not a dict")
    cells = (table.get("data") or {}).get("table_cells") or table.get("table_cells") or []
    raw_md = _resolve_raw_markdown(doc_json, table_index, table)
    page_numbers = _resolve_page_numbers(table)
    caption = table.get("caption") or (table.get("data") or {}).get("caption")
    self_ref = f"#/tables/{table_index}"

    shape = detect_shape(cells, table)
    if shape == Shape.OTHER:
        return NormalizedTable(
            table_index=table_index, self_ref=self_ref, caption=caption,
            page_numbers=page_numbers, shape=Shape.OTHER,
            rows=(), columns=(), sections=(), cells=(), raw_markdown=raw_md,
        )

    rows = _build_rows(cells, shape)
    rows = _assign_sections(rows)
    columns = _build_columns(cells, rows, shape, table_index)
    sections = _build_sections(rows)
    norm_cells = _build_cells(cells, rows, columns, table_index, sections)

    return NormalizedTable(
        table_index=table_index, self_ref=self_ref, caption=caption,
        page_numbers=page_numbers, shape=shape, rows=rows, columns=columns,
        sections=sections, cells=norm_cells, raw_markdown=raw_md,
    )


def _empty_normalized(doc_json: dict, table_index: int) -> NormalizedTable:
    raw_md = ""
    try:
        tables = doc_json.get("tables") or []
        if 0 <= table_index < len(tables) and isinstance(tables[table_index], dict):
            raw_md = _resolve_raw_markdown(doc_json, table_index, tables[table_index])
    except Exception:
        pass
    return NormalizedTable(
        table_index=table_index, self_ref=f"#/tables/{table_index}",
        caption=None, page_numbers=(), shape=Shape.OTHER,
        rows=(), columns=(), sections=(), cells=(), raw_markdown=raw_md,
    )


def _resolve_raw_markdown(doc_json: dict, table_index: int, table: dict) -> str:
    """Per §8 step 7 lookup rule.

    Preference order:
      1. doc_json['texts'][i] where prov[0].$ref == '#/tables/{table_index}'
      2. table['text']
      3. table['data']['table_markdown']
      4. ''
    """
    target_ref = f"#/tables/{table_index}"
    for t in doc_json.get("texts") or []:
        prov = t.get("prov") or []
        if prov and isinstance(prov, list):
            first = prov[0] if isinstance(prov[0], dict) else None
            if first and first.get("$ref") == target_ref:
                txt = t.get("text")
                if isinstance(txt, str) and txt.strip():
                    return txt
    txt = table.get("text")
    if isinstance(txt, str) and txt.strip():
        return txt
    md = (table.get("data") or {}).get("table_markdown") or ""
    if md.strip():
        return md
    logger.debug("table_normalization.normalize: no raw_markdown source found for table %d", table_index)
    return ""


def _resolve_page_numbers(table: dict) -> tuple[int, ...]:
    pages: set[int] = set()
    for p in (table.get("prov") or []):
        page = p.get("page_no") if isinstance(p, dict) else None
        if isinstance(page, int):
            pages.add(page)
    return tuple(sorted(pages))


def _extract_unit(label: str) -> str | None:
    if not label:
        return None
    m = _UNIT_RE.search(label)
    return m.group(1) if m else None


def _is_identity_label(label: str) -> bool:
    norm = (label or "").strip().lower()
    return any(kw in norm for kw in IDENTITY_LABEL_KEYWORDS)


def _is_section_header_cell(cell: dict, num_cols: int) -> bool:
    span = (cell.get("end_col_offset_idx", 0) - cell.get("start_col_offset_idx", 0)) + 1
    text = (cell.get("text") or "").strip().lower()
    if span < max(2, num_cols - 1):
        return False
    return any(kw in text for kw in SECTION_KEYWORDS)


def _build_rows(cells: list[dict], shape: Shape) -> tuple[NormalizedRow, ...]:
    if not cells:
        return ()
    num_rows = max((c.get("end_row_offset_idx", 0) for c in cells), default=-1) + 1
    num_cols = max((c.get("end_col_offset_idx", 0) for c in cells), default=-1) + 1

    rows: list[NormalizedRow] = []
    for r in range(num_rows):
        label_cell = next(
            (c for c in cells if c.get("start_row_offset_idx") == r and c.get("start_col_offset_idx") == 0),
            None,
        )
        label = (label_cell.get("text") if label_cell else "") or ""
        is_section = bool(label_cell and _is_section_header_cell(label_cell, num_cols))
        is_identity = (not is_section) and _is_identity_label(label)
        # Unit detection priority:
        #   1. Inline parenthesized label suffix: "Max Range (km)" → "km"
        #   2. Per-row unit cell in an adjacent column: "Min Range, col 1 = m" → "m"
        # (1) is the original v9 behavior; (2) was added 2026-05-16 for SAM-spec
        # tables that use a dedicated unit column.
        unit = _extract_unit(label)
        if unit is None and not is_section and not is_identity:
            unit = _detect_row_unit_from_cells(r, cells)
        rows.append(NormalizedRow(
            row_idx=r, label=label.strip(),
            is_identity_row=is_identity, is_section_header=is_section,
            section=None, unit=unit,
        ))
    return tuple(rows)


def _assign_sections(rows: tuple[NormalizedRow, ...]) -> tuple[NormalizedRow, ...]:
    current: str | None = None
    out: list[NormalizedRow] = []
    for r in rows:
        if r.is_section_header:
            current = r.label
            out.append(r)
        else:
            out.append(NormalizedRow(
                row_idx=r.row_idx, label=r.label,
                is_identity_row=r.is_identity_row, is_section_header=False,
                section=current, unit=r.unit,
            ))
    return tuple(out)


def _build_sections(rows: tuple[NormalizedRow, ...]) -> tuple[TableSection, ...]:
    grouped: dict[str, list[int]] = {}
    order: list[str] = []
    for r in rows:
        if r.section is None or r.is_section_header:
            continue
        if r.section not in grouped:
            grouped[r.section] = []
            order.append(r.section)
        grouped[r.section].append(r.row_idx)
    return tuple(TableSection(name=name, row_indices=tuple(grouped[name])) for name in order)


def _build_columns(
    cells: list[dict], rows: tuple[NormalizedRow, ...], shape: Shape, table_index: int,
) -> tuple[NormalizedColumn, ...]:
    num_cols = max((c.get("end_col_offset_idx", 0) for c in cells), default=-1) + 1
    identity_rows = [r for r in rows if r.is_identity_row]
    columns: list[NormalizedColumn] = []
    for col_idx in range(1, num_cols):
        identity: dict[str, str] = {}
        for irow in identity_rows:
            cell = next(
                (c for c in cells
                 if c.get("start_row_offset_idx") == irow.row_idx
                 and c.get("start_col_offset_idx") <= col_idx <= c.get("end_col_offset_idx", col_idx)),
                None,
            )
            if cell:
                val = (cell.get("text") or "").strip()
                if val:
                    identity[irow.label] = val
        display = _display_name_for_column(identity, col_idx)
        columns.append(NormalizedColumn(col_idx=col_idx, identity=identity, display_name=display))
    return tuple(columns)


def _display_name_for_column(identity: dict[str, str], col_idx: int) -> str:
    norm = {k.lower(): v for k, v in identity.items()}
    for pref in _DISPLAY_NAME_PREFERENCE:
        for k, v in norm.items():
            if pref in k:
                return v
    return f"col-{col_idx}"


def _build_cells(
    cells: list[dict], rows: tuple[NormalizedRow, ...],
    columns: tuple[NormalizedColumn, ...], table_index: int,
    sections: tuple[TableSection, ...],
) -> tuple[NormalizedCell, ...]:
    out: list[NormalizedCell] = []
    spec_rows = [r for r in rows if not r.is_identity_row and not r.is_section_header]
    for row in spec_rows:
        for col in columns:
            cell = next(
                (c for c in cells
                 if c.get("start_row_offset_idx") == row.row_idx
                 and c.get("start_col_offset_idx") <= col.col_idx <= c.get("end_col_offset_idx", col.col_idx)),
                None,
            )
            if not cell:
                continue
            value = (cell.get("text") or "").strip()
            if not value:
                continue
            cell_pos = cells.index(cell)
            ref = CellRef(
                table_index=table_index, row_idx=row.row_idx, col_idx=col.col_idx,
                self_ref=f"#/tables/{table_index}/data/table_cells/{cell_pos}",
            )
            out.append(NormalizedCell(
                row_idx=row.row_idx, col_idx=col.col_idx,
                row_label=row.label, column_identity=col.identity,
                section=row.section, value=value, unit=row.unit, cell_ref=ref,
            ))
    return tuple(out)
