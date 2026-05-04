"""Pre-chunker pivoting of column-major tables (Phase B / B1+B2 combined).

Detects DoclingDocument tables where the leftmost column holds row labels
(``Industry Designation``, ``Missile Type``, ``Max Range``, ...) and the
remaining columns hold per-entity values. For each such table, synthesizes
one prose summary per data column and appends it to ``texts[]`` so the
chunker sees self-contained row-major paragraphs alongside the original
column-major table.

Each synthesized summary contains the column's identifier cells AND all
its spec cells in one continuous text. Whichever chunk the chunker places
that text in carries both the entity name and its full spec set —
eliminating both the column-arithmetic problem (B1) and the table-split-
around-identifier problem (B2) that the empirical Run 17 measurement
exposed.

Non-destructive: original ``tables[]`` is preserved verbatim. The new
text items are additive; they do not replace the original table chunks.
The chunker may emit both forms into the LLM prompt; that's fine — the
model uses whichever form fits its current pass.
"""
from __future__ import annotations

from collections import defaultdict


# Cell-label substrings (case-insensitive) that name an entity-identifier
# column — the values of these rows uniquely identify the entity that the
# rest of the column's specs belong to. A column-major table is only
# pivoted when at least one of these labels appears in the leftmost column.
_KEY_LABEL_PATTERNS = (
    "missile type",
    "missile variant",
    "industry designation",
    "military designation",
    "nato designation",
    "fan song variant",
    "radar variant",
    "system name",
    "system designation",
    "designation",
    "variant",
)


def _is_key_label(label: str) -> bool:
    if not label:
        return False
    norm = label.strip().lower()
    return any(pat in norm for pat in _KEY_LABEL_PATTERNS)


def _is_column_major_table(table: dict) -> bool:
    """Return True when the table looks column-major.

    Column-major in this codebase means: the leftmost column holds row
    labels (Docling flags these with ``row_header: True``), and the
    remaining columns hold per-entity values.

    Heuristic: at least half of the cells whose ``start_col_offset_idx``
    is 0 carry ``row_header: True`` AND the table has at least 4 rows
    and 4 columns (one label column + three data columns minimum).
    """
    data = (table or {}).get("data") or {}
    cells = data.get("table_cells") or []
    if not cells:
        return False

    num_rows = data.get("num_rows") or 0
    num_cols = data.get("num_cols") or 0
    if num_rows < 4 or num_cols < 4:
        return False

    col0_cells = [c for c in cells if c.get("start_col_offset_idx") == 0]
    if not col0_cells:
        return False
    label_count = sum(1 for c in col0_cells if c.get("row_header") is True)
    return label_count * 2 >= len(col0_cells)


def _label_column_width(table: dict) -> int:
    """Return how many leftmost columns the row-label cells span.

    Most variant-style tables have a 1-column label region, but the SA-2
    PDF variants table puts ``Industry Designation`` in cols 0-1 (col_span
    of 2). We take the max ``end_col_offset_idx`` of the row-header cells
    in row 0 as the boundary.
    """
    cells = (table or {}).get("data", {}).get("table_cells") or []
    width = 1
    for c in cells:
        if c.get("start_col_offset_idx") != 0:
            continue
        if not c.get("row_header"):
            continue
        end_col = c.get("end_col_offset_idx", 1) or 1
        if end_col > width:
            width = end_col
    return width


def _row_labels_of(table: dict) -> dict[int, str]:
    """Map ``start_row_offset_idx`` -> label text for left-column cells.

    Only cells flagged ``row_header: True`` count — they are Docling's
    explicit signal that this cell is a row label rather than data.
    """
    labels: dict[int, str] = {}
    cells = (table or {}).get("data", {}).get("table_cells") or []
    for c in cells:
        if c.get("start_col_offset_idx") != 0:
            continue
        if not c.get("row_header"):
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        row = c.get("start_row_offset_idx")
        if row is None:
            continue
        labels[row] = text
    return labels


def _column_data_of(
    table: dict, *, label_width: int
) -> dict[int, list[tuple[int, str]]]:
    """Map ``start_col_offset_idx`` -> [(row_offset, text), ...] for data columns.

    Columns 0 .. label_width-1 are the label column(s) and skipped.
    """
    by_col: dict[int, list[tuple[int, str]]] = defaultdict(list)
    cells = (table or {}).get("data", {}).get("table_cells") or []
    for c in cells:
        col = c.get("start_col_offset_idx")
        if col is None or col < label_width:
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        row = c.get("start_row_offset_idx")
        if row is None:
            continue
        by_col[col].append((row, text))
    return by_col


def _render_column_summary(
    col_offset: int,
    column: list[tuple[int, str]],
    row_labels: dict[int, str],
) -> str | None:
    """Render one data column as a row-major prose summary.

    Returns ``None`` when the column has no key-label rows — without an
    identity anchor the summary can't be attributed to anything, so emitting
    it would just bloat the prompt.
    """
    by_row = dict(column)  # row_offset -> cell text
    keys: list[tuple[str, str]] = []
    specs: list[tuple[str, str]] = []
    # Iterate row labels in row order (top-to-bottom of the source table).
    for row_offset in sorted(row_labels):
        label = row_labels[row_offset]
        cell = by_row.get(row_offset)
        if cell is None:
            continue
        if _is_key_label(label):
            keys.append((label, cell))
        else:
            specs.append((label, cell))

    if not keys or not specs:
        return None

    key_str = ", ".join(f"{label}={value}" for label, value in keys)
    spec_str = "; ".join(f"{label}={value}" for label, value in specs)
    return (
        f"Variants table column {col_offset} — {key_str}. "
        f"Specs: {spec_str}."
    )


def synthesize_pivoted_table_texts(
    doc_json: dict,
    *,
    max_synthesized: int = 64,
) -> tuple[dict, int]:
    """Append per-column row-major summaries for column-major tables.

    Returns ``(doc_json, n_appended)``. Mutates ``doc_json`` in place: new
    items are appended to ``texts[]`` and ``body.children`` so the chunker
    walks them. Returns ``(doc_json, 0)`` when the document has no
    column-major tables matching the pivot criteria.

    ``max_synthesized`` caps the total number of new text items; covers
    documents with very wide tables (16+ data columns).
    """
    if not isinstance(doc_json, dict):
        return doc_json, 0
    tables = doc_json.get("tables") or []
    if not tables:
        return doc_json, 0

    texts = doc_json.setdefault("texts", [])
    body = doc_json.setdefault("body", {})
    body_children = body.setdefault("children", [])

    appended = 0

    for table_idx, table in enumerate(tables):
        if not _is_column_major_table(table):
            continue
        row_labels = _row_labels_of(table)
        if not row_labels:
            continue
        # Require at least one key-label row before we pivot. Otherwise we
        # might emit summaries for tables that aren't entity-keyed.
        if not any(_is_key_label(label) for label in row_labels.values()):
            continue
        label_width = _label_column_width(table)
        column_data = _column_data_of(table, label_width=label_width)
        if not column_data:
            continue

        for col_offset in sorted(column_data):
            if appended >= max_synthesized:
                return doc_json, appended
            summary = _render_column_summary(
                col_offset, column_data[col_offset], row_labels
            )
            if not summary:
                continue
            new_idx = len(texts)
            # Shape mirrors the upstream-entities preamble item
            # (main.py:619-628) so the DoclingDocument Pydantic union
            # validates as TextItem instead of failing at every label
            # variant. Empty prov[] is intentional — the synthesized
            # summary has no source-document char range; using the
            # source table's prov here breaks charspan invariants.
            text_item = {
                "self_ref": f"#/texts/{new_idx}",
                "parent": {"$ref": "#/body"},
                "children": [],
                "content_layer": "body",
                "label": "text",
                "prov": [],
                "orig": summary,
                "text": summary,
            }
            texts.append(text_item)
            body_children.append({"$ref": f"#/texts/{new_idx}"})
            appended += 1

    return doc_json, appended
