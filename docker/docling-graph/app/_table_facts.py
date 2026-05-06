"""Section-aware per-cell table-fact synthesis (spec
2026-05-05-section-aware-table-fact-synthesis-design.md).

Replaces _table_pivot.py operationally. Emits one TextItem per
(entity, schema_field, value) triple drawn from column-major tables in
DoclingDocument.tables[]. Pass-aware — same document fed to four passes
produces four different fact sets, each scoped to that pass's schema fields.

This module declares only the types and the public synthesize_table_facts
entry-point skeleton; the pipeline's pure functions live in this same file
and are added in subsequent tasks.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TypeAlias, TypedDict


class Shape(str, Enum):
    """Detected shape of a DoclingDocument table.

    COLUMN_MAJOR: leftmost column(s) hold row labels; remaining columns
        hold per-entity values. Variant-specs tables are typically this.
    ROW_MAJOR: top row(s) hold column labels; remaining rows hold per-
        entity values. Financial/comparative tables are typically this.
    HYBRID: column-major with multi-row identity (e.g., row 0 "Industry
        Designation" + row 1 "Missile Type" both labeling each column).
    OTHER: skip — synthesis not applicable (below 4×4 floor, or shape
        signals match neither pattern).
    """

    COLUMN_MAJOR = "column_major"
    ROW_MAJOR = "row_major"
    HYBRID = "hybrid"
    OTHER = "other"


# Section keyword (e.g., "1st Stage") or None when no section context applies.
SectionContext: TypeAlias = str | None

# Key into ALIAS_MAP: (label_normalized, section_ctx, pass_name).
AliasKey: TypeAlias = tuple[str, SectionContext, str]


class LabelRow(TypedDict):
    """Normalized representation of a label-bearing table row.

    Both column-major and row-major paths produce LabelRow records; the
    pipeline downstream of extract_label_rows is shape-agnostic.
    """

    row_idx: int
    label_text: str  # raw, pre-normalization
    label_col_span: int
    data_cells: dict[int, str]  # entity_col -> cell text (raw)


@dataclass(frozen=True)
class ParsedValue:
    """One value extracted from a cell after coercion.

    A single cell may produce multiple ParsedValues (discrete alternatives
    like "1135/1028") or one (single value, range collapsed to midpoint).
    Frozen because instances are value-typed and shared across emit_fact.
    """

    value: float | str
    unit_inferred: str | None
    conversion_factor: float  # 1.0 if no conversion applied
    raw_text: str


@dataclass
class FactStats:
    """Per-call synthesis stats. Mutable so the orchestrator increments
    counters in place. Surfaces in diagnostics["service_table_facts"]."""

    tables_seen: int = 0
    tables_by_shape: dict[str, int] = field(default_factory=dict)
    sections_detected: int = 0  # distinct sections matched (embedded only)
    facts_emitted: int = 0
    rows_skipped_unresolvable: int = 0
    values_skipped_unparseable: int = 0
    multi_value_emissions: int = 0  # cells producing ≥2 facts (alternatives, not ranges)
    hybrid_collisions: int = 0  # composite-id collisions; last-write-wins
    truncated_at_cap: bool = False
    idempotent_skip: bool = False

    @classmethod
    def empty(cls) -> "FactStats":
        return cls()

    def as_dict(self) -> dict:
        return asdict(self)


# Dash-class characters mapped to single ASCII hyphen for stable matching.
# Covers: hyphen, non-breaking hyphen, figure dash, en-dash, em-dash,
# horizontal bar, minus sign, hyphen bullet, two-em / three-em dash,
# small em-dash, small hyphen-minus, fullwidth hyphen-minus.
_DASH_CLASS = re.compile(r"[‐‑‒–—―−⁃﹘﹣－]")

# Punctuation to strip after dash normalization. Keeps ASCII alphanumerics,
# whitespace, and hyphens (dashes already collapsed). Strips:
# . , ; : ! ? ' " ` ( ) [ ] { } / \ | _ * + = & % @ # ^ ~ < >
_PUNCT_TO_STRIP = re.compile(
    r"[\.\,\;\:\!\?\'\"\`\(\)\[\]\{\}\/\\\|_\*\+\=\&\%\@\#\^\~\<\>]"
)


def normalize_label(text: str) -> str:
    """Normalize a label string for ALIAS_MAP lookup and §8.3 drift-guard.

    Steps (spec §5.5):
    1. Unicode NFKC fold (collapses fancy quotes, full-width digits, etc.).
    2. Collapse all dash variants (en/em/figure/etc.) -> ASCII hyphen.
    3. Strip punctuation per _PUNCT_TO_STRIP (hyphens preserved).
    4. Lowercase.
    5. Collapse whitespace runs to single space; strip leading/trailing.

    The same function is used by resolve_alias and the §8.3 drift-guard
    test so prose-side and label-side normalization always agree.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _DASH_CLASS.sub("-", text)
    text = _PUNCT_TO_STRIP.sub(" ", text)
    text = text.lower()
    text = " ".join(text.split())
    return text


# ============================================================
# Pipeline step 1: detect_table_shape (spec §5.2 / D1)
# ============================================================

def detect_table_shape(table: dict) -> Shape:
    """Classify a DoclingDocument table cell-shape into one of four buckets.

    COLUMN_MAJOR: ≥50% of leftmost-col cells flagged row_header=True.
    ROW_MAJOR: ≥50% of top-row cells flagged column_header=True.
    HYBRID: column-major with multiple identity rows (left col has more
        than one row_header cell whose value is a key-label pattern).
    OTHER: below 4×4 floor, or neither pattern fires.
    """
    data = (table or {}).get("data") or {}
    cells = data.get("table_cells") or []
    num_rows = data.get("num_rows") or 0
    num_cols = data.get("num_cols") or 0

    if num_rows < 4 or num_cols < 4 or not cells:
        return Shape.OTHER

    col0_cells = [c for c in cells if c.get("start_col_offset_idx") == 0]
    row0_cells = [c for c in cells if c.get("start_row_offset_idx") == 0]

    col0_rh = sum(1 for c in col0_cells if c.get("row_header") is True)
    row0_ch = sum(1 for c in row0_cells if c.get("column_header") is True)

    is_col_major = col0_cells and col0_rh * 2 >= len(col0_cells)
    is_row_major = row0_cells and row0_ch * 2 >= len(row0_cells)

    if is_col_major and not is_row_major:
        # Distinguish HYBRID by counting row_header cells in col 0 that
        # match identity patterns. The patterns are intentionally local to
        # detect_table_shape (single-purpose); derive_entity_ids has its
        # own list. Both are kept in sync via a constant defined below.
        identity_count = sum(
            1 for c in col0_cells
            if c.get("row_header") is True
            and _looks_like_key_label((c.get("text") or "").strip())
        )
        return Shape.HYBRID if identity_count >= 2 else Shape.COLUMN_MAJOR
    if is_row_major and not is_col_major:
        return Shape.ROW_MAJOR
    return Shape.OTHER


# Identity-row label patterns. Cells matching any of these (case-insensitive
# substring) are treated as entity-naming labels, not spec labels. Shared by
# detect_table_shape (HYBRID detection) and derive_entity_ids (Task 9).
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


def _looks_like_key_label(label: str) -> bool:
    if not label:
        return False
    norm = label.strip().lower()
    return any(pat in norm for pat in _KEY_LABEL_PATTERNS)


# ============================================================
# Pipeline step 2: extract_label_rows (spec §5.3)
# ============================================================

def extract_label_rows(table: dict, shape: Shape) -> list[LabelRow]:
    """Normalize column-major and row-major into a unified LabelRow stream."""
    if shape == Shape.OTHER:
        return []

    cells = (table or {}).get("data", {}).get("table_cells") or []
    if not cells:
        return []

    if shape in (Shape.COLUMN_MAJOR, Shape.HYBRID):
        return _extract_column_major(cells)
    if shape == Shape.ROW_MAJOR:
        return _extract_row_major(cells)
    return []


def _label_column_width(cells: list[dict]) -> int:
    """Return how many leftmost columns the row-label cells span."""
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


def _extract_column_major(cells: list[dict]) -> list[LabelRow]:
    label_width = _label_column_width(cells)

    rows_by_idx: dict[int, LabelRow] = {}
    # First pass: collect labels (row_header cells in label region).
    for c in cells:
        if c.get("start_col_offset_idx") != 0:
            continue
        if not c.get("row_header"):
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        row_idx = c.get("start_row_offset_idx")
        if row_idx is None:
            continue
        rows_by_idx[row_idx] = LabelRow(
            row_idx=row_idx,
            label_text=text,
            label_col_span=c.get("col_span", 1) or 1,
            data_cells={},
        )

    # Second pass: collect data cells (col >= label_width).
    for c in cells:
        col = c.get("start_col_offset_idx")
        if col is None or col < label_width:
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        row_idx = c.get("start_row_offset_idx")
        if row_idx is None or row_idx not in rows_by_idx:
            continue
        rows_by_idx[row_idx]["data_cells"][col] = text

    return [rows_by_idx[k] for k in sorted(rows_by_idx)]


def _extract_row_major(cells: list[dict]) -> list[LabelRow]:
    # Top-row column headers; assume col 0 is identity, cols 1+ are spec labels.
    header_cells = [c for c in cells if c.get("start_row_offset_idx") == 0
                    and c.get("column_header") is True]
    if not header_cells:
        return []
    sorted_headers = sorted(header_cells, key=lambda c: c.get("start_col_offset_idx", 0))
    if not sorted_headers:
        return []
    spec_headers = sorted_headers[1:]  # skip identity column

    rows_by_label: dict[str, LabelRow] = {}
    for header in spec_headers:
        col = header.get("start_col_offset_idx")
        text = (header.get("text") or "").strip()
        if not text or col is None:
            continue
        rows_by_label[text] = LabelRow(
            row_idx=col,  # in row-major, "row_idx" of the synthetic LabelRow is the source col
            label_text=text,
            label_col_span=1,
            data_cells={},
        )

    # Collect data cells (rows below 0, at the columns we care about).
    label_cols = {h["row_idx"]: h["label_text"] for h in rows_by_label.values()}
    for c in cells:
        row = c.get("start_row_offset_idx")
        col = c.get("start_col_offset_idx")
        if row is None or row == 0:
            continue
        if col not in label_cols:
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        rows_by_label[label_cols[col]]["data_cells"][row] = text

    return list(rows_by_label.values())


# ============================================================
# Pipeline step 3: derive_entity_ids (spec §5.3.5)
# ============================================================

def derive_entity_ids(rows: list[LabelRow], shape: Shape) -> dict[int, str]:
    """Map entity_col -> entity_id from key-label rows.

    For COLUMN_MAJOR: single key-label row's data_cells become entity_ids.
    For HYBRID: multiple key-label rows produce composite identities by
        concatenating non-empty cells in row order.

    Composite collisions (two columns producing the same entity_id) are
    resolved last-write-wins: only the latest column with that identity
    appears in the returned dict. The orchestrator detects collisions by
    comparing the count of source columns to the count of returned ids
    (incrementing FactStats.hybrid_collisions for the difference).
    """
    key_rows = [r for r in rows if _looks_like_key_label(r["label_text"])]
    if not key_rows:
        return {}

    # Collect all entity_cols seen across key rows.
    all_cols: set[int] = set()
    for kr in key_rows:
        all_cols.update(kr["data_cells"].keys())

    # Build (col -> composite_id) preserving column iteration order.
    raw: dict[int, str] = {}
    for col in sorted(all_cols):
        parts = []
        for kr in key_rows:  # rows already sorted by row_idx
            cell = kr["data_cells"].get(col, "").strip()
            if cell:
                parts.append(cell)
        if parts:
            raw[col] = " ".join(parts)

    # Apply last-write-wins on duplicate composites: track which
    # composite_id last appeared at which column, then keep only those cols.
    last_col_for_id: dict[str, int] = {}
    for col in sorted(raw):
        last_col_for_id[raw[col]] = col

    return {col: composite for composite, col in last_col_for_id.items()}
