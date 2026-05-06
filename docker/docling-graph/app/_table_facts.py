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
