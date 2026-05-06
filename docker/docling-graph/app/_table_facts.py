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
