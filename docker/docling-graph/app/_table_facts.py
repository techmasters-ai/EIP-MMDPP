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
    # First pass: collect labels.
    #
    # Two kinds of col-0 cells qualify:
    # 1) row_header=True cells — standard spec-row labels (Length, Weight, etc.)
    # 2) section-header cells — col-0 cells with row_header=False whose
    #    col_span bridges INTO the data-column region (col_span > label_width).
    #    The SA-2 variants table contains "1st Stage" / "2nd Stage" cells
    #    with col_span=12, row_header=False that the PDF parser doesn't flag
    #    as row_header but ARE the section markers. Including them lets
    #    detect_section_context's header-row strategy match them.
    #
    # The label_width check on col_span is what discriminates them from
    # ordinary multi-column label cells (e.g., "Industry Designation"
    # spans cols 0-1 with col_span=2 == label_width, so it is NOT picked
    # up by the section-header branch — only the row_header=True branch).
    for c in cells:
        if c.get("start_col_offset_idx") != 0:
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        row_idx = c.get("start_row_offset_idx")
        if row_idx is None:
            continue

        is_row_header = c.get("row_header") is True
        is_section_header = (
            not is_row_header
            and (c.get("col_span", 1) or 1) > label_width
        )
        if not (is_row_header or is_section_header):
            continue

        rows_by_idx[row_idx] = LabelRow(
            row_idx=row_idx,
            label_text=text,
            label_col_span=c.get("col_span", 1) or 1,
            data_cells={},
        )

    # Second pass: collect data cells (col >= label_width).
    # Section-header rows naturally have empty data_cells because the
    # spanning cell occupies all data columns at that row, so no other
    # cells exist there. detect_section_context._is_header_row_marker
    # picks them up via name match against SECTION_KEYWORDS + empty-data
    # check.
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


# ============================================================
# Pipeline step 4: detect_section_context (spec §5.4 / D2)
# ============================================================

def detect_section_context(
    rows: list[LabelRow],
) -> list[tuple[LabelRow, SectionContext]]:
    """Pair each LabelRow with its section_ctx using a two-strategy chain.

    Strategy 1 (embedded): substring scan of label_text against
        SECTION_KEYWORDS. If matched, that row's section_ctx is the matched
        keyword AND the keyword is stripped from label_text in the returned
        row (so resolve_alias keys on the bare label).
    Strategy 2 (header-row): track most recent row whose label_text equals
        a section keyword AND whose data_cells are empty/header-like.
        Subsequent rows inherit until the next header-row or end-of-table.
        Header-row marker rows themselves are dropped from the result.

    Conflict resolution: embedded wins (only for non-header-row-marker rows).
    Header-row markers are checked first so that standalone section keywords
    with no data are treated as markers, not as embedded keywords.
    """
    from app._alias_map import SECTION_KEYWORDS

    out: list[tuple[LabelRow, SectionContext]] = []
    current_header_section: str | None = None

    for row in rows:
        label = row["label_text"]

        # Strategy 2 first: header-row marker detection. A row whose label IS
        # a section keyword (exact match after normalize) AND whose data_cells
        # are empty/header-like acts as a context propagator. Drop the marker.
        if _is_header_row_marker(row, SECTION_KEYWORDS):
            current_header_section = _matching_keyword(label, SECTION_KEYWORDS)
            continue

        # Strategy 1: embedded keyword scan (case-insensitive substring).
        embedded = _find_embedded_keyword(label, SECTION_KEYWORDS)
        if embedded is not None:
            new_label = _strip_keyword(label, embedded)
            new_row: LabelRow = {
                "row_idx": row["row_idx"],
                "label_text": new_label,
                "label_col_span": row["label_col_span"],
                "data_cells": row["data_cells"],
            }
            out.append((new_row, embedded))
            continue

        out.append((row, current_header_section))

    return out


def _find_embedded_keyword(label: str, keywords: tuple[str, ...]) -> str | None:
    label_lower = label.lower()
    # Iterate longest-first so "1st Stage" matches before "Stage".
    for kw in sorted(keywords, key=len, reverse=True):
        if kw.lower() in label_lower:
            return kw
    return None


def _strip_keyword(label: str, keyword: str) -> str:
    """Remove a case-insensitive occurrence of keyword from label, collapsing
    whitespace."""
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    stripped = pattern.sub("", label, count=1)
    return " ".join(stripped.split())


def _is_header_row_marker(row: LabelRow, keywords: tuple[str, ...]) -> bool:
    """Marker rows have a section keyword as their label AND no real data."""
    matching = _matching_keyword(row["label_text"], keywords)
    if matching is None:
        return False
    cells = row["data_cells"]
    if not cells:
        return True
    return all(not (v or "").strip() for v in cells.values())


def _matching_keyword(label: str, keywords: tuple[str, ...]) -> str | None:
    """Return the keyword IF the label IS exactly that keyword (after normalize)."""
    label_norm = " ".join(label.lower().split())
    for kw in keywords:
        if kw.lower() == label_norm:
            return kw
    return None


# ============================================================
# Pipeline step 5: resolve_alias (spec §5.5)
# ============================================================

def resolve_alias(
    label: str, section_ctx: SectionContext, active_pass: str
) -> str | None:
    """Look up the schema field for (label, section, pass). Returns None
    when no entry exists; the synthesizer skips that row silently."""
    from app._alias_map import ALIAS_MAP

    key: AliasKey = (normalize_label(label), section_ctx, active_pass)
    return ALIAS_MAP.get(key)


# ============================================================
# Pipeline step 6: coerce_value (spec §5.6 / D3 + D4)
# ============================================================

# Stop-words returning [] from coerce_value, normalized lowercased.
_STOP_WORDS = frozenset({
    "", "tbd", "n/a", "na", "unknown", "unk", "none",
    "?", "??", "???",
    "-", "--",
    "–", "—", "―",  # en/em/horizontal-bar dashes (pre-normalization)
})

# Range separators (after dash-class normalization to '-'): treat '-' here
# as ASCII hyphen-minus.
_ALTERNATIVE_SEPARATOR = "/"


def coerce_value(
    cell_text: str,
    schema_field: str,
    *,
    row_label: str = "",
) -> list[ParsedValue]:
    """Parse cell into 0+ ParsedValues. See spec §5.6 for the policy.

    `row_label` is the row's label text (e.g., "Length mm"), used as a
    fallback unit hint when the cell value itself has no explicit unit.
    SA-2-style tables store bare numbers in cells with units declared in
    the row label — without this hint, '10726' for body_length_m would
    coerce to 10726 metres instead of 10.726.
    """
    raw = cell_text
    # Stop-word check on stripped/lowered raw text (with dash collapse).
    stop_check = _DASH_CLASS.sub("-", raw.strip()).lower()
    if stop_check in _STOP_WORDS:
        return []

    # Determine if this is a string-typed field (no numeric suffix match).
    from app._alias_map import FIELD_SUFFIX_TO_UNIT_CLASS, UNIT_TABLE
    unit_class = _field_unit_class(schema_field, FIELD_SUFFIX_TO_UNIT_CLASS)
    if unit_class is None:
        # String passthrough.
        return [ParsedValue(value=raw.strip(), unit_inferred=None,
                            conversion_factor=1.0, raw_text=raw)]

    # Numeric path. Normalize dash characters first.
    normalized = _DASH_CLASS.sub("-", raw)
    # Split into value-fragments.
    fragments, is_range = _split_values(normalized)
    if not fragments:
        return []

    # Parse each fragment (number + unit).
    parsed: list[tuple[float, str | None]] = []
    for frag in fragments:
        num_unit = _parse_number_and_unit(frag)
        if num_unit is None:
            continue
        parsed.append(num_unit)
    if not parsed:
        return []

    # Determine the implied unit fallback chain:
    # 1. Explicit unit in the cell fragment (preferred).
    # 2. Implied unit from the row label (e.g., "Length mm" -> "mm").
    # 3. Field-aware bare-number inference for risky length/range fields.
    # 4. Canonical unit fallback only for fields where that is low risk.
    label_implied_unit = _extract_unit_from_label(row_label, unit_class, UNIT_TABLE)

    # Apply unit conversion using the schema field's unit class.
    out: list[ParsedValue] = []
    for value, unit_str in parsed:
        if unit_str is None:
            # No explicit unit in cell fragment — try row label, then a
            # field-aware inference. If the value is ambiguous for risky
            # fields, skip rather than emit a wrong table overlay fact.
            unit_str = (
                label_implied_unit
                or _infer_unit_for_bare_number(value, schema_field, unit_class, UNIT_TABLE)
            )
            if unit_str is None:
                if not _allow_canonical_fallback_for_bare_number(schema_field, unit_class):
                    continue
                unit_str = _canonical_unit_for_class(unit_class, UNIT_TABLE)
        unit_norm = unit_str.lower()
        converted = _convert_numeric_value(value, unit_norm, unit_class, UNIT_TABLE)
        if converted is None:
            # Unknown unit -> skip this value.
            continue
        converted_value, factor = converted
        out.append(ParsedValue(
            value=converted_value,
            unit_inferred=unit_norm,
            conversion_factor=factor,
            raw_text=raw,
        ))

    # Range collapse: if input was detected as a range and we ended up with
    # exactly two parsed values, return the midpoint as a single ParsedValue.
    if is_range and len(out) == 2:
        midpoint = (out[0].value + out[1].value) / 2
        return [ParsedValue(
            value=midpoint,
            unit_inferred=out[0].unit_inferred,
            conversion_factor=out[0].conversion_factor,
            raw_text=raw,
        )]

    return out


def _field_unit_class(schema_field: str, suffix_map: dict[str, str]) -> str | None:
    """Find the unit class by checking each registered suffix (longest first)."""
    for suffix in sorted(suffix_map, key=len, reverse=True):
        if schema_field.endswith(suffix):
            return suffix_map[suffix]
    return None


def _canonical_unit_for_class(unit_class: str, unit_table: dict) -> str:
    """Find a unit with factor 1.0 in the class — that's the canonical."""
    table = unit_table.get(unit_class) or {}
    for unit, factor in table.items():
        if factor == 1.0:
            return unit
    return ""


_BARE_UNIT_CANONICAL_RANGES: dict[str, tuple[float, float]] = {
    # Ranges are expressed in the schema's canonical unit. Candidate source
    # units are accepted only when exactly one conversion lands in range.
    "body_length_m": (0.5, 25.0),
    "body_diameter_m": (0.05, 3.0),
    "max_intercept_km": (1.0, 1000.0),
    "min_intercept_km": (1.0, 1000.0),
    "max_altitude_km": (1.0, 100.0),
    "min_altitude_km": (0.03, 10.0),
}

_BARE_UNIT_CANDIDATES_BY_CLASS: dict[str, tuple[str, ...]] = {
    "length_m": ("m", "mm"),
    "length_km": ("km", "m"),
}


def _infer_unit_for_bare_number(
    value: float,
    schema_field: str,
    unit_class: str,
    unit_table: dict,
) -> str | None:
    """Infer a source unit for bare risky fields when exactly one
    interpretation is plausible. Ambiguous values intentionally return None.
    """
    canonical_range = _BARE_UNIT_CANONICAL_RANGES.get(schema_field)
    candidate_units = _BARE_UNIT_CANDIDATES_BY_CLASS.get(unit_class)
    if canonical_range is None or candidate_units is None:
        return None

    low, high = canonical_range
    plausible: list[str] = []
    for unit in candidate_units:
        converted = _convert_numeric_value(value, unit, unit_class, unit_table)
        if converted is None:
            continue
        converted_value, _factor = converted
        if low <= converted_value <= high:
            plausible.append(unit)

    if len(plausible) == 1:
        return plausible[0]
    return None


def _allow_canonical_fallback_for_bare_number(
    schema_field: str, unit_class: str
) -> bool:
    """Return True when a bare numeric cell can safely use the field's
    canonical unit. Length/range fields covered by table overlays are risky:
    SA-2-style tables often omit row-label units while storing mm or m.
    """
    if schema_field in _BARE_UNIT_CANONICAL_RANGES:
        return False
    return unit_class not in {"length_m", "length_km"}


def _convert_numeric_value(
    value: float,
    unit_norm: str,
    unit_class: str,
    unit_table: dict,
) -> tuple[float, float] | None:
    if unit_class == "power_dbw" and unit_norm == "dbm":
        return value - 30.0, 1.0

    table = unit_table.get(unit_class) or {}
    factor = table.get(unit_norm)
    if factor is None:
        return None
    return value * factor, factor


def _extract_unit_from_label(
    row_label: str, unit_class: str, unit_table: dict
) -> str | None:
    """Scan the row label for any token matching a unit in the field's unit
    class. Used when the cell has no explicit unit but the label does
    (e.g., "Length mm", "Weight kg").

    Returns the matched unit (lowercased) or None. Longest-token-first match
    so 'kg' beats 'g' on labels like 'Weight kg'.

    Uses word-boundary-style matching for ALL unit lengths to prevent false
    matches inside larger words. Real failure modes this guards against:
    - 'min' (factor 60 for time_sec) embedded inside 'minimum'
    - 'rad' (factor 57.29 for angle_deg) embedded inside 'radar'
    - 'ton' (factor 1000 for mass_kg) embedded inside 'stone'
    Word-boundary chars: start/end of label, whitespace, or punctuation
    [,.;:/\\-]. Using a uniform check is cheaper to reason about than a
    length-dependent rule and has the same correctness on the canonical
    'Length mm' case.
    """
    if not row_label:
        return None
    table = unit_table.get(unit_class) or {}
    label_lower = row_label.lower()
    # Sort by descending length so 'kg' wins over 'g' on 'Weight kg'.
    for unit in sorted(table.keys(), key=len, reverse=True):
        pattern = re.compile(
            rf"(?:^|[\s,.;:/\\\-])({re.escape(unit)})(?:$|[\s,.;:/\\\-])"
        )
        if pattern.search(label_lower):
            return unit
    return None


def _split_values(cell_text: str) -> tuple[list[str], bool]:
    """Split a cell into value fragments and report whether it was a range.

    Returns (fragments, is_range). is_range=True means the orchestrator
    should collapse the parsed values to their midpoint.
    """
    text = cell_text.strip()
    # Discrete alternatives via slash.
    if _ALTERNATIVE_SEPARATOR in text:
        parts = [p.strip() for p in text.split(_ALTERNATIVE_SEPARATOR) if p.strip()]
        if len(parts) >= 2:
            return parts, False

    # Range via " to " word separator.
    if " to " in text.lower():
        parts = re.split(r"\s+to\s+", text, flags=re.IGNORECASE)
        if len(parts) == 2:
            return [p.strip() for p in parts], True

    # Range or alternatives via ASCII hyphen-minus (ambiguous).
    hyphen_split = re.split(r"\s*-\s*", text)
    if len(hyphen_split) == 2:
        a = _parse_number_and_unit(hyphen_split[0])
        b = _parse_number_and_unit(hyphen_split[1])
        if a is not None and b is not None:
            is_range = a[0] < b[0]  # X<Y -> range
            return [hyphen_split[0].strip(), hyphen_split[1].strip()], is_range

    # Single value.
    return [text], False


# Number + optional unit parser. Accepts "1135", "1135 kg", "1.5e3", "1,135".
_NUMBER_UNIT_PATTERN = re.compile(
    r"^\s*([+-]?\d+(?:[,\.]\d+)?(?:[eE][+-]?\d+)?)\s*([A-Za-z°/µμ]+)?\s*$"
)


def _parse_number_and_unit(text: str) -> tuple[float, str | None] | None:
    if not text:
        return None
    # Replace thousands separators (commas immediately before 3 digits).
    cleaned = re.sub(r",(\d{3})", r"\1", text)
    m = _NUMBER_UNIT_PATTERN.match(cleaned)
    if not m:
        return None
    num_str, unit_str = m.group(1), m.group(2)
    try:
        value = float(num_str)
    except ValueError:
        return None
    return value, unit_str


# ============================================================
# Pipeline step 7: emit_fact (spec §5.7)
# ============================================================

def emit_fact(
    entity_id: str,
    schema_field: str,
    value: float | int | str,
    source_label: str,
    text_idx: int,
) -> dict:
    """Render a (entity, field, value) triple as a TextItem-shaped dict.

    Format:
        "{entity_id} — {schema_field} = {value} [source: {source_label} row of variants table]"

    The em-dash separator is the schema-keyed prefix; the bracketed source
    preserves traceability without forcing the LLM to re-derive it.

    The TextItem skeleton mirrors the b9fe407 schema-validation fix used by
    _table_pivot.py — same shape so DoclingDocument's Pydantic union
    validates correctly.
    """
    formatted_value = _format_value(value)
    text = (
        f"{entity_id} — {schema_field} = {formatted_value} "
        f"[source: {source_label} row of variants table]"
    )
    return {
        "self_ref": f"#/texts/{text_idx}",
        "parent": {"$ref": "#/body"},
        "children": [],
        "content_layer": "body",
        "label": "text",
        "prov": [],
        "orig": text,
        "text": text,
    }


def _format_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Trim trailing .0 for integer-valued floats; retain decimals otherwise.
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


# ============================================================
# Public entry point: synthesize_table_facts (spec §4.2 / §6)
# ============================================================

import logging

logger = logging.getLogger(__name__)

_IDEMPOTENCE_FLAG = "__synthesized_table_facts__"


def synthesize_table_facts(
    doc_json: dict,
    *,
    active_pass: str,
    max_synthesized: int = 256,
) -> tuple[dict, FactStats]:
    """Append section-aware per-cell table-fact TextItems to doc_json.

    Spec §4.2 / §6 entry point. Pass-aware — same DoclingDocument fed to
    four different passes produces four different fact sets, each scoped
    to that pass's schema fields via ALIAS_MAP[(label, section, pass)].

    Returns (mutated_doc_json, FactStats). Mutates doc_json in place but
    also returns it for chaining. Sets the idempotence flag on first run;
    second call short-circuits with idempotent_skip=True.
    """
    stats = FactStats.empty()

    if not isinstance(doc_json, dict):
        return doc_json, stats

    if doc_json.get(_IDEMPOTENCE_FLAG) is True:
        stats.idempotent_skip = True
        return doc_json, stats

    tables = doc_json.get("tables") or []
    if not tables:
        doc_json[_IDEMPOTENCE_FLAG] = True
        return doc_json, stats

    texts = doc_json.setdefault("texts", [])
    body = doc_json.setdefault("body", {})
    body_children = body.setdefault("children", [])

    sections_seen: set[str] = set()

    for table in tables:
        stats.tables_seen += 1
        shape = detect_table_shape(table)
        stats.tables_by_shape[shape.value] = (
            stats.tables_by_shape.get(shape.value, 0) + 1
        )

        if shape == Shape.OTHER:
            continue

        rows = extract_label_rows(table, shape)
        if not rows:
            continue

        entity_ids = derive_entity_ids(rows, shape)
        if not entity_ids:
            # No identifiable entities — skip the whole table.
            continue

        # Detect collisions: derive_entity_ids deduplicates composites
        # (last-write-wins), so any difference between source-column count
        # and returned-id count indicates collisions. Applies to all shapes
        # (HYBRID is most common but column-major can collide too if two
        # columns happen to share the same identity row value).
        key_rows = [r for r in rows if _looks_like_key_label(r["label_text"])]
        source_cols: set[int] = set()
        for kr in key_rows:
            source_cols.update(
                col for col, cell in kr["data_cells"].items() if cell.strip()
            )
        stats.hybrid_collisions += max(0, len(source_cols) - len(entity_ids))

        sectioned = detect_section_context(rows)

        for row, section_ctx in sectioned:
            # Skip the rows that are themselves identity rows; they don't
            # produce facts (they produce entity_ids).
            if _looks_like_key_label(row["label_text"]):
                continue

            if section_ctx is not None:
                sections_seen.add(section_ctx)

            for entity_col, cell_text in row["data_cells"].items():
                entity_id = entity_ids.get(entity_col)
                if entity_id is None:
                    continue

                schema_field = resolve_alias(
                    row["label_text"], section_ctx, active_pass
                )
                if schema_field is None:
                    stats.rows_skipped_unresolvable += 1
                    continue

                # Pass row_label so coerce_value can extract implied units
                # (e.g., "Length mm" -> mm conversion for body_length_m).
                # detect_section_context strips the section keyword from
                # label_text but leaves the unit token intact ("1st Stage
                # Weight kg" -> "Weight kg"), so the post-strip label still
                # carries the unit hint we need.
                parsed = coerce_value(
                    cell_text, schema_field, row_label=row["label_text"]
                )
                if not parsed:
                    stats.values_skipped_unparseable += 1
                    continue

                if len(parsed) >= 2:
                    stats.multi_value_emissions += 1

                for pv in parsed:
                    if stats.facts_emitted >= max_synthesized:
                        stats.truncated_at_cap = True
                        doc_json[_IDEMPOTENCE_FLAG] = True
                        return doc_json, stats

                    text_idx = len(texts)
                    item = emit_fact(
                        entity_id=entity_id,
                        schema_field=schema_field,
                        value=pv.value,
                        source_label=row["label_text"],
                        text_idx=text_idx,
                    )
                    texts.append(item)
                    body_children.append({"$ref": f"#/texts/{text_idx}"})
                    stats.facts_emitted += 1

    stats.sections_detected = len(sections_seen)
    doc_json[_IDEMPOTENCE_FLAG] = True
    return doc_json, stats


# ============================================================================
# Mechanism A1 helpers (spec §5.2). Pure, deterministic, milliseconds.
#
# Imports of `app._alias_map` symbols are LAZY (inside each helper),
# matching the existing convention used by extract_label_rows /
# detect_section_context / coerce_value above. Module-level
# `from app._alias_map import …` would fail at importlib-spec-load
# time in tests (the existing test_table_facts_*.py files load this
# module via importlib.spec_from_file_location, before the conftest's
# service-root sys.path append has fired for *those* tests).
# ============================================================================
# `unicodedata` is already imported at the top of this module — do NOT
# re-import it here.


def _normalize_label(s: str) -> str:
    """Case-insensitive + NFC-folded label normalization.
    Used everywhere we substring-match a row label against the constants
    in _alias_map.py. Stable across operating-system locale settings."""
    return unicodedata.normalize("NFC", (s or "").strip()).casefold()


def _classify_identity_row(label: str) -> str | None:
    """Return entity type ("MISSILE_SYSTEM" / "RADAR_SYSTEM") if the
    label matches an identity row for that type. Otherwise None.

    Classification order (spec §5.1):
      1. Cross-entity-ref check FIRST — labels like 'Fan Song Variant'
         return None here so the caller routes them to
         _classify_cross_entity_ref instead.
      2. Identity-label check SECOND.
    """
    from app._alias_map import (  # lazy — see module header comment
        MISSILE_IDENTITY_LABELS,
        RADAR_IDENTITY_LABELS,
        CROSS_ENTITY_REF_PATTERNS,
    )
    norm = _normalize_label(label)
    if not norm:
        return None
    # (1) cross-entity-ref short-circuits — the row is NOT an identity row
    # for any entity type.
    if norm in CROSS_ENTITY_REF_PATTERNS:
        return None
    # (2) identity-label check, longest-first to avoid 'designation'
    # eating 'industry designation' (we already removed bare 'designation'
    # from the labels list, but stay defensive).
    for missile_label in MISSILE_IDENTITY_LABELS:
        if missile_label in norm:
            return "MISSILE_SYSTEM"
    for radar_label in RADAR_IDENTITY_LABELS:
        if radar_label in norm:
            return "RADAR_SYSTEM"
    return None


def _classify_cross_entity_ref(label: str) -> str | None:
    """Return target entity type if the label is a cross-entity-ref row
    (e.g., 'Fan Song Variant' → 'RADAR_SYSTEM' in a missile-context
    table). Otherwise None.
    """
    from app._alias_map import CROSS_ENTITY_REF_PATTERNS  # lazy
    norm = _normalize_label(label)
    return CROSS_ENTITY_REF_PATTERNS.get(norm)


def _extract_alias_clusters(
    table: dict,
    *,
    entity_type: str,
) -> list[dict[str, str]]:
    """For each data column in the table, build a {label: value} map of
    cells from rows whose label matches an identity row for entity_type.

    Excludes:
      - Cross-entity-ref rows (Fan Song Variant etc.) — they go to
        cross_entity_hints, NOT into the alias cluster.
      - Empty cells.

    Returns a list parallel to the data columns (in left-to-right order).
    Empty list if the table has no identity rows for entity_type.
    """
    cells = (table or {}).get("data", {}).get("table_cells") or []
    if not cells:
        return []

    # Reuse the existing _label_column_width(cells) helper defined
    # earlier in this module. Same algorithm — max end_col_offset_idx of
    # col-0 row_header cells.
    label_width = _label_column_width(cells)

    # Map (row_idx → label) for identity rows that match entity_type.
    identity_rows: dict[int, str] = {}
    for c in cells:
        if c.get("start_col_offset_idx") != 0:
            continue
        if not c.get("row_header"):
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        if _classify_identity_row(text) != entity_type:
            continue
        row = c.get("start_row_offset_idx")
        if row is None:
            continue
        identity_rows[row] = text

    if not identity_rows:
        return []

    # Enumerate data column starts (cells with start_col_offset_idx >=
    # label_width).
    data_col_starts = sorted({
        c.get("start_col_offset_idx") for c in cells
        if c.get("start_col_offset_idx") is not None
        and c.get("start_col_offset_idx") >= label_width
    })

    clusters: list[dict[str, str]] = []
    for col in data_col_starts:
        cluster: dict[str, str] = {}
        for cell in cells:
            if cell.get("start_col_offset_idx") != col:
                continue
            row = cell.get("start_row_offset_idx")
            if row not in identity_rows:
                continue
            text = (cell.get("text") or "").strip()
            if not text:
                continue
            cluster[identity_rows[row]] = text
        clusters.append(cluster)
    return clusters


def _pick_canonical(
    cluster: dict[str, str],
    *,
    entity_type: str,
) -> str:
    """Pick the canonical name from an alias cluster using
    CANONICAL_PRIORITY[entity_type]. First priority entry that's a
    substring of any cluster label wins.

    Fallback: if no priority entry matches, return the alphabetically-
    first value (NFC + casefold sort) and log INFO
    'canonical_picked_via_fallback'. Empty cluster → empty string.
    """
    from app._alias_map import CANONICAL_PRIORITY  # lazy
    if not cluster:
        return ""
    priority = CANONICAL_PRIORITY.get(entity_type, ())
    for priority_label in priority:
        priority_norm = _normalize_label(priority_label)
        for label, value in cluster.items():
            if priority_norm in _normalize_label(label):
                return value
    # Fallback — alphabetically first.
    return sorted(cluster.values(), key=lambda v: _normalize_label(v))[0]


# ============================================================================
# Mechanism A1 public entry point + qualification gate (spec §3 / §5.2).
#
# Imports of `app._alias_map` and `app.schemas` are LAZY (inside each helper
# function body), matching the convention used by all other helpers in this
# module. Module-level imports would fail at importlib-spec-load time in
# tests that load this module via importlib.spec_from_file_location.
# ============================================================================


def _is_column_major_or_hybrid(table: dict) -> bool:
    """Lightweight shape detector: leftmost column carries row_header
    cells AND ≥4 cols / ≥4 rows. Mirrors the existing _table_pivot.py
    heuristic so we stay consistent with the table-fact synthesizer."""
    data = (table or {}).get("data") or {}
    cells = data.get("table_cells") or []
    if not cells:
        return False
    nr, nc = data.get("num_rows") or 0, data.get("num_cols") or 0
    if nr < 4 or nc < 4:
        return False
    col0 = [c for c in cells if c.get("start_col_offset_idx") == 0]
    if not col0:
        return False
    label_count = sum(1 for c in col0 if c.get("row_header") is True)
    return label_count * 2 >= len(col0)


def _table_qualifies_for_overlay(
    table: dict,
    *,
    entity_type: str,
) -> tuple[bool, str]:
    """Return (qualifies, reason). Reason is one of:
       'qualified', 'wrong_shape', 'too_few_entity_columns',
       'no_identity_rows', 'sparse_identity_cells'.
    Spec §3 four-of-four AND gate.
    """
    if not _is_column_major_or_hybrid(table):
        return False, "wrong_shape"

    cells = (table or {}).get("data", {}).get("table_cells") or []
    label_width = _label_column_width(cells)

    data_col_starts = sorted({
        c.get("start_col_offset_idx") for c in cells
        if c.get("start_col_offset_idx") is not None
        and c.get("start_col_offset_idx") >= label_width
    })
    if len(data_col_starts) < 4:
        return False, "too_few_entity_columns"

    # Identity rows for entity_type
    identity_rows = {
        c.get("start_row_offset_idx")
        for c in cells
        if c.get("start_col_offset_idx") == 0
        and c.get("row_header")
        and _classify_identity_row(c.get("text") or "") == entity_type
    }
    if not identity_rows:
        return False, "no_identity_rows"

    # Every entity column has a non-empty cell in ≥1 identity row
    for col in data_col_starts:
        has_cell = False
        for cell in cells:
            if cell.get("start_col_offset_idx") != col:
                continue
            if cell.get("start_row_offset_idx") not in identity_rows:
                continue
            if (cell.get("text") or "").strip():
                has_cell = True
                break
        if not has_cell:
            return False, "sparse_identity_cells"

    return True, "qualified"


def extract_table_overlay(doc_json: dict) -> tuple["TableOverlay", dict]:
    """Parse the FIRST strictly-qualifying column-major / hybrid table
    in doc.tables[]. Returns (TableOverlay, stats_dict). Spec §5.2.
    """
    # Lazy import to avoid circular schemas → _table_facts dependency
    # at module load.
    from app.schemas import TableOverlay  # noqa: PLC0415

    stats = {
        "tables_processed": 0,
        "tables_skipped_unqualified": 0,
        "tables_skipped_multi": 0,
        "tables_skipped_other": 0,
        "columns_skipped_no_canonical": 0,
        "columns_with_canonical_via_fallback": 0,
        "values_skipped_unparseable": 0,
        "facts_skipped_construct_fail": 0,
    }

    tables = (doc_json or {}).get("tables") or []
    if not tables:
        return TableOverlay(), stats

    winner_table = None
    winner_entity_type = None
    for table in tables:
        # Try MISSILE_SYSTEM first, then RADAR_SYSTEM (entity-type-
        # agnostic in v1; first qualifying-of-any-type wins).
        qualified_for_this_table = False
        for et in ("MISSILE_SYSTEM", "RADAR_SYSTEM"):
            qualifies, _reason = _table_qualifies_for_overlay(
                table, entity_type=et,
            )
            if qualifies:
                qualified_for_this_table = True
                if winner_table is None:
                    winner_table = table
                    winner_entity_type = et
                else:
                    stats["tables_skipped_multi"] += 1
                break
        if not qualified_for_this_table:
            # No entity-type qualified for this table.
            if not _is_column_major_or_hybrid(table):
                stats["tables_skipped_other"] += 1
            else:
                stats["tables_skipped_unqualified"] += 1

    if winner_table is None:
        return TableOverlay(), stats

    stats["tables_processed"] = 1

    # Build alias clusters + canonical names.
    clusters = _extract_alias_clusters(
        winner_table, entity_type=winner_entity_type,
    )
    sub_map: dict[str, str] = {}
    canonical_per_col: list[str] = []
    for cluster in clusters:
        canonical = _pick_canonical(cluster, entity_type=winner_entity_type)
        canonical_per_col.append(canonical)
        if not canonical:
            stats["columns_skipped_no_canonical"] += 1
            continue
        for alias in cluster.values():
            if alias and alias != canonical:
                sub_map[alias] = canonical
            if alias:
                # Identity: also map canonical → canonical so
                # downstream rewrite is idempotent.
                sub_map.setdefault(alias, canonical)

    alias_map_by_entity_type: dict[str, dict[str, str]] = {}
    if sub_map:
        alias_map_by_entity_type[winner_entity_type] = sub_map

    # Build facts from spec rows + cross_entity_hints from cross-entity-
    # ref rows. Reuses existing _table_facts.py primitives:
    # extract_label_rows / detect_section_context / coerce_value /
    # resolve_alias / _looks_like_key_label / _classify_cross_entity_ref.
    cells = winner_table.get("data", {}).get("table_cells") or []
    label_width = _label_column_width(cells)
    data_col_starts = sorted({
        c.get("start_col_offset_idx") for c in cells
        if c.get("start_col_offset_idx") is not None
        and c.get("start_col_offset_idx") >= label_width
    })

    facts, hints, target_alias_maps = _emit_facts_and_hints(
        winner_table,
        canonical_per_col=canonical_per_col,
        winner_entity_type=winner_entity_type,
        data_col_starts=data_col_starts,
        stats=stats,
    )

    # 2026-05-17 (Rec 2): merge target-side alias maps from cross-entity
    # rows so the resolver can bridge OCR-spaced and family-derived alias
    # forms (e.g. `RSN- 75V` → `Fan Song`). Generic — works for any
    # `<family> Variant` row, gated by upstream presence at resolution time.
    for target_type, target_map in target_alias_maps.items():
        if not target_map:
            continue
        existing = alias_map_by_entity_type.setdefault(target_type, {})
        for alias_key, canonical in target_map.items():
            existing.setdefault(alias_key, canonical)

    overlay = TableOverlay(
        alias_map_by_entity_type=alias_map_by_entity_type,
        facts=facts,
        cross_entity_hints=hints,
    )
    return overlay, stats


# 2026-05-17 (Rec 2): two generic alias-shape transforms used to register
# target-side aliases for cross-entity rows. Both are document-agnostic:
# OCR-space collapse handles common Docling artifacts; family-name strip
# handles any `<X> Variant` label, not just specific equipment.

def _normalize_target_alias_key(s: str) -> str:
    """Match the resolver's `normalize_evidence_text` shape: uppercased +
    whitespace-collapsed. Used so registered alias keys collide with
    resolver lookups."""
    if not s:
        return ""
    return " ".join(str(s).upper().split())


def _normalize_ocr_spaces(alias: str) -> str:
    """Collapse `<token>- <token>` (dash followed by whitespace) to
    `<token>-<token>` — a common Docling OCR pattern in hyphenated
    equipment designations. Generic — no equipment-specific logic.

    Example: `RSN- 75V` → `RSN-75V`, `Some-  Token` → `Some-Token`.
    """
    if not alias:
        return alias
    import re as _re
    return _re.sub(r"-\s+", "-", alias.strip())


def _extract_cross_entity_family_name(label: str) -> str | None:
    """If the cross-entity-row label is of the form `<family name>
    Variant(s)`, return `<family name>` stripped. Otherwise return None.

    Generic over any `<X> Variant` label — no equipment names hardcoded.
    The returned family name is only USEFUL as a canonical bridge if the
    same family name happens to be in the upstream catalog for the target
    entity type; the resolver checks that at lookup time.
    """
    if not label:
        return None
    import re as _re
    m = _re.match(r"^(.*?)\s+variants?\s*$", label.strip(), _re.IGNORECASE)
    if not m:
        return None
    family = m.group(1).strip()
    return family or None


def _emit_facts_and_hints(
    table: dict,
    *,
    canonical_per_col: list[str],
    winner_entity_type: str,
    data_col_starts: list[int],
    stats: dict,
) -> tuple[list, list, dict[str, dict[str, str]]]:
    """Walk spec rows + cross-entity-ref rows; emit TableFacts and
    CrossEntityHints. Wraps the existing _table_facts.py primitives:
    extract_label_rows, detect_section_context, resolve_alias,
    coerce_value, _looks_like_key_label, _classify_cross_entity_ref.

    Multi-pass routing: for each non-identity, non-cross-entity row, try
    each missile/radar pass that can own that label-section combo via
    resolve_alias(label, section, pass_name). The first pass that
    resolves it gets the fact. (resolve_alias returns None when a pass
    doesn't own the label; the alias map in _alias_map.py is keyed on
    (label, section, pass) triples, which gives us pass-uniqueness for
    free.)
    """
    from app.schemas import TableFact, CrossEntityHint  # noqa: PLC0415

    facts: list = []
    hints: list = []
    # 2026-05-17 (Rec 2): {target_entity_type → {normalized_alias_key → canonical}}
    # Built from cross-entity rows. Returned to caller for merging into the
    # overlay's alias_map_by_entity_type. Empty dict when there are no
    # cross-entity rows OR no family-name bridge available.
    target_alias_maps: dict[str, dict[str, str]] = {}

    if winner_entity_type == "MISSILE_SYSTEM":
        candidate_passes = (
            "missile_propulsion",
            "missile_kinematics",
            "missile_speed_timing",
            "missile_airframe",
        )
        target_for_cross_ref = "RADAR_SYSTEM"
    elif winner_entity_type == "RADAR_SYSTEM":
        candidate_passes = (
            "radar_antenna", "radar_modulation",
            "radar_power_rf", "radar_timing",
        )
        target_for_cross_ref = "MISSILE_SYSTEM"
    else:
        return [], [], {}

    # Map data_col_starts → canonical
    col_to_canonical = {
        col: canonical_per_col[i]
        for i, col in enumerate(data_col_starts)
        if i < len(canonical_per_col) and canonical_per_col[i]
    }

    shape = detect_table_shape(table)
    if shape == Shape.OTHER:
        return [], [], {}

    rows = extract_label_rows(table, shape)
    if not rows:
        return [], [], {}

    sectioned = detect_section_context(rows)

    for row, section_ctx in sectioned:
        label_text = row["label_text"]

        # Classification order MUST match spec §5.1: cross-entity-ref
        # check FIRST, identity-label check SECOND. The existing
        # _looks_like_key_label considers "fan song variant" a key
        # label (its _KEY_LABEL_PATTERNS includes that string), so
        # without this ordering Fan Song rows would be skipped before
        # we could emit them as CrossEntityHints — re-introducing the
        # exact bug spec §1's empirical example called out (Fan Song
        # accidentally collapsed into the missile alias cluster).
        cross_target = _classify_cross_entity_ref(label_text)
        if cross_target == target_for_cross_ref:
            # 2026-05-17 (Rec 2): family-name bridge — if the row label
            # parses as `<family> Variant(s)`, that family name is the
            # canonical we'll map all this row's aliases to. The resolver
            # gates the actual lookup on upstream presence, so registering
            # the bridge is safe even when no such canonical exists.
            family_canonical = _extract_cross_entity_family_name(label_text)
            target_map = target_alias_maps.setdefault(cross_target, {}) if family_canonical else None
            for entity_col, cell_text in row["data_cells"].items():
                target_alias = (cell_text or "").strip()
                if not target_alias:
                    continue
                source_canonical = col_to_canonical.get(entity_col, "")
                if not source_canonical:
                    continue
                try:
                    hints.append(CrossEntityHint(
                        source_canonical=source_canonical,
                        source_entity_type=winner_entity_type,
                        target_alias=target_alias,
                        target_entity_type=cross_target,
                        relationship_kind="associated_with",
                    ))
                except Exception:
                    stats["facts_skipped_construct_fail"] += 1
                # Register alias forms IFF we have a family_canonical
                # bridge. setdefault preserves first-write-wins so a
                # later identical alias on another row can't clobber.
                if target_map is not None:
                    raw_key = _normalize_target_alias_key(target_alias)
                    target_map.setdefault(raw_key, family_canonical)
                    ocr_norm = _normalize_ocr_spaces(target_alias)
                    if ocr_norm and ocr_norm != target_alias:
                        target_map.setdefault(
                            _normalize_target_alias_key(ocr_norm),
                            family_canonical,
                        )
            continue

        # Identity rows produce alias_map only — already handled in
        # the caller via _extract_alias_clusters + _pick_canonical.
        # Comes AFTER the cross-entity-ref check above so Fan Song
        # rows (which _looks_like_key_label happens to match) reach
        # the cross-entity branch first.
        if _looks_like_key_label(label_text):
            continue

        # Spec rows: try each candidate pass; the alias map will
        # resolve at most one (or zero).
        for entity_col, cell_text in row["data_cells"].items():
            canonical = col_to_canonical.get(entity_col, "")
            if not canonical:
                continue
            for pass_name in candidate_passes:
                schema_field = resolve_alias(
                    label_text, section_ctx, pass_name,
                )
                if schema_field is None:
                    continue
                parsed = coerce_value(
                    cell_text, schema_field, row_label=label_text,
                )
                if not parsed:
                    stats["values_skipped_unparseable"] += 1
                    continue
                for pv in parsed:
                    try:
                        facts.append(TableFact(
                            canonical_entity=canonical,
                            entity_type=winner_entity_type,
                            schema_field=schema_field,
                            value=pv.value,
                            source_label=label_text,
                            section_ctx=section_ctx,
                            pass_name=pass_name,
                            raw_text=cell_text,
                        ))
                    except Exception:
                        stats["facts_skipped_construct_fail"] += 1
                # Stop after the first pass that owns this label —
                # alias map is keyed on (label, section, pass), so at
                # most one pass will match for any given (label,
                # section).
                break

    return facts, hints, target_alias_maps
