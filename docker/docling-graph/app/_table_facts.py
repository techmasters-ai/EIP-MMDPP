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
    # 3. Canonical unit for the field's unit class (factor 1.0).
    label_implied_unit = _extract_unit_from_label(row_label, unit_class, UNIT_TABLE)

    # Apply unit conversion using the schema field's unit class.
    out: list[ParsedValue] = []
    for value, unit_str in parsed:
        if unit_str is None:
            # No explicit unit in cell fragment — try row label, then canonical.
            unit_str = label_implied_unit or _canonical_unit_for_class(unit_class, UNIT_TABLE)
        unit_norm = unit_str.lower()
        unit_table = UNIT_TABLE.get(unit_class) or {}
        factor = unit_table.get(unit_norm)
        if factor is None:
            # Unknown unit -> skip this value.
            continue
        out.append(ParsedValue(
            value=value * factor,
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
    r"^\s*([+-]?\d+(?:[,\.]\d+)?(?:[eE][+-]?\d+)?)\s*([A-Za-z°/]+)?\s*$"
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
