"""Deterministic numeric/table candidate extractors.

Runs BEFORE the LLM call. Sweeps the batch text for `<number><unit>`
patterns and returns the matched substrings so we can inject them into
the user prompt as a hint block. The LLM then decides which field each
candidate maps to (a span-classification task rather than a number-
copying task — much easier for general-purpose models).

Also detects Docling's flattened-table form:

    Column Name, 1 = Header. Column Name, 2 = value.

Cells sharing the same numeric suffix are from the same source row. The
LLM otherwise tends to attach a row's numeric values to whichever named
system column is easiest, instead of the schema's intended identity
column. Rendering row-aligned hints makes this a table attribution task
instead of a prose inference task.

Phase A item 5 of the post-LLM-extraction-fix work. Pure code; no LLM
involvement; runs in <1ms per batch.
"""
from __future__ import annotations

from collections import defaultdict
import re

# Unit alternation ordered longest-first so "GHz" doesn't get split as
# "G" + "Hz". Each entry is the literal unit string the regex matches;
# the unit family is implicit in the LLM's interpretation.
_UNIT_PATTERNS = [
    # Frequency
    r"GHz", r"MHz", r"kHz", r"Hz",
    # Power
    r"MW", r"kW", r"dBW", r"dBm",
    # Gain / loss
    r"dBi", r"dBd", r"dB",
    # Time
    r"µs", r"us", r"ms", r"min", r"sec", r"s",
    # Distance
    r"km", r"nmi", r"NM", r"miles", r"mi", r"ft", r"in", r"cm", r"mm", r"m",
    # Mass
    r"kg", r"lb", r"lbs", r"g", r"tonnes?", r"t",
    # Speed
    r"km/h", r"kph", r"mph", r"m/s", r"Mach",
    # Angle
    r"degrees?", r"deg", r"°", r"radians?", r"rad", r"mil",
    # Pulse counts (no unit but field-relevant patterns)
]

_NUMBER = r"-?\d+(?:\.\d+)?"
_NUM_UNIT = re.compile(
    r"(?<![A-Za-z0-9_])("
    + _NUMBER
    + r")\s*("
    + "|".join(_UNIT_PATTERNS)
    + r")(?![A-Za-z])"
)

_FLATTENED_TABLE_CELL = re.compile(
    r"(?:^|[.\n\r])\s*"
    r"(?P<label>[A-Za-z][A-Za-z0-9/()' \-\u2010-\u2014]{1,80}?)"
    r"\s*,\s*"
    r"(?P<row>\d{1,3})"
    r"\s*=\s*"
    r"(?P<value>[^.\n\r]*)",
    re.MULTILINE,
)

# Substring patterns (case-insensitive) for cell labels that name a row
# identifier \u2014 the column whose value uniquely identifies the entity the
# rest of the row's specs belong to. When a row contains one of these,
# Option A emits per-identifier keyed hints so the LLM can pair specs with
# the schema's identity field directly instead of inferring across columns.
_IDENTIFIER_LABEL_PATTERNS = (
    "missile type",
    "missile variant",
    "fan song variant",
    "radar variant",
    "radar type",
    "system name",
    "system designation",
    "industry designation",
    "military designation",
    "nato designation",
    "designation",
    "variant",
    "missile",
)

# Common unit / placeholder strings that signal a row is a header/units
# row rather than data. Used to skip header rows when emitting keyed hints.
_HEADER_VALUE_TOKENS = frozenset((
    "", "-", "\u2014", "n/a",
    "m", "mm", "cm", "km", "kg", "g", "lb", "lbs", "ft", "in",
    "s", "sec", "ms", "us", "min", "hz", "khz", "mhz", "ghz",
    "deg", "\u00b0", "kw", "mw", "dbi", "dbw", "db",
    "m/s", "km/h", "mph", "kph",
))


def extract_numeric_candidates(text: str, *, max_candidates: int = 40) -> list[str]:
    """Return verbatim "<number> <unit>" substrings found in text.

    Order = order of first appearance. De-duplicates exact repeats.
    Caps at max_candidates to avoid bloating the prompt; in practice
    a single radar parameter table has 10-25 candidates.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for match in _NUM_UNIT.finditer(text):
        num = match.group(1)
        unit = match.group(2)
        candidate = f"{num} {unit}".strip()
        # Normalize spacing around the number so "35dBi" and "35 dBi"
        # collapse to one canonical form.
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
        if len(out) >= max_candidates:
            break
    return out


def _clean_cell_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n;:")


def extract_flattened_table_rows(
    text: str,
    *,
    max_rows: int = 12,
    max_cells_per_row: int = 8,
    max_row_chars: int = 260,
) -> list[str]:
    """Return compact row-aligned hints from flattened Docling tables.

    The returned strings are intentionally schema-agnostic. They preserve
    source labels and row ids only; the LLM still decides which schema
    fields, if any, each value supports.
    """
    if not text:
        return []

    rows: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for match in _FLATTENED_TABLE_CELL.finditer(text):
        label = _clean_cell_text(match.group("label"))
        value = _clean_cell_text(match.group("value"))
        if not label or not value:
            continue
        row = int(match.group("row"))
        rows[row].append((label, value))

    out: list[str] = []
    for row in sorted(rows):
        cells = rows[row]
        if len(cells) < 2:
            continue
        cell_text = " | ".join(
            f"{label}={value}" for label, value in cells[:max_cells_per_row]
        )
        rendered = f"row {row}: {cell_text}"
        if len(rendered) > max_row_chars:
            rendered = rendered[: max_row_chars - 1].rstrip() + "…"
        out.append(rendered)
        if len(out) >= max_rows:
            break
    return out


def _is_identifier_label(label: str) -> bool:
    """Return True when a cell label names a row identifier column."""
    if not label:
        return False
    norm = label.strip().lower()
    return any(pat in norm for pat in _IDENTIFIER_LABEL_PATTERNS)


def _looks_like_header_row(cells: list[tuple[str, str]]) -> bool:
    """Return True when a row's cells look like the header/units row.

    A header row is one where every cell's value is either empty, a unit,
    or echoes the label itself (Docling emits ``Label, 1 = Label`` for
    column-name rows). Used to skip row 1 when emitting keyed hints.
    """
    if not cells:
        return True
    for label, value in cells:
        v = (value or "").strip().lower()
        l = (label or "").strip().lower()
        if v in _HEADER_VALUE_TOKENS:
            continue
        if v == l:
            continue
        return False
    return True


def extract_keyed_table_rows(
    text: str,
    *,
    max_lines: int = 32,
    max_specs_per_line: int = 12,
    max_line_chars: int = 320,
) -> list[str]:
    """Return per-row hints keyed by each identifier cell in that row.

    For a flattened table row containing identifier columns (e.g. Missile
    Type, NATO Designation) and spec columns (e.g. Max Range, Max Alt),
    this emits one line per identifier-cell of the form::

        Missile Type=1D: Max Range=29000, Max Alt=22000, Length=10726

    Multiple identifiers in a single source row produce multiple lines —
    same specs, different keys — so the LLM can pick whichever identifier
    matches the schema's identity field without cross-referencing columns.
    """
    if not text:
        return []

    rows: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for match in _FLATTENED_TABLE_CELL.finditer(text):
        label = _clean_cell_text(match.group("label"))
        value = _clean_cell_text(match.group("value"))
        if not label or not value:
            continue
        row = int(match.group("row"))
        rows[row].append((label, value))

    out: list[str] = []
    for row in sorted(rows):
        cells = rows[row]
        if len(cells) < 2:
            continue
        if _looks_like_header_row(cells):
            continue

        identifiers: list[tuple[str, str]] = []
        specs: list[tuple[str, str]] = []
        for label, value in cells:
            if (value or "").strip().lower() in _HEADER_VALUE_TOKENS:
                continue
            if _is_identifier_label(label):
                identifiers.append((label, value))
            else:
                specs.append((label, value))

        if not identifiers or not specs:
            continue

        spec_str = ", ".join(f"{l}={v}" for l, v in specs[:max_specs_per_line])

        for id_label, id_value in identifiers:
            line = f"{id_label}={id_value}: {spec_str}"
            if len(line) > max_line_chars:
                line = line[: max_line_chars - 1].rstrip() + "…"
            out.append(line)
            if len(out) >= max_lines:
                return out
    return out


def render_candidate_block(candidates: list[str]) -> str:
    """Render a candidate list as a user-prompt section. Returns ''
    when no candidates so the caller can skip emitting the section."""
    if not candidates:
        return ""
    bullets = "\n".join(f"- {c}" for c in candidates)
    return (
        "=== NUMERIC CANDIDATES (verbatim spans found in batch document) ===\n"
        + bullets
        + "\n"
        + "Each candidate above appears verbatim in the BATCH DOCUMENT. "
        "When populating a numeric field, prefer assigning a candidate "
        "from this list to a matching field — don't paraphrase, don't "
        "convert unless the Unit Policy permits, and don't invent values "
        "that aren't on this list.\n"
        "=== END NUMERIC CANDIDATES ===\n\n"
    )


def render_keyed_table_block(rows: list[str]) -> str:
    """Render keyed-by-identifier table-row hints for the LLM prompt."""
    if not rows:
        return ""
    bullets = "\n".join(f"- {row}" for row in rows)
    return (
        "=== KEYED TABLE ROWS (each spec already attributed to a per-row identifier) ===\n"
        + bullets
        + "\n"
        "Each line above shows one source-table row whose specs were already "
        "attributed to a specific identifier (Missile Type, NATO Designation, "
        "Variant, etc.). When the active schema's identity field matches one of "
        "these identifiers, copy the listed spec values directly to that "
        "system. Do NOT shuffle a spec to a different identifier and do NOT "
        "invent values that aren't on these lines.\n"
        "=== END KEYED TABLE ROWS ===\n\n"
    )


def render_flattened_table_block(rows: list[str]) -> str:
    """Render row-aligned flattened-table hints for the LLM prompt."""
    if not rows:
        return ""
    bullets = "\n".join(f"- {row}" for row in rows)
    return (
        "=== FLATTENED TABLE ROWS (cells sharing the same row number belong together) ===\n"
        + bullets
        + "\n"
        "Docling may render tables as repeated `Column, row = value` cells. "
        "Treat cells with the same row number as one source table row. "
        "If row 1 contains headers or units, use it only to interpret later rows. "
        "When a later row contains multiple identity/designation columns, attach "
        "numeric/spec values to the identity required by the active schema, not "
        "to an adjacent system, NATO code, radar, launcher, or caption name unless "
        "that is the schema's identity field. Do not emit values from rows that "
        "lack a directly matching in-scope identity for the current pass.\n"
        "=== END FLATTENED TABLE ROWS ===\n\n"
    )
