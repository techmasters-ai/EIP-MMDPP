# Section-Aware Per-Cell Table-Fact Synthesis — Design

**Status:** Approved 2026-05-05
**Predecessor:** `2026-04-27-radar-field-group-extraction-design.md`
**Replaces operationally:** `_table_pivot.py` (Phase B / B1+B2)
**Related TODO:** #83 (post-extraction `IDENTITY_FILTER` relaxation — deferred; downgraded after this design's empirical findings)

## 1. Problem

The 2026-05-05 alias-patch sweep (T=0.3, T=1.0) and the B1+B2 re-run produced
identical aggregate scorecards on `missile_propulsion`:

```
✓ exact: 0     ✗ wrong: 14–16     — null: 4–6
```

The wrong values are not hallucinations — they are real numbers from the source,
attributed to the wrong schema field via a consistent off-by-one row-to-field
shift:

| Missile | Reported booster_mass_kg | Actual GT booster_mass_kg | What 'booster' actually got |
|---|---|---|---|
| 13DM    | 2283 | 1032 | gt `total_mass_kg` |
| 13DA    | 2289 | 1032 | gt `total_mass_kg` |
| 20D     | 2391 | 1011 | gt `total_mass_kg` |
| 20DSU   | 2397 | 1011 | gt `total_mass_kg` |

The model reads the column-major variants table top-to-bottom, hits "Total
Weight" first, maps it to `booster_mass_kg`, then the next row ("1st Stage
Weight") becomes `sustain_mass_kg`. Same pattern in every variants-table row,
both temperatures, with and without B1+B2 pivot active.

**Root cause:** column-major table row-to-field attribution is the failure
class. Alias hints in the schema description and §12b prose do not override the
LLM's column-arithmetic strategy when navigating a 15-row × 12-column table.
The B1+B2 prose-per-column pivot does not fix this — it emits the same
ambiguous label set in a single sentence per column, which the model still has
to disambiguate.

## 2. Goals

1. **Recover the 0 ✓ propulsion failure mode.** Empirical acceptance: ✓ exact
   ≥ 6 on `missile_propulsion` at T=1.0 against the §20 GT scorecard.
2. **Generalize beyond the SA-2 case.** Support diverse table shapes (D1),
   section structures (D2), multi-value cells (D3), and unit conversions (D4).
   Defer multilingual labels (D5), prose-table hybrids (D6), and cross-table
   consolidation (D7).
3. **Preserve auto-evidence + sanitization wiring.** Synthesis runs after
   sanitization, before chunking, in the same `run_extraction_pass` flow.
4. **Non-critical to extraction.** Synthesizer failure must never break
   `/extract-pass` — the original chunker path runs on whatever's there.

## 3. Non-Goals (deferred)

- D5 (multilingual labels), D6 (prose-table hybrids), D7 (cross-table
  consolidation) — separate designs.
- Generated-from-data §12b prose. Manual sync between the structured map and
  the prose remains acceptable; a unit test guards drift.
- Notebook outcome tracker `facts/pass` column — followup PR after the
  synthesizer ships.

## 4. Architecture

### 4.1 Module layout

| File | Status | Responsibility |
|---|---|---|
| `docker/docling-graph/app/_table_facts.py` | NEW | Synthesizer pipeline (six pure functions) |
| `docker/docling-graph/app/_alias_map.py` | NEW | Structured alias map (Python data); paired with §12b prose, drift-guarded |
| `docker/docling-graph/app/main.py` | MODIFIED | Replace B1+B2 call site; surface `FactStats` in diagnostics |
| `docker/docling-graph/app/_table_pivot.py` | DEPRECATED | Marked DEPRECATED; not imported. Removed next cycle. |
| `docker/docling-graph/tests/test_table_facts.py` | NEW | Unit + integration tests |
| `docker/docling-graph/tests/test_alias_map.py` | NEW | Drift guard against §12b prose |
| `docker/docling-graph/tests/test_table_pivot.py` | PRESERVED 1 cycle | Regression for the deprecated path |

### 4.2 Public API

```python
def synthesize_table_facts(
    doc_json: dict,
    *,
    active_pass: str,
    max_synthesized: int = 256,
) -> tuple[dict, FactStats]
```

The synthesizer is **pass-aware**: same DoclingDocument fed to four different
passes produces four different fact sets, each scoped to that pass's schema
fields via `ALIAS_MAP[(label, section_ctx, active_pass)]`.

### 4.3 Pipeline shape (Approach B — pure functions)

```
DoclingDocument.tables[]
  ↓ detect_table_shape(table)                   # D1
  ↓ extract_label_rows(table, shape)            # row labels + per-column data
  ↓ derive_entity_ids(rows)                     # one entity_id per data column from key-label rows
  ↓ detect_section_context(rows)                # D2
  ↓ for each (entity_col, entity_id, row, section_ctx):
      resolve_alias(label, section_ctx, pass)   # → schema_field | None
        ↓ coerce_value(cell_text, schema_field) # D3 + D4
        ↓ emit_fact(entity_id, schema_field, value, source_label)  # → TextItem
  ↓ append TextItems to doc.texts[] + body.children
```

Each step is a pure function (no shared state, no hidden side effects),
testable in isolation. New behavior is added by inserting a function or
extending an existing one's input set. Strategy registries are not earned by
the D1–D4 scope; revisit if a 4th detection mode per dimension is ever shipped.

## 5. Components

### 5.0 Types

All shapes used across the pipeline. Defined once here; referenced everywhere
else. Stored at the top of `_table_facts.py`.

`LabelRow` is a `TypedDict` because it carries no behavior and has many fields;
`ParsedValue` and `FactStats` are `@dataclass(frozen=True)` so they support
methods (`.empty()`, `.as_dict()`) and positional construction used elsewhere
in the spec.

```python
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import TypeAlias, TypedDict

class Shape(str, Enum):
    COLUMN_MAJOR = "column_major"
    ROW_MAJOR = "row_major"
    HYBRID = "hybrid"
    OTHER = "other"

# `section_ctx` is either a section keyword (e.g., "1st Stage") or None.
SectionContext: TypeAlias = str | None

# `(label_normalized, section_ctx, pass_name)` — keys into ALIAS_MAP.
AliasKey: TypeAlias = tuple[str, SectionContext, str]

class LabelRow(TypedDict):
    row_idx: int
    label_text: str        # raw, pre-normalization
    label_col_span: int
    data_cells: dict[int, str]  # entity_col → cell text (raw)

@dataclass(frozen=True)
class ParsedValue:
    value: float | str
    unit_inferred: str | None
    conversion_factor: float       # 1.0 if no conversion applied
    raw_text: str

@dataclass
class FactStats:
    tables_seen: int = 0
    tables_by_shape: dict[str, int] = field(default_factory=dict)
    sections_detected: int = 0     # distinct sections matched (embedded only)
    facts_emitted: int = 0
    rows_skipped_unresolvable: int = 0
    values_skipped_unparseable: int = 0
    multi_value_emissions: int = 0 # cells that produced ≥2 facts (alternatives, not ranges)
    hybrid_collisions: int = 0     # composite-id collisions; last-write-wins
    truncated_at_cap: bool = False # True if max_synthesized hit
    idempotent_skip: bool = False  # True if doc was already synthesized

    @classmethod
    def empty(cls) -> "FactStats":
        return cls()

    def as_dict(self) -> dict:
        return asdict(self)
```

`tables_by_shape` is per-Shape-enum-value census only (no synthetic keys);
`hybrid_collisions` is its own field for clean observability.

### 5.1 `_alias_map.py` — structured alias map

```python
ALIAS_MAP: dict[AliasKey, str]
# Value: canonical schema field name (e.g., "booster_mass_kg")

SECTION_KEYWORDS: tuple[str, ...]
# ("1st Stage", "2nd Stage", "Booster", "Sustainer", "Sustain", "Ejector", ...)
# Extensible per domain.

UNIT_TABLE: dict[str, dict[str, float]]
# Per-unit-class conversion factors keyed by unit-class name.
# {"length_m": {"mm": 0.001, "cm": 0.01, "in": 0.0254, "ft": 0.3048, "km": 1000.0, "m": 1.0}, ...}

FIELD_SUFFIX_TO_UNIT_CLASS: dict[str, str]
# {"_m": "length_m", "_kg": "mass_kg", "_sec": "time_sec", "_mps": "velocity_mps",
#  "_km": "length_km", "_dbi": "gain_dbi", "_mhz": "frequency_mhz", ...}
```

Keyed on the `AliasKey` triple (label_normalized, section_ctx, pass_name) so
pass- and section-conditionals are first-class. A drift-guard unit test asserts
every entry has a corresponding §12b prose mention.

### 5.2 `detect_table_shape(table) → Shape` (D1)

Returns `COLUMN_MAJOR | ROW_MAJOR | HYBRID | OTHER`.

- Reuses `_table_pivot.py`'s `_is_column_major_table` heuristic for COLUMN_MAJOR
  (≥50% of leftmost-col cells flagged `row_header=True`).
- Mirror heuristic for ROW_MAJOR (≥50% of top-row cells flagged
  `column_header=True`).
- HYBRID: column-major with multi-row left labels (rows 0..K all in label
  column 0 with `row_header=True`, no data values) — partially handled today
  via `_label_column_width`.
- OTHER: tables below 4×4 floor or matching neither shape.

### 5.3 `extract_label_rows(table, shape) → list[LabelRow]`

Normalizes column-major and row-major into the same intermediate `LabelRow`
shape (defined in §5.0):

- **COLUMN_MAJOR:** today's logic in `_table_pivot.py` (leftmost label cols,
  remaining data cols).
- **ROW_MAJOR:** transposed equivalent.
- **HYBRID:** rows whose label belongs to the multi-row identity region (rows
  0..K all in label column 0 with `row_header=True`, no data values) are still
  emitted as `LabelRow`s — entity-id derivation in §5.3.5 consumes them.

### 5.3.5 `derive_entity_ids(rows, shape) → dict[int, str]`

Maps `entity_col → entity_id`. Identity comes from rows whose label matches the
existing `_KEY_LABEL_PATTERNS` from `_table_pivot.py` (`"Missile Type"`,
`"Industry Designation"`, `"Military Designation"`, `"NATO Designation"`,
`"Variant"`, `"System Name"`, `"Designation"`, etc.).

- **One key-label row** (typical column-major): `entity_id = data_cells[col]`.
  E.g., `"Missile Type"` row col 2 = `"1D"` → `entity_id = "1D"`.
- **Multiple key-label rows** (HYBRID composite): concatenate the key cells in
  row order, separated by single space — `"Industry Designation"` row col 2 =
  `"S-75"`, `"Missile Type"` row col 2 = `"1D"` → `entity_id = "S-75 1D"`. If
  two columns produce the same composite, increment `hybrid_collisions` and
  last-write-wins (later column overwrites earlier).
- **No key-label row found**: column is unidentifiable; skip it entirely
  (synthesizer emits no facts for that column). `rows_skipped_unresolvable`
  is incremented by the number of would-have-been emissions for that column.
- **Empty cell at the key-label row**: same as above — skip the column.

This step runs once per table, before the main emission loop. Result feeds
`emit_fact()`.

### 5.4 `detect_section_context(rows) → list[(LabelRow, SectionContext)]` (D2)

Two-strategy chain (in order):

1. **Embedded:** substring scan of `label_text` against `SECTION_KEYWORDS`. If
   matched, that row's `section_ctx` is the matched keyword.
2. **Header-row:** track most recent row whose `label_text` *equals* a section
   keyword (after normalization — see §5.5) AND whose `data_cells` are
   empty/header-like (every cell either empty, repeats the label, or is the
   single section keyword text); subsequent rows inherit that section until
   the next header-row or end-of-table.

Default: `None` if neither matches. Rows with `None` section context can still
resolve aliases that don't require sectioning (e.g., `total_mass_kg`).

**Conflict resolution:** embedded wins (most-specific signal). Header-row
context applies only when the row has no embedded section keyword.

**Header-row fixture example** (for the unit test in §8.1):

```
Row 0: "Missile Type"        | "1D"   | "13D"  | "13DM" | "13DA"   <- key label row
Row 1: "Total Weight kg"     | "2163" | "2283" | "2283" | "2289"   <- no section
Row 2: "1st Stage"           | ""     | ""     | ""     | ""        <- header-row marker
Row 3: "Weight kg"           | "1135" | "1032" | "1032" | "1032"   <- inherits "1st Stage"
Row 4: "Time sec"            | "4.0"  | "4.0"  | "4.0"  | "4.0"    <- inherits "1st Stage"
Row 5: "2nd Stage"           | ""     | ""     | ""     | ""        <- new header-row marker
Row 6: "Weight kg"           | "1028" | "1251" | "1251" | "1257"   <- inherits "2nd Stage"
```

Rows 3, 4 resolve `"Weight kg" + "1st Stage"` → `booster_mass_kg`,
`"Time sec" + "1st Stage"` → `booster_time_sec`. Row 6 resolves
`"Weight kg" + "2nd Stage"` → `sustain_mass_kg`. The SA-2 PDF embeds the
section keyword in the label itself (`"1st Stage Weight kg"`), exercising the
embedded path; the header-row path is exercised by corpora that put section
markers as standalone rows.

### 5.5 `resolve_alias(label, section_ctx, active_pass) → str | None`

Lookup `ALIAS_MAP[(normalize_label(label), section_ctx, active_pass)]`.

**Normalization rules** (single function `normalize_label` exported from
`_table_facts.py`, used by both the resolver AND the drift-guard test in §8.3
so they assert the same thing):

```python
import re
import unicodedata

# Dash-class characters mapped to single ASCII hyphen for stable matching.
_DASH_CLASS = re.compile(r"[‐-―−⁃﹘﹣－]")

# Punctuation to strip after dash normalization. Keeps ASCII alphanumerics,
# whitespace, and hyphens (dashes already collapsed). Specifically removes:
# . , ; : ! ? ' " ` ( ) [ ] { } / \ | _ * + = & % @ # ^ ~ < >
_PUNCT_TO_STRIP = re.compile(r"[\.\,\;\:\!\?\'\"\`\(\)\[\]\{\}\/\\\|_\*\+\=\&\%\@\#\^\~\<\>]")

def normalize_label(text: str) -> str:
    """Normalize a label string for ALIAS_MAP lookup and drift-guard checks.

    1. Unicode NFKC fold (collapses fancy quotes, full-width digits, etc.).
    2. Collapse all dash variants (en/em/figure/non-breaking-hyphen) → "-".
    3. Strip punctuation per _PUNCT_TO_STRIP. Hyphens preserved (so
       "SA-2" stays distinct from "SA 2").
    4. Lowercase.
    5. Collapse whitespace runs to single space; strip leading/trailing.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _DASH_CLASS.sub("-", text)
    text = _PUNCT_TO_STRIP.sub(" ", text)
    text = text.lower()
    text = " ".join(text.split())
    return text
```

This same function is used in §8.3's drift guard so prose-side and label-side
normalization always agree.

**Lookup behavior:**

- Returns `None` when no entry exists; the synthesizer skips that row.
- Pass-conditional: `"Range"` returns `"max_intercept_km"` for
  `missile_kinematics` and `None` for other passes.
- Section-conditional: `"Weight kg"` returns `"booster_mass_kg"` only when
  `section_ctx == "1st Stage"` AND `active_pass == "missile_propulsion"`.

### 5.6 `coerce_value(cell_text, schema_field) → list[ParsedValue]` (D3 + D4)

`ParsedValue` defined in §5.0.

**Numeric fields** (`*_kg`, `*_m`, `*_km`, `*_sec`, `*_mps`, `*_dbi`, `*_mhz`,
`*_deg`, `*_kw`, `*_dbw`, etc.):

1. Strip and normalize cell text (whitespace + dash collapse only — do NOT
   apply full `normalize_label`; cell values may legitimately contain
   punctuation like decimal points).
2. Classify the cell into one of three multi-value patterns. **The
   classification policy is explicit and the synthesizer must distinguish
   them; this is not "default emit all":**

   | Pattern | Examples | Emission policy |
   |---|---|---|
   | **Range** (single physical quantity, two endpoints) | `"4–6 sec"`, `"29–34 km"`, `"4 to 6"`, `"4-6"` | Emit ONE fact with the **midpoint** value (and `unit_inferred` set; `multi_value_emissions` does NOT increment for ranges). Annotate `raw_text` with the original range string. The model needs a single value per `(entity, field)`; emitting both endpoints would tell the LLM the missile has two booster burn times. Acceptable lossy compression. |
   | **Discrete alternatives** (two distinct measurements with separator `/`) | `"1135/1028"`, `"100 / 120"` | Emit TWO facts, each as its own `ParsedValue`. Increment `multi_value_emissions`. Caller (`emit_fact`) emits TWO TextItems. The LLM sees two candidate values and chooses based on context. |
   | **Single value** (no range, no alternative separator) | `"1135"`, `"4.0 sec"`, `"885 m/s"` | Emit ONE fact. |

   **Range-vs-alternatives heuristic:**
   - Separator is en-dash (`–`), em-dash (`—`), or the literal word `"to"` → range.
   - Separator is forward slash (`/`) → alternatives.
   - Separator is ASCII hyphen-minus (`-`): ambiguous. Disambiguate by
     numeric ordering — if `X < Y`, treat as range; if `X >= Y`, treat as
     alternatives. (`"100-120"` X<Y → range; `"1135-1028"` X>Y → alternatives.)
   - More than two values (`"100/110/120"`): treat as alternatives, emit all.

3. Parse number + unit from each value-fragment. Unit comes from explicit
   cell content (`"1135 kg"`) OR from the row label (`"Length mm"` → `"mm"`).
   When both are present and disagree, cell-content wins.
4. Coerce via `UNIT_TABLE`. Schema-field suffix selects the unit class
   (`*_kg` → `mass_kg`); the cell-extracted unit string selects the
   conversion factor within that class.
5. Return `[]` (skip) if cell empty, unit absent and no implied unit, unit
   unknown to `UNIT_TABLE`, or value won't parse as number.

**String fields** (`*_thrust`, `system_name`): pass through verbatim, single-
element list. Multi-value detection does not apply.

**Stop-words** (return `[]`):
- Empty string `""`
- ASCII dash variants: `"-"`, `"--"`
- Unicode dashes: `"–"` (en), `"—"` (em), `"―"` (horizontal bar)
- Words: `"TBD"`, `"N/A"`, `"NA"`, `"unknown"`, `"unk"`, `"none"` (case-insensitive)
- Question/uncertainty markers: `"?"`, `"???"`

### 5.7 `emit_fact(entity_id, schema_field, value, source_label) → TextItem`

Output text format:

```
"{entity_id} — {schema_field} = {value} [source: {source_label} row of variants table]"
```

Concrete:

```
"1D — booster_mass_kg = 1135 [source: 1st Stage Weight kg row of variants table]"
```

Schema-keyed prefix ensures the LLM sees the canonical field name and can
match it directly to its schema. The bracketed source preserves traceability
without forcing the LLM to re-derive it.

TextItem skeleton mirrors the schema-validation fix from b9fe407:

```python
{
    "self_ref": f"#/texts/{new_idx}",
    "parent": {"$ref": "#/body"},
    "children": [],
    "content_layer": "body",
    "label": "text",
    "prov": [],
    "orig": text,
    "text": text,
}
```

**Idempotence:** top-level `doc_json["__synthesized_table_facts__"] = True`
flag set on first run; second call short-circuits with
`stats=FactStats(idempotent_skip=True)`. Cleaner than per-item markers — single
guard, no parsing.

## 6. Data Flow (worked example)

`missile_propulsion` pass on the SA-2 PDF, `tables[]` containing the variants
table at index 0:

```
table[0]:
  detect_table_shape() → COLUMN_MAJOR
  extract_label_rows() → 15 LabelRow records, 10 entity columns (cols 2-11)
    Row labels: ["Industry Designation", "Military Designation", "Missile Type",
                 "Max Range m", "Max Alt m", "Min Alt m", "Min Range m",
                 "Length mm", "Body Diameter mm", "Total Weight kg",
                 "1st Stage Weight kg", "1st Stage Time sec", "1st Stage Thrust",
                 "2nd Stage Weight kg", "2nd Stage Time sec"]
  detect_section_context() →
    Rows 0-9: section_ctx=None
    Rows 10-12: section_ctx="1st Stage" (embedded)
    Rows 13-14: section_ctx="2nd Stage" (embedded)

  Per (entity_col=2 (1D), row, section_ctx) for missile_propulsion:
    ("Max Range m", None, missile_propulsion)
      → resolve_alias() → None [propulsion pass excludes kinematics labels]
      → SKIP, increment rows_skipped_unresolvable

    ("1st Stage Weight kg", "1st Stage", missile_propulsion)
      → resolve_alias() → "booster_mass_kg"
      → coerce_value("1135", "booster_mass_kg") → [ParsedValue(1135, "kg", 1.0, "1135")]
      → emit_fact() → "1D — booster_mass_kg = 1135 [source: 1st Stage Weight kg row of variants table]"

    ("1st Stage Time sec", "1st Stage", missile_propulsion) → "booster_time_sec" = 4.0
    ("1st Stage Thrust",   "1st Stage", missile_propulsion) → "booster_thrust"  = "..." (verbatim)
    ("2nd Stage Weight kg", "2nd Stage", missile_propulsion) → "sustain_mass_kg" = 1028
    ("2nd Stage Time sec",  "2nd Stage", missile_propulsion) → "sustain_time_sec" = ...

  Loop over 10 entity cols × ~5 propulsion-relevant rows = ~50 fact attempts.
  Realistic emission: ~30–40 (some cells empty / unparseable → skip).

  --- Graceful-fail rows (illustrating the §5.6 unit-conversion + §7 skip paths) ---

  ("Length mm", None, missile_propulsion)
    → resolve_alias() → None [propulsion pass excludes airframe labels]
    → SKIP, increment rows_skipped_unresolvable

  ("Length mm", None, missile_airframe)  [a different pass on the same table]
    → resolve_alias() → "body_length_m"
    → coerce_value("10726", "body_length_m"):
        unit class for "_m" = length_m; cell has no explicit unit;
        row label "Length mm" implies "mm"; UNIT_TABLE["length_m"]["mm"] = 0.001
        → [ParsedValue(value=10.726, unit_inferred="mm", conversion_factor=0.001, raw_text="10726")]
    → emit_fact() → "1D — body_length_m = 10.726 [source: Length mm row of variants table]"

  ("Diameter mm", None, missile_airframe)
    cell text = "TBD"
    → coerce_value() → [] (stop-word match)
    → SKIP, increment values_skipped_unparseable

  ("1st Stage Burn", "1st Stage", missile_propulsion)
    cell text = "4–6 sec"
    → resolve_alias() → "booster_time_sec"
    → coerce_value("4-6 sec", "booster_time_sec"):
        range pattern (4 < 6, en-dash variant); midpoint 5.0; unit "sec" matches
        → [ParsedValue(value=5.0, unit_inferred="sec", conversion_factor=1.0, raw_text="4–6 sec")]
    → emit_fact() → "1D — booster_time_sec = 5.0 [source: 1st Stage Burn row of variants table]"
        (multi_value_emissions NOT incremented — range collapsed to midpoint)
```

After synthesis the LLM sees these facts in whatever chunk the chunker places
them in:

```
... [original document text] ...

1D — booster_mass_kg = 1135 [source: 1st Stage Weight kg row of variants table]
1D — booster_time_sec = 4.0  [source: 1st Stage Time sec row of variants table]
1D — sustain_mass_kg = 1028 [source: 2nd Stage Weight kg row of variants table]
13D — booster_mass_kg = 1032 [source: 1st Stage Weight kg row of variants table]
...
```

Extraction becomes classification ("does this fact match my schema? if so,
copy") rather than table-arithmetic. No row counting. No alias mapping. No
section disambiguation. The hard work is done deterministically at synthesis
time.

## 7. Error Handling

Synthesis is best-effort. Any failure mode falls back to "skip this fact" —
never crashes, never blocks `/extract-pass`.

| Failure mode | Behavior | Stats counter | Log level |
|---|---|---|---|
| Empty cell text | Skip silently | `values_skipped_unparseable` | none |
| Cell value `"TBD"`, `"—"`, `"N/A"`, `"unknown"` | Skip silently | `values_skipped_unparseable` | DEBUG |
| Numeric field, value won't parse | Skip | `values_skipped_unparseable` | INFO |
| Numeric field, unit absent and label has no implied unit | Skip | `values_skipped_unparseable` | INFO |
| Unit unknown to `UNIT_TABLE` | Skip | `values_skipped_unparseable` | INFO |
| Multi-value cell, one of N values fails | Emit parseable values; skip the failed | `multi_value_emissions++`, `values_skipped_unparseable++` | INFO |
| Row label not in `ALIAS_MAP` for active pass | Skip silently | `rows_skipped_unresolvable` | none |
| Row label in map but section_ctx mismatches | Skip silently | `rows_skipped_unresolvable` | DEBUG |
| Section detection conflict (embedded vs header-row) | Embedded wins | `sections_detected` (embedded only) | DEBUG |
| HYBRID composite identity collision | Last one wins; previous overwritten | `hybrid_collisions++` | WARNING |
| `max_synthesized` cap reached | Stop emission, return early | `truncated_at_cap=True` | WARNING |
| Idempotence flag set on entry | Return doc unchanged | `idempotent_skip=True` | INFO |
| Synthesizer raises any exception | Caller catches, logs WARNING, continues with original doc | n/a | WARNING + traceback |
| `active_pass` unknown | Skip whole call, return doc unchanged | n/a | WARNING |
| `doc_json` malformed | Return doc preserved as-is, no facts emitted | `tables_seen=0` | DEBUG |

**Caller-side guard** (in `main.py`):

```python
try:
    docling_document_json, fact_stats = synthesize_table_facts(
        docling_document_json, active_pass=pass_name,
    )
    if fact_stats.facts_emitted > 0:
        logger.info("GRAPH_EXTRACTION_FACTS pass=%s ...", ...)
except Exception as exc:
    logger.warning("synthesize_table_facts failed pass=%s: %s — continuing", pass_name, exc)
    fact_stats = FactStats.empty()
```

The synthesizer is non-critical: it augments the document with hints, but the
original chunker path still runs on whatever's there. A synthesizer bug never
breaks `/extract-pass`.

## 8. Testing

### 8.1 Unit tests (`test_table_facts.py`)

| Function | Cases |
|---|---|
| `detect_table_shape` | column-major / row-major / hybrid / OTHER (below 4×4) |
| `extract_label_rows` | column-major / row-major transposition / hybrid composite identity |
| `detect_section_context` | embedded match / header-row tracking / conflict (embedded wins) / no section |
| `resolve_alias` | exact triple match / pass-conditional skip / section-conditional / unknown label → None |
| `coerce_value` | numeric + explicit unit / numeric + implied unit / unit conversion / multi-value `X/Y` / range `X–Y` / `"TBD"` → `[]` / unparseable → `[]` / non-numeric passthrough |
| `emit_fact` | TextItem schema completeness / source-label preservation / entity_id formatting |

### 8.2 Integration tests (`test_table_facts.py`)

- Synthetic SA-2-shaped column-major table (10 entity cols × 15 spec rows,
  section keywords embedded) → for each of 4 passes, assert correct facts
  emitted with correct values (full coverage of expected propulsion facts).
- Synthetic row-major table (5 rows × 4 cols) → entities-as-rows path.
- HYBRID multi-row identity → composite `entity_id`.
- Idempotence: call twice, second returns unchanged with `idempotent_skip=True`.
- `max_synthesized` cap: 200 facts attempted, cap=10 → emits 10, sets
  `truncated_at_cap=True`.

### 8.3 Drift guard (`test_alias_map.py`)

A literal substring assertion (`normalize_label(label) in normalize_label(DELTA_SYSTEM_PROMPT)`)
will not work for stage-conditional aliases — for example, the §12b prose says
*"Under a `1st Stage` or `Booster` section, `Weight`/`Mass` maps to
`booster_mass_kg`"* with the section keyword and the spec keyword in separate
backtick-quoted tokens, not as the contiguous string `"1st stage weight"`. The
ALIAS_MAP key for that entry would normalize to `"weight"` (label only,
without section prefix; the section is the second key-tuple element). So we
check label tokens and section keywords independently.

```python
import re
from app._alias_map import ALIAS_MAP, SECTION_KEYWORDS
from app._table_facts import normalize_label
from ontology_bundles._shared.prompt_rules import DELTA_SYSTEM_PROMPT


def test_alias_map_labels_appear_in_prompt_rule():
    """Every ALIAS_MAP key's label_normalized appears as a token in §12b
    prose. Catches drift where a new alias is added to the structured map
    but the LLM never gets told about it."""
    prose_normalized = normalize_label(DELTA_SYSTEM_PROMPT)
    # Tokenize the normalized prose into whitespace-separated tokens.
    prose_tokens = set(prose_normalized.split())

    for (label_norm, _section, _pass), _field in ALIAS_MAP.items():
        # Each space-separated token of the normalized label must appear
        # as a token in the prose. "weight" and "kg" both must be present.
        for token in label_norm.split():
            assert token in prose_tokens, (
                f"Token {token!r} from ALIAS_MAP label {label_norm!r} "
                f"missing from §12b prose tokens. Either add it to §12b "
                f"or drop it from ALIAS_MAP."
            )


def test_section_keywords_appear_in_prompt_rule():
    """Every SECTION_KEYWORDS entry appears (as a contiguous phrase, since
    they are short stable strings) in §12b prose."""
    prose_normalized = normalize_label(DELTA_SYSTEM_PROMPT)
    for keyword in SECTION_KEYWORDS:
        keyword_norm = normalize_label(keyword)
        assert keyword_norm in prose_normalized, (
            f"Section keyword {keyword!r} (normalized {keyword_norm!r}) "
            f"missing from §12b prose. Add it to §12b before adding to "
            f"SECTION_KEYWORDS."
        )


def test_alias_map_target_fields_exist_on_schemas():
    """Every ALIAS_MAP value (target schema field) must exist as a field on
    the schema for the corresponding pass. Catches drift where a schema is
    refactored and the alias map points at a renamed/removed field."""
    from app.bundles import load_pass_template
    by_pass: dict[str, set[str]] = {}
    for (_label, _section, pass_name), schema_field in ALIAS_MAP.items():
        by_pass.setdefault(pass_name, set()).add(schema_field)
    for pass_name, fields in by_pass.items():
        template_cls = load_pass_template("air_defense_v3", pass_name)
        # Walk the model recursively to collect every field name that exists
        # on any nested entity. Same pattern used by auto-evidence.
        actual_fields = _collect_all_field_names(template_cls)
        missing = fields - actual_fields
        assert not missing, (
            f"ALIAS_MAP entries for pass {pass_name!r} reference fields "
            f"{missing!r} that do not exist on the schema."
        )
```

The token-based label check tolerates the prose's natural grammar (separate
backticked tokens for label and section) while still catching the cases that
matter: a new alias whose label does not appear anywhere in the prose, or a
typo (`"booster_mass_kg"` → `"booser_mass_kg"`). The section-keyword check
uses the contiguous form because section keywords are short, stable
multi-word phrases (`"1st Stage"`, `"2nd Stage"`) that should appear
verbatim. The third test catches schema-side drift independently of §12b.

### 8.4 Prompt-content test (CI proxy for end-to-end)

```python
def test_synthesized_facts_appear_in_extract_pass_prompt():
    """End-to-end smoke test: synthesizer runs, facts land in the
    user-message of the LLM prompt, schema field names are present
    in the exact emitted format."""
    # Mock OllamaChatClient.post to capture the rendered prompt.
    # Run /extract-pass against a fixture DoclingDocument with the SA-2-shaped
    # column-major table from §8.2 (10 entity cols × 15 spec rows, section
    # keywords embedded), pass_name="missile_propulsion".
    captured = mock_post.call_args.kwargs["messages"]
    user_msg = next(m for m in captured if m["role"] == "user")["content"]

    # Assert on the exact emit_fact format — catches both presence AND
    # format drift (a planner choosing a weaker substring like
    # "booster_mass_kg" without the entity_id or value would not catch
    # an entity_id formatting regression).
    assert "1D — booster_mass_kg = 1135 [source: 1st Stage Weight kg row of variants table]" in user_msg
    assert "13DM — sustain_mass_kg = 1251 [source: 2nd Stage Weight kg row of variants table]" in user_msg
    assert "13DM — booster_mass_kg = 1032 [source: 1st Stage Weight kg row of variants table]" in user_msg
```

Catches integration regressions without requiring a live LLM. Runs in ~5s.
The exact-string assertions above are the operative test of the §5.7 emit
format — change the format and these tests fail loud.

### 8.5 End-to-end empirical validation (operator-driven, not CI)

The §20 notebook cell at `T=1.0` is the headline test. After deploy:

- **Acceptance:** `missile_propulsion` ✓ exact ≥ 6 of the 7 listed variants
  where alias-only T=1.0 produced wrong values for `booster_mass_kg`. The 7
  listed: **13DM, 13DA, 13DAM, 20D, 20DP, 20DSU, 5Ya23**. Threshold is a
  lower bound — 6, 7, or all 7 plus other GT entries pass; ≤ 5 fails. One
  variant is permitted to remain wrong/null because of residual non-deterministic
  LLM behavior even with deterministic facts in the prompt.
- **No regression criterion** (also acceptance):
  - `missile_kinematics` ✓ exact ≥ 4 (alias-only T=1.0 baseline = 4 — see §8.5
    Baseline Reference below).
  - `missile_airframe` ✓ exact ≥ 8 (baseline = 8).
  - `missile_speed_timing` ✓ exact ≥ 6 (baseline = 6).
- Wall-time delta ≤ +20% per pass (synthesized texts add some chunks but
  shouldn't double the work). Measured against alias-only T=1.0 baseline on
  the same Ollama-host topology — **the baseline must be re-measured if the
  number of model hosts changes between baseline and validation runs.**

Each `/extract-pass` call is ~25 minutes and depends on a live Ollama, which is
unsuitable for CI. CI uses the prompt-content test in §8.4; the §20 run is the
human-in-the-loop empirical check.

**Baseline Reference:**
The alias-only T=1.0 baseline values cited above (4 / 8 / 6 / 0 ✓ exact for
kinematics/airframe/speed_timing/propulsion) come from the §20 GT scorecard
run logged in the 2026-05-05 session: results paste of the alias-patch sweep
(20 missile passes against the SA-2 PDF), entity counts 41/43/39/40
respectively. Cached pass-output JSON files preserved at
`/tmp/r21_alias_only_backup/` inside `eip-mmdpp-jupyter` for direct comparison
during validation. If those files have been cleaned up before validation
runs, re-derive the baseline by running §20 with the synthesizer disabled
(comment out the §9.1 wire-up block) before judging the synthesizer's deltas.

## 9. Integration

### 9.1 Wire-up in `main.py`

```python
# Imports
- from app._table_pivot import synthesize_pivoted_table_texts
+ from app._table_facts import synthesize_table_facts, FactStats

# run_extraction_pass body (replacing B1+B2 block)
- docling_document_json, _pivoted_count = synthesize_pivoted_table_texts(
-     docling_document_json
- )
- if _pivoted_count > 0:
-     logger.info("GRAPH_EXTRACTION_PIVOTED pass=%s synthesized=%d ...", ...)

+ try:
+     docling_document_json, fact_stats = synthesize_table_facts(
+         docling_document_json,
+         active_pass=pass_name,
+     )
+     if fact_stats.facts_emitted > 0:
+         logger.info(
+             "GRAPH_EXTRACTION_FACTS pass=%s facts=%d tables=%d sections=%d "
+             "skipped_unresolvable=%d unparseable=%d shapes=%s",
+             pass_name, fact_stats.facts_emitted, fact_stats.tables_seen,
+             fact_stats.sections_detected, fact_stats.rows_skipped_unresolvable,
+             fact_stats.values_skipped_unparseable, fact_stats.tables_by_shape,
+         )
+ except Exception as exc:
+     logger.warning(
+         "synthesize_table_facts failed pass=%s: %s — continuing with original doc",
+         pass_name, exc,
+     )
+     fact_stats = FactStats.empty()

# Diagnostics surfacing (extends existing dict at line ~1150)
diagnostics["service_table_facts"] = fact_stats.as_dict()
```

### 9.2 Deprecation pattern

- This PR: `_table_pivot.py` docstring updated to `DEPRECATED — replaced by
  _table_facts.py`. Module not imported by `main.py`. `tests/test_table_pivot.py`
  preserved as regression for the deprecated path.
- Next PR (after one cycle of green production): both `_table_pivot.py` and
  `tests/test_table_pivot.py` deleted.

### 9.3 Container rebuild

Required for the change to land:

```bash
docker compose build docling-graph && docker compose up -d docling-graph
```

### 9.4 Rollback

**Strategy: single-commit revert + container rebuild.** A commit reverting
§9.1's diff restores the prior `main.py` (no `_table_facts` import, original
B1+B2 block — currently the disabled state). The new files
(`_table_facts.py`, `_alias_map.py`) stay on disk; unimported = inert.
Container rebuild (`docker compose build docling-graph && docker compose up -d
docling-graph`) is required because the import set changes.

This is a clean rollback: one revert, one rebuild, no feature flags, no
dual-import staging cruft. The rebuild adds ~30 seconds to the rollback
sequence — acceptable given the operational simplicity. We are NOT staging
both code paths under a feature flag.

### 9.5 Observability

Operator running `docker logs eip-mmdpp-docling-graph-1 | grep
GRAPH_EXTRACTION_FACTS` sees one line per `/extract-pass` call with the per-pass
synthesis volume. Combined with `IDENTITY_FILTER` and existing logs, the
pipeline's per-stage health remains traceable end-to-end.

## 10. Acceptance Criteria

1. `synthesize_table_facts` is wired into `main.py` with the catch-and-continue
   guard.
2. All unit, integration, drift-guard, and prompt-content tests pass in CI.
3. `_table_pivot.py` marked DEPRECATED but preserved with its test for one
   cycle.
4. Operator-driven §20 cell at T=1.0 with the new synthesizer produces
   `missile_propulsion ✓ exact ≥ 6 of the 7 listed variants` (13DM, 13DA,
   13DAM, 20D, 20DP, 20DSU, 5Ya23) per §8.5.
5. Wall-time delta per pass ≤ +20% vs alias-only T=1.0 baseline measured on
   the same Ollama-host topology (entity_count neutral or improved).
6. No regression on `missile_kinematics ≥ 4`, `missile_airframe ≥ 8`,
   `missile_speed_timing ≥ 6` ✓ exact counts (alias-only T=1.0 baseline cited
   in §8.5 Baseline Reference).
7. `IDENTITY_FILTER` drop counts remain in the same range (1–5 per pass) — the
   synthesizer should not produce noise that gets gated.

## 11. Implementation Order (suggested)

1. **Types module** (§5.0) — `Shape` enum, `LabelRow`, `ParsedValue`,
   `FactStats`, `AliasKey`, `SectionContext` — declared first because every
   subsequent step references them. Lives at the top of `_table_facts.py` (or
   in a tiny `_table_facts_types.py` if circular imports demand it).
2. `_alias_map.py` — structured data + `normalize_label` function + drift-guard
   test. Pre-requisite for steps 4 and 5.
3. `_table_facts.py` skeleton — `detect_table_shape`, `extract_label_rows`,
   `derive_entity_ids` with unit tests.
4. `detect_section_context` + `resolve_alias` with unit tests. Depends on (2)
   for alias data.
5. `coerce_value` + `emit_fact` with unit tests (multi-value + unit conversion
   are the most surface-area).
6. Top-level `synthesize_table_facts` orchestrator + integration tests
   (depends on 3–5).
7. `main.py` wire-up + prompt-content test.
8. Deprecation marker on `_table_pivot.py`.
9. Container rebuild, deploy, run §20, verify acceptance.

**Parallelizability:** once steps 1 and 2 land, steps 3, 4, and 5 can proceed
in parallel since each is a self-contained pure function with its own unit
tests. Steps 6–9 are sequential.
