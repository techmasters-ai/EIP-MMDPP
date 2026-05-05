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
  ↓ detect_section_context(rows)                # D2
  ↓ for each (entity_col, row, section_ctx):
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

### 5.1 `_alias_map.py` — structured alias map

```python
ALIAS_MAP: dict[AliasKey, str]
# AliasKey = (label_normalized: str, section_ctx: str | None, pass_name: str)
# Value: canonical schema field name (e.g., "booster_mass_kg")

SECTION_KEYWORDS: tuple[str, ...]
# "1st Stage", "2nd Stage", "Booster", "Sustainer", "Sustain", "Ejector", ...
# Extensible per domain.

UNIT_TABLE: dict[str, dict[str, float]]
# Per-unit-class conversion factors keyed by canonical-field suffix.
# {"_m": {"mm": 0.001, "cm": 0.01, "in": 0.0254, "ft": 0.3048, "km": 1000}, ...}

FIELD_SUFFIX_TO_UNIT_CLASS: dict[str, str]
# {"_m": "length_m", "_kg": "mass_kg", "_sec": "time_sec", "_mps": "velocity_mps",
#  "_km": "length_km", "_dbi": "gain_dbi", "_mhz": "frequency_mhz", ...}
```

Keyed on the (label, section, pass) triple so pass- and section-conditionals
are first-class. A drift-guard unit test asserts every entry has a corresponding
§12b prose mention.

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

Normalizes column-major and row-major into the same intermediate shape:

```python
LabelRow = TypedDict({
    "row_idx": int,
    "label_text": str,
    "label_col_span": int,
    "data_cells": dict[int, str],  # entity_col → cell text
})
```

- **COLUMN_MAJOR:** today's logic in `_table_pivot.py` (leftmost label cols,
  remaining data cols).
- **ROW_MAJOR:** transposed equivalent.
- **HYBRID:** combine multi-row identity labels into a composite — e.g., row 0
  `"Industry Designation"` + row 1 `"Missile Type"` produces `entity_id =
  "S-75 1D"` (concatenated).

### 5.4 `detect_section_context(rows) → list[(LabelRow, SectionContext)]` (D2)

Two-strategy chain (in order):

1. **Embedded:** substring scan of `label_text` against `SECTION_KEYWORDS`. If
   matched, that row's `section_ctx` is the matched keyword.
2. **Header-row:** track most recent row whose `label_text` is a bare section
   keyword AND whose `data_cells` are empty/header-like; subsequent rows inherit
   that section until the next header-row or end-of-table.

Default: `None` if neither matches. Rows with `None` section context can still
resolve aliases that don't require sectioning (e.g., `total_mass_kg`).

**Conflict resolution:** embedded wins (most-specific signal). Header-row
context applies only when the row has no embedded section keyword.

### 5.5 `resolve_alias(label, section_ctx, active_pass) → str | None`

Lookup `ALIAS_MAP[(normalize(label), section_ctx, active_pass)]`.

- `normalize` collapses whitespace, lowercases, strips punctuation.
- Returns `None` when no entry exists; the synthesizer skips that row.
- Pass-conditional: `"Range"` returns `"max_intercept_km"` for
  `missile_kinematics` and `None` for other passes.
- Section-conditional: `"Weight kg"` returns `"booster_mass_kg"` only when
  `section_ctx == "1st Stage"` AND `active_pass == "missile_propulsion"`.

### 5.6 `coerce_value(cell_text, schema_field) → list[ParsedValue]` (D3 + D4)

```python
ParsedValue = TypedDict({
    "value": float | str,
    "unit_inferred": str | None,
    "conversion_factor": float,  # 1.0 if no conversion applied
    "raw_text": str,
})
```

**Numeric fields** (`*_kg`, `*_m`, `*_km`, `*_sec`, `*_mps`, `*_dbi`, `*_mhz`,
`*_deg`, `*_kw`, `*_dbw`, etc.):

1. Strip and normalize cell text.
2. Detect multi-value patterns (`X/Y`, `X–Y`, `X to Y`, `X-Y`); each match
   produces a separate `ParsedValue`. Default emits all values as facts.
3. Parse number + unit from the cell. Unit comes from explicit cell content
   (`"1135 kg"`) OR from the row label (`"Length mm"` → `"mm"`).
4. Coerce via `UNIT_TABLE` keyed by schema-field suffix.
5. Return `[]` (skip) if cell empty, unit absent and no implied unit, unit
   unknown to `UNIT_TABLE`, or value won't parse as number.

**String fields** (`*_thrust`, `system_name`): pass through verbatim, single-
element list.

**Stop-words** (`"TBD"`, `"—"`, `"N/A"`, `"unknown"`, `""`): return `[]`.

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
| HYBRID composite identity collision | Last one wins; previous overwritten | `tables_by_shape["HYBRID_COLLISION"]++` | WARNING |
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

```python
def test_alias_map_entries_have_prompt_rule_mentions():
    """Every ALIAS_MAP entry's source label appears in §12b prose of
    DELTA_SYSTEM_PROMPT. Catches drift between the structured map (synthesizer
    SSoT) and the LLM-facing prose (LLM SSoT)."""
    from app._alias_map import ALIAS_MAP, SECTION_KEYWORDS
    from ontology_bundles._shared.prompt_rules import DELTA_SYSTEM_PROMPT
    for (label, _section, _pass), _field in ALIAS_MAP.items():
        assert _normalize(label) in _normalize(DELTA_SYSTEM_PROMPT), \
            f"label {label!r} missing from §12b prose"
    for keyword in SECTION_KEYWORDS:
        assert _normalize(keyword) in _normalize(DELTA_SYSTEM_PROMPT)
```

### 8.4 Prompt-content test (CI proxy for end-to-end)

```python
def test_synthesized_facts_appear_in_extract_pass_prompt():
    """End-to-end smoke test: synthesizer runs, facts land in the
    user-message of the LLM prompt, schema field names are present."""
    # Mock OllamaChatClient.post to capture the rendered prompt.
    # Run /extract-pass against a fixture DoclingDocument with a known table.
    # Assert the captured user message contains
    # "1D — booster_mass_kg = 1135"
```

Catches integration regressions without requiring a live LLM. Runs in ~5s.

### 8.5 End-to-end empirical validation (operator-driven, not CI)

The §20 notebook cell at `T=1.0` is the headline test. After deploy:

- **Acceptance:** `missile_propulsion` ✓ exact ≥ 6 (recovers at minimum the 6
  variants where alias-only T=1.0 produced wrong values: 13DM, 13DA, 13DAM,
  20D, 20DP, 20DSU, 5Ya23 — `booster_mass_kg`).
- Wall-time delta ≤ +20% per pass (synthesized texts add some chunks but
  shouldn't double the work).

Each `/extract-pass` call is ~25 minutes and depends on a live Ollama, which is
unsuitable for CI. CI uses the prompt-content test in §8.4; the §20 run is the
human-in-the-loop empirical check.

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

Single-commit revert of the `main.py` integration block. The new files
(`_table_facts.py`, `_alias_map.py`) stay on disk — unimported = inert. No
stack-wide rebuild needed if we keep the B1+B2 import in `main.py` available
during rollout (currently disabled).

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
   `missile_propulsion ✓ exact ≥ 6` against the GT scorecard.
5. Wall-time delta per pass ≤ +20% vs alias-only T=1.0 baseline (entity_count
   neutral or improved).
6. No regression on `missile_kinematics`, `missile_airframe`, or
   `missile_speed_timing` ✓ exact counts (4, 8, 6 respectively at T=1.0 in the
   alias-only baseline).
7. `IDENTITY_FILTER` drop counts remain in the same range (1–5 per pass) — the
   synthesizer should not produce noise that gets gated.

## 11. Implementation Order (suggested)

1. `_alias_map.py` — structured data + drift-guard test (no logic, easy first
   step).
2. `_table_facts.py` skeleton — `detect_table_shape`, `extract_label_rows`
   with unit tests.
3. `detect_section_context` + `resolve_alias` with unit tests.
4. `coerce_value` + `emit_fact` with unit tests (multi-value + unit conversion
   are the most surface-area).
5. Top-level `synthesize_table_facts` orchestrator + integration tests.
6. `main.py` wire-up + prompt-content test.
7. Deprecation marker on `_table_pivot.py`.
8. Container rebuild, deploy, run §20, verify acceptance.
