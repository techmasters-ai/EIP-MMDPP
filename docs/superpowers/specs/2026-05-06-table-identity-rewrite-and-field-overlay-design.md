# Table-Derived Identity Rewrite + Per-Cell Field Overlay (Mechanism A1)

**Status:** Draft (in user review — r6)
**Predecessor (parked):** `docs/superpowers/specs/2026-05-05-section-aware-table-fact-synthesis-design.md`
**Reuses:** `docker/docling-graph/app/_table_facts.py` parser primitives (already on disk, 99 tests passing)
**Related future work:** Mechanism B (per-variant prose binding accuracy for V-75-style docs) — separate spec

## 1. Problem

Field extraction across missile/radar entities is bottlenecked by two distinct
patterns. This spec addresses **only the first**.

**Pattern A — Within-column alias scatter (variants tables):**

The SA-2 PDF has a column-major variants table. Each column is one missile
variant. Within a column, cells across multiple identity rows give different
designations FOR THE SAME variant:

```
Column 0:
  Industry Designation = "SA-75"
  Military Designation = "SA-75"
  NATO Designation     = "SA-2A"
  Fan Song Variant     = "RSNA-75"   ← actually a radar reference, NOT a missile alias
  Missile Type         = "1D"        ← canonical
```

The LLM emits multiple `MISSILE_SYSTEM` vertices for this single variant
(under names "SA-75", "SA-2A", "1D"). Field values land on different aliases.
Empirical example from §20 alias-only baseline:

| Vertex | Fields populated |
|---|---|
| `1D` (Missile Type) | body_length_m=10.726, body_diameter_m=0.654 |
| `SA-2A` (NATO) | total_mass_kg=2163 |
| `SA-75` (Industry) | (nothing) |

If these collapsed onto one canonical `1D` vertex, that vertex would carry
both body dimensions AND mass. As-is, queries against the `1D` vertex miss
the mass; queries against `SA-2A` miss the body dimensions.

**Pattern B (NOT addressed here) — Cross-variant prose binding:**

In the V-75 doc (no variants table), 6 distinct variants are mentioned in
prose: V-75, SA-2, SA-2E, HQ-2B, HQ-2J, S-75. These are NOT aliases of one
variant — they are different physical variants of the same family. Each must
remain a distinct entity. The problem is the LLM mis-binds extracted field
values across them. This is a different problem class, addressed by a future
Mechanism B (per-variant constrained extraction or larger LLM); not in scope
here.

## 2. Goals

1. **Recover scattered fields onto canonical entities** for docs with
   column-major variants tables. Each canonical missile variant carries the
   union of fields its aliases held individually, plus deterministic table
   facts where the LLM had nulls.
2. **Maintain entity granularity.** Different columns (1D vs 13D vs 13DM) stay
   distinct. Cross-entity references (Fan Song Variant in a missile table)
   stay distinct AND emit relationship hints, not collapse into the missile.
3. **Bound wall time.** Parser is milliseconds; overlay is milliseconds. Total
   added cost on table-bearing docs ≤ +5%; zero on no-table docs.
4. **Generalize structurally.** A new doc with the same shape (column-major
   variants table with identity rows) works with the same code, no
   domain-specific changes beyond adding to `MISSILE_IDENTITY_LABELS` /
   `CANONICAL_PRIORITY` if new label patterns appear.

## 3. Non-Goals (deferred)

- **Mechanism B** (per-variant prose binding for V-75-style docs)
- **Prose-pattern alias detection** (e.g., parsing "X (also Y, NATO designation Z)" from narrative). Defer until empirical need; tables give bounded equivalence classes, prose does not.
- **Row-major variants tables.** Different shape, separate handling. Add when corpus has examples.
- **Cross-entity relationship pass.** `cross_entity_hints` are collected for the Fan Song / radar association but NOT applied as `ASSOCIATED_WITH` edges in this spec. Reserved for a follow-up that integrates with the existing relationship-pass pipeline.
(Cross-entity relationship-edge writes from `cross_entity_hints` are also
out of scope — see the existing entry above.)
- **Multi-table docs.** A single doc with two distinct variants tables (e.g., a future doc with both a missile variants table AND a separate radar variants table) — **`extract_table_overlay` (parser, §5.2) enforces "first STRICTLY-QUALIFYING table wins"**. The qualification gate (and ALL multi-table acceptance) is a hard AND of:
    1. Table classified as `column_major` or `hybrid` shape (existing detector in `_table_facts.py`).
    2. **`entity_columns ≥ 4`** — at least 4 data columns to the right of the label region (i.e. the table is wide enough to carry a real variants set, not a 2-column spec sheet that happens to look column-major).
    3. **`identity_rows ≥ 1`** — at least one row in column 0 whose label normalizes to a `MISSILE_IDENTITY_LABELS` or `RADAR_IDENTITY_LABELS` entry (so we know it's keyed on entity identity, not just column-of-numbers).
    4. **Each entity column has a non-empty cell in at least one of the matched identity rows** — kills tables where the identity row is sparse and only one column can be aliased.
   A table that fails ANY of (1)-(4) is NOT a candidate; it is silently passed over (counted in `tables_skipped_unqualified`), NOT in `tables_skipped_multi`. Only candidates that pass all four enter the "first wins" race. Subsequent qualifying tables logged at INFO level (`tables_skipped_multi=N`) and skipped. The corpus survey shows zero docs with this shape today; revisit if a future doc requires both. The worker side (`apply_*` functions, `merge_and_resolve`) doesn't need to enforce — it only ever sees the parser's selected first qualifying table.

  *Why strict qualification matters:* the SA-2 PDF has many tables (radar specs, propulsion sub-tables, ejector specs). Without strict gating, a small early table that incidentally looks column-major can starve the real variants table at row 6+. The four-of-four AND keeps that from happening without baking in document-order assumptions.

## 4. Architecture

### 4.1 File layout

| File | Status | Purpose |
|---|---|---|
| `docker/docling-graph/app/_table_facts.py` | MODIFY (~+100 LOC) | Add `extract_table_overlay(doc_json) → TableOverlay`. Reuses existing parser primitives. New private helpers: `_classify_identity_row`, `_classify_cross_entity_ref`, `_extract_alias_clusters`, `_pick_canonical`. |
| `docker/docling-graph/app/_alias_map.py` | MODIFY (~+50 LOC) | Add `MISSILE_IDENTITY_LABELS`, `RADAR_IDENTITY_LABELS`, `CROSS_ENTITY_REF_PATTERNS`, `CANONICAL_PRIORITY` constants. |
| `docker/docling-graph/app/schemas.py` | MODIFY (~+30 LOC) | Add `TableOverlay`, `TableFact`, `CrossEntityHint` Pydantic models. Add `table_overlay: TableOverlay \| None` field to `ExtractPassResponse`. |
| `docker/docling-graph/app/main.py` | MODIFY (~+10 LOC) | Call `extract_table_overlay` after sanitize, before LLM extraction. Attach to response. Wrap in try/except (overlay parsing failure must not break extract-pass). |
| `app/services/table_overlay.py` | NEW (~200 LOC) | Two functions operating on Pydantic instances via `iter_entities_of_type`: `apply_identity_rewrite(pass_results, alias_map, ontology) -> RewriteStats` and `apply_field_overlay(pass_results, table_facts, *, policy="table_wins_for_table_facts") -> OverlayStats`. |
| `app/services/extraction_merge.py` | MODIFY (~+30 LOC) | `canonicalize_cross_pass_identities` accepts new `table_alias_map` argument; calls `apply_identity_rewrite` BEFORE its existing token-overlap loop. New separate call to `apply_field_overlay` from `merge_and_resolve` after canonicalization, before merge. |
| `app/workers/pipeline.py` | MODIFY (~+20 LOC) | `_call_extract_pass` reads `table_overlay` from response and stashes on `PassResult`. `merge_and_resolve` invocation passes through to `canonicalize_cross_pass_identities`. |

**Test files (NEW):**
- `docker/docling-graph/tests/test_table_overlay_extract.py` — parser-side unit tests
- `docker/docling-graph/tests/test_alias_map_overlay_constants.py` — drift guards on identity-label / canonical-priority constants
- `tests/unit/test_table_overlay_worker.py` — worker-side unit tests
- `tests/unit/test_extraction_merge_table_overlay.py` — integration tests on `canonicalize_cross_pass_identities`
- `tests/integration/test_table_overlay_end_to_end.py` — full pass_result-list → merge_and_resolve, on synthetic SA-2 fixture

### 4.2 Two operations cleanly separated

```
1. Identity rewrite (deterministic, alias collapse) ─ runs in
   canonicalize_cross_pass_identities (before existing token-overlap pass)
   ─ Inputs: Pydantic entity instances + entity-type-scoped alias_map
   ─ Output: instances with system_name aliases rewritten to canonical
   ─ Effect: when merge_and_resolve runs, multiple alias-instances collapse
              onto one MergedEntityRecord, fields union
   ─ Wall: O(N entities × M alias-lookups), milliseconds

2. Per-cell field overlay (deterministic, scoped table_wins gate) ─
   runs in merge_and_resolve as Phase 0.5 (between canonicalize and the
   merge loop)
   ─ Inputs: Pydantic instances (post-rewrite) + table_facts
   ─ Output: instances with table-derived field values overriding LLM
              values for fields covered by a TableFact; LLM values for
              other fields untouched
   ─ Wall: O(F facts × E entities), milliseconds
```

The two operations live in different functions deliberately: identity
rewrite consumes the existing canonicalization slot; field overlay slots
into merge_and_resolve so it sees the post-rewrite (and post-token-
overlap) entity instances ready for merge.

The token-overlap canonicalization pass remains as fallback for entities
not matched by tables (e.g., prose-only mentions). Token-overlap runs
AFTER the table-derived rewrite, so table aliases take precedence.

### 4.3 Kill switch

Single env var allows operator-controlled rollback without code change:

```yaml
# docker-compose.yml docling-graph service env block
DOCLING_GRAPH_TABLE_OVERLAY_ENABLED: ${DOCLING_GRAPH_TABLE_OVERLAY_ENABLED:-true}
```

When `false`, `extract_table_overlay` short-circuits to empty `TableOverlay`;
worker-side overlay is a no-op. Restart docling-graph + worker (env var
change) → fully reverted to pre-overlay behavior.

### 4.4 Coordination with concurrent plan: per-pass Celery tasks

A separate plan (`docs/superpowers/plans/2026-05-06-per-pass-celery-tasks.md`)
restructures the worker's `derive_ontology_graph` Celery task into per-pass
tasks plus a fan-in merge callback. That plan introduces a new
`ingest.pipeline_pass_outputs` table that persists each pass's `pass_output`,
metadata, and provenance, with the merge step loading from the DB rather
than from an in-memory `dict[str, PassResult]`.

**Interaction surface:** This spec adds a `table_overlay: TableOverlay | None`
attribute to `PassResult`. If the Celery plan lands first, the per-pass
extraction state is read back from `pipeline_pass_outputs.metadata_json` at
merge time, so the table overlay must travel with that state. The two plans
overlap in:

| File / table | This spec | Celery plan |
|---|---|---|
| `app/workers/pipeline.py` | `_call_extract_pass` reads `table_overlay` from response and stashes on `PassResult` (~10 LOC) | New `derive_ontology_graph_pass` + `derive_ontology_graph_merge` tasks; refactor `derive_ontology_graph` to dispatcher (substantial) |
| `pipeline_pass_outputs` table | Needs to carry `TableOverlay` JSON | Owns the schema |
| `merge_and_resolve` invocation | Adds Phase 0.5 (field overlay) inside the function | Changes WHERE the function is called from (chord callback rather than inline) — the function body change here composes cleanly |

**Persistence approach (agreed coordination): stash inside `metadata_json`.**
Whichever plan lands first writes the glue:

```python
# Celery-plan-side, when persisting a pass's output:
metadata_json["table_overlay"] = (
    response.table_overlay.model_dump(mode="json") if response.table_overlay else None
)

# Merge-callback-side, when loading back:
overlay_dict = pass_output_record.metadata_json.get("table_overlay")
table_overlay = TableOverlay.model_validate(overlay_dict) if overlay_dict else None
# Then attach: pass_result.table_overlay = table_overlay
# _extract_doc_overlay(pass_results) picks it up unchanged.
```

Adding a dedicated `table_overlay_json` column instead is feasible but
requires a separate Alembic migration; not justified at v1 (single doc class
benefits from this overlay; metadata_json is the lower-friction site).

**Sequencing implications:**
- Celery plan first → this spec's worker-side wire-up adds two lines (write/read on `metadata_json["table_overlay"]`).
- This spec first → Celery plan's persistence schema needs to know about `PassResult.table_overlay` and round-trip it through `metadata_json`.
- Both in flight → agree on `metadata_json["table_overlay"]` shape upfront; conflicts limited to `app/workers/pipeline.py` rebase.

The functional logic of identity rewrite + field overlay (this spec's
substance) is independent of how pass results travel through the worker —
in-memory dict, persisted DB rows, or any future storage doesn't change
how `apply_identity_rewrite` and `apply_field_overlay` operate on the
Pydantic instances they're handed. The Celery plan only affects the
plumbing, not the algorithms.

## 5. Components

### 5.1 `_alias_map.py` — Constants

```python
# Entity-type-specific identity-row label patterns. Match is case-insensitive
# substring against the row's label_text.
#
# Bare "variant" and "designation" are DELIBERATELY EXCLUDED from v1 — they
# create false positives via cross-entity-ref rows (e.g., "Fan Song Variant"
# would match "variant", which is wrong). v2 may add them back if a strict
# "cross-entity-ref classification runs first" rule is enforced and tested.
MISSILE_IDENTITY_LABELS: tuple[str, ...] = (
    "missile type", "missile variant",
    "industry designation", "military designation", "nato designation",
    "system designation",
)

RADAR_IDENTITY_LABELS: tuple[str, ...] = (
    "radar variant", "radar designation", "radar type",
)

# Cross-entity reference rows: row labels that name a SIBLING entity type.
# When seen in a missile-context table, the row's cells are not missile
# aliases — they're radar aliases attached to the same column's missile via
# a relationship hint. Emitted as CrossEntityHint.
#
# **Classification order (enforced in _classify_row):**
# 1. Cross-entity-ref check FIRST
# 2. Identity-label check SECOND
# 3. Spec-row check (label-to-schema-field alias) THIRD
# 4. Otherwise: ignored
# This ordering matters because future v2 changes might add broader
# patterns; the cross-entity check shields radar references from being
# mis-classified as missile identity rows.
CROSS_ENTITY_REF_PATTERNS: dict[str, str] = {
    "fan song variant": "RADAR_SYSTEM",
    "spoon rest variant": "RADAR_SYSTEM",
}

# Canonical-name priority per entity type. When a column has aliases from
# multiple identity rows, pick the FIRST priority label that's present.
CANONICAL_PRIORITY: dict[str, tuple[str, ...]] = {
    "MISSILE_SYSTEM": (
        "Missile Type",
        "Industry Designation",
        "Military Designation",
        "NATO Designation",
    ),
    "RADAR_SYSTEM": (
        "Radar Variant",
        "Radar Designation",
        "Radar Type",
    ),
}
```

### 5.2 `_table_facts.py` — `extract_table_overlay`

Public entry point. Pure function, deterministic, milliseconds.

```python
def extract_table_overlay(doc_json: dict) -> TableOverlay:
    """Parse the FIRST STRICTLY-QUALIFYING column-major / hybrid variants
    table in doc.tables[].

    Strict qualification gate (all four MUST hold, see §3 Non-Goals
    "Multi-table docs"):
      (1) shape ∈ {column_major, hybrid}
      (2) entity_columns ≥ 4   (data columns to the right of the label region)
      (3) identity_rows ≥ 1    (col-0 row matches MISSILE_/RADAR_IDENTITY_LABELS,
                                excluding CROSS_ENTITY_REF_PATTERNS)
      (4) every entity column has a non-empty cell in ≥1 matched identity row

    Tables that fail any of (1)-(4): increment `tables_skipped_unqualified`
    and continue scanning (a small qualifying-on-shape-alone earlier table
    must NOT starve a strictly-qualifying later table). Tables that pass:
    if no winner yet → adopt as winner; if winner already set → log INFO
    `tables_skipped_multi++` and skip. Tables of OTHER shape are silently
    skipped (counted in `tables_skipped_other` for diagnostics).

    For the chosen table, for each entity column:
    1. Build alias cluster from MISSILE_IDENTITY_LABELS rows (excluding
       CROSS_ENTITY_REF_PATTERNS).
    2. Pick canonical via CANONICAL_PRIORITY[entity_type].
    3. Map all aliases → canonical → contribute to alias_map.
    4. Walk spec rows with section context → emit TableFact per cell that
       passes the parse-time gate (label-to-field alias resolves, unit
       conversion succeeds, value coerces to a parseable type).
    5. Walk cross-entity-ref rows → emit CrossEntityHint per cell.

    Returns empty TableOverlay if no qualifying table found (no-op for
    docs without column-major / hybrid tables).
    """
```

Internal helpers (all private, all pure):

| Function | Responsibility |
|---|---|
| `_classify_identity_row(label) → str \| None` | "MISSILE_SYSTEM" / "RADAR_SYSTEM" / None based on case-insensitive substring against `MISSILE_IDENTITY_LABELS` and `RADAR_IDENTITY_LABELS`. |
| `_classify_cross_entity_ref(label) → str \| None` | Returns target entity type if label matches `CROSS_ENTITY_REF_PATTERNS`, else None. |
| `_extract_alias_clusters(rows, entity_type, label_width) → dict[entity_col, set[(label, alias_text)]]` | Per-column cluster of (source-label, alias-text) tuples drawn from identity rows of that entity_type only. |
| `_pick_canonical(cluster, entity_type) → str` | Walks `CANONICAL_PRIORITY[entity_type]`; returns first matching alias's text. Falls back to deterministic-first when no priority match: NFC-normalize then casefold cluster members; sort by (normalized form, raw text) tuple to disambiguate Unicode-equivalent forms; return first. Logged at INFO level. |

### 5.3 `app/services/table_overlay.py` — Worker-side overlay

By the time `canonicalize_cross_pass_identities` runs, the worker has
already parsed JSON `pass_output` into Pydantic instances and stored them
under `PassResult` (per `pipeline.py:_parse_pass_response` → the `iter_entities_of_type`
walker yields Pydantic instances). Both functions operate on Pydantic
instances, but **explicitly re-validate values via `TypeAdapter`** because
the current extraction schemas (e.g., `missile_propulsion.py:30`) do NOT
set `validate_assignment=True` in their `model_config`. Plain `setattr`
on those models bypasses validators and field-level coercion. We use
`TypeAdapter(field_type).validate_python(value)` for explicit per-field
validation, then `setattr` only the validated value (or fall back to
`type(inst).model_validate(updated_dict)` if a model-level validator
needs to fire).

```python
def apply_identity_rewrite(
    pass_results: dict[str, "PassResult"],
    alias_map_by_entity_type: dict[str, dict[str, str]],
    ontology: dict,
) -> RewriteStats:
    """Rewrite system_name aliases to canonical names in-place on
    Pydantic instances across all passes.

    Alias map is entity-type-scoped: alias_map_by_entity_type["MISSILE_SYSTEM"]
    is consulted only for MISSILE_SYSTEM entities, etc. This prevents a
    radar-side alias "Fan Song A" from being mistakenly applied to a
    missile entity that happens to be named "Fan Song A" in some other doc.

    Operates on the same set of entity types canonicalize_cross_pass_identities
    walks (those whose graph_id_fields == ('system_name',)). Entity type
    is read from the model's ConfigDict.ontology_name.

    For each pass, for each entity instance of an eligible entity type:
      entity_type = inst.model_config.get('ontology_name')
      sub_map = alias_map_by_entity_type.get(entity_type, {})
      current = getattr(inst, 'system_name', None)
      if current in sub_map and sub_map[current] != current:
          setattr(inst, 'system_name', sub_map[current])  # plain str, no validation needed
          rewrites += 1

    Multiple entities may now share canonical system_name; merge_and_resolve's
    LogicalIdentity-keyed merge collapses them onto one MergedEntityRecord.

    Returns RewriteStats(rewrites=N, unique_canonicals=M, passes_touched=K).
    """

def apply_field_overlay(
    pass_results: dict[str, "PassResult"],
    table_facts: list[TableFact],
    *, policy: str = "table_wins_for_table_facts",
) -> OverlayStats:
    """Apply per-cell table facts to Pydantic entity instances.

    Facts already carry pass_name + canonical_entity + schema_field + value
    + section_ctx + source_label (set by parser). Each fact is routed to
    the matching pass_result. Within that pass, find the entity instance
    whose system_name == fact.canonical_entity (post-rewrite).

    Conflict resolution policy: 'table_wins_for_table_facts' (v1 default).
    The variants table is the authoritative source for spec values when a
    TableFact passed all 4 gates (section + alias + unit + Pydantic
    validation). The LLM's value for that same field is ignored — this is
    deliberate, because the propulsion failure mode is non-null wrong values
    (off-by-one row attribution), not nulls. Additive-only would skip those
    and miss the acceptance target.

    The "scoped" qualifier matters: ONLY (entity, field) pairs covered by
    a TableFact are overridden. Fields the LLM extracted that have NO
    corresponding TableFact (e.g., guidance_type from prose, or any field
    in a doc with no variants table) are entirely untouched. The overlay
    NEVER mutates fields outside its own evidence.

    For each candidate (fact, entity_instance) pair:
      1. Route fact to pass: if fact.pass_name not in pass_results,
         skipped_no_entity++ and continue. Otherwise look up entity in
         pass_results[fact.pass_name] whose system_name == fact.canonical_entity.
         If no such entity, skipped_no_entity++ and continue.
      2. Capture original via getattr(entity_instance, fact.schema_field, None).
      3. Validate fact.value against the field's Pydantic type:
           field_info = type(entity_instance).model_fields[fact.schema_field]
           try:
               coerced = TypeAdapter(field_info.annotation).validate_python(fact.value)
           except (ValidationError, ValueError, TypeError):
               skipped_validation_fail++; continue
      4. Apply: setattr(entity_instance, fact.schema_field, coerced)
         applied++
         if original is not None and original != coerced:
             conflicts_overridden++
             log("FIELD_OVERLAY_OVERRIDE pass=%s entity=%s field=%s "
                 "llm=%r table=%r source=%r",
                 fact.pass_name, fact.canonical_entity, fact.schema_field,
                 original, coerced, fact.source_label)
         else:
             log applied event (DEBUG)
      5. Optional model-level revalidation hook: when extraction schemas
         add `model_validator(mode='after')` constraints in the future
         (e.g., total_mass_kg >= booster_mass_kg + sustain_mass_kg), call
         type(inst).model_validate(inst.model_dump()) once at end of
         per-entity overlay batch. Out of scope for v1 — current schemas
         have no such cross-field validators.

    Returns OverlayStats(applied=N, skipped_no_entity=, skipped_validation_fail=,
    conflicts_overridden=, policy_active="table_wins_for_table_facts").
    """
```

**Why explicit `TypeAdapter` validation, not `setattr` + `validate_assignment`:**

The current extraction schemas (verified at `missile_propulsion.py:30`,
`radar_*.py:31`) configure `ConfigDict(extra="ignore", ontology_name=...,
graph_id_fields=..., identity_scope="global", is_entity=True)` — no
`validate_assignment=True`. Plain `setattr(inst, field, value)` would
silently store the raw value without type coercion or validator
enforcement. That's a real risk: a fact with `value="not a number"`
would land on a `Optional[float]` field as a string, breaking downstream
serialization at unpredictable points.

Two mitigations were considered:
- **(A)** Add `validate_assignment=True` to every extraction schema's
  `ConfigDict`. Touches ~9 files; possible but invasive; might surface
  pre-existing validators that fire on unexpected data shapes elsewhere
  in the pipeline.
- **(B) (chosen)** Validate explicitly via `TypeAdapter(field_type).validate_python(value)`
  before `setattr`. Self-contained in the overlay code path; no schema
  changes; `TypeAdapter` does all the type coercion (str→float etc.) and
  constraint enforcement (`Field(ge=0, le=...)`) that `validate_assignment`
  would have done.

If `model_validator(mode='after')` field-cross constraints land on the
extraction schemas in the future, swap to `model_validate(inst.model_dump())`
at the end of each entity's overlay batch (see step 5 above).

**v1 policy: `table_wins_for_table_facts`.** Scoped, deterministic, and
load-bearing for the propulsion acceptance target.

Why not additive-only:
- Propulsion's failure mode is **wrong values** (off-by-one row
  attribution: `13DM.booster_mass_kg=2283` when GT is `1032`). Additive-only
  would skip these because they're populated; the acceptance target
  (≥6 of 7 listed variants for `booster_mass_kg`) is unreachable.
- The 4-gate test in the parser ensures TableFact emission is conservative:
  a fact only exists if section context AND label-to-field alias AND unit
  conversion AND Pydantic validation all succeed. By construction, an
  emitted TableFact has stronger provenance than an LLM-extracted value
  for the same field.

Why not full table_wins (across all fields):
- Most LLM-extracted fields don't have a corresponding TableFact (prose-
  sourced fields like `guidance_type`, `nomenclature`, `system_status`).
  Those are the LLM's domain entirely.
- Scoping to fields covered by TableFacts limits override blast radius
  to exactly what the variants table can authoritatively answer.

Every override is logged with `(entity, field, llm_value, table_value,
source_label)` so audit/forensics can detect cases where the table itself
is wrong.

### 5.4 Data shapes

The wire-shape types (`TableOverlay`, `TableFact`, `CrossEntityHint`) are
**Pydantic `BaseModel`** because they cross the docling-graph ↔ worker
boundary in the `/extract-pass` HTTP response and (per §4.4) round-trip
through the per-pass-celery-tasks plan's `pipeline_pass_outputs.metadata_json`.
`model_dump(mode="json")` and `model_validate(...)` are required.

The internal stats types (`RewriteStats`, `OverlayStats`) stay as
`@dataclass` — they don't cross any boundary; they're returned from
worker functions and serialized via `asdict()` only for log lines.

```python
from pydantic import BaseModel, ConfigDict

class TableFact(BaseModel):
    """Per-cell deterministic fact derived from a variants table row."""
    model_config = ConfigDict(frozen=True)
    canonical_entity: str        # e.g., "1D"
    entity_type: str             # e.g., "MISSILE_SYSTEM" — scopes which entities receive this fact
    schema_field: str            # e.g., "booster_mass_kg"
    value: float | str
    source_label: str            # e.g., "Weight kg"
    section_ctx: str | None      # e.g., "1st Stage"
    pass_name: str               # e.g., "missile_propulsion" — facts route to this pass
    raw_text: str                # e.g., "1135"

class CrossEntityHint(BaseModel):
    """Row-level cross-entity reference. v1: collected but not applied."""
    model_config = ConfigDict(frozen=True)
    source_canonical: str        # e.g., "1D" (missile canonical)
    source_entity_type: str      # e.g., "MISSILE_SYSTEM"
    target_alias: str            # e.g., "RSNA-75" (radar alias)
    target_entity_type: str      # e.g., "RADAR_SYSTEM"
    relationship_kind: str       # e.g., "associated_with"

class TableOverlay(BaseModel):
    """Doc-level deterministic overlay derived from a variants table."""
    alias_map_by_entity_type: dict[str, dict[str, str]] = {}
    """Entity-type-scoped alias map: {entity_type: {alias: canonical}}.
       E.g., for SA-2 col 0:
       {"MISSILE_SYSTEM": {"SA-75": "1D", "SA-2A": "1D", ...}}.
       Empty if no variants table found. Doc-level: identical across all
       passes from the same DoclingDocument.

       Why entity-type-scoped (not flat dict[str, str]): the same alias
       text could legitimately mean different entities in different
       entity-type contexts (e.g., a missile-naming string colliding with
       a radar-naming string in some future doc). Scoping by entity_type
       makes the rewrite unambiguous and prevents accidental cross-type
       collapse."""

    facts: list[TableFact] = []
    """Per-cell facts ready for direct application to canonical entities.
       Each fact carries its own pass_name and entity_type, allowing the
       overlay to filter by both when applied."""

    cross_entity_hints: list[CrossEntityHint] = []
    """Optional: rows like Fan Song Variant in a missile table. v1: collected
       for future relationship-pass integration but NOT applied as edges."""

@dataclass
class RewriteStats:
    rewrites: int = 0
    unique_canonicals: int = 0
    passes_touched: int = 0
    def as_dict(self) -> dict: return asdict(self)

@dataclass
class OverlayStats:
    applied: int = 0
    skipped_no_entity: int = 0
    skipped_validation_fail: int = 0
    conflicts_overridden: int = 0   # set when LLM had a value AND TableFact overwrote
    policy_active: str = "table_wins_for_table_facts"  # echoed for diagnostics
    def as_dict(self) -> dict: return asdict(self)
```

### 5.5 `extraction_merge.py` integration

The alias map is doc-level (one DoclingDocument → one variants table → one
alias_map identical across all passes against that doc). Pass it as a single
`dict[str, str]`, not per-pass. Per-pass facts (which carry `pass_name`)
travel separately as a `list[TableFact]`. To carry both with minimal
plumbing, accept the full `TableOverlay` object. (Each `PassResult` will
have a `table_overlay` attribute set by `pipeline.py`; `merge_and_resolve`
extracts it from any one PassResult — they should be equivalent.)

**Identity rewrite — slots into existing `canonicalize_cross_pass_identities`:**

```python
def canonicalize_cross_pass_identities(
    pass_results: dict[str, "PassResult"],
    ontology: dict,
    *,
    table_alias_map: dict[str, str] | None = None,  # NEW, keyword-only
) -> int:
    """Mutate entity instances in-place so cross-pass duplicates share a
    single canonical `system_name`.

    Pass 0 (NEW): if table_alias_map is provided, rewrite aliases to
    canonical names FIRST. Catches table-defined aliases that the
    token-overlap heuristic below misses (e.g., SA-2A → 1D, where token
    bags don't overlap).

    Pass 1 (EXISTING): token-overlap canonicalization (unchanged).
    """
    rewrites = 0

    # NEW: table-derived rewrite (runs first; doc-level map applies to all passes)
    if table_alias_map:
        try:
            stats = apply_identity_rewrite(pass_results, table_alias_map, ontology)
            rewrites += stats.rewrites
            logger.info(
                "IDENTITY_REWRITE rewrites=%d unique_canonicals=%d passes_touched=%d",
                stats.rewrites, stats.unique_canonicals, stats.passes_touched,
            )
        except Exception as exc:
            logger.warning(
                "apply_identity_rewrite failed: %s — falling through to "
                "existing token-overlap canonicalization",
                exc,
            )

    # EXISTING: token-overlap canonicalization
    for entity_def in ontology.get("entity_types", []):
        ...  # unchanged code path

    return rewrites
```

**Field overlay — slots into `merge_and_resolve` Phase 0.5:**

```python
def merge_and_resolve(
    pass_results: dict[str, "PassResult"],
    manifest: Any,
    ontology: dict,
    document_id: str,
    pipeline_run_id: str,
) -> MergedExtraction:
    # --- Phase 0: cross-pass identity canonicalization ---
    # Extract doc-level table_overlay from any one PassResult (all should
    # carry the same overlay parsed from the same DoclingDocument). If
    # multiple passes carry different overlays, the parser is buggy; use
    # the first non-empty and log a WARNING.
    table_overlay = _extract_doc_overlay(pass_results)  # NEW helper

    canonicalize_cross_pass_identities(
        pass_results,
        ontology,
        table_alias_map=(table_overlay.alias_map if table_overlay else None),
    )

    # --- Phase 0.5 (NEW): per-pass field overlay ---
    if table_overlay and table_overlay.facts:
        try:
            stats = apply_field_overlay(
                pass_results,
                table_overlay.facts,
                policy="table_wins_for_table_facts",
            )
            logger.info(
                "TABLE_OVERLAY_APPLIED doc_id=%s "
                "field_overlay_applied=%d skipped_no_entity=%d "
                "skipped_validation_fail=%d conflicts_overridden=%d "
                "policy=%s",
                document_id, stats.applied, stats.skipped_no_entity,
                stats.skipped_validation_fail, stats.conflicts_overridden,
                stats.policy_active,
            )
        except Exception as exc:
            logger.warning(
                "apply_field_overlay failed: %s — proceeding with merge "
                "using LLM-extracted values only",
                exc,
            )

    # --- Pass 1: merge entities (existing) ---
    entity_index: dict[LogicalIdentity, MergedEntityRecord] = {}
    ...  # unchanged code path


def _extract_doc_overlay(pass_results: dict[str, "PassResult"]) -> TableOverlay | None:
    """Pick the doc-level TableOverlay carried on any PassResult.

    All PassResults from the same doc should carry equivalent overlays
    (parser is deterministic on doc.tables[]). Defensive: log WARNING if
    multiple non-empty overlays disagree.

    An overlay is "non-empty" if ANY of its three components is populated:
    alias_map_by_entity_type, facts, or cross_entity_hints. (Edge case:
    a doc could have a column-major table with cross-entity-ref rows but
    no proper identity rows — that yields cross_entity_hints + an empty
    alias_map but is still useful when relationship-pass integration ships.)
    """
    def _is_nonempty(ov: TableOverlay | None) -> bool:
        if ov is None:
            return False
        return bool(ov.alias_map_by_entity_type) or bool(ov.facts) or bool(ov.cross_entity_hints)

    seen: list[TableOverlay] = [
        ov for pr in pass_results.values()
        if _is_nonempty(ov := getattr(pr, "table_overlay", None))
    ]
    if not seen:
        return None

    first = seen[0]
    for other in seen[1:]:
        # Compare all three components. Divergence on any field indicates
        # a parser bug — log + use first.
        diverged = (
            other.alias_map_by_entity_type != first.alias_map_by_entity_type
            or other.facts != first.facts
            or other.cross_entity_hints != first.cross_entity_hints
        )
        if diverged:
            logger.warning(
                "PassResults carry divergent table_overlay (parser bug?). "
                "Using first non-empty; first_alias_size=%d other_alias_size=%d "
                "first_facts=%d other_facts=%d",
                sum(len(m) for m in first.alias_map_by_entity_type.values()),
                sum(len(m) for m in other.alias_map_by_entity_type.values()),
                len(first.facts), len(other.facts),
            )
    return first
```

**Failure-mode invariant:** if `apply_identity_rewrite` partially mutates
some Pydantic instances and then raises, those instances stay rewritten —
subsequent token-overlap canonicalization runs against half-rewritten
state. To keep this safe, both functions are written to be **idempotent**
(re-applying the same alias_map is a no-op since `alias_map[canonical] ==
canonical` short-circuits the assignment). Partial state therefore reaches
token-overlap as a STABLE intermediate, not corruption.

## 6. Data Flow (worked example: SA-2 PDF, missile_airframe pass)

```
─────────────────────────────────────────────────────────────────────────
PHASE 1: docling-graph /extract-pass (per-pass)
─────────────────────────────────────────────────────────────────────────

Input: docling_document_json (SA-2 PDF, has variants table)
       active_pass = "missile_airframe"

  Sanitize → extract_table_overlay (NEW) → LLM extraction → response
                ↓
  TableOverlay(
    alias_map={"SA-75":"1D", "SA-2A":"1D", "S-75":"13D", "SA-2C":"13D",
               "S-75M":"13DM", "SA-2D":"13DM", ...},
    facts=[
      TableFact(canonical="1D", schema_field="body_length_m", value=10.726,
                source_label="Length mm", section_ctx=None,
                pass_name="missile_airframe"),
      TableFact(canonical="1D", schema_field="booster_mass_kg", value=1135,
                source_label="Weight kg", section_ctx="1st Stage",
                pass_name="missile_propulsion"),
      ... ~50 facts across all passes
    ],
    cross_entity_hints=[
      CrossEntityHint(source_canonical="1D", target_alias="RSNA-75",
                      relationship_kind="associated_with"),
      ...  (deferred — not applied in v1)
    ]
  )

ExtractPassResponse: pass_output (LLM-extracted) + table_overlay (NEW field)

─────────────────────────────────────────────────────────────────────────
PHASE 2: worker accumulates 4 PassResults (one per missile pass)
─────────────────────────────────────────────────────────────────────────

PassResult.table_overlay attached per pass.

─────────────────────────────────────────────────────────────────────────
PHASE 3: merge_and_resolve → canonicalize_cross_pass_identities
─────────────────────────────────────────────────────────────────────────

  for each pass_result:
    apply_identity_rewrite(pass_output, alias_map)
    # airframe pass_output AFTER rewrite:
    # [{"system_name": "1D",   "body_length_m": 10.726, ...},
    #  {"system_name": "1D",   "total_mass_kg": 2163, ...},
    #  {"system_name": "13DM", "body_length_m": 10.841, ...},
    #  ...]                ← three "1D" entities now ready to merge

  EXISTING: token-overlap canonicalization runs (catches non-table names)

─────────────────────────────────────────────────────────────────────────
PHASE 4: field overlay (called once from merge_and_resolve, all passes routed)
─────────────────────────────────────────────────────────────────────────

  apply_field_overlay(pass_results, table_overlay.facts, policy="table_wins_for_table_facts")
  # Each fact carries pass_name → routed to pass_results[fact.pass_name].
  # Gate: TypeAdapter validation passes (LLM value, if any, gets overridden — scoped table_wins)
  #       (which subsumes the section_ctx + alias + unit checks since
  #        the parser only emitted facts that already passed those).

─────────────────────────────────────────────────────────────────────────
PHASE 5: merge_and_resolve entity loop (existing code, unchanged)
─────────────────────────────────────────────────────────────────────────

Three "1D" instances merge → one MISSILE_SYSTEM vertex with consolidated
fields:
  body_length_m: 10.726     (overlay applied; LLM was null → applied++)
  body_diameter_m: 0.654    (overlay applied; LLM was null → applied++)
  total_mass_kg: 2163       (LLM-extracted via "SA-2A"→"1D" rewrite,
                             no TableFact for total_mass_kg → untouched)
  booster_mass_kg: 1135     (overlay OVERRODE LLM=970 →
                             FIELD_OVERLAY_OVERRIDE log emitted,
                             conflicts_overridden++)
  booster_time_sec: 4.0     (overlay applied; LLM was null → applied++)
  booster_thrust: 19500     (LLM-extracted, no TableFact for this field
                             → never touched by overlay)

Demonstrates scoped-table-wins:
  - body_length_m / body_diameter_m / booster_time_sec → applied (LLM null)
  - booster_mass_kg → overridden (LLM had wrong number; this is the
    propulsion-acceptance unlock that additive_only would have missed)
  - total_mass_kg / booster_thrust → NEVER touched (no TableFact ⇒ no
    interference with LLM-only fields)
```

## 7. Error Handling

The overlay is augmentative at every boundary. Any failure falls back to the
existing LLM-extraction + token-overlap path. **System never gets worse than
the current baseline.**

### Parser-side (docling-graph)

| Failure | Behavior | Stats / log |
|---|---|---|
| Doc has no tables / only OTHER-shape tables | Empty `TableOverlay` | DEBUG; `tables_seen=N, alias_clusters_built=0` |
| Column-major table found but no identity rows match | Skip table for cluster building; column has no canonical → no-op | DEBUG; `columns_skipped_no_canonical++` |
| Identity row found but cell text empty for column | Exclude that column from cluster; other columns proceed | DEBUG |
| `_pick_canonical` finds no priority match in cluster | Fall back to alphabetic-first; log fallback | INFO `canonical_picked_via_fallback` |
| `coerce_value` fails on spec-row cell | Skip that fact | INFO `values_skipped_unparseable++` |
| Pydantic-validation fails on `TableFact(...)` construction | Skip that fact | INFO `facts_skipped_construct_fail++` |
| `extract_table_overlay` raises any uncaught exception | `main.py` catches, sets `table_overlay=None`, logs WARNING | `WARNING: extract_table_overlay failed: <exc>` |

### Worker-side (`table_overlay.py`)

| Failure | Behavior | Stats / log |
|---|---|---|
| `pass_result.table_overlay` is None | Both functions no-op | DEBUG |
| `alias_map` empty | No-op rewrite | DEBUG |
| Entity has no `system_name` field | Skip that entity (existing behavior) | DEBUG |
| `alias_map` points to canonical no entity has | Rewrite still applies (collapses aliases even without canonical seed) | (expected) |
| Field overlay finds no entity matching `fact.canonical_entity` | Skip fact | INFO `skipped_no_entity++` |
| Pydantic validation fails on field overlay | Skip fact | INFO `skipped_validation_fail++` |
| Field already populated AND policy=`table_wins_for_table_facts` AND `original != coerced` | Override LLM value with table fact | INFO `conflicts_overridden++`; `FIELD_OVERLAY_OVERRIDE doc=… pass=… entity=… field=… prior=… new=… fact=…` log line per override |
| Field already populated AND policy=`table_wins_for_table_facts` AND `original == coerced` | No-op (idempotent re-apply) | DEBUG; counted as `applied++`, NOT as override |
| Either function raises uncaught exception | `extraction_merge` catches, logs WARNING, falls through to existing path with original `pass_output` intact | `WARNING: <function> failed: <exc>` |

### Integration-layer (`extraction_merge.py`)

| Failure | Behavior |
|---|---|
| `table_alias_map` not provided (None) | New code path is a no-op; existing token-overlap canonicalization runs as before. Backward-compatible. |
| Multiple passes provide different `alias_map` content (parser bug — same doc should yield same map) | Defensive: use first non-empty alias_map; log WARNING on mismatch. |
| Field overlay applies value, then merge phase rejects via vertex-write validation | Standard merge error path triggers (already exists). Same behavior as if LLM had emitted that value directly. |

### Diagnostics surface

`/extract-pass` response gains `diagnostics["service_table_overlay"]`:

```python
{
    "alias_map_size": int,
    "facts_count": int,
    "cross_entity_hints_count": int,
    "tables_processed": int,
    "columns_skipped_no_canonical": int,
    "columns_with_canonical_via_fallback": int,
    "values_skipped_unparseable": int,
    "tables_skipped_multi": int,         # extras beyond first STRICTLY-QUALIFYING table
    "tables_skipped_unqualified": int,   # column_major/hybrid shape but failed entity_columns ≥ 4 / identity_rows ≥ 1 / sparse-identity gate
    "tables_skipped_other": int,         # other shape (row-major, too small, etc.)
    "kill_switch_active": bool,          # echoes DOCLING_GRAPH_TABLE_OVERLAY_ENABLED
}
```

Worker log lines per merge:

```
IDENTITY_REWRITE doc_id=<id>
  rewrites=N unique_canonicals=M passes_touched=K

TABLE_OVERLAY_APPLIED doc_id=<id>
  field_overlay_applied=K skipped_no_entity=A skipped_validation_fail=B
  conflicts_overridden=C policy=table_wins_for_table_facts
```

Both lines emit even when their counts are zero, so an operator scanning
logs can confirm the overlay path ran (vs ran silently / errored).

## 8. Testing

### 8.1 Parser unit tests (`test_table_overlay_extract.py`)

| Function | Cases |
|---|---|
| `_classify_identity_row` | "Missile Type" / "Industry Designation" / "Radar Variant" / "Length mm" / "Fan Song Variant" / unmatched / empty |
| `_classify_cross_entity_ref` | "Fan Song Variant" / "Spoon Rest Variant" / unmatched |
| `_extract_alias_clusters` | SA-2-shape → 10 clusters; Fan Song row excluded; empty cells properly excluded; column with NO matching identity rows → no cluster |
| `_pick_canonical` | Cluster with Missile Type → "Missile Type" wins; cluster missing Missile Type, present NATO → NATO wins; cluster with no priority match → alphabetic fallback (logged); empty cluster → "" |
| `extract_table_overlay` | SA-2 fixture → ~30 alias entries, ~50 facts, cross_entity_hints for Fan Song; doc with no tables → empty TableOverlay; doc with row-major table → empty (out of scope); malformed cells → empty + WARNING log |

### 8.2 Drift guards (`test_alias_map_overlay_constants.py`)

```python
def test_identity_labels_have_canonical_priority_coverage():
    """Every label in MISSILE_IDENTITY_LABELS appears (case-insensitive
    substring) in CANONICAL_PRIORITY['MISSILE_SYSTEM'] OR is documented as
    intentional fallback. Catches new label without priority entry."""

def test_cross_entity_ref_patterns_dont_overlap_identity_labels():
    """A label can't be both a missile-identity row AND a cross-entity-ref
    row. CROSS_ENTITY_REF_PATTERNS keys must not match MISSILE_IDENTITY_LABELS
    or RADAR_IDENTITY_LABELS after normalization."""

def test_canonical_priority_uses_display_labels():
    """The strings in CANONICAL_PRIORITY are user-facing label patterns
    (Title Case with spaces, e.g., "Missile Type"), not schema field names
    (snake_case with underscores, e.g., "system_name"). Catches accidental
    schema-field-name leakage. Specific assertions:
      - Each entry contains at least one space (multi-word display label)
      - No entry contains an underscore character
      - Each entry's first character is uppercase
      - Each entry, after normalize_label(), is a substring of at least one
        entry in MISSILE_IDENTITY_LABELS or RADAR_IDENTITY_LABELS"""
```

### 8.3 Worker unit tests (`test_table_overlay_worker.py`)

| Function | Cases |
|---|---|
| `apply_identity_rewrite` | empty alias_map → no-op; entity with system_name in map → rewritten; entity without system_name → skipped; multiple entities sharing alias → all rewrite to same canonical; empty pass_output → no-op |
| `apply_field_overlay` (table_wins_for_table_facts, v1 default) | Fact for canonical not in pass_output → skipped (skipped_no_entity++); value coerces via TypeAdapter → applied; value fails TypeAdapter → skipped (skipped_validation_fail++); pre-existing LLM value present → overwritten AND override logged AND conflicts_overridden++; pre-existing was None → applied without log noise; fields without a corresponding TableFact are NEVER touched |
| `apply_field_overlay` entity-type scoping | Fact with entity_type="MISSILE_SYSTEM" never lands on a RADAR_SYSTEM instance even if names happen to collide |
| `RewriteStats` / `OverlayStats` | Counter increments + `.as_dict()` serialization |

### 8.4 Worker integration tests (`test_extraction_merge_table_overlay.py`)

```python
def test_table_alias_map_runs_before_token_overlap():
    """Three entities {SA-75, SA-2A, 1D} with non-overlapping tokens;
    alias_map={SA-75:"1D", SA-2A:"1D"}. After
    canonicalize_cross_pass_identities, all three have system_name="1D";
    merge_and_resolve produces ONE vertex."""

def test_table_overlay_does_not_break_existing_token_overlap():
    """PAC-3 and MIM-104F (existing token-overlap path catches via shared
    tokens). Empty alias_map. Existing canonicalization still runs."""

def test_field_overlay_table_wins_for_table_facts_policy():
    """entity has body_length_m=null + table_fact body_length_m=10.726 →
    after overlay, entity body_length_m=10.726, applied++.

    entity has body_length_m=10.5 + same fact → after overlay,
    body_length_m=10.726 (table wins), conflicts_overridden++,
    FIELD_OVERLAY_OVERRIDE log emitted with field, prior, new, and
    table_fact provenance.

    entity has body_length_m=10.726 + same fact → after overlay,
    body_length_m=10.726 (no-op), applied++, conflicts_overridden NOT
    incremented because original == coerced."""

def test_field_overlay_only_touches_fields_with_facts():
    """entity has body_length_m=10.5 (LLM) and max_speed_m_per_s=600 (LLM);
    table_fact only for body_length_m. After overlay, body_length_m=10.726
    (overridden), max_speed_m_per_s=600 (untouched). Demonstrates scoped
    table_wins: no fact, no change."""

def test_field_overlay_entity_type_scope():
    """table_fact has entity_type='MISSILE_SYSTEM'. pass_output contains a
    RADAR_SYSTEM with the same canonical name. Fact must NOT land on the
    RADAR_SYSTEM. Confirms cross-type collisions are filtered out."""

def test_field_overlay_pydantic_validation_gate():
    """table_fact value="not a number" for numeric field →
    TypeAdapter(field_info.annotation).validate_python raises →
    skipped_validation_fail++, entity field stays at its prior LLM value
    (NOT overwritten with garbage). Verifies validate-and-coerce is a hard
    gate even under table_wins_for_table_facts."""

def test_kill_switch_disables_overlay():
    """DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false → table_overlay=None in
    response → worker no-ops both functions → existing token-overlap
    canonicalization runs as before."""
```

### 8.5 End-to-end fixture (`test_table_overlay_end_to_end.py`)

Synthetic DoclingDocument matching SA-2 variants table structure; full
4-pass simulation through `merge_and_resolve`:

1. Build synthetic DoclingDocument with SA-2 column structure (22×12,
   identity rows + section headers + spec rows + Fan Song row).
2. Construct synthetic 4-pass `pass_results: dict[str, PassResult]` with
   LLM-extracted entities (Pydantic instances) under various aliases
   (SA-75, SA-2A, 1D, S-75, etc.). Each PassResult carries the
   `table_overlay` attribute populated from step 1.
3. Call `merge_and_resolve(pass_results, manifest, ontology, doc_id, run_id)`
   — the new code path picks up the overlay from PassResult attributes.
4. Assert final `MergedExtraction.entities`:
   - One canonical MISSILE_SYSTEM per entity column of the synthetic table.
     The SA-2 PDF has 10 entity columns (verified by 2026-05-05 dump:
     1D, 13D, 13DM, 13DA, 13DAM, 20D, 20DP, 20DSU, 5Ya23, 15D); the
     synthetic fixture matches this exactly.
   - Each carries expected fields from variants table (per-cell overlay
     application).
   - No duplicate `MergedEntityRecord` for alias names — `SA-75`, `SA-2A`,
     etc. must NOT appear as separate entities.
   - `cross_entity_hints` collected (≥5 Fan Song associations from
     applicable columns) but explicit assert: NO `ASSOCIATED_WITH` edges
     between missile and radar entities in this v1 test (v2 / future spec).

### 8.6 Acceptance — empirical operator-driven

`notebooks/extraction_walkthrough.ipynb` §20 cell at `T=1.0` against real
SA-2 PDF. Same scoreboard format used pre-revert.

**Acceptance criteria:**

The targets below are FLOORS (entity counts at the pass level) plus
**field-specific table-overlay acceptance** (per-field correctness on the
fields the variants table actually carries). Both the floor and the
field-specific row must pass.

| Pass | Floor: ✓ exact entity count (post-A1) | Field-specific table-overlay acceptance | vs alias-only baseline |
|---|---|---|---|
| `missile_kinematics` | ≥ 4 | n/a — variants table carries no kinematics fields per the SA-2 PDF | no regression |
| `missile_airframe` | ≥ 14 | for each variant column where the variants table row "Body Length" is non-empty: that variant's `body_length_m` post-overlay equals the table cell to within unit-conversion tolerance (1e-3 m); same for "Diameter" → `body_diameter_m`. Required: ≥ 6 of the 7 listed variants on EACH such field where the cell exists. | +6 over baseline 8 |
| `missile_speed_timing` | ≥ 6 | for each variant column where the variants table row "Max Speed" / "Max Effective Range" is non-empty: that variant's corresponding flat field post-overlay equals the cell within tolerance. Required: ≥ 6 of 7 on EACH such field. | no regression |
| `missile_propulsion` | ≥ 4 (was the regression-baseline floor) | **≥ 6 of 7 listed variants** (13DM, 13DA, 13DAM, 20D, 20DP, 20DSU, 5Ya23) have `booster_mass_kg` matching the variants-table booster row within tolerance, AND ≥ 6 of 7 have `booster_propellant_mass_kg` matching, sourced via `apply_field_overlay` (i.e. with `FIELD_OVERLAY_OVERRIDE` log lines emitted where the LLM had wrong values pre-overlay) | spec target met |

**Why field-specific gates matter (per user pushback on r5):** the floor
counts above are the same metric we used pre-revert and they conflate
"entity exists" with "entity has correct field values." `additive_only`
(rejected here) could have hit the floor counts while leaving the
propulsion field values wrong because the LLM had populated them with
wrong-but-non-null numbers that `additive_only` would not have
overridden. Scoped `table_wins_for_table_facts` is the only policy that
satisfies the field-specific row when the LLM emits wrong numbers; the
field-specific row is therefore the load-bearing acceptance, and the
floor is just a guard against introducing entity-loss regressions.

**No regression on no-table docs:**
- For each of 16 no-table corpus docs and 4 row-major-table docs: re-run extraction post-overlay-deploy at the SAME temperature as the pre-deploy run.
  - **Hard guard:** entity count per pass must match within ±2 (LLM at T=1.0 is non-deterministic; tolerance accommodates run-to-run variance, not regression).
  - **Soft guard:** the system_name set in the post-deploy run must be ⊇ 80% of the pre-deploy set (entities don't disappear; new aliases would be a parser bug since these docs have no variants tables).
  - For temperature-sensitive comparison, run at T=0.0 against a synthetic LLM mock (returns canned response) — exact pass_output match required.

**Wall-time budget:**
- ≤ +5% per /extract-pass call wall time on table-bearing docs.
- 0% on no-table docs.

**Baseline reference for ✓ exact targets — RE-DERIVE BEFORE APPROVAL:**

The floor numbers above are pinned to the post-revert state at commit
`1b71150` (table-fact synthesizer reverted, B1+B2 disabled, alias-only
path active). The cached §20 GT scorecard from 2026-05-06
(`/tmp/r21_alias_only_backup/`) is **stale enough that it must NOT be
treated as authoritative for landing this overlay.** Implementation
sequence (also reflected in §10):

  1. **Re-derive baseline first.** Before merging any code in this spec,
     run notebook §20 at `T=1.0` on `HEAD` (current `main`, with overlay
     wiring DISABLED via `DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false`) and
     record fresh per-pass `✓ exact` counts AND per-variant
     field-correctness on the fields named in the per-pass table above.
     This becomes the *live baseline* the post-overlay run is compared
     against.
  2. **Update the floor row** of the per-pass table above with the live
     baseline numbers (subject to `±2` LLM noise tolerance) before
     declaring acceptance.
  3. **Compare post-overlay run** to the live baseline, NOT the cached
     scorecard.

Until step 1 is done and the floor row reflects fresh numbers, this
spec's acceptance is provisional and **must not be approved** based on
the stale cached values.

## 9. Acceptance Criteria

1. `extract_table_overlay` is wired into `/extract-pass` with catch-and-continue guard.
2. `apply_identity_rewrite` + `apply_field_overlay` exist as pure functions in `app/services/table_overlay.py`.
3. `canonicalize_cross_pass_identities` accepts `table_alias_map` argument; calls rewrite BEFORE its existing token-overlap pass.
4. All unit, integration, drift-guard, and end-to-end tests pass.
5. **Live baseline re-derived** at HEAD with overlay disabled (per §8.6 baseline-reference subsection); floor row in §8.6 updated to live numbers before approval.
6. **Field-specific table-overlay acceptance** per §8.6 satisfied: for each pass and each variants-table field listed there, ≥ 6 of 7 listed variants match the table cell within tolerance, and `FIELD_OVERLAY_OVERRIDE` log lines exist for any variant where the LLM had emitted a wrong value pre-overlay (proves `apply_field_overlay` actually overrode rather than leaving wrong LLM values intact).
7. **Floor (entity-count) targets** from §8.6 met or beaten relative to the live baseline (±2 LLM-noise tolerance). Specifically: `missile_propulsion ≥ live-baseline floor`, `missile_kinematics ≥ live-baseline floor`, `missile_airframe ≥ live-baseline floor`, `missile_speed_timing ≥ live-baseline floor`. The floor is a guard against entity-loss regressions, not the primary gate.
8. No regression on the 20 no-table corpus docs (system_name set within ±2 entities and ⊇ 80% of pre-deploy set per §8.6).
9. Wall-time delta ≤ +5% on table-bearing docs; 0% on no-table docs.
10. Kill switch (`DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false`) cleanly reverts behavior without code change.
11. Diagnostics surfaced: `service_table_overlay` in /extract-pass response (with `tables_skipped_unqualified` / `tables_skipped_multi` / `tables_skipped_other` populated); `IDENTITY_REWRITE` / `TABLE_OVERLAY_APPLIED` / `FIELD_OVERLAY_OVERRIDE` log lines from the worker.

## 10. Implementation Order (suggested)

0. **Re-derive baseline.** Run notebook §20 at `T=1.0` on the current `main` HEAD with `DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false`. Record fresh per-pass ✓ exact counts AND per-variant field-correctness on the fields named in §8.6. Update the floor row of §8.6 with the live numbers. **All subsequent acceptance comparisons compare against THIS live baseline, not the cached `/tmp/r21_alias_only_backup/` scorecard.**
1. **Constants module updates** (`_alias_map.py`) — add `MISSILE_IDENTITY_LABELS` etc. + drift-guard tests. No logic change.
2. **Parser additions** (`_table_facts.py`) — `_classify_identity_row`, `_classify_cross_entity_ref`, `_extract_alias_clusters`, `_pick_canonical` + unit tests.
3. **Schemas** (`schemas.py`) — `TableOverlay`, `TableFact`, `CrossEntityHint` Pydantic models.
4. **Public parser entry** (`extract_table_overlay`) + integration tests on SA-2 fixture.
5. **`main.py` wire-up** — call parser, attach to response, kill-switch env var.
6. **Worker overlay module** (`app/services/table_overlay.py`) — `apply_identity_rewrite` + `apply_field_overlay` + unit tests.
7. **`extraction_merge.py` integration** — extend `canonicalize_cross_pass_identities` signature; call rewrite before token-overlap.
8. **`pipeline.py` wire-up** — read `table_overlay` from response; pass through to merge.
9. **End-to-end fixture test** + manual smoke test on SA-2 fixture.
10. **Container rebuild + §20 acceptance run** on real SA-2 PDF.

**Parallelizability:** steps 1, 2, 3 can run in parallel after step 1 lands the constants. Step 4 depends on 2+3. Step 5 depends on 4. Step 6 is independent of parser path; step 7 depends on 6. Step 8 depends on 5+7. Steps 9-10 sequential after 8.
