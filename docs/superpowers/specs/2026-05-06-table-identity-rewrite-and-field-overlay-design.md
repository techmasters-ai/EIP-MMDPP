# Table-Derived Identity Rewrite + Per-Cell Field Overlay (Mechanism A1)

**Status:** Approved 2026-05-07 — implementation complete on `main`, parser smoke verified on real SA-2 Guideline PDF (43/43 GT-field facts correct, 9 listed missile variants × 5 fields). Worker-side application verified by Task 7+8+11 unit/integration tests. Live worker-side smoke against real SA-2 deferred pending coordination with concurrent `feat/per-pass-celery-fanin` branch (spec §4.4) — the in-memory PassResults path on `main` is shortly to be replaced by the DB-backed fan-in path, so a full real-ingest live smoke is best deferred to the post-merge moment when both architectures are live together.
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
| `app/services/table_overlay.py` | NEW (~200 LOC) | Two functions operating on Pydantic instances via `iter_entities_of_type`: `apply_identity_rewrite(pass_results, alias_map_by_entity_type: dict[str, dict[str, str]], ontology) -> RewriteStats` and `apply_field_overlay(pass_results, table_facts, *, policy="table_wins_for_table_facts") -> OverlayStats`. |
| `app/services/extraction_merge.py` | MODIFY (~+30 LOC) | `canonicalize_cross_pass_identities` accepts new keyword-only `table_alias_map_by_entity_type: dict[str, dict[str, str]] \| None` argument; calls `apply_identity_rewrite` BEFORE its existing token-overlap loop. New separate call to `apply_field_overlay` from `merge_and_resolve` after canonicalization, before merge. |
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

Single env var allows operator-controlled rollback without code change.
**The flag is read on BOTH the parser side and the worker side**, with
the worker-side check being authoritative for what actually mutates
`pass_results`. This matters because, with the per-pass-celery-tasks
plan (§4.4) persisting `table_overlay` into `pipeline_pass_outputs.
metadata_json`, an old already-emitted overlay can outlive a parser-side
flag flip — a worker-side gate is the only way to guarantee the merge
phase respects a freshly disabled kill switch on previously-cached
overlays.

```yaml
# docker-compose.yml — must be set on BOTH services
docling-graph:
  environment:
    DOCLING_GRAPH_TABLE_OVERLAY_ENABLED: ${DOCLING_GRAPH_TABLE_OVERLAY_ENABLED:-true}

worker / worker-graph:
  environment:
    DOCLING_GRAPH_TABLE_OVERLAY_ENABLED: ${DOCLING_GRAPH_TABLE_OVERLAY_ENABLED:-true}
```

Effects:

- **Parser side (docling-graph).** When `false`, `extract_table_overlay`
  short-circuits and returns an empty `TableOverlay`. Fresh extract-pass
  responses carry no overlay payload. Behavior on fresh extractions is
  bit-identical to pre-overlay.
- **Worker side (`extraction_merge.py`).** When `false`, both
  `_extract_doc_overlay(pass_results)` AND the `apply_field_overlay`
  call site short-circuit to None / no-op, **regardless of whether
  `pass_results[*].table_overlay` carries a non-empty payload from a
  prior cached extraction**. Specifically:

      def merge_and_resolve(...):
          if not _table_overlay_enabled():
              # kill-switch authoritative; ignore any cached overlay
              # carried in pass_results from prior runs.
              canonicalize_cross_pass_identities(
                  pass_results, ontology,
                  table_alias_map_by_entity_type=None,
              )
              # ... no Phase 0.5 ...
              return _existing_merge_path(pass_results, ...)
          ...

  Where `_table_overlay_enabled()` reads `os.environ.get(
  "DOCLING_GRAPH_TABLE_OVERLAY_ENABLED", "true").lower() != "false"`.
  This is the operator-controlled "back to baseline" path: a single
  env-flag flip + worker restart is sufficient to drop the system to
  pre-overlay behavior even if `pipeline_pass_outputs.metadata_json`
  already contains overlay payloads from yesterday's runs.
- **Diagnostics.** When the worker-side gate trips on cached overlays,
  log `INFO: TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER pass_count=%d
  cached_overlay_present=%d` once per `merge_and_resolve` invocation.
  The `service_table_overlay.kill_switch_active` field in the
  /extract-pass response reflects the parser-side flag; a separate
  worker-emitted log line reflects the worker-side gate. They are
  redundant by design (defense in depth).

Restart sequence to fully revert: (1) flip both env flags, (2) restart
docling-graph and worker / worker-graph containers (so they pick up
the new env), (3) any pending merge work after the restart respects
the new flag. In-flight Celery tasks finish with their captured-at-
start flag value (acceptable; they were already past the worker-side
gate).

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
# Every entry in MISSILE_/RADAR_IDENTITY_LABELS must appear (case-insensitive
# substring) somewhere in this priority tuple — otherwise the drift guard
# test_identity_labels_have_canonical_priority_coverage fails.
CANONICAL_PRIORITY: dict[str, tuple[str, ...]] = {
    "MISSILE_SYSTEM": (
        "Missile Type",
        "Industry Designation",
        "Military Designation",
        "NATO Designation",
        "System Designation",   # fallback for docs that use this label only
        "Missile Variant",      # fallback for docs that use this label only
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
instances, but **explicitly re-validate values via the full
`cls.model_validate({**inst.model_dump(), field: value})` path** because
the current extraction schemas (e.g., `missile_propulsion.py:30`) do
NOT set `validate_assignment=True` in their `model_config` AND each
schema attaches per-field `field_validator(mode="before")` hooks
(`_v_booster_mass_kg`, `_v_max_intercept_km`, `_v_max_speed_mps`, etc.)
that handle string→float coercion and codebase-specific normalization.
A bare `TypeAdapter(field_info.annotation).validate_python(value)` would
coerce the type but skip those validators; `setattr` would skip
everything. We therefore route every (entity, field) overlay through
`cls.model_validate(...)` so every existing field validator fires, and
any future `model_validator(mode="after")` cross-field constraint also
fires automatically. See §5.3 for the per-step swap pattern that keeps
instance identity stable.

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

    Facts already carry pass_name + entity_type + canonical_entity +
    schema_field + value + section_ctx + source_label (set by parser).
    Each fact is routed to the matching pass_result. Within that pass,
    apply the fact to **every** entity instance of the matching
    `entity_type` whose `system_name == fact.canonical_entity`
    (post-rewrite). After alias rewrite, multiple instances can legitimately
    share a canonical name; we cannot rely on one being "the" instance —
    if we update only one, merge order can preserve a wrong value from a
    sibling. So a fact fans out across all matches.

    Conflict resolution policy: 'table_wins_for_table_facts' (v1 default).
    The variants table is the authoritative source for spec values when a
    TableFact passed all 4 gates (section + alias + unit + Pydantic
    validation). The LLM's value for that same field is ignored — this is
    deliberate, because the propulsion failure mode is non-null wrong values
    (off-by-one row attribution), not nulls. A populated-only policy would
    skip those and miss the acceptance target.

    The "scoped" qualifier matters: ONLY (entity, field) pairs covered by
    a TableFact are overridden. Fields the LLM extracted that have NO
    corresponding TableFact (e.g., guidance_type from prose, or any field
    in a doc with no variants table) are entirely untouched. The overlay
    NEVER mutates fields outside its own evidence.

    For each fact:
      1. Route to pass: if fact.pass_name not in pass_results,
         skipped_no_entity++ and continue. Otherwise enumerate ALL
         entity instances in pass_results[fact.pass_name] of type
         fact.entity_type whose system_name == fact.canonical_entity
         (post-rewrite). If none, skipped_no_entity++ and continue.
      2. For each matching instance (fan-out — see "multi-instance"
         note below):
           a. **Pre-validate the field name against the model.**
              Extraction schemas use `ConfigDict(extra="ignore")`, which
              means `cls.model_validate(candidate)` would silently drop
              an unknown key in the candidate dict. Without this check,
              a stale or pass-mismatched `fact.schema_field` could pass
              `model_validate`, then `getattr(revalidated,
              fact.schema_field)` would either raise AttributeError
              (truly unknown attribute) OR return the entity's prior
              value (silently inherited from the dump), and `applied++`
              would falsely count a no-op as success. Guard explicitly:
                cls = type(inst)
                if fact.schema_field not in cls.model_fields:
                    skipped_unknown_field++
                    log("FIELD_OVERLAY_UNKNOWN_FIELD pass=%s "
                        "entity_type=%s entity=%s schema_field=%s "
                        "model=%s — fact dropped",
                        fact.pass_name, fact.entity_type,
                        fact.canonical_entity, fact.schema_field,
                        cls.__name__)
                    continue
              This catches: (i) fact emitted from a parser bug under a
              field name that does not exist on the routed pass's
              schema; (ii) cross-pass mis-routing (e.g., a propulsion
              fact accidentally tagged `pass_name="missile_airframe"`);
              (iii) schema renames where the parser alias map was not
              updated. All three are caller bugs, not validation
              failures, and are counted separately so the diagnostics
              surface distinguishes them from value-typed failures.
           b. Capture original via
              getattr(inst, fact.schema_field, None).
           c. Build a candidate dict and validate. Plain
              `TypeAdapter(field_info.annotation).validate_python`
              validates the raw annotation but does NOT execute the
              schema's `field_validator(mode='before')` hooks (e.g.
              `_v_booster_mass_kg`, `_v_max_intercept_km`) or any
              `Field(ge=…, le=…)` metadata constraints attached to the
              field. Use Pydantic's full model-validation path on a
              candidate dict so before-validators AND field-level
              constraints AND any future `model_validator(mode='after')`
              all fire:
                candidate = {**inst.model_dump(), fact.schema_field: fact.value}
                try:
                    revalidated = cls.model_validate(candidate)
                except (ValidationError, ValueError, TypeError):
                    skipped_validation_fail++; continue
                coerced = getattr(revalidated, fact.schema_field)
              `extra="ignore"` no longer hides anything here because
              step (a) already rejected unknown field names; the only
              remaining failure mode is value-typed.
           d. Atomic single-field setattr. The model_validate call
              already produced a fully-validated `revalidated`
              instance. Mutate ONLY `fact.schema_field` on `inst`
              using the validated `coerced` value:
                try:
                    setattr(inst, fact.schema_field, coerced)
                except Exception as exc:
                    skipped_validation_fail++
                    log("FIELD_OVERLAY_SETATTR_FAILED ...")
                    continue
              We deliberately do NOT loop over `revalidated.model_dump()`
              and copy every field. That would silently rewrite
              SIBLING fields to whatever shape model_validate produced
              (string→float coercions on un-touched fields, etc.) and
              could surprise downstream code that expects unchanged
              LLM values for fields the overlay didn't touch. The
              "overlay NEVER mutates fields outside its own evidence"
              principle from earlier in this section is load-bearing —
              single-field setattr enforces it.

              This guarantees: if model_validate raised, inst is
              UNCHANGED. If setattr raised (rare; would only happen
              under future `validate_assignment=True`), inst is also
              unchanged. Sibling instances in the same fan-out are
              independent: a validation failure for one does NOT
              block fan-out to the others.
           e. Bookkeeping (per-instance only — `matches_touched` is
              incremented once per fact in step 3, NOT here):
                applied++           # fact-instance count, NOT fact count
                if original is not None and original != coerced:
                    conflicts_overridden++
                    log("FIELD_OVERLAY_OVERRIDE pass=%s entity_type=%s "
                        "entity=%s field=%s llm=%r table=%r source=%r",
                        fact.pass_name, fact.entity_type,
                        fact.canonical_entity, fact.schema_field,
                        original, coerced, fact.source_label)
                else:
                    log applied event (DEBUG)
      3. After the per-instance loop completes for a fact, increment
         matches_touched++ if the fact applied to ≥1 instance. This
         counter answers "how many facts found an entity to land on?"
         independently of fan-out width. With fan-out, `applied`
         (fact-instance count) ≥ matches_touched (fact count); the
         ratio = average post-rewrite duplicate-count per fact.

    **Multi-instance fan-out:** A TableFact applies to ALL matching
    instances of the right entity_type within its pass, not just the
    first. Rationale: after alias rewrite, a pass can contain N >= 1
    instances with the same canonical system_name (e.g., the LLM
    emitted "SA-2A" and "SA-75" separately, both rewritten to "1D").
    If overlay only touched the first, the OTHER N-1 instances retain
    their pre-overlay (potentially wrong) LLM values. The merge layer
    then has no deterministic way to pick the correct one — order-
    dependent. Fanning out to all N gives all N the same correct value
    so any merge order yields the right answer. Same-pass duplicates
    are common in propulsion when columns alias to a single missile.

    **Transactional semantics — bounded, not full:** Each (fact,
    instance) pair is atomic via the model_validate-then-swap pattern;
    a validation failure for that pair leaves the instance untouched.
    BUT the overall apply_field_overlay loop is NOT a single
    transaction across multiple facts: an unhandled exception thrown
    from inside the loop body (e.g. KeyError on a malformed fact)
    leaves earlier successfully-applied mutations in place. The §7
    error-handling table classifies what extraction_merge.py does on
    that path (catch+log, fall through to merge with whatever
    mutations completed). Overlay is bounded-degraded, not bounded-
    none-or-all.

    Returns OverlayStats(
        applied=N,                  # fact-instance count (fan-out)
        matches_touched=M,          # fact count that landed on >=1 inst
        skipped_no_entity=,         # fact had no matching entity in pass
        skipped_unknown_field=,     # fact.schema_field not in cls.model_fields
        skipped_validation_fail=,   # value failed model_validate
        conflicts_overridden=,      # LLM had a value AND TableFact replaced it
        policy_active="table_wins_for_table_facts",
    ).
    """
```

**Why full model-validation, not `TypeAdapter` and not `setattr` + `validate_assignment`:**

The current extraction schemas (verified at `missile_propulsion.py:30`,
`radar_*.py:31`) configure `ConfigDict(extra="ignore", ontology_name=...,
graph_id_fields=..., identity_scope="global", is_entity=True)` — no
`validate_assignment=True`. Plain `setattr(inst, field, value)` would
silently store the raw value without type coercion or validator
enforcement. That's a real risk: a fact with `value="not a number"`
would land on a `Optional[float]` field as a string, breaking downstream
serialization at unpredictable points.

Three mitigations considered:
- **(A)** Add `validate_assignment=True` to every extraction schema's
  `ConfigDict`. Touches ~9 files; possible but invasive; might surface
  pre-existing validators that fire on unexpected data shapes elsewhere
  in the pipeline.
- **(B)** Validate via `TypeAdapter(field_info.annotation).validate_python(value)`
  before `setattr`. **Rejected.** TypeAdapter applied to the bare
  annotation (e.g. `Optional[float]`) coerces the type but does NOT
  execute the schema's `field_validator(mode='before')` hooks
  (`_v_booster_mass_kg = field_validator("booster_mass_kg",
  mode="before")(coerce_optional_float)` and the equivalents on
  every numeric field across propulsion / kinematics / speed_timing /
  airframe). Those validators are where the codebase's float-coercion
  rules and `coerce_optional_text` logic live; bypassing them means
  the overlay applies raw `fact.value` (a Python primitive from the
  parser, but with no schema-level normalization). It also misses any
  `Field(ge=…, le=…)` metadata constraints, which TypeAdapter on the
  raw annotation does not always honor without an `Annotated[...]`
  wrapper. We cannot pretend constraints don't exist; some are
  load-bearing for downstream graph-write validity.
- **(C) (chosen)** Use full Pydantic model validation:
    `cls.model_validate({**inst.model_dump(), fact.schema_field: fact.value})`
  This re-runs the entire schema's validators on a candidate dict —
  including every `field_validator(mode='before')`, every `Field(...)`
  constraint, every existing-but-currently-unused
  `model_validator(mode='after')`. The cost is one extra `model_dump()`
  + `model_validate()` per fact-instance pair (microseconds per
  instance per fact; negligible at the corpus scale). The benefit is
  that whatever the schema currently enforces, the overlay also
  enforces, and future `model_validator(mode='after')` cross-field
  constraints are honored automatically.

  After validation succeeds, a "swap" pattern (copy validated values
  back via `setattr`) keeps the *identity* of the instance stable —
  important because `iter_entities_of_type` may have already yielded
  references the caller holds.

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
from pydantic import BaseModel, ConfigDict, Field

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
    # Use Field(default_factory=...) for mutable defaults so each
    # instance gets its own dict/list. Pydantic v2 deep-copies bare
    # `= {}` / `= []` defaults safely, but default_factory is the
    # idiomatic and unambiguous form, and keeps spec→implementation
    # transcription from accidentally introducing shared-default bugs.
    alias_map_by_entity_type: dict[str, dict[str, str]] = Field(default_factory=dict)
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

    facts: list[TableFact] = Field(default_factory=list)
    """Per-cell facts ready for direct application to canonical entities.
       Each fact carries its own pass_name and entity_type, allowing the
       overlay to filter by both when applied."""

    cross_entity_hints: list[CrossEntityHint] = Field(default_factory=list)
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
    applied: int = 0                # fact-instance count (fan-out width)
    matches_touched: int = 0        # fact count that landed on >=1 instance
    skipped_no_entity: int = 0      # fact had no matching entity in pass
    skipped_unknown_field: int = 0  # fact.schema_field not in cls.model_fields
    skipped_validation_fail: int = 0  # value failed cls.model_validate(...)
    conflicts_overridden: int = 0   # set when LLM had a value AND TableFact overwrote
    policy_active: str = "table_wins_for_table_facts"  # echoed for diagnostics
    def as_dict(self) -> dict: return asdict(self)
```

### 5.5 `extraction_merge.py` integration

The alias map is doc-level (one DoclingDocument → one variants table → one
alias map identical across all passes against that doc). Pass it as a
single **entity-type-scoped** `dict[str, dict[str, str]]` (outer key is
ontology type like `"MISSILE_SYSTEM"`, inner is alias→canonical), NOT a
flat `dict[str, str]` — the scoping prevents radar names from colliding
with missile names per §3 issue 3. Per-pass facts (which carry `pass_name`)
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
    # NEW, keyword-only. Entity-type-scoped: outer key is ontology type
    # name (e.g., "MISSILE_SYSTEM"), inner is alias → canonical.
    table_alias_map_by_entity_type: dict[str, dict[str, str]] | None = None,
) -> int:
    """Mutate entity instances in-place so cross-pass duplicates share a
    single canonical `system_name`.

    Pass 0 (NEW): if table_alias_map_by_entity_type is provided, rewrite
    aliases to canonical names FIRST. Catches table-defined aliases that
    the token-overlap heuristic below misses (e.g., SA-2A → 1D, where
    token bags don't overlap). Scoped per entity type: a missile's alias
    map cannot rewrite a radar's system_name and vice versa.

    Pass 1 (EXISTING): token-overlap canonicalization (unchanged).
    """
    rewrites = 0

    # NEW: table-derived rewrite (runs first; doc-level map applies to all passes)
    if table_alias_map_by_entity_type:
        try:
            stats = apply_identity_rewrite(
                pass_results,
                table_alias_map_by_entity_type,
                ontology,
            )
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
        table_alias_map_by_entity_type=(
            table_overlay.alias_map_by_entity_type if table_overlay else None
        ),
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
                "field_overlay_applied=%d matches_touched=%d "
                "skipped_no_entity=%d skipped_unknown_field=%d "
                "skipped_validation_fail=%d conflicts_overridden=%d "
                "policy=%s",
                document_id, stats.applied, stats.matches_touched,
                stats.skipped_no_entity, stats.skipped_unknown_field,
                stats.skipped_validation_fail, stats.conflicts_overridden,
                stats.policy_active,
            )
        except Exception as exc:
            logger.warning(
                "apply_field_overlay failed mid-loop: %s — proceeding "
                "with merge. NOTE: any successful (fact, instance) swaps "
                "completed before the exception remain in pass_results. "
                "Overlay is bounded-degraded, not all-or-nothing — see §7.",
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
    alias_map_by_entity_type={
      "MISSILE_SYSTEM": {
        "SA-75":"1D", "SA-2A":"1D", "S-75":"13D", "SA-2C":"13D",
        "S-75M":"13DM", "SA-2D":"13DM", ...
      },
      # No "RADAR_SYSTEM" entry — Fan Song / Spoon Rest rows in the
      # missile-context table are emitted as cross_entity_hints, NOT
      # as aliases (per §5.1 classification-order rule).
    },
    facts=[
      TableFact(canonical_entity="1D", entity_type="MISSILE_SYSTEM",
                schema_field="body_length_m", value=10.726,
                source_label="Length mm", section_ctx=None,
                pass_name="missile_airframe", raw_text="10726 mm"),
      TableFact(canonical_entity="1D", entity_type="MISSILE_SYSTEM",
                schema_field="booster_mass_kg", value=1135,
                source_label="Weight kg", section_ctx="1st Stage",
                pass_name="missile_propulsion", raw_text="1135"),
      TableFact(canonical_entity="1D", entity_type="MISSILE_SYSTEM",
                schema_field="sustain_mass_kg", value=896,
                source_label="Weight kg", section_ctx="2nd Stage",
                pass_name="missile_propulsion", raw_text="896"),
      TableFact(canonical_entity="1D", entity_type="MISSILE_SYSTEM",
                schema_field="max_intercept_km", value=29,
                source_label="Max Effective Range km", section_ctx=None,
                pass_name="missile_kinematics", raw_text="29"),
      TableFact(canonical_entity="1D", entity_type="MISSILE_SYSTEM",
                schema_field="max_speed_mps", value=975,
                source_label="Max Speed m/s", section_ctx=None,
                pass_name="missile_speed_timing", raw_text="975"),
      ... ~50 facts across all passes
    ],
    cross_entity_hints=[
      CrossEntityHint(source_canonical="1D",
                      source_entity_type="MISSILE_SYSTEM",
                      target_alias="RSNA-75",
                      target_entity_type="RADAR_SYSTEM",
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

  apply_identity_rewrite(
      pass_results,                         # all 4 passes' Pydantic instances
      alias_map_by_entity_type,             # outer-keyed by ontology type
      ontology,
  )
  # airframe pass_results["missile_airframe"] AFTER rewrite:
    # [{"system_name": "1D",   "body_length_m": 10.726, ...},
    #  {"system_name": "1D",   "total_mass_kg": 2163, ...},
    #  {"system_name": "13DM", "body_length_m": 10.841, ...},
    #  ...]                ← three "1D" entities now ready to merge

  EXISTING: token-overlap canonicalization runs (catches non-table names)

─────────────────────────────────────────────────────────────────────────
PHASE 4: field overlay (called once from merge_and_resolve, all passes routed)
─────────────────────────────────────────────────────────────────────────

  apply_field_overlay(pass_results, table_overlay.facts, policy="table_wins_for_table_facts")
  # Each fact carries (pass_name, entity_type) → routed to
  # pass_results[fact.pass_name], filtered to instances of fact.entity_type.
  # Gate: cls.model_validate(candidate) succeeds (runs all
  #       field_validator(mode="before") hooks + Field(...) constraints).
  #       LLM value, if any, gets overridden — scoped table_wins.
  # Fan-out: a fact applies to ALL post-rewrite instances of the matching
  #          entity_type with system_name == fact.canonical_entity in
  #          pass_results[fact.pass_name], not just one. After alias
  #          rewrite, multiple instances can legitimately share the
  #          canonical name; updating only one would let merge order
  #          preserve a wrong sibling value.

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
    propulsion-acceptance unlock that a populated-only policy would have missed)
  - total_mass_kg / booster_thrust → NEVER touched (no TableFact ⇒ no
    interference with LLM-only fields)
```

## 7. Error Handling

Overlay is **bounded-degraded, not strictly augmentative.** With
`policy="table_wins_for_table_facts"`, an applied fact intentionally
overwrites a populated LLM value when the parser's 4-gate test passed —
that is, by design, a way the post-overlay field value differs from the
pre-overlay LLM value. So the older r1-r5 phrasing ("system never gets
worse than the current baseline") is no longer true *as stated*. The
correct narrower claim is:

  1. **Parser-side bypass.** Any uncaught parser exception causes
     `main.py` to emit `table_overlay=None` and log a WARNING; the
     LLM-extraction path runs untouched. Overlay-off behavior == pre-
     overlay behavior, exactly.
  2. **Worker-side bypass.** Any uncaught exception from
     `apply_identity_rewrite` or `apply_field_overlay` is caught at the
     `merge_and_resolve` boundary, logged WARNING, and merge proceeds
     with whatever `pass_results` state existed at that moment.
  3. **Bounded override risk.** A fact is only emitted when section
     context AND label-to-field alias AND unit conversion AND Pydantic
     model-validation all succeed. An emitted fact therefore has
     stronger provenance than an LLM extraction for the same
     (entity, field). Overrides are not unconditional; they are
     conditional on the 4-gate passing.
  4. **Per-(fact, instance) atomicity, not full-batch atomicity.**
     Each `model_validate`+swap is atomic: a validation failure on one
     pair leaves that instance unchanged. But the apply_field_overlay
     loop is NOT a single transaction across multiple facts. If an
     unhandled exception fires from inside the loop body (e.g., a
     fan-out hitting an instance that was deleted by an unrelated
     code path mid-run — should not happen in practice), earlier
     successfully-applied mutations stay in place. Merge then runs
     with a partially-overlaid `pass_results`, which is a strictly
     better state than pre-overlay for the fields that landed and
     identical to pre-overlay for the fields that didn't. There is
     no rollback. We accept partial-progress over rollback because
     rollback would require deep-copying every Pydantic instance
     before mutation, which is wall-time-significant on the corpus
     and not justified by any observed failure mode.
  5. **Kill switch.** `DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false`
     drops behavior to exactly the pre-overlay path with no code
     change. This is the operator-controlled "get back to baseline"
     escape hatch; the system invariant of "never worse than
     baseline" is preserved through the kill switch, not through the
     overlay code path itself.

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
| `fact.schema_field` not in `cls.model_fields` (stale alias map / wrong pass routing / schema rename) | Skip fact; do NOT call model_validate (would silently no-op via `extra="ignore"`) | INFO `skipped_unknown_field++`; `FIELD_OVERLAY_UNKNOWN_FIELD pass=… entity_type=… entity=… schema_field=… model=…` |
| Pydantic validation fails on field overlay | Skip fact | INFO `skipped_validation_fail++` |
| Field already populated AND policy=`table_wins_for_table_facts` AND `original != coerced` | Override LLM value with table fact | INFO `conflicts_overridden++`; `FIELD_OVERLAY_OVERRIDE doc=… pass=… entity=… field=… prior=… new=… fact=…` log line per override |
| Field already populated AND policy=`table_wins_for_table_facts` AND `original == coerced` | No-op (idempotent re-apply) | DEBUG; counted as `applied++`, NOT as override |
| Either function raises uncaught exception **mid-loop** | `extraction_merge` catches, logs WARNING, falls through to merge with **whatever (fact, instance) swaps had already completed** in `pass_results`. NOT all-or-nothing — overlay is bounded-degraded per §7 invariants. The per-(fact, instance) atomicity guarantees each individual mutation is internally consistent; the overall batch is not transactional. Operator-controlled rollback is via the kill switch (§4.3), not via in-flight rollback. | `WARNING: <function> failed mid-loop: <exc>` |

### Integration-layer (`extraction_merge.py`)

| Failure | Behavior |
|---|---|
| `table_alias_map_by_entity_type` not provided (None) | New code path is a no-op; existing token-overlap canonicalization runs as before. Backward-compatible. |
| Entity-type-scoped alias map provided but contains no key matching this entity's ontology type | `apply_identity_rewrite` skips that entity (via `sub_map = alias_map_by_entity_type.get(entity_type, {})`) — guards against radar maps landing on missiles and vice versa. |
| Multiple passes provide different `alias_map_by_entity_type` content (parser bug — same doc should yield same map) | Defensive: use first non-empty map; log WARNING on mismatch. |
| Field overlay applies value, then merge phase rejects via vertex-write validation | Standard merge error path triggers (already exists). Same behavior as if LLM had emitted that value directly. |

### Diagnostics surface

`/extract-pass` response gains `diagnostics["service_table_overlay"]`:

```python
{
    # parser-side, per-doc:
    "alias_map_size": int,            # sum of len(sub_map) across entity types
    "facts_count": int,
    "cross_entity_hints_count": int,
    "tables_processed": int,
    "columns_skipped_no_canonical": int,
    "columns_with_canonical_via_fallback": int,
    "values_skipped_unparseable": int,
    "tables_skipped_multi": int,         # extras beyond first STRICTLY-QUALIFYING table
    "tables_skipped_unqualified": int,   # column_major/hybrid shape but failed entity_columns ≥ 4 / identity_rows ≥ 1 / sparse-identity gate
    "tables_skipped_other": int,         # other shape (row-major, too small, etc.)
    "kill_switch_active_parser": bool,   # echoes parser-side DOCLING_GRAPH_TABLE_OVERLAY_ENABLED
}
```

Worker emits `worker_table_overlay_stats` separately on each merge
(this surfaces the worker-side stats that the parser-side response
cannot carry, and the worker-side kill-switch state which can differ
from the parser's):

```python
{
    "kill_switch_active_worker": bool,   # worker-side env flag at merge time
    "cached_overlay_present": int,       # how many PassResults carried a non-empty overlay
    "rewrites": int,                     # apply_identity_rewrite stats
    "unique_canonicals": int,
    "passes_touched": int,
    "applied": int,                      # fact-instance count (fan-out)
    "matches_touched": int,              # fact count that landed on >=1 instance
    "skipped_no_entity": int,
    "skipped_unknown_field": int,        # NEW: fact.schema_field not in cls.model_fields
    "skipped_validation_fail": int,
    "conflicts_overridden": int,
    "policy_active": str,                # "table_wins_for_table_facts" | etc.
}
```

Worker log lines per merge:

```
IDENTITY_REWRITE doc_id=<id>
  rewrites=N unique_canonicals=M passes_touched=K

TABLE_OVERLAY_APPLIED doc_id=<id>
  field_overlay_applied=K matches_touched=M skipped_no_entity=A
  skipped_unknown_field=U skipped_validation_fail=B conflicts_overridden=C
  policy=table_wins_for_table_facts

# Emitted ONCE per merge_and_resolve invocation when the worker-side
# kill switch is off and at least one cached overlay was ignored:
TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER doc_id=<id>
  pass_count=N cached_overlay_present=M

# Emitted per (fact, instance) pair where the model_validate dropped
# the field due to extra="ignore" — i.e., skipped_unknown_field path:
FIELD_OVERLAY_UNKNOWN_FIELD doc_id=<id>
  pass=<pass_name> entity_type=<type> entity=<canonical>
  schema_field=<field> model=<cls.__name__>
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

**Strict-qualification starvation tests (`test_table_overlay_qualification.py`):**

These guard the §3 strict 4-of-4 AND gate against the failure mode the
user called out in r6.1 review: a small earlier column-major-shaped
table starving the real variants table.

```python
def test_unqualified_earlier_table_does_not_starve_real_variants_table():
    """doc.tables = [
        # Table 0 — column_major shape but only 2 entity columns,
        # 1 identity row → fails entity_columns ≥ 4 gate.
        small_columnar_spec_sheet,
        # Table 1 — REAL SA-2 variants table (10 cols, 5 identity rows).
        real_variants_table,
    ]. Expected: extract_table_overlay returns alias_map_by_entity_type
    populated from Table 1, tables_skipped_unqualified=1,
    tables_skipped_multi=0, NOT empty."""

def test_entity_columns_gate_under_4_rejects():
    """column_major table with 3 entity columns and 2 identity rows →
    fails (entity_columns < 4). Returns empty TableOverlay,
    tables_skipped_unqualified=1."""

def test_sparse_identity_cells_rejects():
    """column_major table, entity_columns=5, identity_rows=1, but the
    matched identity row has cells filled in only 1 of the 5 columns →
    fails (every-entity-column-must-have-cell gate). Returns empty,
    tables_skipped_unqualified=1."""

def test_radar_qualifying_table_before_missile_table_v1_picks_missile():
    """v1 acceptance is missile-focused (per §1 Pattern A, §8.6 acceptance
    rows). When doc.tables = [
        radar_variants_table_qualifying,   # passes 4-gate, RADAR_SYSTEM-keyed
        missile_variants_table_qualifying, # passes 4-gate, MISSILE_SYSTEM-keyed
    ], extract_table_overlay should still pick the FIRST strictly-
    qualifying table — the radar one — because the picker is
    entity-type-agnostic by design (cross-entity scoping happens later
    via alias_map_by_entity_type and TableFact.entity_type, not at the
    table-pick stage). Result: alias_map_by_entity_type contains a
    "RADAR_SYSTEM" key from the radar table; missile entities are NOT
    affected because no MISSILE_SYSTEM aliases were emitted from the
    chosen table. tables_skipped_multi=1 (the missile table is the
    skipped second-qualifying)."""

def test_two_qualifying_missile_tables_first_wins():
    """Two strictly-qualifying missile variants tables in the same doc.
    First wins; second logged tables_skipped_multi=1. Confirms the
    "no order-dependence problem within a single entity type"
    assumption documented in §3."""
```

**Note on radar-before-missile:** the v1 picker is entity-type-agnostic
(it picks the first strictly-qualifying table of any keying). If a
future SA-2-shaped doc puts a small qualifying radar table before the
real missile variants table, the picker will choose the radar table
and the missile entities will not benefit from overlay. This is
acceptable for v1 because (a) the corpus survey shows zero such docs
today, and (b) the kill switch + diagnostics surface the choice. v2
may revisit by adding entity-type preference (e.g.,
`prefer_entity_type="MISSILE_SYSTEM"`) when a real corpus example
appears.

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
| `apply_identity_rewrite` | empty alias_map → no-op; entity with system_name in map → rewritten; entity without system_name → skipped; multiple entities sharing alias → all rewrite to same canonical; empty pass_output → no-op; entity_type-scoped map: alias for MISSILE_SYSTEM never rewrites a RADAR_SYSTEM instance |
| `apply_field_overlay` validation gate uses model_validate | Fact value passed as a string ("1135") for `booster_mass_kg` → `_v_booster_mass_kg` field_validator coerces to float → applied as 1135.0. Verifies cls.model_validate(...) is used (NOT TypeAdapter on raw annotation, which would skip the validator). |
| `apply_field_overlay` unknown-field precheck | Fact carries `schema_field="bogus_field_not_on_model"` for a propulsion entity. cls.model_validate would silently no-op (extra="ignore") and getattr(revalidated, "bogus_field_not_on_model") would raise/return stale. Expected: `skipped_unknown_field++`, `FIELD_OVERLAY_UNKNOWN_FIELD` log emitted, no setattr, applied stays at 0, instance unchanged. |
| `apply_field_overlay` matches_touched vs applied accounting | One fact, two post-rewrite instances both matching the canonical → applied=2, matches_touched=1. One fact, zero matching instances → applied=0, matches_touched=0, skipped_no_entity=1. Demonstrates fact-instance vs fact accounting under fan-out. |
| `apply_field_overlay` (table_wins_for_table_facts, v1 default) | Fact for canonical not in pass_output → skipped (skipped_no_entity++); value passes model_validate → applied; value fails model_validate → skipped (skipped_validation_fail++); pre-existing LLM value present → overwritten AND override logged AND conflicts_overridden++; pre-existing was None → applied without log noise; fields without a corresponding TableFact are NEVER touched |
| `apply_field_overlay` entity-type scoping | Fact with entity_type="MISSILE_SYSTEM" never lands on a RADAR_SYSTEM instance even if system_names happen to collide |
| `apply_field_overlay` multi-instance fan-out | Two MISSILE_SYSTEM instances in the same pass with system_name="1D" (post-rewrite, originally "SA-75" and "SA-2A"); single TableFact for 1D.booster_mass_kg=1135 → BOTH receive the value, applied++ counted twice |
| `apply_field_overlay` per-(fact, instance) atomicity | Validation failure on instance A leaves A unchanged AND does not block fan-out to instance B which passes validation |
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
    """entity has body_length_m=10.5 (LLM) and total_mass_kg=2200 (LLM);
    table_fact only for body_length_m. After overlay, body_length_m=10.726
    (overridden), total_mass_kg=2200 (untouched). Demonstrates scoped
    table_wins: no fact, no change."""

def test_field_overlay_runs_field_validator_hooks():
    """`missile_propulsion._v_booster_mass_kg` is a
    field_validator(mode='before')(coerce_optional_float) hook. A
    TableFact value passed as a string ("1135") must round-trip through
    the validator to a float. Verifies that apply_field_overlay uses
    `cls.model_validate(...)`, NOT TypeAdapter on the raw annotation —
    the latter would not execute the field-validator and the field
    would end up holding a str."""

def test_field_overlay_fans_out_to_all_matching_instances():
    """After alias rewrite, missile_propulsion pass_output contains TWO
    Pydantic instances both with system_name='1D' (one was originally
    'SA-75', one was originally 'SA-2A'; both have null booster_mass_kg).
    A single TableFact for 1D.booster_mass_kg=1135 → BOTH instances get
    booster_mass_kg=1135, applied++ twice. Verifies fan-out (sibling
    duplicates can't carry stale wrong values into merge)."""

def test_field_overlay_atomic_per_fact_failure():
    """A TableFact's value fails model_validate (e.g., violates Field(ge=0)
    on a future-added constraint). The instance's prior LLM value MUST be
    unchanged after the failure (atomicity per (fact, instance) pair).
    Sibling instances in the same fan-out must still receive the fact if
    they pass validation — one instance's failure does not block siblings."""

def test_field_overlay_entity_type_scope():
    """table_fact has entity_type='MISSILE_SYSTEM'. pass_output contains a
    RADAR_SYSTEM with the same canonical name. Fact must NOT land on the
    RADAR_SYSTEM. Confirms cross-type collisions are filtered out."""

def test_field_overlay_pydantic_validation_gate():
    """table_fact value="not a number" for a numeric field →
    cls.model_validate({**inst.model_dump(), field: value}) raises
    ValidationError → skipped_validation_fail++, entity field stays at
    its prior LLM value (NOT overwritten with garbage). Verifies
    validate-and-coerce is a hard gate even under
    table_wins_for_table_facts, and that the gate uses full model
    validation (so all field_validator(mode='before') hooks AND
    Field(...) constraints fire), not bare TypeAdapter on the raw
    annotation."""

def test_kill_switch_disables_overlay():
    """DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false → table_overlay=None in
    response → worker no-ops both functions → existing token-overlap
    canonicalization runs as before."""

def test_kill_switch_worker_side_overrides_cached_overlay():
    """Critical defense-in-depth case: worker receives a PassResult
    whose .table_overlay field is a fully-populated TableOverlay
    (e.g., loaded from pipeline_pass_outputs.metadata_json from
    yesterday's run). Operator has just set
    DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false on the worker container.
    Expected: merge_and_resolve consults the worker-side env gate,
    treats the cached overlay as if it were None, runs ONLY the
    pre-overlay code path. apply_identity_rewrite NOT called;
    apply_field_overlay NOT called; no IDENTITY_REWRITE log line; no
    TABLE_OVERLAY_APPLIED log line. ONE INFO log line:
    TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER pass_count=N
    cached_overlay_present=N."""

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

Schema field names verified against
`ontology_bundles/air_defense_v3/extraction_schemas/{missile_propulsion,missile_kinematics,missile_speed_timing,missile_airframe}.py`
(2026-05-06 HEAD). Fields not present in the schemas are excluded from
acceptance even if the variants table carries the row label.

**Live-baseline floor numbers pinned 2026-05-06 from §20 cached scoreboards** at `/tmp/baseline_2026-05-06_pre_overlay/` (Task 0 capture; worker code at HEAD with overlay disabled). The floor row below tracks the **count of `✓ exact` GT-field rows** in the §20 aggregate scorecard (NOT entity count — entity counts of 41/41/45/45 reflect alias scatter and will DROP after Mechanism A1 alias collapse, which is the intended improvement). LLM-noise tolerance: ±2 around each floor.

| Pass | Floor: ✓ exact GT-field count (live baseline 2026-05-06) | Field-specific table-overlay acceptance | vs live baseline |
|---|---|---|---|
| `missile_kinematics` | **floor = 4** (live: 1D + 13D have `max_altitude_km` ✓ AND `min_intercept_km` ✓; the other 8 variants are all null) | for each variant column where the variants table has non-empty cells in "Max (Effective) Range", "Min Range", "Max Altitude", "Min Altitude": ≥ 6 of 7 listed variants have `max_intercept_km` / `min_intercept_km` / `max_altitude_km` / `min_altitude_km` matching the table cell within tolerance (1e-2 km). Schema fields per `missile_kinematics.py`: `min_intercept_km`, `max_intercept_km`, `min_altitude_km`, `max_altitude_km`, `max_launch_angle_deg`. Expected post-overlay: **~36 ✓ exact** (9 listed variants × 4 fields filled from null). | +32 expected |
| `missile_airframe` | **floor = 10** (live: all 10 variants have `total_mass_kg` ✓ already; `body_length_m` and `body_diameter_m` are null for ALL 10) | for each variant column where "Length" / "Diameter" is non-empty: ≥ 6 of 7 listed variants have `body_length_m` / `body_diameter_m` matching within tolerance (1e-3 m). For columns where "Total Weight" / "Launch Weight" is non-empty: ≥ 6 of 7 have `total_mass_kg` matching within tolerance (1 kg). Schema fields per `missile_airframe.py`: `body_length_m`, `body_diameter_m`, `total_mass_kg`. Expected post-overlay: **~30 ✓ exact** (10 already-correct + 10 length + 10 diameter). | +20 expected |
| `missile_speed_timing` | **floor = 6** (live: 6 of 7 variants have `max_speed_mps` ✓; only 13D is null) | for each variant column where "Max Speed" is non-empty: ≥ 6 of 7 have `max_speed_mps` (NOT `max_speed_m_per_s` — that field does not exist) matching within tolerance (1 m/s). Schema fields per `missile_speed_timing.py` available for table mapping: `average_speed_mps`, `max_speed_mps`, `max_flyout_time_sec`, `flight_time_sec`, `coast_time_sec`, `intra_salvo_time_sec`, `total_burn_time_sec`, `ejector_time_sec`. **Note:** "Max Effective Range" maps to `missile_kinematics.max_intercept_km`, NOT to any speed_timing field; if r5 wording suggested otherwise, that was wrong. Expected post-overlay: **7 ✓ exact**. | +1 expected |
| `missile_propulsion` | **floor = 0** (live: 0 of 20 GT-field rows correct across the 10 measured variants × 2 fields. The LLM is consistently wrong via off-by-one row attribution: see `/tmp/baseline_2026-05-06_pre_overlay/scorecard.md`) | **≥ 6 of 7 listed variants** (13DM, 13DA, 13DAM, 20D, 20DP, 20DSU, 5Ya23) have `booster_mass_kg` matching the variants-table 1st-stage weight row within tolerance, AND ≥ 6 of 7 have `sustain_mass_kg` matching the 2nd-stage weight row (NOT `booster_propellant_mass_kg` — that field does NOT exist in `missile_propulsion.py`; the schema's flat fields are `ejector_mass_kg`, `booster_mass_kg`, `sustain_mass_kg`, plus `_thrust` text fields and `_time_sec` floats). Each match must be sourced via `apply_field_overlay` with `FIELD_OVERLAY_OVERRIDE` log lines emitted where the LLM had wrong values pre-overlay. Expected post-overlay: **≥ 18 ✓ exact** (9 variants × 2 fields, minus tolerance). Predicted override count: **15** (7 booster + 8 sustain) — see `/tmp/baseline_2026-05-06_pre_overlay/expected_overrides.md`. | +18 expected |

**Why field-specific gates matter (per user pushback on r5):** the floor
counts above are the same metric we used pre-revert and they conflate
"entity exists" with "entity has correct field values." A populated-only
policy (rejected here per user pushback on r5) could have hit the floor
counts while leaving the propulsion field values wrong because the LLM
had populated them with wrong-but-non-null numbers that a populated-only
policy would not have overridden. Scoped `table_wins_for_table_facts` is the only policy that
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
3. `canonicalize_cross_pass_identities` accepts keyword-only `table_alias_map_by_entity_type: dict[str, dict[str, str]] | None` argument; calls rewrite BEFORE its existing token-overlap pass with entity-type scoping enforced.
4. All unit, integration, drift-guard, and end-to-end tests pass.
5. **Live baseline re-derived** at HEAD with overlay disabled (per §8.6 baseline-reference subsection); floor row in §8.6 updated to live numbers before approval.
6. **Field-specific table-overlay acceptance** per §8.6 satisfied: for each pass and each variants-table field listed there, ≥ 6 of 7 listed variants match the table cell within tolerance, and `FIELD_OVERLAY_OVERRIDE` log lines exist for any variant where the LLM had emitted a wrong value pre-overlay (proves `apply_field_overlay` actually overrode rather than leaving wrong LLM values intact).
7. **Floor (entity-count) targets** from §8.6 met or beaten relative to the live baseline (±2 LLM-noise tolerance). Specifically: `missile_propulsion ≥ live-baseline floor`, `missile_kinematics ≥ live-baseline floor`, `missile_airframe ≥ live-baseline floor`, `missile_speed_timing ≥ live-baseline floor`. The floor is a guard against entity-loss regressions, not the primary gate.
8. No regression on the 20 no-table corpus docs (system_name set within ±2 entities and ⊇ 80% of pre-deploy set per §8.6).
9. Wall-time delta ≤ +5% on table-bearing docs; 0% on no-table docs.
10. Kill switch (`DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false`) cleanly reverts behavior without code change. **Both parser and worker honor the flag**; the worker-side gate is authoritative and ignores any cached overlay carried in `pass_results` from prior runs (verified by `test_kill_switch_worker_side_overrides_cached_overlay`).
11. Diagnostics surfaced: `service_table_overlay` in /extract-pass response (with `tables_skipped_unqualified` / `tables_skipped_multi` / `tables_skipped_other` and `kill_switch_active_parser` populated); worker stats include `applied` / `matches_touched` / `skipped_no_entity` / `skipped_unknown_field` / `skipped_validation_fail` / `conflicts_overridden` / `kill_switch_active_worker` / `cached_overlay_present`; log lines `IDENTITY_REWRITE` / `TABLE_OVERLAY_APPLIED` / `FIELD_OVERLAY_OVERRIDE` / `FIELD_OVERLAY_UNKNOWN_FIELD` / `TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER` emitted from the worker.

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
