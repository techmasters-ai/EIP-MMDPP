# Table-Derived Identity Rewrite + Per-Cell Field Overlay (Mechanism A1)

**Status:** Approved 2026-05-06
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
- **`table_wins` conflict resolution policy.** v1 ships with `additive_only` (apply table fact only when entity field is null). Escalation to `table_wins` deferred until empirical evidence shows additive_only insufficient. The `policy` parameter is plumbed through (one-line v2 escalation) but `table_wins` tests are written as `xfail` — flip the marker when v2 ships.
- **Multi-table docs.** A single doc with two distinct variants tables (e.g., a future doc with both a missile variants table AND a separate radar variants table) — **`extract_table_overlay` (parser, §5.2) enforces "first qualifying table wins"**. Subsequent qualifying tables logged at INFO level (`tables_skipped_multi=N`) and skipped. The corpus survey shows zero docs with this shape today; revisit if a future doc requires both. The worker side (`apply_*` functions, `merge_and_resolve`) doesn't need to enforce — it only ever sees the parser's selected first table.

## 4. Architecture

### 4.1 File layout

| File | Status | Purpose |
|---|---|---|
| `docker/docling-graph/app/_table_facts.py` | MODIFY (~+100 LOC) | Add `extract_table_overlay(doc_json) → TableOverlay`. Reuses existing parser primitives. New private helpers: `_classify_identity_row`, `_classify_cross_entity_ref`, `_extract_alias_clusters`, `_pick_canonical`. |
| `docker/docling-graph/app/_alias_map.py` | MODIFY (~+50 LOC) | Add `MISSILE_IDENTITY_LABELS`, `RADAR_IDENTITY_LABELS`, `CROSS_ENTITY_REF_PATTERNS`, `CANONICAL_PRIORITY` constants. |
| `docker/docling-graph/app/schemas.py` | MODIFY (~+30 LOC) | Add `TableOverlay`, `TableFact`, `CrossEntityHint` Pydantic models. Add `table_overlay: TableOverlay \| None` field to `ExtractPassResponse`. |
| `docker/docling-graph/app/main.py` | MODIFY (~+10 LOC) | Call `extract_table_overlay` after sanitize, before LLM extraction. Attach to response. Wrap in try/except (overlay parsing failure must not break extract-pass). |
| `app/services/table_overlay.py` | NEW (~200 LOC) | Two functions operating on Pydantic instances via `iter_entities_of_type`: `apply_identity_rewrite(pass_results, alias_map, ontology) -> RewriteStats` and `apply_field_overlay(pass_results, table_facts, *, policy="additive_only") -> OverlayStats`. |
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
1. Identity rewrite (deterministic, alias collapse)
   ─ Inputs: pass_output entities + alias_map (from doc's variants table)
   ─ Output: pass_output with system_name aliases rewritten to canonical
   ─ Effect: when merge_and_resolve runs, multiple alias-vertices collapse
              onto one canonical vertex, fields union
   ─ Wall: O(N entities × M alias-lookups), milliseconds

2. Per-cell field overlay (deterministic, additive-only gate)
   ─ Inputs: pass_output (post-rewrite) + table_facts + active pass's schema
   ─ Output: pass_output with table-derived field values applied where
              entity[field] was null (additive_only policy)
   ─ Wall: O(F facts × E entities), milliseconds
```

Both run in `extraction_merge.canonicalize_cross_pass_identities` BEFORE
its existing token-overlap pass. The token-overlap pass remains as fallback
for entities not matched by tables (e.g., prose-only mentions).

### 4.3 Kill switch

Single env var allows operator-controlled rollback without code change:

```yaml
# docker-compose.yml docling-graph service env block
DOCLING_GRAPH_TABLE_OVERLAY_ENABLED: ${DOCLING_GRAPH_TABLE_OVERLAY_ENABLED:-true}
```

When `false`, `extract_table_overlay` short-circuits to empty `TableOverlay`;
worker-side overlay is a no-op. Restart docling-graph + worker (env var
change) → fully reverted to pre-overlay behavior.

## 5. Components

### 5.1 `_alias_map.py` — Constants

```python
# Entity-type-specific identity-row label patterns. Match is case-insensitive
# substring against the row's label_text.
MISSILE_IDENTITY_LABELS: tuple[str, ...] = (
    "missile type", "missile variant",
    "industry designation", "military designation", "nato designation",
    "system designation", "designation", "variant",
)

RADAR_IDENTITY_LABELS: tuple[str, ...] = (
    "radar variant", "radar designation", "radar type",
)

# Cross-entity reference rows: row labels that name a SIBLING entity type.
# When seen in a missile-context table, the row's cells are not missile
# aliases — they're radar aliases attached to the same column's missile via a
# relationship hint. Emitted as CrossEntityHint.
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
    """Parse the FIRST column-major / hybrid variants table in doc.tables[].

    Multi-table policy (per §3 Non-Goals): if multiple column-major /
    hybrid tables are encountered, only the FIRST (in doc.tables[] order)
    is processed. Subsequent qualifying tables are logged at INFO with
    `tables_skipped_multi=N` and skipped. Tables of OTHER shape (small,
    row-major-only, no identity rows) are silently skipped — same as today.

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
walker yields Pydantic instances). Both functions therefore operate on
Pydantic instances, mutating in place via `setattr` — same pattern the
existing `_pick_canonical_name` heuristic uses at `extraction_merge.py:1083`.

```python
def apply_identity_rewrite(
    pass_results: dict[str, "PassResult"],
    alias_map: dict[str, str],
    ontology: dict,
) -> RewriteStats:
    """Rewrite system_name aliases to canonical names in-place on
    Pydantic instances across all passes.

    Operates on the same set of entity types canonicalize_cross_pass_identities
    walks (those whose graph_id_fields == ('system_name',)).

    For each pass, for each entity instance of an eligible entity type:
      current = getattr(inst, 'system_name', None)
      if current in alias_map and alias_map[current] != current:
          setattr(inst, 'system_name', alias_map[current])
          rewrites += 1

    Multiple entities may now share canonical system_name; merge_and_resolve's
    LogicalIdentity-keyed merge collapses them onto one MergedEntityRecord.

    Returns RewriteStats(rewrites=N, unique_canonicals=M, passes_touched=K).
    """

def apply_field_overlay(
    pass_results: dict[str, "PassResult"],
    table_facts: list[TableFact],
    *, policy: str = "additive_only",
) -> OverlayStats:
    """Apply per-cell table facts to Pydantic entity instances.

    Facts already carry pass_name (set by parser); each fact is routed to
    the matching pass_result. Within that pass, find the entity instance
    whose system_name == fact.canonical_entity (post-rewrite, so canonical
    names match).

    For each candidate (fact, entity_instance) pair:
      1. Route fact to pass: if fact.pass_name not in pass_results, increment
         skipped_no_entity++ and continue. Otherwise, look up entity in
         pass_results[fact.pass_name] whose system_name == fact.canonical_entity.
         If no such entity exists, skipped_no_entity++ and continue.
      2. Capture original value BEFORE attempting assignment:
           original = getattr(entity_instance, fact.schema_field, None)
      3. Apply per policy (validation gate is implicit in setattr under
         Pydantic v2 validate_assignment=True):
           - 'additive_only' (v1 default):
               If original is not None: skipped_field_populated++; continue
               (no assignment, original preserved by virtue of never writing).
               If original is None:
                   try: setattr(entity_instance, fact.schema_field, fact.value)
                   except (ValidationError, ValueError, TypeError):
                       skipped_validation_fail++; continue
                   applied++; log applied event
           - 'table_wins' (deferred, v2):
                   try: setattr(entity_instance, fact.schema_field, fact.value)
                   except (ValidationError, ValueError, TypeError):
                       skipped_validation_fail++; continue
                   if original is not None: conflicts_overridden++; log override
                   applied++; log applied event
      4. Log per-fact outcome for audit.

    Note on policy ordering: 'additive_only' checks `original is not None`
    BEFORE attempting setattr, so the policy check is the first gate (no
    revert path needed; the original value is never overwritten in the
    skipped case). This avoids the validate-then-revert subtlety where a
    revert to None on a non-Optional field would re-raise.

    Returns OverlayStats(applied=N, skipped_no_entity=, skipped_validation_fail=,
    skipped_field_populated=, conflicts_overridden=).
    """
```

**Implementation note on Pydantic mutation:** all extraction-side
schema models inherit `BaseModel` with `model_config = {"validate_assignment": True}`
so `setattr(inst, field, value)` triggers field-level validation. Type
coercion (str→float, int→float, etc.) and constraint enforcement (Pydantic
field validators on the schema) all run automatically. If a validator
raises, the assignment is rolled back; we increment `skipped_validation_fail`.
This makes the all-4-agree gate self-enforcing — we don't need a separate
schema-introspection step.

**Conflict resolution policy: `additive_only` for v1.**
Rationale: safest first cut, no override of LLM extraction. v1
implementation gates on `original is not None` BEFORE attempting
assignment, so the pre-existing LLM value is never overwritten and
validation only runs on actually-applied facts (single assignment per
applied fact, zero assignments for skipped). Switching to `table_wins`
in v2 is a one-line policy change.

### 5.4 Data shapes

```python
@dataclass(frozen=True)
class TableFact:
    canonical_entity: str       # e.g., "1D"
    schema_field: str           # e.g., "booster_mass_kg"
    value: float | str
    source_label: str           # e.g., "Weight kg"
    section_ctx: str | None     # e.g., "1st Stage"
    pass_name: str              # e.g., "missile_propulsion" — facts route to this pass
    raw_text: str               # e.g., "1135"

@dataclass(frozen=True)
class CrossEntityHint:
    source_canonical: str       # e.g., "1D" (missile canonical)
    target_alias: str           # e.g., "RSNA-75" (radar alias)
    relationship_kind: str      # e.g., "associated_with"

@dataclass(frozen=True)
class TableOverlay:
    alias_map: dict[str, str]
    """Maps any alias name → canonical name. E.g., for SA-2 col 0:
       {"SA-75": "1D", "SA-2A": "1D", ...}. Empty if no variants table found.
       Doc-level: identical across all passes from the same DoclingDocument."""

    facts: list[TableFact]
    """Per-cell facts ready for direct application to canonical entities.
       Each fact carries its own pass_name, allowing the overlay to filter
       by pass when applied."""

    cross_entity_hints: list[CrossEntityHint]
    """Optional: rows like Fan Song Variant in a missile table (deferred:
       collected for future relationship-pass integration but NOT applied in v1)."""

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
    skipped_field_populated: int = 0
    conflicts_overridden: int = 0  # only relevant for policy='table_wins'
    policy_active: str = "additive_only"  # echoed for diagnostics
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
                policy="additive_only",
            )
            logger.info(
                "TABLE_OVERLAY_APPLIED doc_id=%s "
                "field_overlay_applied=%d skipped_no_entity=%d "
                "skipped_validation_fail=%d skipped_field_populated=%d "
                "policy=%s",
                document_id, stats.applied, stats.skipped_no_entity,
                stats.skipped_validation_fail, stats.skipped_field_populated,
                "additive_only",
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
    multiple non-empty overlays disagree."""
    seen: list[TableOverlay] = []
    for pr in pass_results.values():
        ov = getattr(pr, "table_overlay", None)
        if ov is not None and ov.alias_map:
            seen.append(ov)
    if not seen:
        return None
    first = seen[0]
    for other in seen[1:]:
        if other.alias_map != first.alias_map:
            logger.warning(
                "PassResults carry divergent table_overlay.alias_map (parser bug?). "
                "Using first non-empty; first_size=%d other_size=%d",
                len(first.alias_map), len(other.alias_map),
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

  apply_field_overlay(pass_results, table_overlay.facts, policy="additive_only")
  # Each fact carries pass_name → routed to pass_results[fact.pass_name].
  # Gate: original-is-None (additive_only) + Pydantic validate_assignment
  #       (which subsumes the section_ctx + alias + unit checks since
  #        the parser only emitted facts that already passed those).

─────────────────────────────────────────────────────────────────────────
PHASE 5: merge_and_resolve entity loop (existing code, unchanged)
─────────────────────────────────────────────────────────────────────────

Three "1D" instances merge → one MISSILE_SYSTEM vertex with consolidated
fields:
  body_length_m: 10.726     (from "1D" in pass_output)
  body_diameter_m: 0.654    (from overlay, was null)
  total_mass_kg: 2163       (from rewritten "SA-2A" → "1D")
  booster_mass_kg: 1135     (from overlay, was null, propulsion pass)
  booster_time_sec: 4.0     (from overlay)
  booster_thrust: ...       (LLM-extracted, unchanged)
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
| Field already populated AND policy=`additive_only` | Skip fact (LLM value preserved) | DEBUG `skipped_field_populated++` |
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
    "tables_skipped_multi": int,    # multi-table docs: extras beyond first
    "kill_switch_active": bool,     # echoes DOCLING_GRAPH_TABLE_OVERLAY_ENABLED
}
```

Worker log lines per merge:

```
IDENTITY_REWRITE doc_id=<id>
  rewrites=N unique_canonicals=M passes_touched=K

TABLE_OVERLAY_APPLIED doc_id=<id>
  field_overlay_applied=K skipped_no_entity=A skipped_validation_fail=B
  skipped_field_populated=C policy=additive_only
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
| `apply_field_overlay` (additive_only) | Fact for canonical not in pass_output → skipped; value passes Pydantic → applied; value fails Pydantic → skipped; field already populated → skipped (LLM preserved); applied count matches |
| `apply_field_overlay` (table_wins, deferred) | Tests written but disabled / xfail; populated field gets overwritten + override logged. Activated when policy escalates in v2. |
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

def test_field_overlay_additive_only_policy():
    """entity has body_length_m=null + table_fact body_length_m=10.726 →
    after overlay, entity body_length_m=10.726.

    entity has body_length_m=10.5 + same fact → after overlay, body_length_m=10.5
    (LLM preserved). skipped_field_populated++."""

def test_field_overlay_pydantic_validation_gate():
    """table_fact value="not a number" for numeric field → skipped,
    skipped_validation_fail++, entity field stays null."""

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

| Pass | Required ✓ exact (post-A1) | vs alias-only baseline |
|---|---|---|
| `missile_kinematics` | ≥ 4 | no regression |
| `missile_airframe` | ≥ 14 | +6 over baseline 8 |
| `missile_speed_timing` | ≥ 6 | no regression |
| `missile_propulsion` | **≥ 6 of 7 listed variants** (13DM, 13DA, 13DAM, 20D, 20DP, 20DSU, 5Ya23) for `booster_mass_kg` | spec target met |

**No regression on no-table docs:**
- For each of 16 no-table corpus docs and 4 row-major-table docs: re-run extraction post-overlay-deploy at the SAME temperature as the pre-deploy run.
  - **Hard guard:** entity count per pass must match within ±2 (LLM at T=1.0 is non-deterministic; tolerance accommodates run-to-run variance, not regression).
  - **Soft guard:** the system_name set in the post-deploy run must be ⊇ 80% of the pre-deploy set (entities don't disappear; new aliases would be a parser bug since these docs have no variants tables).
  - For temperature-sensitive comparison, run at T=0.0 against a synthetic LLM mock (returns canned response) — exact pass_output match required.

**Wall-time budget:**
- ≤ +5% per /extract-pass call wall time on table-bearing docs.
- 0% on no-table docs.

**Baseline reference for ✓ exact targets:**

The numeric ✓ exact targets in the per-pass table above are pinned to the
post-revert state at commit `1b71150` (table-fact synthesizer reverted,
B1+B2 disabled, alias-only path active). Pre-deploy verification of these
baselines at T=1.0 documented in the 2026-05-06 §20 GT scorecard
(`/tmp/r21_alias_only_backup/`). If those cached results are no longer
available, re-derive by running §20 against commit `1b71150` before
landing the overlay.

## 9. Acceptance Criteria

1. `extract_table_overlay` is wired into `/extract-pass` with catch-and-continue guard.
2. `apply_identity_rewrite` + `apply_field_overlay` exist as pure functions in `app/services/table_overlay.py`.
3. `canonicalize_cross_pass_identities` accepts `table_alias_map` argument; calls rewrite BEFORE its existing token-overlap pass.
4. All unit, integration, drift-guard, and end-to-end tests pass.
5. Operator-driven §20 acceptance run shows: `missile_propulsion ✓ exact ≥ 6 of 7 listed variants` AND no regression on `missile_kinematics ≥ 4`, `missile_airframe ≥ 14`, `missile_speed_timing ≥ 6`.
6. No regression on the 20 no-table corpus docs (system_name set within ±2 entities and ⊇ 80% of pre-deploy set per §8.6).
7. Wall-time delta ≤ +5% on table-bearing docs; 0% on no-table docs.
8. Kill switch (`DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false`) cleanly reverts behavior without code change.
9. Diagnostics surfaced: `service_table_overlay` in /extract-pass response; `IDENTITY_REWRITE` / `TABLE_OVERLAY_APPLIED` log lines from the worker.

## 10. Implementation Order (suggested)

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
