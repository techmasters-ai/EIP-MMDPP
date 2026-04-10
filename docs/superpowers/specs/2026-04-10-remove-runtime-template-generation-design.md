# Design: Remove Runtime Ontology → Template Generation

**Date:** 2026-04-10
**Status:** Draft — awaiting spec review and user final approval
**Branch target:** `feature/extraction-refactor` (off `feature/arcadedb`)

---

## Context

The extraction hot path currently builds Pydantic schemas at runtime from `ontology.yaml` via `docker/docling-graph/app/template_builder.py`. This design has produced a steady stream of bugs:

- Heuristic fallback `id_fields` (first-property) marked arbitrary properties as required, causing Pydantic validation failures when the LLM omitted them.
- Reserved-word remapping (`TABLE` → `TABLE_REF`) created false node-ID collisions.
- The full unified schema exceeds the Ollama 8KB `format=<JSONSchema>` cutoff, silently degrading to `format="json"` — losing strict output validation exactly where it matters most.
- A dynamic `ontology_definition` field on the extraction request allowed each ingest to mutate the schema at will, preventing any static analysis of "what the LLM is being asked to produce."
- The layered extraction code (`app/services/layered_extraction.py` + `app/services/ontology_layers.py`) reads a per-relationship `validation_matrix` shape that does not exist in the actual ontology YAML — the layer subsetting has been producing empty relationship sets against production data since it was written.
- Silent fail-open from layered → single-pass hid layered-extraction failures behind a misleading "success" signal.

At 46 entity types and 50 relationship types, the dynamic template-build path is the highest-brittleness surface in the pipeline.

This spec removes runtime ontology → template generation from the extraction hot path. The ontology remains the domain contract. A new, narrower **extraction contract** lives in hand-authored Pydantic modules under a checked-in **bundle directory**. A checker script enforces consistency at CI time. Bundle selection is threaded through the pipeline so different use cases can run against different bundles without any runtime schema construction.

## Goals

1. Delete runtime `create_model()` / `build_unified_template` calls from the extraction hot path.
2. Replace with hand-authored Pydantic extraction schemas under `ontology_bundles/<bundle_key>/extraction_schemas/`.
3. Make ontology authoring errors fail at CI time instead of at extraction time.
4. Preserve multi-ontology support via a bundle selection mechanism — not runtime schema building from arbitrary payloads.
5. Restructure extraction into 5 source-centric passes with explicit required/optional semantics and honest failure signals.
6. Unify execution, yield, and skip semantics so metrics and gating are unambiguous.
7. Keep the existing `ontology.yaml` in place as the domain contract; only the extraction-side Pydantic layer changes.

## Non-goals

See §8.9 for the full list. Most important:

- Upstream fixes to the LiteLLM or `NodeIDRegistry` monkey-patches. Contract tests are added instead.
- Community detection refactoring.
- Shipping more than one bundle. Only `air_defense_v3` ships.
- Atomic property restoration for global-scoped entities on failed ingest.
- Build-time codegen (Approach B). Reserved as a future optimization.

## Approach

**Approach A — hand-authored fixed schemas, disk-based bundles, worker-side orchestration.** Considered and rejected: (B) build-time codegen, (C) fixing runtime generation in place. A is the only option that removes the brittle feature; C preserves it; B adds tooling complexity without enough payoff at the current scale (23 extract-bucket entities out of 46 total).

Key refinements locked during brainstorming:

1. `manifest.yaml` is authoritative for pass names, required/optional status, primary vs bridge entity declarations, and module paths. The worker and service communicate only in `(bundle_key, pass_name)` — never individual schema filenames. This preserves the ability to bolt on codegen later without renegotiating runtime contracts.
2. Rollout is strictly additive until switchover: new executor path first, worker cutover second, legacy deletion last.

---

## §1 — Architecture

The refactor separates three contracts that the current code conflates, and introduces a **bundle** as the unit of versioning across all three.

**Domain contract — `ontology.yaml`.** Canonical description of the domain: all entity types, all relationship types, and the validation_matrix. Used by ArcadeDB schema sync, graph validation, admin views, query planning, and documentation. Not consumed by the extraction hot path. For `air_defense_v3` this is the current `ontology/ontology.yaml` moved into the bundle directory, with two additive per-entity fields (`identity_fields`, `identity_scope`).

**Extraction contract — hand-authored Pydantic modules under `extraction_schemas/`.** A narrower operational contract. Describes only the entity types and relationships the LLM is actually asked to extract, grouped into source-centric passes. Each module is plain Python — importable, testable, statically analyzable. Validators for messy LLM output (page numbers, confidence, enum normalization) live here. The LLM only ever sees this contract; it never sees `ontology.yaml`.

**Execution contract — `manifest.yaml` + per-pass StageRun semantics.** Describes how the extraction schemas are orchestrated: pass names, order, required/optional status, primary vs bridge entity declarations, module paths, dependency graph. `manifest.yaml` is the wire-facing layer — read by the worker (for metadata) and by the docling-graph service (to resolve `(bundle_key, pass_name)` into a concrete template class). Per-pass execution status (`COMPLETE`/`FAILED`/`SKIPPED`) and yield status (`HIT`/`EMPTY`/`BRIDGES_ONLY`/`DEGRADED`) are persisted on StageRun rows.

**The boundary that matters:** the domain contract can grow without touching the extraction contract. The extraction contract can be narrowed or widened without touching the domain contract. The manifest couples them at build time (via the coverage checker) and at runtime (via the orchestrator). Each is independently diffable, reviewable, and testable.

**Bundle as the unit of versioning.** A bundle is a directory containing one (ontology + extraction + execution + post-merge derivation rules) set. `ontology_bundles/air_defense_v3/` is the only shipped bundle in this spec. Selection is by `bundle_key` string. No dynamic schema construction at any layer.

**Runtime layout:**

```
┌──────────────────────────────────────────────┐
│  API / reingest route                        │
│    - resolves bundle_key from source default │
│      or explicit override                    │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│  Worker: start_ingest_pipeline               │
│    - loads bundle manifest                   │
│    - snapshots bundle audit fields on        │
│      PipelineRun                             │
│    - schedules derive_ontology_graph stage   │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│  Worker: derive_ontology_graph orchestrator  │
│    - iterates manifest.passes in order       │
│    - for each pass: calls docling-graph,     │
│      writes per-pass StageRun, records       │
│      execution + yield status                │
│    - runs required-pass gate (before merge)  │
│    - merges results, runs typed-ref          │
│      resolution, applies derive_rules.py     │
│    - three-phase graph import                │
│    - writes DocumentGraphExtraction snapshot │
└──────────────────────┬───────────────────────┘
                       ↓ HTTP (one call per pass)
┌──────────────────────────────────────────────┐
│  docling-graph service                       │
│    - POST /extract-pass                      │
│    - input: {bundle_key, pass_name,          │
│              docling_document_json,          │
│              upstream_entities?}             │
│    - loads pass module from bundle manifest  │
│    - runs docling_graph.run_pipeline with    │
│      that single fixed template              │
│    - returns normalized extraction result    │
│    - STATELESS: no merge, no ref resolution  │
└──────────────────────────────────────────────┘
```

**What changes in the service:**
1. Loses `build_unified_template` / `build_templates_with_edges` / the entire runtime schema-building surface.
2. Request contract reduced to `(bundle_key, pass_name, docling_document_json, upstream_entities?)`. No ontology override.
3. Holds no orchestration state — all orchestration, merge, and validation logic moves to the worker.

**What changes in the worker:**
1. `derive_ontology_graph` becomes a pass-iterator instead of a single-call.
2. Gains typed-ref merge + the execution/yield status model.
3. Threads bundle selection through `Source`, `PipelineRun`, `DocumentGraphExtraction`.

Everything else — Docling conversion, chunking, embedding, picture descriptions, community detection, search — stays untouched.

---

## §2 — Bundle layout and manifest contract

### Directory layout

```
ontology_bundles/
├── __init__.py                      # makes ontology_bundles importable
└── air_defense_v3/
    ├── __init__.py
    ├── ontology.yaml                # moved from ontology/ontology.yaml
    ├── manifest.yaml                # NEW: pass registry
    ├── coverage.yaml                # NEW: extract/derive buckets
    ├── extraction_schemas/
    │   ├── __init__.py
    │   ├── reference.py
    │   ├── radar_domain.py
    │   ├── missile_domain.py
    │   ├── other_systems.py
    │   └── system_links.py
    ├── validators.py                # shared @field_validator helpers
    └── derive_rules.py              # deterministic post-merge derivations
```

Everything under `ontology_bundles/<bundle_key>/` is checked into git.

### `manifest.yaml` schema

The single source of truth for what passes exist, in what order, with what input semantics. Read by the worker, the docling-graph service, and the coverage checker.

```yaml
bundle_key: air_defense_v3
manifest_schema_version: "1.0.0"
ontology_name: "EIP Military Equipment Ontology"
ontology_version: "3.0.0"
extraction_profile_version: "1.0.0"

passes:
  - name: reference
    required: true
    kind: entities
    input_mode: document_only
    module: extraction_schemas.reference
    template_class: ReferencePass
    primary_entity_types: [SECTION, FIGURE, TABLE, ASSERTION]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: radar_domain
    required: true
    kind: entities_and_relationships
    input_mode: document_only
    module: extraction_schemas.radar_domain
    template_class: RadarDomainPass
    primary_entity_types:
      [RADAR_SYSTEM, ANTENNA, RECEIVER, TRANSMITTER, SIGNAL_PROCESSING_CHAIN,
       FREQUENCY_BAND, WAVEFORM]
    bridge_entity_types: [PLATFORM, SPECIFICATION]
    extracted_relationship_types:
      [INSTALLED_ON, HAS_ANTENNA, HAS_RECEIVER, HAS_TRANSMITTER,
       HAS_PROCESSING_CHAIN, OPERATES_IN_BAND, USES_WAVEFORM, SPECIFIED_BY]
    depends_on: []

  - name: missile_domain
    required: true
    kind: entities_and_relationships
    input_mode: document_only
    module: extraction_schemas.missile_domain
    template_class: MissileDomainPass
    primary_entity_types:
      [MISSILE_SYSTEM, LAUNCHER_SYSTEM, GUIDANCE_METHOD, SEEKER, PROPULSION_STACK]
    bridge_entity_types: [PLATFORM, SPECIFICATION]
    extracted_relationship_types:
      [INSTALLED_ON, HAS_GUIDANCE, HAS_SEEKER, HAS_PROPULSION, LAUNCHES, SPECIFIED_BY]
    depends_on: []

  - name: other_systems
    required: true
    kind: entities_and_relationships
    input_mode: document_only
    module: extraction_schemas.other_systems
    template_class: OtherSystemsPass
    primary_entity_types:
      [AIR_DEFENSE_ARTILLERY_SYSTEM, ELECTRONIC_WARFARE_SYSTEM,
       FIRE_CONTROL_SYSTEM, WEAPON_SYSTEM, INTEGRATED_AIR_DEFENSE_SYSTEM]
    bridge_entity_types: [PLATFORM, SPECIFICATION]
    extracted_relationship_types: [INSTALLED_ON, SPECIFIED_BY]
    depends_on: []

  - name: system_links
    required: true
    kind: relationships_only
    input_mode: document_plus_entity_refs
    module: extraction_schemas.system_links
    template_class: SystemLinksPass
    primary_entity_types: []
    bridge_entity_types: []
    extracted_relationship_types: [ASSOCIATED_WITH, CUES]
    depends_on: [radar_domain, missile_domain, other_systems]
    skip_if_no_upstream_endpoints: true
    skip_justification: >
      When none of the upstream domain passes produce any linkable system
      entities, there is nothing for system_links to link. Skipping is
      preferable to dispatching an LLM call that can only fail.
```

Key fields:
- `manifest_schema_version` — bumped when the manifest structure changes
- `kind` — `entities` | `entities_and_relationships` | `relationships_only`
- `input_mode` — `document_only` | `document_plus_entity_refs`
- `primary_entity_types` — entity types owned by this pass (counted toward `primary_entities_extracted`)
- `bridge_entity_types` — entity types duplicated across passes (counted toward `bridge_entities_extracted`)
- `depends_on` — pass names this pass depends on. For `input_mode=document_plus_entity_refs`, those passes' merged entities are passed as `upstream_entities`. For skip logic, those passes' outputs determine whether this pass has satisfiable work.
- `skip_if_no_upstream_endpoints` — only valid for `relationships_only` passes; triggers the skip logic in `_should_skip()`

### `coverage.yaml` schema

Declares which ontology types are `extract` or `derive`. `validate_only` is computed by the checker as `ontology_types - extract - derive`.

```yaml
bundle_key: air_defense_v3
version: "1.0.0"

entity_types:
  extract:
    - SECTION
    - FIGURE
    - TABLE
    - ASSERTION
    - RADAR_SYSTEM
    - PLATFORM
    - ANTENNA
    - RECEIVER
    - TRANSMITTER
    - SIGNAL_PROCESSING_CHAIN
    - FREQUENCY_BAND
    - WAVEFORM
    - SPECIFICATION
    - MISSILE_SYSTEM
    - LAUNCHER_SYSTEM
    - GUIDANCE_METHOD
    - SEEKER
    - PROPULSION_STACK
    - AIR_DEFENSE_ARTILLERY_SYSTEM
    - ELECTRONIC_WARFARE_SYSTEM
    - FIRE_CONTROL_SYSTEM
    - WEAPON_SYSTEM
    - INTEGRATED_AIR_DEFENSE_SYSTEM
  derive: []

relationship_types:
  extract:
    - INSTALLED_ON
    - HAS_ANTENNA
    - HAS_RECEIVER
    - HAS_TRANSMITTER
    - HAS_PROCESSING_CHAIN
    - OPERATES_IN_BAND
    - USES_WAVEFORM
    - SPECIFIED_BY
    - HAS_GUIDANCE
    - HAS_SEEKER
    - HAS_PROPULSION
    - LAUNCHES
    - ASSOCIATED_WITH
    - CUES
  derive:
    - HAS_PROVENANCE
    - MENTIONED_IN
    - CONTAINS_TEXT
    - CONTAINS_IMAGE
```

**Bucket semantics:**
- **`extract`** — the LLM is asked to produce this type, and the checker enforces that an extraction schema exists for it.
- **`derive`** — produced deterministically post-merge by the worker. The worker writes these via one of two deterministic paths: (a) `derive_rules.derive_structural_edges()` in phase 4 (e.g. `MENTIONED_IN`), or (b) the auto-creation inside `graph_store.upsert_nodes_batch_sync` when a non-None `ProvenanceMetadata` is passed in phase 2 (specifically `HAS_PROVENANCE`). Either path is a valid implementation of "derived" — the key invariant is that the LLM is not asked to produce these, they are computed from extraction provenance or document structure. See §3.8 for the split between the two paths.
- **`validate_only`** (computed) — everything in `ontology.yaml` that is neither `extract` nor `derive`. Present in the domain contract for completeness (used by admin views, schema sync, validation matrix) but never asked of the LLM and never produced by deterministic rules. Includes `DOCUMENT` (see §3.5), `COMPONENT`, `SUBSYSTEM`, `CAPABILITY`, `ENGAGEMENT_TIMELINE`, and ~20 others.

### Bundle loader API

**Worker side — `app/services/ontology_bundles.py`:**

```python
class PassManifest(BaseModel):
    name: str
    required: bool
    kind: Literal["entities", "entities_and_relationships", "relationships_only"]
    input_mode: Literal["document_only", "document_plus_entity_refs"]
    module: str
    template_class: str
    primary_entity_types: list[str]
    bridge_entity_types: list[str]
    extracted_relationship_types: list[str]
    depends_on: list[str]
    skip_if_no_upstream_endpoints: bool = False
    skip_justification: str | None = None

class BundleManifest(BaseModel):
    bundle_key: str
    manifest_schema_version: str
    ontology_name: str
    ontology_version: str
    extraction_profile_version: str
    passes: list[PassManifest]

    def find_pass(self, pass_name: str) -> PassManifest: ...

def load_bundle_manifest(bundle_key: str) -> BundleManifest: ...
def list_available_bundles() -> list[str]: ...
def load_bundle_ontology(bundle_key: str) -> dict: ...

def resolve_bundle_key(
    *,
    explicit_override: str | None,
    source_default: str | None,
    system_default: str,
) -> str: ...

def resolve_bundle_key_for_graph_only(
    *,
    explicit_override: str | None,
    inherited_from_run: str | None,
    source_default: str | None,
    system_default: str,
) -> str: ...
```

The worker-side loader does NOT import extraction schema modules. It only reads manifest metadata (pass names, required flags, primary/bridge declarations) to orchestrate and classify results.

**Service side — `docker/docling-graph/app/bundles.py`:**

```python
def load_bundle_manifest(bundle_key: str) -> BundleManifest: ...

def load_pass_template(bundle_key: str, pass_name: str) -> type[BaseModel]:
    """Import the module and return the template class declared in manifest.
    All bundles are pre-imported on service startup for fast per-request dispatch."""
```

The service-side loader does import the extraction schema modules. It caches loaded templates at service startup. On miss, it raises `UnknownBundleOrPassError` — never silently falls back.

### Load ontology split

`load_ontology()` in `app/services/ontology_templates.py` is refactored. `prefer_active` is dropped from the public signature.

```python
def load_ontology(
    *,
    bundle_key: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Load an ontology definition.

    Resolution order (exactly one applies):
    1. If `path` is given, load directly from that file.
       (For tests and admin tools pointing at arbitrary files.)
    2. Else if `bundle_key` is given, load that bundle's ontology.yaml.
    3. Else load the system default bundle's ontology.yaml.

    This function never consults the registry/version-pinning store.
    For version-pinned loads, call load_registry_ontology() explicitly.
    """

def load_registry_ontology(version_id: str) -> dict[str, Any]:
    """Load a version-pinned ontology snapshot from the
    registry.ontology_versions table. Used only by audit and
    historical-reproduction code paths — never by the live extraction hot path."""
```

All existing no-arg callers get the system default bundle's ontology (the same file they used to read, just at a new location). Callers that previously passed `prefer_active=True` for registry lookups switch to `load_registry_ontology(version_id)`.

### Structured output threshold

The 8KB Ollama `format=<JSONSchema>` cutoff lives in service config, not bundle metadata.

```python
# docker/docling-graph/app/config.py
class ServiceSettings(BaseSettings):
    structured_output_threshold_chars: int = 8000
    # ...
```

Both the runtime monkey-patch in `main.py` and `tools/check_extraction_coverage.py` read the same value. Bundles cannot override it.

### Wire contract

```python
# docker/docling-graph/app/schemas.py
from typing import Any, Literal
from pydantic import BaseModel

class EntityRef(BaseModel):
    ref_id: str                      # opaque, e.g. "E01" — assigned by worker
    entity_type: str
    identity: dict[str, Any]         # values of the entity's identity_fields
    display_label: str
    pass_origin: str

class ExtractPassRequest(BaseModel):
    bundle_key: str
    pass_name: str
    docling_document_json: dict
    upstream_entities: list[EntityRef] | None = None

class ExtractionMetadata(BaseModel):
    schema_size_chars: int
    structured_output_mode: Literal["strict", "json_only"]
    salvaged: bool
    model: str
    provider: str
    duration_ms: int

class ExtractPassResponse(BaseModel):
    bundle_key: str
    pass_name: str
    extraction: dict                 # Pydantic-validated pass template output
    metadata: ExtractionMetadata
```

**Error responses from the service:**
- `400` — input_mode mismatch (document_only with upstream_entities, or vice versa)
- `404` — `UnknownBundleOrPassError`
- `408` — LLM call timeout (retryable at worker)
- `422` — Pydantic validation of extracted JSON failed after salvage (terminal)
- `500` — internal service error (retryable)

### Checker rules (14-slot list; 13 active rules + one removed placeholder)

Rules are numbered 1–14 for stability of cross-references. Rule 7 is a placeholder for a deleted rule and is intentionally skipped — 13 rules are actively enforced.

`tools/check_extraction_coverage.py` runs in CI. For every bundle:

1. **Coverage subset.** `coverage.entity_types.extract ∪ derive ⊆ ontology.yaml entity_types`. Checker computes `validate_only = ontology_types - extract - derive` for reporting.
2. **Manifest entities in extract bucket.** Every entity in `manifest.passes[].primary_entity_types ∪ bridge_entity_types` is in `coverage.entity_types.extract`.
3. **Manifest relationships in extract bucket.** Every relationship in `manifest.passes[].extracted_relationship_types` is in `coverage.relationship_types.extract`.
4. **Validation matrix existence.** Every extracted relationship has at least one row in `ontology.yaml`'s top-level `validation_matrix`.
5. **Schema size.** For each pass, import its template class, measure its JSON Schema size, assert `size <= settings.structured_output_threshold_chars` (same value used by runtime).
6. **Partial safety (recursive).** Every non-system field at every depth of every template class is Optional with `default` or `default_factory`. Walks into `list[T]`, `Optional[T]`, `Union[T, …]`.
7. *(removed — replaced by rule 6)*
8. **Extraction ⊆ ontology.** Every Pydantic field on an extraction model exists as a declared property of that entity type in the bundle's `ontology.yaml`, with a compatible type mapping.
9. **Identity completeness.** Every field listed in `identity_fields` for an extract-bucket entity appears as a Pydantic field on that entity's extraction model.
10. **Display label.** Every extract-bucket entity type produces a non-empty display label from minimal inputs via `build_display_label`.
11. **Identity scope required.** Every entity in `coverage.entity_types.extract` has `identity_scope: document` or `identity_scope: global` in `ontology.yaml`.
12. **Identity scope advisory (warning, not error).** Entity with `identity_fields: []` and `identity_scope: global` emits a CI warning.
13. **Bridge scope consistency.** Bridge entities used in multiple passes have the same `identity_scope` across all passes.
14. **Rejection reason coverage.** Every `RelationshipRejectionReason` enum value has at least one test fixture in `tests/unit/test_extraction_merge.py`.

**Manifest self-consistency sub-checks** (enforced alongside the numbered rules):
- Unique pass names.
- `depends_on` references pass names that exist and are declared earlier in `passes:`.
- `primary_entity_types ∩ bridge_entity_types == ∅` per pass.
- `kind == relationships_only` implies `primary_entity_types == []` and `bridge_entity_types == []`.
- `skip_if_no_upstream_endpoints: true` permitted only when `kind == relationships_only`.
- `input_mode == document_plus_entity_refs` ⇒ `depends_on` non-empty AND template class has no entity-collection fields (reflection check).
- **Satisfiability:** every extracted relationship in a pass has at least one `validation_matrix` row whose `source` and `target` are both in `primary ∪ bridge` of this pass ∪ `primary ∪ bridge` of every pass in `depends_on`.

**Curation advisory (warning, not error):** the checker prints the count and names of ontology properties that exist for each entity type but are NOT included in the extraction schema. Advisory for human review.

### Packaging

1. `docker-compose.yml`: `docling-graph.build.context: .`, `docling-graph.build.dockerfile: docker/docling-graph/Dockerfile`.
2. `docker/docling-graph/Dockerfile`: `COPY docker/docling-graph/app /app/app` and `COPY ontology_bundles /app/ontology_bundles`.
3. `docker/worker/Dockerfile`: `COPY ontology_bundles /app/ontology_bundles`.
4. Both images: `ENV PYTHONPATH=/app`.
5. CI smoke test: `python -c "from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import RadarDomainPass"` inside both worker and docling-graph images.

---

## §3 — Pass definitions and ownership

### §3.1 — Design principles (all passes)

1. **One top-level class per module**, named in `manifest.template_class`.
2. **Every field at every nesting depth is Optional** with a default. `None` for scalars, `list[X] = Field(default_factory=list)` for collections. The checker's rule 6 enforces this recursively.
3. **`ConfigDict(extra="ignore")`** on every model. LLM-emitted extra keys are silently dropped.
4. **No synthetic `name` field.** Entity Pydantic classes mirror `ontology.yaml` property names directly. Identity is declared per entity type via `identity_fields` in `ontology.yaml` — not by convention.
5. **Bridge entities and primary entities have identical Pydantic shapes.** Primary/bridge is a manifest declaration, not a class-level distinction. The worker uses the manifest to bucket counts.
6. **Shared validators** live in `ontology_bundles/air_defense_v3/validators.py` and are wired via `field_validator(..., mode="before")`.
7. **Relationships always use typed refs.** Same-pass relationships carry `from_identity`/`to_identity` dicts; cross-pass `system_links` uses `from_ref_id`/`to_ref_id`. Never bare names.

### §3.2 — Shared validators

`ontology_bundles/air_defense_v3/validators.py`:

```python
def coerce_optional_int(v):
    """Accepts int, None, numeric string, or text with embedded number.
    Returns None for anything unparseable. Never raises."""
    if v is None: return None
    if isinstance(v, int): return v
    if isinstance(v, str):
        s = v.strip()
        if not s: return None
        try: return int(s)
        except ValueError: pass
        import re
        match = re.search(r"-?\d+", s)
        if match:
            try: return int(match.group())
            except ValueError: pass
    return None

def coerce_optional_float(v):
    """Same shape; handles decimals, percentages, scientific notation."""
    ...

def coerce_optional_confidence(v):
    """Accepts 0.0-1.0 floats, 0-100 percentages, or text labels
    ('high'/'medium'/'low'). Returns None for unparseable."""
    if v is None: return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f / 100.0 if f > 1.0 else f
    if isinstance(v, str):
        s = v.strip().lower()
        textual = {"high": 0.9, "confident": 0.9, "medium": 0.6,
                   "moderate": 0.6, "low": 0.3, "uncertain": 0.3}
        if s in textual: return textual[s]
        try:
            f = float(s.replace("%", ""))
            return f / 100.0 if f > 1.0 else f
        except ValueError:
            return None
    return None

def normalize_enum(allowed: set[str]):
    """Factory: returns a validator that maps freeform labels to allowed
    values, or None if no match."""
    def _validate(v):
        if v is None: return None
        if not isinstance(v, str): return None
        canonical = v.strip().upper().replace(" ", "_")
        return canonical if canonical in allowed else None
    return _validate
```

### §3.3 — Pattern A: entities-only pass (`reference.py`)

```python
"""Reference pass: document structure.

Extracts document-level anchors. No LLM-extracted relationships. Post-merge,
MENTIONED_IN edges are produced by derive_rules.derive_structural_edges;
HAS_PROVENANCE edges are auto-created by graph_store.upsert_nodes_batch_sync
in phase 2 (see §3.8 and §5.6 Phase 2) — NOT by derive_rules.
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from ..validators import coerce_optional_int, coerce_optional_confidence


class SectionEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    heading: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    confidence: Optional[float] = None

    _v_ps = field_validator("page_start", mode="before")(coerce_optional_int)
    _v_pe = field_validator("page_end", mode="before")(coerce_optional_int)
    _v_c  = field_validator("confidence", mode="before")(coerce_optional_confidence)


class FigureEntity(BaseModel): ...      # figure_id, page, caption, figure_type
class TableEntity(BaseModel): ...       # table_id, page, caption
class AssertionEntity(BaseModel): ...   # assertion_text, confidence


class ReferencePass(BaseModel):
    """Top-level template for the reference pass."""
    model_config = ConfigDict(extra="ignore")

    sections: list[SectionEntity] = Field(default_factory=list)
    figures: list[FigureEntity]   = Field(default_factory=list)
    tables: list[TableEntity]     = Field(default_factory=list)
    assertions: list[AssertionEntity] = Field(default_factory=list)
    # NO documents field — DOCUMENT is not extracted; see §3.5.
```

**Extraction field set rule:** the Pydantic field set for each entity is a **curated subset** of the ontology's properties — all `identity_fields` plus a hand-picked selection of high-signal properties. NOT an exhaustive mirror. The coverage checker's **curation advisory** (§2 checker, described below the numbered rule list) prints a warning listing omitted properties for human review. It is advisory, not a build failure.

### §3.4 — Pattern B: entities + relationships pass (`radar_domain.py`)

```python
class RadarRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rel_type: Optional[str] = Field(
        default=None,
        description="One of: INSTALLED_ON, HAS_ANTENNA, HAS_RECEIVER, "
                    "HAS_TRANSMITTER, HAS_PROCESSING_CHAIN, OPERATES_IN_BAND, "
                    "USES_WAVEFORM, SPECIFIED_BY",
    )
    from_type: Optional[str] = None                    # e.g. "RADAR_SYSTEM"
    from_identity: Optional[dict[str, Any]] = None     # e.g. {"system_name": "AN/MPQ-53"}
    to_type: Optional[str] = None                      # e.g. "ANTENNA"
    to_identity: Optional[dict[str, Any]] = None
    confidence: Optional[float] = None

    _v_rel = field_validator("rel_type", mode="before")(
        normalize_enum({"INSTALLED_ON", "HAS_ANTENNA", "HAS_RECEIVER",
                        "HAS_TRANSMITTER", "HAS_PROCESSING_CHAIN",
                        "OPERATES_IN_BAND", "USES_WAVEFORM", "SPECIFIED_BY"})
    )
    _v_conf = field_validator("confidence", mode="before")(coerce_optional_confidence)


class RadarDomainPass(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Primary
    radar_systems: list[RadarSystemEntity]         = Field(default_factory=list)
    antennas: list[AntennaEntity]                  = Field(default_factory=list)
    receivers: list[ReceiverEntity]                = Field(default_factory=list)
    transmitters: list[TransmitterEntity]          = Field(default_factory=list)
    signal_processing_chains: list[SPCEntity]      = Field(default_factory=list)
    frequency_bands: list[FrequencyBandEntity]     = Field(default_factory=list)
    waveforms: list[WaveformEntity]                = Field(default_factory=list)

    # Bridges (same Pydantic shape — primary/bridge is manifest metadata)
    platforms: list[PlatformEntity]                = Field(default_factory=list)
    specifications: list[SpecificationEntity]      = Field(default_factory=list)

    relationships: list[RadarRelationship] = Field(default_factory=list)
```

`missile_domain.py` and `other_systems.py` follow this pattern with different entity sets and different `rel_type` enum constraints.

### §3.5 — Pattern C: relationships-only pass (`system_links.py`)

`DOCUMENT` is **not extracted** in v1. The existing structural `Document` vertex at `pipeline.py:2645` carries document-level metadata via existing upstream stages (`derive_document_metadata`). Extracting a separate ontology `DOCUMENT` entity would create a second document concept competing with the structural root. `DOCUMENT` is in `coverage.validate_only`; its properties can be populated by a future `document_metadata_extraction` pass (non-goal for this spec).

```python
class SystemLinkRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rel_type: Optional[str] = None
    from_ref_id: Optional[str] = None
    to_ref_id: Optional[str] = None
    confidence: Optional[float] = None

    _v_rel = field_validator("rel_type", mode="before")(
        normalize_enum({"ASSOCIATED_WITH", "CUES"})
    )
    _v_conf = field_validator("confidence", mode="before")(coerce_optional_confidence)


class SystemLinksPass(BaseModel):
    """Relationships-only pass. Consumes upstream_entities from the request body.
    Has NO entity fields — enforced by the 'input_mode == document_plus_entity_refs
    implies no entity-collection fields' sub-check in the manifest self-consistency
    section of §2 checker rules."""
    model_config = ConfigDict(extra="ignore")

    relationships: list[SystemLinkRelationship] = Field(default_factory=list)
```

**Service prompt construction for this pass kind:** when `input_mode == document_plus_entity_refs`, the service prepends a compact upstream-entities preamble to the document content:

```
Available entities extracted from this document:

[E01] RADAR_SYSTEM     "Fan Song"
[E02] RADAR_SYSTEM     "Low Blow"
[E03] MISSILE_SYSTEM   "SA-2 Guideline"
[E04] PLATFORM         "Su-75"

Identify ASSOCIATED_WITH and CUES relationships. Reference entities by
their bracketed ID (from_ref_id="E01", to_ref_id="E03"). Do not invent new IDs.
```

The worker assigns `ref_id` tokens (`E01`, `E02`, …) when building the upstream list. Post-merge resolution drops relationships whose `ref_id` doesn't match any upstream entry (rejection reason: `UNKNOWN_REF_ID`).

### §3.6 — Identity model

Every entity type in the `extract` bucket declares `identity_fields` and `identity_scope` in `ontology.yaml`:

```yaml
entity_types:
  # Document-scoped: identity meaningful only within the containing document
  - name: SECTION
    identity_fields: [heading, page_start]
    identity_scope: document

  - name: FIGURE
    identity_fields: [figure_id, page]
    identity_scope: document

  - name: SPECIFICATION
    identity_fields: [parameter, value]
    identity_scope: document

  - name: ENGAGEMENT_TIMELINE
    identity_fields: []              # content-hash fallback
    identity_scope: document         # anonymous entities are document-local

  # Global-scoped: identity represents a shared real-world thing
  - name: RADAR_SYSTEM
    identity_fields: [system_name]
    identity_scope: global

  - name: PLATFORM
    identity_fields: [name]
    identity_scope: global

  - name: FREQUENCY_BAND
    identity_fields: [band_name]
    identity_scope: global
```

**Scope semantics:**
- **`document`** — identity tuple effectively includes `document_id`. Two documents extracting identical identity values produce two distinct vertices.
- **`global`** — identity tuple omits `document_id`. Two documents extracting the same identity collapse into one vertex.

**Empty `identity_fields: []`** — content-hash fallback. The merge code hashes the sorted non-system field values of the Pydantic extraction model instance, where "non-system" means "all fields declared on the entity's extraction model except `confidence` and any identity-system fields injected at upsert time (e.g., `document_id` for document-scoped entities)." The hash is deterministic across calls with the same inputs (JSON-serialized with sorted keys, SHA-1 prefixed). Strongly recommended to pair with `identity_scope: document` (checker warns on `[]` + `global` — rule 12). In v1, no extract-bucket entity uses `identity_fields: []` — only `ENGAGEMENT_TIMELINE` does, and it's in `validate_only`, so this code path is reachable only via future additions. Unit tests (§8.4 `test_extraction_merge.py`) lock in the determinism invariant so future additions don't silently regress.

**Logical identity at runtime:**

```python
@dataclass(frozen=True)
class LogicalIdentity:
    entity_type: str
    identity_field_names: tuple[str, ...]   # ordered field names from ontology.yaml
    identity_tuple: tuple[Any, ...]          # parallel values
    scope: Literal["document", "global"]
    document_id: str | None                  # populated iff scope == "document"

    def identity_values_dict(self) -> dict[str, Any]:
        """Identity field names zipped with values. Does NOT include document_id.
        Used for display-label construction and for identity comparison in merge."""
        return dict(zip(self.identity_field_names, self.identity_tuple, strict=True))

    def as_upsert_identity_dict(self) -> dict[str, Any]:
        """Shape expected by GraphStore.NodeRecord.identity_fields. Adds
        document_id for document-scoped entities so the composite identity
        distinguishes same-named entities across documents."""
        d = self.identity_values_dict()
        if self.scope == "document":
            assert self.document_id is not None, "document_id required for scope=document"
            d["document_id"] = self.document_id
        return d
```

Both `identity_values_dict()` and `as_upsert_identity_dict()` are used — the former for display labels and merge comparison, the latter for `NodeRecord.identity_fields`. Do not conflate them.

**Schema sync change:** every document-scoped entity vertex class in ArcadeDB gains a `document_id: STRING` property. The composite unique index on that class becomes `(identity_fields..., document_id)`.

### §3.7 — Merge and resolve

`app/services/extraction_merge.py`:

```python
@dataclass
class PassResult:
    """The normalized output of one /extract-pass call, as seen by the
    worker after parsing the ExtractPassResponse. Handoff type between
    _run_single_pass (producer) and merge_and_resolve (consumer)."""
    pass_name: str
    template_instance: BaseModel           # instantiated Pydantic template class
    metadata: ExtractionMetadata           # from service response (schema size, etc.)
    pre_merge_rejections: list[tuple[Any, RelationshipRejectionReason]]

    def iter_entities_of_type(self, entity_type: str) -> Iterable[BaseModel]:
        """Return the nested entity models matching the given type."""
        ...

    @property
    def relationships(self) -> list[BaseModel]:
        """Return the relationships field (empty list for entities-only passes)."""
        ...


@dataclass
class MergedEntityRecord:
    identity: LogicalIdentity
    properties: dict[str, Any]        # merged from all source passes
    confidence: float                 # highest confidence across merges
    pass_origins: set[str]            # which passes contributed
    display_label: str                # derived from identity_tuple


@dataclass
class MergedEdgeRecord:
    from_identity: LogicalIdentity
    to_identity: LogicalIdentity
    rel_type: str
    confidence: float
    source_pass: str


@dataclass
class MergedExtraction:
    entities: list[MergedEntityRecord]
    edges: list[MergedEdgeRecord]
    rejected_edges: list[tuple[str, Any, RelationshipRejectionReason]]
    rejections_by_pass: dict[str, int]
    pipeline_run_id: str
    document_id: str


def merge_and_resolve(
    pass_results: dict[str, PassResult],
    manifest: BundleManifest,
    ontology: dict,
    document_id: str,
    pipeline_run_id: str,
) -> MergedExtraction:
    """Phase 1: merge entities; resolve edges against logical identity.
    No backend RIDs involved. Pure logical-identity IR."""
    ...
```

**Key properties:**
1. Entities are keyed by `LogicalIdentity`. Bridge entities with identical identities across passes collapse into one `MergedEntityRecord` with both pass names in `pass_origins`.
2. Relationships are resolved post-merge by looking up each endpoint's `LogicalIdentity` in the merged entity index.
3. Same-pass relationships: identity-dict lookup. Cross-pass relationships: `ref_id` lookup against the worker-assembled upstream set.
4. Rejections are counted per pass and per reason.
5. Confidence defaulting: `confidence = 0.8 if rel.confidence is None else rel.confidence`. Explicit `0.0` is preserved.

**Rejection taxonomy:**

```python
class RelationshipRejectionReason(str, Enum):
    MISSING_REL_TYPE          = "missing_rel_type"
    INVALID_IDENTITY_PAYLOAD  = "invalid_identity_payload"
    UNKNOWN_REF_ID            = "unknown_ref_id"
    FROM_ENDPOINT_NOT_FOUND   = "from_endpoint_not_found"
    TO_ENDPOINT_NOT_FOUND     = "to_endpoint_not_found"
    INVALID_TRIPLE            = "invalid_triple"
```

**Ambiguous endpoint match is not a reachable case.** `LogicalIdentity`-keyed merge collapses entities with identical identity tuples by construction. Two passes producing the same entity result in ONE `MergedEntityRecord`; resolution against that record is a dict lookup with exactly one hit or none.

### §3.8 — Derived structural edges

`ontology_bundles/air_defense_v3/derive_rules.py`:

```python
@dataclass
class ChunkForDerivation:
    """DTO used by derive_structural_edges. Distinct from the SQLAlchemy
    TextChunk ORM model — carries only the fields derivation needs.
    Constructed by the worker from TextChunk rows before calling derive_rules."""
    rid: str                    # ArcadeDB vertex RID of this chunk
    text_normalized: str        # lowercased, whitespace-collapsed text

@dataclass
class DerivedEdge:
    """Output of derive_structural_edges. Uses RID-based endpoints because
    both source (extracted entity) and target (Document/TextChunk) RIDs are
    already known at derivation time."""
    from_id: str                # extracted entity RID (from identity_to_rid)
    to_id: str                  # Document or TextChunk RID
    rel_type: str
    confidence: float | None

def derive_structural_edges(
    merged: MergedExtraction,
    identity_to_rid: dict[LogicalIdentity, str],
    chunks: list[ChunkForDerivation],
    document_rid: str,
) -> list[DerivedEdge]:
    """Deterministic edges that are NOT extracted by the LLM.
    Only rules whose outputs are 100% deterministic given the inputs.

    IMPORTANT: HAS_PROVENANCE edges are NOT produced here. They are
    created automatically by graph_store.upsert_nodes_batch_sync via
    its internal _create_provenance_edges_batch_sync helper whenever a
    non-None ProvenanceMetadata is passed. Duplicating them here would
    produce two HAS_PROVENANCE edges per entity. See §5.6 Phase 2 for
    the provenance-metadata handoff."""
    edges: list[DerivedEdge] = []

    # Rule: MENTIONED_IN from each entity to chunks containing its display label
    for entity in merged.entities:
        from_rid = identity_to_rid.get(entity.identity)
        if from_rid is None:
            continue  # shouldn't happen — every merged entity was just upserted
        canonical = normalize_name(entity.display_label)
        if not canonical:
            continue
        for chunk in chunks:
            if canonical in chunk.text_normalized:
                edges.append(DerivedEdge(
                    from_id=from_rid,
                    to_id=chunk.rid,
                    rel_type="MENTIONED_IN",
                    confidence=entity.confidence,
                ))

    # CONTAINS_TEXT / CONTAINS_IMAGE / NEXT_CHUNK are handled by the existing
    # derive_structure_links stage — not duplicated here.
    # HAS_PROVENANCE is handled by upsert_nodes_batch_sync — see above.

    return edges
```

The worker imports `derive_rules.py` directly (not through the service). The identity→RID map is built in phase 2 of the import and passed through. `ChunkForDerivation` is a lightweight DTO constructed by the worker from SQLAlchemy `TextChunk` rows — it is deliberately NOT the ORM model, to avoid coupling derivation logic to persistence concerns.

**HAS_PROVENANCE ownership:** `graph_store.upsert_nodes_batch_sync` auto-creates `HAS_PROVENANCE` edges from each upserted node to the structural `Document` vertex via its internal `_create_provenance_edges_batch_sync` helper whenever a non-None `ProvenanceMetadata` is passed. The worker passes a `ProvenanceMetadata` with `document_id` populated in phase 2; the auto-created edges are the canonical source of HAS_PROVENANCE. `derive_rules.py` is explicitly out of this business.

### §3.9 — Display label helper

`app/services/extraction_merge.py`:

```python
def build_display_label(
    entity_type: str,
    identity_values: dict[str, Any],
    properties: dict[str, Any],
) -> str:
    """Compose a human-readable display label for a NodeRecord.name field.

    Resolution order:
    1. First "name-like" key (system_name, name, title, heading, document_id)
       present in identity_values with a truthy value.
    2. Concatenation of non-empty identity_values joined by " / ".
    3. First "name-like" key present in properties with a truthy value.
    4. Deterministic fallback: "<entity_type>_<short-hash-of-identity-tuple>".
    """
    NAME_LIKE_KEYS = ("system_name", "name", "title", "heading", "document_id")

    for key in NAME_LIKE_KEYS:
        v = identity_values.get(key)
        if v: return str(v)

    non_empty = [str(v) for v in identity_values.values() if v]
    if non_empty:
        return " / ".join(non_empty)

    for key in NAME_LIKE_KEYS:
        v = properties.get(key)
        if v: return str(v)

    identity_hash = hashlib.sha1(
        json.dumps(identity_values, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return f"{entity_type}_{identity_hash}"
```

Checker rule 10 asserts this returns a non-empty string for every extract-bucket entity type given minimal inputs.

**Call-path note:** `build_display_label` is called from §5.6 Phase 2 with `identity_values_dict()` as the second argument — NOT `as_upsert_identity_dict()`. `identity_values_dict()` returns only the ontology-declared identity fields (e.g., `system_name` for RADAR_SYSTEM, `parameter + value` for SPECIFICATION). It does NOT include `document_id`, even for document-scoped entities. The `document_id` key only appears when `as_upsert_identity_dict()` is called at upsert time to build the composite storage identity — that dict is not passed to `build_display_label`. So `document_id` never appears in the `identity_values` argument and never contributes a bare-UUID display label. `NAME_LIKE_KEYS` including `document_id` is a legacy safety fallback for future callers that might pass an alternative identity dict; it has no effect in the current flow.

### §3.10 — Ownership map

| File / directory | Owned by (can import) | Enforced by |
|---|---|---|
| `ontology.yaml` | worker, service, checker, tests | data loaders only |
| `manifest.yaml` | worker, service, checker | data loaders only |
| `coverage.yaml` | checker | data loaders only |
| `validators.py` | `extraction_schemas/*` (intra-bundle) | convention |
| `extraction_schemas/*.py` | **service only** | CI lint: `grep -r "from ontology_bundles\.[^.]*\.extraction_schemas" app/` returns zero |
| `derive_rules.py` | worker (post-merge) | convention |

The worker treats the bundle as a data directory plus one helper module (`derive_rules.py`). The service treats it as a data directory plus a Python package of extraction schemas.

---

## §4 — Data model changes

### §4.1 — `Source` (two new nullable columns)

```python
class Source(Base):
    # ... existing columns ...

    default_ontology_bundle_key: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
        doc="Default bundle_key for documents ingested from this source. "
            "NULL resolves to the system default bundle."
    )
    default_use_case_key: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
        doc="Optional use-case identifier for retrieval alignment. "
            "Not consumed by the extraction hot path in phase 1."
    )
```

### §4.2 — `PipelineRun` (one required + five nullable snapshot columns)

```python
class PipelineRun(Base):
    # ... existing columns ...

    mode: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="full",
        doc="Pipeline mode: 'full' (all stages) or 'graph_only' "
            "(derive_ontology_graph only). CHECK constraint enforces "
            "mode IN ('full', 'graph_only')."
    )
    ontology_bundle_key: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ontology_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ontology_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    use_case_key: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    extraction_profile_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True,
        doc="Run-level metrics blob. Populated after derive_ontology_graph "
            "merge completes. Stores document_extraction_anomaly flag, "
            "pass_degraded_count, overall_relationship_rejection_ratio, "
            "rejected_relationships_sample, bundle_legacy, and related "
            "quality signals. See §6.6."
    )
```

Snapshot fields (bundle_key, ontology_name, etc.) are populated once at pipeline start and never updated. The `metrics` field is populated once at stage exit and holds the run-level quality rollup.

### §4.3 — `DocumentGraphExtraction` (FK + snapshot)

```python
class DocumentGraphExtraction(Base):
    # ... existing columns ...

    pipeline_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ingest.pipeline_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    ontology_bundle_key: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ontology_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    ontology_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    use_case_key: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    extraction_profile_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
```

**Latest-snapshot semantics.** `DocumentGraphExtraction` is keyed on `document_id` — one row per document, reflecting the latest successful `derive_ontology_graph` run. Historical extraction ledger lives on `PipelineRun` + `StageRun`, not here.

**Interaction with `purge_document_derivations` (full mode — important caveat).** The `full` ingest pipeline runs these stages in order: `prepare_document`, `detect_and_translate`, `derive_document_metadata`, **`purge_document_derivations`**, `derive_picture_descriptions`, `derive_text_chunks_and_embeddings`, `derive_image_embeddings`, `derive_ontology_graph`, `finalize_document` (see `app/workers/pipeline.py:440, 443, 444, 448`). When `purge_document_derivations` executes, it deletes the existing `DocumentGraphExtraction` row (`pipeline.py:1707`) AND the document-level graph state in ArcadeDB, including `TextChunk`, `ImageChunk`, and the structural `Document` vertex (`arcadedb_graph.py:401`, `pipeline.py:1717`).

**`full` reingests go through three distinct phases with respect to snapshot preservation:**

1. **Pre-purge phase** — current stage is one of `prepare_document`, `detect_and_translate`, `derive_document_metadata`, and `purge_document_derivations` has not yet executed. The prior `DocumentGraphExtraction` row and ArcadeDB graph state are still intact. A failure in any of these stages leaves the prior snapshot queryable. The authoritative status API pseudocode (§7.10) correctly reports `graph_snapshot != null` and `graph_queryable == true` in this phase because it queries current PostgreSQL state, not historical intent.

2. **Post-purge, pre-`derive_ontology_graph`-success phase** — purge has executed, deleting the prior snapshot row and graph. `derive_ontology_graph` may not have started, may be in progress, or may have failed. During this window the document has no `DocumentGraphExtraction` row at all. `graph_snapshot == null` and `graph_queryable == false`. Any failure in this phase (including `derive_ontology_graph` gate/pre-mutation/mutation-time/pre-import failures and any failure in the intermediate chunking/embedding stages between purge and `derive_ontology_graph`) leaves the document with no snapshot and no queryable graph until a fresh successful ingest runs.

3. **Post-`derive_ontology_graph`-success phase** — the new extraction has written a replacement `DocumentGraphExtraction` row and populated ArcadeDB. Downstream stages (`finalize_document`, etc.) may still fail, but such failures do not affect the extraction snapshot or its graph — the extraction stage's rollback semantics are scoped to the extraction stage itself.

**This spec does NOT change the purge semantics.** Redesigning `purge_document_derivations` to preserve the prior snapshot until the replacement succeeds is a separate architectural change (shadow-write / staging-namespace / atomic-swap on success) and is explicitly out of scope. The spec treats the existing purge behavior as a **first-class graph invalidation event** and propagates that honesty through §5.4 (terminalization), §6.8 (rollback scope), §7.10 (status API semantics), and §8.5 (test assertions).

**Scope of rollback/retention guarantees in this spec:**

| Scenario | Prior snapshot preserved on failure? | Why |
|---|---|---|
| `graph_only` reingest, gate failure | **Yes** | No writes occurred; rollback not triggered. |
| `graph_only` reingest, mutation-time failure | **No** (rollback deletes the document's extraction-layer graph) | See §6.8 accepted limitation. |
| `graph_only` reingest, pre-mutation failure inside phase helper | **Yes** | `tracker.any_mutation_attempted == False`; rollback not triggered. |
| `full` reingest, **pre-purge** stage failure | **Yes** | Purge hasn't run yet. Prior snapshot is still the current `DocumentGraphExtraction` row. |
| `full` reingest, **post-purge** failure (any stage, including `derive_ontology_graph`) | **No** | `purge_document_derivations` already removed the prior snapshot and graph. No rollback can restore it. |
| `full` reingest, success | **N/A** | New snapshot replaces nothing (purge removed the prior one); the new row is just a fresh insert. |

The rollback semantics in §6.8, the status API in §7.10, and the failure-flow tests in §8.5 are all scoped according to this table. For `full` mode the distinction between "pre-purge" and "post-purge" failure is load-bearing.

**`graph_json` carry-forward naming.** The column name is retained from the pre-refactor schema. Its contents in the new path are an **audit blob** (counts, rejection summary, pass outcomes) — NOT a serialized graph payload. The actual graph lives in ArcadeDB. Docstring updated in PR 3 to reflect the new semantics; no column rename.

### §4.4 — `StageRun` (first-class columns, not metrics JSON)

The existing `status` column is the Celery-level task status and is NOT overloaded. New first-class columns carry extraction-specific semantics.

```python
class StageRun(Base):
    # ... existing columns unchanged ...

    pass_name: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
        doc="Extraction pass name, or NULL for non-extraction stages and "
            "for the stage-summary row (one per derive_ontology_graph invocation)."
    )

    execution_status: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True,
        doc="COMPLETE | FAILED | SKIPPED. NULL for non-extraction stages."
    )
    yield_status: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True,
        doc="HIT | EMPTY | BRIDGES_ONLY | DEGRADED. "
            "Meaningful iff execution_status == 'COMPLETE'."
    )
    skip_reason: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True,
        doc="NO_UPSTREAM_ENDPOINTS | ... "
            "Meaningful iff execution_status == 'SKIPPED'."
    )

    primary_entities_extracted: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bridge_entities_extracted:  Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    relationships_extracted:    Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    relationships_rejected:     Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    salvaged:                   Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    schema_size_chars:          Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    structured_output_mode:     Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Populated only on the stage-summary row for derive_ontology_graph
    # (pass_name IS NULL). Indicates whether the failure path called
    # _attempt_rollback(). One of two inputs to the status API's
    # top-level graph_queryable signal (see §7.10 authoritative rule —
    # the other input is snapshot existence). NOT derived from
    # error_message string-prefix heuristics.
    #   - NULL   — summary row not yet finalized, or non-extraction stage
    #   - False  — stage completed without calling rollback (success,
    #              gate failure, merge/import failure before first mutation,
    #              or pre-import failure)
    #   - True   — rollback was invoked (graph mutations had started;
    #              extraction-layer graph for this document was deleted)
    rollback_executed:          Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
```

**Row-kind split:**
- **Pass rows** — `pass_name IS NOT NULL`. One per `(pipeline_run_id, stage_name, pass_name, attempt)`. Uniquely indexed.
- **Summary row** — `pass_name IS NULL` for `stage_name = 'derive_ontology_graph'`. One per stage invocation representing the overall stage outcome. Uniquely indexed separately.
- **Other-stage rows** — `pass_name IS NULL` for `stage_name != 'derive_ontology_graph'`. Existing non-extraction stages, unchanged.

**Indexes (in the Alembic migration):**

```sql
-- Explicitly drop the existing (pipeline_run_id, stage_name, attempt) uniqueness
-- constraint before adding new ones. PR 1 discovers the exact constraint name
-- via pg_constraint lookup and uses it in the migration.
ALTER TABLE ingest.stage_runs DROP CONSTRAINT <existing_uq_name>;

CREATE UNIQUE INDEX uq_stage_runs_run_pass_attempt
    ON ingest.stage_runs (pipeline_run_id, stage_name, pass_name, attempt)
    WHERE pass_name IS NOT NULL;

CREATE UNIQUE INDEX uq_stage_runs_summary_row
    ON ingest.stage_runs (pipeline_run_id, stage_name, attempt)
    WHERE pass_name IS NULL AND stage_name = 'derive_ontology_graph';
-- Scoped narrowly to the derive_ontology_graph summary row. Without the
-- second predicate the index would also govern every existing non-extraction
-- stage row, which could fail the migration on pre-existing duplicates.

CREATE INDEX ix_stage_runs_extraction_pass
    ON ingest.stage_runs (stage_name, pass_name)
    WHERE stage_name = 'derive_ontology_graph';

CREATE INDEX ix_stage_runs_execution_status
    ON ingest.stage_runs (execution_status)
    WHERE execution_status IS NOT NULL;
```

**Latest-attempt view:**

```sql
CREATE VIEW ingest.v_latest_pass_attempts AS
SELECT DISTINCT ON (pipeline_run_id, stage_name, pass_name)
    id, pipeline_run_id, stage_name, pass_name, attempt,
    status, execution_status, yield_status, skip_reason,
    primary_entities_extracted, bridge_entities_extracted,
    relationships_extracted, relationships_rejected, salvaged,
    schema_size_chars, structured_output_mode,
    error_message, started_at, finished_at
FROM ingest.stage_runs
WHERE pass_name IS NOT NULL
ORDER BY pipeline_run_id, stage_name, pass_name, attempt DESC;
```

### §4.5 — Bundle selection threading

```
POST /sources (body includes default_ontology_bundle_key)
    → Source row persisted

POST /documents/upload  (or watcher picks up a file)
    → Document row created; no bundle selection yet
    → start_ingest_pipeline(document_id) scheduled

start_ingest_pipeline(document_id, *, ontology_bundle_key=None, use_case_key=None)
    → resolve_bundle_key(
          explicit_override=ontology_bundle_key,
          source_default=document.source.default_ontology_bundle_key,
          system_default=settings.default_ontology_bundle_key,
      )
    → manifest = load_bundle_manifest(resolved_key)
    → PipelineRun.create(
          ontology_bundle_key=resolved_key,
          ontology_name=manifest.ontology_name,
          ontology_version=manifest.ontology_version,
          ...
      )
    → IngestDispatchResult(pipeline_run_id, celery_task_id) returned

POST /documents/{id}/reingest?mode=graph_only
    → resolve_bundle_key_for_graph_only(
          explicit_override=request.ontology_bundle_key,
          inherited_from_run=latest_run.ontology_bundle_key,
          source_default=source.default_ontology_bundle_key,
          system_default=settings.default_ontology_bundle_key,
      )
    → new PipelineRun row with resolved snapshot
    → derive_ontology_graph.delay(pipeline_run_id)
```

**`graph_only` precedence:** explicit → inherited from latest run → source default → system default. Reproducibility beats source-policy drift. Legacy-NULL latest run (pre-refactor) falls through to source/system default.

### §4.6 — Legacy-NULL semantics

**`NULL` on `ontology_bundle_key` means `legacy/unknown`, not `air_defense_v3`.** New runs always populate a real key after resolution; resolution failure raises `BundleResolutionError` (config bug). API reads return both the raw key and a display label:

```python
LEGACY_BUNDLE_LABEL = "legacy/unknown"

def describe_bundle_for_display(bundle_key: str | None) -> str:
    return bundle_key if bundle_key else LEGACY_BUNDLE_LABEL
```

Response shape:

```json
{
  "ontology_bundle_key": null,
  "ontology_bundle_label": "legacy/unknown"
}
```

`ontology_bundle_key` is never transformed in output — what was stored is what is returned, including `null`. Only `ontology_bundle_label` carries display text.

**Aggregate metrics exclude legacy rows.** Cross-run aggregates like `required_pass_failure_rate`, `domain_entity_extraction_rate`, and `edge_retention_rate` are computed only over runs with `ontology_bundle_key IS NOT NULL`. Legacy rows (pre-refactor, `NULL` key) are counted separately in a `legacy_rows_excluded` counter so the exclusion is visible. Per-run metrics written to `PipelineRun.metrics` (§6.6) include a `bundle_legacy: bool` flag on every row for traceability — that flag is a tag on the individual row, not a contradiction of the exclusion rule. A reader can always answer both "what's the domain extraction rate over non-legacy runs?" and "how many runs were tagged legacy?" from the same data.

### §4.7 — `QueryProfileRegistry` — unchanged

`app/models/query_profiles.py` is explicitly untouched. The `ontology_definition` JSONB column stays in place because retrieval still uses it. The extraction path no longer reads it. Any future alignment with bundles is a separate spec.

### §4.8 — Alembic migration

One migration file adds all of the above in one upgrade transaction.

```python
def upgrade() -> None:
    # Source
    op.add_column('sources', sa.Column('default_ontology_bundle_key', sa.String(100), nullable=True), schema='ingest')
    op.add_column('sources', sa.Column('default_use_case_key', sa.String(100), nullable=True), schema='ingest')

    # PipelineRun
    op.add_column('pipeline_runs', sa.Column('mode', sa.String(32), nullable=False, server_default='full'), schema='ingest')
    op.add_column('pipeline_runs', sa.Column('ontology_bundle_key', sa.String(100), nullable=True), schema='ingest')
    op.add_column('pipeline_runs', sa.Column('ontology_name', sa.String(255), nullable=True), schema='ingest')
    op.add_column('pipeline_runs', sa.Column('ontology_version', sa.String(100), nullable=True), schema='ingest')
    op.add_column('pipeline_runs', sa.Column('use_case_key', sa.String(100), nullable=True), schema='ingest')
    op.add_column('pipeline_runs', sa.Column('extraction_profile_version', sa.String(100), nullable=True), schema='ingest')
    op.add_column('pipeline_runs', sa.Column('metrics', postgresql.JSONB, nullable=True), schema='ingest')
    op.create_check_constraint(
        'chk_pipeline_run_mode', 'pipeline_runs',
        "mode IN ('full', 'graph_only')", schema='ingest',
    )

    # DocumentGraphExtraction
    op.add_column('document_graph_extractions', sa.Column('pipeline_run_id', postgresql.UUID(as_uuid=True), nullable=True), schema='ingest')
    op.create_foreign_key(
        'fk_dge_pipeline_run', 'document_graph_extractions', 'pipeline_runs',
        ['pipeline_run_id'], ['id'],
        source_schema='ingest', referent_schema='ingest', ondelete='SET NULL',
    )
    op.add_column('document_graph_extractions', sa.Column('ontology_bundle_key', sa.String(100), nullable=True), schema='ingest')
    op.add_column('document_graph_extractions', sa.Column('ontology_name', sa.String(255), nullable=True), schema='ingest')
    op.add_column('document_graph_extractions', sa.Column('ontology_version', sa.String(100), nullable=True), schema='ingest')
    op.add_column('document_graph_extractions', sa.Column('use_case_key', sa.String(100), nullable=True), schema='ingest')
    op.add_column('document_graph_extractions', sa.Column('extraction_profile_version', sa.String(100), nullable=True), schema='ingest')

    # StageRun — explicitly drop the old uniqueness constraint first
    # PR 1 verifies the exact constraint name via:
    #   SELECT conname FROM pg_constraint
    #    WHERE conrelid = 'ingest.stage_runs'::regclass AND contype = 'u';
    op.drop_constraint('<existing_uq_name>', 'stage_runs', schema='ingest', type_='unique')

    # StageRun columns
    op.add_column('stage_runs', sa.Column('pass_name', sa.String(64), nullable=True), schema='ingest')
    op.add_column('stage_runs', sa.Column('execution_status', sa.String(16), nullable=True), schema='ingest')
    op.add_column('stage_runs', sa.Column('yield_status', sa.String(16), nullable=True), schema='ingest')
    op.add_column('stage_runs', sa.Column('skip_reason', sa.String(32), nullable=True), schema='ingest')
    op.add_column('stage_runs', sa.Column('primary_entities_extracted', sa.Integer, nullable=True), schema='ingest')
    op.add_column('stage_runs', sa.Column('bridge_entities_extracted', sa.Integer, nullable=True), schema='ingest')
    op.add_column('stage_runs', sa.Column('relationships_extracted', sa.Integer, nullable=True), schema='ingest')
    op.add_column('stage_runs', sa.Column('relationships_rejected', sa.Integer, nullable=True), schema='ingest')
    op.add_column('stage_runs', sa.Column('salvaged', sa.Boolean, nullable=True), schema='ingest')
    op.add_column('stage_runs', sa.Column('schema_size_chars', sa.Integer, nullable=True), schema='ingest')
    op.add_column('stage_runs', sa.Column('structured_output_mode', sa.String(32), nullable=True), schema='ingest')
    op.add_column('stage_runs', sa.Column('rollback_executed', sa.Boolean, nullable=True), schema='ingest')

    # Indexes
    op.create_index('uq_stage_runs_run_pass_attempt', 'stage_runs',
                    ['pipeline_run_id', 'stage_name', 'pass_name', 'attempt'],
                    unique=True, postgresql_where=sa.text('pass_name IS NOT NULL'),
                    schema='ingest')
    op.create_index('uq_stage_runs_summary_row', 'stage_runs',
                    ['pipeline_run_id', 'stage_name', 'attempt'],
                    unique=True,
                    postgresql_where=sa.text(
                        "pass_name IS NULL AND stage_name = 'derive_ontology_graph'"
                    ),
                    schema='ingest')
    op.create_index('ix_stage_runs_extraction_pass', 'stage_runs',
                    ['stage_name', 'pass_name'],
                    postgresql_where=sa.text("stage_name = 'derive_ontology_graph'"),
                    schema='ingest')
    op.create_index('ix_stage_runs_execution_status', 'stage_runs',
                    ['execution_status'],
                    postgresql_where=sa.text('execution_status IS NOT NULL'),
                    schema='ingest')

    # View
    op.execute("""CREATE VIEW ingest.v_latest_pass_attempts AS ...""")


def downgrade() -> None:
    # Reverse order — drop view, drop indexes, drop columns, restore old constraint
    ...
```

All new columns are nullable (except `PipelineRun.mode` which has `server_default='full'`). No backfill. `NULL` = `legacy/unknown` at read time.

---

## §5 — Runtime flow

### §5.1 — Entry points

Three entry points trigger ingest:
1. **`POST /documents/upload`** — upload handler calls `start_ingest_pipeline(document_id)`.
2. **Watch-directory task** — `scan_watch_directories` calls `start_ingest_pipeline(document_id)`.
3. **`POST /documents/{id}/reingest`** — full re-run or `mode=graph_only`. Accepts optional `ontology_bundle_key` and `use_case_key` overrides.

### §5.2 — `start_ingest_pipeline`

```python
@dataclass(frozen=True)
class IngestDispatchResult:
    pipeline_run_id: str
    celery_task_id: str

def start_ingest_pipeline(
    document_id: str,
    *,
    ontology_bundle_key: str | None = None,
    use_case_key: str | None = None,
) -> IngestDispatchResult:
    document = session.get(Document, document_id)
    source = document.source

    resolved_key = resolve_bundle_key(
        explicit_override=ontology_bundle_key,
        source_default=(source.default_ontology_bundle_key if source else None),
        system_default=settings.default_ontology_bundle_key,
    )
    manifest = load_bundle_manifest(resolved_key)

    run = PipelineRun(
        id=uuid.uuid4(),
        document_id=document_id,
        mode="full",
        status="PROCESSING",
        ontology_bundle_key=resolved_key,
        ontology_name=manifest.ontology_name,
        ontology_version=manifest.ontology_version,
        use_case_key=use_case_key or (source.default_use_case_key if source else None),
        extraction_profile_version=manifest.extraction_profile_version,
        started_at=now(),
    )
    session.add(run)
    session.commit()

    async_result = dispatch_ingest_chain(run.id)
    return IngestDispatchResult(
        pipeline_run_id=str(run.id),
        celery_task_id=async_result.id,
    )
```

Callers at `app/api/v1/sources.py` update mechanically:

```python
result = start_ingest_pipeline(document_id)
document.celery_task_id = result.celery_task_id
return {
    "document_id": str(document.id),
    "celery_task_id": result.celery_task_id,
    "pipeline_run_id": result.pipeline_run_id,
}
```

### §5.3 — `graph_only` reingest path

```python
def reingest_graph_only(doc_id: UUID, request: ReingestRequest) -> dict:
    document = session.get(Document, doc_id)
    latest_run = (
        session.query(PipelineRun)
        .filter_by(document_id=doc_id)
        .order_by(PipelineRun.started_at.desc())
        .first()
    )
    inherited_bundle = (
        latest_run.ontology_bundle_key
        if latest_run and latest_run.ontology_bundle_key else None
    )

    resolved_key = resolve_bundle_key_for_graph_only(
        explicit_override=request.ontology_bundle_key,
        inherited_from_run=inherited_bundle,
        source_default=(document.source.default_ontology_bundle_key if document.source else None),
        system_default=settings.default_ontology_bundle_key,
    )

    if inherited_bundle is None and latest_run is not None:
        logger.info(
            "reingest_graph_only: latest run for document %s is legacy "
            "(ontology_bundle_key NULL); bundle inferred from source/system default (%s)",
            doc_id, resolved_key,
        )

    manifest = load_bundle_manifest(resolved_key)
    new_run = PipelineRun(
        id=uuid.uuid4(),
        document_id=doc_id,
        mode="graph_only",
        status="PROCESSING",
        ontology_bundle_key=resolved_key,
        ontology_name=manifest.ontology_name,
        ontology_version=manifest.ontology_version,
        use_case_key=request.use_case_key or (latest_run.use_case_key if latest_run else None),
        extraction_profile_version=manifest.extraction_profile_version,
        started_at=now(),
    )
    session.add(new_run)
    session.commit()

    async_result = derive_ontology_graph.delay(pipeline_run_id=str(new_run.id))
    return {
        "pipeline_run_id": str(new_run.id),
        "celery_task_id": async_result.id,
        "ontology_bundle_key": resolved_key,
    }
```

### §5.4 — `derive_ontology_graph` orchestrator

```python
@celery_app.task(bind=True, name="derive_ontology_graph", queue="graph")
def derive_ontology_graph(self, pipeline_run_id: str) -> dict:
    run = session.get(PipelineRun, pipeline_run_id)

    # Stage-summary row: one per stage invocation, represents overall outcome.
    # StageRun.status uses "RUNNING" / "COMPLETE" / "FAILED" (Celery-level),
    # which is distinct from PipelineRun.status's "PROCESSING" / "COMPLETE" /
    # "FAILED" vocabulary. The spec does not change either set.
    #
    # Note: StageRun.status has a model default of "PENDING"
    # (app/models/ingest.py:252). This row deliberately overrides that default
    # with "RUNNING" at creation time, matching the convention used by existing
    # per-stage writers in app/workers/pipeline.py (e.g., the prepare_document
    # _update_stage_run call with status="RUNNING"). Downstream observers filter
    # on "RUNNING" to find in-flight stages.
    stage_summary = StageRun(
        pipeline_run_id=run.id,
        stage_name="derive_ontology_graph",
        pass_name=None,
        attempt=self.request.retries + 1,
        status="RUNNING",
        started_at=now(),
    )
    session.add(stage_summary)
    session.commit()

    # Tracks whether any ArcadeDB mutation has been ATTEMPTED in this run.
    # The phase-2/3/4 helpers receive this tracker and call .mark() IMMEDIATELY
    # BEFORE issuing their first graph_store mutation. Failure branches below
    # read `tracker.any_mutation_attempted` to decide whether rollback is
    # needed. This handles both pre-import failures AND pre-mutation failures
    # inside the phase helpers — if a helper throws while building records
    # or computing derivations BEFORE attempting any write, the tracker stays
    # False and rollback is skipped.
    tracker = GraphWriteTracker()

    try:
        manifest = load_bundle_manifest(run.ontology_bundle_key)
        ontology = load_ontology(bundle_key=run.ontology_bundle_key)
        doc_json = _build_docling_document_json(run.document_id)

        pass_results: dict[str, PassResult] = {}
        upstream_refs: dict[str, EntityRef] = {}

        for pass_def in manifest.passes:
            _run_single_pass(
                pipeline_run_id=run.id,
                pass_def=pass_def,
                manifest=manifest,
                ontology=ontology,
                bundle_key=run.ontology_bundle_key,
                doc_json=doc_json,
                pass_results=pass_results,
                upstream_refs=upstream_refs,
                document_id=run.document_id,
            )
            # Raises IngestFailed if a required pass fails after retries.

        # Required-pass gate — before merge
        gate = check_required_pass_gate(run.id)
        if not gate.passed:
            raise IngestFailed(f"Required passes failed: {gate.failures}")

        # Merge and resolve
        merged = merge_and_resolve(
            pass_results=pass_results,
            manifest=manifest,
            ontology=ontology,
            document_id=run.document_id,
            pipeline_run_id=str(run.id),
        )

        # Post-merge yield updates (HIT → DEGRADED when rejection ratio high)
        _apply_post_merge_yield_updates(run.id, merged)

        # Run-level metrics: written BEFORE import so merge diagnostics
        # survive any later import failure. See §6.6.
        _write_pipeline_run_metrics(run.id, merged, manifest)

        # Three-phase graph import.
        # Each helper receives the tracker and calls tracker.mark() RIGHT
        # BEFORE its first graph_store mutation. If a helper raises while
        # building records or computing derivations (before the first
        # mutation attempt), tracker.any_mutation_attempted stays False
        # and the except branches below skip rollback.
        identity_to_rid = _import_graph_phase_nodes(
            merged, ontology, run.document_id, tracker,
        )
        _import_graph_phase_domain_edges(merged, ontology, tracker)
        _import_graph_phase_structural_edges(
            merged, identity_to_rid, run.document_id, str(run.id), tracker,
        )

        # DocumentGraphExtraction upsert (PostgreSQL only — not ArcadeDB;
        # does not flip the tracker). If this fails but earlier phases
        # already mutated ArcadeDB, the catch-all still runs rollback
        # because tracker.any_mutation_attempted is True from phase 2.
        _upsert_document_graph_extraction(
            document_id=run.document_id,
            pipeline_run_id=run.id,
            run=run,
            merged=merged,
        )

        # Stage-summary terminalizes COMPLETE on success (always).
        stage_summary.status = "COMPLETE"
        stage_summary.execution_status = "COMPLETE"
        stage_summary.rollback_executed = False  # success path — no rollback
        stage_summary.finished_at = now()

        # PipelineRun + Document terminalization is MODE-SPECIFIC:
        #   - graph_only: this stage IS the whole run. Terminalize both.
        #   - full:       downstream stages follow (finalize_document, etc.).
        #                 Leave PipelineRun.status = "PROCESSING" and
        #                 Document.pipeline_status untouched — the later
        #                 finalize_document stage terminalizes both.
        if run.mode == "graph_only":
            run.status = "COMPLETE"
            run.finished_at = now()
            # Document.pipeline_status uses the EXISTING vocabulary from
            # app/models/ingest.py:60 — "COMPLETE" is the success value.
            # This spec does NOT introduce a new "OK" value; storage and
            # API both use "COMPLETE".
            _update_document_pipeline_status(run.document_id, "COMPLETE")

        session.commit()
        return {"stage": "derive_ontology_graph", "status": "ok",
                "entities": len(merged.entities),
                "edges": len(merged.edges)}

    except IngestFailed as exc:
        # Gate failure — no graph writes occurred, no rollback needed.
        # Terminalizes the run in BOTH modes because the ingest cannot
        # continue once a required pass has failed.
        stage_summary.status = "FAILED"
        stage_summary.execution_status = "FAILED"
        stage_summary.rollback_executed = False  # never reached phase 2
        stage_summary.error_message = f"gate_failed: {exc}"
        stage_summary.finished_at = now()
        run.status = "FAILED"
        run.finished_at = now()
        _update_document_pipeline_status(run.document_id, "PARTIAL_COMPLETE")
        session.commit()
        raise

    except (MergeError, GraphImportError) as exc:
        # Merge/import failure — partial ArcadeDB writes may exist iff
        # tracker.any_mutation_attempted is True. MergeError fires BEFORE
        # phase 2 (tracker=False). GraphImportError fires during a phase
        # helper; if it fires before the phase's first tracker.mark() call
        # (e.g., while building records), tracker stays False; if after,
        # tracker is True and rollback is required.
        logger.exception("derive_ontology_graph merge/import failure")
        should_rollback = tracker.any_mutation_attempted
        rollback_note = _attempt_rollback(run.document_id) if should_rollback else ""

        stage_summary.status = "FAILED"
        stage_summary.execution_status = "FAILED"
        stage_summary.rollback_executed = should_rollback
        stage_summary.error_message = f"merge_or_import_failed: {exc}{rollback_note}"
        stage_summary.finished_at = now()
        run.status = "FAILED"
        run.finished_at = now()
        _update_document_pipeline_status(run.document_id, "PARTIAL_COMPLETE")
        # DocumentGraphExtraction intentionally NOT updated
        session.commit()
        raise

    except Exception as exc:
        # Catch-all for everything else: bundle loading, post-merge yield
        # updates, metrics write, snapshot write, or any unexpected exception.
        # Rollback is conditional on tracker.any_mutation_attempted — any
        # failure before the first graph_store mutation attempt skips
        # rollback entirely.
        logger.exception("derive_ontology_graph unexpected failure")
        should_rollback = tracker.any_mutation_attempted
        rollback_note = _attempt_rollback(run.document_id) if should_rollback else ""

        try:
            stage_summary.status = "FAILED"
            stage_summary.execution_status = "FAILED"
            stage_summary.rollback_executed = should_rollback
            stage_summary.error_message = f"unexpected_failure: {exc}{rollback_note}"
            stage_summary.finished_at = now()
            run.status = "FAILED"
            run.finished_at = now()
            _update_document_pipeline_status(run.document_id, "PARTIAL_COMPLETE")
            session.commit()
        except Exception as bookkeeping_exc:
            # Even the bookkeeping update failed. Log loudly, rollback the
            # session, and re-raise the ORIGINAL exception so Celery marks
            # the task failed with the real root cause.
            logger.error(
                "derive_ontology_graph: bookkeeping update also failed: %s",
                bookkeeping_exc,
            )
            session.rollback()
        raise  # re-raise the original unexpected exception
```

**Helpers used above:**

```python
# Defined as a private helper in app/workers/pipeline.py — NOT in
# app/services/extraction_merge.py. This is worker-local orchestration
# state tied to the derive_ontology_graph task's rollback semantics,
# not merge logic.
@dataclass
class GraphWriteTracker:
    """Mutable tracker passed into every phase helper. Flipped to True at
    the moment of the first graph_store mutation attempt in each phase.
    Failures before .mark() leave the flag False so rollback is skipped."""
    any_mutation_attempted: bool = False

    def mark(self) -> None:
        self.any_mutation_attempted = True


def _attempt_rollback(document_id: str) -> str:
    """Best-effort rollback via the abstract extraction-layer graph delete.
    Callers must gate this on `tracker.any_mutation_attempted` so failures
    before the first graph_store mutation don't trigger document-level
    graph deletion.

    Returns a diagnostic suffix for the stage error_message — empty on
    success, '; ROLLBACK_ALSO_FAILED: <detail>' on failure."""
    try:
        _delete_extraction_layer_graph(document_id)
        return ""
    except Exception as rollback_exc:
        logger.error(
            "rollback during failure handling also failed: %s", rollback_exc,
        )
        return f"; ROLLBACK_ALSO_FAILED: {rollback_exc}"


def _delete_extraction_layer_graph(document_id: str) -> None:
    """Abstract helper wired in PR 1 to the concrete graph-store method
    that satisfies the contract in §6.8:

    MUST delete:
      - document-scoped extracted entity vertices
      - domain edges tagged with this document_id (provenance metadata)
      - structural edges written by derive_rules in phase 4
        (identifiable via document_id / source=derive_rules properties)

    MUST NOT delete:
      - chunks (TextChunk, ImageChunk) — owned by upstream stages
      - the structural Document vertex — owned by earlier stages
      - global-scoped entity vertices (PLATFORM, RADAR_SYSTEM, etc.)

    PR 1 verifies whether the existing graph_store.delete_document_graph_sync
    satisfies this contract. If yes, this helper is a one-line wrapper. If
    no, PR 1 adds a narrower sibling (e.g., delete_extraction_layer_graph_sync)
    and wires this helper to it. The runtime code above is unaffected by
    the choice — it only sees the abstract name. See residual check #1."""
    # PR 1 picks ONE of the following based on residual check #1 verification:
    graph_store.delete_document_graph_sync(document_id)
    # OR:
    # graph_store.delete_extraction_layer_graph_sync(document_id)


def _write_pipeline_run_metrics(
    pipeline_run_id: UUID,
    merged: MergedExtraction,
    manifest: BundleManifest,
) -> None:
    """Populate PipelineRun.metrics with the quality-signal blob from §6.6.

    Reads per-pass StageRun outcomes (via v_latest_pass_attempts), computes
    roll-up signals, and writes them to PipelineRun.metrics. Runs AFTER
    _apply_post_merge_yield_updates (so yields reflect post-merge state)
    and BEFORE three-phase import (so diagnostics survive import failure)."""
    pass_outcomes = _build_pass_outcomes_rollup(pipeline_run_id)
    run = session.get(PipelineRun, pipeline_run_id)
    run.metrics = {
        "pass_outcomes": pass_outcomes,
        "document_extraction_anomaly": all(
            outcome.get("yield") in ("EMPTY", "BRIDGES_ONLY")
            for name, outcome in pass_outcomes.items()
            if name in {"radar_domain", "missile_domain", "other_systems"}
               and outcome.get("execution") == "COMPLETE"
        ),
        "pass_degraded_count": sum(
            1 for outcome in pass_outcomes.values()
            if outcome.get("yield") == "DEGRADED"
        ),
        "overall_relationship_rejection_ratio": _compute_rejection_ratio(merged),
        "rejected_relationships_sample": _build_rejection_sample(merged),
        "bundle_legacy": False,  # New runs always have a real bundle_key
        "bundle_key_display": manifest.bundle_key,
    }
    session.commit()
```

**Terminalization rules, summarized (§5.4 + §6.9):**

| Mode | Outcome | `stage_summary` | `PipelineRun.status` | `PipelineRun.finished_at` | `Document.pipeline_status` |
|---|---|---|---|---|---|
| `graph_only` | success | COMPLETE | `COMPLETE` | `now()` | `COMPLETE` |
| `full` | success | COMPLETE | unchanged (`PROCESSING`) | unchanged | unchanged (downstream `finalize_document` terminalizes) |
| either | gate failure | FAILED | `FAILED` | `now()` | `PARTIAL_COMPLETE` |
| either | merge/import failure | FAILED | `FAILED` | `now()` | `PARTIAL_COMPLETE` |
| either | unexpected exception | FAILED | `FAILED` | `now()` | `PARTIAL_COMPLETE` |

`full`-mode success is the only path that leaves `PipelineRun.status = PROCESSING`. All failure paths terminalize the run immediately because downstream stages cannot proceed with a failed graph extraction.

### §5.5 — Single-pass execution (`_run_single_pass`)

```python
def _run_single_pass(
    pipeline_run_id: UUID,
    pass_def: PassManifest,
    manifest: BundleManifest,
    ontology: dict,
    bundle_key: str,
    doc_json: dict,
    pass_results: dict[str, PassResult],
    upstream_refs: dict[str, EntityRef],
    document_id: str,
) -> None:
    attempt = 1
    while True:
        if _should_skip(pass_def, upstream_refs, ontology):
            _write_stage_run(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                attempt=attempt,
                execution_status="SKIPPED",
                yield_status=None,
                skip_reason="NO_UPSTREAM_ENDPOINTS",
                counts=None,
                error=None,
            )
            return

        try:
            request_body = _build_extract_pass_request(
                bundle_key=bundle_key,
                pass_def=pass_def,
                doc_json=doc_json,
                upstream_refs=upstream_refs if pass_def.input_mode == "document_plus_entity_refs" else None,
            )
            response = _call_extract_pass(request_body, timeout=settings.docling_graph_timeout)
            pass_result = _parse_pass_response(response, pass_def, manifest)

        except PassRetryable as exc:
            _write_stage_run(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                attempt=attempt,
                execution_status="FAILED",
                yield_status=None,
                skip_reason=None,
                counts=None,
                error=str(exc),
            )
            if attempt >= settings.pass_max_retries:
                if pass_def.required:
                    raise IngestFailed(f"Required pass {pass_def.name} exhausted retries") from exc
                return
            _backoff(attempt)
            attempt += 1
            continue

        except PassTerminal as exc:
            _write_stage_run(
                pipeline_run_id=pipeline_run_id,
                pass_def=pass_def,
                attempt=attempt,
                execution_status="FAILED",
                yield_status=None,
                skip_reason=None,
                counts=None,
                error=str(exc),
            )
            if pass_def.required:
                raise IngestFailed(f"Required pass {pass_def.name} terminal failure") from exc
            return

        yield_status = classify_yield(pass_result, pass_def, ontology)
        counts = _count_pass_output(pass_result, pass_def, ontology)
        _write_stage_run(
            pipeline_run_id=pipeline_run_id,
            pass_def=pass_def,
            attempt=attempt,
            execution_status="COMPLETE",
            yield_status=yield_status,
            skip_reason=None,
            counts=counts,
            error=None,
        )
        pass_results[pass_def.name] = pass_result

        if _any_downstream_pass_depends_on(manifest, pass_def.name):
            _extend_upstream_refs(upstream_refs, pass_result, pass_def, ontology)
        return
```

**Skip logic** (consuming actual upstream refs, not yield buckets):

```python
def _should_skip(
    pass_def: PassManifest,
    upstream_refs: dict[str, EntityRef],
    ontology: dict,
) -> bool:
    """Return True iff a relationships_only pass has no satisfiable
    (source_type, rel_type, target_type) triple given the refs actually
    available from its declared dependencies."""
    if pass_def.kind != "relationships_only":
        return False
    if not pass_def.skip_if_no_upstream_endpoints:
        return False

    declared_deps = set(pass_def.depends_on)
    available_types: set[str] = {
        ref.entity_type
        for ref in upstream_refs.values()
        if ref.pass_origin in declared_deps
    }
    if not available_types:
        return True

    allowed_rels = set(pass_def.extracted_relationship_types)
    for row in ontology.get("validation_matrix", []):
        if row.get("relationship") not in allowed_rels:
            continue
        if (row.get("source") in available_types
                and row.get("target") in available_types):
            return False

    return True
```

**Retry policy:**
- Retryable: transport error, timeout, HTTP 5xx, partial response parse, `TransientOllamaBusyError`
- Terminal: HTTP 4xx, Pydantic validation after salvage, `UnknownBundleOrPassError`, `ManifestValidationError`, worker code bugs
- `pass_max_retries` default 3. Backoff: `30s × 2^(attempt-1)`, capped at 300s.
- Each attempt writes a new StageRun row with `attempt+1`. Previous FAILED row is not mutated.

### §5.6 — Three-phase graph import

**Phase 2 — node upsert:**

```python
def _import_graph_phase_nodes(
    merged: MergedExtraction,
    ontology: dict,
    document_id: str,
    tracker: GraphWriteTracker,
) -> dict[LogicalIdentity, str]:
    # Note: ProvenanceMetadata gains a new optional field `pipeline_run_id`
    # in PR 1 as part of this refactor — see §5.6 note below. It is
    # backward-compatible (Optional with default None).
    provenance = ProvenanceMetadata(
        document_id=document_id,
        pipeline_run_id=merged.pipeline_run_id,
    )
    # Build NodeRecord list BEFORE touching the graph store. If any of
    # these calls raises (e.g., identity-serialization bug, display-label
    # bug), the tracker stays False and rollback is skipped.
    node_records = [
        NodeRecord(
            entity_type=e.identity.entity_type,
            identity_fields=e.identity.as_upsert_identity_dict(),
            name=build_display_label(
                e.identity.entity_type,
                e.identity.identity_values_dict(),
                e.properties,
            ),
            properties=e.properties,
            extraction_confidence=e.confidence,
        )
        for e in merged.entities
    ]
    # Flip the tracker RIGHT BEFORE the first graph_store mutation.
    # Any failure after this point triggers rollback in the caller's
    # except branch.
    tracker.mark()
    # upsert_nodes_batch_sync:
    # (1) upserts each NodeRecord and returns list[str] of RIDs in input order
    # (2) auto-creates HAS_PROVENANCE edges from each upserted node to the
    #     structural Document vertex via _create_provenance_edges_batch_sync,
    #     because we passed a non-None ProvenanceMetadata with document_id.
    #     This is the canonical source of HAS_PROVENANCE edges; derive_rules
    #     does NOT also create them.
    node_rids: list[str] = graph_store.upsert_nodes_batch_sync(node_records, provenance)
    identity_to_rid: dict[LogicalIdentity, str] = dict(zip(
        (e.identity for e in merged.entities),
        node_rids,
        strict=True,
    ))
    return identity_to_rid
```

`upsert_nodes_batch_sync` signature unchanged (still returns `list[str]`). `strict=True` on `zip` guards against length mismatch. The `tracker.mark()` call is intentionally positioned after all pure Python work and immediately before the first network call.

**ProvenanceMetadata extension (PR 1, additive):** the existing `ProvenanceMetadata` dataclass in `app/services/graph_store.py` (lines 19–26) gains one new optional field without touching the existing ones:

```python
# Additive change to app/services/graph_store.py:ProvenanceMetadata —
# existing fields (document_id, page_numbers with default_factory=list,
# upload_datetime, document_datetime) keep their exact current types and
# defaults. Only this new field is added:
pipeline_run_id: str | None = None     # NEW in PR 1 — optional, additive
```

This lets downstream code (including `_create_provenance_edges_batch_sync` and future audit queries) correlate provenance writes back to the run that produced them without changing any existing callers or existing field types. Populated by the worker in phase 2; used as metadata on the auto-created HAS_PROVENANCE edges.

**Note on `page_numbers` default:** the current `ProvenanceMetadata.page_numbers` uses `field(default_factory=list)` (not `None`). This stays unchanged. `_create_provenance_edges_batch_sync` relies on `page_numbers` being a list it can iterate, and changing it to `None`-able would break existing callers. The additive change is strictly scoped to `pipeline_run_id`.

**Phase 3 — domain relationship upsert (identity-based):**

```python
def _import_graph_phase_domain_edges(
    merged: MergedExtraction,
    ontology: dict,
    tracker: GraphWriteTracker,
) -> None:
    provenance = ProvenanceMetadata(
        document_id=merged.document_id,
        pipeline_run_id=merged.pipeline_run_id,
    )
    rel_records = [
        RelationshipRecord(
            from_type=e.from_identity.entity_type,
            from_identity=e.from_identity.as_upsert_identity_dict(),
            to_type=e.to_identity.entity_type,
            to_identity=e.to_identity.as_upsert_identity_dict(),
            rel_type=e.rel_type,
            extraction_confidence=e.confidence,
        )
        for e in merged.edges
    ]
    # Phase 2 already marked the tracker; this call is a safety net for
    # any future reordering. Idempotent.
    tracker.mark()
    graph_store.upsert_relationships_batch_sync(rel_records, provenance)
```

**Phase 4 — derived structural edges (RID-based, structural-edge path):**

```python
def _import_graph_phase_structural_edges(
    merged: MergedExtraction,
    identity_to_rid: dict[LogicalIdentity, str],
    document_id: str,
    pipeline_run_id: str,
    tracker: GraphWriteTracker,
) -> None:
    # Build phase — pure Python, no writes. If this raises, tracker state
    # is unchanged (already True from phase 2, or False if phase 2 was
    # somehow skipped in a future refactor).
    chunks = _load_chunks_for_derivation(document_id)  # returns list[ChunkForDerivation]
    document_rid = _get_structural_document_rid(document_id)

    derived = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid=identity_to_rid,
        chunks=chunks,
        document_rid=document_rid,
    )
    # No edges to write → no mutation attempted here, but phase 2 already
    # wrote nodes, so tracker is already True. If derived is empty, we
    # still skip the loop without touching the tracker.
    for edge in derived:
        # First-mutation mark is idempotent; phase 2 already called it.
        tracker.mark()
        # create_structural_edge_sync uses from_id/to_id (not from_rid/to_rid) —
        # matches the existing signature at app/services/graph_store.py:528.
        graph_store.create_structural_edge_sync(
            from_id=edge.from_id,
            to_id=edge.to_id,
            rel_type=edge.rel_type,
            properties={
                "document_id": document_id,
                "pipeline_run_id": pipeline_run_id,
                "extraction_confidence": edge.confidence,
                "source": "derive_rules",
            },
        )
```

Structural edges go through `graph_store.create_structural_edge_sync` (RID-based), NOT `upsert_relationships_batch_sync` (identity-based). The distinction matches the existing graph store's split between domain-edge and structural-edge paths.

**HAS_PROVENANCE note:** `HAS_PROVENANCE` edges are NOT created in phase 4. They are auto-created during phase 2's `upsert_nodes_batch_sync` call because we pass a `ProvenanceMetadata` object. Phase 4 only produces `MENTIONED_IN` (and any future deterministic rules that are not already handled elsewhere).

**Helper signatures** (defined in `app/workers/pipeline.py` as private helpers; not part of the public API):

```python
def _load_chunks_for_derivation(document_id: str) -> list[ChunkForDerivation]:
    """Load TextChunk rows for this document and convert them into the
    lightweight ChunkForDerivation DTO used by derive_rules."""

def _get_structural_document_rid(document_id: str) -> str:
    """Look up the ArcadeDB @rid of the structural Document vertex for
    this document_id. The vertex is guaranteed to exist at this point
    because earlier pipeline stages created it."""

def _build_docling_document_json(document_id: str) -> dict:
    """Construct the enriched DoclingDocument JSON payload passed to
    /extract-pass. Source: the persisted docling_document.json in MinIO
    plus enrichment overlays. This helper already exists in the codebase
    in some form; this spec does not change it."""
```

### §5.7 — DocumentGraphExtraction snapshot write

```python
def _upsert_document_graph_extraction(document_id, pipeline_run_id, run, merged):
    existing = (
        session.query(DocumentGraphExtraction)
        .filter_by(document_id=document_id)
        .first()
    )
    values = dict(
        pipeline_run_id=pipeline_run_id,
        ontology_bundle_key=run.ontology_bundle_key,
        ontology_name=run.ontology_name,
        ontology_version=run.ontology_version,
        use_case_key=run.use_case_key,
        extraction_profile_version=run.extraction_profile_version,
        graph_json=serialize_for_audit(merged),    # audit blob, not serialized graph
        updated_at=now(),
    )
    if existing:
        for k, v in values.items():
            setattr(existing, k, v)
    else:
        session.add(DocumentGraphExtraction(document_id=document_id, **values))
    session.commit()


def serialize_for_audit(merged: MergedExtraction) -> dict:
    """Produce the audit blob stored in DocumentGraphExtraction.graph_json.

    Shape (all counts; no full entity/edge lists):
    {
        "entity_count_by_type": {"RADAR_SYSTEM": 3, "PLATFORM": 2, ...},
        "edge_count_by_type":   {"HAS_ANTENNA": 5, "INSTALLED_ON": 4, ...},
        "primary_entities_total": 12,
        "bridge_entities_total":  3,
        "edges_accepted": 18,
        "edges_rejected": 4,
        "rejection_reasons": {"to_endpoint_not_found": 3, "invalid_triple": 1},
        "pass_summaries": [
            {"pass_name": "reference",    "yield_status": "HIT", ...},
            {"pass_name": "radar_domain", "yield_status": "HIT", ...},
            ...
        ],
    }

    The previous column contents (serialized NetworkX node-link graph) are
    NOT compatible with the new shape. PR 3 updates the column docstring to
    reflect this. No backward-compatibility shim is provided — the column
    is not read by any live retrieval code; only admin/audit queries touch
    it, and those are updated in PR 2 alongside the writer."""
```

One row per document. Subsequent ingests overwrite in place. Previous runs remain in `PipelineRun` + `StageRun` for audit.

---

## §6 — Failure semantics and pass outcomes

### §6.1 — Execution status

| State | Meaning | Triggers |
|---|---|---|
| `COMPLETE` | Pass dispatched, response validated (possibly after salvage), counts computed. | Successful pass invocation. |
| `FAILED` | Pass dispatched but did not reach a validated response. | Transport error, timeout, HTTP 4xx/5xx, parse failure, Pydantic validation failure after salvage, worker exception. |
| `SKIPPED` | Pass never dispatched; no HTTP call was made. | `kind=relationships_only` + `skip_if_no_upstream_endpoints` + no satisfiable `(source_type, rel_type, target_type)` triple given actual `upstream_refs`. |

Exactly one status is written per StageRun row. Retries create new rows; previous terminal statuses are not mutated.

### §6.2 — Yield status

Meaningful only when `execution_status == COMPLETE`. Two related functions live in `app/services/extraction_merge.py`:

```python
def classify_yield_from_counts(
    *,
    primary: int,
    bridge: int,
    extracted_rels: int,
    rejected_rels: int,
) -> YieldStatus:
    """Precedence (top wins). Pure function of counts — has no knowledge
    of the pass definition or ontology. Used by both pre-merge classification
    and post-merge reclassification."""
    total_rels = extracted_rels + rejected_rels
    if total_rels >= 4 and rejected_rels / total_rels >= 0.75:
        return YieldStatus.DEGRADED
    if primary == 0 and bridge == 0 and extracted_rels == 0:
        return YieldStatus.EMPTY
    if primary == 0 and bridge > 0:
        return YieldStatus.BRIDGES_ONLY
    return YieldStatus.HIT


def classify_yield(
    result: PassResult, pass_def: PassManifest, ontology: dict,
) -> YieldStatus:
    """Convenience wrapper used inside _run_single_pass. Extracts counts
    from the PassResult and delegates to classify_yield_from_counts."""
    primary = count_primary_entities(result, pass_def)
    bridge  = count_bridge_entities(result, pass_def)
    extracted_rels = len(result.relationships)
    rejected_pre_merge = len(result.pre_merge_rejections)
    return classify_yield_from_counts(
        primary=primary,
        bridge=bridge,
        extracted_rels=extracted_rels,
        rejected_rels=rejected_pre_merge,
    )
```

`classify_yield` is called by `_run_single_pass` (§5.5) with fresh pass output. `classify_yield_from_counts` is called by `_apply_post_merge_yield_updates` (§5.4) after merge updates `relationships_rejected` on the StageRun row, to recompute yield from the updated totals. Both live in `extraction_merge.py`. `_run_single_pass`'s internal reference is the same function — the `_classify_yield` private alias is dropped in favor of the public `classify_yield` name.

Post-merge yield reclassification may move a `HIT` pass to `DEGRADED` when post-merge rejections push the ratio over threshold. It cannot move the pass in the other direction.

### §6.3 — Skip reasons

| `skip_reason` | Condition |
|---|---|
| `NO_UPSTREAM_ENDPOINTS` | `pass.kind == "relationships_only"` AND `pass.skip_if_no_upstream_endpoints: true` AND no satisfiable `(source_type, rel_type, target_type)` triple exists given the actual refs from `depends_on` passes. Does NOT depend on upstream yield buckets. |

Only one skip reason in v1. Adding new reasons requires explicit design review and spec update.

### §6.4 — Required-pass gate

Run once per ingest, after the pass loop completes and BEFORE merge/import:

```python
def check_required_pass_gate(pipeline_run_id: UUID) -> GateResult:
    run = session.get(PipelineRun, pipeline_run_id)
    manifest = load_bundle_manifest(run.ontology_bundle_key)
    required_passes = [p.name for p in manifest.passes if p.required]
    failures: list[tuple[str, str]] = []

    for pass_name in required_passes:
        latest = (
            session.query(StageRun)
            .filter(
                StageRun.pipeline_run_id == pipeline_run_id,
                StageRun.stage_name == "derive_ontology_graph",
                StageRun.pass_name == pass_name,
            )
            .order_by(StageRun.attempt.desc())
            .first()
        )
        if latest is None:
            raise WorkerInvariantError(f"Required pass {pass_name} has no StageRun")

        if latest.execution_status == "COMPLETE":
            continue  # HIT, EMPTY, BRIDGES_ONLY, DEGRADED all pass the gate
        if latest.execution_status == "FAILED":
            failures.append((pass_name, f"FAILED: {latest.error_message}"))
            continue
        if latest.execution_status == "SKIPPED":
            allowed = {"NO_UPSTREAM_ENDPOINTS"}
            if latest.skip_reason in allowed:
                continue
            failures.append((pass_name, f"unauthorized skip: {latest.skip_reason}"))
            continue

    return GateResult(passed=(not failures), failures=failures)
```

**Gate rules:**
- **Pass:** `COMPLETE` (any yield status) OR `SKIPPED` with authorized skip_reason
- **Fail:** `FAILED` OR `SKIPPED` with unauthorized skip_reason

A failed gate causes (in both `full` and `graph_only` modes):
1. `PipelineRun.status = "FAILED"` and `PipelineRun.finished_at = now()`
2. `Document.pipeline_status = "PARTIAL_COMPLETE"` (using the existing `Document.pipeline_status` vocabulary from `app/models/ingest.py:60`)
3. `IngestFailed` exception propagates
4. `DocumentGraphExtraction` NOT updated by `derive_ontology_graph` — but see the caveat below
5. No rollback needed within `derive_ontology_graph` (no graph writes occurred yet in this stage)

**Caveat — prior snapshot preservation is phase-dependent for `full` mode (§4.3):**
- **`graph_only`:** no earlier stage ran `purge_document_derivations`. The prior snapshot survives the gate failure and remains queryable. Status API returns `graph_snapshot != null`, `graph_queryable=true`.
- **`full`, gate failure:** by the time `derive_ontology_graph` runs and its gate fires, `purge_document_derivations` has already executed as an earlier pipeline stage and deleted the `DocumentGraphExtraction` row + ArcadeDB graph. The gate failure leaves the document with NO snapshot and NO queryable graph. Status API returns `graph_snapshot=null`, `graph_queryable=false`. This is not a behavior of `derive_ontology_graph` — it's a consequence of the existing purge semantics documented in §4.3.
- **`full`, failure in pre-purge stages (handled by those stages, not by `derive_ontology_graph`):** if a `full` ingest fails in `prepare_document`, `detect_and_translate`, or `derive_document_metadata`, purge hasn't run yet and the prior snapshot is still intact. `derive_ontology_graph` doesn't see those failures — they're handled by the earlier stages themselves — but the status API will report `graph_snapshot != null`, `graph_queryable=true` because the prior row is still present.

Gate failure in `derive_ontology_graph` terminalizes the run in both modes because downstream stages cannot proceed with a broken extraction. The mode-conditional terminalization on the SUCCESS path (§5.4) is a separate concern.

### §6.5 — Retry policy

Per pass, per invocation:

| Condition | Action |
|---|---|
| `COMPLETE` | Terminal success. No further attempts. |
| `FAILED` + retryable + `attempt < pass_max_retries` | New StageRun row with `attempt+1` after exponential backoff. |
| `FAILED` + retryable + `attempt == pass_max_retries` | Terminal. If required → ingest fails. |
| `FAILED` + terminal exception | Terminal immediately. If required → ingest fails. |
| `SKIPPED` | Terminal. No retries possible. |

`pass_max_retries` default: 3. Backoff: `30s × 2^(attempt-1)`, capped at 300s. Worker manages retries internally — Celery does NOT retry the `derive_ontology_graph` task on per-pass failures.

### §6.6 — Quality signals (metrics, not gates)

Written to `PipelineRun.metrics` after merge completes:

- `document_extraction_anomaly: bool` — true iff all core entity passes (`radar_domain`, `missile_domain`, `other_systems`) ended `EMPTY` or `BRIDGES_ONLY`.
- `pass_degraded_count: int` — count of passes with `yield_status == DEGRADED`.
- `overall_relationship_rejection_ratio: float` — `total_rejected / (total_extracted + total_rejected)`.
- `rejected_relationships_sample: dict` — up to 20 rejections per pass per reason for diagnostics.
- `bundle_legacy: bool` — true iff `ontology_bundle_key IS NULL`.

Observable, not enforceable. Consumed by the baseline harness (§8.3) and future dashboards.

### §6.7 — Relationship rejection taxonomy

| Reason | Trigger | Phase |
|---|---|---|
| `MISSING_REL_TYPE` | `rel_type` is None or not in pass's allowed enum | Pre-merge |
| `INVALID_IDENTITY_PAYLOAD` | `from_identity` / `to_identity` missing required identity_fields keys, empty, or type-incompatible | Pre-merge |
| `UNKNOWN_REF_ID` | `from_ref_id` / `to_ref_id` doesn't match any `upstream_entities` entry (system_links only) | Pre-merge |
| `FROM_ENDPOINT_NOT_FOUND` | Resolved from-side `LogicalIdentity` doesn't match any merged entity | Post-merge |
| `TO_ENDPOINT_NOT_FOUND` | Same for target | Post-merge |
| `INVALID_TRIPLE` | `(from_type, rel_type, to_type)` not in `validation_matrix` | Post-merge |

**Ambiguous endpoint match is not a reachable case** — `LogicalIdentity`-keyed merge collapses identical-identity entries by construction.

Every rejection is logged with context. Sample persisted in `PipelineRun.metrics["rejected_relationships_sample"]`.

### §6.8 — Rollback scope (v1)

On merge or import failure (and only when `tracker.any_mutation_attempted` is True — see the `GraphWriteTracker` gate in §5.4), rollback calls the abstract helper `_delete_extraction_layer_graph(document_id)`. PR 1 wires this helper to either the existing `graph_store.delete_document_graph_sync(document_id)` or a narrower sibling method — whichever satisfies the contract below (see residual check #1).

**Contract — what `_delete_extraction_layer_graph(document_id)` MUST delete:**
- Document-scoped extracted entity vertices (their identity includes `document_id`, so every such vertex represents extraction-layer work for this document)
- Domain edges written by the extraction stage, identifiable via their `document_id` provenance property
- Structural edges written by `derive_rules` in phase 4, identifiable via their `document_id` / `source=derive_rules` properties

**Contract — what it MUST NOT delete:**
- Global-scoped entity vertices (PLATFORM, RADAR_SYSTEM, etc.). Their properties may have been merged during the failed run; those merges are not reverted in v1.
- Chunks (`TextChunk`, `ImageChunk`) — these are owned by upstream pipeline stages (chunking/embedding), not the extraction stage.
- The structural `Document` vertex — owned by earlier stages, referenced by extraction but not created by it.
- The document's persisted `docling_document.json` in MinIO, or any derivation state outside the graph.

**Accepted limitations (v1 — explicit non-goals in §8.9):**

1. **Global entity property restoration:** global entity properties enriched during a failed run persist until the next successful ingest that references them overwrites them. The spec does not provide atomic property restoration.

2. **Prior successful graph is NOT preserved across a failed `graph_only` reingest that reaches mutation time.** If a prior `derive_ontology_graph` successfully produced graph G-prior for this document, and a later `graph_only` reingest fails AFTER `tracker.any_mutation_attempted` has flipped True, the rollback deletes **all** document-scoped extraction state — including G-prior's document-scoped entities. The `DocumentGraphExtraction` row for G-prior is **not re-written** by the failed new run (it remains the prior row), but its `graph_json` audit blob describes a graph that is no longer fully queryable in ArcadeDB. Document-scoped entities from G-prior are gone; global entities touched by G-prior may remain (property-merge semantics).

   This is a deliberate v1 tradeoff: preserving G-prior across a failed reingest would require a shadow-write strategy (write new state to a transient namespace, atomic swap on success). That is a larger graph_store refactor and is explicitly out of scope.

3. **Prior successful graph is invalidated by `purge_document_derivations` during a `full` reingest — but only after the purge stage actually executes.** The `full` pipeline runs several stages before purge (`prepare_document`, `detect_and_translate`, `derive_document_metadata`); failures in those stages leave the prior `DocumentGraphExtraction` row and its graph intact. Once purge runs, the prior state is gone and cannot be restored by `derive_ontology_graph`'s rollback. See §4.3 for the three-phase breakdown of a full reingest with respect to snapshot preservation.

   Operationally: a `full` reingest that fails in the **pre-purge** stages is equivalent to a `graph_only` reingest gate failure in terms of preservation — the prior graph is still queryable. A `full` reingest that fails **post-purge** (including at `derive_ontology_graph`'s gate, inside its phase helpers, or in any intermediate stage between purge and `derive_ontology_graph`) loses the prior graph. Users who need to preserve a working graph across a reingest should prefer `graph_only` when possible.

4. **Status API consequences (phase-dependent, not strictly mode-dependent).** The status API reflects current PostgreSQL state via the authoritative rule in §7.10. Specific signals depend on which phase of the run has been reached:

   - **Failed `graph_only` reingest with preserved prior snapshot** (gate failure or pre-mutation failure — `rollback_executed=False`): `document_status="PARTIAL_COMPLETE"`, `graph_snapshot != null` (prior run's row), `graph_queryable=true`. Clients can still query the prior extraction.
   - **Failed `graph_only` reingest with rollback fired** (`rollback_executed=True`): `document_status="PARTIAL_COMPLETE"`, `graph_snapshot != null` (prior run's row, historical audit), `graph_queryable=false`. Clients must handle "row exists but vertices are gone" gracefully.
   - **Failed `full` reingest, pre-purge phase** (failure in `prepare_document` / `detect_and_translate` / `derive_document_metadata`): `document_status="PARTIAL_COMPLETE"`, `graph_snapshot != null` (prior run's row — purge never executed), `graph_queryable=true`. Equivalent to a graph_only gate failure from a client's perspective.
   - **Failed `full` reingest, post-purge phase** (failure in any stage from `derive_picture_descriptions` through `derive_ontology_graph`): `document_status="PARTIAL_COMPLETE"`, `graph_snapshot=null` (purged and not yet replaced), `graph_queryable=false`. Clients see no snapshot at all.

   `graph_queryable` is the single authoritative signal for "can I run queries against this document's extraction graph right now?" — computed via §7.10's rule, not by inspecting `rollback_executed` alone or by inferring phase from mode.

5. **Write-log / compensating RID-level rollback** is not implemented. v1 uses the abstract helper only.

6. **Purge-stage redesign** is explicitly a non-goal. Making `purge_document_derivations` non-destructive (shadow/staging/swap) is a separate spec.

### §6.9 — State transitions

**Per-pass StageRun:**

```
        [insert]
           │
           ▼
        RUNNING ──────────────────────────────┐
           │                                  │
           │ (success)         (exception in handler)
           ▼                                  ▼
        COMPLETE                           FAILED
         ├─ execution_status=COMPLETE       ├─ execution_status=FAILED
         ├─ yield_status ∈ {HIT, EMPTY,     ├─ yield_status=NULL
         │  BRIDGES_ONLY, DEGRADED}         └─ error_message set
         └─ counts populated                             │
                  │                                      │
                  │                           (retryable && attempts left)
                  │                                      │
                  │                                      ▼
                  │                               [new row, attempt+1]
                  │
        OR (skipped without dispatch)
                  │
                  ▼
        COMPLETE-as-SKIPPED
         ├─ execution_status=SKIPPED
         ├─ yield_status=NULL
         └─ skip_reason=NO_UPSTREAM_ENDPOINTS
```

**Per-ingest PipelineRun (mode-conditional on success):**

```
     PROCESSING                           (default value per app/models/ingest.py:219)
         │
         ├─ derive_ontology_graph succeeds, mode=graph_only → COMPLETE (terminal)
         ├─ derive_ontology_graph succeeds, mode=full       → stays PROCESSING
         │                                                    until finalize_document
         │                                                    terminalizes
         └─ any failure in derive_ontology_graph             → FAILED (terminal)
            (gate, merge/import, unexpected, either mode)
```

**Document.pipeline_status transitions — from `derive_ontology_graph` only:**

Uses the EXISTING vocabulary in `app/models/ingest.py:60`: `PENDING | PROCESSING | COMPLETE | PARTIAL_COMPLETE | FAILED | PENDING_HUMAN_REVIEW`. This spec does not introduce new values.

| Mode | Trigger | New value |
|---|---|---|
| `graph_only` | success | `COMPLETE` |
| `full` | success | **unchanged** (downstream `finalize_document` terminalizes) |
| either | gate failure | `PARTIAL_COMPLETE` |
| either | merge/import failure | `PARTIAL_COMPLETE` |
| either | unexpected exception | `PARTIAL_COMPLETE` |

Regression from `COMPLETE` → `PARTIAL_COMPLETE` on failed `graph_only` reingest is the locked behavior. `Document.pipeline_status` reflects the latest ingest attempt's health; `DocumentGraphExtraction` reflects the latest successful snapshot; both are surfaced separately in the status API (§7.11).

---

## §7 — Migration and rollout plan

### §7.1 — Stabilization phase (`feature/arcadedb`)

Before any refactor work, commit the current uncommitted changes on `feature/arcadedb` in two commits:

**Commit A — durable ArcadeDB fixes:**
- `app/services/arcadedb_client.py` — event-loop-aware httpx client
- `app/services/arcadedb_graph.py` — Cypher label predicates (no `@class`)

**Commit B — legacy extraction-path stabilizers:**
- `app/workers/pipeline.py` — None-safe confidence defaulting
- `docker/docling-graph/app/template_builder.py` — fallback id_field Optional

`feature/arcadedb` stays a shippable checkpoint.

### §7.2 — Branch creation

```bash
git checkout feature/arcadedb
git checkout -b feature/extraction-refactor
```

Three PRs on `feature/extraction-refactor`.

### §7.3 — PR 1: Baseline + Scaffolding (strictly additive)

**Invariant:** at the end of PR 1, the legacy extraction path still runs production documents end-to-end. The new path exists, loads cleanly, passes unit tests, and is testable directly — but no caller has switched to it.

**Honest failures on the legacy path:**
- `app/config.py`: default `graph_layered_shadow_mode=False`, default `graph_layered_fail_open_to_single_pass=False`
- `app/services/layered_extraction.py`: remove per-pass try/except that swallows errors
- `app/workers/pipeline.py`: persist `attempted_mode`/`completed_mode` on legacy stage metrics

**Symlink compatibility layer:**

1. Move `ontology/ontology.yaml` to `ontology_bundles/air_defense_v3/ontology.yaml` via `git mv`.
2. Apply `identity_fields` and `identity_scope` edits to the moved file.
3. Create a git-tracked symlink at `ontology/ontology.yaml` → `../ontology_bundles/air_defense_v3/ontology.yaml`.
4. Legacy `docker-compose.yml` bind mount of `./ontology` continues to work: container sees the symlink, resolves it to `/app/ontology_bundles/air_defense_v3/ontology.yaml` (which exists via the new Dockerfile COPY).
5. CI smoke test verifies the symlink resolves inside the container.
6. PR 3 removes the symlink and the `./ontology` bind mount.

**Bundle skeleton and contents:**
- `ontology_bundles/__init__.py`, `ontology_bundles/air_defense_v3/__init__.py`
- `manifest.yaml`, `coverage.yaml`
- `validators.py`, `derive_rules.py`
- Five extraction schema modules

**`load_ontology()` refactor:**
- Drop `prefer_active` parameter
- Add `bundle_key` with three-tier resolution
- Split `load_registry_ontology()` into a separate function
- **Caller audit (pre-migration task in PR 1).** The current default behavior is "active registry ontology first, else repository YAML" (see `app/services/ontology_templates.py:111, 155`). The new default is "system default bundle." This is a silent behavior change for every no-arg caller. Before landing the refactor, PR 1 performs a grep audit:

  ```bash
  grep -rn "load_ontology\|get_ontology_cache_signature\|prefer_active" app/ tests/
  ```

  Every hit is classified into one of three buckets:
  1. **Extraction hot path** — migrates to `load_ontology(bundle_key=...)` driven by `PipelineRun.ontology_bundle_key`.
  2. **Live non-extraction path that still needs registry behavior** (retrieval, query profiles, admin views that pin to an historical version) — migrates to `load_registry_ontology(version_id=...)` with an explicit version id.
  3. **Safe no-arg → system default bundle** — no change required beyond dropping `prefer_active=True` if present.

  Every caller in bucket 2 gets a regression test added in the same PR that asserts "this code path still loads the version-pinned ontology it used to load." Callers classified into bucket 3 get a one-line justification comment so a future reader can see why they were considered safe.

  The audit output is committed to PR 1 as a migration note under `docs/superpowers/plans/` (alongside the implementation plan) so reviewers can verify the classification is complete.

**Bundle loader API:**
- `app/services/ontology_bundles.py` (worker side)
- `docker/docling-graph/app/bundles.py` (service side)

**Alembic migration:** all columns and indexes per §4.8. Explicit `op.drop_constraint('<existing_uq_name>', ...)` — PR 1 verifies the exact constraint name via `pg_constraint` lookup before coding the drop (residual check).

**New executor path in docling-graph service:**
- Add `POST /extract-pass` endpoint alongside existing `POST /extract-all`
- Service startup pre-loads all bundle manifests and extraction schema modules
- Input-mode validation; unknown bundle/pass → HTTP 404
- Monkey-patches unchanged
- **New file `docker/docling-graph/app/config.py`** with `ServiceSettings` (pydantic `BaseSettings`) exposing `structured_output_threshold_chars` (default 8000) as an env var. The monkey-patch in `main.py` reads from `settings.structured_output_threshold_chars` instead of the current hardcoded `8000`. The coverage checker reads the same config value so CI and runtime agree by construction.

**Coverage checker:** `tools/check_extraction_coverage.py` implementing all 13 active rules (numbered 1–14 with rule 7 as a deleted placeholder). Runs in CI.

**Docker packaging:**
- `docker/docling-graph/Dockerfile`: build context change, `COPY ontology_bundles`
- `docker/worker/Dockerfile`: `COPY ontology_bundles`
- CI smoke import test inside both images

**PR 1 exit criteria:**
- [ ] Full test suite passes
- [ ] Coverage checker passes
- [ ] Both images build; smoke import test passes
- [ ] Migration applies cleanly on fresh and existing DB
- [ ] Migration downgrade works
- [ ] Old `/extract-all` endpoint still responds
- [ ] New `/extract-pass` endpoint responds for all 5 passes
- [ ] End-to-end ingest via legacy path still produces a graph

### §7.4 — PR 2: Switchover (behind feature flag)

**Invariant:** at merge time, a feature flag selects between legacy and bundle-passes paths per-invocation. Flag defaults to `legacy`. Flipped to `bundle_passes` after baseline comparison passes.

**Feature flag:**
- `app/config.py`: `graph_extraction_engine: Literal["legacy", "bundle_passes"] = "legacy"`
- Read from env var at settings-load time. Cached via existing `get_settings()`.
- **Flipping the flag requires worker and beat restart.** No uncached per-task reader.

**New orchestrator code:**
- `app/services/extraction_merge.py` (new file): `LogicalIdentity`, `PassResult`, `MergedEntityRecord`, `MergedEdgeRecord`, `MergedExtraction`, `ChunkForDerivation`, `DerivedEdge`, `merge_and_resolve`, `build_display_label`, `classify_yield`, `classify_yield_from_counts`, `RelationshipRejectionReason`
- `app/workers/pipeline.py` (new private helpers): `GraphWriteTracker` dataclass, `_attempt_rollback`, `_delete_extraction_layer_graph`, `_write_pipeline_run_metrics`, `_run_single_pass`, `_should_skip`, `_apply_post_merge_yield_updates`, `_import_graph_phase_nodes`, `_import_graph_phase_domain_edges`, `_import_graph_phase_structural_edges`, `_update_document_pipeline_status`, `check_required_pass_gate`, `IngestDispatchResult`. `GraphWriteTracker` is worker-local orchestration state, not merge logic — it lives in the worker module alongside the phase helpers that mutate it.
- `app/workers/pipeline.py`: rewrite `derive_ontology_graph` for the new branch (legacy branch unchanged); new private functions for pass loop, gate, import phases, rollback, document status updates
- `IngestDispatchResult` dataclass and caller updates

**Bundle threading:**
- `Source` create/response schemas gain bundle defaults
- `ReingestRequest` gains optional bundle override
- Upload, reingest, and `graph_only` reingest routes all resolve and snapshot bundle

**Status/read API (three-concept split):**
- Response exposes `document_status`, `latest_run` (with `passes[]` and `stage_summary.attempt`), `graph_snapshot` (with `is_stale`)
- `ontology_bundle_key` and `ontology_bundle_label` emitted as separate fields

**Soak procedure (pre-merge gate for PR 3):**
1. Deploy PR 2 with flag = `legacy`.
2. Verify legacy path still works.
3. Flip flag to `bundle_passes` in dev, rerun baseline corpus, compare to baseline.
4. If dev matches criteria, flip in staging.
5. If staging matches, flip in prod.
6. **At least 7 days** of regular ingests on `bundle_passes` in prod without incident.
7. Baseline rerun at least twice during soak with consistent results.
8. Metric alerts (§7.9) did not fire.

**PR 2 exit criteria:**
- [ ] Full test suite passes including new unit tests
- [ ] End-to-end integration test with flag=`bundle_passes` succeeds
- [ ] Bundle resolution precedence test passes
- [ ] `graph_only` reingest honors inheritance precedence
- [ ] Baseline comparison meets criteria in dev/staging when flag flipped

### §7.5 — PR 3: Deletion and Hardening

**Pre-merge gates** (must all be true):
- PR 2 deployed to production ≥ 7 days
- `graph_extraction_engine = bundle_passes` in prod continuously during that window
- No production incidents attributed to new path
- Baseline rerun at least twice with consistent results
- Metric alerts did not fire
- At least one reingest of a previously-ingested document on the new path with operator-expected result

**Deletions:**
- `docker/docling-graph/app/template_builder.py`
- `docker/docling-graph/tests/test_template_builder.py`
- `app/services/layered_extraction.py`
- `tests/unit/test_layered_extraction.py`
- `app/services/ontology_layers.py`
- `tests/unit/test_ontology_layers.py`
- `ontology/layer_map.yaml`
- `ontology/ontology.yaml` symlink
- `./ontology` bind mount from `docker-compose.yml` (docling-graph service)
- Old `POST /extract-all` endpoint
- `ExtractAllRequest` and `ontology_definition` field from service schemas
- `ontology_definition` parameter from `docling_graph_service.extract_graph_all()` (or the function itself if no longer called)
- Legacy branch of `derive_ontology_graph`
- `settings.graph_extraction_engine` flag
- `settings.graph_layered_*` config keys

**Renames:**
- `HealthResponse.template_count` → `HealthResponse.schema_count`

**New contract tests:**
- `docker/docling-graph/tests/test_monkey_patches.py` — three tests, one per patch
- Detect upstream LiteLLM or docling-graph drift

**CI additions:**
- Import scope lint: `grep -r "from ontology_bundles\.[^.]*\.extraction_schemas" app/` returns zero
- Unsafe confidence-defaulting lint
- `prefer_active` resurrection lint
- `ontology_definition` absence lint
- `/app/ontology/` path absence lint
- `graph_extraction_engine` absence lint
- `graph_layered_` absence lint

**Column docstring update:**
- `DocumentGraphExtraction.graph_json` docstring reflects "audit blob, not graph serialization" semantics

**PR 3 exit criteria (post-merge verification):**
- [ ] All deletions complete; grep shows zero references to deleted modules
- [ ] All lint checks return zero
- [ ] Monkey-patch contract tests pass
- [ ] Full end-to-end integration test with only the new path succeeds
- [ ] Post-merge health check in first 24 hours shows no extraction regressions

### §7.6 — Feature flag lifecycle

| Stage | `graph_extraction_engine` | Restart required? |
|---|---|---|
| PR 1 merged | does not exist | no |
| PR 2 merged | `legacy` (default) | yes (new setting introduced) |
| PR 2 flipped in dev/staging | `bundle_passes` | yes |
| PR 2 flipped in prod | `bundle_passes` | yes |
| PR 3 merged | removed | yes |

Every stage transition involves a worker + beat restart.

### §7.7 — Rollback plan

**After PR 1:** revert the PR. Additive-only; no data issues.

**After PR 2, before PR 3:** flip `graph_extraction_engine=legacy` via env var and restart workers. New runs use legacy path immediately. No code revert needed.

**After PR 3:** legacy code is gone. Git revert PR 3, redeploy with flag, investigate, re-land after fix. This is why PR 3 waits for a soak period — rollback between PR 2 and PR 3 is a one-flag flip; after PR 3 it's a revert-and-redeploy.

### §7.8 — Operational safeguards during soak

**Metric monitoring (alerts, not auto-flips):**
- `silent_format_json_degradations > 0` — alert
- `hidden_mode_drift_incidents > 0` — alert
- `required_pass_failure_rate > 10%` in any 1-hour window — alert
- `domain_entity_extraction_rate < baseline × 0.75` — alert
- Per-pass FAILED rate > 20% for any required pass in any 1-hour window — alert

**Manual rollback mechanism:** env var flip + worker restart. No automated circuit breaker (dropped per scope decision).

### §7.9 — Intentional naming carry-forwards

- **`DocumentGraphExtraction.graph_json`** — column name retained; semantics shift from "serialized graph payload" to "extraction audit blob." Documented in docstring.
- **`StageRun.status`** — existing column retained for Celery-level task state. New columns supplement it with extraction-specific semantics. The two are orthogonal.
- **`template_count` → `schema_count`** — the one rename. Old name is actively misleading.

### §7.10 — Status/read API shape

```json
{
  "document_id": "a1b2c3...",
  "document_status": "PARTIAL_COMPLETE",

  "latest_run": {
    "pipeline_run_id": "e4f5g6...",
    "status": "FAILED",
    "mode": "graph_only",
    "started_at": "2026-04-10T14:22:10Z",
    "finished_at": "2026-04-10T14:31:47Z",
    "ontology_bundle_key": "air_defense_v3",
    "ontology_bundle_label": "air_defense_v3",
    "ontology_name": "EIP Military Equipment Ontology",
    "ontology_version": "3.0.0",
    "passes": [
      {
        "pass_name": "reference",
        "execution_status": "COMPLETE",
        "yield_status": "HIT",
        "attempt": 1,
        "primary_entities_extracted": 12,
        "bridge_entities_extracted": 0,
        "relationships_extracted": 0,
        "relationships_rejected": 0,
        "schema_size_chars": 1842,
        "structured_output_mode": "strict"
      },
      {
        "pass_name": "radar_domain",
        "execution_status": "FAILED",
        "yield_status": null,
        "attempt": 3,
        "error_message": "HTTP 500 from docling-graph service"
      }
    ],
    "stage_summary": {
      "execution_status": "FAILED",
      "attempt": 2,
      "error_message": "Required pass radar_domain failed after 3 attempts"
    }
  },

  "graph_snapshot": {
    "pipeline_run_id": "9h8i7j...",
    "ontology_bundle_key": "air_defense_v3",
    "ontology_bundle_label": "air_defense_v3",
    "ontology_version": "3.0.0",
    "entity_count": 47,
    "edge_count": 32,
    "updated_at": "2026-04-09T09:15:00Z",
    "is_stale": true
  },
  "graph_queryable": false
}
```

**Three concepts exposed separately:**
- `document_status` — latest processing health (`Document.pipeline_status`)
- `latest_run` — most recent `PipelineRun` with pass rollup and `stage_summary.attempt` for Celery-level retry visibility
- `graph_snapshot` — the `DocumentGraphExtraction` row **currently** in PostgreSQL for this document, or `null` if there is no such row. A row may be absent because (a) no successful extraction has ever completed for the document, OR (b) a prior row existed but was deleted by a later operation (most commonly `purge_document_derivations` during an in-flight `full` reingest, before the new `derive_ontology_graph` writes its replacement row). The field reflects current PostgreSQL state, not historical "has this document ever been extracted successfully?" semantics. When non-null, `is_stale` is true iff `latest_run.status != COMPLETE` OR `latest_run.pipeline_run_id != graph_snapshot.pipeline_run_id`. When null, `is_stale` is not exposed.

**`is_stale` vs `graph_queryable` — two independent signals.**

`is_stale` and `graph_queryable` answer different questions, and the status API exposes both. **`graph_queryable` is a top-level field of the status response**, not nested inside `graph_snapshot` — it must remain meaningful when `graph_snapshot == null`.

- **`is_stale: bool`** *(nested inside `graph_snapshot` when non-null)* — the `graph_snapshot` row's `pipeline_run_id` doesn't match the most recent `PipelineRun`, OR the most recent run isn't COMPLETE. Answers "has the latest run confirmed this snapshot?" Has no meaning when `graph_snapshot == null`.

- **`graph_queryable: bool`** *(top-level)* — is there currently a queryable extraction-layer graph for this document in ArcadeDB? Answers "can I run extraction graph queries against this document right now?"

**Authoritative computation rule for `graph_queryable`** (the single source of truth — all prose elsewhere in the spec defers to this pseudocode):

```
graph_queryable = (
    graph_snapshot IS NOT NULL
    AND NOT (latest_stage_summary.rollback_executed IS True)
)
```

In words:
- If `graph_snapshot` is null → `graph_queryable = false`. This covers:
  - Document never had a successful extraction.
  - `full` reingest in progress: `purge_document_derivations` deleted the row before `derive_ontology_graph` ran, and the new extraction hasn't written a replacement yet. Even if rollback didn't fire, there's nothing to query.
  - `full` reingest failed: purge removed the prior row; the failed new extraction left no row behind.
- If `graph_snapshot` exists AND `latest_stage_summary.rollback_executed == True` → `graph_queryable = false`. The row is a historical audit record; the graph it describes has been rolled back.
- If `graph_snapshot` exists AND `rollback_executed` is False or NULL → `graph_queryable = true`. The row and its graph are intact.

`rollback_executed` is one input to the rule, not the whole rule. Snapshot existence is the other input.

**Computed independently of `error_message` prefixes.** The authoritative pseudocode is in the "Pseudocode" block further down (§7.10). The case tables below illustrate the rule in action.

**Status cases and their signals — `graph_only` reingests** (purge does not run; prior snapshot potentially preserved):

| latest_run state | stage_summary.rollback_executed | graph_snapshot | is_stale | graph_queryable |
|---|---|---|---|---|
| COMPLETE (success) | False | latest-run row | False | True |
| FAILED (gate failure) | False | prior-run row (intact) | True | True (no rollback ran) |
| FAILED (pre-mutation failure inside phase 2) | False | prior-run row (intact) | True | True (tracker stayed False, no rollback) |
| FAILED (mutation-time failure) | **True** | prior-run row (still in table, but graph deleted) | True | **False** (rollback ran) |
| FAILED (post-import failure, e.g., snapshot write) | **True** | prior-run row (still in table, but graph deleted) | True | **False** (rollback ran because earlier phases mutated) |
| FAILED (bundle load, metrics write, pre-import) | False | prior-run row (intact) | True | True (no mutations started) |

**Status cases — `full` reingests** (must be split by whether `purge_document_derivations` has run yet in the current run):

The `full` pipeline runs its stages in this order: `prepare_document` → `detect_and_translate` → `derive_document_metadata` → **`purge_document_derivations`** → `derive_picture_descriptions` → `derive_text_chunks_and_embeddings` → `derive_image_embeddings` → `derive_ontology_graph` → `finalize_document`. A `full` run is "pre-purge" until `purge_document_derivations` completes; any state observed before that point leaves the prior `DocumentGraphExtraction` row and graph intact. After purge completes and before a new `derive_ontology_graph` successfully writes its replacement, the document has no snapshot row at all.

**`full` reingest — pre-purge phase** (current run's stage is one of `prepare_document`, `detect_and_translate`, `derive_document_metadata`, and `purge_document_derivations` has not yet executed):

| latest_run state | stage_summary.rollback_executed | graph_snapshot | is_stale | graph_queryable |
|---|---|---|---|---|
| PROCESSING (pre-purge stage in progress) | NULL (no `derive_ontology_graph` row yet) | prior-run row (intact — purge hasn't run) | True | True |
| FAILED (pre-purge stage failed, e.g., OCR error) | NULL (no `derive_ontology_graph` row yet) | prior-run row (intact — purge never ran) | True | True |

**`full` reingest — post-purge, pre-`derive_ontology_graph`-success phase** (purge has completed, deleting the prior row and graph; the new `derive_ontology_graph` has not yet written a replacement):

| latest_run state | stage_summary.rollback_executed | graph_snapshot | is_stale | graph_queryable |
|---|---|---|---|---|
| PROCESSING (mid-chain, after purge, before `derive_ontology_graph` runs or while it runs) | NULL (no summary row, or summary row RUNNING) | **NULL** (purged, not yet replaced) | n/a (snapshot is null) | **False** (no graph) |
| FAILED (post-purge stage failed before `derive_ontology_graph`, e.g., embedding error) | NULL (no `derive_ontology_graph` summary row) | **NULL** (purged) | n/a | **False** |
| FAILED (`derive_ontology_graph` gate failure) | False | **NULL** (purged) | n/a | **False** |
| FAILED (`derive_ontology_graph` pre-mutation failure) | False | **NULL** (purged) | n/a | **False** |
| FAILED (`derive_ontology_graph` mutation-time failure, rollback fired) | True | **NULL** (purged — nothing to preserve) | n/a | **False** |
| FAILED (`derive_ontology_graph` pre-import failure) | False | **NULL** (purged) | n/a | **False** |

**`full` reingest — `derive_ontology_graph` successful and later:**

| latest_run state | stage_summary.rollback_executed | graph_snapshot | is_stale | graph_queryable |
|---|---|---|---|---|
| PROCESSING (`derive_ontology_graph` succeeded, downstream running) | False | new-run row (this run) | True (run not COMPLETE) | True (graph just written) |
| COMPLETE (all stages done) | False | new-run row | False | True |
| FAILED (downstream stage after `derive_ontology_graph` succeeded) | False | new-run row | True | True (graph exists from this run; downstream failure doesn't trigger extraction-stage rollback) |

**`is_stale` is "n/a" where `graph_snapshot` is null.** The status API omits `is_stale` entirely from responses when `graph_snapshot == null` (it lives inside the `graph_snapshot` object — see §7.10 response shape). The "n/a" entries in the tables are for clarity only; they do not appear in API output.

**Key differences between the three tables:**

- **`graph_only`** runs never touch the prior `DocumentGraphExtraction` row via purge. Failures that don't reach mutation time leave the prior snapshot and its graph intact. Queryability depends on whether `rollback_executed == True`.
- **`full` pre-purge** is indistinguishable from "not in the current run yet" — the prior snapshot is still present and queryable because purge hasn't fired.
- **`full` post-purge, pre-success** always has `graph_snapshot == null` because purge deleted the prior row and no new one has been written yet. `rollback_executed` is a diagnostic signal in this phase but cannot change the queryability outcome — there is nothing to roll back to.
- **`full` post-success** has a new-run snapshot row; subsequent downstream failures don't affect it because `derive_ontology_graph`'s rollback is scoped to itself.

**Pseudocode — the authoritative implementation for status-API computation:**

```python
@dataclass
class StatusSignals:
    snapshot: DocumentGraphExtraction | None   # the row, or None
    is_stale: bool                              # meaningful iff snapshot is not None
    graph_queryable: bool                       # top-level — always meaningful

def compute_status_signals(document_id: str, session) -> StatusSignals:
    snapshot = session.query(DocumentGraphExtraction).filter_by(
        document_id=document_id,
    ).first()

    latest_run = session.query(PipelineRun).filter_by(
        document_id=document_id,
    ).order_by(PipelineRun.started_at.desc()).first()

    latest_summary = None
    if latest_run is not None:
        latest_summary = session.query(StageRun).filter(
            StageRun.pipeline_run_id == latest_run.id,
            StageRun.stage_name == "derive_ontology_graph",
            StageRun.pass_name.is_(None),
        ).order_by(StageRun.attempt.desc()).first()

    # is_stale is only meaningful when a snapshot exists.
    is_stale = False
    if snapshot is not None:
        is_stale = (
            latest_run is None
            or latest_run.id != snapshot.pipeline_run_id
            or latest_run.status != "COMPLETE"
        )

    # graph_queryable — authoritative rule:
    #   (graph_snapshot exists) AND (rollback_executed is not True)
    rollback_ran = (
        latest_summary is not None
        and latest_summary.rollback_executed is True
    )
    graph_queryable = (snapshot is not None) and (not rollback_ran)

    return StatusSignals(
        snapshot=snapshot,
        is_stale=is_stale,
        graph_queryable=graph_queryable,
    )
```

Representative scenarios this rule handles correctly:

| Scenario | snapshot | rollback_executed | graph_queryable |
|---|---|---|---|
| Fresh successful extraction | row present | False | **True** |
| `graph_only` gate failure (prior run successful) | row present (prior run's) | False | **True** |
| `graph_only` pre-mutation failure | row present (prior run's) | False | **True** |
| `graph_only` mutation-time failure (rollback ran) | row present (prior run's — we don't re-write on failure) | **True** | **False** |
| `full` reingest in progress, before `derive_ontology_graph` stage | **None** (purged) | NULL (no summary yet) | **False** |
| `full` reingest in progress, after `derive_ontology_graph` success, before `finalize_document` | row present (new run's) | False | **True** |
| `full` reingest failed at `derive_ontology_graph` gate | **None** (purged; new run didn't write) | False | **False** |
| `full` reingest failed at mutation time | **None** (purged; new run didn't write because rollback fired before snapshot upsert) | True | **False** |

**Response shape — `graph_queryable` is a top-level sibling of `graph_snapshot`:**

```json
{
  "document_id": "a1b2c3...",
  "document_status": "PARTIAL_COMPLETE",

  "latest_run": { ... },

  "graph_snapshot": null,              // or the row if present
  "graph_queryable": false             // TOP-LEVEL — always meaningful
}
```

With a snapshot present:

```json
{
  "document_id": "a1b2c3...",
  "document_status": "COMPLETE",

  "latest_run": { ... },

  "graph_snapshot": {
    "pipeline_run_id": "9h8i7j...",
    "ontology_bundle_key": "air_defense_v3",
    "ontology_bundle_label": "air_defense_v3",
    "ontology_version": "3.0.0",
    "entity_count": 47,
    "edge_count": 32,
    "updated_at": "2026-04-09T09:15:00Z",
    "is_stale": false
  },
  "graph_queryable": true              // TOP-LEVEL
}
```

`graph_queryable` is NOT nested inside `graph_snapshot`, because it must remain meaningful when `graph_snapshot == null`. `is_stale` stays nested inside `graph_snapshot` because it only makes sense when a snapshot exists.

A follow-up spec may introduce proper shadow-write rollback that preserves prior graphs even after a failed mutation-time failure. v1 does not; the `rollback_executed=True` case is permanent data loss for that document's extraction-layer graph until a fresh successful ingest runs.

---

## §8 — Testing, measurement, and non-goals

### §8.1 — Definition of Done (functional)

**Bundle layout:**
- [ ] `ontology_bundles/air_defense_v3/` contains `ontology.yaml` (with `identity_fields`/`identity_scope` on every extract-bucket entity), `manifest.yaml`, `coverage.yaml`, `validators.py`, `derive_rules.py`, `extraction_schemas/__init__.py`, and the five pass modules.
- [ ] `ontology/ontology.yaml` symlink exists during PR 1/PR 2; removed in PR 3.

**Coverage checker:**
- [ ] `tools/check_extraction_coverage.py` runs in CI and passes. All 13 active rules (numbered 1–14 with rule 7 as a deleted placeholder) + manifest self-consistency sub-checks enforced.

**Service contract:**
- [ ] Docling-graph service accepts `(bundle_key, pass_name, docling_document_json, upstream_entities?)` on `POST /extract-pass`. Wire field name is `bundle_key` (short), not `ontology_bundle_key`. The longer name is reserved for persisted DB columns (Source, PipelineRun, DocumentGraphExtraction).
- [ ] Input-mode mismatch → HTTP 400. Unknown bundle/pass → HTTP 404. No fallback.
- [ ] `ontology_definition` removed from public request schemas, worker-to-service calls, and extraction hot path.

**Hot-path invariants:**
- [ ] No `create_model()` in the live extraction hot path.
- [ ] Bundle resolution precedence for standard ingest and `graph_only` reingest implemented and tested.
- [ ] `BundleResolutionError` raised when all tiers yield None.

**Persistence (migration applied):**
- [ ] All columns per §4.1–§4.4, including `PipelineRun.metrics` (JSONB, nullable).
- [ ] `ProvenanceMetadata` dataclass extended with `pipeline_run_id: str | None = None` (additive).
- [ ] Old `uq_stage_run` constraint explicitly dropped (name verified via `pg_constraint`).
- [ ] New partial unique indexes and query indexes created.
- [ ] `v_latest_pass_attempts` view created.
- [ ] Partial-index existing-data pre-flight passed (no duplicate summary rows in prod snapshot).

**Pipeline threading:**
- [ ] `start_ingest_pipeline()` returns `IngestDispatchResult(pipeline_run_id, celery_task_id)`. Callers updated.
- [ ] Standard reingest and `graph_only` reingest both resolve bundle and snapshot on `PipelineRun`.
- [ ] `Document.pipeline_status` updates use the existing `app/models/ingest.py:60` vocabulary — `COMPLETE` on `graph_only` success, `PARTIAL_COMPLETE` on any failure, unchanged on `full` success (downstream `finalize_document` terminalizes). No new vocabulary value is introduced.
- [ ] `PipelineRun.status` + `finished_at` terminalization is mode-conditional on the success path per §5.4 and §6.9. Failures terminalize in both modes.
- [ ] `derive_ontology_graph` includes an `except Exception:` catch-all that records stage-summary failure, terminalizes the run, and attempts best-effort rollback **gated on `tracker.any_mutation_attempted`**.
- [ ] `GraphWriteTracker` is instantiated once per `derive_ontology_graph` invocation and passed to all three phase helpers. Each helper calls `tracker.mark()` immediately before its first `graph_store` mutation.
- [ ] `StageRun.rollback_executed` column is set on the stage-summary row in every terminal branch (success=False, gate failure=False, pre-mutation failure=False, post-mutation failure=True).
- [ ] `PipelineRun.metrics` is written by `_write_pipeline_run_metrics()` between post-merge yield updates and graph import, so merge diagnostics survive import failures.
- [ ] Status API `graph_queryable` field is a **top-level** sibling of `graph_snapshot` (not nested), and is computed via the authoritative rule in §7.10: `(graph_snapshot IS NOT NULL) AND NOT (latest_stage_summary.rollback_executed IS True)`. Not derived from `error_message` prefix heuristics. `graph_queryable` must remain meaningful when `graph_snapshot == null`.

**Per-pass StageRun outcomes:**
- [ ] Each pass writes a row per attempt with execution/yield/skip status and counts.
- [ ] One summary row per stage invocation with `pass_name=NULL`.
- [ ] Retry history preserved.

**Required-pass gating:**
- [ ] The gate evaluates the latest attempt of every required pass and applies these rules:
  - `COMPLETE` (any yield) → PASSES
  - `SKIPPED` with authorized `skip_reason` (`{NO_UPSTREAM_ENDPOINTS}` for `relationships_only` + `skip_if_no_upstream_endpoints: true`) → PASSES
  - `FAILED` → FAILS
  - `SKIPPED` with unrecognized `skip_reason` → FAILS
- [ ] Gate runs before merge/import.

**Typed cross-pass references:**
- [ ] Same-pass uses `from_identity`/`to_identity` dicts.
- [ ] Cross-pass (`system_links`) uses `from_ref_id`/`to_ref_id`.
- [ ] Post-merge resolution handles the three reachable cases: exact match, missing endpoint, invalid triple. Ambiguous match is not reachable by construction.

**Merge / import / rollback:**
- [ ] `merge_and_resolve` produces `LogicalIdentity`-keyed IR, no RIDs.
- [ ] Three-phase import: nodes → domain edges → structural edges. `identity_to_rid` from zipped `upsert_nodes_batch_sync` return with `strict=True`.
- [ ] Structural edges via `graph_store.create_structural_edge_sync` (RID-based), not `upsert_relationships_batch_sync`.
- [ ] On merge/import failure, rollback calls `delete_document_graph_sync(document_id)` (or narrower sibling per residual check). Global properties explicitly NOT reverted.

**Monkey-patch contract tests:**
- [ ] Tests exist and pass for `_patched_build_request`, `_patched_call_api`, `NodeIDRegistry.get_node_id`.

**Schema budget:**
- [ ] Every pass schema size asserted below `settings.structured_output_threshold_chars` at CI time.

**Health / status:**
- [ ] `HealthResponse.template_count` → `schema_count`.
- [ ] Status response exposes `document_status`, `latest_run` (with `stage_summary.attempt`), `graph_snapshot`. Bundle key and label as separate fields.

**Legacy deletion (after soak, in PR 3):**
- [ ] All files listed in §7.5 deleted.
- [ ] All lint checks return zero.

### §8.2 — Measurable acceptance criteria

The refactor is successful only if the post-switchover run, measured against the same baseline corpus and methodology, meets all applicable criteria. A formal before/after baseline exists and is committed. Log comparison alone is not sufficient.

- [ ] `silent_format_json_degradations = 0` for the new path.
- [ ] `hidden_mode_drift_incidents = 0`.
- [ ] `required_pass_failure_rate` recorded. Non-zero failures explainable by genuine execution failures.
- [ ] **`domain_entity_extraction_rate ≥ 50%`** on an in-domain corpus. Definition: % of documents producing ≥1 primary non-reference entity. For mixed-domain corpus, apply to a declared in-domain subset.
- [ ] `relationship_hit_rate` recorded. % of documents with ≥1 accepted non-reference relationship. Above baseline after switchover.
- [ ] **`edge_retention_rate ≥ 70%`.** `relationships_accepted / (relationships_accepted + relationships_rejected)` across non-reference passes post-resolution.
- [ ] **`derive_ontology_graph` stage latency: p90 < 30 minutes.** Both p50 and p90 recorded.
- [ ] `document_extraction_anomaly_rate` recorded. Measured, not gated.
- [ ] `bridge_only_pass_rate` recorded per pass. Measured, not gated.
- [ ] `required_pass_skip_rate` recorded per required pass. Diagnostic only.

### §8.3 — Baseline methodology

**When:** after honest-failure changes on legacy path (part of PR 1), before new path activated.

**Corpus:** the 19 existing documents in the environment. Frozen. Recorded with document IDs, selection date, per-document in-domain classification.

**Execution:** `graph_only` reingest when prerequisites exist; full ingest labeled separately otherwise.

**Environment frozen:** branch/SHA, model/provider, `structured_output_threshold_chars`, timeouts, fail-open disabled, shadow mode disabled.

**Per-document fields persisted:**
- document_id, run timestamp, branch, commit SHA, model/provider
- **`derive_ontology_graph` stage duration from the stage-summary StageRun row (`pass_name IS NULL`)** — NOT per-pass rows
- attempted mode, completed mode, schema size
- structured output mode (strict vs json_only)
- entities accepted/rejected, relationships accepted/rejected
- whether the run failed
- whether the document qualifies as `document_extraction_anomaly`

**Publication:**
- `docs/superpowers/specs/2026-04-10-baseline-legacy-path.md` (baseline)
- `docs/superpowers/specs/2026-04-10-baseline-bundle-passes.md` (post-switchover)
- Machine-readable JSON/CSV artifact alongside each, if safe to store in-repo
- Side-by-side diff in PR 3 description

### §8.4 — Unit test coverage

**`tests/unit/test_ontology_bundles.py`** — loader and resolver
- `load_bundle_manifest` succeeds / fails on unknown / fails on malformed
- `resolve_bundle_key` precedence (standard and graph_only)
- `resolve_bundle_key` raises `BundleResolutionError` when all tiers None
- `load_ontology(bundle_key=None)` returns system default bundle
- `load_ontology` no longer accepts `prefer_active`

**`tests/unit/test_extraction_schemas.py`** — template invariants
- Every pass template instantiates with `**{}`
- Recursive partial-safety walk passes
- `build_display_label` returns non-empty for every extract-bucket entity
- Every Pydantic field exists in the corresponding ontology entity's properties

**`tests/unit/test_extraction_merge.py`** — merge/resolve and rejection taxonomy (checker rule 14)
- `LogicalIdentity` for document-scoped entities includes `document_id`
- `LogicalIdentity` for global-scoped entities omits `document_id`
- Bridge entities across passes collapse on identity match
- Content-hash identity deterministic and document-scoped
- Cross-pass edge resolution: exact match → accepted
- Missing endpoint → `FROM_ENDPOINT_NOT_FOUND` or `TO_ENDPOINT_NOT_FOUND`
- Invalid triple → `INVALID_TRIPLE`
- Unknown ref_id → `UNKNOWN_REF_ID`
- Missing rel_type → `MISSING_REL_TYPE`
- `INVALID_IDENTITY_PAYLOAD`: missing identity_field key → raised
- `INVALID_IDENTITY_PAYLOAD`: type-incompatible value → raised
- `INVALID_IDENTITY_PAYLOAD`: empty identity dict → raised
- **Entity merge collapse invariant: two entities with identical `LogicalIdentity` tuples produce ONE `MergedEntityRecord` with both passes in `pass_origins`.** Documents the impossibility of ambiguous match.
- **Confidence defaulting: explicit `0.0` preserved, not defaulted to `0.8`.**

**`tests/unit/test_yield_classification.py`**
- Zero/zero → EMPTY
- Only bridges → BRIDGES_ONLY
- Normal → HIT
- High rejection (≥0.75, ≥4 total) → DEGRADED
- Small pass below denominator cutoff → not DEGRADED
- Post-merge yield update HIT → DEGRADED

**`tests/unit/test_skip_logic.py`**
- Returns False for non-relationships_only
- Returns True when upstream_refs empty
- Returns True when no satisfiable triple
- Returns False when at least one triple satisfiable
- Filters upstream_refs to declared depends_on only

**`tests/unit/test_required_pass_gate.py`**
- Passes on all COMPLETE
- Fails on any FAILED
- Passes on authorized SKIPPED
- Fails on unauthorized SKIPPED
- Uses only latest attempt per pass

**`tests/unit/test_stage_run_retry.py`**
- Retry inserts new row with attempt+1
- Previous FAILED row not mutated
- `v_latest_pass_attempts` view returns highest-attempt row
- Uniqueness constraint prevents duplicates
- Summary row (pass_name=NULL) coexists with pass rows

**`tests/unit/test_display_label.py`**
- `build_display_label` for every extract-bucket entity
- Name-like key precedence
- Concatenation fallback
- Content-hash fallback determinism

### §8.5 — Integration tests

**`tests/integration/test_derive_ontology_graph_task.py`** — task-level (inspects state immediately after `derive_ontology_graph` returns, but does NOT run downstream stages)
- Invoke `derive_ontology_graph` directly against a prepared fixture `PipelineRun` + Document with prerequisite chunks/embeddings already in place
- Assert 5 per-pass StageRun rows exist with correct `execution_status` and `yield_status`
- Assert 1 stage-summary StageRun row exists with `pass_name IS NULL`
- Assert `DocumentGraphExtraction` row written with audit fields + FK to run
- Assert merged graph vertices and edges exist in ArcadeDB
- Assert `HAS_PROVENANCE` (auto-created by `upsert_nodes_batch_sync`) and `MENTIONED_IN` (from `derive_rules`) edges present
- Assert `PipelineRun.metrics` populated with `pass_outcomes`, `document_extraction_anomaly`, `overall_relationship_rejection_ratio`, `bundle_key_display`
- Run with `mode="graph_only"`: assert `PipelineRun.status == "COMPLETE"`, `PipelineRun.finished_at IS NOT NULL`, `Document.pipeline_status == "COMPLETE"`
- Run with `mode="full"`: assert `PipelineRun.status == "PROCESSING"`, `PipelineRun.finished_at IS NULL`, `Document.pipeline_status` **unchanged** by `derive_ontology_graph` — because the downstream stages haven't run in this test

**`tests/integration/test_end_to_end_full_ingest.py`** — true full-pipeline E2E
- Upload a fresh fixture document and run the entire Celery chain to completion (`prepare_document` → Docling convert → chunking/embedding → `derive_ontology_graph` → `finalize_document` → community detection if enabled)
- Waits for Celery chain completion (with sensible timeout)
- Assert `Document.pipeline_status == "COMPLETE"` (set by `finalize_document`, NOT by `derive_ontology_graph`)
- Assert `PipelineRun.status == "COMPLETE"` and `finished_at IS NOT NULL` (set by `finalize_document`)
- Assert `DocumentGraphExtraction` row written and matches the extracted graph
- Assert `PipelineRun.metrics` populated
- Assert 5 per-pass StageRun rows + 1 stage-summary row present for the `derive_ontology_graph` stage
- Assert extracted entities and edges exist in ArcadeDB

**`tests/integration/test_end_to_end_graph_only_reingest.py`** — graph-only E2E
- Prepare a document with a prior successful full ingest
- Trigger `graph_only` reingest via the reingest route
- Assert a new `PipelineRun` row is created with `mode="graph_only"` and bundle snapshot populated via the `explicit → inherited → source → system` precedence
- Assert the `derive_ontology_graph` task runs to completion
- Assert `PipelineRun.status == "COMPLETE"`, `finished_at IS NOT NULL` (because `graph_only` mode terminalizes in `derive_ontology_graph` itself per §5.4)
- Assert `Document.pipeline_status == "COMPLETE"`
- Assert the new `DocumentGraphExtraction` row replaces the prior snapshot (same `document_id`, new `pipeline_run_id`)
- Assert the prior `PipelineRun` row is unchanged (audit preservation)

**`tests/integration/test_bundle_resolution_precedence.py`**
- Source default, system default, explicit override, inherited from latest run, legacy-NULL fallthrough — all precedence orderings

**`tests/integration/test_required_pass_failure_flow_graph_only.py`** — scoped to `graph_only` reingest because `full` reingest already purges the prior snapshot via `purge_document_derivations` before `derive_ontology_graph` runs (see §4.3 purge caveat)
- Setup: document with a prior successful full ingest + intact `DocumentGraphExtraction` row + intact ArcadeDB graph
- Trigger a `graph_only` reingest with forced `radar_domain` HTTP 500 × 3
- Assert 3 FAILED StageRun rows with attempts 1/2/3
- Assert `IngestFailed`, stage-summary `execution_status=FAILED`, `rollback_executed=False`, `error_message` starts with `gate_failed:`
- Assert `PipelineRun.status=FAILED`, `Document.pipeline_status=PARTIAL_COMPLETE`
- Assert prior `DocumentGraphExtraction` row **untouched** (same `pipeline_run_id`, same `graph_json`, same `updated_at`)
- Assert prior ArcadeDB graph still queryable — snapshot's vertices exist
- Assert status API returns `graph_queryable=True`
- Assert `_delete_extraction_layer_graph` NOT called (tracker never marked)

**`tests/integration/test_required_pass_failure_flow_full_post_purge.py`** — documents the accepted limitation from §4.3 and §6.8 item 3 for the post-purge phase of a full reingest
- Setup: document with a prior successful full ingest
- Trigger a new `full` reingest with forced `radar_domain` HTTP 500 × 3
- Wait for `purge_document_derivations` to execute (the test explicitly synchronizes on this stage completing), verifying that the prior `DocumentGraphExtraction` row and ArcadeDB document graph have been deleted. This establishes the post-purge precondition.
- Allow the chain to continue through to `derive_ontology_graph`, which fails at its gate due to the forced `radar_domain` errors.
- Assert `IngestFailed`, stage-summary `execution_status=FAILED`, `rollback_executed=False` (tracker never marked because gate failed before phase 2)
- Assert `PipelineRun.status=FAILED`, `Document.pipeline_status=PARTIAL_COMPLETE`
- Assert **no `DocumentGraphExtraction` row exists** for this document (purge removed it; failed `derive_ontology_graph` did not write a new one)
- Assert status API returns `graph_snapshot=null` and `graph_queryable=false`
- Assert `_delete_extraction_layer_graph` NOT called (nothing to roll back to)
- This test asserts the status quo behavior documented in §4.3, not a desired behavior. If `purge_document_derivations` is ever redesigned to be non-destructive, this test is updated in that follow-up spec.

**`tests/integration/test_required_pass_failure_flow_full_pre_purge.py`** — companion test demonstrating that pre-purge failures leave the prior snapshot intact
- Setup: document with a prior successful full ingest
- Trigger a new `full` reingest with forced failure in `detect_and_translate` (or another pre-purge stage — e.g., an invalid language code)
- Assert `purge_document_derivations` did NOT run (stage was never reached)
- Assert `PipelineRun.status=FAILED`, `Document.pipeline_status=PARTIAL_COMPLETE`
- Assert the prior `DocumentGraphExtraction` row is **still present and unchanged** (same `pipeline_run_id`, `graph_json`, `updated_at` as before the reingest attempt)
- Assert prior ArcadeDB graph state is still queryable — snapshot's vertices exist
- Assert status API returns `graph_snapshot != null` AND `graph_queryable=true`
- This test demonstrates that `full`-mode "snapshot preservation" is phase-dependent, not always-absent.

**`tests/integration/test_merge_import_failure_rollback.py`** — three test cases covering the three rollback regimes

**Test case A: mutation-time failure triggers rollback.**
- Setup: pre-populate graph with one prior successful extraction for this document
- Patch the graph store so `upsert_nodes_batch_sync` raises on the first call. This ensures `_import_graph_phase_nodes` calls `tracker.mark()` and then fails inside the mutation call.
- Assert the abstract rollback primitive `_delete_extraction_layer_graph(document_id)` was invoked exactly once. **Do NOT hardcode the concrete method name** (`delete_document_graph_sync` vs `delete_extraction_layer_graph_sync`) — PR 1 may wire the abstract helper to either per residual check #1, and the test must be resilient to either choice. Patch `_delete_extraction_layer_graph` itself and assert its call args.
- Assert `Document.pipeline_status == "PARTIAL_COMPLETE"`
- Assert stage-summary: `execution_status == "FAILED"`, `rollback_executed == True`, `error_message` starts with `merge_or_import_failed:`
- Assert the status API returns `graph_queryable == False` for this document
- Assert `DocumentGraphExtraction` row still present (historical audit record) but `is_stale == True`
- Assert global-scoped entity vertices still exist as vertices; their properties are NOT asserted against pre-run state (per §8.9 non-goal on property restoration)

**Test case B: pre-mutation failure inside phase 2 does NOT trigger rollback.**
- Setup: same as A
- Patch `build_display_label` to raise on the first call. This fires inside `_import_graph_phase_nodes` BEFORE `tracker.mark()` is reached.
- Assert `_delete_extraction_layer_graph` was NOT invoked
- Assert stage-summary: `execution_status == "FAILED"`, `rollback_executed == False`, `error_message` starts with `merge_or_import_failed:` (or `unexpected_failure:` depending on which exception type the build crash raises)
- Assert the status API returns `graph_queryable == True`
- Assert `DocumentGraphExtraction` row present AND its prior-run data is still queryable

**Test case C: pre-import failure (metrics write crash) does NOT trigger rollback.**
- Setup: same as A
- Patch `_write_pipeline_run_metrics` to raise
- Assert `_delete_extraction_layer_graph` was NOT invoked
- Assert stage-summary: `rollback_executed == False`, `error_message` starts with `unexpected_failure:`
- Assert status API returns `graph_queryable == True`
- Assert prior graph fully intact

**Common invariant across A/B/C:** the test never hardcodes the concrete method name used by `_delete_extraction_layer_graph`. The abstraction layer is the stable contract; the concrete wiring is an implementation choice for PR 1.

**`tests/integration/test_system_links_skip.py`**
- Ingest document where radar_domain, missile_domain, other_systems all return EMPTY
- Assert `system_links` is SKIPPED with `skip_reason=NO_UPSTREAM_ENDPOINTS`
- Assert ingest completes successfully (SKIPPED allowed)
- Assert `system_links` StageRun has `execution_status=SKIPPED` and `yield_status=NULL`

**`tests/integration/test_legacy_null_bundle.py`**
- Create PipelineRun with `ontology_bundle_key=NULL` (legacy)
- Trigger `graph_only` reingest
- Assert inheritance falls through to source/system default (NOT collapsed into `air_defense_v3`)
- Assert INFO log emitted
- Assert new PipelineRun has non-null bundle_key
- Assert status API returns `ontology_bundle_key=null, ontology_bundle_label="legacy/unknown"` for the old run, populated values for the new run

### §8.6 — Contract tests (PR 3)

**`docker/docling-graph/tests/test_monkey_patches.py`** — detect upstream drift

`_patched_build_request`:
- Preserves `format=<schema_dict>` (raw JSON Schema) through the LiteLLM filter
- Sets `stream=False` and preserves `think="low"` for Ollama thinking models
- Falls back to `format="json"` when schema size exceeds `settings.structured_output_threshold_chars`

`_patched_call_api` (preserves current behavior — tests lock it in place; metadata flag names match the existing implementation in `docker/docling-graph/app/main.py`):
- Valid response with non-empty `message.content` → returns content unchanged
- Empty `message.content`, `reasoning_content` present → raises `ClientError`. Error message includes `finish_reason`, usage fields, and `has_reasoning_content=True`, `has_thinking=False` metadata flags. **`reasoning_content` is surfaced as diagnostic metadata but NOT substituted for content.**
- Empty `message.content`, `thinking` field present → raises `ClientError` with `has_thinking=True` flag. Non-substitution rule applies.
- Empty content with both reasoning_content and thinking → raises `ClientError` with `has_reasoning_content=True` and `has_thinking=True`.
- Completely empty response → raises `ClientError` with both flags False.

Test assertion names must use the actual metadata keys `has_reasoning_content` and `has_thinking` as they appear in the live patch — not shortened variants.

`NodeIDRegistry.get_node_id`:
- Correctly handles `TABLE_REF_<fingerprint>` via `rsplit("_", 1)[0]` (regression for collision bug fixed in stabilization commit B)

**`docker/docling-graph/tests/test_extract_pass_endpoint.py`** (added in PR 1)
- `document_only` pass with unexpected `upstream_entities` → HTTP 400
- `document_plus_entity_refs` pass with missing `upstream_entities` → HTTP 400
- `document_plus_entity_refs` pass with empty `upstream_entities` → HTTP 400
- Unknown `bundle_key` → HTTP 404
- Unknown `pass_name` → HTTP 404
- Valid `document_only` request → HTTP 200
- Valid `document_plus_entity_refs` request → HTTP 200

### §8.7 — CI lint rules (grep-based)

**After PR 2:**
- `grep -r "prefer_active" app/` returns zero

**After PR 3:**
- `grep -r "from ontology_bundles\.[^.]*\.extraction_schemas" app/` returns zero
- `grep -r "ontology_definition" app/ docker/` returns zero
- `grep -r "/app/ontology/" app/ docker/` returns zero
- `grep -nE '\.get\("confidence",|"confidence" *or |\.confidence *or '` returns zero in the extraction code paths
- `grep -r "graph_extraction_engine" app/` returns zero
- `grep -r "graph_layered_" app/` returns zero

Each lint is a separate CI job failing the build independently.

### §8.8 — Non-goals

- Upstream fixes to LiteLLM or `NodeIDRegistry` monkey-patches. Contract tests instead.
- Ollama thinking-model response-shape handling. Existing behavior preserved.
- Community detection refactoring.
- UI for extraction inspection or bundle management.
- Anomaly alerting or dashboards. Only raw signals recorded.
- `query_profiles` models or retrieval path refactoring.
- Shipping more than one bundle. Only `air_defense_v3`.
- Semantic / co-occurrence edge derivation. Only deterministic structure edges.
- Backfilling historical rows. `NULL` = `legacy/unknown`, never rewritten.
- Extracting `DOCUMENT`-type ontology entities from LLM text. Structural `Document` vertex carries metadata via existing upstream stages.
- **Atomic property restoration for global-scoped entities on failed ingest.** v1 rollback is scoped to document-local state only.
- Moving `structured_output_threshold_chars` into bundle metadata. Stays in service config.
- Changing `upsert_nodes_batch_sync` or `upsert_relationships_batch_sync` return shapes.
- Renaming `DocumentGraphExtraction.graph_json`. Carries forward pre-refactor name.
- Write log / compensating RID-level rollback. v1 uses `delete_document_graph_sync` only.
- Build-time codegen for extraction schemas (Approach B). Future optimization.
- Multi-bundle A/B at runtime.
- Schema-size per-bundle override. Threshold is global.
- Automated circuit breaker for new-path failure rate. Manual rollback only.

---

## Residual execution-time checks

Carried forward from brainstorm decisions and spec review. These are NOT design open questions — they are verified during implementation and do not reopen the design.

1. **Wire `_delete_extraction_layer_graph` to a concrete method.** The runtime code in §5.4 calls the abstract helper `_delete_extraction_layer_graph(document_id)` for rollback. PR 1 verifies whether the existing `graph_store.delete_document_graph_sync(document_id)` satisfies the contract in §6.8:

   - **Must delete** document-scoped extracted entity vertices, domain edges tagged with this `document_id`, and structural edges produced by `derive_rules.py` in phase 4.
   - **Must not delete** chunks, embeddings, the structural `Document` root, or global-scoped entity vertices.

   If `delete_document_graph_sync` already satisfies this (its current scope matches), PR 1 wires `_delete_extraction_layer_graph` as a one-line call to it. If it over-deletes (removes chunks or the structural Document root), PR 1 adds a narrower sibling method — e.g., `graph_store.delete_extraction_layer_graph_sync` — that satisfies the contract exactly, and wires the abstract helper to the new method. Either way, the runtime code in §5.4 is unchanged.

2. **Resolve the exact existing `uq_stage_run` constraint name.** The Alembic migration in §4.8 issues `op.drop_constraint(...)` before creating the new indexes. Spec review confirmed the current name is `uq_stage_run` (literal) at `app/models/ingest.py:237`, but PR 1 re-verifies via:

   ```sql
   SELECT conname FROM pg_constraint
   WHERE conrelid = 'ingest.stage_runs'::regclass AND contype = 'u';
   ```

   and hardcodes the result into the revision file before the migration is coded.

3. **Existing-data pre-flight for `uq_stage_runs_summary_row`.** The partial unique index is deliberately scoped narrowly to `WHERE pass_name IS NULL AND stage_name = 'derive_ontology_graph'`, so it does NOT govern existing non-extraction stage rows and cannot fail the migration on pre-existing duplicates from other stages. PR 1 still runs a one-line pre-flight query as a sanity check to confirm no existing `derive_ontology_graph` rows have `pass_name IS NULL` (they shouldn't — the current code writes one row per stage invocation without the new summary-row concept):

   ```sql
   SELECT pipeline_run_id, attempt, count(*)
   FROM ingest.stage_runs
   WHERE pass_name IS NULL AND stage_name = 'derive_ontology_graph'
   GROUP BY 1, 2
   HAVING count(*) > 1;
   ```

   If any duplicates exist, PR 1 deduplicates them in a prior migration step before creating the index. The narrowing of the index was done at design time (see §4.4 and §4.8) specifically to avoid the broader blast radius flagged in the user review.

4. **Verify `_create_provenance_edges_batch_sync` semantics.** PR 1 confirms that `upsert_nodes_batch_sync` auto-creates `HAS_PROVENANCE` edges exactly once per upsert call when a non-None `ProvenanceMetadata` is passed. If the behavior is conditional on other fields (e.g., only when `page_numbers` is present), the spec's assumption in §3.8 that `HAS_PROVENANCE` is auto-created on every node needs adjustment. The contract is: "phase 2 produces HAS_PROVENANCE edges exactly once per extracted entity, via exactly one of the two possible paths — never both, never neither." PR 1 picks whichever path satisfies that contract.

   **Synchronized update if this check fails:** if PR 1 discovers that auto-creation is conditional on fields the spec did not account for, four sections must be updated in one commit to stay consistent: §2 (coverage.yaml `derive` bucket), §3.3 (reference-pass module docstring), §3.8 (`derive_structural_edges` signature and body), and §5.6 Phase 2 (provenance passing). Piecemeal fixes will silently drift the contract. PR 1's verification step must produce a single synchronized update commit if the assumption doesn't hold — or leave all four in place if it does.

---

## Approval status

All sections locked during brainstorming:
- §1 Architecture
- §2 Bundle layout and manifest contract
- §3 Pass definitions, identity model, merge semantics, derive rules
- §4 Data model changes
- §5 Runtime flow
- §6 Failure semantics and pass outcomes
- §7 Migration and rollout plan
- §8 Testing, measurement, and non-goals

Next: spec review loop via `spec-document-reviewer` subagent, user final review, then `writing-plans` skill to produce the implementation plan.
