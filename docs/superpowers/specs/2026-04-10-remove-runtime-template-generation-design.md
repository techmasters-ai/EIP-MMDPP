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

**Approach A — hand-authored fixed schemas, disk-based bundles, worker-side orchestration.** Considered and rejected: (B) build-time codegen, (C) fixing runtime generation in place. A is the only option that removes the brittle feature; C preserves it; B adds tooling complexity without enough payoff at the current scale (24 extract-bucket entities out of 46 total).

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
- **`derive`** — produced deterministically post-merge by `derive_rules.py`. Not asked of the LLM. Not part of any pass's extraction contract.
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

### Checker rules (full 14-rule list)

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

Extracts document-level anchors. No LLM-extracted relationships — structural
edges (HAS_PROVENANCE, MENTIONED_IN) are derived post-merge in derive_rules.py.
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

**Empty `identity_fields: []`** — content-hash fallback. The merge code hashes sorted non-system field values. Strongly recommended to pair with `identity_scope: document` (checker warns on `[]` + `global` — rule 12).

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

**Edge case — document-scoped entities with `document_id` in identity_values:** `build_display_label` includes `document_id` in its `NAME_LIKE_KEYS` list as a fallback. For document-scoped entities whose `as_upsert_identity_dict` adds a raw UUID `document_id` to the dict, the display label could end up as a bare UUID string. This is acceptable: the fallback below it (content-hash `<entity_type>_<hash>`) will not fire because `document_id` is never empty on document-scoped entities, and a UUID is a valid if unfriendly human label. Operators viewing the graph see the UUID and can pivot to the Document vertex for a friendly title. If this becomes annoying, a later spec can introduce a "title lookup" step that swaps the UUID for the document's real title at display time.

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

**Latest-snapshot semantics.** `DocumentGraphExtraction` is keyed on `document_id` — one row per document, always reflecting the latest successful `derive_ontology_graph` run. Failed runs do NOT overwrite the previous snapshot. Historical extraction ledger lives on `PipelineRun` + `StageRun`, not here.

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
    WHERE pass_name IS NULL;

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

    # Indexes
    op.create_index('uq_stage_runs_run_pass_attempt', 'stage_runs',
                    ['pipeline_run_id', 'stage_name', 'pass_name', 'attempt'],
                    unique=True, postgresql_where=sa.text('pass_name IS NOT NULL'),
                    schema='ingest')
    op.create_index('uq_stage_runs_summary_row', 'stage_runs',
                    ['pipeline_run_id', 'stage_name', 'attempt'],
                    unique=True, postgresql_where=sa.text('pass_name IS NULL'),
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

    try:
        manifest = load_bundle_manifest(run.ontology_bundle_key)
        ontology = load_ontology(bundle_key=run.ontology_bundle_key)
        doc_json = build_docling_document_json(run.document_id)

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

        # Three-phase graph import
        identity_to_rid = _import_graph_phase_nodes(merged, ontology, run.document_id)
        _import_graph_phase_domain_edges(merged, ontology)
        _import_graph_phase_structural_edges(
            merged, identity_to_rid, run.document_id, str(run.id),
        )

        # DocumentGraphExtraction upsert (latest snapshot only)
        _upsert_document_graph_extraction(
            document_id=run.document_id,
            pipeline_run_id=run.id,
            run=run,
            merged=merged,
        )

        stage_summary.status = "COMPLETE"
        stage_summary.execution_status = "COMPLETE"
        stage_summary.finished_at = now()
        run.status = "COMPLETE"
        _update_document_pipeline_status(run.document_id, "OK")
        session.commit()
        return {"stage": "derive_ontology_graph", "status": "ok",
                "entities": len(merged.entities),
                "edges": len(merged.edges)}

    except IngestFailed as exc:
        # Gate failure — no graph writes occurred, no rollback needed
        stage_summary.status = "FAILED"
        stage_summary.execution_status = "FAILED"
        stage_summary.error_message = f"gate_failed: {exc}"
        stage_summary.finished_at = now()
        run.status = "FAILED"
        _update_document_pipeline_status(run.document_id, "PARTIAL_COMPLETE")
        session.commit()
        raise

    except (MergeError, GraphImportError) as exc:
        # Merge/import failure — partial writes may exist
        logger.exception("derive_ontology_graph merge/import failure")
        rollback_note = ""
        try:
            graph_store.delete_document_graph_sync(run.document_id)
        except Exception as rollback_exc:
            rollback_note = f"; ROLLBACK_ALSO_FAILED: {rollback_exc}"
            logger.error("rollback during merge/import failure also failed: %s", rollback_exc)

        stage_summary.status = "FAILED"
        stage_summary.execution_status = "FAILED"
        stage_summary.error_message = f"merge_or_import_failed: {exc}{rollback_note}"
        stage_summary.finished_at = now()
        run.status = "FAILED"
        _update_document_pipeline_status(run.document_id, "PARTIAL_COMPLETE")
        # DocumentGraphExtraction intentionally NOT updated
        session.commit()
        raise
```

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
) -> dict[LogicalIdentity, str]:
    # Note: ProvenanceMetadata gains a new optional field `pipeline_run_id`
    # in PR 1 as part of this refactor — see §5.6 note below. It is
    # backward-compatible (Optional with default None).
    provenance = ProvenanceMetadata(
        document_id=document_id,
        pipeline_run_id=merged.pipeline_run_id,
    )
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

`upsert_nodes_batch_sync` signature unchanged (still returns `list[str]`). `strict=True` on `zip` guards against length mismatch.

**ProvenanceMetadata extension (PR 1, additive):** the existing `ProvenanceMetadata` dataclass in `app/services/graph_store.py` gains one new optional field:

```python
@dataclass
class ProvenanceMetadata:
    document_id: str
    page_numbers: list[int] | None = None
    upload_datetime: datetime | None = None
    document_datetime: datetime | None = None
    pipeline_run_id: str | None = None     # NEW in PR 1 — optional, additive
```

This lets downstream code (including `_create_provenance_edges_batch_sync` and future audit queries) correlate provenance writes back to the run that produced them without changing any existing callers. Populated by the worker in phase 2; used as metadata on the auto-created HAS_PROVENANCE edges.

**Phase 3 — domain relationship upsert (identity-based):**

```python
def _import_graph_phase_domain_edges(merged, ontology):
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
    graph_store.upsert_relationships_batch_sync(rel_records, provenance)
```

**Phase 4 — derived structural edges (RID-based, structural-edge path):**

```python
def _import_graph_phase_structural_edges(
    merged: MergedExtraction,
    identity_to_rid: dict[LogicalIdentity, str],
    document_id: str,
    pipeline_run_id: str,
) -> None:
    chunks = _load_chunks_for_derivation(document_id)  # returns list[ChunkForDerivation]
    document_rid = _get_structural_document_rid(document_id)

    derived = derive_rules.derive_structural_edges(
        merged=merged,
        identity_to_rid=identity_to_rid,
        chunks=chunks,
        document_rid=document_rid,
    )
    for edge in derived:
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

A failed gate causes:
1. `PipelineRun.status = "FAILED"`
2. `Document.pipeline_status = "PARTIAL_COMPLETE"`
3. `IngestFailed` exception propagates
4. `DocumentGraphExtraction` NOT updated — prior snapshot intact
5. No rollback needed (no graph writes occurred yet)

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

On merge or import failure, rollback calls `graph_store.delete_document_graph_sync(document_id)` (or its narrower sibling if PR 1 discovers the existing method over-deletes chunks/embeddings — see residual check).

**Scope:**
- **Deleted:** document-scoped entity vertices (their identity includes `document_id`), structural vertices with this document_id, edges incident to any deleted vertex, domain edges tagged with this document_id in provenance metadata.
- **NOT deleted:** global-scoped entity vertices (PLATFORM, RADAR_SYSTEM, etc.). Their properties may have been merged during the failed run; those merges are not reverted in v1.

**Accepted limitation:** global entity properties enriched during a failed run persist until the next successful ingest that references them overwrites them. Atomic property restoration is explicitly a non-goal (§8.9).

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

**Per-ingest PipelineRun:**

```
      RUNNING
         │
         ├─ all required passes pass the gate, merge + import OK → COMPLETE
         └─ any failure → FAILED
```

**Document.pipeline_status transitions:**

| Trigger | New value |
|---|---|
| Gate failure | `PARTIAL_COMPLETE` |
| Merge/import failure | `PARTIAL_COMPLETE` |
| Unexpected exception | `PARTIAL_COMPLETE` |
| Success | `OK` |

Regression from `OK` → `PARTIAL_COMPLETE` on failed `graph_only` reingest is the locked behavior. `Document.pipeline_status` reflects the latest ingest attempt's health; `DocumentGraphExtraction` reflects the latest successful snapshot; both are surfaced separately in the status API.

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

**Coverage checker:** `tools/check_extraction_coverage.py` implementing all 14 rules. Runs in CI.

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
  }
}
```

**Three concepts exposed separately:**
- `document_status` — latest processing health (`Document.pipeline_status`)
- `latest_run` — most recent `PipelineRun` with pass rollup and `stage_summary.attempt` for Celery-level retry visibility
- `graph_snapshot` — latest successful `DocumentGraphExtraction`, may be older than `latest_run` if the latest run failed. `is_stale` is true when `latest_run.status != COMPLETE` OR `latest_run.pipeline_run_id != graph_snapshot.pipeline_run_id`. `null` if no successful extraction has ever completed for this document.

---

## §8 — Testing, measurement, and non-goals

### §8.1 — Definition of Done (functional)

**Bundle layout:**
- [ ] `ontology_bundles/air_defense_v3/` contains `ontology.yaml` (with `identity_fields`/`identity_scope` on every extract-bucket entity), `manifest.yaml`, `coverage.yaml`, `validators.py`, `derive_rules.py`, `extraction_schemas/__init__.py`, and the five pass modules.
- [ ] `ontology/ontology.yaml` symlink exists during PR 1/PR 2; removed in PR 3.

**Coverage checker:**
- [ ] `tools/check_extraction_coverage.py` runs in CI and passes. All 14 rules + manifest self-consistency sub-checks enforced.

**Service contract:**
- [ ] Docling-graph service accepts `(ontology_bundle_key, pass_name, docling_document_json, upstream_entities?)` on `POST /extract-pass`.
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
- [ ] `Document.pipeline_status` updated on all terminal states (`OK`/`PARTIAL_COMPLETE`).

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

**`tests/integration/test_end_to_end_bundle_extraction.py`** — the smoke test
- Ingest fixture document through bundle/pass path
- Assert bundle resolution, 5 per-pass StageRun rows, stage-summary row, `DocumentGraphExtraction` with audit fields, merged graph in ArcadeDB, `Document.pipeline_status=OK`, `HAS_PROVENANCE` + `MENTIONED_IN` edges present

**`tests/integration/test_bundle_resolution_precedence.py`**
- Source default, system default, explicit override, inherited from latest run, legacy-NULL fallthrough — all precedence orderings

**`tests/integration/test_required_pass_failure_flow.py`**
- Simulate `radar_domain` returning HTTP 500 three times
- Assert 3 FAILED StageRun rows with attempts 1/2/3
- Assert `IngestFailed`, stage-summary FAILED, `PipelineRun.status=FAILED`, `Document.pipeline_status=PARTIAL_COMPLETE`
- Assert prior `DocumentGraphExtraction` untouched
- Assert rollback NOT called (gate failed before merge)

**`tests/integration/test_merge_import_failure_rollback.py`**
- Setup: pre-populate graph with one prior successful extraction
- Force `merge_and_resolve` to raise
- Assert `delete_document_graph_sync` called once with correct `document_id`
- Assert `Document.pipeline_status = PARTIAL_COMPLETE`
- Assert prior `DocumentGraphExtraction` untouched
- Assert stage-summary FAILED with `merge_or_import_failed:` prefix
- Assert document-scoped entity vertices from this run no longer exist after rollback
- Assert global-scoped entity vertices still exist (not deleted by rollback)
- **Explicitly NOT asserted:** whether global entity properties match pre-run state. Property restoration is intentionally undefined (§6.8, §8.9). In-code comment pointing to those sections.

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

1. **Confirm `delete_document_graph_sync(document_id)` scope.** In PR 1, verify whether the existing method over-deletes chunks/embeddings/structural-root state that should be preserved during extraction-stage rollback. If it does, add a narrower sibling (`delete_document_extraction_graph_sync`) that targets only extraction-layer contributions. The rollback call in §5.4 and §6.8 uses whichever of the two is correct — the contract is unchanged ("delete only this run's extraction-layer contributions for this document, without touching chunks/embeddings or global entities").

2. **Resolve the exact existing `uq_stage_run` constraint name.** The Alembic migration in §4.8 issues `op.drop_constraint(...)` before creating the new indexes. Spec review confirmed the current name is `uq_stage_run` (literal) at `app/models/ingest.py:237`, but PR 1 re-verifies via:

   ```sql
   SELECT conname FROM pg_constraint
   WHERE conrelid = 'ingest.stage_runs'::regclass AND contype = 'u';
   ```

   and hardcodes the result into the revision file before the migration is coded.

3. **Existing-data pre-flight for `uq_stage_runs_summary_row`.** The new partial unique index `uq_stage_runs_summary_row` covers `(pipeline_run_id, stage_name, attempt) WHERE pass_name IS NULL`. That `WHERE` clause matches **every existing non-extraction StageRun row** in production. PR 1's migration must verify no existing non-extraction stages have duplicate `(pipeline_run_id, stage_name, attempt)` rows, or the migration will fail with a unique-violation on existing data. Pre-flight query:

   ```sql
   SELECT pipeline_run_id, stage_name, attempt, count(*)
   FROM ingest.stage_runs
   WHERE pass_name IS NULL
   GROUP BY 1, 2, 3
   HAVING count(*) > 1;
   ```

   PR 1 runs this against the production snapshot before coding the index creation. If any duplicates exist, the migration either (a) deduplicates them in a prior step, or (b) narrows the partial-index `WHERE` clause to target only derive_ontology_graph summary rows: `WHERE pass_name IS NULL AND stage_name = 'derive_ontology_graph'`.

4. **Verify `_create_provenance_edges_batch_sync` semantics.** PR 1 confirms that `upsert_nodes_batch_sync` auto-creates `HAS_PROVENANCE` edges exactly once per upsert call when a non-None `ProvenanceMetadata` is passed. If the behavior is conditional on other fields (e.g., only when `page_numbers` is present), the spec's assumption in §3.8 that `HAS_PROVENANCE` is auto-created on every node needs adjustment. The contract is: "phase 2 produces HAS_PROVENANCE edges exactly once per extracted entity, via exactly one of the two possible paths — never both, never neither." PR 1 picks whichever path satisfies that contract.

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
