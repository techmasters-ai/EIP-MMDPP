# Flat-Schema Profile Refactor — Design

**Date:** 2026-04-25
**Status:** Draft, awaiting user review
**Scope:** Bring the four starter query profiles (System Dossier, System Components, System RF Parameters, System Performance) onto the flat-checklist extraction schema, and sync the canonical ontology entities so the schema-drift xfails clear at the same time.

---

## 1. Background

### 1.1 What changed under us

The Phase 5/6/7 Pydantic ontology SSoT work refactored the radar/missile extraction schemas to a **flat checklist** model: instead of emitting nested `ANTENNA`, `RECEIVER`, `BOOSTER`, `SEEKER`, `SPECIFICATION` etc. entities and connecting them to `RADAR_SYSTEM` / `MISSILE_SYSTEM` via typed edges (`HAS_ANTENNA`, `EMITS`, `OPERATES_IN_BAND`, …), the new schemas put every parameter as a **property field** on the parent system entity. The bundle manifest now declares 3 passes (`radar_domain`, `missile_domain`, `system_links`) and `kind="entities"` everywhere — no typed RF/component edges.

### 1.2 What this broke

After the latest re-ingest:

- `0` vertices of every former subtype (`ANTENNA`, `FREQUENCY_BAND`, `IF_AMPLIFIER`, `MODULATION`, `RECEIVER`, `RF_EMISSION`, `RF_SIGNATURE`, `SCAN_PATTERN`, `SEEKER`, `SIGNAL_PROCESSING_CHAIN`, `SPECIFICATION`, `TRANSMITTER`, `WAVEFORM`).
- `0` edges of `EMITS`, `HAS_ANTENNA`, `HAS_PROCESSING_CHAIN`, `HAS_RECEIVER`, `HAS_SCAN`, `HAS_SEEKER`, `HAS_SIGNATURE`, `HAS_TRANSMITTER`, `OPERATES_IN_BAND`, `RADIATES`, `RECEIVES`, `SPECIFIED_BY`, `USES_MODULATION`, `USES_WAVEFORM`.
- The four starter profiles, which are configured to traverse those types/edges, return `0` results.
- 5 contract tests are carrying `xfail(strict=False)` markers because the canonical Pydantic ontology under `ontology_bundles/air_defense_v3/entities.py` still defines the old nested `RadarSystemEntity` / `MissileSystemEntity` plus orphaned `AntennaEntity`, `RadarPerformanceEntity`, `MissilePerformanceEntity`, `MissilePhysicalCharacteristicsEntity`, `FrequencyBandEntity`, `SpecificationEntity` — none of which are produced by the flat extraction. The drift is real and the extraction schemas have ~60 fields the canonical doesn't know about.

### 1.3 Why bundle the ontology sync with the profile refactor

The profile refactor needs to know "which flat fields belong to which profile section" (e.g., `gain_dbi` → RF Parameters and Performance). That mapping is naturally a property of the field, not the profile. If we declare it on the canonical Pydantic field via `json_schema_extra={"profile_sections": [...]}`, the profile code introspects it and the mapping has one source of truth. Doing the ontology sync as a separate plan would force us to either duplicate the mapping in `query_profiles.py` (drift waiting to happen) or block the profile work until the sync lands. Bundling avoids both.

### 1.4 Decisions log (from brainstorming)

| # | Decision |
|---|---|
| 1 | One spec, two-phase plan: ontology sync first, then profile refactor. Phase 3 (field-level evidence) added per user request. |
| 2 | Section result shape: one result with structured `field_groups`, **not** a list of pseudo-entities. |
| 3 | Field → profile-section mapping lives on the Pydantic field via `json_schema_extra={"profile_sections": [...], "profile_subgroup": "..."}`. |
| 4 | `System Components` becomes property-groups (antenna, booster, seeker, …) **plus** a small `related_systems` block fed from `CHILD_OF` edges. |
| 5 | A field can belong to multiple profile sections (`profile_sections` is a list, not a string). |
| 6 | Field-level evidence is in scope (Phase 3). |
| 7 | Field-level evidence implementation: option 1 — real LLM-extracted source snippets, deterministically resolved to `element_uid` post-LLM. |
| 8 | Dossier composition: single `resolved_root` plus a list of per-section field-group blocks. The root is not duplicated per section. |

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Ontology sync                                         │
│  ────────────────────────                                       │
│  ontology_bundles/air_defense_v3/entities.py                    │
│    + flat fields on RadarSystemEntity / MissileSystemEntity     │
│    + json_schema_extra={"profile_sections":[...],               │
│                          "profile_subgroup":"..."}              │
│    − orphan classes (AntennaEntity, RadarPerformanceEntity, …)  │
│  → 5 xfail'd schema-drift tests pass; remove markers            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Profile refactor                                      │
│  ──────────────────────────                                     │
│  app/services/query_profiles.py                                 │
│    + new kind="section_properties"                              │
│    + _project_field_groups(canonical_cls, instance, section)    │
│    + System Components: get_child_of_systems(node_id)           │
│    + Dossier: single resolved_root + per-section blocks         │
│  app/schemas/query_profiles.py                                  │
│    + QueryProfileFieldGroup, QueryProfileFieldEntry             │
│    + .field_groups, .related_systems on section response        │
│  Frontend                                                        │
│    + <FieldGroupTable> render path; legacy list still works     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Field-level evidence                                  │
│  ──────────────────────────────                                 │
│  docker/docling-graph/app/                                      │
│    + LLM prompt asks for per-field supporting snippets          │
│    + Service post-process resolves snippet → element_uid        │
│    + ExtractionFieldProvenance rows on ExtractPassResponse      │
│  app/services/extraction_merge.py                               │
│    + parse field_provenance, attach to MergedEntityRecord       │
│  app/services/arcadedb_graph.py                                 │
│    + persist _field_evidence JSON on entity vertex              │
│  app/services/query_profiles.py                                 │
│    + surface evidence: list[ChunkExcerpt] on each FieldEntry    │
│  Frontend                                                        │
│    + per-field evidence popover                                  │
└─────────────────────────────────────────────────────────────────┘
```

Phase boundaries are sequential. Phase 2 cannot land before Phase 1 (depends on the canonical entity tags). Phase 3 cannot land before Phase 2 (extends `QueryProfileFieldEntry`). Within each phase, work is independently testable.

---

## 3. Phase 1: Ontology sync

### 3.1 Files touched

- `ontology_bundles/air_defense_v3/entities.py` — `RadarSystemEntity`, `MissileSystemEntity` get all flat fields; orphan classes deleted.
- `tests/unit/test_docs_compliance_contracts.py`, `tests/unit/contracts/test_extraction_schema_contract.py`, `tests/unit/test_coverage_checker.py` — drop the 5 `xfail(strict=False)` markers added during the prior debugging pass.

### 3.2 Field migration

For every non-system field on `extraction_schemas/radar_domain.py:RadarSystemEntity` and `extraction_schemas/missile_domain.py:MissileSystemEntity`:

1. Copy the declaration onto the canonical class (`entities.py`) with the same type and default.
2. Add a meaningful `description` and (where applicable) `examples` — required by the docs-compliance contract.
3. Add `json_schema_extra={"profile_sections": [...], "profile_subgroup": "<group>"}`.

The `confidence` system field on each canonical class is unchanged.

### 3.3 `profile_sections` and `profile_subgroup` mapping

**`RadarSystemEntity` (canonical):**

| Fields | `profile_sections` | `profile_subgroup` |
|---|---|---|
| `nominal_rf_mhz`, `frequency_excursion_mhz`, `nominal_pri_usec`, `nominal_pd_usec`, `inter_pulse`, `pulses_per_dwell`, `dwell_time`, `intra_pulse_mop`, `num_bits_in_code` | `["rf_parameters"]` | `"waveform"` |
| `antenna_dim_az_m`, `antenna_dim_el_m`, `beamwidth_az_deg`, `beamwidth_el_deg`, `gain_dbi`, `antenna_photo`, `spoiled`, `coverage_limits_el_deg` | `["rf_parameters","components"]` | `"antenna"` |
| `tx_peak_power_kw`, `erp_dbw` | `["rf_parameters","performance"]` | `"transmit"` |
| `scan_type`, `scan_period_sec` | `["rf_parameters","performance"]` | `"scan"` |
| `elnot`, `dieqp`, `asrd`, `system_status`, `responsible_agency`, `review_cycle`, `next_review_date` | `[]` | n/a (system metadata) |

**`MissileSystemEntity` (canonical):**

| Fields | `profile_sections` | `profile_subgroup` |
|---|---|---|
| `body_length_m`, `body_diameter_m`, `total_mass_kg`, `missile_photo` | `["components","performance"]` | `"airframe"` |
| `seeker_type` | `["components","performance"]` | `"seeker"` |
| `booster_time_sec`, `booster_thrust`, `booster_mass_kg` | `["components","performance"]` | `"booster"` |
| `sustain_time_sec`, `sustain_thrust`, `sustain_mass_kg` | `["components","performance"]` | `"sustain"` |
| `ejector_time_sec`, `ejector_thrust`, `ejector_mass_kg` | `["components","performance"]` | `"ejector"` |
| `min_intercept_km`, `max_intercept_km`, `min_altitude_km`, `max_altitude_km`, `max_launch_angle_deg` | `["performance"]` | `"engagement"` |
| `average_speed_mps`, `max_speed_mps`, `max_flyout_time_sec`, `flight_time_sec`, `coast_time_sec`, `total_burn_time_sec`, `intra_salvo_time_sec` | `["performance"]` | `"kinematics"` |
| `guidance_type` | `["performance"]` | `"guidance"` |

A field with empty `profile_sections` is "system metadata" — it is not surfaced by any starter profile but remains a real part of the entity (and is still extracted, indexed, and visible on the entity-detail view).

### 3.4 Orphan canonical entities

Delete the following from `entities.py`:

- `AntennaEntity` (line ~944)
- `FrequencyBandEntity` (line ~842)
- `MissilePerformanceEntity` (line ~1101)
- `MissilePhysicalCharacteristicsEntity` (line ~1127)
- `RadarPerformanceEntity` (line ~1176)
- `SpecificationEntity` (line ~1233)

These have no extraction path and no surviving incoming references after Phase 1.2. Grep across the whole tree to confirm and remove dead imports. If any tests reference them, either delete the test (orphan coverage) or repoint to the parent system class.

### 3.5 Drop xfails

Once 3.2–3.4 land, the 5 schema-drift tests pass without their `xfail` markers. Remove each marker in the same commit. If a marker can't be removed, that's a real regression to fix before merging Phase 1.

### 3.6 Phase 1 success criteria

- `pytest tests/unit -q` shows the 5 previously-xfail'd tests passing without markers.
- `check_bundle(ontology_bundles/air_defense_v3)` returns 0 errors.
- A re-ingest of one previously-ingested doc still produces `RADAR_SYSTEM` / `MISSILE_SYSTEM` vertices with all the new fields populated where the LLM had values (no extraction regression).

### 3.7 Risks

- Renaming or repurposing a field on the canonical class while the extraction schema diverges silently. Mitigation: keep the canonical and extraction class field lists 1:1 in this phase; future divergence requires a deliberate spec.
- Deleting an orphan class that turns out to still be imported somewhere obscure (e.g., a Jupyter notebook). Mitigation: full-tree grep before deletion, plus run the unit suite after each removal.

---

## 4. Phase 2: Profile refactor

### 4.1 Files touched

- `app/services/query_profiles.py` — starter definitions, `_fetch_section_items`, `execute_section_search`, `execute_dossier_search`, new helpers.
- `app/schemas/query_profiles.py` — new `QueryProfileFieldGroup`, `QueryProfileFieldEntry`; `QueryProfileSectionResponse.field_groups`, `.related_systems`; `QueryProfileDossierSection`, `QueryProfileDossierResponse` updated per Q7-B.
- `app/services/arcadedb_graph.py` — new `get_child_of_systems(node_id)`, new `get_entity_by_rid(node_id)` (if not already present in usable shape).
- `frontend/src/components/QueryPage.tsx` (or co-located components) — new `<FieldGroupTable>` render path; existing list path retained for legacy traversal profiles.
- `tests/unit/test_query_profiles.py`, `tests/unit/test_dossier_service.py` — rewrite for property-projection paths; new helper tests.

### 4.2 New profile kind

Add `"section_properties"` to `QueryProfileDefinition.kind`. Schema additions:

- `profile_sections: list[str]` — which `json_schema_extra["profile_sections"]` tags this profile pulls.
- `include_child_of: bool = False` — when true, the section response includes `related_systems` populated from `CHILD_OF` edges. Only `system_components` sets it.

`validate_shape` model_validator updated:

- `kind=="section"` requires non-empty `traversals` (existing rule, unchanged).
- `kind=="section_properties"` requires non-empty `profile_sections`.
- `kind=="dossier"` requires non-empty `section_profile_ids` (existing rule, unchanged).

### 4.3 Property-projection helper

```python
def _project_field_groups(
    canonical_cls: type[BaseModel],
    instance_data: dict[str, Any],
    profile_section: str,
) -> list[QueryProfileFieldGroup]:
    """Walk canonical_cls.model_fields, pick fields whose
    json_schema_extra['profile_sections'] contains profile_section,
    group by 'profile_subgroup', and return the populated values.

    - Skips fields where instance_data[field_name] is None.
    - Returns groups in a deterministic order (subgroup name asc;
      within a subgroup, field name asc).
    - Each FieldEntry carries description, examples, enum metadata
      from json_schema_extra so the UI can show tooltips.
    - Fields with empty/missing profile_sections are never included.
    """
```

### 4.4 `_fetch_section_items` branching

```python
async def _fetch_section_items(graph_store, resolved, request, profile):
    if profile.kind == "section_properties":
        instance = await graph_store.get_entity_by_rid(resolved.node_id)
        canonical = _canonical_class_for(resolved.entity_type)
        groups: list[QueryProfileFieldGroup] = []
        for section in profile.profile_sections:
            groups.extend(_project_field_groups(canonical, instance, section))
        return groups   # caller treats this as field_groups, not items
    # existing traversal branch unchanged for kind="section"
```

`execute_section_search` packages the result into `QueryProfileSectionResponse` — populating `field_groups` for `section_properties` profiles, `items` for legacy `section` profiles, and `related_systems` only when `profile.include_child_of`.

### 4.5 `_canonical_class_for` resolver

```python
_CANONICAL_BY_ENTITY_TYPE: dict[str, type[BaseModel]] = {
    "RADAR_SYSTEM": RadarSystemEntity,
    "MISSILE_SYSTEM": MissileSystemEntity,
}

def _canonical_class_for(entity_type: str) -> type[BaseModel]:
    cls = _CANONICAL_BY_ENTITY_TYPE.get(entity_type)
    if cls is None:
        raise ValueError(
            f"No canonical Pydantic class registered for entity_type={entity_type!r}; "
            "field-projection profiles only run against types listed in _CANONICAL_BY_ENTITY_TYPE."
        )
    return cls
```

If a future profile lands on a different entity type (e.g., `INTEGRATED_AIR_DEFENSE_SYSTEM`), register the class here. The error message is explicit so the failure mode is obvious.

### 4.6 `get_child_of_systems`

```python
async def get_child_of_systems(self, node_id: str) -> list[GraphEntityResult]:
    """Return systems linked by CHILD_OF in either direction.

    out('CHILD_OF') → parent platforms; in('CHILD_OF') → variants.
    Returns deduplicated list with direction annotation in
    relationship_types (`["CHILD_OF"]` or `["PARENT_OF"]`).
    """
```

Implemented as a typed-MATCH against `RADAR_SYSTEM` and `MISSILE_SYSTEM` (per the type-required pattern established in the recent ArcadeDB fixes). Resolves the node's `@type` first, builds a typed seed, traverses both directions, returns up to 25 systems.

### 4.7 Dossier composition (per Q7-B)

```python
async def execute_dossier_search(graph_store, db, request) -> QueryProfileDossierResponse:
    profile = _resolve_profile(request.profile_id)              # validated kind=="dossier"
    resolved = await _resolve_root(graph_store, profile, request)
    if resolved is None:
        raise QueryRootNotFoundError(...)

    sections: list[QueryProfileDossierSection] = []
    for section_id in profile.section_profile_ids:
        section_profile = _resolve_profile(section_id)          # kind in ("section","section_properties")
        section_resp = await execute_section_search(
            graph_store, db,
            QueryProfileSearchRequest(profile_id=section_id, query_text=request.query_text, top_k=request.top_k),
            _override_resolved=resolved,                          # avoid re-resolving the root
        )
        sections.append(QueryProfileDossierSection(
            profile_id=section_id,
            profile_label=section_profile.label,
            field_groups=section_resp.field_groups,
            related_systems=section_resp.related_systems,
        ))

    return QueryProfileDossierResponse(
        registry_id=...,
        profile_id=profile.id,
        profile_label=profile.label,
        resolved_root=resolved,
        sections=sections,
        total=sum(len(g.fields) for s in sections for g in s.field_groups),
    )
```

`_override_resolved` is a new internal parameter on `execute_section_search` that lets the dossier path reuse the already-resolved root without re-running entity resolution. Public callers don't pass it.

### 4.8 Starter profile redefinitions

```python
QueryProfileDefinition(
    id="system_rf_parameters", label="System RF Parameters", kind="section_properties",
    description="Frequency, antenna, scan, modulation, and other RF descriptors of the resolved system.",
    root_entity_types=["RADAR_SYSTEM", "MISSILE_SYSTEM"],
    profile_sections=["rf_parameters"],
    placeholder_query="e.g. Fan Song",
),
QueryProfileDefinition(
    id="system_components", label="System Components", kind="section_properties",
    description="Antenna, propulsion, seeker, ejector, body, and other physical components of the resolved system.",
    root_entity_types=["RADAR_SYSTEM", "MISSILE_SYSTEM"],
    profile_sections=["components"],
    include_child_of=True,
    placeholder_query="e.g. SA-2",
),
QueryProfileDefinition(
    id="system_performance", label="System Performance", kind="section_properties",
    description="Engagement envelope, kinematics, transmit power, and propulsion timing for the resolved system.",
    root_entity_types=["RADAR_SYSTEM", "MISSILE_SYSTEM"],
    profile_sections=["performance"],
    placeholder_query="e.g. SA-2",
),
QueryProfileDefinition(
    id="system_dossier", label="System Dossier", kind="dossier",
    description="Composite report of RF parameters, components, and performance for the resolved system.",
    root_entity_types=["RADAR_SYSTEM", "MISSILE_SYSTEM"],
    section_profile_ids=["system_rf_parameters", "system_components", "system_performance"],
    placeholder_query="e.g. SA-2",
),
```

The legacy traversal-based starter profiles (and any registry-overridden user profiles of `kind="section"`) keep working unchanged.

### 4.9 Frontend

`QueryPage.tsx` (or wherever section results render) gets a render branch:

- If response carries `field_groups` non-empty → render `<FieldGroupTable>`: stacked collapsible cards keyed by `subgroup_label`, each with a property table (`label : value`, value formatted by type, description as tooltip). When the profile has `related_systems` non-empty (Components only), render a chip row above the field groups linking each related system back into the same profile search.
- Else if response carries `items` non-empty → existing result-card list (legacy).
- Else → "No results" empty state with the profile-specific placeholder hint.

Dossier rendering is the per-section block list under one entity header, each block calling the same `<FieldGroupTable>` component.

### 4.10 Phase 2 success criteria

- `/v1/query-profiles/search/section` for each starter profile against the running ArcadeDB returns non-empty `field_groups` for at least one of: `SA-2`, `Fan Song`, `Engagement and Fire Control Radar`.
- `/v1/query-profiles/search/dossier` returns one `resolved_root` and 3 populated `sections`.
- The legacy `kind="section"` profile path still produces `items` correctly; the new branch is purely additive on the response and inert on legacy profiles.
- Unit suite green; no xfail regressions.

### 4.11 Risks

- A canonical field that was meant to belong to a profile but was tagged `[]` by mistake silently disappears from the UI. Mitigation: a contract test that asserts every domain field on `RadarSystemEntity` / `MissileSystemEntity` either belongs to ≥1 profile section or is explicitly tagged `system_metadata=True`.
- `_canonical_class_for` doesn't know a new entity type and the section endpoint 500s. Mitigation: explicit error message; profile registry validation catches the unknown type at registration time.
- Frontend renders a giant flat property table when an entity has 30+ populated fields. Mitigation: subgroup-level collapse defaults to expanded for the first group and collapsed thereafter; user can override.

---

## 5. Phase 3: Field-level evidence

### 5.1 Why snippet, not chunk_id

The LLM is asked for **the supporting text snippet** for each field value, not for an opaque chunk identifier. LLMs are reliable at quoting source text and unreliable at remembering opaque IDs. Post-extraction, a deterministic substring matcher resolves snippet → chunk_id. Three benefits:

1. Independently verifiable by a human reading the snippet.
2. No risk of hallucinated IDs.
3. The snippet is the citation we want to display anyway — chunk_id is plumbing.

### 5.2 Files touched

- `docker/docling-graph/app/schemas.py` — new `ExtractionFieldProvenance`; `ExtractPassResponse` gets `field_provenance: list[ExtractionFieldProvenance]`.
- `docker/docling-graph/app/main.py` (or wherever the extraction prompt + structured-output schema is composed) — extend output schema and prompt with per-field source-snippet requirement.
- `docker/docling-graph/app/prompt_rules.py` — add a "field provenance" instruction block.
- `docker/docling-graph/app/provenance.py` — extend the existing post-LLM provenance pass to also resolve `ExtractionFieldProvenance` snippets to `element_uid`.
- `app/services/extraction_merge.py` — parse `field_provenance` from `ExtractPassResponse`, attach to `MergedEntityRecord`, dedup on `(instance_id, field_name)`.
- `app/services/arcadedb_graph.py` — `upsert_nodes_batch_sync` writes `_field_evidence: dict[field_name, list[{chunk_id, snippet, element_uid}]]` as a JSON property on the entity vertex.
- `app/services/query_profiles.py` — `_project_field_groups` reads `_field_evidence`, fills `QueryProfileFieldEntry.evidence`.
- `app/schemas/query_profiles.py` — `QueryProfileFieldEntry.evidence: list[ChunkExcerpt] = []`.
- Frontend — per-field evidence popover in `<FieldGroupTable>`.

### 5.3 Wire schema

```python
# docker/docling-graph/app/schemas.py
class ExtractionFieldProvenance(BaseModel):
    instance_id: str                # joins ExtractionProvenance.instance_id
    field_name: str                 # canonical field name on the entity model
    value: Any                      # the value the LLM extracted (sanity check)
    supporting_snippet: str         # exact-quoted text from the source the LLM used
    element_uid: str | None = None  # filled by post-process; None if no chunk match

class ExtractPassResponse(BaseModel):
    ...
    provenance: list[ExtractionProvenance] = ...        # entity-level, existing
    field_provenance: list[ExtractionFieldProvenance] = ...  # NEW
```

### 5.4 Prompt and structured-output changes

The structured-output JSON schema for each entity gains an additional `_field_provenance` map: `dict[str, str]` from canonical field name to supporting snippet. The system prompt adds:

> For every field you populate on an extracted entity, also include in `_field_provenance` the exact verbatim snippet from the source text that established that value. The snippet must appear verbatim somewhere in the chunks provided. Do not paraphrase or summarize. If you cannot quote a source for a field, omit that field rather than guess.

Empty `_field_provenance` for an entity is allowed (e.g., entities derived from headings); fields without an entry get no field-evidence row but are not dropped.

The service unpacks each entity's `_field_provenance` into `ExtractionFieldProvenance` rows, one per field, with `instance_id` set to the entity's instance id (same id used by `ExtractionProvenance`).

### 5.5 Snippet → element_uid resolver

In `docker/docling-graph/app/provenance.py` post-process:

```python
def resolve_field_provenance_uids(
    field_provenance: list[ExtractionFieldProvenance],
    input_chunks: list[InputChunk],   # what the service fed the LLM
) -> None:
    """For each row, set element_uid by substring match against
    input_chunks[*].text. Whitespace-collapsed, case-insensitive.
    Multiple matches: pick the chunk with the longest unique-prefix
    match. No match: leave element_uid=None. Mutates rows in place."""
```

The service already tracks the LLM's input chunks for `ExtractionProvenance`; this reuses that surface.

### 5.6 Worker-side merge

`app/services/extraction_merge.py` parses `field_provenance` alongside the existing `provenance` parsing. Each `MergedEntityRecord` gains:

```python
@dataclass
class MergedEntityRecord:
    ...
    field_evidence: dict[str, list[FieldEvidenceRow]] = field(default_factory=dict)

@dataclass
class FieldEvidenceRow:
    chunk_id: str
    snippet: str
    element_uid: str | None
```

`chunk_id` is resolved from `element_uid` via the existing chunk lookup (the same path that builds `EXTRACTED_FROM` edges). Rows where `element_uid` is None get `chunk_id=None` and the row still ships — the snippet alone is useful.

### 5.7 Persistence

`upsert_nodes_batch_sync` adds a `_field_evidence` JSON property to the upsert payload. ArcadeDB stores it as a generic JSON value:

```jsonc
{
  "name": "SA-2",
  "max_speed_mps": 1100,
  "_field_evidence": {
    "max_speed_mps": [
      {"chunk_id": "af701ee3-...", "snippet": "maximum speed of 1100 m/s", "element_uid": "#/texts/14"}
    ]
  }
}
```

On re-ingest the JSON is replaced wholesale (the merger's per-field provenance reflects the union across this run's passes; we don't carry forward stale rows from a previous ingest).

### 5.8 Section-endpoint surfacing

`_project_field_groups` reads `instance_data["_field_evidence"][field_name]` and converts each row to a `ChunkExcerpt` using the same `_lookup_chunk_by_type` helper that retrieval uses. Old data: `_field_evidence` missing → `evidence=[]`. The UI renders an empty cell with "no per-field evidence" tooltip.

### 5.9 Frontend

The `<FieldGroupTable>` row gets a small "evidence" affordance — an icon button that opens a popover listing each `(snippet, chunk_id)` pair with a deep link into the document viewer at the matching element. Empty cell when `evidence: []`.

### 5.10 Migration

No DB schema migration. `_field_evidence` is a JSON property; default missing on old vertices. Re-ingest is the migration path. Recommended (not required) one-time re-ingest of all corpora after Phase 3 lands.

### 5.11 LLM cost / latency impact

Per-field snippets enlarge each entity's output by roughly the number of populated fields × average snippet length. For the densest case (a fully-populated `RadarSystemEntity` with ~30 fields, ~30-token snippets) this adds ~900 output tokens per entity. Empirically order-of-percent on extraction cost; no schedule impact.

### 5.12 Phase 3 success criteria

- A re-ingested test doc has populated `_field_evidence` on at least one `MISSILE_SYSTEM` or `RADAR_SYSTEM` vertex for ≥3 fields.
- `/v1/query-profiles/search/section` returns `evidence` populated for those fields in `field_groups[*].fields[*]`.
- A snippet that doesn't substring-match any input chunk results in `element_uid=None` and `chunk_id=None` on the row — the snippet still surfaces; nothing is fabricated.
- Old (pre-Phase-3) data continues to render with empty per-field evidence and no errors.
- Re-running the unit suite stays green.

### 5.13 Risks

- LLM ignores or paraphrases snippets despite the prompt. Mitigation: a service-side validator drops snippets that don't substring-match any input chunk, logging the field; this prevents fabricated quotes.
- Output size growth degrades extraction throughput. Mitigation: budget tracked in Phase 3 acceptance test; if over 5%, prompt is split into "structured fields first, then provenance" to avoid LLM wandering.
- Snippet collisions across chunks (same text appears in multiple chunks). Mitigation: longest-unique-prefix tiebreaker; if still ambiguous, attach all candidate `element_uid`s as a list.

---

## 6. API & data shapes (consolidated)

### 6.1 Section endpoint — `POST /v1/query-profiles/search/section`

Request unchanged.

```python
class QueryProfileFieldEntry(APIModel):
    name: str
    label: str
    value: Any
    description: str | None = None
    examples: list[Any] | None = None
    enum: list[str] | None = None
    evidence: list[ChunkExcerpt] = []      # Phase 3 — empty until re-ingest

class QueryProfileFieldGroup(APIModel):
    subgroup: str | None = None
    subgroup_label: str | None = None
    fields: list[QueryProfileFieldEntry]

class QueryProfileSectionResponse(APIModel):
    registry_id: uuid.UUID
    profile_id: str
    profile_label: str
    resolved_root: GraphEntityResult
    field_groups: list[QueryProfileFieldGroup] = []     # Phase 2
    related_systems: list[GraphEntityResult] = []       # Phase 2 — Components only
    items: list[GraphEntityResult] = []                  # legacy traversal profiles
    total: int
```

### 6.2 Dossier endpoint — `POST /v1/query-profiles/search/dossier`

Request unchanged.

```python
class QueryProfileDossierSection(APIModel):
    profile_id: str
    profile_label: str
    field_groups: list[QueryProfileFieldGroup] = []
    related_systems: list[GraphEntityResult] = []

class QueryProfileDossierResponse(APIModel):
    registry_id: uuid.UUID
    profile_id: str
    profile_label: str
    resolved_root: GraphEntityResult        # single root, per Q7-B
    sections: list[QueryProfileDossierSection]
    total: int
```

### 6.3 Profile registry — `GET /v1/query-profiles`

```python
class QueryProfileDefinition(APIModel):
    id: str
    label: str
    description: str | None = None
    kind: Literal["section", "section_properties", "dossier"] = "section"
    exposed: bool = True
    root_entity_types: list[str] = []
    target_entity_types: list[str] = []      # only meaningful for kind=section
    traversals: list[QueryProfileTraversal] = []   # only meaningful for kind=section
    profile_sections: list[str] = []          # NEW — only for kind=section_properties
    include_child_of: bool = False            # NEW — Components only
    section_profile_ids: list[str] = []       # only meaningful for kind=dossier
    placeholder_query: str | None = None
```

### 6.4 docling-graph wire (Phase 3)

`ExtractPassResponse.field_provenance` additive; existing consumers ignore unknown fields.

### 6.5 Entity vertex (Phase 3)

`_field_evidence: dict[str, list[FieldEvidenceRow]]` JSON property on `RADAR_SYSTEM` and `MISSILE_SYSTEM` vertices. Missing on pre-Phase-3 ingests.

---

## 7. Migration & backward compatibility

- **Phase 1:** ontology-only; no data migration. Tests gate.
- **Phase 2:** API is purely additive — `field_groups`, `related_systems` default `[]`. Legacy `kind="section"` profiles unaffected. Frontend renders both branches. **Phase 1+2 deliver the visible fix without a re-ingest** because the flat extraction has been writing the field values onto the entity vertices all along; the canonical entity definitions just didn't recognize them. Sections become populated as soon as Phase 1+2 ship.
- **Phase 3:** wire and storage are additive. Old vertices have no `_field_evidence`; UI shows empty evidence cells with a tooltip explaining "no per-field evidence; re-ingest to populate." A one-time re-ingest of all corpora after Phase 3 lands populates evidence; not required.

---

## 8. Testing

| Phase | Unit | Integration | Contract |
|---|---|---|---|
| 1 | Field-level introspection: every domain field has either ≥1 `profile_sections` tag or `system_metadata=True`. Bundle checker contract. | None (ontology-only). | 5 xfail'd schema-drift tests pass; remove markers. |
| 2 | `_project_field_groups` table-driven; `_canonical_class_for` resolution; `validate_shape` for `kind="section_properties"`; `_fetch_section_items` branching. | `/search/section` for each starter profile against running ArcadeDB returns non-empty `field_groups` on a known SA-2 / Fan Song; `/search/dossier` returns one root + 3 section blocks. | New: `kind="section_properties"` profile shape requires `profile_sections` non-empty. |
| 3 | docling-graph `_field_provenance` parser; snippet→element_uid resolver; `MergedEntityRecord.field_evidence` union; upsert serialization round-trip; `_project_field_groups` evidence pass-through. | End-to-end re-ingest of one doc → `_field_evidence` populated; `/search/section` surfaces ≥1 per-field evidence row. | Field-evidence rows joined to chunks 1:1; missing snippet → `element_uid=None, chunk_id=None`, no fabrication. |

---

## 9. Out of scope

- Reintroducing nested ANTENNA / RECEIVER / etc. entities. The flat-checklist refactor is final for this spec.
- Any change to retrieval (text/image vector search, hybrid, global). Those endpoints are unaffected.
- Re-extraction of community reports. (Phase 3's per-field evidence is for entity properties, not for community-report content.)
- Authoring a UI for editing `profile_sections` tags directly in the registry. Tags live in code (Pydantic field declarations) — they're an ontology-engineering concern, not a runtime registry concern.
- New profiles beyond the four listed in §1. Adding more profiles after this spec is a follow-on.

---

## 10. Open questions

None expected after the brainstorming pass — all design decisions were captured in §1.4. If reviewers surface new ones during the spec-review loop, we add them here and revise.
