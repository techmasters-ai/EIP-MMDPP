# Flat-Schema Profile Refactor — Design

**Date:** 2026-04-25
**Status:** Revised after first review pass — 8 findings addressed (see §11)
**Scope:** Bring the four starter query profiles (System Dossier, System Components, System RF Parameters, System Performance) onto the flat-checklist extraction schema, and sync the canonical ontology entities so the schema-drift xfails clear at the same time.

---

## 1. Background

### 1.1 What changed under us

The Phase 5/6/7 Pydantic ontology SSoT work refactored the radar/missile extraction schemas to a **flat checklist** model: instead of emitting nested `ANTENNA`, `RECEIVER`, `BOOSTER`, `SEEKER`, `SPECIFICATION` etc. entities and connecting them to `RADAR_SYSTEM` / `MISSILE_SYSTEM` via typed edges (`HAS_ANTENNA`, `EMITS`, `OPERATES_IN_BAND`, …), the new schemas put every parameter as a **property field** on the parent system entity. The bundle manifest now declares 3 passes — `radar_domain` (`kind="entities"`), `missile_domain` (`kind="entities"`), and `system_links` (`kind="relationships_only"`, `extracted_relationship_types: [ASSOCIATED_WITH, CUES]`). No typed RF/component edges; only the two coarse system-to-system edges from `system_links`.

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
| 4 | `System Components` becomes property-groups (antenna, booster, seeker, …) **plus** a small `related_systems` block fed from `ASSOCIATED_WITH` edges. (Original draft said `CHILD_OF`; review pointed out `CHILD_OF` is declared `SECTION → SECTION` in `validation_matrix.py:152` and `system_links` only emits `ASSOCIATED_WITH` and `CUES` between systems.) |
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
│  ontology_bundles/.../extraction_schemas/                       │
│    + RadarDomainPass / MissileDomainPass gain                   │
│      field_provenance: list[FieldProvenanceRow]                 │
│      (top-level on the pass template — survives extra="ignore") │
│  docker/docling-graph/app/                                      │
│    + LLM prompt asks for per-field supporting snippets          │
│    + Service post-process resolves snippet → element_uid        │
│    + ExtractionFieldProvenance rows on ExtractPassResponse      │
│  app/services/extraction_merge.py                               │
│    + parse field_provenance, attach to MergedEntityRecord       │
│  app/services/arcadedb_graph.py                                 │
│    + persist _field_evidence JSON on entity vertex              │
│  app/services/query_profiles.py                                 │
│    + surface evidence: list[GraphEvidenceItem] on each FieldEntry    │
│  Frontend                                                        │
│    + per-field evidence popover                                  │
└─────────────────────────────────────────────────────────────────┘
```

Phase boundaries are sequential. Phase 2 cannot land before Phase 1 (depends on the canonical entity tags). Phase 3 cannot land before Phase 2 (extends `QueryProfileFieldEntry`). Within each phase, work is independently testable.

---

## 3. Phase 1: Ontology sync

### 3.1 Files touched

Phase 1 is broader than the original draft. The orphan canonical classes are referenced by enough sibling files that a partial deletion leaves an incoherent bundle and `check_bundle()` will fail. Treat these as one atomic change set:

- `ontology_bundles/air_defense_v3/entities.py`
  - **Add** flat fields onto `RadarSystemEntity` (lines 411-…) and `MissileSystemEntity` (lines 568-…) per §3.2/§3.3.
  - **Delete** the now-unused `edge(...)` declarations on `RadarSystemEntity` and `MissileSystemEntity` (e.g., `waveforms`, `rf_emissions`, `antennas`, `receivers`, `transmitters`, `frequency_bands`, `modulations`, `scan_patterns`, `signal_processing_chains`, `if_amplifiers`, `seekers`, `specifications`, `propulsion_*`, `radar_performance`, `missile_performance`, `missile_physical_characteristics`, `capabilities`, etc.) — see §3.4 for the full deletion list.
  - **Delete** the orphan canonical classes themselves (the standalone `AntennaEntity`, `ReceiverEntity`, `WaveformEntity`, etc.) — full list in §3.4.
  - **Remove** the deleted classes from the `ENTITY_TYPES` registry at `entities.py:1346`. Anything that's no longer importable must not appear in the dict.
- `ontology_bundles/air_defense_v3/coverage.yaml` — remove the orphan entity types from the section coverage lists; remove the orphan `HAS_*`, `EMITS`, `RADIATES`, `RECEIVES`, etc. relationship types from the relationship coverage lists.
- `ontology_bundles/air_defense_v3/validation_matrix.py` — drop every tuple whose subject or object is an orphan type. The `ASSERTION` rows are already TODO'd as dead per the file's own comment; remove them in the same pass.
- `ontology_bundles/air_defense_v3/relationships.py` — delete `RelationshipType` members for the dropped edge labels (`HAS_ANTENNA`, `HAS_RECEIVER`, `HAS_TRANSMITTER`, `EMITS`, `RADIATES`, `RECEIVES`, `OPERATES_IN_BAND`, `USES_WAVEFORM`, `USES_MODULATION`, `SPECIFIED_BY`, `HAS_SCAN`, `HAS_SEEKER`, `HAS_SIGNATURE`, `HAS_PROCESSING_CHAIN`, …). Keep `IS_A`, `PART_OF`, `INSTALLED_ON`, `ASSOCIATED_WITH`, `CUES`, `CHILD_OF`, the document/structure edges, and any other relationship still emitted by current passes.
- `app/services/dossier_service.py` — delete the stale `RF_ENTITY_TYPES`, `PERFORMANCE_ENTITY_TYPES`, `COMPONENT_ENTITY_TYPES` constants (around lines 38-65). They're hard-coded lists of the now-deleted entity types and are dead code after Phase 1. Any caller that imports them is repointed to read `profile_sections` introspectively (the same way `_project_field_groups` will, in Phase 2).
- `tests/unit/test_docs_compliance_contracts.py`, `tests/unit/contracts/test_extraction_schema_contract.py`, `tests/unit/test_coverage_checker.py` — drop the 5 `xfail(strict=False)` markers from the prior debugging pass.

After the first `check_bundle()` run lands a green result on the modified bundle, the Phase 1 file scope is verified. Anything that still references a deleted symbol surfaces as an `ImportError` or a `check_bundle()` failure; both are deterministic gates we can iterate on.

### 3.2 Field migration

For every non-system field on `extraction_schemas/radar_domain.py:RadarSystemEntity` and `extraction_schemas/missile_domain.py:MissileSystemEntity`:

1. Copy the declaration onto the canonical class (`entities.py`) with the same type and default.
2. Add a meaningful `description` and (where applicable) `examples` — required by the docs-compliance contract.
3. Add `json_schema_extra={"profile_sections": [...], "profile_subgroup": "<group>"}`.

The `confidence` system field on each canonical class is unchanged.

### 3.3 `profile_sections` and `profile_subgroup` mapping

**Convention:** every non-system field on the canonical class falls into one of three buckets, and the contract is enforced by a Phase 1 unit test (§3.6 success criteria):

1. **Profile-mapped** — `profile_sections: list[str]` non-empty, `profile_subgroup: str` set.
2. **System metadata** — `profile_sections: []` AND `system_metadata: True` (new flag in `json_schema_extra`). The field is real and indexed but never surfaced by a starter profile (e.g., audit trails like `responsible_agency`, identity adjuncts like `nomenclature`, internal classifiers like `dieqp`).
3. **System field** (the existing `system_field: True` marker) — bookkeeping like `confidence`, `extraction_confidence`. Unchanged.

A field with `profile_sections=[]` AND no `system_metadata` flag is a bug — the contract test fails the build.

**`RadarSystemEntity` (canonical) full mapping:**

| Fields | `profile_sections` | `profile_subgroup` |
|---|---|---|
| `nominal_rf_mhz`, `frequency_excursion_mhz`, `nominal_pri_usec`, `nominal_pd_usec`, `inter_pulse`, `pulses_per_dwell`, `dwell_time`, `intra_pulse_mop`, `num_bits_in_code` | `["rf_parameters"]` | `"waveform"` |
| `antenna_dim_az_m`, `antenna_dim_el_m`, `beamwidth_az_deg`, `beamwidth_el_deg`, `gain_dbi`, `antenna_photo`, `spoiled`, `coverage_limits_el_deg` | `["rf_parameters","components"]` | `"antenna"` |
| `tx_peak_power_kw`, `erp_dbw` | `["rf_parameters","performance"]` | `"transmit"` |
| `scan_type`, `scan_period_sec` | `["rf_parameters","performance"]` | `"scan"` |
| `emitter_function` | `["rf_parameters"]` | `"classification"` |
| `nomenclature` | `[]`, `system_metadata=True` | n/a (identity adjunct, surfaced on entity header) |
| `elnot`, `dieqp`, `asrd`, `system_status`, `responsible_agency`, `review_cycle`, `next_review_date` | `[]`, `system_metadata=True` | n/a |

**`MissileSystemEntity` (canonical) full mapping:**

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
| `emitter_function` | `["performance"]` | `"classification"` |
| `nomenclature` | `[]`, `system_metadata=True` | n/a (identity adjunct, surfaced on entity header) |
| `asrd` | `[]`, `system_metadata=True` | n/a |

**Identity-adjunct surfacing.** `nomenclature` is not part of any profile section but is high-value identity context (formal designator vs. NATO common name). The entity header on the section/dossier UI renders the resolved root's `nomenclature` next to the system name when populated. No special schema work — `nomenclature` rides along on `resolved_root.properties` like any other field.

### 3.4 Orphan canonical entities — full audit

The original draft listed 6 classes. Reviewer (correctly) flagged that this is too narrow: the broader set of subtypes registered in `ENTITY_TYPES` at `entities.py:1346` are also orphaned by the flat-checklist refactor and have to be removed coherently for `check_bundle()` to pass.

**Full deletion set** (every class registered in `ENTITY_TYPES` that has no extraction path in `radar_domain.py`, `missile_domain.py`, or `system_links.py`, and is not a structural anchor or top-level system type):

`FrequencyBandEntity`, `ModulationEntity`, `RfSignatureEntity`, `RfEmissionEntity`, `WaveformEntity`, `ScanPatternEntity`, `AntennaEntity`, `TransmitterEntity`, `ReceiverEntity`, `IfAmplifierEntity`, `SignalProcessingChainEntity`, `GuidanceMethodEntity`, `SeekerEntity`, `MissilePerformanceEntity`, `MissilePhysicalCharacteristicsEntity`, `PropulsionStackEntity`, `PropulsionStageEntity`, `CapabilityEntity`, `RadarPerformanceEntity`, `EngagementTimelineEntity`, `ForceStructureEntity`, `AssemblyEntity`, `SpecificationEntity`, `StandardEntity`, `ProcedureEntity`, `FailureModeEntity`, `TestEventEntity`.

**Retained classes** (used by the structural / system layer, not orphaned):

`DocumentEntity`, `SectionEntity`, `FigureEntity`, `TableEntity`, `ImageEntity`, `TextBlockEntity`, `OrganizationEntity`, `PlatformEntity`, `WeaponSystemEntity`, `EquipmentSystemEntity`, `SubsystemEntity`, `ComponentEntity`, `RadarSystemEntity`, `MissileSystemEntity`, `AirDefenseArtillerySystemEntity`, `ElectronicWarfareSystemEntity`, `FireControlSystemEntity`, `IntegratedAirDefenseSystemEntity`, `LauncherSystemEntity`, `Alias`.

**Audit gate:** before deletion, grep each candidate name across the entire repo (`grep -rn "AntennaEntity\|ANTENNA\b" --include='*.py' --include='*.yaml' --include='*.md'`) and inspect each hit. Anything that's a test fixture / notebook reference / orphaned import gets cleaned up in the same commit; anything that turns out to be a live consumer (extraction path I missed, fixtures we still want to keep) is escalated and the deletion is reconsidered for that specific class.

**Validation matrix + relationship enum:** every tuple in `validation_matrix.py` whose subject or object is in the deletion set goes away, and every relationship label that only occurred between deleted classes (`HAS_ANTENNA`, `EMITS`, `RADIATES`, `RECEIVES`, `OPERATES_IN_BAND`, `USES_WAVEFORM`, `USES_MODULATION`, `SPECIFIED_BY`, `HAS_SCAN`, `HAS_SEEKER`, `HAS_SIGNATURE`, `HAS_PROCESSING_CHAIN`, `HAS_RECEIVER`, `HAS_TRANSMITTER`, `HAS_IF_AMPLIFIER`, `HAS_PROPULSION`, `HAS_GUIDANCE`, `MEASURES`, `MANUFACTURED_BY` if no longer reachable, …) is removed from `RelationshipType` in `relationships.py` and from `coverage.yaml`. The `system_links` pass continues to emit `ASSOCIATED_WITH` and `CUES`; those stay.

**Edge fields on retained entities:** `RadarSystemEntity` and `MissileSystemEntity` currently declare ~20 `edge(...)` fields each pointing at the deleted classes (e.g., `waveforms: List["WaveformEntity"]`). All such fields are deleted. The retained edge fields on these classes are: `documents` (`DERIVED_FROM` → DocumentEntity), `organizations` (`MANUFACTURED_BY` → OrganizationEntity, if kept), `platform` (`INSTALLED_ON` → PlatformEntity), and any structural edges that survive the relationship-type audit.

### 3.5 Drop xfails

Once 3.2–3.4 land, the 5 schema-drift tests pass without their `xfail` markers. Remove each marker in the same commit. If a marker can't be removed, that's a real regression to fix before merging Phase 1.

### 3.6 Phase 1 success criteria

- `pytest tests/unit -q` shows the 5 previously-xfail'd tests passing without markers.
- New contract test: every domain field on `RadarSystemEntity` and `MissileSystemEntity` either has non-empty `profile_sections` OR carries `system_metadata=True` (no field falls through the cracks).
- `check_bundle(ontology_bundles/air_defense_v3)` returns 0 errors.
- `python -c "from ontology_bundles.air_defense_v3 import entities, relationships, validation_matrix"` succeeds with no `ImportError` (i.e., we deleted everything we said we'd delete and nothing references a deleted name).
- A re-ingest of one previously-ingested doc still produces `RADAR_SYSTEM` / `MISSILE_SYSTEM` vertices with all the new fields populated where the LLM had values (no extraction regression).

### 3.7 Risks

- Renaming or repurposing a field on the canonical class while the extraction schema diverges silently. Mitigation: keep the canonical and extraction class field lists 1:1 in this phase; future divergence requires a deliberate spec.
- Deleting an orphan class that turns out to still be imported somewhere obscure (e.g., a Jupyter notebook). Mitigation: full-tree grep before deletion, plus run the unit suite after each removal.

---

## 4. Phase 2: Profile refactor

### 4.1 Files touched

- `app/services/query_profiles.py` — starter definitions, `_fetch_section_items`, `execute_section_search`, `execute_dossier_search`, new helpers.
- `app/schemas/query_profiles.py` — new `QueryProfileFieldGroup`, `QueryProfileFieldEntry`; `QueryProfileSectionResponse.field_groups`, `.related_systems`; `QueryProfileDossierSection`, `QueryProfileDossierResponse` updated per Q7-B.
- `app/services/arcadedb_graph.py` — new `get_associated_systems(node_id)`, new `get_entity_by_rid(node_id)` (if not already present in usable shape).
- `frontend/src/api/client.ts` — extend `QueryProfileDefinition.kind` literal from `"section" | "dossier"` (line ~520) to `"section" | "section_properties" | "dossier"`; add new TS interfaces for `QueryProfileFieldGroup`, `QueryProfileFieldEntry`, `QueryProfileDossierSection`; update `QueryProfileSectionResponse` and `QueryProfileDossierResponse` interfaces to expose `field_groups`, `related_systems`, and the new dossier shape (one root + per-section blocks).
- `frontend/src/components/QueryPage.tsx` (around the `selectedIsGraphProfile` branch at ~line 827) — split the result-rendering switch on `profile.kind`: `"section"` keeps the existing `items`-flattening render; `"section_properties"` calls a new `<FieldGroupTable>` render path; `"dossier"` calls a new `<DossierSectionList>` that renders one entity header + N stacked field-group cards. The existing `setResults` / `setTotalResults` plumbing is generalized: `setSectionResponse` / `setDossierResponse` carry the typed payloads.
- `frontend/src/components/FieldGroupTable.tsx` — new component. Stacked collapsible cards keyed by `subgroup_label`; each card renders a property table with `label : value` rows; canonical `description` shown on hover; per-field `evidence` (Phase 3) renders a small chip that opens an evidence popover.
- `frontend/src/components/FieldEvidencePopover.tsx` — new component (Phase 3). Lists each `(snippet, chunk_id)` pair with a deep link to the document viewer at the matching `element_uid`.
- `tests/unit/test_query_profiles.py`, `tests/unit/test_dossier_service.py` — rewrite for property-projection paths; new helper tests.

### 4.2 New profile kind

Add `"section_properties"` to `QueryProfileDefinition.kind`. Schema additions:

- `profile_sections: list[str]` — which `json_schema_extra["profile_sections"]` tags this profile pulls.
- `include_associated_systems: bool = False` — when true, the section response includes `related_systems` populated from `ASSOCIATED_WITH` / `CUES` edges. Only `system_components` sets it. (Renamed from `include_child_of` per review.)

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

`execute_section_search` packages the result into `QueryProfileSectionResponse` — populating `field_groups` for `section_properties` profiles, `items` for legacy `section` profiles, and `related_systems` only when `profile.include_associated_systems`.

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

### 4.6 `get_associated_systems`

(Renamed from the original draft's `get_child_of_systems`. Reviewer correctly noted that `CHILD_OF` is declared `SECTION → SECTION` in `validation_matrix.py:152` — it does not connect systems. The relationship `system_links` actually emits between systems is `ASSOCIATED_WITH` (and `CUES`).)

```python
async def get_associated_systems(self, node_id: str) -> list[GraphEntityResult]:
    """Return systems linked by ASSOCIATED_WITH or CUES in either direction.

    Used by the System Components profile's `related_systems` block.
    Resolves @type for typed MATCH, traverses bothE() across the two
    relevant edge labels, deduplicates by RID, returns up to 25.
    Direction is annotated on each result via `relationship_types`,
    e.g. `["ASSOCIATED_WITH"]` or `["CUES_IN"]` / `["CUES_OUT"]`.
    """
```

Why not add a new `RELATED_TO` / `CHILD_OF` between systems? Because:

1. The relationship-extraction prompt for `system_links` would have to be redesigned to produce it, which lands a moving target on top of the refactor.
2. `ASSOCIATED_WITH` already captures the "this radar pairs with that missile" semantics the user actually wants from the Components panel (e.g., Fan Song ↔ SA-2). It's coarser than a typed `CHILD_OF`, but it's real data. We can refine post-Phase-2 if it's not enough.

Implementation pattern matches the recent typed-MATCH fixes (`get_ontology_linked_chunks`, `get_relationships_between_entities`): resolve the seed's `@type` with a quick `SELECT @type FROM <rid>`, then build a typed-seed MATCH; ArcadeDB MATCH first-node-without-`type:` throws `UnsupportedOperationException`.

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
    include_associated_systems=True,
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

Concrete touch points (all in `frontend/src/`):

- **`api/client.ts:520`** — extend the `QueryProfileDefinition.kind` literal type from `"section" | "dossier"` to include `"section_properties"`. Without this, TypeScript rejects every section-properties profile shape on receipt.
- **`api/client.ts`** — add interfaces for `QueryProfileFieldEntry`, `QueryProfileFieldGroup`, `QueryProfileDossierSection`. Update `QueryProfileSectionResponse` to add optional `field_groups: QueryProfileFieldGroup[]` and `related_systems: GraphEntityResult[]`. Update `QueryProfileDossierResponse` to the new single-root + per-section-blocks shape (per Q7-B).
- **`components/QueryPage.tsx`** — at the `selectedIsGraphProfile` branch (~line 827), the response handler currently flattens dossier results through `items`. Replace with a `kind`-typed switch:
  - `kind === "section"` → existing `items`-flattening (legacy).
  - `kind === "section_properties"` → render `<FieldGroupTable>` for `field_groups`; if `related_systems` non-empty, render a chip row above.
  - `kind === "dossier"` → render `<DossierSectionList>`: one entity header + a stacked list of field-group blocks per section.
  - The result-state hook (`setResults` / `setTotalResults`) is generalized to a `result: SectionPayload | DossierPayload | LegacyItemsPayload` discriminated union.
- **`components/FieldGroupTable.tsx`** *(new)* — stacked collapsible cards keyed by `subgroup_label`. Property table rows show `label : value`; canonical `description` is the tooltip; canonical `examples` show as a placeholder for empty rows. The first subgroup card defaults expanded; the rest collapsed.
- **`components/FieldEvidencePopover.tsx`** *(new, Phase 3)* — per-field evidence chip; on click opens a popover listing each `(snippet, chunk_id, element_uid)` row, with a "Open in document viewer" deep link.
- **`components/DossierSectionList.tsx`** *(new)* — one entity-header card (using `resolved_root.name` + `nomenclature` if populated, plus type chip), then N stacked section blocks each containing a `<FieldGroupTable>`. Empty sections render with a "no data extracted" placeholder rather than disappearing.

Phase 1+2 land all of the Phase-2-tagged components above. Phase 3 only adds `<FieldEvidencePopover>` and a per-row chip in `<FieldGroupTable>` — no further structural change.

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

### 5.1.1 Where the LLM emits provenance — top-level wrapper, not per-entity

The original draft suggested per-entity `_field_provenance: dict[str, str]` on each `RadarSystemEntity` / `MissileSystemEntity` instance. **That doesn't work.** Reviewer caught this: the extraction entity classes set `model_config = ConfigDict(extra="ignore", ...)` (see `extraction_schemas/radar_domain.py:94`, `missile_domain.py:86`), and the docling-graph service serializes via `template_instance.model_dump(mode="json")` at `docker/docling-graph/app/main.py:850`. An undeclared `_field_provenance` key on an entity would be silently dropped during Pydantic validation before it ever reaches `pass_output`.

**Revised design:** put provenance at the **pass-template level**, parallel to the entities, not nested inside them.

`RadarDomainPass` (and `MissileDomainPass`) are the structured-output template classes — they already carry the list of extracted entities. Add a sibling field:

```python
class RadarDomainPass(BaseModel):
    primary_entities: list[RadarSystemEntity] = Field(default_factory=list, ...)
    field_provenance: list[FieldProvenanceRow] = Field(
        default_factory=list,
        description=(
            "Per-field source attribution. One row per (entity, field) pair "
            "for which the LLM identified a verbatim source snippet. The "
            "service post-processes these rows to resolve element_uid by "
            "substring-matching supporting_snippet against the input chunks."
        ),
    )

class FieldProvenanceRow(BaseModel):
    entity_index: int          # position in primary_entities
    field_name: str            # canonical field on the entity model
    supporting_snippet: str    # exact verbatim quote from source
```

Because `field_provenance` is a declared field on the pass template, `model_dump(mode="json")` carries it through the wire intact. The structured-output JSON schema the LLM sees is updated accordingly — the prompt asks for two top-level keys per pass response: `primary_entities` (existing) and `field_provenance` (new).

The service then converts each `FieldProvenanceRow` into a wire-shape `ExtractionFieldProvenance` row (joining `entity_index` to the matching `instance_id` from the entity-level provenance the service already tracks) before returning the response.

### 5.2 Files touched

- `ontology_bundles/air_defense_v3/extraction_schemas/radar_domain.py`, `missile_domain.py` — add `field_provenance: list[FieldProvenanceRow]` field to `RadarDomainPass` and `MissileDomainPass` (the pass-template classes that the docling-graph service serializes). Add `FieldProvenanceRow` itself in a shared module (e.g. `ontology_bundles/air_defense_v3/extraction_schemas/_field_provenance.py`).
- `docker/docling-graph/app/schemas.py` — new wire-shape `ExtractionFieldProvenance`; `ExtractPassResponse` gets `field_provenance: list[ExtractionFieldProvenance]`. Built from each pass response's `template_instance.field_provenance` rows in the service post-process.
- `docker/docling-graph/app/main.py` (or wherever the extraction prompt + structured-output schema is composed) — extend output schema and prompt with per-field source-snippet requirement.
- `docker/docling-graph/app/prompt_rules.py` — add a "field provenance" instruction block.
- `docker/docling-graph/app/provenance.py` — extend the existing post-LLM provenance pass to also resolve `ExtractionFieldProvenance` snippets to `element_uid`.
- `app/services/extraction_merge.py` — parse `field_provenance` from `ExtractPassResponse`, attach to `MergedEntityRecord`, dedup on `(instance_id, field_name)`.
- `app/services/arcadedb_graph.py` — `upsert_nodes_batch_sync` writes `_field_evidence: dict[field_name, list[{chunk_id, snippet, element_uid}]]` as a JSON property on the entity vertex.
- `app/services/query_profiles.py` — `_project_field_groups` reads `_field_evidence`, fills `QueryProfileFieldEntry.evidence`.
- `app/schemas/query_profiles.py` — `QueryProfileFieldEntry.evidence: list[GraphEvidenceItem] = []`.
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

The structured-output JSON schema gets a new top-level `field_provenance` array (per §5.1.1). The system prompt is extended with:

> After populating `primary_entities`, fill `field_provenance`. For every field you populated on an entity for which you can quote a source, emit one `field_provenance` row containing:
>
> - `entity_index`: the 0-based position of the entity in `primary_entities`
> - `field_name`: the canonical field name on that entity (e.g. `gain_dbi`, `max_speed_mps`)
> - `supporting_snippet`: an exact verbatim quote from the input text that established the field's value. The snippet must appear verbatim somewhere in the chunks provided. Do not paraphrase or summarize. Whitespace differences are acceptable; word substitution is not.
>
> If you cannot quote a source for a field, simply omit that field's row from `field_provenance` — never invent or paraphrase. An empty `field_provenance` array is acceptable.

The service post-process converts each `FieldProvenanceRow` (with `entity_index`) into an `ExtractionFieldProvenance` row (with `instance_id`) by indexing into the same `primary_entities` array the service already iterates over to emit entity-level provenance.

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

`_project_field_groups` reads `instance_data["_field_evidence"][field_name]` and converts each row to a `GraphEvidenceItem` using the same `_lookup_chunk_by_type` helper that retrieval uses. Old data: `_field_evidence` missing → `evidence=[]`. The UI renders an empty cell with "no per-field evidence" tooltip.

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
    evidence: list[GraphEvidenceItem] = []      # Phase 3 — empty until re-ingest

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
    include_associated_systems: bool = False  # NEW — Components only (ASSOCIATED_WITH / CUES)
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
| 3 | `RadarDomainPass.field_provenance` round-trip through `model_dump`; snippet→element_uid resolver; `MergedEntityRecord.field_evidence` union; upsert serialization round-trip; `_project_field_groups` evidence pass-through. | End-to-end re-ingest of one doc → `_field_evidence` populated on the entity vertex; `/search/section` surfaces ≥1 per-field evidence row. | Field-evidence rows joined to chunks 1:1; missing snippet → `element_uid=None, chunk_id=None`, no fabrication. |

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

---

## 11. Review responses (revision 1)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | High | Phase 1 file scope incomplete: stale references in `coverage.yaml`, `validation_matrix.py`, `entities.py:ENTITY_TYPES`, `dossier_service.py:38`. `check_bundle()` won't pass. | §3.1 expanded to enumerate all six files (coverage.yaml, validation_matrix.py, relationships.py, entities.py + ENTITY_TYPES catalog, dossier_service.py constants, plus the canonical-class `edge(...)` field deletions on `RadarSystemEntity` / `MissileSystemEntity`). §3.6 adds an explicit import-time gate to catch any remaining stale reference. |
| 2 | High | `CHILD_OF` doesn't connect systems — declared `SECTION → SECTION` only. `system_links` emits `ASSOCIATED_WITH` and `CUES`. Components' `related_systems` would be empty. | §1.4 #4 updated. §4.6 renamed to `get_associated_systems`, switched to `ASSOCIATED_WITH`/`CUES`. Profile flag renamed `include_child_of` → `include_associated_systems`. Rationale documented inline. |
| 3 | High | Phase 1 mapping omits `nomenclature`, `emitter_function`, and (on missile) extra fields. Will fail the contract or vanish silently. | §3.3 mapping now has explicit rows for `nomenclature` (system_metadata=True, surfaced on entity header), `emitter_function` (mapped to `rf_parameters` for radar / `performance` for missile), `asrd` and other audit-trail fields (system_metadata=True). New convention spelled out: every domain field is in exactly one of {profile-mapped, system_metadata, system_field}. New contract test gates the build. |
| 4 | Medium | Orphan-deletion list too narrow — radar/missile entities still have nested edge fields and `ENTITY_TYPES` registers ~33 classes, only 6 are listed for deletion. | §3.4 expanded to a full audit: 27 classes deleted, 20 retained, with the criterion stated (orphan = no extraction path AND not a structural anchor / system top type). The relationship-type cleanup in `relationships.py` and `validation_matrix.py` is in scope alongside the class deletions. |
| 5 | Medium | Phase 3 per-entity `_field_provenance` is dropped by `extra="ignore"` before the service unpacks it. | §5.1.1 added: provenance moves to a top-level `field_provenance: list[FieldProvenanceRow]` on the pass-template class (`RadarDomainPass` / `MissileDomainPass`), parallel to `primary_entities`. Declared field, no `extra` swallowing. The wire `ExtractionFieldProvenance` is built in the service post-process by joining `entity_index` to `instance_id`. §5.4 prompt instructions rewritten accordingly. |
| 6 | Medium | Frontend impact understated: TS API only allows `kind: "section" | "dossier"`; `QueryPage` flattens through `items`. | §3.1 (under Phase 2 file list) and §4.9 expanded with concrete file paths (`api/client.ts:520` literal-type extension, `QueryPage.tsx:827` switch refactor) and three new components (`FieldGroupTable`, `FieldEvidencePopover`, `DossierSectionList`). |
| 7 | Low | §1.1 incorrectly says manifest is `kind="entities"` everywhere; `system_links` is `relationships_only`. | §1.1 corrected to spell out the per-pass kinds and the two relationship types `system_links` actually emits. |
| 8 | Low | `ChunkExcerpt` is not an existing schema name — current evidence shape is `GraphEvidenceItem`. | All references updated to `GraphEvidenceItem` throughout (§§4-6). |
