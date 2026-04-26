# Flat-Schema Profile Refactor — Design

**Date:** 2026-04-25
**Status:** Frozen after sixth review pass — 2 cleanup edits applied (see §16). Ready for implementation planning.
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
│    + System Components: get_associated_systems(node_id)         │
│    + Dossier: single resolved_root + per-section blocks         │
│    + Starter-profile registry migration (alembic) for the 4     │
│      starter profile IDs — see §4.10                            │
│    + Phase 2 also removes 4 legacy /graph/system-* endpoints    │
│      and dossier_service.py — see §4.13                         │
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
│    + surface evidence: list[QueryProfileFieldEvidence]               │
│      on each FieldEntry (snippet + element_uid + chunk meta)         │
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
- `app/services/dossier_service.py` — **kept untouched in Phase 1** (per pass-3 review).

  Reviewer (pass 3) correctly flagged that deleting the legacy `/graph/system-dossier` endpoint inside Phase 1 contradicts §7's "Phase 1 is ontology-only" backward-compat story. The constants reference type-name **strings** (`"FREQUENCY_BAND"`, `"RF_EMISSION"`, …), not the deleted Pydantic classes, so deleting the orphan classes does NOT break `dossier_service.py` at import time — the legacy endpoint continues to load and runs against an empty graph for those types (post-flat-refactor reality), returning empty results. That's the same behavior the user already sees today; we're not regressing.

  Phase 2 owns the deletion of `dossier_service.py` and **all four** `/graph/system-*` routes (`/system-dossier`, `/system-components`, `/system-rf-parameters`, `/system-performance`) plus the `_system_section` helper — see §4.13. This restores §7's purity: Phase 1 is ontology-only; the legacy-endpoint removal is a deliberate Phase 2 breaking change with migration notes.
- `tests/unit/test_docs_compliance_contracts.py`, `tests/unit/contracts/test_extraction_schema_contract.py`, `tests/unit/test_coverage_checker.py` — drop the 5 `xfail(strict=False)` markers from the prior debugging pass.

After the first `check_bundle()` run lands a green result on the modified bundle, the Phase 1 file scope is verified. Anything that still references a deleted symbol surfaces as an `ImportError` or a `check_bundle()` failure; both are deterministic gates we can iterate on.

### 3.2 Field migration

For every non-system field on `extraction_schemas/radar_domain.py:RadarSystemEntity` and `extraction_schemas/missile_domain.py:MissileSystemEntity`:

1. Copy the declaration onto the canonical class (`entities.py`) with the same type and default.
2. Add a meaningful `description` and (where applicable) `examples` — required by the docs-compliance contract.
3. Add `json_schema_extra={"profile_sections": [...], "profile_subgroup": "<group>"}`.

The `confidence` system field on each canonical class is unchanged.

### 3.3 `profile_sections` and `profile_subgroup` mapping

**Convention:** every field on the canonical class falls into one of four buckets, and the contract is enforced by a Phase 1 unit test (§3.6 success criteria):

1. **Profile-mapped** — `profile_sections: list[str]` non-empty, `profile_subgroup: str` set.
2. **System metadata** — `profile_sections: []` AND `system_metadata: True` (new flag in `json_schema_extra`). The field is real and indexed but never surfaced by a starter profile (e.g., audit trails like `responsible_agency`, classifier IDs like `dieqp`, status flags like `system_status`, review cadence like `review_cycle`/`next_review_date`).
3. **Identity** — fields named in `model_config["graph_id_fields"]` (e.g., `system_name`) or marked `identity_field: True` (e.g., `nomenclature`, the missile-schema `name` alias field). Identity fields are surfaced on the entity header in the UI; they're excluded from the contract's "must be profile-mapped or metadata" rule because they're identity, not profile content.
4. **System field** (the existing `system_field: True` marker) — bookkeeping like `confidence`, `extraction_confidence`. Unchanged.

A field that's none of profile-mapped / system_metadata / identity / system_field is a bug — the contract test fails the build. Reviewer (pass 3) flagged that the prior two-bucket taxonomy didn't account for identity fields like `system_name` (a `graph_id_field`) and that `MissileSystemEntity` has more metadata than the original mapping enumerated; this revision is the four-bucket replacement.

**`RadarSystemEntity` (canonical) full mapping:**

| Fields | `profile_sections` | `profile_subgroup` |
|---|---|---|
| `nominal_rf_mhz`, `frequency_excursion_mhz`, `nominal_pri_usec`, `nominal_pd_usec`, `inter_pulse`, `pulses_per_dwell`, `dwell_time`, `intra_pulse_mop`, `num_bits_in_code` | `["rf_parameters"]` | `"waveform"` |
| `antenna_dim_az_m`, `antenna_dim_el_m`, `beamwidth_az_deg`, `beamwidth_el_deg`, `gain_dbi`, `antenna_photo`, `spoiled`, `coverage_limits_el_deg` | `["rf_parameters","components"]` | `"antenna"` |
| `tx_peak_power_kw`, `erp_dbw` | `["rf_parameters","performance"]` | `"transmit"` |
| `scan_type`, `scan_period_sec` | `["rf_parameters","performance"]` | `"scan"` |
| `emitter_function` | `["rf_parameters"]` | `"classification"` |
| `system_name` | identity (`graph_id_field`) | n/a — surfaced on entity header |
| `nomenclature` | identity (`identity_field=True`) | n/a — surfaced on entity header |
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
| `system_name` | identity (`graph_id_field`) | n/a — surfaced on entity header |
| `nomenclature`, `name` | identity (`identity_field=True`) | n/a — `name` is the missile-schema secondary alias field at `missile_domain.py:131`; rendered after `nomenclature` on the entity header |
| `dieqp`, `asrd`, `system_status`, `responsible_agency`, `review_cycle`, `next_review_date` | `[]`, `system_metadata=True` | n/a — full list expanded per pass-3 reviewer note (`MissileSystemEntity` extraction at `missile_domain.py:122,153,174,186,195` includes all of these as Optional fields) |

**Identity-adjunct surfacing.** `nomenclature` is not part of any profile section but is high-value identity context (formal designator vs. NATO common name). The entity header on the section/dossier UI renders the resolved root's `nomenclature` next to the system name when populated. No special schema work — `nomenclature` rides along on `resolved_root.properties` like any other field.

### 3.4 Orphan canonical entities — full audit

The original draft listed 6 classes. Reviewer (correctly) flagged that this is too narrow: the broader set of subtypes registered in `ENTITY_TYPES` at `entities.py:1346` are also orphaned by the flat-checklist refactor and have to be removed coherently for `check_bundle()` to pass.

**Full deletion set** (every class registered in `ENTITY_TYPES` that has no extraction path in `radar_domain.py`, `missile_domain.py`, or `system_links.py`, and is not a structural anchor or top-level system type):

`FrequencyBandEntity`, `ModulationEntity`, `RfSignatureEntity`, `RfEmissionEntity`, `WaveformEntity`, `ScanPatternEntity`, `AntennaEntity`, `TransmitterEntity`, `ReceiverEntity`, `IfAmplifierEntity`, `SignalProcessingChainEntity`, `GuidanceMethodEntity`, `SeekerEntity`, `MissilePerformanceEntity`, `MissilePhysicalCharacteristicsEntity`, `PropulsionStackEntity`, `PropulsionStageEntity`, `CapabilityEntity`, `RadarPerformanceEntity`, `EngagementTimelineEntity`, `ForceStructureEntity`, `AssemblyEntity`, `SpecificationEntity`, `StandardEntity`, `ProcedureEntity`, `FailureModeEntity`, `TestEventEntity`.

**Retained classes** (used by the structural / system layer, not orphaned):

`DocumentEntity`, `SectionEntity`, `FigureEntity`, `TableEntity`, `ImageEntity`, `TextBlockEntity`, `OrganizationEntity`, `PlatformEntity`, `WeaponSystemEntity`, `EquipmentSystemEntity`, `SubsystemEntity`, `ComponentEntity`, `RadarSystemEntity`, `MissileSystemEntity`, `AirDefenseArtillerySystemEntity`, `ElectronicWarfareSystemEntity`, `FireControlSystemEntity`, `IntegratedAirDefenseSystemEntity`, `LauncherSystemEntity`.

(Pass-3 reviewer note: `Alias` is **not** a Pydantic canonical class — it's an ArcadeDB schema-level vertex type used for entity-aliasing in the graph layer. It stays in the ArcadeDB schema, not in this canonical-class list.)

**Audit gate:** before deletion, grep each candidate name across the entire repo (`grep -rn "AntennaEntity\|ANTENNA\b" --include='*.py' --include='*.yaml' --include='*.md'`) and inspect each hit. Anything that's a test fixture / notebook reference / orphaned import gets cleaned up in the same commit; anything that turns out to be a live consumer (extraction path I missed, fixtures we still want to keep) is escalated and the deletion is reconsidered for that specific class.

**Validation matrix + relationship enum:** every tuple in `validation_matrix.py` whose subject or object is in the deletion set goes away, and every relationship label that only occurred between deleted classes (`HAS_ANTENNA`, `EMITS`, `RADIATES`, `RECEIVES`, `OPERATES_IN_BAND`, `USES_WAVEFORM`, `USES_MODULATION`, `SPECIFIED_BY`, `HAS_SCAN`, `HAS_SEEKER`, `HAS_SIGNATURE`, `HAS_PROCESSING_CHAIN`, `HAS_RECEIVER`, `HAS_TRANSMITTER`, `HAS_IF_AMPLIFIER`, `HAS_PROPULSION`, `HAS_GUIDANCE`, `MEASURES`, `MANUFACTURED_BY` if no longer reachable, …) is removed from `RelationshipType` in `relationships.py` and from `coverage.yaml`. The `system_links` pass continues to emit `ASSOCIATED_WITH` and `CUES`; those stay.

**Edge fields on retained entities — full audit (per pass-2 reviewer note):** every retained class is audited for `edge(...)` fields whose target is in the deletion set. Confirmed offenders (incomplete — exact list determined by grep at implementation time):

- `EquipmentSystemEntity` (`entities.py:259-340`) — delete `capabilities`, `standards`, `specifications`, `test_events`.
- `SubsystemEntity` (`entities.py:340-410`) — delete `capabilities`, `standards`, `test_events`.
- `RadarSystemEntity` (`entities.py:411-560`) — delete `waveforms`, `rf_emissions`, `antennas`, `receivers`, `transmitters`, `scan_patterns`, `signal_processing_chains`, `radar_performances`, `frequency_bands`, `capabilities`, `rf_signatures`, `engagement_timelines`, `specifications`, `test_events` (~14 edges).
- `MissileSystemEntity` (`entities.py:568-660`) — delete `guidance_method`, `seeker`, `propulsion_stacks`, `missile_performances`, `capabilities`, `specifications`, `test_events` (~7 edges).
- `WeaponSystemEntity` (`entities.py:225-258`) — audit per grep; likely `capabilities`, `specifications` if present.
- `PlatformEntity`, `AirDefenseArtillerySystemEntity`, `ElectronicWarfareSystemEntity`, `FireControlSystemEntity`, `IntegratedAirDefenseSystemEntity`, `LauncherSystemEntity` — same audit (`ElectronicWarfareSystemEntity` at `entities.py:716+` has at least `frequency_bands`, `capabilities`, `rf_emissions`).

**The audit gate is mechanical:** for each retained class, walk `model_fields` and flag any field whose annotation references a deletion-set name; delete those fields and rebuild forward refs. The first `python -c "from ontology_bundles.air_defense_v3 import entities"` after deletion that succeeds confirms the audit is complete.

### 3.5 Drop xfails

Once 3.2–3.4 land, the 5 schema-drift tests pass without their `xfail` markers. Remove each marker in the same commit. If a marker can't be removed, that's a real regression to fix before merging Phase 1.

### 3.6 Phase 1 success criteria

- `pytest tests/unit -q` shows the 5 previously-xfail'd tests passing without markers.
- New contract test: every field on `RadarSystemEntity` and `MissileSystemEntity` falls into exactly one of the four buckets defined in §3.3 — profile-mapped (non-empty `profile_sections`), system_metadata (`system_metadata=True`), identity (declared in `model_config["graph_id_fields"]` or marked `identity_field=True`), or system field (existing `system_field=True`). No field falls through the cracks.
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
- (Listed elsewhere — **`FieldEvidencePopover.tsx` is built in Phase 3, not Phase 2.** See §5.2 for its file entry. Phase 2 leaves the per-row evidence chip on `<FieldGroupTable>` inert; Phase 3 wires it up.)
- `tests/unit/test_query_profiles.py` — rewrite for property-projection paths; new helper tests for `_project_field_groups`, `_canonical_class_for`, the `_CANONICAL_ROOT_ENTITY_TYPES` ↔ `_CANONICAL_BY_ENTITY_TYPE` sync contract.
- `tests/unit/test_dossier_service.py` — **delete** (per pass-3 reviewer note: §4.13 deletes `app/services/dossier_service.py`, so its test counterpart goes with it). Coverage of dossier behavior moves into `tests/unit/test_query_profiles.py` under the dossier execute path.

### 4.2 New profile kind

Add `"section_properties"` to `QueryProfileDefinition.kind`. Schema additions:

- `profile_sections: list[str]` — which `json_schema_extra["profile_sections"]` tags this profile pulls.
- `include_associated_systems: bool = False` — when true, the section response includes `related_systems` populated from `ASSOCIATED_WITH` / `CUES` edges. Only `system_components` sets it. (Renamed from `include_child_of` per review.)

`validate_shape` model_validator updated:

- `kind=="section"` requires non-empty `traversals` (existing rule, unchanged).
- `kind=="section_properties"` requires non-empty `profile_sections` AND every entry in `root_entity_types` MUST appear in the allowed-root-types set (per pass-2 reviewer note).

  **Where the validation lives** (per pass-3 reviewer note): the schema-side Pydantic validator references a module-local constant `_CANONICAL_ROOT_ENTITY_TYPES: frozenset[str] = frozenset({"RADAR_SYSTEM", "MISSILE_SYSTEM"})` declared at the top of `app/schemas/query_profiles.py`. The service-side dispatch dict `_CANONICAL_BY_ENTITY_TYPE` in `app/services/query_profiles.py` is kept in sync via a contract test that asserts `set(_CANONICAL_BY_ENTITY_TYPE.keys()) == _CANONICAL_ROOT_ENTITY_TYPES`. This avoids a backward dep from schema → service (the service module already imports from the schema module; the reverse direction would create a circular import even with lazy resolution). Single source of truth in the schema layer; the service mirrors it.
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
            kind=section_profile.kind,                # "section" | "section_properties"
            field_groups=section_resp.field_groups,    # populated for section_properties
            related_systems=section_resp.related_systems,
            items=section_resp.items,                  # populated for legacy section
        ))

    return QueryProfileDossierResponse(
        registry_id=...,
        profile_id=profile.id,
        profile_label=profile.label,
        resolved_root=resolved,
        aliases=resolved.aliases or [],   # preserved for back-compat per pass-4 review
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
- **`components/DossierSectionList.tsx`** *(new)* — one entity-header card (using `resolved_root.name` + `nomenclature` if populated, plus type chip), then N stacked section blocks each containing a `<FieldGroupTable>`. Empty sections render with a "no data extracted" placeholder rather than disappearing.

Phase 2 builds `<FieldGroupTable>` and `<DossierSectionList>` but leaves `<FieldGroupTable>`'s per-row evidence chip inert (the API returns `evidence: []` until Phase 3). Phase 3 (§5.2) builds `<FieldEvidencePopover>` and wires the chip to open it; no further structural change.

### 4.10 Active-registry reconciliation

Reviewer (pass 2) flagged that starter profile definitions live in `build_default_registry_template()` at `query_profiles.py:157`, but the search endpoints read from the **DB-persisted active registry** via `get_required_active_registry()` at `query_profiles.py:372`. Existing active registries already contain the four starter profiles in their old `kind="section"` traversal-based shape; merely changing `build_default_registry_template()` does nothing for them.

A startup reconciliation step is required. Two options:

| Option | Mechanism | Trade-offs |
|---|---|---|
| **A — Alembic data migration** | New migration that updates the JSON column for the four well-known starter profile IDs (`system_dossier`, `system_components`, `system_rf_parameters`, `system_performance`) inside any existing active registry row. | Idempotent, runs once on `alembic upgrade head`, no per-startup cost. Hard-codes the four IDs, but they're well-known starter IDs. |
| **B — Startup hook** | A `_reconcile_starter_profiles_on_active_registry()` call from API startup that diffs each active registry against `build_default_registry_template()` for those four IDs and overwrites if shape doesn't match. | Self-healing for any future drift, runs every startup, slightly more code. |

**Recommend A.** The migration is one-time, deterministic, and tracked in source control alongside other schema changes. Option B's "self-healing" benefit is only useful if we expect the starter definitions to evolve frequently after this spec lands; if we do, that becomes a separate concern.

The migration should:
1. Locate every row in the registry table whose JSON column carries one of the four starter profile IDs.
2. For each, replace the entire profile-definition object for that ID with the new shape from `build_default_registry_template()`.
3. Leave all non-starter profiles in the registry untouched.
4. Be **structurally reversible** — `down()` writes back the old traversal-based JSON shape from the migration file. Reviewer (pass 4) correctly noted this is *structural* reversal, not *behavioral* compatibility: Phase 1 deleted the ontology types, relationships, and validation rows the old traversal-based profiles depend on. A profile-only `down()` produces parseable rows but the underlying graph traversal will return empty results because there are no `ANTENNA` / `RECEIVER` / `HAS_*` etc. anymore. To fully roll back to the old behavior the operator must roll back Phase 1 too (the ontology + code changes). The migration file's docstring spells this out explicitly so an operator running `alembic downgrade -1` can't be surprised.

User-defined custom profiles inheriting from the old `kind="section"` shape continue to work as before; only the four well-known starter IDs are touched.

### 4.11 Phase 2 success criteria

- `/v1/query-profiles/search/section` for each starter profile against the running ArcadeDB returns non-empty `field_groups` for at least one of: `SA-2`, `Fan Song`, `Engagement and Fire Control Radar`.
- `/v1/query-profiles/search/dossier` returns one `resolved_root`, populated `aliases` (back-compat), and 3 populated `sections`.
- The legacy `kind="section"` profile path still produces `items` correctly; the new branch is purely additive on the response and inert on legacy profiles.
- After §4.13's deletions, **`grep -rn "from app.services.dossier_service\|/graph/system-dossier\|/graph/system-components\|/graph/system-rf-parameters\|/graph/system-performance\|SystemQueryRequest\|SystemSectionResponse\|SystemDossierResponse\|_system_section\|build_system_dossier" {app,tests}/`** returns zero hits. (Phase 2 success gate per pass-5 reviewer note — leftover imports for the deleted shapes/handlers must not slip through.)
- Unit suite green; no xfail regressions.

### 4.13 Legacy `/graph/system-*` endpoint removal (Phase 2 breaking change)

`app/services/dossier_service.py` backs **four** legacy routes (per pass-4 reviewer note — the prior draft only mentioned the dossier route):

| Route | Handler at `graph_store.py` | Migration |
|---|---|---|
| `POST /graph/system-dossier` | line 177 → `build_system_dossier` | `POST /v1/query-profiles/search/dossier {profile_id: "system_dossier"}` |
| `POST /graph/system-components` | line 204 → `_system_section(..., "components")` | `POST /v1/query-profiles/search/section {profile_id: "system_components"}` |
| `POST /graph/system-rf-parameters` | line 209 → `_system_section(..., "rf_parameters")` | `POST /v1/query-profiles/search/section {profile_id: "system_rf_parameters"}` |
| `POST /graph/system-performance` | line 214 → `_system_section(..., "performance")` | `POST /v1/query-profiles/search/section {profile_id: "system_performance"}` |

Phase 2 removes **all four** routes plus the `_system_section` helper (`graph_store.py:195`), plus `dossier_service.py`. One packaged breaking change:

- Delete `app/services/dossier_service.py`.
- Remove all four `/graph/system-*` route handlers and `_system_section` from `app/api/v1/graph_store.py` (lines 177-216 region).
- Remove the request/response schemas they exclusively use (`SystemQueryRequest`, `SystemSectionResponse`, `SystemDossierResponse`) if no other callers — grep audit.
- Drop any `app.services.dossier_service` imports from other modules (grep audit).
- `app/services/query_profiles.py` previously imported one helper from `dossier_service.py` (around line 359) for the entity_evidence path — inline the equivalent or fold it into `query_profiles.py` directly.

Frontend has zero references to any of the four routes (`grep -rn "/graph/system-" frontend/src/` returns nothing); no frontend coordination needed. External API consumers (if any) get the migration table above in CHANGELOG.

### 4.14 Risks

- A canonical field that was meant to belong to a profile but was tagged `[]` by mistake silently disappears from the UI. Mitigation: the four-bucket contract test from §3.6 — every field must land in exactly one of profile-mapped / system_metadata / identity / system_field; mistyping a profile field as system_metadata fails the test on a different axis (the diff against the previous bucket-assignment dump).
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

`RadarDomainPass` (and `MissileDomainPass`) are the structured-output template classes — they already carry the list of extracted entities, but the list field is **pass-specific**: `RadarDomainPass.radar_systems` (`radar_domain.py:560`) and `MissileDomainPass.missile_systems` (`missile_domain.py:573`). There is no generic `primary_entities` field.

The reviewer (pass 2) caught this. The provenance design must accommodate that pass-specific naming.

```python
# In a shared module, e.g. ontology_bundles/air_defense_v3/extraction_schemas/_field_provenance.py
class FieldProvenanceRow(BaseModel):
    entity_index: int          # 0-based index into the pass's primary entity list
    field_name: str            # canonical field on the entity model
    supporting_snippet: str    # exact verbatim quote from source
```

Each pass-template gains a sibling list (no rename of the existing primary list field):

```python
class RadarDomainPass(BaseModel):
    radar_systems: list[RadarSystemEntity] = Field(default_factory=list, ...)
    field_provenance: list[FieldProvenanceRow] = Field(default_factory=list, ...)

class MissileDomainPass(BaseModel):
    missile_systems: list[MissileSystemEntity] = Field(default_factory=list, ...)
    field_provenance: list[FieldProvenanceRow] = Field(default_factory=list, ...)
```

The service post-process resolves `entity_index` to the right primary list by introspecting which list field holds the pass's primary entities — the manifest already declares `primary_entity_types` (e.g., `[RADAR_SYSTEM]` for `RadarDomainPass`), and a small helper finds the model field whose annotation is `list[<class with ontology_name=primary_type>]`:

```python
def _primary_list_field_name(template_cls: type[BaseModel], primary_type: str) -> str:
    """Return the name of the pass-template's primary entity list field
    (e.g. 'radar_systems' for RadarDomainPass when primary_type='RADAR_SYSTEM')."""
```

Because `field_provenance` is a declared field on the pass template, `model_dump(mode="json")` carries it through the wire intact. The structured-output JSON schema the LLM sees is updated accordingly — the prompt asks for two top-level keys per pass response: the pass-specific primary list (e.g. `radar_systems`, existing) and `field_provenance` (new).

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
- `app/schemas/query_profiles.py` — `QueryProfileFieldEntry.evidence: list[QueryProfileFieldEvidence] = []`. The new `QueryProfileFieldEvidence` shape (defined in §5.8) carries chunk metadata + `supporting_snippet` + `element_uid` — `GraphEvidenceItem` lacks the latter two fields and is left untouched on retrieval / dossier paths where they're meaningless.
- `frontend/src/components/FieldEvidencePopover.tsx` — *new component, built here in Phase 3.* Lists each evidence row as `(supporting_snippet, chunk_id, element_uid)` with a deep link to the document viewer at the matching element. "Unverified source" badge on rows where post-process couldn't substring-match the snippet.
- `frontend/src/components/FieldGroupTable.tsx` — Phase 2 already created this component but left the evidence column inert. Phase 3 wires the per-row evidence chip to open `<FieldEvidencePopover>`.

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

> After populating the pass's primary entity list (e.g. `radar_systems` for the radar pass), fill `field_provenance`. For every field you populated on an entity for which you can quote a source, emit one `field_provenance` row containing:
>
> - `entity_index`: the 0-based position of the entity in the pass's primary entity list
> - `field_name`: the canonical field name on that entity (e.g. `gain_dbi`, `max_speed_mps`)
> - `supporting_snippet`: an exact verbatim quote from the input text that established the field's value. The snippet must appear verbatim somewhere in the chunks provided. Do not paraphrase or summarize. Whitespace differences are acceptable; word substitution is not.
>
> If you cannot quote a source for a field, simply omit that field's row from `field_provenance` — never invent or paraphrase. An empty `field_provenance` array is acceptable.

The service post-process converts each `FieldProvenanceRow` (with `entity_index`) into an `ExtractionFieldProvenance` row (with `instance_id`) by indexing into the pass's primary list (resolved via `_primary_list_field_name`, see §5.1.1).

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
    chunk_id: str | None       # None when element_uid couldn't be resolved
    snippet: str
    element_uid: str | None
```

`chunk_id` is resolved from `element_uid` via the existing chunk lookup (the same path that builds `EXTRACTED_FROM` edges). Rows where `element_uid` is None get `chunk_id=None` and the row still ships — the snippet alone is useful. (`chunk_id: str | None` per pass-2 reviewer note; the original draft typed it `str` but the design always allowed missing matches.)

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

`GraphEvidenceItem` (`app/schemas/graph_store.py:59`) carries chunk metadata and `content_text` but **does not** carry `supporting_snippet` or `element_uid`, both of which the Phase 3 UI needs (snippet for the popover quote; `element_uid` for the deep link into the document viewer). Reviewer (pass 3) correctly flagged this.

A dedicated `QueryProfileFieldEvidence` shape is added to `app/schemas/query_profiles.py`:

```python
class QueryProfileFieldEvidence(APIModel):
    # Chunk metadata — same surface as GraphEvidenceItem.
    chunk_id: Optional[uuid.UUID] = None         # None when snippet didn't resolve to a chunk
    chunk_type: Optional[str] = None
    artifact_id: Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    document_name: Optional[str] = None
    modality: Optional[str] = None
    page_number: Optional[int] = None
    classification: str = "UNCLASSIFIED"
    content_text: Optional[str] = None
    source_characterization: Optional[str] = None
    date_of_information: Optional[str] = None
    extraction_confidence: Optional[float] = None
    # Field-evidence specific.
    supporting_snippet: str                       # the LLM's exact-quoted source text
    element_uid: Optional[str] = None             # DoclingDocument element_uid; None when unresolved
```

`QueryProfileFieldEntry.evidence: list[QueryProfileFieldEvidence] = []`. `_project_field_groups` reads `instance_data["_field_evidence"][field_name]`, looks up the chunk via the same `_lookup_chunk_by_type` retrieval helper, and combines chunk metadata + `supporting_snippet` + `element_uid` into one `QueryProfileFieldEvidence` row per evidence entry.

Old data: `_field_evidence` missing → `evidence=[]`. The UI renders an empty cell with a "no per-field evidence" tooltip.

Why not extend `GraphEvidenceItem`: it's used across retrieval and other paths where `supporting_snippet` is meaningless. Adding optional fields there would pollute every other consumer's TypeScript surface. The dedicated schema is purely additive.

### 5.9 Frontend

The `<FieldGroupTable>` row gets a small "evidence" affordance — an icon button that opens a popover listing each evidence entry as `(supporting_snippet, chunk_id, element_uid)`. The popover shows the verbatim snippet, the source chunk preview, and a deep link to the document viewer at `element_uid` when present. When the LLM emitted a snippet but post-processing couldn't substring-match it to a chunk, both `chunk_id` and `element_uid` are `None`; the popover still shows the snippet, prefixed with an "Unverified source" badge so the reader knows the citation is the LLM's quote without a confirmed chunk anchor. Empty cell when `evidence: []`.

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

- LLM ignores or paraphrases snippets despite the prompt. Mitigation: **keep the row, set `element_uid=None` and `chunk_id=None`, log a structured warning** (per pass-3 reviewer note resolving the §5.12/§5.13 contradiction in the prior draft). The UI renders unmatched snippets with a visible "unverified source" badge so a reviewer can decide whether to trust them. Dropping the row would silently lose informative quoted evidence; the badge is a better defense against fabrication because a human can audit it.
- Output size growth degrades extraction throughput. Mitigation: budget tracked in Phase 3 acceptance test; if over 5%, prompt is split into "structured fields first, then provenance" to avoid LLM wandering.
- Snippet collisions across chunks (same text appears in multiple chunks). Mitigation: longest-unique-prefix tiebreaker. If still ambiguous after that, the resolver picks the **first** chunk by stable order and emits a `ambiguous_snippet` log row carrying all candidate `element_uid`s. This keeps the schema (`element_uid: Optional[str]`) intact — the spec doesn't promise a list, and per pass-4 review we don't widen the schema to one. Ambiguous-snippet incidents stay visible via the log; if they turn out to be common in practice we add a per-row diagnostic field in a follow-up, not in this spec.

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
    evidence: list[QueryProfileFieldEvidence] = []   # Phase 3 — empty until re-ingest

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
    kind: Literal["section", "section_properties"]    # so the UI knows which payload to render
    field_groups: list[QueryProfileFieldGroup] = []   # populated when kind == "section_properties"
    related_systems: list[GraphEntityResult] = []     # populated when kind == "section_properties" + Components
    items: list[GraphEntityResult] = []                # populated when kind == "section" (legacy traversal)

class QueryProfileDossierResponse(APIModel):
    registry_id: Optional[uuid.UUID] = None
    profile_id: str
    profile_label: str
    resolved_root: GraphEntityResult                          # single root, per Q7-B
    aliases: list[str] = Field(default_factory=list)          # preserved from current schema for back-compat (pass-4 review)
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

- **Phase 1:** ontology-only — class deletions and field migrations on `entities.py`, `relationships.py`, `validation_matrix.py`, `coverage.yaml`, plus the `ENTITY_TYPES` registry. **No API changes.** No data migration. The legacy `/graph/system-dossier` endpoint and `dossier_service.py` are **kept in place** here (their constants reference type-name strings, not deleted Pydantic classes, so they don't break at import time); the legacy endpoint already returns empty results today and continues to return empty results after Phase 1. Phase 1 success is gated by the contract tests + a clean `python -c "from ontology_bundles.air_defense_v3 import entities, relationships, validation_matrix"`.
- **Phase 2:** purely additive on the **new** API surface — `/v1/query-profiles/search/section` gains optional `field_groups`/`related_systems`, `/v1/query-profiles/search/dossier` gains the new single-root + per-section-blocks shape (with `aliases` preserved for back-compat). Plus a **packaged set of deliberate breaking changes** (per pass-3 + pass-5 review): all four legacy `/graph/system-*` endpoints (`/system-dossier`, `/system-components`, `/system-rf-parameters`, `/system-performance`), the `_system_section` helper, and `app/services/dossier_service.py` are removed in Phase 2 — see §4.13 for the full per-route migration table. Frontend has zero references to any of the four routes, so no frontend coordination is needed; external API consumers (if any) get the §4.13 migration table in CHANGELOG. **Phase 1+2 deliver the visible fix without a re-ingest** because the flat extraction has been writing the field values onto the entity vertices all along; the canonical entity definitions just didn't recognize them. Sections become populated as soon as Phase 1+2 ship.
- **Phase 3:** wire and storage are additive. Old vertices have no `_field_evidence`; UI shows empty evidence cells with a tooltip explaining "no per-field evidence; re-ingest to populate." A one-time re-ingest of all corpora after Phase 3 lands populates evidence; not required.

---

## 8. Testing

| Phase | Unit | Integration | Contract |
|---|---|---|---|
| 1 | Four-bucket contract (§3.3): every field on `RadarSystemEntity` / `MissileSystemEntity` is profile-mapped OR system_metadata OR identity OR system_field. Bundle checker contract. | None (ontology-only). | 5 xfail'd schema-drift tests pass; remove markers. |
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

---

## 12. Review responses (revision 2)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | High | Phase 1 still under-scopes edge cleanup on retained classes. `EquipmentSystemEntity`, `SubsystemEntity`, `ElectronicWarfareSystemEntity`, etc. point to `CapabilityEntity`, `StandardEntity`, `SpecificationEntity`, `TestEventEntity`, `FrequencyBandEntity`, `RfEmissionEntity` — all in deletion set. Imports will fail. | §3.4 expanded with a per-class enumeration of stale edges on retained entities (Equipment, Subsystem, Radar, Missile, WeaponSystem, Platform, AirDefenseArtillery, ElectronicWarfare, FireControl, IntegratedAirDefense, Launcher) and a mechanical audit gate (`python -c "from ontology_bundles.air_defense_v3 import entities"` must succeed post-deletion). |
| 2 | High | `dossier_service.py` constants are not dead code — legacy `/graph/system-dossier` endpoint still uses them. Deleting only the constants leaves a runtime `NameError`. | §3.1 reworked: delete `dossier_service.py` and the legacy `/graph/system-dossier` route handler entirely in Phase 1. Frontend has zero references to the legacy endpoint (verified via `grep -rn "/graph/system-dossier" frontend/src/`); the new `/v1/query-profiles/search/dossier` is the replacement and ships in the same release. |
| 3 | High | "Phase 1+2 deliver visible fix without re-ingest" misses persisted active registries. Search endpoints read DB-backed `get_required_active_registry()`, not `build_default_registry_template()`. | New §4.10 ("Active-registry reconciliation"). Recommended path: alembic data migration that updates the JSON column for the four well-known starter profile IDs (`system_dossier`, `system_components`, `system_rf_parameters`, `system_performance`) inside any existing active registry row. Idempotent, reversible, leaves user-defined custom profiles untouched. |
| 4 | Medium | Phase 3 used `primary_entities` but actual fields are pass-specific (`radar_systems`, `missile_systems`). | §5.1.1 corrected: each pass-template gets `field_provenance` alongside its existing `radar_systems` / `missile_systems` field (no rename). New `_primary_list_field_name(template_cls, primary_type)` helper resolves the primary list per pass via the manifest's `primary_entity_types`. §5.4 prompt instructions reworked to reference "the pass's primary entity list" rather than a fictitious `primary_entities`. |
| 5 | Medium | `kind="section_properties"` validation only checks `profile_sections` non-empty, not `root_entity_types ⊆ _CANONICAL_BY_ENTITY_TYPE`. Endpoint will 500 instead of failing at registration. | §4.2 `validate_shape` rule extended: every entry in `root_entity_types` for a `section_properties` profile must appear in `_CANONICAL_BY_ENTITY_TYPE.keys()`. Lazy import to dodge the circular dependency. |
| 6 | Low | `FieldEvidenceRow.chunk_id` typed `str` but design allows None for unmatched snippets. | §5.6 dataclass updated to `chunk_id: str | None`. |
| 7 | Low | Architecture overview still showed `get_child_of_systems(node_id)` despite §4.6 rename. | §2 diagram updated to `get_associated_systems(node_id)`. |

---

## 13. Review responses (revision 3)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | High | `MissileSystemEntity` mapping omits `dieqp`, `name`, `system_status`, `responsible_agency`, `review_cycle`, `next_review_date`. Identity fields like `system_name` need explicit handling. | §3.3 reworked from a three-bucket convention to a four-bucket convention: profile-mapped / system_metadata / **identity** / system_field. `system_name` is identity (`graph_id_field`); `nomenclature` and the missile-schema `name` field are identity (`identity_field=True`); the six metadata fields are explicitly listed under `system_metadata=True`. Contract test enforces all four buckets. |
| 2 | High | Deleting `/graph/system-dossier` in Phase 1 contradicts §7's "Phase 1 ontology-only" backward-compat story. | §3.1 reverts: `dossier_service.py` and the legacy endpoint are **kept** in Phase 1 (they don't break at import time because the constants reference type-name strings, not deleted classes). New §4.13 owns the removal as a deliberate Phase 2 breaking change with explicit migration notes. §7 rewritten to call the Phase 2 removal out as a breaking change. |
| 3 | High | `GraphEvidenceItem` doesn't carry `supporting_snippet` or `element_uid` — Phase 3 needs both for the per-field popover and deep link. | §5.8 rewritten: introduce a dedicated `QueryProfileFieldEvidence` Pydantic shape in `app/schemas/query_profiles.py` that carries chunk metadata + `supporting_snippet` + `element_uid`. `QueryProfileFieldEntry.evidence` switches to `list[QueryProfileFieldEvidence]`. `GraphEvidenceItem` is left untouched on retrieval / dossier paths where snippet is meaningless. |
| 4 | Medium | §5.12 and §5.13 contradicted on unmatched snippets (surface-with-None vs. drop-and-log). | §5.13 risk mitigation rewritten: keep the row, set `element_uid=None`/`chunk_id=None`, log a structured warning. UI renders an "unverified source" badge. Dropping would silently lose informative quoted evidence; badge + log is the better defense against fabrication because it's auditable. |
| 5 | Medium | `_CANONICAL_BY_ENTITY_TYPE` import inside `app/schemas/query_profiles.py` creates a service-to-schema backward dep, brittle even with lazy imports. | §4.2 reworked: schema layer declares its own `_CANONICAL_ROOT_ENTITY_TYPES: frozenset[str]` constant; service-side `_CANONICAL_BY_ENTITY_TYPE` keys are kept in sync via a contract test. Schema-side validator references only the schema-local constant. No backward dep. |
| 6 | Medium | `QueryProfileDossierSection` lacked `items` — a dossier composing a legacy `kind="section"` profile would silently drop results. | §6.2 schema and the `execute_dossier_search` snippet add `items: list[GraphEntityResult] = []` and a `kind: Literal["section", "section_properties"]` discriminator on `QueryProfileDossierSection`. UI branches on `kind` like the section endpoint does. |
| 7 | Low | Retained-class list included `Alias`, but `Alias` is an ArcadeDB schema vertex type, not a Pydantic canonical class. | §3.4 retained list updated; explicit note added that `Alias` is a graph-schema concept, not an ontology-class concept. |
| 8 | Low | Testing section referenced `tests/unit/test_dossier_service.py`, but Phase 2 deletes the underlying module. | §3.1/§4.x test list now explicitly **deletes** `tests/unit/test_dossier_service.py` in Phase 2 alongside `dossier_service.py`. Dossier behavior coverage moves into `tests/unit/test_query_profiles.py` under the dossier execute path. |

---

## 14. Review responses (revision 4)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | High | §4.13 only removes `/graph/system-dossier`, but `dossier_service.py` also backs `/graph/system-components`, `/graph/system-rf-parameters`, `/graph/system-performance` (and the shared `_system_section` helper). | §4.13 rewritten to enumerate all four routes + `_system_section`, with a per-route migration table. The Phase 2 deletion is one packaged breaking change; CHANGELOG migration table covers all four. |
| 2 | High | Field-evidence schema inconsistent — §5.8 introduces `QueryProfileFieldEvidence`, but §5.2 / §6.1 / architecture diagram still said `list[GraphEvidenceItem]`. | All references updated: architecture diagram (§2), Phase 3 file list (§5.2), consolidated schema (§6.1) now consistently say `list[QueryProfileFieldEvidence]`. |
| 3 | Medium | §3.6 + §8 contract-test wording stale — said "profile_sections OR system_metadata=True" (two-bucket), but §3.3 has four buckets (profile-mapped, metadata, identity, system field). | §3.6 success criteria + §4.14 risks + §8 testing-table row 1 all rewritten to reference the four-bucket convention; the test must check exactly-one-of all four, not just the original two. |
| 4 | Medium | `/v1/query-profiles/search/dossier` already exposes `aliases` (per `app/schemas/query_profiles.py:131`) — the prior draft dropped it, making the response shape change a breaking change rather than additive. | `QueryProfileDossierResponse.aliases: list[str] = []` preserved in §6.2 schema and the `execute_dossier_search` populator snippet (§4.7). Compatibility restored. |
| 5 | Medium | §4.10 alembic rollback claim was over-promised — `down()` would write back old traversal-based starter profiles, but Phase 1 deleted the ontology types those profiles depend on. | §4.10 step 4 rewritten to call this out as **structurally reversible** but **not behaviorally compatible** without paired rollback of Phase 1. The migration's docstring is the place an operator running `alembic downgrade -1` will see this warning. |
| 6 | Low | §5.13 said ambiguous snippet collisions "attach all candidate `element_uid`s as a list" but `ExtractionFieldProvenance.element_uid` and `QueryProfileFieldEvidence.element_uid` are single optional strings. | §5.13 rewritten: resolver picks the first chunk by stable order after the longest-unique-prefix tiebreaker; emits an `ambiguous_snippet` log row with all candidates. Schema stays `element_uid: Optional[str]`. Multi-element future case explicitly deferred to a follow-up if it becomes common. |

---

## 15. Review responses (revision 5)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | High | §7 still said Phase 2 removes only `/graph/system-dossier` after §4.13 was expanded to four routes in revision 4. | §7 Phase 2 bullet rewritten to enumerate all four legacy `/graph/system-*` endpoints, the `_system_section` helper, and `dossier_service.py`. Aliases-back-compat note added. CHANGELOG-pointer remains §4.13's per-route migration table. |
| 2 | Medium | The schema-import cleanup ("if no other callers") needed to be a Phase 2 success gate, not a hand-wave. | §4.11 success criteria gain a concrete grep-audit gate: zero hits across `{app,tests}/` for any of `from app.services.dossier_service`, `/graph/system-*`, `SystemQueryRequest`, `SystemSectionResponse`, `SystemDossierResponse`, `_system_section`, `build_system_dossier`. |
| 3 | Medium | §5.9 popover text said `(snippet, chunk_id)` but the API now also carries `element_uid` and unresolved-state requires a badge. | §5.9 rewritten: popover lists `(supporting_snippet, chunk_id, element_uid)`; "Unverified source" badge for rows where snippet didn't substring-match (both `chunk_id` and `element_uid` are `None`); snippet still shown so the citation is verifiable by the reader. |
| 4 | Low | Architecture diagram said "Active-registry reconciliation on startup" but §4.10 recommended an alembic migration. | Diagram (§2) updated to "Starter-profile registry migration (alembic) — see §4.10". Also adds a one-line callout that Phase 2 removes 4 legacy endpoints (§4.13) so the diagram reflects the breaking change. |
| 5 | Low | §14 row 3 cross-referenced "§4.12 risks" — risks are now §4.14. | Cross-reference fixed in §14 row 3. |

---

## 16. Review responses (revision 6 — freeze cleanup)

| # | Finding | Resolution |
|---|---|---|
| 1 | §3.1 said "Phase 2 owns deletion of `dossier_service.py` and the `/graph/system-dossier` route" — should match §4.13 / §7's enumeration of all four routes. | §3.1 enumerates all four `/graph/system-*` routes plus `_system_section`. |
| 2 | §4.1 listed `FieldEvidencePopover.tsx` inside Phase 2's file list with a "(Phase 3)" parenthetical — confusing for implementation tracking. | Phase 2's file list now contains a placeholder entry pointing to §5.2 (Phase 3) for `FieldEvidencePopover.tsx`. The component itself is listed in §5.2's file list as Phase 3 work. §4.9's component list also clarified — Phase 2 builds `FieldGroupTable` (with inert evidence chip) and `DossierSectionList`; Phase 3 builds `FieldEvidencePopover` and activates the chip. |

**Spec is frozen.** Ready to hand off to `writing-plans`.
