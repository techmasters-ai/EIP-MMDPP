# Design Spec — docling-graph docs-alignment refactor

**Date:** 2026-04-16
**Status:** Draft for review
**Blocks:** completion of [`2026-04-14-docling-graph-schema-compliance.md`](./2026-04-14-docling-graph-schema-compliance.md) — that plan's Phase 7 Task 53 live-extraction gate cannot pass until this plan lands.
**Authoritative reference:** [`/docling-graph-docs.md`](../../../docling-graph-docs.md) at repo root.

## 1. Scope & approach

This plan realigns the `air_defense_v3` canonical ontology and extraction schemas with the docling-graph library's documented patterns. The motivating finding (2026-04-16 live-extraction investigation) is that the `reference` LLM pass cannot reliably populate its identity fields (`heading`, `page_start`) because the library's docs explicitly call this out as an anti-pattern:

> Prefer short, document-derived ID examples (section numbers, figure/table labels, named items). **Do not use section or chapter titles as entity identities.** — docling-graph-docs.md:18470

### Core theses

1. **Document structure comes from Docling, not the LLM.** SECTION/FIGURE/TABLE/DOCUMENT are emitted by a new deterministic worker pre-pass that walks `DoclingDocument` structure. The LLM reference pass is deleted.
2. **Every canonical entity gets a docs-compliant identity.** No `graph_id_fields=[]`. No long-text identities. No multi-field identities for document structure. Value objects become `is_entity=False` components.
3. **ASSERTION is dropped** from the ontology entirely (YAGNI — reintroduce later with real design if a concrete use-case emerges).
4. **Schemas are fault-tolerant by construction.** Strict required identity + lenient coercers on non-identity fields + library-level identity filter and quality gate (R22) handle messy LLM output.
5. **Full corpus re-ingest after canonical changes.** 21 docs re-processed via full pipeline to validate end-to-end on new schemas.

### Out of scope

- Upstream `docling-graph` library patches (we handle everything at the schema level).
- New entity types beyond the audit.
- Retrieval-algorithm changes.
- UI redesign (except the single ASSERTION filter-list deletion).
- Phase 8 provenance from the current plan — that resumes after this plan lands.

### Sequencing relative to the current plan

- **This plan precedes** the current `2026-04-14-docling-graph-schema-compliance.md`.
- Current plan's Phases 1–6 are already done, Phase 7 Task 52 done. Phase 7 Task 53 (live E2E gate) and Phase 8 (provenance) stay paused until this plan lands and merges.
- After this plan merges, the current plan resumes from Task 53 on stable schemas; Phase 8 provenance is additive on top.

## 2. Canonical entity classification

The audit examined all 46 canonical entities in `ontology_bundles/air_defense_v3/entities.py` (verified via `grep -c "^class .*(BaseModel):" entities.py`). Below is the classification table that drives every rewrite in Chunk B. Every canonical class is accounted for in exactly one bucket.

### 2.1 Dropped from ontology (2)

| Entity | Reason |
|---|---|
| ASSERTION | Q3 decision + longstanding KNOWN ANTI-PATTERN (`graph_id_fields=["assertion_text"]` violates R2 "short identities"). Drop entirely; reintroduce later if a concrete use-case emerges with a better design. |
| SPREADSHEET | Redundant with DOCUMENT. Consolidate by adding `SPREADSHEET` to `DocumentEntity.source_type` enum. `workbook_name` and `sheet_name` become optional properties on DocumentEntity. |

### 2.2 Give proper identity (15)

| Entity | Current | New `graph_id_fields` | Scope | Notes |
|---|---|---|---|---|
| DOCUMENT | `[]` | `["document_id"]` required | global | Internal UUID assigned at upload. Populated by `derive_document_anchors`, never by LLM. Not subject to R14 "descriptive ID" because it's system-constructed, not LLM-emitted. |
| SECTION | `["heading","page_start"]` | `["section_number"]` required | document | Positional enumeration from Docling `section_path` walk (e.g. `"1"`, `"1.1"`, `"2.3.4"`). R17 exempted because Docling-derived (see §2.5 rationale). `heading` becomes descriptive property. |
| FIGURE | `["figure_id","page"]` | `["figure_ref"]` required | document | Docling `self_ref` (e.g. `"#/pictures/3"`). `figure_label: Optional[str]` added as descriptive property for human-readable labels (`"Figure 3-12"` pulled from Docling caption when available). |
| TABLE | `["table_id","page"]` | `["table_ref"]` required | document | Docling `self_ref` (e.g. `"#/tables/1"`). `table_label: Optional[str]` added as descriptive property. |
| ORGANIZATION | `[]` | `["name"]` required | global | `cage_code` stays as an optional secondary property. |
| STANDARD | `[]` | `["designation"]` required | global | `"MIL-STD-1553B"`, `"MIL-DTL-31000G"`. |
| EQUIPMENT_SYSTEM | `[]` | `["name"]` required | global | Named trackable systems. |
| COMPONENT | `[]` | `["part_number"]` required | global | Components without a part_number in the source drop via library identity filter; that's acceptable (R18 "when possible"). |
| ASSEMBLY | `[]` | `["assembly_number"]` required | global | Same pattern as COMPONENT. |
| CAPABILITY | `[]` | `["capability_name"]` required | global | Named operational features. |
| PROCEDURE | `[]` | `["name"]` required | document | Doc-scoped — same-named procedure across unrelated docs ≠ same procedure. |
| FAILURE_MODE | `[]` | `["name"]` required | document | Doc-scoped. |
| TEST_EVENT | `[]` | `["name"]` required | global | Named events, globally unique. |
| FORCE_STRUCTURE | `[]` | `["name"]` required | global | Named military units. |
| SUBSYSTEM | `[]` | `["name"]` required | document | Doc-scoped hierarchical unit (docs Pattern 7 precedent, L16645). "Guidance Section" in Missile A ≠ "Guidance Section" in Missile B. |

### 2.3 Demote to `is_entity=False` (12 components)

Deduplicate by full content per docs L17225+ pattern.

| Entity | Pattern match |
|---|---|
| SPECIFICATION | Measurement — parameter+value+unit+condition is definitionally content (docs Address/Measurement pattern). |
| MODULATION | Value bag: chirp type + bandwidth + code bits. |
| RF_SIGNATURE | Bundle of RF observations. |
| RF_EMISSION | Observation snapshot. |
| SCAN_PATTERN | Value bag: scan type + timing. |
| IF_AMPLIFIER | Electrical stage description. |
| MISSILE_PERFORMANCE | Performance envelope data. |
| MISSILE_PHYSICAL_CHARACTERISTICS | Dimensions + mass. |
| PROPULSION_STACK | Container for stages (KNOWN ANTI-PATTERN in current code). |
| PROPULSION_STAGE | Sub-unit of stack; content-dedup on stage_type+params. |
| RADAR_PERFORMANCE | Performance envelope. |
| ENGAGEMENT_TIMELINE | Timing envelope. |

### 2.4 Unchanged (17 entities, already docs-compliant)

| # | Entity | Identity | Scope |
|---|---|---|---|
| 1 | PlatformEntity | `["name"]` | global |
| 2 | WeaponSystemEntity | `["system_name"]` | global |
| 3 | RadarSystemEntity | `["system_name"]` | global |
| 4 | MissileSystemEntity | `["system_name"]` | global |
| 5 | AirDefenseArtillerySystemEntity | `["system_name"]` | global |
| 6 | ElectronicWarfareSystemEntity | `["system_name"]` | global |
| 7 | FireControlSystemEntity | `["system_name"]` | global |
| 8 | IntegratedAirDefenseSystemEntity | `["name"]` | global |
| 9 | LauncherSystemEntity | `["system_name"]` | global |
| 10 | FrequencyBandEntity | `["band_name"]` | global |
| 11 | WaveformEntity | `["waveform_name"]` | document |
| 12 | AntennaEntity | `["name"]` | document |
| 13 | TransmitterEntity | `["name"]` | document |
| 14 | ReceiverEntity | `["name"]` | document |
| 15 | SignalProcessingChainEntity | `["name"]` | document |
| 16 | GuidanceMethodEntity | `["guidance_type"]` (enum) | global |
| 17 | SeekerEntity | `["seeker_nomenclature"]` | document |

All have single-field short identities on already-extracted named concepts. **These still receive docs-compliance touch-ups in Chunk B**: identity field examples get the §4.3 treatment (distinct, no duplicates, R16/R17-aligned), non-identity fields confirmed `Optional[T] = None` per §4.4, `edge(label=...)` fields gain `description` + `examples` kwargs per §7.1. But the identity shape doesn't change.

**Totals:** 46 canonical classes today → 2 dropped + 15 give-identity + 12 demoted + 17 unchanged = **46 accounted for**. Post-refactor: **32 entities + 12 components = 44 models** in `entities.py`.

### 2.5 Rule-conflict reconciliation for SECTION

Docs rules R14 ("prefer descriptive IDs"), R17 ("avoid examples like `3.1`"), and R21 ("prefer short document-derived IDs like section numbers") appear to conflict for SECTION.

The reconciliation: **R14/R17 apply to LLM-emitted identities** (the intent is to steer model output via examples). **R21 applies to system-constructed identities** (section numbers ARE listed as preferred). SECTION is populated deterministically by the Docling anchor walker; the LLM never sees SECTION's `section_number` field. Therefore R21 governs, and `section_number` with positional-enumeration examples (`["1", "1.1", "2.3.4"]`) is correct.

Captured in a `# Docs rule note:` comment on the SECTION class in entities.py, referencing this section.

### 2.6 Per-entity Chunk B task list

For the implementation plan's Chunk B, this is the per-entity rewrite task list. 27 entities need structural changes (15 give-identity + 12 demote) + 17 touch-ups. Ordered by dependency (entities referenced by others come first):

**Drops (2 tasks):**
1. Delete `AssertionEntity` + remove all references.
2. Delete `SpreadsheetEntity` + merge into `DocumentEntity.source_type` enum.

**Give-identity batch (15 tasks):**
3. `DocumentEntity.graph_id_fields=["document_id"]` required.
4. `SectionEntity.graph_id_fields=["section_number"]` required.
5. `FigureEntity.graph_id_fields=["figure_ref"]` required; `figure_label` added as optional.
6. `TableEntity.graph_id_fields=["table_ref"]` required; `table_label` added as optional.
7. `OrganizationEntity.graph_id_fields=["name"]` required.
8. `StandardEntity.graph_id_fields=["designation"]` required.
9. `EquipmentSystemEntity.graph_id_fields=["name"]` required.
10. `ComponentEntity.graph_id_fields=["part_number"]` required.
11. `AssemblyEntity.graph_id_fields=["assembly_number"]` required.
12. `CapabilityEntity.graph_id_fields=["capability_name"]` required.
13. `ProcedureEntity.graph_id_fields=["name"]` required, scope=document.
14. `FailureModeEntity.graph_id_fields=["name"]` required, scope=document.
15. `TestEventEntity.graph_id_fields=["name"]` required.
16. `ForceStructureEntity.graph_id_fields=["name"]` required.
17. `SubsystemEntity.graph_id_fields=["name"]` required, scope=document.

**Demote batch (12 tasks):**
18. `SpecificationEntity`: `is_entity=False`.
19. `ModulationEntity`: `is_entity=False`.
20. `RfSignatureEntity`: `is_entity=False`.
21. `RfEmissionEntity`: `is_entity=False`.
22. `ScanPatternEntity`: `is_entity=False`.
23. `IfAmplifierEntity`: `is_entity=False`.
24. `MissilePerformanceEntity`: `is_entity=False`.
25. `MissilePhysicalCharacteristicsEntity`: `is_entity=False`.
26. `PropulsionStackEntity`: `is_entity=False`.
27. `PropulsionStageEntity`: `is_entity=False`.
28. `RadarPerformanceEntity`: `is_entity=False`.
29. `EngagementTimelineEntity`: `is_entity=False`.

**Touch-up batch (aggregated, 3 tasks — batched not per-entity):**
30. All 17 unchanged entities: apply R16/R17 example-list cleanup (2-5 distinct, no duplicated examples). One commit.
31. All entities: confirm non-identity fields are `Optional[T] = None`; flip any that aren't. One commit.
32. All entities with `edge(label=...)` fields: extend `edge()` helper to accept `description` + `examples` kwargs, update every call site. One commit.

**Chunk B total: 32 tasks, 32 commits.** (Up from the earlier 18 estimate.)

## 3. Docling-derived anchors — architecture

### 3.1 New Celery task

File: `app/workers/pipeline.py`
Task: `derive_document_anchors(document_id, run_id)` with queue `graph` (co-located with `derive_ontology_graph`).

### 3.2 Position in Celery chain

Between `derive_image_embeddings` and `derive_ontology_graph`:

```
prepare → translate → metadata → purge → picture_descriptions
  → text_chunks → image_embeddings
  → [NEW] derive_document_anchors
  → derive_ontology_graph → collect → structure_links → canonicalization → finalize
```

Rationale:
- Runs after purge (no concurrency with cleanup).
- Runs before `derive_ontology_graph` so downstream merge can observe anchor nodes.
- Runs before `derive_structure_links` so `EXTRACTED_FROM` edges from TextChunks have live SECTION/FIGURE/TABLE target vertices.

### 3.3 Walker logic

Inputs:
- `docling_document.json` from MinIO (via existing `_build_docling_document_json`).
- Document UUID (for `DocumentEntity.document_id`).

Outputs:
- 1 `DocumentEntity` with `document_id = UUID`.
- N `SectionEntity` records — one per unique `section_path` encountered (deduplicated across elements sharing that path).
- M `FigureEntity` records — one per entry in `docling_document.pictures`.
- K `TableEntity` records — one per entry in `docling_document.tables`.
- Edges: `(DOCUMENT)-[:HAS_SECTION]->(SECTION)`, `(DOCUMENT)-[:HAS_FIGURE]->(FIGURE)`, `(DOCUMENT)-[:HAS_TABLE]->(TABLE)`, `(SECTION)-[:CHILD_OF]->(SECTION)`.

#### Deterministic algorithm (pseudocode)

```python
def walk(docling_doc: dict, document_uuid: str) -> MergedExtraction:
    doc_entity = DocumentEntity(document_id=document_uuid, ...)

    # --- Sections: dedup + enumerate by document-order first-occurrence --------
    # Walk texts (and any other body descendants with prov) in document order.
    # For each element, read prov[0].section_path — a list of heading-stem strings
    # that Docling assigns as the element's ancestor chain.
    # First occurrence of each unique tuple-path establishes a SECTION.
    # section_number is the 1-based positional enumeration within the parent path.

    section_by_path: OrderedDict[tuple[str, ...], SectionEntity] = OrderedDict()
    sibling_counters: dict[tuple[str, ...], int] = defaultdict(int)

    for elem in iter_body_descendants_in_order(docling_doc):
        path_tuple = tuple(prov_section_path(elem))  # () for root-level elements
        if path_tuple and path_tuple not in section_by_path:
            parent_tuple = path_tuple[:-1]
            sibling_counters[parent_tuple] += 1
            idx = sibling_counters[parent_tuple]
            if parent_tuple and parent_tuple in section_by_path:
                parent_number = section_by_path[parent_tuple].section_number
                section_number = f"{parent_number}.{idx}"
            else:
                section_number = str(idx)
            section_by_path[path_tuple] = SectionEntity(
                section_number=section_number,
                heading=path_tuple[-1],  # descriptive-only; not identity
                ...,
            )

    # --- Figures: one per Docling self_ref --------------------------------------
    figures = [
        FigureEntity(
            figure_ref=item["self_ref"],
            figure_label=extract_caption_label(item),  # "Figure 3-12" or None
            ...,
        )
        for item in docling_doc.get("pictures", [])
    ]

    # --- Tables: one per Docling self_ref ---------------------------------------
    tables = [
        TableEntity(
            table_ref=item["self_ref"],
            table_label=extract_caption_label(item),
            ...,
        )
        for item in docling_doc.get("tables", [])
    ]

    # --- Edges ------------------------------------------------------------------
    edges = []
    for section in section_by_path.values():
        edges.append(MergedEdgeRecord(from=doc_entity, to=section, label="HAS_SECTION"))
    for section in section_by_path.values():
        if "." in section.section_number:
            parent_number = section.section_number.rsplit(".", 1)[0]
            parent = find_by_number(section_by_path, parent_number)
            if parent is not None:
                edges.append(MergedEdgeRecord(from=section, to=parent, label="CHILD_OF"))
    for fig in figures:
        edges.append(MergedEdgeRecord(from=doc_entity, to=fig, label="HAS_FIGURE"))
    for tbl in tables:
        edges.append(MergedEdgeRecord(from=doc_entity, to=tbl, label="HAS_TABLE"))

    return MergedExtraction(
        merged_entities=[doc_entity, *section_by_path.values(), *figures, *tables],
        merged_edges=edges,
    )
```

#### Determinism properties

- `iter_body_descendants_in_order`: uses Docling's own `body.children` → recursive descent via `$ref` lookups, in declared order. Identical input JSON → identical output order.
- Dedup key for SECTION: the tuple of strings from `prov[0].section_path`. Two elements that share that exact tuple attach to the same SECTION.
- `section_number` tie-breaking: document-order first-occurrence. If `section_path=["A", "B"]` appears at elem 10 and `section_path=["A", "C"]` appears at elem 20, then B gets number "1.1" and C gets "1.2" (assuming A is "1").
- Parent-child edge resolution: by parsed `section_number` prefix (splitting on `"."`), not by raw path. Deterministic because `section_number` is assigned by the algorithm above.
- `figure_ref` / `table_ref` are Docling's own refs — globally unique within a single `DoclingDocument`.

### 3.4 Fallback for weak structure

If `section_path` is completely absent for every element in the DoclingDocument (rare — Docling emits at minimum a body root), emit one `SectionEntity(section_number="0")` representing the whole document body. All TextChunks attach to this anchor. Prevents zero-SECTION docs.

### 3.5 Write path

Uses the existing `graph_store.upsert_nodes_batch_sync` + `upsert_relationships_batch_sync`. Identity resolution is automatic because these helpers consult `graph_id_fields` from the entity's `model_config`.

Creates a `DocumentGraphExtraction` audit row with `pass_name="document_anchors"` for bookkeeping parity with other passes.

### 3.6 Manifest change

Remove the `reference` pass from `ontology_bundles/air_defense_v3/manifest.yaml`. Remaining passes: `radar_domain`, `missile_domain`, `other_systems`, `system_links`.

### 3.7 File deletions

- `ontology_bundles/air_defense_v3/extraction_schemas/reference.py` — deleted.

### 3.8 Testing

Unit tests in `tests/unit/test_docling_anchor_walker.py`:
- Given a fixture `docling_document.json`, verify SECTION/FIGURE/TABLE records emit with exact identities.
- Verify `CHILD_OF` edges wire up for nested sections.
- Verify fallback emits single `section_number="0"` when structure is absent.
- Deterministic; no LLM mocks.

## 4. Schema patterns

### 4.1 Lenient field coercers (R10)

`ontology_bundles/air_defense_v3/validators.py` updates:

```python
import logging
logger = logging.getLogger(__name__)

def coerce_optional_int(value) -> int | None:
    # ... existing coerce attempts ...
    logger.warning("coerce_optional_int: unrecoverable input %r → None", value)
    return None
```

Same update pattern for `coerce_optional_float`, `coerce_optional_confidence`.

New helper: `coerce_identity_str(value) -> str | None` — strips, rejects multi-line, rejects empty-after-strip. Applied via `@field_validator` on every required `graph_id_fields` string field.

### 4.2 Pass-root `mode="after"` dedup (R20)

The existing `_dedupe_entities_by_identity` model_validator is hoisted from `extraction_schemas/reference.py` into `validators.py` as shared code. Converted from the earlier `mode="before"` drop-and-dedup to `mode="after"` pure dedup (per docs L16977).

No pass-root drop-invalid-identity validator is added. Library-level identity_filter + quality_gate handle bad LLM output for us (R22).

### 4.3 Identity field examples (R16, R17)

All `examples=[...]` lists on identity fields:
- 2–5 distinct values.
- Short.
- Document-derived style for LLM-emitted identities (e.g. `examples=["AN/MPQ-65", "AN/TPY-2", "AN/SPY-6"]` for radar nomenclature).
- Positional for Docling-derived identities (e.g. `examples=["1", "1.1", "2.3.4"]` for section_number).
- No duplicates (catches current `examples=[42, 42]` anti-pattern).

### 4.4 Optional non-identity fields (R19)

Every field NOT in `graph_id_fields` becomes `Optional[T]` with `default=None`. Only identity fields are required. Eliminates the docling-graph salvage-bug hit (library fills missing required-int with `""`, which fails coercion) — there are no required non-identity fields to salvage.

### 4.5 Component parent re-attachment

12 entities demoting to `is_entity=False` stay attached to their natural parent entities via existing `edge(label=...)` fields in `entities.py`. No edge rewiring needed; only the `is_entity=False` flip + `graph_id_fields` removal on each demoted class.

Affected parent entities (no edge changes, just receive components):
- RADAR_SYSTEM receives MODULATION, RF_SIGNATURE, RF_EMISSION, SCAN_PATTERN, IF_AMPLIFIER, RADAR_PERFORMANCE, ENGAGEMENT_TIMELINE, SPECIFICATION.
- MISSILE_SYSTEM receives MISSILE_PERFORMANCE, MISSILE_PHYSICAL_CHARACTERISTICS, PROPULSION_STACK, PROPULSION_STAGE, SPECIFICATION.
- EQUIPMENT_SYSTEM, WEAPON_SYSTEM, AIR_DEFENSE_ARTILLERY_SYSTEM etc. receive SPECIFICATION + SUBSYSTEM (SUBSYSTEM stays entity per §2.2).

### 4.6 Library defense tuning (R22)

Pipeline config set:
- `delta_identity_filter_enabled = True` (default, keep on).
- `delta_identity_filter_strict = False` (default, keep off — want real document-derived values, not just allowlisted examples).
- `delta_quality_min_instances` — tune per pass. Default 20 is too strict for short docs. Set to 3 for domain passes (radar_domain/missile_domain/other_systems), 1 for system_links.

Config lives in `docker/docling-graph/repo/docling_graph/cli/config_builder.py` or passed per-request via `/extract-pass`.

### 4.7 Extraction schema rewrites

`extraction_schemas/radar_domain.py`, `missile_domain.py`, `other_systems.py`:
- Match new canonical identity shapes (single short identity field per entity).
- Import shared `_deduplicate_by_identity` helper from `validators.py`.
- Remove local duplicated dedup code.
- All examples lists rewritten per 4.3.
- All non-identity fields become Optional per 4.4.
- Components (demoted entities) carry `is_entity=False` in model_config, no `graph_id_fields`.

`extraction_schemas/system_links.py`: minimal change — Decision-4 DTO pattern preserved. Examples cleaned up.

`extraction_schemas/reference.py`: **deleted**.

## 5. Migration (full purge + re-ingest)

### 5.1 Pre-migration state to wipe

**Postgres — explicit per-table action** (ordered by FK-dependency for truncate):

| Table | Action |
|---|---|
| `ingest.stage_runs` | TRUNCATE |
| `ingest.pipeline_runs` | TRUNCATE |
| `ingest.document_graph_extractions` | TRUNCATE |
| `ingest.document_elements` | TRUNCATE |
| `ingest.artifacts` | TRUNCATE |
| `retrieval.chunk_links` | TRUNCATE |
| `retrieval.text_chunks` | TRUNCATE |
| `retrieval.image_chunks` | TRUNCATE |
| `retrieval.chunks_legacy` | TRUNCATE |
| `retrieval.community_runs` | TRUNCATE |
| `governance.feedback` | TRUNCATE |
| `governance.patch_approvals` | TRUNCATE |
| `governance.patch_events` | TRUNCATE |
| `governance.patches` | TRUNCATE |
| `governance.query_profile_registries` | TRUNCATE |
| `governance.trusted_data_submissions` | TRUNCATE |
| `ontology.entity_types` | TRUNCATE |
| `ontology.relationship_types` | TRUNCATE |
| `ontology.versions` | TRUNCATE |
| `ingest.documents` | UPDATE pipeline_status='PENDING', pipeline_stage=NULL, failed_stages=NULL, error_message=NULL, celery_task_id=NULL (preserve rows) |
| `ingest.watch_logs` | TRUNCATE |
| `ingest.sources` | PRESERVE |
| `ingest.watch_dirs` | PRESERVE |
| `auth.users`, `auth.user_roles` | PRESERVE |
| `public.alembic_version` | PRESERVE |

Any tables added to the schema after this spec is written get TRUNCATE by default; the migration script reads `pg_tables` and truncates anything not in the preserve-list. A dry-run flag prints the list before executing.

**ArcadeDB:** DROP + recreate schema.

**MinIO:** empty `derived/*` bucket; preserve `originals/*`.

**Redis:** `FLUSHALL` to clear Celery queues.

### 5.2 Migration script

`scripts/full_purge_and_reingest.py`:

1. Safety flag check (`--i-understand-this-deletes-derived-data`).
2. Stop worker + beat containers.
3. Truncate Postgres per 5.1 list.
4. DROP + recreate ArcadeDB schema via `arcadedb_schema.ensure_schema`.
5. Empty MinIO derived bucket.
6. FLUSHALL Redis.
7. Apply `alembic upgrade head`.
8. Restart worker + beat.
9. For each doc in `ingest.documents`: reset `pipeline_status='PENDING'`, clear stage + error fields.
10. Enqueue the full pipeline chain for each document.
11. Poll until all pipelines reach terminal status.
12. Emit `/tmp/migration-report-{timestamp}.md`.

### 5.3 Migration report structure

- Per-doc: UUID, filename, final `pipeline_status`, per-pass `yield_status`, entity counts by type, edge counts by label, `extraction_quality` (ok/degraded/anomaly).
- Summary: total docs, counts by outcome, any FAILED stages with error snippets.

### 5.4 Acceptance gate

- All 21 docs reach `COMPLETE` or `PARTIAL_COMPLETE` (not `FAILED`).
- ≥3 radar/missile-heavy docs produce `extraction_quality="ok"`.
- Every doc produces ≥1 SECTION, ≥1 TextChunk, and a DOCUMENT node (deterministic anchor-walker guarantee).
- Zero `page_start: Input should be a valid integer`-style errors in docling-graph logs.

If the gate fails, root-cause before merging the branch.

### 5.5 Rollback

Single-branch design. Git-revert + re-run the same migration script on reverted code returns to pre-plan state in ~15–25 min.

## 6. Consumer updates

### 6.1 Hardcoded entity-type lists — complete inventory

Grep of `app/`, `frontend/src/` (excluding tests, migrations, `__pycache__`, `node_modules`) for hardcoded entity-type name strings returns the following concrete hits. Each gets an update in Chunk E.

| File:Line | Current string | Action |
|---|---|---|
| `frontend/src/components/GraphExplorer.tsx:27` | `"SUBSYSTEM"` | Keep (SUBSYSTEM stays entity per §2.2). |
| `frontend/src/components/GraphExplorer.tsx:29` | `"SPECIFICATION"` | Remove — SPECIFICATION demotes to component. |
| `frontend/src/components/GraphExplorer.tsx:33` | `"ASSERTION"` | Remove — entity dropped. |
| `app/services/dossier_service.py:40` | `"RF_EMISSION"` in a `_RF_ENTITY_TYPES` list | Keep (still valid entity_type for query); verify list still queries correctly since RF_EMISSION is now a component. |
| `app/services/dossier_service.py:42–63` | `"MODULATION"`, `"RF_SIGNATURE"`, `"SCAN_PATTERN"`, `"IF_AMPLIFIER"`, `"SPECIFICATION"`, `"RADAR_PERFORMANCE"`, `"ENGAGEMENT_TIMELINE"`, `"MISSILE_PERFORMANCE"`, `"MISSILE_PHYSICAL_CHARACTERISTICS"`, `"PROPULSION_STACK"`, `"PROPULSION_STAGE"`, `"SPECIFICATION"` (duplicated) | Audit: these lists filter dossier sections by ontology_name. Components still have ontology_name; the filters still match. Verify Cypher query emits component-kind vertices correctly. Remove the duplicated `"SPECIFICATION"` entry. |
| `app/services/query_profiles.py:51–74` | Same set as dossier_service.py | Same audit. |
| `app/services/arcadedb_graph.py:2020–2026` | Docstring listing SECTION/FIGURE/TABLE/ASSERTION/WAVEFORM/... as document-scoped entity classes | Drop ASSERTION, update identity-scope list to match new canonical. |

**Audit note for query_profiles + dossier_service:** these files' entity-type filters are load-bearing for retrieval. Demoting SPECIFICATION/MODULATION/etc. to components changes how they appear in ArcadeDB vertex classes. Components are still stored as vertices (with content-based dedup), so filter-by-ontology_name still works. **Verification step in Chunk E**: write a small integration test that executes the dossier's filtered query against a re-ingested doc and confirms non-zero vertex matches for each of these ontology_names. If any filter fails, the fix is either (a) the filter's entity_type enumeration needs updating, or (b) the component's vertex class name in ArcadeDB differs from expected (unlikely; `arcadedb_schema` keys vertex class off `ontology_name`).

### 6.2 Identity-field references

A focused grep for attribute reads `\.heading\b`, `\.figure_id\b`, `\.table_id\b`, `\.page_start\b`, `\.assertion_text\b` across `app/` and `frontend/src/` (excluding tests, migrations, `__pycache__`, `node_modules`) returned **zero hits**. No consumer code currently reads these as entity attributes. The identity fields only exist inside `entities.py`, `extraction_schemas/reference.py`, and test files.

- `app/services/extraction_merge.py:305` — change `_NAME_LIKE_KEYS` tuple's `"heading"` → `"section_number"`.
- `ontology_bundles/air_defense_v3/extraction_schemas/reference.py` — deleted entirely per §3.7.
- Test files updated as part of Chunk B/C schema rewrites.

### 6.3 Retrieval / dossier / query profiles

Most query paths filter by `ontology_name` which is unchanged. The concrete consumer surface is documented in §6.1 above — the two entity-type lists in `dossier_service.py` and `query_profiles.py`.

No `.heading` / `.figure_id` / `.table_id` / `.page_start` identity-field attribute reads exist (confirmed in §6.2 grep). All retrieval consumers query by `ontology_name`.

### 6.4 Derive-rules + structure-links

`app/services/derive_rules.py :: derive_structural_edges`:
- Lookup SECTION/FIGURE/TABLE vertices by new single-field identities.

`app/workers/pipeline.py :: derive_structure_links`:
- Remove DOCUMENT-node creation (now done upstream by `derive_document_anchors`).
- SAME_SECTION edge resolution keys on new SECTION identity.

### 6.5 Canonicalization

`app/services/canonicalization.py` — verified via grep: contains no `"SPECIFICATION"` or other demoted-entity-name hardcoded references. The canonicalization pathway is ontology-agnostic (it canonicalizes entity names generically), so demotions don't break it. No change required beyond what the re-ingest naturally produces.

### 6.6 Pipeline config audit

Verify `delta_identity_filter_enabled` and `delta_quality_min_instances` are set as §4.6 specifies in whatever config-builder path the worker's `/extract-pass` request uses.

### 6.7 Tests updated

Every test constructing old-identity entities updated to new shape. Parity tests deleted (§7.2).

## 7. Contract tests + xfail resolution

### 7.1 Remaining xfails (both resolved by this plan)

- `test_identity_fields_have_examples` — resolved when `GuidanceMethodEntity.guidance_type` gains `examples=["COMMAND", "SARH", "ARH"]` during the entities.py rewrite.
- `test_descriptions_and_examples_on_extraction_relevant_fields` — resolved when canonical rewrite adds description + examples to every field. `edge()` helper extended with optional `description` + `examples` kwargs that forward to `Field`.

### 7.2 Parity tests deleted

Tests introduced by the YAML→Pydantic migration (current plan) compared Pydantic introspection to the frozen `tests/fixtures/ontology/air_defense_v3_snapshot.yaml`. After canonical rewrite the fixture has no valid oracle. Complete list of parity tests to evaluate for deletion (actual files confirmed via `ls tests/unit/*parity*.py`):

| File | Disposition |
|---|---|
| `tests/unit/test_arcadedb_schema_ontology_source_parity.py` | Delete (YAML vs Pydantic parity). |
| `tests/unit/test_canonicalization_ontology_source_parity.py` | Delete. |
| `tests/unit/test_dossier_service_ontology_source_parity.py` | Delete. |
| `tests/unit/test_extraction_merge_ontology_source_parity.py` | Delete. |
| `tests/unit/test_graph_store_ontology_source_parity.py` | Delete. |
| `tests/unit/test_main_api_ontology_source_parity.py` | Delete. |
| `tests/unit/test_pipeline_ontology_source_parity.py` | Delete. |
| `tests/unit/test_query_profiles_ontology_source_parity.py` | Delete. |
| `tests/unit/test_relationships_parity.py` | Delete. |
| `tests/unit/test_validation_matrix_parity.py` | Delete. |
| `tests/unit/test_introspect_entity_types.py` | Delete. |
| `tests/unit/test_introspect_ontology_dict.py` | Delete. |
| `tests/unit/test_introspect_relationship_types.py` | Delete. |
| `tests/unit/test_introspect_validation_and_weights.py` | Delete. |
| `tests/unit/test_ontology_source_flag.py` | Delete (ONTOLOGY_SOURCE is a no-op post-Task-51). |
| `tests/unit/test_arcadedb_schema.py` | Keep file. Remove YAML-comparison assertions only; keep schema-creation assertions. |
| `tests/fixtures/ontology/air_defense_v3_snapshot.yaml` | Delete. |

**15 full test-file deletions + 1 in-place edit + 1 fixture deletion.** Each deleted file gets its own task in Chunk F so the diff stays reviewable.

### 7.3 New contract tests

Split existing `tests/unit/test_docs_compliance_contracts.py` (~500 lines) into:
- `tests/unit/contracts/test_identity_contract.py`
- `tests/unit/contracts/test_component_contract.py`
- `tests/unit/contracts/test_extraction_schema_contract.py`

Add 10 new tests (R-rule references link to docs):

| Test | Rule(s) | Assertion |
|---|---|---|
| `test_entity_has_identity_or_is_component` | R1, R3 | Every `is_entity=True` model has non-empty `graph_id_fields`; every `is_entity=False` has absent/empty. |
| `test_identity_fields_are_required` | R18 | Every `graph_id_fields` field is `...` required. |
| `test_identity_field_examples_are_short` | R2, R17, R21 | Every identity-field example ≤ 80 chars, no `\n`. |
| `test_identity_fields_not_named_heading_or_title` | R21 | No `graph_id_fields` includes `heading`/`title`/`caption`/`description`. |
| `test_identity_examples_are_distinct` | R16 | Identity examples list has no duplicates. |
| `test_non_identity_fields_are_optional` | R19 | Every non-identity field is `Optional[T]` with `default=None`. |
| `test_edge_fields_have_edge_label` | R4 | Every `List[Entity]`/`Optional[Entity]` field on `is_entity=True` model has `json_schema_extra["edge_label"]`. |
| `test_no_nested_property_dicts` | R11 | Non-edge, non-BaseModel properties are primitive or `list[primitive]`. |
| `test_component_fields_attached_via_edge_helper` | R3 | Components used in catalog paths attach via `edge(label=...)` on parent entities. |
| `test_identity_example_values_populated_for_library_filter` | R22 | Every `is_entity=True` model has ≥2 examples on every identity field. |

### 7.4 New anchor-walker tests

`tests/unit/test_docling_anchor_walker.py`:
- 6–8 tests with fixture `docling_document.json` inputs of varying complexity.
- Assert SECTION/FIGURE/TABLE records with exact identities + hierarchy edges.
- Assert fallback-emit when `section_path` is absent.

### 7.5 Acceptance gate for test suite

- All 19 contract tests pass (**9 existing** in `test_docs_compliance_contracts.py` — count verified via `grep -c "^def test_"` — **+ 10 new**); 0 xfails.
- All 15 parity test files + snapshot fixture deleted.
- Anchor walker tests pass.
- Full `pytest tests/unit/` run passes, zero failures.

## 8. Plan chunks and task count estimate

| Chunk | Theme | Tasks |
|---|---|---|
| A | Prep: 10 new contract tests (added xfailed), `edge()` helper extension, lenient-coercer logging, pipeline-config knobs | 6 |
| B | Canonical `entities.py` rewrite — 2 drops + 15 give-identity + 12 demote + 3 batched touch-ups (see §2.6) | 32 |
| C | Extraction schemas rewrite + manifest change + reference.py delete | 5 |
| D | Docling anchor walker + new worker task + fixtures + tests | 4 |
| E | Consumer updates — derive_rules, structure_links, arcadedb_graph, frontend (3 lines), extraction_merge (1 line), dossier_service filter lists, query_profiles filter lists, canonicalization verification | 8 |
| F | Test cleanup — 15 parity test deletions + 1 in-place edit + 1 fixture delete + un-xfail 2 contract tests + verify 19/19 | 6 |
| G | Migration — write script, dry-run, execute on 21-doc corpus, produce report, acceptance gate | 4 |

**Total: ~65 tasks, ~65 commits, ~12–20 days of execution.**

Sequencing: A → B → C → D, E after B/C/D, F after E, G last. Chunk B's 32 tasks can technically parallelize by entity but per-entity sequential commits keep diffs reviewable.

## 9. Risks + mitigations

| Risk | Mitigation |
|---|---|
| 29 canonical entity rewrites + 3 batched touch-ups cause reviewer fatigue | One-entity-per-commit discipline in Chunk B. Each commit is small and individually reviewable. Grouped batches (touch-ups 30–32) land as three small commits. |
| Docling's `section_path` sparse on some docs → zero SECTIONs | §3.4 fallback emits single `section_number="0"` for document body. |
| Migration on 21 docs is slow | MinIO originals preserved → no Docling re-conversion needed. ~1–3 hours LLM time total (acceptable). |
| Frontend graph visualization shifts due to component demotion | Visually different ≠ broken. Document in migration report. Frontend doesn't need explicit changes beyond ASSERTION filter. |
| Upstream docling-graph salvage bug for required int fields | Avoided by §4.4 (no required non-identity fields) + §4.6 (library identity_filter catches empty-string identities). |
| Canonicalization path for SPECIFICATION breaks with is_entity=False | §6.5 audit during implementation. Components aren't canonicalized; remove SPECIFICATION from any canonicalization registry. |

## 10. Handoff to current plan

After Chunk G's acceptance gate passes and the branch merges:

- The current plan (`2026-04-14-docling-graph-schema-compliance.md`) resumes.
- Its Task 53 (live E2E on b1b0d596) is unblocked — reference pass is deleted, domain passes produce non-zero entities on the new schemas.
- Phase 8 provenance tasks (67–72 in its tracker) execute against the new schemas without conflict; `ExtractionProvenance` is a per-instance additive field, orthogonal to identity redesign.

## 11. Docs-adherence traceability

Every design decision traces to an explicit rule in `docling-graph-docs.md`. A full audit table is in the brainstorm session's memory. Summary:

- **R1, R2, R18** — §2 identity assignments.
- **R3** — §2 component demotions.
- **R4** — §3 anchor edges use explicit `HAS_SECTION`/`HAS_FIGURE`/`HAS_TABLE`/`CHILD_OF` labels; §4.5 component attachment via `edge()`.
- **R7, R9** — §3 Docling-derived anchors (no invention).
- **R10, R19** — §4.1 and §4.4 lenient coercers + Optional non-identity.
- **R11** — §4 flat properties verified (all demoted components are primitives).
- **R14, R15, R16, R17, R21** — §2.5 reconciliation note; §4.3 example style.
- **R20** — §4.2 mode="after" dedup.
- **R22** — §4.6 library defense tuning.
