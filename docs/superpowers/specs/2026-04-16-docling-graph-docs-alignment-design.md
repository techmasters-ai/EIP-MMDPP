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

The audit examined all 48 canonical entities in `ontology_bundles/air_defense_v3/entities.py`. Below is the classification table that drives every rewrite in Chunk B.

### 2.1 Dropped from ontology (2)

| Entity | Reason |
|---|---|
| ASSERTION | Q3 decision + longstanding KNOWN ANTI-PATTERN (`graph_id_fields=["assertion_text"]` violates R2 "short identities"). Drop entirely; reintroduce later if a concrete use-case emerges with a better design. |
| SPREADSHEET | Redundant with DOCUMENT. Consolidate by adding `SPREADSHEET` to `DocumentEntity.source_type` enum. `workbook_name` and `sheet_name` become optional properties on DocumentEntity. |

### 2.2 Give proper identity (14)

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

### 2.4 Unchanged (20 entities, already docs-compliant)

PLATFORM, WEAPON_SYSTEM, RADAR_SYSTEM, MISSILE_SYSTEM, AIR_DEFENSE_ARTILLERY_SYSTEM, ELECTRONIC_WARFARE_SYSTEM, FIRE_CONTROL_SYSTEM, INTEGRATED_AIR_DEFENSE_SYSTEM, LAUNCHER_SYSTEM, FREQUENCY_BAND, WAVEFORM, ANTENNA, TRANSMITTER, RECEIVER, SIGNAL_PROCESSING_CHAIN, GUIDANCE_METHOD, SEEKER, (plus DOCUMENT/SECTION/FIGURE/TABLE/SUBSYSTEM/etc. after revision as entities — but counted separately above).

All have single-field short identities on already-extracted named concepts (`system_name`, `name`, `band_name`, `designation`, etc.).

**Totals:** 48 → 46 entities → **34 entities + 12 components** after classification.

### 2.5 Rule-conflict reconciliation for SECTION

Docs rules R14 ("prefer descriptive IDs"), R17 ("avoid examples like `3.1`"), and R21 ("prefer short document-derived IDs like section numbers") appear to conflict for SECTION.

The reconciliation: **R14/R17 apply to LLM-emitted identities** (the intent is to steer model output via examples). **R21 applies to system-constructed identities** (section numbers ARE listed as preferred). SECTION is populated deterministically by the Docling anchor walker; the LLM never sees SECTION's `section_number` field. Therefore R21 governs, and `section_number` with positional-enumeration examples (`["1", "1.1", "2.3.4"]`) is correct.

This reconciliation is captured in a `# Docs rule note:` comment on the SECTION class in entities.py, referencing the spec section.

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
- N `SectionEntity` records — one per unique `section_path` encountered. `section_number` is the positional enumeration walked in order: `"1"`, `"1.1"`, `"1.2"`, `"2"`, `"2.1"`, etc.
- M `FigureEntity` records — one per entry in `docling_document.pictures`. `figure_ref = item.self_ref`. `figure_label` pulled from Docling annotations if present.
- K `TableEntity` records — one per entry in `docling_document.tables`.
- Edges:
  - `(DOCUMENT)-[:HAS_SECTION]->(SECTION)` for each section.
  - `(DOCUMENT)-[:HAS_FIGURE]->(FIGURE)` for each figure.
  - `(DOCUMENT)-[:HAS_TABLE]->(TABLE)` for each table.
  - `(SECTION)-[:CHILD_OF]->(SECTION)` for hierarchical nesting (parent section_path is a prefix of child).

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

- Postgres: truncate all tables except `auth.*`, `alembic_version`, `ingest.sources`, `ingest.documents` (metadata + storage keys), `ingest.watch_dirs`.
- ArcadeDB: DROP + recreate schema.
- MinIO `derived/*` bucket: empty.
- MinIO `originals/*` bucket: **preserved** (source PDFs stay; no re-upload).
- Redis: `FLUSHALL` to clear Celery queues.

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

### 6.1 Hardcoded entity-type lists (must update)

| File | Change |
|---|---|
| `frontend/src/components/GraphExplorer.tsx:33` | Remove `"ASSERTION"` from filter list. |
| `app/services/arcadedb_graph.py:2020–2026` | Update `delete_extraction_layer_graph_sync` docstring — drop ASSERTION, update identity-scope entity list. |
| Any additional hardcoded list found during implementation (grep for `"ASSERTION"`, `"SPECIFICATION"`, `"SUBSYSTEM"`, `"MODULATION"` etc. across `app/`, `frontend/`) | Case-by-case. |

### 6.2 Identity-field references

- `app/services/extraction_merge.py:305` — change `_NAME_LIKE_KEYS` tuple's `"heading"` → `"section_number"`.
- All other `heading`/`page_start`/`assertion_text`/`figure_id`/`table_id` references found outside migrations/tests: updated or deleted.

### 6.3 Retrieval / dossier / query profiles

Most query paths filter by `ontology_name` which is unchanged. Verified no `.heading`/`.figure_id` identity-field reads in `app/api/v1/query_profiles.py` or `app/services/dossier_service.py`; re-grep during implementation.

### 6.4 Derive-rules + structure-links

`app/services/derive_rules.py :: derive_structural_edges`:
- Lookup SECTION/FIGURE/TABLE vertices by new single-field identities.

`app/workers/pipeline.py :: derive_structure_links`:
- Remove DOCUMENT-node creation (now done upstream by `derive_document_anchors`).
- SAME_SECTION edge resolution keys on new SECTION identity.

### 6.5 Canonicalization

`app/services/canonicalization.py`:
- Verify SPECIFICATION canonicalization path handles is_entity=False demotion (components don't canonicalize; if SPECIFICATION was keyed during canonicalization, remove it).

### 6.6 Pipeline config audit

Verify `delta_identity_filter_enabled` and `delta_quality_min_instances` are set as §4.6 specifies in whatever config-builder path the worker's `/extract-pass` request uses.

### 6.7 Tests updated

Every test constructing old-identity entities updated to new shape. Parity tests deleted (§7.2).

## 7. Contract tests + xfail resolution

### 7.1 Remaining xfails (both resolved by this plan)

- `test_identity_fields_have_examples` — resolved when `GuidanceMethodEntity.guidance_type` gains `examples=["COMMAND", "SARH", "ARH"]` during the entities.py rewrite.
- `test_descriptions_and_examples_on_extraction_relevant_fields` — resolved when canonical rewrite adds description + examples to every field. `edge()` helper extended with optional `description` + `examples` kwargs that forward to `Field`.

### 7.2 Parity tests deleted

Tests introduced in Task 44b of the current plan compared Pydantic introspection to the frozen `tests/fixtures/ontology/air_defense_v3_snapshot.yaml`. After canonical rewrite the fixture has no valid oracle. Delete:

- `tests/unit/test_introspect_entity_types.py`
- `tests/unit/test_introspect_ontology_dict.py`
- `tests/unit/test_introspect_relationship_types.py`
- `tests/unit/test_introspect_validation_and_weights.py`
- `tests/unit/test_ontology_source_flag.py`
- `tests/unit/test_relationships_parity.py`
- `tests/unit/test_validation_matrix_parity.py`
- YAML-comparison assertions within `tests/unit/test_arcadedb_schema.py` (schema-creation assertions stay)
- `tests/fixtures/ontology/air_defense_v3_snapshot.yaml`

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

- All 19 contract tests pass (9 existing + 10 new); 0 xfails.
- All parity tests deleted.
- Anchor walker tests pass.
- Full `pytest tests/unit/` run passes.

## 8. Plan chunks and task count estimate

| Chunk | Theme | Approx tasks |
|---|---|---|
| A | Prep: contract tests (xfailed), `edge()` helper extension, lenient-coercer logging, pipeline-config knobs | 6 |
| B | Canonical `entities.py` rewrite — per-entity commits | 18 |
| C | Extraction schemas rewrite + manifest change + reference.py delete | 5 |
| D | Docling anchor walker + new worker task + fixtures + tests | 4 |
| E | Consumer updates — derive_rules, structure_links, arcadedb_graph, frontend, extraction_merge, canonicalization | 6 |
| F | Test cleanup — delete parity tests, un-xfail contract tests, verify 19/19 | 3 |
| G | Migration — write script, execute on 21-doc corpus, produce report, acceptance gate | 4 |

**Total: ~46 tasks, ~46 commits, ~8–15 days of execution.**

Sequencing: A → B → C → D, E after B/C/D, F after E, G last.

## 9. Risks + mitigations

| Risk | Mitigation |
|---|---|
| 18 canonical entity rewrites cause reviewer fatigue | One-entity-per-commit discipline in Chunk B. Each commit is small and individually reviewable. |
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
