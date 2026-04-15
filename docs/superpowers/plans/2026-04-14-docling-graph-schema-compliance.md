# Pydantic as Single Source of Truth — Docs-Compliant Schema Consolidation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Pydantic the single authoritative schema for the air_defense_v3 bundle, full docs-compliant per the Docling-Graph schema-design documentation, including the documented `edge(label=...)` relationship pattern — not the current DTO-record pattern. Full parity with `ontology.yaml` (entity_types **plus** rich relationship-type metadata **plus** `scoring_weights`). Every consumer reads from Pydantic via an introspection layer. `ontology.yaml` is deleted.

**Architecture:** Three layers, kept separate:

1. **Canonical ontology registry** — `ontology_bundles/air_defense_v3/entities.py` holds one Pydantic class per ontology entity_type (48 classes). Relationships are declared as **typed fields with `edge(label=...)` between entity classes** (the docs pattern), not as DTO records in a separate `relationships` list. Each class carries an explicit, refactor-stable `model_config["ontology_name"]` (e.g. `"RADAR_SYSTEM"`). `relationships.py` holds the `RelationshipType` `str` Enum (48 values), a `RelationshipMetadata` Pydantic model per type (label, description, source_type, target_type, cardinality), `VALIDATION_MATRIX` as a frozenset of 128 triples using ontology_names (not class names), and `SCORING_WEIGHTS` dict.

2. **Introspection** — `introspect.py` walks canonical classes and emits the exact dict shape today's `load_ontology()` returns, including every key the YAML carries (`entity_types` with properties+examples, full relationship metadata, validation_matrix, scoring_weights). Parity is enforced by byte-level dict equality tests for the canonical bundle.

3. **Narrow extraction-pass templates** — the five `extraction_schemas/*.py` files are rewritten per the docs' "Template Basics" and "Relationships" sections: required imports, `edge()` helper defined identically in every file, components-before-entities ordering, `List[...]` from typing, required identity fields with ≥2 examples, typed `edge(label=...)` relationships between entity classes. The current DTO relationship classes (`RadarRelationship`, `MissileRelationship`, `OtherSystemsRelationship`, `SystemLinkRelationship`) are deleted. Extraction templates are narrow views onto canonical entities — not subclasses of the full canonical schema — so the LLM's extraction prompt stays focused (protecting extraction quality).

**Merge-path refactor (required for docs compliance):** `app/services/extraction_merge.py:merge_and_resolve` currently reads `PassResult.relationships` (a list of DTO records) at line 126. That consumption path is rewritten to traverse the extracted entity graph and harvest relationships from fields marked with `edge_label` metadata. This is the biggest single behavior-touching change in the plan.

**Controlled behavioral changes (gated by expanded verification):**
- Identity fields become required per docs ("Avoid optional identity fields in staged and delta extraction"). Not backward-compatible with prior `Optional[str] = None` identity.
- Relationship consumption path rewrites `merge_and_resolve`.
- DTO relationship classes are deleted.

These are not behind a feature flag — they land together with the extraction-template rewrites, verified by before/after delta-catalog diffs, prompt-block diffs, per-pass entity/relationship counts, `yield_status` distribution, rejection-reason counts, and merged node/edge counts by type.

**Tech Stack:** Pydantic v2 (`BaseModel`, `ConfigDict`, `Field`, typed field relationships); Python 3.11 `str` Enums; `frozenset[tuple]` for validation matrix; pytest; existing Celery worker, ArcadeDB, docling-graph library — unchanged.

**Strict docs compliance — no project-level extensions.** Where the docs and the current code disagree, the docs win. No "this is how our project does it" carve-outs.

---

## The split-brain we must eliminate

Audit of the code during planning surfaced three drifts that this plan fixes:

1. **Two identity sources.** The LLM-facing extraction contract reads identity from `model_config["graph_id_fields"]` on Pydantic. The merge-time dedup reads it from ontology.yaml `identity_fields` in `extraction_merge.py:302`. Today these happen to mostly agree because comments were hand-copied; nothing enforces it. Plan: introspection returns identity from Pydantic; YAML gets deleted.

2. **Two relationship sources.** The docs describe typed `edge(label=...)` fields between entity classes as the single source of truth for relationships. The current code stores relationships as DTO records with opaque `from_identity: dict` / `to_identity: dict` and reads them via `PassResult.relationships` at `extraction_merge.py:126`. Plan: rewrite the merge path to traverse typed edges; delete DTO classes.

3. **Rich metadata in YAML, not on Pydantic.** `ontology.yaml` carries descriptions, examples, and property schemas that never reach the LLM because `docker/docling-graph/app/main.py:491–493` loads only the Pydantic template class. Plan: move descriptions and examples onto `Field(...)` calls.

## Residual brittleness — fixed under this plan

Items surfaced during review that are real bugs, not consolidation targets, but are cheap to repair alongside this work:

1. **`app/workers/watcher.py:163`** does `task_id = start_ingest_pipeline(str(document.id))` and stores the result in `celery_task_id`. But `start_ingest_pipeline` is declared at `app/workers/pipeline.py:1283–1288` to return an `IngestDispatchResult`, not a `str`. The watcher stores a repr of the dataclass (or crashes on update depending on Pydantic coercion). Fix: consume `.celery_task_id` / `.pipeline_run_id` properly.

2. **`app/workers/pipeline.py:4347–4363`** (`derive_structure_links`) reads `graph_extraction.graph_json.get("mentions", [])` but the serializer at `pipeline.py:233` (`_serialize_for_audit`) writes only `entity_count_by_type` / `primary_total` / `bridge_total` — no `mentions`, no `nodes`. The read is always empty. Either the serializer must write mentions/nodes, or the reader must consume the counts-only shape. Fix: align the two — choose the richer shape that supports the downstream structural-edge builder, and test.

Both bugs are unrelated to the schema consolidation but would silently regress extraction quality if left in place during the rebuild. They get their own commits under Phase 7.

## Validator bug also fixed

3. **`ontology_bundles/air_defense_v3/validators.py`** `coerce_optional_int` / `coerce_optional_float` / `coerce_optional_confidence` each check `isinstance(value, bool)` with a comment saying "reject" but then return `int(value)` / `float(value)`. The opposite of reject. If the LLM emits `true` for a numeric field it becomes `1` instead of `None`. Fix: return `None`. Phase 1.

---

## Files touched

**Create:**
- `ontology_bundles/air_defense_v3/entities.py` — 48 canonical Pydantic classes with typed `edge(label=...)` relationships and stable `ontology_name` metadata.
- `ontology_bundles/air_defense_v3/relationships.py` — `RelationshipType` Enum, `RelationshipMetadata` per type, `VALIDATION_MATRIX`, `SCORING_WEIGHTS`.
- `ontology_bundles/air_defense_v3/introspect.py` — Pydantic → full-ontology-dict layer, byte-equivalent to YAML load.
- Multiple test files (`tests/unit/test_entities.py`, `tests/unit/test_relationships.py`, `tests/unit/test_pydantic_ontology_introspect.py`, per-consumer parity test files).

**Modify:**
- `ontology_bundles/air_defense_v3/extraction_schemas/reference.py`, `radar_domain.py`, `missile_domain.py`, `other_systems.py`, `system_links.py` — full docs-compliant rewrite: `edge()` helper in each file (verbatim from docs), `List[...]` from typing, components-before-entities, required identity with examples, typed `edge(label=...)` between entity classes. DTO relationship classes deleted.
- `ontology_bundles/air_defense_v3/validators.py` — fix bool-coercion; add `_normalize_enum(enum_cls, v)` helper per docs signature.
- `app/services/extraction_merge.py` — rewrite `merge_and_resolve` relationship consumption to traverse typed edges; delete `PassResult.relationships` DTO-list access path.
- `app/services/ontology_templates.py` — add `ONTOLOGY_SOURCE={yaml,pydantic}` flag on `load_ontology()`; flip default to `pydantic`; delete YAML path last.
- `app/services/extraction_merge.py`, `app/services/arcadedb_schema.py`, `app/services/arcadedb_graph.py`, `app/services/canonicalization.py`, `app/services/query_profiles.py`, `app/services/dossier_service.py`, `app/services/graph_store.py`, `app/services/ontology_bundles.py`, `app/workers/pipeline.py`, `app/main.py`, `app/api/v1/graph_store.py`, `app/api/v1/_retrieval_helpers.py`, `app/schemas/query_profiles.py` — consumer parity tests; no code changes unless introspection shape reveals a gap.
- `app/workers/watcher.py` — fix `start_ingest_pipeline` return-value handling.
- `app/workers/pipeline.py` (`_serialize_for_audit` + `derive_structure_links`) — align `graph_json` write/read shape.
- `tests/unit/test_extraction_schemas.py`, `tests/unit/test_bundle_validators.py` — updated assertions + new contract tests.

**Delete (at end of Phase 6):**
- `ontology_bundles/air_defense_v3/ontology.yaml`
- YAML-reading branch of `load_ontology()` in `ontology_templates.py`

**Unchanged:**
- `ontology_bundles/air_defense_v3/manifest.yaml` — bundle pass metadata stays YAML.
- `ontology_bundles/air_defense_v3/coverage.yaml`, `derive_rules.py` — bundle-specific rules.
- `docker/docling-graph/` — library code unchanged; service reads Pydantic templates via `load_pass_template`.

---

## Design decisions (locked — each justified against docs or reviewer guidance)

1. **Full docs compliance, no project-level extensions.** Where docs and current code disagree, docs win. Includes `edge(label=...)` between typed entity classes instead of DTO relationship records.

2. **`edge()` helper defined identically in every template file.** Docs "Template Basics → Edge Helper Function → Required Definition": *"This function must be defined identically in every template."* Three-line helper copy-pasted verbatim into each of the 5 extraction_schemas files AND `entities.py`. No shared module.

3. **Typed edges between entity classes.** Relationship-bearing fields in entity classes use `edge(label="<RELATIONSHIP_TYPE>")` with the target class as the type. Example: `RadarSystemEntity.antenna: AntennaEntity = edge(label="HAS_ANTENNA", ...)`. Required by docs "Relationships → Using the edge() Function" and "Advanced Patterns → Pattern 2: Nested List with Edges".

4. **DTO relationship classes deleted.** `RadarRelationship`, `MissileRelationship`, `OtherSystemsRelationship`, `SystemLinkRelationship` do not survive this plan.

5. **Canonical and extraction-projection layers stay separate.** `entities.py` holds full-property canonical classes for 48 entity types. `extraction_schemas/*.py` declare narrow views — each pass template has its own entity classes that `import` or `subclass-with-narrower-fields` from canonical but present a smaller surface to the LLM. This preserves extraction quality per the reviewer's finding #5.

6. **Stable entity-type keys via `model_config["ontology_name"]`**, not `class.__name__`. Ontology names (`"RADAR_SYSTEM"`, `"PLATFORM"`, etc.) are the persisted keys in ArcadeDB and `manifest.yaml`. Class names are free to change without breaking the graph. Contract test enforces presence.

7. **Required identity fields (`str`, not `Optional[str] = None`)**, with ≥2 examples per docs "Field Definitions → Identity fields" and "Schema design for staged/delta extraction". Existing `test_all_fields_optional_or_default_recursive` updated to exempt identity.

8. **Full ontology parity via introspection.** Introspection must produce a dict byte-equivalent to today's `load_ontology()` output for every key: `entity_types`, `relationship_types` (with `label`/`description`/`source_type`/`target_type`/`cardinality`), `validation_matrix`, `scoring_weights`. Not just key-set coverage — byte-level equality on a fixed seed.

9. **No identity redesigns in this plan.** `ASSERTION.identity_fields` stays `[assertion_text]` (docs-flagged as anti-pattern but corpus-grounded alternative isn't ready). `PROPULSION_STACK.identity_fields` stays `[]` (knowingly broken, same reason). Both land in Plan 2 after corpus-grounded investigation. Reviewer finding #2 accepted.

10. **No "no behavioral changes" claim.** Identity-required + typed-edge relationships + `merge_and_resolve` rewrite = controlled behavioral change gated by expanded verification. Reviewer finding #5 accepted.

11. **Verification bar raised.** Before merging Phase 5 (rewrite), capture on a fixed test doc: (a) delta-catalog bytes per pass, (b) prompt-block bytes per pass, (c) per-pass `primary_entities_extracted` / `bridge_entities_extracted` / `relationships_extracted` / `relationships_rejected` / `yield_status`, (d) merged `entity_count_by_type` / edge counts. After merge: same capture. Diff reported in the Phase 9 task. Reviewer finding #8 accepted.

12. **Loader switch point is `app/services/ontology_templates.py:load_ontology()`**, not `app/services/ontology_bundles.py`. The latter only wraps the former. Reviewer finding #7 accepted.

---

## Phases

- **Phase 1** — Bug fixes + test infrastructure. Land fixes for the two residual brittleness items (watcher + graph_json mentions) and `validators.py` bool coercion, add `_normalize_enum` helper, update the partial-safety test, add xfail-marked contract tests. YAML still authoritative.

- **Phase 2** — Build canonical `entities.py` + `relationships.py` with typed edges, stable `ontology_name` metadata, and full metadata parity. Not wired yet — just data.

- **Phase 3** — Build `introspect.py`; wire `load_ontology()` to a `ONTOLOGY_SOURCE` feature flag that returns introspection-backed data. Byte-equivalence parity test vs YAML load.

- **Phase 4** — Migrate consumers off YAML one at a time under the feature flag. Each commit has a parity test. `ONTOLOGY_SOURCE=pydantic` default flipped at the end.

- **Phase 5** — Rewrite `merge_and_resolve` relationship consumption to traverse typed edges. Delete DTO relationship classes. Extraction templates rewritten per docs with typed-edge relationships. Xfail markers removed.

- **Phase 6** — Delete `ontology.yaml` + YAML code path.

- **Phase 7** — Expanded end-to-end verification with before/after metric diffs, multi-doc regression.

---

## Chunk 1: Phase 1 — Bug fixes + test infrastructure

### Task 0: Capture expanded baseline

**Files:** observational.

- [ ] **Step 1:** Run unit suite; record pass/fail.
- [ ] **Step 2:** Capture current catalog for each pass:

```bash
docker exec eip-mmdpp-docling-graph-1 python3 -c "
from app.bundles import load_pass_template
from docling_graph.core.extractors.contracts.delta.catalog import build_delta_node_catalog
from docling_graph.core.extractors.contracts.delta.schema_mapper import build_catalog_prompt_block
for p in ('reference','radar_domain','missile_domain','other_systems','system_links'):
    tcls = load_pass_template('air_defense_v3', p)
    cat = build_delta_node_catalog(tcls)
    print(f'=== {p} ===')
    print(build_catalog_prompt_block(cat))
" > /tmp/baseline-catalog.txt
```

- [ ] **Step 3:** Capture current ontology-dict shape (YAML load):

```bash
docker compose exec worker python3 -c "
import json
from app.services.ontology_templates import load_ontology
o = load_ontology(bundle_key='air_defense_v3')
json.dump(o, open('/tmp/baseline-ontology.json','w'), default=str, sort_keys=True, indent=2)
print('bytes:', __import__('os').path.getsize('/tmp/baseline-ontology.json'))
"
```

- [ ] **Step 4:** Capture per-pass metrics by re-running `derive_ontology_graph` on one fixed doc (`b1b0d596`) and recording `primary_entities_extracted`, `relationships_extracted`, `yield_status`, rejection reasons, `entity_count_by_type`. These are the before-baseline for Phase 7 verification.

- [ ] **Step 5:** No commit — evidence only.

### Task 1: Fix `watcher.py:163` `IngestDispatchResult` handling

**Files:**
- Modify: `app/workers/watcher.py:162–171`

- [ ] **Step 1:** Write failing test asserting `celery_task_id` and `pipeline_run_id` are set correctly after watcher enqueue.
- [ ] **Step 2:** Change:

```python
task_id = start_ingest_pipeline(str(document.id))
# ...
.values(celery_task_id=task_id)
```

to:

```python
dispatch = start_ingest_pipeline(str(document.id))
# ...
.values(
    celery_task_id=dispatch.celery_task_id,
    pipeline_run_id=dispatch.pipeline_run_id,
)
```

- [ ] **Step 3:** Run the test — PASS.
- [ ] **Step 4:** Commit: `fix(watcher): consume IngestDispatchResult correctly`

### Task 2: Align `graph_json` mentions/nodes write/read

**Files:**
- Modify: `app/workers/pipeline.py` (`_serialize_for_audit` at ~233, `derive_structure_links` at ~4347)

- [ ] **Step 1:** Decide the shape. Two options: (a) have `_serialize_for_audit` write `mentions: [...]` and `nodes: [...]` so `derive_structure_links` works as-is; (b) rewrite `derive_structure_links` to consume counts-only. Choose (a) because the mentioned-entities-to-chunks edge builder needs per-mention resolution — counts-only can't express that.
- [ ] **Step 2:** Failing test: create a merged result with two entities each mentioned in two chunks; assert `graph_json["mentions"]` has 4 entries with `entity_name`/`entity_type`/`element_uid`.
- [ ] **Step 3:** Update `_serialize_for_audit` to write `mentions` + `nodes` along with existing counts.
- [ ] **Step 4:** Run test — PASS. Regression-run `derive_structure_links` on `b1b0d596` to confirm it now creates per-mention chunk edges (currently creates zero because `mentions` is `[]`).
- [ ] **Step 5:** Commit: `fix(pipeline): _serialize_for_audit writes mentions/nodes for structural-edge builder`

### Task 3: Fix bool-coercion in `validators.py`

Per earlier spec: three functions return `int(value)` / `float(value)` for bool despite "reject" docstring. Return `None`.

- [ ] Failing tests (3) + fix + commit: `fix(bundle/validators): reject bool in int/float/confidence coercion`

### Task 4: Add `_normalize_enum(enum_cls, v)` helper per docs signature

Per docs "Validation → Enum Normalization Helper". Keep existing `normalize_enum(set[str])` for back-compat until callers migrate.

- [ ] Helper + tests + commit: `feat(bundle/validators): add _normalize_enum per docs signature`

### Task 5: Update `test_all_fields_optional_or_default_recursive` to exempt identity

Identity must be required per docs; existing test forbids required fields. Exempt `graph_id_fields` members. Add companion `test_identity_fields_are_required`.

- [ ] Update + companion test + commit.

### Task 6: Contract test — entity-or-component declaration (xfail)

Every nested `BaseModel` must declare `graph_id_fields` or `is_entity=False`. Xfail until Phase 2.

- [ ] Test + commit.

### Task 7: Contract test — `edge_label` on every list-of-entity field (xfail)

**NOT just pass-root list fields.** Every `List[EntityClass]` anywhere in the schema must be declared via `edge(label=..., default_factory=list)`. Docs "Relationships" treats all multi-valued entity references as edges. Xfail until Phase 5.

- [ ] Test + commit.

### Task 8: Contract test — identity fields have ≥2 examples (xfail)

Per docs. Xfail until Phase 2.

- [ ] Test + commit.

### Task 9: Contract test — every canonical entity class declares `ontology_name`

Every class in `entities.py` (Phase 2) must declare `model_config["ontology_name"] == "<ONTOLOGY_NAME>"`. Enforces stable refactor-resistant keys per reviewer finding #6. Xfail until Phase 2.

- [ ] Test + commit.

---

## Chunk 2: Phase 2 — Canonical entities + relationships

### Task 10: `RelationshipType` Enum

`relationships.py` — `str` Enum with every ontology.yaml relationship_type (48 members). Parity test: enum value set equals YAML name set.

- [ ] Test + enum + commit: `feat(bundle): RelationshipType enum matches ontology.yaml 1:1`

### Task 11: `RelationshipMetadata` Pydantic model + per-type registry

Each relationship carries the metadata the YAML holds: `label`, `description`, `source_type`, `target_type`, `cardinality`. Define `RelationshipMetadata` model, and a `RELATIONSHIP_METADATA: dict[RelationshipType, RelationshipMetadata]` populated from YAML. Parity test asserts every YAML field matches for every type.

- [ ] Tests + model + registry + commit: `feat(bundle): RelationshipMetadata — full rel metadata parity`

### Task 12: `VALIDATION_MATRIX` frozenset (128 triples)

`frozenset[tuple[str, RelationshipType, str]]` with ontology names (not class names). Use a throwaway script `scripts/yaml_to_validation_matrix.py` to emit the Python source; paste; verify. Parity test: exact triple-set equivalence with YAML.

- [ ] Test + matrix + commit: `feat(bundle): VALIDATION_MATRIX frozenset matches ontology.yaml 1:1`

### Task 13: `SCORING_WEIGHTS` dict

`dict[str, float]` mirroring YAML `scoring_weights`, keyed by relationship name. Parity test: exact dict equality.

- [ ] Test + dict + commit: `feat(bundle): SCORING_WEIGHTS mirrors ontology.yaml 1:1`

### Task 14: `entities.py` Layer 1 — reference (6 classes)

DocumentEntity, SectionEntity, FigureEntity, TableEntity, SpreadsheetEntity, AssertionEntity. Each declares:
- `model_config = ConfigDict(ontology_name="<NAME>", graph_id_fields=[...], identity_scope=..., dodaf_parent=..., is_entity=True/False)`
- Full property set from YAML (30+ per entity in some cases), each as `Field(..., description=..., examples=[...])` with YAML content verbatim
- Required identity fields (`str`); Optional non-identity
- Docstring per docs "Docstring Standards" template
- Relationship fields using `edge(label=...)` to other entity classes (forward-ref strings for circular types)

`ASSERTION.graph_id_fields = ["assertion_text"]` — stays as-is (reviewer finding #2). Document this in the class docstring as a known anti-pattern to be fixed in Plan 2.

- [ ] Tests asserting identity + examples + ontology_name + typed edges for each class. Commit.

### Task 15: `entities.py` Layer 2 — military equipment (12 classes)

PLATFORM, RADAR_SYSTEM, MISSILE_SYSTEM, AIR_DEFENSE_ARTILLERY_SYSTEM, ELECTRONIC_WARFARE_SYSTEM, FIRE_CONTROL_SYSTEM, INTEGRATED_AIR_DEFENSE_SYSTEM, LAUNCHER_SYSTEM, WEAPON_SYSTEM, SUBSYSTEM, COMPONENT, ORGANIZATION. Same pattern as Task 14.

- [ ] Tests + classes + commit.

### Task 16: `entities.py` Layer 3 — EM/RF (11 classes)

FREQUENCY_BAND, RF_EMISSION, WAVEFORM, MODULATION, RF_SIGNATURE, SCAN_PATTERN, ANTENNA, TRANSMITTER, RECEIVER, IF_AMPLIFIER, SIGNAL_PROCESSING_CHAIN.

- [ ] Tests + classes + commit.

### Task 17: `entities.py` Layer 4 — weapon/missile (6 classes)

SEEKER, GUIDANCE_METHOD, MISSILE_PERFORMANCE, MISSILE_PHYSICAL_CHARACTERISTICS, PROPULSION_STACK, PROPULSION_STAGE.

`PROPULSION_STACK.graph_id_fields = []` — stays as-is (reviewer finding #2). Document as known issue in docstring.

- [ ] Tests + classes + commit.

### Task 18: `entities.py` Layer 5 — operational (11+ classes)

CAPABILITY, RADAR_PERFORMANCE, ENGAGEMENT_TIMELINE, FORCE_STRUCTURE, EQUIPMENT_SYSTEM, ASSEMBLY, SPECIFICATION, STANDARD, PROCEDURE, FAILURE_MODE, TEST_EVENT.

- [ ] Tests + classes + commit.

### Task 19: `ALL_ENTITIES` registry + coverage test

`entities.py` gets `ALL_ENTITIES: dict[str, type[BaseModel]]` mapping ontology_name → class. Coverage test asserts full 1:1 with YAML `entity_types`.

- [ ] Test + registry + commit: `feat(bundle): ALL_ENTITIES registry covers ontology.yaml 1:1`

---

## Chunk 3: Phase 3 — Introspection layer + feature flag

### Task 20: `introspect.build_entity_types_list`

Returns list matching YAML `entity_types` dict shape: `name`, `label`, `identity_fields`, `identity_scope`, `parent`, `description`, `properties`. Reads from `ALL_ENTITIES` + `model_config` + `model_fields`. Parity test: every YAML field deep-equal for every entity type.

- [ ] Test + builder + commit.

### Task 21: `introspect.build_relationship_types_list` — FULL metadata

Returns list matching YAML `relationship_types` shape: `name`, `label`, `description`, `source_type`, `target_type`, `cardinality` (all fields). Reads from `RELATIONSHIP_METADATA`. Byte-equivalence test.

- [ ] Test + builder + commit: `feat(bundle): introspect relationship_types — full YAML parity`

### Task 22: `introspect.build_validation_matrix_list` + `build_scoring_weights`

Two builders returning YAML-shaped output for `validation_matrix` and `scoring_weights`. Parity tests per builder.

- [ ] Tests + builders + commit.

### Task 23: `introspect.build_ontology_dict` — full parity

Composes all four builders into the full ontology dict, including `version`. **Byte-equivalence test with YAML load** via `json.dumps(..., sort_keys=True)` equality.

- [ ] Parity test + wrapper + commit: `feat(bundle): build_ontology_dict byte-equivalent to load_ontology()`

### Task 24: `ONTOLOGY_SOURCE` feature flag in `ontology_templates.py`

**Correct file: `app/services/ontology_templates.py:123` (`load_ontology`)** — reviewer finding #7 accepted.

```python
def load_ontology(bundle_key: str | None = None) -> dict:
    source = os.environ.get("ONTOLOGY_SOURCE", "yaml").lower()
    if source == "pydantic":
        from ontology_bundles.air_defense_v3.introspect import build_ontology_dict
        return build_ontology_dict()
    return _load_from_yaml(bundle_key)
```

Parametrize the full unit suite over `["yaml", "pydantic"]`. All tests pass under both.

- [ ] Flag + parametrized fixture + commit: `feat(bundle): ONTOLOGY_SOURCE flag — introspection backend wired`

---

## Chunk 4: Phase 4 — Consumer migration

Each task: (1) read the consumer, (2) write a parity test running under both sources, (3) verify behavior equivalent, (4) commit. No consumer migrates without a green parity test.

### Task 25: Migrate `app/services/extraction_merge.py` (identity + validation_matrix consumers)

Lines 297, 335, 472 read `ontology.get("entity_types", ...)`; line 360 reads `ontology.get("validation_matrix", ...)`. If `load_ontology()` returns byte-equivalent Pydantic-backed data, these call sites need no code change — parity test confirms.

- [ ] Parity test + commit.

### Task 26: Migrate `app/services/arcadedb_schema.py` + `app/services/arcadedb_graph.py` (CRITICAL)

Registers ArcadeDB vertex/edge types. Parity test: compare registered types under both sources against a fresh ArcadeDB. If mismatch, introspection has a gap.

- [ ] Parity test + verification + commit.

### Task 27: Migrate `app/workers/pipeline.py` (lines 1831, 1868, 1928, 1965)

- [ ] Parity test + commit.

### Task 28: Migrate `app/services/canonicalization.py`

- [ ] Parity test + commit.

### Task 29: Migrate `app/services/query_profiles.py`

- [ ] Parity test + commit.

### Task 30: Migrate `app/services/dossier_service.py`

- [ ] Parity test + commit.

### Task 31: Migrate `app/services/graph_store.py`

- [ ] Parity test + commit.

### Task 32: Migrate `app/api/v1/_retrieval_helpers.py` (`scoring_weights` consumer)

`_retrieval_helpers.py:47` reads `scoring_weights` — verified via Task 22's parity. Add consumer-level parity test.

- [ ] Parity test + commit.

### Task 33: Migrate `app/services/ontology_templates.py` internals + remaining files

`ontology_templates.py:175, 195, 207`, plus `app/services/ontology_bundles.py`, `app/main.py`, `app/api/v1/graph_store.py`, `app/schemas/query_profiles.py`.

- [ ] Parity tests + commits.

### Task 34: Flip `ONTOLOGY_SOURCE` default to `pydantic`

Edit default in `ontology_templates.py` + worker `docker-compose.yml` env. Rebuild container. Full suite PASS.

- [ ] Commit: `feat(bundle): flip default ONTOLOGY_SOURCE=pydantic`

---

## Chunk 5: Phase 5 — Merge-path refactor + extraction-template rewrites

### Task 35: Capture before-metrics for Phase 5 gate

Run `derive_ontology_graph` on `b1b0d596` with current (pre-refactor) code (under `ONTOLOGY_SOURCE=pydantic`). Capture:
- delta-catalog bytes per pass → `/tmp/phase5-before-catalog/<pass>.txt`
- prompt-block bytes per pass → `/tmp/phase5-before-prompts/<pass>.txt`
- per-pass metrics from `StageRun` rows → `/tmp/phase5-before-metrics.json`
- merged graph `entity_count_by_type` + edge counts → `/tmp/phase5-before-graph.json`

No commit — evidence.

### Task 36: Design + write the new `merge_and_resolve` relationship-consumption path

**Files:**
- Modify: `app/services/extraction_merge.py`

- [ ] **Step 1:** Write failing tests for the new behavior:
  - Given a pass result containing a `RadarSystemEntity` with `antenna: AntennaEntity = edge(label="HAS_ANTENNA")`, `merge_and_resolve` produces a `MergedEdgeRecord` with `rel_type="HAS_ANTENNA"`, `from_identity` = radar identity, `to_identity` = antenna identity.
  - Given a `List[FigureEntity]` under `ReferencePass.figures` with `edge_label="HAS_FIGURE"`, each figure becomes a `MergedEdgeRecord`.
  - Given two chained edges (`radar → antenna → frequency_band`), all three edges are emitted.

- [ ] **Step 2:** Implement `harvest_edges_from_entity(entity, ontology) -> Iterable[MergedEdgeRecord]`: walk `entity.model_fields`, find fields with `json_schema_extra["edge_label"]`, yield edges.

- [ ] **Step 3:** Rewrite `merge_and_resolve` to call `harvest_edges_from_entity` instead of reading `PassResult.relationships`. Keep identity dedup via `model_config["graph_id_fields"]` (already in place from Phase 2).

- [ ] **Step 4:** Run tests — PASS.
- [ ] **Step 5:** Commit: `feat(merge): harvest edges from typed entity fields per docs pattern`

### Task 37: Delete DTO relationship classes

- [ ] Delete `RadarRelationship`, `MissileRelationship`, `OtherSystemsRelationship`, `SystemLinkRelationship` from their respective `extraction_schemas/*.py` files.
- [ ] Delete `PassResult.relationships` property at `extraction_merge.py:126`.
- [ ] Run full unit suite; expect failures in consumers still referencing these. Fix each.
- [ ] Commit: `refactor(bundle): delete DTO relationship classes; typed edges are now authoritative`

### Task 38: Rewrite `extraction_schemas/reference.py` per docs

Full docs-compliant rewrite:
- Module docstring per docs "Docstring Standards" with "Key entities" / "Key relationships" template
- Required imports per docs "Standard Import Block"
- `edge()` helper defined identically (verbatim from docs)
- Components before entities per docs "Standard File Organization"
- `List[...]` from typing (not `list[...]`)
- Required identity with ≥2 examples
- Typed `edge(label=...)` between entity classes (e.g. figures + tables appear as typed edges from the root)
- `ontology_name` on every class

Imports canonical from `entities.py` where the extraction shape equals canonical; otherwise declares narrow views.

- [ ] Rewrite + tests pass for this pass + commit.

### Task 39: Rewrite `extraction_schemas/radar_domain.py` per docs

Same pattern. Typed edges between `RadarSystemEntity`, `AntennaEntity`, `ReceiverEntity`, `TransmitterEntity`, `SPCEntity`, `FrequencyBandEntity`, `WaveformEntity`, `PlatformEntity`. Each edge carries a docs-valid `RelationshipType` label (`HAS_ANTENNA`, `HAS_RECEIVER`, etc.). Narrow field lists per pass (not full canonical breadth).

- [ ] Rewrite + tests + commit.

### Task 40: Rewrite `extraction_schemas/missile_domain.py`

Same pattern. `MissileSystemEntity`, `LauncherSystemEntity`, `GuidanceMethodEntity`, `SeekerEntity`, `PropulsionStackEntity`, `PlatformEntity`. Typed edges: `HAS_GUIDANCE`, `HAS_SEEKER`, `HAS_PROPULSION`, `LAUNCHES`, etc.

- [ ] Rewrite + tests + commit.

### Task 41: Rewrite `extraction_schemas/other_systems.py`

Same. `ADAEntity`, `EWSystemEntity`, `FireControlSystemEntity`, `WeaponSystemEntity`, `IADSEntity`. Typed edges: `INSTALLED_ON`, `SPECIFIED_BY`, etc.

- [ ] Rewrite + tests + commit.

### Task 42: Rewrite `extraction_schemas/system_links.py`

Smallest. Typed-edge-only pass: `ASSOCIATED_WITH`, `CUES`. No entities of its own.

- [ ] Rewrite + tests + commit.

### Task 43: Remove xfail markers

Drop `@pytest.mark.xfail` from the four Phase 1 contract tests (entity-or-component, edge_label, identity-examples, ontology_name). Full suite runs clean.

- [ ] Commit: `test(schemas): drop xfail — schemas fully docs-compliant`

### Task 44: Capture after-metrics + Phase 5 gate

Re-run `derive_ontology_graph` on `b1b0d596`. Capture the same four artifacts as Task 35 → `/tmp/phase5-after-*`. Diff:
- Catalog: expect `ids=[none]` to drop to zero; every entity path has real identity.
- Prompt block: expect field descriptions to appear (now on `Field(description=...)`); pass block structure unchanged.
- Per-pass metrics: `primary_entities_extracted` > 0 for radar/missile/other passes on a radar/SAM document; `relationships_extracted` > 0.
- Merged graph: entity counts > 0 per type represented in the doc.

**Gate: if any metric regresses vs Task 35 baseline in a way we can't explain, STOP and investigate before Phase 6.**

- [ ] Diff report committed to `/tmp/phase5-diff.md`. Commit nothing under source control.

---

## Chunk 6: Phase 6 — Delete `ontology.yaml`

### Task 45: Delete `ontology.yaml` + YAML path in `load_ontology()`

- [ ] Full unit + integration suite with `ONTOLOGY_SOURCE=pydantic` (already default from Task 34). PASS.
- [ ] Delete `ontology_bundles/air_defense_v3/ontology.yaml`.
- [ ] Simplify `load_ontology()` to always use introspection; remove `ONTOLOGY_SOURCE` env var references.
- [ ] Run full suite — PASS.
- [ ] Commit: `refactor(bundle): delete ontology.yaml — Pydantic is the single source of truth`

---

## Chunk 7: Phase 7 — Worker quality check + expanded verification

### Task 46: Worker `_classify_extraction_quality`

Worker-side aggregate: report `degraded` when reference pass has structure but domain passes returned zero entities. Three tests + helper + wiring. Same as earlier plan variant.

- [ ] Tests + helper + commit.

### Task 47: Rebuild + full end-to-end on `b1b0d596` with expanded verification

- [ ] `docker compose build worker docling-graph && docker compose up -d`
- [ ] Purge derivations for `b1b0d596`.
- [ ] Enqueue `derive_ontology_graph`.
- [ ] Capture the same four artifacts as Phase 5 gate.
- [ ] Assert all four diffs are improvements (catalog has identity, prompt has descriptions, per-pass counts > 0, graph has nodes by type).
- [ ] Verify ArcadeDB contains nodes of `RADAR_SYSTEM`, `MISSILE_SYSTEM`, `PLATFORM` types with `document_id` = b1b0d596.
- [ ] **If any metric regresses or graph is empty, STOP. Do not proceed.**
- [ ] Commit nothing — observational.

### Task 48: Regression across multiple docs

Repeat Task 47 for 3 more docs of varying sizes. Capture per-doc metrics table. All docs must produce non-zero entities and edges.

- [ ] Metrics table saved to `/tmp/phase7-regression.md`.

### Task 49: Final summary + MEMORY update

- [ ] `git log --oneline` — expect ~48 commits.
- [ ] Optional squash to one feature-branch commit if team policy prefers.
- [ ] Update MEMORY: create `project_pydantic_ontology_ssot.md` capturing the architecture change, the DTO→typed-edge migration, and the deletion of `ontology.yaml`.
- [ ] Commit memory file.

---

## Out of scope — tracked as follow-ups

- **Plan 2 (identity redesigns):** `ASSERTION` and `PROPULSION_STACK` identity must be redesigned with corpus-grounded evidence. Separate plan.
- **`num_ctx` / model-selection:** `OLLAMA_NUM_CTX=64000` is ignored by litellm client; llama3.3:70b loads at 32K. Separate plan.
- **`derive_rules.py` + `coverage.yaml`:** bundle-specific rules; leave in place.
- **`manifest.yaml`:** bundle pass metadata; orthogonal to schema consolidation.

---

## Risk register

| Risk | Mitigation |
|---|---|
| ArcadeDB schema diverges under Pydantic vs YAML → existing graph unreachable | Task 26 parity test compares registered types against fresh ArcadeDB. |
| Validation matrix miscounts → valid edges rejected | Task 12 parity test asserts exact triple-set equivalence. |
| Entity identity drift → dedup changes | Tasks 14–18 per-entity tests + Task 25 consumer parity test pin behavior. |
| Scoring_weights missing → retrieval ranking changes | Task 13 + Task 22 parity tests enforce exact equality. |
| Rewriting `merge_and_resolve` breaks relationship extraction | Tasks 35+44 before/after metric diffs gate Phase 5. Any regression blocks merge. |
| Typed-edge schemas produce a larger LLM surface → extraction quality drops | Canonical vs extraction-projection layer separation (decision #5). Extraction templates are narrow views with only domain-relevant fields, not full canonical breadth. |
| Deleting DTO relationship classes breaks some consumer not surfaced by tests | Task 37 runs full suite expecting failures; fix each before committing. |
| `graph_json` mentions-write regresses `derive_structure_links` | Task 2 regression test on `b1b0d596` asserts non-zero chunk edges. |

---

## Acceptance criteria

1. Every consumer reads ontology data from Pydantic introspection. `ontology.yaml` does not exist.
2. Full unit + integration suite passes.
3. Phase 5 gate (Task 44): after-metrics strictly better-or-equal to before-metrics on catalog identity, prompt content, per-pass counts, graph counts.
4. Four docs through `derive_ontology_graph` (Task 48) produce `entities > 0, edges > 0` with realistic type distributions per doc.
5. Docs-compliance contract tests (entity-or-component, edge_label, identity-examples, ontology_name) all pass without `xfail`.
6. Residual brittleness items resolved: watcher consumes `IngestDispatchResult`; `_serialize_for_audit` writes `mentions` that `derive_structure_links` consumes into non-zero chunk edges.
7. `validators.py` bool coercion returns `None`, not `0` / `1`.
8. No DTO relationship classes exist anywhere.
9. `ASSERTION.identity_fields` remains `[assertion_text]` and `PROPULSION_STACK.identity_fields` remains `[]` (both docstrings flag these as Plan 2 scope).
