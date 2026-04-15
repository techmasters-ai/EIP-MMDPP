# Pydantic as Single Source of Truth — Docs-Compliant Schema Consolidation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Pydantic the single authoritative schema for the air_defense_v3 bundle. Docs-compliant **within each extraction pass** (typed `edge(label=...)` between entities, required identity, full field metadata). Full YAML parity (entity_types + rich relationship-type metadata + `scoring_weights`). Every consumer reads from Pydantic via an introspection layer. `ontology.yaml` is deleted.

**Scoped compliance claim (stated honestly):**
- **Within-pass extraction:** docs-compliant typed-edge Pydantic templates.
- **Cross-pass linking (`system_links`):** one intentional, documented non-docs-compliant exception using ref-id DTOs (`SystemLinkRelationship`). Multi-pass architecture requires it; collapsing that architecture is out of scope here and tracked as a separate follow-up.
- **Pass-root collection fields:** plain `Field(default_factory=list)`, not `edge()`. Pass-roots are wrappers, not ontology entities, so they have no `from_identity` for graph edges.
- **Known identity anti-patterns (`ASSERTION`, `PROPULSION_STACK`):** explicitly preserved as-is, with class-level docstring flags. Corpus-grounded redesign deferred to Plan 2.
- **Extraction templates stay narrow.** Canonical entities carry the full property set (30+ fields per radar etc.); extraction passes declare narrow views with only the fields the LLM can reasonably extract. Narrow views use canonical's `ontology_name` but are not full-canonical subclasses.

**Architecture:** Three layers, kept separate:

1. **Canonical ontology registry** — `ontology_bundles/air_defense_v3/entities.py` holds one Pydantic class per ontology entity_type (48 classes). Relationships are declared as **typed fields with `edge(label=...)` between entity classes** (the docs pattern), not as DTO records in a separate `relationships` list. Each class carries an explicit, refactor-stable `model_config["ontology_name"]` (e.g. `"RADAR_SYSTEM"`). `relationships.py` holds the `RelationshipType` `str` Enum (48 values), a `RelationshipMetadata` Pydantic model per type (label, description, source_type, target_type, cardinality), `VALIDATION_MATRIX` as a frozenset of 128 triples using ontology_names (not class names), and `SCORING_WEIGHTS` dict.

2. **Introspection** — `introspect.py` walks canonical classes and emits the exact dict shape today's `load_ontology()` returns, including every key the YAML carries (`entity_types` with properties+examples, full relationship metadata, validation_matrix, scoring_weights). Parity is enforced by canonical-JSON dict equality tests for the canonical bundle.

3. **Narrow extraction-pass templates** — `reference.py`, `radar_domain.py`, `missile_domain.py`, `other_systems.py` are rewritten per the docs' "Template Basics" and "Relationships" sections: required imports, `edge()` helper defined identically in every file, components-before-entities ordering, `List[...]` from typing, required identity fields with ≥2 examples, typed `edge(label=...)` relationships between entity classes. **Intra-pass DTO relationship classes (`RadarRelationship`, `MissileRelationship`, `OtherSystemsRelationship`) are deleted.** `system_links.py` gets docs-housekeeping only; its `SystemLinkRelationship` DTO is preserved as the documented multi-pass exception (Decision 4). Extraction templates are narrow views onto canonical entities — parallel classes with matching `ontology_name`, not subclasses of canonical — so the LLM's extraction prompt stays focused (protecting extraction quality).

**Merge-path refactor (required for docs compliance):** `app/services/extraction_merge.py:merge_and_resolve` currently reads `PassResult.relationships` (a list of DTO records) at line 126. That consumption path is rewritten to traverse the extracted entity graph and harvest relationships from fields marked with `edge_label` metadata. **system_links remains on the DTO path** — the merge function branches by pass: intra-pass relationships come from typed-edge harvesting; `system_links` relationships come from the existing `SystemLinkRelationship` DTO list. Both feed the same `MergedEdgeRecord` output. This is the biggest single behavior-touching change in the plan.

**Typed-edge harvester — cycle and dedup rules (two separate concerns):**
- **Cycle prevention:** visited set keyed by **Python object identity** (`id(entity_object)`). Prevents infinite recursion when the in-memory object graph contains cycles (mutual or self-references). Object identity is correct here because the same Python instance won't be visited twice; two distinct instances representing the "same logical entity" should each be traversed.
- **Edge dedup:** emitted edges deduped by `(from_logical_identity, rel_type, to_logical_identity)`. This prevents output inflation when the same logical edge is reachable via multiple traversal paths in the object graph, while still letting traversal proceed through every distinct object.

Crucially, do NOT use `(entity_identity, field_name)` as a visited key — that breaks for empty-identity entities (e.g. `PROPULSION_STACK` with `graph_id_fields=[]`, all instances would dedupe to one) and incorrectly suppresses real traversal when the same logical entity legitimately appears as the target of multiple distinct edges.

Explicit test coverage:
- (a) Self-reference cycle (entity A's field points to A).
- (b) Mutual reference cycle (entity A's field points to B; B's field points to A).
- (c) Same logical edge reachable via two object-graph paths → emitted once.
- (d) Empty-identity entity (PROPULSION_STACK shape) → multiple distinct instances each traversed; edges emitted per (from, rel, to) where to_identity is the empty tuple — output dedup collapses to one edge per source.
- (e) Same logical entity as target of multiple distinct edges → all edges emitted (dedup is on the edge tuple, not the target).

**Relationship-placement design step (new Phase 1.5):** Before `entities.py` Layers 1–5 are written, there is an explicit design task producing a Relationship Placement Table. For each of the 128 triples in `VALIDATION_MATRIX`, the table specifies: which class owns the typed field, the field name, the cardinality (`Entity` vs `List[Entity]`), and whether it's nullable. YAML provides *that* a relationship is valid; it doesn't prescribe *where* the field lives. Table is reviewed and committed as a design doc before class-definition tasks start.

**Provenance plumbing (not a token fix):** Task 1b of the earlier plan — writing `mentions` into `graph_json` — requires upstream work at the service boundary. The docling-graph service must return an additive provenance payload alongside `template_instance.model_dump()`: per-entity `ontology_name` + identity values + `element_uid` + optional page/chunk. `ExtractPassResponse` gets a new `provenance: list[ExtractionProvenance]` field. `_parse_pass_response` carries that through to `PassResult`. `_serialize_for_audit` writes real `mentions` from it. `derive_structure_links` finally produces non-zero chunk-to-entity edges. This spans the service/worker boundary; it is not a local `_serialize_for_audit` tweak.

**Risk note — `graph_json` `mentions`/`nodes` gap persists through Phases 2–7.** `_serialize_for_audit` at `app/workers/pipeline.py:233` today writes counts/rejections only. The `mentions`/`nodes` keys that `derive_structure_links` at `:4347, :4373` expects are absent, so the fallback artifact-wide path at `:4393–4420` is the ONLY active path today. The primary mention-driven path does nothing. Phase 8 (Tasks 50–55, 51b) is what activates that path by landing the full provenance pipeline. **If Phase 8 is deprioritized or slips, the plan ships a 7-phase window where the audit blob is impoverished.** Mitigations:
- **Option A (accepted):** Phase 8 is non-optional and must land in-plan. The migration is not complete without it. Listed on the Phase 8 gate (Task 54). No separate interim task — the plan is already iteration-converged around a full provenance pipeline; a minimum-viable-mentions shim would duplicate work for no correctness gain (the fallback already covers every entity artifact-wide).
- **Option B (fallback if Phase 8 slips):** the existing artifact-wide fallback continues working for entity-chunk linking. Users lose per-element mention granularity; chunk-linking coverage is preserved because the fallback keys on `artifact_id` at the artifact level, not on individual `element_uid`s.
- **No silent regression:** Phase 7's E2E re-verification (Task 47) runs `derive_structure_links` on a live document and counts EXTRACTED_FROM edges. A zero-edge result at that gate would catch the Phase 8 omission loudly, not silently.

**Controlled behavioral changes (gated by expanded verification):**
- Identity fields become required per docs ("Avoid optional identity fields in staged and delta extraction"). Not backward-compatible with prior `Optional[str] = None` identity.
- Relationship consumption path rewrites `merge_and_resolve`.
- Intra-pass DTO relationship classes are deleted (`RadarRelationship`, `MissileRelationship`, `OtherSystemsRelationship`). `SystemLinkRelationship` is preserved as the documented multi-pass exception.

These are not behind a feature flag — they land together with the extraction-template rewrites, verified by before/after delta-catalog diffs, prompt-block diffs, per-pass entity/relationship counts, `yield_status` distribution, rejection-reason counts, and merged node/edge counts by type.

**Tech Stack:** Pydantic v2 (`BaseModel`, `ConfigDict`, `Field`, typed field relationships); Python 3.11 `str` Enums; `frozenset[tuple]` for validation matrix; pytest; existing Celery worker, ArcadeDB, docling-graph library — unchanged.

**Strict docs compliance within scope.** Where the docs and the current code disagree, the docs win *within a single extraction pass*. Cross-pass linking (`system_links`) and pass-root collection containers are outside the scope the docs describe; they get explicit, named exceptions documented in the class docstrings and the plan. No hidden "project-level carve-outs" — every deviation from literal docs compliance is enumerated in the Scoped Compliance list above.

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

Both bugs are unrelated to the schema consolidation but would silently regress extraction quality if left in place during the rebuild. They get their own commits: **watcher** in Phase 1 (Task 1a); **graph_json mentions** in Phase 8 (Tasks 50–55, since the fix requires service-side provenance plumbing that doesn't exist yet).

## Validator bug also fixed

3. **`ontology_bundles/air_defense_v3/validators.py`** `coerce_optional_int` / `coerce_optional_float` / `coerce_optional_confidence` each check `isinstance(value, bool)` with a comment saying "reject" but then return `int(value)` / `float(value)`. The opposite of reject. If the LLM emits `true` for a numeric field it becomes `1` instead of `None`. Fix: return `None`. Phase 1.

---

## Files touched

**Create:**
- `ontology_bundles/air_defense_v3/entities.py` — 48 canonical Pydantic classes with typed `edge(label=...)` relationships and stable `ontology_name` metadata.
- `ontology_bundles/air_defense_v3/relationships.py` — `RelationshipType` Enum, `RelationshipMetadata` per type, `VALIDATION_MATRIX`, `SCORING_WEIGHTS`.
- `ontology_bundles/air_defense_v3/introspect.py` — Pydantic → full-ontology-dict layer, canonical-JSON-equivalent to YAML load.
- Multiple test files (`tests/unit/test_entities.py`, `tests/unit/test_relationships.py`, `tests/unit/test_pydantic_ontology_introspect.py`, per-consumer parity test files).

**Modify:**
- `ontology_bundles/air_defense_v3/extraction_schemas/reference.py`, `radar_domain.py`, `missile_domain.py`, `other_systems.py` — docs-compliant rewrite within scope: `edge()` helper in each file (verbatim from docs), `List[...]` from typing, components-before-entities, required identity with examples, typed `edge(label=...)` between entity classes. Intra-pass DTO relationship classes deleted.
- `ontology_bundles/air_defense_v3/extraction_schemas/system_links.py` — docs-housekeeping only (imports, docstring, `List[...]`, `_normalize_enum`); `SystemLinkRelationship` DTO retained per Decision 4.
- `ontology_bundles/air_defense_v3/validators.py` — fix bool-coercion; add `_normalize_enum(enum_cls, v)` helper per docs signature.
- `app/services/extraction_merge.py` — rewrite `merge_and_resolve` relationship consumption: typed-edge traversal for intra-pass; **`PassResult.relationships` DTO-list property RETAINED** for the system_links branch (Decision 4). Add `walk_entities_only()` (Task 35a) + `walk_entity_graph()` (Task 35b) for single-traversal entity+edge collection; add `PreMergeWalkSummary` dataclass (Task 34b) as the shared pre-merge carrier; add `MergedEntityRecord.provenance` aggregation (Task 52a); update `MergedEdgeRecord.source_pass: str` → `pass_origins: set[str]` and add `PerPassEdgeMetrics` carrier on `MergedExtraction` (Task 36 Step 2.dataclass).
- `app/services/ontology_templates.py` — add `ONTOLOGY_SOURCE={yaml,pydantic}` flag on `load_ontology()`; flip default to `pydantic`; delete YAML path last.
- `app/services/extraction_merge.py`, `app/services/arcadedb_schema.py`, `app/services/arcadedb_graph.py`, `app/services/canonicalization.py`, `app/services/query_profiles.py`, `app/services/dossier_service.py`, `app/services/graph_store.py`, `app/services/ontology_bundles.py`, `app/workers/pipeline.py`, `app/main.py`, `app/api/v1/graph_store.py`, `app/api/v1/_retrieval_helpers.py`, `app/schemas/query_profiles.py` — consumer parity tests; no code changes unless introspection shape reveals a gap.
- `app/workers/watcher.py` — fix `start_ingest_pipeline` return-value handling.
- `app/workers/pipeline.py` (`_serialize_for_audit` + `derive_structure_links`) — align `graph_json` write/read shape.
- `tests/unit/test_extraction_schemas.py`, `tests/unit/test_bundle_validators.py` — updated assertions + new contract tests.

**Delete (at end of Phase 6):**
- `ontology_bundles/air_defense_v3/ontology.yaml` — the authoritative source for this bundle moves to Pydantic introspection.
- `ONTOLOGY_SOURCE` env var reads inside `load_ontology()` — introspection becomes unconditional for the `air_defense_v3` default-lookup path.

**Preserved (NOT deleted despite Phase 6's YAML-removal scope):**
- **`load_ontology()`'s YAML-reading code path** stays intact. Explicit `path=<file>` calls must continue to load the referenced YAML file (test fixtures, `load_validation_matrix(path=...)` at `:186`, and any future non-introspected bundle depend on this). Only the `air_defense_v3` default-lookup path stops reading YAML; the path-based and non-default-bundle branches are unchanged. See Task 45 for the exact signature/behavior preserved.

**Unchanged:**
- `ontology_bundles/air_defense_v3/manifest.yaml` — bundle pass metadata stays YAML.
- `ontology_bundles/air_defense_v3/coverage.yaml`, `derive_rules.py` — bundle-specific rules.
- `docker/docling-graph/` — library code unchanged; service reads Pydantic templates via `load_pass_template`.

---

## Design decisions (locked — each justified against docs or reviewer guidance)

1. **Scoped docs compliance.** Docs win *within a single extraction pass* — typed `edge(label=...)` between entity classes; required identity; full field metadata. Outside that scope (cross-pass `system_links`, pass-root collection wrappers, two known identity anti-patterns), explicit named exceptions documented in code and plan, never hidden carve-outs.

2. **`edge()` helper defined identically in every template file.** Docs "Template Basics → Edge Helper Function → Required Definition": *"This function must be defined identically in every template."* Three-line helper copy-pasted verbatim into each of the 5 extraction_schemas files AND `entities.py`. No shared module.

3. **Typed edges between entity classes.** Relationship-bearing fields in entity classes use `edge(label="<RELATIONSHIP_TYPE>")` with the target class as the type. Example: `RadarSystemEntity.antenna: AntennaEntity = edge(label="HAS_ANTENNA", ...)`. Required by docs "Relationships → Using the edge() Function" and "Advanced Patterns → Pattern 2: Nested List with Edges".

4. **Intra-pass DTO relationship classes deleted; `SystemLinkRelationship` preserved.** `RadarRelationship`, `MissileRelationship`, `OtherSystemsRelationship` are replaced by typed `edge(label=...)` fields between entity classes within their pass. **`SystemLinkRelationship` stays**, because cross-pass linking operates on `from_ref_id` / `to_ref_id` pointing at entities extracted in prior passes, which the typed-edge pattern cannot express. `SystemLinkRelationship` gets a clear docstring marking it the one intentional non-docs-compliant exception, tied to the multi-pass architecture. Contract tests (edge_label, DTO-absence) carve it out explicitly.

4a. **Pass-root list fields are plain collections, not edges.** `ReferencePass.figures`, `RadarDomainPass.radar_systems`, etc. stay as `List[EntityClass] = Field(default_factory=list)`. Pass-roots are wrappers (`is_entity=False`), not ontology entities — they have no `from_identity` for an outbound edge. Edges to the `DOCUMENT` vertex are created post-merge by `derive_rules.derive_structural_edges`, not at extraction time.

5. **Canonical and extraction-projection layers stay separate.** `entities.py` holds full-property canonical classes for 48 entity types. `extraction_schemas/*.py` declare narrow extraction views — each pass template has its own entity classes declaring `model_config["ontology_name"]` matching a canonical class but exposing *only* the fields the LLM can reasonably extract (5–10 per entity). Narrow views are not subclasses of canonical; they're parallel classes tied together by the `ontology_name` key. Protects extraction-prompt size; protects extraction quality; preserves canonical as the full-property source of truth for graph/retrieval consumers.

6. **Stable entity-type keys via `model_config["ontology_name"]`**, not `class.__name__`. Ontology names (`"RADAR_SYSTEM"`, `"PLATFORM"`, etc.) are the persisted keys in ArcadeDB and `manifest.yaml`. Class names are free to change without breaking the graph. Contract test enforces presence.

7. **Required identity fields (`str`, not `Optional[str] = None`)**, with ≥2 examples per docs "Field Definitions → Identity fields" and "Schema design for staged/delta extraction". Existing `test_all_fields_optional_or_default_recursive` updated to exempt identity.

8. **Full ontology parity via introspection — canonical-JSON equality, not raw byte equality.** Introspection emits the full dict (`entity_types`, `relationship_types` with `label`/`description`/`source_type`/`target_type`/`cardinality`, `validation_matrix`, `scoring_weights`). Parity test uses a deterministic canonicalization: object keys sorted; list elements sorted by stable key (entity_types by `name`, relationship_types by `name`, validation_matrix by `(source, relationship, target)` tuple). Both YAML load and Pydantic introspection are passed through the canonicalizer; equality is asserted on the canonical bytes. Eliminates flake from `frozenset` ordering or dict iteration order.

9. **No identity redesigns in this plan.** `ASSERTION.identity_fields` stays `[assertion_text]` (docs-flagged as anti-pattern but corpus-grounded alternative isn't ready). `PROPULSION_STACK.identity_fields` stays `[]` (knowingly broken, same reason). Both land in Plan 2 after corpus-grounded investigation. Reviewer finding #2 accepted.

10. **No "no behavioral changes" claim.** Identity-required + typed-edge relationships + `merge_and_resolve` rewrite = controlled behavioral change gated by expanded verification. Reviewer finding #5 accepted.

11. **Verification bar raised.** Before merging Phase 5 (rewrite), capture on a fixed test doc: (a) delta-catalog bytes per pass, (b) prompt-block bytes per pass, (c) per-pass `primary_entities_extracted` / `bridge_entities_extracted` / `relationships_extracted` / `relationships_rejected` / `yield_status`, (d) merged `entity_count_by_type` / edge counts. After merge: same capture. Diff reported in Task 44 (Phase 5 gate) to `/tmp/phase5-diff.md`, and re-verified against the same baseline in Task 47 (Phase 7 end-to-end). There is no Phase 9 — the plan stops at Phase 8. Reviewer finding #8 accepted.

12. **Loader switch point is `app/services/ontology_templates.py:load_ontology()`**, not `app/services/ontology_bundles.py`. The latter only wraps the former. Reviewer finding #7 accepted.

---

## Phases

- **Phase 1** — Bug fixes + test infrastructure. Land fixes for the watcher (IngestDispatchResult) and `validators.py` bool coercion, add `_normalize_enum` helper, update the partial-safety test, add xfail-marked contract tests. YAML still authoritative. The `graph_json` mentions fix moves to Phase 8 (provenance plumbing).

- **Phase 1.5** — Relationship Placement Design. Walk every triple in `VALIDATION_MATRIX`; for each, decide which canonical class owns the typed `edge()` field, the field name, cardinality, nullability. Output: `docs/design/relationship-placement-table.md` reviewed and committed before any class-definition task starts.

- **Phase 2** — Build canonical `entities.py` + `relationships.py` per the placement table. Typed edges, stable `ontology_name` metadata, full metadata parity. Not wired yet — just data.

- **Phase 3** — Build `introspect.py`; wire `load_ontology()` to a `ONTOLOGY_SOURCE` feature flag. Canonical-JSON parity test vs YAML load (deterministic ordering — keys sorted, list elements sorted by stable key).

- **Phase 4** — Migrate consumers off YAML one at a time under the feature flag. `ONTOLOGY_SOURCE=pydantic` default flipped at the end.

- **Phase 5** — Rewrite `merge_and_resolve` relationship consumption: harvest typed edges from intra-pass entity graphs (cycle/dedup rules); keep `system_links` DTO branch. Delete `RadarRelationship`/`MissileRelationship`/`OtherSystemsRelationship` (NOT `SystemLinkRelationship`). Extraction templates rewritten with narrow typed-edge views. Xfail markers removed (with carve-outs for `SystemLinkRelationship` and pass-root list fields).

- **Phase 6** — Delete `air_defense_v3/ontology.yaml` (snapshot first as fixture via Task 44b). YAML-reading code path in `load_ontology()` is PRESERVED — explicit `path=` callers continue to work. See Task 45.

- **Phase 7** — Worker `_classify_extraction_quality` + expanded end-to-end verification with before/after metric diffs, multi-doc regression.

- **Phase 8** — Provenance plumbing for `graph_json` mentions. Extend `ExtractPassResponse` with additive `provenance: list[ExtractionProvenance]` payload. Service captures `element_uid` per extracted entity; worker `_serialize_for_audit` writes real `mentions`; `derive_structure_links` produces non-zero entity-chunk edges. Verified by regression test on a known-good doc.

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

- [ ] **Step 1 (verify field types before edit):** read the `Document` model around the `celery_task_id` / `pipeline_run_id` columns. Confirm:
  - `celery_task_id` is `Mapped[Optional[str]]` (or similar — accepts the string UUID from `dispatch.celery_task_id`).
  - `pipeline_run_id` exists as a column and is typed to accept the UUID from `dispatch.pipeline_run_id`.
  If either column is missing OR typed as something incompatible (e.g. declared as `int`), the plan's fix would introduce a new runtime error. In that case: add a migration for the missing/retyped column BEFORE applying the watcher fix. Document the verification outcome in the commit message.
- [ ] **Step 2:** Write failing test asserting `celery_task_id` and `pipeline_run_id` are set correctly after watcher enqueue.
- [ ] **Step 3:** Change:

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

- [ ] **Step 4:** Run the test — PASS.
- [ ] **Step 5:** Commit: `fix(watcher): consume IngestDispatchResult correctly`

### Task 2: (DEFERRED to Phase 8 — provenance plumbing)

The original "fix `_serialize_for_audit` to write `mentions`" plan is incomplete: `_serialize_for_audit` cannot synthesize `element_uid` provenance from `template_instance.model_dump()` alone. The service must return that data. See Phase 8 (Tasks 51–55) for the full work.

**No code change in Phase 1.** `derive_structure_links` continues producing zero entity-chunk edges until Phase 8 completes. This is a known degraded state for the duration of Phases 2–7.

### Task 3: Fix bool-coercion in `validators.py`

Per earlier spec: three functions return `int(value)` / `float(value)` for bool despite "reject" docstring. Return `None`.

- [ ] Failing tests (3) + fix + commit: `fix(bundle/validators): reject bool in int/float/confidence coercion`

### Task 4: Add `_normalize_enum(enum_cls, v)` helper per docs signature

Per docs "Validation → Enum Normalization Helper". Keep existing `normalize_enum(set[str])` for back-compat until callers migrate.

- [ ] Helper + tests + commit: `feat(bundle/validators): add _normalize_enum per docs signature`

### Task 5: Update `test_all_fields_optional_or_default_recursive` to exempt identity

Identity must be required per docs; existing test forbids required fields. Exempt `graph_id_fields` members. Add companion `test_identity_fields_are_required`.

- [ ] Update + companion test + commit.

**Note on contract-test scoping:** Tasks 6, 7 (edge_label), and 9c (descriptions/examples) scope their assertions to classes where `model_config.get("is_entity") is True` — **not** "classes with non-empty `graph_id_fields`". This matches Task 9b (explicit `is_entity` required on every class) and preserves empty-identity entities like `PROPULSION_STACK` as in-scope.

### Task 6: Contract test — every `is_entity=True` class declares `graph_id_fields` key (xfail)

Every class with `model_config["is_entity"] is True` must explicitly declare the `graph_id_fields` key in `model_config` — but the value MAY be an empty list (`[]`). Documented anti-pattern entities like `PROPULSION_STACK` intentionally have empty identity and must still pass this test; they just need to declare the key explicitly so the absence of identity is a deliberate choice, not an oversight.

This replaces the earlier "entity-or-component" framing, which was a proxy for the explicit `is_entity` check now handled by Task 9b. Separating the two tests keeps each concern focused: Task 9b enforces explicit entity/component classification; Task 6 enforces that entities declare their identity key (even if empty).

Xfail until Phase 2.

- [ ] Test + commit.

### Task 7: Contract test — `edge_label` on entity-to-entity fields (xfail)

**Scoped:** every `List[EntityClass]` or single `EntityClass` field declared on a class where `model_config["is_entity"] is True` must be declared via `edge(label=..., default_factory=list)` (or `edge(label=...)` for non-list). Scoped via `is_entity=True` so classes like `PROPULSION_STACK` (empty identity but a real entity) are correctly in-scope. **Carve-outs:**
- Pass-root classes (`is_entity=False`) are exempt — their list fields are containers.
- `SystemLinkRelationship` is exempt — DTO ref-id pattern is the documented multi-pass exception.

Xfail until Phase 5. Test explicitly skips wrapper/exception classes.

### Task 8: Contract test — identity fields have ≥2 examples (xfail)

Per docs. Xfail until Phase 2.

- [ ] Test + commit.

### Task 9: Contract test — every canonical entity class declares `ontology_name`

Every class in `entities.py` (Phase 2) must declare `model_config["ontology_name"] == "<ONTOLOGY_NAME>"`. Enforces stable refactor-resistant keys per reviewer finding #6. Xfail until Phase 2.

- [ ] Test + commit.

### Task 9a: Contract test — extraction views are subsets of canonical (xfail)

**The cornerstone of "Pydantic is the single source of truth" claim.** Without this, drift just moves from YAML-vs-template to canonical-vs-pass-template.

For every Pydantic class in `extraction_schemas/*.py` declaring `model_config["ontology_name"]`, find the matching canonical class in `entities.py` with the same `ontology_name`. Assert:
- `graph_id_fields` are equal (same fields, same order).
- For every field name present on both: same Python type annotation; same `description` (if any); same `examples` (if any).
- For every field with `edge_label`: the canonical class has the same field with the same `edge_label`.
- Extraction view's field set is a subset of canonical's field set (no fields on the extraction view that don't exist on canonical).
- **Shared validator parity:** for every field present on both, any `field_validator` (mode `before` or `after`) on the canonical side must have a counterpart on the extraction-view side with the same coercion/normalization function. Walk Pydantic's `__pydantic_decorators__.field_validators` on both classes; compare the validator function objects (or their qualified names) per field name. If canonical normalizes `SPECIFICATION.value` via `coerce_optional_text` but the extraction view doesn't, that is drift — fail the test.

Xfail until Phase 5. Test asserts each extraction class against its canonical peer; explicitly skips `SystemLinkRelationship` (DTO exception).

- [ ] Test + commit: `test(schemas): canonical-vs-extraction subset contract`

### Task 9b: Contract test — every class declares `is_entity` explicitly (xfail)

Per docs "Entities vs Components": the entity/component classification is a deliberate decision per class. **Do not infer non-entity status from empty `graph_id_fields`** — that breaks for intentional empty-identity entities like PROPULSION_STACK (Decision 9 anti-pattern).

The contract test walks every `BaseModel` in `extraction_schemas/*.py` and `entities.py` and asserts that `model_config["is_entity"]` is **explicitly set to `True` or `False`** on every class. No `is_entity=True` default; no inference from `graph_id_fields`. This forces every class author to make a conscious choice and prevents PROPULSION_STACK-style misclassifications when identity happens to be empty.

- [ ] Test + commit (xfail until Phase 2).

### Task 9c: Contract test — descriptions and examples on important fields

Per docs "Field Definitions": fields used for extraction must have `description=` and `examples=`. **Scope: `is_entity=True` classes only.** Required on:
- every identity field (fields listed in `graph_id_fields`) — description requirement here; Task 8 already requires ≥2 examples.
- every field with `edge_label` (the LLM needs to know what the edge represents).
- every domain-specific field on an `is_entity=True` class (excludes system fields like `confidence`).

**Exclusions:** `is_entity=False` classes (pass-roots, components, DTO records) are out of scope — their fields are wrappers or value-object properties, not extraction-driving metadata. Empty-identity entities like `PROPULSION_STACK` are still in scope (they're `is_entity=True`); their identity is empty but their other domain fields still need descriptions/examples.

Test asserts presence on all in-scope fields. Xfail until Phase 5.

- [ ] Test + commit.

### Task 9e: Contract test — `edge_label` fields target `is_entity=True` classes only (xfail)

Per docs "Entities vs Components": entities are graph endpoints; components are value-object properties embedded on their parent entity. A field with `edge_label` pointing at an `is_entity=False` class violates this model — it would either emit a component as a graph node (wrong) or drop the edge silently (loss of data).

Contract test walks every `BaseModel` in `extraction_schemas/*.py` and `entities.py`; for each field with `json_schema_extra["edge_label"]`, resolves the field's annotation to a target class; asserts `target_class.model_config["is_entity"] is True`. Skips the `SystemLinkRelationship` DTO exception.

- [ ] Test + commit (xfail until Phase 5).

### Task 9d: Contract test — pass-root list dedup is merge-preserving (xfail)

Per docs "Best Practices → Deduplicating root-level lists (chunked extraction)": pass-root list fields susceptible to chunked-extraction duplication need a dedup mechanism. **Dedup must be merge-preserving, not lossy** — if two duplicates have different non-null fields, the survivor carries the union.

**Scope: schema-layer only.** The `@model_validator` runs at Pydantic validation time inside each extraction template — it has no access to merge-layer `LogicalIdentity` or Phase-8 `ExtractionProvenance`. Keep the test schema-local. Provenance/`LogicalIdentity` concerns live in Tasks 36 / 52a / 53 (merge-layer), not here.

Dedup key at the schema layer: the tuple of values for the target class's `graph_id_fields` (from `model_config`). If the class has empty `graph_id_fields`, skip dedup (documented anti-pattern per Decision 9).

For each pass-root class with an entity-list field whose element class has non-empty `graph_id_fields`:
- **Test 1 (count collapse):** build two entity instances with identical identity values; validate pass-root; assert surviving list length is 1.
- **Test 2 (scalar union):** build two duplicates where each fills different scalar fields (e.g. dup A has `nomenclature="SA-2"` and null `radar_type`; dup B has null `nomenclature` and `radar_type="fire_control"`). Assert the surviving entry has BOTH non-null values populated.
- **Test 3 (nested entity-ref union ON PYDANTIC INSTANCE):** if duplicates have `edge_label` child lists (e.g. `antennas: List[Antenna] = edge(...)`), the surviving Pydantic instance's child list contains the union of both duplicates' children, **keyed by the child class's `graph_id_fields` values when non-empty**. **This dedup operates on the Pydantic instance's field** — the child list attribute on a `BaseModel`, before the instance ever enters the merge layer. **It does NOT write anything to `MergedEntityRecord.properties`.** Task 36 Step 2b is strictly-scalar entity-property merging at the merger layer; Task 9d here is schema-layer preservation of nested children ON THE INSTANCE, handled entirely inside a Pydantic `@model_validator`. The two passes operate at different layers on different objects; both preserve information rather than discard it; neither writes edge structure into `MergedEntityRecord.properties`.

- **Test 3b (empty-identity child entity fallback):** when the child class has **empty `graph_id_fields`** (e.g. `MissileSystem.propulsion_stacks: List[PropulsionStack] = edge(...)` where `PropulsionStack.graph_id_fields = []`), there is no schema-local key for union. Fallback rule: **skip dedup and concatenate both parents' child lists**. Each child instance survives individually — the anti-pattern propagates at the schema layer too, consistent with the empty-identity anti-pattern documented for `PROPULSION_STACK` in Decision 9. Test asserts: two `MissileSystem` duplicates with one `PropulsionStack` each → surviving `MissileSystem.propulsion_stacks` has 2 entries (not collapsed). A warning log line notes the fallback was used, tagged with the child's `ontology_name` for observability.
- **Test 4 (max confidence):** duplicates with `confidence=0.6` and `confidence=0.9` → surviving entry has `confidence=0.9`.
- **Test 5 (conflict warning):** if two duplicates have **different non-null** values for the same scalar field, the dedup path logs a WARNING and keeps first-seen. Assert via `caplog`.

**Provenance union is out of scope for Task 9d** — that's Task 52a's merge-layer responsibility, which operates on post-validation `PassResult` with provenance attached. Schema-layer dedup cannot union what the schema never saw.

Lossy dedup (Test 1 only) would pass a simple count check but silently discard extracted data. This test set ensures schema-layer richness is preserved.

- [ ] Test + commit (xfail until Phase 5).

---

---

## Chunk 1.5: Phase 1.5 — Relationship Placement Design (must complete before Phase 2)

### Task 9.5: Relationship Placement Table

**Files:**
- Create: `docs/design/2026-04-14-relationship-placement-table.md`

YAML's `validation_matrix` says *which* triples `(source, rel, target)` are valid. It does not say *where* the typed `edge(label=...)` field lives. Two implementations both "matching YAML" can place edges differently and produce different extraction catalogs and merge behavior. This task makes the placement explicit and reviewable before any class-definition work starts.

For every triple in `VALIDATION_MATRIX` (128 entries), the table specifies:

| triple `(source, rel, target)` | owning class | field name | cardinality | nullable | rationale |
|---|---|---|---|---|---|
| `(RADAR_SYSTEM, HAS_ANTENNA, ANTENNA)` | `RadarSystemEntity` | `antennas` | `List[AntennaEntity]` | yes (default empty) | a radar typically lists multiple antennas; doc text usually places them under the radar's section |
| `(MISSILE_SYSTEM, HAS_GUIDANCE, GUIDANCE_METHOD)` | `MissileSystemEntity` | `guidance` | `Optional[GuidanceMethodEntity]` | yes | typically one guidance method per missile; doc may not state it |
| ... | ... | ... | ... | ... | ... |

**Placement rules (consistent across the table):**
1. Direction follows the edge name when natural ("HAS_X" → owning class is the parent; X is the field).
2. Inverse fields are NOT added (no `Antenna.radar_system` back-pointer). Single direction; back-traversal happens at query time via the graph DB.
3. Cardinality from the doc-text norm: "one X per Y" → `Optional[X]`; "many Xs per Y" → `List[X]`.
4. Edges between non-system entities (e.g. `HAS_PROVENANCE` → DOCUMENT) are NOT extraction-time edges — they're created post-merge by `derive_rules`. Mark such triples as "post-merge" in the table; no Pydantic field is generated for them.
5. Cross-pass relationships (those that show up in `system_links` today: `ASSOCIATED_WITH`, `CUES`) — mark as "system_links DTO" in the table; no Pydantic typed-edge field. These remain in the documented DTO exception.
6. **Maximum extraction nesting depth: 3 levels.** Per docs "Best practices → Template checklist": *"Limit nesting depth (2-4 recommended)."* If placing an edge would require depth > 3 from any pass-root container, **flatten the extraction shape**: place the entity at a shallower position (e.g. as a sibling at the pass root rather than nested two levels deep). Table includes a `depth_from_root` column and explicit nesting-violation flag.

**Do not use `system_links` as a general fallback for deep nesting.** The `system_links` pass is contractually scoped to `ASSOCIATED_WITH` and `CUES` per `manifest.yaml` and cannot represent arbitrary relationship types. Using it to "split chains" for depth purposes requires expanding the pass's contract, which is out of scope for this plan.

When applying rule 6, prefer flatter shapes: an extraction template where 95% of edges are at depth 1 (entity-on-pass-root → sibling-edge) is more reliable for LLM extraction than one where edges chain three levels deep. Deep nesting concentrates schema complexity in single LLM calls and hurts extraction quality.

- [ ] **Step 1:** Generate a starting table by joining `VALIDATION_MATRIX` with the entity-type list. Output to `docs/design/2026-04-14-relationship-placement-table.md`.
- [ ] **Step 2:** For each row, fill in owning class, field name, cardinality. Apply rules above. Keep concise rationale per row.
- [ ] **Step 3:** Mark post-merge and system_links rows.
- [ ] **Step 4:** Sanity check: every `VALIDATION_MATRIX` triple appears exactly once in the table.
- [ ] **Step 5:** Commit: `docs(design): relationship placement table — input for entities.py Phase 2`

This document becomes the reference Tasks 14–18 follow when declaring typed-edge fields. If a Phase 2 task needs to deviate, the table is updated in the same commit.

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

**Critical exclusion rule:** when building `properties`, **skip fields whose `json_schema_extra` contains `edge_label`**. YAML `entity_types[*].properties` only describes *scalar* properties (e.g. `ASSERTION` has `assertion_text` and `confidence` — no relationship fields). If introspection emits typed-edge fields under `properties`, parity breaks against YAML and downstream consumers that expect scalar property dicts will see relationships leak into property iteration.

**Property-schema mapping rules (required for canonical-JSON parity — Task 23):**

YAML `entity_types[*].properties` uses a JSON-schema-like shape with **singular `example`** and optional `enum`, not Pydantic's plural `examples`:
```yaml
title: {type: string, description: "Full title of the document", example: "Operator Manual for Patriot Missile System"}
source_type: {type: string, description: "...", enum: [MANUAL, REPORT, BRIEFING, ...]}
```

Introspection must map Pydantic `FieldInfo` to this exact shape. The mapping rules:

| YAML key | Pydantic source | Rule |
|---|---|---|
| `type` | `finfo.annotation` | `str`/`Optional[str]` → `"string"`; `int`/`Optional[int]` → `"integer"`; `float`/`Optional[float]` → `"number"`; `bool` → `"boolean"`; `list[X]` → `"array"`. Optionality is carried via `default=None`; the YAML `type` stays the underlying scalar. |
| `description` | `finfo.description` | Copy verbatim; omit the key entirely if description is `None` (matches YAML which elides missing keys). |
| `example` (singular) | `finfo.examples[0]` if present | **Take the FIRST example only** — YAML carries one; Pydantic carries many. Omit the key entirely when `finfo.examples` is empty or None. Required example for identity fields (Task 21 contract) is enforced separately; this mapping rule is about format only. |
| `enum` | derived from `Enum` annotation or `normalize_enum({...})` validator's allowed set | If the field's annotation is a `str` Enum subclass, emit `enum: [member.value for member in EnumCls]` in YAML-declared order. If the field uses `_normalize_enum(EnumCls, v)` as a validator but the annotation is `str`, walk the class's `__pydantic_decorators__.field_validators` to recover `EnumCls`; same emission. Omit the key if no enum constraint exists. |
| `pattern` | from `Field(pattern=...)` if present | Regex constraint copied verbatim. Omit if absent. |

Implementation skeleton:
```python
for fname, finfo in cls.model_fields.items():
    extra = finfo.json_schema_extra or {}
    if "edge_label" in extra:
        continue  # Edges not in properties
    prop: dict[str, Any] = {"type": _pydantic_type_to_yaml(finfo.annotation)}
    if finfo.description is not None:
        prop["description"] = finfo.description
    if finfo.examples:
        prop["example"] = finfo.examples[0]          # singular, first only
    enum_values = _resolve_enum_values(cls, fname, finfo)
    if enum_values is not None:
        prop["enum"] = enum_values
    if getattr(finfo, "pattern", None):
        prop["pattern"] = finfo.pattern
    properties[fname] = prop
```

Parity tests:
- Entity with typed-edge fields (`RadarSystemEntity.antennas`): those fields do NOT appear in `properties`.
- Entity with plural Pydantic `examples=["a", "b", "c"]`: YAML emits singular `example: "a"` (first only, not a list).
- Entity with an `str`-Enum field (`DOCUMENT.source_type`): `enum: [MANUAL, REPORT, ...]` matches YAML-declared order.
- Entity with a `normalize_enum({"A","B"})` validator: `enum` resolved from the validator's class — same output shape.
- Entity with no description on a field: key elided from the emitted dict (NOT emitted as `description: null`).

- [ ] Test + builder + commit.

### Task 21: `introspect.build_relationship_types_list` — FULL metadata

Returns list matching YAML `relationship_types` shape: `name`, `label`, `description`, `source_type`, `target_type`, `cardinality` (all fields). Reads from `RELATIONSHIP_METADATA`. Canonical-JSON parity test.

- [ ] Test + builder + commit: `feat(bundle): introspect relationship_types — full YAML parity`

### Task 22: `introspect.build_validation_matrix_list` + `build_scoring_weights`

Two builders returning YAML-shaped output for `validation_matrix` and `scoring_weights`. Parity tests per builder.

- [ ] Tests + builders + commit.

### Task 23: `introspect.build_ontology_dict` — full parity

Composes all four builders into the full ontology dict, including `version`. **Canonical-JSON parity test with YAML load:** both sides passed through a deterministic canonicalizer (sort dict keys; sort list elements by stable key — entity_types by `name`, relationship_types by `name`, validation_matrix by `(source, relationship, target)`, properties dicts by key) before `json.dumps(...)` equality. Eliminates flake from `frozenset` ordering or dict iteration order.

- [ ] Parity test + wrapper + commit: `feat(bundle): build_ontology_dict canonical-JSON-equivalent to load_ontology()`

### Task 24: `ONTOLOGY_SOURCE` feature flag in `ontology_templates.py`

**Correct file: `app/services/ontology_templates.py:123` (`load_ontology`).**

The current public signature is `load_ontology(bundle_key: str | None = None, path: Path | str | None = None) -> dict`. `load_validation_matrix()` at `:186` forwards `path` through. The feature-flag wrapper MUST preserve the full resolution order: **`path` first, then `bundle_key`, then default bundle.** Hardcoding `air_defense_v3` or dropping `path` silently bypasses explicit-path loads and non-default bundles.

```python
def load_ontology(
    bundle_key: str | None = None,
    path: Path | str | None = None,
) -> dict:
    source = os.environ.get("ONTOLOGY_SOURCE", "yaml").lower()
    if source == "pydantic":
        # Pydantic introspection only applies when neither path nor a
        # non-default bundle_key is specified — an explicit path or a
        # non-canonical bundle_key signals the caller wants that specific
        # YAML source, and overriding that silently would be a regression.
        from ontology_bundles.air_defense_v3.introspect import build_ontology_dict
        # Resolve requested bundle via the existing helper so behavior matches
        # the YAML path when callers omit arguments.
        resolved_bundle = bundle_key or SYSTEM_DEFAULT_BUNDLE_KEY
        if path is None and resolved_bundle == "air_defense_v3":
            return build_ontology_dict()
        # Any other combination falls through to YAML — same loader contract
        # as today for explicit-path / non-default-bundle requests.
    return _load_from_yaml(bundle_key=bundle_key, path=path)
```

**Scope note:** Pydantic introspection only handles the `air_defense_v3` bundle (the only bundle in this repo). Other bundles keep using YAML until/unless they're migrated; the plan doesn't widen introspection scope. Tests assert the fallback-to-YAML path for:
- explicit `path=<somepath>` argument (regardless of `ONTOLOGY_SOURCE`),
- explicit `bundle_key="something_other_than_air_defense_v3"`,
- the default no-args call with `ONTOLOGY_SOURCE=pydantic` (must return introspection data).

Parametrize the full unit suite over `["yaml", "pydantic"]`. All tests pass under both.

- [ ] Flag + signature-preserving wrapper + parametrized fixture + commit: `feat(bundle): ONTOLOGY_SOURCE flag — introspection backend for air_defense_v3; loader contract preserved`

**Note on test lifecycle — see Task 45b below.** The `ONTOLOGY_SOURCE`-parametrized suite is transitional. Task 45 removes the env-var read from `load_ontology`, which would turn the `"yaml"` parametrization into a dead branch. Task 45b converts those tests to a non-env-var form before the removal lands.

### Task 24b: Convert backend-parametrized tests to default-vs-fixture assertions (blocks Task 45)

**Files:**
- Modify: any test file using `@pytest.mark.parametrize("source", ["yaml", "pydantic"])` or reading `ONTOLOGY_SOURCE` (identify via grep during implementation).

After Task 45 removes the `ONTOLOGY_SOURCE` env-var branch, env-parametrized tests either:
- Become no-ops (both "branches" call introspection) — stops testing two real code paths.
- Fail outright if they rely on env-var-driven dispatch.

Convert each parametrized test into two explicit assertions against the two surviving code paths:
1. **Default-lookup path (introspection):** `load_ontology()` with no args → introspection-backed dict.
2. **Explicit-path YAML path:** `load_ontology(path=tests/fixtures/ontology/air_defense_v3_snapshot.yaml)` → YAML-backed dict (the frozen fixture from Task 44b).

Both paths return canonical-JSON-equivalent dicts today (per the parity contract from Tasks 20–23); if they diverge, the parity test from Task 23 catches it. These tests exercise the LOADER CONTRACT itself — that both code paths survive Task 45 and return usable data.

- [ ] **Step 1:** Identify all ONTOLOGY_SOURCE-parametrized tests via grep.
- [ ] **Step 2:** Rewrite each to assert against the two surviving code paths (default vs fixture path), replacing env-var toggles with explicit call forms.
- [ ] **Step 3:** Suite runs green under Task 24's feature-flag code AND under Task 45's post-removal code (both must continue to pass).
- [ ] **Step 4:** Commit: `test(bundle): convert backend-parametrized tests to default-vs-fixture form`

Task 45 is blocked by Task 24b.

---

## Chunk 4: Phase 4 — Consumer migration

Each task: (1) read the consumer, (2) write a parity test running under both sources, (3) verify behavior equivalent, (4) commit. No consumer migrates without a green parity test.

### Task 25: Migrate `app/services/extraction_merge.py` (identity + validation_matrix consumers)

Lines 297, 335, 472 read `ontology.get("entity_types", ...)`; line 360 reads `ontology.get("validation_matrix", ...)`. If `load_ontology()` returns canonical-JSON-equivalent Pydantic-backed data, these call sites need no code change — parity test confirms.

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

### Task 34b: `PreMergeWalkSummary` — shared pre-merge carrier for entities + raw edge count

**Files:**
- Modify: `app/services/extraction_merge.py` — add `PreMergeWalkSummary` dataclass
- Modify: `app/workers/pipeline.py` — `_count_pass_output`, pass loop around `:449`, `classify_yield` integration

The pre-merge pass loop needs BOTH entity counts (for `primary_entities_extracted`/`bridge_entities_extracted`) AND raw edge counts (for provisional `relationships_extracted`). Running the walker twice — once for entities, once for edges — would waste work AND risk drift between the two counts. Introduce a single shared carrier:

```python
@dataclass
class PreMergeWalkSummary:
    entities: list[BaseModel]        # every entity reachable in the pass result (walker output)
    raw_edge_count: int              # count of edge emissions during the walk (no validation yet)
```

The pre-merge path runs `walk_entity_graph` ONCE per `PassResult` with both `on_entity` and `on_edge` callbacks hooked up. `on_entity` appends to `entities`; `on_edge` increments `raw_edge_count`. The resulting `PreMergeWalkSummary` is passed into both `_count_pass_output` and `classify_yield` — both consume the pre-built summary, neither re-traverses.

Because the pre-merge path DOES need `ontology` and `document_id` to invoke `walk_entity_graph` with identities (required for edges), the pass loop at `pipeline.py:449` is the right place to construct the summary — it has both in scope. The summary gets attached to `PassResult` as an optional field so later callers (inside merge or downstream) can reuse it without re-walking. Fallback: if the summary isn't attached (e.g. tests constructing `PassResult` manually), `_cached_entities` uses `walk_entities_only` to populate just the entity list as before.

**`system_links` DTO special-case (important):** `system_links` is not on the walker path — its relationships live in `SystemLinkRelationship` DTO records (Decision 4 exception). A naive walker-only `raw_edge_count` would always report `0` for `system_links`, so `classify_yield_from_counts` would see `primary=0, bridge=0, extracted_rels=0` and classify the pass as `EMPTY` pre-merge — even when the LLM emitted candidate DTOs. Current `_apply_post_merge_yield_updates()` at `pipeline.py:506` is a **one-way HIT → DEGRADED** downgrade for entity-bearing passes only (guarded by `if row.yield_status == "HIT"`); it never promotes `EMPTY` upward and does not touch `system_links` at all today. Left unchanged, every `system_links` pass would stay `EMPTY` regardless of actual output. This is the pre-merge half of the bug — Task 36's relationships-only post-merge branch (which delegates to `classify_yield_from_counts`) is the other half.

For `system_links` passes the pass loop at `pipeline.py:449` constructs `PreMergeWalkSummary` differently:
- `entities` = `[]` (system_links has no entities of its own — it's the documented relationships-only pass).
- `raw_edge_count` = `len(pass_result.template_instance.relationships)` (DTO-list length — every candidate `SystemLinkRelationship` counts as a provisional edge, same semantics as typed-edge passes counting walker emissions pre-validation).

This makes `system_links`'s pre-merge classification match typed-edge passes: `classify_yield()` sees the DTO length as the provisional `relationships_extracted` (so non-zero candidate DTOs produce pre-merge `HIT`); Task 36's post-merge relationships-only branch then overwrites `yield_status` authoritatively from `per_pass_edge_metrics.accepted`/`.rejected` via `classify_yield_from_counts`.

- [ ] **Step 1:** Add `PreMergeWalkSummary` dataclass; add optional `pre_merge_walk: PreMergeWalkSummary | None` field on `PassResult`.
- [ ] **Step 2:** Update the pass loop at `pipeline.py:449` to construct the summary once per pass and attach it to the `PassResult`.
- [ ] **Step 3:** Update `_count_pass_output` to consume `pass_result.pre_merge_walk.entities` for entity counts and `.raw_edge_count` for the provisional `relationships_extracted`.
- [ ] **Step 4:** Update `classify_yield` to consume the same summary.
- [ ] **Step 5:** Failing test: patched walker increments a counter on each invocation; construct one `PassResult` with a nested-entity template; verify walker runs exactly once for the pre-merge phase (not twice).
- [ ] **Step 6:** Run test + full unit suite — PASS.
- [ ] **Step 7:** Commit: `refactor(merge,pipeline): PreMergeWalkSummary — single-traversal pre-merge carrier for entities and raw edges`

### Task 35a: Split traversal into entity-only and full walkers; rewrite `PassResult.iter_entities_of_type`

**Files:**
- Modify: `app/services/extraction_merge.py:94` (`PassResult`), `:104` (`iter_entities_of_type`)
- New unified public walker: `walk_entity_graph` (single function — see below). NO separate `walk_entities_only` function; mode is determined by whether the `on_edge` callback is provided.

**The problem:** `PassResult` at `extraction_merge.py:94` only carries `pass_name`, `template_instance`, `metadata`, `pre_merge_rejections`, `upstream_refs`. No `ontology`, no `document_id`. `iter_entities_of_type` is called from many sites — including pre-merge `_count_pass_output` at `pipeline.py:1753` (call site `:449`) — where `ontology` and `document_id` aren't yet in scope or aren't available. But the full walker (Task 35b) needs both to build logical identities for edges.

**The split — single walker with optional edge emission, NOT two separate functions:**

Commit to one public walker: **`walk_entity_graph(node, on_entity, *, ontology=None, document_id=None, on_edge=None, visited_objects=None, at_pass_root=True)`**.
- When `on_edge is None`: entity-only traversal — walker skips edge-tuple construction and never touches `ontology` / `document_id`. `ontology=None, document_id=None` are tolerated in this mode.
- When `on_edge is not None`: full traversal — walker emits edges with logical identities; `ontology` and `document_id` are required (asserted at the top of the function with a clear error message on None).

Thin adapters at the call sites:
- `PassResult.iter_entities_of_type(entity_type)` at `:104` calls `walk_entity_graph(..., on_entity=..., on_edge=None)`. No external context required; `ontology`/`document_id` remain `None`. Recursive nested-entity behavior is inherited from the unified walker.
- `merge_and_resolve`'s edge-consumption loop calls `walk_entity_graph(..., on_entity=..., on_edge=..., ontology=..., document_id=...)`.

**Rationale:** the prior plan left "two thin wrappers vs one internal helper" as an implementation choice. That choice matters for test surface: two public functions means duplicate tests, duplicate recursion bugs if they drift, and an unclear contract for new callers ("which do I use?"). A single walker with a keyword-only optional `on_edge` resolves that — one public surface, one test matrix, one recursion path. The mode flag is the presence of the callback, not a separate function.

**Iterative-vs-recursive note (addresses Finding 9):** Python's default recursion limit is 1000. Deeply nested extractions in this ontology (radar → antenna → transmitter → receiver → signal_processing_chain) are well below that (max observed depth ≈ 8). Recursive implementation is acceptable. If a future ontology change introduces deeper chains, the walker's single entry point makes iterative conversion a local change — no call-site impact.

**`PassResult` changes (minimal):** add a lazy cache field that prefers pre-merge walk output if present:
```python
@dataclass
class PassResult:
    ...
    pre_merge_walk: PreMergeWalkSummary | None = None      # populated by pass loop (Task 34b)
    _walker_entities_cache: list[BaseModel] | None = field(default=None, init=False)

    def _cached_entities(self) -> list[BaseModel]:
        if self._walker_entities_cache is not None:
            return self._walker_entities_cache
        # Prefer the pre-merge walk if the pass loop already built one —
        # guarantees iter_entities_of_type and _count_pass_output see the
        # same traversal, avoiding count-vs-upstream-ref drift.
        if self.pre_merge_walk is not None:
            self._walker_entities_cache = self.pre_merge_walk.entities
            return self._walker_entities_cache
        # Fallback: test-built PassResult or contexts without ontology/document_id.
        # Call the single unified walker in entity-only mode (on_edge=None).
        out: list[BaseModel] = []
        walk_entity_graph(
            self.template_instance,
            on_entity=out.append,
            on_edge=None,
            ontology=None,
            document_id=None,
        )
        self._walker_entities_cache = out
        return self._walker_entities_cache

    def iter_entities_of_type(self, type_name: str) -> Iterator[BaseModel]:
        # Filter on emit (O(total_entities)): the unified walker emits all
        # entities; per-type filtering happens here. For _count_pass_output
        # at pipeline.py:1756-1762 which calls this once per entity type N,
        # the naive pattern is O(N × total). Use _cached_entities() (memoized
        # per PassResult) so the walk runs ONCE per PassResult — total cost
        # is O(total) amortized across all type filters. Verified with a
        # patched-walker counter test in Task 34b: walker invoked exactly
        # once per PassResult regardless of how many iter_entities_of_type
        # calls fire against it.
        for e in self._cached_entities():
            cfg = getattr(e, "model_config", {}) or {}
            if cfg.get("ontology_name") == type_name:
                yield e
```

No `ontology` / `document_id` needed at this level for the fallback path — entity-only mode tolerates both as `None`. The common path (post-pass-loop) reuses `pre_merge_walk.entities` — one traversal feeds both count-writes at `pipeline.py:449` and the later `_extend_upstream_refs` call. No drift between counts and upstream-ref discovery.

**Walker filtering contract (addresses reviewer finding C2):** the unified `walk_entity_graph` does NOT filter by entity type — it emits every reachable entity via `on_entity`. Per-type filtering is the caller's responsibility. This is deliberate:
- Adding an `entity_type_filter` parameter to the walker would pollute the signature and leak ontology concerns into the walker.
- Memoized `_cached_entities()` means the walk runs once per `PassResult`; subsequent `iter_entities_of_type` calls for different types reuse the cache. Amortized O(total), NOT O(N × total).
- Test from Task 34b already pins this: patched walker increments a counter; assert counter == 1 after multiple `iter_entities_of_type` invocations on one `PassResult`.

This resolves the `system_links` primary-ref gap: manifest `radar_domain` lists `ANTENNA`, `RECEIVER`, `TRANSMITTER`, `SIGNAL_PROCESSING_CHAIN`, `FREQUENCY_BAND`, `WAVEFORM` as primary entity types; after Phase 5 these live nested inside `RadarSystem` via typed edges, and upstream refs for cross-pass linking must reach them.

**Upstream-ref dedup rule (merge-preserving, required alongside recursive discovery):** once `iter_entities_of_type` yields nested entities, the same logical entity can be reached via multiple graph paths — e.g., two `RadarSystem` instances each nesting an `Antenna` with the same identity but different complementary non-null fields (one filled `gain_dbi`, the other filled `antenna_type`). `_extend_upstream_refs` at `pipeline.py:1851` currently allocates a fresh ref id per yielded instance AND builds `display_label` from non-identity properties of the first-seen instance. A naive first-seen dedup would collapse the duplicate endpoint correctly but discard the better label.

The fix lives in `_extend_upstream_refs`, not in `iter_entities_of_type`. Merge-preserving dedup with a **detached scratch accumulator** (never mutate the live extracted entity, which merge will consume later):

**Dedup key (precise):** `(entity_type, identity_tuple)` where `identity_tuple` is constructed from the ontology's declared `identity_fields` list **in ontology-declared order**, matching `_select_upstream_refs_for_pass()` at `pipeline.py:1939` and `_build_logical_identity()` at `extraction_merge.py:288`. Do not use `tuple(identity_dict.values())` — dict iteration order is an implementation detail and will produce inconsistent keys across Python versions / insertion paths. Always look up the ontology's `identity_fields` list and iterate values in that declared sequence.

**Algorithm (does NOT mutate live model instances):**
1. First yield at a given key: allocate a ref id; initialize a scratch `dict[str, Any]` capturing the instance's non-null non-identity fields (copy-by-value, no reference to the original). Record the ref id + scratch dict in a local accumulator.
2. Subsequent yields at the same key: iterate the new instance's non-null non-identity fields; for each field not yet in the scratch dict, copy the value in (first-non-null wins). If the field is already in the scratch dict with a different non-null value, log a warning (same conflict semantics as Task 36 Step 2b entity dedup). Do NOT touch the original Python model instance.
3. After iteration completes, build a synthetic property bag from each scratch dict (combined with the known identity values) and pass that to the existing `display_label` builder. The emitted upstream ref references the first-seen instance's identity but carries the merged display label.

Merge-path consumers later see the unchanged live entity objects from `template_instance`; they union fields via the separate Task 36 Step 2b path on their own data. Upstream-ref label enrichment is cosmetic-only for the LLM prompt; it does not leak into the merged entity graph.

**Empty-identity entities:** the bypass path is *conceptual* — `_is_valid_upstream_ref()` at `pipeline.py:1813` already rejects any type whose manifest declares empty `identity_fields`, so empty-identity entities (`PROPULSION_STACK` etc.) never reach ref allocation under the current worker contract regardless. No upstream refs are emitted for those types. Plan 2's identity redesign will revisit whether empty-identity types should become eligible. This plan does not change that gate.

- [ ] **Step 2.dedup:** Failing test (count collapse): two `RadarSystem` instances in one pass result, both nesting an `Antenna` with identity `{"name": "FF-1"}`; assert `_extend_upstream_refs` produces exactly one upstream ref for that antenna.
- [ ] **Step 2.union:** Failing tests (merge-preserving accumulator, no mutation, exact-label under today's builder):

  Note on label builder: `build_display_label()` at `extraction_merge.py:198` currently walks identity values first (steps 1–2 of its resolution order), then falls back to name-like keys in the properties dict (step 3). Fields that aren't in `_NAME_LIKE_KEYS = ("system_name", "name", "title", "heading", "document_id")` — e.g. `gain_dbi`, `antenna_type` — never appear in the output. Extending the builder to use arbitrary scalars is out of scope for this plan; the merge-preserving accumulator stores richer data so a future label-builder refactor can use it, but the tests here assert exact behavior under today's code, not aspirational behavior.

  - **Accumulator contents (not output):** duplicate A has `gain_dbi=38.0` and null `antenna_type`; duplicate B has null `gain_dbi` and `antenna_type="phased-array"`. Patch `build_display_label` with a capturing wrapper; call `_extend_upstream_refs`; assert the scratch dict passed to the wrapper has BOTH `gain_dbi=38.0` AND `antenna_type="phased-array"` populated. Asserts the union happened at the accumulator layer, independent of what the builder does with the data.
  - **No-mutation:** assert the original Python instances are unchanged after `_extend_upstream_refs` returns — `duplicate_A.antenna_type is None` and `duplicate_B.gain_dbi is None`. Guards against accidental pre-merge of live entity data that later merge stages will consume.
  - **Exact label under today's builder, step 1 hit (AntennaEntity with name-like identity):** identity `{"name": "FF-1"}` on both duplicates, with the non-identity-field split from the accumulator-contents case above. Assert the emitted upstream ref's `display_label == "FF-1"` — exact equality, because step 1 of the builder's resolution order returns the `name` identity value before properties are consulted. This is a hard contract under today's code: identity-named entities must produce the identity name verbatim regardless of whether the accumulator carries extra scalars.
  - **Exact label under today's builder, step 2 hit (non-name-like identity):** construct a test-only entity with `graph_id_fields=["label_id"]` (NOT in `_NAME_LIKE_KEYS`) and identity value `{"label_id": "x-42"}`. Both duplicates carry any non-null non-identity scalars (e.g. duplicate A has `name="Primary"`, duplicate B has `title="Backup"`, to verify step 3 is NOT reached). Assert `display_label == "x-42"` — exact: step 1 finds no name-like identity key and falls through; step 2 joins non-empty identity values, yielding `"x-42"` (single value, no separator needed); step 3 never runs because step 2 returned a truthy result. This locks the step-1→step-2 transition and proves the accumulator's extra name-like properties are NOT consulted when identity is non-empty.
  - **Exact label under today's builder, step 3 hit (empty-identity case, optional):** only meaningful if the fixture also exercises empty-identity entities via upstream refs. Under the current worker contract, `_is_valid_upstream_ref` at `pipeline.py:1813` rejects empty-identity types BEFORE `_extend_upstream_refs` runs, so this path is unreachable in production. If the test shims past that gate purely to exercise step 3, it should assert `display_label == "Primary"` (first duplicate's name-like property wins after accumulator merge; `_NAME_LIKE_KEYS[0] == "system_name"`, absent here; `_NAME_LIKE_KEYS[1] == "name"` hits duplicate A's value). Mark this sub-test `@pytest.mark.xfail(reason="empty-identity upstream refs gated off in production")` so it documents intent without claiming reachability.

  If a future plan extends the builder to consume arbitrary scalars, the step-2-hit test will fail and must be updated there. That's intentional: the test is the contract between "what the accumulator carries" and "what the builder emits today," and both must change together.
- [ ] **Step 2.order:** Failing test (identity-order stability): compose an ontology with `identity_fields: [band_name, designation]`; build two instances with same field values but different dict-insertion orders; assert `_extend_upstream_refs` produces a single ref (keys match despite insertion-order variance) because the key tuple follows ontology field order, not dict order.
- [ ] Update `_extend_upstream_refs` to use a detached scratch-dict accumulator keyed on ontology-field-ordered identity tuples; derive `display_label` from the merged scratch dict; never write into the live instance.

- [ ] **Step 1:** Failing test (upstream refs): build a `PassResult` for `radar_domain` with a nested antenna inside a radar; call `_extend_upstream_refs` through the existing code path; assert the antenna appears in `upstream_refs` with a real ref id. (Currently fails because `iter_entities_of_type("ANTENNA")` is empty.)
- [ ] **Step 2:** Failing test (rename): rename a pass-root field (`radar_systems` → `radars`); assert `iter_entities_of_type("RADAR_SYSTEM")` still yields the same items.
- [ ] **Step 3:** Failing test (no-context): construct a `PassResult` without `ontology` or `document_id`; `iter_entities_of_type("RADAR_SYSTEM")` works — no `AttributeError`, no `None`-identity crash.
- [ ] **Step 4:** Implement `walk_entities_only` + lazy `_walker_entities_cache` on `PassResult`.
- [ ] **Step 5:** Run tests — PASS. Run full extraction_merge + worker test suite.
- [ ] **Step 6:** Commit: `refactor(merge): split walker — entity-only for PassResult, full walker for merge; iter_entities_of_type recursive`

### Task 35b: Unified entity-graph walker (`walk_entity_graph`) emitting entities AND edges in one pass

**Files:**
- Modify: `app/services/extraction_merge.py`

When relationships move inside entity classes (e.g. `RadarSystem.antennas: List[Antenna] = edge(label="HAS_ANTENNA")`), nested child entities live ONLY behind typed-edge fields, never at the pass root. Without recursive collection, merge sees zero antennas even though the LLM extracted them. Task 35a alone fixes the naming heuristic but walks only the pass root.

**Design (combines Task 36's edge harvester with entity collection — single traversal, single visited set):**

Two separate traversals would break each other: if entity collection marks objects visited, the later edge harvester sees every object as visited and emits zero edges. Combine them into one walker with callbacks for entities and edges:

```python
def walk_entity_graph(
    node: Any,
    ontology: dict,
    document_id: str,
    *,
    visited_objects: set[int],
    on_entity: Callable[[Any], None],
    on_edge: Callable[[Any, str, Any], None],
    at_pass_root: bool = True,  # True only for the pass-root container at top-level
) -> None:
    """Walk the typed-edge graph rooted at `node`. Emits entities via on_entity
    and edges via on_edge in a single pass. Cycle prevention via id(object).

    Traversal rules (graph-only, per docs):
    - `at_pass_root=True` (only the initial pass-root container): walk plain
      list/scalar BaseModel fields to reach top-level entities. Children are
      entered with `at_pass_root=False`.
    - Entity nodes (`is_entity=True`): emit via on_entity; then follow ONLY
      fields marked with `json_schema_extra.edge_label`. Components reached
      via `edge_label` are a contract violation — enforced by contract test
      Task 9e (edge_label targets must be entities).
    - Component nodes (`is_entity=False`) encountered *inside* the graph
      (not at pass-root): treat as embedded data. Do NOT recurse, do NOT
      emit as entity. This matches docs "Entities vs Components": components
      are value objects attached to their parent entity's properties, not
      graph endpoints.
    - Plain nested `BaseModel` fields without `edge_label`: embedded data,
      not graph-relevant. Do NOT recurse."""

    if id(node) in visited_objects:
        return
    visited_objects.add(id(node))

    cfg = getattr(node, "model_config", {}) or {}

    if at_pass_root:
        # Pass-root container: walk plain fields to reach top-level entities.
        # Do NOT emit as entity.
        for fname, finfo in node.model_fields.items():
            value = getattr(node, fname, None)
            if value is None:
                continue
            for child in (value if isinstance(value, list) else [value]):
                if isinstance(child, BaseModel):
                    walk_entity_graph(
                        child, ontology, document_id,
                        visited_objects=visited_objects,
                        on_entity=on_entity, on_edge=on_edge,
                        at_pass_root=False,
                    )
        return

    if cfg.get("is_entity") is False:
        # Component encountered inside the entity graph → embedded data.
        # Do NOT emit, do NOT recurse. Contract test Task 9e forbids
        # components being reached via edge_label; unexpected components
        # here indicate a schema bug.
        return

    # Entity node — emit, then follow edge_label fields only.
    on_entity(node)
    # NOTE on signature: _build_logical_identity at extraction_merge.py:288
    # is (entity_type: str, entity_instance, ontology, document_id). Derive
    # entity_type from the canonical ontology_name on model_config.
    entity_type = cfg["ontology_name"]
    parent_identity = _build_logical_identity(entity_type, node, ontology, document_id)

    for fname, finfo in node.model_fields.items():
        extra = finfo.json_schema_extra or {}
        edge_label = extra.get("edge_label")
        if not edge_label:
            continue  # Non-edge field; embedded data or scalar, not graph-relevant.
        value = getattr(node, fname, None)
        if value is None:
            continue
        for child in (value if isinstance(value, list) else [value]):
            if not isinstance(child, BaseModel):
                continue
            child_cfg = getattr(child, "model_config", {}) or {}
            if child_cfg.get("is_entity") is not True:
                # Defensive runtime guard: edges must target is_entity=True
                # classes (Task 9e enforces this at schema-validation time).
                # If a schema bug slips past, log a warning and skip — never
                # emit an edge to a component or unclassified class.
                logger.warning(
                    "walk_entity_graph: edge_label=%r on %s.%s points at %s "
                    "which is not is_entity=True; skipping (contract violation)",
                    edge_label, type(node).__name__, fname, type(child).__name__,
                )
                continue
            on_edge(parent_identity, edge_label, child)
            walk_entity_graph(
                child, ontology, document_id,
                visited_objects=visited_objects,
                on_entity=on_entity, on_edge=on_edge,
                at_pass_root=False,
            )
```

**No lossy collection-time dedup.** `on_entity` appends to a `list[entity_instance]` (duplicates allowed). Dedup with smart field merging (non-null union, max-confidence, provenance union, nested-edge union) is the merger's existing responsibility in `merge_and_resolve`. Collection stays pure.

- [ ] **Step 1:** Failing tests for the walker in isolation:
  - (a) `RadarDomainPass` with one `RadarSystem` containing two nested `Antenna` items → `on_entity` called 3× (1 radar + 2 antennas); `on_edge` called 2× (radar→ant1, radar→ant2).
  - (b) Self-reference cycle (entity points to itself) → walker terminates; `on_entity` called once; `on_edge` called once.
  - (c) Same logical entity reachable via two paths → `on_entity` called twice (same Python object prevents third visit; distinct objects still visited); merger handles logical dedup later.
  - (d) **Component nested inside an entity via plain field (no edge_label)** → NOT recursed into, NOT emitted (it's embedded data, per docs).
  - (e) **Component reached via edge_label field** → Task 9e contract test fails at schema-validation time; this combination must not occur. Walker-level test asserts that if it somehow does occur, the walker returns early without emitting the component as an entity (defensive: it's still not graph-relevant).
  - (f) Plain nested BaseModel entity field (no edge_label) on an entity → NOT recursed into, NOT emitted.
  - (g) `at_pass_root=True` root iteration enters children with `at_pass_root=False` → prevents the root-container branch from re-engaging inside the graph (e.g. nested `PassResult`-like wrappers would otherwise be walked as if they were at the top).
- [ ] **Step 2:** Failing tests for duplicate-preserving collection:
  - Same logical entity appears twice in one pass with different non-null fields (e.g. pass 1 fills `nomenclature`, another path fills `radar_type`) → collector emits both instances; merger (in Task 36) unions non-null fields, keeps max confidence, unions provenance.
- [ ] **Step 3:** Implement `walk_entity_graph`.
- [ ] **Step 4:** Update `merge_and_resolve` to call `walk_entity_graph` over each pass-root with `on_entity` appending to a per-pass entities list and `on_edge` appending to a per-pass edges list. **Separate `visited_objects` set per pass** (resets between passes — same Python object reachable in two passes is legitimately visited per-pass).
- [ ] **Step 5:** Run tests + full extraction_merge suite — PASS.
- [ ] **Step 6:** Commit: `feat(merge): unified walk_entity_graph — entities + edges in one traversal`

### Task 35c: Rewrite pass-level counters to consume `PreMergeWalkSummary` (no fresh walks)

**Files:**
- Modify: `app/services/extraction_merge.py` (`classify_yield` at line 257)
- Modify: `app/workers/pipeline.py` (`_count_pass_output` at line 1753)

Both functions currently count entities via `PassResult.iter_entities_of_type()`, which is now pass-root-only (after Task 35a). With nested children living behind typed edges after Phase 5, these counters would undercount and yield_status would misreport (`EMPTY`/`BRIDGES_ONLY` when real content exists).

**Consume the shared `PreMergeWalkSummary` from Task 34b, DO NOT re-walk.** Task 34b's entire point is single-traversal: the pass loop at `pipeline.py:449` constructs one `PreMergeWalkSummary` per `PassResult` and attaches it to `PassResult.pre_merge_walk`. Both `classify_yield` and `_count_pass_output` read that pre-built summary:

- `pass_result.pre_merge_walk.entities` → list of every emitted entity (recursive, nested children included). Classify each by `model_config["ontology_name"]` against the pass's `primary_entity_types` / `bridge_entity_types` manifest metadata.
- `pass_result.pre_merge_walk.raw_edge_count` → provisional `relationships_extracted` (pre-validation; will be overwritten authoritatively by `_apply_post_merge_yield_updates`).
- **`relationships_rejected` is forced to `0` at pre-merge** — explicitly overriding the legacy `len(pass_result.pre_merge_rejections)` derivation. This matches the new lifecycle contract: VALIDATION_MATRIX triple-check and per-reason rejection accounting happen at merge time; pre-merge StageRun rows record raw walker emissions only, no rejected counts. The `pre_merge_rejections` list is still carried on `PassResult` for observability, but `_count_pass_output` no longer consumes it; post-merge `_apply_post_merge_yield_updates` is the single authority for `relationships_rejected`.

Running `walk_entity_graph` a second time from either function would reintroduce the exact double-walk and count-drift risk Task 34b eliminated.

- [ ] **Step 1:** Failing tests:
  - (a) Nested-entity pass (radar with nested antennas): assert `_count_pass_output` returns `primary_entities_extracted` including the nested antennas (currently would return only the radar).
  - (b) Walker-invocation counter: patch `walk_entity_graph` to increment a counter on each call; run the pass loop for one `PassResult`; assert the counter is exactly 1 after both `_count_pass_output` AND `classify_yield` have been invoked (both consumed the shared summary; neither re-walked).
  - (c) **Pre-merge rejected always 0:** construct a `PassResult` with `pre_merge_rejections=[<fake rejection dict>, <another>]` (populating the legacy field); call `_count_pass_output`; assert the returned `relationships_rejected` is `0`, not `2`. The post-merge path (`_apply_post_merge_yield_updates`) is the authoritative source; the pre-merge row reports raw extraction only.
- [ ] **Step 2:** Rewrite `classify_yield` and `_count_pass_output` to read from `pass_result.pre_merge_walk` exclusively and return `relationships_rejected=0` at pre-merge. If `pre_merge_walk` is `None` (test-built `PassResult`), fall back to `walk_entities_only` for entities and `raw_edge_count=0` for edges.
- [ ] **Step 3:** Run tests — PASS (nested and flat pass structures; walker invoked once; rejected stays 0).
- [ ] **Step 4:** Commit: `refactor(merge,pipeline): pass-level counters from PreMergeWalkSummary; relationships_rejected=0 at pre-merge`

### Task 36: Design + write the new `merge_and_resolve` relationship-consumption path

**Files:**
- Modify: `app/services/extraction_merge.py`

- [ ] **Step 1:** Write failing tests for the new behavior, including cycle/dedup:
  - Given a pass result containing a `RadarSystemEntity` with `antenna: AntennaEntity = edge(label="HAS_ANTENNA")`, harvester produces a `MergedEdgeRecord` with `rel_type="HAS_ANTENNA"`, `from_identity` = radar identity, `to_identity` = antenna identity.
  - Given two chained edges (`radar → antenna → frequency_band`), all three edges are emitted.
  - **Cycle: self-reference.** An entity with a self-referential edge (e.g. `Section.parent_section: Section = edge(...)`) does not infinite-loop; visited set keyed by **`id(entity_object)` (Python object identity)** prevents revisit.
  - **Cycle: mutual reference.** Two entities A↔B referencing each other yield exactly one edge in each direction, no infinite recursion. Object-id guard.
  - **Dedup: same edge from two paths.** If `(rad_1, HAS_ANTENNA, ant_1)` is reachable both via direct field walk and via a back-reference traversal, the merger emits it once. Edges deduped by `(from_logical_identity, rel_type, to_logical_identity)`.
  - **Empty-identity entity (PROPULSION_STACK shape).** Multiple distinct instances each get traversed (object-id guard does not collapse them); edges emitted per (from, rel, to). Output dedup keyed on logical identity collapses outbound edges from empty-identity sources to a single edge — known anti-pattern, documented.
  - **system_links DTO branch.** Given a `SystemLinksPass` with one `SystemLinkRelationship(from_ref_id="X", to_ref_id="Y", rel_type="ASSOCIATED_WITH")`, the merger resolves both refs to upstream entities (via existing `upstream_refs` mechanism) and emits a `MergedEdgeRecord` — using the DTO branch, not the typed-edge harvester. Both branches feed the same output type.
  - **DTO ↔ typed-edge normalization parity (addresses reviewer finding C3).** Build a fixture with ONE edge emitted via each path: (a) a typed-edge harvest (`RadarSystemEntity.antenna = AntennaEntity(...)`) producing `MergedEdgeRecord(from_identity=radar_id, rel_type="HAS_ANTENNA", to_identity=antenna_id, pass_origins={"radar_domain"})`; (b) a system_links DTO producing the SAME edge via ref-resolution (`SystemLinkRelationship(from_ref_id=radar_ref, to_ref_id=antenna_ref, rel_type="HAS_ANTENNA")` where refs resolve to identical identities). Assert the two `MergedEdgeRecord` instances compare equal on every field (`from_identity`, `rel_type`, `to_identity`, and that `pass_origins` each contain their respective pass name). If the two paths drift — different field ordering, different identity construction, different dedup-key computation — one path silently loses edges to VALIDATION_MATRIX rejection or edge-dedup collapse. This test pins: **both branches produce structurally identical `MergedEdgeRecord` outputs for equivalent inputs.**

- [ ] **Step 2:** Edge harvesting is subsumed by `walk_entity_graph` (Task 35b). Edges come from the `on_edge` callback — they're built as `MergedEdgeRecord` and appended to a per-pass list. No separate edge-harvester function.

- [ ] **Step 2.dataclass:** Update `MergedEdgeRecord` (at `extraction_merge.py:146`) — replace single `source_pass: str` with `pass_origins: set[str]`. Symmetric with `MergedEntityRecord.pass_origins`.

  **All consumers of `edge.source_pass` must be updated. Full blast radius:**
  - `app/workers/pipeline.py:506` (`_apply_post_merge_yield_updates`) — per-pass yield accounting iterates edges by `source_pass`.
  - `app/workers/pipeline.py:519` — same function, same pattern.
  - Audit-blob serialization in `_serialize_for_audit`.
  - ArcadeDB graph-import in `_import_graph_phase_domain_edges`.
  - Any test fixture or mock that constructs a `MergedEdgeRecord` with `source_pass=...`.

  **Per-pass edge-count semantics (needed because edges now have multiple origins):**
  Pre-reduction pass-local counting. Each pass's walker output yields its own per-pass edge list (before cross-pass reduction). Per-pass StageRun metrics (`relationships_extracted`, `relationships_rejected`) count from the pre-reduction per-pass list — each pass gets its own contribution counted separately. The cross-pass reducer runs afterward, unions `pass_origins`, and produces the final merged edge set.

  **Carrier for per-pass edge metrics:** `MergedExtraction` (at `extraction_merge.py:156`) gets a new field:
  ```python
  per_pass_edge_metrics: dict[str, PerPassEdgeMetrics]  # keyed by pass name

  @dataclass
  class PerPassEdgeMetrics:
      attempted: int              # raw count from walker (typed-edge passes) or DTO list (system_links) — before any validation
      accepted: int               # after VALIDATION_MATRIX triple check + Pydantic parse — authoritative extraction count
      rejected: int               # attempted - accepted (includes INVALID_TRIPLE + any pass-specific rejection reasons)
      rejection_sample: list[dict]  # up to N sampled rejected tuples via _rel_to_dict shape
      rejections_by_reason: dict[str, int]  # {"INVALID_TRIPLE": ..., etc.} — authoritative per-reason counts
  ```

  **Field semantics disambiguated:** `attempted` / `accepted` / `rejected` replace the ambiguous single `extracted`. `StageRun.relationships_extracted` maps to `accepted` (matches current authoritative semantics at `pipeline.py:506`, which counts accepted edges only — the old `extracted = attempted` would have overcounted because it included edges that fail VALIDATION_MATRIX). `StageRun.relationships_rejected` maps to `rejected`.

  **Carrier is populated by BOTH edge-producing branches — uniform input to `_apply_post_merge_yield_updates`:**
  - **Typed-edge passes** (radar/missile/other, reference): populated from the per-pass walker output. `attempted` = raw walker edge emissions. Validation (VALIDATION_MATRIX triple check) happens in `merge_and_resolve` before cross-pass reduction; rejected edges append to `rejection_sample` and increment `rejections_by_reason`.
  - **`system_links` DTO branch**: populated from the DTO-list consumption path. `attempted` = len of raw `SystemLinkRelationship` list; `accepted` = count of successfully-resolved ref-id pairs that also pass VALIDATION_MATRIX; `rejected` + `rejection_sample` + `rejections_by_reason` mirror the current `pre_merge_rejections` shape for parity. Both branches feed identical-shape `PerPassEdgeMetrics` into the carrier — `_apply_post_merge_yield_updates` reads the carrier uniformly without caring which branch produced which entry.

  The `rejections_by_reason` field preserves parity with the current DTO path: `_build_rejections_by_reason()` at `pipeline.py:1788` and `_apply_post_merge_yield_updates()` at `:506` currently persist per-reason metrics into `StageRun.metrics`; the new carrier must continue feeding that observability path without regression.

  `_apply_post_merge_yield_updates()` (`pipeline.py:506`) is rewritten to read `merged.per_pass_edge_metrics[pass_name]` for both typed-edge AND system_links passes. Without the carrier there's nothing for the yield-updater to consume; without `rejections_by_reason` specifically, passes lose per-reason trending; without uniform population across both branches, system_links metrics regress.

  **Pre-merge vs post-merge count lifecycle (stated explicitly — decision locked):**
  - **Pre-merge** (`_count_pass_output` at `pipeline.py:1753`, called at `:449` per pass): writes **provisional** counts into `StageRun.metrics` alongside the `COMPLETE` row that `_write_stage_run` (at `pipeline.py:1566`) emits when the pass parses successfully. `metrics["relationships_extracted"]` = raw walker edge count (mirror of the top-level column); `metrics["relationships_rejected"]` = 0 at this point because invalid-triple validation happens later; `metrics["rejection_sample"]` empty; `metrics["counts_authoritative"]` = `False`. The row's `status`/`execution_status` columns stay at the existing `COMPLETE`/`FAILED`/`SKIPPED` semantics — unchanged.
  - **Post-merge** (`_apply_post_merge_yield_updates` at `pipeline.py:506`): reads `merged.per_pass_edge_metrics[pass_name]` and **overwrites the same `metrics` keys authoritatively** — now including `rejected` count, `rejection_sample`, and `rejections_by_reason`; flips `metrics["counts_authoritative"]` to `True`. The row's existing `COMPLETE` status does not change.

  **Concrete pre-merge writer change (required — do NOT leave to implementer discretion):**
  Today the pass loop at `pipeline.py:449` only sets `counts["metrics"] = {"rejections_by_reason": _build_rejections_by_reason(...)}`. That's why the regression test's pre-merge assertion set is currently stronger than the writer. Explicit steps to align them:

  - [ ] **Step 2.premerge-writer:** Edit the pass loop at `pipeline.py:449` so `counts["metrics"]` is constructed as:
    ```python
    counts["metrics"] = {
        "counts_authoritative": False,
        "relationships_extracted": counts["relationships_extracted"],  # mirror top-level column
        "relationships_rejected": counts["relationships_rejected"],    # mirror top-level column (0 at pre-merge)
        "rejection_sample": [],                                        # populated post-merge only
        "rejections_by_reason": _build_rejections_by_reason(
            getattr(pass_result, "pre_merge_rejections", None),
        ),
    }
    ```
    All five keys are present on every pre-merge `StageRun.metrics` JSONB write. `_write_stage_run` at `pipeline.py:1566` is unchanged — it just persists whatever metrics dict it receives.
  - [ ] **Step 2.premerge-writer-test:** Add a focused unit test `test_premerge_metrics_jsonb_shape` that runs one pass through the pass loop, queries the row, and asserts the exact five-key shape (each key present; values match the spec above). This test is independent of Step 2.authtest's post-merge lockstep assertion — if the writer regresses to the old single-key shape, this test fails first and localizes the break.
  - [ ] **Step 2.postmerge-writer:** Symmetric change at `pipeline.py:506` — `_apply_post_merge_yield_updates` must overwrite all five keys (not just `rejections_by_reason` as today) and flip `counts_authoritative=True` in the same `row.metrics` assignment. Already implicit in Step 2.authtest's assertions; lifted here as a concrete implementation step so it's not missed.

  **Decision: record provisional-vs-authoritative state inside the existing `metrics` JSON column, not via the `status` column.** Add a single boolean `metrics["counts_authoritative"]` — pre-merge writes set it to `False`; `_apply_post_merge_yield_updates` flips it to `True` when it overwrites the numeric keys. The existing `StageRun.status` and `_write_stage_run` contract at `pipeline.py:1566` stay untouched (that code sets `status="COMPLETE"` immediately with `finished_at`; reworking it would be a much broader lifecycle change). Readers that want authoritative counts filter on `metrics["counts_authoritative"] == True`.

  **Rejected alternatives:**
  - Reusing `status` to mean "authoritative yet": would require rewriting `_write_stage_run` to hold rows in `RUNNING` until post-merge, changing `finished_at` semantics, and updating every consumer that treats `COMPLETE` as "pass finished parsing successfully." Significantly larger blast radius than this plan covers.
  - Adding a new top-level `is_authoritative: bool` column on `StageRun`: requires a schema migration for one bit of state that already fits inside the existing `metrics` JSON.

  Both phases write the same metric keys, differentiated by `counts_authoritative`. The lifecycle is documented in the `StageRun` class docstring. Tests and observability queries that want "extraction counts after validation" filter on `metrics->>'counts_authoritative' = 'true'`.

  **Explicit regression test for the `counts_authoritative` contract** (ensures provisional/authoritative separator doesn't silently break):

  - [ ] **Step 2.authtest:** End-to-end test `test_counts_authoritative_lifecycle`. Must assert **lockstep** updates across BOTH the JSONB `metrics` keys AND the top-level `StageRun` columns (`relationships_extracted`, `relationships_rejected` at `app/models/ingest.py:279-280`) — they are two projections of the same counter and must never drift:
    1. Run one pass through `_write_stage_run()` at `pipeline.py:1566` via the pass-loop path that invokes `_count_pass_output`. Query the `StageRun` row:
       - JSONB: `metrics["counts_authoritative"] is False`; `metrics["relationships_extracted"]` equals the pre-merge walker count; `metrics["relationships_rejected"] == 0`; `metrics["rejection_sample"] == []`.
       - Top-level columns: `row.relationships_extracted == metrics["relationships_extracted"]` (lockstep); `row.relationships_rejected == 0`.
    2. Run `_apply_post_merge_yield_updates(pipeline_run_id, merged, manifest)` against the same pass-name in a `MergedExtraction` whose `per_pass_edge_metrics[pass_name]` has known `attempted=5, accepted=3, rejected=2, rejection_sample=[{...}, {...}], rejections_by_reason={"INVALID_TRIPLE": 2}`. Re-query the `StageRun` row and assert **all three surfaces move together**:
       - JSONB: `metrics["counts_authoritative"] is True`; `metrics["relationships_extracted"] == 3`; `metrics["relationships_rejected"] == 2`; `metrics["rejection_sample"]` has the 2 overwritten entries; `metrics["rejections_by_reason"] == {"INVALID_TRIPLE": 2}`.
       - Top-level columns: `row.relationships_extracted == 3`; `row.relationships_rejected == 2`. These MUST equal the JSONB values (lockstep assertion) — an XOR of the two surfaces is a test failure, even if each individually matches an expected value.
       - Unchanged: `row.status` / `row.execution_status` remain `"COMPLETE"`.
    3. Query before-and-after of a row that has `counts_authoritative=True` — ensure a second `_apply_post_merge_yield_updates` call is idempotent (no double-accounting) on both JSONB metrics and top-level columns.
    4. **Relationships-only lockstep sub-case:** feed a `system_links` pass with `accepted=0, rejected=3`. Assert post-merge: `row.relationships_extracted == 0`, `metrics["relationships_extracted"] == 0`, `row.yield_status == "EMPTY"`, `metrics["counts_authoritative"] is True` — proving the new relationships-only branch also updates all surfaces in lockstep.

  This test gates the entire lifecycle contract with concrete database assertions across both projections rather than relying on readers to notice a silent regression on one surface while the other stays correct.

  **Rejection semantics for typed-edge path** (previously undefined, now explicit):
  - **Schema-time contract violations** — `edge_label` targeting `is_entity=False` class; field declaring impossible cardinality; etc. — are caught by Task 9a/9e contract tests in CI. If they escape to production the walker's runtime defense (log-and-skip) prevents an invalid edge from being emitted, but this path does NOT append to `rejected_edges` because it represents a developer bug, not extraction data.
  - **`VALIDATION_MATRIX` triple-check rejections** — the walker emits an edge, then the worker's post-walker validator checks `(from_type, rel_type, to_type) in VALIDATION_MATRIX`. Invalid triples append to `rejected_edges` with `reason="INVALID_TRIPLE"` — same mechanism as the current DTO path. This preserves `relationships_rejected` semantics and rejection sampling (`_rel_to_dict` at `pipeline.py:170`, `_build_rejection_sample` at `:181`) for typed-edge passes.
  - **Pydantic validation failures at pass-template parsing** — same as today: `_parse_pass_response` raises `PassTerminal`; whole-pass failure, not per-edge rejection.

  **`system_links` authoritative-yield promotion** (fixes latent bug in current `_apply_post_merge_yield_updates`):
  Current `_apply_post_merge_yield_updates` only allows `HIT → DEGRADED` for entity-bearing passes, and takes no action on `system_links`. For a `system_links` pass with 1–3 rejected DTOs and 0 accepted edges, the post-merge authoritative `yield_status` would stay at `HIT` — but `classify_yield_from_counts()` at `extraction_merge.py:234` would classify zero-accepted + zero-entity as `EMPTY`. For 4+ DTOs with ≥75% rejected, the classifier would return `DEGRADED`. Pre-merge and post-merge disagree, silently.

  **Signature change:** `_apply_post_merge_yield_updates(pipeline_run_id, merged)` at `pipeline.py:506` must be extended to `_apply_post_merge_yield_updates(pipeline_run_id, merged, manifest: BundleManifest)`. The real caller is the derive-task execution path `_derive_ontology_graph_bundle_passes` at `pipeline.py:3853` — specifically the call at `pipeline.py:3942` (immediately after `merge_and_resolve` at `:3934`). That scope already holds `manifest = load_bundle_manifest(bundle_key)` at `pipeline.py:3910`; threading it into the helper is a one-line change at the call site. (`start_ingest_pipeline` at `pipeline.py:1283` is the Celery-dispatch function that returns `IngestDispatchResult` — it does NOT run merge or post-merge StageRun updates; the earlier plan draft misnamed it.) Inside the helper we dispatch on `manifest.find_pass(pass_name).kind` — note the correct helper name is `BundleManifest.find_pass` at `app/services/ontology_bundles.py:65`, NOT `get_pass`.

  Add a **relationships-only branch** that delegates back to `classify_yield_from_counts` rather than hardcoding the answer — this preserves the DEGRADED-on-high-rejection rule (≥4 total_rels with ≥75% rejected) without re-implementing it here:
  ```python
  # After reading merged.per_pass_edge_metrics[pass_name]:
  pass_def = manifest.find_pass(pass_name)  # BundleManifest.find_pass — ontology_bundles.py:65
  if pass_def.kind == "relationships_only":
      # No entity counts for this pass-kind. Delegate to the canonical classifier
      # so EMPTY (accepted==0 with <4 total rejected) and DEGRADED (total>=4 and
      # >=75% rejected) both come out consistent with pre-merge semantics.
      authoritative_yield = classify_yield_from_counts(
          primary=0,
          bridge=0,
          extracted_rels=metrics.accepted,
          rejected_rels=metrics.rejected,
      )
      row.yield_status = authoritative_yield.value  # overwrite unconditionally for this pass-kind
  else:
      # Existing HIT → DEGRADED rule for entity-bearing passes (guarded on yield_status == "HIT").
      if row.yield_status == "HIT":
          new_yield = classify_yield_from_counts(
              primary=row.primary_entities_extracted or 0,
              bridge=row.bridge_entities_extracted or 0,
              extracted_rels=metrics.accepted,
              rejected_rels=metrics.rejected,
          )
          if new_yield == YieldStatus.DEGRADED:
              row.yield_status = "DEGRADED"
  ```

  `system_links`'s `kind: relationships_only` flag in `manifest.yaml` is the discriminator. Post-merge now produces an authoritative `yield_status` fully consistent with `classify_yield_from_counts()` across the DTO/typed-edge split, with no hardcoded values in the update helper.

  Test in the `counts_authoritative` regression suite — three sub-cases for the relationships-only branch, all asserted post-`_apply_post_merge_yield_updates`:
  - **EMPTY promotion:** 3 DTOs, all fail VALIDATION_MATRIX triple check (`accepted=0, rejected=3`). Pre-merge `yield_status="HIT"` (3 provisional). Post-merge `yield_status="EMPTY"`.
  - **DEGRADED promotion:** 4 DTOs, all rejected (`accepted=0, rejected=4`). Pre-merge `"HIT"`. Post-merge `"DEGRADED"` (total_rels=4, rejected/total=1.0 ≥ 0.75).
  - **HIT retained:** 4 DTOs, 3 accepted + 1 rejected. Post-merge `"HIT"` (rejection ratio 0.25 < 0.75).

  Commit as its own sub-task: `refactor(merge): MergedEdgeRecord.pass_origins + per-pass metrics carrier + typed-edge rejection rules + relationships-only yield delegation`.

- [ ] **Step 2a:** Post-traversal edge reducer. Given `list[MergedEdgeRecord]` from the walker:
  - Group by `(from_logical_identity, rel_type, to_logical_identity)`.
  - Keep max confidence.
  - Union `pass_origins` across all contributing passes.
  - Emit one `MergedEdgeRecord` per unique triple.

- [ ] **Step 2b:** Per-pass entity dedup with **merge-preserving scalar-only** semantics. Given `list[entity_instance]` from the walker (duplicates allowed), group by `LogicalIdentity` and merge:
  - Union **scalar/non-edge** fields across duplicates (first-non-null wins per field; never overwrite non-null with non-null — log warning if two non-null values disagree).
  - Keep maximum `confidence`.
  - Union `provenance` lists (populated in Phase 8).
  - Track `pass_origins` as a set.
  - **Do NOT merge edge-backed child lists into entity properties.** Edges live in `MergedEdgeRecord` with their own dedup (Step 2a). Edge-backed child fields are visited by the walker and emitted via `on_edge`; the entity-merge phase does not need to re-store those. Keeping entity merge scalar-only preserves the existing `MergedEntityRecord` contract (`properties: dict[str, Any]`) and avoids leaking graph structure into entity properties.

  Output: `dict[LogicalIdentity, MergedEntityRecord]` per pass. Edge dedup produces the companion `list[MergedEdgeRecord]`.

- [ ] **Step 3:** Rewrite `merge_and_resolve` to:
  - For each non-`system_links` pass: call `walk_entity_graph` with `on_entity`/`on_edge` callbacks; apply Step 2a + 2b reducers. Use a **fresh `visited_objects` set per pass** (reviewer finding #1 — sharing across passes would skip edges).
  - For `system_links`: keep the existing `PassResult.relationships` DTO consumption path, resolving `from_ref_id` / `to_ref_id` against `upstream_refs` (mechanism already in place).
  - Both branches produce `MergedEdgeRecord` lists and `MergedEntityRecord` dicts, fed through the existing cross-pass merge.
  - Cross-pass merge uses the same merge-preserving rules as Step 2b against the already-deduped per-pass results.

- [ ] **Step 4:** Run tests — PASS, including cycle, dedup, and merge-preserving cases.

- [ ] **Step 5:** Commit: `feat(merge): unified walker + merge-preserving dedup; system_links DTO branch retained`

### Task 37: Delete intra-pass DTO relationship classes (preserve `SystemLinkRelationship`)

- [ ] Delete `RadarRelationship` from `extraction_schemas/radar_domain.py`.
- [ ] Delete `MissileRelationship` from `extraction_schemas/missile_domain.py`.
- [ ] Delete `OtherSystemsRelationship` from `extraction_schemas/other_systems.py`.
- [ ] **Keep `SystemLinkRelationship`** in `extraction_schemas/system_links.py`. Add module-level docstring marking it as the documented multi-pass-architecture exception per Decision 4.
- [ ] **Keep `PassResult.relationships`** at `extraction_merge.py:126` — used only by the system_links branch in `merge_and_resolve`. Add inline comment marking that.
- [ ] Run full unit suite; expect failures in consumers referencing the deleted classes. Fix each.
- [ ] Commit: `refactor(bundle): delete intra-pass DTO relationships; SystemLinkRelationship retained as documented exception`

### Task 38: Rewrite `extraction_schemas/reference.py` per docs

Docs-compliant rewrite within scope:
- Module docstring per docs "Docstring Standards" with "Key entities" / "Key relationships" template
- Required imports per docs "Standard Import Block"
- `edge()` helper defined identically (verbatim from docs)
- Components before entities per docs "Standard File Organization"
- `List[...]` from typing (not `list[...]`)
- Required identity with ≥2 examples on entity classes
- `ontology_name` on every entity class
- **Pass-root collection fields stay as plain `List[EntityClass] = Field(default_factory=list)`** — `ReferencePass` is `is_entity=False`, not an ontology entity, so its `figures` / `tables` / `sections` / `assertions` are containers, not edges. (Decision 4a.)
- **Typed `edge(label=...)` only between entity classes** — e.g. if `SectionEntity` references a `Subsection`, that's an edge. The reference pass currently has no inter-entity relationships; nothing in this file uses `edge()`. That's correct and explicit; it's not a missing requirement.

Imports canonical from `entities.py` for narrow extraction views (parallel classes with matching `ontology_name`, not subclasses). Subset-enforcement contract test (Task 38a) verifies the parallel views align with canonical.

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

### Task 42: Update `extraction_schemas/system_links.py` (DTO retained as exception, helper present)

`system_links` is the documented exception. The pass keeps its `SystemLinkRelationship` DTO with `from_ref_id` / `to_ref_id` because cross-pass linking operates on already-extracted refs that aren't in-scope Python objects.

Updates needed:
- Add module docstring marking the file as the documented multi-pass exception per Decision 4. Cite this plan and the docs sections that don't address cross-pass linking.
- Apply the docs-compliant *housekeeping* that doesn't conflict with the DTO pattern: required imports per docs "Standard Import Block", `List[...]` from typing (for any list fields in supporting models), module docstring template, lenient validators kept.
- **Define the `edge()` helper verbatim from docs, even though this file does not call it.** Per docs "Template Basics → Edge Helper Function → Required Definition": *"This function must be defined identically in every template"*. Closest possible adherence to the literal docs even inside the documented exception — the helper is present; it's just unused in this file because the DTO pattern takes its place.
- `SystemLinksPass.relationships: List[SystemLinkRelationship] = Field(default_factory=list)` — plain Field (not `edge()`), since its items are DTOs, not entities. Inline comment cites Decision 4.
- `rel_type: RelationshipType` field uses `_normalize_enum(RelationshipType, v)` validator (per docs "Validation → Enum Normalization Helper").

- [ ] Rewrite + tests + commit: `feat(bundle/system_links): docs-housekeeping (incl. edge() helper) + DTO pattern retained as documented multi-pass exception`

### Task 43: Remove xfail markers

Drop `@pytest.mark.xfail` from all **nine** contract tests. Each is a real release gate. After this commit, every contract test runs as a normal pass-or-fail without skip; CI fails if any of them break.

Tests that must have xfail removed (listed by test function name — plan task numbers are not durable; test names are):
1. `test_is_entity_true_classes_declare_graph_id_fields_key`
2. `test_edge_label_on_entity_to_entity_fields`
3. `test_identity_fields_have_examples`
4. `test_canonical_entities_declare_ontology_name`
5. `test_extraction_views_subset_of_canonical_with_validator_parity`
6. `test_every_class_declares_is_entity_explicitly`
7. `test_descriptions_and_examples_on_extraction_relevant_fields`
8. `test_pass_root_list_dedup_schema_local`
9. `test_edge_label_targets_are_is_entity_true`

- [ ] Commit: `test(schemas): drop xfail — nine contract tests are now release gates`

### Task 44: Capture after-metrics + Phase 5 gate

Re-run `derive_ontology_graph` on `b1b0d596`. Capture the same four artifacts as Task 35 → `/tmp/phase5-after-*`. Diff:

- **Catalog:** `ids=[none]` drops to zero **for every entity type whose canonical `graph_id_fields` is non-empty AND whose pass is not the system_links exception**. PROPULSION_STACK (intentionally `graph_id_fields=[]`) and SystemLinkRelationship (the DTO exception) are explicitly excluded from this gate — they are not bugs, they are documented anti-patterns / exceptions. Test specifies the exclusion list explicitly.
- **Prompt block:** field descriptions appear (now on `Field(description=...)`); pass block structure unchanged.
- **Per-pass metrics:** `primary_entities_extracted` > 0 for radar / missile / other passes on a radar/SAM document; `relationships_extracted` > 0.
- **Merged graph:** entity counts > 0 per type represented in the doc (excluding PROPULSION_STACK).

**Gate: if any non-excluded metric regresses vs Task 35 baseline in a way we can't explain, STOP and investigate before Phase 6.**

- [ ] Diff report committed to `/tmp/phase5-diff.md`. Commit nothing under source control.

---

## Chunk 6: Phase 6 — Delete `ontology.yaml`

### Task 44b: Snapshot `ontology.yaml` as a frozen test fixture (must precede Task 45)

**Files:**
- Create: `tests/fixtures/ontology/air_defense_v3_snapshot.yaml` (committed byte-for-byte copy of `ontology_bundles/air_defense_v3/ontology.yaml` at Phase 5 completion)

The Phase 3 parity tests (Tasks 20–23) are built around "Pydantic introspection output == YAML `load_ontology(bundle_key='air_defense_v3')` output". After Task 45 deletes the YAML file, those tests lose their oracle. Option space:
- Delete the parity tests too — but that loses the anti-drift guard permanently; any future introspection bug would go undetected.
- Regenerate YAML-shaped output from introspection and compare against itself — tautological; tests nothing.
- **Freeze a YAML snapshot as a committed fixture and point parity tests at it via `load_ontology(path=...)`** — preserves the oracle, reuses the path-based loader contract Task 45 keeps intact, and turns the parity tests into a stable anti-drift guard.

We pick the fixture approach.

- [ ] **Step 1:** Copy `ontology_bundles/air_defense_v3/ontology.yaml` → `tests/fixtures/ontology/air_defense_v3_snapshot.yaml`. Verify byte-for-byte equality via `cmp`.
- [ ] **Step 2:** Update Phase 3 parity tests (Tasks 20–23) to load the oracle via `load_ontology(path=Path("tests/fixtures/ontology/air_defense_v3_snapshot.yaml"))` instead of `load_ontology(bundle_key="air_defense_v3")`. Assertion logic unchanged — both return the same dict shape.
- [ ] **Step 3:** Run parity tests under current (pre-deletion) state to confirm no change in outcome. PASS baseline established.
- [ ] **Step 4:** Commit: `test(bundle): freeze ontology.yaml snapshot fixture for parity tests`

This task MUST land before Task 45. The tasks.json dependency graph is updated so Task 45 is blocked by Task 44b.

**Fixture lifecycle:** the snapshot is intentionally frozen — it does NOT get updated when the Pydantic canonical schema evolves. When a Pydantic change produces a deliberate ontology-dict change (e.g. Plan 2's identity redesigns), the workflow is:
1. Make the Pydantic change; parity test fails (expected).
2. Regenerate the fixture from introspection: `python -m ontology_bundles.air_defense_v3.introspect dump > tests/fixtures/ontology/air_defense_v3_snapshot.yaml`.
3. Review the diff manually — this is the point where the change becomes visible and reviewable.
4. Commit both the Pydantic change and the fixture bump in the same commit.

The fixture is a review gate, not an automation target.

### Task 45: Delete `air_defense_v3/ontology.yaml` while preserving the public loader contract

**The loader contract (`bundle_key`, `path`) is preserved permanently.** `load_ontology()` keeps its current signature and resolution order. What changes:
- Only the **`air_defense_v3` bundle's `ontology.yaml`** is deleted; when that bundle is loaded without an explicit `path`, introspection serves the data.
- Explicit `path=<file>` calls continue to load the referenced YAML file unchanged — test fixtures, path-based tooling, and `load_validation_matrix(path=...)` at `:186` all keep working.
- Other bundle keys (none exist today, but the signature must not foreclose them) continue to resolve through the normal bundle-lookup path, which can still find a YAML file per bundle directory.
- The `ONTOLOGY_SOURCE` env var becomes obsolete for `air_defense_v3` (introspection is the only source when no path is given) and is removed. For future bundles, this plan doesn't decide their migration path — they can live as YAML under `bundle_key` until they're separately migrated.

**Why not delete the YAML-reading code path entirely:** callers pass explicit `path=` today (fixtures at a minimum), and the reviewer's contract-regression risk is real. Deleting that path breaks tests and tooling with no benefit — the YAML loader is still 40 lines of stable code, and preserving it is strictly additive.

- [ ] Full unit + integration suite with `ONTOLOGY_SOURCE=pydantic` (default from Task 34). PASS.
- [ ] Delete `ontology_bundles/air_defense_v3/ontology.yaml`.
- [ ] Simplify `load_ontology()`: remove the `ONTOLOGY_SOURCE` env read (introspection is unconditional for the `air_defense_v3` default-lookup path); KEEP `path=` handling; KEEP `bundle_key` resolution for any future non-default bundles.
- [ ] Regression test: `load_ontology(path=<fixture_path>)` returns the fixture contents (not introspection), proving path-based loads survive the YAML deletion.
- [ ] Run full suite — PASS.
- [ ] Commit: `refactor(bundle): delete air_defense_v3/ontology.yaml; preserve path-based loader contract`

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

- [ ] `git log --oneline` — expect ~55 commits including Phase 8.
- [ ] Optional squash to one feature-branch commit if team policy prefers.
- [ ] Update MEMORY: create `project_pydantic_ontology_ssot.md` capturing the architecture change, the intra-pass DTO→typed-edge migration, the system_links DTO exception, and the deletion of `ontology.yaml`.
- [ ] Commit memory file.

---

## Chunk 8: Phase 8 — Provenance plumbing (`graph_json` mentions fix)

Goal: extend the docling-graph service's `ExtractPassResponse` with an additive provenance payload so `_serialize_for_audit` can write real `mentions` and `derive_structure_links` can produce non-zero entity-chunk edges. Pure additive changes — `pass_output` is unchanged; old consumers ignore the new field.

### Task 50: Define `ExtractionProvenance` schema

**Files:**
- Create: `docker/docling-graph/app/schemas.py` — add `ExtractionProvenance` model
- Modify: `docker/docling-graph/app/main.py` — add field to `ExtractPassResponse`

```python
class ExtractionProvenance(BaseModel):
    """Per-extracted-entity-instance provenance link to source DoclingDocument elements.

    Additive payload — does not change pass_output. Consumers that don't read
    provenance ignore this field.

    The `instance_id` disambiguates between distinct extracted instances that
    happen to share the same identity tuple (same `ontology_name` +
    `identity_values`). This matters for:
      - Same-identity duplicates: two separate extractions with same identity
        won't collapse to one provenance bucket.
      - Empty-identity entities (e.g. PROPULSION_STACK): every instance is
        separately trackable even though logical identity is an empty tuple.
    Downstream merge-preserving dedup (Step 2b) unions provenance by
    instance_id, not by identity, so information is retained."""
    instance_id: str = Field(..., description="Unique id per extracted instance in this response (e.g. UUID).")
    ontology_name: str = Field(..., description="Canonical entity_type name (e.g. RADAR_SYSTEM).")
    identity_values: dict[str, Any] = Field(..., description="Field-name → value for the entity's graph_id_fields (may be empty).")
    element_uid: str = Field(..., description="DoclingDocument element where the entity was extracted. REQUIRED for chunk-linking; rows without element_uid cannot produce mentions and are dropped by the worker before provenance aggregation (Task 52).")
    page: Optional[int] = Field(None, description="Observational secondary field. NOT sufficient on its own for chunk linking — element_uid is the authoritative handle.")
    chunk_index: Optional[int] = Field(None)


class ExtractPassResponse(...):
    ...
    provenance: list[ExtractionProvenance] = Field(default_factory=list)
```

**Worker-side wiring (Task 52a):** `MergedEntityRecord.provenance: list[ExtractionProvenance]` accumulates *all* `ExtractionProvenance` entries whose logical identity matches the record — one per distinct `instance_id`. No bucket collapse even when identities match. Empty-identity entities still carry per-instance provenance. `_serialize_for_audit` (Task 53) iterates the full provenance list to emit `mentions`, so every extracted instance contributes a mention row.

- [ ] **Step 1:** Add the model. Existing tests still pass (additive).
- [ ] **Step 2:** Commit: `feat(docling-graph/schemas): ExtractionProvenance — additive payload for mentions`

### Task 51: Service captures element_uid during extraction

**Files:**
- Modify: `docker/docling-graph/app/main.py:546` (after `template_instance.model_dump()`)

The DoclingDocument JSON the service receives has `texts: [{self_ref, content, ...}]` etc. The extracted entities reference these via internal pointers. After `run_extraction_pass` returns the context, walk `context.knowledge_graph` (a NetworkX graph), and for each node with an entity-type label, pull its `element_uid` provenance from the graph metadata. Build the `ExtractionProvenance` list and attach to the response.

**Strengthened contract: `element_uid` is required, not optional.** The downstream `derive_structure_links` mention path at `pipeline.py:4347` resolves chunks exclusively via `element_uid`; a provenance row with only `page` produces zero mention-based edges AND cannot be used to compute `artifact_ids` for fallback. Making `element_uid` optional at the schema level would let page-only rows pass service tests while silently degrading chunk-linking. Drop them at service emission time instead: if the service cannot resolve an element_uid for a node in `context.knowledge_graph`, that node simply doesn't produce a provenance row (log WARNING, surface in service metrics). `page` remains as an observational secondary field for operator visibility.

- [ ] **Step 1:** Failing test: hit `/extract-pass` against a small fixture doc; assert `response.provenance` is non-empty and every entry has `element_uid` (not nullable) plus `ontology_name`, `identity_values`. Page is optional.
- [ ] **Step 2:** Second failing test: fixture doc with a knowledge_graph node whose metadata lacks element_uid. Assert the service drops that node from provenance (does NOT emit a provenance row with element_uid=null or with only page populated). Service log contains a WARNING.
- [ ] **Step 3:** Implement: walk `context.knowledge_graph`; skip nodes without resolvable element_uid (WARN); populate page where available.
- [ ] **Step 4:** Run tests. PASS.
- [ ] **Step 5:** Commit: `feat(docling-graph/service): emit ExtractionProvenance with required element_uid`

### Task 51b: `pass_output` ↔ `provenance` round-trip contract test

**Files:**
- Create: `docker/docling-graph/tests/test_provenance_roundtrip.py`
- Reference: `docker/docling-graph/app/main.py:523` (pass_output construction) + `:546` (provenance capture) + `app/services/extraction_merge.py:316` (`logical_identity_from_dict` worker-side join).

`pass_output` is built from `template_instance.model_dump()`; `provenance` is built from `context.knowledge_graph`. These are two independent code paths in the same handler. The worker later joins them via `logical_identity_from_dict` — if the service's provenance emission disagrees on `ontology_name` or `identity_values` with what lands in `pass_output`, the join silently drops rows in `Task 52a`'s drop-with-WARNING contract. Task 51's service-side test and Task 52a's worker-side tests each verify one half; neither catches cross-boundary drift.

This test asserts: for every `provenance[i]` row, there exists a matching entity in `pass_output` such that worker-side normalization (`logical_identity_from_dict(provenance[i].ontology_name, provenance[i].identity_values, ontology, doc_id)`) returns a `LogicalIdentity` equal to what `_build_logical_identity(...)` produces for the corresponding `pass_output` entity. If any provenance row fails the lookup, the test fails — proves the service-emitted keys are actually joinable by the worker.

**Scope decision (pinned — CI-realistic):** this is a **service-side unit test with a stubbed context**, NOT an end-to-end test against a live `/extract-pass` endpoint. Rationale:
- Live `/extract-pass` requires a running docling-graph service, which requires the vLLM backend, which requires a GPU. That infrastructure does not exist in standard CI; the test would either be skipped or flake.
- The contract being protected is the identity-join between two code paths WITHIN the service handler: `template_instance.model_dump()` at `main.py:523` and `context.knowledge_graph` walk at `:546`. Both paths run in-process against in-memory objects; a live HTTP round-trip adds no additional coverage.
- The `context` object (a `DoclingDocument`-shaped input + extraction state) can be built directly in the test from a committed fixture DoclingDocument JSON file and a synthesized `template_instance`. Call the service's provenance-emission function and its pass_output-emission function on the same context; assert the join.

Concrete test shape:
```python
def test_pass_output_provenance_roundtrip():
    # Load fixture: a DoclingDocument JSON + an expected extraction template_instance.
    context = _build_test_context(fixture_dir / "radar_page.json")
    template_instance = _build_test_template_instance(fixture_dir / "radar_extracted.json")
    pass_output = _build_pass_output(template_instance)       # same code path as main.py:523
    provenance = _build_provenance(context.knowledge_graph)   # same code path as main.py:546
    ontology = load_ontology()
    doc_id = "test-doc-001"
    for prov in provenance:
        identity = logical_identity_from_dict(
            prov.ontology_name, prov.identity_values, ontology, doc_id,
        )
        assert identity is not None, f"provenance row failed normalization: {prov}"
        matching = [e for e in pass_output.entities
                    if _build_logical_identity(type(e).ontology_name, e, ontology, doc_id) == identity]
        assert len(matching) >= 1, f"no pass_output match for provenance {prov}"
```

- [ ] **Step 1:** Create fixture DoclingDocument JSON + expected extraction template_instance JSON in `docker/docling-graph/tests/fixtures/provenance_roundtrip/`. Keep small (one page, one extracted entity) for test speed.
- [ ] **Step 2:** Failing test with deliberate mismatch (provenance `identity_values={"name": "X"}`, pass_output entity `name="Y"`). Assert the test fails.
- [ ] **Step 3:** Correct fixture: test PASSES under current (Phase 8) service code.
- [ ] **Step 4:** Commit: `test(docling-graph): pass_output ↔ provenance identity round-trip — service-unit, CI-safe`

**NOT required:** an additional live-service integration test. If the service's in-process functions pass the round-trip, HTTP serialization cannot break the identity contract — the same objects are emitted through the response body. If future work adds response-body transformation (e.g. protobuf serialization), a live-service test becomes warranted; not today.

### Task 52: Worker carries provenance into `PassResult`

**Files:**
- Modify: `app/services/extraction_merge.py` (`PassResult` dataclass)
- Modify: `app/workers/pipeline.py:1707` (`_parse_pass_response`)

Add `provenance: list[ExtractionProvenance]` to `PassResult`. `_parse_pass_response` reads it from the response body.

- [ ] Failing test + impl + commit: `feat(worker): carry provenance through PassResult`

### Task 52a: Aggregate provenance into `MergedEntityRecord` with `instance_id` dedup

**Files:**
- Modify: `app/services/extraction_merge.py` — `MergedEntityRecord` and `merge_and_resolve`

Without this step, `_serialize_for_audit` has no path from a `MergedEntityRecord` to its source `element_uid`s. Aggregation has two layers:

1. **Collection by canonical `LogicalIdentity`** (merge-layer join): every `ExtractionProvenance` entry is converted to a `LogicalIdentity` via the **existing** `logical_identity_from_dict(entity_type, identity_dict, ontology, document_id)` helper at `app/services/extraction_merge.py:316`. This is the same helper the worker already uses to convert upstream-entity-ref payloads into `LogicalIdentity` for `PassResult.upstream_refs`. The produced `LogicalIdentity` is compared for equality against `MergedEntityRecord.identity` — a `@dataclass(frozen=True)` with value-based equality. If the two logical identities are equal, the provenance entry is appended to `merged_entity.provenance`.

   **Do NOT invent a parallel `(ontology_name, identity_values)` dict comparison path.** The codebase has three canonical identity surfaces: `_build_logical_identity(entity_type, instance, ontology, document_id)` at `extraction_merge.py:288` (from Pydantic instance), `logical_identity_from_dict(entity_type, identity_dict, ontology, document_id)` at `extraction_merge.py:316` (from raw dict — the call site relevant here), and the `LogicalIdentity` dataclass itself. Matching on raw dicts would be a fourth identity representation that drifts on `identity_scope` handling (document-scoped vs global), on missing/extra keys (the canonical helper returns `None` and the caller drops the entry; a naive dict comparison would silently miscompare), and on key ordering (dict iteration order vs ontology-declared `identity_fields` order). Using `logical_identity_from_dict` guarantees parity with the upstream-refs path, the entity-merge path, and the edge-dedup key.

   **Cross-caller identity-helper parity contract (addresses reviewer finding S2):** three separate call sites invoke `logical_identity_from_dict` / `_build_logical_identity`:
   - upstream-refs path (Phase 1/earlier)
   - entity-merge path (`_build_logical_identity` from Pydantic instances in `merge_and_resolve`)
   - provenance aggregation (this task, from `ExtractionProvenance` dicts)

   Add a **single shared contract test** that generates a `LogicalIdentity` for the SAME entity via all three helpers with equivalent inputs, and asserts all three return objects that compare equal via dataclass `__eq__`. Test shape:
   ```python
   def test_logical_identity_cross_caller_parity():
       antenna = AntennaEntity(name="FF-1")
       ontology = load_ontology()
       doc_id = "doc-test"
       from_instance = _build_logical_identity("ANTENNA", antenna, ontology, doc_id)
       from_dict     = logical_identity_from_dict("ANTENNA", {"name": "FF-1"}, ontology, doc_id)
       # Upstream refs path: constructs identity from a different dict form
       from_ref      = logical_identity_from_dict("ANTENNA", {"name": "FF-1"}, ontology, doc_id)
       assert from_instance == from_dict == from_ref
   ```
   If any of the three drifts (different identity_scope handling, different key-tuple order, different document_id coupling), the test fails. Pins the three-caller contract that otherwise relies on each implementation getting every detail right independently.

   **Failure-to-normalize contract:** if `logical_identity_from_dict(...)` returns `None` for a `ExtractionProvenance` entry (missing required identity key, unknown `entity_type`), the entry is **dropped with a WARNING log** citing the provenance instance_id + entity_type — same "drop ref" semantic the upstream-refs consumer uses, with an observability hook so provenance-loss from malformed service payloads surfaces rather than hiding.

   **Join cardinality contract (pass_output ↔ provenance):** the join semantics between an entity in `pass_output` and its provenance rows are many-to-one (provenance → record) and one-to-many (record → provenance):
   - **Many-to-one:** multiple provenance rows can join to the same `MergedEntityRecord` (the normal case — a single merged entity collects all its mentions across passes). The `logical_identity_from_dict(prov.ontology_name, prov.identity_values, ontology, doc_id)` result must equal the record's `LogicalIdentity` for the row to attach. Each attaching row keeps its distinct `(instance_id, element_uid)` pair; dedup only collapses true duplicates (same pair).
   - **One-to-many:** a single `MergedEntityRecord` can hold unbounded provenance rows. No upper cap; consumers iterate the list.
   - **Zero matches case (no MergedEntityRecord with that identity):** the provenance row is dropped with a WARNING distinct from the malformed case. Reason: an `ontology_name`/`identity_values` tuple that doesn't map to any merged entity indicates the service extracted something the worker-side merge filtered out (e.g. VALIDATION_MATRIX rejection, post-merge pruning). This is not a malformed payload — it's a legitimate drift between pass_output and post-merge state. Log format distinguishes: `"drop provenance for post-merge-absent entity identity=... instance_id=..."` vs `"drop malformed provenance ..."` for observability.
   - **Worker-side drop counters:** both drop paths (malformed + post-merge-absent) increment dedicated counters on the merge run, surfaced in `PassResult` or the post-merge metrics. Without aggregate counters, the WARNINGs are invisible; with them, a monitoring threshold can trigger on elevated drop rates. Counters are populated alongside the existing `rejection_reasons` dict in the pass-run metrics.

2. **Deduplication by `(instance_id, element_uid)`** (within-entity cleanup, implemented in `merge_and_resolve` at `app/services/extraction_merge.py`): dedup happens inside the merge-aggregation loop where provenance rows are first gathered onto their matching `MergedEntityRecord`. The implementation uses a set-during-accumulation pattern rather than a post-collection cleanup pass:

   ```python
   # Inside merge_and_resolve's provenance-aggregation loop:
   seen_keys: set[tuple[str, str]] = set()  # (instance_id, element_uid)
   for prov in incoming_provenance_rows:
       identity = logical_identity_from_dict(
           prov.ontology_name, prov.identity_values, ontology, document_id,
       )
       if identity is None:
           logger.warning("drop malformed provenance instance_id=%s type=%s",
                          prov.instance_id, prov.ontology_name)
           continue
       record = merged_index.get(identity)
       if record is None:
           continue  # provenance for an entity that didn't merge through — drop
       key = (prov.instance_id, prov.element_uid)
       if key in seen_keys:
           continue  # duplicate echo — pure redundancy, drop
       seen_keys.add(key)
       record.provenance.append(prov)
   ```

   The set-during-accumulation form replaces the earlier "accumulate then dedup" two-phase description: cleaner contract, same result, one loop instead of two, and the invariant "no two entries share `(instance_id, element_uid)`" is obvious from the code. Different `instance_id`s with the same identity (separate extractions of the same logical entity) are all retained; empty-identity entities (PROPULSION_STACK shape) each keep distinct entries because every extraction has a unique `instance_id` regardless of identity.

   **Performance note (set vs sort-dedup):** for the realistic workload — typical document has ~1K entities × ~3 mentions each = ~3K provenance rows per merge — Python's `set[tuple[str, str]]` lookup is O(1) amortized and the set itself stays small (tuples are hashable with no collisions for normal UUIDs). Sort-then-dedup would be O(N log N) with better cache locality but requires an extra allocation and a second pass. At 3K rows the difference is microseconds either way; set is chosen for code clarity. If a future workload pushes provenance lists into the 100K+ range, revisit — but that volume also implies rethinking the per-pass-run aggregation model, not just the dedup algorithm.

   **Location pin:** this loop replaces the current `MergedEntityRecord` construction path inside `merge_and_resolve`. NOT `_upsert_document_graph_extraction` and NOT `_serialize_for_audit` — those are later consumers and must see a deduped list already in `record.provenance`.

   **Why composite, not bare `instance_id`:** `ExtractionProvenance` is a single-`element_uid` row. If one extracted instance legitimately spans multiple source elements (e.g. a service refactor that emits one provenance row per referenced element while keeping a single `instance_id`), bare-`instance_id` dedup would drop real distinct mentions and under-link chunks. Composite key preserves each `(instance_id, element_uid)` pair as a separate mention while still collapsing pure duplicate echoes. `element_uid` is required (Task 51), so the composite key is always well-defined.

   Without this dedup step, `_serialize_for_audit` would emit the same extracted-instance/element pair as multiple mentions when cross-pass merging echoes the same instance through multiple paths. The `instance_id` introduced in Task 50, paired with `element_uid`, forms a stable, serialization-safe per-mention key.

Add `provenance: list[ExtractionProvenance]` to `MergedEntityRecord`. `_serialize_for_audit` (Task 53) reads `merged_entity.provenance` directly — no side map.

- [ ] **Step 1:** Failing tests:
  - (a) Two passes each provide one `ExtractionProvenance` entry for the same logical entity (distinct `instance_id`s) → `MergedEntityRecord.provenance` has 2 entries.
  - (b) Same pass emits two `ExtractionProvenance` entries with the SAME `(instance_id, element_uid)` for the same logical entity (duplicate echo) → after composite-key dedup, `MergedEntityRecord.provenance` has 1 entry.
  - (b2) **Multi-element same instance:** service emits two `ExtractionProvenance` entries with the SAME `instance_id` but DIFFERENT `element_uid`s (one logical instance referenced from multiple source elements). After composite-key dedup, both entries are RETAINED (`MergedEntityRecord.provenance` has 2 entries) — each contributes a distinct mention row in `_serialize_for_audit`. Bare-`instance_id` dedup would falsely drop one; the test pins the composite-key contract.
  - (c) Empty-identity entity (PROPULSION_STACK): two separate extractions, each with its own `instance_id` → `MergedEntityRecord.provenance` has 2 entries (not collapsed despite identical empty-identity tuple).
  - (d) **Identity-helper contract:** a provenance entry with `identity_values={"name": "FF-1"}` for `ontology_name="ANTENNA"` is aggregated under a `MergedEntityRecord` whose `identity` was built by `_build_logical_identity` from a Pydantic `AntennaEntity(name="FF-1")`. Assert the two `LogicalIdentity` instances compare equal via dataclass `__eq__`, and the provenance lands on that record. Proves the `logical_identity_from_dict` / `_build_logical_identity` paths produce equal objects for matching inputs.
  - (e) **Document-scoped identity:** same fixture as (d) but with the ontology marking `ANTENNA` as `identity_scope="document"`. Both identities must carry the same `document_id`; assert aggregation still lands the provenance entry on the matching record. If `document_id` differed the test would fail — the identity dataclass includes `document_id` for document-scoped types.
  - (f) **Malformed payload drops with warning:** a provenance entry with `ontology_name="ANTENNA"` but `identity_values={}` (missing the required `name` key). `logical_identity_from_dict` returns `None`; assert the entry is NOT appended to any `MergedEntityRecord.provenance` list, AND a WARNING log line cites the instance_id + entity_type (assert via `caplog`).
  - (g) **Unknown entity_type drops with warning:** `ontology_name="BOGUS_ENTITY"` not in ontology → same drop-with-warning semantics as (f).
- [ ] **Step 2:** Add the field; update merger aggregation logic to: (1) call `logical_identity_from_dict` on each `ExtractionProvenance`, (2) drop-with-warning on `None` return, (3) find matching `MergedEntityRecord` by `LogicalIdentity` equality, (4) append, (5) dedup by `instance_id`.
- [ ] **Step 3:** Run tests — PASS.
- [ ] **Step 4:** Commit: `feat(merge): aggregate provenance into MergedEntityRecord; dedup by instance_id`

### Task 53: `_serialize_for_audit` writes real `mentions` AND `nodes`

**Prerequisite:** Task 52b (LogicalIdentity.serialize_as_entity_id) MUST land before this task — Task 53 calls `record.identity.serialize_as_entity_id()` to compute `entity_id` values for both `nodes[]` and `mentions[]`. Tasks.json `blockedBy` reflects this ordering; the narrative defines Task 52b further down for topical grouping with other Phase 8 serializer concerns, but the execution order is **52b → 53 → 53b**.

**Files:**
- Modify: `app/workers/pipeline.py:233` (`_serialize_for_audit`)

Two keys must be written. `derive_structure_links` at `:4362–4377` reads both `mentions` (primary path) and `nodes` (fallback path). Writing only `mentions` leaves the fallback dead — `all_extracted_entities` stays empty, and any entity without mentions gets no fallback chunk links.

**`mentions`:** walk `merged.entities`; for each `MergedEntityRecord`, iterate its `provenance: list[ExtractionProvenance]` (populated by Task 52a). For each entry emit `{entity_name, entity_type, element_uid, page}` into `graph_json["mentions"]`. `entity_name` derives from `record.display_label`; `entity_type` from `record.identity.entity_type`.

**`nodes`:** walk `merged.entities` once; for each `MergedEntityRecord` emit `{name, entity_type, entity_id, rid, artifact_ids}` into `graph_json["nodes"]`, where:

- `name` = `record.display_label` (human-readable label from `build_display_label`).
- `entity_type` = `record.identity.entity_type`.
- `entity_id` = `record.identity.serialize_as_entity_id()` — see the new canonical serializer introduced by **Task 52b** below.
- `rid` = `identity_to_rid[record.identity]` — populated via the signature changes below.
- `artifact_ids` = derived from provenance — see the full threading spec below.

**Mentions also carry `rid` and `entity_id`.** Each `mentions[]` entry includes `{entity_name, entity_type, entity_id, rid, element_uid, page, chunk_index}`. The `derive_structure_links` mention-path at `pipeline.py:4362` builds `EntityChunkEdge` records directly from mention iteration; if `rid` is only on nodes, the mention path would need an extra `entity_id → rid` dict lookup per mention. Putting `rid` on mentions too avoids that join and means the single mention iteration produces complete edge records.

**Consistency invariant (tested):** for any mention `m` and the matching node `n` (same `entity_id`), `m["rid"] == n["rid"]`. Both values come from the same `identity_to_rid` map at serialization time. A contract test asserts this across a multi-entity multi-mention fixture — if the two ever diverge, a bug exists in the serializer.

`entity_id` is what disambiguates same-name same-type entities — e.g. two `SECTION`s both with `heading="Overview"` but different `page_start` values produce different `entity_id` strings. `build_display_label` collapses such sections to `"Overview"`, but `entity_id` preserves `(heading, page_start)`. Also add the same `entity_id` to each `mentions` entry emitted.

### Task 52b: Canonical `LogicalIdentity -> str` serializer

**Files:**
- Modify: `app/services/extraction_merge.py:49` (`LogicalIdentity` dataclass) — add method.

Currently `LogicalIdentity` has two outward conversions: `identity_values_dict()` at `:63` and `as_upsert_identity_dict()` at `:67`. There is **no canonical string serializer**. Multiple new sites (`_serialize_for_audit` node/mention emission, `derive_structure_links` suppression, persisted `EntityChunkEdge.entity_id` column) each need a stable `LogicalIdentity -> str` — inventing that inline at each call site produces drift on escaping, ordering, and document-id handling.

Add:
```python
ENTITY_ID_FORMAT_VERSION = "v1"

def serialize_as_entity_id(self) -> str:
    """Canonical stable string for this identity.

    Format: "v1::{entity_type}::{k1}={v1}|{k2}={v2}|...[|__doc__={document_id}]"
    - Leading version token "v1" is the format-schema version. Future format
      changes (e.g. new identity-field semantics, different separators) bump
      to v2 and coexist with v1-persisted entity_ids during migration. A
      reader that sees v1 strings under a v2 code base uses a pinned v1
      parser — no data surgery required.
    - Identity fields in declared order (matches identity_field_names tuple order).
    - Values converted via repr() for unambiguous round-trip across ints, strings
      with colons/pipes, None, etc. repr() escapes embedded quotes, so values
      containing "::" or "|" round-trip without collision.
    - Document-scoped identities carry __doc__=document_id so the same
      identity tuple in different documents produces distinct entity_ids.

    Used by _serialize_for_audit.nodes/mentions emission, derive_structure_links
    suppression, EntityChunkEdge persistence. Single canonical surface — no
    inline f-strings inventing the format at individual call sites.
    """
    parts = [f"{k}={v!r}" for k, v in zip(self.identity_field_names, self.identity_tuple, strict=True)]
    if self.scope == "document" and self.document_id is not None:
        parts.append(f"__doc__={self.document_id}")
    return f"{ENTITY_ID_FORMAT_VERSION}::{self.entity_type}::{'|'.join(parts)}"
```

- [ ] **Step 1:** Failing tests:
  - (a) Stable: same identity input twice → same string (no randomness, no dict ordering).
  - (b) Distinct: two identities differing only in identity_tuple values → different strings.
  - (c) Document-scoped: same tuple, different `document_id` → different strings.
  - (d) Global-scoped: `__doc__` suffix absent; `document_id=None` tolerated.
  - (e) Embedded delimiter: a value containing `::` or `|` round-trips without collision (repr() quoting handles it).
  - (f) **Write-only contract — no deserialization allowed.** `entity_id` is a stable opaque string for comparison and edge-property persistence. It is NOT a parseable identity carrier. If a future consumer needs `(entity_type, identity_tuple)` from a persisted edge, it should join against the audit blob's `nodes[]` entries (which carry the fields separately and explicitly) rather than parse `entity_id`. Document this in the method's docstring as an explicit ban. Rationale: versioning the format (Task 52b's `v1::` prefix) anticipates future format changes, and a parser would couple consumers to a specific version's format. A "parse entity_id" method would also invert the point of having a canonical helper — callers would parse and then re-construct, re-introducing the drift risk this helper exists to prevent.
- [ ] **Step 2:** Implement method.
- [ ] **Step 3:** Refactor any existing inline `f"{entity_type}::..."` pattern (if planning doc drafts introduced any) to call `serialize_as_entity_id()`.
- [ ] **Step 4:** Run tests — PASS.
- [ ] **Step 5:** Commit: `feat(merge): LogicalIdentity.serialize_as_entity_id — canonical id string for audit/structure-links/chunk-edges`

**Persistence decision (locks the reviewer's "column or JSON metadata" open question):** `entity_id` becomes a **first-class property on the `EntityChunkEdge` ArcadeDB edge class** at `app/services/graph_store.py:106`. ArcadeDB edge classes are schema-flexible; adding a property does not require a SQL migration. The `EntityChunkEdge` dataclass gets an `entity_id: str` field; `batch_create_entity_chunk_edges` (or whatever the existing insert API is) accepts it; `derive_structure_links` populates it from the `mentions[]`/`nodes[]` entries written in Task 53.

Rejected alternative: JSON metadata blob on the existing edge. Not queryable, not indexable, opaque to downstream consumers that want to resolve chunk-link edges back to specific entities. The first-class property is strictly better-scoped and costs the same to write.

- [ ] Edge-class property addition lives under Task 53b (not a separate task) since it's tightly coupled to the suppression-key change.

- [ ] Failing test 4 (same-name same-type distinction): two `SECTION` entities with `heading="Overview"` but `page_start=3` and `page_start=47`. Both appear in `nodes` with distinct `entity_id`s. Both appear in `mentions` under their respective `entity_id`s. `derive_structure_links` runs end-to-end; assert both sections get their own chunk-link edges, not merged.

- [ ] Failing test 1 (mentions): fixture merged result with provenance → assert `mentions` populated.
- [ ] Failing test 2 (nodes): fixture merged result → assert `nodes` has one entry per merged entity with `name`, `entity_type`, `id`.
- [ ] Failing test 3 (fallback for unmentioned-but-provenance-bearing entity): a merged entity has provenance (non-empty `artifact_ids` on its node) but is absent from `mentions[]` for the chunks covering that artifact. Assert it receives artifact-scoped fallback chunk links. **Important:** this test exercises the `artifact_ids != []` branch only. The old draft of this test allowed entities with NO provenance at all to get fallback — that contract is dropped. Task 53b's fail-closed rule is the single fallback contract: `artifact_ids=[]` → skip + WARNING. Fallback only runs for entities with at least one resolvable artifact.

- [ ] Failing test 3b (empty-identity + unmapped element_uid): PROPULSION_STACK entity (identity_fields=[], empty-identity) with a provenance row whose `element_uid` is NOT present in the `DocumentElement` table for that document_id. Aggregation succeeds (identity matches), but `_serialize_for_audit` → `element_uid_to_artifact_id` lookup returns empty → `artifact_ids=[]`. Assert: (a) the entity's `nodes[]` entry has `artifact_ids=[]`; (b) `derive_structure_links` hits the fail-closed skip path with a WARNING naming the entity's `entity_id` + element_uid; (c) no chunk-link edges are created for it; (d) a drop counter `fallback_skipped_no_artifact_ids` is incremented. **This is intended behavior:** an element_uid the worker cannot resolve to an artifact means provenance data is stale or the element was purged — fail-closed is correct over fan-out-to-all-chunks. Test documents the contract so a future reader doesn't mistake it for a bug.

- [ ] Failing test 3c (zero-extractions document): extraction pipeline returns zero entities from every pass (degenerate case — empty document, all-image document, or catastrophic LLM failure). `_serialize_for_audit` writes `nodes=[]`, `mentions=[]`, `element_to_artifact={}` into the audit blob. `derive_structure_links` runs: outer loop over `nodes[]` is no-op; no edges created; no WARNINGs logged (empty is not an error). Assert: (a) audit blob persists successfully with the empty keys; (b) `derive_structure_links` returns without exceptions; (c) downstream dashboards reading `graph_json->nodes` see `[]` (length 0) not KeyError. This pins the "a document can legitimately have zero extracted entities" contract — degenerate docs shouldn't break the pipeline or crash the reader.
- [ ] Commit: `feat(worker): _serialize_for_audit writes mentions AND nodes; restores derive_structure_links fallback`

### Task 53b: Fix `derive_structure_links` fallback suppression — key by `entity_id`

**Files:**
- Modify: `app/workers/pipeline.py:4359, :4370, :4383, :4406, :4430` (`derive_structure_links`)

The fallback uses `mentioned_entities: set[str]` keyed by name alone (`:4359`) and filters with `if n not in mentioned_entities` at `:4383`. Two failure modes:
- **Different types, same name** (e.g. `PLATFORM` "Fan Song" + `RADAR_SYSTEM` "Fan Song"): one mention suppresses the other.
- **Same type, same name, different identity** (e.g. two `SECTION`s with `heading="Overview"` but `page_start=3` and `:47`): `build_display_label` collapses both to `"Overview"`; even `(name, type)` conflates them; only the full `LogicalIdentity` (`heading + page_start`) disambiguates.

The primary fix is to key suppression on the `entity_id` string that Task 53 writes into every `nodes[]` and `mentions[]` entry — that string IS the serialized `LogicalIdentity`, so it distinguishes both cases correctly. Persisted chunk-link edges also carry `entity_id` so downstream consumers can resolve back to the specific entity.

```python
mentioned_entity_ids: set[str] = set()
...
# For each mention:
mentioned_entity_ids.add(mention["entity_id"])
edge_tuples.append((mention["entity_name"], mention["entity_type"], mention["entity_id"], chunk_id))
...
# For fallback:
entities_needing_fallback = [
    node for node in all_extracted_entities
    if node["entity_id"] not in mentioned_entity_ids
]
```

Persist `entity_id` as a **first-class property** on the `EntityChunkEdge` ArcadeDB edge class at `app/services/graph_store.py:106` (decision locked in Task 52b). ArcadeDB edge classes are schema-flexible; adding a property is not a SQL migration. Update:
- `EntityChunkEdge` dataclass at `graph_store.py:106`: add `entity_id: str` field AND `source_rid: str` field (see next bullet for why source_rid is critical).
- Batch-insert API: accept and write the new fields.
- `derive_structure_links` at `:4430`: populate from the corresponding `mentions[]` / `nodes[]` entry's `entity_id` + the entity's already-resolved RID.
- Existing rows: no backfill required — Phase 7 E2E re-runs `derive_structure_links` on the target doc under the new code path. Legacy rows carry `entity_id=None`, `source_rid=None`; readers tolerate nullables. Cleanup is out of scope.

Call sites that compute or compare `entity_id` MUST use `LogicalIdentity.serialize_as_entity_id()` from Task 52b — no inline f-strings.

**Source-RID resolution fix (critical — `entity_id` property alone is not sufficient):**

The ArcadeDB writer at `app/services/arcadedb_graph.py:1920` currently resolves the source vertex by a subquery keyed on `name` + `entity_type`:
```sql
CREATE EDGE EXTRACTED_FROM FROM (SELECT FROM {entity_type} WHERE name = :name_i AND entity_type = :etype_i LIMIT 1) TO :rid_i SET ...
```

For two entities with the same `name` and `entity_type` but different identity tuples (e.g. `SECTION("Overview", page_start=3)` vs `SECTION("Overview", page_start=47)`), both match the `WHERE`; the `LIMIT 1` picks whichever vertex the storage engine returns first. Attach-to-wrong-vertex bug — independent of whether `entity_id` is persisted on the *edge*, because the edge's FROM-clause never consults it.

**Task-boundary constraint:** `identity_to_rid: dict[LogicalIdentity, str]` is built in `derive_ontology_graph` during `_import_graph_phase_nodes` at `pipeline.py:3945`. `derive_ontology_graph` then calls `_upsert_document_graph_extraction(...)` at `:3972` to write the audit snapshot and returns. The map does not survive the task boundary. `derive_structure_links` runs later as a separate Celery task and reconstructs its work from `DocumentGraphExtraction.graph_json` + artifact metadata at `:4347`. Threading an in-memory map across that boundary is not possible.

**Signature changes to get `identity_to_rid` AND `element_uid_to_artifact_id` into `_serialize_for_audit`:** the current signatures don't pass either, so both have to be threaded through. Concrete changes:

- `_upsert_document_graph_extraction(...)` at `pipeline.py:881`: add two required parameters — `identity_to_rid: dict[LogicalIdentity, str]` and `element_uid_to_artifact_id: dict[str, str]`. Every call site must pass both.
- Call site at `pipeline.py:3972` inside `derive_ontology_graph`:
  - `identity_to_rid` is in scope there (built at `:3945` by `_import_graph_phase_nodes`).
  - `element_uid_to_artifact_id` is NOT currently built anywhere — it must be constructed from `DocumentElement`, the only model that carries both `element_uid` (the docling element handle) and `artifact_id` (the artifact that element belongs to). `TextChunk` and `ImageChunk` have `artifact_id`/`document_id` only and do NOT carry `element_uid` (verified against `app/models/retrieval.py:11` and `:60`). The prior draft's "query text_chunks + image_chunks" was incorrect.
  - Build via shared helper `_build_element_uid_to_artifact_id(db, document_id) -> dict[str, str]` that queries `DocumentElement`:
    ```python
    def _build_element_uid_to_artifact_id(db, document_id: str) -> dict[str, str]:
        """Map every DoclingDocument element_uid to its owning artifact_id.

        DocumentElement is the only model that carries both. Used by
        _serialize_for_audit to derive artifact_ids on each node from
        its provenance element_uids, and by derive_structure_links for
        any element→artifact lookups the later task still needs.
        """
        rows = db.execute(
            select(DocumentElement.element_uid, DocumentElement.artifact_id)
            .where(DocumentElement.document_id == uuid.UUID(document_id))
        ).all()
        return {uid: str(aid) for uid, aid in rows if uid and aid}
    ```
  - **Index note:** `DocumentElement` already has `UniqueConstraint("document_id", "element_uid")` at `app/models/ingest.py:294`. PostgreSQL auto-creates a composite B-tree index on the constraint. The leading column (`document_id`) is the filter column here, so the query is an index range scan — no additional index required. Verified via the model definition; confirm during implementation with `EXPLAIN` on one document's row.
  - **Persistence (upgraded from follow-up to required):** the map is persisted in the audit blob as `graph_json["element_to_artifact"]: dict[str, str]`. `derive_ontology_graph` builds it once and writes it alongside `nodes[]`/`mentions[]`. `derive_structure_links` reads it from `DocumentGraphExtraction.graph_json` — no re-query. This makes the two tasks share a stable, snapshot-consistent view.
  - **Snapshot-consistency contract (pinned explicitly):** the persisted `element_to_artifact` is a snapshot of `DocumentElement` state at the moment `derive_ontology_graph` ran. If `DocumentElement` rows change between that run and a later retry of `derive_structure_links` (e.g. artifact deletion setting `artifact_id=None`, CASCADE from document deletion), the snapshot becomes stale relative to the live database. **This is accepted, not a bug:** the audit blob is explicitly the ingestion's view-of-the-world. If underlying elements change in a way that invalidates that view, the whole ingestion must re-run end-to-end — fixing it via live re-query in `derive_structure_links` would produce an internally inconsistent result (nodes[] and mentions[] still reference the ingestion-time elements, but artifact_ids would reflect current state). The alternative — live re-query every time — was the v30 design; v31 deliberately trades freshness for consistency. For the realistic failure mode (document deletion during retry), both designs produce bad output because both tasks read from a document being torn down; the remediation is the same (accept the retry fails, let the pipeline re-run).
  - **Signature implication:** `_serialize_for_audit` writes `element_to_artifact` into the audit blob root; `derive_structure_links` reads it from `graph_extraction.graph_json["element_to_artifact"]` instead of calling `_build_element_uid_to_artifact_id` again. The helper still exists (used only by `derive_ontology_graph`); `derive_structure_links` becomes a pure consumer of the audit blob.
  - **Legacy-blob fallback:** if `graph_json` lacks `element_to_artifact` (pre-Phase-8 audit blob), `derive_structure_links` WARNS and falls back to calling the helper — keeps pre-Phase-8 documents operable during the migration window without requiring backfill. The WARNING log message explicitly says: `"derive_structure_links: audit blob missing element_to_artifact; rebuilding from DocumentElement. Document pre-dates Phase 8; re-run derive_ontology_graph to regenerate audit blob."` This surfaces the migration state loudly to operators.
  - **Legacy-blob fallback test (explicit, addresses reviewer finding C1):** failing test loads a mock `DocumentGraphExtraction` with `graph_json = {"nodes": [...], "mentions": [...]}` but NO `element_to_artifact` key. Run `derive_structure_links`. Assert: (a) WARNING log emitted with the exact message above; (b) `_build_element_uid_to_artifact_id` helper IS called as the fallback; (c) fresh edges ARE created using the freshly-rebuilt map (pre-Phase-8 doc remains operable); (d) a counter `legacy_audit_blob_rebuilds` is incremented for operator visibility.
  - **Size measurement (addresses reviewer finding P1):** at audit-blob write time, `_serialize_for_audit` emits a log line: `"audit_blob_size doc_id=%s bytes=%d nodes=%d mentions=%d element_to_artifact=%d"`. Gives operators visibility into TOAST pressure without a separate metrics pipeline. Rough cost estimate for a typical 10-page document with dense extraction: ~650 KB JSONB (500 entities × 6 mentions × ~100 bytes + 10K element_to_artifact × ~30 bytes). PostgreSQL TOAST threshold is 2 KB; blobs of this size go out-of-line automatically. Not catastrophic; worth measuring.
- `_serialize_for_audit(merged, manifest)` at `pipeline.py:233`: signature becomes `_serialize_for_audit(merged, manifest, identity_to_rid, element_uid_to_artifact_id)`. `_upsert_document_graph_extraction` forwards both maps.
- Inside `_serialize_for_audit`:
  - `rid` on each `nodes[]` entry comes from `identity_to_rid[record.identity]`.
  - `artifact_ids` on each `nodes[]` entry comes from walking `record.provenance[*].element_uid` and mapping each through `element_uid_to_artifact_id`; deduped + sorted for determinism. Empty list if no provenance row resolves to an artifact.
- Failing test 1: call `_upsert_document_graph_extraction` with fake merged + fake `identity_to_rid`; assert persisted `graph_json["nodes"][*]["rid"]` matches the map.
- Failing test 2: call `_upsert_document_graph_extraction` with fake merged (each record has provenance with element_uids) + fake `element_uid_to_artifact_id`; assert persisted `graph_json["nodes"][*]["artifact_ids"]` matches the mapped+deduped+sorted list.
- Failing test 3 (helper parity): `_build_element_uid_to_artifact_id` produces identical output whether called from `derive_ontology_graph` or `derive_structure_links` for the same doc. Proves the shared helper eliminates drift between the two tasks.

Alternative rejected: adding `artifact_id` directly to `ExtractionProvenance` at service emission. The service doesn't know about artifacts — that's a worker-side concept. Adding it would require the worker to post-populate the field at `_parse_pass_response` time, which is earlier than the pipeline stage where the element→artifact map naturally exists (text_chunks query). The signature-threading approach keeps `ExtractionProvenance` focused on what the service actually captures (element_uid, page, identity, instance_id) and builds artifact enrichment once at the audit-assembly boundary.

Alternative also rejected: building the nodes-with-rid payload in `derive_ontology_graph` itself and passing a pre-built list into `_upsert_document_graph_extraction`. Works but splits audit-blob assembly across two functions and complicates unit testing of `_serialize_for_audit`. The signature-extension approach keeps assembly in one place.

**The fix: persist source RID in `graph_json["nodes"]` entries.** Task 53's `nodes[]` shape is extended to include `rid: str`:
```json
{"name": "Overview", "entity_type": "SECTION", "entity_id": "SECTION::heading='Overview'|page_start=3|__doc__=<doc>", "rid": "#42:17", "artifact_ids": ["<uuid-a>", "<uuid-b>"]}
```
`_serialize_for_audit` (now receiving `identity_to_rid`) reads the RID from the map and writes it into each node entry. The audit snapshot becomes the single loadable surface for source-vertex lookup in the downstream task. `derive_structure_links` already reads `graph_json["nodes"]` at `:4373`; it now picks up `rid` + `artifact_ids` from the same records it was already consuming. No cross-task in-memory map needed.

**`artifact_ids` on each `nodes[]` entry (needed for fallback boundary — see below):** `_serialize_for_audit` derives the list from the entity's provenance by walking `record.provenance[*].element_uid` and mapping to artifact ids via the worker's element-to-artifact lookup (already materialized earlier in the pipeline for chunk/element resolution). Entities with no provenance (pre-Phase-8 legacy or corner cases) get `artifact_ids: []` — the fallback treats empty as "unknown scope" and skips, rather than silently fanning out to all artifacts.

Writer changes at `arcadedb_graph.py:1920`:
```sql
CREATE EDGE EXTRACTED_FROM FROM :source_rid_i TO :rid_i SET entity_id = :entity_id_i, document_id = :doc_id_i, pipeline_run_id = :run_id_i, created_at = sysdate()
```
No subquery; direct RID-to-RID edge. Same-name same-type entities each get their own correctly-attached edge. **New edge properties `document_id` and `pipeline_run_id` scope each edge to a specific run — required for stale-edge handling and Task 54/55 verification (see below).**

**Singular-edge API parity (also fixed in this task):** the singular `create_entity_chunk_edge_sync(entity_name, entity_type, chunk_rid)` at `arcadedb_graph.py:1963` still uses the name+type-LIMIT-1 resolution path via `resolve_root_entity_sync`. Leaving it unfixed means any caller (tests, scripts, retry paths) that still uses the singular helper reintroduces the same-name same-type attach bug the batch path just eliminated.

Grep first to determine if the singular helper has any callers:
- If dead: delete the method; commit separately as `refactor(graph): delete unused create_entity_chunk_edge_sync`.
- If live: convert its signature to `create_entity_chunk_edge_sync(source_rid, chunk_rid, entity_id, document_id, pipeline_run_id)` and update callers to pass the pre-resolved RID (same way the batch path now does). Do NOT leave a name+type-lookup variant around — any future caller would hit the bug again.

- [ ] Grep `create_entity_chunk_edge_sync` across `app/`, `tests/`, `scripts/`. Inventory callers.
- [ ] Either delete or convert per the grep result. Explicit in the commit which path was taken.

**Stale-edge handling for verification integrity (addresses Task 54/55 false-pass risk):**

The current writers are append-only — `batch_create_entity_chunk_edges_sync` and `create_entity_chunk_edge_sync` both issue bare `CREATE EDGE EXTRACTED_FROM` without any delete-first step. The narrower cleanup `delete_extraction_layer_graph_sync` at `arcadedb_graph.py:2021` does NOT list `EXTRACTED_FROM` among the deleted edges — cascades only reach edges bound to deleted document-scoped entity vertices. For **global-scope entities** (`RADAR_SYSTEM`, `PLATFORM`, `MISSILE_SYSTEM`, …) whose vertices are preserved across documents, their `EXTRACTED_FROM` edges to chunks of this document survive re-ingestion.

Consequence: re-running `derive_structure_links` during Phase 8 validation stacks new edges on top of old ones. Task 54's "non-zero EXTRACTED_FROM edges" gate is satisfiable by **stale edges from pre-plan runs**, even if the new provenance path emits none. False-pass risk.

Fix (two-part):

1. **Scope each edge to its pipeline run.** The new `document_id` + `pipeline_run_id` edge properties (added to the writer above) let verification queries filter to the fresh run only. Without them, the only discriminator is `created_at`, which is unreliable across test reruns.
2. **Purge-before-write in `derive_structure_links`.** Before populating new `EXTRACTED_FROM` edges for a document, delete all existing `EXTRACTED_FROM` edges bound to that document_id. Add a new cleanup primitive `delete_extracted_from_edges_by_document_sync(document_id)` to `arcadedb_graph.py` that issues `DELETE FROM EXTRACTED_FROM WHERE document_id = :doc_id`. Call it at the top of `derive_structure_links` (after audit blob is loaded, before new edge construction).
3. **Extend `delete_extraction_layer_graph_sync`** at `:2021` to include `EXTRACTED_FROM` edges filtered by `document_id` — brings the narrower cleanup path into parity with the new edge-scoping contract.

Task 54/55 verification queries must filter by `pipeline_run_id = :current_run` (or `document_id = :current_doc` for multi-run regressions) to assert edges from THIS run, not accumulated history.

**Cross-document global-entity edge semantics (explicit contract):** a global-scope entity like `RADAR_SYSTEM "Fan Song"` can appear in many documents over time. Each time a document is ingested, `derive_structure_links` runs and creates `EXTRACTED_FROM` edges from that entity vertex to the document's chunk vertices. Those edges carry `document_id = :this_doc`. Re-ingesting document D2 purges only D2-scoped edges (`document_id = :d2`); edges from the same `RADAR_SYSTEM` vertex to D1 and D3 chunks are untouched — correct, since we're not re-running D1 or D3.

Consequence: a global entity's EXTRACTED_FROM edge count grows monotonically as more documents reference it, with re-ingestion of any single document resetting only that document's share. This is **intended behavior** — it's the data model that lets dossier queries ask "every chunk that mentions this RADAR_SYSTEM across the corpus" by traversing EXTRACTED_FROM edges from the entity vertex.

What's NOT a concern:
- Edges never go stale for un-reingested documents: the document's chunks and entity mentions remain valid while the document exists.
- Size blowup: each edge carries a small constant payload (entity_id, source_rid, document_id, pipeline_run_id, created_at). For a global entity referenced by 100 documents with 50 chunks each = 5000 edges. Trivial for ArcadeDB.

What IS a concern, handled separately:
- Document deletion: when a document is deleted, its EXTRACTED_FROM edges must also be deleted. This is already handled by `delete_document_graph_sync` and (after this plan) `delete_extraction_layer_graph_sync`, which include EXTRACTED_FROM by `document_id` in their delete scope. NOT a new task — already in scope via the existing cleanup primitives extended in this plan.

- [ ] Add `document_id` + `pipeline_run_id` properties on `EntityChunkEdge` dataclass + writer SQL.
- [ ] Add `delete_extracted_from_edges_by_document_sync(document_id)` primitive.
- [ ] Call it at the top of `derive_structure_links` before new edge construction.
- [ ] Extend `delete_extraction_layer_graph_sync` to include EXTRACTED_FROM by document_id.
- [ ] Task 54/55 verification queries filter by `pipeline_run_id` / `document_id`.
- [ ] Failing test: run `derive_structure_links` twice on the same document; after the second run, query EXTRACTED_FROM edges filtered to the current document_id. Count must equal ONLY the fresh-run edges, not double.

`EntityChunkEdge` dataclass at `graph_store.py:106` adds `source_rid: str` AND `entity_id: str` fields. `derive_structure_links` builds each edge record with values looked up from the `nodes[]` / `mentions[]` entries (both of which now carry `entity_id` + `rid`).

Alternative considered and rejected: moving `EXTRACTED_FROM` creation into `derive_ontology_graph`. Works, but splits the structural-edge logic — `derive_structure_links` currently also creates NEXT_CHUNK / SAME_PAGE / SAME_SECTION links (at `pipeline.py:4129+`) that depend on text+image chunks materialized earlier in the pipeline. Keeping all structural-edge creation in one task is cleaner; the audit-snapshot carrier is the right boundary handoff.

- [ ] **Step A (audit-blob extension):** update `_serialize_for_audit` at `pipeline.py:233` to include `rid` on every `nodes[]` entry (looked up from `identity_to_rid` which IS in scope there). Tests assert RID present and matches.
- [ ] **Step B (writer):** update `batch_create_entity_chunk_edges_sync` at `arcadedb_graph.py:1920` to accept `source_rid` + `entity_id` on each edge and emit RID-to-RID SQL with entity_id property.
- [ ] **Step C (worker):** `derive_structure_links` reads `node["rid"]` + `node["entity_id"]` from the audit blob when building `EntityChunkEdge`. If `rid` missing (legacy audit blob pre-plan), WARN and skip that edge rather than fail the batch.
- [ ] **Step D (critical test):** two `SECTION` entities with `heading="Overview"`, `page_start=3` and `:47`. Both imported as separate vertices with distinct RIDs. Both appear in `nodes[]` with distinct `rid`s. Run `derive_structure_links`; verify via SQL that each section's EXTRACTED_FROM edges attach to its OWN vertex — group by source RID, confirm expected per-section chunk counts. Under the old `name+type+LIMIT 1` path both sections' edges would land on the same vertex; this test distinguishes the real fix from a cosmetic one.

**Artifact-local fallback — replace, don't patch; preserve artifact boundary via `nodes[].artifact_ids`:**

`derive_structure_links` has a legacy fallback path at `pipeline.py:4393–4420` that reads per-artifact `content_metadata.docling_graph_data` or `extracted_entities`. `grep` confirms there is **no current writer** for those keys anywhere in `app/` — the fallback reads legacy data from the prior extraction architecture; for documents ingested under the current code, the path is dead. Reducing to `(name, entity_type)` at `:4400–:4416` would still collide for same-name same-type entities if those keys did get populated, but that's a downstream symptom of using a stale data source.

**Critical behavior to preserve: the fallback is ARTIFACT-SCOPED.** The current code walks each `Artifact` separately and fans out each entity only to that artifact's chunk ids (`:4398–:4420`). A naive replacement that reads only document-level `graph_json["nodes"]` has no artifact binding and would either fan out every unmentioned entity to every artifact's chunks (behavior regression on multi-artifact documents) or fall back to fanning-out-to-all (loses the intended scope).

Fix: `_serialize_for_audit` writes `artifact_ids: list[str]` on each `nodes[]` entry (see the Phase 8 plumbing above). Fallback respects that scope.

**Loop structure — pinned:** outer loop over `nodes[]`, inner loop over that node's `artifact_ids`. NOT per-artifact outer loop with nodes filter inside. Rationale: `O(nodes × avg_artifact_ids_per_node)` is strictly less than `O(artifacts × nodes)` for documents with many artifacts and narrow per-entity artifact scope (the typical case — an entity usually appears in 1–2 artifacts, while a document may have 100+ artifacts). The structure also keeps the fail-closed check at the node level (natural place for the WARNING log) rather than scattered across the artifact loop.

```python
# Artifact-scoped fallback — use document-level nodes[] + artifact_ids,
# not legacy per-artifact metadata. Preserves the current artifact boundary.
# Outer: nodes (one pass); Inner: that node's artifact_ids.
for node in graph_extraction.graph_json.get("nodes", []):
    eid = node.get("entity_id")
    if not eid or eid in mentioned_entity_ids:
        continue
    source_rid = node.get("rid")
    artifact_ids = node.get("artifact_ids", [])
    if not artifact_ids:
        # Entity has no resolvable artifact — skip rather than fan out
        # to all artifacts. Increment drop counter (observability).
        fallback_skipped_no_artifact_ids += 1
        logger.warning(
            "derive_structure_links: unmentioned entity %s has empty "
            "artifact_ids; fallback skipped. entity_id=%s",
            node.get("name"), eid,
        )
        continue
    for aid in artifact_ids:
        for chunk_id in artifact_chunk_map.get(aid, []):
            edge_tuples.append((
                node["name"], node["entity_type"],
                eid, source_rid, chunk_id,
            ))
```

Performance: single `nodes[]` traversal, one `artifact_chunk_map` lookup per artifact per node. If a document has `N` nodes averaging `K` artifacts each and `M` chunks per artifact, total edge-tuple appends are `N × K × M` — unavoidable since every chunk in scope needs an edge. The inversion to "outer nodes, inner artifacts" adds no asymptotic cost vs the alternative and eliminates a redundant per-artifact filter over the nodes list.

Same disambiguation (`entity_id`), same source resolution (`source_rid`), and — crucially — the artifact boundary is preserved via the new `artifact_ids` list on each node. One consumer surface; no behavior regression on multi-artifact documents.

Empty-`artifact_ids` case is fail-closed (skip + WARNING) rather than fan-out-to-all. For legacy audit blobs that pre-date this plan (`artifact_ids` absent): the skip means those docs lose fallback entirely on re-run, but their mention-primary path still works; the re-run through Phase 7's E2E regenerates the audit blob with proper `artifact_ids`, so this is a transient migration-state behavior only.

- [ ] **Step E (fallback replacement):** rewrite `pipeline.py:4393–4420` to consume `graph_json["nodes"]` with `artifact_ids` filtering. Delete the `content_metadata.docling_graph_data` / `extracted_entities` reads.
- [ ] **Step F (fallback tests):**
  - **F1 (artifact boundary preserved):** document with two artifacts A and B. Entity E1 has provenance pointing at A (so `artifact_ids=[A]`) but zero mentions. Fallback links E1 to A's chunks ONLY — not B's.
  - **F2 (multi-artifact entity):** entity E2 provenance spans A and B (`artifact_ids=[A, B]`), zero mentions. Fallback links E2 to both.
  - **F3 (empty artifact_ids = skip):** entity E3 with `artifact_ids=[]`, zero mentions. Fallback skips; WARNING logged. No edges created for E3.
  - **F4 (mixed mentioned + unmentioned):** one entity mentioned in one artifact; a second entity with no mentions but `artifact_ids=[A, B]`. First gets primary-path links; second gets fallback links scoped to A and B. Each lands on its own `source_rid`, verifiable via SQL.

- [ ] Failing test 1 (homonym different types): two entities `display_label="Fan Song"`, types `PLATFORM` and `RADAR_SYSTEM`. Only `PLATFORM` has a mention. Assert `RADAR_SYSTEM` still receives artifact-wide fallback chunk-links. (Resolved by distinct `entity_id`s even without changing suppression key.)
- [ ] Failing test 2 (same-name same-type distinct identity): two `SECTION`s `heading="Overview"`, `page_start=3` and `:47`. Only the first has a mention. Assert the SECOND still receives fallback chunk-links. (Resolved ONLY by `entity_id` — (name, type) alone still collides.)
- [ ] Change `mentioned_entities` → `mentioned_entity_ids`; update `.add(...)`, filter predicate, and edge-record shape (include `entity_id` on the chunk-link row).
- [ ] If schema migration needed: add alembic migration adding `entity_id: str` column; backfill null for old rows.
- [ ] Run failing tests + full `derive_structure_links` suite — PASS.
- [ ] Commit: `fix(structure_links): key chunk-link suppression by entity_id — full LogicalIdentity disambiguation`

### Task 54: Verify `derive_structure_links` produces non-zero entity-chunk edges

**Files:** observational.

**Critical sequencing:** `derive_structure_links` reads `DocumentGraphExtraction.graph_json` at `pipeline.py:4347`. That blob is only refreshed by `_upsert_document_graph_extraction(...)` inside `derive_ontology_graph`. Running `derive_structure_links` alone against a pre-Phase-8 audit row will see no `rid` / `artifact_ids` on the `nodes[]` entries and will hit the fail-closed skip paths by design — zero edges would be expected, which would falsely suggest a bug. The audit blob must be regenerated first.

- [ ] **Step 1:** Re-run `derive_ontology_graph` on `b1b0d596` — this regenerates the audit blob with the Phase 8 shape (`rid`, `artifact_ids`, `entity_id` on nodes/mentions).
- [ ] **Step 2:** Query `document_graph_extraction.graph_json` for `b1b0d596`; assert `nodes[0]` has non-null `rid` and non-empty `artifact_ids`. Proves the audit blob is in the new shape before testing the downstream consumer.
- [ ] **Step 3:** Re-run `derive_structure_links` on `b1b0d596`.
- [ ] **Step 4:** Query ArcadeDB for `EXTRACTED_FROM` edges. Expect non-zero; expect each edge to carry non-null `entity_id` property and direct source RID (not a name-based resolution artifact).
- [ ] **Step 5:** If zero: bug in provenance flow or audit-blob regeneration. Debug before considering Phase 8 done.
- [ ] No commit — observational.

### Task 55: Phase 8 regression on multiple docs

- [ ] Repeat Task 54 **including Step 1's `derive_ontology_graph` regeneration** across the same 3 docs from Task 48.
- [ ] Per-doc mentions count + entity-chunk-edge count saved to `/tmp/phase8-regression.md`.
- [ ] No commit — observational.

### Task 55a: Update `DocumentGraphExtraction.graph_json` model-level contract comment

**Files:**
- Modify: `app/models/ingest.py:343` (SQLAlchemy `comment=` on `graph_json` column)

The current comment reads: *"Audit blob (entity/edge counts, rejection reasons, pass summaries). NOT a serialized graph — ..."* This was accurate before Phase 8. After this plan, the blob carries `nodes[]` with per-entity `rid` / `artifact_ids` / `entity_id` plus `mentions[]` with element-level provenance. A reader seeing only counts/rejection-reasons language will not realize they can resolve entity chunk-linking downstream from this column.

Rewrite to reflect the new shape:
```python
comment=(
    "Audit blob carrying pass summaries, entity counts, rejection reasons, "
    "AND post-merge provenance artifacts. Shape includes: nodes[] "
    "(per-entity entity_id + source RID + artifact_ids), mentions[] "
    "(per-instance element_uid + chunk linkage), counts/metrics, yield_status. "
    "Written by _serialize_for_audit; consumed by derive_structure_links "
    "for chunk-link edge creation. Read spec §5.7 + Phase 8 plan "
    "(docs/superpowers/plans/2026-04-14-docling-graph-schema-compliance.md) "
    "for the authoritative shape."
),
```

- [ ] Update the comment in the model definition.
- [ ] Regenerate or alembic-annotate (if applicable) — this is a metadata-only column change.
- [ ] Commit: `docs(models): refresh graph_json comment for Phase 8 shape`

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
| `graph_json` mentions-write regresses `derive_structure_links` | Phase 8 Task 54 regression on `b1b0d596` asserts non-zero chunk edges after the provenance pipeline lands. Between Phase 1 and Phase 8 the bug remains; it's a known degraded state, not a regression introduced by this plan. |

---

## Rollback plan

Phase 8 introduces durable schema changes to three persisted stores. A clean revert to pre-Phase-8 code must leave production operable. Each change is designed to be additive AND nullable-tolerant so revert is clean.

### 1. `DocumentGraphExtraction.graph_json` audit blob (Postgres JSONB)

**Additions:** `nodes[]`, `mentions[]`, `element_to_artifact` top-level keys.

**Revert tolerance:** pre-Phase-8 `_serialize_for_audit` writes only counts/rejections and reads nothing from the Phase 8 keys. If Phase 8 code writes the richer blob and a revert happens, pre-Phase-8 reads ignore the unknown keys (Python dict access tolerates absence). No data corruption; the extra keys become dead payload until re-ingestion overwrites.

**Verified consumers (no Pydantic strict-schema layer):**
- `app/models/ingest.py:343` — `graph_json: Mapped[Optional[dict]]`, raw JSONB with no Pydantic validation layer at the ORM.
- `app/api/v1/sources.py:335–342` — reads `graph_json` via `getattr(..., "graph_json", None) or {}` and then `.get("entity_count_by_type", {}).values()` / `.get("edges_accepted", 0)`. Raw dict access with defaults — tolerant of missing keys.
- `app/workers/pipeline.py` (derive_structure_links readers) — `.get("nodes", [])`, `.get("mentions", [])`, `.get("element_to_artifact", {})` — all default to empty. Tolerant.
- No frontend/analytics consumer traverses the blob with strict schema validation (checked via grep — no Pydantic model class declares `graph_json: X` where X is a typed model).

Rollback tolerance claim is verified for all known consumers. If a future consumer adds a strict-schema layer over `graph_json`, the rollback hazard returns — document in that consumer's ADR that it couples to Phase 8 shape.

**Consumer impact on revert:** `derive_structure_links` reverts to reading legacy `content_metadata.docling_graph_data` / `extracted_entities` (the paths this plan retires). Those paths are currently dead (no writer) for post-Phase-2 ingestions but harmless for the revert window — fallback still runs at artifact scope because the code is unchanged, and any new ingestions under reverted code will repopulate the legacy metadata.

### 2. `EntityChunkEdge` ArcadeDB edge class

**Additions:** `entity_id`, `source_rid`, `document_id`, `pipeline_run_id` edge properties.

**Revert tolerance:** ArcadeDB edge properties are schema-flexible. Pre-Phase-8 code writes only `created_at`; the new properties become unread. Existing edges with the new properties survive revert — they just carry extra fields the reverted code ignores.

**Consumer impact on revert:** `batch_create_entity_chunk_edges_sync` reverts to `FROM (SELECT FROM {type} WHERE name = ... LIMIT 1)` subquery — reintroduces the same-name same-type attach bug. Revert is operationally safe (no crash) but reopens the correctness gap. Accept this as the cost of rollback; a second Phase 8 attempt would require the source-RID fix.

### 3. `LogicalIdentity.serialize_as_entity_id` + `EntityChunkEdge.entity_id` column

**Additions:** `entity_id: str` carried on every new edge.

**Revert tolerance:** the string is opaque to pre-Phase-8 code. Persisted edges survive revert with their `entity_id` string unread. Version prefix (`v1::`) on every string means a future `v2::` re-implementation coexists — no data surgery.

### 4. New worker-side drop counters

**Additions:** `fallback_skipped_no_artifact_ids`, `malformed_provenance_drops`, `post_merge_absent_provenance_drops` on pipeline-run metrics.

**Revert tolerance:** metrics columns are optional. Reverted code skips writing them; dashboards querying them get null. No crash; graphs go flat.

### Rollback procedure

1. Revert code to the commit before Phase 8 landed.
2. Deploy. Existing audit blobs remain readable; existing EXTRACTED_FROM edges remain readable; global entities' cross-document edges unaffected.
3. New ingestions under reverted code write pre-Phase-8 audit blobs (without `nodes`/`mentions`/`element_to_artifact`). Fallback runs artifact-wide, same as pre-plan baseline.
4. Known limitation during revert window: same-name same-type attach bug returns; primary mention path produces zero edges; chunk-linking falls back entirely to artifact scope. Users lose per-element mention granularity but retain artifact-level coverage.

**No manual data migration needed on revert.** If the project later re-attempts Phase 8, the audit blobs from the first Phase 8 run are still valid (additive shape); `derive_structure_links` reruns would regenerate consistent state.

---

## Acceptance criteria

1. Every consumer reads ontology data from Pydantic introspection. `ontology.yaml` does not exist.
2. Full unit + integration suite passes.
3. Phase 5 gate (Task 44): **dedup-aware** metric review. With merge-preserving dedup (Task 9d / Step 2b), stricter identity requirements, and edge-dedup reduction, some raw counts may legitimately drop (duplicates collapsing) while correctness improves. The gate requires: (a) no *unexplained* regressions — every count delta traced to a named cause (dedup, stricter identity, etc.); (b) non-empty expected entity and edge output for the test doc; (c) catalog identity fidelity improved (non-`ids=[none]` coverage strictly increased, exclusions-list respected); (d) prompt-block field-description coverage strictly increased.
4. Four docs through `derive_ontology_graph` (Task 48) produce `entities > 0, edges > 0` with realistic type distributions per doc.
5. All **nine** docs-compliance contract tests pass without `xfail`, listed by concrete test function name:
   - `test_is_entity_true_classes_declare_graph_id_fields_key`
   - `test_edge_label_on_entity_to_entity_fields`
   - `test_identity_fields_have_examples`
   - `test_canonical_entities_declare_ontology_name`
   - `test_extraction_views_subset_of_canonical_with_validator_parity`
   - `test_every_class_declares_is_entity_explicitly`
   - `test_descriptions_and_examples_on_extraction_relevant_fields`
   - `test_pass_root_list_dedup_schema_local`
   - `test_edge_label_targets_are_is_entity_true`

   Carve-outs for `SystemLinkRelationship` and pass-root list fields are coded into each test's fixture/parametrization — tests explicitly skip those classes rather than relying on implicit type inference.
6. **Residual brittleness items resolved:** watcher consumes `IngestDispatchResult` (Phase 1). Provenance plumbing complete: `ExtractPassResponse.provenance` populated, `_serialize_for_audit` writes `mentions`, `derive_structure_links` produces non-zero entity-chunk edges (Phase 8).
7. `validators.py` bool coercion returns `None`, not `0` / `1`.
8. **Intra-pass DTO relationship classes deleted** (`RadarRelationship`, `MissileRelationship`, `OtherSystemsRelationship`). `SystemLinkRelationship` retained as the documented multi-pass-architecture exception per Decision 4.
9. `ASSERTION.identity_fields` remains `[assertion_text]` and `PROPULSION_STACK.identity_fields` remains `[]` (both docstrings flag these as Plan 2 scope; they're NOT claimed as compliant).
10. **Compliance claim restated honestly in MEMORY:** docs-compliant within each extraction pass; one named non-compliant exception (`system_links`); two acknowledged identity anti-patterns deferred. No "full docs compliance" assertions anywhere in the merged code or docs.
