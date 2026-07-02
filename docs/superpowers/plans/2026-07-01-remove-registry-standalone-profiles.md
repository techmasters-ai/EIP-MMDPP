# Remove Query-Profile Registry → Standalone Ontology-Driven Profiles — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the query-profile *registry* layer; make query profiles first-class rows built directly against MMDP's single live extraction ontology, each with a Project-Source scope.

**Architecture:** Replace the one `governance.query_profile_registries` table (which bundled a frozen ontology snapshot + embedded JSONB profiles + an `is_active` gate) with a flat `governance.query_profiles` table. The ontology is served live from the `air_defense_v3` Pydantic SSoT via `build_ontology_dict()` (no stored copy). Profile executors (`section_properties`/`dossier`) gain a `source_id` filter (Global = unfiltered). The dead `ontology.*` Postgres tables + their seeder are removed.

**Tech Stack:** FastAPI, SQLAlchemy + Alembic, PostgreSQL (JSONB), ArcadeDB (graph), React + Cytoscape (UI), pytest.

**User decisions (already made):**
- "I want to keep query profiles" — profiles stay, re-homed to their own table.
- "remove the registry piece" — no registry, no activation/exposure indirection, no stored ontology copy.
- "the ontology the UI shows should read straight from the extraction ontology" — single SSoT (`air_defense_v3`), read-only in UI.
- "select the Project Source (or leave it global)" — per-profile `source_id` scope (null = Global); filters resolved root + evidence to that source's documents.
- Decision 1: migrate the 4 seeded profiles into the new table.
- Decision 2: keep a per-profile `enabled` toggle.
- Decision 3: keep the `RADAR_SYSTEM` + `MISSILE_SYSTEM` root-type limit as-is (do NOT expand root types).
- Decision 4: clean up the dead `ontology.*` tables + `seed_ontology.py` + docker-compose seed step + the `full_purge_and_reingest.py` truncate list.
- Decision 5: Project-Source scope = filter the profile's resolved root + evidence to entities/chunks whose documents belong to the selected source; Global = unfiltered.

---

## Reference — current code (READ before editing)

- `app/models/query_profiles.py:11` — `QueryProfileRegistry` (columns: id, name, description, source_id FK `ingest.sources`, ontology_name, ontology_version, ontology_definition JSONB, **profiles JSONB list**, is_active, created_by, timestamps).
- `app/schemas/query_profiles.py:90` — `QueryProfileDefinition`; `:109-129` `validate_shape`; `:85` `_CANONICAL_ROOT_ENTITY_TYPES` (RADAR_SYSTEM, MISSILE_SYSTEM).
- `app/services/query_profiles.py`: `get_active_registry :396`, `build_default_registry_template :259`, `_ontology_subset :245`, `_ensure_profile_editable :46`, `_CURRENT_ONTOLOGY_NAME :136`, `resolve_root_entity :544`, `_fetch_section_items :608`, `_project_field_groups :64`, `_CANONICAL_BY_ENTITY_TYPE :42`, `execute_section_search :786`, `execute_dossier_search :836`, `attach_evidence :748`, `_fetch_chunk_evidence :697/731`, `active_registry_payload :383`.
- `app/api/v1/query_profiles.py` — 12 routes (registries list/get/create/update/activate/default-template at 98/115/129/168/204/344; profiles append/update/delete at 224/255/297; active `GET /query-profiles` at 336; `search/section` 352; `search/dossier` 371).
- `alembic/versions/0011_query_profile_registries.py` (table), `0018_starter_profiles_to_section_properties.py` (`NEW_PROFILES` = the 4 seeded profiles), `0001_initial_schemas.py:442-513` (`ontology.*` tables).
- `ontology_bundles/air_defense_v3/introspect.py:26` — `build_ontology_dict()` (`ONTOLOGY_VERSION="3.0.0"`); emits `entity_types`, `relationship_types`, etc.
- `scripts/seed_ontology.py` (writes `ontology.*`), `docker-compose.yml:338` (runs the seeder on api start), `scripts/full_purge_and_reingest.py` (truncates `ontology.*`).
- UI: `frontend/src/components/QueryProfileRegistryPage.tsx` (registry+profile authoring), `GraphExplorer.tsx:209` (profiles tab), `QueryPage.tsx:791/862/874/886` (`getActiveQueryProfiles` + search calls), `frontend/src/api/client.ts:663-794` (registry/profile API functions).

**The 4 profiles to preserve** (all `kind=section_properties`/`dossier`, root `["RADAR_SYSTEM","MISSILE_SYSTEM"]`, `exposed=True`): `system_rf_parameters` (rf_parameters), `system_components` (components, include_associated_systems), `system_performance` (performance), `system_dossier` (dossier → the other three).

---

## Task 0: `QueryProfile` model + table + data-migrate the 4 profiles

**Goal:** A flat `governance.query_profiles` table holding first-class profiles, seeded from the 4 profiles embedded in the existing registry row (Global scope, enabled).

**Files:**
- Modify: `app/models/query_profiles.py` (add `QueryProfile`; keep `QueryProfileRegistry` for now — dropped in Task 5)
- Create: `alembic/versions/0019_query_profiles_table.py`
- Test: `tests/unit/test_query_profile_model.py` (create)

**Acceptance Criteria:**
- [ ] `governance.query_profiles` exists with columns: `id` (uuid pk), `name` (unique), `description`, `kind` (str), `root_entity_types` (JSONB), `definition` (JSONB — the profile body: profile_sections / subgroups / referenced_profiles / include_associated_systems), `source_id` (uuid FK `ingest.sources` ON DELETE SET NULL, nullable), `enabled` (bool default true), `created_by`, `created_at`, `updated_at`.
- [ ] The migration copies the 4 profiles out of the single `query_profile_registries` row's `profiles` JSONB into `query_profiles` (source_id NULL, enabled TRUE), preserving `name`/`kind`/`root_entity_types`/section-body.
- [ ] Downgrade drops the table (data loss acceptable — it's derived).

**Verify:** `alembic upgrade head` then `docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tAc "SELECT name,kind,source_id,enabled FROM governance.query_profiles ORDER BY name"` → 4 rows (`system_components`, `system_dossier`, `system_performance`, `system_rf_parameters`), all `source_id` NULL / `enabled` t.

**Steps:**
- [ ] **Step 1:** Add `QueryProfile` to `app/models/query_profiles.py` mirroring the existing model's SQLAlchemy style (same `Base`, `TIMESTAMP`, `UUID`, `JSONB` imports). Columns per AC. `source_id = Column(UUID, ForeignKey("ingest.sources.id", ondelete="SET NULL"), nullable=True)`; index on `source_id` and `enabled`.
- [ ] **Step 2:** Write `alembic/versions/0019_query_profiles_table.py` (`down_revision` = the current head — find via `alembic heads`). `upgrade()`: `op.create_table("query_profiles", schema="governance", ...)` with the columns; then a data step: `conn = op.get_bind(); rows = conn.execute(sa.text("SELECT profiles FROM governance.query_profile_registries LIMIT 1")).fetchone()` → for each profile dict in `rows[0]`, `conn.execute(sa.text("INSERT INTO governance.query_profiles (id,name,description,kind,root_entity_types,definition,source_id,enabled,created_at,updated_at) VALUES (gen_random_uuid(), :name, :desc, :kind, :roots, :defn, NULL, TRUE, now(), now())"), {...})` — map the embedded profile fields (name, description, kind, root_entity_types → `:roots` as json, and the remaining body → `:defn` as json). Guard for zero registry rows (skip seeding). `downgrade()`: `op.drop_table("query_profiles", schema="governance")`.
- [ ] **Step 3:** Test `tests/unit/test_query_profile_model.py`: assert the model maps to `governance.query_profiles`, `source_id` nullable, `enabled` default True. (Model-level, no live DB — mirror an existing model test.)
- [ ] **Step 4:** Run `alembic upgrade head`; run the Verify query → 4 rows. `python3 -m pytest tests/unit/test_query_profile_model.py -v`.
- [ ] **Step 5:** Commit `feat(profiles): query_profiles table + data-migrate 4 profiles off the registry`.

---

## Task 1: Live ontology service + `GET /v1/ontology`

**Goal:** Serve the ontology straight from `air_defense_v3` (`build_ontology_dict()`) so the UI builds profiles from the live SSoT — no stored copy.

**Files:**
- Create: `app/services/ontology_service.py`
- Modify: `app/schemas/query_profiles.py` (add `OntologyResponse`), `app/api/v1/query_profiles.py` (add route)
- Test: `tests/unit/test_ontology_service.py` (create)

**Acceptance Criteria:**
- [ ] `get_live_ontology()` returns `{version, entity_types:[{name,label}], relationship_types:[{name}], profile_sections:[...]}` derived from `build_ontology_dict()` and the `profile_sections` tags on the canonical Pydantic classes (RADAR_SYSTEM/MISSILE_SYSTEM).
- [ ] `GET /v1/ontology` returns that payload (200).
- [ ] `profile_sections` is derived from the Pydantic `json_schema_extra["profile_sections"]` tags (NOT hardcoded) — so the current set surfaces `rf_parameters`, `components`, `performance`.

**Verify:** `curl -s localhost:8005/v1/ontology | python3 -m json.tool` → shows entity_types incl RADAR_SYSTEM/MISSILE_SYSTEM, relationship_types, and profile_sections `["rf_parameters","components","performance"]`.

**Steps:**
- [ ] **Step 1:** `app/services/ontology_service.py`: `get_live_ontology()` calls `build_ontology_dict()` (from `ontology_bundles.air_defense_v3.introspect`), extracts entity/relationship types; derive `profile_sections` by walking `_CANONICAL_BY_ENTITY_TYPE` classes' `model_fields` and collecting every `json_schema_extra["profile_sections"]` value (dedup, sorted). Reuse the same canonical-class map the service uses (`app/services/query_profiles.py:_CANONICAL_BY_ENTITY_TYPE`) — import it, don't duplicate.
- [ ] **Step 2:** Add `OntologyResponse` schema (entity_types, relationship_types, profile_sections, version).
- [ ] **Step 3:** Add `GET /v1/ontology` (in `query_profiles.py` router, or a small `ontology.py` router — match where the app registers routers) returning `get_live_ontology()`.
- [ ] **Step 4:** Test `tests/unit/test_ontology_service.py`: `get_live_ontology()` returns non-empty entity_types containing RADAR_SYSTEM & MISSILE_SYSTEM and profile_sections containing rf_parameters/components/performance.
- [ ] **Step 5:** Run pytest + curl verify. Commit `feat(ontology): live GET /v1/ontology from air_defense_v3 SSoT`.

---

## Task 2: Service — profiles from the table + Project-Source filtering

**Goal:** Read/CRUD profiles from `governance.query_profiles`; thread `source_id` into resolve + evidence so a scoped profile only sees its source's documents. Delete registry/template/active-gate logic.

**Files:**
- Modify: `app/services/query_profiles.py`
- Test: `tests/unit/test_query_profile_source_scope.py` (create)

**Acceptance Criteria:**
- [ ] New `list_profiles(db, enabled_only=False)`, `get_profile(db, id)`, `create_profile`, `update_profile`, `delete_profile` operate on `QueryProfile` rows (dossier-referenced-profile delete-guard preserved).
- [ ] Removed: `get_active_registry`, `active_registry_payload`, `build_default_registry_template`, `_ontology_subset`, `_ensure_profile_editable`, `_CURRENT_ONTOLOGY_NAME` (and any other registry-only helpers).
- [ ] `resolve_root_entity(...)` accepts an optional `source_id`; when set, the resolved root must belong to a document of that source (filter the alias/fulltext/direct resolution by the entity's documents' `source_id`).
- [ ] `attach_evidence`/`_fetch_chunk_evidence` accept `source_id`; when set, only chunks whose `ingest.documents.source_id` matches are hydrated.
- [ ] With `source_id=None` behavior is byte-identical to today (Global = unfiltered).

**Verify:** `python3 -m pytest tests/unit/test_query_profile_source_scope.py -v` — asserts the resolve/evidence SQL gains a `source_id` predicate only when scoped, and none when Global.

**Steps:**
- [ ] **Step 1:** Read `resolve_root_entity :544` and `_fetch_chunk_evidence :697/731`. The evidence hydration already joins `ingest.documents` (`:697`) — add `AND documents.source_id = :source_id` when `source_id` is provided. For resolve, after finding candidate entities, filter to those with an `EXTRACTED_FROM` chunk whose document's `source_id` matches (or join through the doc filter). Keep the None path unchanged.
- [ ] **Step 2:** Replace `get_active_registry`/template helpers with table-backed CRUD (`list_profiles`/`get_profile`/`create_profile`/`update_profile`/`delete_profile`). `execute_section_search`/`execute_dossier_search` now take a loaded `QueryProfile` (or its id) + read its `source_id`, and pass it through resolve+evidence.
- [ ] **Step 3:** Test `tests/unit/test_query_profile_source_scope.py`: build the resolve/evidence SQL with and without `source_id`; assert the predicate is present iff scoped (use the SQL-string builders or mock the client and inspect the emitted SQL/params, mirroring existing service tests).
- [ ] **Step 4:** Run the touched service tests (`python3 -m pytest tests/ -k "query_profile or section or dossier" -v`) + the new test; report the delta vs pre-existing failures.
- [ ] **Step 5:** Commit `feat(profiles): table-backed CRUD + Project-Source filtering; drop registry service logic`.

---

## Task 3: API — delete registry routes; standalone profile CRUD + search

**Goal:** The API exposes profiles directly (no registry parent) + the two search endpoints, wired to the table + source scope.

**Files:**
- Modify: `app/api/v1/query_profiles.py`, `app/schemas/query_profiles.py`
- Test: `tests/integration/test_query_profiles_api.py` (create or update)

**Acceptance Criteria:**
- [ ] Deleted routes: `GET/POST /query-profiles/registries`, `GET/PUT /registries/{id}`, `POST /registries/{id}/activate`, `GET /query-profiles/default-template`, and the registry-nested profile routes.
- [ ] New/kept routes operate on profiles directly: `GET /v1/query-profiles` (list all, `?enabled_only=`), `POST /v1/query-profiles` (create), `GET/PUT/DELETE /v1/query-profiles/{id}`, `POST /v1/query-profiles/search/section`, `POST /v1/query-profiles/search/dossier`. Each create/update accepts `source_id` (nullable) + `enabled`.
- [ ] `search/section` + `search/dossier` load the profile by id from the table and honor its `source_id`.
- [ ] Schemas drop `QueryProfileRegistryCreate/Update/Response`; add `QueryProfileCreate/Update/Response` (incl. `source_id`, `enabled`).

**Verify:** `curl -s localhost:8005/v1/query-profiles | python3 -m json.tool` → 4 profiles (flat list, no registry wrapper); `curl -sf -X POST localhost:8005/v1/query-profiles/registries` → 404 (route gone).

**Steps:**
- [ ] **Step 1:** In `query_profiles.py` delete the 6 registry routes + registry-nested profile routes. Add flat profile CRUD (list/create/get/update/delete) calling Task-2 service functions. Repoint `search/section`/`search/dossier` to load a `QueryProfile` by id.
- [ ] **Step 2:** In `app/schemas/query_profiles.py` remove registry schemas; add `QueryProfileCreate/Update/Response` (fields: name, description, kind, root_entity_types, definition body, source_id, enabled). Keep `validate_shape` + the RADAR/MISSILE root frozenset (Decision 3).
- [ ] **Step 3:** Test: list returns 4; create with `source_id` persists it; `POST /registries` → 404; section/dossier search runs for a profile.
- [ ] **Step 4:** Restart api (`docker restart eip-mmdpp-api-1`), run the curl verifies + `python3 -m pytest tests/integration/test_query_profiles_api.py -v`.
- [ ] **Step 5:** Commit `feat(api): standalone query-profile CRUD + search; remove registry routes`.

---

## Task 4: UI — Query Profiles page (ontology-driven builder + Project Source)

**Goal:** Replace registry authoring with a profile authoring page: read-only ontology from `/v1/ontology`, profile list + CRUD, a Project Source dropdown, and an enabled toggle. Consumers list profiles directly.

**Files:**
- Modify: `frontend/src/components/QueryProfileRegistryPage.tsx` (→ Query Profiles page), `frontend/src/components/QueryPage.tsx`, `frontend/src/api/client.ts`, `frontend/src/components/GraphExplorer.tsx` (tab label/wiring)
- Test: manual UI verification (frontend has no unit-test harness for these; verify in the running app)

**Acceptance Criteria:**
- [ ] The "Ontology and Query Profiles" tab shows the **read-only ontology** (entity types, relationship types, profile_sections) fetched from `GET /v1/ontology` — no ontology JSON editor, no registry create/activate controls.
- [ ] Profile list + create/edit/delete against `/v1/query-profiles`; the create/edit form has a **Project Source dropdown** (options from `ingest.sources` + a "Global" default) bound to `source_id`, and an **Enabled** toggle.
- [ ] `profile_sections` options in the builder come from `/v1/ontology` (NOT the hardcoded `["rf_parameters","components","performance"]` at `QueryProfileRegistryPage:864`).
- [ ] `QueryPage` builds query modes from the flat profile list (drop the active-registry gate in `getActiveQueryProfiles`).
- [ ] `client.ts` registry functions removed; profile CRUD + `getOntology()` + `listSources()` added.

**Verify:** In the running UI (`:3100`): the tab shows the ontology read-only + 4 profiles; creating a profile with a Project Source persists and appears; the Search Documents page shows the profile query modes. Backend confirm: `curl -s localhost:8005/v1/query-profiles` reflects UI edits.

**Steps:**
- [ ] **Step 1:** `client.ts`: remove registry functions (`createQueryProfileRegistry`, `activate...`, etc.); add `listQueryProfiles`, `createQueryProfile`, `updateQueryProfile`, `deleteQueryProfile`, `getOntology`, and a `listSources` (reuse the existing sources endpoint if present — check `client.ts` for a sources fetch).
- [ ] **Step 2:** Reshape `QueryProfileRegistryPage.tsx`: drop registry/ontology-edit UI; render ontology read-only from `getOntology()`; profile list + form with Project Source `<select>` (from `listSources()` + Global) + Enabled checkbox + section multiselect from ontology `profile_sections`.
- [ ] **Step 3:** `QueryPage.tsx`: replace `getActiveQueryProfiles()` usage with `listQueryProfiles({enabled_only:true})`; drop the null-active-registry empty state.
- [ ] **Step 4:** `GraphExplorer.tsx`: keep the tab; ensure it renders the reshaped page; remove registry-derived ontology option builders if now redundant (leave the DEFAULT fallbacks).
- [ ] **Step 5:** Verify in the running UI + curl. Commit `feat(ui): ontology-driven query-profile builder with Project Source; remove registry UI`.

---

## Task 5: Drop the registry table + dead `ontology.*` tables & seeder

**Goal:** Remove the now-unused registry table and the write-only `ontology.*` tables + their seeder.

**Files:**
- Create: `alembic/versions/0020_drop_registry_and_ontology_tables.py`
- Modify: `docker-compose.yml` (remove seed step ~:338), `scripts/full_purge_and_reingest.py` (remove `ontology.*` truncate), delete `scripts/seed_ontology.py`
- Test: grep-based verification (no runtime readers remain)

**Acceptance Criteria:**
- [ ] Migration drops `governance.query_profile_registries` and `ontology.versions` / `ontology.entity_types` / `ontology.relationship_types` (and the `ontology` schema if empty).
- [ ] `scripts/seed_ontology.py` deleted; `docker-compose.yml` no longer invokes it on api start; `full_purge_and_reingest.py` no longer truncates `ontology.*`.
- [ ] `grep -rn "query_profile_registries\|ontology.entity_types\|seed_ontology" app/ scripts/ docker-compose.yml` returns only this migration / nothing live.

**Verify:** `alembic upgrade head`; `docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -tAc "SELECT to_regclass('governance.query_profile_registries'), to_regclass('ontology.entity_types')"` → both NULL. `docker restart eip-mmdpp-api-1` starts clean (no seeder error in logs).

**Steps:**
- [ ] **Step 1:** Confirm (grep) nothing outside the removed code reads `query_profile_registries` or `ontology.*` (the analysis found none — re-verify post Tasks 2/3). Remove the `QueryProfileRegistry` model class.
- [ ] **Step 2:** Write `0020_drop_registry_and_ontology_tables.py`: `op.drop_table("query_profile_registries", schema="governance")`; drop the three `ontology.*` tables; `op.execute("DROP SCHEMA IF EXISTS ontology CASCADE")` if desired. `downgrade()` recreates minimally (or a no-op documented as irreversible).
- [ ] **Step 3:** Delete `scripts/seed_ontology.py`; remove the seed command from `docker-compose.yml:~338`; remove the `ontology.*` truncate lines from `scripts/full_purge_and_reingest.py`.
- [ ] **Step 4:** `alembic upgrade head`; restart api; run the Verify queries + grep. Commit `chore(db): drop registry + dead ontology.* tables and seeder`.

---

## Task 6: Cleanup + end-to-end verification gate

**Goal:** Remove residual dead code and prove the reshaped feature works end-to-end incl. Project-Source filtering.

**Files:**
- Modify: `app/services/ontology_templates.py` (remove dead `load_registry_ontology`), any leftover registry imports; contract tests
- Test: `tests/integration/test_query_profiles_e2e.py` (create)

**Acceptance Criteria:**
- [ ] Dead `load_registry_ontology` (`ontology_templates.py:162`) and any other now-orphaned registry helpers removed; `python3 -c "import app.services.query_profiles, app.api.v1.query_profiles"` imports clean.
- [ ] E2E: create a profile scoped to a specific source → `search/section` returns only entities/evidence from that source's documents; the same profile as Global returns the unfiltered superset.
- [ ] The 4 migrated profiles run via `search/section`/`search/dossier` without error.
- [ ] Full touched-module suite: no NEW failures vs the known pre-existing set.

**Verify:** `python3 -m pytest tests/integration/test_query_profiles_e2e.py -v` → pass; captured before/after showing scoped ⊆ global.

**Steps:**
- [ ] **Step 1:** Remove `load_registry_ontology` + any orphaned registry references; run the import smoke.
- [ ] **Step 2:** E2E test: pick a source with known entities (e.g. an SA-2 source); create a `system_rf_parameters`-style profile scoped to it; run section search; assert results' documents all belong to that source. Re-run Global; assert scoped result set ⊆ global result set.
- [ ] **Step 3:** Run the E2E + `python3 -m pytest tests/ -k "query_profile or ontology or section or dossier" -q`; report delta.
- [ ] **Step 4:** Commit `test(profiles): e2e source-scope + cleanup dead registry code`.

---

## Rollout notes
- Tasks 0-4 keep `query_profile_registries` intact (read only by the data migration); the drop is Task 5, after all code stops referencing it — so a mid-plan halt is safe.
- Deployment: api/worker bind-mount `app/`; `docker restart eip-mmdpp-api-1` after backend tasks. Frontend via Vite HMR. Alembic migrations run against the live Postgres — take a DB backup before Task 0 and Task 5 (schema/data changes).
- `air_defense_v3` remains the extraction SSoT; this plan only makes the query-profile subsystem *read* it live instead of a frozen copy — no extraction change.
