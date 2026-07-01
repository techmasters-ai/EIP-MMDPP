# Case-Insensitive Entity Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make entity identity case/whitespace-insensitive so extraction never creates case-variant duplicate vertices, and clean up the 2 existing case-collisions — no re-ingest.

**Architecture:** Add a shared `norm()` and derive a **normalized identity key** used for (a) in-memory `LogicalIdentity` equality/merge and (b) the DB upsert `WHERE` + a new unique index — while raw first-seen identity values stay for the display name. A one-time migration merges the 2 existing case-pairs, backfills `_key` on every existing vertex, then creates the unique index.

**Tech Stack:** Python 3.11, ArcadeDB (SQL/UPSERT), pytest, Celery workers.

**User decisions (already made):**
- "code fix only" — no re-ingest; existing dups handled by a tiny targeted merge + backfill.
- "Include case-normalization" — `norm(x)=" ".join(str(x).strip().casefold().split())`.
- Display name = **first-seen casing** (not a canonical-casing algorithm).
- Spec authority: `docs/superpowers/specs/2026-07-01-entity-dedup-extraction-design.md` (v3, vetted over 2 review rounds).

---

## Reference: current code anchors (read before editing)

- `app/services/extraction_merge.py:72` — `LogicalIdentity.as_upsert_identity_dict()` (adds `document_id` only when `scope=="document"`).
- `app/services/extraction_merge.py:482` — `build_display_label()` (prioritizes identity values for the display name).
- `app/services/extraction_merge.py:~631` — `LogicalIdentity` identity resolution (raw identity values feed equality/merge/provenance/edges/audit).
- `app/services/arcadedb_graph.py:168` — `_build_where(identity_fields)` — shared by node identity (`:915`) AND relationship endpoints (`from_identity` `:311`, `to_identity` `:317`).
- `app/services/arcadedb_graph.py:180` — `_build_upsert_node_script(records)` — the **batch** node upsert SQL builder (used by `upsert_nodes_batch_sync` `:2568`, the production write path via `pipeline.py:1541/7811`).
- `app/services/arcadedb_graph.py:2524` — `_upsert_node_impl_sync(record)` — single-node sync upsert.
- `app/services/arcadedb_schema.py:471` — the upsert-UNIQUE-index DDL phase (currently builds `(identity_fields[, document_id], entity_type)` UNIQUE).
- `app/workers/pipeline.py:1528` — `NodeRecord.name = build_display_label(entity_type, identity.identity_values_dict(), properties)`.
- Deployed graph facts: `RADAR_SYSTEM[system_name, entity_type]` global index; **2** case-collisions (`fan song`, `spoon rest`); 16 "Guideline" rows are **distinct** `system_name` variants (must stay separate).

---

## Task 0: `norm()` + LogicalIdentity in-memory normalization

**Goal:** A shared `norm()` and a normalized identity key that makes `FAN SONG`/`Fan Song` the *same* `LogicalIdentity` (so they merge in memory), while raw identity values still drive the display name.

**Files:**
- Modify: `app/services/extraction_merge.py`
- Test: `tests/services/test_identity_norm.py` (create)

**Acceptance Criteria:**
- [ ] `norm("  FAN   Song ") == "fan song"` (trim, casefold, collapse internal whitespace).
- [ ] Two `LogicalIdentity` with identity values differing only in case/whitespace are **equal** and **hash-equal**; distinct `system_name`s are not.
- [ ] `identity_values_dict()` / `identity_tuple` still return the **raw** values (display name unaffected — `build_display_label` still yields `Fan Song`, not `fan song`).

**Verify:** `python3 -m pytest tests/services/test_identity_norm.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write `tests/services/test_identity_norm.py`:**
```python
from app.services.extraction_merge import norm, LogicalIdentity

def test_norm():
    assert norm("  FAN   Song ") == "fan song"
    assert norm("Spoon Rest") == "spoon rest"
    assert norm(None) == ""  # defensive

def _li(system_name):
    # construct a minimal LogicalIdentity for RADAR_SYSTEM keyed on system_name
    return LogicalIdentity(entity_type="RADAR_SYSTEM",
                           identity_field_names=("system_name",),
                           identity_tuple=(system_name,), scope="global", document_id=None)

def test_case_variants_equal_and_hash_equal():
    a, b = _li("Fan Song"), _li("FAN SONG")
    assert a == b and hash(a) == hash(b)

def test_distinct_names_not_equal():
    assert _li("Fan Song") != _li("Low Blow")

def test_display_values_stay_raw():
    a = _li("Fan Song")
    assert a.identity_values_dict()["system_name"] == "Fan Song"  # NOT casefolded
```
(Adjust the `LogicalIdentity(...)` constructor call to the real dataclass signature — read `extraction_merge.py` for the exact fields; keep the intent: build one keyed on `system_name`.)

- [ ] **Step 2: Run** `python3 -m pytest tests/services/test_identity_norm.py -v` → FAIL (`norm` missing / equality case-sensitive).

- [ ] **Step 3: Add `norm()` to `extraction_merge.py`** (module level):
```python
def norm(value) -> str:
    """Case/whitespace-insensitive identity key: trim, casefold, collapse whitespace."""
    return " ".join(str(value).strip().casefold().split()) if value is not None else ""
```

- [ ] **Step 4: Add a normalized key to `LogicalIdentity`** and route equality/hash through it, leaving `identity_tuple`/`identity_values_dict()` raw. Add a property:
```python
    @property
    def norm_key(self) -> tuple:
        # normalized identity for equality/merge/upsert; scope-aware (keep document_id raw
        # for document-scoped types since it's a UUID, not a display string).
        base = tuple(norm(v) for v in self.identity_tuple)
        return (self.entity_type, base, self.document_id if self.scope == "document" else None)
```
Then implement `__eq__`/`__hash__` on `LogicalIdentity` in terms of `norm_key` (if the class is a frozen dataclass, override with `eq=False` on the decorator and add explicit `__eq__`/`__hash__`; match the existing class style). Do NOT change `identity_tuple`/`identity_values_dict()`.

- [ ] **Step 5: Run** `python3 -m pytest tests/services/test_identity_norm.py -v` → PASS.
- [ ] **Step 6: Commit** `git add app/services/extraction_merge.py tests/services/test_identity_norm.py && git commit -m "feat(extraction): case-insensitive LogicalIdentity (norm key; raw display preserved)"`

---

## Task 1: DB write paths derive + match the normalized `_key`

**Goal:** Every DB node-write path and every relationship endpoint resolves on the normalized `<field>_key`, so case-variants dedup at write time and edges connect to the merged vertex.

**Files:**
- Modify: `app/services/arcadedb_graph.py` (`_build_where`, `_build_upsert_node_script`, `_upsert_node_impl_sync`, async single-node upsert)
- Modify: `app/workers/pipeline.py` (where `identity_fields`/`from_identity`/`to_identity` dicts are built, if `_key` must be injected there — see Step 2)

**Acceptance Criteria:**
- [ ] A single helper derives `{f"{field}_key": norm(value)}` from an identity dict; used by all node-write paths and relationship endpoint clauses.
- [ ] Node upsert `WHERE` matches on `<field>_key` (+ `entity_type`), and the upsert `SET`s the `_key` field(s); display fields (`name`, `system_name`) are set **only on insert** (first-seen wins).
- [ ] Relationship endpoint `WHERE` (`from_identity`/`to_identity`) matches on `<field>_key`.
- [ ] `entity_type` and `document_id` (for document-scoped types) are unchanged.

**Verify:** `python3 -m pytest tests/services/test_arcadedb_upsert_key.py -v` (create a focused test that builds the SQL and asserts it references `<field>_key` in WHERE) → pass.

**Steps:**

- [ ] **Step 1: Add a key-derivation helper** in `arcadedb_graph.py` (module level), reusing `norm`:
```python
from app.services.extraction_merge import norm

_DISPLAY_IDENTITY_FIELDS = {"name", "system_name"}  # kept raw for display; keyed via _key

def _key_fields(identity_fields: dict) -> dict:
    """Map each identity field to its normalized <field>_key (skip document_id/UUIDs)."""
    return {f"{k}_key": norm(v) for k, v in identity_fields.items() if k != "document_id"}
```

- [ ] **Step 2: `_build_where` matches on `_key`.** In `_build_where` (`:168`), build the clause over the **`_key`** field(s) instead of the raw identity field(s): for each identity field `k` (except `document_id`), emit `{k}_key = :{k}_key`; keep `document_id = :document_id` as-is for document-scoped. Ensure the caller supplies the `_key` params — inject them via `_key_fields(identity_fields)` at each call site (node identity `:915`, relationship `from_identity` `:311` / `to_identity` `:317`). Since relationship endpoints come from `RelationshipRecord.from_identity/to_identity` (built in `pipeline.py:2573` from `LogicalIdentity`), derive `_key` in `_build_where` itself so no upstream change is needed.

- [ ] **Step 3: Batch upsert (`_build_upsert_node_script` `:180`).** For each record, add `_key_fields(record.identity_fields)` to the SET map and to the WHERE params; change the `UPSERT … WHERE` to the `_key` clause. Keep `name`/`system_name` display fields set only when inserting (use ArcadeDB's create-only semantics: exclude display fields from the `UPDATE SET`, or set them via `SET name = COALESCE(name, :name)` so an existing display value is preserved).

- [ ] **Step 4: Single sync (`_upsert_node_impl_sync` `:2524`) and async single-node upsert.** Apply the same `_key`-based WHERE + `_key` SET + display-first-seen. Centralize by having both call the shared helpers so the logic isn't duplicated.

- [ ] **Step 5: Focused test** `tests/services/test_arcadedb_upsert_key.py`: call `_build_upsert_node_script([record])` (or `_build_where`) with a record whose `system_name="Fan Song"`, assert the produced SQL/params contain `system_name_key` matching `norm("Fan Song")=="fan song"` in the WHERE, and that `name`/`system_name` are not force-overwritten on update. Run → pass.

- [ ] **Step 6: Commit** `git commit -am "feat(graph): dedup node upserts + relationship endpoints on normalized _key"`

---

## Task 2: Schema — `_key` property + unique index

**Goal:** The schema builder declares the `<field>_key` STRING property and a UNIQUE index on the normalized key, so ArcadeDB UPSERT can dedup on it.

**Files:**
- Modify: `app/services/arcadedb_schema.py` (the upsert-index phase, `:471`)

**Acceptance Criteria:**
- [ ] For each domain-entity type with `identity_fields`, DDL emits `CREATE PROPERTY <type>.<field>_key STRING IF NOT EXISTS` for each identity field **before** the index DDL.
- [ ] The UNIQUE index is `(…<field>_key…, entity_type)` (global) — matching the new upsert WHERE.
- [ ] Component/content classes (`is_entity=False`, empty `identity_fields`) are skipped (unchanged).

**Verify:** `python3 -m pytest tests/services/test_schema_key_ddl.py -v` (assert the generated DDL list contains the `_key` property + `_key` unique index for a domain type, and nothing for a component type) → pass.

**Steps:**

- [ ] **Step 1: Test** `tests/services/test_schema_key_ddl.py` — feed a minimal ontology dict (one domain type with `identity_fields=["system_name"]`, one component type with `identity_fields=[]`) to the DDL-building function; assert the output contains `CREATE PROPERTY RADAR_SYSTEM.system_name_key STRING` and `CREATE INDEX … ON RADAR_SYSTEM (system_name_key, entity_type) UNIQUE`, and no `_key` DDL for the component. Run → FAIL.

- [ ] **Step 2: Edit `arcadedb_schema.py:471` upsert-index phase.** For each entity type with `id_fields`: emit `CREATE PROPERTY {type}.{field}_key STRING IF NOT EXISTS` per field, then build the UNIQUE index over `[f"{field}_key" for field in id_fields] + (["document_id"] if scope=="document") + ["entity_type"]`. (Keep the existing raw-field indexes for now if other code still reads them; the new `_key` unique index is what UPSERT uses.)

- [ ] **Step 3: Run** `python3 -m pytest tests/services/test_schema_key_ddl.py -v` → PASS.
- [ ] **Step 4: Commit** `git commit -am "feat(schema): normalized _key property + unique index for entity upsert"`

---

## Task 3: One-time migration (merge 2 pairs → backfill → index) — MIGRATION GATE

**USER-ORDERED GATE — NON-SKIPPABLE.** This task performs a destructive merge and a graph-wide backfill on the live database; it was requested by the user. It MUST NOT be closed by walking around it or substituting a cheaper check. Close only after every acceptance criterion is re-validated independently, with captured output.

**Goal:** Bring the existing graph to the new scheme without re-ingest: merge the 2 case-collisions, backfill `_key` on all existing domain vertices, then create the unique index.

**Files:**
- Create: `scripts/migrate_entity_key_dedup.py`

**Acceptance Criteria:**
- [ ] The 2 case-pairs (`Fan Song`/`FAN SONG`, `Spoon Rest`/`SPOON REST`) are merged: the data-bearing vertex keeps all edges from both (relationships, `EXTRACTED_FROM`, aliases); the empty duplicate is deleted. **2 vertices removed, zero edges lost.**
- [ ] Every existing vertex of each indexed domain type has a non-null `<field>_key` (`… WHERE <field>_key IS NULL` → count 0).
- [ ] The `(…_key, entity_type)` UNIQUE index exists and creation succeeded.
- [ ] The 16 SA-2 "Guideline" variants (distinct `system_name`) are still **16** vertices (not merged).

**Verify:** run the script with `--dry-run` first (prints the plan), then live; then the checks in Task 4.

**Steps:**

- [ ] **Step 1: Write `scripts/migrate_entity_key_dedup.py`** with three ordered phases and a `--dry-run` flag:
  1. **Merge pairs:** for each of the 2 known pairs, find both vertices; pick the survivor (non-null specs); for the loser, re-point every in/out edge to the survivor (ArcadeDB: `MOVE`/recreate edges), then `DELETE VERTEX` the loser. Log edge counts before/after (must match).
  2. **Backfill:** for each indexed domain type + identity field, `UPDATE <type> SET <field>_key = <norm(field)>` for all rows (compute `norm` in Python per row, or an ArcadeDB expression if available; Python loop is safest). Idempotent.
  3. **Create index:** run the `_key` UNIQUE index DDL (now safe: collisions merged, keys populated).
  Order is mandatory: merge → backfill → index.
- [ ] **Step 2: Dry-run** `docker exec eip-mmdpp-api-1 python scripts/migrate_entity_key_dedup.py --dry-run` → prints the 2 merges + backfill counts, no writes. Capture output.
- [ ] **Step 3: CONFIRM with the user** before the live run (destructive). On go-ahead: `docker exec eip-mmdpp-api-1 python scripts/migrate_entity_key_dedup.py` → capture output (edge counts preserved; backfill count; index created).
- [ ] **Step 4: Commit** `git add scripts/migrate_entity_key_dedup.py && git commit -m "feat(migration): merge case-pairs + backfill _key + create unique index"`

```json:metadata
{"files": ["scripts/migrate_entity_key_dedup.py"], "verifyCommand": "docker exec eip-mmdpp-api-1 python scripts/migrate_entity_key_dedup.py --dry-run", "acceptanceCriteria": ["2 case-pairs merged, zero edges lost", "no null _key on any indexed domain vertex", "_key unique index created", "16 Guideline variants still 16 vertices"], "userGate": true, "tags": ["user-gate"], "requiresUserSpecification": false, "gateScope": "migration", "modelTier": "complex"}
```

---

## Task 4: Verification gate (dedup works, variants preserved) — VERIFICATION GATE

**USER-ORDERED GATE — NON-SKIPPABLE.** Requested by the user across two review rounds. Close only after every acceptance criterion is independently re-validated with captured output.

**Goal:** Prove case-dedup works end to end, existing data is clean, and distinct variants are preserved.

**Files:**
- Test: `tests/integration/test_entity_dedup.py`

**Acceptance Criteria:**
- [ ] Grouping by normalized key per domain type → every count = 1: `fan song` = 1 vertex, `spoon rest` = 1 vertex.
- [ ] The merged `Fan Song` carries `nominal_rf_mhz`/`tx_peak_power_kw` and `EXTRACTED_FROM` lineage from all its former source docs.
- [ ] The 16 SA-2 variants (distinct `system_name`, shared `name="Guideline"`) remain **16** separate vertices.
- [ ] No existing vertex of an indexed domain type has a null `<field>_key`.
- [ ] Re-extracting/upserting an entity named `FAN SONG` upserts onto the existing `fan song` vertex (no new vertex) and its relationship endpoints resolve to it.

**Verify:** `python3 -m pytest tests/integration/test_entity_dedup.py -v` + captured graph queries.

**Steps:**

- [ ] **Step 1:** Post-migration graph checks (capture output): group RADAR_SYSTEM by `system_name_key` → assert `fan song`/`spoon rest` each count 1; count MISSILE_SYSTEM `system_name='Guideline'` → still 16; `… WHERE system_name_key IS NULL` → 0 across domain types.
- [ ] **Step 2:** Idempotent-upsert test: call `upsert_nodes_batch_sync` (or the batch script) with a record `system_name="FAN SONG"` (RADAR_SYSTEM) → assert it returns the RID of the existing merged `Fan Song` vertex (no new vertex created), display name stays `Fan Song`.
- [ ] **Step 3:** Relationship-endpoint test: upsert an edge whose endpoint identity is `FAN SONG` → assert it attaches to the merged vertex.
- [ ] **Step 4:** Run the full RRF/community suites are unaffected; run `python3 -m pytest tests/integration/test_entity_dedup.py -v`. Capture into close notes.
- [ ] **Step 5: Commit** `git commit -am "test(extraction): entity case-dedup integration gate"`

```json:metadata
{"files": ["tests/integration/test_entity_dedup.py"], "verifyCommand": "python3 -m pytest tests/integration/test_entity_dedup.py -v", "acceptanceCriteria": ["fan song/spoon rest each 1 vertex by normalized key", "merged Fan Song keeps specs + all EXTRACTED_FROM", "16 Guideline variants stay 16", "no null _key", "upsert FAN SONG hits existing vertex + edges resolve"], "userGate": true, "tags": ["user-gate"], "requiresUserSpecification": false, "gateScope": "verification", "requireEvidenceTokens": [["before", "raw-name", "2 case-pairs"], ["after", "normalized-key", "1 per designator"]], "modelTier": "complex"}
```

---

## Rollout note

Tasks 0–2 are safe code changes (tested, no data touch). Task 3 is the one-time destructive migration (2-vertex merge) + graph-wide backfill + index creation — run once, gated on explicit confirmation. No re-ingest. If anything regresses, the `_key` unique index can be dropped and the raw-field indexes still exist (Task 2 keeps them), reverting to pre-change upsert behavior.
