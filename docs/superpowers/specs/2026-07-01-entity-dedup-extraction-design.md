# Case-Insensitive Entity Identity — Design

- **Date:** 2026-07-01
- **Status:** Approved (brainstorming, v3 after 2 review rounds) → ready for implementation plan
- **Scope:** entity identity normalization at the `LogicalIdentity` (merge) layer + the ArcadeDB
  upsert/index layer. **Code fix only** — no re-ingest.

## Problem (corrected after review)

Earlier framing ("entities are document-scoped → 16 'Guideline' duplicates") was a
**misdiagnosis**. Verified against the deployed graph (`air_defense_v3_narrowing_v1`):

- Entities are **already global-scoped**: the live indexes are
  `RADAR_SYSTEM[system_name, entity_type]` and `MISSILE_SYSTEM[system_name, entity_type]` —
  **no `document_id`**. No document-scope bug exists.
- The "16 Guideline" rows are **not duplicates**: all 16 have *different* `system_name`
  values (SA-2, V-750, V-750V, V-751, V-755, V-757, V-759, V-760, …) — 16 **distinct** SA-2
  missile variants that share the NATO reporting name "Guideline" in the `name` field.
  Grouping on `name` (a shared reporting name) falsely conflated them.
- The **only real duplication** is **case-sensitivity of the identity field** (`system_name`).
  Grouping by case-normalized `system_name`: **RADAR_SYSTEM = 2 true dups** (`fan song`,
  `spoon rest`), **MISSILE_SYSTEM = 0**.

So the genuine defect: two vertices for one entity when the identity differs only in
case/whitespace (`FAN SONG` vs `Fan Song`; `SPOON REST` vs `Spoon Rest`).

## Goal

Make entity identity **case/whitespace-insensitive** so extraction never creates
case-variant duplicates — at both the in-memory merge layer and the DB upsert layer. Clean up
the 2 existing case-pairs so the new unique index can be created. **No re-ingest.**

## Non-goals

- Any `identity_scope` / document-vs-global change (entities are correctly global; DOCUMENT is
  intentionally global via `document_number` — leave scoping alone).
- A full re-ingest or graph wipe.
- Deduping distinct variants that share a `name` reporting value (they are correctly separate).
- A canonical-casing display algorithm — display value is **first-seen**.
- Normalizing component/content identities (`is_entity=False`) — normalization applies to
  **domain entity identities only**.

---

## Design

`norm(x) = " ".join(str(x).strip().casefold().split())` (trim, casefold, collapse internal
whitespace).

### Component 1 — Normalize at the `LogicalIdentity` (merge) layer  *(required — review finding #3)*

`LogicalIdentity` currently uses **raw** identity values for equality, hashing, merge,
provenance aggregation, edge/relationship resolution, audit IDs, and RID maps
(`app/services/extraction_merge.py:631`, `app/workers/pipeline.py:1589`). If only the DB layer
normalizes, then `FAN SONG` and `Fan Song` stay **separate `MergedEntityRecord`s** in memory
and later collide on the DB upsert (last-write-wins → lost fields/provenance).

- Add a **separate normalized key** (`norm()` applied to each identity field value) used for
  equality / hashing / merge / edge-resolution / audit serialization. **Do NOT normalize
  `identity_tuple` / `identity_values_dict()` in place** — those still hold the **raw** values
  and feed `build_display_label()` → `NodeRecord.name` (`extraction_merge.py:482`,
  `pipeline.py:1528`). Normalizing them would casefold the display name (review finding #3).
- The **raw first-seen** identity values are preserved for display; a merge update from a later
  case variant must **not** overwrite the display casing (first-seen wins).
- Effect: within a run, `FAN SONG` and `Fan Song` produce the **same** normalized key →
  they **merge into one** `MergedEntityRecord` (aggregating fields + provenance), keeping the
  first-seen display casing, before write.

### Component 2 — Normalized `_key` property + unique index + all write paths  *(review findings #2, #4)*

For cross-run dedup (doc A ingested now, doc B later), every DB write path must set and match on
the normalized key.

- Schema build: **`CREATE PROPERTY <type>.<identity_field>_key STRING`** for each domain-entity
  identity field, **before** the unique-index DDL (`app/services/arcadedb_schema.py:471`).
- UNIQUE index on the **normalized** `_key` field(s) + `entity_type` (global scope preserved).
- **All node-write paths** must compute+set the `_key` field(s) and match `WHERE` on them —
  not just one function. Covers: the **batch** builder `_build_upsert_node_script`
  (`arcadedb_graph.py:180`, used by `upsert_nodes_batch_sync` — the production path via
  `pipeline.py:1541/7811`), the single sync `_upsert_node_impl_sync`, and the async single-node
  upsert. Centralize `_key` derivation so all paths share it.
- **Relationship endpoint resolution** must also match on the normalized key. `_build_where`
  (`arcadedb_graph.py:168`) builds both node identity clauses (line 915) **and** relationship
  endpoint clauses from `RelationshipRecord.from_identity`/`to_identity` (lines 311/317). If
  endpoints resolve on raw identity while nodes dedup on normalized, edges won't connect to the
  merged vertex. The `from_identity`/`to_identity` dicts (produced upstream, `pipeline.py:2573`)
  must carry the normalized `_key` field(s), and `_build_where` must use them.
- **Display fields (`name`, `system_name`) are preserved first-seen** — the upsert must not
  overwrite them on update (set-on-insert only for display fields).

### Component 3 — One-time migration on the existing graph *(review finding #1)*

With **no re-ingest**, existing vertices have a `NULL` `_key`, so a new `WHERE _key = …` upsert
would miss every old row and re-create duplicates. So the migration must backfill `_key` on
**all** existing vertices of each indexed domain type, in this order:

1. **Merge the 2 existing case-collisions** (`Fan Song`/`FAN SONG`, `Spoon Rest`/`SPOON REST`):
   keep the vertex with data, transfer the other's edges (relationships, `EXTRACTED_FROM`,
   aliases) onto it, then delete the empty duplicate. *(The only destructive step — 2 vertices,
   bounded; explicitly confirmed at execution time.)*
2. **Non-destructive `_key` backfill for ALL existing vertices** of each indexed domain type:
   `UPDATE <type> SET <field>_key = norm(<field>)` (an idempotent update over every row). This
   also fixes existing **relationship endpoints** if any endpoint identity is stored denormalized
   (verify edges resolve to the backfilled vertices).
3. **Create the UNIQUE index** on the `_key` field(s) — now creatable (collisions merged,
   `_key` populated).

Step 2 must precede step 3 (index) and step 1 (merge) must precede step 2 (so the merge doesn't
leave a colliding `_key`). Steps 2–3 are non-destructive; step 1 is the only destructive part.

---

## Verification  *(review finding #5 — group by normalized key, not raw name)*

- **True dups gone:** `SELECT <field>_key, count(*) … GROUP BY <field>_key` per domain type →
  every count = 1. Specifically `fan song` and `spoon rest` each resolve to **one** vertex.
- **Variants preserved:** the 16 SA-2 variants (distinct `system_name`) remain **16** separate
  entities — the fix must NOT collapse distinct-designator entities that share a `name`.
- **Merged Fan Song** carries `nominal_rf_mhz`/`tx_peak_power_kw` and the union of both former
  vertices' `EXTRACTED_FROM` lineage.
- **Backfill complete:** no existing vertex of an indexed domain type has a `NULL` `_key`
  (`SELECT count(*) … WHERE <field>_key IS NULL` = 0), so old rows are matchable.
- **Future-proof:** re-extracting a document that mentions `FAN SONG` upserts onto the existing
  `fan song` `_key` (no new vertex) — and its relationship endpoints resolve to that same vertex.

## Testing

- **Unit:** `norm()`; `LogicalIdentity` equality/hash — two case/whitespace variants of the
  same identity are equal and merge into one record; distinct `system_name`s stay distinct.
- **Integration:** extract a fixture where one designator appears in two casings across two
  chunks/docs → exactly **one** `MergedEntityRecord` → **one** vertex, provenance from both.
- **Regression:** distinct variants sharing a `name` (e.g. two missiles both named "Guideline"
  with different `system_name`) remain **two** vertices.

## Risks / notes

- The **2-vertex targeted merge** (Component 3) is the only destructive step; explicitly
  confirmed at execution time. It is required for the unique index to be creatable.
- Normalization applies to **domain entity identities only**; component/content classes
  (`is_entity=False`) are excluded (review finding #6).
- First-seen display casing depends on ingest order — accepted; a canonical-casing rule is a
  possible additive follow-up.
- `air_defense_v3` is the source-of-truth bundle; the identity-normalization logic lives in
  shared code (`extraction_merge.py` / `arcadedb_schema.py` / `arcadedb_graph.py`), so it
  applies across bundles without per-bundle edits.
