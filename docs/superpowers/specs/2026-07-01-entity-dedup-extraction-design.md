# Extraction-Time Entity Dedup — Design

- **Date:** 2026-07-01
- **Status:** Approved (brainstorming) → pending implementation plan
- **Scope:** entity identity/upsert in extraction commit + schema index build; remediation via re-ingest

## Problem

The knowledge graph contains many duplicate entity vertices for the same real-world
designator: **`Guideline` = 16 vertices**, **`Fan Song` = ~6** (3 `FAN SONG` + 3 `Fan Song`),
`Spoon Rest` similarly. This fragments provenance, dilutes Louvain communities, and splits
extracted specs across copies (only one `Fan Song` vertex carries `nominal_rf_mhz` /
`tx_peak_power_kw`; the rest are empty).

### Root cause (verified)

A **default-value mismatch** across the identity-scope resolution:

- Domain schemas author entities as global — `radar_power_rf.py:34`, `missile_identity.py:38`,
  `radar_identity.py:41`, etc. all set `identity_scope="global"`.
- `introspect.py:110` also **defaults** to `"global"`, and lines `111-113` only write the
  `identity_scope` key into the compiled ontology when it is **non-global** (global is
  omitted as the assumed default).
- But the consumers default a **missing** key to `"document"`:
  - `extraction_merge.py` (LogicalIdentity scope resolution, ~lines 616/632/670/772) →
    `LogicalIdentity.as_upsert_identity_dict()` appends `document_id` for `scope=="document"`.
  - `arcadedb_schema.py:476` → the UPSERT UNIQUE index gets `document_id` appended.
  - `docling_anchors.py:496` → same default.

So a global-authored domain entity is **silently document-scoped**: its upsert identity
becomes `(identity_fields, document_id, entity_type)` → **one vertex per document**. A
designator mentioned in 16 documents → 16 vertices. Cross-document
`canonicalize_document_entities` only adds `HAS_ALIAS` edges; it never merges the vertices.

Separately, identity matching is **case/whitespace-sensitive**, so even when global,
`FAN SONG` and `Fan Song` remain two vertices.

## Goal

Domain entities dedupe **at write time** across (a) documents and (b) case/whitespace, so one
real-world designator = one vertex carrying merged provenance and specs. Remediate existing
data by **re-ingest** (clean slate — no in-place merge script).

## Non-goals

- In-place merge migration of existing duplicates (rejected in favor of re-ingest).
- Changing structural-entity scoping (SECTION / FIGURE / TABLE / DOCUMENT stay `document`).
- A canonical-casing/“preferred display name” algorithm — display name is **first-seen** casing.
- Re-designing canonicalization's alias/fuzzy layers (orthogonal).

---

## Design

### Component 1 — Fix the global/document default mismatch

- **`introspect.py`:** always emit `identity_scope` into the compiled ontology entry (remove
  the "omit when global" special-case at `:111-113`) — the compiled ontology becomes the single
  explicit source of truth; no consumer relies on a default.
- **Align consumer fallbacks to `"global"`** (belt-and-suspenders, in case a path bypasses the
  compiled ontology): `arcadedb_schema.py:476`, `extraction_merge.py` (the ~4 scope-resolution
  sites), `docling_anchors.py:496` — change `.get("identity_scope", "document")` →
  `.get("identity_scope", "global")`.
- Structural types explicitly set `identity_scope="document"`, so they are unaffected.
- **Result:** domain-entity upsert identity = `(identity_fields, entity_type)` — no
  `document_id` → cross-document dedup.

### Component 2 — Case/whitespace-normalized identity

- Define `norm(x) = " ".join(str(x).strip().casefold().split())` (trim, casefold, collapse
  internal whitespace).
- Store a **normalized identity value** on the vertex for each identity field (e.g.
  `system_name_key = norm(system_name)`), and build the UNIQUE index + upsert `WHERE` on the
  **normalized** field(s) + `entity_type` (global scope from Component 1).
- Preserve the original **display name** (`name` / `system_name`) as **first-seen**: the upsert
  must NOT overwrite the display name on update (only set it on insert). Concretely, keep the
  display fields out of the UPSERT `SET`-on-update path (e.g. set them only when the record is
  new, or via a create-only column), so the first casing seen wins deterministically per
  ingest order.
- `FAN SONG`, `Fan Song`, ` Fan  Song ` → identical `_key` → one vertex.

### Component 3 — Schema/index build (fresh graph)

- The corrected `arcadedb_schema.py` upsert-index phase emits the new **global + normalized**
  UNIQUE index DDL. Because remediation is a **fresh re-ingest** into an empty graph, there is
  **no in-place index swap** — the schema is built correctly from scratch.

### Component 4 — Remediation (re-ingest)

1. Land Components 1–3 (code).
2. **Blow away the graph data** (destructive — gated behind explicit execution-time
   confirmation; use the project's standard teardown/reset, not per-document `/cancel`).
3. **Re-ingest all ~21 documents** through the normal pipeline → global, normalized,
   deduped entities from the start.
4. **Re-run community detection** on the clean graph (produces reports over deduped entities).

---

## Verification

- **No duplicates:** for each domain type, `SELECT name, count(*) … GROUP BY name` → every
  count = 1. `Guideline` = 1 (was 16); `Fan Song` = 1 (was ~6); `Spoon Rest` = 1.
- **Merged provenance & specs:** the single `Fan Song` vertex carries `nominal_rf_mhz` /
  `tx_peak_power_kw`, and its `EXTRACTED_FROM` lineage points to **all** of its source
  documents (upsert-merge preserved every document's edges).
- **Case merge:** no `FAN SONG` vertex distinct from `Fan Song`.
- **Downstream:** community detection produces reports; Global Research "Fan Song" returns a
  synthesis that cites the specs.

## Testing

- **Unit:** `norm()` (trim/casefold/whitespace); identity-scope resolution returns `global`
  for a domain type and `document` for a structural type given a compiled ontology entry.
- **Integration:** re-ingest a **2-document** fixture where the same designator appears in both
  and in different casing → assert exactly **one** vertex, with `EXTRACTED_FROM` provenance
  from **both** documents and the display name = the first-seen casing.
- **Regression:** structural entities (SECTION/FIGURE/TABLE) remain document-scoped (still one
  per document) — the fix must not collapse them across documents.

## Risks / notes

- **Destructive wipe** is required for remediation — must be explicitly confirmed at execution
  time (per project rule on destructive actions).
- **Full re-extraction** of ~21 docs is time-consuming (hours) and re-runs community detection.
- **Display-name determinism** depends on ingest order (first-seen casing); acceptable per
  design decision. If a canonical-casing rule is wanted later, it's an additive follow-up.
- The narrowed sibling bundles inherit the fix via the shared `introspect.py`
  (air_defense_v3 is the source of truth; propagate per project convention).
