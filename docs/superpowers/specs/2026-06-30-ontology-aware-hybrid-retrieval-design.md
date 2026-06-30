# Ontology-Aware Hybrid Retrieval — Design

**Date:** 2026-06-30
**Status:** Approved (brainstorming) — pending implementation plan
**Scope:** The multi-modal **hybrid** search path only (`strategy=hybrid` → `_multi_modal_pipeline`). Does NOT touch the basic single-modality vector path or the deterministic graph-profile query mode.

---

## 1. Problem

Graph expansion in hybrid search was intended to retrieve **ontologically** similar chunks — content related through the air-defense domain ontology (a radar that `CUES`/`ASSOCIATED_WITH` a missile, a `VARIANT_OF`, a `PART_OF` component) — not merely semantically similar ones. It does not do this today:

1. **Expansion never walks the domain ontology.** `_expand_via_ontology` → `get_ontology_linked_chunks` (`app/services/arcadedb_graph.py`) does a 2-hop **`EXTRACTED_FROM` co-mention bounce**: `seed_chunk ←EXTRACTED_FROM— entity —EXTRACTED_FROM→ other_chunks`. It only finds *other chunks mentioning the same entity*. It never follows a domain relation (`CUES`, `ASSOCIATED_WITH`, `VARIANT_OF`, …), so querying an SA-2 chunk never reaches the related Fan Song radar chunks even though those edges exist in the graph.
2. **The per-relation weights are dead code for retrieval.** `_expand_via_ontology` hardcodes `ontology_rel_type="EXTRACTED_FROM"` (`retrieval.py:795`), which isn't in the weight table → every expanded chunk gets the flat `default=0.70`. The 0.75–0.95 relationship priors are never applied.
3. **The weight table is mis-keyed.** `SCORING_WEIGHTS` (air_defense_v3) lists `IS_VARIANT_OF` (0.95) but the live predicate is `VARIANT_OF` → falls to 0.70; `ASSOCIATED_WITH`/`CUES` are absent → 0.70.
4. **The reranker erases the ontological signal.** `_apply_reranker` overwrites the fused (semantic+ontology) score with a **pure cross-encoder semantic** score (`retrieval.py:249`) and the `top_k` cap runs *before* rerank. So even an ontology chunk that survives the cap is re-scored purely semantically and demoted.
5. **Structural under-weighting.** The global ontology fusion weight is `0.15` (vs semantic `0.65`): a perfect 1-hop `VARIANT_OF` contributes only `0.15×0.95≈0.14` — by construction it cannot compete with a strong semantic match.

Net: in the hybrid path, "ontological similarity" today means "shares an extracted entity," scored at a flat default and then erased by the reranker.

## 2. Goal

For the hybrid search path, retrieve chunks that are **ontologically** related through the air-defense domain relations, weight them by the relation traversed, and carry **both** the semantic and ontological signal through the `top_k` cap and the reranker into the final returned context — with a per-query UI control over how strongly ontology is guaranteed a seat.

## 3. Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Relations to traverse | **Curated high-value subset**: `VARIANT_OF, ASSOCIATED_WITH, CUES, PART_OF, CONTAINS, USES_COMPONENT` (config-overridable) |
| Ontology strength | **Both** — soft additive weighting **and** reserved slots (hard membership guarantee) |
| Hop depth | **1 domain hop** (seed entity → directly-related entity → its chunks) |
| Architecture | **Dedicated unit** — new traversal method + new expansion function, *augmenting* (not replacing) the existing co-mention expansion |
| Reserved-slot count `M` | **Per-query UI control** for hybrid search, env default `3` |

## 4. Components

### 4.1 Traversal — `get_related_entity_chunks` (new, `app/services/arcadedb_graph.py`)

Mirrors `get_ontology_linked_chunks` but walks one **domain** hop:

```
seed_chunk ←(EXTRACTED_FROM)— entity —[curated domain rel, both directions]— related_entity —(EXTRACTED_FROM)→ chunks
```

- **Signature:** `get_related_entity_chunks(node_id: str, rel_types: list[str], limit: int) -> list[dict]`.
- **Undirected traversal** (`.both(<rel_types>)`) over the curated set: domain edges are stored directionally (`CUES` radar→missile, `VARIANT_OF` variant→base, `ASSOCIATED_WITH` symmetric); retrieval wants the related content regardless of stored direction.
- **True relation captured per chunk.** Each returned chunk carries the actual relation traversed + the related-entity name. Implementation: bind the edge in the MATCH and read its `@type`; if ArcadeDB MATCH edge-binding proves unreliable (the existing code documents MATCH quirks), fallback is one MATCH per relation type, unioned — relation known by construction.
- **Dedup by chunk**, keeping the **highest-weighted** relation when a chunk is reachable via several.
- **Bounded** by `limit` (new `RETRIEVAL_DOMAIN_EXPAND_K`, default 5), independent of the co-mention cap.
- `node_id` may be RID or chunk UUID (resolve as `get_ontology_linked_chunks` does); returns `[]` for non-chunk seeds.

### 4.2 Domain expansion — `_expand_via_domain_relations` (new, `app/api/v1/retrieval.py`)

Parallel to `_expand_via_ontology`, called from `_expand_seeds` alongside it (augments — co-mention stays untouched):

- For each seed, call `get_related_entity_chunks` with the curated rel set.
- For each result, look up chunk data (`_lookup_chunk_by_type`), stamp `context = {"source": "ontology_relation", "rel_type": <true relation>, "related_entity": <name>, "reserved": <set later>}`.
- Initial score via `compute_fusion_score(semantic_score=<seed score>, ontology_rel_type=<true relation>, ontology_hops=1, ...)`; re-scored in §4.4.

### 4.3 Retrieval relation-weight table (new, decoupled from extraction)

A **retrieval-specific** relation-weight map (env / bundle `retrieval_relation_weights` block), keyed on live predicates — leaves the extraction `SCORING_WEIGHTS` untouched:

| Relation | Weight |
|---|---|
| `VARIANT_OF` | 0.95 |
| `USES_COMPONENT` | 0.92 |
| `CONTAINS` / `PART_OF` | 0.90 |
| `CUES` | 0.88 |
| `ASSOCIATED_WITH` | 0.85 |
| `default` (incl. `EXTRACTED_FROM` co-mention) | 0.70 |

`compute_fusion_score` / `_rescore_expanded_chunks` consume this for ontology-relation chunks; co-mention chunks keep `default`. Separately (optional, independent cleanup): fix the dead `IS_VARIANT_OF→VARIANT_OF` key in the extraction `SCORING_WEIGHTS`.

Global fusion rebalance: `semantic 0.65 / doc 0.20 / ontology 0.15` → **`0.60 / 0.15 / 0.25`** so domain edges have real voice in non-reserved slots. Env-tunable; validated by a quick pre-flight experiment.

### 4.4 Reserved slots + blended rerank (`_multi_modal_pipeline`, `_apply_reranker`)

Operate on the post-dedup/diversify/modality-filter pool, before the `top_k` cap.

**Membership — reserved slots.** Reserve up to `M` of `top_k` slots for domain-expanded chunks that qualify:
- relation weight ≥ `RESERVE_MIN_REL_WEIGHT` (default 0.85), **and**
- **raw query-chunk cosine** (the `semantic_score` component computed in `_rescore_expanded_chunks`, *not* the fused score) ≥ `RESERVE_MIN_COSINE` (default 0.15 — low, to admit ontologically-central but semantically-distant chunks while excluding garbage).

Note: with the §4.3 curated weights all ≥ 0.85, the rel-weight gate currently admits every curated-relation chunk; it exists as a forward-safe floor if the curated set later includes weaker relations. The cosine floor is the active discriminator against off-topic expansions.

Among qualifiers, take the top-`M` by fused score, dedup against the normally-ranked picks, then fill `top_k − M_used` by fused score. `M=0` reverts to pure ranking.

**Ordering — blended rerank.** Change `_apply_reranker` from overwrite to blend:
```
score = α · rerank_semantic + (1 − α) · fused_score        # α = RERANK_BLEND_ALPHA, default 0.6
```
The reranker's input pool is already ≤ `top_k`, so it never drops — reserved chunks stay in; the blend keeps the ontological signal in the final ordering for all chunks. `α=1.0` reproduces today's pure-semantic behavior.

**Provenance.** Reserved chunks carry `context.reserved=true` + `rel_type`/`related_entity`, surfaced in the source list and logs ("ontological reservation via ASSOCIATED_WITH → Fan Song") — satisfies the data-lineage rule and makes behavior debuggable.

### 4.5 Config surface — `M` exposed in the UI

**Request parameter.** `UnifiedQueryRequest.ontology_reserved_slots: int | None = None` (`app/schemas/retrieval.py`), resolved in `_multi_modal_pipeline` as `body.ontology_reserved_slots ?? settings.retrieval_ontology_reserved_slots`, clamped to `[0, top_k]`.
- `POST /retrieval/query` carries it via the body.
- `GET /agent/context` adds a `reserved_slots` query param plumbed into the `UnifiedQueryRequest` it builds (`agent.py:70`).

**UI control.** Hybrid-mode-only numeric stepper "Ontology reserved slots" in `QueryPage` (range `0…top_k`, default seeded from the server). Sent on the query request.

**Settings endpoint.** Extend `GET /settings/retrieval` (`retrieval.py:1274`) to return `ontology_reserved_slots` (and other defaults) so the UI seeds from the server.

**Env vars (both `.env` + `.env.example`, with comments):**

| Env var | Default | UI? |
|---|---|---|
| `RETRIEVAL_ONTOLOGY_RESERVED_SLOTS` | 3 | per-query |
| `RETRIEVAL_ONTOLOGY_RESERVE_MIN_REL_WEIGHT` | 0.85 | no |
| `RETRIEVAL_ONTOLOGY_RESERVE_MIN_COSINE` | 0.15 | no |
| `RETRIEVAL_RERANK_BLEND_ALPHA` | 0.6 | no |
| `RETRIEVAL_DOMAIN_EXPAND_K` | 5 | no |
| `RETRIEVAL_ONTOLOGY_WEIGHT` | 0.25 (was 0.15) | no |
| `RETRIEVAL_SEMANTIC_WEIGHT` | 0.60 (was 0.65) | no |
| `RETRIEVAL_DOC_STRUCTURE_WEIGHT` | 0.15 (was 0.20) | no |
| `RETRIEVAL_DOMAIN_RELATION_WEIGHTS` | curated table (§4.3) | no |

**Kill-switch:** `RETRIEVAL_ONTOLOGY_RESERVED_SLOTS=0` + `RETRIEVAL_RERANK_BLEND_ALPHA=1.0` → byte-identical to today's behavior.

## 5. Files affected

- `app/services/arcadedb_graph.py` — new `get_related_entity_chunks` (+ GraphStore Protocol decl in `app/services/graph_store.py`).
- `app/api/v1/retrieval.py` — new `_expand_via_domain_relations`, call it in `_expand_seeds`; reserved-slot logic in `_multi_modal_pipeline`; blended `_apply_reranker`; extend `GET /settings/retrieval`.
- `app/api/v1/_retrieval_helpers.py` — retrieval relation-weight table + `compute_fusion_score` consuming the true relation.
- `app/schemas/retrieval.py` — `ontology_reserved_slots` on `UnifiedQueryRequest`.
- `app/api/v1/agent.py` — `reserved_slots` query param plumbing.
- `app/config.py` + `.env` + `.env.example` — new settings.
- `frontend/src/components/QueryPage.tsx` (+ api client) — hybrid-mode reserved-slots stepper.
- Tests: `tests/unit/` (weight table, reserved-slot selection, blended rerank, request plumbing, traversal shaping), integration (MATCH traversal vs seeded graph), E2E acceptance.

## 6. Acceptance criteria

1. A hybrid query on a known air-defense system (e.g. SA-2) returns at least one chunk reached via a **domain** relation (`ASSOCIATED_WITH`/`CUES`/`VARIANT_OF` — e.g. a Fan Song chunk) that does **not** appear in the pre-change result, marked `context.reserved=true` with its `rel_type`.
2. Domain-expanded chunks are weighted by the **true** relation (`VARIANT_OF`=0.95 ≠ co-mention `EXTRACTED_FROM`=0.70), verified in the fused score.
3. Reserved-slot selection admits up to `M` qualifying (top-tier rel ≥0.85, cosine ≥0.15) domain chunks into `top_k`; below-floor / non-tier chunks are not reserved.
4. The reranker **blends** (`α·rerank + (1−α)·fused`) rather than overwrites; reserved chunks are never dropped.
5. `M` is settable per query from the hybrid-search UI and via the request parameter, defaulting from the server.
6. Kill-switch (`RESERVED_SLOTS=0`, `ALPHA=1.0`) reproduces today's results exactly.
7. The co-mention (`EXTRACTED_FROM`) expansion is preserved (augmented, not replaced).

## 7. Non-goals

- Changing the basic single-modality vector path or the deterministic graph-profile query mode.
- A custom ontology-aware cross-encoder model.
- Multi-hop (>1) domain traversal (deferred; hop-penalty infra exists if revisited).
- Exposing the advanced knobs (rel-weight gate, cosine floor, α) in the UI.
