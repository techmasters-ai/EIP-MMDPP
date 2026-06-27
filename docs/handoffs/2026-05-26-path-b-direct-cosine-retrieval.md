# Handoff: Path B — Direct-Cosine Retrieval (Bypass HNSW for Per-Run Search)

**Repo:** `/home/josh/development/EIP-MMDPP`
**Worktree:** `/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry`
**Branch:** `walltime/c0-telemetry`
**Last commit:** `8792f6a` (token cap rollback)
**Status:** plan only — implementation not started.
**Review:** Codex review applied 2026-05-26; sections marked `[Rev1]` updated
based on findings. See "Codex review — applied corrections" near the bottom
for the original findings list.

---

## TL;DR

Add `search_extraction_chunks_direct()` — a non-HNSW retrieval path that pulls
all `ExtractionChunk` vertices for one `pipeline_run_id` via SQL, then computes
cosine similarity client-side in numpy. Wire it via a feature-flag env var
(`VECTOR_ROUTER_RETRIEVAL_MODE=hnsw|direct`, default `hnsw`). Eliminates a
documented HNSW post-filter starvation bug.

**Latency framing [Rev1]:** The retrieval stage is ~50,000× faster at our
scale (numpy cosine on 300 vectors = 0.04 ms vs HNSW + retry = ~1.9 s).
**End-to-end is NOT necessarily faster** because the downstream reranker
scores every returned candidate. Direct may return MORE candidates than
HNSW's starved 185 (e.g. up to 317 for SA-2), so reranker wall-time can
go UP. The acceptance A/B must measure `vector_search_ms` and `rerank_ms`
separately; the quality win (no starvation) is the primary benefit.

Acceptance gated on an A/B that matches or beats C.7f quality on Dvina +
SA-2 with `mode=direct`, with separate retrieval / rerank / wall budgets.

---

## Background — why this is needed

**Documented HNSW post-filter starvation** (see
`app/services/extraction_chunk_search.py:5-50` module docstring + the
integration test `tests/integration/test_extraction_chunk_filter_starvation.py`):

- ArcadeDB indexes every `ExtractionChunk` across every `pipeline_run` in **one
  shared HNSW graph**.
- `vectorNeighbors()` traverses the global graph and returns globally-top-k
  nearest chunks, then applies `WHERE pipeline_run_id='...'` as a **post-filter**.
- At our scale (~300 chunks/run, ~3000 total across the index), the global
  top-2000 mostly belongs to OTHER runs, leaving only ~185 of our run's 317 to
  survive the filter.

**Concrete evidence (run `3a00e695-2d55-4855-ad33-9d9f41fd39eb`, SA-2 C.7e
radar_power_rf):**

- 317 chunks total, all with embeddings (verified by direct query).
- Computed cosine similarity for every chunk against the radar_power_rf query
  (bypassing HNSW): score distribution `[0.35..0.60]`.
- VR selected 185, excluded 132.
- Within the `0.45-0.50` bin: 91 selected, **76 excluded at the same similarity**.
- Top excluded chunks scored as high as **0.555** (`#/texts/215` — "Fan Song
  Engagement Radar" — clearly relevant).
- Estimated 10 entity loss on radar_power_rf (36 baseline → 26 C.7e) traces to
  these high-similarity excluded chunks.

**`min_similarity` is effectively a no-op** for our queries: all chunks score
above the configured floor; the actual filtering happens inside HNSW's
non-deterministic post-filter behavior, not at the configured threshold.

**Per-run cleanup already exists** — `cleanup_extraction_index(run_id)` is
called at pipeline terminal + an hourly janitor backstops it (24-h TTL). This
work does NOT need to change the cleanup pipeline; per-run isolation is
already enforced at the database level via the `pipeline_run_id` column with a
B-tree secondary index.

---

## Goal

Replace HNSW-based per-run retrieval with **direct cosine over the per-run
filtered chunk set**, keeping the same interface so callers don't change.

Non-goals:
- Removing HNSW entirely. HNSW stays for any future cross-run retrieval. The
  feature flag lets us preserve the old path.
- Per-document vertex types (the "Option A" from earlier brainstorming). The
  current `WHERE pipeline_run_id=X` index already gives us per-run isolation
  at the SQL level — no schema-level isolation needed.
- Re-architecting the reranker. The reranker is downstream of retrieval; this
  change only affects which N chunks reach the reranker.

---

## Architecture

### Current path (`mode=hnsw`)

```
chunk-scope endpoint
  ↓
embed query (bge-m3 1024-dim)
  ↓
search_extraction_chunks() [extraction_chunk_search.py:117]
  ↓
  store.vector_search()  ←  HNSW traversal over GLOBAL graph
  ↓                          post-filter WHERE pipeline_run_id=X
  ↓                          retry once at top_k=2000 if short
  returns top-185-or-so chunks
  ↓
rerank() (bge-reranker-v2-m3, CPU, ~10-30s for 185 candidates)
  ↓
top_k slice
  ↓
selected_refs → apply_chunk_scope → scoped DoclingDocument
```

### Proposed path (`mode=direct`)

```
chunk-scope endpoint
  ↓
embed query (bge-m3 1024-dim)        ← unchanged
  ↓
search_extraction_chunks_direct()    ← NEW
  ↓
  SELECT FROM ExtractionChunk WHERE pipeline_run_id=X
    (already B-tree-indexed; ~50ms for 300 rows)
  ↓
  numpy cosine: query · chunks.embedding (already L2-normalized at write)
    (~0.04ms for 300 × 1024)
  ↓
  filter score ≥ score_threshold
  ↓
  sort desc, take desired_top_n
  ↓
  returns top-N chunks (exact, no approximation, no starvation)
  ↓
rerank()                              ← unchanged (no-op when top_k ≥ returned)
  ↓
top_k slice → selected_refs → apply_chunk_scope → scoped DoclingDocument
```

### Dispatcher

```
search_extraction_chunks(...)  [public entry point — unchanged signature]
  ↓
  if settings.vector_router_retrieval_mode == "direct":
      return await search_extraction_chunks_direct(...)
  else:
      return await search_extraction_chunks_hnsw(...)   # current code, renamed
```

Same return shape `(list[GraphEntityResult], ChunkSearchDiagnostics)` so all
callers (chunk-scope endpoint, tests) work unchanged. Diagnostics field
`filter_strategy` becomes `"direct_cosine"` when the new path is used (vs
existing `"overfetch_post_filter"`).

---

## Implementation steps (numbered)

### 1. Settings + env var

**File:** `app/config.py`

Add to `Settings`:

```python
vector_router_retrieval_mode: Literal["hnsw", "direct"] = Field(
    default="hnsw",
    description=(
        "How VR fetches per-run ExtractionChunks. 'hnsw' uses ArcadeDB's "
        "global HNSW + post-filter (the legacy path, subject to documented "
        "post-filter starvation). 'direct' pulls all chunks for the run via "
        "SQL and computes cosine in numpy — exact, no starvation, ~50,000× "
        "faster at our scale. Switch to 'direct' once the A/B passes."
    ),
)
```

**Files:** `.env` AND `.env.example` (per project rule that env vars live in
both — see `feedback_env_vars_must_appear_in_dotenv_files` memory).

Append after the existing `VECTOR_ROUTER_*` block:

```bash
# VR retrieval mode — 'hnsw' (default, legacy global-graph path) or
# 'direct' (per-run SQL filter + numpy cosine; no HNSW starvation).
# A/B test gate: flip to 'direct' once C.7g shows quality + latency parity
# with C.7f. See docs/handoffs/2026-05-26-path-b-direct-cosine-retrieval.md.
VECTOR_ROUTER_RETRIEVAL_MODE=hnsw
```

### 2. Rename existing function (avoid behavior changes during refactor)

**File:** `app/services/extraction_chunk_search.py`

Rename `async def search_extraction_chunks(...)` →
`async def search_extraction_chunks_hnsw(...)`. **No behavior change.**

Keep the existing module docstring (HNSW post-filter rationale) attached to
the renamed function — it documents that function's purpose.

### 3. Add the new direct function [Rev1 — Codex corrections applied]

**File:** `app/services/extraction_chunk_search.py`

```python
async def search_extraction_chunks_direct(
    *,
    store: "ArcadeDBGraphStore",
    query_vector: list[float],
    pipeline_run_id: str,
    desired_top_n: int,
    score_threshold: float | None = None,
) -> "tuple[list[GraphEntityResult], ChunkSearchDiagnostics]":
    """Per-run vector search via direct cosine in Python.

    Pulls ALL ExtractionChunk vertices for `pipeline_run_id` via SQL
    (B-tree-indexed; trivial). Computes cosine similarity client-side
    against `query_vector`. Returns top-`desired_top_n` chunks above
    `score_threshold`, sorted descending by score with `self_ref` as
    stable tiebreaker.

    Same return shape as `search_extraction_chunks_hnsw` for drop-in
    swap behind the `vector_router_retrieval_mode` env var. Eliminates
    the HNSW global-graph post-filter starvation documented at the top
    of this module. Exact (no approximation) and deterministic. Retrieval
    stage is ~50,000× faster at our scale (300 chunks); end-to-end may be
    similar or slower due to the reranker scoring more candidates.
    """
    import numpy as np
    from app.services.graph_store import GraphEntityResult

    # 1. Pull every chunk for this run. SQL index on pipeline_run_id makes
    #    this O(matching rows). No global scan. ORDER BY self_ref ASC for
    #    stable tiebreaker on equal cosine scores. Match the project's
    #    canonical accessor pattern (named param via :run_id) — see
    #    app/api/v1/extraction_routing.py:139 for the prevailing style.
    rows = await store._client.query(
        store._database,
        "sql",
        (
            "SELECT self_ref, chunk_text, embedding, page_number, modality, "
            "pipeline_run_id "
            "FROM ExtractionChunk "
            "WHERE pipeline_run_id = :run_id "
            "ORDER BY self_ref ASC"
        ),
        {"run_id": pipeline_run_id},
    )

    if not rows:
        return [], ChunkSearchDiagnostics(
            filter_strategy="direct_cosine",
            ann_top_k_requested=0,
            post_filter_candidate_count=0,
            post_filter_retry_count=0,
            short_fetch=False,
        )

    # 2. Build paired valid-rows / embeddings — same length, same order.
    #    Filtering embeddings without filtering rows would misalign scores.
    valid_rows = [r for r in rows if r.get("embedding")]
    if not valid_rows:
        return [], ChunkSearchDiagnostics(
            filter_strategy="direct_cosine",
            ann_top_k_requested=len(rows),
            post_filter_candidate_count=0,
            post_filter_retry_count=0,
            short_fetch=False,
        )

    embeddings = np.array(
        [r["embedding"] for r in valid_rows], dtype=np.float32,
    )
    q = np.array(query_vector, dtype=np.float32)
    # Defensive normalize — bge-m3 emits L2-normalized vectors but this
    # protects against future writer-side drift.
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    q /= np.linalg.norm(q) + 1e-12

    scores = embeddings @ q  # shape: (N,)

    # 3. Threshold first (defines the post-threshold candidate count
    #    used by diagnostics), then top-N slice with stable tiebreaker.
    if score_threshold is not None:
        keep = scores >= float(score_threshold)
        kept_rows = [r for r, k in zip(valid_rows, keep.tolist()) if k]
        kept_scores = scores[keep]
    else:
        kept_rows = valid_rows
        kept_scores = scores

    candidate_count = len(kept_rows)

    # Stable ordering: primary -score, secondary self_ref ASC. lexsort
    # last key is primary.
    if candidate_count > 0:
        self_refs = np.array([r["self_ref"] for r in kept_rows])
        # lexsort uses LAST key as primary; secondary then primary.
        order = np.lexsort((self_refs, -kept_scores))[:desired_top_n]
        selected = [kept_rows[i] for i in order.tolist()]
        selected_scores = kept_scores[order].tolist()
    else:
        selected, selected_scores = [], []

    # 4. Materialize as GraphEntityResult — match the dataclass signature
    #    at app/services/graph_store.py:67 (requires node_id, name,
    #    entity_type; score+score_type for ranking; properties for the
    #    chunk metadata downstream code reads).
    results = [
        GraphEntityResult(
            node_id=str(row.get("@rid", row.get("vertex_id", row["self_ref"]))),
            name=row["self_ref"],
            entity_type="ExtractionChunk",
            extraction_confidence=score,
            score=score,
            score_type="vector",
            properties={
                "self_ref": row["self_ref"],
                "chunk_text": row.get("chunk_text", ""),
                "page_number": row.get("page_number"),
                "modality": row.get("modality"),
                "pipeline_run_id": row.get("pipeline_run_id"),
            },
        )
        for row, score in zip(selected, selected_scores)
    ]

    # short_fetch in direct mode means "fewer matching chunks exist than
    # desired_top_n" (not "retrieval was incomplete" — direct retrieval is
    # exact). Downstream code uses short_fetch to log + adjust expectations.
    return results, ChunkSearchDiagnostics(
        filter_strategy="direct_cosine",
        ann_top_k_requested=len(rows),
        post_filter_candidate_count=candidate_count,
        post_filter_retry_count=0,
        short_fetch=(candidate_count < desired_top_n),
    )
```

**Corrections applied (Codex review):**

- **SQL accessor** [Codex finding 1]: uses
  `store._client.query(store._database, "sql", sql, {"run_id": ...})` with
  `:run_id` named param. Verified prevailing pattern in
  `app/api/v1/extraction_routing.py:139`. Do NOT add a `?` placeholder; not
  used elsewhere in this repo.
- **`GraphEntityResult` construction** [finding 2]: required fields
  `node_id`, `name`, `entity_type` set (`ExtractionChunk` literal for
  entity_type); `extraction_confidence` mirrors `score`; `score_type=
  "vector"`; `properties` includes `pipeline_run_id`. Confirmed against
  dataclass signature at `app/services/graph_store.py:67`.
- **Null-embedding handling** [finding 3]: filters `valid_rows` BEFORE
  building the embeddings matrix; both lists stay length-aligned. Also
  short-circuits returning empty if every embedding is null (would otherwise
  crash on `np.linalg.norm` over empty array).
- **Determinism** [finding 8]: SQL `ORDER BY self_ref ASC` plus
  `np.lexsort((self_refs, -scores))` give deterministic results even when
  multiple chunks tie on score.
- **Diagnostics semantics** [finding 5]: `post_filter_candidate_count` =
  count surviving the threshold (BEFORE top-N slice); `short_fetch` =
  "fewer matches than requested" (exact retrieval — not "incomplete").
- **Performance hygiene** [finding 9]: at 300 rows `argsort` is fine; if
  per-run chunk counts grow beyond ~10K, swap to `argpartition` for the
  top-N partition + sort the partition only.

All ExtractionChunk vertices currently have non-null `embedding`
post-`build_extraction_index` (verified during C.7f investigation: 100%
coverage on 317-chunk SA-2 run). The `if r.get("embedding")` filter is
defensive; should never filter rows in practice but protects against
schema drift.

### 4. Wire the dispatcher

**File:** `app/services/extraction_chunk_search.py`

Add a public entry point that picks the mode based on settings:

```python
async def search_extraction_chunks(
    *,
    store: "ArcadeDBGraphStore",
    query_vector: list[float],
    pipeline_run_id: str,
    desired_top_n: int,
    score_threshold: float | None = None,
) -> "tuple[list[GraphEntityResult], ChunkSearchDiagnostics]":
    """Dispatch to HNSW or direct-cosine retrieval based on settings.

    See `vector_router_retrieval_mode` in `app/config.py`. Default is
    'hnsw' for safe rollback; flip to 'direct' once the A/B verifies
    quality + latency parity (see Path B handoff).
    """
    from app.config import get_settings
    mode = get_settings().vector_router_retrieval_mode

    if mode == "direct":
        return await search_extraction_chunks_direct(
            store=store, query_vector=query_vector,
            pipeline_run_id=pipeline_run_id,
            desired_top_n=desired_top_n,
            score_threshold=score_threshold,
        )
    return await search_extraction_chunks_hnsw(
        store=store, query_vector=query_vector,
        pipeline_run_id=pipeline_run_id,
        desired_top_n=desired_top_n,
        score_threshold=score_threshold,
    )
```

All existing callers (chunk-scope endpoint at
`app/api/v1/extraction_routing.py:275`, integration tests) call
`search_extraction_chunks` — they pick up the dispatch with zero code change.

### 5. TDD — write failing tests FIRST [Rev1 — expanded]

**File:** `tests/unit/test_extraction_chunk_search_direct.py` (NEW)

Write **before** the implementation function exists. Per the
`superpowers-extended-cc:test-driven-development` skill's iron law: production
code only after a failing test.

Required test cases (each one written + run-to-fail before the next):

**Direct-function correctness tests:**

1. **`test_direct_returns_empty_when_no_chunks_for_run`** — mock the SQL
   client to return `[]`; assert empty list + diagnostics fields
   (`filter_strategy="direct_cosine"`, `ann_top_k_requested=0`).
2. **`test_direct_returns_top_n_by_score_descending`** — mock returns 5
   chunks with known embeddings + a query vector; assert the function
   returns the right 3 in the right order when `desired_top_n=3`.
3. **`test_direct_filters_by_score_threshold`** — same setup; with
   `score_threshold=0.5`, assert only the chunks scoring ≥0.5 are
   returned AND `post_filter_candidate_count` reflects the post-threshold
   count.
4. **`test_direct_filters_by_pipeline_run_id_via_sql`** — assert the SQL
   query string contains `WHERE pipeline_run_id = :run_id` and the params
   dict includes `run_id`.
5. **`test_direct_return_shape_matches_hnsw_path`** — call both paths
   with the same mock data; assert `type(result[0]) is GraphEntityResult`
   AND the required fields are set (`node_id`, `name`, `entity_type`,
   `score`, `score_type`, `properties.pipeline_run_id`).
6. **`test_direct_handles_null_embedding_alignment`** — one mock row has
   `embedding=None`; assert it's skipped AND that the returned `node_id`s
   correctly map to the non-null rows (catches the alignment bug Codex
   flagged in finding 3).
7. **`test_direct_returns_empty_when_all_embeddings_null`** — all rows
   have `embedding=None`; assert empty result with diagnostics intact
   (no `np.linalg.norm` crash).
8. **`test_direct_normalizes_query_vector`** — pass an unnormalized
   query; assert scores are bounded in `[-1, 1]` (cosine invariant).
9. **`test_direct_short_fetch_when_candidates_less_than_top_n`** — only
   5 chunks pass threshold but `desired_top_n=10`; assert `short_fetch=
   True` and the 5 are returned.
10. **`test_direct_tiebreaker_stable_on_equal_scores`** — two chunks have
    IDENTICAL embeddings (= identical scores against any query); assert
    they are returned in `self_ref` ASC order deterministically across
    multiple invocations.

**Dispatcher tests (per Codex finding 7):**

11. **`test_dispatcher_routes_to_direct_when_env_is_direct`** — monkeypatch
    `get_settings().vector_router_retrieval_mode = "direct"`, clear the
    `lru_cache` on `get_settings`, then call `search_extraction_chunks`
    and assert the direct variant ran (e.g., assert the diagnostics
    `filter_strategy == "direct_cosine"`).
12. **`test_dispatcher_routes_to_hnsw_when_env_is_hnsw`** — same pattern,
    mode `"hnsw"`; assert `filter_strategy == "overfetch_post_filter"`.
13. **`test_dispatcher_invalid_mode_fails_validation`** — set env var to
    `"banana"`; assert `Settings()` instantiation raises a Pydantic
    validation error.
14. **`test_dispatcher_respects_cleared_settings_cache`** — first call
    with `mode=hnsw`, then change setting, call `get_settings.cache_clear()`,
    set `mode=direct`, second call routes to direct (catches the `lru_cache`
    pitfall Codex flagged).

**Existing-test handling (per Codex finding 6):**

The integration test
`tests/integration/test_extraction_chunk_filter_starvation.py` asserts
`filter_strategy == "overfetch_post_filter"`. While the default stays
`hnsw`, this remains correct. When the default flips to `direct` (step 11),
those two assertions (lines 426 and 577) need EITHER:

- Update the test to call `search_extraction_chunks_hnsw()` directly
  (preferred — the test is specifically about HNSW behavior).
- OR force `monkeypatch.setenv("VECTOR_ROUTER_RETRIEVAL_MODE", "hnsw")` in
  the test fixture and clear the settings cache.

Choose path 1 (direct call) — keeps the test honest about which code path
it's exercising.

Run each test before writing code; confirm the failure is "function not
defined" / "wrong return shape" / "settings field missing" etc. Then
implement, watch GREEN, refactor.

### 6. Integration test (optional but recommended)

**File:** `tests/integration/test_extraction_chunk_search_direct_starvation.py` (NEW)

Mirror of the existing starvation test, but exercises `mode=direct`. Insert
~600 ExtractionChunk rows split across 3 fake `pipeline_run_id`s; assert that
`search_extraction_chunks_direct(pipeline_run_id=A)` returns ONLY chunks from
run A regardless of how chunks from runs B+C score. The HNSW path
demonstrably fails this in `test_extraction_chunk_filter_starvation.py`; the
direct path should pass it cleanly.

### 7. Verify unit tests pass

```bash
docker exec eip-mmdpp-worker-1 python -m pytest \
  /tmp/test_ecs_direct.py -v
```

(Copy the test file into the container via `docker cp` since `tests/` isn't
bind-mounted — see the C.7e/C.7f session log for the established pattern.)

### 8. Force-recreate worker + api to pick up the new code

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
COMPOSE_PROJECT_NAME=eip-mmdpp docker compose up -d --force-recreate \
  worker worker-graph api
```

(The dispatcher lives in `app/services/extraction_chunk_search.py`, which is
imported by `app/api/v1/extraction_routing.py` in the api container. The api
also needs the recreate.)

Verify the new function is loaded:

```bash
docker exec eip-mmdpp-worker-1 python -c "
from app.services.extraction_chunk_search import (
    search_extraction_chunks, search_extraction_chunks_direct,
    search_extraction_chunks_hnsw,
)
print('all three functions importable')
"
```

### 9. A/B run — gated by the env var

**Step A (still on `mode=hnsw`):** confirm a fresh run reproduces C.7f
numbers. Trigger Dvina + SA-2; expect ~234m SA-2 wall + 52 system_links
committed edges. (Optional sanity step; skip if confidence is high.)

**Step B (flip to `mode=direct`):**

```bash
# in .env
VECTOR_ROUTER_RETRIEVAL_MODE=direct
```

Then `docker compose up -d --force-recreate docling-graph api worker worker-graph`
(api + worker re-read settings on process start; docling-graph re-reads `.env`).

Trigger Dvina + SA-2 reingest via the existing watcher script pattern (see
`/tmp/c7f_dvina_then_sa2.sh` for the model). Name this C.7g.

### 10. Acceptance criteria [Rev1 — retrieval-specific gates added]

**Per-pass retrieval gates** (from
`ingest.pipeline_pass_outputs.diagnostics_json->'router'` on the narrowed
passes `radar_power_rf` + `missile_kinematics`):

| Gate | Threshold | Source |
|---|---|---|
| R1. Filter strategy switched | `filter_strategy == "direct_cosine"` on every narrowed pass | router diagnostics |
| R2. No fallback regression | `fallback_reason != "no_chunks_above_threshold"` on any narrowed pass | router diagnostics |
| R3. Candidate count plausible | `post_filter_candidate_count` ≥ HNSW candidate count (185 on SA-2 in C.7f) | router diagnostics |
| R4. Retrieval stage faster | `vector_search_ms` < C.7f's value for the same pass | router diagnostics |
| R5. Rerank not regressed unboundedly | `rerank_ms` ≤ 1.5× C.7f's value | router diagnostics |

**Per-pass quality gates** (catches scope leak from including more chunks):

| Gate | Threshold | Source |
|---|---|---|
| Q1. Quality parity | per-pass entity counts within ±10% of C.7f | `ingest.v_latest_pass_attempts` |
| Q2. system_links committed | ≥ C.7f's 52 | ArcadeDB ASSOCIATED_WITH + VARIANT_OF + CUES + ALIAS_OF counts since C.7g start |
| Q3. Evidence refs survive | for each `radar_power_rf` + `missile_kinematics` entity in C.7f, its source self_ref appears in C.7g's `selected_refs` | manual cross-check via `pipeline_pass_outputs.extract_pass_response_json.provenance` |

**Run-level gates** (noisy, lower priority):

| Gate | Threshold | Source |
|---|---|---|
| W1. SA-2 wall not worse | ≤ 1.1× C.7f's 233.5m | `pipeline_runs.finished_at - started_at` |
| W2. Dvina wall not worse | ≤ 1.1× C.7f's 14.7m | same |

**C.7f baseline numbers** (run UUIDs for direct comparison):
- Dvina C.7f: `f155d08e-edb3-44da-bdaa-8ed7f78c4f5d` — wall 14.7m
- SA-2 C.7f: `62065621-0375-4bf5-88c0-a459109a8da1` — wall 233.5m

**Why retrieval gates matter more than wall time** [Codex finding 4]: SA-2
wall is dominated by `missile_identity` (the non-narrowed full-doc pass).
Path B has zero impact on that pass — it only affects narrowed passes
(`radar_power_rf`, `missile_kinematics`). Run-level wall is therefore a
NOISY signal for Path B; the per-pass retrieval diagnostics are the
falsifiable measurement.

### 11. If all 3 gates pass

Flip the default in `app/config.py` to `mode="direct"` and remove the
deprecation note. Update `.env.example` to match (`hnsw` → `direct`). Keep the
HNSW function in the file as a documented fallback — don't delete; future
cross-run use cases (e.g., UI search across all docs) may need it.

Commit + advance to Task #48 (C.8 — Two-doc PROMOTABLE GATE).

### 12. If a gate fails

| Failure | Likely cause | Action |
|---|---|---|
| Quality regressed >10% | direct returns DIFFERENT chunks than HNSW even when both should agree | check normalize, dimension mismatch, NULL embedding rate — start with the score distribution histogram from C.7e analysis as a sanity sample |
| system_links < 52 | NOT a Path B regression (system_links is upstream of retrieval) — would indicate the identity_aliases fix is sensitive to which entity passes saw which chunks | re-run; if reproducible, narrow with the C.7e/C.7f diff |
| Latency worse than HNSW | SQL pull is slower than expected (network / row size) | profile the SQL; consider streaming OR caching the embedding matrix per run inside the chunk-scope endpoint |

---

## Out of scope for Path B (file as separate work)

- Per-document vertex types (the original "Option A" — rejected as unnecessary
  given SQL-level isolation works).
- Removing HNSW entirely — keep for future cross-run retrieval.
- Reranker rework — separate decision (currently a no-op when `top_k ≥
  candidates`; latency cost worth investigating but orthogonal to Path B).
- ArcadeDB filterable HNSW upstream patch — would close the gap but on an
  unknown timeline.

---

## Risks & mitigations [Rev1]

| Risk | Likelihood | Mitigation |
|---|---|---|
| Direct cosine returns slightly different top-N than HNSW (because HNSW is approximate) | Medium — expected | This is a CORRECTNESS improvement, not a regression. Quality gates Q1+Q3 measure entity output, not retrieved-chunk overlap. |
| End-to-end latency worse than HNSW because rerank scores more candidates | Medium [Codex 4] | Direct returns up to ~317 candidates for SA-2 vs HNSW's starved 185. Rerank cost grows linearly with candidate count. Gate R5 measures this directly; if rerank_ms regresses unboundedly, consider raising `min_similarity` or skipping rerank when `top_k ≥ candidates`. |
| Memory spike pulling 300 × 1024-float arrays for every query | Low | Cap is the per-run chunk count (~300-500); peak working set ~2-5MB per call. Negligible vs the worker's GB-range footprint. |
| ArcadeDB SQL `_client.query(...)` row-count limit | Low | The current chunk indexer (`extraction_chunk_index.py`) already pulls all rows for cleanup with no observed limit. If hit, page or batch. |
| Embeddings not L2-normalized at write time | Low | Re-normalize defensively in step 3. Confirmed bge-m3 emits normalized vectors but defensive normalize protects against drift. |
| `lru_cache` on `get_settings()` masks env changes across test cases [Codex 7] | Medium | Test fixtures must `get_settings.cache_clear()` between cases that vary the mode. Documented in dispatcher tests. |
| Equal-score tie ordering changes between runs [Codex 8] | Low | `ORDER BY self_ref ASC` in SQL + `np.lexsort((self_refs, -scores))` for stable secondary sort. Tested in #10. |
| Existing starvation test breaks when default flips [Codex 6] | Confirmed | The test currently asserts `filter_strategy == "overfetch_post_filter"`. Step 5 directs the implementer to migrate those assertions to `search_extraction_chunks_hnsw()` direct calls before flipping the default. |
| `numpy` becomes a direct first-class dependency [Codex 10] | Low | numpy is already an effective dependency (transitive via ML libraries). Add an explicit entry to `pyproject.toml` dependencies for hygiene; not a runtime concern. |

---

## Files this touches [Rev1]

| File | Action | Size |
|---|---|---|
| `app/config.py` | add 1 setting (Literal type) | ~10 lines |
| `app/services/extraction_chunk_search.py` | rename + add 2 functions (direct + dispatcher) | ~130 lines |
| `.env` | add 1 var | 3 lines |
| `.env.example` | add 1 var | 3 lines |
| `pyproject.toml` | optional: add explicit `numpy` dependency [Codex 10] | 1 line |
| `tests/unit/test_extraction_chunk_search_direct.py` | new TDD tests (10 direct + 4 dispatcher) | ~300-400 lines |
| `tests/integration/test_extraction_chunk_filter_starvation.py` | migrate 2 assertions to call `search_extraction_chunks_hnsw()` directly | ~10 lines edited |
| `tests/integration/test_extraction_chunk_search_direct_starvation.py` | new (optional but recommended) | ~80 lines |

Approximate total: 450-550 lines added, 1 function renamed (no behavior
change), 1 existing test migrated. Suggested commit order:

1. Config setting + .env (the feature flag, defaulted to `hnsw` — safe no-op)
2. Function rename `search_extraction_chunks` → `search_extraction_chunks_hnsw` + dispatcher
3. New direct function + TDD tests (one commit per RED→GREEN cycle is overkill; batch the 10+4 tests)
4. Migrate existing starvation test to call `_hnsw` variant explicitly
5. Integration test for direct (optional)
6. Flip default to `direct` — SEPARATE commit so A/B can revert by reverting one commit [Codex 5]

---

## Pre-flight checklist (before starting work)

- [ ] On branch `walltime/c0-telemetry` (confirm via `git status`)
- [ ] HEAD is `8792f6a` (token cap rollback) or later
- [ ] C.7f acceptance gates all pass (verified at session end 2026-05-26):
  - SA-2 wall 233.5m ✓
  - system_links 52 committed ✓
  - per-pass entity counts within ±10% of C.7e ✓
- [ ] `docker ps` shows worker/worker-graph/api/docling-graph all healthy
- [ ] Worktree's `.env` has `DOCLING_GRAPH_LLM_MAX_TOKENS=16384` (rolled back)
- [ ] No in-flight pipeline_run (verify via the SQL in step 9)
- [ ] Read this entire handoff before starting; don't shortcut the TDD order

---

## How to verify the actual SQL accessor name before implementing

The `store.query()` call in step 3 of the plan is the one detail I want
Codex (or whoever picks this up) to verify against current source before
typing the code:

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
grep -n "async def query\|def run_query\|def execute" app/services/graph_store.py app/db/arcadedb*.py 2>/dev/null
```

Use the canonical async accessor that ALL other `app/services/*` modules use.
If multiple candidates exist, pick the one used by `extraction_chunk_index.py`
since that module reads/writes the same `ExtractionChunk` type and is the
natural pattern partner.

---

## Open questions for review

1. **Should `score_threshold=None` mean "no threshold" or "use manifest
   default"?** The HNSW path treats `None` as "no filter at this step"; the
   manifest's `min_similarity` is applied earlier in
   `extraction_routing.py`. Suggest keeping the same semantics for parity.
2. **Should we surface the full score distribution in diagnostics?** Cheap
   to add and would give us the histogram data without re-running the
   instrumentation script we used for the C.7e analysis. Currently the
   diagnostics dataclass only exposes summary fields (`score_range`,
   `candidate_count`). Adding `score_histogram: list[int]` (10 bins) seems
   low-risk.
3. **Default should stay `hnsw` for one A/B cycle (C.7g)**, then flip to
   `direct` if green. Agreed?
4. **Do we need a per-pass override?** Some passes are non-narrowed
   (run on full doc); the retrieval mode is irrelevant for those. The
   chunk-scope endpoint already short-circuits when `input_mode !=
   "document_only"`-narrowed, so no override needed. Confirm by reading
   `extraction_routing.py:apply_chunk_scope_endpoint` to verify the
   short-circuit covers all non-narrowed cases.

---

## Reviewer prompt (paste this to Codex)

> Review the plan in
> `docs/handoffs/2026-05-26-path-b-direct-cosine-retrieval.md` against the
> code at HEAD (`walltime/c0-telemetry` branch, commit `8792f6a` or later).
> Focus on:
> - Is `search_extraction_chunks_direct` the cleanest place to add this, or
>   does it belong elsewhere (e.g., a new module)?
> - Is the SQL accessor name (`store.query()` in step 3) correct? If not,
>   what's the right one?
> - Are the 7 unit tests + 1 integration test sufficient? Anything missing?
> - Any concerns with the `numpy` dependency at this layer? (Already used
>   elsewhere; should be fine.)
> - Should we ship the default flip in the same commit as the
>   implementation, or keep them separate? Plan suggests separate; reviewer
>   may disagree.
> - Are the acceptance gates measurable + falsifiable?
> Report concise findings (<400 words).

---

## Codex review — applied corrections

Codex reviewed the original plan on 2026-05-26 and surfaced 10 findings.
All 10 verified against the source and incorporated into this revision.

| # | Finding | Section updated |
|---|---|---|
| 1 | `store.query(...)` is not the accessor; use `store._client.query(store._database, "sql", sql, {"run_id": ...})` with named `:run_id` param | Step 3 — SQL accessor |
| 2 | `GraphEntityResult` requires `node_id`, `name`, `entity_type`; include `pipeline_run_id` in properties | Step 3 — GraphEntityResult construction |
| 3 | Null-embedding handling was buggy: filtered embeddings but not rows → misalignment + crash on all-null | Step 3 — `valid_rows` pre-filter |
| 4 | Latency claim overstated — reranker still scores every returned candidate; end-to-end may regress | TL;DR + Step 10 (gate R5) + Risks |
| 5 | Diagnostics semantics for `post_filter_candidate_count` + `short_fetch` need clarification in direct mode | Step 3 + Step 10 — gate definitions |
| 6 | Existing starvation test will break when default flips; needs migration to `_hnsw()` direct call | Step 5 — existing-test handling |
| 7 | Add dispatcher tests (mode routing, settings cache, invalid values) | Step 5 — 4 new dispatcher tests |
| 8 | Equal-score tie ordering needs explicit `ORDER BY self_ref` + stable sort secondary key | Step 3 — SQL `ORDER BY` + `np.lexsort` |
| 9 | `argpartition` is hygiene at 300 rows; note for future scale | Step 3 — performance hygiene note |
| 10 | `numpy` arrives transitively; add explicit `pyproject.toml` entry for hygiene | Files this touches |

### Acceptance-gate revision (Codex finding 4)

Original gates were run-level wall time. Whole-pipeline wall is noisy because
SA-2 wall is dominated by `missile_identity` (full-doc, not narrowed) — a
pass Path B doesn't touch. Revised gates are split:

- **R1-R5** (retrieval-stage, from `pipeline_pass_outputs.diagnostics_json->'router'`)
  — directly measure what Path B changes.
- **Q1-Q3** (quality, per pass + ArcadeDB edge count) — catch any scope leak
  from including more chunks.
- **W1-W2** (run wall time, lower priority) — confirm no end-to-end blow-up
  even though missile_identity dominates.
