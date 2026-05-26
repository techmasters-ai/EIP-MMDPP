# Handoff: Path B — Implementation Review (post-implementation)

**Repo:** `/home/josh/development/EIP-MMDPP`
**Worktree:** `/home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry`
**Branch:** `walltime/c0-telemetry`
**Implementation commit:** `be29e02` (`feat(vr): Path B — direct-cosine retrieval bypasses HNSW starvation`)
**Status:** Implementation complete and tested. Default still `mode=hnsw`. **No A/B run yet.** Awaiting code review before flipping default + running C.7g A/B.
**Related plan:** `docs/handoffs/2026-05-26-path-b-direct-cosine-retrieval.md` (the original spec — already Codex-reviewed once, all 10 findings applied as [Rev1])

This document describes **what I actually built**, not what was planned. Use it to verify the implementation matches the spec.

---

## 1. Summary of changes

| File | Lines | What changed |
|---|---|---|
| `app/config.py` | +11 | `vector_router_retrieval_mode: Literal["hnsw","direct"] = "hnsw"` added next to the existing `vector_router_mode` field |
| `app/services/extraction_chunk_search.py` | +219 / -3 | Renamed `search_extraction_chunks` → `search_extraction_chunks_hnsw`; added `search_extraction_chunks_direct` (~170 lines); added new public dispatcher `search_extraction_chunks` (~30 lines) |
| `.env` + `.env.example` | +6 each | `VECTOR_ROUTER_RETRIEVAL_MODE=hnsw` with explanatory comments |
| `tests/integration/test_extraction_chunk_filter_starvation.py` | +20 / -2 | 2 call sites migrated from `search_extraction_chunks` → `search_extraction_chunks_hnsw as search_extraction_chunks` |
| `tests/unit/test_extraction_chunk_search_direct.py` | +362 (new file) | 14 tests: 10 direct-function + 4 dispatcher |

Single commit: `be29e02` (6 files changed, 1507 insertions, 3 deletions — the 1507 includes the planning doc `2026-05-26-path-b-direct-cosine-retrieval.md` which was committed in the same change).

---

## 2. The new function — what it does, line by line

`app/services/extraction_chunk_search.py:247-410` — `search_extraction_chunks_direct`

### Signature

```python
async def search_extraction_chunks_direct(
    *,
    store: "ArcadeDBGraphStore",
    query_vector: list[float],
    pipeline_run_id: str,
    desired_top_n: int,
    score_threshold: float | None = None,
) -> "tuple[list[GraphEntityResult], ChunkSearchDiagnostics]":
```

**Identical to** `search_extraction_chunks_hnsw` signature — drop-in swap.

### Behavior

1. **SQL pull** (`store._client.query`): `SELECT self_ref, chunk_text, embedding, page_number, modality, pipeline_run_id FROM ExtractionChunk WHERE pipeline_run_id = :run_id ORDER BY self_ref ASC`.
   - Named `:run_id` param (matches `extraction_routing.py:139` pattern).
   - `ORDER BY self_ref ASC` for stable iteration / tiebreaker.
   - Returns 0 rows → returns `(empty list, diag with short_fetch=desired_top_n>0)`.

2. **Build valid-rows + embeddings** (paired length-aligned):
   ```python
   valid_rows = [r for r in rows if r.get("embedding")]
   embeddings = np.asarray([r["embedding"] for r in valid_rows], dtype=np.float32)
   ```
   All-null short-circuits with a separate return path (no empty-array crash on `np.linalg.norm`).

3. **Normalize both sides** to L2 unit length (`+ 1e-12` guard against zero):
   ```python
   embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
   q /= np.linalg.norm(q) + 1e-12
   ```
   Cosine = dot product after normalization.

4. **Threshold filter** (defines `candidate_count` for diagnostics):
   ```python
   if score_threshold is not None:
       keep_mask = scores >= float(score_threshold)
       kept_rows = [r for r, k in zip(valid_rows, keep_mask.tolist()) if k]
       kept_scores = scores[keep_mask]
   ```
   `candidate_count = len(kept_rows)` — surviving the threshold, BEFORE the top-N slice.

5. **Stable sort + top-N slice**:
   ```python
   self_refs = np.asarray([r["self_ref"] for r in kept_rows])
   order = np.lexsort((self_refs, -kept_scores))[:desired_top_n]
   ```
   `np.lexsort` uses the LAST key as primary; primary = `-scores` (descending), secondary = `self_refs` ASC (stable tiebreaker).

6. **Materialize `GraphEntityResult`** with all required fields:
   ```python
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
   ```
   `node_id` cascade: prefer `@rid` (ArcadeDB's internal vertex id), fall back to `vertex_id` (the synthetic `f"{pipeline_run_id}:{self_ref}"` PK), then `self_ref` as a string id of last resort.

7. **Return** `(results, ChunkSearchDiagnostics(filter_strategy="direct_cosine", ...))`.

### Diagnostics field semantics (direct path)

| Field | Value |
|---|---|
| `filter_strategy` | `"direct_cosine"` (vs HNSW's `"overfetch_post_filter"`) |
| `ann_top_k_requested` | Total rows pulled for the run (proxy for candidate pool size; not an HNSW top_k) |
| `post_filter_candidate_count` | Rows surviving the threshold, BEFORE the `desired_top_n` slice |
| `post_filter_retry_count` | Always `0` (direct path doesn't retry) |
| `short_fetch` | `True` iff `candidate_count < desired_top_n` (means "not enough matches in this run", NOT "retrieval was incomplete") |

---

## 3. The dispatcher

`app/services/extraction_chunk_search.py:414-447` — public `search_extraction_chunks`

```python
async def search_extraction_chunks(...):
    from app.config import get_settings
    mode = get_settings().vector_router_retrieval_mode
    if mode == "direct":
        return await search_extraction_chunks_direct(...)
    return await search_extraction_chunks_hnsw(...)
```

Same signature + return shape as the historical entry point. Routes by setting. All existing callers (chunk-scope endpoint at `extraction_routing.py:275`, integration tests) work unchanged because they import the public name.

`get_settings` is `@lru_cache`-decorated. Callers that change the env var between calls must invoke `get_settings.cache_clear()`. This is tested explicitly (`test_dispatcher_respects_cleared_settings_cache`).

---

## 4. Test suite — 14 unit tests

`tests/unit/test_extraction_chunk_search_direct.py`

### Direct-function tests (10)

| # | Test | Asserts |
|---|---|---|
| 1 | `test_direct_returns_empty_when_no_chunks_for_run` | Empty rows → empty result + `ann_top_k_requested=0`, `short_fetch=True` |
| 2 | `test_direct_returns_top_n_by_score_descending` | 5 chunks with multi-axis embeddings ranked correctly by cosine |
| 3 | `test_direct_filters_by_score_threshold` | `threshold=0.5` keeps only chunks scoring ≥0.5; `candidate_count` reflects post-threshold count |
| 4 | `test_direct_filters_by_pipeline_run_id_via_sql` | SQL string contains `WHERE pipeline_run_id = :run_id`; params dict is `{"run_id": "run-XYZ"}` |
| 5 | `test_direct_return_shape_matches_hnsw_path` | `GraphEntityResult` has `node_id`, `name`, `entity_type="ExtractionChunk"`, `score`, `score_type="vector"`, `properties["pipeline_run_id"]` |
| 6 | `test_direct_handles_null_embedding_alignment` | Row with `embedding=None` skipped without misaligning scores/rows for remaining rows |
| 7 | `test_direct_returns_empty_when_all_embeddings_null` | All-null embeddings → empty result + `candidate_count=0` (no `np.linalg.norm` crash) |
| 8 | `test_direct_normalizes_query_vector` | Unnormalized query → scores still in `[-1, 1]`; aligned vectors score ~1.0 |
| 9 | `test_direct_short_fetch_when_candidates_less_than_top_n` | 2 chunks pass threshold but `desired_top_n=10` → `short_fetch=True` |
| 10 | `test_direct_tiebreaker_stable_on_equal_scores` | Two chunks with IDENTICAL embeddings + SQL feeding them in `z, a` order → `["a_chunk", "z_chunk"]` (ASC tiebreaker holds) |

### Dispatcher tests (4)

| # | Test | Asserts |
|---|---|---|
| 11 | `test_dispatcher_routes_to_direct_when_setting_is_direct` | `monkeypatch.setenv("VECTOR_ROUTER_RETRIEVAL_MODE", "direct")` + `get_settings.cache_clear()` → diagnostics show `filter_strategy="direct_cosine"` |
| 12 | `test_dispatcher_routes_to_hnsw_when_setting_is_hnsw` | Same with `"hnsw"` → direct NOT called (`monkeypatched` spy); HNSW returns `filter_strategy="overfetch_post_filter"` |
| 13 | `test_dispatcher_invalid_mode_fails_settings_validation` | `VECTOR_ROUTER_RETRIEVAL_MODE="banana"` → `Settings()` raises `pydantic.ValidationError` |
| 14 | `test_dispatcher_respects_cleared_settings_cache` | Demonstrates the `lru_cache` pitfall: changing the env var without `cache_clear` keeps the old route; calling `cache_clear` picks up the new route |

**Test result:** `14 passed in 0.22s` (verified post-implementation).

---

## 5. Existing-test migration

`tests/integration/test_extraction_chunk_filter_starvation.py` had two assertions that depend on the HNSW path being the public default:

- Line 426: `assert diag.filter_strategy == "overfetch_post_filter"`
- Line 577: `assert diag.filter_strategy == "overfetch_post_filter"`

Both call sites of `search_extraction_chunks` (lines 341, 487) migrated to:

```python
from app.services.extraction_chunk_search import (
    search_extraction_chunks_hnsw as search_extraction_chunks,
)
```

This keeps the test's HNSW-specific intent intact regardless of what the dispatcher defaults to. Path B's default-flip commit (still TBD) won't break this test.

---

## 6. What is NOT in this commit

Per the plan (and Codex finding 5), the following are explicit follow-ups:

- **Flipping the default to `mode=direct`** — separate commit so the A/B can revert by reverting one commit. The C.7g A/B run is the gate for this flip.
- **Optional integration test** (`test_extraction_chunk_search_direct_starvation.py`) — the unit tests already cover the per-run-isolation behavior via mocked SQL; an integration variant would need real ArcadeDB. Deferred until after C.7g shows the implementation works in production.
- **Explicit `numpy` in `pyproject.toml`** (Codex finding 10) — numpy is already a transitive dependency (via ML libs); adding the explicit entry is hygiene, not correctness. Deferred.
- **`argpartition` performance hygiene** (Codex finding 9) — at our scale of ~300 chunks/run, `np.argsort` is fine. Note in the implementation docstring + ready to swap if per-run counts grow to 10K+.

---

## 7. Verification — what I ran

```bash
# 1. Imports + dispatcher routing (no test framework)
docker exec eip-mmdpp-worker-1 python -c "
from app.services.extraction_chunk_search import (
    search_extraction_chunks,
    search_extraction_chunks_hnsw,
    search_extraction_chunks_direct,
)
# All three functions importable, signatures match.
"
# Output: all three params match: ['store', 'query_vector', 'pipeline_run_id',
# 'desired_top_n', 'score_threshold']. Default mode = 'hnsw'.

# 2. New unit tests
docker cp tests/unit/test_extraction_chunk_search_direct.py eip-mmdpp-worker-1:/tmp/test_ecs.py
docker exec eip-mmdpp-worker-1 python -m pytest /tmp/test_ecs.py -v
# Output: 14 passed in 0.22s

# 3. Existing chunk-scope endpoint regression
docker cp tests/unit/test_v1_extraction_routing.py eip-mmdpp-worker-1:/tmp/test_v1.py
docker exec eip-mmdpp-worker-1 python -m pytest /tmp/test_v1.py /tmp/test_ecs.py -q
# Output: 18 passed (v1) + 14 passed (ecs_direct), 19 errors (pre-existing
# missing-fixture issues — `client` fixture not picked up when tests run from /tmp).
# Zero new failures.
```

The 19 errors in `test_v1` are pre-existing — caused by running tests from `/tmp` instead of the project root (the `conftest.py` providing the `client` fixture isn't visible). Not Path B regressions. Verified by running a single error case manually: `ERROR at setup of TestAsyncEmbedViaExecutor.test_async_embed_via_run_in_executor — fixture 'client' not found`.

---

## 8. What I want you to verify

Codex review focus areas (please respond to each):

### 8.1 Correctness

1. **SQL accessor pattern** — `store._client.query(store._database, "sql", sql, {"run_id": pipeline_run_id})`. Is this the canonical method now? I verified one prevailing call site at `extraction_routing.py:139`. Are there other patterns in the codebase I should match instead (e.g., a higher-level wrapper)?

2. **`GraphEntityResult` field values** — I set `entity_type="ExtractionChunk"`, `score_type="vector"`. Both consistent with downstream consumers? Specifically the chunk-scope endpoint reads `r.properties["self_ref"]` and `r.score`; do other consumers read `.entity_type` or `.score_type` for routing?

3. **`node_id` cascade** — `row.get("@rid", row.get("vertex_id", row["self_ref"]))`. Will `@rid` always be present in the SQL result from `_client.query`? If not, the fallback path is safe (still produces a non-empty string), but worth a sanity check.

4. **Normalize-twice safety** — Embeddings are already L2-normalized at write time (bge-m3). Defensively normalizing again should be a no-op for normalized vectors but adds a tiny numerical drift (the `+1e-12` safety term). Acceptable, or should we skip the re-normalize when we trust the writer?

### 8.2 Diagnostics semantics

5. **`post_filter_candidate_count`** — I defined this as "rows surviving the threshold, BEFORE the top-N slice." This is what the test `test_direct_filters_by_score_threshold` asserts. Is that the same semantic the HNSW path uses? (HNSW's docstring says "How many results survived the pipeline_run_id post-filter" which is slightly different framing.)

6. **`short_fetch=True` when `desired_top_n > 0` and no rows returned** — Edge case: empty result set with `desired_top_n=5` gets `short_fetch=True`. Is that the right signal for callers? The HNSW path doesn't fire `short_fetch` on a pure-empty result (it only triggers when the retry path runs out). Consistent or worth aligning?

### 8.3 Test coverage

7. **Mocking pattern** — I mock `store._client.query` to return a list of dicts directly. The real ArcadeDB driver may return a different shape (e.g., `{"result": [...]}` wrapper). Worth confirming via a quick integration test? Or is the mock-shape assumption fine because the unit test scope is "given dicts, does direct path produce the right output"?

8. **Concurrency under monkeypatch** — `test_dispatcher_respects_cleared_settings_cache` chains 3 dispatcher calls with env mutations between them. Pytest-asyncio shouldn't interleave but worth a sanity check that the test's intent is captured (the test passed; this is about clarity).

### 8.4 Production readiness

9. **Backwards-compatibility** — The dispatcher routes by setting; default is `hnsw`. The chunk-scope endpoint and the integration starvation test are migrated. Is there any OTHER consumer of `search_extraction_chunks` in the codebase I missed? `grep` shows only `app/api/v1/extraction_routing.py:275` and the test files I touched.

10. **Performance at the boundary** — At ~300 chunks the direct path is trivially fast. What's a sensible chunk-count ceiling where we should bail to HNSW? My estimate: 100K+ per run. Worth gating in code with a fall-back, or just documenting?

---

## 9. Pre-A/B checklist (for after review approves)

- [ ] All Codex findings from this review addressed
- [ ] `git log --oneline -3` confirms commit `be29e02` is at HEAD or HEAD~1
- [ ] In `.env`: `VECTOR_ROUTER_RETRIEVAL_MODE=hnsw` (current state — no env flip yet)
- [ ] Worker + worker-graph containers know the new setting:
   ```bash
   docker exec eip-mmdpp-worker-1 env | grep VECTOR_ROUTER_RETRIEVAL_MODE
   # → VECTOR_ROUTER_RETRIEVAL_MODE=hnsw
   ```
- [ ] Worker import resolves the new functions:
   ```bash
   docker exec eip-mmdpp-worker-1 python -c "
   from app.services.extraction_chunk_search import (
       search_extraction_chunks_hnsw, search_extraction_chunks_direct
   )"
   ```
- [ ] If any of the above fails: `COMPOSE_PROJECT_NAME=eip-mmdpp docker compose up -d --force-recreate worker worker-graph api docling-graph` (force-recreate is required for env to propagate per memory `feedback_compose_env_force_recreate`)

---

## 10. A/B procedure (planned, NOT executed)

Once review approves:

1. **Commit the default flip** as a separate commit:
   ```python
   # app/config.py
   vector_router_retrieval_mode: Literal["hnsw", "direct"] = "direct"  # was "hnsw"
   ```
   Plus `.env` and `.env.example`: `VECTOR_ROUTER_RETRIEVAL_MODE=direct`.

2. **Force-recreate worker + worker-graph + api + docling-graph**.

3. **Trigger Dvina + SA-2 via the existing watcher pattern** (template: `/tmp/c7f_dvina_then_sa2.sh`). Name this C.7g.

4. **Acceptance gates** (already in the original plan, repeated for reviewer convenience):
   - **R1**: `pipeline_pass_outputs.diagnostics_json->'router'->>'filter_strategy' == "direct_cosine"` on all narrowed passes (radar_power_rf, missile_kinematics).
   - **R3**: SA-2 `radar_power_rf` candidate_count ≥ 185 (HNSW's starved C.7f number).
   - **R4**: `vector_search_ms` per pass < C.7f's value.
   - **R5**: `rerank_ms` per pass ≤ 1.5× C.7f's value.
   - **Q1**: per-pass entity counts within ±10% of C.7f.
   - **Q2**: ArcadeDB ASSOCIATED_WITH + VARIANT_OF + CUES committed edges ≥ 52 since C.7g start.
   - **W1**: SA-2 wall ≤ 1.1× C.7f's 233.5m.
   - **W2**: Dvina wall ≤ 1.1× C.7f's 14.7m.

5. **If all gates pass** → flip is permanent. Path B task #61 marked complete. Advance to C.8 (Task #48 — Two-doc PROMOTABLE GATE).

6. **If any gate fails** → revert the default-flip commit (`git revert HEAD`) leaves the implementation in place but reverts to `mode=hnsw`. Investigate before re-attempting.

---

## 11. Reviewer prompt (paste to Codex)

> Review the Path B implementation at commit `be29e02` on branch
> `walltime/c0-telemetry`. The handoff at
> `docs/handoffs/2026-05-26-path-b-implementation-review.md` describes
> what was built (not what was planned — that's the sibling doc
> `2026-05-26-path-b-direct-cosine-retrieval.md`).
>
> Verify against §8 "What I want you to verify" — 10 specific questions.
> Focus on:
> - SQL accessor pattern matches repo convention (§8.1.1)
> - `GraphEntityResult` field choices are downstream-safe (§8.1.2)
> - Diagnostics semantics match HNSW path where they should (§8.2.5-6)
> - No missed callers of `search_extraction_chunks` (§8.4.9)
>
> Report concise findings (<400 words), one per question if needed.
