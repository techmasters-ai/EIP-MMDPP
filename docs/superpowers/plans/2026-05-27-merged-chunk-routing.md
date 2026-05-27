# Merged-Chunk Routing Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the retrieval/rerank/LLM units. Top-k scoring runs against the same merged HybridChunker chunks that the LLM consumes — not against per-element fragments.

**Architecture (post-implementation):**
- Phase 1: `build_extraction_index` populates `ExtractionChunk` rows from real `HybridChunker.chunk(...)` output. One row = one merged chunk. Vector router scores merged chunks. `apply_chunk_scope` receives expanded constituent refs; docling-graph still rechunks downstream.
- Phase 2: docling-graph accepts pre-selected chunk texts directly (new request shape); narrowed-mode passes skip downstream rechunking. Retrieval unit = rerank unit = LLM input unit, byte-for-byte.

**Tech stack:** docling `HybridChunker` (`docling.chunking.HybridChunker`, already in `worker-1` env), bge-m3 embeddings, bge-reranker-v2-m3 cross-encoder, ArcadeDB `ExtractionChunk` vertex, FastAPI on docling-graph.

---

## Why this plan exists

Current state — observable on Dvina (`d23fa85d`) and SA-2 (`1862d234`, stopped):
- Layer-1 + Layer-2 filter fixes (Options E + G) successfully retain more doc content in the index pool, but **`sel_refs` is unchanged** between baseline and post-filter runs because the router scores **per-element** chunks and small fragments don't outscore the existing top-K.
- Merged HybridChunker chunks (which the LLM actually consumes) live downstream of `apply_chunk_scope` — the router never sees them during scoring.
- Outcome: filter fixes can't materially improve narrowed-pass recall while the retrieval granularity remains misaligned with the consumer granularity.

The cleanest fix is to make HybridChunker the source of truth for the index, not the post-chunk-scope LLM batcher. This plan does that.

---

## Pre-flight verification

Before starting Phase 1 implementation, confirm:

- [ ] **HybridChunker order is deterministic** on identical input. Run `HybridChunker.chunk(dl_doc)` twice on the Dvina doc; assert chunk count + `meta.doc_items[*].self_ref` tuples match.

- [ ] **Exact config reuse**. Read `docling_graph/core/extractors/document_chunker.py` and capture the exact constructor parameters in use today:
  - tokenizer (`AutoTokenizer.from_pretrained(model_name=...)` wrapped via `HuggingFaceTokenizer(tokenizer=tok, max_tokens=512)` — the repo pattern; do NOT invent `HuggingFaceTokenizer.from_pretrained` since it doesn't exist)
  - `max_tokens=512`
  - `merge_peers=True`
  - `repeat_table_header=True`
  - `omit_header_on_overflow=False`
  - `always_emit_headings=False`

  These become the canonical config for the shared helper introduced in Task 2a. **Phase 1 keeps `max_tokens=512`**. Do not lower up-front to placate the reranker concern — that would change the chunk unit to solve a downstream problem, which conflicts with the goal.

- [ ] **bge-reranker-v2-m3 truncation behavior** on 512-token-pair input. Score a 200-token merged chunk vs a 512-token merged chunk against the kinematics query; capture both scores. Cross-encoder typically right-truncates — long body text may be silently dropped.

  **Decision tree if truncation is harmful** (i.e. A/B shows missed chunks correlated with body-right being truncated):
  1. Keep 512 and accept/measure the truncation impact
  2. Move both router index AND docling-graph to a lower token budget (joint change so byte identity holds)
  3. Use a reranker model with longer context
  4. Rerank with a derived preview (e.g. first N tokens) but still select/store the full HybridChunker chunk for downstream

  **Default**: option 1 (keep 512 for Phase 1) unless measurement forces a change.

- [ ] **Docling chunk has `meta.doc_items` provenance**. Confirm `chunk.meta.doc_items` exists and each item has a `self_ref` attribute resolvable to `texts[i]/tables[i]/pictures[i]`.

- [ ] **Doc-shape parity**. The worker indexes `doc_json` as-loaded-from-MinIO (post-Layer-1 filter). docling-graph receives the same shape via its extraction request. Confirm both services consume **identical doc_json shape** before the chunker runs — if either side enriches/sanitizes the doc differently (sanitizer in docling-graph's `main.py:482` or worker's `filter_docling_document`), the chunker outputs will diverge and "exact HybridChunker" no longer holds.

---

## Chunk 1: Phase 1 — Real HybridChunker chunks in the index

Replace per-element `_walk_docling_elements` with real HybridChunker output via a shared helper. Router scores merged chunks; `apply_chunk_scope` semantics preserved by expanding to constituent refs.

**Phase 1 fixes selection granularity.** It does NOT yet give byte identity between router-selected chunks and LLM input — that's Phase 2. The downstream rechunk in docling-graph still happens after `apply_chunk_scope`, so merged-chunk boundaries may shift slightly when neighbors are removed from the scoped doc. Acceptable for Phase 1; closed by Phase 2.

### Task 1: Add fields to `ExtractionChunk` vertex schema

**Files:**
- Modify: `app/services/graph_store.py` (or wherever `ExtractionChunk` schema is declared)
- Modify: corresponding ArcadeDB schema migration
- Test: `tests/integration/test_extraction_chunk_schema.py`

New columns on `ExtractionChunk`:
- `chunk_index: int` — position of this chunk in HybridChunker output for a given pipeline_run_id; used for deterministic vertex IDs.
- `source_refs` — element self_refs that contributed to this merged chunk (e.g. `["#/texts/35", "#/texts/36", ..., "#/texts/51"]`). **Verify ArcadeDB list-property support** for the existing `ExtractionChunk` schema. If ArcadeDB list properties are awkward (per prior bug history, e.g. `system_links` typed-edge issues), store as a JSON-encoded string column `source_refs_json` plus a thin helper that lazily decodes — but the router code must NOT do ad hoc parsing scattered across endpoints; encapsulate access in `app/services/extraction_chunk_index.py:read_chunk_source_refs(row) -> list[str]`.
- `token_count: int` — diagnostics field; output of `tokenizer.count_tokens(text)`. Used by Task 6 calibration sweep and Phase 2 diagnostics.

Vertex id changes from `pipeline_run_id:<element_self_ref>` to `pipeline_run_id:chunk_<chunk_index>`. Stable hash alternative (`hash(tuple(source_refs))`) is rejected because hash-collision handling adds complexity for no gain — `chunk_index` is already deterministic if HybridChunker is (pre-flight verifies this).

- [ ] **Step 1: Write the failing test**

```python
def test_extraction_chunk_has_chunk_index_and_source_refs():
    store = GraphStore(...)
    store.insert_extraction_chunk(
        pipeline_run_id="r1", chunk_index=3, source_refs=["#/texts/35", "#/texts/36"],
        text="...", embedding=[0.0]*1024, document_id="d1", page_no="2",
        token_count=312,
    )
    row = store.read_extraction_chunk(pipeline_run_id="r1", chunk_index=3)
    assert row.chunk_index == 3
    assert read_chunk_source_refs(row) == ["#/texts/35", "#/texts/36"]
    assert row.token_count == 312
```

- [ ] **Step 2: Add the columns** to the vertex schema with default values (`chunk_index = -1`, `source_refs = []` or `source_refs_json = "[]"`, `token_count = 0`) so existing per-element rows remain queryable during rollout.

- [ ] **Step 3: Run integration test**. Expected: PASS.

- [ ] **Step 4: Commit** `feat(extraction-chunk): add chunk_index + source_refs + token_count columns`.

### Task 2a: Shared helper — `build_hybrid_chunks_for_extraction`

**Files:**
- Create: `app/services/hybrid_chunking.py`
- Test: `tests/unit/test_hybrid_chunking.py`

Avoid copy-pasting `DocumentChunker` config between the indexer and any test/preview harness. Centralize it in one helper that mirrors `docling_graph/core/extractors/document_chunker.py` exactly:

```python
# app/services/hybrid_chunking.py
from dataclasses import dataclass
from transformers import AutoTokenizer  # already a transitive dep via docling
from docling_core.types.doc import DoclingDocument
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


@dataclass(frozen=True)
class HybridChunkConfig:
    """Canonical HybridChunker config — mirror docling-graph's DocumentChunker."""
    tokenizer_model_name: str = "BAAI/bge-m3"  # match retrieval embedder
    max_tokens: int = 512
    merge_peers: bool = True
    repeat_table_header: bool = True
    omit_header_on_overflow: bool = False
    always_emit_headings: bool = False


@dataclass(frozen=True)
class HybridExtractionChunk:
    chunk_index: int
    text: str                  # output of chunker.contextualize(chunk)
    source_refs: list[str]     # [item.self_ref for item in chunk.meta.doc_items]
    page_no: str | None        # first prov page if resolvable
    token_count: int           # tokenizer.count_tokens(text) — for diagnostics


def build_hybrid_chunks_for_extraction(
    doc_json: dict,
    config: HybridChunkConfig | None = None,
) -> list[HybridExtractionChunk]:
    """Run HybridChunker against doc_json using the canonical config.

    NOTE: doc_json must be the same shape docling-graph receives — apply
    the same Layer-1 worker-side filter (filter_docling_document) BEFORE
    calling this. Doc-shape parity is asserted by the caller, not here.
    """
    cfg = config or HybridChunkConfig()
    raw_tok = AutoTokenizer.from_pretrained(cfg.tokenizer_model_name)
    tokenizer = HuggingFaceTokenizer(tokenizer=raw_tok, max_tokens=cfg.max_tokens)
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=cfg.merge_peers,
        # NB: HybridChunker takes its token budget from the tokenizer's
        # max_tokens, NOT a chunk_max_tokens kwarg. See docling source.
    )

    dl_doc = DoclingDocument.model_validate(doc_json)
    out: list[HybridExtractionChunk] = []
    for idx, chunk in enumerate(chunker.chunk(dl_doc=dl_doc)):
        text = chunker.contextualize(chunk=chunk)
        source_refs = [item.self_ref for item in chunk.meta.doc_items]
        page_no = _resolve_first_page_no(chunk)
        token_count = tokenizer.count_tokens(text=text)
        out.append(HybridExtractionChunk(
            chunk_index=idx,
            text=text,
            source_refs=source_refs,
            page_no=page_no,
            token_count=token_count,
        ))
    return out
```

- [ ] **Step 1: Write failing tests** that pin the canonical config (counts merged chunks on a Dvina fixture; asserts deterministic order across two calls).
- [ ] **Step 2: Implement** + verify constructor invocations match docling-graph (no `from_pretrained` on HuggingFaceTokenizer; no `chunk_max_tokens` kwarg on HybridChunker).
- [ ] **Step 3: Run tests**.
- [ ] **Step 4: Commit** `feat(hybrid-chunking): shared chunker helper mirroring docling-graph config`.

### Task 2b: New indexer — `build_extraction_index_hybrid`

**Files:**
- Modify: `app/services/extraction_chunk_index.py`
- Test: `tests/unit/test_extraction_chunk_index_hybrid.py`

Add a new function alongside the existing one (do NOT delete `build_extraction_index` yet; feature-flagged rollout).

```python
def build_extraction_index_hybrid(
    doc_json: dict,
    pipeline_run_id: str,
    document_id: str,
    store: GraphStore,
) -> BuildIndexDiagnostics:
    """Index merged chunks via the shared HybridChunker helper.

    One ExtractionChunk per merged chunk. Each row stores:
      - chunk_index, source_refs, page_no, text (contextualized merged text)
      - embedding: bge-m3 of text

    Vertex id: f"{pipeline_run_id}:chunk_{chunk_index}"
    """
    from app.services.hybrid_chunking import build_hybrid_chunks_for_extraction

    chunks = build_hybrid_chunks_for_extraction(doc_json)
    embeddings = embed_texts([c.text for c in chunks])
    for c, emb in zip(chunks, embeddings):
        store.insert_extraction_chunk(
            pipeline_run_id=pipeline_run_id,
            document_id=document_id,
            chunk_index=c.chunk_index,
            source_refs=c.source_refs,
            text=c.text,
            embedding=emb,
            page_no=c.page_no,
            token_count=c.token_count,
        )
    return BuildIndexDiagnostics(
        chunk_count=len(chunks),
        mean_tokens=mean([c.token_count for c in chunks]) if chunks else 0,
        ...
    )
```

- [ ] **Step 1: Write failing tests** covering:
  - merged chunk count matches the shared helper's output
  - each row's `source_refs` non-empty and references real `texts[]/tables[]/pictures[]` indices
  - `text` includes the heading prefix (contains the parent section heading string)
  - deterministic chunk_index across two runs on same doc

- [ ] **Step 2: Implement** above.

- [ ] **Step 3: Run tests**.

- [ ] **Step 4: Commit** `feat(extraction-chunk-index): build_extraction_index_hybrid for merged-chunk routing`.

### Task 3: Feature-flag the indexer choice in worker

**Files:**
- Modify: `app/workers/pipeline.py:~8617` (the index-build site)
- Modify: `.env.example` + `.env`
- Test: `tests/unit/test_pipeline_index_flag.py`

```python
# In derive_ontology_graph:
USE_MERGED_INDEX = os.getenv("EXTRACTION_INDEX_MODE", "per_element") == "merged"
if USE_MERGED_INDEX:
    from app.services.extraction_chunk_index import build_extraction_index_hybrid
    diag = build_extraction_index_hybrid(doc_json_for_index, run_id, doc_id, store)
else:
    diag = build_extraction_index(doc_json_for_index, run_id, doc_id, store)
```

- [ ] **Step 1: Write failing test** that asserts env var routing works.
- [ ] **Step 2: Wire** env var + branching.
- [ ] **Step 3: Add env var** to `.env` (default `per_element`) and `.env.example`.
- [ ] **Step 4: Commit** `feat(worker): EXTRACTION_INDEX_MODE flag for merged vs per-element indexer`.

### Task 4: Chunk-scope endpoint expands `source_refs`

**Files:**
- Modify: `app/api/v1/extraction_routing.py:~168` (`chunk_scope` endpoint)
- Modify: `app/services/extraction_chunk_index.py` (read helper for `source_refs`)
- Test: `tests/integration/test_chunk_scope_endpoint_merged.py`

Today the endpoint returns selected self_refs via `selected_refs`. In merged mode it must expand each selected merged chunk's `source_refs` into that same list. **Preserve the existing response contract** — same field name `selected_refs`, same `mode` value semantics. Only ADD optional fields.

```python
# After router selects top-K merged-chunk rows (gated on EXTRACTION_INDEX_MODE=merged):
expanded_refs: list[str] = []
seen: set[str] = set()
# IMPORTANT: ordering. Don't sort lexicographically — '#/texts/100' would
# precede '#/texts/35'. apply_chunk_scope currently re-orders by body walk
# (per the C.9c-blocker fix), but make the order intent explicit at the
# endpoint level too. Preserve chunk-encounter order so that selected refs
# are grouped by their merged chunk's position in HybridChunker output.
for chunk_row in selected:    # iteration order = bge-m3+reranker top-K order
    for ref in read_chunk_source_refs(chunk_row):
        if ref not in seen:
            seen.add(ref)
            expanded_refs.append(ref)

return ChunkScopeResponse(
    mode=existing_mode_value,                  # do NOT change ("full" / "narrowed" semantics unchanged)
    selected_refs=expanded_refs,               # existing field, populated from expansion
    selected_chunk_ids=[c.chunk_id for c in selected],            # NEW (optional)
    selected_chunks=[{"chunk_index": c.chunk_index, "token_count": c.token_count} for c in selected],  # NEW (optional, used by Phase 2)
    ...
)
```

- [ ] **Step 1: Write failing test** with a fixture pipeline_run that has merged chunks; query returns expanded `selected_refs` in chunk-encounter order, plus the new optional fields.
- [ ] **Step 2: Implement** expansion (gated on `EXTRACTION_INDEX_MODE=merged`).
- [ ] **Step 3: Snapshot test** the response shape for both modes — must be backward-compatible (no field deletions, no semantic changes to existing fields).
- [ ] **Step 4: Confirm** `apply_chunk_scope` still produces a correctly-ordered scoped doc when given the chunk-encounter-ordered ref list. If it does NOT reorder by body walk, the doc will be out-of-order; either keep apply_chunk_scope's reordering OR have the endpoint sort by `parse_ref_index(ref)`.
- [ ] **Step 5: Commit** `feat(chunk-scope): expand source_refs when indexer is in merged mode`.

### Task 5: Janitor + cleanup for new chunk_index keys

**Files:**
- Modify: `app/services/extraction_chunk_index.py:cleanup_extraction_index`
- Modify: Beat schedule entry (`#65`)
- Test: `tests/integration/test_extraction_chunk_cleanup_merged.py`

Verify the existing janitor `purge_terminated_extraction_chunks` correctly removes merged-chunk rows (the vertex-id format change must not break the DELETE WHERE clause).

- [ ] **Step 1: Run janitor against a synthetic pipeline_run with merged chunks**; assert all removed.
- [ ] **Step 2: Fix any vertex-id-format assumptions** in the janitor SQL.
- [ ] **Step 3: Commit** `fix(janitor): handle merged-chunk vertex ids`.

### Task 6: Phase 1 calibration sweep

**Files:**
- New: `notebooks/c10-phase1-merged-chunk-calibration.ipynb` (or equivalent script)
- Output: `docs/handoffs/2026-05-27-phase1-merged-chunk-sweep.md`

Re-run the C.9a-style offline retrieval sweep on merged chunks. The previous bundle `air_defense_v3_narrowing_v1` was tuned for per-element scoring; thresholds need re-calibration.

Sweep dimensions:
- `min_similarity`: 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
- `top_n_candidates`: 25, 50, 75, 100
- `top_k`: 5, 10, 15, 20, 30

`top_n_candidates > top_k` is preserved on purpose: even with merged chunks, the reranker (bge-reranker-v2-m3, cross-encoder) can still improve ordering of the candidate pool before final top_k selection. The candidate set being larger than the final pick is what gives the reranker something to reorder.

Output: per-pass `min_similarity` / `top_n_candidates` / `top_k` values for the new `air_defense_v3_merged_v1` bundle.

Diagnostics to capture per (min_sim, top_n, top_k) cell:
- ground-truth coverage at top_k (GT entities retrievable)
- `selected_chunk_count`
- `expanded_ref_count` (after `source_refs` union; useful to predict apply_chunk_scope output size)
- `selected_chunk_token_estimate` (sum of merged chunk token_counts)

- [ ] **Step 1: Run the sweep**.
- [ ] **Step 2: Pick the knee** for each narrowed pass (`missile_kinematics`, `radar_power_rf`, `radar_antenna`, `radar_timing`, `radar_modulation`, `missile_airframe`, `missile_propulsion`, `missile_speed_timing`, `missile_guidance`).
- [ ] **Step 3: Write handoff doc** with chosen values.

### Task 7: Phase 1 A/B against C.10 baseline

**Files:**
- Bundle: `ontology_bundles/air_defense_v3_merged_v1/manifest.yaml` — calibrated values from Task 6.

- [ ] **Step 1: Trigger Dvina graph_only with `EXTRACTION_INDEX_MODE=merged`** + `air_defense_v3_merged_v1` bundle.
- [ ] **Step 2: Trigger SA-2 graph_only** with same.
- [ ] **Step 3: Compare against C.10 baselines** (`cfcc9539` Dvina, `7d46c487` SA-2). Required gains:
  - Narrowed-pass recall: kinematics + at least one other narrowed pass shows ≥+50% entity count
  - Wall time: ≤120% of baseline (Phase 1 alone is allowed to be slightly slower; Phase 2 closes the gap)
- [ ] **Step 4: Promote `EXTRACTION_INDEX_MODE=merged`** as default if gates pass.

### Phase 1 gate

After Task 7, **discuss with user** (per [[feedback-phase-discussion-before-implementation]]):
- A/B results
- Whether to proceed to Phase 2 or call it done

---

## Chunk 2: Phase 2 — Direct selected-chunk feed (byte-level identity)

Eliminate docling-graph's downstream re-chunking for narrowed passes. The chunk text that the LLM sees is byte-equal to the row stored in `ExtractionChunk` — the worker reads it from the vertex and forwards it; no second HybridChunker invocation anywhere.

### Task 8: New docling-graph endpoint accepting pre-chunked input

**Files:**
- Modify: docling-graph `/extract-pass` endpoint (or add `/extract-pass-chunked`)
- Modify: docling-graph request schema (`docker/docling-graph/app/schemas.py`)
- Test: `tests/integration/test_extract_pass_chunked.py`

New request shape:
```json
{
  "pass_name": "missile_kinematics",
  "bundle_key": "air_defense_v3_merged_v1",
  "selected_chunks": [
    {"chunk_index": 12, "text": "## SECTION\n\nLength:\n35 feet\n...", "source_refs": ["#/texts/35", "..."]},
    {"chunk_index": 28, "text": "...", "source_refs": [...]},
    ...
  ],
  "upstream_refs": [...],
  ...
}
```

Behavior: skip the `DocumentChunker` step in `pipeline/stages.py`; iterate `selected_chunks` directly as LLM batches. Preserve `source_refs` into extracted-fact `evidence_units`.

- [ ] **Step 1: Write failing integration test** that POSTs to `/extract-pass-chunked` with synthetic merged chunks.
- [ ] **Step 2: Implement** branch in docling-graph extraction pipeline.
- [ ] **Step 3: Provenance check** — extracted entity/fact records cite `source_refs` from the chunk that produced them.
- [ ] **Step 4: Commit** in docling-graph `feat(extract): /extract-pass-chunked accepts pre-selected chunks`.

### Task 9: Worker uses chunked endpoint for narrowed passes — byte-identical chunk text

**Files:**
- Modify: `app/workers/pipeline.py:derive_ontology_graph_pass` (the per-pass dispatcher)
- Test: `tests/unit/test_pipeline_chunked_dispatch.py`

**Critical invariant for byte identity**: the worker MUST pass the **stored chunk text** from `ExtractionChunk.text` (the field populated by the indexer's `chunker.contextualize(chunk)` call), NOT re-run HybridChunker on the worker side. Re-chunking locally would mean two HybridChunker invocations against potentially-divergent doc states, breaking the "Top-K stored chunks = Gemma input" invariant.

Flow:
1. Worker calls `/v1/extraction/chunk-scope` (returns `selected_chunk_ids` + `selected_chunks` per Task 4)
2. Worker reads `text`, `source_refs`, `chunk_index` for each selected chunk_id from the `ExtractionChunk` vertices in ArcadeDB
3. Worker POSTs `{selected_chunks: [{chunk_index, text, source_refs}, ...]}` to `/extract-pass-chunked`
4. docling-graph iterates `selected_chunks` directly as LLM batches (per Task 8). NO `DocumentChunker` invocation.

- [ ] **Step 1: Write failing test** mocking docling-graph: asserts the request body's `selected_chunks[*].text` is byte-equal to the corresponding `ExtractionChunk.text` row.
- [ ] **Step 2: Implement** the read-and-forward path in `derive_ontology_graph_pass` (gated on `EXTRACTION_INDEX_MODE=merged` so per-element mode still uses the old path).
- [ ] **Step 3: Sanity check** that a Dvina run produces extracted entities whose `evidence_units` contain `source_refs` from the merged chunk that produced them (not a docling-graph-locally-rechunked chunk).
- [ ] **Step 4: Commit** `feat(worker): narrowed passes send stored chunk text byte-identically to docling-graph`.

### Task 10: Phase 2 A/B against Phase 1

**Bundle:** `air_defense_v3_merged_v1` (same as Phase 1; only the wire-protocol differs).

- [ ] **Step 1: Dvina graph_only with Phase 1 enabled** (record baseline).
- [ ] **Step 2: Dvina graph_only with Phase 2 enabled** (chunked endpoint).
- [ ] **Step 3: Compare**:
  - Entity counts: should be **identical or higher** (Phase 2 doesn't reduce content, just removes the redundant re-chunk)
  - Wall time: should drop (fewer markdown export + rechunk cycles in docling-graph)
  - Provenance: every extracted entity has `source_refs` populated from a real merged chunk's source_refs

### Phase 2 gate

After Task 10, **discuss with user**:
- A/B results
- Whether to retire the per-element index path entirely (and remove the `EXTRACTION_INDEX_MODE` flag) or keep both paths as configurable

---

## Rollback plan

Each task is feature-flagged, so rollback is trivial:

- `EXTRACTION_INDEX_MODE=per_element` (default until Phase 1 promoted) — falls back to the existing `build_extraction_index` + standard `apply_chunk_scope` + standard `/extract-pass` flow.
- If Phase 1 ships but produces worse recall on a real doc, flip the env var.
- ExtractionChunk schema additions (`chunk_index`, `source_refs`) are backward-compatible (default-valued); old per-element rows continue to function.

If `air_defense_v3_merged_v1` calibration turns out wrong, revert to `air_defense_v3_narrowing_v1` bundle without code changes — just a manifest swap.

---

## Open questions (to resolve before kickoff)

1. **Should `system_links` use merged chunks?** It's non-narrowed today (sees full doc). With merged-mode, do we still skip the index for it, or also index it for completeness? Recommend: same as today, non-narrowed bypasses the index.

2. **Identity passes (`*_identity`)** are non-narrowed. They go through `derive_ontology_graph` (worker-1) not `derive_ontology_graph_pass` (worker-graph). They don't use the index for routing at all today. Phase 1 doesn't change this; identity passes still see the full filtered doc. Phase 2 doesn't change this either. Confirm this is desired.

3. **`evidence_units` schema** in extracted records: does the existing field accept `source_refs: List[str]` or does it require richer provenance (page_no, span)? Need to align with the field-provenance prompt block (per `docs/superpowers/plans/2026-04-25-flat-schema-profile-refactor.md:3285`).

4. **Document deduplication semantics**: Layer-1's per-element dedup (Rule 3) still applies to `texts[]` before chunking. But per-merged-chunk dedup makes no semantic sense — each merged chunk is unique by construction. Verify `build_extraction_index_hybrid` does NOT carry over Layer-2's in-loop dedup (its motivation was per-element duplicates; merged chunks can't duplicate).

5. **ArcadeDB list-property reliability for `source_refs`**: if list-typed properties have known limitations (cf. prior `system_links` typed-edge issues), fall back to JSON-string storage. Decide before Task 1 implementation so the schema migration is one-shot.

6. **Reranker truncation outcome**: gated by pre-flight measurement, not theory. The plan defaults to `max_tokens=512` and the four contingent options listed in the pre-flight decision tree.

---

## Acceptance criteria summary

| Phase | Gate | Pass condition |
|---|---|---|
| Pre-flight | Determinism + config-reuse + reranker truncation check + doc-shape parity | All hold; truncation decision logged with measurement |
| Phase 1, Task 1-5 | Implementation tests | All pass; flag works; response contract preserved |
| Phase 1, Task 6-7 | Calibration + A/B | Recall ≥ baseline + ≥1 narrowed pass +50% ent; wall ≤ 120% baseline; `selected_chunk_token_estimate` recorded |
| Phase 2, Task 8-10 | A/B + byte identity | Recall ≥ Phase 1; wall < Phase 1; LLM batch input byte-equal to `ExtractionChunk.text` for narrowed passes |
| Final | Production rollout | Promote `EXTRACTION_INDEX_MODE=merged`; deprecate per-element path on next stable release |

## Diagnostics expected on every merged-mode pass

Per-pass diagnostics dict (alongside existing `doc_filter`, `router`):

```json
{
  "selected_chunk_count": 12,            // merged chunks returned by the router
  "selected_ref_count": 87,              // expanded source_refs (sum of source_refs across selected chunks)
  "expanded_ref_count": 87,              // alias for selected_ref_count for clarity in merged mode
  "selected_chunk_token_estimate": 4823, // sum of token_counts across selected chunks
  "min_similarity": 0.30,                // value actually used (from manifest)
  "top_n_candidates": 50,
  "top_k": 12,
  "index_mode": "merged"                 // distinguishes from "per_element" baseline runs
}
```

Keep `selected_ref_count` populated so existing dashboards and the `narrowing-ineffective` heuristic (bug #66) continue to work without rewrite.

---

## Time estimate

- Pre-flight: 1h
- Phase 1 (Tasks 1-5 implementation): ~1 day
- Phase 1 (Task 6 calibration + Task 7 A/B): ~1 day (mostly waiting on Dvina + SA-2 runs)
- Phase 2 (Tasks 8-10): ~1-2 days
- Total: **3-5 days focused effort**, plus calibration wall-time which is gated on Dvina/SA-2 runs (~5 hours each).
