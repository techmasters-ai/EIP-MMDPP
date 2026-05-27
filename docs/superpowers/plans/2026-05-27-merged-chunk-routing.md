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

Before starting Phase 1 implementation, confirm and resolve:

### Determinism + provenance

- [ ] **HybridChunker order is deterministic** on identical input. Run `HybridChunker.chunk(dl_doc)` twice on the Dvina doc; assert chunk count + `meta.doc_items[*].self_ref` tuples match. (HybridChunker source review shows no RNG — it walks `body.children` in deterministic order. This pre-flight is a property-level pin, also re-asserted as a unit test in Task 2a.)

- [ ] **Docling chunk has `meta.doc_items` provenance**. Confirm `chunk.meta.doc_items` exists and each item has a `self_ref` attribute resolvable to `texts[i]/tables[i]/pictures[i]`.

### Tokenizer alignment (BLOCKING) — required for Phase 2 byte identity

- [ ] **Pin docling-graph's `chunker_config` tokenizer to `BAAI/bge-m3`** so the worker indexer and docling-graph use the same tokenizer. Today docling-graph's `DocumentChunker` defaults to `"sentence-transformers/all-MiniLM-L6-v2"` (`docker/docling-graph/repo/docling_graph/core/extractors/document_chunker.py:73-74`) and `many_to_one.py:70` constructs `chunker_config = {"chunk_max_tokens": int(chunk_max_tokens or 512)}` with no `tokenizer_name` override. Phase 2 byte identity is impossible without this. One-line fix in `many_to_one.py:70`:

      chunker_config = {
          "chunk_max_tokens": int(chunk_max_tokens or 512),
          "tokenizer_name": "BAAI/bge-m3",
      }

  Ship this fix as a pre-Phase-1 prerequisite. The worker indexer and docling-graph then share both tokenizer AND chunker config.

- [ ] **Confirm production tokenizer model name** by reading `docker/docling-graph/app/config_builder.py:293-319` for any override of `tokenizer_name` in production manifests. If a non-bge-m3 override exists, decide explicitly: change the canonical config OR change production.

### Exact config reuse (corrected wording)

- [ ] **Capture exact `DocumentChunker` constructor params** at `docker/docling-graph/repo/docling_graph/core/extractors/document_chunker.py:60-127`:
  - tokenizer: `AutoTokenizer.from_pretrained(model_name=...)` wrapped via `HuggingFaceTokenizer(tokenizer=tok, max_tokens=512)`. NOTE: `HuggingFaceTokenizer.from_pretrained` DOES exist as a classmethod, but the existing code uses the wrap pattern specifically so it can also call `_raise_tokenizer_max_length(...)` (`document_chunker.py:30-46`) which suppresses HF "Token indices sequence length is longer than..." warnings on oversize chunks. The shared helper MUST also call `_raise_tokenizer_max_length` for parity.
  - `max_tokens=512`
  - `merge_peers=True`
  - `repeat_table_header=True`
  - `omit_header_on_overflow=False`
  - `always_emit_headings=False`

  These become the canonical config for the shared helper (Task 2a). **Phase 1 keeps `max_tokens=512`**.

- [ ] **`HybridChunker` token budget**: confirmed via source review — `HybridChunker.__init__` takes `tokenizer` and `merge_peers` (no `chunk_max_tokens` kwarg); the token budget is read from `tokenizer.get_max_tokens()`. The plan's helper must pass the budget through the tokenizer, not as a chunker kwarg.

### Sanitizer alignment (BLOCKING) — required for Phase 2 byte identity

- [ ] **Worker `filter_docling_document` vs docling-graph `_sanitize_docling_document` are NOT byte-equivalent.** Worker filter uses `chunk_quality.classify_chunk` (Layer-1 short/dedup/after_strip predicates). docling-graph sanitizer at `docker/docling-graph/app/main.py:472-514` uses a separate `_looks_like_nav_or_tracking` predicate to blank `text`/`orig`/`hyperlink` of nav/tracking elements. Even with identical tokenizers, the resulting `texts[]` arrays differ → HybridChunker output diverges.

  **Resolution (single source of truth on worker side):**
  - Add a code path to docling-graph that skips `_sanitize_docling_document` when the new `selected_chunks` field is provided (Phase 2 Task 8). Confirm via reading `main.py:638-657` whether sanitize is in the skip-able block.
  - Add an integration test in Phase 1 Task 2a that asserts: `build_hybrid_chunks_for_extraction(filtered_doc_json) == DocumentChunker.chunk_document(filtered_doc_json)` for a fixture Dvina doc. Without this test, byte identity is unverifiable.

### Embedder context budget (Ollama bge-m3)

- [ ] **Verify Ollama bge-m3 `num_ctx`** is ≥1024. Default may be 512, which silently truncates a 512-tokenized merged chunk after the BGE retrieval prefix `"Represent this sentence: "` is prepended. Check `.env*` and `app/services/embedding.py` for the prefix logic + any `num_ctx` override. If default is 512, set per-host or in the embedder config to ≥1024.

### Reranker truncation

- [ ] **bge-reranker-v2-m3 truncation behavior** on 512-token-pair input. Score a 200-token merged chunk vs a 512-token merged chunk against the kinematics query; capture both scores. Cross-encoder typically right-truncates — long body text may be silently dropped.

  **Decision tree if truncation is harmful** (i.e. A/B shows missed chunks correlated with body-right being truncated):
  1. Keep 512 and accept/measure the truncation impact
  2. Move both router index AND docling-graph to a lower token budget (joint change so byte identity holds)
  3. Use a reranker model with longer context
  4. Rerank with a derived preview (e.g. first N tokens) but still select/store the full HybridChunker chunk for downstream

  **Default**: option 1 (keep 512 for Phase 1) unless measurement forces a change.

### Hard prerequisites — must be landed before Phase 1 Task 4

- [ ] **TaskList #53-56 (apply_chunk_scope cleanups)** must be merged. Task 4's expansion ordering relies on `apply_chunk_scope` reordering by body walk (`scoped_docling_document.py:178-220`). If cleanups #53/#54 change this ordering or the retained-group scope, Task 4's expanded `self_refs` list semantics shift. Verify these are landed before starting Task 4.

- [ ] **Verify TaskList #65 status**. `app/workers/celery_app.py:107-114` already has `vr-purge-terminated-extraction-chunks` in the beat schedule. #65 may be stale. Confirm by checking `git log -- app/workers/celery_app.py`. If actually done, mark #65 complete; if not, schedule its fix before merged-mode rollout so stale per-element rows don't survive the cross-mode transition.

- [ ] **TaskList #64 (inline cleanup never fires on COMPLETE)** must be fixed before A/B. Otherwise stale per-element rows from baseline runs will pollute the index pool until the periodic janitor sweeps them 24h later.

### A/B environment lockdown (BLOCKING)

- [ ] **Pin `VECTOR_ROUTER_RETRIEVAL_MODE=direct`** (Path B, TaskList #61, already implemented at `app/services/extraction_chunk_search.py:265-436`) for the merged-mode A/B (Task 7 + Task 10). Otherwise HNSW post-filter starvation (`app/services/extraction_chunk_search.py:5-26`) confounds the recall measurement because HNSW returns globally top-K then post-filters by `pipeline_run_id` — `min_similarity` becomes a no-op and selection is non-deterministic. Document this in Task 7 + Task 10 environment specs.

---

## Chunk 1: Phase 1 — Real HybridChunker chunks in the index

Replace per-element `_walk_docling_elements` with real HybridChunker output via a shared helper. Router scores merged chunks; `apply_chunk_scope` semantics preserved by expanding to constituent refs.

**Phase 1 fixes selection granularity.** It does NOT yet give byte identity between router-selected chunks and LLM input — that's Phase 2. The downstream rechunk in docling-graph still happens after `apply_chunk_scope`, so merged-chunk boundaries may shift slightly when neighbors are removed from the scoped doc. Acceptable for Phase 1; closed by Phase 2.

### Task 1: Add fields to `ExtractionChunk` vertex schema

**Files (verified):**
- Modify: `app/services/arcadedb_schema.py:38` — vertex schema declaration (`ExtractionChunk` block)
- Modify: `app/services/extraction_chunk_index.py:749-760` — inline INSERT SQL string that hard-codes `f"{pipeline_run_id}:{self_ref}"` vertex id format. Replace with `f"{pipeline_run_id}:chunk_{chunk_index}"` only in merged-mode path (Task 3 introduces the flag).
- Modify: `app/services/extraction_chunk_index.py` — add the `read_chunk_source_refs(row) -> list[str]` accessor helper (signature below)
- Test: `tests/integration/test_extraction_chunk_schema.py`

New columns on `ExtractionChunk`:
- `chunk_index: int` — position of this chunk in HybridChunker output for a given pipeline_run_id; used for deterministic vertex IDs. Default `-1` for legacy per-element rows.
- `source_refs` — element self_refs that contributed to this merged chunk. **Storage decision deferred to pre-flight** (Task 0). `CommunityReport` at `arcadedb_schema.py:91,94` already uses ArcadeDB `LIST` type successfully, so committing to `LIST` from day one is recommended. If list-typing fails for any reason, fall back to JSON-encoded string `source_refs_json`. Either way the row interface is the same accessor.
- `token_count: int` — diagnostics field; output of `tokenizer.count_tokens(text)`. Default `0` for legacy rows.

**Accessor helper (canonical signature, single source of truth):**

```python
# app/services/extraction_chunk_index.py
def read_chunk_source_refs(row: dict | object) -> list[str]:
    """Return source_refs as a list[str] regardless of underlying storage.

    Handles both:
      - native ArcadeDB LIST property (returns row["source_refs"] directly)
      - JSON-encoded string fallback (row["source_refs_json"] → json.loads)
      - legacy per-element rows (returns [] since they have no merged source_refs)

    NEVER returns None. Empty list means "no constituent refs known" — caller
    decides whether that's an error or a legacy row.
    """
```

This helper is the ONLY place that knows about underlying storage shape. Endpoint code, worker code, and test code all call it.

Vertex id format changes from `pipeline_run_id:<element_self_ref>` to `pipeline_run_id:chunk_<chunk_index>`. The stable-hash alternative (`hash(tuple(source_refs))`) is rejected — `chunk_index` is already deterministic given HybridChunker determinism (pre-flight verifies this), and hash-collision handling adds complexity for no gain.

- [ ] **Step 1: Write the failing test**

```python
def test_extraction_chunk_has_chunk_index_source_refs_token_count():
    store = GraphStore(...)
    store.insert_extraction_chunk(
        pipeline_run_id="r1", chunk_index=3,
        source_refs=["#/texts/35", "#/texts/36"],
        text="...", embedding=[0.0]*1024, document_id="d1", page_no="2",
        token_count=312,
    )
    row = store.read_extraction_chunk(pipeline_run_id="r1", chunk_index=3)
    assert row["chunk_index"] == 3
    assert read_chunk_source_refs(row) == ["#/texts/35", "#/texts/36"]
    assert row["token_count"] == 312


def test_read_chunk_source_refs_handles_legacy_rows():
    # Legacy per-element row (chunk_index=-1, source_refs absent)
    legacy = {"chunk_index": -1, "self_ref": "#/texts/0"}
    assert read_chunk_source_refs(legacy) == []
```

- [ ] **Step 2: Add the columns** to `arcadedb_schema.py:38` block with defaults. Add a property migration (ALTER TYPE if ArcadeDB needs it) so existing per-element rows remain queryable.

- [ ] **Step 3: Update inline INSERT SQL** at `extraction_chunk_index.py:749-760` (per-element path still uses the existing vertex_id format; merged-mode path uses the new format — branched by Task 3 flag).

- [ ] **Step 4: Implement `read_chunk_source_refs` accessor** with the storage-agnostic semantics above.

- [ ] **Step 5: Run integration test**. Expected: PASS.

- [ ] **Step 6: Commit** `feat(extraction-chunk): add chunk_index + source_refs + token_count columns + read_chunk_source_refs accessor`.

### Task 2a: Shared helper — `build_hybrid_chunks_for_extraction`

**Files:**
- Create: `app/services/hybrid_chunking.py`
- Test: `tests/unit/test_hybrid_chunking.py`
- Test: `tests/integration/test_hybrid_chunker_parity.py` (doc-shape parity with docling-graph — see Pre-flight)

Avoid copy-pasting `DocumentChunker` config between the worker indexer and docling-graph. Centralize it in one helper that mirrors `docker/docling-graph/repo/docling_graph/core/extractors/document_chunker.py:60-127` exactly. The canonical helper also `_raises_tokenizer_max_length` (mirrors the same function at `document_chunker.py:30-46`) to suppress HF oversize-warning noise.

```python
# app/services/hybrid_chunking.py
from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from docling_core.types.doc import DoclingDocument
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


_TOKENIZER_COUNTING_MAX_LENGTH = 8192  # mirror docling-graph's value


def _raise_tokenizer_max_length(
    hf_tokenizer: PreTrainedTokenizerBase, chunk_max_tokens: int
) -> None:
    """Suppress HF 'Token indices sequence length is longer than...' warnings
    on oversize chunks. Mirrors docling-graph's helper exactly.

    See docker/docling-graph/repo/docling_graph/core/extractors/document_chunker.py:30-46.
    """
    new_max = max(chunk_max_tokens, _TOKENIZER_COUNTING_MAX_LENGTH)
    hf_tokenizer.model_max_length = new_max


@dataclass(frozen=True)
class HybridChunkConfig:
    """Canonical HybridChunker config — mirror docling-graph's DocumentChunker.

    Sync requirement: this config MUST match docker/docling-graph/repo/
    docling_graph/core/extractors/document_chunker.py constructor params.
    A regression test (test_chunker_config_parity in tests/integration/)
    diff-checks the two so they cannot drift.
    """
    tokenizer_model_name: str = "BAAI/bge-m3"
    max_tokens: int = 512
    merge_peers: bool = True
    # NB: repeat_table_header, omit_header_on_overflow, always_emit_headings
    # are HybridChunker defaults today (verified at docling source). If
    # docling-graph customizes them, they MUST be added here too.


@dataclass(frozen=True)
class MergedChunk:
    """In-memory representation of one HybridChunker-merged chunk.

    Renamed from "HybridExtractionChunk" to avoid namespace collision with
    the ArcadeDB ExtractionChunk vertex (the persistence type owns that
    name). MergedChunk is the value object; ExtractionChunk is the row.
    """
    chunk_index: int
    text: str                  # output of chunker.contextualize(chunk)
    source_refs: list[str]     # [item.self_ref for item in chunk.meta.doc_items]
    page_no: str | None        # first prov page if resolvable
    token_count: int           # tokenizer.count_tokens(text)


def build_hybrid_chunks_for_extraction(
    doc_json: dict,
    config: HybridChunkConfig | None = None,
) -> list[MergedChunk]:
    """Run HybridChunker against doc_json using the canonical config.

    CALLER CONTRACT:
      - doc_json MUST be the post-sanitize, post-Layer-1-filter shape that
        docling-graph would receive. Producing this shape is the caller's job;
        doc-shape parity is asserted by integration test, not by this helper.

    Failure semantics (fail-loud, mirrors build_extraction_index strict mode):
      - DoclingDocument.model_validate(doc_json) failure → ValueError raised
        (caller's try/except converts to RUN_FULL fallback per the C.4 wrapper)
      - HybridChunker.chunk() returning zero chunks → returns [] (NOT an error;
        caller's BuildIndexDiagnostics.chunks_inserted=0 signals empty-doc)
    """
    cfg = config or HybridChunkConfig()
    raw_tok = AutoTokenizer.from_pretrained(cfg.tokenizer_model_name)
    _raise_tokenizer_max_length(raw_tok, cfg.max_tokens)
    tokenizer = HuggingFaceTokenizer(tokenizer=raw_tok, max_tokens=cfg.max_tokens)
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=cfg.merge_peers,
        # NB: HybridChunker reads its token budget from tokenizer.get_max_tokens(),
        # NOT a chunk_max_tokens kwarg. Verified in HybridChunker source.
    )

    dl_doc = DoclingDocument.model_validate(doc_json)  # may raise → caller catches
    out: list[MergedChunk] = []
    for idx, chunk in enumerate(chunker.chunk(dl_doc=dl_doc)):
        text = chunker.contextualize(chunk=chunk)
        source_refs = [item.self_ref for item in chunk.meta.doc_items]
        page_no = _resolve_first_page_no(chunk)
        token_count = tokenizer.count_tokens(text=text)
        out.append(MergedChunk(
            chunk_index=idx,
            text=text,
            source_refs=source_refs,
            page_no=page_no,
            token_count=token_count,
        ))
    return out


def _resolve_first_page_no(chunk) -> str | None:
    """Walk chunk.meta.doc_items[0].prov[0].page_no. Returns None if absent.

    Mirrors the per-element resolution in extraction_chunk_index.py:669.
    """
    items = getattr(chunk.meta, "doc_items", None) or []
    if not items:
        return None
    prov = getattr(items[0], "prov", None) or []
    if not prov:
        return None
    page_no = getattr(prov[0], "page_no", None)
    if isinstance(page_no, int):
        return str(page_no)
    return None
```

**Note on `HuggingFaceTokenizer.from_pretrained`**: this classmethod DOES exist in docling_core. The reason for the wrap-via-`AutoTokenizer` pattern is that `_raise_tokenizer_max_length` needs to mutate the underlying HF tokenizer's `model_max_length` directly. Using `HuggingFaceTokenizer.from_pretrained` would not expose the raw HF tokenizer object for that mutation.

- [ ] **Step 1: Write failing tests** that:
  - Count merged chunks on a Dvina fixture; assert deterministic order across two calls
  - Assert `MergedChunk.source_refs` non-empty and reference valid texts/tables/pictures indices
  - Assert `chunker.contextualize(chunk)` text contains the parent heading string
  - Empty-doc (no texts/tables/pictures) → returns `[]`, not crash
  - Property test: `chunk_index` dense from 0 to len-1
- [ ] **Step 2: Write failing parity test** at `tests/integration/test_hybrid_chunker_parity.py`: build `MergedChunk` list from worker side and `DocumentChunker.chunk_document` list from docling-graph side on the same post-sanitize fixture; assert byte-by-byte equality of chunk text. This locks the sync requirement in CI.
- [ ] **Step 3: Implement** the helper. Verify constructor invocations match docling-graph (`HuggingFaceTokenizer(tokenizer=..., max_tokens=...)` + `HybridChunker(tokenizer=..., merge_peers=True)`).
- [ ] **Step 4: Run tests**.
- [ ] **Step 5: Commit** `feat(hybrid-chunking): shared chunker helper mirroring docling-graph config + MergedChunk + parity test`.

### Task 2b: New indexer — `build_extraction_index_hybrid`

**Files:**
- Modify: `app/services/extraction_chunk_index.py`
- Test: `tests/unit/test_extraction_chunk_index_hybrid.py`

Add a new function alongside the existing one (do NOT delete `build_extraction_index` yet; feature-flagged rollout per Task 3).

**Failure semantics (mirror existing `build_extraction_index` contract at `extraction_chunk_index.py:524-744`):**
- `DELETE` failures propagate (strict=True), same as existing function.
- `DoclingDocument.model_validate(doc_json)` failure → propagates `ValueError`. The C.4 dispatcher's try/except (which wraps `build_extraction_index` calls) catches and falls back to `RUN_FULL` mode.
- `chunker.chunk()` returning zero chunks → returns `BuildIndexDiagnostics(chunks_inserted=0, ...)` with `chunks_inserted_zero_reason="empty_chunker_output"`. The dispatcher sees zero and treats as "narrowing-ineffective" (per the existing pattern), falling back to RUN_FULL.
- `embed_texts` partial response (returned length ≠ requested) → raises `RuntimeError` to mirror the length-guard at `extraction_chunk_index.py:737`. Strict; do not silently truncate.

**Diagnostics:** extend the existing `BuildIndexDiagnostics` dataclass at `extraction_chunk_index.py:74` with two new fields (`mean_token_count: int = 0`, `chunks_inserted_zero_reason: str | None = None`) rather than introducing a new type. Keeps one diagnostic shape across both indexers.

```python
def build_extraction_index_hybrid(
    doc_json: dict,
    pipeline_run_id: str,
    document_id: str,
    store: GraphStore,
) -> BuildIndexDiagnostics:
    """Index merged chunks via the shared HybridChunker helper.

    One ExtractionChunk row per merged chunk. Each row stores:
      - chunk_index, source_refs, page_no, text (contextualized merged text)
      - embedding: bge-m3 of text
      - token_count: tokenizer.count_tokens(text) for diagnostics

    Vertex id: f"{pipeline_run_id}:chunk_{chunk_index}"

    Failure semantics — see Task 2b spec in the plan for full contract.
    """
    from app.services.hybrid_chunking import build_hybrid_chunks_for_extraction

    # Mirror build_extraction_index Step 1: idempotent DELETE for this run_id
    _delete_by_run_id(store, pipeline_run_id, strict=True)

    chunks = build_hybrid_chunks_for_extraction(doc_json)
    if not chunks:
        return BuildIndexDiagnostics(
            chunks_inserted=0,
            chunks_inserted_zero_reason="empty_chunker_output",
        )

    embeddings = embed_texts([c.text for c in chunks])
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"embed_texts returned {len(embeddings)} embeddings for {len(chunks)} chunks"
        )

    for c, emb in zip(chunks, embeddings):
        store.insert_extraction_chunk(
            pipeline_run_id=pipeline_run_id,
            document_id=document_id,
            vertex_id_format="merged",   # branches the INSERT vertex_id format
            chunk_index=c.chunk_index,
            source_refs=c.source_refs,
            text=c.text,
            embedding=emb,
            page_no=c.page_no,
            token_count=c.token_count,
        )

    return BuildIndexDiagnostics(
        chunks_inserted=len(chunks),
        mean_token_count=int(mean([c.token_count for c in chunks])),
    )
```

- [ ] **Step 1: Write failing tests** covering:
  - merged chunk count matches the shared helper's output
  - each row's `source_refs` non-empty and references real `texts[]/tables[]/pictures[]` indices
  - `text` includes the heading prefix (contains the parent section heading string)
  - deterministic chunk_index across two runs on same doc (pin to a Dvina fixture)
  - empty doc → returns `BuildIndexDiagnostics(chunks_inserted=0, chunks_inserted_zero_reason="empty_chunker_output")`
  - malformed doc_json → `DoclingDocument.model_validate` raises `ValueError`, NOT silently swallowed
  - `embed_texts` partial → `RuntimeError`

- [ ] **Step 2: Extend `BuildIndexDiagnostics`** with `mean_token_count` and `chunks_inserted_zero_reason` fields (default values preserve backward compat).

- [ ] **Step 3: Implement** indexer with failure semantics above.

- [ ] **Step 4: Run tests**.

- [ ] **Step 5: Commit** `feat(extraction-chunk-index): build_extraction_index_hybrid for merged-chunk routing`.

### Task 3: Feature-flag the indexer choice in worker — via pydantic `Settings`

**Files:**
- Modify: `app/config.py` (add typed Settings field — mirror the pattern of `vector_router_mode` at `app/config.py:565`)
- Modify: `app/workers/pipeline.py:~8617` (the index-build site)
- Modify: `.env.example` + `.env` (add new env var per [[feedback-env-vars-must-appear-in-dotenv-files]])
- Test: `tests/unit/test_pipeline_index_flag.py`

**Settings field (pydantic — NOT raw `os.getenv`)**:

```python
# app/config.py (alongside vector_router_mode and friends)
extraction_index_mode: Literal["per_element", "merged"] = Field(
    default="per_element",
    description=(
        "Granularity of ExtractionChunk index rows. 'per_element' indexes "
        "one row per docling element (legacy). 'merged' indexes one row "
        "per HybridChunker output chunk (Phase 1 of merged-chunk routing)."
    ),
)
```

**Worker branching**:

```python
# In derive_ontology_graph (worker pipeline.py:~8617):
from app.config import settings
if settings.extraction_index_mode == "merged":
    from app.services.extraction_chunk_index import build_extraction_index_hybrid
    diag = build_extraction_index_hybrid(doc_json_for_index, run_id, doc_id, store)
else:
    diag = build_extraction_index(doc_json_for_index, run_id, doc_id, store)
```

- [ ] **Step 1: Write failing test** that asserts:
  - default `settings.extraction_index_mode == "per_element"`
  - setting `EXTRACTION_INDEX_MODE=merged` env var → `settings.extraction_index_mode == "merged"`
  - invalid value (e.g. `"foo"`) → pydantic validation error at startup (Literal enforces this)
- [ ] **Step 2: Add the Settings field** + verify it's picked up by `app/config.py`'s settings instance.
- [ ] **Step 3: Wire** the branching at `pipeline.py:~8617`.
- [ ] **Step 4: Add env var** to `.env` (default `per_element`) and `.env.example` with a short comment.
- [ ] **Step 5: Commit** `feat(worker): EXTRACTION_INDEX_MODE Settings field for merged vs per-element indexer`.

### Task 4: Chunk-scope endpoint expands `source_refs` AND populates `text_by_ref` with merged chunk text

**Files (verified):**
- Modify: `app/api/v1/extraction_routing.py:168-407` (`chunk_scope` endpoint)
- Modify: `app/schemas/extraction_routing.py:82-99` (`ChunkScopeResponse` + `ChunkScopeDiagnostics`)
- Modify: `app/services/extraction_chunk_index.py` (already adds `read_chunk_source_refs` in Task 1)
- Test: `tests/integration/test_chunk_scope_endpoint_merged.py`

**IMPORTANT FIELD-NAME CORRECTION**: the existing field is `self_refs` (NOT `selected_refs`). Worker consumes `self_refs` at `app/workers/pipeline.py:7293-7297`. `ChunkScopeDiagnostics` already has `selected_ref_count` at `app/schemas/extraction_routing.py:30`; keep populating it.

**Reuse `text_by_ref`**: `ChunkScopeResponse` already has `text_by_ref: dict[str, str]` at `app/schemas/extraction_routing.py:91-98`. Today it's populated with per-element rendered text. **In merged mode, populate it with merged-chunk text** keyed by a synthetic chunk key (`"chunk_{chunk_index}"`). The worker reads `text_by_ref` at `app/workers/pipeline.py:7298-7303` already, which means **Phase 2's byte-identity goal is achievable without adding a new field**. Add only ONE optional field: `selected_chunks: list[SelectedChunk] | None` — a structured form for Phase 2 to read `(chunk_index, source_refs, token_count, text_by_ref_key)` together.

```python
# After router selects top-K merged-chunk rows (gated on settings.extraction_index_mode == "merged"):
expanded_refs: list[str] = []
seen: set[str] = set()
text_by_ref: dict[str, str] = {}
selected_chunks: list[SelectedChunk] = []

# IMPORTANT: ordering. Don't sort lexicographically — '#/texts/100' would
# precede '#/texts/35'. apply_chunk_scope reorders by body walk
# (scoped_docling_document.py:178-220) so the output doc is correct, but
# preserve chunk-encounter order in the response so consumers see refs
# grouped by their merged chunk's position in HybridChunker output.

for chunk_row in selected:    # iteration order = bge-m3+reranker top-K order
    chunk_key = f"chunk_{chunk_row['chunk_index']}"
    text_by_ref[chunk_key] = chunk_row["text"]   # merged chunk text for Phase 2 direct feed
    refs_for_chunk = read_chunk_source_refs(chunk_row)
    for ref in refs_for_chunk:
        if ref not in seen:
            seen.add(ref)
            expanded_refs.append(ref)
    selected_chunks.append(SelectedChunk(
        chunk_index=chunk_row["chunk_index"],
        chunk_key=chunk_key,                # cross-ref to text_by_ref
        source_refs=refs_for_chunk,         # element refs covered by this chunk
        token_count=chunk_row["token_count"],
    ))

return ChunkScopeResponse(
    mode=existing_mode_value,                # unchanged
    self_refs=expanded_refs,                 # existing field, populated from expansion
    text_by_ref=text_by_ref,                 # existing field, now contains merged chunk text
    selected_chunks=selected_chunks,         # NEW optional structured form
    diagnostics=ChunkScopeDiagnostics(
        selected_ref_count=len(expanded_refs),
        selected_chunk_count=len(selected),
        expanded_ref_count=len(expanded_refs),   # alias for clarity in merged mode
        selected_chunk_token_estimate=sum(c["token_count"] for c in selected),
        ...
    ),
)
```

**`SelectedChunk` pydantic model** (new, in `app/schemas/extraction_routing.py`):

```python
class SelectedChunk(BaseModel):
    """Structured form of a router-selected merged chunk. Phase 2 uses this
    to forward chunk text + provenance to docling-graph without re-deriving.
    """
    chunk_index: int
    chunk_key: str             # key into ChunkScopeResponse.text_by_ref
    source_refs: list[str]     # element self_refs covered by this merged chunk
    token_count: int           # tokenizer.count_tokens(text)
```

- [ ] **Step 1: Write failing test** asserting:
  - response `self_refs` populated from `read_chunk_source_refs` expansion in chunk-encounter order
  - `text_by_ref[chunk_key]` equals the merged chunk text byte-for-byte
  - `selected_chunks` items match selected merged-chunk rows
  - `diagnostics.selected_chunk_count` and `expanded_ref_count` reported correctly
- [ ] **Step 2: Add `SelectedChunk` model + new optional field on `ChunkScopeResponse`**. Existing fields untouched.
- [ ] **Step 3: Implement** the merged-mode population path (gated on `settings.extraction_index_mode == "merged"`).
- [ ] **Step 4: Snapshot test** the response shape for both modes — must be backward-compatible (no field deletions, no semantic changes to existing fields).
- [ ] **Step 5: Confirm** `apply_chunk_scope` (`scoped_docling_document.py:178-220`) still produces a correctly-ordered scoped doc when given chunk-encounter-ordered ref list. **Hard prerequisite**: apply_chunk_scope cleanups #53-56 must be merged first (Pre-flight).
- [ ] **Step 6: Commit** `feat(chunk-scope): merged-mode expansion populates self_refs + text_by_ref + selected_chunks`.

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

### Task 6.5: Bundle migration scope decision

**Files:**
- Decision: which of `air_defense_v3` / `air_defense_v3_baseline_subset` / `air_defense_v3_narrowing_v1` adopts merged-mode retrieval values, and which intentionally stays on per-element for regression-test stability.

Production `SA-2_Sources` points at `air_defense_v3_baseline_subset` (per `[[regression-test-source]]` memory). The C.9b `air_defense_v3_narrowing_v1` was tuned for per-element scoring.

Decision matrix:

| Bundle | Recommended action | Reason |
|---|---|---|
| `air_defense_v3` (production) | Keep per-element retrieval values; do NOT auto-promote to merged until Phase 1 A/B passes | Production stability |
| `air_defense_v3_baseline_subset` | Keep per-element — preserves the regression-test baseline so C.10 baselines remain comparable | A/B uses a different bundle |
| `air_defense_v3_narrowing_v1` | Keep per-element — it's the C.9b sweep-winner; preserve as historical control | Comparison reference |
| `air_defense_v3_merged_v1` (NEW) | Sibling of `_baseline_subset`, same 5 passes, calibrated values from Task 6 sweep | Phase 1 A/B target |

- [ ] **Step 1: Create `ontology_bundles/air_defense_v3_merged_v1/`** as a sibling of `_baseline_subset`. Shared extraction schemas via module imports (same pattern as `_baseline_subset` — see `[[regression-test-source]]` for the existing bundle layout).
- [ ] **Step 2: Write `manifest.yaml`** with Task 6 calibrated values + the `index_mode: merged` declaration at the bundle level (the bundle declaration informs which `extraction_index_mode` value the run uses, optional override).
- [ ] **Step 3: Smoke-test bundle load**: `load_bundle_manifest("air_defense_v3_merged_v1")` succeeds, all 5 passes resolve their template classes.
- [ ] **Step 4: Commit** `feat(bundle): air_defense_v3_merged_v1 sibling for Phase 1 merged-mode A/B`.

### Task 7: Phase 1 A/B against C.10 baseline

**Files:**
- Bundle: `ontology_bundles/air_defense_v3_merged_v1/manifest.yaml` — calibrated values from Task 6.

**Environment lockdown for both runs** (per Pre-flight):
- `EXTRACTION_INDEX_MODE=merged`
- `VECTOR_ROUTER_RETRIEVAL_MODE=direct` (avoid HNSW post-filter starvation)
- Workers actually restarted (`docker restart eip-mmdpp-worker-1 eip-mmdpp-worker-graph-1`; `docker compose restart` from worktree silently no-ops per [[docker-restart-from-worktree-silent-noop]])
- ExtractionChunk vertex table swept clean of stale per-element rows before kickoff

- [ ] **Step 1: Trigger Dvina graph_only** with `air_defense_v3_merged_v1` bundle + env lockdown above.
- [ ] **Step 2: Trigger SA-2 graph_only** with same.
- [ ] **Step 3: Compare against C.10 baselines** (`cfcc9539` Dvina, `7d46c487` SA-2). Required gains:
  - Narrowed-pass recall: kinematics + at least one other narrowed pass shows ≥+50% entity count
  - Wall time: ≤120% of baseline (Phase 1 alone is allowed to be slightly slower; Phase 2 closes the gap)
  - Diagnostic delta: `selected_chunk_count` materially lower than baseline `sel_refs` (merged chunks are denser)
- [ ] **Step 4: Promote `EXTRACTION_INDEX_MODE=merged`** as default if gates pass.

### Phase 1 gate

After Task 7, **discuss with user** (per [[feedback-phase-discussion-before-implementation]]):
- A/B results
- Whether to proceed to Phase 2 or call it done

---

## Chunk 2: Phase 2 — Direct selected-chunk feed (byte-level identity)

Eliminate docling-graph's downstream re-chunking for narrowed passes. The chunk text that the LLM sees is byte-equal to the row stored in `ExtractionChunk` — the worker reads it from the vertex and forwards it; no second HybridChunker invocation anywhere.

### Task 8: New docling-graph endpoint accepting pre-chunked input — also skips sanitize

**Files (verified):**
- Modify: docling-graph endpoint code at `docker/docling-graph/app/main.py:638-657` (extraction handler) — branch on new `selected_chunks` field
- Modify: docling-graph request schema (`docker/docling-graph/app/schemas.py`)
- Modify: `docker/docling-graph/repo/docling_graph/core/extractors/factory.py:31` and `strategies/many_to_one.py:46-66` — the `use_chunking=False` path already exists but `many_to_one.py:65` raises `ValueError("Delta extraction requires use_chunking=True.")`. Gate accordingly.
- Test: `tests/integration/test_extract_pass_chunked.py`

**Critical: skip `_sanitize_docling_document` when `selected_chunks` is provided.** Today `main.py:437-516` sanitizes the doc unconditionally on extract requests (per `main.py:638-657`). Phase 2 byte identity requires this skip — the chunk text was produced by the worker against a doc that already went through `filter_docling_document`; running docling-graph's sanitize again would mutate the doc whose refs the chunk text claims. Add a guard:

```python
# In main.py extract handler (line ~638):
if request.selected_chunks:
    # Skip sanitize entirely — the worker already applied Layer-1 filter
    # and built chunks from that exact post-filter doc.
    # Also skip DocumentChunker — selected_chunks ARE the LLM batches.
    pass
else:
    doc_json = _sanitize_docling_document(doc_json)
    chunks = DocumentChunker(...).chunk_document(doc)
```

New request shape:
```json
{
  "pass_name": "missile_kinematics",
  "bundle_key": "air_defense_v3_merged_v1",
  "selected_chunks": [
    {
      "chunk_index": 12,
      "chunk_key": "chunk_12",
      "text": "## SECTION\n\nLength:\n35 feet\n...",
      "source_refs": ["#/texts/35", "#/texts/36", "..."],
      "token_count": 287
    },
    {"chunk_index": 28, "chunk_key": "chunk_28", "text": "...", "source_refs": [...], "token_count": 412},
    ...
  ],
  "upstream_refs": [...]
}
```

Behavior:
1. If `selected_chunks` present: skip sanitize, skip DocumentChunker, iterate `selected_chunks` directly as LLM batches.
2. If `selected_chunks` absent: existing path (sanitize + DocumentChunker.chunk_document).
3. Preserve `source_refs` into extracted-fact `evidence_units` (see open question #3 — schema alignment with field-provenance prompt block).
4. Bypass the `many_to_one.py:65` ValueError when `selected_chunks` is set (the constraint was "delta needs chunking"; we've already chunked).

- [ ] **Step 1: Write failing integration test** that POSTs to docling-graph with `selected_chunks` and asserts:
  - the doc_json is NOT mutated by sanitize (compare pre/post)
  - LLM batches submitted are exactly the texts from `selected_chunks` (byte-equal)
  - extracted entities cite `source_refs` from the chunk that produced them
- [ ] **Step 2: Extend the request schema** with the optional `selected_chunks` field.
- [ ] **Step 3: Implement** sanitize-skip + chunker-skip branch in `main.py:638-657`.
- [ ] **Step 4: Bypass** the `many_to_one.py:65` ValueError when `selected_chunks` is provided.
- [ ] **Step 5: Provenance check** — assertion in integration test that `entity.evidence_units` (or equivalent provenance field) carries `source_refs` from the producing merged chunk.
- [ ] **Step 6: Commit** in docling-graph `feat(extract): accept selected_chunks; bypass sanitize+chunker on chunked path`.

### Task 9: Worker uses chunked endpoint for narrowed passes — byte-identical chunk text via chunk-scope response

**Files (verified):**
- Modify: `app/workers/pipeline.py:derive_ontology_graph_pass` (the per-pass dispatcher)
- Test: `tests/unit/test_pipeline_chunked_dispatch.py`

**Critical invariant for byte identity**: the worker forwards the **chunk text already returned by the chunk-scope endpoint** (Task 4 populates `text_by_ref[chunk_key]` with merged chunk text + `selected_chunks` with the structured form). The worker does NOT re-read from ArcadeDB and does NOT re-run HybridChunker. The chunk text the LLM sees is byte-equal to:
1. What `build_extraction_index_hybrid` wrote into `ExtractionChunk.text`
2. What `/v1/extraction/chunk-scope` returned in `text_by_ref` / `selected_chunks`
3. What the worker POSTs to `/extract-pass-chunked.selected_chunks[*].text`

These three are byte-identical by construction because step 2 reads from step 1 and step 3 forwards step 2 verbatim.

Flow:
1. Worker calls `/v1/extraction/chunk-scope` (already happens today at `app/workers/pipeline.py:7280+`)
2. Worker reads `response.selected_chunks` (new field from Task 4) — each item has `chunk_index`, `chunk_key`, `source_refs`, `token_count`
3. Worker reads `response.text_by_ref[chunk_key]` for the merged chunk text
4. Worker POSTs to `/extract-pass-chunked` with the request shape from Task 8 — `selected_chunks: [{chunk_index, text, source_refs, token_count, chunk_key}]`
5. docling-graph iterates `selected_chunks` directly (Task 8). NO `DocumentChunker` invocation, NO `_sanitize_docling_document`.

**No worker-side ArcadeDB re-read.** The plan's earlier draft proposed "read each chunk's text from ExtractionChunk vertices" — that's an unnecessary extra round-trip. The chunk-scope response already carries the text.

- [ ] **Step 1: Write failing test** mocking the chunk-scope response with merged chunks and the docling-graph endpoint:
  - asserts the POSTed `selected_chunks[*].text` is byte-equal to `chunk_scope_response.text_by_ref[chunk_key]`
  - asserts no ArcadeDB read for chunk text between chunk-scope and docling-graph dispatch
- [ ] **Step 2: Implement** the read-and-forward path in `derive_ontology_graph_pass` (gated on `settings.extraction_index_mode == "merged"` so per-element mode still uses the existing path).
- [ ] **Step 3: Sanity check** on a Dvina run — extracted entities' `evidence_units` (or equivalent provenance) contain `source_refs` from the merged chunk that produced them.
- [ ] **Step 4: Commit** `feat(worker): narrowed passes forward chunk text byte-identically to docling-graph`.

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

- Pre-flight: 2-3h (additional items added in rev 2 — tokenizer pin in docling-graph, sanitize-skip wiring, #65 status check, apply_chunk_scope cleanups #53-56 prereq)
- Phase 1 (Tasks 1-6.5 implementation): ~1.5 days
- Phase 1 (Task 6 calibration + Task 7 A/B): ~1 day (mostly waiting on Dvina + SA-2 runs)
- Phase 2 (Tasks 8-10): ~1-2 days
- Total: **4-6 days focused effort**, plus calibration wall-time which is gated on Dvina/SA-2 runs (~5 hours each).

---

## Revision log

### Rev 2 (2026-05-27) — three-reviewer architectural review

Absorbed all 8 BLOCKERs and all 9 MAJORs from the parallel review by Technical Implementation / Gaps+Performance / Code Quality subagents:

- **B1** Response field is `self_refs` (not `selected_refs`) → Task 4 corrected.
- **B2** `text_by_ref` already exists in `ChunkScopeResponse` → Task 4 reuses it for merged chunk text; Phase 2 Task 9 forwards from `text_by_ref` instead of re-reading ArcadeDB.
- **B3** docling-graph defaults to MiniLM tokenizer, not bge-m3 → Pre-flight requires pinning docling-graph's `chunker_config["tokenizer_name"]` to `BAAI/bge-m3` (one-line change in `many_to_one.py:70`) as a pre-Phase-1 prerequisite.
- **B4** `filter_docling_document` and `_sanitize_docling_document` are not byte-equivalent → Pre-flight resolution: Task 8 adds a sanitize-skip branch when `selected_chunks` is provided; Task 2a adds an integration parity test.
- **B5** Wrong vertex schema file paths → Task 1 file list corrected (`arcadedb_schema.py:38` + `extraction_chunk_index.py:749-760`).
- **B6** Env-var pattern uses pydantic `Settings`, not raw `os.getenv` → Task 3 adds typed `extraction_index_mode: Literal["per_element","merged"]` field to `app/config.py`.
- **B7** Naming collision `HybridExtractionChunk` vs `ExtractionChunk` → renamed to `MergedChunk` throughout.
- **B8** Error handling unspecified → Task 2b spec'd explicitly: `model_validate` propagates, empty chunks return `chunks_inserted=0` with reason, embed length mismatch raises `RuntimeError`.

- **M1** apply_chunk_scope cleanups #53-56 → promoted to Pre-flight hard prerequisite.
- **M2** Pin `VECTOR_ROUTER_RETRIEVAL_MODE=direct` for A/B → Pre-flight + Task 7 environment spec.
- **M3** `HuggingFaceTokenizer.from_pretrained` DOES exist → Pre-flight wording corrected; rationale for wrap pattern restated (mutate `model_max_length` for warning suppression).
- **M4** Existing bundle migration scope → new Task 6.5 with decision matrix; preserve `_baseline_subset` and `_narrowing_v1` as per-element controls; new `_merged_v1` sibling for A/B.
- **M5** Ollama bge-m3 `num_ctx` may be 512 → Pre-flight check item.
- **M6** `HybridChunkConfig` sync between worker and docling-graph → Task 2a includes parity integration test diff-checking the two constructor arg sets.
- **M7** `read_chunk_source_refs(row)` helper signature/location → Task 1 includes the canonical signature + behavior contract.
- **M8** `_raise_tokenizer_max_length` not mirrored → Task 2a adds the helper alongside the chunker config.
- **M9** Janitor TaskList #65 may already be done → Pre-flight verification step (`git log -- app/workers/celery_app.py`).

### Rev 1 (2026-05-27) — initial review

Earlier user-provided architectural review applied:
- Removed premature 390-token recommendation; replaced with measured decision tree
- Pinned exact `DocumentChunker` constructor params; corrected pseudo-code constructor patterns
- Added doc-shape parity pre-flight check
- Introduced `build_hybrid_chunks_for_extraction` shared helper at `app/services/hybrid_chunking.py`
- Preserved existing `mode` / `selected_refs` response contract (later corrected to `self_refs` in Rev 2)
- ArcadeDB list-property fallback guidance
- Chunk-encounter ordering preserved (no lex sort)
- Sweep dimensions expanded
- Byte-identity invariant added to Phase 2

### Rev 0 (2026-05-27) — initial draft

Initial 2-phase plan based on user's design (Phase 1: real HybridChunker chunks in router index; Phase 2: direct selected-chunk feed to LLM).
