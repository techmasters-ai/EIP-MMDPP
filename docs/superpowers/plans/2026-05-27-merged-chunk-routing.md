# Merged-Chunk Routing Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the retrieval/rerank/LLM units. Top-k scoring runs against the same merged HybridChunker chunks that the LLM consumes — not against per-element fragments.

**Architecture (post-implementation):**
- Phase 1: `build_extraction_index` populates `ExtractionChunk` rows from real `HybridChunker.chunk(...)` output. One row = one merged chunk. Vector router scores merged chunks. `apply_chunk_scope` receives expanded constituent refs; docling-graph still rechunks downstream.
- Phase 2: docling-graph accepts pre-selected chunk texts directly (new request shape); narrowed-mode passes skip downstream rechunking. Retrieval unit = rerank unit = LLM input unit, byte-for-byte.

**Tech stack:** docling `HybridChunker` (`docling.chunking.HybridChunker`, already in `worker-1` env), bge-m3 embeddings, bge-reranker-v2-m3 cross-encoder, ArcadeDB `ExtractionChunk` vertex, FastAPI on docling-graph.

---

## Glossary

Five names appear across the plan in slightly overlapping roles. Pin them here:

| Name | Type | Where it lives | Meaning |
|---|---|---|---|
| `chunk_index` | int (dense from 0 per pipeline_run_id) | column on `ExtractionChunk` vertex | Position of this merged chunk in HybridChunker output |
| `vertex_id` | string PK | column on `ExtractionChunk` vertex | `f"{pipeline_run_id}:chunk_{chunk_index}"` in merged mode; `f"{pipeline_run_id}:{self_ref}"` in legacy per-element mode |
| `source_refs` | list[str] | column on `ExtractionChunk` vertex | Element self_refs that contributed to this merged chunk (e.g. `["#/texts/35", "#/texts/36"]`) |
| `self_refs` | list[str] | field on `ChunkScopeResponse` | Existing response field carrying the per-element refs the worker scopes the doc to. In merged mode, populated from the **union** of source_refs across selected merged chunks |
| `chunk_key` | string | computed | `f"chunk_{chunk_index}"`; a stable identifier for one merged chunk in API payloads (request and response). NOT used as a `text_by_ref` key — see Task 6 |

`text_by_ref` keeps its existing semantics: `dict[self_ref → element text]`, consumed by `apply_chunk_scope` to override retained `TextItem.text`. Merged chunk text rides on `SelectedChunk.text`, NOT on `text_by_ref`.

---

## Why this plan exists

Current state — observable on Dvina (`d23fa85d`) and SA-2 (`1862d234`, stopped):
- Layer-1 + Layer-2 filter fixes (Options E + G) successfully retain more doc content in the index pool, but **`sel_refs` is unchanged** between baseline and post-filter runs because the router scores **per-element** chunks and small fragments don't outscore the existing top-K.
- Merged HybridChunker chunks (which the LLM actually consumes) live downstream of `apply_chunk_scope` — the router never sees them during scoring.
- Outcome: filter fixes can't materially improve narrowed-pass recall while the retrieval granularity remains misaligned with the consumer granularity.

The cleanest fix is to make HybridChunker the source of truth for the index, not the post-chunk-scope LLM batcher. This plan does that.

---

## Pre-flight verification

Read-only checks. None of these involve a code change — those are split into "Phase 0 prerequisite tasks" below.

### Determinism + provenance

- [ ] **HybridChunker order is deterministic** on identical input. Run `HybridChunker.chunk(dl_doc)` twice on the Dvina doc; assert chunk count + `meta.doc_items[*].self_ref` tuples match. (HybridChunker source review shows no RNG — it walks `body.children` in deterministic order.)

- [ ] **Docling chunk has `meta.doc_items` provenance**. Confirm `chunk.meta.doc_items` exists and each item has a `self_ref` attribute resolvable to `texts[i]/tables[i]/pictures[i]`.

### Tokenizer + chunker signatures

- [ ] **`HybridChunker.__init__` signature**: takes `tokenizer` and `merge_peers` (no `chunk_max_tokens` kwarg). Token budget read from `tokenizer.get_max_tokens()`. Confirm via:
  ```
  docker exec eip-mmdpp-worker-1 python -c \
    "import inspect; from docling.chunking import HybridChunker; print(inspect.signature(HybridChunker.__init__))"
  ```

- [ ] **`HuggingFaceTokenizer.from_pretrained` DOES exist** as a classmethod but we use `HuggingFaceTokenizer(tokenizer=tok, max_tokens=N)` to mirror docling-graph (`document_chunker.py:60-127`) — that pattern lets us also call `_raise_tokenizer_max_length(tok, N)` which mutates the underlying HF tokenizer's `model_max_length` to suppress noisy oversize warnings.

### Reranker truncation behavior

- [ ] **bge-reranker-v2-m3 truncation behavior** on 512-token-pair input. Score a 200-token merged chunk vs a 512-token merged chunk against the kinematics query; capture both scores. Cross-encoder typically right-truncates — long body text may be silently dropped.

  **Decision tree if truncation is harmful** (i.e. A/B shows missed chunks correlated with body-right being truncated):
  1. Keep 512 and accept/measure the truncation impact
  2. Move both router index AND docling-graph to a lower token budget (joint change so byte identity holds)
  3. Use a reranker model with longer context
  4. Rerank with a derived preview (e.g. first N tokens) but still select/store the full HybridChunker chunk for downstream

  **Default**: option 1 (keep 512 for Phase 1) unless measurement forces a change.

### Environment + dependency checks

- [ ] **Verify Ollama bge-m3 `num_ctx`** is ≥1024. Default may be 512, which silently truncates a 512-tokenized merged chunk after the BGE retrieval prefix `"Represent this sentence: "` is prepended. This is OPERATOR-LEVEL (Ollama daemon config), not app code — flag for environment setup, not code work. Read `app/services/embedding.py:40-44` to confirm the prefix logic + that no app-level `num_ctx` override exists.

- [ ] **Confirm production tokenizer override does not exist**. Read `docker/docling-graph/app/config_builder.py:293-319` for any override of `tokenizer_name`. If a non-default override exists, surface it before Phase 0 Task 0a (below).

### apply_chunk_scope and Path B status checks

- [ ] **Inspect `apply_chunk_scope`** at `app/services/scoped_docling_document.py:178-220`. Confirm it walks `body.children` in document order and reorders the retained ref set by that walk (so the endpoint's chunk-encounter-ordered output is OK). The plan does NOT depend on any unmerged cleanup tasks — earlier rev-2 prereq citations to TaskList #53-56 were wrong and are removed in rev 3.

- [ ] **Confirm `VECTOR_ROUTER_RETRIEVAL_MODE=direct` is already the production default** at `app/config.py:587` and `.env.example:479`. Path B direct-cosine retrieval is implemented at `app/services/extraction_chunk_search.py:265-436`. The earlier rev-2 citation to TaskList #61 as "in_progress" was also wrong — direct mode is the default.

- [ ] **Verify TaskList #65 status**. `app/workers/celery_app.py:107-114` already has `vr-purge-terminated-extraction-chunks` in the beat schedule. Confirm by checking `git log -- app/workers/celery_app.py`. If actually done, mark #65 complete. If TaskList #64 (inline cleanup never fires on COMPLETE) IS still open, schedule that fix before merged-mode A/B so stale per-element rows don't pollute the index pool across mode transitions.

---

## Phase 0: Prerequisite code changes (BLOCKING for Phase 1)

These ARE code changes, not verifications. Each gets its own commit. They must land before any Phase 1 task starts.

### Task 0a: Pin docling-graph's `chunker_config` tokenizer to `BAAI/bge-m3`

**Files:**
- Modify: `docker/docling-graph/repo/docling_graph/core/extractors/strategies/many_to_one.py:68-75`

**Why**: Today `DocumentChunker` defaults `tokenizer_name` to `"sentence-transformers/all-MiniLM-L6-v2"` (`document_chunker.py:73-74`), and `many_to_one.py:70` constructs the chunker config WITHOUT a `tokenizer_name` override. So docling-graph chunks against MiniLM tokens while the worker indexer (Task 2) would chunk against bge-m3 tokens. Phase 2's byte-identity invariant is impossible until both sides use the same tokenizer.

```python
# many_to_one.py:70 — current:
chunker_config = {"chunk_max_tokens": int(chunk_max_tokens or 512)}

# After:
chunker_config = {
    "chunk_max_tokens": int(chunk_max_tokens or 512),
    "tokenizer_name": "BAAI/bge-m3",
}
```

- [ ] **Step 1: Read** `many_to_one.py:46-75` to confirm current state.
- [ ] **Step 2: Add the `tokenizer_name` key** to `chunker_config`.
- [ ] **Step 3: Smoke-test** a docling-graph extraction call still succeeds on a fixture doc (no API regression).
- [ ] **Step 4: Commit** in docling-graph `fix(chunker-config): pin tokenizer_name to BAAI/bge-m3 to match worker indexer`.

### Task 0b: Sanitize-skip + chunker-skip when `selected_chunks` is provided

**Files:**
- Modify: `docker/docling-graph/app/main.py:638-657` (extract handler) — add a guard that skips `_sanitize_docling_document` and `DocumentChunker.chunk_document` when the new optional `selected_chunks` field is present in the request.
- Modify: docling-graph request schema (`docker/docling-graph/app/schemas.py`) — add optional `selected_chunks: list[SelectedChunkInput] | None = None` field.
- Modify: `docker/docling-graph/repo/docling_graph/core/extractors/strategies/many_to_one.py:66` — `ValueError("Delta extraction requires use_chunking=True.")` must NOT fire when `selected_chunks` is set (the constraint was "delta needs chunking"; we've already chunked).

**Why**: Worker's `filter_docling_document` and docling-graph's `_sanitize_docling_document` are NOT byte-equivalent (worker uses `chunk_quality.classify_chunk`; docling-graph uses `_looks_like_nav_or_tracking` defined at `main.py:395-434`, invoked from `_sanitize_docling_document` body at `main.py:437-522`). Running sanitize on a doc the worker already filtered breaks the byte-identity chain Phase 2 depends on. Skip both transforms when `selected_chunks` rides the request — the chunks ARE the LLM batches.

```python
# main.py:638-657 (extract handler) — branch on selected_chunks:
if request.selected_chunks:
    # Worker provided pre-built merged chunks. Skip both sanitize AND
    # DocumentChunker — selected_chunks ARE the LLM batches.
    llm_batches = [c.text for c in request.selected_chunks]
    batch_provenance = [c.source_refs for c in request.selected_chunks]
else:
    # Existing path: sanitize + chunk
    doc_json = _sanitize_docling_document(doc_json)
    chunks = DocumentChunker(...).chunk_document(doc)
    llm_batches = [c.text for c in chunks]
    batch_provenance = [c.source_refs for c in chunks]  # or per-element refs
```

- [ ] **Step 1: Extend request schema** with optional `selected_chunks` field + `SelectedChunkInput` model. Receiver-side fields: `chunk_index` (int), `text` (str), `source_refs` (list[str]), `token_count` (int). **`chunk_key` is OMITTED on the receiver side** — docling-graph iterates `selected_chunks` by list order, doesn't index by key, so the field would be unused noise. Keep it only on the worker-side `SelectedChunk` (Task 6) for log correlation if needed.
- [ ] **Step 2: Add the guard branch** at `main.py:638-657`.
- [ ] **Step 3: Bypass** the `many_to_one.py:66` ValueError when `selected_chunks` is set.
- [ ] **Step 4: Integration test** that POSTs to `/extract-pass` with `selected_chunks` and asserts: (a) `_sanitize_docling_document` not called (mock + assert); (b) `DocumentChunker.chunk_document` not called; (c) LLM batches submitted are exactly `selected_chunks[*].text` byte-equal.
- [ ] **Step 5: Commit** in docling-graph `feat(extract): accept selected_chunks; bypass sanitize+chunker on chunked path`.

(Task 0b also unblocks the actual `/extract-pass` chunked-mode wiring used in Phase 2 Task 12 — but Phase 1 needs the sanitize-skip path EXISTING because the parity test in Task 2 calls it.)

### Phase 0 atomicity + release semantics

Task 0a (tokenizer pin) is backward-compatible alone: MiniLM→bge-m3 narrows the existing default but does not break the per-element worker path because the worker doesn't yet read the docling-graph chunker output. Task 0b adds an optional request field; absent `selected_chunks` keeps existing behavior. So either change can ship independently without breaking production.

**Recommended deploy**: ship 0a + 0b in the **same docling-graph image release**. Order within the release doesn't matter. If 0a deploys first and 0b's deploy fails, production stays functional (just running bge-m3 tokenizer instead of MiniLM on the existing chunked path — no API surface change). Same for the reverse.

### Phase 0 gate

After Task 0a + 0b: re-run pre-flight HybridChunker determinism check; smoke-test docling-graph extraction against a Dvina fixture using `selected_chunks=None` (existing path) AND `selected_chunks=[...]` (new path). Both must succeed. **Discuss with user** before kicking off Phase 1 Task 1.

---

## Chunk 1: Phase 1 — Real HybridChunker chunks in the index

Replace per-element `_walk_docling_elements` with real HybridChunker output via a shared helper. Router scores merged chunks; `apply_chunk_scope` semantics preserved by expanding to constituent refs.

**Phase 1 fixes selection granularity.** It does NOT yet give byte identity between router-selected chunks and LLM input — that's Phase 2. The downstream rechunk in docling-graph still happens after `apply_chunk_scope`, so merged-chunk boundaries may shift slightly when neighbors are removed from the scoped doc. Acceptable for Phase 1; closed by Phase 2.

### Task 1: Add fields to `ExtractionChunk` vertex schema

**Files (verified):**
- Modify: `app/services/arcadedb_schema.py:38` — vertex schema declaration (`ExtractionChunk` block)
- Modify: `app/services/extraction_chunk_index.py:749-760` — inline INSERT SQL string that hard-codes `f"{pipeline_run_id}:{self_ref}"` vertex id format. Branched by merged-mode flag (Task 4).
- Modify: `app/services/extraction_chunk_index.py` — add `read_chunk_source_refs(row) -> list[str]` accessor helper (signature below).
- Test: `tests/integration/test_extraction_chunk_schema.py`

New columns on `ExtractionChunk` (see Glossary for term definitions):
- `chunk_index: int` — position of this chunk in HybridChunker output. Default `-1` for legacy per-element rows.
- `source_refs` — element self_refs that contributed to this merged chunk. `CommunityReport` at `arcadedb_schema.py:91,94` already uses ArcadeDB `LIST` type successfully → use `LIST` from day one. If list-typing fails for any reason, fall back to JSON-encoded string `source_refs_json` — the row interface (the accessor) is the same either way.
- `token_count: int` — diagnostics field. Default `0` for legacy rows.

**Accessor helper (canonical signature, single source of truth):**

```python
# app/services/extraction_chunk_index.py
def read_chunk_source_refs(row: dict | object) -> list[str]:
    """Return source_refs as a list[str] regardless of underlying storage.

    Handles:
      - native ArcadeDB LIST property (returns row["source_refs"] directly)
      - JSON-encoded string fallback (row["source_refs_json"] → json.loads)
      - legacy per-element rows (returns [] since they have no merged source_refs)

    NEVER returns None. Empty list means "no constituent refs known" — caller
    decides whether that's an error or a legacy row.
    """
```

Vertex id format changes from `pipeline_run_id:<element_self_ref>` to `pipeline_run_id:chunk_<chunk_index>`.

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
    legacy = {"chunk_index": -1, "self_ref": "#/texts/0"}
    assert read_chunk_source_refs(legacy) == []
```

- [ ] **Step 2a: Add the column declarations** to `arcadedb_schema.py:38` block with defaults (`chunk_index = -1`, `source_refs = []` or `source_refs_json = "[]"`, `token_count = 0`).
- [ ] **Step 2b: Implement schema migration** — `ALTER TYPE ExtractionChunk ADD PROPERTY chunk_index INTEGER DEFAULT -1`, etc. ArcadeDB does NOT auto-fill columns added later, so this step also runs an explicit **backfill UPDATE** for legacy rows:

      UPDATE ExtractionChunk SET chunk_index = -1
        WHERE chunk_index IS NULL;
      UPDATE ExtractionChunk SET source_refs = []
        WHERE source_refs IS NULL;
      UPDATE ExtractionChunk SET token_count = 0
        WHERE token_count IS NULL;

  Without this, post-Task-5 `read_chunk_source_refs` returns `[]` (safe via the accessor) BUT `chunk_row["token_count"]` is `None` and Task 6's `sum(c["token_count"] for c in selected)` raises `TypeError`. Belt-and-suspenders: also have a `read_chunk_token_count(row) -> int` helper that coalesces None → 0; document the rule that callers go through accessors, never raw dict access on these columns.

- [ ] **Step 2c: Verify against running DB** — `DESCRIBE TYPE ExtractionChunk` shows all three new columns; `SELECT count(*) FROM ExtractionChunk WHERE chunk_index IS NULL` returns 0. Lock the verification in CI via an idempotent migration test.
- [ ] **Step 3: Update inline INSERT SQL** at `extraction_chunk_index.py:749-760` (per-element path still uses the existing vertex_id format; merged-mode path uses the new format — branched by Task 4 flag).
- [ ] **Step 4: Implement `read_chunk_source_refs` (and `read_chunk_token_count`) accessors** with the storage-agnostic, None-coalescing semantics above.
- [ ] **Step 5: Run integration test**. Expected: PASS.
- [ ] **Step 6: Commit** `feat(extraction-chunk): add chunk_index + source_refs + token_count columns + accessors + backfill migration`.

**Rollback semantics**: if Step 2a (declarations) lands but Step 2b (migration + backfill) fails, the schema-declared columns won't exist in the live DB. Treat 2a + 2b as a single deployment unit — either both run or neither. The CI migration test in 2c gates this.

### Task 2: Shared chunker helper — `build_hybrid_chunks_for_extraction`

**Files:**
- Create: `app/services/hybrid_chunking.py`
- Test: `tests/unit/test_hybrid_chunking.py`
- Test: `tests/integration/test_hybrid_chunker_parity.py` (byte-equal output vs docling-graph's `DocumentChunker`)

Avoid copy-pasting `DocumentChunker` config between the worker indexer and docling-graph. Centralize it in one helper that mirrors `docker/docling-graph/repo/docling_graph/core/extractors/document_chunker.py:60-127` exactly.

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
    on oversize chunks. Mirrors document_chunker.py:30-46 exactly.
    """
    hf_tokenizer.model_max_length = max(chunk_max_tokens, _TOKENIZER_COUNTING_MAX_LENGTH)


@dataclass(frozen=True)
class HybridChunkConfig:
    """Canonical HybridChunker config — mirror docling-graph's DocumentChunker.

    Sync requirement: MUST match docker/docling-graph/repo/docling_graph/core/
    extractors/document_chunker.py constructor params. Parity locked in CI
    by tests/integration/test_hybrid_chunker_parity.py.
    """
    tokenizer_model_name: str = "BAAI/bge-m3"
    max_tokens: int = 512
    merge_peers: bool = True


@dataclass(frozen=True)
class MergedChunk:
    """In-memory representation of one HybridChunker-merged chunk.

    Named MergedChunk (not HybridExtractionChunk) to avoid namespace
    collision with the ArcadeDB ExtractionChunk vertex. MergedChunk is
    the value object; ExtractionChunk is the row.
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
      - doc_json MUST be the post-Layer-1-filter shape. Doc-shape parity
        with docling-graph (after Task 0b's sanitize-skip) is asserted
        by the parity test, not by this helper.

    Failure semantics (fail-loud, mirrors build_extraction_index strict mode):
      - DoclingDocument.model_validate(doc_json) failure → ValueError raised
        (caller's try/except converts to RUN_FULL fallback per the C.4 wrapper)
      - HybridChunker.chunk() returning zero chunks → returns [] (NOT an error)
    """
    cfg = config or HybridChunkConfig()
    raw_tok = AutoTokenizer.from_pretrained(cfg.tokenizer_model_name)
    _raise_tokenizer_max_length(raw_tok, cfg.max_tokens)
    tokenizer = HuggingFaceTokenizer(tokenizer=raw_tok, max_tokens=cfg.max_tokens)
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=cfg.merge_peers,
    )

    dl_doc = DoclingDocument.model_validate(doc_json)  # may raise → caller catches
    out: list[MergedChunk] = []
    for idx, chunk in enumerate(chunker.chunk(dl_doc=dl_doc)):
        text = chunker.contextualize(chunk=chunk)
        source_refs = [item.self_ref for item in chunk.meta.doc_items]
        page_no = _resolve_first_page_no(chunk)
        token_count = tokenizer.count_tokens(text=text)
        out.append(MergedChunk(
            chunk_index=idx, text=text, source_refs=source_refs,
            page_no=page_no, token_count=token_count,
        ))
    return out


def _resolve_first_page_no(chunk) -> str | None:
    """Walk chunk.meta.doc_items[0].prov[0].page_no. Returns None if absent."""
    items = getattr(chunk.meta, "doc_items", None) or []
    if not items:
        return None
    prov = getattr(items[0], "prov", None) or []
    if not prov:
        return None
    page_no = getattr(prov[0], "page_no", None)
    return str(page_no) if isinstance(page_no, int) else None
```

**Note on `HuggingFaceTokenizer.from_pretrained`**: this classmethod DOES exist in docling_core. The wrap-via-`AutoTokenizer` pattern is used because `_raise_tokenizer_max_length` needs to mutate the underlying HF tokenizer's `model_max_length` directly.

- [ ] **Step 1a: Failing test — chunk count determinism**. Run twice on a Dvina fixture; assert chunk count + `[item.self_ref for item in chunk.meta.doc_items]` tuples match.
- [ ] **Step 1b: Failing test — provenance**. Assert each `MergedChunk.source_refs` non-empty and references real `texts[]/tables[]/pictures[]` indices.
- [ ] **Step 1c: Failing test — heading prefix**. Assert `MergedChunk.text` contains the parent heading string.
- [ ] **Step 1d: Failing test — empty doc**. `doc_json` with empty `texts/tables/pictures` returns `[]`, not crash.
- [ ] **Step 1e: Failing test — chunk_index density**. `[c.chunk_index for c in chunks]` is `[0, 1, ..., len-1]`.
- [ ] **Step 2: Failing parity test** at `tests/integration/test_hybrid_chunker_parity.py`. Build `MergedChunk` list from worker side AND call docling-graph's `/extract-pass` with sanitize-skip (Task 0b) on the same post-Layer-1-filter fixture; assert byte-equal chunk text. **This locks the Task 0a + 0b prerequisites in CI.**
- [ ] **Step 3: Implement** the helper.
- [ ] **Step 4: Run tests**.
- [ ] **Step 5: Commit** `feat(hybrid-chunking): shared chunker helper mirroring docling-graph config + MergedChunk + parity test`.

### Task 3: New indexer — `build_extraction_index_hybrid`

**Files:**
- Modify: `app/services/extraction_chunk_index.py`
- Test: `tests/unit/test_extraction_chunk_index_hybrid.py`

Add a new function alongside the existing one (do NOT delete `build_extraction_index` yet; feature-flagged rollout per Task 4).

**Failure semantics (mirror existing `build_extraction_index` contract at `extraction_chunk_index.py:524-744`):**
- `DELETE` failures propagate (strict=True).
- `DoclingDocument.model_validate(doc_json)` failure → propagates `ValueError`. The C.4 dispatcher's try/except catches and falls back to `RUN_FULL` mode.
- `chunker.chunk()` returning zero chunks → returns `BuildIndexDiagnostics(chunks_inserted=0, chunks_inserted_zero_reason="empty_chunker_output")`. The dispatcher sees zero and treats as "narrowing-ineffective" (per existing pattern).
- `embed_texts` partial response (returned length ≠ requested) → raises `RuntimeError` (mirrors the length-guard at `extraction_chunk_index.py:737-744`). Fail-loud; do not silently truncate.

**Diagnostics:** extend the existing `BuildIndexDiagnostics` dataclass at `extraction_chunk_index.py:74` with two new fields (`mean_token_count: int = 0`, `chunks_inserted_zero_reason: str | None = None`) rather than introducing a new type.

```python
def build_extraction_index_hybrid(
    doc_json: dict,
    pipeline_run_id: str,
    document_id: str,
    store: GraphStore,
) -> BuildIndexDiagnostics:
    """Index merged chunks via the shared HybridChunker helper.

    Vertex id: f"{pipeline_run_id}:chunk_{chunk_index}"
    """
    from app.services.hybrid_chunking import build_hybrid_chunks_for_extraction

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
            vertex_id_format="merged",
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

- [ ] **Step 1a: Failing test — chunk count matches helper output**.
- [ ] **Step 1b: Failing test — source_refs populated** and reference valid texts/tables/pictures indices.
- [ ] **Step 1c: Failing test — text has heading prefix**.
- [ ] **Step 1d: Failing test — deterministic chunk_index across two runs** on same Dvina fixture.
- [ ] **Step 1e: Failing test — empty doc** returns `BuildIndexDiagnostics(chunks_inserted=0, chunks_inserted_zero_reason="empty_chunker_output")`.
- [ ] **Step 1f: Failing test — malformed doc_json** raises `ValueError` (not silently swallowed).
- [ ] **Step 1g: Failing test — `embed_texts` partial** raises `RuntimeError`.
- [ ] **Step 2: Extend `BuildIndexDiagnostics`** with `mean_token_count` and `chunks_inserted_zero_reason` fields.
- [ ] **Step 3: Implement** indexer with failure semantics above.
- [ ] **Step 4: Run tests**.
- [ ] **Step 5: Commit** `feat(extraction-chunk-index): build_extraction_index_hybrid for merged-chunk routing`.

### Task 4: Feature-flag the indexer choice in worker — via pydantic `Settings`

**Files:**
- Modify: `app/config.py` (add typed Settings field — mirror the pattern of `vector_router_mode` at `app/config.py:565`)
- Modify: `app/workers/pipeline.py:~8617` (the index-build site)
- Modify: `.env.example` + `.env` (per [[feedback-env-vars-must-appear-in-dotenv-files]])
- Test: `tests/unit/test_pipeline_index_flag.py`

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

```python
# In derive_ontology_graph (worker pipeline.py:~8617):
from app.config import settings
if settings.extraction_index_mode == "merged":
    from app.services.extraction_chunk_index import build_extraction_index_hybrid
    diag = build_extraction_index_hybrid(doc_json_for_index, run_id, doc_id, store)
else:
    diag = build_extraction_index(doc_json_for_index, run_id, doc_id, store)
```

- [ ] **Step 1: Failing test** for default `per_element`, `merged` override, and invalid-value rejection (Literal enforces).
- [ ] **Step 2: Add the Settings field** to `app/config.py`.
- [ ] **Step 3: Wire** the branching at `pipeline.py:~8617`.
- [ ] **Step 4: Add env var** to `.env` and `.env.example` with a short comment.
- [ ] **Step 5: Commit** `feat(worker): EXTRACTION_INDEX_MODE Settings field for merged vs per-element indexer`.

### Task 5: Extend direct-cosine SQL projection for merged-mode columns

**Dependency**: Task 5 must land before Task 6. Task 6 reads `chunk_row["chunk_index"]` / `chunk_row["token_count"]` / source_refs from the dict returned by `search_extraction_chunks_direct`. Without Task 5's extended projection, those keys are absent → `KeyError`. Do NOT parallelize 5 and 6.

**Files:**
- Modify: `app/services/extraction_chunk_search.py:262-434` — `search_extraction_chunks_direct` SQL SELECT (around line 315-320) and result projection.
- Modify: any caller / `GraphEntityResult.properties` construction (line ~407-413) that flattens rows into a dict.
- Test: `tests/integration/test_extraction_chunk_search_merged.py`

**Why**: Pre-flight mandates `VECTOR_ROUTER_RETRIEVAL_MODE=direct` for the A/B (Task 10). Today the direct-cosine SQL at `extraction_chunk_search.py:315-320` selects `chunk_text, embedding, page_number, modality, pipeline_run_id` (and `@rid`, `vertex_id`, `self_ref`). It does NOT select the new `chunk_index`, `source_refs`, `token_count` columns Task 1 adds. Without this extension, Task 6's `chunk_scope` endpoint will `KeyError` on `chunk_row["chunk_index"]` when trying to expand merged chunks.

```sql
-- current (extraction_chunk_search.py:~315):
SELECT chunk_text, embedding, page_number, modality, pipeline_run_id, ...

-- after Task 5:
SELECT chunk_text, embedding, page_number, modality, pipeline_run_id,
       chunk_index, source_refs, token_count,
       ...
```

`GraphEntityResult.properties` must carry the new fields so Task 6 can read them via the dict path it uses today.

- [ ] **Step 1: Failing test** — insert merged-mode rows in fixture; call `search_extraction_chunks_direct`; assert returned dicts contain `chunk_index`, `source_refs`, `token_count`.
- [ ] **Step 2: Extend SELECT** at `extraction_chunk_search.py:~315`.
- [ ] **Step 3: Extend `GraphEntityResult.properties`** flattening to include the new fields.
- [ ] **Step 4: Snapshot test** that per-element-mode rows still flow through unchanged (legacy `chunk_index=-1`, empty `source_refs` — caller handles via `read_chunk_source_refs`).
- [ ] **Step 5: Commit** `feat(extraction-chunk-search): project chunk_index/source_refs/token_count for merged-mode router`.

### Task 6: Chunk-scope endpoint expands `source_refs` and rides merged chunk text on `SelectedChunk.text`

**Files (verified):**
- Modify: `app/api/v1/extraction_routing.py:168-407` (`chunk_scope` endpoint)
- Modify: `app/schemas/extraction_routing.py:82-99` (`ChunkScopeResponse` + `ChunkScopeDiagnostics`)
- Test: `tests/integration/test_chunk_scope_endpoint_merged.py`

**Field name**: existing response field is `self_refs` (NOT `selected_refs`). Worker consumes `self_refs` at `app/workers/pipeline.py:7278` (read) and `:7293-7303` (dispatch).

**`text_by_ref` policy**: in merged mode the endpoint LEAVES `text_by_ref` unchanged (existing self_ref-keyed semantics consumed by `apply_chunk_scope` at `scoped_docling_document.py:225-228`). Merged chunk text rides on a new `SelectedChunk.text` field. The earlier rev-2 proposal to key `text_by_ref` by `chunk_{i}` was wrong — `apply_chunk_scope` reads by self_ref and would silently drop chunk-keyed entries.

```python
# After router selects top-K merged-chunk rows (gated on settings.extraction_index_mode == "merged"):
expanded_refs: list[str] = []
seen: set[str] = set()
selected_chunks: list[SelectedChunk] = []

# Preserve chunk-encounter order — don't lex sort
# ('#/texts/100' would precede '#/texts/35'). apply_chunk_scope reorders
# by body walk so output is correct; we preserve encounter order for
# diagnostics + Phase 2 dispatch.
for chunk_row in selected:    # iteration order = bge-m3+reranker top-K order
    refs_for_chunk = read_chunk_source_refs(chunk_row)
    for ref in refs_for_chunk:
        if ref not in seen:
            seen.add(ref)
            expanded_refs.append(ref)
    selected_chunks.append(SelectedChunk(
        chunk_index=chunk_row["chunk_index"],
        chunk_key=f"chunk_{chunk_row['chunk_index']}",
        text=chunk_row["chunk_text"],           # merged chunk text rides here
        source_refs=refs_for_chunk,
        token_count=chunk_row["token_count"],
    ))

return ChunkScopeResponse(
    mode=existing_mode_value,                # unchanged
    self_refs=expanded_refs,                 # existing field, populated from expansion
    text_by_ref=existing_text_by_ref,        # UNCHANGED — still self_ref-keyed
    selected_chunks=selected_chunks,         # NEW optional field; carries merged chunk text
    diagnostics=ChunkScopeDiagnostics(
        selected_ref_count=len(expanded_refs),
        selected_chunk_count=len(selected),
        expanded_ref_count=len(expanded_refs),
        selected_chunk_token_estimate=sum(c["token_count"] for c in selected),
        selected_token_estimate=...,         # see "legacy field semantics" below
        ...
    ),
)
```

**Legacy `selected_token_estimate` semantics in merged mode** (resolves rev-2 gap): the existing `ChunkScopeDiagnostics.selected_token_estimate` field at `app/schemas/extraction_routing.py:46` was defined for per-element mode as "sum of token estimates across selected per-element chunks." In merged mode, it MUST equal `selected_chunk_token_estimate` so existing dashboards keep working. Two implementation options:

- **Preferred — `@computed_field`**: rewrite `selected_token_estimate` as a pydantic `@computed_field` that returns `selected_chunk_token_estimate` when `index_mode == "merged"` and the existing per-element computation otherwise. Single source of truth; no "two fields with the same value" smell.
- **Fallback — populate both identically** with a `Field(description=...)` on the legacy field explaining the merged-mode equivalence inline. Acceptable but smell-prone.

**Zero-retrieval edge case**: the existing empty-retrieval path at `extraction_routing.py:297-310` already sets `selected_token_estimate = 0` when no chunks are returned, regardless of mode. The merged-mode population path MUST NOT overwrite or double-set this when the candidate list is empty — verify in Step 4's snapshot test.

**`SelectedChunk` pydantic model** (new, in `app/schemas/extraction_routing.py`):

```python
class SelectedChunk(BaseModel):
    """Router-selected merged chunk. Phase 2 worker reads .text directly
    and forwards to docling-graph via the chunked-extract path.
    """
    chunk_index: int           # dense int per pipeline_run_id (see Glossary)
    chunk_key: str             # "chunk_{chunk_index}"; Phase 2 payload pre-staging
    text: str                  # merged chunk text (chunker.contextualize output)
    source_refs: list[str]     # element self_refs covered by this merged chunk
    token_count: int           # tokenizer.count_tokens(text)
```

**Note on `chunk_key` in Phase 1**: this field has no Phase-1 consumer — `apply_chunk_scope` doesn't read it, and Phase 1's LLM-batch path goes through the existing self_ref-keyed pipeline. We populate `chunk_key` in Phase 1 strictly as forward-compat payload pre-staging for Phase 2's `/extract-pass` chunked-mode dispatch. If Phase 2 is cancelled, `chunk_key` can be dropped from `SelectedChunk` with no Phase-1 functional impact.

- [ ] **Step 1: Failing test** asserting:
  - response `self_refs` populated from `read_chunk_source_refs` expansion in chunk-encounter order
  - `selected_chunks[i].text` equals the row's `chunk_text` byte-for-byte
  - `text_by_ref` is UNCHANGED in merged mode (still self_ref-keyed)
  - `diagnostics.selected_chunk_count` and `expanded_ref_count` reported correctly
  - `selected_token_estimate` equals `selected_chunk_token_estimate` in merged mode (backward-compat)
- [ ] **Step 2: Add `SelectedChunk` model + new optional `selected_chunks` field on `ChunkScopeResponse`**. Existing fields untouched.
- [ ] **Step 3: Implement** the merged-mode population path (gated on `settings.extraction_index_mode == "merged"`).
- [ ] **Step 4: Snapshot test** the response shape for both modes — must be backward-compatible.
- [ ] **Step 5: Commit** `feat(chunk-scope): merged-mode expands self_refs and rides merged text on SelectedChunk.text`.

### Task 7: Janitor + cleanup for new vertex_id format

**Files:**
- Modify: `app/services/extraction_chunk_index.py:cleanup_extraction_index` (line ~820)
- Test: `tests/integration/test_extraction_chunk_cleanup_merged.py`

Existing janitor at `extraction_chunk_index.py:820` deletes `WHERE pipeline_run_id = :run_id` — run-id scoped, NOT vertex-id scoped. So the new vertex_id format is automatically handled.

But: TaskList #64 (inline cleanup never fires on COMPLETE) is still open per pre-flight. If unresolved, stale per-element rows from baseline runs survive 24h after the run completes — they would pollute the merged-mode A/B index pool until the periodic janitor sweeps them.

- [ ] **Step 1: Run janitor against a synthetic pipeline_run with merged chunks**; assert all removed by run-id scope.
- [ ] **Step 2: Pre-A/B sweep** — manually trigger janitor before Task 10 A/B to clear all rows older than the experiment start.
- [ ] **Step 3: Commit** if any janitor changes were needed (likely none — run-id scope already covers both formats).

### Task 8: Phase 1 calibration sweep

**Files:**
- New: `notebooks/c10-phase1-merged-chunk-calibration.ipynb` (or script)
- Output: `docs/handoffs/2026-05-27-phase1-merged-chunk-sweep.md`

Sweep dimensions:
- `min_similarity`: 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
- `top_n_candidates`: 25, 50, 75, 100
- `top_k`: 5, 10, 15, 20, 30

`top_n_candidates > top_k` preserved so the reranker has a pool to reorder.

Diagnostics per cell:
- ground-truth coverage at top_k
- `selected_chunk_count`
- `expanded_ref_count`
- `selected_chunk_token_estimate`

- [ ] **Step 1: Run the sweep**.
- [ ] **Step 2: Pick the knee** for each narrowed pass.
- [ ] **Step 3: Write handoff doc**.

### Task 9: Create `air_defense_v3_merged_v1` bundle

**Files:**
- New: `ontology_bundles/air_defense_v3_merged_v1/` (full sibling of `_baseline_subset`)

**Bundle layout reality check (rev-3 correction)**: rev 2 claimed "shared extraction schemas via module imports." That's false — inode check + commit `4b42bad` confirm all three bundles (`air_defense_v3`, `_baseline_subset`, `_narrowing_v1`) have **independent file copies** of extraction_schemas. Creating `_merged_v1` adds a **4th sync surface** — every future schema-description retune (like 4b42bad) now must touch 4 bundles.

**Follow-up debt** (NOT in this plan's scope, but worth noting): extract `extraction_schemas/` into a single shared package importable by all bundles. That's a one-time refactor cheaper than another full copy. File a separate task; not blocking for Phase 1.

Production `SA-2_Sources` points at `air_defense_v3_baseline_subset` per `[[regression-test-source]]` memory.

Decision matrix:

| Bundle | Action | Reason |
|---|---|---|
| `air_defense_v3` (production) | Keep per-element retrieval values | Production stability until Phase 1 A/B passes |
| `air_defense_v3_baseline_subset` | Keep per-element — preserves regression-test baseline | C.10 baselines remain comparable |
| `air_defense_v3_narrowing_v1` | Keep per-element — historical sweep-winner | Comparison reference |
| `air_defense_v3_merged_v1` (NEW) | Sibling of `_baseline_subset`; calibrated values from Task 8 | Phase 1 A/B target |

- [ ] **Step 1: Create directory** with the same 12 schema files as `_baseline_subset` (copy + verify identical content with `diff`).
- [ ] **Step 2: Write `manifest.yaml`** with Task 8 calibrated values.
- [ ] **Step 3: Smoke-test bundle load**: `load_bundle_manifest("air_defense_v3_merged_v1")` succeeds; all 5 passes resolve their template classes.
- [ ] **Step 4: Commit** `feat(bundle): air_defense_v3_merged_v1 sibling for Phase 1 merged-mode A/B`.

### Task 10: Phase 1 A/B against C.10 baseline

**Files:**
- Bundle: `ontology_bundles/air_defense_v3_merged_v1/manifest.yaml` (Task 9)

**Environment lockdown** (per Pre-flight):
- `EXTRACTION_INDEX_MODE=merged`
- `VECTOR_ROUTER_RETRIEVAL_MODE=direct` (already production default; verify env shows `direct`)
- Workers ACTUALLY restarted via `docker restart eip-mmdpp-worker-1 eip-mmdpp-worker-graph-1` (`docker compose restart` from worktree silently no-ops per [[docker-restart-from-worktree-silent-noop]]). Verify with `docker inspect ... --format '{{.State.StartedAt}}'`.
- ExtractionChunk vertex table swept clean of stale per-element rows before kickoff.

- [ ] **Step 1: Trigger Dvina graph_only** with `air_defense_v3_merged_v1` bundle + env lockdown.
- [ ] **Step 2: Trigger SA-2 graph_only** with same.
- [ ] **Step 3: Compare against C.10 baselines** (`cfcc9539` Dvina, `7d46c487` SA-2):
  - Narrowed-pass recall: kinematics + at least one other narrowed pass shows ≥+50% entity count
  - Wall time: ≤120% of baseline (Phase 1 alone is allowed to be slightly slower; Phase 2 closes the gap)
  - Diagnostic delta: `selected_chunk_count` materially lower than baseline `sel_refs` (merged chunks denser)
- [ ] **Step 4: Promote `EXTRACTION_INDEX_MODE=merged`** as default if gates pass.

### Phase 1 gate

After Task 10, **discuss with user** (per [[feedback-phase-discussion-before-implementation]]):
- A/B results
- Whether to proceed to Phase 2 or call it done

---

## Chunk 2: Phase 2 — Direct selected-chunk feed (byte-level identity)

Eliminate docling-graph's downstream re-chunking for narrowed passes. The chunk text the LLM sees is byte-equal to `ExtractionChunk.text` → `SelectedChunk.text` → docling-graph input.

Task 0b already wired the sanitize-skip + chunker-skip path. Phase 2 wires the worker to use it.

### Task 11: Worker forwards `SelectedChunk.text` to docling-graph chunked path

**Files (verified):**
- Modify: `app/workers/pipeline.py:derive_ontology_graph_pass` (the per-pass dispatcher)
- Test: `tests/unit/test_pipeline_chunked_dispatch.py`

**Critical invariant for byte identity** (see Glossary for `SelectedChunk`, `chunk_key`, `source_refs`): the worker reads `text` directly from `SelectedChunk` (returned by `/v1/extraction/chunk-scope` per Task 6). No worker-side ArcadeDB re-read. No worker-side HybridChunker re-run. The chunk text the LLM sees is byte-equal to:
1. What `build_extraction_index_hybrid` wrote into `ExtractionChunk.text`
2. What `/v1/extraction/chunk-scope` returned in `selected_chunks[*].text`
3. What the worker POSTs to docling-graph's chunked path as `selected_chunks[*].text`

Identity holds because step 2 reads from step 1 (direct cosine SQL returns `chunk_text`) and step 3 forwards step 2 verbatim.

Flow:
1. Worker calls `/v1/extraction/chunk-scope` (already happens at `app/workers/pipeline.py:7280+`).
2. Worker reads `response.selected_chunks` (new field from Task 6) — each has `chunk_index`, `chunk_key`, `text`, `source_refs`, `token_count`.
3. Worker POSTs to docling-graph's `/extract-pass` with the optional `selected_chunks` field set (Task 0b wired the receiving side).
4. docling-graph iterates `selected_chunks` directly (Task 0b). NO `DocumentChunker` invocation, NO `_sanitize_docling_document`.

- [ ] **Step 1: Failing test** mocking the chunk-scope response with merged `selected_chunks` and the docling-graph endpoint:
  - asserts the POSTed `selected_chunks[*].text` is byte-equal to `chunk_scope_response.selected_chunks[i].text`
  - asserts no ArcadeDB read for chunk text between chunk-scope and docling-graph dispatch
- [ ] **Step 2: Implement** the forward path in `derive_ontology_graph_pass` (gated on `settings.extraction_index_mode == "merged"`).
- [ ] **Step 3: Sanity check** on a Dvina run — extracted entities' `evidence_units` carry `source_refs` from the producing merged chunk.
- [ ] **Step 4: Commit** `feat(worker): narrowed passes forward chunk text byte-identically to docling-graph`.

### Task 12: Phase 2 A/B against Phase 1

**Bundle:** `air_defense_v3_merged_v1` (same as Phase 1; only the wire-protocol differs).

- [ ] **Step 1: Dvina graph_only with Phase 1 only** (record baseline).
- [ ] **Step 2: Dvina graph_only with Phase 2 enabled** (chunked endpoint).
- [ ] **Step 3: Compare**:
  - Entity counts: should be **identical or higher** (Phase 2 doesn't reduce content)
  - Wall time: should drop (fewer markdown export + rechunk cycles in docling-graph)
  - Provenance: every extracted entity has `source_refs` populated from a real merged chunk's source_refs

### Phase 2 gate

After Task 12, **discuss with user**:
- A/B results
- Trigger to retire per-element path (see Acceptance criteria)

---

## Rollback plan

- `EXTRACTION_INDEX_MODE=per_element` (default until Phase 1 promoted) — falls back to existing `build_extraction_index` + standard `apply_chunk_scope` + standard `/extract-pass` flow.
- ExtractionChunk schema additions (`chunk_index`, `source_refs`, `token_count`) are backward-compatible (default-valued); old per-element rows continue to function.
- If `air_defense_v3_merged_v1` calibration turns out wrong, revert by switching the bundle ref in the trigger — no code change.
- Phase 0 Task 0b is backward-compatible (sanitize-skip only activates when `selected_chunks` is present; absent request keeps existing behavior).

---

## Concrete backwards-compat removal trigger

The `extraction_index_mode` flag and the entire per-element path (`build_extraction_index`, `_walk_docling_elements`, `_render_text_chunk`, the per-element branches in Tasks 4/6) are retired when **all of the following hold**:

1. Phase 1 A/B passes (Task 10 gates) on Dvina + SA-2.
2. Phase 2 A/B passes (Task 12 gates) on Dvina + SA-2.
3. **2 consecutive weeks** of `EXTRACTION_INDEX_MODE=merged` as production default. "No regression" is validated by **manual review of routine traffic** comparing entity/relationship counts and wall-time per pass against the Phase 1 baseline diagnostic capture; this is NOT a scheduled job. If a recurring eval harness is built later it supersedes the manual check.
4. No open bug citing the per-element path.

Retirement task list (one PR):
- Delete `build_extraction_index` + `_walk_docling_elements` + `_render_text_chunk` from `app/services/extraction_chunk_index.py`.
- Delete `extraction_index_mode` Settings field from `app/config.py` and `.env*` entries.
- Remove per-element branches from Task 4 (worker dispatch) and Task 6 (chunk-scope endpoint) wires.
- Drop the `vertex_id_format` parameter on `store.insert_extraction_chunk` (always merged-mode key).
- Drop schema column defaults — `chunk_index`/`source_refs`/`token_count` become non-nullable; backfill UPDATE from Task 1 Step 2b is no longer needed since no per-element rows can exist.
- Remove `read_chunk_source_refs`'s legacy-row branch (was: returns `[]` for `chunk_index = -1`).
- Drop the parity test's per-element-mode snapshot column.

---

## Open questions (to resolve before kickoff)

1. **Should `system_links` use merged chunks?** Non-narrowed today (sees full doc). Recommend: same as today — non-narrowed bypasses the index.

2. **Identity passes (`*_identity`)** are non-narrowed; bypass the index. Phase 1/2 don't change this. Confirm desired.

3. **`evidence_units` schema** in extracted records: does it accept `source_refs: List[str]` or require richer provenance (page_no, span)? Align with field-provenance prompt block (per `docs/superpowers/plans/2026-04-25-flat-schema-profile-refactor.md:3285`).

4. **Document deduplication**: Layer-1's per-element dedup (Rule 3) still applies to `texts[]` before chunking. Per-merged-chunk dedup makes no sense (chunks unique by construction). Verify `build_extraction_index_hybrid` does NOT carry over Layer-2's in-loop dedup.

5. **ArcadeDB list-property for `source_refs`**: `CommunityReport` already uses LIST successfully. Commit to LIST from day one; fall back to JSON-string only if migration fails.

6. **Reranker truncation outcome**: gated by pre-flight measurement. Default decision: keep `max_tokens=512`.

(Closed in rev 4: the `EXTRACTION_INDEX_MODE` naming question. Reviewer suggested `EXTRACTION_CHUNK_GRANULARITY=element|hybrid_merged`. **Decision: keep `EXTRACTION_INDEX_MODE` with values `per_element|merged`.** Rationale: pairs with `vector_router_mode` and `vector_router_retrieval_mode` in `app/config.py`; the subsystem prefix `EXTRACTION_INDEX_*` clearly anchors it to the ExtractionChunk index. `CHUNK_GRANULARITY` would be flatter but loses the subsystem anchor.)

---

## Acceptance criteria summary

| Phase | Gate | Pass condition |
|---|---|---|
| Pre-flight | Determinism + reranker truncation + Path B status | All hold; truncation decision logged with measurement |
| Phase 0 (Tasks 0a, 0b) | Tokenizer pin + sanitize-skip | Both committed in docling-graph; smoke tests pass |
| Phase 1, Tasks 1-7 | Implementation tests | All pass; flag works; response contract preserved |
| Phase 1, Tasks 8-10 | Calibration + A/B | Recall ≥ baseline + ≥1 narrowed pass +50% ent; wall ≤ 120% baseline; `selected_chunk_token_estimate` recorded |
| Phase 2, Tasks 11-12 | A/B + byte identity | Recall ≥ Phase 1; wall < Phase 1; LLM batch input byte-equal to `SelectedChunk.text` for narrowed passes |
| Final | Production rollout | Promote `EXTRACTION_INDEX_MODE=merged`; retirement trigger above |

---

## Diagnostics expected on every merged-mode pass

```json
{
  "selected_chunk_count": 12,            // merged chunks returned by the router
  "selected_ref_count": 87,              // expanded source_refs union (also equals expanded_ref_count)
  "expanded_ref_count": 87,              // alias for clarity in merged mode
  "selected_chunk_token_estimate": 4823, // sum of token_counts across selected chunks
  "selected_token_estimate": 4823,       // backward-compat — equals selected_chunk_token_estimate in merged mode
  "min_similarity": 0.30,
  "top_n_candidates": 50,
  "top_k": 12,
  "index_mode": "merged"
}
```

Keep `selected_ref_count` + `selected_token_estimate` populated so existing dashboards + the `narrowing-ineffective` heuristic (bug #66) continue to work without rewrite.

---

## Time estimate

- Pre-flight: 1-2h (read-only checks)
- Phase 0 (Tasks 0a, 0b): ~0.5 day (docling-graph changes + smoke test)
- Phase 1 (Tasks 1-7 implementation): ~1.5 days
- Phase 1 (Task 8 calibration + Tasks 9-10 A/B): ~1 day (mostly waiting on Dvina + SA-2 runs)
- Phase 2 (Tasks 11-12): ~1 day
- Total: **4-6 days focused effort**, plus calibration wall-time gated on Dvina/SA-2 runs (~5 hours each).

---

## Revision log

### Rev 4 (2026-05-27) — third-pass reviewer findings absorbed

Two new BLOCKERs + nine smaller items from the third 3-reviewer pass:

- **B11**: ArcadeDB ALTER TYPE doesn't auto-fill columns added later. Without explicit UPDATE backfill, legacy rows return `None` for `chunk_index`/`source_refs`/`token_count` → `TypeError` in Task 6's token sum. → Task 1 Step 2b now spec's explicit `UPDATE ExtractionChunk SET … WHERE … IS NULL`; new `read_chunk_token_count` coalescing accessor; Step 2c verification via `DESCRIBE TYPE` + `SELECT count(*) WHERE … IS NULL == 0`.
- **B12**: Task 5 → Task 6 dependency now declared explicitly ("Task 5 must land before Task 6; do NOT parallelize"). Without it, Task 6 KeyErrors on the new dict keys.
- **Q1**: `_looks_like_nav_or_tracking` citation corrected — defined at `main.py:395-434`; called from `_sanitize_docling_document` body at `main.py:437-522`.
- **Q2**: `selected_token_estimate` zero-retrieval edge documented — merged-mode population path must NOT overwrite the empty-retrieval default at `extraction_routing.py:297-310`.
- **Q3**: Removal trigger weakened to "manual review of routine traffic" since no recurring eval harness exists. If a harness is later built it supersedes the manual check.
- **Q4**: Phase 0 atomicity stated explicitly — Task 0a backward-compatible alone; Task 0b adds optional field; both can ship independently; recommended same release for clarity.
- **Q5**: Task 1 Step 2c added — `DESCRIBE TYPE ExtractionChunk` + null-count verification; locked in CI.
- **N1**: `SelectedChunk.chunk_key` Phase-1 orphaning called out explicitly (forward-compat payload pre-staging). On the docling-graph receiver side (Task 0b), `chunk_key` is OMITTED from `SelectedChunkInput` since the receiver iterates by list order.
- **N2**: `selected_token_estimate` legacy field smell addressed — preferred `@computed_field` approach documented; fallback inline `Field(description=...)` allowed.
- **N3**: Open question #7 (`EXTRACTION_INDEX_MODE` naming) closed in-doc — committed to `EXTRACTION_INDEX_MODE` with rationale.
- **N4**: Glossary cross-references added on first task mentions (`see Glossary`).
- **N5**: Rev 4 revision log entry consolidates B11/B12/Q1-Q5/N1-N5; future-reader-friendly.

### Rev 3 (2026-05-27) — second-pass reviewer findings absorbed

Two new BLOCKERs and several quality issues surfaced in the second 3-reviewer pass:

- **B9 (new)**: Direct-cosine SQL projection at `extraction_chunk_search.py:315` doesn't SELECT the new `chunk_index`/`source_refs`/`token_count` columns. Pre-flight mandates `VECTOR_ROUTER_RETRIEVAL_MODE=direct`, so Task 6 would KeyError. → **New Task 5** extends the SELECT projection.
- **B10 (new)**: `apply_chunk_scope` reads `text_by_ref` keyed by `self_ref`, not by `chunk_{i}`. Rev 2's chunk-keyed `text_by_ref` population would be silently dropped. → Task 6 reverts to leaving `text_by_ref` untouched in merged mode; merged chunk text rides on `SelectedChunk.text` directly. Task 11 reads from there.
- **False prerequisites**: TaskList #53-56 and #61 cited in rev-2 pre-flight don't refer to the apply_chunk_scope / Path B work — they're already-completed items in other subsystems. Pre-flight rewritten as read-only inspection of the current code's behavior; no prereqs gated on unmerged TaskList items.
- **Bundle layout claim corrected**: rev 2 said "shared schemas via module imports"; actually all bundles have independent copies (per commit 4b42bad). Task 9 acknowledges the 4-way sync burden and flags an extract-to-shared-package follow-up.
- **Glossary added** at the top of the doc to disambiguate `chunk_index` / `chunk_key` / `source_refs` / `self_refs` / `vertex_id` / `text_by_ref`.
- **Phase 0 prerequisite tasks** extracted from pre-flight verification. Task 0a (tokenizer pin) and Task 0b (sanitize-skip) are now first-class tasks with commits, not `[ ]` verification rows.
- **Concrete backwards-compat removal trigger** spec'd in its own section.
- **`selected_token_estimate` legacy semantics** defined for merged mode (equals `selected_chunk_token_estimate`, preserves dashboard compat).
- **Task numbering** flattened: Tasks 0a, 0b, 1–12 (no fractional numbers).
- **Step splitting**: Task 1 Step 2 split into 2a (declarations) + 2b (migration). Task 2 and Task 3 Step 1 expanded into per-concern Step 1a–1g for bite-sized TDD.
- **Line citations corrected**: `_sanitize_docling_document` at `main.py:437-522` (was `437-516`); ValueError at `many_to_one.py:66` (was `:65`); worker reads `self_refs` at `pipeline.py:7278` (was `:7293-7297` — that's dispatch, not the read site).

### Rev 2 (2026-05-27) — three-reviewer architectural review

Absorbed all 8 BLOCKERs and 9 MAJORs:
- **B1-B8**: response field name `self_refs`, `text_by_ref` reuse (later corrected in rev 3), docling-graph tokenizer pin, sanitize alignment, vertex schema file paths, pydantic Settings pattern, `MergedChunk` rename, error handling spec.
- **M1-M9**: apply_chunk_scope cleanups (later corrected in rev 3), `VECTOR_ROUTER_RETRIEVAL_MODE=direct` mandate, `HuggingFaceTokenizer.from_pretrained` correction, bundle migration scope, Ollama `num_ctx` check, `HybridChunkConfig` parity test, `read_chunk_source_refs` signature, `_raise_tokenizer_max_length` mirror, janitor #65 status check.

### Rev 1 (2026-05-27) — initial review

Earlier user-provided architectural review applied: removed premature 390-token recommendation; pinned exact `DocumentChunker` constructor params; introduced `build_hybrid_chunks_for_extraction` shared helper; preserved existing `mode`/`selected_refs` response contract (later corrected to `self_refs`); ArcadeDB list-property fallback; chunk-encounter ordering; sweep dimensions expanded; byte-identity invariant added to Phase 2.

### Rev 0 (2026-05-27) — initial draft

Initial 2-phase plan based on user's design.
