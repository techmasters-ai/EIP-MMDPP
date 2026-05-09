# Minimal Provenance Fix Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make graph-extraction provenance reliable by sourcing it from the exact chunks the LLM saw, not from a second `HybridChunker()` re-chunk.

**Architecture:** Each extraction-time chunk carries `self_refs`, `evidence_ids`, `evidence_units`, and the chunker's config. That metadata flows through the trace events, the Delta IR normalizer, and the response schemas, so entity / field / relationship provenance can resolve without ever re-chunking. Embedding chunks (Pass A) stay independent in size but persist the same evidence-bearing fields so RAG retrieval can cite source units.

**Tech Stack:** Python 3.11+, Docling/HybridChunker, FastAPI (docling-graph service), Pydantic v2, pytest, Celery (worker-embed), Ollama (BGE-M3 embeddings, gemma4:31b extraction).

**Working directory:** `/home/josh/development/EIP-MMDPP/.worktrees/provenance/`

**Reference:** Implementation handoff in this conversation (12 spec steps); `docling-chunker.md` in repo root for HybridChunker semantics.

---

## Chunk 1: Extraction-time evidence units (Tasks 0–4)

### Task 0: Environment setup + baseline verification

**Why this task exists:** Confirm the worktree has working Python tooling for the docling-graph subrepo and the parent app, and that the existing test suite passes from a clean checkout. Without this baseline, we can't tell whether new failures are ours or pre-existing.

**Files:**
- Read: `docker/docling-graph/repo/pyproject.toml`
- Read: `pyproject.toml` (parent)

**Steps:**

- [ ] **Step 0.1: Install docling-graph subrepo deps**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance/docker/docling-graph/repo
pip install -e '.[dev]' 2>&1 | tail -3
```

Expected: clean install or "already satisfied". If it fails because of missing system deps (e.g. `poppler`), install at the OS level and retry — do NOT skip this step.

- [ ] **Step 0.2: Install parent app deps**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance
pip install -e '.[dev]' 2>&1 | tail -3
```

Expected: clean install.

- [ ] **Step 0.3: Run docling-graph subrepo unit tests as baseline**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance/docker/docling-graph/repo
pytest tests/unit -x -q 2>&1 | tail -20
```

Expected: ALL tests pass (zero failures). If any fail, STOP — investigate before continuing. Capture the count: `<N> passed`.

- [ ] **Step 0.4: Run docling-graph service unit tests as baseline**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance
pytest docker/docling-graph/tests -x -q 2>&1 | tail -20
```

Expected: ALL pass. Capture the count.

- [ ] **Step 0.5: Run parent app unit tests as baseline**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance
pytest tests/unit -x -q 2>&1 | tail -20
```

Expected: ALL pass. Capture the count.

- [ ] **Step 0.6: Record baseline counts**

Append to a temp file `/tmp/provenance-baseline.txt`:
```
docling-graph subrepo: <N> passed
docling-graph service: <N> passed
parent app:            <N> passed
```

These are the floor counts every later task must preserve. No commit needed — this is local scratch.

**Acceptance:** All three test suites pass on a clean `provenance` branch with 0 failures. Baseline counts recorded.

---

### Task 1: Make HybridChunker construction explicit

**Why this task exists:** The current `DocumentChunker` constructs `HybridChunker(tokenizer=…, merge_peers=…)` without passing the table-handling or heading knobs. That makes the production behavior depend on whichever defaults the upstream library happens to ship. Acceptance criterion: every production `HybridChunker` construction is explicit.

**Files:**
- Modify: `docker/docling-graph/repo/docling_graph/core/extractors/document_chunker.py:113-116` (constructor) and `:331-343` (`get_config_summary`)
- Test: `docker/docling-graph/repo/tests/unit/core/extractors/test_document_chunker.py` (new test cases appended)

**Steps:**

- [ ] **Step 1.1: Write failing test — explicit HybridChunker args**

Append to `docker/docling-graph/repo/tests/unit/core/extractors/test_document_chunker.py`:

```python
def test_document_chunker_constructs_hybridchunker_with_explicit_args():
    """Every HybridChunker production construction must pass merge_peers,
    repeat_table_header, omit_header_on_overflow, always_emit_headings
    explicitly so behavior cannot drift on a library default change."""
    from unittest.mock import patch
    with patch(
        "docling_graph.core.extractors.document_chunker.HybridChunker"
    ) as mock_hybrid:
        from docling_graph.core.extractors.document_chunker import DocumentChunker
        DocumentChunker(chunk_max_tokens=512, merge_peers=True)
        kwargs = mock_hybrid.call_args.kwargs
        assert "tokenizer" in kwargs
        assert kwargs["merge_peers"] is True
        assert kwargs["repeat_table_header"] is True
        assert kwargs["omit_header_on_overflow"] is False
        assert kwargs["always_emit_headings"] is False


def test_document_chunker_summary_includes_new_knobs():
    from docling_graph.core.extractors.document_chunker import DocumentChunker
    summary = DocumentChunker(chunk_max_tokens=512).get_config_summary()
    assert summary["repeat_table_header"] is True
    assert summary["omit_header_on_overflow"] is False
    assert summary["always_emit_headings"] is False
    # Existing keys must still be present (backward compatibility).
    assert "tokenizer_name" in summary
    assert "chunk_max_tokens" in summary
    assert "merge_peers" in summary
```

- [ ] **Step 1.2: Run tests — verify they fail**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance/docker/docling-graph/repo
pytest tests/unit/core/extractors/test_document_chunker.py::test_document_chunker_constructs_hybridchunker_with_explicit_args tests/unit/core/extractors/test_document_chunker.py::test_document_chunker_summary_includes_new_knobs -v 2>&1 | tail -10
```

Expected: both FAIL — `KeyError: 'repeat_table_header'` (or similar) and an `AssertionError` on the kwargs check.

- [ ] **Step 1.3: Make HybridChunker construction explicit**

In `docker/docling-graph/repo/docling_graph/core/extractors/document_chunker.py` replace lines 113-116:

```python
# Initialize HybridChunker
self.chunker = HybridChunker(
    tokenizer=self.tokenizer,
    merge_peers=merge_peers,
)
```

with:

```python
# Initialize HybridChunker — every relevant knob passed explicitly
# so production behavior is independent of upstream default drift.
# See docling-chunker.md (repo root) for knob semantics.
self.chunker = HybridChunker(
    tokenizer=self.tokenizer,
    merge_peers=merge_peers,
    repeat_table_header=True,
    omit_header_on_overflow=False,
    always_emit_headings=False,
)
```

Note: `max_tokens` stays on the tokenizer wrapper (HuggingFaceTokenizer/OpenAITokenizer); per `docling-chunker.md`, `HybridChunker.max_tokens` is derived from `tokenizer.get_max_tokens()`.

- [ ] **Step 1.4: Extend `get_config_summary()`**

In the same file, locate the `get_config_summary()` method (around line 331-343) and add the three new keys to its returned dict:

```python
def get_config_summary(self) -> dict:
    return {
        "tokenizer_name": self.tokenizer_name,
        "chunk_max_tokens": self.chunk_max_tokens,
        "merge_peers": self.merge_peers,
        "tokenizer_class": type(self.tokenizer).__name__,
        # NEW — reflect explicit HybridChunker knobs:
        "repeat_table_header": True,
        "omit_header_on_overflow": False,
        "always_emit_headings": False,
    }
```

If `merge_peers` isn't already an instance attribute in `__init__`, also store it: `self.merge_peers = merge_peers`.

- [ ] **Step 1.5: Run tests — verify they pass**

```bash
pytest tests/unit/core/extractors/test_document_chunker.py -v 2>&1 | tail -15
```

Expected: both new tests PASS, all pre-existing `test_document_chunker.py` tests still PASS.

- [ ] **Step 1.6: Run full subrepo unit tests — no regression**

```bash
pytest tests/unit -x -q 2>&1 | tail -5
```

Expected: same `<N> passed` count as Task 0 baseline (plus the 2 new tests).

- [ ] **Step 1.7: Commit**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance/docker/docling-graph/repo
git add docling_graph/core/extractors/document_chunker.py tests/unit/core/extractors/test_document_chunker.py
git commit -m "feat(chunker): pass HybridChunker knobs explicitly + surface in config summary

Removes silent dependence on upstream defaults. repeat_table_header=True,
omit_header_on_overflow=False, always_emit_headings=False match the prior
implicit behavior, but are now part of the contract."
```

**Acceptance:** Both new unit tests pass. `get_config_summary()` returns 3 additional keys. Subrepo baseline test count preserved + 2.

---

### Task 2: Extend extraction chunk metadata with evidence units

**Why this task exists:** Provenance must come from the exact chunks the LLM saw. To resolve a node back to its source DoclingDocument element, we need `self_refs` (which Docling items each chunk references) carried through the pipeline. We bundle that with `evidence_ids` (stable IDs the LLM can cite) and `evidence_units` (text + page payload for human/LLM consumption).

**Files:**
- Modify: `docker/docling-graph/repo/docling_graph/core/extractors/document_processor.py:210-271` (`extract_chunks_with_metadata`)
- Test: `docker/docling-graph/repo/tests/unit/core/extractors/test_document_processor.py`

**Evidence unit shape (used by Steps 2, 3, 4, 8):**

```python
{
    "evidence_id": str,         # == self_ref of the source DoclingItem
    "self_ref": str,            # same value, kept for symmetry
    "page_numbers": list[int],  # pages this item spans
    "text": str,                # item.text or item.orig (whichever is non-empty)
    "label": str | None,        # str(item.label) if present
}
```

**Steps:**

- [ ] **Step 2.1: Write failing test — metadata fields present**

Append to `docker/docling-graph/repo/tests/unit/core/extractors/test_document_processor.py`:

```python
def test_extract_chunks_with_metadata_includes_evidence_fields(small_docling_doc):
    """Every chunk metadata row must carry chunk_kind, self_refs, evidence_ids,
    evidence_units, and chunker_config."""
    from docling_graph.core.extractors.document_processor import DocumentProcessor
    proc = DocumentProcessor(chunker_config={"chunk_max_tokens": 4096})
    chunks, metadata = proc.extract_chunks_with_metadata(small_docling_doc)
    assert len(chunks) == len(metadata)
    for cmeta in metadata:
        assert cmeta["chunk_kind"] == "graph_extraction"
        assert isinstance(cmeta["self_refs"], list)
        assert isinstance(cmeta["evidence_ids"], list)
        assert isinstance(cmeta["evidence_units"], list)
        assert "chunker_config" in cmeta
        # evidence_ids must mirror self_refs (1:1 by construction)
        assert cmeta["evidence_ids"] == cmeta["self_refs"]
        # each evidence_unit must have the documented shape
        for unit in cmeta["evidence_units"]:
            assert {"evidence_id", "self_ref", "page_numbers", "text"} <= unit.keys()
            assert unit["evidence_id"] == unit["self_ref"]


def test_extract_chunks_with_metadata_fallback_split_inherits_evidence(large_docling_doc):
    """When a chunk exceeds chunk_max_tokens and is split via chunk_text_fallback,
    every sub-chunk inherits the parent's self_refs / evidence_ids / evidence_units."""
    from docling_graph.core.extractors.document_processor import DocumentProcessor
    # Force fallback splitting by capping max_tokens very low.
    proc = DocumentProcessor(chunker_config={"chunk_max_tokens": 64})
    chunks, metadata = proc.extract_chunks_with_metadata(large_docling_doc)
    # Find groups of sub-chunks that share a parent (same self_refs set).
    from collections import defaultdict
    by_refs = defaultdict(list)
    for cmeta in metadata:
        by_refs[tuple(sorted(cmeta["self_refs"]))].append(cmeta)
    fallback_groups = [g for g in by_refs.values() if len(g) > 1]
    assert fallback_groups, "fixture should produce at least one fallback split"
    for group in fallback_groups:
        # All sub-chunks share evidence metadata.
        first = group[0]
        for sib in group[1:]:
            assert sib["self_refs"] == first["self_refs"]
            assert sib["evidence_ids"] == first["evidence_ids"]
            assert sib["evidence_units"] == first["evidence_units"]
```

If fixtures `small_docling_doc` / `large_docling_doc` don't exist, add them to `tests/unit/core/extractors/conftest.py`:

```python
import pytest
from docling_core.types.doc.document import DoclingDocument, TextItem
# Use existing test fixtures elsewhere in repo if available; otherwise build
# a minimal DoclingDocument with N TextItems where item.text length forces
# the desired chunk count. See tests/fixtures/ for prior examples.

@pytest.fixture
def small_docling_doc():
    """Tiny doc, 3-5 text items, fits in one HybridChunker chunk at 4096 tokens."""
    # If a sample DoclingDocument JSON exists in tests/fixtures, prefer
    # loading that via DoclingDocument.model_validate(json.load(f)).
    ...

@pytest.fixture
def large_docling_doc():
    """Doc with at least one text item whose enriched length exceeds 64 tokens
    so chunk_text_fallback() fires and produces 2+ sub-chunks for one source item."""
    ...
```

If realistic fixtures are non-trivial to build, search `docker/docling-graph/repo/tests/fixtures/` for existing JSON exports of DoclingDocument and reuse them; reach for the simplest one that satisfies each constraint.

- [ ] **Step 2.2: Run tests — verify they fail**

```bash
pytest tests/unit/core/extractors/test_document_processor.py::test_extract_chunks_with_metadata_includes_evidence_fields tests/unit/core/extractors/test_document_processor.py::test_extract_chunks_with_metadata_fallback_split_inherits_evidence -v 2>&1 | tail -15
```

Expected: both FAIL — `KeyError: 'chunk_kind'` (or similar).

- [ ] **Step 2.3: Build a `_evidence_units_for_chunk` helper**

In `docker/docling-graph/repo/docling_graph/core/extractors/document_processor.py`, near the top of the file (after imports), add:

```python
def _evidence_units_for_chunk(chunk_obj) -> list[dict]:
    """Extract evidence units from a DocChunk's doc_items.

    Each unit references one source DoclingItem (text/table/picture). The
    LLM can cite an evidence_id verbatim and the resolver can map it back
    to the document element via DoclingDocument.lookup(self_ref).
    """
    units: list[dict] = []
    items = getattr(getattr(chunk_obj, "meta", None), "doc_items", None) or []
    for item in items:
        ref = getattr(item, "self_ref", None)
        if not isinstance(ref, str) or not ref:
            continue
        prov = getattr(item, "prov", None) or []
        page_numbers = sorted({
            p.page_no for p in prov if getattr(p, "page_no", None) is not None
        })
        text = getattr(item, "text", None) or getattr(item, "orig", None) or ""
        label = getattr(item, "label", None)
        units.append({
            "evidence_id": ref,
            "self_ref": ref,
            "page_numbers": page_numbers,
            "text": text,
            "label": str(label) if label is not None else None,
        })
    return units
```

- [ ] **Step 2.4: Extend the metadata dict in `extract_chunks_with_metadata`**

In `extract_chunks_with_metadata()`, locate the per-chunk metadata construction (around lines 247-253 and the fallback path at 259-266) and extend each metadata dict.

For the structure-aware (non-fallback) chunk:

```python
evidence_units = _evidence_units_for_chunk(chunk_obj)
self_refs = [u["evidence_id"] for u in evidence_units]
chunker_config = self.chunker.get_config_summary()
metadata.append({
    "chunk_id": chunk_id,
    "chunk_kind": "graph_extraction",
    "page_numbers": page_numbers,
    "token_count": enriched_tokens,
    "self_refs": self_refs,
    "evidence_ids": list(self_refs),       # 1:1 with self_refs by construction
    "evidence_units": evidence_units,
    "chunker_config": chunker_config,
})
```

For the fallback-split path: COMPUTE `evidence_units`, `self_refs`, `chunker_config` once per parent chunk (outside the inner `for sub_text in sub_chunks` loop), then attach the SAME object to every sub-chunk metadata. Example:

```python
# Outside the inner loop:
parent_units = _evidence_units_for_chunk(chunk_obj)
parent_refs = [u["evidence_id"] for u in parent_units]
parent_cfg = self.chunker.get_config_summary()

# Inside the per-sub-chunk loop:
metadata.append({
    "chunk_id": sub_chunk_id,
    "chunk_kind": "graph_extraction",
    "page_numbers": page_numbers,
    "token_count": sub_token_count,
    "self_refs": list(parent_refs),
    "evidence_ids": list(parent_refs),
    "evidence_units": list(parent_units),  # shallow copy is fine — units are read-only downstream
    "chunker_config": parent_cfg,
})
```

Use `list(...)` not the same reference, so accidental in-place mutation downstream doesn't propagate to siblings.

- [ ] **Step 2.5: Run tests — verify they pass**

```bash
pytest tests/unit/core/extractors/test_document_processor.py -v 2>&1 | tail -15
```

Expected: both new tests PASS, all pre-existing tests in this file still PASS.

- [ ] **Step 2.6: Run full subrepo unit tests — no regression**

```bash
pytest tests/unit -x -q 2>&1 | tail -5
```

Expected: same baseline count + 2 (Task 1) + 2 (this task).

- [ ] **Step 2.7: Commit**

```bash
git add docling_graph/core/extractors/document_processor.py tests/unit/core/extractors/test_document_processor.py tests/unit/core/extractors/conftest.py
git commit -m "feat(extractor): emit self_refs / evidence_ids / evidence_units per chunk

Each chunk metadata row now carries the source DoclingDocument items it
covers, so downstream provenance can resolve without a re-chunk pass.
Fallback-split sub-chunks inherit the parent chunk's evidence metadata."
```

**Acceptance:** Both new unit tests pass. Each metadata row includes `chunk_kind`, `self_refs`, `evidence_ids`, `evidence_units`, `chunker_config`. Fallback sub-chunks inherit parent evidence.

---

### Task 3: Embed evidence markers in LLM chunk text

**Why this task exists:** For the LLM to cite a stable evidence ID, the IDs must be visible to it inside the chunk text. We append a compact `=== EVIDENCE UNITS ===` block to each chunk before it leaves `extract_chunks_with_metadata`. The marker format is intentionally minimal so it costs few tokens.

**Files:**
- Modify: `docker/docling-graph/repo/docling_graph/core/extractors/document_processor.py:210-271` (same function as Task 2, append-only change to chunk text)
- Test: `docker/docling-graph/repo/tests/unit/core/extractors/test_document_processor.py`

**Marker format:**

```text
<chunk text as before>

=== EVIDENCE UNITS ===
[EVIDENCE id="#/texts/12" page="3"]
Missile diameter: 0.37 m
[/EVIDENCE]
[EVIDENCE id="#/texts/13" page="3"]
…
[/EVIDENCE]
```

Each unit gets one `[EVIDENCE id="…" page="…"]…[/EVIDENCE]` block. Multi-page units list pages comma-separated: `page="3,4"`. Units with empty `text` are skipped (nothing to quote).

**Steps:**

- [ ] **Step 3.1: Write failing test — marker block present**

Append to `tests/unit/core/extractors/test_document_processor.py`:

```python
def test_extract_chunks_text_contains_evidence_markers(small_docling_doc):
    """Each emitted chunk text must end with an EVIDENCE UNITS block listing
    each unit it carries, so the LLM can cite stable IDs."""
    from docling_graph.core.extractors.document_processor import DocumentProcessor
    proc = DocumentProcessor(chunker_config={"chunk_max_tokens": 4096})
    chunks, metadata = proc.extract_chunks_with_metadata(small_docling_doc)
    for chunk_text, cmeta in zip(chunks, metadata):
        if not cmeta["evidence_units"]:
            continue  # no items → no marker block
        assert "=== EVIDENCE UNITS ===" in chunk_text
        for unit in cmeta["evidence_units"]:
            if not unit["text"]:
                continue
            marker = f'[EVIDENCE id="{unit["evidence_id"]}"'
            assert marker in chunk_text, (
                f"chunk missing marker for {unit['evidence_id']}"
            )
            assert "[/EVIDENCE]" in chunk_text
```

- [ ] **Step 3.2: Run test — verify it fails**

```bash
pytest tests/unit/core/extractors/test_document_processor.py::test_extract_chunks_text_contains_evidence_markers -v 2>&1 | tail -10
```

Expected: FAIL — `=== EVIDENCE UNITS ===` not in chunk text.

- [ ] **Step 3.3: Add a `_render_evidence_block` helper**

In `document_processor.py`, near `_evidence_units_for_chunk`:

```python
def _render_evidence_block(units: list[dict]) -> str:
    """Render evidence units into the marker block appended to chunk text."""
    if not units:
        return ""
    lines = ["", "=== EVIDENCE UNITS ==="]
    for unit in units:
        text = unit.get("text") or ""
        if not text.strip():
            continue
        pages = unit.get("page_numbers") or []
        page_attr = ",".join(str(p) for p in pages) if pages else ""
        page_clause = f' page="{page_attr}"' if page_attr else ""
        lines.append(f'[EVIDENCE id="{unit["evidence_id"]}"{page_clause}]')
        lines.append(text)
        lines.append("[/EVIDENCE]")
    return "\n".join(lines) if len(lines) > 2 else ""
```

- [ ] **Step 3.4: Append the block to chunk text**

In `extract_chunks_with_metadata()`, both in the structure-aware path and the fallback-split path, after the chunk text is finalized but before the metadata is appended, do:

```python
evidence_block = _render_evidence_block(evidence_units)
chunk_text_with_evidence = chunk_text + evidence_block

chunks.append(chunk_text_with_evidence)
# … then the metadata.append(...) from Task 2
```

For the fallback path, use `parent_units` (the same object inherited by all sub-chunks): every sub-chunk gets the same marker block. The LLM can cite any evidence_id from any sub-chunk; we don't try to per-sub-chunk slice the markers because Stage-3 splits are arbitrary text breakpoints — slicing markers would be unsound.

Update the per-chunk token count to use the post-marker text:

```python
enriched_tokens = self.tokenizer.count_tokens(chunk_text_with_evidence)
```

Move the `token_count` field to use the post-marker count. NOTE: this may push a chunk over `chunk_max_tokens` after marker addition; that's acceptable per the spec ("graph extraction chunks may stay large"). If you observe runaway sizes from over-rich evidence units, future work can post-trim, but the current scope leaves this alone.

- [ ] **Step 3.5: Run tests — verify pass**

```bash
pytest tests/unit/core/extractors/test_document_processor.py -v 2>&1 | tail -15
```

Expected: PASS for the new test plus all prior tests.

- [ ] **Step 3.6: Run full subrepo unit tests — no regression**

```bash
pytest tests/unit -x -q 2>&1 | tail -5
```

Expected: baseline + 5 (1, 2, 2 from Tasks 1 & 2 & 3).

- [ ] **Step 3.7: Commit**

```bash
git add docling_graph/core/extractors/document_processor.py tests/unit/core/extractors/test_document_processor.py
git commit -m "feat(extractor): append [EVIDENCE id=…] markers to chunk text

LLM can now cite stable evidence_ids that map back to DoclingDocument
self_refs without prompt-side string matching. Marker block is appended
once per chunk; fallback sub-chunks share the parent's evidence units."
```

**Acceptance:** Each chunk text containing items ends with an `=== EVIDENCE UNITS ===` block. Each unit (with non-empty text) appears as `[EVIDENCE id="…" page="…"]…[/EVIDENCE]`. Token counts in metadata reflect the post-marker text.

---

### Task 4: Emit evidence metadata in `chunk_created` trace events

**Why this task exists:** `main.py` needs to recover extraction-time provenance after `run_pipeline` completes. The cleanest channel is the trace stream that's already emitted per chunk — we just need to widen its payload. With this in place, Step 5 can stop calling `_build_chunk_to_self_refs_map` (the bare `HybridChunker()` re-chunk).

**Files:**
- Modify: `docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/strategy_ops.py:40-51`
- Test: `docker/docling-graph/repo/tests/unit/core/extractors/contracts/delta/test_strategy_ops.py` (create if needed)

**Steps:**

- [ ] **Step 4.1: Write failing test — trace payload widened**

Create or append `docker/docling-graph/repo/tests/unit/core/extractors/contracts/delta/test_strategy_ops.py`:

```python
def test_chunk_created_trace_includes_evidence_metadata():
    """The chunk_created trace event must carry chunk_kind, self_refs,
    evidence_ids, evidence_units, and chunker_config so downstream
    consumers (main.py) can resolve provenance without re-chunking."""
    from unittest.mock import MagicMock
    from docling_graph.core.extractors.contracts.delta.strategy_ops import (
        _emit_chunk_created_traces,  # extracted in Step 4.3
    )
    trace = MagicMock()
    chunk_metadata = [{
        "chunk_id": 0,
        "chunk_kind": "graph_extraction",
        "token_count": 42,
        "page_numbers": [1, 2],
        "self_refs": ["#/texts/0", "#/texts/1"],
        "evidence_ids": ["#/texts/0", "#/texts/1"],
        "evidence_units": [
            {"evidence_id": "#/texts/0", "self_ref": "#/texts/0",
             "page_numbers": [1], "text": "hello", "label": "text"},
        ],
        "chunker_config": {"chunk_max_tokens": 4096, "merge_peers": True},
    }]
    chunks = ["hello world\n=== EVIDENCE UNITS ===\n[EVIDENCE id=\"#/texts/0\" page=\"1\"]\nhello\n[/EVIDENCE]"]
    _emit_chunk_created_traces(trace, chunk_metadata, chunks)
    trace.emit.assert_called_once()
    args, _ = trace.emit.call_args
    name, scope, payload = args
    assert name == "chunk_created"
    assert payload["chunk_kind"] == "graph_extraction"
    assert payload["self_refs"] == ["#/texts/0", "#/texts/1"]
    assert payload["evidence_ids"] == ["#/texts/0", "#/texts/1"]
    assert len(payload["evidence_units"]) == 1
    assert payload["chunker_config"]["chunk_max_tokens"] == 4096
```

- [ ] **Step 4.2: Run test — verify it fails**

```bash
pytest tests/unit/core/extractors/contracts/delta/test_strategy_ops.py -v 2>&1 | tail -10
```

Expected: FAIL — `_emit_chunk_created_traces` doesn't exist.

- [ ] **Step 4.3: Extract the trace emission into a named helper + extend payload**

In `docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/strategy_ops.py`, replace the inline trace block at lines 40-51 with:

```python
def _emit_chunk_created_traces(trace_data, chunk_metadata, chunks) -> None:
    """Emit one chunk_created event per chunk, carrying enough metadata
    to recover provenance downstream (no re-chunking required)."""
    for cmeta, chunk_text in zip(chunk_metadata, chunks, strict=False):
        trace_data.emit(
            "chunk_created",
            "extraction",
            {
                "chunk_id": cmeta.get("chunk_id"),
                "chunk_kind": cmeta.get("chunk_kind"),
                "token_count": cmeta.get("token_count"),
                "page_numbers": cmeta.get("page_numbers"),
                "self_refs": cmeta.get("self_refs"),
                "evidence_ids": cmeta.get("evidence_ids"),
                "evidence_units": cmeta.get("evidence_units"),
                "chunker_config": cmeta.get("chunker_config"),
                "text_content": chunk_text,
            },
        )
```

Then at the original call site (around line 40):

```python
if trace_data is not None:
    _emit_chunk_created_traces(trace_data, chunk_metadata, chunks)
```

- [ ] **Step 4.4: Run test — verify pass**

```bash
pytest tests/unit/core/extractors/contracts/delta/test_strategy_ops.py -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 4.5: Run full subrepo unit tests — no regression**

```bash
pytest tests/unit -x -q 2>&1 | tail -5
```

Expected: baseline + 6 new tests across Tasks 1-4.

- [ ] **Step 4.6: Commit**

```bash
git add docling_graph/core/extractors/contracts/delta/strategy_ops.py tests/unit/core/extractors/contracts/delta/test_strategy_ops.py
git commit -m "feat(trace): widen chunk_created payload with evidence metadata

Adds chunk_kind, self_refs, evidence_ids, evidence_units, and
chunker_config to every chunk_created trace event. main.py can now
recover extraction-time provenance from the trace stream instead of
re-chunking with a bare HybridChunker()."
```

**Acceptance:** `chunk_created` trace event payload includes all 5 new keys. Pre-existing keys (`chunk_id`, `token_count`, `page_numbers`, `text_content`) preserved.

---

## Chunk 2: Provenance pipeline (Tasks 5–8)

### Task 5: Replace re-chunking with trace-based `chunk_to_self_refs`

**Why this task exists:** `_build_chunk_to_self_refs_map()` calls `HybridChunker()` with no arguments — the chunks it produces have NO relationship to the chunks the LLM actually saw. Different `max_tokens` → different boundaries → wrong `self_refs` mapping → silently broken provenance. With Task 4 done, the trace stream now carries the real chunk → self_refs map; this task wires `main.py` to read from there.

**Files:**
- Modify: `docker/docling-graph/app/main.py:755-761` (assignment site) and `:879-913` (the helper itself)
- Test: `docker/docling-graph/tests/test_main_provenance_source.py` (create)

**Steps:**

- [ ] **Step 5.1: Write failing test — provenance built from trace, not re-chunk**

Create `docker/docling-graph/tests/test_main_provenance_source.py`:

```python
def test_chunk_to_self_refs_built_from_trace_events(monkeypatch):
    """main.py must build context._chunk_to_self_refs from chunk_created
    trace events (extraction-time), NOT by re-chunking with HybridChunker()."""
    from app.main import _chunk_to_self_refs_from_trace
    trace_events = [
        ("chunk_created", "extraction", {
            "chunk_id": 0,
            "self_refs": ["#/texts/0", "#/texts/1"],
            "evidence_units": [],
        }),
        ("chunk_created", "extraction", {
            "chunk_id": 1,
            "self_refs": ["#/texts/2"],
            "evidence_units": [],
        }),
    ]
    out = _chunk_to_self_refs_from_trace(trace_events)
    assert out == {0: ["#/texts/0", "#/texts/1"], 1: ["#/texts/2"]}


def test_no_production_path_calls_bare_hybridchunker():
    """Grep main.py for `HybridChunker()` (no args) — only diagnostic
    fallback paths may use it, marked with a `# diagnostic-only:` comment."""
    import pathlib
    main_py = pathlib.Path("docker/docling-graph/app/main.py").read_text()
    # Naive but effective: any line with `HybridChunker()` must include
    # the diagnostic-only marker on the same line or the prior line.
    lines = main_py.splitlines()
    bad = []
    for i, line in enumerate(lines):
        if "HybridChunker()" not in line:
            continue
        prev = lines[i - 1] if i > 0 else ""
        if "diagnostic-only" in line or "diagnostic-only" in prev:
            continue
        bad.append((i + 1, line.strip()))
    assert not bad, f"production HybridChunker() call sites: {bad}"
```

- [ ] **Step 5.2: Run tests — verify they fail**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance
pytest docker/docling-graph/tests/test_main_provenance_source.py -v 2>&1 | tail -10
```

Expected: FAIL — `_chunk_to_self_refs_from_trace` doesn't exist; the second test fails because line 899 is a bare construction without the marker.

- [ ] **Step 5.3: Add the trace-based builder**

In `docker/docling-graph/app/main.py`, ABOVE `_build_chunk_to_self_refs_map`, add:

```python
def _chunk_to_self_refs_from_trace(trace_events) -> dict[int, list[str]]:
    """Build {chunk_id: [self_ref, ...]} from chunk_created trace events.

    Authoritative provenance source — uses the EXACT chunks the LLM saw,
    keyed by their extraction-time chunk_id. Replaces the deprecated
    re-chunking path that produced a different boundary set.
    """
    out: dict[int, list[str]] = {}
    for evt in trace_events or []:
        # trace_events may be tuples (name, scope, payload) OR dicts —
        # support both shapes the framework emits.
        if isinstance(evt, tuple) and len(evt) >= 3:
            name, _scope, payload = evt[0], evt[1], evt[2]
        elif isinstance(evt, dict):
            name = evt.get("event") or evt.get("name")
            payload = evt.get("payload") or evt
        else:
            continue
        if name != "chunk_created":
            continue
        cid = payload.get("chunk_id")
        refs = payload.get("self_refs")
        if cid is None or not isinstance(refs, list):
            continue
        out[int(cid)] = [r for r in refs if isinstance(r, str)]
    return out


def _chunk_to_evidence_units_from_trace(trace_events) -> dict[int, list[dict]]:
    """Build {chunk_id: [evidence_unit, ...]} from chunk_created trace events."""
    out: dict[int, list[dict]] = {}
    for evt in trace_events or []:
        if isinstance(evt, tuple) and len(evt) >= 3:
            name, _scope, payload = evt[0], evt[1], evt[2]
        elif isinstance(evt, dict):
            name = evt.get("event") or evt.get("name")
            payload = evt.get("payload") or evt
        else:
            continue
        if name != "chunk_created":
            continue
        cid = payload.get("chunk_id")
        units = payload.get("evidence_units")
        if cid is None or not isinstance(units, list):
            continue
        out[int(cid)] = list(units)
    return out
```

- [ ] **Step 5.4: Mark `_build_chunk_to_self_refs_map` diagnostic-only**

Update its docstring and the bare `HybridChunker()` line:

```python
def _build_chunk_to_self_refs_map(docling_document: Any) -> dict[int, list[str]] | None:
    """DIAGNOSTIC-ONLY fallback. Re-chunks the document with a default
    HybridChunker.

    DO NOT use as a normal provenance source — the re-chunked boundaries
    do NOT match the extraction-time chunks (different default max_tokens,
    independent merge_peers state). Trace-event-derived
    `_chunk_to_self_refs_from_trace` is authoritative.
    """
    if docling_document is None:
        return None
    try:
        from docling.chunking import HybridChunker
    except ImportError as exc:
        logger.warning("HybridChunker import failed; provenance self_refs unavailable: %s", exc)
        return None

    try:
        chunker = HybridChunker()  # diagnostic-only: re-chunk for fallback only
        # … rest unchanged …
```

The `# diagnostic-only:` comment on the same line satisfies the test in Step 5.1.

- [ ] **Step 5.5: Switch the call site to trace-based source**

At the assignment site (around lines 755-761), replace:

```python
chunk_to_self_refs = _build_chunk_to_self_refs_map(
    getattr(context, "docling_document", None)
)
try:
    context._chunk_to_self_refs = chunk_to_self_refs
except AttributeError:
    pass
```

with:

```python
trace_events = getattr(getattr(context, "trace", None), "events", None) or []
chunk_to_self_refs = _chunk_to_self_refs_from_trace(trace_events)
chunk_to_evidence_units = _chunk_to_evidence_units_from_trace(trace_events)

if not chunk_to_self_refs:
    logger.warning(
        "no chunk_created trace events found — provenance will be empty. "
        "Check that strategy_ops._emit_chunk_created_traces ran."
    )
try:
    context._chunk_to_self_refs = chunk_to_self_refs
    context._chunk_to_evidence_units = chunk_to_evidence_units
except AttributeError:
    pass
```

The exact attribute path to trace events depends on how the docling-graph framework exposes them — verify by reading `context.trace` shape via a one-off `print` test if needed. If the framework uses `context.delta_trace` or `context._trace`, adjust accordingly. The failing-fast warning lets us catch wiring bugs immediately.

- [ ] **Step 5.6: Confirm fallback at line 596 is updated**

In `main.py`, the existing `ctx._chunk_to_self_refs = None` fallback (around line 596) when no document is available — leave it as-is. Add a parallel `ctx._chunk_to_evidence_units = {}` line right next to it, so consumers always see a dict shape.

- [ ] **Step 5.7: Run tests — verify pass**

```bash
pytest docker/docling-graph/tests/test_main_provenance_source.py -v 2>&1 | tail -10
```

Expected: PASS for both tests.

- [ ] **Step 5.8: Run all docling-graph service tests — no regression**

```bash
pytest docker/docling-graph/tests -x -q 2>&1 | tail -5
```

Expected: baseline + 2.

- [ ] **Step 5.9: Commit**

```bash
git add docker/docling-graph/app/main.py docker/docling-graph/tests/test_main_provenance_source.py
git commit -m "fix(provenance): build chunk_to_self_refs from trace, not re-chunk

Bare HybridChunker() re-chunk produced boundaries unrelated to the
chunks the LLM actually saw — silently broken provenance. Now reads
self_refs and evidence_units from chunk_created trace events
(extraction-time, authoritative).

_build_chunk_to_self_refs_map kept as diagnostic-only fallback,
marked with explicit comment so the regression test catches any new
production callers."
```

**Acceptance:** `_chunk_to_self_refs_from_trace` builds the map from trace events. Production path no longer calls bare `HybridChunker()`. The regression test (`test_no_production_path_calls_bare_hybridchunker`) prevents new violations.

---

### Task 6: Add evidence fields to Delta IR models

**Why this task exists:** Steps 7 (prompt) and 8 (normalizer) want the LLM to return `evidence_ids` per node and per relationship. The Pydantic models that catch the LLM's JSON output need to accept those fields. This is a small, additive Pydantic change — pure schema work.

**Files:**
- Modify: `docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/models.py:29-68`
- Test: `docker/docling-graph/repo/tests/unit/core/extractors/contracts/delta/test_models.py` (create or extend)

**Steps:**

- [ ] **Step 6.1: Write failing test — fields accept evidence**

Create or append `docker/docling-graph/repo/tests/unit/core/extractors/contracts/delta/test_models.py`:

```python
def test_delta_node_accepts_evidence_fields():
    from docling_graph.core.extractors.contracts.delta.models import DeltaNode
    n = DeltaNode(
        path="/foo/bar",
        ids={"id_field": "X1"},
        evidence_ids=["#/texts/0", "#/texts/1"],
        property_evidence={"name": ["#/texts/0"], "designation": ["#/texts/1"]},
    )
    assert n.evidence_ids == ["#/texts/0", "#/texts/1"]
    assert n.property_evidence == {"name": ["#/texts/0"], "designation": ["#/texts/1"]}


def test_delta_node_evidence_fields_default_empty():
    """Backward-compat: existing payloads without evidence fields parse cleanly."""
    from docling_graph.core.extractors.contracts.delta.models import DeltaNode
    n = DeltaNode(path="/foo/bar", ids={"id_field": "X1"})
    assert n.evidence_ids == []
    assert n.property_evidence == {}


def test_delta_relationship_accepts_evidence_ids():
    from docling_graph.core.extractors.contracts.delta.models import DeltaRelationship
    r = DeltaRelationship(
        edge_label="HAS_PART",
        source_path="/foo",
        target_path="/foo/bar",
        evidence_ids=["#/texts/2"],
    )
    assert r.evidence_ids == ["#/texts/2"]


def test_delta_relationship_evidence_default_empty():
    from docling_graph.core.extractors.contracts.delta.models import DeltaRelationship
    r = DeltaRelationship(edge_label="HAS_PART", source_path="/a", target_path="/b")
    assert r.evidence_ids == []
```

- [ ] **Step 6.2: Run tests — verify fail**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance/docker/docling-graph/repo
pytest tests/unit/core/extractors/contracts/delta/test_models.py -v 2>&1 | tail -10
```

Expected: FAIL — fields don't exist.

- [ ] **Step 6.3: Add fields to `DeltaNode` and `DeltaRelationship`**

In `docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/models.py`:

In `DeltaNode` (around lines 29-52), add:

```python
evidence_ids: list[str] = Field(default_factory=list)
property_evidence: dict[str, list[str]] = Field(default_factory=dict)
```

In `DeltaRelationship` (around lines 54-68), add:

```python
evidence_ids: list[str] = Field(default_factory=list)
```

Both fields are additive with defaults — no validators needed; existing payloads parse unchanged.

- [ ] **Step 6.4: Run tests — verify pass**

```bash
pytest tests/unit/core/extractors/contracts/delta/test_models.py -v 2>&1 | tail -10
```

Expected: 4 PASS.

- [ ] **Step 6.5: Run full subrepo unit tests — no regression**

```bash
pytest tests/unit -x -q 2>&1 | tail -5
```

Expected: baseline + 6 (Tasks 1-4) + 4 (this task) = baseline + 10.

- [ ] **Step 6.6: Commit**

```bash
git add docling_graph/core/extractors/contracts/delta/models.py tests/unit/core/extractors/contracts/delta/test_models.py
git commit -m "feat(delta-ir): add evidence_ids + property_evidence fields

DeltaNode and DeltaRelationship now accept LLM-emitted evidence IDs.
Both fields default to empty so existing payloads parse unchanged."
```

**Acceptance:** Both models accept and default the new fields. Existing payloads unaffected.

---

### Task 7: Update delta extraction prompt to request evidence IDs

**Why this task exists:** Without the prompt asking for `evidence_ids` and `property_evidence`, the LLM won't emit them and Step 8's normalizer will always fall back to batch-level provenance. This task is the prompt + schema update.

**Files:**
- Modify: `docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/prompts.py:40-68`
- Test: `docker/docling-graph/repo/tests/unit/core/extractors/contracts/delta/test_prompts.py` (create)

**Prompt-engineering constraint per the spec:** Do NOT ask the model to nest provenance inside property values — the current normalizer strips nested properties. Evidence stays at the node and relationship level only.

**Steps:**

- [ ] **Step 7.1: Write failing test — prompt mentions evidence**

Create `docker/docling-graph/repo/tests/unit/core/extractors/contracts/delta/test_prompts.py`:

```python
def test_delta_prompt_requests_evidence_ids():
    """The user prompt's JSON schema instruction must require evidence_ids
    on each node and each relationship, plus property_evidence on nodes."""
    from docling_graph.core.extractors.contracts.delta import prompts
    # Prompts module exposes the schema instruction string. Find it via the
    # module's exported builder OR by reading the module source directly.
    # If a builder function exists, call it; if not, read the file source.
    import inspect, pathlib
    src = pathlib.Path(inspect.getsourcefile(prompts)).read_text()
    # Required mentions:
    assert "evidence_ids" in src, "prompt must mention evidence_ids"
    assert "property_evidence" in src, "prompt must mention property_evidence"
    # Anti-pattern: nested provenance inside property values (normalizer strips).
    assert "evidence inside property values" not in src
```

- [ ] **Step 7.2: Run test — verify fail**

```bash
pytest tests/unit/core/extractors/contracts/delta/test_prompts.py -v 2>&1 | tail -5
```

Expected: FAIL — `evidence_ids` not present.

- [ ] **Step 7.3: Update the prompt**

In `docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/prompts.py`, locate the JSON schema instruction (around line 67) and extend the per-node and per-relationship descriptions. The current line:

```python
'Return JSON: {"nodes": [...], "relationships": [...]} with each node: {path, node_type?, ids, parent, properties}.'
```

Replace with:

```python
'Return JSON: {"nodes": [...], "relationships": [...]}.\n'
'Each node: {path, node_type?, ids, parent, properties, evidence_ids, property_evidence}.\n'
'  - evidence_ids: list of evidence_id strings (from [EVIDENCE id="…"] markers in the chunk text) supporting this node\'s identity.\n'
'  - property_evidence: object mapping each property field name to a list of evidence_ids supporting that field\'s value. Omit fields you cannot cite.\n'
'Each relationship: {edge_label, source_path, source_ids, target_path, target_ids, properties, evidence_ids}.\n'
'  - evidence_ids: list of evidence_id strings supporting this relationship.\n'
'Use ONLY evidence_id values that appear in [EVIDENCE id="…"] markers above. Do not invent IDs. Do not nest evidence inside property values.'
```

Add this addition to the system prompt (around lines 20-38) too, so the model knows the rule from the start:

```python
"- Evidence: every node must include evidence_ids citing the [EVIDENCE id=\"…\"] markers in the chunk; every property in property_evidence must map to one or more of those IDs. Never invent IDs. Relationships also include evidence_ids."
```

- [ ] **Step 7.4: Run test — verify pass**

```bash
pytest tests/unit/core/extractors/contracts/delta/test_prompts.py -v 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 7.5: Run full subrepo unit tests — no regression**

```bash
pytest tests/unit -x -q 2>&1 | tail -5
```

Expected: baseline + 11.

- [ ] **Step 7.6: Commit**

```bash
git add docling_graph/core/extractors/contracts/delta/prompts.py tests/unit/core/extractors/contracts/delta/test_prompts.py
git commit -m "feat(delta-prompt): instruct LLM to emit evidence_ids per node + relationship

Adds evidence_ids and property_evidence to the per-node JSON schema
instruction, evidence_ids to relationships, and a system-prompt rule
forbidding ID invention. Pairs with the [EVIDENCE id=\"…\"] markers
appended to chunk text by the document processor."
```

**Acceptance:** Prompt source includes `evidence_ids`, `property_evidence`, and the anti-invention rule. Existing extraction prompt structure preserved.

---

### Task 8: Preserve evidence in IR normalizer + validate against batch

**Why this task exists:** The LLM might emit invalid or hallucinated evidence_ids. The normalizer needs to (a) preserve good evidence_ids on the normalized node/relationship, (b) reject IDs that don't appear in the batch's chunk_metadata, (c) fall back to batch-level evidence_ids when validation fails, and (d) emit a diagnostic counter so we know how often the model is hallucinating.

**Files:**
- Modify: `docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/ir_normalizer.py:513-838` (`normalize_delta_ir_batch_results`)
- Test: `docker/docling-graph/repo/tests/unit/core/extractors/contracts/delta/test_ir_normalizer.py`

**Steps:**

- [ ] **Step 8.1: Write failing tests — node + rel evidence preserved, validation, fallback**

Append to `tests/unit/core/extractors/contracts/delta/test_ir_normalizer.py`:

```python
def test_normalizer_preserves_valid_node_evidence_ids():
    """When LLM emits evidence_ids that exist in the batch, preserve them
    verbatim on the normalized node."""
    from docling_graph.core.extractors.contracts.delta.ir_normalizer import (
        normalize_delta_ir_batch_results,
    )
    chunk_metadata = [{
        "chunk_id": 0, "page_numbers": [1],
        "self_refs": ["#/texts/0", "#/texts/1"],
        "evidence_ids": ["#/texts/0", "#/texts/1"],
    }]
    batch_plan = {0: [(0, 0, 0)]}  # exact tuple shape: (chunk_index, ?, ?) — match prod
    raw_results = {
        0: {"nodes": [{
            "path": "/foo", "ids": {"id": "X"},
            "evidence_ids": ["#/texts/1"],
            "property_evidence": {"name": ["#/texts/1"]},
        }], "relationships": []},
    }
    out = normalize_delta_ir_batch_results(
        raw_results=raw_results, batch_plan=batch_plan, chunk_metadata=chunk_metadata,
    )
    node = out["nodes"][0]
    assert node["provenance"]["evidence_ids"] == ["#/texts/1"]
    assert node["provenance"]["property_evidence"] == {"name": ["#/texts/1"]}
    assert "#/texts/0" in node["provenance"]["self_refs"]


def test_normalizer_falls_back_when_evidence_id_invalid():
    """Hallucinated evidence_ids (not in batch) get rejected; node falls back
    to batch-level evidence_ids and a diagnostic counter increments."""
    from docling_graph.core.extractors.contracts.delta.ir_normalizer import (
        normalize_delta_ir_batch_results,
    )
    chunk_metadata = [{
        "chunk_id": 0, "page_numbers": [1],
        "self_refs": ["#/texts/0"],
        "evidence_ids": ["#/texts/0"],
    }]
    batch_plan = {0: [(0, 0, 0)]}
    raw_results = {
        0: {"nodes": [{
            "path": "/foo", "ids": {"id": "X"},
            "evidence_ids": ["#/texts/999"],  # not in batch
        }], "relationships": []},
    }
    out = normalize_delta_ir_batch_results(
        raw_results=raw_results, batch_plan=batch_plan, chunk_metadata=chunk_metadata,
    )
    node = out["nodes"][0]
    assert node["provenance"]["evidence_ids"] == ["#/texts/0"], "fallback to batch IDs"
    diag = out.get("diagnostics") or {}
    assert diag.get("invalid_evidence_ids", 0) >= 1


def test_normalizer_preserves_relationship_evidence_ids():
    from docling_graph.core.extractors.contracts.delta.ir_normalizer import (
        normalize_delta_ir_batch_results,
    )
    chunk_metadata = [{
        "chunk_id": 0, "page_numbers": [1],
        "self_refs": ["#/texts/0", "#/texts/1"],
        "evidence_ids": ["#/texts/0", "#/texts/1"],
    }]
    batch_plan = {0: [(0, 0, 0)]}
    raw_results = {
        0: {"nodes": [
            {"path": "/a", "ids": {"id": "A"}},
            {"path": "/b", "ids": {"id": "B"}},
        ], "relationships": [
            {"edge_label": "REL", "source_path": "/a", "target_path": "/b",
             "evidence_ids": ["#/texts/0"]},
        ]},
    }
    out = normalize_delta_ir_batch_results(
        raw_results=raw_results, batch_plan=batch_plan, chunk_metadata=chunk_metadata,
    )
    rel = out["relationships"][0]
    assert rel["provenance"]["evidence_ids"] == ["#/texts/0"]
```

The exact `batch_plan` tuple shape and the `out` dict shape need to match the production code — verify by reading the function signature and existing tests in this file before adopting these literals. Adjust the test wiring (not the assertions) to match.

- [ ] **Step 8.2: Run tests — verify fail**

```bash
pytest tests/unit/core/extractors/contracts/delta/test_ir_normalizer.py::test_normalizer_preserves_valid_node_evidence_ids tests/unit/core/extractors/contracts/delta/test_ir_normalizer.py::test_normalizer_falls_back_when_evidence_id_invalid tests/unit/core/extractors/contracts/delta/test_ir_normalizer.py::test_normalizer_preserves_relationship_evidence_ids -v 2>&1 | tail -15
```

Expected: 3 FAIL — provenance dict missing `evidence_ids` / `self_refs`.

- [ ] **Step 8.3: Extend provenance dict in the normalizer**

In `docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/ir_normalizer.py`, locate the provenance dict construction (around lines 558-562):

```python
provenance = {
    "batch_index": batch_index,
    "chunk_indexes": chunk_indexes,
    "page_numbers": page_numbers,
}
```

Replace with:

```python
# Aggregate self_refs and evidence_ids across all chunks in this batch.
batch_self_refs: list[str] = []
batch_evidence_ids: list[str] = []
for chunk_index in chunk_indexes:
    cmeta = chunk_metadata[chunk_index] if chunk_index < len(chunk_metadata) else {}
    for ref in cmeta.get("self_refs") or []:
        if isinstance(ref, str) and ref not in batch_self_refs:
            batch_self_refs.append(ref)
    for eid in cmeta.get("evidence_ids") or []:
        if isinstance(eid, str) and eid not in batch_evidence_ids:
            batch_evidence_ids.append(eid)

provenance = {
    "batch_index": batch_index,
    "chunk_indexes": chunk_indexes,
    "page_numbers": page_numbers,
    "self_refs": batch_self_refs,
    "evidence_ids": batch_evidence_ids,
}
```

- [ ] **Step 8.4: Add per-node evidence preservation + validation**

At the node-build site (around line 742), update:

```python
if config.attach_provenance:
    normalized_node["provenance"] = provenance
```

to:

```python
if config.attach_provenance:
    # Per-node provenance is a SHALLOW COPY of the batch provenance so we
    # can override per-node evidence without mutating siblings.
    node_prov = dict(provenance)

    raw_evidence = raw_node.get("evidence_ids") or []
    valid_evidence, invalid_evidence = _partition_evidence(
        raw_evidence, batch_evidence_ids
    )
    node_prov["evidence_ids"] = valid_evidence or list(batch_evidence_ids)

    raw_property_evidence = raw_node.get("property_evidence") or {}
    if isinstance(raw_property_evidence, dict):
        node_prov["property_evidence"] = {
            k: [eid for eid in v if eid in batch_evidence_ids]
            for k, v in raw_property_evidence.items()
            if isinstance(v, list)
        }
    else:
        node_prov["property_evidence"] = {}

    if invalid_evidence:
        diagnostics["invalid_evidence_ids"] = (
            diagnostics.get("invalid_evidence_ids", 0) + len(invalid_evidence)
        )

    normalized_node["provenance"] = node_prov
```

At the relationship-build site (around line 832):

```python
if config.attach_provenance:
    rel_prov = dict(provenance)
    raw_evidence = raw_rel.get("evidence_ids") or []
    valid_evidence, invalid_evidence = _partition_evidence(
        raw_evidence, batch_evidence_ids
    )
    rel_prov["evidence_ids"] = valid_evidence or list(batch_evidence_ids)
    if invalid_evidence:
        diagnostics["invalid_evidence_ids"] = (
            diagnostics.get("invalid_evidence_ids", 0) + len(invalid_evidence)
        )
    normalized_rel["provenance"] = rel_prov
```

Add the helper near the top of the module:

```python
def _partition_evidence(
    raw: list, valid_pool: list[str]
) -> tuple[list[str], list[str]]:
    """Split raw evidence IDs into (valid, invalid) buckets relative to the
    batch's evidence pool. Non-string entries are treated as invalid."""
    valid_set = set(valid_pool)
    valid: list[str] = []
    invalid: list[str] = []
    for eid in raw or []:
        if isinstance(eid, str) and eid in valid_set:
            if eid not in valid:
                valid.append(eid)
        else:
            invalid.append(eid)
    return valid, invalid
```

The `diagnostics` variable: if the existing function already has a diagnostics dict accumulator (read its return shape first), use it. Otherwise, initialize `diagnostics: dict = {}` near the top of the function and include it in the return value: `return {"nodes": ..., "relationships": ..., "diagnostics": diagnostics}`.

- [ ] **Step 8.5: Run tests — verify pass**

```bash
pytest tests/unit/core/extractors/contracts/delta/test_ir_normalizer.py -v 2>&1 | tail -15
```

Expected: 3 new tests PASS, all pre-existing tests still PASS.

- [ ] **Step 8.6: Run full subrepo unit tests — no regression**

```bash
pytest tests/unit -x -q 2>&1 | tail -5
```

Expected: baseline + 14.

- [ ] **Step 8.7: Commit**

```bash
git add docling_graph/core/extractors/contracts/delta/ir_normalizer.py tests/unit/core/extractors/contracts/delta/test_ir_normalizer.py
git commit -m "feat(normalizer): preserve evidence_ids on nodes + relationships

Batch provenance now aggregates self_refs and evidence_ids across the
batch's chunks. Per-node and per-relationship evidence_ids are validated
against the batch pool — invalid (hallucinated) IDs are dropped and
counted in diagnostics; nodes with zero valid evidence_ids fall back to
the batch-level set so provenance is never empty."
```

**Acceptance:** Node and relationship provenance carry valid evidence_ids. Invalid IDs trigger fallback + diagnostic counter. property_evidence preserved on nodes.

---

## Chunk 3: Response & embedding (Tasks 9–12)

### Task 9: Extend `ExtractionProvenance` + provenance.py resolution

**Why this task exists:** The wire-shape `ExtractionProvenance` needs to carry the new evidence IDs. We add OPTIONAL fields (preserves backward compat) and update `_resolve_element_uid` to prefer extraction-time `self_refs` / `evidence_ids` over the chunk-index lookup that depended on the deleted re-chunk.

**Files:**
- Modify: `docker/docling-graph/app/schemas.py:105-160` (`ExtractionProvenance`)
- Modify: `docker/docling-graph/app/provenance.py:155-196` (`_resolve_element_uid`)
- Test: `docker/docling-graph/tests/test_extraction_provenance_schema.py` (create)

**Steps:**

- [ ] **Step 9.1: Write failing tests — backward compat + new fields**

Create `docker/docling-graph/tests/test_extraction_provenance_schema.py`:

```python
def test_extraction_provenance_backward_compat():
    """Existing callers must construct ExtractionProvenance with no new
    fields and not break."""
    from app.schemas import ExtractionProvenance
    p = ExtractionProvenance(
        instance_id="i1",
        ontology_name="RADAR_SYSTEM",
        identity_values={"designation": "AN/MPQ-65"},
        element_uid="#/texts/12",
    )
    assert p.evidence_ids == []
    assert p.page_numbers == []
    assert p.evidence_text is None


def test_extraction_provenance_accepts_new_fields():
    from app.schemas import ExtractionProvenance
    p = ExtractionProvenance(
        instance_id="i1",
        ontology_name="RADAR_SYSTEM",
        identity_values={},
        element_uid="#/texts/12",
        evidence_ids=["#/texts/12", "#/texts/13"],
        page_numbers=[3, 4],
        evidence_text="The AN/MPQ-65 radar...",
    )
    assert p.evidence_ids == ["#/texts/12", "#/texts/13"]
    assert p.page_numbers == [3, 4]
    assert p.evidence_text == "The AN/MPQ-65 radar..."
```

Append to `docker/docling-graph/tests/test_provenance.py` (or create):

```python
def test_resolve_element_uid_prefers_provenance_self_refs():
    """Resolution order: direct → nested provenance.element_uid →
    provenance.self_refs[0] → provenance.evidence_ids[0] (if self_ref-shaped)
    → chunk_index lookup."""
    from app.provenance import _resolve_element_uid
    node_data = {
        "provenance": {
            "self_refs": ["#/texts/9", "#/texts/10"],
        },
    }
    assert _resolve_element_uid(node_data, chunk_to_self_refs={}) == "#/texts/9"


def test_resolve_element_uid_falls_back_to_evidence_ids():
    from app.provenance import _resolve_element_uid
    node_data = {
        "provenance": {
            "evidence_ids": ["#/texts/9"],
        },
    }
    assert _resolve_element_uid(node_data, chunk_to_self_refs={}) == "#/texts/9"


def test_resolve_element_uid_skips_non_selfref_evidence():
    """evidence_ids that aren't shaped like Docling self_refs (don't start
    with '#/') get ignored at the resolver level."""
    from app.provenance import _resolve_element_uid
    node_data = {
        "provenance": {
            "evidence_ids": ["llm-generated-noise-id"],
        },
    }
    assert _resolve_element_uid(node_data, chunk_to_self_refs={}) is None
```

- [ ] **Step 9.2: Run tests — verify fail**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance
pytest docker/docling-graph/tests/test_extraction_provenance_schema.py docker/docling-graph/tests/test_provenance.py -v 2>&1 | tail -15
```

Expected: 5 FAIL — fields don't exist or resolver doesn't check `self_refs`/`evidence_ids`.

- [ ] **Step 9.3: Add fields to `ExtractionProvenance`**

In `docker/docling-graph/app/schemas.py` (around line 160, end of `ExtractionProvenance`):

```python
# Additive fields — extraction-time evidence carried through to the
# response. All optional / default-empty for backward compatibility
# with callers that don't yet read them.
evidence_ids: list[str] = Field(
    default_factory=list,
    description=(
        "DoclingDocument self_refs (e.g. '#/texts/12') the LLM cited "
        "as evidence for this entity instance. Populated when the "
        "delta normalizer found valid evidence_ids in the LLM output."
    ),
)
page_numbers: list[int] = Field(
    default_factory=list,
    description="Pages the cited evidence units span. Sorted, deduped.",
)
evidence_text: Optional[str] = Field(
    default=None,
    description=(
        "Concatenated text of the evidence units (best-effort, "
        "truncated to ~500 chars). For human review and downstream "
        "snippet display."
    ),
)
```

Keep `element_uid` REQUIRED (the spec says "downstream currently depends on it").

- [ ] **Step 9.4: Update `_resolve_element_uid`**

In `docker/docling-graph/app/provenance.py:155-196`, replace the function with:

```python
def _resolve_element_uid(
    node_data: dict[str, Any],
    chunk_to_self_refs: dict[int, list[str]] | None,
) -> str | None:
    """Return element_uid for a knowledge-graph node, or None if none.

    Resolution order:
      1. Direct `element_uid` attribute on the node dict.
      2. Nested `provenance.element_uid`.
      3. Nested `provenance.self_refs[0]` (extraction-time, authoritative).
      4. Nested `provenance.evidence_ids[0]` IF it is a Docling self_ref
         (starts with '#/').
      5. `provenance.chunk_indexes[0]` → first self_ref via
         chunk_to_self_refs (extraction-time map from main.py).
    """
    direct = node_data.get("element_uid")
    if isinstance(direct, str) and direct:
        return direct

    prov = node_data.get("provenance")
    if not isinstance(prov, dict):
        return None

    nested = prov.get("element_uid")
    if isinstance(nested, str) and nested:
        return nested

    self_refs = prov.get("self_refs")
    if isinstance(self_refs, list) and self_refs:
        first = self_refs[0]
        if isinstance(first, str) and first:
            return first

    evidence_ids = prov.get("evidence_ids")
    if isinstance(evidence_ids, list):
        for eid in evidence_ids:
            if isinstance(eid, str) and eid.startswith("#/"):
                return eid

    if chunk_to_self_refs:
        chunk_indexes = prov.get("chunk_indexes")
        if isinstance(chunk_indexes, list) and chunk_indexes:
            first = chunk_indexes[0]
            if isinstance(first, int):
                refs = chunk_to_self_refs.get(first)
                if refs:
                    return refs[0]

    return None
```

- [ ] **Step 9.5: Populate the new `ExtractionProvenance` fields in `build_provenance_from_context`**

In `docker/docling-graph/app/provenance.py`, locate the `provenance_cls(...)` construction in `build_provenance_from_context` (around line 318) and extend the kwargs:

```python
prov_dict = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
out.append(
    provenance_cls(
        instance_id=str(instance_id),
        ontology_name=str(label),
        identity_values=identity_values,
        element_uid=element_uid,
        page=_resolve_page(data),
        chunk_index=_resolve_chunk_index(data),
        # NEW:
        evidence_ids=[
            eid for eid in (prov_dict.get("evidence_ids") or [])
            if isinstance(eid, str)
        ],
        page_numbers=sorted({
            p for p in (prov_dict.get("page_numbers") or [])
            if isinstance(p, int)
        }),
        evidence_text=_join_evidence_text(
            chunk_to_self_refs=None,  # see Step 9.6 if we have evidence_units
            evidence_units_for_node=data.get("evidence_units"),
        ),
    )
)
```

The `_join_evidence_text` helper (place near `_resolve_element_uid`):

```python
def _join_evidence_text(
    chunk_to_self_refs=None,
    evidence_units_for_node=None,
    cap: int = 500,
) -> str | None:
    """Concatenate evidence-unit text into a single snippet, capped at `cap`
    chars. Returns None if no text available."""
    if not evidence_units_for_node:
        return None
    parts = [
        u.get("text", "") for u in evidence_units_for_node
        if isinstance(u, dict) and isinstance(u.get("text"), str)
    ]
    joined = " ".join(p.strip() for p in parts if p.strip())
    if not joined:
        return None
    return joined[:cap] + ("…" if len(joined) > cap else "")
```

(If you want richer wiring of evidence_units to nodes, that's Task 10 territory — for this task, leaving `_join_evidence_text` accepting `None` and returning `None` is fine.)

- [ ] **Step 9.6: Run tests — verify pass**

```bash
pytest docker/docling-graph/tests/test_extraction_provenance_schema.py docker/docling-graph/tests/test_provenance.py -v 2>&1 | tail -15
```

Expected: 5 PASS.

- [ ] **Step 9.7: Run all docling-graph service tests — no regression**

```bash
pytest docker/docling-graph/tests -x -q 2>&1 | tail -5
```

Expected: baseline + 5.

- [ ] **Step 9.8: Commit**

```bash
git add docker/docling-graph/app/schemas.py docker/docling-graph/app/provenance.py docker/docling-graph/tests/test_extraction_provenance_schema.py docker/docling-graph/tests/test_provenance.py
git commit -m "feat(provenance): carry evidence_ids/page_numbers through ExtractionProvenance

Adds optional evidence_ids, page_numbers, evidence_text to the wire shape
(backward compatible). _resolve_element_uid prefers extraction-time
self_refs and evidence_ids over the chunk_index → re-chunk lookup."
```

**Acceptance:** New fields default-empty. Resolver order is direct → nested → self_refs → evidence_ids (self_ref-shaped) → chunk_index lookup. All 5 new tests pass.

---

### Task 10: Extend `ExtractionFieldProvenance` + `build_auto_field_evidence`

**Why this task exists:** Field provenance currently carries (instance_id, field_name, value, supporting_snippet, element_uid). The spec wants optional `evidence_id`, `page`, `document_id` — same idea as Task 9 but for fields. `build_auto_field_evidence` accepts `(element_uid, text)` tuples; we widen it (additively) to also accept evidence-unit shapes.

**Files:**
- Modify: `docker/docling-graph/app/schemas.py:162-189` (`ExtractionFieldProvenance`)
- Modify: `docker/docling-graph/app/provenance.py:344-418` (`build_auto_field_evidence`)
- Modify: `docker/docling-graph/app/main.py` (caller — pass evidence units when available)
- Test: `docker/docling-graph/tests/test_field_provenance.py` (create or extend)

**Steps:**

- [ ] **Step 10.1: Write failing test — field provenance new fields + evidence-unit input**

Create `docker/docling-graph/tests/test_field_provenance.py`:

```python
def test_field_provenance_backward_compat():
    from app.schemas import ExtractionFieldProvenance
    p = ExtractionFieldProvenance(
        instance_id="i1",
        field_name="diameter",
        value=0.37,
        supporting_snippet="diameter: 0.37 m",
    )
    assert p.evidence_id is None
    assert p.page is None
    assert p.document_id is None


def test_field_provenance_accepts_new_fields():
    from app.schemas import ExtractionFieldProvenance
    p = ExtractionFieldProvenance(
        instance_id="i1",
        field_name="diameter",
        value=0.37,
        supporting_snippet="diameter: 0.37 m",
        element_uid="#/texts/12",
        evidence_id="#/texts/12",
        page=3,
        document_id="doc-uuid-abc",
    )
    assert p.evidence_id == "#/texts/12"
    assert p.page == 3
    assert p.document_id == "doc-uuid-abc"


def test_build_auto_field_evidence_accepts_evidence_units():
    """build_auto_field_evidence must accept evidence-unit-shaped inputs
    (dicts with evidence_id/text/page_numbers) in addition to the legacy
    (element_uid, text) tuple shape. The extra metadata flows into the
    emitted ExtractionFieldProvenance."""
    from app.provenance import build_auto_field_evidence
    from app.schemas import ExtractionFieldProvenance
    units = [
        {"evidence_id": "#/texts/12", "self_ref": "#/texts/12",
         "text": "diameter: 0.37 m", "page_numbers": [3]},
    ]
    primary_entities = [{"instance_id": "i1", "diameter": 0.37}]
    rows = build_auto_field_evidence(
        primary_entities=primary_entities,
        instance_ids=["i1"],
        input_chunks=units,  # accepts evidence-unit dicts now
        skip_fields=set(),
        provenance_cls=ExtractionFieldProvenance,
    )
    assert any(
        r.field_name == "diameter" and r.evidence_id == "#/texts/12" and r.page == 3
        for r in rows
    )
```

- [ ] **Step 10.2: Run tests — verify fail**

```bash
pytest docker/docling-graph/tests/test_field_provenance.py -v 2>&1 | tail -10
```

Expected: 3 FAIL.

- [ ] **Step 10.3: Add new fields to `ExtractionFieldProvenance`**

In `docker/docling-graph/app/schemas.py` (around line 189), append:

```python
evidence_id: Optional[str] = Field(
    default=None,
    description="Stable evidence_id (== DoclingDocument self_ref) for the source unit.",
)
page: Optional[int] = Field(
    default=None,
    description="Page number of the supporting evidence unit.",
)
document_id: Optional[str] = Field(
    default=None,
    description="Document UUID this field's evidence came from.",
)
```

- [ ] **Step 10.4: Make `build_auto_field_evidence` accept evidence-unit inputs**

In `docker/docling-graph/app/provenance.py:344-418`, update the signature and implementation. Choose ONE of:

(a) **Detect input shape inside the function**: if `input_chunks[0]` is a dict with `evidence_id`, treat as evidence units; else treat as legacy `(element_uid, text)` tuples. Convert internally.

(b) **Add a small wrapper** `_normalize_field_evidence_inputs(input_chunks)` that returns a normalized list of dicts: `[{"element_uid": ..., "text": ..., "evidence_id": ..., "page": ..., "document_id": ...}, ...]`. Call from inside `build_auto_field_evidence`.

Recommended: (b). Add the wrapper near the top of `provenance.py`:

```python
def _normalize_field_evidence_inputs(input_chunks) -> list[dict]:
    """Accept either (element_uid, text) tuples (legacy) or evidence-unit
    dicts (new). Return a normalized list of dicts ready for matching."""
    out: list[dict] = []
    for entry in input_chunks or []:
        if isinstance(entry, dict):
            out.append({
                "element_uid": entry.get("self_ref") or entry.get("evidence_id"),
                "text": entry.get("text") or "",
                "evidence_id": entry.get("evidence_id"),
                "page": (entry.get("page_numbers") or [None])[0],
                "document_id": entry.get("document_id"),
            })
        elif isinstance(entry, tuple) and len(entry) >= 2:
            out.append({
                "element_uid": entry[0],
                "text": entry[1],
                "evidence_id": entry[0] if isinstance(entry[0], str) and entry[0].startswith("#/") else None,
                "page": None,
                "document_id": None,
            })
    return out
```

In `build_auto_field_evidence`, at the top, replace any direct iteration over `input_chunks` with `normalized = _normalize_field_evidence_inputs(input_chunks)` and read `text`/`element_uid`/`evidence_id`/`page`/`document_id` from `normalized` rows.

When constructing each `ExtractionFieldProvenance` row, pass the new fields:

```python
provenance_cls(
    instance_id=...,
    field_name=...,
    value=...,
    supporting_snippet=...,
    element_uid=row.get("element_uid"),
    evidence_id=row.get("evidence_id"),
    page=row.get("page"),
    document_id=row.get("document_id"),
)
```

DO NOT remove the deterministic value-matching logic — the spec says to keep it because it verifies field value against source text even when the LLM gives weak evidence_ids.

- [ ] **Step 10.5: Update the caller in `main.py`**

Where `build_auto_field_evidence` is called from `main.py`, update the input to pass evidence units when they're available:

```python
# Where input_chunks was built as [(element_uid, text), ...] previously,
# prefer the extraction-time evidence units if present:
evidence_units_by_chunk = getattr(context, "_chunk_to_evidence_units", None) or {}
all_evidence_units: list[dict] = []
for units in evidence_units_by_chunk.values():
    for u in units:
        # Annotate each unit with the document_id for downstream persistence.
        u_copy = dict(u)
        u_copy.setdefault("document_id", document_id)
        all_evidence_units.append(u_copy)

field_provenance_rows = build_auto_field_evidence(
    primary_entities=primary_entities,
    instance_ids=instance_ids,
    input_chunks=all_evidence_units or legacy_input_chunks,  # fallback to legacy if no units
    skip_fields=skip_fields,
    provenance_cls=ExtractionFieldProvenance,
)
```

Find the existing call site by grepping `main.py` for `build_auto_field_evidence`. If multiple call sites exist, prefer the production path (NOT diagnostic ones).

- [ ] **Step 10.6: Run tests — verify pass**

```bash
pytest docker/docling-graph/tests/test_field_provenance.py -v 2>&1 | tail -10
```

Expected: 3 PASS.

- [ ] **Step 10.7: Run all docling-graph service tests — no regression**

```bash
pytest docker/docling-graph/tests -x -q 2>&1 | tail -5
```

Expected: baseline + 8.

- [ ] **Step 10.8: Commit**

```bash
git add docker/docling-graph/app/schemas.py docker/docling-graph/app/provenance.py docker/docling-graph/app/main.py docker/docling-graph/tests/test_field_provenance.py
git commit -m "feat(field-provenance): accept evidence units + carry evidence_id/page/document_id

ExtractionFieldProvenance gains optional evidence_id, page, document_id
fields. build_auto_field_evidence accepts both legacy (element_uid, text)
tuples and the new evidence-unit dicts; main.py prefers evidence units
from chunk_created traces when available. Deterministic value matching
preserved unchanged."
```

**Acceptance:** Field provenance carries evidence_id/page/document_id when present. `build_auto_field_evidence` works with both input shapes. Backward compat preserved.

---

### Task 11: Add `ExtractionRelationshipProvenance` to `ExtractPassResponse`

**Why this task exists:** Entities and fields have provenance rows; relationships don't. Step 11 adds the third axis.

**Files:**
- Modify: `docker/docling-graph/app/schemas.py:226-276` (`ExtractPassResponse`) + new `ExtractionRelationshipProvenance` class
- Modify: `docker/docling-graph/app/main.py` — assemble `relationship_provenance` from normalized relationships
- Modify: `docker/docling-graph/app/provenance.py` — add `build_relationship_provenance_from_context`
- Test: `docker/docling-graph/tests/test_relationship_provenance.py` (create)

**Steps:**

- [ ] **Step 11.1: Write failing tests**

Create `docker/docling-graph/tests/test_relationship_provenance.py`:

```python
def test_relationship_provenance_schema():
    from app.schemas import ExtractionRelationshipProvenance
    p = ExtractionRelationshipProvenance(
        relationship_type="HAS_PART",
        source_instance_id="i1",
        target_instance_id="i2",
        evidence_ids=["#/texts/0"],
        self_refs=["#/texts/0"],
        page_numbers=[3],
        supporting_snippet="The radar consists of...",
    )
    assert p.relationship_type == "HAS_PART"
    assert p.evidence_ids == ["#/texts/0"]


def test_relationship_provenance_defaults():
    from app.schemas import ExtractionRelationshipProvenance
    p = ExtractionRelationshipProvenance(relationship_type="REL")
    assert p.source_instance_id is None
    assert p.evidence_ids == []
    assert p.self_refs == []
    assert p.page_numbers == []
    assert p.supporting_snippet is None


def test_extract_pass_response_includes_relationship_provenance():
    from app.schemas import ExtractPassResponse, ExtractionRelationshipProvenance
    resp = ExtractPassResponse(
        bundle_key="x", pass_name="y", pass_output={},
        relationship_provenance=[
            ExtractionRelationshipProvenance(relationship_type="REL"),
        ],
    )
    assert len(resp.relationship_provenance) == 1
    assert resp.relationship_provenance[0].relationship_type == "REL"


def test_extract_pass_response_relationship_provenance_default_empty():
    """Backward compat: callers not setting relationship_provenance get []."""
    from app.schemas import ExtractPassResponse
    resp = ExtractPassResponse(bundle_key="x", pass_name="y", pass_output={})
    assert resp.relationship_provenance == []


def test_build_relationship_provenance_from_normalized_rels():
    from app.provenance import build_relationship_provenance_from_context
    from app.schemas import ExtractionRelationshipProvenance
    from types import SimpleNamespace
    import networkx as nx
    g = nx.DiGraph()
    g.add_node("n1", instance_id="i1")
    g.add_node("n2", instance_id="i2")
    g.add_edge("n1", "n2", label="HAS_PART", provenance={
        "evidence_ids": ["#/texts/0"],
        "self_refs": ["#/texts/0"],
        "page_numbers": [3],
    })
    ctx = SimpleNamespace(knowledge_graph=g)
    rows = build_relationship_provenance_from_context(
        ctx, ExtractionRelationshipProvenance,
    )
    assert len(rows) == 1
    assert rows[0].relationship_type == "HAS_PART"
    assert rows[0].source_instance_id == "i1"
    assert rows[0].target_instance_id == "i2"
    assert rows[0].evidence_ids == ["#/texts/0"]
```

- [ ] **Step 11.2: Run tests — verify fail**

```bash
pytest docker/docling-graph/tests/test_relationship_provenance.py -v 2>&1 | tail -10
```

Expected: 5 FAIL.

- [ ] **Step 11.3: Add `ExtractionRelationshipProvenance` to `schemas.py`**

In `docker/docling-graph/app/schemas.py`, after `ExtractionFieldProvenance` (around line 189):

```python
class ExtractionRelationshipProvenance(BaseModel):
    """Per-extracted-relationship provenance link to source DoclingDocument
    elements. Mirrors ExtractionProvenance but for edges."""
    relationship_type: str
    source_instance_id: Optional[str] = None
    target_instance_id: Optional[str] = None
    evidence_ids: list[str] = Field(default_factory=list)
    self_refs: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    supporting_snippet: Optional[str] = None
```

In `ExtractPassResponse` (around line 276), add:

```python
relationship_provenance: list[ExtractionRelationshipProvenance] = Field(
    default_factory=list,
    description=(
        "Per-relationship provenance, mirrors `provenance` for entities. "
        "Built from delta-IR relationship.provenance when present. "
        "Empty by default — additive on the wire shape."
    ),
)
```

- [ ] **Step 11.4: Add `build_relationship_provenance_from_context` to `provenance.py`**

In `docker/docling-graph/app/provenance.py`, add a new public function near `build_provenance_from_context`:

```python
def build_relationship_provenance_from_context(
    context: Any,
    provenance_cls: type,
) -> list[Any]:
    """Walk context.knowledge_graph edges and emit
    ExtractionRelationshipProvenance per edge with non-trivial provenance."""
    graph = getattr(context, "knowledge_graph", None)
    if graph is None:
        return []
    out: list[Any] = []
    for source, target, edge_data in graph.edges(data=True):
        label = edge_data.get("label") or edge_data.get("edge_label")
        if not label:
            continue
        prov = edge_data.get("provenance") if isinstance(edge_data.get("provenance"), dict) else {}
        source_instance = (
            graph.nodes[source].get("instance_id") if source in graph else None
        )
        target_instance = (
            graph.nodes[target].get("instance_id") if target in graph else None
        )
        out.append(
            provenance_cls(
                relationship_type=str(label),
                source_instance_id=str(source_instance) if source_instance else None,
                target_instance_id=str(target_instance) if target_instance else None,
                evidence_ids=[
                    eid for eid in (prov.get("evidence_ids") or [])
                    if isinstance(eid, str)
                ],
                self_refs=[
                    r for r in (prov.get("self_refs") or [])
                    if isinstance(r, str)
                ],
                page_numbers=sorted({
                    p for p in (prov.get("page_numbers") or [])
                    if isinstance(p, int)
                }),
                supporting_snippet=None,
            )
        )
    return out
```

- [ ] **Step 11.5: Wire it into `main.py` response assembly**

Find the place in `main.py` where `ExtractPassResponse(...)` is constructed (search for `ExtractPassResponse(`) and add:

```python
from app.provenance import build_relationship_provenance_from_context
from app.schemas import ExtractionRelationshipProvenance

relationship_provenance_rows = build_relationship_provenance_from_context(
    context, ExtractionRelationshipProvenance,
)

return ExtractPassResponse(
    # … existing fields …
    relationship_provenance=relationship_provenance_rows,
)
```

- [ ] **Step 11.6: Run tests — verify pass**

```bash
pytest docker/docling-graph/tests/test_relationship_provenance.py -v 2>&1 | tail -10
```

Expected: 5 PASS.

- [ ] **Step 11.7: Run all docling-graph service tests — no regression**

```bash
pytest docker/docling-graph/tests -x -q 2>&1 | tail -5
```

Expected: baseline + 13.

- [ ] **Step 11.8: Commit**

```bash
git add docker/docling-graph/app/schemas.py docker/docling-graph/app/provenance.py docker/docling-graph/app/main.py docker/docling-graph/tests/test_relationship_provenance.py
git commit -m "feat(provenance): add ExtractionRelationshipProvenance to /extract-pass response

Mirrors ExtractionProvenance for relationships. Built from delta-IR
relationship.provenance when present; field on ExtractPassResponse
defaults to [] for backward compat."
```

**Acceptance:** `relationship_provenance` field present on the response, populated from normalized relationships.

---

### Task 12: Persist evidence metadata in embedding pipeline (Pass A)

**Why this task exists:** RAG retrieval needs to cite source units the same way graph extraction does. Embedding chunks may have different boundaries than extraction chunks (that's fine), but they MUST persist `evidence_ids`, `self_refs`, `page_numbers`, `document_id`, `text` so downstream retrieval surfaces the same lineage.

**Files:**
- Modify: `app/workers/pipeline.py:4501-4605` (native HybridChunker path)
- Possibly Modify: `app/db/models.py` or wherever `TextChunk` is defined — needs new columns OR a JSON properties bag with these fields
- Test: `tests/unit/workers/test_embedding_chunk_evidence.py` (create)

**Steps:**

- [ ] **Step 12.1: Inventory `TextChunk` fields**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance
grep -n "class TextChunk" app/db/models.py 2>/dev/null || grep -rn "class TextChunk" app/db/ 2>/dev/null | head -5
```

Read the class definition. Note which existing columns are usable, especially any JSON `properties` bag.

If a JSON properties column exists (most likely): persist evidence metadata there to keep this change additive (no Alembic migration). If not: add a single `properties: dict[str, Any]` JSON column via migration as the smallest possible schema delta. Choose the migration-free path if possible.

- [ ] **Step 12.2: Write failing test — embedding chunk persists evidence**

Create `tests/unit/workers/test_embedding_chunk_evidence.py`:

```python
def test_embedding_chunk_metadata_includes_evidence_fields():
    """Chunks emitted by the native HybridChunker path in pipeline.py
    must carry evidence_ids, self_refs, page_numbers, document_id, text
    in their persisted metadata."""
    # This is a unit-level test that exercises just the chunk-meta
    # construction part of pipeline.py — extract that into a small
    # helper in Step 12.3 so the test can call it directly without
    # spinning up the full Celery task.
    from app.workers.pipeline import _build_native_chunk_meta
    from types import SimpleNamespace
    chunk = SimpleNamespace(
        text="hello world",
        meta=SimpleNamespace(doc_items=[
            SimpleNamespace(
                self_ref="#/texts/0",
                prov=[SimpleNamespace(page_no=3)],
            ),
        ]),
    )
    meta = _build_native_chunk_meta(
        chunk_idx=0, chunk=chunk, document_id="doc-uuid-abc",
        model_version="bge-m3:latest",
    )
    assert meta["evidence_ids"] == ["#/texts/0"]
    assert meta["self_refs"] == ["#/texts/0"]
    assert meta["page_numbers"] == [3]
    assert meta["document_id"] == "doc-uuid-abc"
    assert meta["modality"] == "text"
```

- [ ] **Step 12.3: Extract `_build_native_chunk_meta` helper**

In `app/workers/pipeline.py`, near the native-chunking path (around line 4524-4555), extract the per-chunk metadata construction into a module-level helper:

```python
def _build_native_chunk_meta(
    chunk_idx: int,
    chunk,
    document_id: str,
    model_version: str,
) -> dict:
    """Build per-chunk metadata for the native HybridChunker path.

    Carries evidence_ids/self_refs/page_numbers so embedding chunks share
    the same source-unit lineage as graph-extraction chunks (independent
    boundaries, identical lineage shape).
    """
    self_refs: list[str] = []
    page_numbers: set[int] = set()
    for item in (getattr(getattr(chunk, "meta", None), "doc_items", None) or []):
        ref = getattr(item, "self_ref", None)
        if isinstance(ref, str) and ref:
            self_refs.append(ref)
        for p in (getattr(item, "prov", None) or []):
            pn = getattr(p, "page_no", None)
            if pn is not None:
                page_numbers.add(pn)
    chunk_key = hashlib.sha256(
        f"{document_id}:native:{chunk_idx}:{model_version}".encode()
    ).hexdigest()
    chunk_id = uuid.UUID(hashlib.md5(chunk_key.encode()).hexdigest())
    return {
        "chunk_id": chunk_id,
        "chunk_index": chunk_idx,
        "page_number": min(page_numbers) if page_numbers else None,
        "page_numbers": sorted(page_numbers),
        "modality": "text",
        "self_refs": self_refs,
        "evidence_ids": list(self_refs),
        "document_id": document_id,
    }
```

Replace the existing inline construction with a call to this helper:

```python
all_chunk_metas.append(
    _build_native_chunk_meta(
        chunk_idx=chunk_idx,
        chunk=chunk,
        document_id=document_id,
        model_version=model_version,
    )
)
```

- [ ] **Step 12.4: Persist evidence in the DB row**

In the persistence loop (around lines 4564-4600), enrich the `properties` field of `_TCR` (or whatever shape your codebase uses) with the new keys. The exact field names depend on `TextChunk` schema — ideally a `properties: dict` JSON column already exists; add `self_refs`, `evidence_ids`, `page_numbers` to it:

```python
text_chunk_records.append(_TCR(
    chunk_id=str(meta["chunk_id"]),
    text=text,
    document_id=document_id,
    properties={
        "artifact_id": None,
        "modality": meta["modality"],
        "page_number": meta["page_number"],
        "page_numbers": meta["page_numbers"],
        "classification": doc_classification,
        # NEW — evidence lineage:
        "self_refs": meta["self_refs"],
        "evidence_ids": meta["evidence_ids"],
    },
    embedding=embedding,
))
```

Inspect `_TCR` to confirm the shape — if it has a flat schema with no JSON bag, you may need a separate persistence path or a small migration. Per the spec, this is allowed to be additive: `If embedding is outside /extract-pass, this service should at least expose evidence metadata in diagnostics/response so the caller can persist it.` So the minimum required is that the metadata exists in `meta` and is available; downstream persistence can be done in a follow-up if the schema delta is large.

- [ ] **Step 12.5: Run test — verify pass**

```bash
pytest tests/unit/workers/test_embedding_chunk_evidence.py -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 12.6: Run all parent app unit tests — no regression**

```bash
pytest tests/unit -x -q 2>&1 | tail -5
```

Expected: parent app baseline + 1.

- [ ] **Step 12.7: Commit**

```bash
git add app/workers/pipeline.py tests/unit/workers/test_embedding_chunk_evidence.py
git commit -m "feat(embedding): persist self_refs/evidence_ids/page_numbers per chunk

Embedding chunks now carry the same source-unit lineage as graph
extraction chunks (different boundaries, identical metadata shape).
Extracts _build_native_chunk_meta helper for testability and adds
evidence fields to the persisted TextChunk properties."
```

**Acceptance:** `_build_native_chunk_meta` exposes evidence fields. Persistence layer carries them through to TextChunk. Parent app baseline tests preserved.

---

## Chunk 4: Integration verification (Task 13)

### Task 13: End-to-end smoke + simplify + docs

**Why this task exists:** Per the user's standing rule (`feedback_post_code_workflow.md`), every code update gets: (1) simplify, (2) full test suite, (3) verify VERIFICATION_CHECKLIST.md, (4) update README. Task 13 runs that workflow against the cumulative changes from Tasks 1-12.

**Files:**
- Touch: README.md (only if observable behavior changed in user-facing way)
- Touch: VERIFICATION_CHECKLIST.md (add provenance checks if missing)

**Steps:**

- [ ] **Step 13.1: Run full subrepo unit + integration tests**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance/docker/docling-graph/repo
pytest tests/ -q 2>&1 | tail -10
```

Expected: clean run. Total count = baseline + (Task 1: 2) + (Task 2: 2) + (Task 3: 1) + (Task 4: 1) + (Task 6: 4) + (Task 7: 1) + (Task 8: 3) = baseline + 14.

- [ ] **Step 13.2: Run full docling-graph service tests**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance
pytest docker/docling-graph/tests -q 2>&1 | tail -10
```

Expected: baseline + (Task 5: 2) + (Task 9: 5) + (Task 10: 3) + (Task 11: 5) = baseline + 15.

- [ ] **Step 13.3: Run full parent app tests**

```bash
pytest tests/ -q 2>&1 | tail -10
```

Expected: parent baseline + 1 (Task 12).

- [ ] **Step 13.4: Invoke `simplify` skill on the diff**

```
Skill: simplify
```

Apply with the stated set of changed files. Address any recommendations BEFORE merging.

- [ ] **Step 13.5: Update VERIFICATION_CHECKLIST.md**

Add a new section at the bottom of `VERIFICATION_CHECKLIST.md`:

```markdown
## Provenance Pipeline (Pass B → response)

- [ ] /extract-pass response includes non-empty `provenance` for at least one entity per pass on a known-good fixture
- [ ] /extract-pass response includes `field_provenance` rows that resolve to evidence_id when LLM cited valid IDs
- [ ] /extract-pass response includes `relationship_provenance` for at least one edge
- [ ] No log warnings of `no chunk_created trace events found`
- [ ] `_build_chunk_to_self_refs_map` is NOT in the production call stack (search logs for the diagnostic-only warning)
- [ ] Diagnostics counter `invalid_evidence_ids` reports a tractable rate (single-digit per pass; spikes indicate prompt drift)

## Provenance Pipeline (Pass A — embedding)

- [ ] TextChunk rows in PostgreSQL carry `self_refs` and `evidence_ids` in their properties JSON
- [ ] Retrieval API responses can dereference an evidence_id back to a DoclingDocument self_ref
```

- [ ] **Step 13.6: Update README — only if user-visible behavior changed**

Read existing README. If the section about /extract-pass response shape mentions provenance, append a one-line mention of the new `relationship_provenance` field and the additive evidence_ids fields. Otherwise leave alone — internal refactoring needs no README change.

- [ ] **Step 13.7: Final commit (only if 13.4 / 13.5 / 13.6 produced changes)**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/provenance
git add VERIFICATION_CHECKLIST.md README.md  # whichever changed
git commit -m "docs: add provenance pipeline verification entries

Captures the end-to-end checks for the minimal provenance fix so
future changes don't silently regress the trace-based provenance source."
```

**Acceptance:** All test suites green. simplify skill recommendations addressed. VERIFICATION_CHECKLIST has provenance section.

---

## Final Cross-Task Acceptance Criteria

(Mirrors the spec's "Acceptance Criteria" — verify all are true at end.)

- [ ] Graph extraction chunks can remain large (4096 tokens) — confirmed in Task 1's new `chunker_config` summary
- [ ] Embedding/RAG chunks can be smaller and independent — confirmed in Task 12 (independent `EMBEDDING_CHUNK_MAX_TOKENS=512` path)
- [ ] Every production `HybridChunker` construction is explicit — Task 1 (DocumentChunker) + Task 5 regression test (no bare `HybridChunker()` in production paths)
- [ ] Production provenance does not depend on re-running `HybridChunker` — Task 5 (replaces `_build_chunk_to_self_refs_map` with trace-based source)
- [ ] Entity provenance resolves to document/page/self_ref/evidence — Task 9 (`ExtractionProvenance` carries all 4 axes)
- [ ] Field provenance resolves to document/page/self_ref/snippet/evidence — Task 10
- [ ] Relationship provenance resolves to document/page/self_ref/evidence — Task 11
- [ ] Diagnostics report when exact evidence IDs are missing and batch-level provenance was used — Task 8 (`invalid_evidence_ids` counter)
