# EXTRACTED_FROM Precise-Lineage Fix — store chunk metadata + prefer per-node evidence + resolve to chunk

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore PRECISE entity→chunk lineage (`EXTRACTED_FROM` edges where each entity links to its actual source chunk(s)+page) so graph-query and RAG endpoints return per-entity provenance — meeting the hard requirement that every value trace to its exact source chunk + document + page.

**Architecture:** Three coupled root causes, all verified against live code + real gemma4 output (memory `project_extracted_from_root_cause.md`; survived three dissent-review rounds).
- **Part A (gate unblock, docling-graph):** patch 0002's CHUNKED-BATCHES branch (production path for DoclingDocument input) builds correct `chunk_metadata` (chunk_id+self_refs+page_numbers) but never assigns `doc_processor.last_chunk_metadata`. `app/main.py` reads that to build the `chunk_to_self_refs`/`chunk_to_page_numbers` maps the provenance builders need; empty → `element_uid=""` → `_parse_pass_response` drops every provenance row (pipeline.py:3744) → lineage gate rejects all → `EXTRACTED_FROM`=0.
- **Part B (precision, docling-graph):** `_resolve_element_uid` (provenance.py:232-242) returns the batch-wide `provenance.self_refs[0]` (Strategy 3) BEFORE the per-node `provenance.evidence_ids[0]` (Strategy 4). The IR normalizer attaches the SAME batch-wide `self_refs` to every node (ir_normalizer.py:611) but narrows `evidence_ids` per node — so Strategy 3 makes every entity in a batch resolve to one identical self_ref (coarse). Real gemma4 output confirmed: nodes carry DISTINCT per-node `evidence_ids` that are Docling self_refs (`#/texts/59`,`#/texts/61`,`#/texts/60`). Reorder to prefer per-node `evidence_ids[0]` (when it is a `#/` self_ref) over batch `self_refs[0]`.
- **Part C (worker resolution):** `derive_structure_links` resolves chunks via `element_uid_chunk_map` keyed on `DocumentElement.element_uid` (`{page}-{order}-{type}-{hash}`) — a different namespace than the synthesizer's `#/texts/N`. A `#/` value misses the map → fans out to ALL chunks (coarse, pipeline.py:9329-9334). Translate `#/` self_refs → concrete `element_uid` via the `identity_map` (self_ref→element_uid) that ingest already persists in `docling_document.json._enrichments.identity_map` (pipeline.py:4836-4841); the all-chunks fan-out becomes a flagged last resort.

Together: Part A makes provenance non-empty so entities commit; Part B makes the committed `element_uid` the precise per-entity self_ref; Part C maps that self_ref to the exact chunk.

**Tech Stack:** Python, FastAPI (docling-graph, port 8002, COPY image + `patch`-applied library patches), Celery worker (`app/**` bind-mounted, loads code at process start), ArcadeDB, pytest.

**Scope / non-goals:** Precise lineage on the production DoclingDocument chunked path for paginated docs. OUT: the salvage/synthesize FALLBACK path (`synthesize_provenance_from_pass_output`, cid=0) stays coarse — it only fires under full salvage when `build_provenance_from_context` yields [] (Task 6 reports which path was live); page-less sources (gate needs `page is not None`; SA-2 chunks carry pages); raw-text/pre-built paths (`page_numbers=[]`); relationship/edge lineage (system_links / bug #59) is VERIFIED-OR-SCOPED in Task 5, not fixed here.

---

## File Structure

- `docker/docling-graph/patches/0002-...patch` — **MODIFY** (Task 1): add bare `doc_processor.last_chunk_metadata = chunk_metadata` store.
- `docker/docling-graph/Dockerfile` — **MODIFY** (Task 1): patch loop fails loud (`--fuzz=0 || exit 1`).
- `docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py` — **CREATE** (Task 1).
- `docker/docling-graph/app/provenance.py` — **MODIFY** (Task 2): reorder `_resolve_element_uid` to prefer per-node `evidence_ids[0]` (`#/` self_ref) over batch `self_refs[0]`.
- `docker/docling-graph/tests/test_resolve_element_uid_prefers_evidence.py` — **CREATE** (Task 2).
- `app/workers/pipeline.py` — **MODIFY** (Task 3): in `derive_structure_links`, translate `#/` self_ref element_uids via `identity_map` to concrete element_uid before the all-chunks fan-out; fan-out becomes flagged last resort with WARN.
- `tests/unit/test_extracted_from_self_ref_resolution.py` — **CREATE** (Task 3).
- `scripts/run_tests.sh` — **MODIFY** (Task 4): collect the single new docling-graph test (NOT the whole dir — it red-fails on host).
- `scripts/verify_lineage_e2e.py` — **MODIFY** (Task 6): run-scope EXTRACTED_FROM by `pipeline_run_id`; per-entity precision bound (not median-<25%); EXTRACTED_FROM-only trace; run-windowed warnings.

---

## Task 1: Part A — store last_chunk_metadata in patch 0002 + harden Dockerfile (TDD)

**Goal:** The patched CHUNKED-BATCHES branch stores its `chunk_metadata` on `doc_processor.last_chunk_metadata` (proven by a host-runnable test); the Dockerfile patch loop fails the build on any mis-applied patch.

**Files:**
- Create: `docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py`
- Modify: `docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch`
- Modify: `docker/docling-graph/Dockerfile`

**Acceptance Criteria:**
- [ ] A test applies all `docker/docling-graph/patches/*.patch` to a temp copy of `docker/docling-graph/repo`, imports patched `docling_graph.pipeline.stages`, drives `ExtractionStage._extract_from_docling_document` through the CHUNKED-BATCHES branch with fakes, and asserts `fake_doc_processor.last_chunk_metadata` carries the returned `self_refs` + `page_numbers`. FAILS before the patch edit, PASSES after.
- [ ] The store is a BARE assignment `doc_processor.last_chunk_metadata = chunk_metadata` (no try/except — matches `strategy_ops.py:47`/`many_to_one.py:589`), inserted right after `extract_chunks_with_metadata(...)`, before `if context.trace_data:`.
- [ ] The Dockerfile patch loop uses `patch -p1 --fuzz=0 -i "$p" || exit 1`.
- [ ] `patch --fuzz=0 --dry-run` of all five patches (0001–0005) applies in sequence against a clean repo copy with no FAILED and no fuzz.

**Verify:** `python3 -m pytest docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py -v` → 1 passed

**Steps:**

- [ ] **Step 1: Write the failing test** — create `docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py`:

```python
"""Patch-0002 regression: the CHUNKED-BATCHES branch of
ExtractionStage._extract_from_docling_document MUST publish chunk_metadata to
doc_processor.last_chunk_metadata. app/main.py builds the chunk_to_self_refs /
chunk_to_page_numbers provenance maps from it; empty -> element_uid="" -> rows
dropped -> no EXTRACTED_FROM. The fix is in a library PATCH applied at Docker
build, so this test applies the patch stack to a temp repo copy and imports the
PATCHED module in a clean subprocess.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SERVICE_ROOT = _HERE.parent
_REPO = _SERVICE_ROOT / "repo"
_PATCHES = _SERVICE_ROOT / "patches"


def _apply_patches(dst_repo: Path) -> None:
    for p in sorted(_PATCHES.glob("*.patch")):
        subprocess.run(
            ["patch", "-p1", "--fuzz=0", "-i", str(p)],
            cwd=str(dst_repo), check=True, capture_output=True, text=True,
        )


_DRIVER = textwrap.dedent(
    '''
    import sys
    sys.path.insert(0, sys.argv[1])
    from docling_graph.pipeline.stages import ExtractionStage

    CHUNK_META = [
        {"chunk_id": 0, "self_refs": ["#/texts/3"], "page_numbers": [1],
         "evidence_ids": ["#/texts/3"], "evidence_units": [], "token_count": 10},
        {"chunk_id": 1, "self_refs": ["#/texts/4"], "page_numbers": [2],
         "evidence_ids": ["#/texts/4"], "evidence_units": [], "token_count": 12},
    ]

    class FakeDocProcessor:
        chunker = object()
        last_chunk_metadata = []
        def extract_chunks_with_metadata(self, document):
            return (["chunk one text", "chunk two text"], CHUNK_META)

    class FakeBackend:
        def extract_from_chunk_batches(self, *, chunks, chunk_metadata, template, context):
            return {"ok": True}
        def extract_from_markdown(self, **kw):
            raise AssertionError("must take CHUNKED-BATCHES path, not single-blob")

    class FakeExtractor:
        _extraction_contract = "delta"
        def __init__(self):
            self.doc_processor = FakeDocProcessor()
            self.backend = FakeBackend()

    class FakeDoc:
        def export_to_markdown(self):
            raise AssertionError("single-blob path should not run")

    class FakeContext:
        def __init__(self):
            self.extractor = FakeExtractor()
            self.docling_document = FakeDoc()
            self.template = type("T", (), {})
            self.trace_data = None

    stage = ExtractionStage()
    ctx = FakeContext()
    models = stage._extract_from_docling_document(ctx)
    dp = ctx.extractor.doc_processor
    assert models, "expected an extracted model from the chunked path"
    assert dp.last_chunk_metadata, (
        "REGRESSION: chunked-batches path did not store chunk_metadata on "
        "doc_processor.last_chunk_metadata (empty => no EXTRACTED_FROM lineage)"
    )
    refs = [r for row in dp.last_chunk_metadata for r in (row.get("self_refs") or [])]
    pages = [p for row in dp.last_chunk_metadata for p in (row.get("page_numbers") or [])]
    assert "#/texts/3" in refs, f"self_refs not preserved: {refs!r}"
    assert 1 in pages and 2 in pages, f"page_numbers not preserved: {pages!r}"
    print("DRIVER_OK")
    '''
)


def test_chunked_batches_stores_last_chunk_metadata(tmp_path):
    dst = tmp_path / "repo"
    subprocess.run(["cp", "-a", str(_REPO), str(dst)], check=True)
    _apply_patches(dst)
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)
    proc = subprocess.run([sys.executable, str(driver), str(dst)],
                          capture_output=True, text=True)
    assert "DRIVER_OK" in proc.stdout, (
        f"patched-artifact check failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
```

- [ ] **Step 2: Run the test to verify it FAILS.**

Run: `python3 -m pytest docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py -v`
Expected: FAIL — driver's `AssertionError: REGRESSION ...` in captured STDERR.

> Fallback if a host import error occurs: `docker cp docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py eip-mmdpp-docling-graph-1:/app/tests/ && docker exec eip-mmdpp-docling-graph-1 python3 -m pytest /app/tests/test_chunked_batches_stores_chunk_metadata.py -v` (earlier check: `docling_graph.pipeline.stages` imports via the repo path on host).

- [ ] **Step 3: Regenerate patch 0002 with the store line.** Each patch touches a distinct file (0001=orchestrator.py, 0002=stages.py, 0003=prompts.py, 0004=llm_backend.py, 0005=many_to_one.py), so 0002's baseline is the CLEAN repo:

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
rm -rf /tmp/dg_base /tmp/dg_fixed
cp -a docker/docling-graph/repo /tmp/dg_base
cp -a docker/docling-graph/repo /tmp/dg_fixed
patch -p1 -d /tmp/dg_fixed -i "$(pwd)/docker/docling-graph/patches/0002-"*.patch >/dev/null
# hand-edit /tmp/dg_fixed/docling_graph/pipeline/stages.py: insert the 5 lines below
# AFTER `chunks, chunk_metadata = doc_processor.extract_chunks_with_metadata(...)`
# inside `if can_chunk_batch:`, BEFORE `if context.trace_data:`.
```
  The 5 lines to insert (16-space indent, BARE assignment):
```python
                # PATCH 2026-05-31: publish chunk_metadata so app/main.py builds
                # chunk_to_self_refs / chunk_to_page_numbers provenance maps.
                # Mirrors strategy_ops.extract_delta_from_document / many_to_one.py:589.
                # Empty maps -> element_uid="" -> rows dropped -> no EXTRACTED_FROM.
                doc_processor.last_chunk_metadata = chunk_metadata
```
  Regenerate (restore `a/`…`b/` prefixes for `-p1`):
```bash
( cd /tmp && diff -u dg_base/docling_graph/pipeline/stages.py dg_fixed/docling_graph/pipeline/stages.py \
    | sed -e 's#^--- dg_base/#--- a/#' -e 's#^+++ dg_fixed/#+++ b/#' ) > /tmp/0002.new
head -8 /tmp/0002.new
cp /tmp/0002.new docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch
rm -rf /tmp/dg_base /tmp/dg_fixed /tmp/0002.new
```

- [ ] **Step 4: Harden the Dockerfile patch loop** — edit `docker/docling-graph/Dockerfile`:

```dockerfile
RUN cd /app/repo && for p in /app/patches/*.patch; do \
        echo "Applying patch: $p"; \
        patch -p1 --fuzz=0 -i "$p" || exit 1; \
    done \
    && pip install --no-cache-dir --no-deps -e /app/repo
```

- [ ] **Step 5: Run the test to verify it PASSES.**

Run: `python3 -m pytest docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py -v`
Expected: PASS (`DRIVER_OK`).

- [ ] **Step 6: Verify the whole patch stack applies cleanly with zero fuzz.**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
rm -rf /tmp/dgpatchcheck && cp -a docker/docling-graph/repo /tmp/dgpatchcheck
for p in docker/docling-graph/patches/*.patch; do echo "== $p =="; patch -p1 -d /tmp/dgpatchcheck --fuzz=0 --dry-run -i "$(pwd)/$p" || echo "FAILED: $p"; done
rm -rf /tmp/dgpatchcheck
```
Expected: every patch `checking file ...`, no FAILED, no fuzz. (An "offset N lines" note is benign — `--fuzz=0` rejects context fuzz, not offset.)

- [ ] **Step 7: Commit.**

```bash
git add docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py docker/docling-graph/Dockerfile
git commit -m "fix(extraction): patch 0002 must store last_chunk_metadata (+ fail-loud patch loop)

Part A of EXTRACTED_FROM=0 fix: chunked-batches path computed chunk_metadata
(self_refs+pages) but never published doc_processor.last_chunk_metadata, so
app/main.py built empty chunk maps -> element_uid='' -> rows dropped -> no
EXTRACTED_FROM. Mirror the sibling store; harden Dockerfile loop (--fuzz=0||exit1).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Part B — prefer per-node evidence_ids over batch self_refs in `_resolve_element_uid` (TDD)

**Goal:** `_resolve_element_uid` returns the per-node `evidence_ids[0]` (a Docling `#/` self_ref) — the precise per-entity anchor — in preference to the batch-wide `self_refs[0]` that is identical across all nodes in a batch.

**Files:**
- Modify: `docker/docling-graph/app/provenance.py` (`_resolve_element_uid`, lines 205-253)
- Create: `docker/docling-graph/tests/test_resolve_element_uid_prefers_evidence.py`

**Acceptance Criteria:**
- [ ] When `provenance.evidence_ids[0]` is a `#/`-prefixed self_ref, `_resolve_element_uid` returns it INSTEAD of `provenance.self_refs[0]`. When `evidence_ids` is empty / not a `#/` ref, it falls back to the existing order (direct → nested element_uid → self_refs[0] → evidence_ids `#/` scan → chunk_indexes).
- [ ] A node carrying batch-wide `self_refs=["#/texts/10"]` but per-node `evidence_ids=["#/texts/42"]` resolves to `#/texts/42` (precise), not `#/texts/10` (batch-coarse).
- [ ] Existing strategies still work: direct `element_uid` wins over everything; a node with only `self_refs` (no evidence_ids) still resolves to `self_refs[0]`; a node with neither resolves via `chunk_indexes` → `chunk_to_self_refs`.
- [ ] All existing `_resolve_element_uid` behaviors keep passing (run the existing provenance test module if present).

**Verify:** `python3 -m pytest docker/docling-graph/tests/test_resolve_element_uid_prefers_evidence.py -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing test** — create `docker/docling-graph/tests/test_resolve_element_uid_prefers_evidence.py`. Import the resolver directly from the service module (the function is module-level, no app.* dependency):

```python
"""Part B: _resolve_element_uid must prefer the per-node evidence_ids[0]
(precise Docling self_ref) over the batch-wide self_refs[0] (identical across a
batch). Verified empirically: real gemma4 nodes carry distinct per-node
evidence_ids (#/texts/59, #/texts/61, ...) while self_refs is batch-wide.
"""
import importlib.util
from pathlib import Path

_PROV = Path(__file__).resolve().parent.parent / "app" / "provenance.py"
_spec = importlib.util.spec_from_file_location("dg_provenance_under_test", _PROV)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_resolve_element_uid = _mod._resolve_element_uid


def test_prefers_per_node_evidence_id_over_batch_self_refs():
    node = {"provenance": {
        "self_refs": ["#/texts/10"],          # batch-wide, same for all nodes
        "evidence_ids": ["#/texts/42"],        # per-node, this entity's source
    }}
    assert _resolve_element_uid(node, None) == "#/texts/42"


def test_falls_back_to_self_refs_when_no_evidence_ids():
    node = {"provenance": {"self_refs": ["#/texts/10"], "evidence_ids": []}}
    assert _resolve_element_uid(node, None) == "#/texts/10"


def test_non_selfref_evidence_id_does_not_win():
    # evidence_ids that are not "#/" refs must not be chosen as element_uid;
    # fall through to self_refs[0].
    node = {"provenance": {"self_refs": ["#/texts/10"], "evidence_ids": ["e1", "e2"]}}
    assert _resolve_element_uid(node, None) == "#/texts/10"


def test_direct_element_uid_still_wins():
    node = {"element_uid": "p1-2-text-abcd",
            "provenance": {"evidence_ids": ["#/texts/42"], "self_refs": ["#/texts/10"]}}
    assert _resolve_element_uid(node, None) == "p1-2-text-abcd"


def test_chunk_indexes_fallback_unchanged():
    node = {"provenance": {"chunk_indexes": [0]}}
    assert _resolve_element_uid(node, {0: ["#/texts/7"]}) == "#/texts/7"
```

- [ ] **Step 2: Run the test to verify it FAILS.**

Run: `python3 -m pytest docker/docling-graph/tests/test_resolve_element_uid_prefers_evidence.py -v`
Expected: `test_prefers_per_node_evidence_id_over_batch_self_refs` FAILS (current code returns `#/texts/10` from self_refs Strategy 3 before reaching evidence_ids). Others pass.

- [ ] **Step 3: Reorder `_resolve_element_uid`.** In `docker/docling-graph/app/provenance.py`, move the per-node evidence_ids `#/` check ABOVE the self_refs check. Replace the body from the `nested = prov.get("element_uid")` block onward (lines 228-253) with:

```python
    nested = prov.get("element_uid")
    if isinstance(nested, str) and nested:
        return nested

    # Prefer the per-node evidence_ids self_ref: the IR normalizer narrows
    # evidence_ids PER NODE (ir_normalizer._attach_evidence_to_prov) but copies
    # self_refs BATCH-WIDE onto every node, so self_refs[0] is identical across a
    # batch (coarse) while evidence_ids[0] is this entity's actual source element.
    evidence_ids = prov.get("evidence_ids")
    if isinstance(evidence_ids, list):
        for eid in evidence_ids:
            if isinstance(eid, str) and eid.startswith("#/"):
                return eid

    self_refs = prov.get("self_refs")
    if isinstance(self_refs, list) and self_refs:
        first = self_refs[0]
        if isinstance(first, str) and first:
            return first

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
  Also update the docstring resolution-order list (lines 211-218) to reflect that evidence_ids `#/` now precedes self_refs.

- [ ] **Step 4: Run the test to verify it PASSES.**

Run: `python3 -m pytest docker/docling-graph/tests/test_resolve_element_uid_prefers_evidence.py -v`
Expected: all 5 pass.

- [ ] **Step 5: Run any existing provenance tests to confirm no regression.**

Run: `ls docker/docling-graph/tests/ | grep -i provenance` then run each found module via `python3 -m pytest docker/docling-graph/tests/<that_file>.py -v`. Expected: pass (no behavior other than the evidence-before-self_refs reorder changed).

- [ ] **Step 6: Commit.**

```bash
git add docker/docling-graph/app/provenance.py docker/docling-graph/tests/test_resolve_element_uid_prefers_evidence.py
git commit -m "fix(lineage): prefer per-node evidence_ids over batch self_refs (Part B)

_resolve_element_uid returned the batch-wide self_refs[0] (identical for every
node in a batch) before the per-node evidence_ids[0] -> all entities in a batch
got the same coarse anchor. Real gemma4 output carries distinct per-node
evidence_ids (Docling self_refs); prefer them so element_uid is per-entity precise.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Part C — resolve `#/` self_ref element_uids to precise chunks in the worker (TDD)

**Goal:** In `derive_structure_links`, a mention whose `element_uid` is a `#/` self_ref resolves to its SPECIFIC chunk(s) via the persisted `identity_map`, instead of fanning out to all text chunks.

**Files:**
- Modify: `app/workers/pipeline.py` (`derive_structure_links`: helper near line 9253; mention loop ~9323-9334)
- Create: `tests/unit/test_extracted_from_self_ref_resolution.py`

**Acceptance Criteria:**
- [ ] A pure helper `_resolve_mention_chunks(element_uid, element_uid_chunk_map, identity_map, all_text_chunk_ids) -> (chunk_ids, is_coarse)` resolves: (1) direct hit in `element_uid_chunk_map`; (2) `#/` self_ref → `identity_map[self_ref]` → `element_uid_chunk_map`; (3) last resort = all chunks, `is_coarse=True`.
- [ ] Unit test: identity_map `#/texts/5→X`, element_uid_chunk_map `X→[chunkA]`, element_uid `"#/texts/5"` → `(["chunkA"], False)`; a `#/` self_ref absent from identity_map → `(all_text_chunk_ids, True)`; a concrete `{page}-{order}-...` element_uid in the map → `([chunkA], False)`.
- [ ] `derive_structure_links` loads `identity_map` once (from the persisted docling_document.json `_enrichments.identity_map`; `{}` on any miss) and calls the helper for each mention; when `is_coarse` is True it logs a WARNING naming the unresolved self_ref (no silent coarsening).
- [ ] No regression: a mention with a concrete element_uid still resolves via `element_uid_chunk_map` unchanged.

**Verify:** `python3 -m pytest tests/unit/test_extracted_from_self_ref_resolution.py -v` → all pass

**Steps:**

- [ ] **Step 1: Confirm the identity_map load path.**

```bash
sed -n '4832,4848p' app/workers/pipeline.py     # identity_map written into _enrichments at ingest
grep -n "_build_docling_document_json\|docling_document.json\|_enrichments\|download_bytes_sync\|get_object" app/workers/pipeline.py | grep -iE "docling_document.json|_enrichments|download" | head
```
  Identify the existing way the worker reads the derived `docling_document.json` from MinIO (the same base path it is written to at pipeline.py:4851-4856). Use that to implement `_load_identity_map`; do not invent a new storage path.

- [ ] **Step 2: Write the failing unit test** — create `tests/unit/test_extracted_from_self_ref_resolution.py`:

```python
"""Part C: a synthesizer self_ref element_uid must resolve to its SPECIFIC chunk
via identity_map, not fan out to all chunks."""
from app.workers.pipeline import _resolve_mention_chunks


def test_self_ref_resolves_to_specific_chunk_via_identity_map():
    resolved, coarse = _resolve_mention_chunks(
        "#/texts/5",
        {"p1-2-text-abcd": ["chunkA"], "p1-3-text-ef01": ["chunkB"]},
        {"#/texts/5": "p1-2-text-abcd"},
        ["chunkA", "chunkB", "chunkC"],
    )
    assert resolved == ["chunkA"]
    assert coarse is False


def test_unmapped_self_ref_falls_back_to_all_chunks_flagged_coarse():
    resolved, coarse = _resolve_mention_chunks(
        "#/texts/99", {"x": ["chunkA"]}, {"#/texts/5": "x"}, ["chunkA", "chunkB"],
    )
    assert resolved == ["chunkA", "chunkB"]
    assert coarse is True


def test_concrete_element_uid_resolves_directly_no_regression():
    resolved, coarse = _resolve_mention_chunks(
        "p1-2-text-abcd", {"p1-2-text-abcd": ["chunkA"]}, {}, ["chunkA", "chunkB"],
    )
    assert resolved == ["chunkA"]
    assert coarse is False
```

- [ ] **Step 3: Run the test to verify it FAILS.**

Run: `python3 -m pytest tests/unit/test_extracted_from_self_ref_resolution.py -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_mention_chunks'`.

- [ ] **Step 4: Implement the helper + wire it in.** Add the pure helper near `derive_structure_links` in `app/workers/pipeline.py`:

```python
def _resolve_mention_chunks(
    element_uid: str,
    element_uid_chunk_map: dict[str, list[str]],
    identity_map: dict[str, str],
    all_text_chunk_ids: list[str],
) -> tuple[list[str], bool]:
    """Resolve a mention's element_uid to concrete chunk ids. Returns
    (chunk_ids, is_coarse). Order: (1) direct hit in element_uid_chunk_map;
    (2) Docling self_ref ('#/...') -> identity_map[self_ref] -> element_uid_chunk_map;
    (3) last resort: all text chunks (is_coarse=True), caller WARNs."""
    direct = element_uid_chunk_map.get(element_uid)
    if direct:
        return direct, False
    if isinstance(element_uid, str) and element_uid.startswith("#/"):
        mapped_uid = identity_map.get(element_uid)
        if mapped_uid:
            resolved = element_uid_chunk_map.get(mapped_uid)
            if resolved:
                return resolved, False
        return all_text_chunk_ids, True
    return [], False
```
  Then in `derive_structure_links`, load the map once (using the Step 1 method) before the mention loop and replace the inline `#/`-fan-out (~9329-9334):

```python
            identity_map = _load_identity_map(document_id)  # {} if unavailable
            ...
            for mention in graph_extraction.graph_json.get("mentions", []):
                ...
                euid = mention.get("element_uid", "")
                resolved_chunks, is_coarse = _resolve_mention_chunks(
                    euid, element_uid_chunk_map, identity_map, all_text_chunk_ids,
                )
                if is_coarse:
                    logger.warning(
                        "derive_structure_links: self_ref %r unresolved via "
                        "identity_map; coarse fan-out across %d chunks (entity=%s)",
                        euid, len(all_text_chunk_ids), mention.get("entity_name"),
                    )
                for chunk_id in resolved_chunks:
                    edge_records.append((name, etype, chunk_id, eid, src_rid))
                    if eid:
                        mentioned_entity_ids.add(eid)
```
  Implement `_load_identity_map(document_id) -> dict[str,str]` using the existing derived-JSON fetch; read `["_enrichments"]["identity_map"]`; return `{}` on any miss/exception (never raise).

- [ ] **Step 5: Run the test to verify it PASSES.**

Run: `python3 -m pytest tests/unit/test_extracted_from_self_ref_resolution.py -v`
Expected: all 3 pass.

- [ ] **Step 6: Commit.**

```bash
git add app/workers/pipeline.py tests/unit/test_extracted_from_self_ref_resolution.py
git commit -m "fix(lineage): resolve #/ self_ref element_uids to precise chunks (Part C)

derive_structure_links keyed element_uid_chunk_map on DocumentElement.element_uid
({page}-{order}-...) but mentions carry #/texts/N self_refs -> missed -> fanned
out to ALL chunks. Translate self_ref -> element_uid via the persisted
identity_map (_enrichments.identity_map); fan-out is now a flagged last resort.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Wire the single new docling-graph test into the suite

**Goal:** The new patch-regression test is collected by the standard runner WITHOUT turning the suite red (the full `docker/docling-graph/tests/` dir errors on the host — namespace collision — so collect only the safe new tests).

**Files:**
- Modify: `scripts/run_tests.sh`

**Acceptance Criteria:**
- [ ] `scripts/run_tests.sh` runs the two new host-safe tests explicitly: `docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py` and `docker/docling-graph/tests/test_resolve_element_uid_prefers_evidence.py` — NOT the whole dir.
- [ ] The added block's exit status is wired into the runner's pass/fail aggregation like sibling blocks.
- [ ] Running the runner collects and passes both new tests; it does NOT collect the ~360 container-only tests in that dir (which fail on host).

**Verify:** `grep -q "test_chunked_batches_stores_chunk_metadata" scripts/run_tests.sh && echo WIRED` → WIRED

**Steps:**

- [ ] **Step 1: Read the runner** to mirror its style: `sed -n '120,170p' scripts/run_tests.sh`.

- [ ] **Step 2: Add an explicit two-file block** next to the `pytest tests/unit` block, matching its env/exit handling:

```bash
PYTHONPATH="docker/docling-graph/repo:${PYTHONPATH:-}" pytest \
  docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py \
  docker/docling-graph/tests/test_resolve_element_uid_prefers_evidence.py -q
```
  Wire its exit code into the aggregate exactly like the surrounding `pytest` blocks. Do NOT use the bare directory (`docker/docling-graph/tests`) — earlier verification showed ~73 host failures from an `app` namespace collision; those are container-only.

- [ ] **Step 3: Verify it collects + passes both, and does not pull the red dir.**

Run: `PYTHONPATH="docker/docling-graph/repo:${PYTHONPATH:-}" pytest docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py docker/docling-graph/tests/test_resolve_element_uid_prefers_evidence.py -q`
Expected: 6 passed (1 + 5), 0 errors.

- [ ] **Step 4: Commit.**

```bash
git add scripts/run_tests.sh
git commit -m "test(harness): collect the new docling-graph lineage tests in run_tests.sh

Wire the two host-safe patch/resolver tests into the standard runner (the full
docker/docling-graph/tests dir is container-only and errors on host, so collect
the two new files explicitly).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Verify (or scope out) relationship/edge lineage

**Goal:** Determine whether Part A+B+C also restores lineage for relationship edges (`system_links` / bug #59), and either fix it or document the boundary so Task 6's gate is not misread as covering edges.

**Files:** (investigation; code only if a trivial parallel gap is found)
- Possibly Modify: `app/workers/pipeline.py` (relationship-provenance resolution, `_import_graph_phase_domain_edges` ~1460-1521)

**Acceptance Criteria:**
- [ ] Documented finding in memory `project_extracted_from_root_cause.md`: does relationship provenance flow through the same `evidence_ids`/`element_uid`/`last_chunk_metadata` path; will Part A+B+C populate edge lineage, or is it separate?
- [ ] If edges share the exact gap and the same fix applies → fix + a unit test; otherwise → explicit scope-out note referencing bug #59.

**Verify:** `grep -q "relationship" /home/josh/.claude/projects/-home-josh-development-EIP-MMDPP/memory/project_extracted_from_root_cause.md && echo DOCUMENTED` → DOCUMENTED

**Steps:**

- [ ] **Step 1: Trace relationship provenance** response→merge→commit:

```bash
sed -n '1460,1521p' app/workers/pipeline.py
grep -n "relationship_provenance\|ExtractionRelationshipProvenance" app/services/extraction_merge.py | head
```
  `ExtractionRelationshipProvenance` carries `evidence_ids` + `self_refs` (extraction_merge.py:192-194); determine whether committed edges derive chunk lineage from them and whether the same self_ref→element_uid resolution is needed.

- [ ] **Step 2: Decide + record.** If the same gap and helper apply → add resolution + a unit test mirroring Task 3 and commit. Else → write a precise scope-out paragraph in `project_extracted_from_root_cause.md` (what's covered, what's not, why) referencing bug #59. Commit the doc/code either way.

---

## Task 6: Harden the verifier + end-to-end PRECISE-lineage gate (USER-ORDERED)

**Goal:** Make `verify_lineage_e2e.py` prove THIS run restored PRECISE, run-attributed lineage (not pre-satisfied by existing MENTIONED_IN / stale entities, not false-failed by idempotent upsert), then run it on a fresh SA-2 graph_only run on the fully-deployed fix and confirm precise entity→chunk→page lineage on the graph-query endpoint.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Modify: `scripts/verify_lineage_e2e.py`

**Acceptance Criteria:**
- [ ] **Deploy precondition (captured):** docling-graph rebuilt with Tasks 1+2 (`docker compose -p eip-mmdpp build docling-graph` shows `Applying patch` for 0002, no failure) and recreated; worker-1 AND worker-graph-1 restarted for Task 3; `StartedAt` advanced on all three; live container has `last_chunk_metadata = chunk_metadata` in stages.py AND the reordered evidence_ids check in provenance.py; gemma4 pool idle (~0.24s probe). Also: the persisted `docling_document.json._enrichments.identity_map` for the SA-2 doc is NON-EMPTY (else Part C degrades to fan-out — abort and require a full reingest).
- [ ] **Verifier #1 — run-scoped EXTRACTED_FROM:** accept `--run`; count `EXTRACTED_FROM` filtered by `WHERE pipeline_run_id = :run` (the edge carries `pipeline_run_id`, arcadedb_graph.py:808/2641). PASS requires run-scoped count > 0. This is the hard discriminator.
- [ ] **Verifier #2 — drop the false-FAIL entity-count delta** (idempotent upsert → 0 on re-run); keep an absolute `entities ≥ merged_baseline` recall sanity check, not as the lineage proof.
- [ ] **Verifier #3 — per-entity precision (run-scoped):** for entities with run-scoped `EXTRACTED_FROM`, assert the MAX per-entity distinct target-chunk count is small (≤ 5 chunks, NOT a median/percent — a max bounds the worst case). Report the full distribution.
- [ ] **Verifier #4 — EXTRACTED_FROM-only trace + run-windowed signals:** the field-trace check traverses `out('EXTRACTED_FROM')` ONLY (drop MENTIONED_IN from both the trace and the edge-existence check); the provenance-drop / gate-rejection warning counts are taken from the run's Postgres window (run started_at→finished_at), not a fixed `--since 3h`.
- [ ] **Fresh SA-2 graph_only run** on the deployed fix: run-scoped `EXTRACTED_FROM` > 0; 0 provenance-drop warnings in the run window; provenance rows with non-empty `element_uid` > 0; precision MAX ≤ 5; `/v1/graph/query` SNR-75 returns `sources[]` with non-null `document_id`+`page_number`.
- [ ] `python3 scripts/verify_lineage_e2e.py --run <run_id>` → "ALL CHECKS PASS".

**Verify:** `python3 scripts/verify_lineage_e2e.py --run <run_id>` → "ALL CHECKS PASS"

**Steps:**

- [ ] **Step 1: Deploy all parts + capture preconditions.**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
docker compose -p eip-mmdpp build docling-graph 2>&1 | grep -E "Applying patch|Built|ERROR|FAILED"
docker compose -p eip-mmdpp up -d --force-recreate docling-graph
docker restart eip-mmdpp-worker-1 eip-mmdpp-worker-graph-1
for c in docling-graph-1 worker-1 worker-graph-1; do docker inspect eip-mmdpp-$c --format "$c StartedAt: {{.State.StartedAt}}"; done
docker exec eip-mmdpp-docling-graph-1 grep -c "last_chunk_metadata = chunk_metadata" /app/repo/docling_graph/pipeline/stages.py
docker exec eip-mmdpp-docling-graph-1 grep -c "per-node evidence_ids" /app/app/provenance.py
for h in 10.0.1.121 10.0.1.109; do printf "%s " "$h"; curl -s --max-time 30 -o /dev/null -w '%{time_total}s\n' http://$h:11434/api/generate -d '{"model":"gemma4:31b","prompt":"hi","stream":false,"options":{"num_predict":1}}'; done
```
  Then assert the identity_map is present for the doc (fetch the persisted docling_document.json from MinIO via the same path the worker uses; confirm `_enrichments.identity_map` is non-empty). If empty → STOP; a graph_only run cannot fix it (identity_map is written only on full ingest) — escalate to the user for a full reingest decision.

- [ ] **Step 2: Apply verifier changes #1–#4** per the acceptance criteria in `scripts/verify_lineage_e2e.py`. Run-scope the EXTRACTED_FROM count and the per-entity precision query by `pipeline_run_id`; drop the entity-delta check; add the MAX-per-entity precision bound; make the trace EXTRACTED_FROM-only; derive the log window from the run's Postgres `started_at`/`finished_at`. Keep the `[PASS]/[FAIL]` line format + "ALL CHECKS PASS" sentinel.

- [ ] **Step 3: Capture run-scoped pre-state** (the run doesn't exist yet, so pre is implicitly 0 for its run_id — note global EXTRACTED_FROM for context):

```bash
ADB() { curl -s -u root:eip_arcadedb_secret -X POST http://localhost:2480/api/v1/command/eip_knowledge_graph -H "Content-Type: application/json" -d "{\"language\":\"sql\",\"command\":\"$1\"}"; }
echo -n "EXTRACTED_FROM global pre: "; ADB "SELECT count(*) AS c FROM EXTRACTED_FROM" | python3 -c "import sys,json;print(json.load(sys.stdin)['result'][0]['c'])"
```

- [ ] **Step 4: Trigger the SA-2 graph_only reingest.**

```bash
DOC=ddaa9e36-2854-47c3-bc94-ff38d531dafd
curl -s -X POST "http://localhost:8005/v1/documents/$DOC/reingest" -H "Content-Type: application/json" \
  -d '{"mode":"graph_only","ontology_bundle_key":"air_defense_v3_merged_v1"}' | python3 -m json.tool
```
  Record `pipeline_run_id`. Multi-hour run; monitor in background (lives in Celery). NOTE: EXTRACTED_FROM is append-only and graph_only does not purge it — run-scoping by `pipeline_run_id` (Verifier #1/#3) is what isolates this run from prior coarse edges; do NOT rely on a global delta.

- [ ] **Step 5: At terminal status, run the hardened verifier + endpoint + precision checks.**

```bash
RUN=<pipeline_run_id>
ADB() { curl -s -u root:eip_arcadedb_secret -X POST http://localhost:2480/api/v1/command/eip_knowledge_graph -H "Content-Type: application/json" -d "{\"language\":\"sql\",\"command\":\"$1\"}"; }
echo -n "EXTRACTED_FROM run-scoped: "; ADB "SELECT count(*) AS c FROM EXTRACTED_FROM WHERE pipeline_run_id='$RUN'" | python3 -c "import sys,json;print(json.load(sys.stdin)['result'][0]['c'])"
python3 scripts/verify_lineage_e2e.py --run "$RUN"
curl -s -X POST "http://localhost:8005/v1/graph/query" -H "Content-Type: application/json" \
  -d '{"query":"SNR-75","top_k":1,"hop_count":1}' | python3 -c "import sys,json;r=json.load(sys.stdin);print(json.dumps(r[0].get('sources'),indent=2) if r else 'NO RESULT')"
```
  Expected: run-scoped EXTRACTED_FROM > 0; verifier "ALL CHECKS PASS"; `sources[]` carries document_id+page; per-entity max target count ≤ 5 (precise).

- [ ] **Step 6: Record outcome in memory** (`project_extracted_from_root_cause.md`): fix verified; run-scoped EXTRACTED_FROM count; precision distribution (per-entity target counts, max); which provenance path was live (build_provenance_from_context vs synthesize); any residual coarse cases.

---

## Notes for the executor

- **Sequencing / deploy gate:** Tasks 1+2 (docling-graph: patch + resolver) and Task 3 (worker) are independent edits. ALL THREE must be deployed before Task 6's gate — this is Task 6 Step 1, captured as a precondition (not buried in notes). Task 1/2 need an image rebuild + recreate; Task 3 is worker `app/**` (bind-mounted) → `docker restart eip-mmdpp-worker-1 eip-mmdpp-worker-graph-1` (Celery loads code at process start; restart BOTH — worker-1 is a catch-all that also consumes the `graph` queue).
- **Deploy semantics** (memory): docling-graph is COPY-based → rebuild; compose from this worktree needs `-p eip-mmdpp`; `docker compose restart` does NOT reload code into a running worker → `docker restart <container>`. Verify `StartedAt` advanced.
- **identity_map is full-ingest-only** (pipeline.py:4836, in `prepare_document`); `reingest_graph_only` does NOT rewrite it. Task 6 Step 1 asserts it is present for SA-2 before running; if absent, a graph_only run can't fix lineage — a full reingest is required.
- **EXTRACTED_FROM is append-only, never purged**, built with blind `CREATE EDGE` (arcadedb_graph.py:2650); graph_only does not purge the extraction layer. The verifier MUST scope by `pipeline_run_id` (the edge carries it) for both the count and precision — a global delta is meaningless and a re-run accumulates duplicate edges.
- **Never** use `POST /v1/documents/{id}/cancel` — it hard-deletes the document + all derived data. To stop a run, revoke tasks + reset status; ask first.
- **Truncation caveat:** Part A's store is in the chunked branch BEFORE the LLM call, so it survives truncation. But if a pass fully salvages and `build_provenance_from_context` yields [], the `synthesize_provenance_from_pass_output` fallback (cid=0) is coarse and Parts B/C don't help it — Task 6 Step 6 records which path was live.
