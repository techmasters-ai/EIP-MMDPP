# EXTRACTED_FROM Precise-Lineage Fix — `last_chunk_metadata` store + self_ref resolution

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore PRECISE entity→chunk lineage (`EXTRACTED_FROM` edges where each entity links to its actual source chunk(s)+page, not all chunks) by (A) making the docling-graph chunked-batches path publish `chunk_metadata` to `doc_processor.last_chunk_metadata`, and (B) resolving the synthesizer's Docling self_refs to concrete chunks in the worker so the all-chunks fan-out is avoided.

**Architecture:** Two coupled root causes, both verified (memory `project_extracted_from_root_cause.md`, task #26; confirmed by 3 dissent reviews of the prior single-part plan).
- **Part A (gate unblock):** `docker/docling-graph/patches/0002-...patch`'s CHUNKED-BATCHES branch (production path for DoclingDocument input) builds a correct `chunk_metadata` (chunk_id+self_refs+page_numbers) but never assigns `doc_processor.last_chunk_metadata`. `app/main.py` reads that field to build the `chunk_to_self_refs`/`chunk_to_page_numbers` maps both provenance builders depend on; empty → `element_uid=""`/`page=null` on every entity → the worker lineage gate (`_partition_entities_by_lineage`, pipeline.py:481, requires element_uid AND page≠None) rejects everything → `EXTRACTED_FROM` never built. Sibling chunking paths (`strategy_ops.py:47`/`:73`, `many_to_one.py:589`) all do this store with a bare assignment; patch 0002 omitted it.
- **Part B (precision):** Even with Part A, the synthesizer emits Docling self_refs (`#/texts/N`) as `element_uid`. The worker's `derive_structure_links` resolves chunks via `element_uid_chunk_map`, keyed on `DocumentElement.element_uid` (`{page}-{order}-{type}-{hash}`) — a DIFFERENT namespace (code comment, pipeline.py:9310-9320). A `#/...` value misses the map and hits the `euid.startswith("#/")` fallback (pipeline.py:9329-9334) that fans the entity out to ALL text chunks (coarse). The bridge to fix this ALREADY EXISTS: ingest persists a `self_ref → element_uid` `identity_map` in `docling_document.json._enrichments.identity_map` (pipeline.py:4836-4841). Part B loads it and translates `#/...` self_refs → concrete `element_uid` → existing `element_uid_chunk_map` → precise chunk(s).

**Tech Stack:** Python, FastAPI (docling-graph service, port 8002, COPY image + `patch`-applied library patches), Celery worker (`app/**` bind-mounted, loads code at process start), ArcadeDB graph, pytest.

**Scope / non-goals:** Restores precise lineage on the production DoclingDocument chunked path for paginated docs. OUT: page-less sources (gate also requires `page is not None`; SA-2 chunks carry pages so it passes — page-less chunks remain a separate policy call); raw-text/pre-built-chunk paths that set `page_numbers=[]`; the gemma4 truncation/contention behavior (truncation-independent); relationship/edge lineage for `system_links` (bug #59) is VERIFIED-OR-SCOPED-OUT in Task 4, not fixed here.

---

## File Structure

- `docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch` — **MODIFY** (Task 1): add the `doc_processor.last_chunk_metadata = chunk_metadata` store (bare assignment, matching siblings).
- `docker/docling-graph/Dockerfile` — **MODIFY** (Task 1): harden the patch loop to fail loud (`--fuzz=0 || exit 1`) so a mis-applied patch fails the build instead of silently shipping.
- `docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py` — **CREATE** (Task 1): host-runnable test that applies the patch stack to a temp repo copy and asserts the store happens.
- `app/workers/pipeline.py` — **MODIFY** (Task 3): in `derive_structure_links`, translate synthesizer `#/...` self_ref element_uids via the persisted `identity_map` to concrete `element_uid` before the all-chunks fan-out; the fan-out becomes a last-resort only.
- `tests/unit/test_extracted_from_self_ref_resolution.py` — **CREATE** (Task 3): unit test that a `#/texts/N` mention resolves to the specific chunk via identity_map, NOT all chunks.
- `scripts/verify_lineage_e2e.py` — **MODIFY** (Task 5): add run-attributed `EXTRACTED_FROM` pre→post DELTA as the hard discriminator; replace the entity-count-delta check (false-FAILs on idempotent upsert) with a lineage-edge delta; add a per-entity precision assertion (entity links to a SMALL chunk set, not all).

---

## Task 1: Part A — store `last_chunk_metadata` in patch 0002 + harden Dockerfile (TDD)

**Goal:** The patched CHUNKED-BATCHES branch stores its `chunk_metadata` on `doc_processor.last_chunk_metadata`, proven by a host-runnable test; and the Dockerfile patch loop fails the build on any mis-applied patch.

**Files:**
- Create: `docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py`
- Modify: `docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch`
- Modify: `docker/docling-graph/Dockerfile`

**Acceptance Criteria:**
- [ ] A test applies all `docker/docling-graph/patches/*.patch` to a temp copy of `docker/docling-graph/repo`, imports patched `docling_graph.pipeline.stages`, drives `ExtractionStage._extract_from_docling_document` through the CHUNKED-BATCHES branch with fakes, and asserts `fake_doc_processor.last_chunk_metadata` carries the returned `self_refs` + `page_numbers`. FAILS before the patch edit, PASSES after.
- [ ] The store is a BARE assignment `doc_processor.last_chunk_metadata = chunk_metadata` (no try/except — matches `strategy_ops.py:47`/`many_to_one.py:589`), inserted right after `extract_chunks_with_metadata(...)`, before `if context.trace_data:`.
- [ ] The Dockerfile patch loop uses `patch -p1 --fuzz=0 -i "$p" || exit 1` (fail loud on fuzz/reject) instead of `patch -p1 < "$p"`.
- [ ] `patch --dry-run` of all five patches (0001–0005) applies in sequence against a clean repo copy with **offset 0, no fuzz, no FAILED**.

**Verify:** `python3 -m pytest docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py -v` → 1 passed

**Steps:**

- [ ] **Step 1: Write the failing test.** Create `docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py`:

```python
"""Patch-0002 regression: the CHUNKED-BATCHES branch of
ExtractionStage._extract_from_docling_document MUST publish its chunk_metadata
to doc_processor.last_chunk_metadata. app/main.py builds the
chunk_to_self_refs / chunk_to_page_numbers provenance maps from it; missing the
store yields empty maps -> element_uid="" -> no EXTRACTED_FROM lineage.

The fix lives in a library PATCH applied at Docker build, so this test applies
the full patch stack to a temp copy of docker/docling-graph/repo and imports the
PATCHED module in a clean subprocess, exercising the real patched artifact.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

_HERE = Path(__file__).resolve().parent           # docker/docling-graph/tests
_SERVICE_ROOT = _HERE.parent                       # docker/docling-graph
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
    sys.path.insert(0, sys.argv[1])  # patched repo copy first

    from docling_graph.pipeline.stages import ExtractionStage

    CHUNK_META = [
        {"chunk_id": 0, "self_refs": ["#/texts/3"], "page_numbers": [1],
         "evidence_ids": ["#/texts/3"], "evidence_units": [], "token_count": 10},
        {"chunk_id": 1, "self_refs": ["#/texts/4"], "page_numbers": [2],
         "evidence_ids": ["#/texts/4"], "evidence_units": [], "token_count": 12},
    ]

    class FakeDocProcessor:
        chunker = object()              # truthy -> can_chunk_batch True
        last_chunk_metadata = []        # field under test; starts empty
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
        "doc_processor.last_chunk_metadata (app/main.py reads this; empty => "
        "no EXTRACTED_FROM lineage)"
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
    proc = subprocess.run(
        [sys.executable, str(driver), str(dst)],
        capture_output=True, text=True,
    )
    assert "DRIVER_OK" in proc.stdout, (
        f"patched-artifact check failed.\nSTDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
```

- [ ] **Step 2: Run the test to verify it FAILS.**

Run: `python3 -m pytest docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py -v`
Expected: FAIL — driver's `AssertionError: REGRESSION ...` in captured STDERR (patched branch computes `chunk_metadata` but never assigns `last_chunk_metadata`).

> Fallback if a host import error occurs (host venv missing an optional dep): run inside the container —
> `docker cp docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py eip-mmdpp-docling-graph-1:/app/tests/ && docker exec eip-mmdpp-docling-graph-1 python3 -m pytest /app/tests/test_chunked_batches_stores_chunk_metadata.py -v`
> (Earlier check: `docling_graph.pipeline.stages` imports via the repo path on the host, so the host run is expected to work.)

- [ ] **Step 3: Regenerate patch 0002 with the store line.** Each patch touches a distinct file (verified: 0001=orchestrator.py, 0002=stages.py, 0003=prompts.py, 0004=llm_backend.py, 0005=many_to_one.py), so 0002's baseline is the CLEAN repo:

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
rm -rf /tmp/dg_base /tmp/dg_fixed
cp -a docker/docling-graph/repo /tmp/dg_base
cp -a docker/docling-graph/repo /tmp/dg_fixed
patch -p1 -d /tmp/dg_fixed -i "$(pwd)/docker/docling-graph/patches/0002-"*.patch >/dev/null  # apply CURRENT 0002
# hand-edit ONLY /tmp/dg_fixed/docling_graph/pipeline/stages.py: insert the 5 lines below
# immediately AFTER `chunks, chunk_metadata = doc_processor.extract_chunks_with_metadata(...)`
# inside the `if can_chunk_batch:` branch, BEFORE the following `if context.trace_data:`.
```
  The 5 lines to insert (16-space indent, BARE assignment — matches siblings; no try/except):
```python
                # PATCH 2026-05-31: publish chunk_metadata so app/main.py can build
                # chunk_to_self_refs / chunk_to_page_numbers provenance maps.
                # Mirrors strategy_ops.extract_delta_from_document / many_to_one.py:589.
                # Empty maps -> element_uid="" -> lineage gate rejects all -> no EXTRACTED_FROM.
                doc_processor.last_chunk_metadata = chunk_metadata
```
  Regenerate 0002 as the clean→fixed diff of stages.py, restoring the `a/`…`b/` prefixes `-p1` expects:
```bash
( cd /tmp && diff -u dg_base/docling_graph/pipeline/stages.py dg_fixed/docling_graph/pipeline/stages.py \
    | sed -e 's#^--- dg_base/#--- a/#' -e 's#^+++ dg_fixed/#+++ b/#' ) > /tmp/0002.new
head -8 /tmp/0002.new   # sanity: `--- a/docling_graph/pipeline/stages.py`, `+++ b/...`, then `@@`
cp /tmp/0002.new docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch
rm -rf /tmp/dg_base /tmp/dg_fixed /tmp/0002.new
```

- [ ] **Step 4: Harden the Dockerfile patch loop (fail-loud).** Edit `docker/docling-graph/Dockerfile` — change the patch loop:

```dockerfile
RUN cd /app/repo && for p in /app/patches/*.patch; do \
        echo "Applying patch: $p"; \
        patch -p1 --fuzz=0 -i "$p" || exit 1; \
    done \
    && pip install --no-cache-dir --no-deps -e /app/repo
```
  (Was `patch -p1 < "$p"` with no fuzz guard and no failure propagation — a mis-applied patch built green.)

- [ ] **Step 5: Run the test to verify it PASSES.**

Run: `python3 -m pytest docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py -v`
Expected: PASS (driver prints `DRIVER_OK`).

- [ ] **Step 6: Verify the whole patch stack applies cleanly with zero fuzz.**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
rm -rf /tmp/dgpatchcheck && cp -a docker/docling-graph/repo /tmp/dgpatchcheck
for p in docker/docling-graph/patches/*.patch; do echo "== $p =="; patch -p1 -d /tmp/dgpatchcheck --fuzz=0 --dry-run -i "$(pwd)/$p" || echo "FAILED: $p"; done
rm -rf /tmp/dgpatchcheck
```
Expected: every patch `checking file ...`, no "FAILED", no Hunk failures, no "offset"/"fuzz" notes.

- [ ] **Step 7: Commit.**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
git add docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py docker/docling-graph/Dockerfile
git commit -m "fix(extraction): patch 0002 must store last_chunk_metadata (+ fail-loud patch loop)

Part A of the EXTRACTED_FROM=0 fix: the DoclingDocument chunked-batches path
computed chunk_metadata (self_refs+pages) but never published it to
doc_processor.last_chunk_metadata, so app/main.py built empty chunk maps ->
element_uid=''/page=null -> lineage gate rejected all -> no EXTRACTED_FROM.
Mirror the store the sibling chunking paths do. Also harden the Dockerfile
patch loop (--fuzz=0 || exit 1) so a mis-applied patch fails the build.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Wire the docling-graph test dir into the suite

**Goal:** The new `docker/docling-graph/tests/` test (and the ~60 existing ones there) are collected by the standard runner, so this fix can't silently regress.

**Files:**
- Modify: `scripts/run_tests.sh`

**Acceptance Criteria:**
- [ ] `scripts/run_tests.sh` runs `pytest docker/docling-graph/tests` (in addition to `tests/unit|integration|e2e`), so `test_chunked_batches_stores_chunk_metadata.py` is collected by the standard run.
- [ ] Running `scripts/run_tests.sh` (or the targeted pytest line it adds) collects and passes the new test.

**Verify:** `grep -q "docling-graph/tests" scripts/run_tests.sh && echo WIRED` → WIRED

**Steps:**

- [ ] **Step 1: Read the existing runner** to mirror its style.

Run: `sed -n '120,170p' scripts/run_tests.sh`

- [ ] **Step 2: Add a collection block** for the docling-graph tests next to the existing `pytest tests/unit` block, following the same pattern (same venv/env, same failure handling). Use this invocation (these tests need the repo clone on path; conftest.py there handles app/ shimming):

```bash
PYTHONPATH="docker/docling-graph/repo:${PYTHONPATH:-}" pytest docker/docling-graph/tests -q
```
  Wire its exit status into the script's pass/fail aggregation exactly like the other `pytest` blocks (do not let it run unguarded — match the surrounding `if`/exit-code handling so a failure here fails the suite).

- [ ] **Step 3: Verify it collects + passes.**

Run: `PYTHONPATH="docker/docling-graph/repo:${PYTHONPATH:-}" pytest docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py -q`
Expected: 1 passed.

- [ ] **Step 4: Commit.**

```bash
git add scripts/run_tests.sh
git commit -m "test(harness): collect docker/docling-graph/tests in the standard runner

The docling-graph patch/library tests (incl. the new last_chunk_metadata
regression) lived outside testpaths=[tests] and were never run by
run_tests.sh -> silent regression risk. Wire the dir into the suite.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Part B — resolve synthesizer self_refs to precise chunks in `derive_structure_links` (TDD)

**Goal:** A mention whose `element_uid` is a Docling self_ref (`#/texts/N`) resolves to its SPECIFIC chunk(s) via the persisted `identity_map`, instead of fanning out to all text chunks — so committed `EXTRACTED_FROM` edges are per-entity precise.

**Files:**
- Modify: `app/workers/pipeline.py` (`derive_structure_links`, the `#/`-fallback block at ~9310-9334; build/load the resolver near the `element_uid_chunk_map` construction ~9253-9270)
- Create: `tests/unit/test_extracted_from_self_ref_resolution.py`

**Acceptance Criteria:**
- [ ] In `derive_structure_links`, before the `euid.startswith("#/")` all-chunks fan-out, a `#/...` self_ref is translated to a concrete `element_uid` via the document's persisted `identity_map` (`docling_document.json._enrichments.identity_map`, written at pipeline.py:4836-4841), then resolved through the existing `element_uid_chunk_map` to specific chunk(s).
- [ ] The all-chunks fan-out (`resolved_chunks = all_text_chunk_ids`) becomes a LAST resort only — used when neither `element_uid_chunk_map` nor the identity_map resolves the self_ref. When it fires, log a WARNING naming the unresolved self_ref (no silent coarsening).
- [ ] A unit test proves: given an `identity_map` mapping `#/texts/5 → <element_uid X>` and an `element_uid_chunk_map` mapping `X → [chunkA]`, a mention with `element_uid="#/texts/5"` produces an edge to `chunkA` ONLY (not all chunks). And a self_ref absent from identity_map still falls back to all-chunks (with the warning).
- [ ] No regression: a mention whose `element_uid` is already a concrete `{page}-{order}-...` value still resolves through `element_uid_chunk_map` unchanged.

**Verify:** `python3 -m pytest tests/unit/test_extracted_from_self_ref_resolution.py -v` → all pass

**Steps:**

- [ ] **Step 1: Confirm the identity_map shape + load path.** Read how it's written and where the document_json is fetched in the worker:

```bash
sed -n '4832,4848p' app/workers/pipeline.py          # identity_map written into _enrichments
grep -n "_enrichments\|identity_map\|docling_document.json\|download_bytes\|minio" app/workers/pipeline.py | grep -iE "enrichment|identity_map|docling_document.json" | head
```
  Determine the in-worker way to obtain `identity_map` for `document_id` (load `docling_document.json` from MinIO derived bucket and read `_enrichments.identity_map`, OR an existing helper). Use whatever the codebase already does to fetch derived docling JSON; do not invent a new storage path.

- [ ] **Step 2: Write the failing unit test.** Create `tests/unit/test_extracted_from_self_ref_resolution.py`. Test the pure resolution helper (extracted in Step 3) so it needs no DB/MinIO:

```python
"""Part B: a synthesizer self_ref element_uid must resolve to its SPECIFIC
chunk via identity_map, not fan out to all chunks (coarse lineage)."""
from app.workers.pipeline import _resolve_mention_chunks


def test_self_ref_resolves_to_specific_chunk_via_identity_map():
    element_uid_chunk_map = {"p1-2-text-abcd": ["chunkA"], "p1-3-text-ef01": ["chunkB"]}
    identity_map = {"#/texts/5": "p1-2-text-abcd"}
    all_text_chunk_ids = ["chunkA", "chunkB", "chunkC"]
    resolved, coarse = _resolve_mention_chunks(
        "#/texts/5", element_uid_chunk_map, identity_map, all_text_chunk_ids,
    )
    assert resolved == ["chunkA"], f"expected precise [chunkA], got {resolved!r}"
    assert coarse is False


def test_unmapped_self_ref_falls_back_to_all_chunks_flagged_coarse():
    resolved, coarse = _resolve_mention_chunks(
        "#/texts/99", {"x": ["chunkA"]}, {"#/texts/5": "x"}, ["chunkA", "chunkB"],
    )
    assert resolved == ["chunkA", "chunkB"], "unmapped self_ref should fan out"
    assert coarse is True, "fan-out must be flagged so the caller can WARN"


def test_concrete_element_uid_resolves_directly_no_regression():
    resolved, coarse = _resolve_mention_chunks(
        "p1-2-text-abcd", {"p1-2-text-abcd": ["chunkA"]}, {}, ["chunkA", "chunkB"],
    )
    assert resolved == ["chunkA"]
    assert coarse is False
```

- [ ] **Step 3: Run the test to verify it FAILS.**

Run: `python3 -m pytest tests/unit/test_extracted_from_self_ref_resolution.py -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_mention_chunks'` (helper not yet defined).

- [ ] **Step 4: Implement the helper + wire it in.** Add the pure helper near `derive_structure_links` in `app/workers/pipeline.py`:

```python
def _resolve_mention_chunks(
    element_uid: str,
    element_uid_chunk_map: dict[str, list[str]],
    identity_map: dict[str, str],
    all_text_chunk_ids: list[str],
) -> tuple[list[str], bool]:
    """Resolve a mention's element_uid to concrete chunk ids.

    Returns (chunk_ids, is_coarse). Resolution order:
      1. Direct hit in element_uid_chunk_map (concrete {page}-{order}-... uid).
      2. Docling self_ref ("#/..."): translate via identity_map (self_ref ->
         element_uid) then resolve through element_uid_chunk_map.
      3. Last resort: fan out across all text chunks (is_coarse=True) so the
         entity stays reachable from the document — caller logs a WARNING.
    """
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
  Then replace the inline fan-out block (~pipeline.py:9329-9334) so it loads `identity_map` once (Step 1's method) and calls the helper:

```python
            # identity_map: self_ref -> element_uid (persisted at ingest,
            # docling_document.json _enrichments.identity_map). Bridges the
            # synthesizer's "#/texts/N" namespace to DocumentElement.element_uid.
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
  Implement `_load_identity_map(document_id)` using the codebase's existing derived-docling-JSON fetch (from Step 1); return `{}` on any miss so behavior degrades to the flagged fan-out, never crashes.

- [ ] **Step 5: Run the test to verify it PASSES.**

Run: `python3 -m pytest tests/unit/test_extracted_from_self_ref_resolution.py -v`
Expected: all 3 pass.

- [ ] **Step 6: Commit.**

```bash
git add app/workers/pipeline.py tests/unit/test_extracted_from_self_ref_resolution.py
git commit -m "fix(lineage): resolve synthesizer self_refs to precise chunks (Part B)

derive_structure_links keyed EXTRACTED_FROM on DocumentElement.element_uid
({page}-{order}-{type}-{hash}) but the synthesizer emits Docling self_refs
(#/texts/N) -> every mention missed the map and fanned out to ALL chunks
(coarse lineage). Translate self_ref -> element_uid via the persisted
identity_map (_enrichments.identity_map) so each entity links to its actual
chunk(s); the all-chunks fan-out is now a flagged last resort that WARNs.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Verify (or explicitly scope out) relationship/edge lineage

**Goal:** Determine whether the Part A+B fix also restores lineage for relationship edges (`system_links` pass / bug #59), and either confirm it or document the boundary so Task 5's gate isn't misread as covering edges.

**Files:** (investigation; code change only if a trivial parallel gap is found — otherwise document)
- Possibly Modify: `app/workers/pipeline.py` (relationship-provenance resolution, ~1472-1521) — only if it shares the identical self_ref-namespace gap and the same `_resolve_mention_chunks` applies.

**Acceptance Criteria:**
- [ ] Documented finding (in memory `project_extracted_from_root_cause.md`): does relationship provenance flow through `last_chunk_metadata`/self_refs the same way, and will Part A+B populate edge lineage, or is it a separate path?
- [ ] If edges share the exact gap and the helper applies cleanly → fix + a unit test; otherwise → explicit scope-out note so Task 5 PASS is not read as proving edge lineage.

**Verify:** `grep -q "relationship" /home/josh/.claude/projects/-home-josh-development-EIP-MMDPP/memory/project_extracted_from_root_cause.md && echo DOCUMENTED` → DOCUMENTED

**Steps:**

- [ ] **Step 1: Trace relationship provenance** from the response through merge to commit:

```bash
sed -n '1460,1521p' app/workers/pipeline.py     # _import_graph_phase_domain_edges resolution
grep -n "relationship_provenance" app/services/extraction_merge.py | head
```
  Determine whether committed edges (e.g. VARIANT_OF/HAS_COMPONENT) derive their chunk lineage from the same self_ref/`last_chunk_metadata` source as entities.

- [ ] **Step 2: Decide + record.** If the same gap and helper applies → add the resolution + a unit test mirroring Task 3 and commit. If different/larger → write a precise scope-out paragraph in `project_extracted_from_root_cause.md` (what's covered, what's not, why) and reference bug #59. Either way, commit the doc/code.

---

## Task 5: Harden the verifier + end-to-end PRECISE-lineage gate (USER-ORDERED)

**Goal:** Make `verify_lineage_e2e.py` actually prove THIS run restored PRECISE lineage (not pre-satisfied by existing MENTIONED_IN / stale entities), then run it on a fresh SA-2 graph_only run on the fixed build and confirm precise entity→chunk→page lineage on the graph-query endpoint.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Modify: `scripts/verify_lineage_e2e.py`

**Acceptance Criteria:**
- [ ] Verifier change #1 — **run-attributed EXTRACTED_FROM delta:** accept a `--pre-extracted-from <N>` arg; PASS requires post `EXTRACTED_FROM` count > pre (pre=0 for SA-2). This is the hard discriminator (the old trace/edge checks pass today on 173 MENTIONED_IN with EXTRACTED_FROM=0 — they must no longer be the only gate).
- [ ] Verifier change #2 — **drop the false-FAIL entity-count delta:** replace the `post - pre entities > 0` check (idempotent upsert makes it 0 on re-run) with the EXTRACTED_FROM delta above as the commit-evidence signal; keep an absolute `entities ≥ merged_baseline` sanity check (no recall regression) but not as the lineage proof.
- [ ] Verifier change #3 — **precision assertion:** for committed entities with `EXTRACTED_FROM`, assert the median per-entity target-chunk count is well below the document chunk count (precise), not ≈all (coarse). Threshold: an entity links to ≤ a small fraction (e.g. < 25%) of the document's text chunks; report the distribution.
- [ ] Verifier change #4 — **trace must use EXTRACTED_FROM specifically** (not `MENTIONED_IN,EXTRACTED_FROM`) for the chunk+document+page trace, so the trace proves the new edge, and scope the "no chunk metadata available" warning count to the run window.
- [ ] On a fresh SA-2 `graph_only` run on the fixed build: `EXTRACTED_FROM` post > 0; 0 "no chunk metadata available" warnings in the run window; provenance rows with non-empty `element_uid` > 0; precision assertion passes; `/v1/graph/query` for `SNR-75` returns `sources[]` with non-null `document_id` + `page_number` AND the entity's `EXTRACTED_FROM` target count is small (precise).
- [ ] `python3 scripts/verify_lineage_e2e.py --run <run_id> --pre-extracted-from 0` → "ALL CHECKS PASS".

**Verify:** `python3 scripts/verify_lineage_e2e.py --run <run_id> --pre-extracted-from 0` → "ALL CHECKS PASS"

**Steps:**

- [ ] **Step 1: Read the current verifier** and identify the pre-satisfied checks:

Run: `sed -n '60,132p' scripts/verify_lineage_e2e.py`
Confirm: the trace check (line ~107-123) uses `out('MENTIONED_IN','EXTRACTED_FROM')` and the edge check (line ~95-99) is `any(>0)` — both pass today. The entity-delta check (line ~75-84) false-FAILs on idempotent upsert.

- [ ] **Step 2: Apply verifier changes #1–#4** per the acceptance criteria. Add `--pre-extracted-from` arg; compute post `EXTRACTED_FROM` count; make `post > pre` a hard PASS condition; switch the commit-evidence check off entity-delta; add the precision distribution (per-entity `out('EXTRACTED_FROM').size()` vs doc TextChunk count); change the trace query to `out('EXTRACTED_FROM')` only; scope the warning grep to the run window. Keep output format (`[PASS]/[FAIL]` lines + "ALL CHECKS PASS").

- [ ] **Step 3: Capture pre-snapshot** (EXTRACTED_FROM pre = 0; record committed entity baseline):

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
ADB() { curl -s -u root:eip_arcadedb_secret -X POST http://localhost:2480/api/v1/command/eip_knowledge_graph -H "Content-Type: application/json" -d "{\"language\":\"sql\",\"command\":\"$1\"}"; }
echo -n "EXTRACTED_FROM pre: "; ADB "SELECT count(*) AS c FROM EXTRACTED_FROM" | python3 -c "import sys,json;print(json.load(sys.stdin)['result'][0]['c'])"
```
Expected: 0.

- [ ] **Step 4: Trigger the SA-2 graph_only reingest** (Task 1+3 fix must already be deployed — see note below):

```bash
DOC=ddaa9e36-2854-47c3-bc94-ff38d531dafd
curl -s -X POST "http://localhost:8005/v1/documents/$DOC/reingest" -H "Content-Type: application/json" \
  -d '{"mode":"graph_only","ontology_bundle_key":"air_defense_v3_merged_v1"}' | python3 -m json.tool
```
Record `pipeline_run_id`. Multi-hour run; monitor in background (lives in Celery, session-independent).

- [ ] **Step 5: At terminal status, run the hardened verifier + endpoint + precision checks.**

```bash
RUN=<pipeline_run_id>
ADB() { curl -s -u root:eip_arcadedb_secret -X POST http://localhost:2480/api/v1/command/eip_knowledge_graph -H "Content-Type: application/json" -d "{\"language\":\"sql\",\"command\":\"$1\"}"; }
echo -n "EXTRACTED_FROM post: "; ADB "SELECT count(*) AS c FROM EXTRACTED_FROM" | python3 -c "import sys,json;print(json.load(sys.stdin)['result'][0]['c'])"
python3 scripts/verify_lineage_e2e.py --run "$RUN" --pre-extracted-from 0
# endpoint + precision
curl -s -X POST "http://localhost:8005/v1/graph/query" -H "Content-Type: application/json" \
  -d '{"query":"SNR-75","top_k":1,"hop_count":1}' | python3 -c "import sys,json;r=json.load(sys.stdin);print(json.dumps(r[0].get('sources'),indent=2) if r else 'NO RESULT')"
echo -n "doc TextChunk count: "; ADB "SELECT count(*) AS c FROM TextChunk WHERE document_id='ddaa9e36-2854-47c3-bc94-ff38d531dafd'" | python3 -c "import sys,json;print(json.load(sys.stdin)['result'][0]['c'])"
ADB "SELECT system_name, out('EXTRACTED_FROM').size() AS ef FROM RADAR_SYSTEM WHERE out('EXTRACTED_FROM').size() > 0 LIMIT 5" | python3 -m json.tool
```
Expected: EXTRACTED_FROM post > 0; verifier "ALL CHECKS PASS"; `sources[]` carries document_id+page; per-entity `ef` is a SMALL number (≪ doc chunk count) = precise.

- [ ] **Step 6: Record outcome in memory** (`project_extracted_from_root_cause.md`): fix verified; EXTRACTED_FROM pre→post; precision distribution (per-entity target counts); any residual coarse cases.

---

## Notes for the executor

- **Order:** Task 1 (Part A patch) and Task 3 (Part B worker code) are independent edits but BOTH must be deployed before Task 5's gate. Task 1 needs an image rebuild; Task 3 is worker `app/**` (bind-mounted) → effective on `docker restart eip-mmdpp-worker-1 eip-mmdpp-worker-graph-1` (Celery loads code at process start; restart BOTH — worker-1 is a catch-all that also consumes the `graph` queue). Before Task 5: rebuild docling-graph (`docker compose -p eip-mmdpp build docling-graph && docker compose -p eip-mmdpp up -d --force-recreate docling-graph`), restart both workers, confirm `StartedAt` advanced on docling-graph + both workers, verify the live container has `last_chunk_metadata = chunk_metadata` in stages.py, and confirm the gemma4 pool is idle (~0.24s probe).
- **Deploy semantics** (memory): docling-graph is COPY-based → rebuild; compose from this worktree needs `-p eip-mmdpp`; `docker compose restart` does NOT reload code into a running worker → use `docker restart <container>`.
- **Never** use `POST /v1/documents/{id}/cancel` — it hard-deletes the document and all derived data. To stop a run, revoke tasks + reset status; ask first.
- **Why this is two parts:** Part A alone makes EXTRACTED_FROM fire but COARSE (entity→all chunks) — verified via the self_ref↔DocumentElement.element_uid namespace mismatch (pipeline.py:9310-9334). Part B restores precision using the identity_map that already exists at ingest (pipeline.py:4836). The hard requirement is exact-chunk lineage, so both are required.
