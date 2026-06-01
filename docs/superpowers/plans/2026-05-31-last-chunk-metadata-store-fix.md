# EXTRACTED_FROM Lineage Fix — `last_chunk_metadata` store in patch 0002

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore entity→chunk lineage (`EXTRACTED_FROM` edges, non-empty `element_uid`+`page`) by making the docling-graph chunked-batches path publish its `chunk_metadata` to `doc_processor.last_chunk_metadata`, which the provenance builders read.

**Architecture:** The verified root cause (memory `project_extracted_from_root_cause.md`, task #26) is a single missing assignment in `docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch`. That patch's CHUNKED-BATCHES branch (the production path for DoclingDocument input) computes a correct `chunk_metadata` (chunk_id + self_refs + page_numbers from `extract_chunks_with_metadata`) but never stores it on `doc_processor.last_chunk_metadata`. The service's `app/main.py` reads `context.extractor.doc_processor.last_chunk_metadata` to build the `chunk_to_self_refs` / `chunk_to_page_numbers` maps that BOTH provenance builders depend on; with the store missing the maps are always empty → `element_uid=""`/`page=null` on every entity → the worker lineage gate rejects everything → `EXTRACTED_FROM` is never built. Sibling chunking paths (`strategy_ops.py:47`/`:73`, `many_to_one.py:589`) all perform this store immediately before `extract_from_chunk_batches`; patch 0002 omitted it. Fix = add the one store line to patch 0002, mirroring the siblings.

**Tech Stack:** Python, FastAPI (docling-graph service, port 8002), Celery worker, ArcadeDB graph, `patch(1)`-applied library patches baked into the docling-graph Docker image (COPY + `pip install -e`), pytest.

**Scope / non-goals:** This restores lineage on the production DoclingDocument chunked path. OUT of scope: page-less sources (the gate also requires `page is not None`; SA-2's 102/102 TextChunks carry pages so it passes — genuinely page-less chunks remain a separate policy decision); the raw-text/pre-built-chunk paths that set `page_numbers=[]`; the gemma4 truncation/contention behavior (truncation-independent — see task #26). Whether the restored `EXTRACTED_FROM` is per-entity-precise vs coarse is **measured** in Task 3, not assumed.

---

## File Structure

- `docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch` — **MODIFY**: add the `doc_processor.last_chunk_metadata = chunk_metadata` store into the CHUNKED-BATCHES branch.
- `docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py` — **CREATE**: host-runnable test that applies the patch stack to a temp repo copy, imports the patched `docling_graph.pipeline.stages`, drives `_extract_from_docling_document` through the CHUNKED-BATCHES branch with fakes, and asserts `doc_processor.last_chunk_metadata` is populated with self_refs + page_numbers. RED before the fix, GREEN after.
- `scripts/verify_lineage_e2e.py` — **REUSE** (exists): end-to-end verifier (entities committed delta, 0 gate rejections, EXTRACTED_FROM>0, field→chunk→page trace). Task 3 runs it plus a precision measurement and a `/graph/query` endpoint check.

---

## Task 1: Fix patch 0002 to store `last_chunk_metadata` (TDD)

**Goal:** The patched CHUNKED-BATCHES branch in `_extract_from_docling_document` stores its `chunk_metadata` on `doc_processor.last_chunk_metadata`, proven by a host-runnable test that exercises the patched artifact.

**Files:**
- Create: `docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py`
- Modify: `docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch`

**Acceptance Criteria:**
- [ ] A test applies all `docker/docling-graph/patches/*.patch` to a temp copy of `docker/docling-graph/repo`, imports the patched `docling_graph.pipeline.stages`, calls `ExtractionStage._extract_from_docling_document` with fakes that take the CHUNKED-BATCHES branch, and asserts `fake_doc_processor.last_chunk_metadata` equals the metadata returned by `extract_chunks_with_metadata` (non-empty, carries `self_refs` + `page_numbers`).
- [ ] The test FAILS before editing patch 0002 (store line absent) and PASSES after.
- [ ] The store line is added inside patch 0002's `if can_chunk_batch:` branch, immediately after the `extract_chunks_with_metadata(...)` call and before `extract_from_chunk_batches(...)`.
- [ ] `patch --dry-run` of patch 0002 against a clean repo copy applies with no fuzz/offset errors; all five patches (0001–0005) still apply cleanly in sequence.

**Verify:** `python3 -m pytest docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py -v` → 1 passed

**Steps:**

- [ ] **Step 1: Write the failing test.** Create `docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py`:

```python
"""Patch-0002 regression: the CHUNKED-BATCHES branch of
ExtractionStage._extract_from_docling_document MUST publish its chunk_metadata
to doc_processor.last_chunk_metadata, because app/main.py builds the
chunk_to_self_refs / chunk_to_page_numbers provenance maps from it. Missing the
store yields empty maps -> element_uid="" -> no EXTRACTED_FROM lineage.

The fix lives in a library PATCH applied at Docker build, so this test applies
the full patch stack to a temp copy of docker/docling-graph/repo and imports the
PATCHED module in a clean subprocess (PYTHONPATH=<tmp>), exercising the real
patched artifact rather than a mock of it.
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
    """Mirror the Dockerfile loop: apply every patch with -p1 under dst_repo."""
    for p in sorted(_PATCHES.glob("*.patch")):
        subprocess.run(
            ["patch", "-p1", "-i", str(p)],
            cwd=str(dst_repo), check=True, capture_output=True, text=True,
        )


# Subprocess body: import the PATCHED stages module and drive the chunked path.
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
        # truthy chunker so can_chunk_batch is True
        chunker = object()
        last_chunk_metadata = []  # the field under test; starts empty
        def extract_chunks_with_metadata(self, document):
            return (["chunk one text", "chunk two text"], CHUNK_META)

    class FakeBackend:
        def extract_from_chunk_batches(self, *, chunks, chunk_metadata, template, context):
            return {"ok": True}  # truthy model so the branch completes
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
        "doc_processor.last_chunk_metadata (app/main.py reads this to build "
        "chunk_to_self_refs/page maps; empty => no EXTRACTED_FROM lineage)"
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
    # copy the repo clone, then apply the patch stack to it
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
Expected: FAIL — the driver's `AssertionError: REGRESSION: chunked-batches path did not store chunk_metadata ...` surfaces in captured STDERR (the patched branch computes `chunk_metadata` but never assigns `last_chunk_metadata`, so it stays `[]`).

> If `import docling_graph.pipeline.stages` errors in the subprocess for a missing dependency (host venv lacks an optional dep), run the SAME test inside the running container, which has the full env:
> `docker cp docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py eip-mmdpp-docling-graph-1:/app/tests/ && docker exec eip-mmdpp-docling-graph-1 python3 -m pytest /app/tests/test_chunked_batches_stores_chunk_metadata.py -v`
> (Earlier check confirmed `docling_graph` imports via the repo path on the host, so the host run is expected to work; the container run is the fallback.)

- [ ] **Step 3: Add the store line to patch 0002.** Edit `docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch`. Locate the added CHUNKED-BATCHES block where it calls `extract_chunks_with_metadata` and insert the store immediately after the call returns, before the trace emit / `extract_from_chunk_batches`. The relevant added (`+`) lines change from:

```diff
+                chunks, chunk_metadata = doc_processor.extract_chunks_with_metadata(
+                    context.docling_document
+                )
+                if context.trace_data:
```

to:

```diff
+                chunks, chunk_metadata = doc_processor.extract_chunks_with_metadata(
+                    context.docling_document
+                )
+                # PATCH 2026-05-31: publish chunk_metadata so app/main.py can build
+                # chunk_to_self_refs / chunk_to_page_numbers (provenance maps).
+                # Mirrors strategy_ops.extract_delta_from_document / many_to_one.py:589.
+                # Without this the maps are empty -> element_uid="" -> no EXTRACTED_FROM.
+                try:
+                    doc_processor.last_chunk_metadata = chunk_metadata
+                except AttributeError:
+                    pass
+                if context.trace_data:
```

  These are added lines in a unified diff (each new line prefixed `+`), inserted into an existing hunk whose header is `@@ -756,33 +756,90 @@`. The new-side line count (`90`) must grow by the 9 added lines → `@@ -756,33 +756,99 @@`. The old-side (`-756,33`) is unchanged (no original lines removed).

  **Hand-editing the hunk header is error-prone. Prefer regenerating the patch instead** — it guarantees a correct header:

  **Each patch touches a distinct file (verified: 0001=orchestrator.py, 0002=stages.py, 0003=prompts.py, 0004=llm_backend.py, 0005=many_to_one.py), so 0002's baseline is the CLEAN repo** — no other patch modifies `stages.py`. Regenerate:

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
rm -rf /tmp/dg_base /tmp/dg_fixed
cp -a docker/docling-graph/repo /tmp/dg_base          # clean baseline (0002's reference)
cp -a docker/docling-graph/repo /tmp/dg_fixed
patch -p1 -d /tmp/dg_fixed -i "$(pwd)/docker/docling-graph/patches/0002-"*.patch >/dev/null  # apply CURRENT 0002
# now hand-edit ONLY /tmp/dg_fixed/docling_graph/pipeline/stages.py: insert the 9 lines below
# immediately AFTER the `chunks, chunk_metadata = doc_processor.extract_chunks_with_metadata(...)`
# call inside the `if can_chunk_batch:` branch, BEFORE the following `if context.trace_data:`.
```
  The 9 lines to insert (plain Python, correct indentation = 16 spaces, no `+` prefix — editing source, not a diff):
```python
                # PATCH 2026-05-31: publish chunk_metadata so app/main.py can build
                # chunk_to_self_refs / chunk_to_page_numbers (provenance maps).
                # Mirrors strategy_ops.extract_delta_from_document / many_to_one.py:589.
                # Without this the maps are empty -> element_uid="" -> no EXTRACTED_FROM.
                try:
                    doc_processor.last_chunk_metadata = chunk_metadata
                except AttributeError:
                    pass
```
  Then regenerate 0002 as the clean→fixed diff of stages.py, restoring the `a/`…`b/` prefixes that `-p1` expects:
```bash
( cd /tmp && diff -u dg_base/docling_graph/pipeline/stages.py dg_fixed/docling_graph/pipeline/stages.py \
    | sed -e 's#^--- dg_base/#--- a/#' -e 's#^+++ dg_fixed/#+++ b/#' ) > /tmp/0002.new
head -8 /tmp/0002.new   # sanity: `--- a/docling_graph/pipeline/stages.py`, `+++ b/...`, then `@@` hunks
cp /tmp/0002.new docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch
rm -rf /tmp/dg_base /tmp/dg_fixed /tmp/0002.new
```
  Step 5's dry-run is the backstop that the regenerated patch applies cleanly in sequence.

- [ ] **Step 4: Run the test to verify it PASSES.**

Run: `python3 -m pytest docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py -v`
Expected: PASS (driver prints `DRIVER_OK`; `last_chunk_metadata` now equals `CHUNK_META`).

- [ ] **Step 5: Verify the whole patch stack still applies cleanly.**

Run:
```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
rm -rf /tmp/dgpatchcheck && cp -a docker/docling-graph/repo /tmp/dgpatchcheck
for p in docker/docling-graph/patches/*.patch; do echo "== $p =="; patch -p1 -d /tmp/dgpatchcheck --dry-run -i "$(pwd)/$p" || echo "FAILED: $p"; done
```
Expected: every patch prints `checking file ...` with no "FAILED", no "Hunk #N FAILED", no fuzz/offset errors. Then `rm -rf /tmp/dgpatchcheck`.

- [ ] **Step 6: Commit.**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
git add docker/docling-graph/patches/0002-stages-chunked-batches-for-docling-document-input.patch docker/docling-graph/tests/test_chunked_batches_stores_chunk_metadata.py
git commit -m "fix(extraction): patch 0002 chunked-batches path must store last_chunk_metadata

Root cause of EXTRACTED_FROM=0 (0/2711 non-empty element_uid, every run):
the DoclingDocument chunked-batches branch computed chunk_metadata with
self_refs+pages but never published it to doc_processor.last_chunk_metadata,
so app/main.py built empty chunk_to_self_refs/page maps -> provenance
builders emitted element_uid=''/page=null -> lineage gate rejected all ->
no EXTRACTED_FROM edges. Mirror the store the sibling chunking paths already do.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Rebuild docling-graph image and redeploy (no stale consumers)

**Goal:** The running `docling-graph` container has the corrected patch 0002 applied, and both graph-queue workers run current code, so the next extraction exercises the fix.

**Files:** (no source files — deploy task)
- Touches: docling-graph image (rebuild), containers `eip-mmdpp-docling-graph-1`, `eip-mmdpp-worker-1`, `eip-mmdpp-worker-graph-1`.

**Acceptance Criteria:**
- [ ] `docker compose -p eip-mmdpp build docling-graph` logs `Applying patch: /app/patches/0002-...patch` and finishes without a patch failure.
- [ ] The running container's `/app/repo/docling_graph/pipeline/stages.py` contains `last_chunk_metadata = chunk_metadata` within the chunked-batches branch.
- [ ] `eip-mmdpp-docling-graph-1`, `eip-mmdpp-worker-1`, and `eip-mmdpp-worker-graph-1` all show a `StartedAt` newer than the rebuild (no stale graph-queue consumer — see memory `project_catchall_worker_stale_code_trap`).
- [ ] gemma4 pool idle probe ≈ 0.24s per host (no contention before the verification run).

**Verify:** `docker exec eip-mmdpp-docling-graph-1 grep -c "last_chunk_metadata = chunk_metadata" /app/repo/docling_graph/pipeline/stages.py` → ≥ 1

**Steps:**

- [ ] **Step 1: Rebuild the docling-graph image** (COPY-based service — code change requires rebuild):

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
docker compose -p eip-mmdpp build docling-graph 2>&1 | grep -E "Applying patch|Built|ERROR|FAILED"
```
Expected: lists `Applying patch:` for 0001–0005 (incl. 0002) and a successful build; no FAILED.

- [ ] **Step 2: Recreate docling-graph + restart BOTH graph-queue workers** (worker-1 is a catch-all that also consumes `graph`; restarting only worker-graph leaves a stale consumer):

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
docker compose -p eip-mmdpp up -d --force-recreate docling-graph
docker restart eip-mmdpp-worker-1 eip-mmdpp-worker-graph-1
```

- [ ] **Step 3: Verify the fix is live in the container and workers are fresh.**

```bash
docker exec eip-mmdpp-docling-graph-1 grep -n "last_chunk_metadata = chunk_metadata" /app/repo/docling_graph/pipeline/stages.py
for c in docling-graph-1 worker-1 worker-graph-1; do docker inspect eip-mmdpp-$c --format "$c StartedAt: {{.State.StartedAt}}"; done
```
Expected: the grep prints the matching line (within the chunked-batches branch); all three `StartedAt` are newer than the build.

- [ ] **Step 4: Confirm gemma4 pool is idle** (clean signal for Task 3):

```bash
for h in 10.0.1.121 10.0.1.109; do printf "%s " "$h"; curl -s --max-time 30 -o /dev/null -w '%{time_total}s\n' http://$h:11434/api/generate -d '{"model":"gemma4:31b","prompt":"hi","stream":false,"options":{"num_predict":1}}'; done
```
Expected: ~0.2–0.4s per host (idle). If multi-second, wait for the pool to drain before Task 3.

---

## Task 3: End-to-end lineage verification gate (USER-ORDERED)

**Goal:** Prove on a fresh SA-2 `graph_only` run on the fixed build that lineage is restored end-to-end: non-empty `element_uid`+`page` in pass outputs, entities committed, zero lineage-gate rejections, `EXTRACTED_FROM > 0`, and a field value traces entity→chunk→page on the graph-query endpoint; and MEASURE whether the lineage is per-entity-precise or coarse.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Reuse: `scripts/verify_lineage_e2e.py` (exists)

**Acceptance Criteria:**
- [ ] Capture `SNAPSHOT_PRE` = committed entity count for SA-2's document before the run (baseline: 22; EXTRACTED_FROM pre = 0).
- [ ] Trigger SA-2 `graph_only` reingest (`POST /v1/documents/ddaa9e36-2854-47c3-bc94-ff38d531dafd/reingest` with `{"mode":"graph_only","ontology_bundle_key":"air_defense_v3_merged_v1"}`) and let it reach a terminal status.
- [ ] After the run: across the run's pass outputs, the count of provenance rows with non-empty `element_uid` is > 0 (pre-fix this was 0/141); docling-graph emits 0 "no chunk metadata available" warnings in the run window.
- [ ] `EXTRACTED_FROM` edge count (post) > 0 (pre = 0).
- [ ] `scripts/verify_lineage_e2e.py --run <run_id> --pre <SNAPSHOT_PRE>` reports PASS for: entities committed (delta>0), zero provenance-drop warnings, zero LINEAGE_GATE rejections, entity→chunk lineage edges present, and a field value traced to chunk+document+page.
- [ ] `/v1/graph/query` for `SNR-75` returns `sources[]` with a non-null `document_id` and `page_number` (proves the consumer endpoint surfaces the restored lineage).
- [ ] PRECISION MEASUREMENT (reported, not pass/fail): for ≥3 committed entities, compare the entity's `EXTRACTED_FROM` target chunk(s) against the chunk count of the document; record whether each entity links to a small chunk set (precise) or to all/most chunks (coarse fan-out). Note the result in the close comment; if coarse, file a follow-up rather than blocking.

**Verify:** `python3 scripts/verify_lineage_e2e.py --run <run_id> --pre <SNAPSHOT_PRE>` → "ALL CHECKS PASS"

**Steps:**

- [ ] **Step 1: Capture pre-snapshot.**

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
ADB() { curl -s -u root:eip_arcadedb_secret -X POST http://localhost:2480/api/v1/command/eip_knowledge_graph -H "Content-Type: application/json" -d "{\"language\":\"sql\",\"command\":\"$1\"}"; }
echo -n "EXTRACTED_FROM pre: "; ADB "SELECT count(*) AS c FROM EXTRACTED_FROM" | python3 -c "import sys,json;print(json.load(sys.stdin)['result'][0]['c'])"
python3 scripts/diagnose_lineage_commit.py --snapshot pre 2>&1 | grep -i SNAPSHOT_PRE
```
Record the SNAPSHOT_PRE entity count (expected 22) and EXTRACTED_FROM pre (expected 0).

- [ ] **Step 2: Trigger the graph_only reingest.**

```bash
DOC=ddaa9e36-2854-47c3-bc94-ff38d531dafd
curl -s -X POST "http://localhost:8005/v1/documents/$DOC/reingest" -H "Content-Type: application/json" \
  -d '{"mode":"graph_only","ontology_bundle_key":"air_defense_v3_merged_v1"}' | python3 -m json.tool
```
Record the returned `pipeline_run_id`. (SA-2 is a multi-hour run; monitor in the background — do not block the session. Restart-safe: the run lives in Celery.)

- [ ] **Step 3: Wait for terminal status, then check the fresh chunk-map signal.**

```bash
RUN=<pipeline_run_id>; DOC=ddaa9e36-2854-47c3-bc94-ff38d531dafd
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -c "SELECT status FROM ingest.pipeline_runs WHERE id='$RUN';"
echo -n "no-chunk-metadata warnings this run: "; docker logs eip-mmdpp-docling-graph-1 --since 6h 2>&1 | grep -c "no chunk metadata available"
docker exec eip-mmdpp-postgres-1 psql -U eip -d eip -t -A -F'|' -c "
  SELECT pass_name,
    (SELECT count(*) FROM jsonb_array_elements(extract_pass_response_json->'provenance') p WHERE COALESCE(p->>'element_uid','')<>'') AS euid_set
  FROM ingest.pipeline_pass_outputs WHERE pipeline_run_id='$RUN' ORDER BY pass_name;"
```
Expected: status terminal (COMPLETE); 0 warnings; `euid_set` > 0 for the chunk-bearing passes.

- [ ] **Step 4: Run the e2e verifier + EXTRACTED_FROM count.**

```bash
RUN=<pipeline_run_id>; PRE=<SNAPSHOT_PRE>
ADB() { curl -s -u root:eip_arcadedb_secret -X POST http://localhost:2480/api/v1/command/eip_knowledge_graph -H "Content-Type: application/json" -d "{\"language\":\"sql\",\"command\":\"$1\"}"; }
echo -n "EXTRACTED_FROM post: "; ADB "SELECT count(*) AS c FROM EXTRACTED_FROM" | python3 -c "import sys,json;print(json.load(sys.stdin)['result'][0]['c'])"
python3 scripts/verify_lineage_e2e.py --run "$RUN" --pre "$PRE"
```
Expected: EXTRACTED_FROM post > 0; verifier prints "ALL CHECKS PASS".

- [ ] **Step 5: Confirm the consumer endpoint surfaces lineage + measure precision.**

```bash
curl -s -X POST "http://localhost:8005/v1/graph/query" -H "Content-Type: application/json" \
  -d '{"query":"SNR-75","top_k":1,"hop_count":1}' | python3 -c "import sys,json; r=json.load(sys.stdin); print(json.dumps(r[0].get('sources'), indent=2) if r else 'NO RESULT')"
# Precision: chunks in the doc vs distinct EXTRACTED_FROM targets per entity
ADB() { curl -s -u root:eip_arcadedb_secret -X POST http://localhost:2480/api/v1/command/eip_knowledge_graph -H "Content-Type: application/json" -d "{\"language\":\"sql\",\"command\":\"$1\"}"; }
echo -n "doc TextChunk count: "; ADB "SELECT count(*) AS c FROM TextChunk WHERE document_id='ddaa9e36-2854-47c3-bc94-ff38d531dafd'" | python3 -c "import sys,json;print(json.load(sys.stdin)['result'][0]['c'])"
ADB "SELECT system_name, out('EXTRACTED_FROM').size() AS ef_targets FROM RADAR_SYSTEM WHERE out('EXTRACTED_FROM').size() > 0 LIMIT 5" | python3 -m json.tool
```
Expected: `sources[]` carries a non-null `document_id` + `page_number`. Record per-entity `ef_targets` vs doc chunk count — small set = precise; ≈all chunks = coarse. Put the verdict in the close comment.

- [ ] **Step 6: Record outcome in memory** (`project_extracted_from_root_cause.md`): mark the fix verified, note EXTRACTED_FROM pre→post, and the precision finding (+ follow-up if coarse).

---

## Notes for the executor

- **Deploy semantics** (memory): docling-graph is COPY-based → rebuild required; worker `app/**` is bind-mounted but Celery loads modules at process start → restart needed; compose from this worktree needs `-p eip-mmdpp`; `docker compose restart` does NOT re-read code into a running worker → use `docker restart <container>`.
- **Never** use `POST /v1/documents/{id}/cancel` to stop a run — it hard-deletes the document and all derived data. To stop a run, revoke tasks + reset status; ask first.
- **Pool contention** can still cause truncation on concurrent passes; truncation is now lineage-safe ONLY if the chunked path stored metadata (this fix). If a pass still salvages to raw-text (no DoclingDocument), that path sets `page_numbers=[]` and remains out of scope here.
