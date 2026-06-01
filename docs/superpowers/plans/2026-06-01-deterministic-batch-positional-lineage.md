# Deterministic Batch-Positional Lineage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make entity, field, and relationship lineage PRECISE by stamping each item with the chunk-set of the batch it was extracted from (positional, deterministic, model-independent) and resolving those self_refs to concrete chunks in the worker — never fanning out to all document chunks.

**Architecture:** The delta IR normalizer already computes each batch's positional chunk lineage (`chunk_indexes`/`self_refs`/`page_numbers` from `batch_plan` + `chunk_metadata`, ir_normalizer.py:586-613) but lets the LLM's cited `evidence_ids` overwrite it. We invert the priority: the batch's positional self_refs are AUTHORITATIVE; LLM citations survive only as a `cited_refs` diagnostic. We carry the self_ref/chunk_index LISTS end-to-end (service `ExtractionProvenance` → worker `ExtractionProvenance` → `mentions[]` → `_resolve_mention_chunks`) instead of collapsing to one scalar, give fields a resolved `chunk_id` (was hardcoded `None`), and write relationship source-chunk refs as edge properties. The worker resolves each self_ref to its chunk and, on miss, attributes to the batch's chunk-set (never all-document).

**Tech Stack:** Python, FastAPI (docling-graph, port 8002, COPY image + `patch`-applied library patches, loop hardened `--fuzz=0 || exit 1`), Celery worker (`app/**` bind-mounted, loads code at process start), ArcadeDB, pytest. Authoritative contract: `docs/superpowers/specs/2026-06-01-deterministic-batch-positional-lineage-design.md` §2.5.

**Precision guarantee:** exact single chunk when batch=1 chunk (the configured norm, `batch_token_size==chunk_max_tokens==512`); a correct small K-chunk set when a batch spans multiple chunks; always real page(s). Single-chunk-everywhere narrowing is an explicit non-goal (measured follow-on).

---

## File Structure

- `docker/docling-graph/repo/.../delta/ir_normalizer.py` — **library patch** (new `0006-positional-node-rel-provenance.patch`): stamp authoritative `self_refs`/`chunk_indexes`/`page_numbers` + `cited_refs` diagnostic onto node.provenance, property_evidence, rel.provenance. (Task 1)
- `docker/docling-graph/app/schemas.py` — `ExtractionProvenance`/`ExtractionFieldProvenance` gain `self_refs`/`chunk_indexes`/`cited_refs`. (Task 2)
- `docker/docling-graph/app/provenance.py` — `_resolve_element_uid` demoted (no longer prefers single cited self_ref); new `_resolve_self_refs(node) -> list[str]`; `build_*` emit the lists; `build_auto_field_evidence` sourced positionally. (Task 2)
- `app/services/extraction_merge.py` — worker `ExtractionProvenance` + `FieldEvidenceRow` gain `self_refs`/`chunk_indexes` lists. (Task 3)
- `app/workers/pipeline.py` — `_parse_pass_response` populate lists; `mentions[]` forward lists; `_resolve_mention_chunks` batch-set param + fail-to-batch-set (both fan-out sites); field `chunk_id` resolution; capture rel-upsert RIDs + write edge source-chunk props. (Tasks 3, 4, 5)
- `app/services/arcadedb_graph.py` + `app/services/arcadedb_schema.py` — register + write relationship `source_chunk_ids`/`source_pages`/`source_self_refs` edge properties. (Task 5)
- `scripts/verify_lineage_e2e.py` — all-three-targets + per-target-recall gate. (Task 6)
- Tests under `docker/docling-graph/tests/` + `tests/unit/`. Wire new docling-graph tests into `scripts/run_tests.sh` (Task 1/2).

---

## Task 1: Positional stamp in the delta normalizer (library patch 0006, TDD)

**Goal:** Each normalized delta node, its `property_evidence`, and each relationship carry the batch's authoritative positional `self_refs`/`chunk_indexes`/`page_numbers`, with LLM citations preserved separately as `cited_refs`.

**Files:**
- Create: `docker/docling-graph/patches/0006-positional-node-rel-provenance.patch`
- Create: `docker/docling-graph/tests/test_positional_provenance_stamp.py`
- Modify: `scripts/run_tests.sh` (add the new test file to the explicit docling-graph collection block)

**Acceptance Criteria:**
- [ ] Every normalized node's `provenance` carries `self_refs` (= the batch's `batch_self_refs`), `chunk_indexes` (= the batch's `chunk_indexes`), `page_numbers` (= the batch's), AND `cited_refs` (= the LLM's `valid` evidence_ids for that node, possibly empty). `evidence_ids` is NO LONGER overwritten by the batch pool — it equals `cited_refs` (explicit only).
- [ ] `node.provenance.property_evidence[field]` entries are preserved AND each field's anchor resolves to the batch chunk-set (the per-field `cited_refs` stay as-is; positional `self_refs`/`chunk_indexes` available at node level for the worker to attach per-field).
- [ ] Each normalized relationship's `provenance` carries the same `self_refs`/`chunk_indexes`/`page_numbers` + `cited_refs`.
- [ ] A node/rel that the LLM did NOT cite still gets the full batch `self_refs`/`chunk_indexes` (positional), with `cited_refs=[]` — NOT the old "whole batch pool as evidence_ids" coarsening.
- [ ] Test applies the full patch stack (0001–0006) to a temp repo copy, drives `normalize_delta_ir_batch_results` with a 2-chunk batch where one node cites `#/texts/3` and another cites nothing, asserts: cited node `cited_refs=["#/texts/3"]` + `self_refs` has both batch refs; uncited node `cited_refs=[]` + `self_refs` has both batch refs. FAILS before patch, PASSES after.
- [ ] All six patches `patch --fuzz=0 --dry-run` apply in sequence, no fuzz/FAILED.

**Verify:** `python3 -m pytest docker/docling-graph/tests/test_positional_provenance_stamp.py -v` → passed

**Steps:**

- [ ] **Step 1: Write the failing test** — create `docker/docling-graph/tests/test_positional_provenance_stamp.py`:

```python
"""Positional batch lineage: normalize_delta_ir_batch_results must stamp the
batch's self_refs/chunk_indexes/page_numbers on EVERY node + relationship
(authoritative, model-independent), and preserve LLM citations only as
cited_refs. Applies the patch stack to a temp repo and imports the patched lib.
"""
import subprocess, sys, textwrap
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SERVICE_ROOT = _HERE.parent
_REPO = _SERVICE_ROOT / "repo"
_PATCHES = _SERVICE_ROOT / "patches"


def _apply_patches(dst: Path) -> None:
    for p in sorted(_PATCHES.glob("*.patch")):
        subprocess.run(["patch", "-p1", "--fuzz=0", "-i", str(p)],
                       cwd=str(dst), check=True, capture_output=True, text=True)


_DRIVER = textwrap.dedent('''
    import sys
    sys.path.insert(0, sys.argv[1])
    from docling_graph.core.extractors.contracts.delta.ir_normalizer import (
        normalize_delta_ir_batch_results,
    )
    # Minimal catalog/config doubles: the function needs allowed_paths +
    # config.attach_provenance True. Build the smallest real inputs.
    from docling_graph.core.extractors.contracts.delta.ir_normalizer import (
        IRNormalizerConfig,
    )
    # One batch, two chunks (indices 0,1), each with a self_ref+page.
    chunk_metadata = [
        {"chunk_id": 0, "self_refs": ["#/texts/3"], "page_numbers": [1],
         "evidence_ids": ["#/texts/3"], "evidence_units": [], "token_count": 5},
        {"chunk_id": 1, "self_refs": ["#/texts/4"], "page_numbers": [2],
         "evidence_ids": ["#/texts/4"], "evidence_units": [], "token_count": 5},
    ]
    batch_plan = [[(0, "c0", 5), (1, "c1", 5)]]   # one batch spanning chunks 0,1
    # batch_results: one graph dict; node A cites #/texts/3, node B cites nothing.
    batch_results = [{
        "nodes": [
            {"path": "root", "node_type": "Root", "ids": {}, "properties": {}},
            {"path": "root.radar", "node_type": "RADAR_SYSTEM", "ids": {"system_name": "A"},
             "properties": {"system_name": "A"}, "evidence_ids": ["#/texts/3"]},
            {"path": "root.radar", "node_type": "RADAR_SYSTEM", "ids": {"system_name": "B"},
             "properties": {"system_name": "B"}},  # no evidence_ids => uncited
        ],
        "relationships": [],
    }]
    # NOTE: exact catalog/config construction is determined when writing the test
    # against the real normalize_delta_ir_batch_results signature (see Step 1b).
    print("DRIVER_SHELL_OK")
''')


def test_positional_stamp(tmp_path):
    dst = tmp_path / "repo"
    subprocess.run(["cp", "-a", str(_REPO), str(dst)], check=True)
    _apply_patches(dst)
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)
    proc = subprocess.run([sys.executable, str(driver), str(dst)],
                          capture_output=True, text=True)
    assert "DRIVER_SHELL_OK" in proc.stdout, proc.stdout + "\n" + proc.stderr
```

- [ ] **Step 1b: Resolve the real `normalize_delta_ir_batch_results` signature** before finishing the test. Run:

```bash
sed -n '551,615p' docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/ir_normalizer.py
grep -n "def normalize_delta_ir_batch_results\|catalog\|dedup_policy\|IRNormalizerConfig\|allowed_paths" docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/ir_normalizer.py | head
```
Build the minimal real `catalog`/`dedup_policy`/`config` doubles the function requires (the function takes `batch_results, batch_plan, chunk_metadata, catalog, dedup_policy, config`). Replace the `DRIVER_SHELL_OK` placeholder with the real call + assertions:

```python
    out = normalize_delta_ir_batch_results(
        batch_results=batch_results, batch_plan=batch_plan,
        chunk_metadata=chunk_metadata, catalog=<built>, dedup_policy=<built>,
        config=<built with attach_provenance=True, validate_paths=False>,
    )
    nodes = out[0]["nodes"]  # normalize returns (normalized_results, stats)? confirm shape
    by_name = {n["ids"].get("system_name"): n for n in nodes if n.get("ids")}
    a, b = by_name["A"], by_name["B"]
    assert set(a["provenance"]["self_refs"]) == {"#/texts/3", "#/texts/4"}, a["provenance"]
    assert set(a["provenance"]["chunk_indexes"]) == {0, 1}
    assert a["provenance"]["cited_refs"] == ["#/texts/3"]
    assert a["provenance"]["evidence_ids"] == ["#/texts/3"]   # explicit only, NOT batch pool
    assert set(b["provenance"]["self_refs"]) == {"#/texts/3", "#/texts/4"}  # positional even uncited
    assert b["provenance"]["cited_refs"] == []
    assert b["provenance"]["evidence_ids"] == []              # NOT the whole batch pool
    print("DRIVER_OK")
```
(Confirm the return shape — `normalize_delta_ir_batch_results` returns `(normalized_results, stats)`; index accordingly. Assert on `DRIVER_OK`.)

- [ ] **Step 2: Run the test — verify it FAILS** (current code sets `evidence_ids = valid or batch_pool` and has no `self_refs`/`cited_refs` on node provenance):

Run: `python3 -m pytest docker/docling-graph/tests/test_positional_provenance_stamp.py -v`
Expected: FAIL (uncited node B's `evidence_ids` == whole batch pool, no `cited_refs` key, no node-level `self_refs`).

- [ ] **Step 3: Regenerate patch 0006 from a clean-repo baseline.** Each existing patch touches a distinct file; 0006 also touches `ir_normalizer.py` — which NO other patch touches (verified: 0001=orchestrator, 0002=stages, 0003=prompts, 0004=llm_backend, 0005=many_to_one). So 0006's baseline is the clean repo:

```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
rm -rf /tmp/dg_base /tmp/dg_fixed
cp -a docker/docling-graph/repo /tmp/dg_base
cp -a docker/docling-graph/repo /tmp/dg_fixed
# hand-edit /tmp/dg_fixed/.../delta/ir_normalizer.py per Step 3a/3b below
```

- [ ] **Step 3a: Change `_attach_evidence_to_prov` to stop the batch-pool overwrite + record cited separately.** In `/tmp/dg_fixed/.../ir_normalizer.py:52-59`, replace:

```python
    out = dict(base_provenance)
    raw = raw_obj.get("evidence_ids") or []
    valid, invalid = _partition_evidence(raw, batch_evidence_ids)
    out["evidence_ids"] = valid or list(batch_evidence_ids)
    if invalid:
        stats["invalid_evidence_ids"] = stats.get("invalid_evidence_ids", 0) + len(invalid)
    return out
```
with (evidence_ids = EXPLICIT only; cited_refs mirrors it; positional self_refs/chunk_indexes/page_numbers come from the `base_provenance` the caller passes = the batch `provenance` dict):

```python
    out = dict(base_provenance)
    raw = raw_obj.get("evidence_ids") or []
    valid, invalid = _partition_evidence(raw, batch_evidence_ids)
    # AUTHORITATIVE lineage is positional (self_refs/chunk_indexes/page_numbers
    # already in base_provenance = the batch's chunk-set). The LLM's cited refs
    # are EXPLICIT-only and kept as a diagnostic; they never replace the
    # positional batch lineage and never expand to the whole batch pool.
    out["evidence_ids"] = list(valid)        # explicit only (was: valid or batch_pool)
    out["cited_refs"] = list(valid)          # diagnostic alias
    if invalid:
        stats["invalid_evidence_ids"] = stats.get("invalid_evidence_ids", 0) + len(invalid)
    return out
```
(`base_provenance` is the per-batch `provenance` dict built at ir_normalizer.py:607-613, which already carries `self_refs`/`chunk_indexes`/`page_numbers` — so `out` already has them via `dict(base_provenance)`. The only bug was `evidence_ids` clobbering. Confirm `_attach_evidence_to_prov` is called with that batch `provenance` as `base_provenance` for BOTH nodes (line 794) and rels (line 895) — it is.)

- [ ] **Step 3b: Regenerate the patch** restoring `a/`…`b/` prefixes:

```bash
( cd /tmp && diff -u dg_base/docling_graph/core/extractors/contracts/delta/ir_normalizer.py \
    dg_fixed/docling_graph/core/extractors/contracts/delta/ir_normalizer.py \
    | sed -e 's#^--- dg_base/#--- a/#' -e 's#^+++ dg_fixed/#+++ b/#' ) > /tmp/0006.new
head -6 /tmp/0006.new
cp /tmp/0006.new docker/docling-graph/patches/0006-positional-node-rel-provenance.patch
rm -rf /tmp/dg_base /tmp/dg_fixed /tmp/0006.new
```

- [ ] **Step 4: Run the test — verify it PASSES.**

Run: `python3 -m pytest docker/docling-graph/tests/test_positional_provenance_stamp.py -v`
Expected: PASS (DRIVER_OK).

- [ ] **Step 5: Verify the whole patch stack applies fuzz-clean + wire the test into run_tests.sh.**

```bash
rm -rf /tmp/dgpatchcheck && cp -a docker/docling-graph/repo /tmp/dgpatchcheck
for p in docker/docling-graph/patches/*.patch; do echo "== $p =="; patch -p1 -d /tmp/dgpatchcheck --fuzz=0 --dry-run -i "$(pwd)/$p" || echo "FAILED: $p"; done
rm -rf /tmp/dgpatchcheck
```
Expected: 0001-0006 all clean, no FAILED. Then add `docker/docling-graph/tests/test_positional_provenance_stamp.py` to the explicit docling-graph test list in `scripts/run_tests.sh` (next to `test_chunked_batches_stores_chunk_metadata.py`).

- [ ] **Step 6: Commit.**

```bash
git add docker/docling-graph/patches/0006-positional-node-rel-provenance.patch docker/docling-graph/tests/test_positional_provenance_stamp.py scripts/run_tests.sh
git commit -m "fix(lineage): positional batch self_refs authoritative; LLM cited_refs diagnostic (patch 0006)

normalize_delta_ir_batch_results stamps the batch's self_refs/chunk_indexes/
page_numbers on every node+rel provenance (already computed at :607-613, just
needed to stop evidence_ids='valid or batch_pool' clobbering it). LLM citations
kept only as cited_refs. Uncited nodes now carry positional batch lineage, not
the whole-batch evidence pool.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Service schema + provenance builders carry the lists (TDD)

**Goal:** The docling-graph `/extract-pass` response carries `self_refs`/`chunk_indexes`/`cited_refs` lists on entity provenance, field provenance is sourced positionally, and `_resolve_element_uid` no longer prefers the single cited self_ref.

**Files:**
- Modify: `docker/docling-graph/app/schemas.py` (`ExtractionProvenance` ~304, `ExtractionFieldProvenance`)
- Modify: `docker/docling-graph/app/provenance.py` (`_resolve_element_uid` ~205 demote; new `_resolve_self_refs`; `build_provenance_from_context` ~342 + `synthesize_provenance_from_pass_output` ~84 emit lists; `build_auto_field_evidence` ~481 positional)
- Create: `docker/docling-graph/tests/test_provenance_lists.py`
- Modify: `scripts/run_tests.sh`

**Acceptance Criteria:**
- [ ] `ExtractionProvenance` gains `self_refs: list[str] = []`, `chunk_indexes: list[int] = []`, `cited_refs: list[str] = []` (additive, defaulted). `element_uid` stays (= `self_refs[0]` when present, for back-compat).
- [ ] New `_resolve_self_refs(node_data, chunk_to_self_refs) -> list[str]` returns the node's positional batch self_refs (from `provenance.self_refs`, else `chunk_indexes`→`chunk_to_self_refs`, else `[]`).
- [ ] `_resolve_element_uid` is demoted: drop Strategy 3 (the "prefer numerically-smallest cited #/ self_ref" block); it now returns `self_refs[0]` for back-compat (direct → nested element_uid → `self_refs[0]` → chunk_indexes). The cited-evidence preference is GONE.
- [ ] `build_provenance_from_context` AND `synthesize_provenance_from_pass_output` populate `self_refs`/`chunk_indexes`/`page_numbers`/`cited_refs` on each emitted `ExtractionProvenance` (lists, not just scalar element_uid).
- [ ] `build_auto_field_evidence` (or its replacement) sources each field's anchor from the node's positional batch `self_refs`/`chunk_indexes`, NOT value-text-match over chunks. `ExtractionFieldProvenance` carries `self_refs`/`chunk_indexes` lists.
- [ ] Unit test: a node whose provenance has `self_refs=["#/texts/3","#/texts/4"]`, `cited_refs=["#/texts/3"]` → emitted `ExtractionProvenance.self_refs == ["#/texts/3","#/texts/4"]`, `.cited_refs==["#/texts/3"]`, `.element_uid=="#/texts/3"`; field provenance for a property carries the same positional self_refs.

**Verify:** `python3 -m pytest docker/docling-graph/tests/test_provenance_lists.py -v` → passed

**Steps:**

- [ ] **Step 1: Read the current builders + field-evidence path** to ground the edits:

```bash
sed -n '84,205p' docker/docling-graph/app/provenance.py    # synthesize + _resolve_element_uid
sed -n '342,540p' docker/docling-graph/app/provenance.py    # build_provenance_from_context + build_auto_field_evidence
grep -n "class ExtractionFieldProvenance\|class ExtractionProvenance" docker/docling-graph/app/schemas.py
```

- [ ] **Step 2: Write the failing test** — `docker/docling-graph/tests/test_provenance_lists.py`. Use the existing `dg_app_module`/`dg_schemas` conftest fixtures (they load docling-graph `app/` under a path swap). Build a fake `context.knowledge_graph` node (or call the builder with a delta-node dict) carrying `provenance={self_refs:[...], chunk_indexes:[...], cited_refs:[...], page_numbers:[...]}`, assert the emitted `ExtractionProvenance` carries all lists + `element_uid==self_refs[0]`. Assert `_resolve_self_refs` returns the full list and `_resolve_element_uid` returns `self_refs[0]` (NOT a cited-evidence pick). Run → FAIL (fields/function don't exist).

- [ ] **Step 3: Implement.** In `schemas.py`, add the three list fields to `ExtractionProvenance` + `ExtractionFieldProvenance` (defaulted, additive). In `provenance.py`: add `_resolve_self_refs`; remove `_resolve_element_uid` Strategy 3 (the evidence_ids min-pick block, lines ~239-249) so it falls straight to `self_refs[0]`; have both builders set the list fields; rewrite `build_auto_field_evidence` to read `node.provenance.property_evidence` + node positional `self_refs`/`chunk_indexes` instead of value-text-match. Run test → PASS.

- [ ] **Step 4: Wire test into run_tests.sh + commit.**

```bash
git add docker/docling-graph/app/schemas.py docker/docling-graph/app/provenance.py docker/docling-graph/tests/test_provenance_lists.py scripts/run_tests.sh
git commit -m "fix(lineage): service provenance carries self_refs/chunk_indexes/cited_refs lists; demote _resolve_element_uid

ExtractionProvenance + ExtractionFieldProvenance gain positional lists.
_resolve_element_uid no longer prefers the single cited self_ref (Part B
demoted); cited refs are diagnostics. Field evidence sourced positionally,
not by value text-match.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Worker carries the lists through parse → mentions (TDD)

**Goal:** The worker `ExtractionProvenance` + `FieldEvidenceRow` carry `self_refs`/`chunk_indexes` lists, `_parse_pass_response` populates them from the response, and `_serialize_for_audit`'s `mentions[]` forward the lists (closing the scalar-collapse boundary).

**Files:**
- Modify: `app/services/extraction_merge.py` (`ExtractionProvenance` ~175, `FieldEvidenceRow`)
- Modify: `app/workers/pipeline.py` (`_parse_pass_response` ~3781/3855 populate lists; `mentions[]` ~362-371 forward lists)
- Create: `tests/unit/test_provenance_lists_worker.py`

**Acceptance Criteria:**
- [ ] Worker `ExtractionProvenance` gains `self_refs: list[str] = field(default_factory=list)`, `chunk_indexes: list[int] = field(default_factory=list)`, `cited_refs: list[str] = field(default_factory=list)`.
- [ ] `FieldEvidenceRow` gains `self_refs: list[str]` + `chunk_indexes: list[int]` (defaulted).
- [ ] `_parse_pass_response` populates `self_refs`/`chunk_indexes`/`cited_refs` on each `ExtractionProvenance` from the response row's lists (defaulting to `[element_uid]`/`[chunk_index]` when the response is an older scalar-only shape — back-compat).
- [ ] `mentions[]` entries (`_serialize_for_audit`) include `self_refs`, `chunk_indexes`, `page_numbers` lists in ADDITION to the existing scalar `element_uid`/`chunk_index`/`page`.
- [ ] Unit test: a parsed response row with `self_refs=["#/texts/3","#/texts/4"]` produces a worker `ExtractionProvenance` with those lists, and a `mentions[]` dict built from it carries the lists.

**Verify:** `python3 -m pytest tests/unit/test_provenance_lists_worker.py -v` → passed

**Steps:**

- [ ] **Step 1: Read** `_parse_pass_response` provenance-row construction (`app/workers/pipeline.py` ~3770-3880) and `_serialize_for_audit` mentions loop (362-372) to ground exact field names.

- [ ] **Step 2: Write the failing test** — `tests/unit/test_provenance_lists_worker.py`: construct a response-row dict with `self_refs`/`chunk_indexes`/`cited_refs` lists, call the parse path (or a small extracted helper), assert the worker `ExtractionProvenance` carries them; build a `mentions[]` entry and assert the lists are present. Run → FAIL.

- [ ] **Step 3: Implement** the dataclass fields + parse population + mentions forwarding (lists added alongside the existing scalars; scalar `element_uid` kept = `self_refs[0]`). Run → PASS.

- [ ] **Step 4: Commit.**

```bash
git add app/services/extraction_merge.py app/workers/pipeline.py tests/unit/test_provenance_lists_worker.py
git commit -m "fix(lineage): worker provenance + mentions carry self_refs/chunk_indexes lists

Closes the scalar-collapse boundary: worker ExtractionProvenance + FieldEvidenceRow
gain positional lists; _parse_pass_response populates them; _serialize_for_audit
mentions[] forward them so the batch chunk-set survives to derive_structure_links.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Worker resolution — fail to batch chunk-set, never all-document (TDD)

**Goal:** `_resolve_mention_chunks` resolves each of a mention's self_refs to its chunk and, on miss, attributes to the mention's BATCH chunk-set (passed in) — the all-document fan-out is deleted. Both fan-out sites use this. The `identity_map`↔`#/texts/N` namespace gap is closed.

**Files:**
- Modify: `app/workers/pipeline.py` (`_resolve_mention_chunks` ~1771 signature + body; call site ~9323-9334; legacy `content_metadata` fan-out ~9416-9436; `identity_map` build)
- Modify: `tests/unit/test_extracted_from_self_ref_resolution.py` (extend existing)

**Acceptance Criteria:**
- [ ] `_resolve_mention_chunks` signature gains `batch_chunk_ids: list[str]` (the mention's batch chunk-set, resolved upstream from its `self_refs`/`chunk_indexes`). New order: (1) resolve EACH self_ref via `element_uid_chunk_map`/`identity_map` → union of concrete chunks; (2) on any unresolved self_ref, fall back to `batch_chunk_ids` (NOT `all_text_chunk_ids`); (3) only if `batch_chunk_ids` is also empty, return `([], is_coarse=True)` and the caller WARNs — the all-document fan-out is REMOVED.
- [ ] The mention loop (pipeline.py ~9323) resolves the mention's `self_refs` LIST (not one scalar) and derives `batch_chunk_ids` from the mention's `chunk_indexes` via the chunk-index→chunk map.
- [ ] The legacy `content_metadata` fan-out (~9416-9436) either (a) routes through the same `_resolve_mention_chunks` batch-set path, or (b) is confirmed inert for the delta pipeline and gated off with a comment + WARN if it ever fires. Decide via Step 1; do not leave an all-document fan-out anywhere.
- [ ] `identity_map`↔`#/texts/N` gap: verify the `identity_map` keys ARE `#/texts/N` self_refs (Step 1); if ingest stored a different namespace, add the bridge so resolution hits. If they already match, document that and add a test asserting a real `#/texts/N` resolves.
- [ ] Unit tests: (a) two self_refs both resolve → union of their two chunks (precise, is_coarse=False); (b) one self_ref unresolved → falls back to `batch_chunk_ids` (the batch's chunks), NOT all-document; (c) batch_chunk_ids empty + unresolved → `([], True)` with WARN; (d) concrete `{page}-{order}-...` element_uid still resolves directly (no regression).

**Verify:** `python3 -m pytest tests/unit/test_extracted_from_self_ref_resolution.py -v` → all pass

**Steps:**

- [ ] **Step 1: Investigate the namespace + legacy fan-out.** Run:
```bash
grep -n "identity_map\[" app/workers/pipeline.py | head
sed -n '4870,4890p' app/workers/pipeline.py     # where identity_map keys are built (self_ref source)
sed -n '9396,9440p' app/workers/pipeline.py     # legacy content_metadata fan-out
```
Confirm whether `identity_map` keys are `#/texts/N` (if not, the gap is real → add bridge) and whether the legacy block has a live writer (reviewer found only readers — likely inert; confirm).

- [ ] **Step 2: Write/extend the failing tests** in `tests/unit/test_extracted_from_self_ref_resolution.py` per the four acceptance cases above (new `batch_chunk_ids` param; union resolution; batch-set fallback; no all-document). Run → FAIL (signature lacks `batch_chunk_ids`; fallback still all-document).

- [ ] **Step 3: Implement** the new `_resolve_mention_chunks` signature + body (union over self_refs; batch-set fallback; remove `all_text_chunk_ids` fan-out), update the mention loop to pass `batch_chunk_ids` derived from the mention's `chunk_indexes`, and route/gate the legacy block per Step 1. Run → PASS.

- [ ] **Step 4: Commit.**

```bash
git add app/workers/pipeline.py tests/unit/test_extracted_from_self_ref_resolution.py
git commit -m "fix(lineage): resolve mention self_refs to batch chunk-set; delete all-document fan-out

_resolve_mention_chunks resolves each self_ref to its chunk and falls back to the
mention's BATCH chunk-set (passed in) on miss — never all-document. Closes the
identity_map<->#/texts/N namespace gap. Both fan-out sites routed through it.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Field chunk_id resolution + relationship source-chunk edge properties (TDD)

**Goal:** Field evidence rows get a RESOLVED `chunk_id`+`page` (not `None`), and committed relationship edges carry `source_chunk_ids`/`source_pages`/`source_self_refs` properties resolved from the relationship's positional self_refs.

**Files:**
- Modify: `app/workers/pipeline.py` (`FieldEvidenceRow` chunk_id resolution ~3855/where `_field_evidence` is built; capture rel-upsert RIDs ~1521; resolve + pass rel source-chunk refs)
- Modify: `app/services/arcadedb_graph.py` (`upsert_relationships_batch_sync` ~2448 — write source-chunk props), `app/services/arcadedb_schema.py` (~335 — register the rel-edge properties)
- Create: `tests/unit/test_field_and_rel_lineage.py`

**Acceptance Criteria:**
- [ ] Field: each `FieldEvidenceRow.chunk_id` is RESOLVED from its `self_refs`/`chunk_indexes` via the same resolver as Task 4 (no longer hardcoded `None`); `page` set from the resolved chunk. A field whose self_ref resolves to chunk C has `chunk_id == C`.
- [ ] Relationship: `RelationshipRecord`/upsert carries the rel's positional `self_refs`/`chunk_indexes`; the worker resolves them to chunk ids and the committed edge has `source_chunk_ids: list`, `source_pages: list`, `source_self_refs: list` properties.
- [ ] `arcadedb_schema.py` registers `source_chunk_ids`/`source_pages`/`source_self_refs` as properties on the ontology relationship edge types (Phase 2 rel-edge DDL ~335). Edge delete behavior unchanged (properties die with the edge).
- [ ] The relationship-upsert return RIDs (arcadedb_graph.py:2452) are captured at the call site (pipeline.py:1521), not discarded (needed to attach/verify the props).
- [ ] Unit tests: (a) field self_ref `#/texts/3` → `FieldEvidenceRow.chunk_id == <chunk for #/texts/3>`, page set; (b) a relationship with positional `self_refs=["#/texts/3"]` produces an edge whose `source_chunk_ids` contains that chunk.

**Verify:** `python3 -m pytest tests/unit/test_field_and_rel_lineage.py -v` → all pass

**Steps:**

- [ ] **Step 1: Read** the field-evidence build (`app/workers/pipeline.py` ~3848-3866, where `FieldEvidenceRow(chunk_id=None,...)` is constructed and where `_field_evidence` is later resolved/written ~1303), the rel upsert call (pipeline.py:1510-1521), `upsert_relationships_batch_sync` (arcadedb_graph.py:2448-2520), and the rel-edge DDL (arcadedb_schema.py ~329-340).

- [ ] **Step 2: Write the failing tests** — `tests/unit/test_field_and_rel_lineage.py` per cases (a)/(b). Run → FAIL (chunk_id None; no source_chunk_ids prop).

- [ ] **Step 3: Implement.** Resolve field `chunk_id`/`page` from `self_refs`/`chunk_indexes` where `_field_evidence` is finalized (reuse Task 4's resolution). For relationships: thread positional self_refs into `RelationshipRecord`, resolve to chunk ids in the worker, pass `source_chunk_ids`/`source_pages`/`source_self_refs` into `upsert_relationships_batch_sync`, write them as edge props in the upsert SQL; register the props in `arcadedb_schema.py`; capture the upsert RIDs at the call site. Run → PASS.

- [ ] **Step 4: Commit.**

```bash
git add app/workers/pipeline.py app/services/arcadedb_graph.py app/services/arcadedb_schema.py tests/unit/test_field_and_rel_lineage.py
git commit -m "fix(lineage): resolve field chunk_id + write relationship source-chunk edge props

FieldEvidenceRow.chunk_id resolved from positional self_refs (was hardcoded None).
Relationship edges carry source_chunk_ids/source_pages/source_self_refs props
(resolved from positional self_refs); upsert RIDs captured. Schema registers the
new edge props. Completes entity+field+relationship precise lineage.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Deploy, harden the all-three-targets gate, and verify end-to-end (USER-ORDERED)

**Goal:** Rebuild+redeploy, harden `verify_lineage_e2e.py` to prove PRECISE lineage for entity + field + relationship with per-target recall baselines, and confirm on a fresh SA-2 graph_only run that lineage is precise (not all-document) on all three targets.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Modify: `scripts/verify_lineage_e2e.py`

**Acceptance Criteria:**
- [ ] Deploy: docling-graph rebuilt (patches 0001–0006 apply, build green), `up -d --force-recreate -p eip-mmdpp docling-graph`, BOTH `worker-1` and `worker-graph-1` restarted, all three `StartedAt` advanced past the build; container `ir_normalizer.py` shows the `cited_refs` line; gemma4 pool idle (~0.24s probe).
- [ ] Entity precision (positional): per committed entity (run-scoped by `pipeline_run_id`), `EXTRACTED_FROM` target-chunk count ≤ its source batch width (≈1 common case), NOT ~all chunks. Report the K-distribution. FAIL if any entity links to ≥ (e.g.) 50% of document chunks.
- [ ] Field precision: `_field_evidence` rows for committed entities carry a RESOLVED `chunk_id` (count with `chunk_id != None` == count of extracted property-evidence entries; 0 hardcoded-None).
- [ ] Relationship precision: committed edges carry non-empty `source_chunk_ids`; each resolves within the source nodes' batch chunk-set.
- [ ] Per-target recall (baselines from spec §5): entity committed == run audit-blob merged-entity count; field resolved-rows == extracted property-evidence count; relationship committed == post-validation accepted rels (rejected reported separately, NOT counted as loss).
- [ ] Run-scoping & honesty: all counts filtered by `pipeline_run_id`; trace uses `EXTRACTED_FROM` ONLY (not `MENTIONED_IN`); warnings scoped to run window; fail-closed on no-evidence (missing log/empty result FAILS).
- [ ] `python3 scripts/verify_lineage_e2e.py --run <run_id> --pre-extracted-from 0` → "ALL CHECKS PASS".

**Verify:** `python3 scripts/verify_lineage_e2e.py --run <run_id> --pre-extracted-from 0` → "ALL CHECKS PASS"

**Steps:**

- [ ] **Step 1: Deploy.**
```bash
cd /home/josh/development/EIP-MMDPP/.worktrees/walltime-c0-telemetry
docker compose -p eip-mmdpp build docling-graph 2>&1 | grep -E "Applying patch|Built|FAILED"
docker compose -p eip-mmdpp up -d --force-recreate docling-graph
docker restart eip-mmdpp-worker-1 eip-mmdpp-worker-graph-1
for c in docling-graph-1 worker-1 worker-graph-1; do docker inspect eip-mmdpp-$c --format "$c {{.State.StartedAt}}"; done
docker exec eip-mmdpp-docling-graph-1 grep -c "cited_refs" /app/repo/docling_graph/core/extractors/contracts/delta/ir_normalizer.py
for h in 10.0.1.121 10.0.1.109; do printf "%s " "$h"; curl -s --max-time 30 -o /dev/null -w '%{time_total}s\n' http://$h:11434/api/generate -d '{"model":"gemma4:31b","prompt":"hi","stream":false,"options":{"num_predict":1}}'; done
```
Expected: patches incl 0006 applied; all StartedAt advanced; cited_refs present; pool idle.

- [ ] **Step 2: Harden the gate.** Read `scripts/verify_lineage_e2e.py`; add the three precision checks (entity K-distribution vs batch width, field resolved-chunk_id count, relationship source_chunk_ids) and the three per-target recall baselines, all run-scoped, `EXTRACTED_FROM`-only trace, fail-closed. Keep the `[PASS]/[FAIL]` + "ALL CHECKS PASS" output.

- [ ] **Step 3: Run SA-2 graph_only on the fixed build.**
```bash
DOC=ddaa9e36-2854-47c3-bc94-ff38d531dafd
curl -s -X POST "http://localhost:8005/v1/documents/$DOC/reingest" -H "Content-Type: application/json" \
  -d '{"mode":"graph_only","ontology_bundle_key":"air_defense_v3_merged_v1"}' | python3 -m json.tool
```
Record `pipeline_run_id`. Multi-hour; monitor in background (lives in Celery).

- [ ] **Step 4: At terminal, run the gate + spot-check precision.**
```bash
RUN=<pipeline_run_id>
python3 scripts/verify_lineage_e2e.py --run "$RUN" --pre-extracted-from 0
# spot-check entity is NOT all-document:
ADB() { curl -s -u root:eip_arcadedb_secret -X POST http://localhost:2480/api/v1/command/eip_knowledge_graph -H "Content-Type: application/json" -d "{\"language\":\"sql\",\"command\":\"$1\"}"; }
echo -n "doc chunks: "; ADB "SELECT count(*) AS c FROM TextChunk WHERE document_id='ddaa9e36-2854-47c3-bc94-ff38d531dafd'" | python3 -c "import sys,json;print(json.load(sys.stdin)['result'][0]['c'])"
ADB "SELECT system_name, out('EXTRACTED_FROM').size() AS ef FROM RADAR_SYSTEM WHERE out('EXTRACTED_FROM').size()>0 LIMIT 5" | python3 -m json.tool
```
Expected: ALL CHECKS PASS; per-entity `ef` is small (≈1, not ≈doc-chunk-count).

- [ ] **Step 5: Record outcome** in memory `project_extracted_from_root_cause.md`: precision before→after (SNR-75 736 edges → small set), field chunk_id resolved count, relationship source_chunk_ids present, per-target recall held.

---

## Notes for the executor

- **Order/deploy:** Tasks 1+2 are docling-graph (rebuild required); Tasks 3-5 are worker `app/**` (bind-mounted → restart, no rebuild). ALL must be deployed before Task 6. Restart BOTH workers (worker-1 is a catch-all that also consumes `graph`); confirm `StartedAt`. Compose from this worktree needs `-p eip-mmdpp`.
- **Never** `POST /v1/documents/{id}/cancel` (hard-deletes). To stop a run, revoke tasks + reset status; ask first.
- **Positional ≠ truncation-dependent:** a truncated batch still has a known chunk-set, so lineage holds.
- **Non-goals (measured follow-ons):** single-chunk-everywhere narrowing; `/v1/graph/query sources=null` (verify after precision).
