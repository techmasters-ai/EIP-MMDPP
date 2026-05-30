# Entity-Commit + Field-Value Lineage Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make extracted entities commit to ArcadeDB with full, resolvable lineage (element_uid + document + page) for every field value, and enforce — at the worker import boundary — that no entity commits without lineage.

**Architecture:** Two root failures, fixed independently then gated. (B) docling-graph emits empty `element_uid`/`page` because `doc_processor.last_chunk_metadata` is empty at the `main.py` provenance build → fix the chunk-metadata threading on both delta routes. (A) merged entities don't reach ArcadeDB → diagnose the post-merge commit path and fix the specific break. (C) add a strict lineage-required gate at `_import_graph_phase_nodes` so any entity lacking resolvable lineage is rejected, never committed. A live two-config diagnostic (Task 0) precedes all fixes and pins exact locations + production scope.

**Tech Stack:** Python, FastAPI (docling-graph service), Celery (worker), ArcadeDB (graph), pytest. docling-graph clone code under `docker/docling-graph/repo/` is gitignored → changes ship as a tracked patch (`docker/docling-graph/patches/0005-*.patch`); `docker/docling-graph/app/*.py` is tracked → edited directly; worker `app/**` is bind-mounted → edited directly, no rebuild.

**Spec:** `docs/superpowers/specs/2026-05-30-element-uid-lineage-fix-design.md` (rev 3).

---

## File map (where each change lands + deployment semantics)

| File | Repo location | Edit mode | Deploy |
|---|---|---|---|
| `app/workers/pipeline.py` (worker) | bind-mounted `app/` | edit directly | restart worker-graph (no rebuild) |
| `docker/docling-graph/app/provenance.py` | tracked (outer) | edit directly | COPY → docling-graph rebuild |
| `docker/docling-graph/app/main.py` | tracked (outer) | edit directly | COPY → docling-graph rebuild |
| `docker/docling-graph/repo/.../strategy_ops.py` | gitignored clone | via patch 0005 | applied at build → rebuild |
| `docker/docling-graph/repo/.../many_to_one.py` | gitignored clone | via patch 0005 | applied at build → rebuild |
| `docker/docling-graph/repo/.../document_processor.py` | gitignored clone | via patch 0005 | applied at build → rebuild |
| `tests/unit/**` | tracked | edit directly | n/a (host pytest via scripts/run_tests.sh) |

---

### Task 0: Two-config diagnostic — localize the commit break + provenance-empty source + production scope

**Goal:** Before any code change, prove (A) exactly where merged entities fail to reach ArcadeDB, (B) exactly why `element_uid`/`page` are empty and on which delta route, and (C) whether production-config (`air_defense_v3`, non-narrowed) is affected — so the fixes target real locations, not assumptions.

**USER-ORDERED GATE — NON-SKIPPABLE.** This is a verification gate the user requested ("diagnose service post-filter / postprocess / merge / graph import separately"). It MUST NOT be closed by reasoning inline; close only after the diagnostic run output is captured for each instrumented stage.

**Files:**
- Create: `scripts/diagnose_lineage_commit.py` (read-only/instrumented harness; no production code changes)
- Read (do not edit yet): `app/workers/pipeline.py:1217-1308` (`_import_graph_phase_nodes`), `docker/docling-graph/app/main.py:1095-1140` (chunk_to_self_refs build), `docker/docling-graph/app/provenance.py:84-160` (synthesizer), `docker/docling-graph/repo/.../strategies/many_to_one.py:340-370,561` (route selection + pre-built metadata)

**Acceptance Criteria:**
- [ ] For an SA-2 graph_only run, the diagnostic reports, per field-group pass: docling-graph response entity count, `merge_and_resolve` output entity count, whether `_import_graph_phase_nodes`/`upsert_nodes_batch_sync` executed and how many RIDs it returned, and the final ArcadeDB entity-vertex count for the run's types.
- [ ] The diagnostic reports which delta route ran (`extract_delta_from_document` vs `_extract_delta_from_pre_built_chunks`) and whether `doc_processor.last_chunk_metadata` was non-empty at the `main.py` read, with `self_refs`/`page_numbers` counts.
- [ ] The SAME diagnostic run is repeated under a production-representative config (`air_defense_v3` bundle, `VECTOR_ROUTER_MODE=shadow`) and the commit + provenance results are recorded for comparison.
- [ ] A short findings note is written to `reports/collection/lineage_diagnostic_findings.md` stating: the exact commit-break stage, the exact provenance-empty cause + route, and whether production config is affected.

**Verify:** `python3 scripts/diagnose_lineage_commit.py --run <sa2_graph_only_run_id>` → prints the per-stage table; `reports/collection/lineage_diagnostic_findings.md` exists with the three conclusions filled in.

**Steps:**

- [ ] **Step 1: Write the diagnostic harness.** `scripts/diagnose_lineage_commit.py` (subprocess list-form; reads postgres + ArcadeDB; replays merge in-process inside worker-graph via `docker exec`). For a given run_id it must print:
  - per pass: `_cached_entities()` count from rehydrated pass output (via `load_completed_pass_outputs` + `_rehydrate_pass_result`)
  - `merge_and_resolve(...).entities` count (in-process replay)
  - ArcadeDB committed entity-vertex count for the run's entity types (sum over RADAR_SYSTEM/MISSILE_SYSTEM/FIRE_CONTROL_SYSTEM/etc.)
  - docling-graph log scan for this run: `no chunk metadata available` warnings, `synthesized N provenance rows`, route markers (`CHUNKED-BATCHES`, pre-built vs document)

```python
#!/usr/bin/env python3
"""Diagnose the entity-commit + lineage break for a graph_only run.
Read-only. Run via host python; uses docker exec for in-container merge replay."""
import argparse, json, subprocess

ENTITY_TYPES = ["RADAR_SYSTEM","MISSILE_SYSTEM","FIRE_CONTROL_SYSTEM","WEAPON_SYSTEM",
                "EQUIPMENT_SYSTEM","LAUNCHER_SYSTEM","ELECTRONIC_WARFARE_SYSTEM",
                "AIR_DEFENSE_ARTILLERY_SYSTEM","INTEGRATED_AIR_DEFENSE_SYSTEM"]

def adb(sql):
    out = subprocess.run(["curl","-s","--max-time","10","-u","root:eip_arcadedb_secret",
        "-X","POST","http://localhost:2480/api/v1/command/eip_knowledge_graph",
        "-H","Content-Type: application/json","-d",json.dumps({"language":"sql","command":sql})],
        capture_output=True, text=True, timeout=20).stdout
    try: return json.loads(out).get("result", [])
    except Exception: return []

def committed_entity_count(run_id):
    total = 0
    for t in ENTITY_TYPES:
        r = adb(f"SELECT count(*) AS n FROM `{t}`")
        total += (r[0]["n"] if r else 0)
    return total

def merge_replay(run_id, bundle, doc_id):
    # in-process replay inside worker-graph (real code paths)
    py = (
        "from app.workers.pipeline import _rehydrate_pass_result,_get_db,load_completed_pass_outputs;"
        "from app.services.ontology_bundles import load_bundle_manifest;"
        "from app.services.ontology_templates import load_ontology;"
        "from app.services.extraction_merge import merge_and_resolve;"
        f"db=_get_db();m=load_bundle_manifest('{bundle}');o=load_ontology(bundle_key='{bundle}');"
        f"outs=load_completed_pass_outputs(db,'{run_id}');"
        f"reh={{n:_rehydrate_pass_result(r,m,o,'{doc_id}',db=db,run_id='{run_id}') for n,r in outs.items()}};"
        "per={n:len(p._cached_entities()) for n,p in reh.items()};"
        f"mg=merge_and_resolve(pass_results=reh,manifest=m,ontology=o,document_id='{doc_id}',pipeline_run_id='{run_id}');"
        "import json;print('DIAG',json.dumps({'per_pass':per,'merged_entities':len(mg.entities),"
        "'merged_rels':len(getattr(mg,'relationships',[]) or [])}));db.close()"
    )
    out = subprocess.run(["docker","exec","eip-mmdpp-worker-graph-1","python","-c",py],
                         capture_output=True, text=True, timeout=300).stdout
    for line in out.splitlines():
        if line.startswith("DIAG "):
            return json.loads(line[5:])
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bundle", default="air_defense_v3_merged_v1")
    ap.add_argument("--doc", required=True)
    a = ap.parse_args()
    rep = merge_replay(a.run, a.bundle, a.doc)
    committed = committed_entity_count(a.run)
    print(f"run={a.run} bundle={a.bundle}")
    print(f"  per-pass rehydrated entities: {rep.get('per_pass')}")
    print(f"  merge_and_resolve entities:   {rep.get('merged_entities')}  rels: {rep.get('merged_rels')}")
    print(f"  ArcadeDB committed entities:  {committed}")
    print(f"  >>> COMMIT BREAK: merged={rep.get('merged_entities')} but committed={committed}"
          if (rep.get('merged_entities') or 0) > committed else "  >>> commit OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run against the existing SA-2 graph_only run** (`9d48fc1e-62fd-4b03-a98c-98a35bda3b8e`, doc `ddaa9e36-2854-47c3-bc94-ff38d531dafd`).

Run: `python3 scripts/diagnose_lineage_commit.py --run 9d48fc1e-62fd-4b03-a98c-98a35bda3b8e --doc ddaa9e36-2854-47c3-bc94-ff38d531dafd`
Expected: prints merged≈22 vs committed=0 → COMMIT BREAK confirmed; localizes whether merge replay succeeds (isolating commit-path vs merge-logic).

- [ ] **Step 3: Localize the commit break precisely.** With the merge-replay number in hand, add temporary `logger.info` instrumentation to `_import_graph_phase_nodes` (pipeline.py:1295-1308) logging `len(merged.entities)`, `len(node_records)`, and `len(node_rids)` after `upsert_nodes_batch_sync`. Trigger a fresh SA-2 graph_only run on an idle pool; capture the worker log. (This temp logging is removed in Task 4.)

Run: `docker logs eip-mmdpp-worker-graph-1 --since 30m 2>&1 | grep -iE "import_graph_phase|upsert_nodes|node_rids"`
Expected: identifies whether (i) merge wasn't dispatched, (ii) `merged.entities` is empty at import, (iii) `upsert_nodes_batch_sync` raised/returned 0, or (iv) it returned RIDs but they aren't queryable (DB/type mismatch).

- [ ] **Step 4: Production-config comparison run.** Set `VECTOR_ROUTER_MODE=shadow` and bundle `air_defense_v3` (recreate worker-graph + docling-graph from worktree, `-p eip-mmdpp`, force-recreate), run a graph_only extraction on the SA-2 doc, idle pool, and re-run the diagnostic.

Run: `python3 scripts/diagnose_lineage_commit.py --run <new_shadow_run_id> --bundle air_defense_v3 --doc ddaa9e36-2854-47c3-bc94-ff38d531dafd`
Expected: records whether the commit break + empty provenance also occur under production config (systemic) or not (narrowed-path-specific).

- [ ] **Step 5: Write findings + commit.** Write `reports/collection/lineage_diagnostic_findings.md` with the three conclusions (commit-break stage; provenance-empty cause + route; production-affected y/n). Revert the temp instrumentation from Step 3 (keep it ONLY if Task 4 will formalize it). Commit the diagnostic script.

```bash
git add scripts/diagnose_lineage_commit.py
git commit -m "diag(lineage): two-config diagnostic harness for entity-commit + provenance break

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
(`reports/` is gitignored — the findings note is a working artifact, not committed.)

---

### Task 1: Lineage population (B) — populate element_uid + page on both delta routes (docling-graph patch 0005)

**Goal:** Make docling-graph emit a non-empty `element_uid` AND a resolved `page` for every entity, on BOTH the from-document route and the pre-built-chunk route, by ensuring chunk metadata (self_refs + page_numbers) is available to the provenance builder and threading page into the synthesizer.

**Files:**
- Modify (via patch 0005, repo clone): `docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/strategy_ops.py` — ensure `last_chunk_metadata` carries `self_refs` AND `page_numbers` for both `extract_delta_from_text`/`extract_delta_from_document`
- Modify (via patch 0005, repo clone): `docker/docling-graph/repo/docling_graph/core/extractors/strategies/many_to_one.py:561` — pre-built-chunk metadata must populate `page_numbers` (today `[]`) from `entry["source_refs"]` resolved against the DoclingDocument or carried from worker selected chunks
- Modify (tracked, direct): `docker/docling-graph/app/provenance.py` — change `synthesize_provenance_from_pass_output` signature to accept a `chunk_to_page_numbers: dict[int, list[int]] | None` (or full chunk-metadata map); emit resolved `page` instead of `None`; align `_resolve_page` + `build_provenance_from_context` to the same page source
- Modify (tracked, direct): `docker/docling-graph/app/main.py:1095-1140` — build a `chunk_to_page_numbers` map alongside `chunk_to_self_refs` and pass it to the synthesizer call (currently `main.py` ~1702)
- Create: `tests/unit/test_provenance_page_resolution.py`
- Create: `docker/docling-graph/repo/tests/test_strategy_ops_chunk_metadata.py` (clone-side) AND mirror the host-runnable algorithm test to `tests/unit/`

**Acceptance Criteria:**
- [ ] `synthesize_provenance_from_pass_output`, given a populated chunk-to-page map, emits non-empty `element_uid` AND non-null `page` for every entity (no `page=None` when page data exists).
- [ ] `_resolve_page` and `build_provenance_from_context` read page from the same source so primary and fallback paths agree.
- [ ] Both delta routes produce chunk metadata with non-empty `self_refs` and `page_numbers`.
- [ ] `element_uid=None`/`page=None` only when the source genuinely lacks them (e.g. a doc element with no page), never due to a missing map.

**Verify:** `python3 -m pytest tests/unit/test_provenance_page_resolution.py -v` → PASS; patch dry-run `cd /tmp/cleanclone && patch -p1 --dry-run < .../0005-*.patch` → applies clean.

**Steps:**

- [ ] **Step 1: Write the failing provenance page test.**

```python
# tests/unit/test_provenance_page_resolution.py
import importlib.util, sys
from pathlib import Path
import pytest
pytestmark = pytest.mark.unit

# Load provenance.py directly (docling-graph app dir, not a host package).
_PROV = Path(__file__).resolve().parents[2] / "docker/docling-graph/app/provenance.py"

def _load():
    spec = importlib.util.spec_from_file_location("dg_provenance", _PROV)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

class _ProvStub:
    def __init__(self, **kw): self.__dict__.update(kw)

def test_synthesize_emits_resolved_page_when_map_present():
    prov = _load()
    pass_output = {"radar_systems": [{"system_name": "Fan Song"}, {"system_name": "SNR-75"}]}
    class _Rec:
        model_config = {"ontology_name": "RADAR_SYSTEM"}
    class _Tpl:
        model_fields = {"radar_systems": type("F", (), {"annotation": list[_Rec]})()}
    rows = prov.synthesize_provenance_from_pass_output(
        pass_output=pass_output, template_cls=_Tpl,
        chunk_to_self_refs={0: ["#/texts/12"]},
        chunk_to_page_numbers={0: [3]},          # NEW arg
        provenance_cls=_ProvStub,
    )
    assert rows, "must synthesize a row per entity"
    assert all(r.element_uid for r in rows), "element_uid must be non-empty"
    assert all(r.page == 3 for r in rows), "page must resolve from chunk_to_page_numbers, not None"
```

- [ ] **Step 2: Run it — fails** (signature has no `chunk_to_page_numbers`).

Run: `python3 -m pytest tests/unit/test_provenance_page_resolution.py -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'chunk_to_page_numbers'`.

- [ ] **Step 3: Implement in `docker/docling-graph/app/provenance.py`.** Add `chunk_to_page_numbers` param; resolve page from it (first page of the entity's chunk); keep `element_uid` resolution. Align `_resolve_page` to also accept page_numbers from the same map when present.

```python
def synthesize_provenance_from_pass_output(
    pass_output, template_cls, chunk_to_self_refs, provenance_cls,
    chunk_to_page_numbers=None,   # NEW
):
    ...
    if chunk_to_self_refs:
        first_refs = chunk_to_self_refs.get(0) or next(iter(chunk_to_self_refs.values()), None)
        first_element_uid = first_refs[0] if first_refs else ""
    else:
        first_element_uid = ""
    # NEW: resolve page from the same chunk index
    first_page = None
    if chunk_to_page_numbers:
        pages = chunk_to_page_numbers.get(0) or next(iter(chunk_to_page_numbers.values()), None)
        first_page = pages[0] if pages else None
    ...
    out.append(provenance_cls(
        instance_id=str(uuid.uuid4()), ontology_name=ontology_name,
        identity_values=identity_values, element_uid=first_element_uid,
        page=first_page,                                   # was None
        chunk_index=0 if chunk_to_self_refs else None,
    ))
```

- [ ] **Step 4: Run test — passes.**

Run: `python3 -m pytest tests/unit/test_provenance_page_resolution.py -v`
Expected: PASS.

- [ ] **Step 5: Thread `chunk_to_page_numbers` in `docker/docling-graph/app/main.py`.** In the chunk_to_self_refs build block (~1107), build a parallel `chunk_to_page_numbers` from `cmeta.get("page_numbers")`, store on context, and pass it to the `synthesize_provenance_from_pass_output(...)` call (~1702). (Verify line numbers against the file at edit time.)

- [ ] **Step 6: Generate patch 0005 for the repo-clone changes** (strategy_ops + many_to_one + document_processor self_refs/page_numbers threading). Diff each touched repo file against its clean upstream base (per the 0003/0004 method), write `docker/docling-graph/patches/0005-lineage-chunk-metadata.patch`, and dry-run it against a fresh clean clone.

Run: `cd /tmp && rm -rf cc && git clone <clone-origin> cc && cd cc && for p in <worktree>/docker/docling-graph/patches/*.patch; do patch -p1 --dry-run < "$p"; done`
Expected: all patches (0001-0005) apply clean.

- [ ] **Step 7: Commit** (tracked app files + patch + tests).

```bash
git add docker/docling-graph/app/provenance.py docker/docling-graph/app/main.py \
  docker/docling-graph/patches/0005-lineage-chunk-metadata.patch \
  tests/unit/test_provenance_page_resolution.py
git commit -m "fix(lineage): populate element_uid + page on both delta routes (Task 1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Strict lineage-required commit gate (C) at the worker import boundary

**Goal:** At `_import_graph_phase_nodes`, refuse to upsert any entity whose lineage cannot be resolved (no resolvable `element_uid` + `page`), record the rejected set as a hard failure signal, and upsert only the lineage-complete set — so no entity ever commits without lineage, independent of Task 1.

**Files:**
- Modify: `app/workers/pipeline.py:1295-1308` (`_import_graph_phase_nodes`)
- Test: `tests/unit/test_lineage_commit_gate.py`

**Acceptance Criteria:**
- [ ] An entity whose `MergedEntityRecord.provenance` has at least one row with non-empty `element_uid` AND non-null `page` is upserted.
- [ ] An entity with no such provenance row is NOT in `node_records` (not upserted) and is recorded in a returned/logged `rejected_for_lineage` list (count + identities).
- [ ] `identity_to_rid` is built only from the upserted (lineage-complete) entities, preserving zip-strict correctness.
- [ ] The gate logs a loud WARNING/ERROR with the rejected count (never a silent skip).

**Verify:** `python3 -m pytest tests/unit/test_lineage_commit_gate.py -v` → PASS.

**Steps:**

- [ ] **Step 1: Write failing test.** A fake `merged` with two entities — one with resolvable provenance, one with empty `element_uid` — asserts only the resolvable one reaches `upsert_nodes_batch_sync` and the other is rejected.

```python
# tests/unit/test_lineage_commit_gate.py
import pytest
pytestmark = pytest.mark.unit
from app.workers import pipeline

class _Prov:
    def __init__(self, element_uid, page): self.element_uid=element_uid; self.page=page
class _Ident:
    def __init__(self, n): self._n=n
    def identity_values_dict(self): return {"system_name": self._n}
class _Ent:
    def __init__(self, n, prov): self.identity=_Ident(n); self.properties={}; self.confidence=0.9; self.provenance=prov
class _Merged:
    def __init__(self, ents): self.entities=ents

def test_gate_rejects_entity_without_lineage(monkeypatch):
    captured = {}
    class _Store:
        def upsert_nodes_batch_sync(self, records, provenance):
            captured["records"] = records
            return [f"#1:{i}" for i in range(len(records))]
    monkeypatch.setattr(pipeline, "get_graph_store", lambda: _Store())
    merged = _Merged([
        _Ent("Fan Song", [_Prov("#/texts/12", 3)]),     # resolvable
        _Ent("Ghost",    [_Prov("", None)]),             # no lineage
    ])
    class _Tracker:
        def mark(self): pass
    result = pipeline._import_graph_phase_nodes(merged, ontology=None,
        document_id="doc", tracker=_Tracker(), provenance={})
    names = {r.identity_values.get("system_name") if hasattr(r,'identity_values') else None
             for r in captured["records"]}
    # only the resolvable entity upserts
    assert len(captured["records"]) == 1
```

- [ ] **Step 2: Run — fails** (current code upserts both).

Run: `python3 -m pytest tests/unit/test_lineage_commit_gate.py -v`
Expected: FAIL — 2 records upserted, expected 1.

- [ ] **Step 3: Implement the gate in `_import_graph_phase_nodes`** (pipeline.py:1295).

```python
def _has_resolvable_lineage(e) -> bool:
    return any(
        getattr(p, "element_uid", "") and getattr(p, "page", None) is not None
        for p in (getattr(e, "provenance", None) or [])
    )

lineage_ok = [e for e in merged.entities if _has_resolvable_lineage(e)]
rejected = [e for e in merged.entities if not _has_resolvable_lineage(e)]
if rejected:
    logger.error(
        "LINEAGE_GATE: rejecting %d/%d entities lacking resolvable lineage "
        "(element_uid+page); identities=%r",
        len(rejected), len(merged.entities),
        [e.identity.identity_values_dict() for e in rejected][:20],
    )

node_records = [_build_node_record(e) for e in lineage_ok]
tracker.mark()
graph_store = get_graph_store()
node_rids = graph_store.upsert_nodes_batch_sync(node_records, provenance)
identity_to_rid = dict(zip((e.identity for e in lineage_ok), node_rids, strict=True))
return identity_to_rid
```

- [ ] **Step 4: Run — passes.**

Run: `python3 -m pytest tests/unit/test_lineage_commit_gate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add app/workers/pipeline.py tests/unit/test_lineage_commit_gate.py
git commit -m "feat(lineage): strict commit gate — reject entities without resolvable lineage (Task 2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Entity-commit fix (A) — fix the specific break Task 0 localized

**Goal:** Fix the actual post-merge commit failure identified by the Task-0 diagnostic so lineage-complete merged entities reach ArcadeDB. (Exact code change depends on Task-0 findings; this task formalizes + tests it.)

**Files:**
- Modify: the file/function Task 0 identifies (candidates, in likelihood order): `app/workers/pipeline.py` (`_import_graph_phase_nodes` / merge dispatch / `upsert_nodes_batch_sync` error handling), or `app/services/graph_store.py` (`upsert_nodes_batch_sync`)
- Test: `tests/unit/test_entity_commit_regression.py`

**Acceptance Criteria:**
- [ ] The specific break from Task 0 (e.g. swallowed upsert exception, type mismatch, merge not dispatched on this path) is fixed with a focused change.
- [ ] A regression test reproduces the pre-fix failure and passes post-fix.
- [ ] After the fix, a lineage-complete entity set upserts and is queryable in ArcadeDB.

**Verify:** `python3 -m pytest tests/unit/test_entity_commit_regression.py -v` → PASS. (End-to-end committed-count proof is Task 5.)

**Steps:**

- [ ] **Step 1: Read `reports/collection/lineage_diagnostic_findings.md`** (Task 0 output) for the exact break.
- [ ] **Step 2: Write a failing regression test** reproducing that break (e.g. `upsert_nodes_batch_sync` raising on a known input must surface, not be swallowed; or merge-dispatch fires on the graph_only path). Code the assertion to the actual failure mode found.
- [ ] **Step 3: Run — fails.** Run: `python3 -m pytest tests/unit/test_entity_commit_regression.py -v` → FAIL.
- [ ] **Step 4: Implement the minimal fix** at the located site.
- [ ] **Step 5: Run — passes.** Run: `python3 -m pytest tests/unit/test_entity_commit_regression.py -v` → PASS.
- [ ] **Step 6: Commit.**

```bash
git add app/workers/pipeline.py tests/unit/test_entity_commit_regression.py
git commit -m "fix(commit): <specific break from Task 0> — merged entities reach ArcadeDB (Task 3)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

> NOTE: if Task 0 proves the ONLY commit break was the empty-lineage drop (i.e. with Task 1 populating lineage and Task 2's gate, all 22 entities now have lineage and commit), this task reduces to writing the regression test that pins that behavior — no separate fix needed. Record that outcome explicitly rather than inventing a fix.

---

### Task 4: Deploy + remove temp instrumentation

**Goal:** Build docling-graph with patch 0005 + tracked app changes, deploy worker + docling-graph from the worktree, and remove any temporary diagnostic logging added in Task 0 Step 3.

**Files:**
- Modify: `app/workers/pipeline.py` (remove temp Task-0 logging if not formalized)

**Acceptance Criteria:**
- [ ] `docker compose -p eip-mmdpp build docling-graph` succeeds with all patches 0001-0005 applying (log shows each `Applying patch`).
- [ ] The built image contains the page-resolving provenance + patched strategy_ops/many_to_one (grep inside the image).
- [ ] worker-graph + docling-graph recreated from the worktree path (`-p eip-mmdpp`); both healthy; pool idle.
- [ ] No leftover temporary `logger.info` debug lines from Task 0.

**Verify:** build exits 0 with `Applying patch: /app/patches/0005-*`; `docker exec eip-mmdpp-docling-graph-1 grep -c chunk_to_page_numbers /app/app/provenance.py` ≥ 1.

**Steps:**

- [ ] **Step 1: Remove temp instrumentation** added in Task 0 Step 3 (if any remained). `git diff` to confirm only intended changes.
- [ ] **Step 2: Rebuild docling-graph.** Run: `cd <worktree> && docker compose -p eip-mmdpp build --build-arg CACHE_BUST=$(date +%s) docling-graph 2>&1 | grep -E "Applying patch|Built|error"` → all 5 patches apply, image Built.
- [ ] **Step 3: Verify image contents.** Run: `docker run --rm --entrypoint sh eip-mmdpp-docling-graph -c 'grep -c chunk_to_page_numbers /app/app/provenance.py; grep -c page_numbers /app/repo/docling_graph/core/extractors/strategies/many_to_one.py'` → both ≥ 1.
- [ ] **Step 4: Recreate + restart.** Run: `cd <worktree> && docker compose -p eip-mmdpp up -d --force-recreate docling-graph && docker restart eip-mmdpp-worker-graph-1`; wait healthy; probe pool idle (active-inference latency, both Ollama nodes ~0.2s).
- [ ] **Step 5: Commit** any instrumentation removal.

```bash
git add app/workers/pipeline.py
git commit -m "chore(lineage): remove Task-0 temp instrumentation; deploy fix build (Task 4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: End-to-end verification gate

**Goal:** Prove on a real SA-2 graph_only run (fixed build, idle pool) that entities commit with full lineage and a field value traces to chunk + document + page — and that a genuinely-empty doc still yields 0 legitimately.

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user (the hard data-lineage requirement). It MUST NOT be closed by inline reasoning — close only after each acceptance check is run against ArcadeDB/postgres with output captured.

**Files:**
- Create: `scripts/verify_lineage_e2e.py` (read-only checks)

**Acceptance Criteria:**
- [ ] ArcadeDB entity vertices > 0 for the SA-2 run's types, count consistent with merged-entity count from the diagnostic.
- [ ] Worker log for the run shows ZERO "dropping provenance row missing required fields" warnings AND zero `LINEAGE_GATE: rejecting` for entities that should have lineage.
- [ ] `EXTRACTED_FROM` and/or `MENTIONED_IN` edges > 0 for the run.
- [ ] For one sample entity field value: trace it to its `element_uid` → TextChunk/ExtractionChunk → Document + page number (printed end-to-end).
- [ ] Discriminator: a known image/empty doc (e.g. `cw_radar.jpg`) run yields 0 entities legitimately with docling-graph `raw_node_count`=1 (not a silent drop) — confirm the gate did NOT reject populated entities there.

**Verify:** `python3 scripts/verify_lineage_e2e.py --run <fixed_sa2_run_id>` → all checks print PASS; the field→chunk→doc→page trace prints a concrete example.

**Steps:**

- [ ] **Step 1: Run a fresh SA-2 graph_only extraction** on the fixed build, idle pool. Capture the run_id.
- [ ] **Step 2: Write `scripts/verify_lineage_e2e.py`** that runs the five acceptance checks against ArcadeDB + postgres + docling-graph logs and prints PASS/FAIL per check plus the concrete field→chunk→document→page trace for one entity.
- [ ] **Step 3: Run it.** Run: `python3 scripts/verify_lineage_e2e.py --run <run_id>` → all PASS.
- [ ] **Step 4: Run the empty-doc discriminator** on `cw_radar.jpg`'s run; confirm legitimate 0 entities + `raw_node_count`=1.
- [ ] **Step 5: Commit the verifier + update the defect memory** to "resolved, verified."

```bash
git add scripts/verify_lineage_e2e.py
git commit -m "test(lineage): end-to-end lineage verification gate (Task 5)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Post-plan follow-ups (not tasks in this plan)
- Restart the paused notebooks-collection ingest (all 21 docs) on the fixed build — the user requested this AFTER the fix lands.
- Re-assess scope of `project_recall_gate_collapse_investigation` recall numbers (entity-count vs value-fill) now that entities commit.
