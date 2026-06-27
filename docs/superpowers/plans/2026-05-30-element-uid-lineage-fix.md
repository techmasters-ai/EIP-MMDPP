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
- [ ] For an SA-2 graph_only run, the diagnostic reports, per field-group pass: docling-graph response entity count, `merge_and_resolve` output entity count, whether `_import_graph_phase_nodes`/`upsert_nodes_batch_sync` executed and how many RIDs it returned, and — because entity vertices carry no run_id (review finding High-1) — a **run-attributable committed-entity delta** via a pre/post global-count snapshot around a fresh extraction (NOT a global absolute count, which can false-OK when other entities exist).
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

def committed_entity_count():
    """GLOBAL entity-vertex count across all entity types. Entity vertices carry
    NO run_id/document_id property (verified: RADAR_SYSTEM/MISSILE_SYSTEM schemas
    have only domain + housekeeping fields), so this count is NOT run-scoped.
    Review finding High-1: a global count can false-OK a broken run when other
    entities already exist. Therefore the diagnostic uses a PRE/POST SNAPSHOT
    delta around a fresh extraction, NOT an absolute count, to attribute commits
    to the run under test."""
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
        # MergedExtraction has .edges (NOT .relationships) — review finding Low.
        "import json;print('DIAG',json.dumps({'per_pass':per,'merged_entities':len(mg.entities),"
        "'merged_edges':len(getattr(mg,'edges',[]) or [])}));db.close()"
    )
    out = subprocess.run(["docker","exec","eip-mmdpp-worker-graph-1","python","-c",py],
                         capture_output=True, text=True, timeout=300).stdout
    for line in out.splitlines():
        if line.startswith("DIAG "):
            return json.loads(line[5:])
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="existing run to replay merge for (offline)")
    ap.add_argument("--bundle", default="air_defense_v3_merged_v1")
    ap.add_argument("--doc", required=True)
    ap.add_argument("--snapshot", choices=["pre", "post"], default=None,
                    help="for a FRESH run: capture global entity count before/after to get a run-attributable delta (High-1)")
    a = ap.parse_args()
    if a.snapshot:
        # pre/post snapshot mode: prints the global count; caller diffs pre vs post
        # around a fresh extraction to attribute commits to THIS run (entity
        # vertices carry no run_id, so absolute counts are not run-scoped).
        print(f"SNAPSHOT_{a.snapshot.upper()} global_entity_count={committed_entity_count()}")
        return
    rep = merge_replay(a.run, a.bundle, a.doc)
    committed = committed_entity_count()
    print(f"run={a.run} bundle={a.bundle}")
    print(f"  per-pass rehydrated entities: {rep.get('per_pass')}")
    print(f"  merge_and_resolve entities:   {rep.get('merged_entities')}  edges: {rep.get('merged_edges')}")
    print(f"  ArcadeDB GLOBAL entities:     {committed}  (NOT run-scoped — use --snapshot pre/post around a fresh run for run-attributable delta)")
    print(f"  >>> merge replay yields {rep.get('merged_entities')} entities; "
          f"if a fresh run's post-pre snapshot delta is < that, COMMIT BREAK is downstream of merge")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Merge-replay on the existing SA-2 run + snapshot-delta on a FRESH run.** Two parts, because committed counts are global (not run-scoped):
  - (a) **Offline merge replay** on the existing run to get the expected entity count:
    Run: `python3 scripts/diagnose_lineage_commit.py --run 9d48fc1e-62fd-4b03-a98c-98a35bda3b8e --doc ddaa9e36-2854-47c3-bc94-ff38d531dafd`
    Expected: prints `merge_and_resolve entities: ~22` (the count that SHOULD commit). The printed "GLOBAL entities" is informational only.
  - (b) **Snapshot-delta around a fresh extraction** to measure what actually commits for THAT run:
    Run: `python3 scripts/diagnose_lineage_commit.py --snapshot pre` → record `SNAPSHOT_PRE global_entity_count=<P>`; trigger a fresh SA-2 graph_only run (idle pool), wait terminal; `python3 scripts/diagnose_lineage_commit.py --snapshot post` → record `SNAPSHOT_POST global_entity_count=<Q>`.
    Expected: delta `Q-P` is the entities this run committed. If `Q-P` ≈ 0 while merge replay ≈ 22 → **COMMIT BREAK confirmed** (downstream of merge).

- [ ] **Step 3: Localize the commit break precisely.** With the merge-replay number + the fresh-run delta in hand, add temporary `logger.info` instrumentation to `_import_graph_phase_nodes` (pipeline.py:1295-1308) logging `len(merged.entities)`, `len(node_records)`, and `len(node_rids)` after `upsert_nodes_batch_sync`. Re-trigger a fresh SA-2 graph_only run on an idle pool; capture the worker log. (This temp logging is removed in Task 4.)

Run: `docker logs eip-mmdpp-worker-graph-1 --since 30m 2>&1 | grep -iE "import_graph_phase|upsert_nodes|node_rids"`
Expected: identifies whether (i) merge wasn't dispatched, (ii) `merged.entities` is empty at import, (iii) `upsert_nodes_batch_sync` raised/returned 0, or (iv) it returned RIDs but they aren't queryable (DB/type mismatch).

- [ ] **Step 4: Production-config comparison run (snapshot-delta).** Set `VECTOR_ROUTER_MODE=shadow` and bundle `air_defense_v3` (recreate worker-graph + docling-graph from worktree, `-p eip-mmdpp`, force-recreate). `--snapshot pre`, run a graph_only extraction on the SA-2 doc (idle pool, wait terminal), `--snapshot post`; also run the offline merge replay with `--bundle air_defense_v3`.

Run: `python3 scripts/diagnose_lineage_commit.py --snapshot pre` … (fresh run) … `python3 scripts/diagnose_lineage_commit.py --snapshot post`; then `python3 scripts/diagnose_lineage_commit.py --run <new_shadow_run_id> --bundle air_defense_v3 --doc ddaa9e36-2854-47c3-bc94-ff38d531dafd`
Expected: records whether the commit break (post-pre delta ≈ 0 vs merge replay) + empty provenance also occur under production config (systemic) or not (narrowed-path-specific).

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
- Modify (via patch 0005, repo clone): `docker/docling-graph/repo/docling_graph/core/extractors/strategies/many_to_one.py` — pre-built-chunk metadata at :561 sets `page_numbers: []`; populate it. **Review finding Medium-2 — this route needs explicit plumbing, not just a one-liner:** `_extract_delta_from_pre_built_chunks` (:535) returns `[], document` and does NOT currently use the DoclingDocument for page resolution, and the worker→DG selected-chunk schemas carry `source_refs` but NO `page_numbers`. Two sub-changes required: (1) **schema** — add `page_numbers: list[int]` to `SelectedChunk` (`app/schemas/extraction_routing.py:156`, worker side) and `SelectedChunkInput` (`docker/docling-graph/app/schemas.py:72`, DG side) so the worker can carry per-chunk pages it already has from the ExtractionChunk index; (2) **builder** — in `_extract_delta_from_pre_built_chunks`, read `entry["page_numbers"]` into the chunk metadata (fallback: resolve `entry["source_refs"]` against the passed DoclingDocument's element `prov[].page_no`). Worker side: populate the new `page_numbers` field when building `selected_chunks` (the ExtractionChunk rows already have `page_number`).
- Modify (tracked, direct): `app/schemas/extraction_routing.py` — add `page_numbers: list[int] = []` to `SelectedChunk` (worker side, ungated forward-compat)
- Modify (tracked, direct): `docker/docling-graph/app/schemas.py` — add `page_numbers: list[int] = []` to `SelectedChunkInput` (DG receiver side)
- Modify (tracked, direct): `app/api/v1/extraction_routing.py` — populate `page_numbers` on each `SelectedChunk` from the ExtractionChunk `page_number` when building the chunk-scope response
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
    # Fixtures MUST be real pydantic models with is_entity config — _find_model_class
    # (provenance.py:69) returns only BaseModel subclasses, and synthesis skips models
    # without model_config['is_entity'] (provenance.py:133). A bare class would be
    # silently skipped → test would pass for the wrong reason (review finding Medium).
    from pydantic import BaseModel
    from typing import Optional, List
    prov = _load()

    class _Rec(BaseModel):
        model_config = {"is_entity": True, "ontology_name": "RADAR_SYSTEM",
                        "graph_id_fields": ["system_name"]}
        system_name: str
    class _Tpl(BaseModel):
        radar_systems: List[_Rec] = []

    pass_output = {"radar_systems": [{"system_name": "Fan Song"}, {"system_name": "SNR-75"}]}
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

- [ ] **Step 6: Add `page_numbers` to the selected-chunk schemas + populate worker-side** (pre-built route, Medium-2). Add `page_numbers: list[int] = []` to `SelectedChunk` (`app/schemas/extraction_routing.py`) and `SelectedChunkInput` (`docker/docling-graph/app/schemas.py`); populate `page_numbers` from `ExtractionChunk.page_number` when the chunk-scope endpoint builds `selected_chunks` (`app/api/v1/extraction_routing.py`). These ride into patch 0005's `many_to_one.py` change which reads `entry["page_numbers"]`.

- [ ] **Step 7: Generate patch 0005 for the repo-clone changes** (strategy_ops + many_to_one `entry["page_numbers"]` + document_processor self_refs/page_numbers threading). Diff each touched repo file against its clean upstream base (per the 0003/0004 method), write `docker/docling-graph/patches/0005-lineage-chunk-metadata.patch`, dry-run against a fresh clean clone.

Run: `cd /tmp && rm -rf cc && git clone <clone-origin> cc && cd cc && for p in <worktree>/docker/docling-graph/patches/*.patch; do patch -p1 --dry-run < "$p"; done`
Expected: all patches (0001-0005) apply clean.

- [ ] **Step 8: Commit** (tracked app/schema files + patch + tests).

```bash
git add docker/docling-graph/app/provenance.py docker/docling-graph/app/main.py \
  docker/docling-graph/app/schemas.py app/schemas/extraction_routing.py \
  app/api/v1/extraction_routing.py \
  docker/docling-graph/patches/0005-lineage-chunk-metadata.patch \
  tests/unit/test_provenance_page_resolution.py
git commit -m "fix(lineage): populate element_uid + page on both delta routes; carry page_numbers through selected-chunk schemas (Task 1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Strict lineage-required commit gate (C) — partition the MergedExtraction BEFORE node import

**Goal:** Reject any entity lacking resolvable lineage (no provenance row with non-empty `element_uid` AND non-null `page`) so it is NOT committed AND NOT seen by any downstream consumer (domain edges, structural edges, audit serialization). Because `derive_ontology_graph_merge` passes the SAME `merged` object to `_import_graph_phase_nodes`, `_import_graph_phase_domain_edges` (pipeline.py:7938), `_import_graph_phase_structural_edges` (7944), and `_build_provenance_envelope`/`_serialize_for_audit` (which iterates `merged.entities` and would write rejected entities with `rid=None`, pipeline.py:343/354), the gate must filter `merged.entities` itself, NOT just `node_records`.

**Design (review finding High-2):** add a gate helper that partitions `merged.entities` in place at the **top of `derive_ontology_graph_merge`, immediately after `merge_and_resolve` returns and BEFORE `_build_provenance_envelope`** (pipeline.py ~7915). Replace `merged.entities` with the lineage-complete list; collect the rejected list; if any were rejected, record a hard failure signal on the run (the strict policy — surfaced, never silent). All downstream calls then operate on the filtered `merged` automatically. Also prune `merged.edges` whose endpoints reference a rejected entity identity (so domain edges don't dangle).

**Files:**
- Modify: `app/workers/pipeline.py` — add `_partition_entities_by_lineage(merged)` helper; call it in `derive_ontology_graph_merge` right after `merge_and_resolve` (~7915), before `_build_provenance_envelope` (~7918)
- Test: `tests/unit/test_lineage_commit_gate.py`

**Acceptance Criteria:**
- [ ] After the gate, `merged.entities` contains ONLY entities with ≥1 provenance row having non-empty `element_uid` AND non-null `page`.
- [ ] Rejected entities are removed from `merged.entities` (so domain/structural/audit paths never see them) and collected into a `rejected` list logged at ERROR with count + identities.
- [ ] `merged.edges` referencing a rejected entity identity are pruned (no dangling edges).
- [ ] The run is marked with a hard lineage-rejection signal (e.g. metrics/diagnostics field + loud log) when any entity is rejected — strict, never silent.
- [ ] When all entities have lineage (the normal post-Task-1 case), the gate is a no-op pass-through.

**Verify:** `python3 -m pytest tests/unit/test_lineage_commit_gate.py -v` → PASS.

**Steps:**

- [ ] **Step 1: Write failing test.** Build a fake `MergedExtraction`-like object with two entities (one lineage-complete, one not) + an edge between them; assert the gate keeps only the lineage-complete entity, prunes the dangling edge, and records the rejection. Fixtures match the real contracts: `identity` exposes `entity_type`, `as_upsert_identity_dict()`, `identity_values_dict()` (used by `_build_node_record` pipeline.py:1283 and the logger); `provenance` is a list of rows with `.element_uid`/`.page`.

```python
# tests/unit/test_lineage_commit_gate.py
import pytest
pytestmark = pytest.mark.unit
from app.workers import pipeline

class _Prov:
    def __init__(self, element_uid, page): self.element_uid = element_uid; self.page = page
class _Ident:
    def __init__(self, n): self._n = n; self.entity_type = "RADAR_SYSTEM"
    def identity_values_dict(self): return {"system_name": self._n}
    def as_upsert_identity_dict(self): return {"system_name": self._n}
    def __hash__(self): return hash(self._n)
    def __eq__(self, o): return isinstance(o, _Ident) and o._n == self._n
class _Ent:
    def __init__(self, n, prov):
        self.identity = _Ident(n); self.properties = {}; self.confidence = 0.9; self.provenance = prov
class _Edge:
    # Match real MergedEdgeRecord field names (extraction_merge.py:366-367).
    def __init__(self, src, dst): self.from_identity = src; self.to_identity = dst
class _Merged:
    def __init__(self, ents, edges): self.entities = ents; self.edges = edges

def test_gate_filters_merged_and_prunes_edges():
    ok = _Ent("Fan Song", [_Prov("#/texts/12", 3)])      # lineage-complete
    bad = _Ent("Ghost", [_Prov("", None)])               # no lineage
    merged = _Merged([ok, bad], [_Edge(ok.identity, bad.identity)])  # edge to rejected
    rejected = pipeline._partition_entities_by_lineage(merged)
    assert [e.identity._n for e in merged.entities] == ["Fan Song"]   # only lineage-ok kept
    assert len(rejected) == 1 and rejected[0].identity._n == "Ghost"
    assert merged.edges == []                                          # dangling edge pruned

def test_gate_noop_when_all_have_lineage():
    a = _Ent("A", [_Prov("#/texts/1", 1)]); b = _Ent("B", [_Prov("#/texts/2", 2)])
    merged = _Merged([a, b], [_Edge(a.identity, b.identity)])
    rejected = pipeline._partition_entities_by_lineage(merged)
    assert rejected == [] and len(merged.entities) == 2 and len(merged.edges) == 1
```

- [ ] **Step 2: Run — fails** (`_partition_entities_by_lineage` not defined).

Run: `python3 -m pytest tests/unit/test_lineage_commit_gate.py -v`
Expected: FAIL — `AttributeError: module 'app.workers.pipeline' has no attribute '_partition_entities_by_lineage'`.

- [ ] **Step 3: Implement the gate helper** in `app/workers/pipeline.py`.

```python
def _has_resolvable_lineage(e) -> bool:
    return any(
        getattr(p, "element_uid", "") and getattr(p, "page", None) is not None
        for p in (getattr(e, "provenance", None) or [])
    )

def _partition_entities_by_lineage(merged) -> list:
    """STRICT lineage gate (spec Component 2C). Mutates `merged` in place to keep
    only lineage-complete entities; prunes edges referencing rejected identities;
    returns the rejected entity list. Runs BEFORE provenance envelope / node
    import so NO downstream consumer (domain edges, structural edges, audit
    serialization) ever sees a lineage-less entity."""
    keep, rejected = [], []
    for e in merged.entities:
        (keep if _has_resolvable_lineage(e) else rejected).append(e)
    if rejected:
        rejected_ids = {e.identity for e in rejected}
        merged.entities = keep
        # MergedEdgeRecord fields are from_identity / to_identity
        # (extraction_merge.py:366-367) — NOT source_identity/target_identity.
        merged.edges = [
            ed for ed in (getattr(merged, "edges", None) or [])
            if getattr(ed, "from_identity", None) not in rejected_ids
            and getattr(ed, "to_identity", None) not in rejected_ids
        ]
        logger.error(
            "LINEAGE_GATE: rejected %d/%d entities lacking resolvable lineage "
            "(element_uid+page) — NOT committed; identities=%r",
            len(rejected), len(keep) + len(rejected),
            [e.identity.identity_values_dict() for e in rejected][:20],
        )
    return rejected
```

Wire it into `derive_ontology_graph_merge`. The real ordering is `merge_and_resolve` → `_apply_post_merge_yield_updates` → `_write_pipeline_run_metrics` → `_build_provenance_envelope`. The gate must run **right after merge** (so yield/metrics/envelope/import all operate on filtered entities), but the rejection signal must be written **after** `_write_pipeline_run_metrics` because that function does `run.metrics = {...}` (a full REPLACE, pipeline.py:544 — review finding Medium-3) and would clobber an earlier write:

```python
        merged = merge_and_resolve(...)
        _lineage_rejected = _partition_entities_by_lineage(merged)   # STRICT gate — BEFORE yield/metrics/envelope
        _apply_post_merge_yield_updates(run_id, merged, manifest)
        _write_pipeline_run_metrics(run_id, merged, manifest)        # REPLACES run.metrics
        if _lineage_rejected:
            _record_lineage_rejection(run_id, _lineage_rejected)     # AFTER metrics; merge-updates run.metrics (Step 4)
        provenance_envelope = _build_provenance_envelope(...)
        ...
```

- [ ] **Step 4: Add the run-level hard signal with MERGE semantics.** Implement `_record_lineage_rejection(run_id, rejected)` to MERGE `lineage_rejected_count` + sample identities into the existing `run.metrics` (read-modify-write, NOT replace — `_write_pipeline_run_metrics` already replaced it just before). Pattern:

```python
def _record_lineage_rejection(run_id, rejected):
    """Surface the strict lineage rejection on run.metrics (merge, not replace —
    _write_pipeline_run_metrics ran just before and assigns run.metrics wholesale)."""
    from app.models.ingest import PipelineRun
    db = _get_db()
    try:
        run = db.get(PipelineRun, uuid.UUID(str(run_id)))
        metrics = dict(run.metrics or {})              # copy existing
        metrics["lineage_rejected_count"] = len(rejected)
        metrics["lineage_rejected_sample"] = [
            e.identity.identity_values_dict() for e in rejected
        ][:20]
        run.metrics = metrics                          # merged write
        db.commit()
    finally:
        db.close()
```

- [ ] **Step 5: Run — passes.**

Run: `python3 -m pytest tests/unit/test_lineage_commit_gate.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Commit.**

```bash
git add app/workers/pipeline.py tests/unit/test_lineage_commit_gate.py
git commit -m "feat(lineage): strict gate — partition MergedExtraction before import; prune dangling edges (Task 2)

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
- [ ] A **pre/post snapshot-delta** around the fresh SA-2 run (entity vertices carry no run_id) shows a committed-entity increase consistent with the merge-replay count (~22) — NOT measured by a global absolute count.
- [ ] Worker log for the run shows ZERO "dropping provenance row missing required fields" warnings AND zero `LINEAGE_GATE: rejected` for entities that should have lineage.
- [ ] `EXTRACTED_FROM` and/or `MENTIONED_IN` edges > 0 for the run.
- [ ] For one sample entity field value: trace it to its `element_uid` → TextChunk/ExtractionChunk → Document + page number (printed end-to-end).
- [ ] Discriminator: a known image/empty doc (e.g. `cw_radar.jpg`) run yields 0 entities legitimately with docling-graph `raw_node_count`=1 (not a silent drop) — confirm the gate did NOT reject populated entities there.

**Verify:** `python3 scripts/verify_lineage_e2e.py --run <fixed_sa2_run_id>` → all checks print PASS; the field→chunk→doc→page trace prints a concrete example.

**Steps:**

- [ ] **Step 1: Snapshot-pre, run, snapshot-post.** `python3 scripts/diagnose_lineage_commit.py --snapshot pre` (record P); run a fresh SA-2 graph_only extraction on the fixed build (idle pool), capture run_id, wait terminal; `--snapshot post` (record Q). The committed-entity delta `Q-P` is the run's contribution.
- [ ] **Step 2: Write `scripts/verify_lineage_e2e.py`** that runs the acceptance checks against ArcadeDB + postgres + docling-graph logs and prints PASS/FAIL per check plus the concrete field→chunk→document→page trace for one entity. It takes `--pre <P>` so the committed-delta check compares `Q-P` to the merge-replay count, not a global absolute.
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
