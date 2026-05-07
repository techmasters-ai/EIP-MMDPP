# Mechanism A1 — Notes for Merging `feat/per-pass-celery-fanin`

**Audience:** Whoever merges the per-pass-celery-fanin branch (or any future branch that refactors `app/services/extraction_merge.py` / `app/workers/pipeline.py`) back to `main`.

**Premise:** Mechanism A1 (table-derived identity rewrite + per-cell field overlay) landed on `main` 2026-05-06/07 across 38 commits. Spec at `docs/superpowers/specs/2026-05-06-table-identity-rewrite-and-field-overlay-design.md` (status: Approved 2026-05-07). The worker-side surface was deliberately collapsed into a 1-call orchestrator API to make this merge easy.

---

## TL;DR — what your merge needs to do

Mechanism A1's worker-side integration is **one function call** that you must place in your new merge dispatcher (likely `derive_ontology_graph_merge` or wherever Phase 1 entity-merge happens):

```python
from app.services.table_overlay import apply_table_overlay_phases

apply_table_overlay_phases(
    pass_results,                                        # dict[pass_name, PassResult]
    ontology=ontology,
    document_id=document_id,
    canonicalize_fn=canonicalize_cross_pass_identities,  # the existing canonical fn
)
```

Place this **between** `pass_results` is fully assembled **and** the entity-merge phase begins. That's it. Everything else (env-flag check, kill-switch authority over cached overlays, Phase 0 alias rewrite, Phase 0.5 field overlay, log-line emission) is encapsulated inside the orchestrator.

---

## What A1 added to `main`

### Files added
- `app/services/table_overlay.py` — **canonical worker-side home**. Contains:
  - Wire types: `TableOverlay`, `TableFact`, `CrossEntityHint` (mirror of parser-side `docker/docling-graph/app/schemas.py`)
  - Stats: `RewriteStats`, `OverlayStats`
  - Module-level env helper: `is_overlay_enabled_worker()`
  - Per-pass overlay extraction: `extract_doc_overlay(pass_results)`
  - Apply functions: `apply_identity_rewrite`, `apply_field_overlay`
  - **Orchestrator (THE INTEGRATION SURFACE)**: `apply_table_overlay_phases(...)`

- Parser-side files in `docker/docling-graph/`:
  - `app/_table_facts.py` — `extract_table_overlay()` + 4-of-4 strict qualification gate + helpers
  - `app/_alias_map.py` — `MISSILE_IDENTITY_LABELS`, `RADAR_IDENTITY_LABELS`, `CROSS_ENTITY_REF_PATTERNS`, `CANONICAL_PRIORITY` constants
  - `app/schemas.py` — Pydantic mirrors + `ExtractPassResponse.table_overlay` field
  - `app/main.py` — `/extract-pass` handler wires `extract_table_overlay()` between sanitize and LLM call

- Tests:
  - `tests/unit/test_table_overlay_worker.py` (10 tests)
  - `tests/unit/test_extraction_merge_table_overlay.py` (4 tests including the kill-switch defense-in-depth)
  - `tests/integration/test_table_overlay_end_to_end.py` (1 e2e synthetic test)
  - `docker/docling-graph/tests/test_alias_map_overlay_constants.py` (4 drift guards)
  - `docker/docling-graph/tests/test_table_overlay_extract.py` (13 helper unit tests)
  - `docker/docling-graph/tests/test_table_overlay_qualification.py` (5 starvation tests)
  - `docker/docling-graph/tests/test_table_overlay_schemas.py` (7 wire-type tests including parser↔worker drift guard)
  - `docker/docling-graph/tests/test_main_table_overlay_integration.py` (2 endpoint tests)

### Files modified on `main`

- **`app/services/extraction_merge.py`:**
  - Line ~23: `from app.services.table_overlay import TableOverlay` import
  - Line ~202: `PassResult.table_overlay: TableOverlay | None = None` field on the dataclass
  - Line ~1015: `canonicalize_cross_pass_identities` accepts new keyword-only `table_alias_map_by_entity_type: dict[str, dict[str, str]] | None = None` argument; new Phase 0 alias-rewrite block runs BEFORE existing token-overlap pass
  - Line ~1196 (inside `merge_and_resolve`): single `apply_table_overlay_phases(...)` call — the integration surface

- **`app/workers/pipeline.py`:**
  - Lines ~2615-2625: `_parse_pass_response` reads `response_json["table_overlay"]` via `TableOverlay.model_validate(...)`, with try/except WARN-and-None on malformed
  - Line ~2639: `PassResult(...)` constructor receives `table_overlay=table_overlay_obj` kwarg

- **`docker-compose.yml`:** `DOCLING_GRAPH_TABLE_OVERLAY_ENABLED` env var on docling-graph + worker + worker-graph services. Default `true`. The worker-side check is **authoritative over cached overlays** per spec §4.3.

---

## Where the per-pass-celery-fanin merge will collide

Both branches modify `app/services/extraction_merge.py` and `app/workers/pipeline.py`. The collisions:

### `app/services/extraction_merge.py`

| Line range (main) | A1 added | Per-pass-fanin likely |
|---|---|---|
| ~23 | `from app.services.table_overlay import TableOverlay` import | might add new imports nearby |
| ~202 | `PassResult.table_overlay: TableOverlay \| None = None` field | might restructure PassResult to load from DB rows |
| ~1015 | `canonicalize_cross_pass_identities` adds keyword `table_alias_map_by_entity_type` arg + Phase 0 alias-rewrite block | unchanged signature OR refactor to module |
| ~1133 | Stub doc-comment pointing to `extract_doc_overlay` (originally `_extract_doc_overlay` lived here, promoted to `table_overlay.py`) | unchanged |
| ~1196 (inside `merge_and_resolve`) | Single `apply_table_overlay_phases(...)` call | replaces `merge_and_resolve` body with `derive_ontology_graph_merge` dispatcher |

**Resolution:** Take A1's PassResult field + canonicalize signature change verbatim. For the orchestrator call: drop it into your new `derive_ontology_graph_merge` between fan-in PassResults assembly and the entity-merge phase. The call site is THE thing that absolutely must survive the merge — it's the load-bearing integration point.

### `app/workers/pipeline.py`

| Line range (main) | A1 added | Per-pass-fanin likely |
|---|---|---|
| ~2615 | `_parse_pass_response` overlay-parse block | restructured into per-pass response handling, possibly different file |
| ~2639 | `PassResult(...)` constructor receives `table_overlay=…` kwarg | constructor call moved or replaced with DB-row hydration |

**Resolution:** Wherever the per-pass-fanin branch parses an `/extract-pass` response into a PassResult-like object, ensure it reads `response_json.get("table_overlay")` and passes it through. Drift guard: `docker/docling-graph/tests/test_table_overlay_schemas.py::test_parser_and_worker_table_overlay_classes_round_trip` will catch wire-type-shape drift.

If your branch persists pass outputs to the DB (`pipeline_pass_outputs.metadata_json`), include `table_overlay` in the persisted JSON so it round-trips on retry. The orchestrator's `extract_doc_overlay()` reads `getattr(pr, "table_overlay", None)` — so as long as the rehydrated PassResult carries the field, the orchestrator is happy.

### `docker-compose.yml`

A1 added `DOCLING_GRAPH_TABLE_OVERLAY_ENABLED` to docling-graph + worker + worker-graph environment blocks. Per-pass-fanin likely doesn't touch these, but if it adds new worker services, those should also receive the env var.

---

## Spec §4.3 — kill-switch invariant after merge

The worker-side env check (`is_overlay_enabled_worker()`) is **authoritative over cached overlays**. After per-pass-fanin lands and `pipeline_pass_outputs.metadata_json` becomes the source of truth for PassResult.table_overlay, this becomes the production scenario:

1. Operator sets `DOCLING_GRAPH_TABLE_OVERLAY_ENABLED=false`
2. A retry/reconciler picks up an old `pipeline_pass_outputs` row with `metadata_json.table_overlay` populated from yesterday's run
3. The PassResult is rehydrated WITH a populated `table_overlay`
4. `apply_table_overlay_phases()` calls `is_overlay_enabled_worker()` → False → skips both Phase 0 and Phase 0.5
5. Logs `TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER doc_id=… pass_count=… cached_overlay_present=…`

This invariant must survive the merge. The unit test `test_kill_switch_worker_side_overrides_cached_overlay` asserts it via a stub PassResult that carries a populated cached overlay; it should still pass after the merge as long as `apply_table_overlay_phases()` is called from the new dispatcher.

---

## Acceptance tests post-merge

After merging both branches, run these tests in order:

1. **Worker-side unit tests:**
   ```
   pytest tests/unit/test_table_overlay_worker.py -v
   ```
   Expected: 10 PASSED.

2. **Worker-side integration tests:**
   ```
   pytest tests/unit/test_extraction_merge_table_overlay.py -v
   ```
   Expected: 4 PASSED. **Critical:** `test_kill_switch_worker_side_overrides_cached_overlay` must still pass — that's the spec §4.3 invariant.

3. **End-to-end synthetic test:**
   ```
   pytest tests/integration/test_table_overlay_end_to_end.py -v
   ```
   Expected: 1 PASSED. Asserts alias collapse + propulsion override + `FIELD_OVERLAY_OVERRIDE` log emission.

4. **Parser-side smoke (if you also rebuilt docling-graph):**
   ```
   docker compose exec docling-graph pytest /app/tests/test_table_overlay_*.py /app/tests/test_alias_map_overlay_constants.py -v
   ```
   Expected: ~30 tests pass.

5. **Parser smoke against real SA-2 doc** (proves parser-side end-to-end on real data):
   ```
   docker exec eip-mmdpp-docling-graph python3 -c "
   import json, sys
   sys.path.insert(0, '/app/app')
   import _table_facts as tf
   doc = json.load(open('/tmp/sa2_doc.json'))  # see Task 0 baseline notes
   overlay, stats = tf.extract_table_overlay(doc)
   assert stats['tables_processed'] == 1
   assert 'MISSILE_SYSTEM' in overlay.alias_map_by_entity_type
   assert len(overlay.facts) > 40
   print('OK')
   "
   ```

6. **Live full ingest** (the deferred Task 12 step):
   - Re-ingest the SA-2 Guideline PDF (doc_id `8275cab1-98d1-41e8-b0cb-fed7409fced1`) through the new per-pass dispatcher
   - Watch worker logs for: `IDENTITY_REWRITE rewrites=… …`, `TABLE_OVERLAY_APPLIED doc_id=… …`, ~15 `FIELD_OVERLAY_OVERRIDE pass=missile_propulsion …` lines
   - Query ArcadeDB for the merged `1D` MISSILE_SYSTEM vertex; verify `booster_mass_kg=1135.0`, `sustain_mass_kg=1028.0` (post-overlay correct values)
   - Compare against `/tmp/baseline_2026-05-06_pre_overlay/expected_overrides.md` for the predicted 15 override log lines

---

## Common merge mistakes to avoid

1. **Dropping the `apply_table_overlay_phases()` call.** That's THE integration. If your new merge dispatcher doesn't invoke it, Mechanism A1 is silently disabled — no overlay logs will fire and the SA-2 propulsion fix won't happen. The acceptance tests in §3 will catch this.

2. **Changing the order of phases.** Phase 0 (alias rewrite) MUST run BEFORE Phase 0.5 (field overlay) MUST run BEFORE Phase 1 (entity merge). The orchestrator handles 0 and 0.5 in correct order; you just need to place the orchestrator call before whatever Phase 1 looks like in your new dispatcher.

3. **Persisting PassResult to DB without `table_overlay`.** If `pipeline_pass_outputs.metadata_json` doesn't include the parser-emitted overlay payload, retries lose it. The retry path then runs without overlay context and propulsion fixes regress.

4. **Forgetting the env var on new worker services.** If per-pass-fanin spins up new worker containers (e.g., per-pass workers with a different name), they need `DOCLING_GRAPH_TABLE_OVERLAY_ENABLED` set on their env block.

5. **Dropping the parser-side `table_overlay` field on the response.** The `/extract-pass` handler in `docker/docling-graph/app/main.py` collapses empty overlays to `None` before returning. If your branch reshapes the response, preserve this behavior — the worker-side `extract_doc_overlay` checks for non-empty.

---

## A1's parser-side smoke results (reference baseline)

Real SA-2 Guideline PDF (doc `8275cab1-98d1-41e8-b0cb-fed7409fced1`), table 0 (22×12 column-major variants table), measured 2026-05-07 after unit-fix commit `8d91986`:

- `tables_processed=1`, `tables_skipped_other=1` (table 1 correctly skipped as non-column-major)
- `alias_map_by_entity_type.MISSILE_SYSTEM`: 21 entries spanning 9 listed variants (1D, 13D, 13DM, 13DA, 13DAM, 20D, 20DP, 20DSU, 5Ya23) plus 15D
- 53 facts emitted across 4 missile passes
- 10 cross-entity hints (Fan Song variants → RADAR_SYSTEM)
- **43/43 GT-field facts correct (100%)** when scored against ground-truth values

These numbers are what a successful post-merge live smoke should approximately reproduce.

---

## Contact-the-spec-author cases

Surface these to the spec maintainer (likely whoever last touched the spec doc) before merging IF you find:

1. The per-pass-fanin DB persistence path makes `_extract_doc_overlay` semantics ambiguous (e.g., what if two pass rows on the same doc have divergent overlays from re-extraction?). Spec §4.4 covers the in-memory case; the DB case may need a tiebreaker rule.
2. The new dispatcher needs to preserve `OverlayStats` post-merge for telemetry. Currently `apply_table_overlay_phases()` returns `OverlayStats | None` but the call site in `merge_and_resolve` discards it. If you want pipeline-run-level overlay metrics, capture the return value and persist alongside the merge stats.
3. Cross-entity hints (`CrossEntityHint`) are collected but NOT applied as edges in v1 (spec §3 non-goals). If the per-pass-fanin branch surfaces a clean place to fire those into the relationship pass, that's the v2 follow-up.

---

End of merge notes. The whole point is: **the merge surface is one function call.** Everything else is implementation detail behind that call.
