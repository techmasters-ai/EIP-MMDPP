# dataset_v2 capture audit (pre-export, 2026-06-13)

Read-only audit of all re-collection runs to confirm the Phase-3 dataset is
fully capturable BEFORE export/eval. Run while Engagement retry (66a2afef) was
still in flight.

Runs (terminal=COMPLETE unless noted): museum 003abaec, SNR-75 ac87fa3d,
V-75 5380268e, Images_Demo 6b63d765, Dvina 534b8364, SA2_SR71 39adb2f1,
SA2_RU(doc8) 4f53397c, Engagement(retry, IN FLIGHT) 66a2afef. FAILED de4d0c11
(original Engagement) excluded.

## GREEN — core capture is sound on all completed runs

- **score_components_all present** for every one of the 9 field-group passes on
  every completed run, each with the full **23 COMPONENT_KEYS** (no missing
  keys anywhere). Row counts = chunk counts per doc.
- **ExtractionChunk rows survive run completion** in ArcadeDB (graph_only does
  NOT delete them on success — the FAILED-run cleanup that deleted 175 rows was
  the failure path only). `chunk_text` and `page_number` populated on every row
  → export + lineage intact. `is_table` column non-NULL on every row.
- **Label inputs present**: `extract_pass_response_json` non-null on every pass
  → value-grounded label is computable.
- **No degraded extractions**: every pass `execution_status=COMPLETE`; the
  BATCH_HARD_TIMEOUT ceiling bug fired on doc 8 (SA2_RU) but did NOT prevent
  completion or empty the output (436 entities). Engagement retry (post ceiling
  fix) shows ZERO BATCH_HARD_TIMEOUT — the fix holds.
- **Code identical across all runs**: workers up since 06-11 01:41 (no restart);
  docling-graph image unchanged since 06-10 20:41 (the 06-12 13:12 recreate only
  swapped an env var). So feature differences across docs are NOT deploy skew.
- **New features fire where document content warrants**: is_table + table_gate
  fire on the table-rich docs (SA2_RU 7 table chunks, Engagement 8);
  unit_gate fires per-pass on numeric content (SA2_SR71 missile_propulsion
  31/42, SA2_RU missile_propulsion 57/58). SA2_SR71 has zero `#/tables/` refs →
  is_table=0 there is genuine, not a gap.

## AMBER — one real capture gap: fallback-path docs miss gate-union + cosines

The three SMALL docs — **SNR-75 (7 chunks), V-75 (6), Dvina (7)** — show
`max_field_cosine=0` and `unit_gate=0` across ALL passes/chunks. Root cause
(confirmed): docs with ≤ `field_query_top_k` (=8) chunks trip the E2 fallback
ladder (`fallback_level=degraded` in their router diagnostics; museum and the
larger docs show `fallback_level=none`). The fallback rungs rebuild the pool via
`build_pool_from_multi_channel_state`, which:
- passes `row_cosines=None` (`extraction_chunk_search.py:941-944`, "Task 7…
  Task 18 will wire it") → `max_field_cosine`/`mean_top3_field_cosine` stay 0;
- does NOT run the G1/G2 gate union (Task 7 left fallback-path gating to Task 18)
  → `unit_gate`/`table_gate` stay 0 and `gate_flags` empty.

This is a **documented Task-18 deferral**, surfacing in the capture because three
of the eight docs are small enough to hit fallback.

### Impact — bounded, and the recall floor is still SOUND BY CONSTRUCTION

Positives by doc (v1_relabel labels): Engagement 23, SA2_RU(=SA-2 RU) 5,
SA2_SR71 3, museum 1, Images_Demo 1, **V-75 1, Dvina 1**, SNR-75 0.
- The fallback-affected positives are **V-75 (1) and Dvina (1)** — SNR-75 has 0.
- **The recall floor itself is not at risk**: the gate predicate is a pure
  function of (chunk_text, pass unit-signature) and is a strict superset of the
  value-grounded label BY CONSTRUCTION (unit-tested: every label-positive chunk
  contains a numeric+unit value → contains a digit + a signature unit token →
  passes the gate). The fallback path merely failed to RECORD the flag at
  runtime; the property holds.
- Consequence: `check_gate_coverage` as written reads the captured `unit_gate`
  flag and would **false-fail** on the V-75/Dvina positives. It must recompute
  the gate from `chunk_text` to be construction-faithful — OR the fallback path
  must record the flag.
- `max_field_cosine=0` on these 3 docs is a genuine feature-sparsity region
  (3 docs, 2 positives) — minor for LODO calibration; cannot be backfilled
  offline (needs the field-query embeddings at capture time).

## Remediation options (decision pending; execute AFTER Engagement finishes)

**A. Pull the Task-18 fallback gate-union + row_cosines forward (recommended).**
Wire `build_pool_from_multi_channel_state` to run the gate scan/union and accept
row_cosines, redeploy, re-run only SNR-75/V-75/Dvina (6-7 chunks each, ~minutes
total). Result: uniform dataset, recall floor recorded in capture AND enforced
in production, `max_field_cosine` populated everywhere. Cost: one code change +
redeploy + ~5 min re-collection. NOTE: must NOT touch worker code while the
Engagement run is in flight (would affect its remaining passes).

**B. Proceed as-is + make `check_gate_coverage` recompute the gate offline.**
Faster: no re-run. The recall floor is proven by construction via offline
recompute; accept `max_field_cosine`=0 on 3 small docs as a known region; fix
the fallback path in Task 18 proper before any production flip. Leaves an
asterisk on 3 docs' feature completeness.

Both are sound. A removes the asterisk for trivial cost; B is faster and the
floor is still provably correct. The production fallback-path fix (Task 18) is
REQUIRED under either option before `narrow_only` is enabled.

## Also noted (minor)

- `unit_gate_total` / `unit_gate_added` aggregate diagnostics are NULL even on
  primary-path runs (museum): the per-chunk `unit_gate` component IS written
  (gate ran), but the MultiChannelDiagnostics aggregate counts aren't surfaced
  into the persisted router diagnostics dict. Cosmetic — per-chunk flags are
  what the eval uses; fix the surfacing in Task 18.
