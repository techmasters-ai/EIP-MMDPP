# Walltime Root-Cause Findings + Follow-ups

**Status:** documented 2026-05-29. These are walltime / LLM-infra issues found while validating the wire-off recall fix. They are **out of scope** for the retrieval/routing plan (`docs/superpowers/plans/2026-05-28-schema-wide-retrieval-routing-upgrade.md`) and should be addressed as a **separate workstream after that plan executes.**

## Trigger / evidence

Run `f4e03e7e` (SA-2 `ddaa9e36`, `air_defense_v3_merged_v1`, top_k=15, wire-OFF kill-switch, COMPLETE, 03:17→10:45 2026-05-29) took **447.8 min** vs the `7d46c487` baseline **228.5 min** (~2×). A systematic per-batch decomposition (from `ingest.pipeline_pass_outputs.diagnostics_json.batch_timings`) shows the slowdown is **NOT chunk count and NOT chunk size.**

### Per-pass batch decomposition (f4e03e7e vs baseline 7d46c487)

| Pass | run | chunks | wall | batches >1500s | max batch | avg batch |
|---|---|---|---|---|---|---|
| radar_identity | base | 60 | 111m | 0 | 1015s | 123s |
| radar_identity | f4e0 | 70 | 160m | **2** | **2131s** | 163s |
| radar_power_rf | base | 13 | 13.5m | 0 | 200s | 77s |
| radar_power_rf | f4e0 | 22 | 30.9m | 0 | 242s | 99s |
| missile_identity | base | 60 | 200m | 0 | 1064s | 207s |
| missile_identity | f4e0 | 70 | 410m | **4** | **2061s** | 289s |
| missile_kinematics | base | 10 | 23m | 0 | 332s | 162s |
| missile_kinematics | f4e0 | 20 | 180m | **3** | **3675s** | 523s |
| system_links | base | 19 | 28m | 0 | 242s | 131s |
| system_links | f4e0 | 20 | 37m | 0 | 279s | 131s |

`missile_kinematics` batch times, sorted (s): `3675 2119 2012 | 372 265 248 215 208 183 151 137 132 130 123 118 109 104 68 52 48` — **3 of 20 batches = ~75% of the 180m pass wall.**

Chunk sizes (small — not a factor): radar_power_rf ~270 tok/chunk, missile_kinematics ~310, identity passes ~430 (cap 512).

## Root cause (ranked)

1. **[DOMINANT] Catastrophic stalled batches — the 1800s stream-wall-timeout + retry under intermittent pool contention.** Baseline had **0** batches >1500s; f4e03e7e had **9** (max 3675s). When the `gemma4:31b` pool is contended, a call's generation stretches past `OLLAMA_STREAM_WALL_TIMEOUT` (1800s), gets killed, and **retries into the same contention** → 2000-3675s per batch. The within-pass variance (48s → 3675s for same-size chunks) is the fingerprint of *intermittent contention*, not content complexity.
2. **[SECONDARY, ~2×] Wire-off re-chunk + table-synth raises chunk count** +17-100% (clean datapoint: `radar_power_rf` 13→22 chunks, 13.5→30.9m, 0 stalls). This is the *honest* wire-off cost and is exactly what Phase C targets (fewer/better chunks at top_k=15).
3. **[WALL FLOOR] Identity passes dominate absolute wall.** `missile_identity` 410m + `radar_identity` 160m. They are `document_only` (full-doc, RUN_FULL, no `retrieval:` block) → never touched by routing. The single largest pass (missile_identity) is untouched by the entire retrieval plan.

### Structural amplifiers
- **Wall ≈ Σ(batch times)** in both runs (missile_kin Σ=174m, wall=180m) → batches run **effectively serial** despite `parallel_workers=2`, because the GPU serializes concurrent `gemma4:31b` generations. A stalled batch therefore cannot be hidden behind others; it adds directly to the pass wall.
- **~24 MB request payload per pass** — the full `docling_document_json` is POSTed on every `/extract-pass` call (wire-off requires it). Serialization/network overhead; minor vs LLM time but non-zero.

## Follow-up actions (address AFTER the retrieval plan)

1. **Timeout policy (highest leverage).** The 1800s `OLLAMA_STREAM_WALL_TIMEOUT` kill+retry is counterproductive under contention — it converts a slow-but-progressing call into wasted work plus a retry that re-fights the same contention. Options: (a) detect "still progressing" (tokens arriving) vs truly stalled before killing; (b) adaptive/longer timeout under known contention; (c) cap/back off retries. See `docker/docling-graph/app/config_builder.py:118` (`docling_graph_llm_timeout=1800`) + `docker/docling-graph/app/ollama_pool_client.py`.
2. **Pool contention / scheduling.** The run self-contends (concurrent passes + co-tenant load on `10.0.1.121`/`10.0.1.109`). Options: serialize passes (lower `pass_concurrency_per_document`), cap per-doc pool usage, or dedicate gemma4 capacity for extraction. Always measure wall on an **uncontended** pool.
3. **GPU parallelism mismatch.** `parallel_workers=2` yields ~no speedup because Ollama serializes generations on the GPU (`OLLAMA_NUM_PARALLEL`). Investigate true concurrency (more GPU parallel slots) or drop the parallel_workers coordination overhead and accept serial.
4. **Identity-pass walltime (the wall floor).** `missile_identity`/`radar_identity` are full-doc and dominate total wall. Narrow or subset-schema them (Phase F could reach them) or parallelize — a separate workstream from field-group routing.
5. **Request payload.** Trim the ~24 MB `docling_document_json` per pass if it proves to add meaningful latency.

## Measurement caveat (load-bearing)

Any walltime comparison — **including the retrieval plan's Acceptance #3** ("no wall regression beyond 120% of baseline") — is **only valid on an uncontended LLM pool**. On a contended pool the 1800s-timeout stalls inflate wall 2-4× as an infra artifact unrelated to the path under test. **Re-measure on an idle pool before drawing any wall conclusion.**

## References
- Diagnostics source: `ingest.pipeline_pass_outputs.diagnostics_json` (`batch_timings`, `chunk_count`, `batch_count`, `pass_wall_ms`).
- Recall result for the same run (validated wire-off recovers/exceeds recall): radar_id 24, radar_rf 29, missile_id 49, missile_kin 35, system_links 85 rels — see the comparison in the session / `docs/sa2_extraction_runs.md`.
- Related plan: `docs/superpowers/plans/2026-05-28-schema-wide-retrieval-routing-upgrade.md` — "Walltime scope caveat" + "Open Risks".
