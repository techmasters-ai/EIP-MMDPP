# SA-2 extraction run log

Append-only ledger of every SA-2 reference-doc extraction run from v9 forward.
Each entry records: commit SHA, config, per-pass results, verdict, and the
exact revert command to restore that configuration.

**Doc:** `78673393-639b-4fde-9bda-9e7bfd43ccda` (SA-2 / S-75 Guideline reference)
**Model:** `gemma4:31b` via Ollama (`http://10.0.1.121:11434` + `http://10.0.1.109:11434`)

When a regression is detected, locate the most recent ★STRONGEST★ entry and
follow its `revert` instructions. The log is in chronological order; the most
recent entry is at the bottom.

---

## Run V9 — table-aware chunking v9 quality + stability bundle

- **Date:** ~2026-05-14 (per memory)
- **Commit:** `666084b feat(table-norm): v9 — quality + stability bundle for table-aware extraction`
- **Branch:** main (merged to main via `7d09a90`)
- **Captured fixtures:** `tests/fixtures/sa2/78673393-...-{pass}_response_v9.json` for all 5 passes; summary at `..._extraction_counts_phase2_v9.json`
- **Wall-clock:** 5h 42m
- **Config:**
  - `DOCLING_GRAPH_CHUNK_MAX_TOKENS=512`
  - `DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE=512`
  - `DOCLING_GRAPH_LLM_MAX_TOKENS=16384`
  - `DOCLING_GRAPH_LLM_TRUNCATION_RETRY_MAX_TOKENS=65536`
  - `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=true`
  - `DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN=true`
  - Synthesis appends synth chunks to `body.children`; raw `#/tables/N` refs NOT removed → both representations walked by chunker
  - System prompt: strict Unit Policy ("Apply mechanical conversion ONLY when value AND unit are both explicit")
  - Schema field descriptions: 33 instances of "Emit only when source states value AND unit"
  - `min_altitude_km` / `max_launch_angle_deg`: hard-cleared by `evidence_gate.py:742-748`
  - `_alias_map.py`: no abbreviated `Min Alt` / `Max Alt` aliases
- **Per-pass results:**

| pass | entities | total fills | avg_fill | relationships |
|------|---------:|------------:|---------:|--------------:|
| `radar_identity` | 24 | 26 | 1.08 | — |
| `radar_power_rf` | 42 | 25 | 0.60 | — |
| `missile_identity` | 44 | 62 | 1.41 | — |
| `missile_kinematics` | 40 | 20 | 0.50 | — |
| `system_links` | — | — | — | 30 |
| **TOTAL** | **150 ent** | **133 fills** | — | **30 rel** |

- **Verdict:** ★STRONGEST★ on identity/RF/relationship recall. Weak on kinematics (20 fills, `min_altitude_km=0/40`, `max_launch_angle_deg=0/40`).
- **Revert command:**
  ```bash
  git checkout 666084b -- .
  # OR (preserves later docs/test fixtures):
  git revert <commits-after-666084b-affecting-extraction>
  ```

---

## Run V9-POST-EDIT — UNIT_HINT + relaxed Unit Policy + schema gate removal

- **Date:** 2026-05-15 (this session)
- **Commit:** `d48756c feat(extraction): unblock kinematic numerics — relax Unit Policy + inject SI unit hint`
- **Branch:** main
- **Captured fixtures:** `tmp/claude_vs_gemma4/missile_kinematics_response_post_edit.json` (kinematics-only run)
- **Wall-clock:** ~1h 50m (kinematics pass only)
- **Changes vs V9:**
  - Macro Unit Policy in `prompt_rules.py`: relaxed — now permits unit assumption from preamble/column-label
  - 33 schema field descriptions: "Emit only when source states value AND unit" sentence removed
  - `render_graph.py`: UNIT_HINT preamble injected at top of every synth table block ("UNITS: Numeric values in this block are in SI base units...")
  - `BATCH_HARD_TIMEOUT_SECONDS`: bumped 3600 → 10800
  - Synth chunks STILL appended (not replaced); raw `#/tables/N` refs STILL not removed
  - `min_altitude_km` still hard-cleared
  - Alias map still lacks `Min Alt` / `Max Alt`
- **Per-pass results:** (kinematics only; other passes not run in this session)

| pass | entities | total fills | avg_fill | notes |
|------|---------:|------------:|---------:|-------|
| `missile_kinematics` | 40 | 31 | 0.78/5 | +11 fills vs V9, but `min_altitude_km=0/40` unchanged |

- **Verdict:** Partial improvement. Confirms the relaxed Unit Policy + UNIT_HINT do help once synth blocks reach the LLM. But the synth blocks were still being out-competed by raw flat in the chunker output — the gain came from prompt-side changes only.
- **Revert:** `git checkout d48756c -- .`

---

## Run V9-POST-FIX-GLOBAL — global raw-ref drop + Min Alt alias

- **Date:** 2026-05-15 → 2026-05-16 overnight (this session)
- **Commit:** `d48756c` + uncommitted edits (`_pipeline_hooks.py` adds `_drop_raw_table_refs_from_body_children`; `main.py` calls it unconditionally when suppress is on)
- **Captured fixtures:**
  - `tests/fixtures/sa2/78673393-..._{pass}_response.json` for all 5 passes
  - `tests/fixtures/sa2/78673393-..._extraction_counts_today.json`
- **Wall-clock:** 10h 03m (all 5 passes)
- **Changes vs V9-POST-EDIT:**
  - New helper `_drop_raw_table_refs_from_body_children` in `_pipeline_hooks.py` removes raw `#/tables/N` refs after synth append
  - `main.py`: calls drop helper unconditionally for all passes when suppress is on (GLOBAL — this is the regression)
  - `Min Alt` / `Max Alt` aliases added to schema and prompt_rules.py rule 12b (not to `_alias_map.py`)
  - `min_altitude_km` STILL hard-cleared by `evidence_gate.py:742`
- **Per-pass results vs V9 baseline:**

| pass | V9 ent | new ent | Δ | V9 fills | new fills | Δ |
|------|-------:|--------:|---:|---------:|----------:|---|
| `radar_identity` | 24 | 28 | +4 (+17%) | 26 | 23 | −3 (−12%) |
| `radar_power_rf` | 42 | 36 | −6 (−14%) | 25 | 23 | −2 (−8%) |
| `missile_identity` | 44 | 35 | −9 (−20%) | 62 | 45 | −17 (−27%) |
| `missile_kinematics` | 40 | 42 | +2 (+5%) | 20 | 35 | **+15 (+75%)** |
| `system_links` | 30 rel | 17 rel | −13 (−43%) | — | — | — |
| **TOTAL ent** | **150** | **141** | **−9** | — | — | — |
| **TOTAL rel** | **30** | **17** | **−13** | — | — | — |

- **Verdict:** REGRESSED. Kinematics win (+75% fills) didn't offset losses on identity/prose passes and the cascade hit to system_links (−43%). User constraint "do not regress performance" violated.
- **Min altitude diagnosis:** Still 0/42. Even with synth blocks reaching LLM, `evidence_gate.py:742` was hard-clearing the value post-extraction.
- **Revert:** revert the global drop call in `main.py:675`, return to V9-POST-EDIT state. Done in next run.

---

## Run V9-POST-FIX-GATED — Phase B kinematics-only spike (per-table relevance)

- **Date:** 2026-05-16 (this session)
- **Status:** Phase B complete; Phase C (regression spot-checks) next
- **Commit:** unstaged (Steps 1-4 + tests)
- **Captured fixtures:** `tmp/claude_vs_gemma4/missile_kinematics_response_post_edit.json`
- **Wall-clock (kinematics pass only):** 4219.2s = 70m (faster than prior 81m despite larger prompts)
- **Changes vs V9-POST-FIX-GLOBAL:**
  - **Step 1+2 (per-pass + per-table gate)**: `_drop_raw_table_refs_from_body_children` retired in favor of per-table `is_table_relevant_for_pass(pass_name, nt)` check in `_pipeline_hooks.py`. Eligible passes: missile_kinematics/airframe/speed_timing/propulsion + radar_antenna/timing/modulation/power_rf. Each table independently checked against `PASS_TABLE_ROW_ALIASES` — only relevant tables get synth-only in-place substitution; non-relevant tables keep raw + get synth appended-to-end.
  - **Step 2 (in-place substitution)**: New `_replace_raw_table_refs_in_body_children` preserves document order — synth refs install at the same body.children position as the original raw ref.
  - **Step 3 (min_altitude_km unblocked)**: removed from unconditional-null branch in `evidence_gate.py:742`; added to mechanical-override loop; `_mechanically_supported_missile_fields` extended to parse `Min Alt:` / `Max Alt:` / `Min Range:` / `Max Range:` patterns with SI-base unit assumption (metres → km).
  - **Step 4 (alias map)**: `Min Alt`, `Max Alt`, plus `*_km`/`*_m` variants added to `_alias_map.py` (drift guard satisfied by §12b prose update earlier).
- **Per-pass results (kinematics only):**

| field | v9 baseline | Phase B | Δ |
|-------|------------:|--------:|---:|
| `min_intercept_km` | 4 | 12 | +8 |
| `max_intercept_km` | 10 | 13 | +3 |
| **`min_altitude_km`** | **0** | **11** | **+11 (UNBLOCKED)** |
| `max_altitude_km` | 6 | 13 | +7 |
| `max_launch_angle_deg` | 0 | 0 | tied (not in source) |
| **TOTAL FILLS** | **20** | **49** | **+29 (+145%)** |
| entities | 40 | 43 | +3 |
| avg_fill/5 | 0.50 | 1.14 | +0.64 (+128%) |

- **Diagnostics:** 0 TRUNCATION_PERSISTS events, 0 BATCH_HARD_TIMEOUT events. Per-table relevance gate fired correctly on startup: SA-2 has 2 normalized tables; 1 (missile-spec) → synth-only in-place, 1 (Fan Song / radar) → append-to-end.
- **Verdict:** ★STRONGEST★ on kinematics. Other passes not yet measured under this design.
- **Revert if regression on other passes:** drop entries from `SYNTH_ELIGIBLE_PASSES` in `_pipeline_hooks.py` and/or add entries to `RAW_ONLY_PASSES`; per-table gating means the helper is a no-op for tables that don't match the pass.

---

## (PENDING) Run V9-POST-FIX-GATED — Phase C+D regression check + full 5-pass

- Phase C: targeted missile_identity + radar_power_rf single-pass runs (~1.5h each) to confirm no recall regression
- Phase D: full 5-pass run on SA-2 only after Phase C passes

---

## Run OPTIONS-A+C+REVIEW-FIXES — Steps 1-3 from outside-reviewer handoff

- **Date:** 2026-05-16 20:54 → 2026-05-17 02:39 (local)
- **Commit:** `d9647e3 feat(extraction): restore system_links + generalize table relevance + production-shape deterministic min_altitude_km`
- **Wall-clock:** 5h 45m
- **Captured fixtures:** `tests/fixtures/sa2/78673393-..._{pass}_response.json` for all 5 passes; summary at `..._extraction_counts_today.json`
- **Full handoff doc:** `docs/sa2_run_post_review_handoff.md` (stats + analysis + 5 ranked fixes)
- **Changes vs OPTIONS-A+C-FULL:**
  - Type-segregated `_resolve_ref` + `_build_upstream_name_map_by_type` for system_links cross-entity-hint resolution (prevents cross-type leak — fixed via outside review)
  - Entity-type qualification on `is_table_relevant_for_pass` (row labels AND identity context AND caption-keyword scoping; logistics/comms/platform false positives blocked)
  - Production-shape `_mechanically_supported_missile_fields` (entity-scoped via `_extract_synth_block_for_entity`; unit-evidence gated via `_evidence_has_si_unit_hint`)
  - Narrowed caption hints (dropped `"sensor"`, `"weapon"` per review)
- **Per-pass results vs v9:**

| pass | v9 ent | new ent | Δent | v9 fills | new fills | Δfills | v9 avg | new avg | truncations | hard-timeouts |
|------|------:|--------:|-----:|---------:|----------:|-------:|-------:|--------:|------:|------:|
| `radar_identity` | 24 | **26** | **+2** | 26 | **32** | **+6** | 1.08 | **1.23** | 0 | 0 |
| `radar_power_rf` | 42 | 42 | tied | 25 | 24 | −1 | 0.60 | 0.57 | 0 | 0 |
| `missile_identity` | 44 | 43 | −1 | 62 | 61 | −1 | 1.41 | 1.42 | 0 | 0 |
| `missile_kinematics` | 40 | **42** | **+2** | 20 | **44** | **+24** | 0.50 | **1.05** | 3 | **3** |
| `system_links` | 30 rel | **8 rel** | **−22** | — | — | — | 1.00 | 1.00 | 0 | 0 |
| **TOTAL ENT** | **150** | **153** | **+3 (+2%)** | | | | | | | |
| **TOTAL FILLS** | | | | **133** | **161** | **+28 (+21%)** | | | | |
| **TOTAL REL** | **30** | **8** | **−22 (−73%)** | | | | | | | |

- **Verdict:** **Mixed — fills improved 21% but system_links regressed further (8 vs prior 19).** Diagnosis: NOT a regression from my type-segregation. Of 10 cross-entity hints, only 1 resolved because the overlay's `alias_map_by_entity_type` has only `MISSILE_SYSTEM` keys — no `RADAR_SYSTEM` bridge for OCR'd radar variants like `RSN- 75V`. The other 7 relationships are LLM-emitted prose CUES (stochastic vs prior run's 18).
- **Diagnostics:** 3 TRUNCATION_PERSISTS_RETRYING (recovered at 65K). 3 BATCH_HARD_TIMEOUTs in `missile_kinematics` (batches 37/51/52 hung >3h, library soft-failed silently). 145 unit tests pass.
- **★STRONGEST★ on:** total filled properties (+28 vs v9). Single-pass Phase B still ★STRONGEST★ on kinematics fill count (49 vs this run's 44 — likely dropped 5 due to the 3 hard-timeouts).
- **Revert:** the per-table relevance + min_altitude fixes are pure wins and should stay. If system_links regression is unacceptable, revert ONLY the `_resolve_ref` change in `evidence_gate.py:1046-1101` and the call-site changes in `_postprocess_air_defense_system_links` — would restore prior 19-rel baseline (still below v9's 30 but closer). Better: implement Fix A from the handoff doc (populate RADAR_SYSTEM alias map).
- **Pending follow-up (handoff doc):**
  1. **Fix A** — populate `alias_map_by_entity_type["RADAR_SYSTEM"]` from cross-entity-ref columns (recovers ~9 promoted hints)
  2. **Fix C** — add `service_postprocess.unresolved_hint_samples` diagnostic (observability)
  3. **Fix B** — investigate the 3 hung kinematics batches (could add 5-10 fills)

---

## Run OPTIONS-A+C-FULL — full 5-pass with per-pass+per-table relevance gate + caption/prose unit detection

- **Date:** 2026-05-16 15:50 → 21:00 UTC (this session)
- **Commit:** `b8a9ccd feat(extraction): per-pass+per-table synth-only policy + min_altitude_km unblock + unit-convention detection`
- **Captured fixtures:** `tests/fixtures/sa2/78673393-..._{pass}_response.json` for all 5 passes; summary at `..._extraction_counts_today.json`
- **Wall-clock:** 5h 10m total (radar_id 56.7m + rf 37.5m + missile_id 121m + kinematics 76m + system_links 14m)
- **Changes vs V9-POST-FIX-GATED (Phase B kinematics-only):**
  - **Step C (per-row unit detection)**: `_detect_row_unit_from_cells` in normalize.py — scans non-label columns for known unit tokens (m, km, m/s, kg, ft, lb, etc.) and promotes to NormalizedRow.unit
  - **Step A (caption + adjacent-prose unit-convention)**: `detect_unit_convention(table_idx, doc_json)` in `_pipeline_hooks.py` — regex matches imperial/metric markers; metric is default. Threaded through `render_for_graph(..., unit_convention=...)`.
  - **Three UNIT_HINT variants in render_graph.py**: `UNIT_HINT_METRIC` (default), `UNIT_HINT_IMPERIAL`, `UNIT_HINT_MIXED`
- **Per-pass results:**

| pass | v9 ent | new ent | Δent | v9 fills | new fills | Δfills | v9 avg | new avg | truncations |
|------|------:|--------:|-----:|---------:|----------:|-------:|-------:|--------:|------------:|
| `radar_identity` | 24 | 27 | **+3** | 26 | 29 | **+3** | 1.08 | 1.07 | 0 |
| `radar_power_rf` | 42 | 43 | +1 | 25 | 23 | −2 | 0.60 | 0.53 | 0 |
| `missile_identity` | 44 | 43 | −1 | 62 | 61 | −1 | 1.41 | 1.42 | 0 |
| `missile_kinematics` | 40 | 45 | **+5** | 20 | 40 | **+20** | 0.50 | 0.89 | **1** (32K→65K, recovered) |
| `system_links` | 30 rel | **19 rel** | **−11** | — | — | — | 1.00 | 1.00 | 0 |
| **TOTAL ENT** | **150** | **158** | **+8 (+5%)** | | | | | | |
| **TOTAL FILLS** | | | | **133** | **153** | **+20 (+15%)** | | | |
| **TOTAL REL** | **30** | **19** | **−11 (−37%)** | | | | | | |

- **`missile_kinematics` per-field detail vs v9 / vs Phase B (kinematics-only):**

| field | v9 | Phase B | new | Δ vs v9 | Δ vs Phase B |
|-------|---:|--------:|----:|--------:|-------------:|
| `min_intercept_km` | 4 | 12 | 10 | +6 | −2 |
| `max_intercept_km` | 10 | 13 | 10 | tied | −3 |
| `min_altitude_km` | 0 | 11 | 10 | **+10** | −1 |
| `max_altitude_km` | 6 | 13 | 10 | +4 | −3 |
| `max_launch_angle_deg` | 0 | 0 | 0 | tied | tied |
| **TOTAL** | **20** | **49** | **40** | **+20 (+100%)** | **−9 (−18%)** |

- **Diagnostics:** 1 TRUNCATION_PERSISTS_RETRYING during kinematics (recovered at 65K). 0 HARD_TIMEOUT events. Per-pass synth-only / append-to-end classification: `radar_power_rf` saw 0 synth-only / 13 append-to-end (correct — no RF rows in SA-2 missile table); other passes likewise classified per `is_table_relevant_for_pass`.
- **Verdict:** **Mixed — kinematics wins (+100% fills) but system_links regression persists (−37% relationships).** Total entities +5%, total fills +15%, but system_links is the blocker for ★STRONGEST★ designation.
- **Root cause of system_links regression (per third-party diagnostic):** `cross_entity_hints_count=10` in both runs, but only `1` hint resolved this run vs `7` in v9. The resolution path in `evidence_gate.py:929` only checks `name_to_ref` from upstream entities; table-overlay alias clusters (`alias_map_by_entity_type`) are NOT consulted as fallback. When per-pass canonical-name cleanup reduces upstream entity-alias diversity, table-derived hints stop resolving.
- **Revert if needed:** revert to `d48756c` (post-edit) or `364b593` (V9-POST-FIX-GATED baseline) per their entries above. Per-pass policy in `_pipeline_hooks.SYNTH_ELIGIBLE_PASSES` can also be narrowed in-place.
- **Pending follow-up (handoff received 2026-05-16):**
  1. **Fix system_links alias resolution** — wire `table_overlay_obj.alias_map_by_entity_type` into `_postprocess_air_defense_system_links` as fallback resolver
  2. **Strengthen table relevance with entity-type qualification** — require BOTH row-label match AND identity context match (prevent false positives on logistics/comms tables)
  3. **Make `min_altitude_km` deterministic in production-shape evidence** — fix regex for collapsed (no-newline) evidence; scope by entity via ENTITY block extraction

---

## Append below as runs complete

(Add new entries here. Update "★STRONGEST★" marker on the entry with the
best balance of recall + fill for the user's quality bar.)

---

## Run RECS-1+2+3+5 — full 5-pass with cross-entity hint resolution + Rec 3 metric/relevance + Rec 5 timeout env passthrough  ★ STRONGEST FIELD-FILL BASELINE (relationship-recall regression open) ★

- **Date:** 2026-05-17 (06:30 → 12:25 local; ~5h 55m wall-clock)
- **Commit:** `767d62f feat(extraction): Recs 1+2+3+5 — unresolved-hint diag, target alias maps, RF metric separation, no-synth-append on irrelevant, batch-timeout env passthrough`
- **Branch:** main (HEAD)
- **Captured fixtures:** `tests/fixtures/sa2/78673393-...-{pass}_response.json` (these are the unsuffixed current fixtures; v9 baseline preserved at `..._response_v9.json`)
- **Config:**
  - `DOCLING_GRAPH_CHUNK_MAX_TOKENS=512`
  - `DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE=512`
  - `DOCLING_GRAPH_LLM_MAX_TOKENS=16384`
  - `DOCLING_GRAPH_LLM_TRUNCATION_RETRY_MAX_TOKENS=65536`
  - `DOCLING_GRAPH_BATCH_HARD_TIMEOUT_SECONDS=10800` (3h) — **env passthrough now wired** in docker-compose.yml (Rec 5)
  - `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=true`
  - `DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN=true`
  - `DOCLING_GRAPH_LLM_THINK=false`
  - Per-pass + per-table synth-only policy (`is_table_relevant_for_pass` requires BOTH row-label AND identity-context match)
  - Rec 1: unresolved-hint diagnostics in postprocess (`unresolved_cross_entity_hints.count` now reported)
  - Rec 2: target-side alias maps for cross-entity rows in `_table_facts.py` — overlay's `alias_map_by_entity_type` now bridges table-local missile names to canonical radars
  - Rec 3A: separate `radar_power_rf` metric categories (identity-merge crumbs no longer count as RF fills)
  - Rec 3B: numeric/spec passes skip non-relevant tables entirely; identity/prose passes still append-to-end (v9 behavior preserved)
  - `_resolve_ref` now type-segregated (`_build_upstream_name_map_by_type`) — no cross-type leaks
- **Per-pass results vs v9 baseline:**

| pass | v9 ent | new ent | Δent | v9 fills | new fills | Δfills | v9 avg | new avg | Δavg |
|------|------:|--------:|-----:|--------:|----------:|-------:|-------:|--------:|-----:|
| `radar_identity` | 24 | 19 | **−5** | 50 | 45 | −5 | 2.08 | 2.37 | **+0.29 ✓** |
| `radar_power_rf` | 42 | 38 | **−4** | 67 | 61 | −6 | 1.60 | 1.61 | tied |
| `missile_identity` | 44 | 44 | **0 ✓** | 106 | 108 | +2 | 2.41 | 2.45 | +0.05 ✓ |
| `missile_kinematics` | 40 | 39 | −1 | 60 | **86** | **+26 ★** | 1.50 | **2.21** | **+0.71 ★** |
| `system_links` rels | 30 | 21 | **−9** | — | — | — | — | — | — |
| **TOTAL ENT** | **150** | **140** | **−10 (−7%)** | | | | | | |
| **TOTAL FILLS** | | | | **283** | **300** | **+17 (+6%)** | | | |
| **TOTAL REL** | **30** | **21** | **−9 (−30%)** | | | | | | |

- **`missile_kinematics` per-field detail (the historically weak pass):**

| field | v9 | new | Δ |
|---|--:|--:|--:|
| `min_altitude_km` | **0** | **10** | **+10 ★** |
| `min_intercept_km` | 4 | 11 | +7 |
| `max_intercept_km` | 10 | 13 | +3 |
| `max_altitude_km` | 6 | 13 | +7 |
| `max_launch_angle_deg` | 0 | 0 | tied |

- **`radar_power_rf` per-field detail (Rec 3A correctly categorizes — still emitter_function/nomenclature dominate; no `tx_peak_power_kw`/`nominal_rf_mhz`/`erp_dbw` recall yet):**

| field | v9 | new | Δ |
|---|--:|--:|--:|
| `system_name` | 42 | 38 | −4 |
| `emitter_function` | 23 | 21 | −2 |
| `nomenclature` | 2 | 2 | tied |
| `erp_dbw` / `nominal_rf_mhz` / `tx_peak_power_kw` | 0 | 0 | tied |

- **`system_links` decomposition (the 30→21 gap):**

| | v9 | new | Δ |
|---|--:|--:|--:|
| Total rels | 30 | 21 | −9 |
| **Hint-promoted (deterministic)** | **7** | **10** | **+3 ✓** |
| **LLM-emitted (rels − promoted)** | **23** | **11** | **−12** |
| `unresolved_cross_entity_hints.count` | (not tracked) | **0** | — |
| `cross_entity_hints` total | — | 10 | — |

Per-rel-type:

| rel_type | v9 | new |
|---|--:|--:|
| `ASSOCIATED_WITH` | 27 | 20 |
| `CUES` | 3 | 1 |

- **Diagnostics this run:** 2 TRUNCATION_PERSISTS_RETRYING events (both recovered at 65K, `final=recovered_content`); 14 TRUNCATION_AT_NUM_PREDICT events (all recovered at 32K); **0 BATCH_HARD_TIMEOUT events** (vs 3 in prior run — Rec 5 env-passthrough fix paid off).
- **Verdict (corrected per third-party review):** **Strongest field-fill baseline so far** — fills +17, kinematics fully unblocked (avg fill 1.50 → 2.21), hint-resolution is solved (`unresolved_cross_entity_hints.count=0`). **NOT a clean win on relationship recall.** The system_links 30→21 decomposes to: deterministic side gained +3, LLM-emitted side lost 12. Some of the 12 are granularity shifts (missile→radar-family vs missile→parent-SAM-system) but the lost CUES rels (Side Net/Spoon Rest/Amazonka → SNR-75) are **real recall losses**, not artifacts. Radar entity drops at the pre-filter LLM stage (radar_identity 30→25, radar_power_rf 48→39, missile_identity 50→47) are LLM-side regressions caused by **two separate mechanisms**:
  1. **`radar_identity` / `missile_identity`** (in `RAW_ONLY_PASSES`): non-relevant synth tables go to `_append_synth_refs` (append-to-end). Those synth chunks now carry the `UNIT_HINT` preamble, which may bias the LLM toward numeric extraction over name extraction.
  2. **`radar_power_rf`** (in `SYNTH_ELIGIBLE_PASSES`): Rec 3B `continue`s irrelevant tables entirely. The previously-appended chunks (which carried extra entity-name context that v9 was over-emitting on) are now gone. This is an intentional Rec 3B trade-off — less noise, but also less name context.
- **Decision needed (graph semantics) before this run can be marked unconditionally strongest:**
  1. Are missile-variant → parent-SAM-system relationships (e.g. `1D → S-75`) part of the desired graph? v9 had them; new doesn't.
  2. Should `CUES` radar-to-radar handoff edges (Side Net / Spoon Rest / Amazonka → SNR-75) be preserved? v9 had 3; new has 1 with inverted direction.
- **Pure-loss recovery candidates (not addressed by Rec 1-5):**
  1. Side Net → SNR-75 (CUES) — search→fire-control handoff
  2. Spoon Rest → SNR-75 (CUES) — early-warning→engagement handoff
  3. Amazonka → SNR-75 (CUES; latest has Fan Song → Amazonka, direction inverted)
- **Revert command (if regression deemed unacceptable):**
  ```bash
  git checkout 666084b -- .   # full v9 restore
  # OR keep Recs 1-5 + bring back only the v9-era LLM input flavor (no known partial-revert recipe)
  ```

---

## Run ITEMS-1+3+4+5 INTEGRATED (two-doc) — UNIT_HINT gate + role-aware CUES + deterministic VARIANT_OF + display-name canonicalization  ★ NEW STRONGEST INTEGRATED BASELINE ★

- **Date:** 2026-05-18 14:47 local → 2026-05-19 01:34 UTC (~10h 47m chained wall-clock across 2 docs)
- **Working-tree code (pre-commit):** Items 1+3+4+5 atop `767d62f` (RECS-1+2+3+5)
- **Branch:** main (working tree)
- **Docs run:**
  - SA-2 reference (`78673393-639b-4fde-9bda-9e7bfd43ccda`) — 308 texts / 2 tables / 34 pictures
  - S-75 Dvina (`b77c48f9-3a27-473f-be05-fa7e73e5d6f5`) — 182 texts / 0 tables / 3 pictures (first time tested with current code)
- **Captured fixtures:** `tests/fixtures/sa2/{doc_id}_{pass}_response.json` (both docs)
- **Pre-run state backup:** `tests/fixtures/sa2/items_1_3_4_5_pre_full_run/`
- **Item-specific code changes (vs RECS-1+2+3+5):**
  - **Item 1** — `_emit_unit_hint = _is_relevant and _pass_is_numeric_spec` in `main.py:697`; explicit `emit_unit_hint: bool = True` kwarg threaded through `render_for_graph` → identity/raw-only passes no longer get UNIT_HINT preamble
  - **Item 3** — `_retype_radar_radar_to_cues` in `evidence_gate.py`; reads `properties.emitter_function` on upstream RADAR_SYSTEM refs; retypes RADAR→RADAR ASSOCIATED_WITH to CUES when source ∈ `{SEARCH, HEIGHT_FINDER}` AND target ∈ `{FIRE_CONTROL}`. Adds `EntityRef.properties` schema field
  - **Item 4** — new `RelationshipType.VARIANT_OF` enum value + `_STATIC_RELATIONSHIP_METADATA` entry + `manifest.yaml extracted_relationship_types: [ASSOCIATED_WITH, CUES, VARIANT_OF]`. `_emit_variant_of_relationships` in evidence_gate emits boundary-aware structural matches (parent system_name ∈ child's alias with non-alphanumeric boundary; parent ≥3 chars, letter+digit). Zero equipment names in code path
  - **Item 5** — `_canonicalize_display_name()` in evidence_gate.py at postprocess tail; conservative OCR cleanup: hyphen-space collapse, slash-spacing collapse, repeated whitespace. Runs AFTER `_clear_unsupported_*` so evidence checks operate on raw LLM output

### SA-2 — per-pass results vs v9 + prior

| pass | v9 ent/fills | prior ent/fills | **new ent/fills** | Δ ent vs v9 | Δ fills vs v9 |
|------|--:|--:|--:|--:|--:|
| `radar_identity` | 24 / 50 | 32 / 69 | **29 / 54** | +5 | +4 |
| `radar_power_rf` | 42 / 67 | 39 / 61 | **37 / 58** | −5 | −9 |
| `missile_identity` | 44 / 106 | 44 / 97 | **42 / 102** | −2 | −4 |
| `missile_kinematics` | 40 / 60 | 39 / 86 | **38 / 77** | −2 | **+17** ★ |
| **TOTAL** | **150 / 283** | **154 / 313** | **146 / 291** | **−4** | **+8 (+3%)** |

### SA-2 — `system_links` composition (28 rels)

| rel_type | v9 | prior | **new** |
|---|--:|--:|--:|
| ASSOCIATED_WITH | 27 | 13 | 11 |
| CUES | 3 | 6 | 2 |
| **VARIANT_OF** | **0** | **11** | **15** ★ |
| **total** | **30** | **30** | **28** |
| hint-promoted | 7 | 10 | **10** |
| unresolved cross-entity hints | – | 0 | **0** |

**Structural acceptance (all ✅):** 0 cross-type VARIANT_OF, 0 self-loops, 0 boundary violations, 0 HARD_TIMEOUT, 1 PERSISTS recovered at 65K (1476 chars).

### S-75 Dvina — per-pass results (no v9 baseline; first run with current code)

| pass | entities | fills | wall-clock |
|---|--:|--:|--:|
| `radar_identity` | 1 (`RSNA-75M`) | 1 | ~3 min |
| `radar_power_rf` | 1 (`RSNA-75M`) | 1 | ~2 min |
| `missile_identity` | 1 (`S-75`) | 4 | ~8 min |
| `missile_kinematics` | 2 (`S-75 Dvina`, `S-75`) | 2 | ~7 min |
| `system_links` | 1 rel (`ASSOCIATED_WITH`) | — | ~1 min |

Pipeline ran clean — 0 errors, 0 hangs. Total wall ~22 min.

### What was missed (source vs extracted)

Surveyed designation-like tokens in source prose via regex and compared to extracted entity rosters.

**SA-2** (22,607 source chars; 70 distinct designation tokens in prose):
- **Extracted across all passes:** ~80 entities (many duplicates due to spelling variants the dedup couldn't collapse — e.g. `Fan Song`, `Fan Song E`, `SNR-75 Fan Song`, `RSNA-75/SNR-75 Fan Song` are the same physical radar)
- **Captured cleanly:** S-75, V-750, 1D, 13D, 15D, 20D, 20DP, 5Ya23, DM/DA/DAM/DP/DSU, SA-2 family, Fan Song family, Spoon Rest, Side Net, Amazonka, Konus, Vershina, Parol, Flat Face, Squat Eye, P-12, P-15/19, P-18-x, PRV-10, PRV-11
- **Likely missed or under-captured:**
  - **`SM-90`** — 8 mentions in prose (the launcher/erector); not in missile fixtures
  - **`SM-63`** — 2 mentions (earlier launcher variant)
  - **`AK-20F`** — 3 mentions (variant designation)
  - **`PR-11A`** (3x), **`PR-11AM`** (5x), **`PRD-18`** (2x), **`11A`** (3x), **`11AM`** (5x), **`11D`** (2x) — secondary launcher/control designations
  - **`105D`**, **`18M`**, **`20F`** (3x), **`2A`**, **`4C`**, **`75M`** — short variant suffixes that may be missile/radar sub-variants
  - **`FIFB-22`**, **`NMF-2`**, **`OT-155`** — auxiliary designations, possibly off-domain
  - **`APA-S-75-*`** — 28 mentions of various subtype suffixes (test/maintenance equipment)
  - **`F-105D`**, **`RF-4C`** — target aircraft (out of ontology scope, correctly excluded)

**S-75 Dvina** (5,814 source chars; 25 distinct designation tokens in prose):
- **Source prominence:** `S-75` 16x, `SA-2` 4x, `Dvina` extensively in prose
- **Extracted:** 1 S-75, 1 RSNA-75M radar, 2 missile_kinematics entries
- **Severely under-extracted** — given 16 mentions of S-75 + the doc is literally about S-75 Dvina family, we'd expect identity passes to capture multiple variants (Dvina, Desna, Volkhov, V-750 series, 1D variant, NATO `Guideline` codenames). Almost none surfaced.
- **Likely cause:** The prose is sparse text snippets (182 texts × ~32 chars avg) — likely Wikipedia-extracted with heavy boilerplate. LLM has less context per chunk; missile_identity extracted from one chunk and stopped. Need to investigate chunking + prompt behavior on short-text docs.

### Verdict

✅ **★ NEW STRONGEST INTEGRATED BASELINE ★** for typed-graph semantics. SA-2 fills up +3% vs v9, kinematics still +28% (Item 1 win preserved), 0 structural violations, system_links now produces typed VARIANT_OF + CUES + ASSOCIATED_WITH instead of v9's flat ASSOCIATED_WITH conflation.

**Open work surfaced by this run:**
1. **Dvina prose-only under-extraction** — pipeline emitted only 1-2 entities per pass on a doc that mentions S-75 ~16x. Possibly chunking/prompt issue for short-text Wikipedia-style content. Not a regression (no prior baseline on this doc) but a quality gap to investigate.
2. **SA-2 launcher/secondary designations** — `SM-90` (8x mentions), `PR-11AM` (5x), `11AM` (5x), `11A` (3x), `PR-11A` (3x). These are launchers and ancillary equipment, which may need a new entity-type pass (LAUNCHER_SYSTEM exists in `RelationshipType.LAUNCHES`).
3. **Radar entity duplication** — `Fan Song`, `Fan Song E`, `SNR-75 Fan Song`, `RSNA-75/SNR-75 Fan Song` all represent the same physical radar but appear as separate entities. Item 5 helps post-extraction; pre-extraction or cross-pass merge would help more.
4. **Item 2 (compact identity roster)** — still pending; would help radar_power_rf entity recall (currently −5 vs v9).
