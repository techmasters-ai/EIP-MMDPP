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
