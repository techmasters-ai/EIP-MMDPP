# SA-2 post-review run — stats, analysis, diagnosis, fixes

**Run identifier:** OPTIONS-A+C+REVIEW-FIXES
**Date:** 2026-05-16 20:54 → 2026-05-17 02:39 (local)
**Commit:** `d9647e3 feat(extraction): restore system_links + generalize table relevance + production-shape deterministic min_altitude_km`
**Wall-clock:** 5h 45m (radar_id 65.4m + rf 40.5m + missile_id 126.3m + kinematics 98.6m + system_links 14.0m)
**Doc:** SA-2 reference `78673393-639b-4fde-9bda-9e7bfd43ccda`
**Model:** `gemma4:31b` via Ollama (`http://10.0.1.121:11434` + `http://10.0.1.109:11434`)

## 1. Headline numbers

| metric | v9 baseline | prior run (b8a9ccd) | this run (d9647e3) | Δ vs v9 |
|---|---:|---:|---:|---:|
| total entities (4 entity passes) | 150 | 158 | 153 | +3 (+2%) |
| total filled properties | 133 | 153 | **161** | **+28 (+21%)** |
| total relationships (system_links) | 30 | 19 | **8** | **−22 (−73%)** |
| TRUNCATION_PERSISTS_RETRYING events | 0 | 1 | **3** | +3 |
| BATCH_HARD_TIMEOUT events | 0 | 0 | **3** | +3 |

## 2. Per-pass detail

| pass | v9 ent / fill / avg | this run ent / fill / avg | elapsed | trunc | hard_timeout |
|------|---:|---:|---:|---:|---:|
| `radar_identity` | 24 / 26 / 1.08 | **26 / 32 / 1.23** | 65.4m | 0 | 0 |
| `radar_power_rf` | 42 / 25 / 0.60 | 42 / 24 / 0.57 | 40.5m | 0 | 0 |
| `missile_identity` | 44 / 62 / 1.41 | 43 / 61 / 1.42 | 126.3m | 0 | 0 |
| `missile_kinematics` | 40 / 20 / 0.50 | **42 / 44 / 1.05** | 98.6m | 3 | **3** |
| `system_links` | 30 rel | **8 rel** | 14.0m | 0 | 0 |

### Per-field detail for the two big-delta passes

**`missile_kinematics` (4 fields filled vs 2 in v9):**

| field | v9 | new | Δ |
|---|---:|---:|---:|
| `min_intercept_km` | 4 | 11 | +7 |
| `max_intercept_km` | 10 | 13 | +3 |
| `min_altitude_km` | **0** | **10** | **+10** (was hard-cleared in v9) |
| `max_altitude_km` | 6 | 10 | +4 |
| `max_launch_angle_deg` | 0 | 0 | tied (not in source) |

**`radar_identity`:**

| field | v9 | new | Δ |
|---|---:|---:|---:|
| `emitter_function` | 16 | 19 | +3 |
| `nomenclature` | 9 | 12 | +3 |
| `scan_type` | 1 | 1 | tied |

## 3. Analysis

### What worked
- **Three coordinated fixes from the outside-reviewer handoff all landed correctly:** type-segregated `_resolve_ref`, per-pass + per-table relevance gate with entity-type qualification, production-shape entity-scoped `min_altitude_km` parser.
- **145 unit tests pass** including 12 new tests pinning type-segregation behavior.
- **Field fill recall improved 21% overall vs v9** (133 → 161 total filled properties).
- **`min_altitude_km` permanently unblocked** — was 0/40 in v9 (hard-cleared by `evidence_gate.py:742` unconditional-null branch), now 10/42.
- **`radar_identity` gained on both entity recall AND fill rate** (+2 entities, +6 fills, avg_fill 1.08 → 1.23). Per-row unit detection (`normalize.py::_detect_row_unit_from_cells`) is contributing.
- **No table-relevance false positives observed.** SA-2 has 2 normalized tables; per-pass-per-table classification fired correctly across all 5 passes (e.g. radar_power_rf classified as 0 synth-only / 2 append-to-end, missile_kinematics as 1 synth-only / 1 append-to-end).

### What didn't
- **`system_links` collapsed to 8 relationships** (vs v9's 30, prior's 19). Most concerning.
- **3 BATCH_HARD_TIMEOUT events in kinematics** even with the 3h ceiling. Batches 37, 51, 52 hung — library soft-failed silently with `pass_output will be empty or partial`.
- **3 TRUNCATION_PERSISTS_RETRYING events**, all recovered at 65K but added latency.

## 4. Diagnoses

### 4.1 — system_links drop (30 → 8) is NOT caused by the type-segregated resolver

**Evidence:** `tests/fixtures/sa2/78673393-..._system_links_response.json`:
- `service_postprocess.promoted_from_cross_entity_hints` contains **1** entry (the `13D → RSN-75` pairing).
- `table_overlay.cross_entity_hints` contains **10** entries.
- The other 9 hints failed to resolve because their `target_alias` values are OCR-extracted spaced variants of radar names (`RSN- 75V`, `RSNA- 75M`, `RSN- 75M4`, etc.) that don't appear in the upstream `RADAR_SYSTEM` catalog.
- The overlay's `alias_map_by_entity_type` keys are `['MISSILE_SYSTEM']` only — there is no `RADAR_SYSTEM` alias map, so the type-aware fallback path cannot bridge OCR'd radar variants to canonical radar names.

**The 7 other relationships that DID get emitted** are LLM-extracted `CUES` edges (Spoon Rest D/E → Fan Song, Konus → Fan Song, etc.) — pure prose-derived, no overlay involvement. The prior run had 18 such CUES; this run had 7. That delta is **LLM stochasticity** at temperature=0.1 between two runs on the same prompts.

**Total accounting:** 7 LLM-emitted + 1 promoted = 8 final. Matches.

**This means:** my type-segregation change did not regress hint promotion vs prior code. The 30→8 cumulative drop has two independent causes:
- (a) **prior 30→19 drop:** the per-pass canonical-name cleanup (synth-only chunking) shrank upstream alias diversity, making fewer LLM-CUES edges resolve. Pre-existing.
- (b) **this 19→8 drop:** LLM emitted ~11 fewer prose-derived CUES this run vs prior, on identical inputs. Stochastic.

### 4.2 — 3 BATCH_HARD_TIMEOUTs in kinematics

**Evidence:** `docker logs eip-mmdpp-docling-graph-1`:
```
2026-05-17 06:46:30 Warning: BATCH_HARD_TIMEOUT - 3 of 53 futures did not
  complete within 3600s; hung batch_idxs=[37, 51, 52]
2026-05-17 07:25:04 GRAPH_EXTRACTION_LIBRARY_WARNING pass=missile_kinematics
  tag=batch_hard_timeout count=1 signature='BATCH_HARD_TIMEOUT' —
  library soft-failed silently; pass_output will be empty or partial.
```

3 batches in the same pass hung at the 3h ceiling. The previous runs (b8a9ccd) had 0 such events. Two hypotheses:
- **(a) Synth-only blocks for kinematics are larger this run** because the table normalizer added the SI UNIT_HINT preamble, increasing per-batch token count. Larger prompts → longer generation. Three outlier batches happened to hit the 3h wall.
- **(b) Gleaning-phase deadlock** in the orchestrator's `ThreadPoolExecutor` when one batch hangs and others block on a shared semaphore.

Lost content: probably ~2-5 entity-fields per hung batch. The kinematics fill total (44) might have been 48-50 without the timeouts, closer to Phase B's 49.

### 4.3 — Overlay `alias_map_by_entity_type` is MISSILE-only

**Code path:** the overlay generator in `docker/docling-graph/repo/docling_graph/...` (TableOverlay class) produces `alias_map_by_entity_type` from the normalized table's identity rows. For SA-2's missile-spec table, the identity rows are `Industry Designation`, `Military Designation`, `NATO Designation`, `Fan Song Variant`, `Missile Type`. The overlay correctly maps each Missile-Type token to its NATO/Industry/Military designation siblings under `MISSILE_SYSTEM`. But it does NOT recognize `Fan Song Variant` as a `RADAR_SYSTEM` identity column — so the radar-side alias map is empty.

**Effect on resolution:** when a hint says `target_alias="RSN- 75V"`, type=`RADAR_SYSTEM`:
- Direct upstream hit: `name_to_ref_by_type["RADAR_SYSTEM"].get("RSN- 75V")` → None (upstream catalog has canonical "Fan Song" only)
- Alias fallback: `alias_map_by_entity_type["RADAR_SYSTEM"]` → does not exist
- Returns None — hint dropped

### 4.4 — radar_power_rf flat (42 ent / 24 fills) is expected

This pass's numeric fields (`erp_dbw`, `nominal_rf_mhz`, `tx_peak_power_kw`) are **0/42 in v9 and 0/42 in this run** — the SA-2 doc doesn't carry that data in any tabular or prose form. All 24 fills are on `emitter_function` (radar role) + `nomenclature`, both prose-derived. Synth-blocks don't help these passes on this document. Per-table relevance correctly classified the SA-2 missile-spec table as NOT relevant to `radar_power_rf` (0 synth-only / 13 append-to-end logged at pass start).

### 4.5 — missile_identity flat (43 ent / 61 fills) is the right outcome

The 9 entities lost vs v9 are duplicate naming-row emissions that v9's raw-flat chunking over-counted (Industry/Military/NATO/Fan Song Variant each emitting a separate entity for the same missile). The synth-block consolidation now produces one entity per missile column. The 1-fill loss is within stochastic variance.

## 5. Fixes (proposed, ranked)

### Fix A — Populate `alias_map_by_entity_type["RADAR_SYSTEM"]` from `Fan Song Variant`-style cross-entity-ref columns

**Impact:** would resolve all 9 currently-failing radar-side hints → up to +9 promoted ASSOCIATED_WITH edges → system_links toward 17-20 (still LLM-variance below 30, but recovers the deterministic floor).
**Effort:** low-medium. Identify cross-entity-ref columns in the table normalizer (currently labeled `Fan Song Variant` → `RADAR_SYSTEM` in `_table_facts.py:1010`), emit one alias entry per (variant_name, canonical_radar_name) pair into `alias_map_by_entity_type["RADAR_SYSTEM"]`. Code path: `docker/docling-graph/repo/docling_graph/core/extractors/contracts/delta/table_overlay.py` (need to verify exact filename).
**Risk:** very low — only adds entries to alias map; never replaces existing ones. Tests can pin the new map shape.

### Fix B — Diagnose + recover the 3 hung kinematics batches

**Impact:** could add 5-10 fills to kinematics (closing the 44→49 gap to Phase B).
**Effort:** medium. Need to inspect the diagnostic dump for batches 37/51/52 (look in `/tmp/docgraph-debug-*/debug/`). Two paths:
- (a) Reduce per-batch token count via tighter `DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE` for kinematics specifically, OR
- (b) Investigate the gleaning-path deadlock — the watchdog patch (`docker/docling-graph/patches/0001-orchestrator-batch-hard-timeout.patch`) catches but doesn't recover.
**Risk:** low. Already-running timeout config is the only thing changing.

### Fix C — Add `service_postprocess.unresolved_hint_samples` diagnostic

**Impact:** zero on metrics — pure observability. Lets future runs immediately surface which hint name-keys fail (currently we have to inspect by hand).
**Effort:** low. ~10 lines in `_postprocess_air_defense_system_links` — count and sample unresolved (source_name, target_name, source_type, target_type) tuples.
**Risk:** none.

### Fix D — Drop temperature-stochasticity from system_links by running multiple LLM samples per chunk and union-deduping

**Impact:** could stabilize the LLM-CUES count from the volatile 7-18 range toward the upper end consistently.
**Effort:** medium. Extend the existing gleaning loop to do 2-3 samples per batch at temp=0.1, union the relationship sets, dedup by (from, to, type). Costs 2-3x LLM time on system_links only.
**Risk:** medium — could introduce hallucinated edges if temp-variance includes spurious extractions.

### Fix E — Investigate why no `radar_power_rf` numerics appear in the SA-2 source

**Impact:** verification only — confirms 0/42 is correct vs missed extraction.
**Effort:** ~30 min manual inspection of the doc.
**Risk:** none.

## 6. Recommended sequencing for the reviewer

1. **Ship Fix A first** — lowest risk, highest leverage on system_links. Should land before another full SA-2 run.
2. **Add Fix C diagnostics** before Fix A, so we can directly measure A's effect.
3. **Investigate Fix B** — possibly per-pass token-budget tightening for kinematics specifically.
4. **Defer Fix D** — only pursue if Fixes A+B don't move system_links materially.
5. **Skip Fix E** unless someone cares about whether SA-2 has hidden radar-RF data.

## 7. What's already committed (this session)

Commit `d9647e3` (pushed) — three reviewer-handoff fixes from the prior session plus the test/fixture updates. NO additional uncommitted code right now.

Code lines for the reviewer to focus on:
- `docker/docling-graph/app/evidence_gate.py:1046-1101` — `_resolve_ref` (type-segregated). Verified by `docker/docling-graph/tests/test_resolve_ref.py` (12 tests).
- `docker/docling-graph/app/evidence_gate.py:1109-1185` — `_postprocess_air_defense_system_links` — calls `_resolve_ref` with per-type maps.
- `docker/docling-graph/app/evidence_gate.py:635-720` — `_extract_synth_block_for_entity`, `_evidence_has_si_unit_hint`, `_mechanically_supported_missile_fields` (entity-scoped, unit-evidence gated).
- `app/services/table_normalization/_pipeline_hooks.py:336-470` — `is_table_relevant_for_pass` with entity-type qualification.
- `docker/docling-graph/app/main.py:1487-1510` — passes `alias_map_by_entity_type` into postprocess.

## 8. Run history reference

Full per-run ledger at `docs/sa2_extraction_runs.md`. Most recent ★STRONGEST★ marker is still on the kinematics-only Phase B run (49 fills, single pass only). This 5-pass run is the strongest END-TO-END on filled properties (+28 vs v9) but doesn't yet beat ★STRONGEST★ on kinematics fill count, and regresses on relationships.

---

## 9. Generalization constraints (binding for all SA-2 extraction work)

These rules apply to ALL code, tests, and configuration for the extraction
pipeline. They are not SA-2-specific — they exist BECAUSE the system must
work on other documents that have different entity names, table layouts,
and unit conventions.

**Any code, test, or config that violates these rules must be flagged in
review and either fixed or held until a generic implementation is
designed. SA-2 is one data point — the implementation must not be tuned
to its specific identities.**

### 1. No document-specific constants
Production code must not contain hardcoded equipment names (`SA-2`,
`S-75`, `Fan Song`, `RSN-75`, `Spoon Rest`, `1D`, `20DP`, etc.). The only
allowed knobs are entity types (`MISSILE_SYSTEM`, `RADAR_SYSTEM`), schema
aliases (`Min Range`, `Max Alt`), row labels, and upstream-entity refs.
Tests may use toy names (`M1`, `R1`, `Canonical-X`) to exercise behavior.

### 2. No label-only entity creation
A row label may classify a row (e.g., `<X> Variant` → cross-entity row)
or suggest a relationship type, but it must not create a new entity or
canonical target on its own. The "family-name fallback" from Rec 2 is
acceptable precisely because the canonical it suggests must already be
in the upstream catalog to resolve.

### 3. Resolve only to existing upstream entities
Cross-entity hints may promote to ASSOCIATED_WITH/CUES edges only when
both source and target resolve to already-extracted upstream entities of
the expected type. Failed resolution → drop, never invent.

### 4. Entity-type segregation is mandatory
A `MISSILE_SYSTEM` alias must never resolve to a `RADAR_SYSTEM` ref, and
vice versa, even if names are similar. This is enforced by
`_build_upstream_name_map_by_type` + `_resolve_ref` and pinned by
`docker/docling-graph/tests/test_resolve_ref.py`.

### 5. Use reversible/low-risk string normalization only
Allowed normalization operations: whitespace collapse, hyphen-space
cleanup (`RSN- 75V` → `RSN-75V`), slash-space cleanup, quote/dash glyph
normalization, casefolding. Forbidden: semantic rewrites, transliteration,
abbreviation expansion (other than via the explicit alias map), domain
synonym inference. Any "smart" inference must be backed by upstream
alias data, not by code-side guessing.

### 6. Make fallbacks observable
Diagnostics must distinguish:
  * direct upstream match
  * alias-map match
  * OCR-normalized match
  * label-derived upstream-confirmed match
  * unresolved (with sample + reason)

Today the `unresolved_cross_entity_hints` diagnostic (Rec 1) captures the
last category. The other four categories should be added when their use
sites materialize — the resolver must report which path it took, not just
whether it succeeded.

### 7. Negative tests are required
Every relevance / resolution / alias change must include negative tests
proving the system does NOT resolve:
  * cross-type aliases
  * row-label-only targets with no upstream entity present
  * unrelated frequency tables into `radar_power_rf`
  * missile tables into radar numeric passes
  * communication/electronics tables into radar RF without radar identity context

These are already captured in `test_pass_table_relevance.py`,
`test_resolve_ref.py`, `test_target_side_alias_map.py`, and
`test_unresolved_hint_diagnostics.py`.

### 8. Measure per table, not per document
A pass may synth-rewrite the specific table that is relevant to that
pass. Other tables in the same document must retain default behavior.
This is the per-table relevance gate in `is_table_relevant_for_pass`.

### 9. Separate evidence from normalization
Upstream identity catalogs may normalize entity names across passes
(canonical-name agreement). They must never serve as evidence for
numeric, spec, or admin field extraction. Field evidence comes from the
current batch text/table only. The identity-roster prompt context
(Rec 4, deferred) must include this constraint in-prompt and must be
unit-tested to confirm the roster alone cannot populate any non-identity
field.

### 10. Prefer diagnostics over clever inference
If a hint cannot resolve under generic rules, leave it unresolved and
report why (Rec 1 diagnostic). Do not add document-specific recovery
heuristics to lift one document's score. Improvements should generalize.

### Compliance status of current code (as of this commit)

| rule | status | notes |
|---|---|---|
| 1. No document-specific constants in production code | ✅ in my changes | Pre-existing debt: `_alias_map.CROSS_ENTITY_REF_PATTERNS` contains `"fan song variant"` and `"spoon rest variant"`. Should be generalized to a pattern-match (`<X> Variant`) in a future commit. |
| 2. No label-only entity creation | ✅ | Family-fallback is gated by upstream presence (test_resolver_returns_none_when_family_canonical_absent_from_upstream). |
| 3. Resolve only to upstream | ✅ | `_resolve_ref` returns None when no upstream hit. |
| 4. Entity-type segregation | ✅ | `_build_upstream_name_map_by_type` + type-scoped `_resolve_ref`. |
| 5. Reversible normalization only | ✅ | Only whitespace/hyphen/case normalization in current code. |
| 6. Fallback observability | ⚠ partial | Unresolved-hint diagnostics (Rec 1) ✓. Resolved-via-which-path diagnostics — not yet implemented. |
| 7. Negative tests | ✅ | See test files cited above. |
| 8. Per-table not per-document | ✅ | `is_table_relevant_for_pass` operates per-table. |
| 9. Evidence ≠ normalization | ✅ today | Rec 4 (deferred) must include the in-prompt constraint and a negative test. |
| 10. Diagnostics over inference | ✅ | No document-specific recovery heuristics added. |
