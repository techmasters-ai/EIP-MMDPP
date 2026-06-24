# Absolute Chunk-Selection (signal-union) — Design Spec

Status: **DRAFT for review** (2026-06-24). Supersedes the `guarded_quantile` selector for routable extraction passes.
Owner goal (verbatim, this session): *"I don't want to be fixed to a certain number of chunks or certain percentage of chunks for a document. If no chunks are appropriate, then the number should be 0. If all chunks are relevant, then all chunks should be chosen. This median calibration is another variable of top-k. I want an absolute scoring mechanism."*

---

## 1. Problem

The current per-pass selector is `guarded_quantile` (`app/services/extraction_candidate_scoring.py::select_candidates`):
`threshold = np.quantile(ranker_scores, q=0.5)` → keep ranker-scores ≥ threshold, floored to `k_min=3`, unioned with unit/table gates.

This is **relative, not absolute**:
- `q=0.5` is the median → "keep ≥ median" removes ~half the pool **by construction**, regardless of content. It is a re-parameterised top-k.
- It **cannot return 0** (the `k_min=3` floor forces ≥3 chunks even when nothing is relevant — e.g. it selected pure boilerplate *"Related Fact Sheets … hyperlinks …"* for `missile_speed_timing`).
- It **cannot return all** except on score ties.
- The ranker score (`Σ ranker_weight · raw_feature`) over-weights lexical/numeric density (`unit_token_count` weight 1.24) and under-weights semantics (`max_field_cosine` 0.22, `rerank_norm` 0.055), so it drops semantically-relevant entity chunks (e.g. a chunk with the 2nd-highest cosine in its pool, 0.52, cut at rank 7).

**Measured cost (offline, bake-off ground truth — 6,130 chunks, 115 used):** the quantile drops **~36% of entity-bearing chunks** (chunk-recall 88.7%) AND wastes **~97% of its selections** (precision 2.6%, selects 64.8% of all chunks). Live A/B (NMUSAF, gate forced off) corroborated: narrowed recall ~14–17 vs full-doc 25/25/26 (~-40%).

## 2. Goal

A selection mechanism that is **absolute and per-chunk**: each chunk is judged on its own merit, independent of the rest of the pool. Properties:
- **0-to-all**: 0 chunks when none are appropriate, all chunks when all are relevant.
- **Content-driven**: keep a chunk because it actually holds a field value, not because of its rank.
- **Recall-honest**: do not silently drop entity-bearing chunks.
- **Tunable**: a single knob trades recall vs narrowing; no hard-coded percentage.

## 3. Design — per-chunk absolute signal-union

For each candidate chunk of a **routable** pass, select it iff **any** signal fires:

```
keep(chunk) ⇔  measurement(pass)          # numeric fields: a number + a unit of one of the pass's dimensions
            OR categorical(pass)           # enum fields: an enum value / prose-mapping phrase
            OR image_presence(pass)        # _photo fields: chunk's source_refs contains a #/pictures/ ref
            OR max_field_cosine ≥ τ         # semantic catch-all (the only tunable signal)
```

- Pure per-chunk; no quantile, no median, no `k_min` floor. Returns 0 when no chunk fires any signal; returns all when every chunk does. **The 0-chunk case requires an explicit endpoint→worker contract — see §3.5; a `select_candidates`-only change will NOT produce 0-yield (it falls open to full-doc).**
- `τ` is the **single tunable knob** (recall ↑ as τ ↓). Default **0.55–0.60** ≈ quantile-level recall at ~3× the narrowing.
- The three lexical/structural signals (measurement, categorical, image) are the precision workhorses; the cosine floor is the recall catch-all.

### 3.1 Signal: measurement (numeric fields)

A number immediately followed by a unit belonging to one of the **pass's dimensions**. Dimensions are derived from the pass's field-name unit suffixes (`_km`→length, `_m`→length, `_deg`→angle, `_sec`/`_usec`→time, `_mhz`→frequency, `_mps`→velocity, `_kg`→mass, `_dbi`→gain, `_kw`→power).

- **Dimension-grouped vocabulary** (a length field accepts *any* length unit): abbreviations + spelled-out + plurals + imperial. E.g. length = {m, meter(s), metre(s), km, kilometer(s), mm/cm + spelled, mile(s), feet/foot, yard(s), inch(es), nmi}; time = {s, sec, ms, µs/us, ns, second(s), minute(s), hour(s)}; frequency = {Hz/kHz/MHz/GHz + hertz forms}; velocity = {m/s, mps, km/s, km/h, mph, knots, Mach, "meters per second"}; mass = {kg, g, t, gram(s), kilogram(s), ton(s), lb(s)}; angle = {deg, °, degree(s), rad, radian(s), mrad}; gain = {dBi, dBm, dBW, dB, decibel(s)}; power = {W, kW, MW, watt(s)+}. (Full lists derived at build time; see §6.)
- **Use the existing bounded unit matcher, not a hand-rolled permissive regex.** Reuse the existing bounded unit matcher: `SUFFIX_UNITS` is **defined in `field_value_grounding.py:100,150`**; `extraction_unit_gate.py` imports/uses it (`_compiled_unit_re`). These already encode unit boundaries. The exploratory prototype regex `\d\s{0,2}-?\s{0,2}(unit)` is **too permissive for short single-letter units**: optional whitespace before `m`/`s`/`g`/`w` makes designators like `S-75M`, `13DM`, `5Ya23` casefold into apparent measurements, which directly undercuts the "0 chunks when none appropriate" property (false positives). Required hardening: (a) **single-letter units** (`m`, `s`, `g`, `w`) require a real separator (space or explicit boundary) between number and unit and a trailing word boundary — no `\s{0,2}` zero-space match; (b) reject matches where the digit is part of an alphanumeric designator token (preceded by `-`/letter within a token like `S-75`); (c) prefer multi-char/spelled forms which are unambiguous. Validate the matcher against a designator block (`S-75M`, `V-88`, `1D/13DM/5Ya23`) → 0 measurement hits.
- Pass-specific cut: a chunk with `190 kg` is NOT selected for `radar_antenna` (dimensions {length, angle, gain}); it IS for `missile_airframe` ({length, mass}). This halved false-positive selection vs a generic detector (40% → 21%) at ~equal recall.

### 3.2 Signal: categorical (enum fields)

For passes with enum fields, match the chunk against the field's **enum values + the prose-mapping phrases already written into the schema field descriptions**. Examples:
- `scan_type`: CIRCULAR/SECTOR/RASTER/ELECTRONIC/DWELL_AND_SWITCH/HELICAL + "rotating antenna", "phased array", "AESA/PESA", "sector scan", "helical scan"…
- `emitter_function`: "search radar", "early warning", "tracking radar", "fire-control radar", "illuminator", "multi-function/MFR/AMDR", "height finder"…
- `guidance_type`: "command guidance/CLOS", "semi-active radar homing/SARH", "active radar homing/ARH", "track-via-missile/TVM", "beam-rider", "IR homing", "home-on-jam/HOJ"…
- `seeker_type`: "SARH/ARH/EO/IR/IIR seeker", "mmW seeker", "dual-mode"…
- `system_status`: "operational", "in service", "deployed", "prototype", "decommissioned", "modernized", "FMS"…

Match = multi-word phrases as case-insensitive substrings; short acronyms as `\bACR\b`.

### 3.3 Signal: image-presence (`_photo` fields)

For passes with a `_photo`/image field (`missile_guidance.missile_photo`, `radar_antenna.antenna_photo`), select the chunk iff its `ExtractionChunk.source_refs` contains a `#/pictures/` ref. (Image entities are inferred from imagery, not text — measurement/categorical are blind to them.)

### 3.4 Signal: cosine floor

`max_field_cosine ≥ τ` (per-chunk similarity to the pass's field set; already computed in `score_components_all`). The only tunable signal and the catch-all for entity chunks not caught lexically (names/designations, unlabeled prose).

### 3.5 Empty-selection contract (0 chunks) — REQUIRED, not just `select_candidates`

The "return 0 → clean ZERO_YIELD" property does **not** come for free from changing `select_candidates`. The current chunk-scope endpoint and worker actively convert an empty selection into **full-doc extraction**, which would silently defeat the 0-to-all goal:
- The endpoint emits `mode="full"` on its empty/no-selected-refs paths (`extraction_routing.py:710`, `:1279`).
- `ChunkScopeResponse.self_refs` is only populated for `mode="selected_refs"`.
- `_compute_effective_chunk_scope` only narrows when `resp_mode == "selected_refs" AND self_refs` is non-empty — anything else → `effective_chunk_scope=None` → **full-doc**.

So a 0-chunk absolute-union result on the current code path falls open to full-doc, not ZERO_YIELD. The redesign therefore **must add an explicit empty-selection contract** spanning three layers:
1. **Endpoint:** when the absolute union selects 0, return a NEW response mode (e.g. `mode="empty_selection"`) — distinct from both `selected_refs` and `full` — carrying the diagnostics (which signals were evaluated) so it's auditable and not confused with the existing `would_skip`/`full` fall-opens.
2. **Worker (`_compute_effective_chunk_scope`):** map `mode="empty_selection"` to "extract nothing for this pass" (a sentinel scope), NOT to `None` (which means full-doc).
3. **Extraction/finalization:** a pass that legitimately selected 0 chunks terminates as **ZERO_YIELD → COMPLETE/EMPTY**, never FAILED or full-doc — reusing/extending `_is_clean_empty_pipeline_error`'s clean-empty path (ties to the deferred legitimate-empty→COMPLETE remediation). 

Each layer needs a unit test; the end-to-end check is an off-domain pass on an in-domain doc → 0 chunks → run COMPLETE with that pass ZERO_YIELD (not full-doc, not FAILED).

## 4. Routing — which passes use this

Only the **9 routable field-group passes** (those carrying a `RetrievalProfile` with `selection_mode` in the manifest):
`radar_power_rf, radar_antenna, radar_timing, radar_modulation, missile_kinematics, missile_guidance, missile_airframe, missile_speed_timing, missile_propulsion`.

Per-pass signal mix:
- **8 numeric passes** → measurement + cosine (+ image for `radar_antenna`'s `antenna_photo`).
- **`missile_guidance`** → categorical + image + cosine.
- **`missile_identity`, `radar_identity`, `system_links`** are **NOT routable** — they run full-doc and are untouched by this design. (Confirmed: no selection diagnostics, no narrowing.)

Per-pass config (derived from the bundle schema, see §6):
| pass | dimensions | categorical | image |
|---|---|---|---|
| missile_airframe | length, mass | — | — |
| missile_propulsion | time, mass | — | — |
| missile_kinematics | length, angle | — | — |
| missile_speed_timing | velocity, time | — | — |
| radar_antenna | length, angle, gain | — | antenna_photo |
| radar_power_rf | frequency, power | — | — |
| radar_modulation | frequency, time | — | — |
| radar_timing | time, length | — | — |
| missile_guidance | — | guidance_type, seeker_type | missile_photo |

## 5. Validation (offline, bake-off ground truth)

Bake-off dataset: 6,130 chunks across 243 (run,pass) groups, 115 ground-truth `used` chunks (1.9%); "perfect" = 2% selected @ 100% recall.

**Numeric passes (measurement + cosine):**
| method | chunk-recall | frac-selected | precision | empty-passes |
|---|---:|---:|---:|---:|
| quantile-0.5 (baseline) | 88.7% | 64.8% | 2.6% | 0 |
| C-lexical (rejected) | 83.5% | 53.7% | 2.9% | 28 |
| measurement pass-spec alone | 86.1% | 21.4% | 7.6% | 76 |
| **measurement OR cosine≥0.55** | **89.6%** | **25.7%** | 6.5% | — |
| measurement OR cosine≥0.60 | 89.6% | 21.9% | 7.7% | 70 |

→ matches quantile recall at **~3× the narrowing and ~3× the precision**, and returns 0 on 70 passes. Dense doc (Engagement, 5.2% used): measurement-alone 92.6% recall @ 43% selected.

**missile_guidance (categorical + image + cosine), 42 ground-truth chunks:**
| signal | recall |
|---|---:|
| measurement (none — no units) | 0% |
| categorical | 43% |
| image-presence | 88% |
| cosine≥0.50 | 81% |
| **categorical + image** | **95%** |

**Rejected alternatives:** C-lexical (field_label OR label_value_line OR unit_gate) — *worse* than quantile (83.5% @ 53.7%); broad pass-keyword matching — recovers recall but balloons selection to 71% (no narrowing).

## 6. Implementation outline (for the plan)

- **Where:** replace the `guarded_quantile` branch in `select_candidates` (`extraction_candidate_scoring.py:724`) with the absolute union. **The inputs are already in-loop:** `select_candidates` receives `MergedCandidate` objects, which carry **`chunk_text` and `source_refs`** directly (`extraction_candidate_scoring.py:73`), plus the per-chunk features (`max_field_cosine`, `unit_gate`, `field_label_norm`) in the components. So the measurement/categorical/image booleans can be computed from `MergedCandidate.chunk_text`/`.source_refs` (or precomputed as features) **with no new endpoint plumbing** — earlier draft note about passing them in was wrong. (The empty-selection RESULT still needs the §3.5 endpoint→worker contract — that's the part that changes outside `select_candidates`.)
- **Per-pass config derivation (single source of truth = the bundle schema):**
  - dimensions ← parse unit suffixes of each pass's field names.
  - categorical phrases ← enum values + the prose-mapping phrases in field `description`s.
  - image flag ← pass has a `_photo`/image-typed field.
  Prefer deriving at bundle-load time over hardcoding, so new bundles work without code edits (cf. generalization guardrails: no equipment names; operate on schema, not literals).
- **`selection_mode`:** add `"absolute_union"` alongside `topk`/`guarded_quantile`; keep `topk` as the byte-identical default so the change is opt-in per manifest.
- **Config:** `cosine_tau` (the τ knob) on `RetrievalProfile`, default 0.55; optional per-pass override. No `k_min`/`k_max`/`quantile_q` for this mode.
- **Retain `narrow_min_doc_tokens` (do NOT drop until §7.4 validation passes)** — it exists because narrowing lost recall on small docs (NMUSAF 24->16). Its token-estimate-undercounts-tables flaw (see cross-ref) is a reason to eventually replace its sizing signal, not to drop the guard now. — the absolute union's 0-to-all behaviour is *expected* to subsume the small-doc fall-open, but that is unproven until §7.4 validation passes — so the gate stays active in the first ship.

## 7. Open decisions (resolve in the plan / with user)

1. **Default `τ`** and whether it is global, per-pass, or per-bundle. (Lean: global 0.55, per-pass override allowed.)
2. **Vocabulary location** — derived-from-schema (preferred) vs a reviewed code constant for the measurement/categorical vocab.
3. **Recall floor?** The design intentionally drops `k_min` (to allow 0). Confirm we accept 0-chunk passes for off-domain docs (they should then terminate as clean ZERO_YIELD, not FAILED — see the legitimate-empty deferred work).
4. **Online validation** — re-run the dense docs (NMUSAF/SA-2/Engagement) end-to-end under `absolute_union` and confirm entity-recall vs the full-doc baseline before flipping default.

## 8. Cross-refs
- `docs/operational/diverse-corpus-ingest-analysis-2026-06.md` — oversized-table batching defect + legitimate-empty→COMPLETE remediation (the 0-chunk passes must terminate clean).
- Live A/B (this session): full-doc vs narrowed quantile on NMUSAF (25/25/26 vs 14/17/17) + SA-2/SR-71 — the "before" baseline.
- `app/services/extraction_candidate_scoring.py::select_candidates` — the implementation site.
