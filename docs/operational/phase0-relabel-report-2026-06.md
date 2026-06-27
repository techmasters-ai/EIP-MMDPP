# Phase 0 relabel report — suspect-label audit + dataset re-export + re-baseline (2026-06-10)

USER-GATE evidence package for the guarded-ranker plan (Task 4 + Task 4b).
Everything below was produced READ-ONLY against postgres
(`ingest.pipeline_pass_outputs`, `ingest.pipeline_runs`, `ingest.documents`) and
ArcadeDB (`ExtractionChunk`). The frozen pre-Phase-0 dataset at
`reports/dataset/` was not touched; the re-export lives at
`reports/dataset_v1_relabel/`.

**History of this report:** the first Task-4 re-export produced 116 positives,
81 "gained". The Task-4 investigation showed 82 of those 116 existed only via an
unhardened substring fallback tier inside the label builder (§2) — label noise,
not signal. Task 4b removed that tier (the shared `value_in_chunk` is now the
single label authority), added a two-decimal `num_variants` rendering, and
re-exported. This document is the CORRECTED baseline; the 116-positive interim
numbers appear only as comparison columns.

- Old dataset: `reports/dataset/bakeoff_dataset.csv` — 1692 candidates, **35 positives**
- Corrected dataset: `reports/dataset_v1_relabel/bakeoff_dataset.csv` — 1692 candidates (identical candidate set), **35 positives**
- vs the ORIGINAL 35: lost **7** · gained **7** · persisting **28** (§4)

---

## 1. What changed in the label machinery

Phase 0 (Task 1-3) hardened `app/services/field_value_grounding.py`: unit
synonyms now match token-bounded ("50 sites" no longer grounds unit "s"; "2391
times" no longer grounds "kg") in both ADJACENT and SAME_CHUNK tiers, and
`num_variants` rejects non-finite values. It also added new groundable suffixes
— `sec` (s/sec/second/seconds), `usec` (µs/us/microsec), `dbi` — plus plural
word forms (knots, metres/meters), unblinding radar_timing
(`nominal_pri_usec`, `nominal_pd_usec`, `scan_period_sec`), six `*_time_sec`
missile_speed_timing fields, `booster_time_sec`/`sustain_time_sec`, and
`gain_dbi`.

Task 4b (this commit) made two further changes:

1. **Removed the substring fallback tier** from the dataset label builder
   (`scripts/a0_captured_separation.py::build_extracted_from_target_grounded`,
   inner `_grounded()`) — see §2. `value_in_chunk` is now the SINGLE label
   authority; the builder adds no tiers of its own.
2. **Added a two-decimal rendering to `num_variants`** (`10.6` → also search
   `"10.60"`; `2391` → also `"2391.00"`), mirrored verbatim in
   `docker/docling-graph/app/provenance.py::_vg_num_variants` (production
   lineage matching changes consistently with the label — intended). This
   recovers the one genuine positive the fallback had been carrying: the V-75
   doc prints `Length (m) 10.60` and `%g` renders 10.6 only (§4).

Commits: `66f73f9` (token-bounded unit matching), `e3c807e` (num_variants
finite-guard + compiled-pattern cache), `ad54232` (docling-graph mirror),
`e33e1ee` (sec/usec/dbi suffixes + plurals + groundable-fields audit),
`89f6bfc` (audit review fixes), `d3d8805` (Task-4 interim re-export + fallback
discovery), Task 4b: `fix(label): value_in_chunk is the single label authority`
(this commit).

## 2. PROMINENT FINDING — the label builder's substring fallback tier (removed)

**What it was.** `build_extracted_from_target_grounded`'s inner `_grounded()`
had a second grounding tier beyond the shared matcher: after `value_in_chunk`
failed, it accepted raw substring co-occurrence — `_strnorm(unit) in text AND
_strnorm(number) in text`, NFKD-stripped, casefolded, with **no token
boundaries and no digit-boundary guards**. It predated the Phase-0 hardening
(it was meant to catch flattened tables, which the shared matcher's SAME_CHUNK
tier now covers with proper guards) and silently bypassed every guard Phase 0
added.

**Why it had to go.** With the new `sec`/`usec` suffixes carrying 1-2 character
synonyms ("s", "us"), `_strnorm("s") in text` is true for virtually any English
text, so any chunk containing the digit of a small extracted value became a
positive. In the Task-4 interim export this minted **82 junk positives of 116**
(71%): every signal degraded (cosine pooled 0.865→0.681, final_score
0.584→0.481 — below coin-flip) and the recall-1.0 frontier collapsed to "keep
100%". It also retained 7 designation-embedded substring artifacts among the
original 35 ("25" inside "S-**125**", "22" inside "SA-**22**", "km" inside
"AK/A**KM**" — §4) that the hardened matcher correctly rejects.

**The fix (user decision, Task 4b).** Delete the tier entirely — the builder
now returns the shared matcher's verdict and nothing else (single label
authority, guarded-ranker spec §2/§3) — and recover the tier's one genuine
catch (V-75 "10.60") with a principled two-decimal `num_variants` rendering
that goes through the guarded matcher like every other variant.

## 3. Old → corrected positives per (doc, pass)

Only (doc, pass) cells with at least one positive in either version are shown;
all other cells are 0 → 0. Candidate counts per cell in parentheses. The
"interim (Task 4)" column is the discarded 116-positive export, for reference.

| doc | pass | old | interim (Task 4) | **corrected (4b)** | Δ vs old |
|---|---|---:|---:|---:|---:|
| Engagement and Fire Control Radars (S/X-band) | missile_kinematics (50) | 16 | 16 | **9** | −7 |
| Engagement and Fire Control Radars (S/X-band) | missile_speed_timing (50) | 0 | 33 | **4** | +4 |
| Engagement and Fire Control Radars (S/X-band) | radar_antenna (50) | 5 | 5 | **5** | 0 |
| Engagement and Fire Control Radars (S/X-band) | radar_power_rf (50) | 3 | 3 | **3** | 0 |
| Engagement and Fire Control Radars (S/X-band) | radar_timing (50) | 0 | 36 | **2** | +2 |
| Images_Demo_Doc | radar_power_rf (16) | 1 | 1 | **1** | 0 |
| S-75 Dvina | missile_kinematics (7) | 1 | 1 | **1** | 0 |
| SA-2 Guideline (RU, С-75 Двина/Десна/Волхов) | missile_airframe (50) | 1 | 1 | **1** | 0 |
| SA-2 Guideline (RU) | missile_kinematics (50) | 1 | 1 | **1** | 0 |
| SA-2 Guideline (RU) | missile_propulsion (50) | 1 | 1 | **1** | 0 |
| SA-2 Guideline (RU) | missile_speed_timing (50) | 2 | 2 | **2** | 0 |
| SA-2 National Museum USAF | missile_kinematics (10) | 1 | 1 | **1** | 0 |
| SA-2 National Museum USAF | missile_propulsion (10) | 0 | 3 | **0** | 0 |
| SA-2_and_SR-71_17_Apr_2020 | missile_speed_timing (42) | 0 | 9 | **1** | +1 |
| SA-2_and_SR-71_17_Apr_2020 | radar_antenna (42) | 1 | 1 | **1** | 0 |
| SA-2_and_SR-71_17_Apr_2020 | radar_power_rf (42) | 1 | 1 | **1** | 0 |
| SNR-75 Wikipedia | (all passes) | 0 | 0 | **0** | 0 |
| V-75 SA-2 GUIDELINE | missile_airframe (6) | 1 | 1 | **1** | 0 |
| **TOTAL** | | **35** | **116** | **35** | **0 net (−7/+7)** |

Per-doc totals (old → corrected): Engagement 24→23, SR-71 2→3, National Museum
1→1, SA-2 RU 5→5, Dvina 1→1, V-75 1→1, Images_Demo 1→1, SNR-75 0→0.

## 4. Lost / gained vs the ORIGINAL 35 — every row explained

**Lost (7)** — all seven are Engagement `missile_kinematics` rows, and they are
exactly the substring artifacts the Task-4 investigation flagged as
fallback-retained (it predicted they "*would have been lost* under the hardened
matcher alone" — they now are):

| doc / pass / chunk | what the fallback latched onto |
|---|---|
| Engagement, missile_kinematics, chunk 3 (references index) | "25" inside "SNR-**125**/S-**125**", "22" inside "SA-**22**", "35" inside "9S**35**M Fire Dome", "3" inside "SA-**3** Goa"; "km" inside "AK/A**KM**" |
| Engagement, missile_kinematics, chunk 32 (SA-3 Low Blow prose) | "25" inside "S-**125**", "3" inside "SA-**3**"; "km" inside "d**km**" |
| Engagement, missile_kinematics, chunk 53 (SA-4 Pat Hand prose) | "22" inside "**220** nmi", "3" inside "1S**3**2"; "км" inside "ф**км**" (waveform name) |
| Engagement, missile_kinematics, chunk 89 (SA-8 Gecko prose) | "45" inside "9S**45**6M3", "3" inside "9K**3**3"; "km" inside "A**KM**" |
| Engagement, missile_kinematics, chunk 91 (SA-8 image prose) | "3" inside "9K**3**3/9K**3**3M2"; "km" inside "A**KM**" |
| Engagement, missile_kinematics, chunk 93 (radar spec table) | "22"/"25" inside "0.**225**", "3" digit-embedded; "km" is a real token but every number match is digit-embedded — SAME_CHUNK's `(?<!\d)N(?!\d)(?!\.\d)` guard rejects them |
| Engagement, missile_kinematics, chunk 162 (Pantsir RCS prose) | "3" inside "5**3** km" — digit-embedded |

The 8th fallback-retained row among the original 35 — **V-75 missile_airframe
chunk 1** (`body_length_m=10.6`, text "Length (m) **10.60**") — was a TRUE
positive and is **retained**, now via the principled path: the Task-4b
two-decimal rendering puts `"10.60"` in `num_variants(10.6)` and the chunk
grounds as `SAME_CHUNK` ("10.60" passes the ≥2-digit and digit-boundary
guards; unit "m" is a real token in "length (m)").

**Gained (7)** — exactly the matcher-grounded (non-fallback) sec/usec gains
identified in Task 4; all pass the hardened `value_in_chunk`:

| doc / pass / chunk | grounding | assessment |
|---|---|---|
| SR-71, missile_speed_timing, chunk 16 | `max_flyout_time_sec=63` ADJACENT: "At 55-**63 seconds** after launch … self-destruct" | genuine spec statement |
| Engagement, missile_speed_timing, chunk 142 | `intra_salvo_time_sec=2.0` ADJACENT: "missiles can be launched **2 seconds** apart" | genuine spec statement |
| Engagement, missile_speed_timing, chunk 122 | `intra_salvo_time_sec=2.0` ADJACENT: "**2 seconds** to sweep a 10° x 7° solid angle" | small-integer coincidence (radar sweep time, not salvo spacing) |
| Engagement, missile_speed_timing, chunk 160 | `scan_period_sec=1.0`-family ADJACENT: "within **1 second** of coordinate transfer" | small-integer coincidence |
| Engagement, missile_speed_timing, chunk 80 | SAME_CHUNK: "PRF is **2.0** kHz" / "~**1.0**°" co-occurring with "°/**sec**" slew-rate units | small-value coincidence |
| Engagement, radar_timing, chunk 80 | SAME_CHUNK (same mechanism as above) | small-value coincidence |
| Engagement, radar_timing, chunk 160 | ADJACENT "**1 second**" | small-integer coincidence |

~2 of the 7 gains are unambiguous spec statements; the other 5 are
small-integer time-value coincidences that the ADJACENT tier permits because
its ≥2-digit guard only applies to SAME_CHUNK. This is the one residual label
softness (§7) — it is bounded (5 rows, 0.3% of candidates), goes through the
guarded matcher, and is a property of the shared production matcher (not a
builder-private tier), so it stays until the ADJACENT-guard discussion.

The interim export's other 74 "gains" (Engagement radar_timing 34,
missile_speed_timing 29, SR-71 8, National Museum 3) were all fallback-only
and are gone.

## 5. Suspect-positive verdict: **legitimate** (unchanged, persists)

The suspect row is in run `28a58eb9` = `SA-2_and_SR-71_17_Apr_2020.pdf`, pass
`radar_antenna`, chunk_index 16, cosine 0.4475. It is **still positive** in the
corrected export, grounded by the hardened matcher (no fallback involved):

- Extracted values: `beamwidth_az_deg=1.1`, `beamwidth_el_deg=1.1`.
- `value_in_chunk({"1.1",…}, ["deg","°","degree","degrees","град"], chunk16)` →
  **ADJACENT** via "1.1" + "degree": *"…At 40 nm range, the **1.1-degree** beam
  width resulted in about 4,400 ft accuracy in azimuth and elevation."*

The chunk opens with missile-airframe prose but its tail is genuine Fan Song
angular-accuracy text; the 1.1° beamwidth stated there is exactly the value the
radar_antenna pass extracted. Verdict: **legitimate** — no removal, no special
rule. Note it is also the **lowest-cosine positive in both label sets** — it
alone sets the raw-cosine recall-1.0 frontier at threshold 0.4475 (§6).

## 6. Re-baselined per-feature AUROC — old vs corrected labels

Conventions (stated exactly as the scripts name them):

- **pooled** — one AUROC over all 1692 rows on the raw feature (recomputed from
  the frozen CSV it reproduces the old numbers exactly: pass_keyword 0.459,
  cosine 0.865, final_score 0.584).
- **mean-per-doc** — AUROC computed within each doc having both classes
  (7 of 8 docs in both versions), then averaged.
- `per_metric_signal` "**honest (LODO)**" — pooled-OOF: univariate LogReg
  out-of-fold probabilities under leave-one-doc-out, one AUROC over the pool.
- `a0_captured_separation` bake-off "**leave-one-doc-out AUROC**" —
  mean-per-fold (per-held-out-doc AUROC, averaged); its "pooled CV AUROC" is
  stratified 5-fold (not group-aware → same-doc leakage, optimistic).

Raw-feature AUROC, old (original 35) vs corrected (Task-4b 35) labels; the
interim-116 pooled column shows what the fallback noise had done:

| feature | pooled old | pooled interim-116 | **pooled corrected** | mean-per-doc old | **mean-per-doc corrected** | per_metric LODO (corrected) |
|---|---:|---:|---:|---:|---:|---:|
| cosine | **0.865** | 0.681 | **0.817** | 0.821 | **0.822** | **0.817** |
| rerank_norm | 0.656 | 0.528 | 0.642 | 0.671 | 0.679 | 0.547 |
| field_label_norm | 0.541 | 0.509 | 0.581 | 0.602 | 0.628 | 0.444 |
| pass_keyword_norm | **0.459** | 0.323 | **0.492** | 0.758 | 0.766 | **0.258** |
| anchor_text_norm | 0.458 | 0.463 | 0.495 | 0.671 | 0.669 | 0.289 |
| anchor_section_norm | 0.598 | 0.575 | 0.625 | 0.706 | 0.712 | 0.555 |
| section_norm | 0.598 | 0.575 | 0.625 | 0.706 | 0.712 | 0.555 |
| is_table | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| pattern_norm | 0.605 | 0.546 | 0.619 | 0.626 | 0.639 | 0.433 |
| negative_norm | 0.624 | 0.566 | 0.637 | 0.569 | 0.573 | 0.698 |
| final_score (production baseline) | **0.584** | 0.481 | **0.595** | 0.727 | 0.736 | n/a |

Headlines: **cosine recovers** (pooled 0.681→0.817 vs old 0.865; mean-per-doc
0.822 ≈ old 0.821; per_metric LODO honest 0.817, verdict "STRONG standalone —
viable as a primary discriminator"). **Keyword stays anti-predictive** (pooled
0.492 ≈ coin-flip; LODO honest 0.258 — worse than chance cross-doc).
final_score stays barely-better-than-coin-flip pooled (0.595).

Model bake-off (`a0_captured_separation --fit-runs <8> --target
lineage_grounded`, outputs in `reports/dataset_v1_relabel/a0_bakeoff_e864ba84.*`):

| model | pooled CV AUROC | LODO mean-per-fold | interim-116 LODO | old reference |
|---|---:|---:|---:|---|
| LogisticRegression | 0.860 | 0.869 | 0.536 | old mean-per-fold **0.852** (frozen `reports/dataset/README.md`) |
| RandomForest | 0.852 | 0.879 | 0.705 | — |
| GradientBoosting | 0.867 | 0.858 | 0.697 | old mean-per-fold 0.885 |
| HistGradientBoosting | 0.807 | **0.881** (best) | 0.666 | — |
| MLP | 0.713 | 0.728 | 0.551 | — |

Every model's LODO returns to (or above) the old-label level; the interim-116
collapse (best 0.705) is fully explained by the fallback noise.

**Recall-1.0 frontier — exact, rank-based** (threshold = min positive score on
leave-one-doc-out out-of-fold probabilities; computed directly because the
script's 41-point linear threshold grid under-resolves the compressed
probability region — its "keep 100% @ thr 0.000" line is a grid artifact, the
exact numbers below are the honest ones):

| ranker | old labels: keep @ recall 1.0 | corrected labels: keep @ recall 1.0 |
|---|---:|---:|
| raw cosine (no model) | **68.7%** (thr 0.4475) | **68.7%** (thr 0.4475 — same suspect SR-71 row is min) |
| LogReg LODO-OOF | 70.7% | 99.2% |
| MLP LODO-OOF | 73.8% | 94.3% |
| HistGB LODO-OOF | 93.7% | 99.4% |
| RF / GBM LODO-OOF | 100% / 100% | 100% / 99.8% |

Two honest observations: (1) the earlier "~24% kept @ recall 1.0" figure
sometimes quoted for this corpus came from the single-doc per-field-lineage
calibration (SNR-75 e79a4866), not from this 8-doc pooled dataset — on this
dataset even the OLD labels never had a 24% model frontier (best exact 70.7%).
(2) On both label sets, learned-probability thresholds are frontier-fragile
(single hard positives score near zero OOF), while **raw cosine gives the best
recall-1.0 frontier at ~69% kept** — i.e. a cosine floor alone can cut ~31% of
candidate (pass,chunk) sends without losing a positive, on either label set.
Mid-recall, the corrected labels are far friendlier: the univariate cosine
threshold (per_metric LODO) keeps 29% of chunks at recall 0.77 (savings 71%).

Caveat: the auto-generated `reports/dataset_v1_relabel/README.md` embeds the
old narrative numbers (0.852/0.885/0.706) as hardcoded prose from the export
script; trust `dataset_meta.json` + this report for v1_relabel numbers.

## 7. Implications for Phase 1

- **The dataset is now fit for Phase 1 fitting.** Single label authority holds:
  every positive is grounded by the shared, hardened `value_in_chunk` (28
  persisting + 7 sec/usec gains); the 7 substring artifacts are gone; dataset
  size is unchanged (35/1692, 2.1%) so old/new comparisons stay like-for-like.
- **Sanity gates (Task 4b) all PASS:** (a) positives 35 ∈ [25,45]; (b) cosine
  pooled AUROC 0.817 ≥ 0.80; (c) **35/35** positives' chunks pass the
  unit-token gate predicate (digit present + `has_unit_token` over the union of
  the pass's schema-field units — the Task-5 G1 gate cannot drop a known
  positive by construction); (d) matcher + audit + mirror test suites green
  (67 tests).
- **Cosine-led, keyword-guarded premise confirmed on clean labels:** cosine is
  the only STRONG standalone signal (LODO honest 0.817); keyword remains
  anti-predictive cross-doc (0.258). Nothing rehabilitates lexical ranking;
  keywords stay guards, not rankers.
- **Selection design hint:** raw cosine's recall-1.0 frontier (~69% kept) beats
  every learned-probability threshold on both label sets; learned models only
  help mid-recall. Favor cosine-floor + unit-token-gate + dynamic-k over a
  global learned threshold for the recall-critical cut.
- **Residual label softness (bounded, documented):** 5 of the 7 gained
  positives are small-integer ADJACENT time-value coincidences ("2 seconds",
  "1 second"). Worth a discussion gate on extending the ≥2-digit (or a
  value-specificity) guard to the ADJACENT tier — but it changes the
  PRODUCTION matcher, so it is a deliberate follow-up, not a Task-4b rider.
- **`is_table` is still a dead stub** (always 0; AUROC 0.500) — wiring it
  (issue #70) remains the top feature-engineering candidate for the
  table-derived positives.

---

### Reproduction

```
A0_DATABASE_URL=postgresql+psycopg2://eip:eip_secret@localhost:5437/eip \
python3 -m scripts.export_bakeoff_dataset --runs <8 ids from dataset_meta.json> \
  --target lineage_grounded --out-dir reports/dataset_v1_relabel

A0_DATABASE_URL=... python3 -m scripts.a0_captured_separation \
  --fit-runs <8 ids> --target lineage_grounded --out-dir reports/dataset_v1_relabel

A0_DATABASE_URL=... python3 -m scripts.per_metric_signal --runs <8 ids> --target lineage_grounded

PYTHONPATH="docker/docling-graph/repo:$PYTHONPATH" python3 -m pytest \
  tests/unit/test_field_value_grounding.py tests/unit/test_groundable_fields_audit.py \
  docker/docling-graph/tests/test_value_grounding_mirror.py
```

Run IDs (run → doc mapping in both `dataset_meta.json` files):
`e864ba84` SA-2 Guideline RU · `28a58eb9` SA-2_and_SR-71 · `58767f3f` SA-2
National Museum · `e79a4866` SNR-75 Wikipedia · `ff35b0e2` S-75 Dvina ·
`295aea8e` V-75 SA-2 GUIDELINE · `de6f44d9` Images_Demo_Doc · `1329caf5`
Engagement and Fire Control Radars.
