# Phase 0 relabel report — suspect-label audit + dataset re-export + re-baseline (2026-06-10)

USER-GATE evidence package for the guarded-ranker plan (Task 4). Everything below
was produced READ-ONLY against postgres (`ingest.pipeline_pass_outputs`,
`ingest.pipeline_runs`, `ingest.documents`) and ArcadeDB (`ExtractionChunk`).
The frozen pre-Phase-0 dataset at `reports/dataset/` was not touched; the
re-export lives at `reports/dataset_v1_relabel/`.

- Old dataset: `reports/dataset/bakeoff_dataset.csv` — 1692 candidates, **35 positives**
- New dataset: `reports/dataset_v1_relabel/bakeoff_dataset.csv` — 1692 candidates (identical candidate set), **116 positives**
- Lost: **0** · Gained: **81** (of which only **7** are grounded by the hardened matcher; **74** come from an unhardened substring fallback tier in the label builder — see §3/§4)

---

## 1. What changed in the label machinery

Phase 0 hardened `app/services/field_value_grounding.py`: unit synonyms now match
token-bounded ("50 sites" no longer grounds unit "s"; "2391 times" no longer
grounds "kg") in both ADJACENT and SAME_CHUNK tiers, and `num_variants` rejects
non-finite values. It also added new groundable suffixes — `sec`
(s/sec/second/seconds), `usec` (µs/us/microsec), `dbi` — plus plural word forms
(knots, metres/meters), unblinding radar_timing (`nominal_pri_usec`,
`nominal_pd_usec`, `scan_period_sec`), six `*_time_sec` missile_speed_timing
fields, `booster_time_sec`/`sustain_time_sec`, and `gain_dbi`.

Commits: `66f73f9` (token-bounded unit matching), `e3c807e` (num_variants
finite-guard + compiled-pattern cache), `ad54232` (docling-graph mirror of the
token-bounded matcher), `e33e1ee` (sec/usec/dbi suffixes + plurals +
groundable-fields audit), `89f6bfc` (audit review fixes).

## 2. Old → new positives per (doc, pass)

Only (doc, pass) cells with at least one positive in either version are shown;
all other cells are 0 → 0. Candidate counts per cell in parentheses.

| doc | pass | old | new | Δ |
|---|---|---:|---:|---:|
| Engagement and Fire Control Radars (S/X-band) | missile_kinematics (50) | 16 | 16 | 0 |
| Engagement and Fire Control Radars (S/X-band) | missile_speed_timing (50) | 0 | 33 | +33 |
| Engagement and Fire Control Radars (S/X-band) | radar_antenna (50) | 5 | 5 | 0 |
| Engagement and Fire Control Radars (S/X-band) | radar_power_rf (50) | 3 | 3 | 0 |
| Engagement and Fire Control Radars (S/X-band) | radar_timing (50) | 0 | 36 | +36 |
| Images_Demo_Doc | radar_power_rf (16) | 1 | 1 | 0 |
| S-75 Dvina | missile_kinematics (7) | 1 | 1 | 0 |
| SA-2 Guideline (RU, С-75 Двина/Десна/Волхов) | missile_airframe (50) | 1 | 1 | 0 |
| SA-2 Guideline (RU) | missile_kinematics (50) | 1 | 1 | 0 |
| SA-2 Guideline (RU) | missile_propulsion (50) | 1 | 1 | 0 |
| SA-2 Guideline (RU) | missile_speed_timing (50) | 2 | 2 | 0 |
| SA-2 National Museum USAF | missile_kinematics (10) | 1 | 1 | 0 |
| SA-2 National Museum USAF | missile_propulsion (10) | 0 | 3 | +3 |
| SA-2_and_SR-71_17_Apr_2020 | missile_speed_timing (42) | 0 | 9 | +9 |
| SA-2_and_SR-71_17_Apr_2020 | radar_antenna (42) | 1 | 1 | 0 |
| SA-2_and_SR-71_17_Apr_2020 | radar_power_rf (42) | 1 | 1 | 0 |
| SNR-75 Wikipedia | (all passes) | 0 | 0 | 0 |
| V-75 SA-2 GUIDELINE | missile_airframe (6) | 1 | 1 | 0 |
| **TOTAL** | | **35** | **116** | **+81** |

Per-doc totals: Engagement 24→93, SR-71 2→11, National Museum 1→4, SA-2 RU 5→5,
Dvina 1→1, V-75 1→1, Images_Demo 1→1, SNR-75 0→0.

## 3. Lost positives — zero, and why that is NOT what it looks like

**No old positive was lost.** This is *not* because the token-bounded hardening
found nothing to remove. The dataset label is built by
`scripts/a0_captured_separation.py::build_extracted_from_target_grounded`, whose
inner `_grounded()` has a second tier beyond the shared matcher: a plain
**substring co-occurrence fallback** (`_strnorm(unit) in text and
_strnorm(number) in text`) that was never token-bounded. Rows that the hardened
`value_in_chunk` now correctly rejects are silently retained by that fallback.

Decomposing every new positive into *matcher-grounded* (passes the hardened
`value_in_chunk`) vs *fallback-only* (label exists solely via the substring
tier) — the decomposition reproduces the exported labels with **0 mismatches**
on all 8 runs:

| positives (new export) | matcher-grounded | fallback-only |
|---|---:|---:|
| 35 persisting | 27 | **8** |
| 81 gained | 7 | **74** |
| **116 total** | **34** | **82 (71%)** |

The 8 persisting fallback-only rows are exactly the substring artifacts Task 1
targeted — they *would have been lost* under the hardened matcher alone:

| doc / pass / chunk | grounding pair(s) | why the hardened matcher rejects it / what the substring fallback latches onto |
|---|---|---|
| Engagement, missile_kinematics, chunk 3 (references index) | max_altitude_km=22, 25; max_intercept_km=35; min_intercept_km=3 | "25" inside "SNR-**125** Low Blow / S-**125**", "22" inside "SA-**22**", "35" inside "9S**35**/9S**35**M Fire Dome", "3" inside "SA-**3** Goa"; unit "km" as substring of "9K33M2/M3 Osa AK/A**KM**". A reference listing with no kinematics content. |
| Engagement, missile_kinematics, chunk 32 (SA-3 Low Blow antenna prose) | max_altitude_km=25; min_intercept_km=3 | "25" inside "S-**125**", "3" inside "SA-**3**"; "km" as substring of "d**km** (ballistic)". |
| Engagement, missile_kinematics, chunk 53 (SA-4 Pat Hand prose) | max_altitude_km=22; min_intercept_km=3 | "22" inside "**220** nmi", "3" inside "1S**3**2 Pat Hand"; Cyrillic unit "км" as substring of "ф**км**" (фазокодовая манипуляция — a waveform name). |
| Engagement, missile_kinematics, chunk 89 (SA-8 Gecko prose) | max_intercept_km=45; min_intercept_km=3 | "45" inside "9S**45**6M3 computer", "3" inside "9K**3**3"; "km" inside "AK/A**KM**". |
| Engagement, missile_kinematics, chunk 91 (SA-8 Gecko image prose) | min_intercept_km=3 | "3" inside "9K**3**3/9K**3**3M2"; "km" inside "AK/A**KM**". |
| Engagement, missile_kinematics, chunk 93 (Engagement-radar spec table) | max_altitude_km=22, 25; min_intercept_km=3 | "22"/"25" inside "0.**225**" (IF figure), "3" inside "10 -1**3**" / "[MHz] **3**0.0"; "km" is a real token here ("Target Range [km]") but the numbers are coincidental digit-embeddings — the new SAME_CHUNK digit-boundary guard `(?<!\d)N(?!\d)(?!\.\d)` rejects them. |
| Engagement, missile_kinematics, chunk 162 (Pantsir RCS plot prose) | min_intercept_km=3 | "3" inside "5**3** km"; "km" is a real token, the value match is digit-embedded. |
| V-75 SA-2 GUIDELINE, missile_airframe, chunk 1 (spec table) | body_length_m=10.6 | Text reads "Length (m) **10.60**" — semantically a TRUE positive, but `num_variants(10.6)` = {"10.6"} and the trailing-zero rendering "10.60" fails the `(?!\d)` digit-boundary guard; unit-then-number order also defeats ADJACENT. Only the substring fallback keeps it. |

So: 7 of the 8 would-be-lost rows are genuine substring false positives
(designation-embedded digits — SA-22, S-125, 9S35, 9K33 — plus unit "km"
hiding inside "AKM"/"dkm"/"фкм"), and 1 (V-75) is a true positive that the
hardened matcher misses due to a trailing-zero numeric rendering. At dataset
level nothing was lost only because the fallback tier bypasses the hardening.

## 4. Gained positives by pass

All 81 gains come from the new `sec`/`usec` suffixes activating previously
ungroundable fields (no `dbi` gains; no gains from the plural word forms).

| doc / pass | gained | matcher-grounded | fallback-only | activating fields |
|---|---:|---:|---:|---|
| Engagement / radar_timing | 36 | 2 | 34 | scan_period_sec, nominal_pri_usec, nominal_pd_usec |
| Engagement / missile_speed_timing | 33 | 4 | 29 | intra_salvo_time_sec (2.0), max_flyout_time_sec |
| SR-71 / missile_speed_timing | 9 | 1 | 8 | max_flyout_time_sec (63), intra_salvo_time_sec (5.0) |
| National Museum / missile_propulsion | 3 | 0 | 3 | booster_time_sec (6.0) |
| **total** | **81** | **7** | **74** |

**Fallback-only gains are noise.** The `sec` synonym list includes the
single letter "s" and `usec` includes "us"; the fallback's substring check
makes `_strnorm("s") in text` true for virtually any English text, so any chunk
containing the digit of a small extracted value becomes positive.
Representative examples:

- SR-71 chunk 3 (`intra_salvo_time_sec=5.0`): *"'The SA-2 and SR-71' By John A.
  Schell 17 April 2020. After three more U-2 overflights … the Soviets made an
  official protest…"* — "5" appears somewhere, "s" appears everywhere.
- National Museum chunk 8 (`booster_time_sec=6.0`): *"Technical notes: Range:
  minimum 5 miles… ceiling: up to 60,000 ft. warhead: 288-lb…"* — "6" inside
  "60,000", unit "s" as substring.

**Matcher-grounded gains (7) are mixed.** Two are clearly genuine:

- SR-71 chunk 16, `max_flyout_time_sec=63` ADJACENT: *"At 55-**63 seconds**
  after launch if the missile did not intercept a target, the warhead would
  automatically self-destruct."*
- Engagement chunk 142, `intra_salvo_time_sec=2.0` ADJACENT: *"missiles can be
  launched **2 seconds** apart."*

The other five are small-integer coincidences the ADJACENT tier permits because
its ≥2-digit guard only applies to SAME_CHUNK:

- Engagement chunk 122, `intra_salvo_time_sec=2.0` ADJACENT: *"will take **2
  seconds** to sweep a 10° x 7° az/elev solid angle"* — a radar sweep time, not
  salvo spacing.
- Engagement chunk 160, `scan_period_sec=1.0` ADJACENT: *"90% probability of
  initial target acquisition within **1 second** of coordinate transfer"* — not
  a scan period.
- Engagement chunk 80 (SAME_CHUNK ×2): "PRF is **2.0** kHz" / "~**1.0**°
  mainlobe" co-occurring with "°/**sec**" slew-rate units.

## 5. Suspect-positive verdict: **legitimate** (and it persists)

The suspect row is in run `28a58eb9` = `SA-2_and_SR-71_17_Apr_2020.pdf`
(the task description's `58767f3f` is the National Museum doc per
`dataset_meta.json`; the SR-71 doc is `28a58eb9`), pass `radar_antenna`,
chunk_index 16, cosine 0.4475. It is **still positive** in the re-export, and
the grounding is **matcher-grounded, not fallback**:

- Extracted values (latest attempt of `extract_pass_response_json`, run
  `28a58eb9`, pass `radar_antenna`): `beamwidth_az_deg=1.1` and
  `beamwidth_el_deg=1.1`.
- `value_in_chunk({"1.1"}, ["deg","°","degree","degrees","град"], chunk16)` →
  **ADJACENT** via pair "1.1" + "degree": *"…At 40 nm range, the **1.1-degree**
  beam width resulted in about 4,400 ft accuracy in azimuth and elevation."*

The chunk was flagged because it *opens* with missile-airframe prose ("second
stage was 1.5 feet in diameter and 35 feet long… 88,000 feet"), but it is a
mixed-topic chunk whose tail is genuine Fan Song angular-accuracy text: the
1.1° beamwidth stated there is exactly the value the radar_antenna pass
extracted. Verdict: **legitimate** — no removal, no special rule needed for this
row. (The generic over-matching concern it raised is real, but it lives in the
fallback tier and the single-digit ADJACENT gap documented in §3/§4, not in
this row.)

## 6. Re-baselined per-feature AUROC — old vs new labels

Conventions (stated exactly as the scripts name them):

- **pooled** — one AUROC over all 1692 rows on the raw feature (the old
  headline convention; recomputed from the frozen CSV it reproduces the old
  numbers exactly: pass_keyword 0.459, cosine 0.865, final_score 0.584).
- **mean-per-doc** — AUROC computed within each doc having both classes
  (7 of 8 docs in both versions), then averaged.
- `per_metric_signal` "**honest (LODO)**" — pooled-OOF: univariate LogReg
  out-of-fold probabilities under leave-one-doc-out, one AUROC over the pool.
- `a0_captured_separation` bake-off "**leave-one-doc-out AUROC**" —
  mean-per-fold (per-held-out-doc AUROC, averaged); its "pooled CV AUROC" is
  stratified 5-fold (not group-aware → same-doc leakage, optimistic).

Raw-feature AUROC, old (35 pos) vs new (116 pos) labels:

| feature | pooled old | pooled new | mean-per-doc old | mean-per-doc new | per_metric LODO (new) |
|---|---:|---:|---:|---:|---:|
| cosine | **0.865** | **0.681** | 0.821 | 0.744 | 0.674 |
| rerank_norm | 0.656 | 0.528 | 0.671 | 0.623 | 0.351 |
| field_label_norm | 0.541 | 0.509 | 0.602 | 0.619 | 0.299 |
| pass_keyword_norm | **0.459** | **0.323** | 0.758 | 0.662 | 0.311 |
| anchor_text_norm | 0.458 | 0.463 | 0.671 | 0.619 | 0.248 |
| anchor_section_norm | 0.598 | 0.575 | 0.706 | 0.636 | 0.511 |
| section_norm | 0.598 | 0.575 | 0.706 | 0.636 | 0.511 |
| is_table | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| pattern_norm | 0.605 | 0.546 | 0.626 | 0.603 | 0.277 |
| negative_norm | 0.624 | 0.566 | 0.569 | 0.565 | 0.443 |
| final_score (production baseline) | **0.584** | **0.481** | 0.727 | 0.685 | n/a |

(No old-label `per_metric_signal` column: that script recomputes labels live
from the DB, which now yields the new labels; old-label comparisons use the
frozen CSV.)

Model bake-off (`a0_captured_separation --fit-runs <8> --target
lineage_grounded`, outputs in `reports/dataset_v1_relabel/a0_bakeoff_e864ba84.*`):

| model | pooled CV AUROC (new) | LODO mean-per-fold (new) | old reference |
|---|---:|---:|---|
| LogisticRegression | 0.746 | 0.536 | old mean-per-fold **0.852**, pooled-OOF ≈0.706 (frozen `reports/dataset/README.md`) |
| RandomForest | 0.863 | **0.705** (best) | — |
| GradientBoosting | 0.849 | 0.697 | old mean-per-fold 0.885 |
| HistGradientBoosting | 0.830 | 0.666 | — |
| MLP | 0.772 | 0.551 | — |

**Recall-1.0 frontier:** old honest LODO frontier was ~**24% of chunks kept @
recall 1.0**; on the new labels the best-model LODO frontier **collapses to
keep 100% of chunks @ recall 1.0 (threshold 0.000)** — i.e. zero savings is
achievable without dropping a positive. With fallback-noise positives scattered
across topically unrelated chunks (lowest-cosine chunks included), no ranking
can cut anything at recall 1.0.

Caveat: the auto-generated `reports/dataset_v1_relabel/README.md` embeds the
old narrative numbers (0.852 / 0.706) because they are hardcoded prose in the
export script; trust `dataset_meta.json` + this report for v1_relabel numbers.

## 7. Implications for Phase 1

- **Do not fit Phase 1 on `dataset_v1_relabel` as-is.** 82 of 116 positives
  (71%) exist only via the label builder's unhardened substring fallback tier
  (`scripts/a0_captured_separation.py::build_extracted_from_target_grounded`,
  inner `_grounded()`), which the new 1-2 character unit synonyms ("s", "us")
  turn into a near-universal matcher. Every signal degrades under these labels
  (cosine pooled 0.865→0.681, final_score 0.584→0.481 — below coin-flip) and
  the recall-1.0 frontier collapses from ~24% kept to 100% kept. Proposed
  principled fix (NOT implemented, per task constraints): delete the fallback
  tier so the shared hardened `value_in_chunk` is the single label authority —
  it already contains a principled SAME_CHUNK tier (token-bounded unit +
  digit-boundary-guarded ≥2-digit number) — and recover the one real loss
  (V-75 "10.60" vs 10.6) by extending `num_variants` with trailing-zero decimal
  renderings. Secondary rule worth discussing: extend the ≥2-digit guard (or a
  value-specificity guard) to the ADJACENT tier for generic small-integer time
  values ("2 seconds", "1 second"), which produced 5 of the 7 matcher-grounded
  gains as coincidences.
- **Keyword anti-predictiveness persists and worsens** (pooled 0.459→0.323;
  per_metric LODO 0.311 with negative coefficient sign). Nothing in Phase 0
  rehabilitates lexical keyword features; the Phase 1 guarded-ranker premise
  (cosine-led, keyword-guarded) still stands.
- **Cosine holds as the only generalizing signal, but is diluted by label
  noise** (pooled 0.865→0.681; mean-per-doc 0.821→0.744). On a matcher-only
  label set (34 positives: 27 persisting + 7 gained) cosine's standing should
  recover toward the old numbers — re-export after the fallback decision and
  re-check before fitting.
- **Zero-positive passes did activate** (radar_timing 0→36, missile_speed_timing
  0→33/+9, missile_propulsion 0→3) — the suffix work did its job — but only 7
  of those 81 gains survive the hardened matcher, and only ~2 of those 7 are
  unambiguous spec statements. The honest matcher-only dataset would be ~34
  positives / 1692, i.e. the new suffixes roughly replace the 8 substring
  artifacts with 7 time-field positives at equal dataset size.

---

### Reproduction

```
A0_DATABASE_URL=postgresql+psycopg2://eip:eip_secret@localhost:5437/eip \
python3 -m scripts.export_bakeoff_dataset --runs <8 ids from dataset_meta.json> \
  --target lineage_grounded --out-dir reports/dataset_v1_relabel

A0_DATABASE_URL=... python3 -m scripts.a0_captured_separation \
  --fit-runs <8 ids> --target lineage_grounded --out-dir reports/dataset_v1_relabel

A0_DATABASE_URL=... python3 -m scripts.per_metric_signal --runs <8 ids> --target lineage_grounded
```

Run IDs (run → doc mapping in both `dataset_meta.json` files):
`e864ba84` SA-2 Guideline RU · `28a58eb9` SA-2_and_SR-71 · `58767f3f` SA-2
National Museum · `e79a4866` SNR-75 Wikipedia · `ff35b0e2` S-75 Dvina ·
`295aea8e` V-75 SA-2 GUIDELINE · `de6f44d9` Images_Demo_Doc · `1329caf5`
Engagement and Fire Control Radars.
