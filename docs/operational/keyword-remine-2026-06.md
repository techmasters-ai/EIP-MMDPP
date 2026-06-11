# Keyword Re-mine Against Value-Grounded Labels — June 2026

## Interpretation

**Run date:** 2026-06-10  
**Positives source:** `reports/dataset_v1_relabel/bakeoff_dataset.csv` (35 positives across 8 runs)  
**Filters:** pos-fire ≥ 0.15, lift ≥ 0.10, docspread ≥ 2 (`--allow-units` active)

**Passes with candidates:** 5 of 9 — `radar_power_rf` (17), `radar_antenna` (24),
`missile_kinematics` (57), `missile_airframe` (3), `missile_speed_timing` (44).

**Passes with zero candidates:** 4 — `radar_modulation` (0 positives total; skip),
`radar_timing` (2 positives, both in a single run → docspread < 2 for every term),
`missile_propulsion` (1 positive, same constraint), `missile_guidance` (0 positives).
This is expected: 35 positives across 8 docs and 9 passes leaves several passes with
too few cross-document coverage to pass the docspread ≥ 2 gate. Lowering the gate
would risk SA-2-specific overfit.

**Unit tokens (`--allow-units`):** "kw" fires in `radar_power_rf` (posf=0.80, lift=+0.79,
docs=2) and "mhz" (posf=0.60, lift=+0.60, docs=2); "km" fires in `missile_kinematics`
(posf=0.75, lift=+0.71, docs=2); "sec", "seconds", "metres" fire in
`missile_speed_timing`. Units are not dominating the lists but they do appear and
would have been silently dropped without `--allow-units`. They are plausible signals
(a chunk mentioning "kw" is a strong cue for RF power content) but marginal given
only 2-doc spread — hold for curation.

**Genuinely new prose terms vs curated lists:**
- `radar_antenna`: "elevation azimuth" bigram (posf=0.50, lift=+0.49), "scanners"
  (posf=0.33, lift=+0.33, 2 docs), "tilted" (0.30, 2 docs) — these are not in current
  manifests and generalize across the Engagement doc (24 positives) + SA-2_SR-71.
- `missile_airframe`: "length", "weight", "specifications" all fire at posf=1.0 with
  lift > 0.88 across 2 docs — the most confident candidates in the entire run; likely
  already covered by field label matching but worth a manifest check.
- `missile_kinematics`: "cruise" (posf=0.33, 2 docs), "engagement envelope" bigram —
  the bigram is clean and cross-document.
- `missile_speed_timing`: "command link" bigram (posf=0.43, lift=+0.37, 2 docs),
  "trigger" unigram (posf=0.29, 2 docs), "early variants" bigram (posf=0.29, 2 docs)
  — "early variants" is interesting but risks being a doc-description artifact.

**Curation read — merit for manifest addition:**
- **Likely worth adding:** "elevation azimuth" and "azimuth elevation" bigrams for
  `radar_antenna` (highest lifts, 2 doc spread, structural not equipment-specific);
  "length" + "weight" + "specifications" for `missile_airframe` (confirm not already
  present); "engagement envelope" for `missile_kinematics`.
- **Hold/monitor:** unit tokens ("kw", "mhz", "km", "sec") — signal is real but 2-doc
  spread is the bare minimum; wait for the 21-doc collection before committing.
- **Reject:** "used" (artifact of prose like "missiles used"), "provided/providing" —
  generic prose that will fire across many unrelated chunks at scale.
- **Passes with 0 candidates are working as designed** — not a bug, just thin corpus.

---

## Raw Output

```
## radar_power_rf  (5 pos / 318 neg)  candidates: 17
    lift  posf  negf docs  term
   +0.81  1.00  0.19    3  frequency
   +0.79  0.80  0.01    2  kw
   +0.72  0.80  0.08    2  power
   +0.60  0.60  0.00    2  mhz
   +0.57  0.80  0.23    3  range
   +0.38  0.40  0.02    2  ratio
   +0.38  0.60  0.22    2  used
   +0.37  0.40  0.03    2  jamming
   +0.35  0.40  0.05    2  coverage
   +0.34  0.40  0.06    2  degree
   +0.34  0.40  0.06    2  provide
   +0.32  1.00  0.68    3  radar
   +0.32  0.40  0.08    2  providing
   +0.22  0.40  0.18    2  provided
   +0.20  0.60  0.40    2  target
   +0.15  0.60  0.45    2  antenna
   +0.14  0.40  0.26    2  acquisition

## radar_antenna  (6 pos / 317 neg)  candidates: 24
    lift  posf  negf docs  term
   +0.90  1.00  0.10    2  elevation
   +0.75  0.83  0.08    2  azimuth
   +0.60  0.67  0.07    2  beam
   +0.49  0.50  0.01    2  elevation azimuth
   +0.46  0.50  0.04    2  azimuth elevation
   +0.45  0.50  0.05    2  degree
   +0.43  0.67  0.23    2  range
   +0.39  0.83  0.45    2  antenna
   +0.33  0.33  0.00    2  range accuracy
   +0.33  0.33  0.00    2  scanners
   +0.33  0.33  0.01    2  range azimuth
   +0.32  0.33  0.02    2  effective
   +0.31  0.33  0.02    2  accuracy
   +0.30  0.33  0.03    2  tilted
   +0.30  0.33  0.03    2  proximity fuse
   +0.29  0.33  0.04    2  receive
   +0.29  0.33  0.04    2  fuse
   +0.27  0.67  0.39    2  target

## radar_modulation: 0 positives — skip

## radar_timing  (2 pos / 321 neg)  candidates: 0
    lift  posf  negf docs  term

## missile_kinematics  (12 pos / 311 neg)  candidates: 57
    lift  posf  negf docs  term
   +0.71  0.75  0.04    2  km
   +0.61  0.83  0.22    2  range
   +0.37  0.50  0.13    2  performance
   +0.34  0.50  0.16    2  against
   +0.31  0.33  0.02    2  cruise
   +0.31  0.83  0.52    4  missile
   +0.29  0.42  0.13    2  missiles
   +0.28  0.42  0.14    3  command
   +0.25  0.58  0.33    2  engagement
   +0.25  0.25  0.00    2  range altitude
   +0.24  0.25  0.01    2  engagement envelope
   +0.23  0.25  0.02    3  rotated
   +0.22  0.25  0.03    2  envelope
   +0.22  0.25  0.03    2  various
   +0.21  0.25  0.04    2  maximum
   +0.19  0.33  0.14    2  capability
   +0.18  0.25  0.07    2  altitude
   +0.18  0.25  0.07    2  flight

## missile_airframe  (2 pos / 321 neg)  candidates: 3
    lift  posf  negf docs  term
   +0.98  1.00  0.02    2  length
   +0.97  1.00  0.03    2  weight
   +0.88  1.00  0.12    2  specifications

## missile_propulsion  (1 pos / 322 neg)  candidates: 0
    lift  posf  negf docs  term

## missile_guidance: 0 positives — skip

## missile_speed_timing  (7 pos / 316 neg)  candidates: 44
    lift  posf  negf docs  term
   +0.50  0.71  0.22    3  used
   +0.49  0.57  0.08    2  targets
   +0.49  0.57  0.09    2  azimuth
   +0.46  0.57  0.11    2  elevation
   +0.43  0.57  0.14    2  command
   +0.41  0.43  0.02    2  width
   +0.41  0.43  0.02    2  seconds
   +0.41  0.43  0.02    2  metres
   +0.40  0.43  0.03    2  sec
   +0.38  0.43  0.04    2  azimuth elevation
   +0.38  0.43  0.05    3  point
   +0.38  0.43  0.05    2  warhead
   +0.37  0.43  0.05    2  command link
   +0.37  0.43  0.06    2  link
   +0.34  0.57  0.23    2  range
   +0.30  0.43  0.13    2  missiles
   +0.29  0.29  0.00    2  trigger
   +0.28  0.29  0.01    2  early variants

filters: pos-fire≥0.15, lift≥0.1, docs≥2 (generalizable). Numbers/units/stopwords/designations excluded (non-unit). unit-tokens=ALLOWED positives-source=csv:bakeoff_dataset.csv. REVIEW before committing.

REVIEW ONLY — hand-curate into manifests via inject/union; see guarded-ranker spec §5.4.
```
