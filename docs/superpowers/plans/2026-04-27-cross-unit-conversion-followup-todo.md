# Cross-Unit Conversion — Follow-up TODO

> **Status: OPTION A COMPLETE (2026-04-27).**
>
> Implementation: commit `6fa8f80` (`feat(docling-graph): cross-unit conversion in numeric evidence helper (Option A)`).
> Integration tests: commit `d9c7cf0` (`test(extraction): cross-unit smoke harnesses for radar + missile`).
>
> Live verification: 4/4 radar + 2/2 missile cross-unit smoke tests pass. The two original failure cases from the radar harder-test investigation (Tombstone's `nominal_rf_mhz=10000` from "10 GHz" and `tx_peak_power_kw=1400` from "1.4 megawatts") now populate end-to-end.
>
> **Remaining out-of-scope** (would need Option C or schema changes, not Option A): logarithmic conversions (dBW ↔ dBm, dBi ↔ dBd), Mach ↔ m/s (requires altitude/temp), generalized lbs ↔ kg (already handled case-specifically by `_mechanically_supported_missile_fields()`).

**Surfaced by:** harder-test run against the rebuilt docling-graph after the radar manifest cutover (commits a88c353 + 833c7eb). Result was 21/23 probes passed with 2 documented-limitation hits, both cross-unit:

- "Tombstone operates at 10 GHz" → expected `nominal_rf_mhz=10000.0`, got `null`
- "Tombstone peak transmitter output of 1.4 megawatts" → expected `tx_peak_power_kw=1400.0`, got `null`

## The problem

A numeric value stated in a non-canonical unit (GHz when the field is `_mhz`, megawatts when the field is `_kw`, tonnes when the field is `_kg`, kilometers when the field is `_m`, etc.) gets dropped by the §4.8 evidence-gate even when the LLM extracts the value correctly with the conversion applied.

**Two coupled mechanisms cause the drop:**

1. **LLM-side.** The field descriptions for numeric fields say "See Unit Policy in DELTA_SYSTEM_PROMPT for conversions." Whether the prompt-rules block actually nudges gemma4:31b strongly enough to convert "10 GHz" → 10000 has not been verified directly. The harder test result alone can't distinguish "LLM didn't convert" from "LLM converted, gate nulled."

2. **Evidence-gate side.** `value_match_candidates(value, field_name)` in `docker/docling-graph/app/_numeric_evidence.py` generates only same-unit-suffix variants. For `value=10000.0` and `field_name="nominal_rf_mhz"`, the candidate forms are `["10000", "10000.0", "10000 MHz", "10000Mhz", "10000mhz", ...]`. None of these appear in the source text "operates at 10 GHz." So `value_is_supported_by_text` returns False and `_clear_unsupported_radar_properties` nulls the value.

**The "out of scope for Session 1" docstring** in `_numeric_evidence.py` documented this deliberately:

> Cross-unit conversion (1.5 tonnes <-> 1500 kg, 43 km <-> 43000 m, etc.) is OUT OF SCOPE for Session 1. If a doc states a value in a non-canonical unit and the LLM doesn't normalize, this predicate returns False and the caller will null the value. Tracked as Session 2 follow-up if false-negatives become a real problem in production.

This TODO IS that follow-up.

---

## Concrete observed examples

From the harder test (Tall King + Tombstone, commit-equivalent of 833c7eb run):

| Source text | Field | Expected | Got |
|---|---|---|---|
| "10 GHz" | `nominal_rf_mhz` | 10000.0 | null |
| "1.4 megawatts" | `tx_peak_power_kw` | 1400.0 | null |

Plausible additional cases (not yet probed):

| Source text | Field | Expected | Likely result |
|---|---|---|---|
| "1.5 tonnes" / "1500 kg" | `total_mass_kg` (missile) | 1500.0 | null on "1.5 tonnes" |
| "43 km" / "43000 m" | (any `_m` field stating in km) | 43000.0 | null on "43 km" |
| "Mach 3.5" / "1200 m/s" | `max_speed_mps` (missile) | 1200.0 | null on "Mach 3.5" |
| "2300 lbs" / "1043 kg" | `total_mass_kg` (missile) | 1043.x | possibly preserved by `_mechanically_supported_missile_fields()` regex (existing behavior) |

The missile postprocessor's existing `_mechanically_supported_missile_fields()` already has narrow regex paths for specific unit conversions ("WEIGHT: X LBS" → kg). That's a partial precedent — but it's case-specific and doesn't generalize.

---

## Three architectural options (ordered by complexity)

### Option A: Real unit conversion in `value_match_candidates`

Extend `_UNIT_HINTS_BY_SUFFIX` and `value_match_candidates` to generate cross-magnitude candidates with the value converted.

For `value=10000.0` and `field_name="nominal_rf_mhz"`, also generate:
- "10 GHz", "10.0 GHz", "10GHz" (10000 MHz / 1000 = 10 GHz)
- "10000000 kHz", "10000000kHz" (10000 MHz × 1000 = 10000000 kHz)

Per-field-suffix conversion table:
- `_mhz`: × 0.001 → GHz; × 1000 → kHz
- `_kw`: × 0.001 → MW; × 1000 → W
- `_km`: × 1000 → m
- `_m`: × 0.001 → km
- `_kg`: × 0.001 → tonnes; × 0.4536 → lbs (or × 2.205 reverse)
- `_mps`: × 3.6 → km/h; × 0.001 → km/s

**Pros:**
- Single-source change in `_numeric_evidence.py`
- Both the auto-evidence resolver AND the evidence gate inherit it
- Tests for new cross-unit cases drop into existing test file

**Cons:**
- Doesn't help if the LLM emits the value in the wrong unit (e.g. emits 10.0 instead of 10000.0 for "10 GHz")
- Generated candidate list grows ~5-10× — substring-match cost rises (negligible at current scale, but worth noting)
- "Mach" → m/s requires assumption about Mach speed that isn't a constant (depends on altitude/temp), so this option doesn't cover that case

### Option B: Stronger LLM prompting for unit conversion

Make DELTA_SYSTEM_PROMPT's Unit Policy block more aggressive about unit normalization. Inline conversion rules per field, not just a generic reference.

**Pros:**
- Smaller change scope (prompt-only)
- Helps even when the source uses an entirely-non-canonical unit

**Cons:**
- Was tried in earlier prompt-tuning iterations and didn't move the needle on numeric extraction (per Phase A diagnosis); re-trying is unlikely to help in a regime where the LLM already hits schema-pressure issues
- Doesn't help the evidence-gate side at all — even if the LLM converts perfectly, the gate still can't find the converted value in source text
- Tightly coupled to gemma4:31b's behavior; brittle to model swaps

### Option C: Spec §10 candidate-mapping architecture

The radar spec §10 fallback architecture (deterministic candidate-mapping) is exactly designed for this. Extract numeric candidates from text via regex/parser, hand them to the LLM with their units AND the field they map to, and have the LLM only choose mappings — not values.

**Pros:**
- Solves both sides (LLM no longer responsible for conversion; gate verifies via the candidate, not via stringified value)
- Native handling of unit conversion at the candidate-extraction layer
- Already documented in spec §10
- Generalizes to all numeric fields, not just the ones with conversions

**Cons:**
- Largest scope change — touches `provenance.py`, `evidence_gate.py`, and the LLM extraction contract itself
- Needs its own design pass for what the candidate JSON shape looks like
- Multi-session work (probably Session 3+)

### Recommendation

**Option A as a near-term mitigation** (covers the common "stated in `<larger> unit`, want `<smaller> unit`" cases — GHz↔MHz, MW↔kW, km↔m, tonnes↔kg). Adds same-magnitude-different-unit candidates with conversion, leaves "Mach speed" and similar non-trivial conversions as residual gaps.

**Option C as the long-term architecture** if Option A doesn't push the false-negative rate below the product threshold. Option C also helps with other Phase A failure modes (numeric collisions get easier when each field gets its candidate explicitly), so its value compounds.

**Skip Option B.** It was tried; didn't move the needle. Don't burn another iteration on prompt-only fixes for this.

---

## TODO list (Option A — DONE 2026-04-27)

### Foundation
- [x] Add a `_UNIT_CONVERSIONS_BY_SUFFIX: dict[str, list[tuple[str, float]]]` table in `_numeric_evidence.py`. **Landed in 6fa8f80.** Covers frequency / power / length / mass / speed / time / angle.
- [x] Extend `value_match_candidates(value, field_name)` to also generate cross-unit candidates. **Landed in 6fa8f80.** Uses `:.10g` formatting (better than `:g` — avoids scientific notation for large magnitudes).
- [x] Update the docstring's "out of scope" note. **Landed in 6fa8f80.** Now reads "Cross-unit conversion (Option A) implemented; logarithmic / Mach conversions remain out of scope."

### Tests (all in 6fa8f80)
- [x] Promote `test_value_match_candidates_for_int_with_mhz_suffix` to assert positive cross-unit candidates ("3 GHz", "3000000 kHz") while keeping the same-magnitude-wrong-unit negative assertions ("3000 GHz", "3000 kHz" still forbidden).
- [x] `test_value_is_supported_by_text_cross_unit_frequency_GHz`: value 10000.0 + evidence "10 GHz" → True.
- [x] `test_value_is_supported_by_text_cross_unit_power_megawatts`: value 1400.0 + evidence "1.4 megawatts" → True.
- [x] `test_value_is_supported_by_text_cross_unit_mass_tonnes`: value 1500.0 + evidence "1.5 tonnes" → True.
- [x] `test_value_is_supported_by_text_cross_unit_negative`: assertions for 999/9999/88 in different fields with cross-unit evidence — all False (no false positives).
- [x] Update `_clear_unsupported_radar_properties` regression with 2 cross-unit cases (Tombstone GHz + megawatts).
- [x] Update `_clear_unsupported_missile_properties` regression with 2 cross-unit cases (tonnes + meters→km).

### Verification harness (in d9c7cf0)
- [x] Re-ran the harder-test text against the rebuilt docling-graph — Tombstone's `nominal_rf_mhz=10000.0` and `tx_peak_power_kw=1400.0` now populate. **Confirmed live, 23/23 probes pass (was 21/23 before Option A).**
- [x] Added `tests/integration/test_radar_cross_unit_smoke.py` — 4 cases, all passing live.

### Cross-bundle (in d9c7cf0)
- [x] Confirmed missile bundle inherits the conversion automatically (same `_numeric_evidence.py` is used by both postprocessors).
- [x] Added `tests/integration/test_missile_cross_unit_smoke.py` — 2 cases (`total_mass_kg=1500` from "1.5 tonnes", `max_intercept_km=43` from "43000 m"), both passing live.

### Still out of scope (for a future Option C / Session 3+)
- [ ] Mach-number → m/s conversion (requires altitude/temp assumption)
- [ ] Imperial units → metric: lbs → kg already done case-specifically by missile's `_mechanically_supported_missile_fields`; not generalized at `_numeric_evidence` layer unless need arises.
- [ ] Compound units like "km/h" — currently handled via the `_mps` suffix's `("km/h", 1/3.6)` entry. If field-name conventions change to include `_kmh`, would need a new entry.
- [ ] Logarithmic-domain conversions (dBW ↔ dBm, dBi ↔ dBd) — fundamentally need offset additions, not scale factors. Different mechanism than Option A's table-driven design.

### Bare-number false-positive (orthogonal limitation noted but not fixed)
- [ ] `value_is_supported_by_text(3000, "nominal_rf_mhz", "operates at 3000 GHz")` returns True because the bare number "3000" matches "3000" in the evidence regardless of unit context. This is a pre-existing limitation of substring matching, not introduced by Option A. Tightening would break legitimate bare-number matches for fields like `num_bits_in_code` and `pulses_per_dwell`. Track for future investigation if false-positive rates climb in production.

---

## Decision log entry (when work starts)

When this TODO is promoted to a real plan, capture:

1. Which option was selected (A / C / hybrid)
2. The conversion table's exact contents (audited against canonical schemas)
3. Whether the existing `_mechanically_supported_missile_fields()` regex path is preserved or absorbed into the new helper
4. The minimum acceptance criterion (e.g. "harder-test smoke 23/23 passes" — currently 21/23)

---

## Cross-references

- **Surfaced by:** harder test in this session (commit-equivalent of 833c7eb run)
- **Documented limitation in:** `docker/docling-graph/app/_numeric_evidence.py` docstrings on `value_match_candidates` and `value_is_supported_by_text`
- **Related architectural option:** spec §10 fallback (candidate-mapping) — `docs/superpowers/specs/2026-04-27-radar-field-group-extraction-design.md`
- **Will affect:** both radar (`_clear_unsupported_radar_properties`) and missile (`_clear_unsupported_missile_properties`) evidence-gate paths
- **Existing partial precedent:** `_mechanically_supported_missile_fields()` regex for "WEIGHT: X LBS" → kg in `evidence_gate.py`
