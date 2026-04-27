# Cross-Unit Conversion — Follow-up TODO

> **Status:** Not yet started. Documented Session 1 limitation surfaced by the radar field-group harder test on 2026-04-27. To be promoted to a real plan when prioritized — likely after the missile field-group plan lands and the candidate-mapping architecture is on the table (spec §10 fallback territory).

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

## TODO list (mechanical for Option A — promote to real plan when ready)

### Foundation
- [ ] Add a `_UNIT_CONVERSIONS_BY_SUFFIX: dict[str, list[tuple[str, float]]]` table in `_numeric_evidence.py`. Each entry maps a canonical suffix to a list of `(other_unit_text, scale_factor_to_canonical)` tuples. Example: `"_mhz": [("GHz", 1000.0), ("kHz", 0.001)]`.
- [ ] Extend `value_match_candidates(value, field_name)` to also generate cross-unit candidates: for each `(unit_text, scale)` in the field's conversion table, emit `f"{value/scale:g} {unit_text}"` and `f"{value/scale:g}{unit_text}"` (with `:g` formatting to avoid trailing zeros).
- [ ] **Update the docstring's "out of scope" note** to reflect that cross-unit conversion is now in scope (with the conversion table as the contract).

### Tests
- [ ] **Promote the existing `test_value_match_candidates_for_int_with_mhz_suffix` negative assertion** to a positive assertion: `"10 GHz"` SHOULD now appear as a candidate for value 10000 in field `nominal_rf_mhz`.
- [ ] Add `test_value_is_supported_by_text_cross_unit_frequency_GHz`: value 10000.0 in nominal_rf_mhz, evidence "operates at 10 GHz" → True.
- [ ] Add `test_value_is_supported_by_text_cross_unit_power_megawatts`: value 1400.0 in tx_peak_power_kw, evidence "1.4 megawatts" → True.
- [ ] Add `test_value_is_supported_by_text_cross_unit_negative`: value 999.0 in nominal_rf_mhz, evidence "operates at 10 GHz" → False (still rejects unrelated values).
- [ ] Update `_clear_unsupported_radar_properties` regression tests to include the cross-unit case (was previously excluded "out of scope").

### Verification harness
- [ ] Re-run the harder-test text against the rebuilt docling-graph; expect Tombstone's `nominal_rf_mhz=10000.0` and `tx_peak_power_kw=1400.0` to now populate.
- [ ] Add the harder-test text as a permanent integration test (`tests/integration/test_radar_cross_unit_smoke.py`) so the regression is locked in.

### Cross-bundle
- [ ] Confirm the missile bundle inherits the conversion automatically (same `_numeric_evidence.py` is used by both bundles' postprocessors). Add a missile-flavored test case (e.g., `total_mass_kg=1500.0` from "1.5 tonnes").

### Out of scope for Option A (defer to Option C / Session 3+)
- [ ] Mach-number → m/s conversion (requires altitude/temp assumption)
- [ ] Imperial units → metric (lbs → kg already done by missile's `_mechanically_supported_missile_fields`; no need to duplicate at the `_numeric_evidence` layer unless we want it generalized)
- [ ] Compound units like "km/h" handled via existing `_mps` suffix vs need for a `_kmh` table entry

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
