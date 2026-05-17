# SA-2 Extraction Next-Steps Implementation Handoff

**Context:** latest reviewed run is commit `d9647e3` (`feat(extraction): restore system_links + generalize table relevance + production-shape deterministic min_altitude_km`).

**Goal:** preserve the entity/field-fill gains from `d9647e3`, recover deterministic relationship recall, make `radar_power_rf` metrics trustworthy, and reduce kinematics runtime failures without adding SA-2-specific logic.

## Current State

The latest end-to-end run is directionally good for entity extraction and field fills:

| metric | v9 baseline | latest `d9647e3` | delta |
|---|---:|---:|---:|
| entity-pass entities | 150 | 153 | +3 |
| filled properties | 133 | 161 | +28 |
| system_links relationships | 30 | 8 | -22 |
| missile_kinematics fills | 20 | 44 | +24 |
| radar_identity fills | 26 | 32 | +6 |

Do **not** revert `d9647e3`. The table relevance policy, type-segregated resolver, and production-shape deterministic `min_altitude_km` support are all useful and generalizable.

The main unresolved problems are:

1. `system_links` has no stable deterministic relationship floor.
2. `radar_power_rf` reporting is misleading because it counts unstable extra fields, not just RF fields.
3. numeric/spec passes still receive irrelevant synthesized table blocks as append-to-end context.
4. `missile_kinematics` has strong fill recall but hit 3 hard timeouts.

## Recommendation 1: Add Unresolved Cross-Entity Hint Diagnostics

**Priority:** first  
**Impact:** observability only  
**Risk:** very low

### Why

`system_links` dropped to 8 relationships. The fixture shows only 1 of 10 table-overlay cross-entity hints was promoted. The other hints failed to resolve, but the postprocessor currently does not expose enough structured information about the failures.

The current code promotes hints in:

`docker/docling-graph/app/evidence_gate.py`

Relevant functions:

- `_resolve_ref(...)`
- `_postprocess_air_defense_system_links(...)`

### What To Implement

In `_postprocess_air_defense_system_links`, collect unresolved hint samples when either source or target fails to resolve.

Recommended diagnostic shape:

```json
{
  "unresolved_cross_entity_hints": {
    "count": 9,
    "samples": [
      {
        "source_alias": "1D",
        "source_type": "MISSILE_SYSTEM",
        "target_alias": "RSN- 75V",
        "target_type": "RADAR_SYSTEM",
        "source_resolved": true,
        "target_resolved": false,
        "reason": "target_unresolved"
      }
    ]
  }
}
```

Keep the sample count bounded, e.g. first 20 unresolved hints.

### Acceptance Criteria

- Existing `promoted_from_cross_entity_hints` behavior remains unchanged.
- Diagnostics appear even when zero hints promote.
- Diagnostics include entity type, raw alias, and which side failed.
- Add unit tests for:
  - unresolved source
  - unresolved target
  - both unresolved
  - all resolved

## Recommendation 2: Populate Target-Side Alias Maps From Cross-Entity Table Rows

**Priority:** second  
**Impact:** high for `system_links`  
**Risk:** low if implemented generically

### Why

The overlay currently creates `alias_map_by_entity_type` only for the table's primary/winner entity type. For the SA-2 missile spec table, that produces `MISSILE_SYSTEM` aliases but no `RADAR_SYSTEM` aliases.

That is why hints like this fail:

```text
source: 20DP / MISSILE_SYSTEM
target: RSN- 75V / RADAR_SYSTEM
```

The source resolves through the missile alias map. The target does not resolve because `alias_map_by_entity_type["RADAR_SYSTEM"]` is missing.

### Code Area

Primary implementation area:

`docker/docling-graph/app/_table_facts.py`

Relevant behavior:

- alias clusters are extracted for the winner entity type
- cross-entity rows emit `CrossEntityHint`
- cross-entity rows currently do not add target aliases to `alias_map_by_entity_type`

Resolver/postprocess area:

`docker/docling-graph/app/evidence_gate.py`

Relevant behavior:

- `_resolve_ref` is type-segregated and should stay that way
- alias lookup should remain scoped by `entity_type`

### What To Implement

When a table row is classified as a cross-entity reference row, register aliases for the target entity type as well.

This must be generic. Do **not** hardcode `Fan Song`, `RSN-75`, SA-2, or any document-specific equipment names.

Use two generic mechanisms:

1. **OCR-normalized alias variants**

   Register aliases with obvious OCR spacing variants normalized:

   - `RSN- 75V` -> `RSN-75V`
   - `RSNA- 75M` -> `RSNA-75M`

   Keep the original raw alias too.

2. **Label-derived family fallback**

   If the cross-entity row label is of the form:

   ```text
   <family name> Variant
   ```

   then treat `<family name>` as a possible target canonical name only if it exists in the upstream catalog for the target entity type.

   Example:

   ```text
   Fan Song Variant
   ```

   may bridge target aliases to `Fan Song` only when `Fan Song` is already present as a `RADAR_SYSTEM` upstream entity.

This gives the SA-2 behavior we need without making the code SA-2-specific. For another document, `Foo Variant` only bridges to `Foo` if `Foo` is already an extracted upstream entity.

### Expected Result

Current latest run:

- 7 LLM-emitted relationships
- 1 promoted table hint
- 8 total relationships

If the 9 failed hints resolve, expect roughly:

- 7 LLM-emitted relationships
- up to 10 promoted table hints
- about 16-18 total relationships after dedupe

This probably will not fully recover the v9 count of 30. It should restore a deterministic floor and make the remaining gap easier to diagnose.

### Acceptance Criteria

- `alias_map_by_entity_type` can contain more than the winner entity type.
- Cross-entity target aliases are type-scoped under their target type.
- Existing missile alias behavior is unchanged.
- Type-segregated resolver remains type-segregated.
- Add tests for:
  - cross-entity target alias map generation
  - OCR-spaced alias normalization
  - label-derived fallback only when upstream canonical exists
  - no fallback when upstream canonical is absent

## Recommendation 3: Fix `radar_power_rf` Reporting And Guard Against Irrelevant Synth Noise

**Priority:** third  
**Impact:** medium  
**Risk:** low-medium

### Why

The latest `radar_power_rf` run looks like a possible regression, but direct fixture comparison shows the actual RF fields are unchanged:

| field | v9 | latest |
|---|---:|---:|
| `system_name` | 42 | 42 |
| `erp_dbw` | 0 | 0 |
| `tx_peak_power_kw` | 0 | 0 |
| `nominal_rf_mhz` | 0 | 0 |
| extra `emitter_function` | 23 | 22 |
| extra `nomenclature` | 2 | 2 |

The schema for `radar_power_rf` only declares:

- `system_name`
- `erp_dbw`
- `tx_peak_power_kw`
- `nominal_rf_mhz`

But the fixture contains extra fields such as `emitter_function` and `nomenclature`. Those are not RF fields and should not be used to judge RF extraction quality.

### Code Areas

Schema:

`ontology_bundles/air_defense_v3/extraction_schemas/radar_power_rf.py`

Field group:

`ontology_bundles/air_defense_v3/extraction_schemas/_field_groups.py`

Table policy:

`app/services/table_normalization/_pipeline_hooks.py`

Table injection:

`docker/docling-graph/app/main.py`

### What To Implement

#### 3.1 Separate Metrics By Field Category

For `radar_power_rf`, report at least:

- entity count
- RF schema fills: `erp_dbw`, `tx_peak_power_kw`, `nominal_rf_mhz`
- identity fills: `system_name`
- out-of-schema/extra fills, if present

Do not use `emitter_function` or `nomenclature` to claim RF extraction improved or regressed.

#### 3.2 Do Not Append Irrelevant Synth Tables To Numeric/Spec Passes

Current behavior:

- relevant table -> synth-only in-place replacement
- non-relevant table -> raw table remains and synth refs are appended to the end

For numeric/spec passes, irrelevant synthesized table blocks are noise. They can destabilize entity naming and compete with prose/table evidence.

Recommended policy:

| pass shape | relevant table | non-relevant table |
|---|---|---|
| identity/prose-heavy | raw preserved; current behavior acceptable | raw preserved; append synth context acceptable |
| numeric/spec pass | synth-only in-place | raw preserved only; **do not append synth refs** |

Keep `radar_power_rf` in `SYNTH_ELIGIBLE_PASSES`. It should use synth-only when a document actually has a relevant radar RF table. The fix is to stop adding irrelevant synthesized context when the table does not match the pass.

### Acceptance Criteria

- `radar_power_rf` metric output distinguishes RF fields from extra fields.
- `radar_power_rf` remains synth-eligible for relevant radar RF tables.
- irrelevant missile tables are not appended as synthesized text for `radar_power_rf`.
- identity passes retain their existing recall-friendly behavior.
- Add table relevance tests:
  - radar table with `Radar` + `Frequency` -> relevant
  - radar table with `Emitter` + `Peak Power` -> relevant
  - missile table with `Frequency` -> not relevant
  - communications table with `Frequency` but no radar identity context -> not relevant
  - radar identity-only table with no RF/power row -> not relevant for `radar_power_rf`

## Recommendation 4: Use Identity Pass Outputs As Name Context For Numeric Passes

**Priority:** fourth  
**Impact:** medium  
**Risk:** medium if evidence rules are not strict

### Why

The numeric/spec passes can churn entity names because they rediscover systems from local chunks. A compact upstream identity roster can stabilize names across passes.

This should help `radar_power_rf`, `radar_antenna`, `radar_timing`, `radar_modulation`, and missile numeric passes.

### Rule

Upstream identity-pass output may help normalize the entity name, but must not serve as evidence for any property value.

Allowed:

- use upstream `radar_identity` aliases to normalize `SNR-75 "Fan Song"` and `Fan Song`
- use upstream `missile_identity` aliases to normalize missile variant names

Forbidden:

- filling `nominal_rf_mhz` because upstream says the entity is a radar
- filling `tx_peak_power_kw` from a known equipment profile
- emitting a property without current-batch support

### Implementation Direction

Add a compact roster section to numeric/spec pass prompts if the pipeline already has upstream entities available.

The roster should include:

- entity type
- canonical name
- directly extracted aliases/nomenclature

It should explicitly state:

```text
The roster is for entity name normalization only. It is not evidence for any property value.
```

### Acceptance Criteria

- Numeric property evidence still comes only from current batch text/table content.
- Name normalization improves or remains stable on SA-2.
- No increase in unsupported RF/missile numeric fills.
- Add a test or fixture-level assertion that upstream roster content alone cannot populate numeric fields.

## Recommendation 5: Investigate And Reduce Kinematics Hard Timeouts

**Priority:** fifth  
**Impact:** medium  
**Risk:** low

### Why

`missile_kinematics` is now the strongest field-fill win:

- v9: 20 fills
- latest: 44 fills

But the latest run had:

- 3 `TRUNCATION_PERSISTS_RETRYING` events
- 3 `BATCH_HARD_TIMEOUT` events
- hung batches: 37, 51, 52

That means the current score is probably below what the implementation could produce. It also makes the run less reliable.

### What To Do

1. Inspect debug dumps for batches 37, 51, and 52.
2. Determine whether the prompts are unusually large, malformed, or normal-sized.
3. If prompts are too large, reduce the kinematics-specific batch token size.
4. If prompts are normal-sized, investigate the orchestrator/gleaning path for a hang or deadlock.

Do not raise the timeout again as the first fix. The current ceiling is already high enough that a 3-hour batch is operationally unhealthy.

### Acceptance Criteria

- No hard timeouts on a targeted `missile_kinematics` rerun.
- Kinematics fills remain above v9 by at least 50%.
- `min_altitude_km` remains nonzero.

## Recommendation 6: Defer Multi-Sample `system_links`

**Priority:** defer  
**Impact:** potentially high  
**Risk:** medium

Do not implement multi-sample union/dedup for `system_links` yet.

Reason:

- current deterministic table-hint promotion is broken
- multi-sampling is expensive
- multi-sampling can add hallucinated relationships

First restore the deterministic floor with Recommendations 1 and 2. Then rerun `system_links`. Only consider multi-sample extraction if:

- cross-entity hints promote correctly
- deterministic relationship count is stable
- prose-derived `CUES` edges remain too volatile

## Recommended Implementation Sequence

1. Add unresolved cross-entity hint diagnostics.
2. Add target-side alias maps for cross-entity rows.
3. Rerun `system_links` only using the same upstream entity outputs.
4. Add `radar_power_rf` metric separation.
5. Change numeric/spec pass table policy so irrelevant tables are raw-only, not raw plus synthesized append.
6. Add table relevance tests for `radar_power_rf`.
7. Investigate kinematics hard-timeout batches and tune batch sizing or orchestrator behavior.
8. Consider identity-roster prompt context for numeric/spec passes.
9. Defer multi-sample `system_links` unless the deterministic fixes are insufficient.

## Expected Results After These Changes

Expected near-term SA-2 results:

| area | expected result |
|---|---|
| `system_links` | improves from 8 toward 16-18 relationships if all 10 table hints promote |
| total field fills | should remain near latest run's 161, with small stochastic variance |
| `missile_kinematics` | should remain much stronger than v9; may recover some lost fills if timeouts are fixed |
| `radar_power_rf` | RF fields likely remain 0 on SA-2 unless the source actually contains RF data |
| radar_power_rf reporting | becomes reliable because RF fields are separated from extra role/name crumbs |
| future documents | relevant radar RF tables get synth-only treatment; irrelevant frequency tables do not pollute the pass |

The main metric to watch after Recommendations 1 and 2 is not total relationship count alone. Watch:

- number of cross-entity hints generated
- number promoted
- number unresolved by source vs target
- number of LLM-emitted prose relationships
- final deduped relationship count

That breakdown will distinguish deterministic overlay failures from LLM stochasticity.

