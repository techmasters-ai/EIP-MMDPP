# Missile Field-Group Extraction — Follow-up TODO

> **Status:** Not yet started. Do **not** begin until the radar field-group split (`docs/superpowers/plans/2026-04-27-radar-field-group-extraction.md`) is implemented and the smoke harness shows ≥2/3 numeric extraction. If the radar smoke is <2/3, switch tracks per spec §10 (Fallback) instead of mirroring the split for missile.
>
> **Pattern source:** Mirrors the radar plan exactly. Same architecture, same files, same merge contract — only the field partition and forbidden-name set differ.

**Prerequisite:** Radar plan tasks 1-21 complete and committed; cutover landed; baseline test suite green.

---

## What this is

The radar field-group refactor splits `radar_domain` into 5 sub-passes so each LLM call sees a smaller schema. `missile_domain` has the same shape problem (38 fields on `MissileSystemEntity`, mixed string + numeric) and the same hypothesised cause for missing numeric extraction. This TODO captures the exact mirror so the missile session is mechanical, not a re-design.

Once started, this becomes a real plan at `docs/superpowers/plans/YYYY-MM-DD-missile-field-group-extraction.md` modeled on the radar plan.

---

## Field partition (proposed — confirm before starting)

`MissileSystemEntity` (38 fields) splits into 6 groups. `system_name` is replicated as the identity in every group.

### `missile_identity` (10 fields, mostly strings — discovery pass)
- system_name *(identity)*
- nomenclature
- dieqp
- name
- emitter_function
- system_status
- asrd
- responsible_agency
- review_cycle
- next_review_date

### `missile_kinematics` (5 numerics + identity)
- system_name *(identity)*
- min_intercept_km
- max_intercept_km
- min_altitude_km
- max_altitude_km
- max_launch_angle_deg

### `missile_guidance` (3 fields + identity)
- system_name *(identity)*
- guidance_type
- seeker_type
- missile_photo

### `missile_airframe` (3 numerics + identity)
- system_name *(identity)*
- body_length_m
- body_diameter_m
- total_mass_kg

### `missile_speed_timing` (8 numerics + identity)
- system_name *(identity)*
- average_speed_mps
- max_speed_mps
- max_flyout_time_sec
- flight_time_sec
- coast_time_sec
- intra_salvo_time_sec
- total_burn_time_sec
- ejector_time_sec

### `missile_propulsion` (8 fields, mixed numeric + str + identity)
- system_name *(identity)*
- ejector_thrust *(str)*
- ejector_mass_kg
- booster_time_sec
- booster_thrust *(str)*
- booster_mass_kg
- sustain_time_sec
- sustain_thrust *(str)*
- sustain_mass_kg

**Excluded from groups:** `confidence` (meta — emitted on every record, not assigned to any group).

**Sanity check during planning:**
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.missile_domain import MissileSystemEntity
declared = set(MissileSystemEntity.model_fields.keys())
grouped = {
  'system_name','nomenclature','dieqp','name','emitter_function','system_status','asrd','responsible_agency','review_cycle','next_review_date',
  'min_intercept_km','max_intercept_km','min_altitude_km','max_altitude_km','max_launch_angle_deg',
  'guidance_type','seeker_type','missile_photo',
  'body_length_m','body_diameter_m','total_mass_kg',
  'average_speed_mps','max_speed_mps','max_flyout_time_sec','flight_time_sec','coast_time_sec','intra_salvo_time_sec','total_burn_time_sec','ejector_time_sec',
  'ejector_thrust','ejector_mass_kg','booster_time_sec','booster_thrust','booster_mass_kg','sustain_time_sec','sustain_thrust','sustain_mass_kg',
}
meta = {'confidence'}
missing = declared - grouped - meta
extra = grouped - declared
print('declared:', len(declared), 'grouped:', len(grouped), 'meta:', len(meta))
print('missing from groups:', missing)
print('extra in groups (not on schema):', extra)
"
```
Expected: missing == set(), extra == set(). If `MissileSystemEntity` has gained/lost fields since 2026-04-27, update the partition before starting.

---

## File-level mirror (parallel to radar plan)

| Radar artifact | Missile equivalent | Notes |
|----------------|--------------------|-------|
| `ontology_bundles/.../extraction_schemas/_field_groups.py` already exists from radar | Add `MISSILE_FIELD_GROUPS` constant in same file | Reuse the file — don't create a sibling |
| `ontology_bundles/.../extraction_schemas/_radar_shared.py` | `_missile_shared.py` (sibling) | Mirrors `_radar_shared.py`: `MISSILE_FORBIDDEN_SYSTEM_NAMES`, `validate_missile_system_name`, `make_missile_root_sanitizer`, `MISSILE_OPTIONAL_TEXT_FIELDS` |
| `ontology_bundles/.../extraction_schemas/radar_identity.py` (and 4 siblings) | `missile_identity.py` + 5 siblings | One file per sub-pass, same template-class shape |
| `_clear_unsupported_radar_properties` post-processor | `_clear_unsupported_missile_properties` (already exists in `evidence_gate.py` — refactor analogously to radar) | **Critical:** verify it doesn't unconditionally null missile numerics. Check before cutover. |
| Description-quality contract test for radar | Add missile fixtures to the same test file | Same checks: no numeric examples, no FORBIDDEN-block leakage, no typical-value ranges |
| Smoke harness `test_radar_smoke.py` (3 cases) | `test_missile_smoke.py` (3 cases — pick 3 known-failing missiles) | Suggested cases: 5V55K (SA-2 missile), 5V28 (SA-5 missile), 9M82 (SA-12 missile) — confirm with current corpus |
| Manifest entry `radar_domain` | Replace `missile_domain` with 6 entries (`missile_identity`, `missile_kinematics`, `missile_guidance`, `missile_airframe`, `missile_speed_timing`, `missile_propulsion`) | Same `depends_on` pattern as radar |

---

## TODO list (mechanical mirror of radar plan tasks 1-21)

Mark these off as you build the actual plan document. Do not check them in this file — copy them into the new plan.

### Foundation chunk
- [ ] Add `MISSILE_FIELD_GROUPS: dict[str, list[str]]` to `_field_groups.py` (radar's location). Six entries, exact partition above.
- [ ] Create `_missile_shared.py` mirroring `_radar_shared.py`: `MISSILE_FORBIDDEN_SYSTEM_NAMES`, `MISSILE_OPTIONAL_TEXT_FIELDS`, `validate_missile_system_name`, `make_missile_root_sanitizer`. Source the forbidden set from existing `_MISSILE_FORBIDDEN_SYSTEM_NAMES` in `missile_domain.py`.
- [ ] Confirm `_numeric_evidence.py` (built in radar plan task 3) is reused as-is — do not fork.
- [ ] Unit tests for the new shared validators/sanitizers, parallel to the radar `_radar_shared` test file.

### Sub-pass modules chunk
- [ ] `missile_identity.py` — discovery pass, 10 string fields. `MissileIdentityEntity` + `MissileIdentityPass`. Emits `MISSILE_SYSTEM[]`.
- [ ] `missile_kinematics.py` — 5 numerics + identity. `MissileKinematicsEntity` + `MissileKinematicsPass`. Auto-evidence wired via `_numeric_evidence.py`.
- [ ] `missile_guidance.py` — 3 fields (str/str/bool) + identity.
- [ ] `missile_airframe.py` — 3 numerics + identity.
- [ ] `missile_speed_timing.py` — 8 numerics + identity.
- [ ] `missile_propulsion.py` — 8 fields (4 numeric + 3 str + identity).
- [ ] Description-quality contract test extended with missile fixtures (no FORBIDDEN-block leakage, no numeric examples in field descriptions, no typical-value ranges).

### Sanitization rules during description copy
For every missile sub-pass, when copying descriptions from canonical `missile_domain.py`:
1. **Strip the FORBIDDEN-values block** (lines listing forbidden synonyms).
2. **Strip typical-value ranges** (e.g. "typically 5-50 km") — keep the field's *meaning* and *unit*, not example numbers.
   These are the same two rules the radar plan applies; the description-quality contract test enforces them.

### Cutover prep chunk
- [ ] Refactor `_clear_unsupported_missile_properties` in `docker/docling-graph/app/evidence_gate.py` to use `verify_numeric_against_evidence()` from `_numeric_evidence.py` (the radar refactor's pattern). **Critical:** this is the missile equivalent of the §4.8 numeric-erasure blocker. Before this refactor, missile numerics may be silently nulled even when correctly extracted.
- [ ] Update bundle `__init__.py` exports to include the six missile sub-pass classes.
- [ ] Coverage checker / `check_bundle()` passes with new schemas registered.
- [ ] Worker `_parse_pass_response` regression sweep with `pytest tests/unit -k missile -q`.

### Manifest cutover + verification chunk
- [ ] Replace single `missile_domain` manifest entry with 6 sub-pass entries. Mirror the radar `depends_on` topology — if radar serializes via depends-on chain due to Ollama concurrency limits (per spec §8 risk row), missile must serialize too.
- [ ] Smoke harness `test_missile_smoke.py` runs and reports baseline (expected: 0/3 or 1/3 before refactor lands; ≥2/3 after).
- [ ] Re-ingest a missile-bearing fixture; verify `MISSILE_SYSTEM` vertices have populated numeric properties.
- [ ] Description-quality contract test passes for all 6 sub-passes.
- [ ] Full pytest suite green.
- [ ] Commit each task; final commit references this TODO doc.

---

## Decisions to confirm before promoting this TODO into a real plan

1. **Group count.** Six groups vs. five vs. four — depends on whether `missile_propulsion` (8 mixed fields) is too large for the LLM under the same schema-pressure hypothesis. If radar's 5-pass split worked at 4-11 fields per pass, six missile groups at 4-10 fields each fits the same envelope.
2. **`missile_propulsion`'s 3 string-thrust fields.** `ejector_thrust`, `booster_thrust`, `sustain_thrust` are typed `Optional[str]` today. If they're actually meant to be numeric (kN, lbf), promote them to numerics in the missile session — but track that as schema-correction work, not part of the field-group split.
3. **Smoke-case selection.** Current candidates (5V55K, 5V28, 9M82) are guesses. Before starting the missile session, confirm against the corpus that these missiles appear in ingest-able sources with extractable numeric values.
4. **Concurrency cap.** If radar's 5-pass parallel topology hit Ollama backend limits and was serialized via `depends_on` chain, missile's 6 passes will too. Decide once based on radar's observed runtime.
5. **`_clear_unsupported_missile_properties` blast radius.** Before the missile session starts, grep for it: `grep -n "_clear_unsupported_missile" docker/docling-graph/app/`. If it nulls numerics unconditionally (parallel to the radar bug discovered in spec §4.8), refactor it the same way. If it doesn't exist, no refactor needed — still verify nothing else is silently dropping missile numerics.

---

## Cross-reference

- **Radar plan (source pattern):** `docs/superpowers/plans/2026-04-27-radar-field-group-extraction.md`
- **Radar spec (architecture decisions):** `docs/superpowers/specs/2026-04-27-radar-field-group-extraction-design.md`
- **Spec §9 deferral note:** "Missile field-group split — same pattern, follow-up session."
- **Fallback track if radar split underperforms:** spec §10 — do **not** start missile if radar smoke is <2/3; switch architecture instead.
