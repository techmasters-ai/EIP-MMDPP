# Radar field-group extraction refactor — design

**Author:** Josh / Claude (brainstorming session 2026-04-27)
**Status:** Approved (5/5 sections)
**Predecessor:** `docs/superpowers/specs/2026-04-25-flat-schema-profile-refactor-design.md`
**Predecessor commits:** `dda4862` (Phase A: prompt sanitization + numeric-candidate hints + Unit Policy)

---

## 1. Background

Phase A of the prompt-extraction work landed deterministic auto-evidence
(no LLM verbatim quotes), prompt sanitization (numeric examples stripped,
typical-value ranges removed, FORBIDDEN-values block compressed), the
Unit Policy block in `DELTA_SYSTEM_PROMPT`, and a regex-based numeric-
candidate hint block in the user prompt. After those changes, controlled
testing on a rich Fan Song spec doc produced:

- **5 string-typed fields** extracted by gemma4:31b (`system_name`,
  `nomenclature`, `emitter_function`, `scan_type`, `intra_pulse_mop`).
- **0 numeric-typed fields**, despite the doc explicitly stating
  "3000 MHz / 600 kW / 35 dBi / 6 meters / 1.5 degrees / 10 seconds"
  AND those values appearing as verbatim spans in the injected numeric-
  candidate hint block.
- Identical behavior in strict JSON-schema mode and loose JSON mode →
  the constrained-decoding grammar is not the bottleneck.

The remaining gap is **schema pressure**: `RadarSystemEntity` has 30+
fields, each with multi-sentence descriptions. Even after sanitization,
gemma4:31b consistently emits `null` for `Optional[float]` fields. The
established remedy is to **split the radar extraction into smaller LLM-
call boundaries**, where each call sees ~5 fields with focused
descriptions and a small structured-output schema.

This spec covers Session 1 of a multi-session refactor. Sessions 2 and
3 (identity/parameter split, group-scoped retry, evidence-span
verification, diagnostics persistence, golden test harness) are
deferred and out of scope here.

## 2. Goal

Replace the single `radar_domain` extraction pass with five focused sub-
passes — `radar_identity`, `radar_power_rf`, `radar_antenna`,
`radar_timing`, `radar_modulation` — each emitting `RADAR_SYSTEM[]` with
the same `system_name` identity so the existing merge layer collapses
partial records onto one vertex. Each sub-pass is its own
`/extract-pass` call against a smaller schema (5-11 fields) with a
focused semantic guide. No worker-side, merge, or vertex-persistence
code changes.

## 3. Non-goals (Session 1)

- Do **not** split `missile_domain`. Same pattern; deferred to a follow-
  up session.
- Do **not** introduce identity-fed parameter passes (Item #3). All
  sub-passes remain `document_only` for Session 1.
- Do **not** add group-scoped retry (Item #7).
- Do **not** add evidence-span verification (Item #6).
- Do **not** persist per-pass diagnostics beyond what's already logged
  (Item #9).
- Do **not** build a full per-field recall/precision golden harness
  (Item #10) — Session 1 ships a 3-case smoke harness only.

## 4. Architecture

### 4.1 File layout

```
ontology_bundles/air_defense_v3/extraction_schemas/
├── _radar_shared.py            ← NEW
├── _field_groups.py            ← NEW
├── radar_identity.py           ← NEW
├── radar_power_rf.py           ← NEW
├── radar_antenna.py            ← NEW
├── radar_timing.py             ← NEW
├── radar_modulation.py         ← NEW
├── radar_domain.py             ← KEPT (legacy; not in manifest after cutover)
├── missile_domain.py           ← UNCHANGED
└── system_links.py             ← UNCHANGED
```

### 4.2 `_field_groups.py`

Single source of truth for which fields each sub-pass extracts. Hand-
authored, contract-tested.

```python
RADAR_FIELD_GROUPS: dict[str, list[str]] = {
    "radar_identity": [
        "system_name", "nomenclature", "elnot", "dieqp",
        "emitter_function", "system_status", "asrd",
        "responsible_agency", "review_cycle", "next_review_date",
        "scan_type",
    ],
    "radar_power_rf": [
        "system_name", "erp_dbw", "tx_peak_power_kw", "nominal_rf_mhz",
    ],
    "radar_antenna": [
        "system_name", "antenna_photo", "gain_dbi",
        "antenna_dim_az_m", "antenna_dim_el_m",
        "beamwidth_az_deg", "beamwidth_el_deg",
        "spoiled", "coverage_limits_el_deg",
    ],
    "radar_timing": [
        "system_name", "nominal_pri_usec", "nominal_pd_usec",
        "scan_period_sec", "dwell_time",
    ],
    "radar_modulation": [
        "system_name", "intra_pulse_mop", "inter_pulse",
        "frequency_excursion_mhz", "num_bits_in_code",
        "pulses_per_dwell",
    ],
}
```

### 4.3 `_radar_shared.py`

Centralizes the items every sub-pass uses identically. Public-facing
constants drop the leading underscore so cross-module imports are
explicit:

- `edge` — the existing field decorator (copied / re-exported).
- `RADAR_FORBIDDEN_SYSTEM_NAMES` — frozen set of forbidden identities.
  Single source of truth.
- `RADAR_OPTIONAL_TEXT_FIELDS` — set of optional-text field names that
  `sanitize_entity_list` should normalize. **Note**: this set is the
  **superset** across sub-passes; each `make_root_sanitizer` call passes
  the subset relevant to its record class.
- `validate_radar_system_name(value)` — `field_validator("system_name",
  mode="before")` body. Centralized so identity validation is
  consistent across sub-passes.
- `make_root_sanitizer(list_field, optional_text_fields)` — factory
  returning a `model_validator(mode="before")` body wired with the
  caller's list-field name and per-class text-field set. Defaults the
  forbidden-identities argument to `RADAR_FORBIDDEN_SYSTEM_NAMES` so
  sub-pass modules don't have to import the constant directly.
  **The returned validator runs both `sanitize_entity_list` and
  `dedupe_entities_by_identity`**, mirroring the legacy
  `_sanitize_and_dedupe_root_entities` body in `radar_domain.py`.
  Sanitize-only factories silently break duplicate-emission handling.

### 4.4 Per-sub-pass module shape

Each sub-pass module is ~80 lines and follows a uniform template
(example: `radar_power_rf.py`):

```python
"""radar_power_rf extraction pass — RF carrier + transmit power.

Spec §4.4. One of 5 sub-passes splitting the legacy radar_domain into
smaller LLM call boundaries. Emits RADAR_SYSTEM[] with the same identity
(system_name) as the other sub-passes; merge_and_resolve collapses
partial records onto one vertex.
"""
from __future__ import annotations
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._field_groups import RADAR_FIELD_GROUPS
from ._radar_shared import edge, make_root_sanitizer, validate_radar_system_name
from ..validators import coerce_optional_float

_GROUP_NAME = "radar_power_rf"
_FIELDS = RADAR_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class RadarPowerRfRecord(BaseModel):
    model_config = ConfigDict(
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
        extra="ignore",
    )

    system_name: str = Field(..., description="...", examples=["Fan Song"])
    erp_dbw: Optional[float] = Field(default=None, description="...")
    tx_peak_power_kw: Optional[float] = Field(default=None, description="...")
    nominal_rf_mhz: Optional[float] = Field(default=None, description="...")

    _v_system_name      = field_validator("system_name", mode="before")(validate_radar_system_name)
    _v_erp_dbw          = field_validator("erp_dbw", mode="before")(coerce_optional_float)
    _v_tx_peak_power_kw = field_validator("tx_peak_power_kw", mode="before")(coerce_optional_float)
    _v_nominal_rf_mhz   = field_validator("nominal_rf_mhz", mode="before")(coerce_optional_float)


class RadarPowerRfPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    radar_systems: List[RadarPowerRfRecord] = edge(
        label="CONTAINS",
        description="Top-level radar systems with RF carrier + transmit power values.",
        examples=[["Fan Song"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_root_sanitizer(
            list_field="radar_systems",
            optional_text_fields=set(),   # no string fields in this group
        )
    )
```

Key shape points:

- **`extra="ignore"`** on the record so out-of-group LLM emissions are
  silently dropped rather than failing validation.
- **`graph_id_fields=["system_name"]`** identical across all 5 sub-pass
  record classes; this is what makes `merge_and_resolve` collapse
  records by `LogicalIdentity`.
- **Validators** use the canonical `field_validator(...)` decorator
  pattern. Per-pass `coerce_optional_float` validators copied from
  canonical for numeric fields.
- **Description text** is hand-copied from existing canonical /
  extraction-side schemas. Per-Session-1 acceptance: explicit
  duplication preferred over a description-drift contract test.

### 4.5 Manifest changes

Single commit at the cutover step. Removes `radar_domain`, adds 5 sub-
passes, updates `system_links.depends_on`:

```yaml
passes:
  - name: radar_identity
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.radar_identity
    template_class: RadarIdentityPass
    primary_entity_types: [RADAR_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: radar_power_rf
    # ... analogous ...

  - name: radar_antenna
    # ... analogous ...

  - name: radar_timing
    # ... analogous ...

  - name: radar_modulation
    # ... analogous ...

  - name: missile_domain
    # ... unchanged ...

  - name: system_links
    required: true
    kind: relationships_only
    input_mode: document_plus_entity_refs
    module: extraction_schemas.system_links
    template_class: SystemLinksPass
    primary_entity_types: []
    bridge_entity_types: []
    extracted_relationship_types: [ASSOCIATED_WITH, CUES]
    depends_on:
      - radar_identity
      - radar_power_rf
      - radar_antenna
      - radar_timing
      - radar_modulation
      - missile_domain
    skip_if_no_upstream_endpoints: true
```

`system_links` depends on **all 6** entity passes — not just
`radar_identity` — so the relationship pass sees the fullest set of
identities including any radar that appears only in a parameter table
and was missed by the identity-discovery pass.

### 4.6 Pass execution order

```
radar_identity ──┐
radar_power_rf ──┤
radar_antenna ───┼─→ all run in parallel (depends_on: [])
radar_timing ────┤
radar_modulation ┤
missile_domain ──┘
                 ↓
            system_links (waits on all 6)
```

The orchestrator already supports parallel sibling execution via
`depends_on`. No orchestrator code changes needed.

### 4.7 Wiring preserved

Zero code change in any of the following:

- **`prompt_rules.install()`** — wraps `get_delta_batch_prompt` and
  `build_compact_semantic_guide` per LLM call. Each new sub-pass gets
  the same `DELTA_SYSTEM_PROMPT`, Unit Policy block, sanitization, and
  numeric-candidate hint. Smaller schemas mean each sub-pass's semantic
  guide fits comfortably within the 30K-char budget.
- **`build_auto_field_evidence`** — runs per-pass after the LLM call.
  Each sub-pass produces evidence rows for the fields it extracts. The
  worker-side merger aggregates `_field_evidence` across passes by
  `(instance_id, field_name)`, so a single `RADAR_SYSTEM` vertex ends
  up with `_field_evidence` keys spanning all groups.
- **`merge_and_resolve`** — already keys by `LogicalIdentity`
  (`system_name`). 5 sub-passes emitting the same `system_name="Fan
  Song"` collapse to one `MergedEntityRecord`. Property merge unions
  per-pass props into one dict; confidence is `max()` across passes.
- **`_import_graph_phase_nodes`** — writes `_field_evidence` JSON
  property on the vertex. Already merges across passes; works as-is.

### 4.8 Touchpoints requiring code change

**Three pass-name literals in app code reference `"radar_domain"` and
must be updated as part of the manifest cutover. Missing any of these
is a silent correctness regression, not a test failure.**

- **`app/workers/pipeline.py::_DOMAIN_PASS_NAMES`** (around line 382) —
  frozenset feeds `_classify_extraction_quality`. After cutover,
  `radar_domain` no longer appears in `pass_outcomes`; a successful
  radar extraction via the 5 sub-passes wouldn't match the `domain_hit`
  check, and the pipeline run could be classified `degraded` /
  `anomaly` even when the new passes returned HIT. Update to include
  all 5 new pass names: `{"radar_identity", "radar_power_rf",
  "radar_antenna", "radar_timing", "radar_modulation",
  "missile_domain"}`. Verify `_classify_extraction_quality`'s OR-logic
  treats *any* radar sub-pass returning HIT as a domain hit (not a
  per-pass AND).

- **`docker/docling-graph/app/evidence_gate.py::_run_post_extraction_evidence_gate`**
  (around line 317) — dispatches to `_postprocess_air_defense_radars`
  only when `pass_name == "radar_domain"`. After cutover, none of the
  5 sub-passes match this string, so radar-side evidence-gate post-
  processing (status validation, identity sanity checks) is silently
  bypassed. Update the dispatch to recognize the 5 new names. Confirm
  the post-processor function is idempotent across multiple sub-pass
  invocations on the same document (since 5 sub-passes will each
  trigger it; merge dedup happens later).

- **`docker/docling-graph/app/schemas.py`** (around line 55) — docstring
  example references `pass_name="radar_domain"`. Low-risk; update the
  example to a current pass name (`radar_identity` is a fine
  representative).

**Test fixtures with hardcoded pass names:**

- `tests/docker/docling-graph/test_extract_pass_endpoint.py`
- `tests/docker/docling-graph/test_service_identity_gate.py`

Both pin `pass_name="radar_domain"` in their request fixtures. Either
update fixtures to a sub-pass name (e.g. `radar_identity`) or document
in a regression-fixture justification why the legacy name is preserved
(e.g. these tests verify the legacy radar_domain.py module still loads
even though it's not in the manifest). Default plan: update to
`radar_identity`.

## 5. Tests

### 5.1 Contract tests — `tests/unit/test_radar_field_groups_contract.py`

**Import target for assertions 2-5:**
`ontology_bundles.air_defense_v3.extraction_schemas.radar_domain.RadarSystemEntity`
(the extraction-side schema). Field names in `RADAR_FIELD_GROUPS` match
the extraction-side schema's flat-checklist fields (`tx_peak_power_kw`,
`nominal_pri_usec`, etc.); the canonical `entities.RadarSystemEntity`
is a superset that also includes structural / system fields outside
extraction scope. The contract is about partitioning the **extraction
schema's** fields, not the canonical entity's fields. Importing the
canonical instead would cause spurious failures from non-extraction
fields not appearing in any group.

Group-membership assertions (5):

1. **`test_every_group_includes_system_name`** — every group's field
   list contains `system_name`.
2. **`test_every_listed_field_exists_on_canonical`** (where "canonical"
   here means the extraction-side schema per the import target above) —
   every name in `RADAR_FIELD_GROUPS` resolves on
   `RadarSystemEntity.model_fields`.
3. **`test_no_non_identity_field_in_multiple_groups`** — no field
   except `system_name` appears in more than one group.
4. **`test_every_flat_checklist_field_appears_exactly_once`** — every
   non-`system_field` field on the extraction-side `RadarSystemEntity`
   is in exactly one group; no field listed in groups is missing from
   the schema. **Expected-set formula:**
   ```python
   expected = {
       fname for fname, finfo in RadarSystemEntity.model_fields.items()
       if not (
           isinstance(finfo.json_schema_extra, dict)
           and finfo.json_schema_extra.get("system_field") is True
       )
   }
   expected |= set(RadarSystemEntity.model_config.get("graph_id_fields", []) or [])
   ```
   The grouped set is the union of all 5 group field-lists. Test
   asserts `expected == grouped`.
5. **`test_system_fields_are_excluded`** — fields tagged
   `system_field=True` (e.g. `confidence`) appear in no group.

Description-quality assertions (per sub-pass record class,
parametrized):

- Every field has a non-empty description.
- For numeric-typed fields (detected via recursive
  `typing.get_origin/get_args` traversal, not string matching on
  annotation strings), `field_info.examples` must be empty/None.
- Description text (lower-cased) must not contain `"typical"`,
  `"common radar bands"`, or `"forbidden values"` — these markers
  indicate prompt sanitization didn't run.

### 5.2 Smoke harness — `tests/integration/test_radar_field_groups_smoke.py`

Marked `@pytest.mark.integration`; skipped when docling-graph is offline.
Posts minimal `DoclingDocument` payloads to
`http://localhost:8002/extract-pass` for the **3 known-failing-in-Phase-A
cases**:

| Pass | Source text | Field | Expected range |
|---|---|---|---|
| `radar_power_rf` | "Fan Song transmitter peak power is 600 kW." | `tx_peak_power_kw` | [400, 800] |
| `radar_power_rf` | "Fan Song operates at 3000 MHz." | `nominal_rf_mhz` | [2900, 3100] |
| `radar_antenna` | "Fan Song antenna gain is 35 dBi." | `gain_dbi` | [33, 37] |

Per-case assertions:

- HTTP 200.
- `pass_output.radar_systems` non-empty.
- The matching entity is selected via `next((e for e in radar_systems
  if "Fan Song" in (e.get("system_name") or "")), None)` — never assume
  `[0]` is the right entity.
- Target field is a non-`None` `int`/`float` within the expected range.
- On failure, prints the full `pass_output` dict to stdout (visible in
  pytest `-v` output) for debugging.

Smoke request body **omits** `upstream_entities` for `document_only`
passes. The endpoint earlier rejected `document_only` requests when the
key was present, even with an empty list.

**Model assumption:** the smoke harness's expected ranges
(`tx_peak_power_kw [400, 800]`, `nominal_rf_mhz [2900, 3100]`,
`gain_dbi [33, 37]`) are calibrated against `gemma4:31b` (the Phase A
baseline model). If the live docling-graph service is rebuilt against
a different `DOCLING_GRAPH_LLM_MODEL`, the thresholds may need
recalibration — re-run the harness once on a known-good extraction
under the new model and update the ranges to bracket the observed
value rather than the source-text value verbatim.

## 6. Rollout

In commit order. Main stays green at every step.

1. **Add `_field_groups.py`** with `RADAR_FIELD_GROUPS` and the 5 group-
   membership contract tests. No manifest changes; tests pass against
   the still-active `radar_domain` because they assert field-set
   partitioning, independent of the manifest.

2. **Add `_radar_shared.py`** exporting `edge`,
   `RADAR_FORBIDDEN_SYSTEM_NAMES`, `RADAR_OPTIONAL_TEXT_FIELDS`,
   `validate_radar_system_name`, `make_root_sanitizer`. Marked legacy
   `radar_domain.py` keeps its own copies until step 4.

3. **Add the 5 sub-pass modules** + the description-quality contract
   test. Each module importable; description-quality test passes.

4. **Manifest cutover** — single commit:
   - Add 5 new pass entries.
   - Remove `radar_domain` entry.
   - Update `system_links.depends_on` to list all 6 entity passes.
   - Mark `extraction_schemas/radar_domain.py` legacy in module
     docstring (leave file in source).
   - Update the **3 app-code touchpoints** enumerated in §4.8:
     - `app/workers/pipeline.py::_DOMAIN_PASS_NAMES`
     - `docker/docling-graph/app/evidence_gate.py::_run_post_extraction_evidence_gate`
       dispatch
     - `docker/docling-graph/app/schemas.py` docstring example
   - Update **test fixtures** in
     `tests/docker/docling-graph/test_extract_pass_endpoint.py` and
     `tests/docker/docling-graph/test_service_identity_gate.py` from
     `pass_name="radar_domain"` to `pass_name="radar_identity"`.
   - Sweep remaining tests + dashboards for hardcoded `"radar_domain"`
     literal references; update to either generic phrasing or
     `radar_identity`.
   - Run the relevant unit + pipeline regression sweep; document any
     pre-existing unrelated failures.

5. **Add smoke harness** — `tests/integration/test_radar_field_groups_smoke.py`.

6. **Rebuild docling-graph + worker-graph**, re-ingest the SNR-75
   Wikipedia PDF, run the smoke harness against the live service.
   Compare against the Phase A baseline (5 strings, 0 numerics).

## 7. Success criteria

- All 5 group-membership contract tests pass.
- Description-quality contract passes for all 5 sub-pass record classes.
- Full relevant unit + pipeline regression sweep green; any pre-
  existing unrelated failures documented.
- At least 2 of 3 smoke cases extract a numeric value within the
  expected range. (`0/3` is not a code-correctness failure but signals
  that field grouping alone did not change gemma4:31b's behavior; in
  that case, Session 2's identity/parameter split + retry becomes
  mandatory.)
- Section endpoint smoke test on the FAN SONG vertex shows ≥1 numeric
  flat-checklist field (`gain_dbi`, `nominal_rf_mhz`, or
  `tx_peak_power_kw`) populated post re-ingest.

## 8. Risks + mitigations

| Risk | Mitigation |
|---|---|
| `system_links` upstream-refs builder reads `radar_domain` pass-name literal | Step 4 manifest sweep grep + targeted update before the cutover commit lands. |
| App-code pass-name literals silently bypass new sub-passes (`_DOMAIN_PASS_NAMES`, evidence-gate dispatch) | §4.8 enumerates the 3 known touchpoints; step 4 explicitly updates each. Smoke test in step 6 catches any missed dispatch (radar-side evidence-gate would no-op silently otherwise). |
| 5 parallel sub-passes overwhelm the Ollama backend (gemma4:31b can't do 5 concurrent) | If Ollama cannot handle sibling radar passes concurrently, cap worker/service concurrency or temporarily serialize `radar_*` passes via a depends_on chain (`radar_identity` → `radar_power_rf` → `radar_antenna` → ... ). Verify before step 6 by watching Ollama logs during a test ingest. |
| Worker `_parse_pass_response` chokes on new pass names | The function is pass-name-agnostic and loads `template_class` via the manifest. Verify in step 4 regression sweep. |
| Coverage checker / bundle-checker rejects the new manifest layout | Run `check_bundle()` after step 4 commit; expected 0 errors based on prior Phase 1 work. |
| Sub-pass record's `radar_systems` list-field name conflicts with merge expectations | All 5 sub-passes use the same `radar_systems` list-field name → uniform shape downstream. Verified compatible with library prompt-rendering pattern. |
| Description duplication drifts from canonical over time | Description-quality contract test catches the *failure modes* (numeric examples, typical ranges, FORBIDDEN-block leakage); description-prose drift accepted as Session 1 cost. |
| Smoke test runs on every PR, slowing CI | Marked `@pytest.mark.integration`; default `pytest tests/unit tests/pipeline` doesn't pick it up. |
| Manifest pass count changes break status/progress assumptions | Grep tests + UI + status-render code for assumed pass names/counts; verify `StageRun` rows and status signals render all new pass names without literal `radar_domain` references. |

## 9. Out-of-scope / deferred

- **Missile field-group split** — same pattern, follow-up session.
- **Identity/parameter split (Item #3)** — once field groups exist,
  `radar_identity` becomes a discovery pass and parameter passes
  receive its identities. Currently all sub-passes are `document_only`;
  Item #3 changes them to `document_plus_entity_refs`. Structural
  change worth its own session.
- **Group-scoped retry (Item #7)** — depends on Item #3 landing.
- **Evidence-span verification (Item #6)** — layers on whatever
  extraction shape exists; deferred to Session 3.
- **Per-pass diagnostics persistence (Item #9)** — Session 3.
- **Golden test harness with per-field recall/precision (Item #10)** —
  Session 3. Session 1's 3-case smoke is not a substitute, just an
  immediate-signal guardrail.
