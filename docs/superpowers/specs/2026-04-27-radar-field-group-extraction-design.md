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
focused semantic guide.

**Code-change scope:**
- **No merge / import / vertex-persistence changes** in the worker
  layer (`extraction_merge.py`, `_import_graph_phase_nodes`, ArcadeDB
  upsert path).
- **Worker quality-classification:** one frozenset in
  `app/workers/pipeline.py::_DOMAIN_PASS_NAMES` needs updating
  (§4.8).
- **docling-graph evidence-gate:** several pass-name dispatches in
  `apply_bundle_postprocessing` plus a refactor of
  `_clear_unsupported_radar_properties` to verify numeric values
  against batch evidence text instead of nulling them unconditionally
  (§4.8).
- **Orchestrator (potentially):** the upstream-ref builder for
  `system_links` may need a dedupe step if it doesn't already collapse
  duplicate identities across dependency-pass outputs (§4.5).

Full enumeration of touchpoints in §4.8.

## 3. Non-goals (Session 1)

- Do **not** split `missile_domain`. Same pattern; deferred to a follow-
  up session.
- Do **not** introduce identity-fed parameter passes (Item #3). All
  sub-passes remain `document_only` for Session 1.
- Do **not** add group-scoped retry (Item #7).
- Do **not** build the **full per-field evidence-span verification
  system** that Item #6 will eventually add (post-extraction
  validation that every persisted field's snippet appears in its
  source chunk, with mismatch logging and rejection thresholds).
  However, Session 1 **does require a lightweight evidence-text
  presence check** inside `_clear_unsupported_radar_properties` so
  explicit numeric values aren't nulled when they appear in the
  batch (§4.8). That narrow check is in-scope for correctness; the
  full Item #6 system is deferred.
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
- `validate_radar_system_name(value)` —
  `field_validator("system_name", mode="before")` body. **Scope:
  normalization + non-empty-identity check only** (whitespace strip,
  canonicalize via `canonicalize_identity_text`, reject empty / None).
  Does **not** enforce the forbidden-names list — that authority lives
  exclusively in `make_root_sanitizer` / `sanitize_entity_list`.
  Splitting concerns this way keeps a single source of truth for
  forbidden-name enforcement and prevents the two layers from drifting.
- `make_root_sanitizer(list_field, optional_text_fields)` — factory
  returning a `model_validator(mode="before")` body wired with the
  caller's list-field name and per-class text-field set. **The single
  authority for forbidden-name enforcement** — internally calls
  `sanitize_entity_list(forbidden_identities=RADAR_FORBIDDEN_SYSTEM_NAMES,
  ...)` so sub-pass modules don't have to import the forbidden-set
  constant directly. **The returned validator runs both
  `sanitize_entity_list` and `dedupe_entities_by_identity`**,
  mirroring the legacy `_sanitize_and_dedupe_root_entities` body in
  `radar_domain.py`. Sanitize-only factories silently break duplicate-
  emission handling.

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
  extraction-side schemas, **with two sanitization rules applied at
  copy time**:
  a. Strip the FORBIDDEN-values block from `system_name`'s
     description. Forbidden-name enforcement is delegated to
     `make_root_sanitizer` / `sanitize_entity_list` (validator-side,
     not prompt-side), so the description doesn't need the verbose
     list. Replace with a one-sentence summary.
  b. Strip "Typical X-band ground radars: ..." / "Common radar bands"
     / similar typical-value-range prose from numeric field
     descriptions (these confuse the model per Phase A diagnosis).
  The description-quality assertions in §5.1 enforce both rules.
  Per-Session-1 acceptance: explicit duplication preferred over a
  description-drift contract test (description-prose drift outside
  the two sanitization rules is accepted Session-1 cost).

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

**Upstream-ref dedupe requirement:** when `system_links`'s upstream
references are built from the dependency passes' outputs, the builder
**MUST dedupe by `(entity_type, normalized identity_values)`** across
all 6 dependency pass outputs. Otherwise five partial `RADAR_SYSTEM`
records with `system_name="Fan Song"` (one per radar sub-pass) would
become five upstream refs in the LLM prompt, wasting tokens and
encouraging duplicate emissions. The post-extraction normalizer in
`evidence_gate.py:720` collapses duplicates server-side, but the
prompt-side dedup is what saves token budget and keeps the LLM seeing
"there are 2 radars in this doc" not "there are 10". Locate the
upstream-ref builder in the orchestrator path that fires on
`input_mode == "document_plus_entity_refs"` and confirm or add the
dedup.

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
`depends_on`. No scheduling change is expected, but the
`system_links` upstream-ref builder must be verified and may need
identity dedupe per §4.5.

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
- **Evidence-gate post-processor logic** — the
  `_postprocess_air_defense_radars` function body's identity
  sanitization, status validation, and exact-text-field clearing
  logic (lines 401-417) are preserved as-is. Only two things change
  per §4.8:
    1. The `pass_name` dispatch in `apply_bundle_postprocessing`
       expands to recognize the 5 new sub-pass names.
    2. `_clear_unsupported_radar_properties`'s strict-null branch
       (lines 419-442) is refactored to verify each numeric value
       against batch evidence text via shared substring-matching
       helpers (§4.8 details). Without that refactor, every
       numeric extraction is nulled before the response leaves the
       service.
- **`merge_and_resolve`** — already keys by `LogicalIdentity`
  (`system_name`). 5 sub-passes emitting the same `system_name="Fan
  Song"` collapse to one `MergedEntityRecord`. Property merge unions
  per-pass props into one dict; confidence is `max()` across passes.
- **`_import_graph_phase_nodes`** — writes `_field_evidence` JSON
  property on the vertex. Already merges across passes; works as-is.

### 4.8 Touchpoints requiring code change

**Pass-name literals + radar numeric-postprocessing refactor.** Missing
any of these is a silent correctness regression, not a test failure.

- **`app/workers/pipeline.py::_DOMAIN_PASS_NAMES`** (around line 382) —
  frozenset feeds `_classify_extraction_quality`. Current value:
  `{"radar_domain", "missile_domain", "other_systems", "system_links"}`.
  After cutover, `radar_domain` no longer appears in `pass_outcomes`;
  a successful radar extraction via the 5 sub-passes wouldn't match
  the `domain_hit` check, and the pipeline run could be classified
  `degraded` / `anomaly` even when the new passes returned HIT.
  **Replacement set:**
  `{"radar_identity", "radar_power_rf", "radar_antenna",
  "radar_timing", "radar_modulation", "missile_domain",
  "system_links"}`. Note the replacement preserves `"system_links"`
  (otherwise system-link-only HITs lose domain-hit credit) and drops
  `"other_systems"` (no longer a manifest pass; dead in the existing
  set). Verify `_classify_extraction_quality`'s OR-logic treats *any*
  radar sub-pass returning HIT as a domain hit (not a per-pass AND).

- **`docker/docling-graph/app/evidence_gate.py::apply_bundle_postprocessing`**
  (around line 307) — dispatches to `_postprocess_air_defense_radars`
  only when `pass_name == "radar_domain"`. After cutover, none of the
  5 sub-passes match this string, so radar-side evidence-gate post-
  processing (status validation, identity sanity checks) is silently
  bypassed. Update the dispatch to recognize the 5 new names.
  **Idempotency contract:** each sub-pass invocation hands the post-
  processor only that group's slice of fields. The post-processor
  must (a) not assume all radar fields are present on a single record
  — only the group's subset is — and (b) merge gracefully across
  invocations on the same `system_name`.

- **`docker/docling-graph/app/evidence_gate.py::_clear_unsupported_radar_properties`**
  (around line 398) — **THIS IS A CORRECTNESS BLOCKER. Currently
  unconditionally sets 18 numeric fields to `None`** (lines 419-442:
  `erp_dbw`, `tx_peak_power_kw`, `gain_dbi`, `antenna_dim_*`,
  `beamwidth_*`, `nominal_rf_mhz`, `nominal_pri_usec`, `nominal_pd_usec`,
  `scan_period_sec`, `frequency_excursion_mhz`, `num_bits_in_code`,
  `pulses_per_dwell`, plus `antenna_photo`, `spoiled`,
  `coverage_limits_el_deg`, `confidence`). This was a "delete unsupported
  guesses" rule from before Phase A's auto-evidence + numeric-candidate
  system. Even a perfectly-extracted `gain_dbi=35.0` from a doc that
  says "antenna gain is 35 dBi" gets nulled here before the response
  leaves the service.

  **Refactor required as part of Session 1's cutover commit:** verify
  each numeric value against `evidence_text` via substring match with
  unit-aware variants. The exact-text branch (lines 401-417, for
  string fields like `nomenclature`) keeps its current
  `_value_is_quoted_in_text` check — that path is correct. Failing to
  fix this guarantees Session 1's smoke tests fail even when the LLM
  extracts correctly.

  **Implementation: extract a shared helper, do not duplicate.** The
  unit-variant matching logic already exists in
  `docker/docling-graph/app/provenance.py::_value_match_candidates`
  (with `_UNIT_HINTS_BY_SUFFIX` and `_normalize_text`). Refactor those
  helpers into a shared module — proposed location:
  `docker/docling-graph/app/_numeric_evidence.py` — and have BOTH
  `provenance.build_auto_field_evidence` AND the new
  `_clear_unsupported_radar_properties` consume the same
  `value_is_supported_by_text(value, field_name, evidence_text)`
  predicate. Two consumers, one source of truth. Avoid copy-pasting
  the unit table or the normalization regex.

- **`docker/docling-graph/app/schemas.py`** (around line 55) — docstring
  example references `pass_name="radar_domain"`. Low-risk; update the
  example to a current pass name (`radar_identity` is a fine
  representative).

**Test fixtures with hardcoded pass names:**

- `docker/docling-graph/tests/test_extract_pass_endpoint.py`
- `docker/docling-graph/tests/test_service_identity_gate.py`

Both pin `pass_name="radar_domain"` in their request fixtures. Default
plan: update to `pass_name="radar_identity"`.

**Special case:** `test_service_identity_gate.py` line 5 has
`from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain
import RadarDomainPass`. This is a **class import**, not a pass-name
literal. Because `radar_domain.py` is kept in source (legacy
reference per §6 step 4), the import still resolves. **Keep the
import as-is** — it explicitly verifies the legacy module still loads
even though it's not in the manifest. Add a one-line comment marking
the test as a legacy-loadability regression check.

**Manifest-shape and active-schema tests that must be updated:**

- `tests/unit/test_ontology_bundles.py` (lines 5-19) hard-asserts
  `len(m.passes) == 3` and the set
  `{"radar_domain", "missile_domain", "system_links"}`. After cutover
  this becomes 7 passes with a different set. **Update the test** to
  assert the new shape: `len(m.passes) == 7` and the set
  `{"radar_identity", "radar_power_rf", "radar_antenna",
  "radar_timing", "radar_modulation", "missile_domain",
  "system_links"}`.

- `tests/unit/test_extraction_schemas.py` (line 8 + line 14) imports
  `radar_domain, missile_domain, system_links` and parametrizes a
  `PASS_MODULES` list containing `(radar_domain, "RadarDomainPass")`.
  After cutover the active-schema parametrization should iterate the
  5 new sub-pass modules + missile_domain + system_links. Update
  imports + the `PASS_MODULES` list. The existing assertions
  (parameterized `model_validate(...)` smoke tests on each pass
  module) should pass on the new modules unchanged.

- `tests/integration/test_pr1_scaffolding_smoke.py` (lines 36-45,
  85-95 — both blocks contain `"reference", "radar_domain",
  "missile_domain"` literal lists). Update the literal pass-name lists
  to the new 7-pass shape, or rewrite the test to read from the
  manifest dynamically.

- `extraction_schemas/system_links.py` module docstring references
  upstream passes by name. Low-risk; update if it mentions
  `radar_domain` literally.

These tests assert against **active schema shape**, not generic
synthetic pass names. Distinct from the many `tests/unit/...` files
that use `"radar_domain"` as a fixture-only string and should be left
as-is.

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

**DoclingDocument shape:** in the synthesized doc, each text element
must use `"label": "text"`, **not** `"label": "paragraph"`. Existing
fixtures and service-injected text use `"text"`; `"paragraph"` may be
accepted by some Docling versions but the local fixture pattern is
`"text"`. Using the wrong label leads to debugging document-shape
issues instead of measuring extraction.

**Range calibration policy:** expected ranges bracket the
**source-text value with tolerance for unit-conversion rounding**, NOT
the model's observed output:

| Source text | Field | Expected range | Why these bounds |
|---|---|---|---|
| "600 kW" | `tx_peak_power_kw` | [400, 800] | ±33% to allow 600 / 600.0 / unit-conversion artefacts |
| "3000 MHz" | `nominal_rf_mhz` | [2900, 3100] | ±3% — frequency literal expected verbatim |
| "35 dBi" | `gain_dbi` | [33, 37] | ±6% — float-encoding tolerance |

If a future model emits `3500 MHz` for a doc that says `3000 MHz`,
the smoke test SHOULD fail. Recalibrating ranges to model output
would mask that as a regression. The ranges are the ground-truth
contract, not a model-fit metric. If a model legitimately can't hit
these tolerances, that's a model-rejection signal, not a range-
adjustment signal.

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
   - Update the **app-code touchpoints** enumerated in §4.8:
     - `app/workers/pipeline.py::_DOMAIN_PASS_NAMES` (frozenset update)
     - `docker/docling-graph/app/evidence_gate.py::apply_bundle_postprocessing`
       (pass-name dispatch — recognize 5 new sub-pass names)
     - `docker/docling-graph/app/evidence_gate.py::_clear_unsupported_radar_properties`
       (refactor numeric-field clearing to honor batch evidence text)
     - `docker/docling-graph/app/schemas.py` docstring example
     - Possibly the orchestrator's upstream-ref builder for
       `system_links` (dedupe by identity if not already)
   - Update **test fixtures** in
     `docker/docling-graph/tests/test_extract_pass_endpoint.py` and
     `docker/docling-graph/tests/test_service_identity_gate.py` from
     `pass_name="radar_domain"` to `pass_name="radar_identity"`.
     (Keep the `from ...radar_domain import RadarDomainPass` line in
     `test_service_identity_gate.py` as a legacy-loadability check;
     add a one-line comment.)
   - Update **manifest-shape assertions** in
     `tests/unit/test_ontology_bundles.py` (lines 5-19) from
     `len(m.passes) == 3` and the radar_domain/missile_domain/system_links
     set to `len(m.passes) == 7` and the new 7-pass set.
   - Sweep remaining tests for hardcoded `"radar_domain"` literal
     references. Most occurrences in `tests/unit/test_extraction_merge.py`,
     `test_pipeline_metrics.py`, `test_classify_extraction_quality.py`,
     etc. use `"radar_domain"` as a **generic synthetic pass name** in
     fixtures and **should be left as-is** unless they assert against
     the live manifest.
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
| App-code pass-name literals silently bypass new sub-passes (`_DOMAIN_PASS_NAMES`, evidence-gate dispatch) | §4.8 enumerates the known touchpoints; step 4 explicitly updates each. Smoke test in step 6 catches any missed dispatch (radar-side evidence-gate would no-op silently otherwise). |
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
