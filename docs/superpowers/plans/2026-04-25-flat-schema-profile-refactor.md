# Flat-Schema Profile Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the four starter query profiles (System Dossier, System Components, System RF Parameters, System Performance) onto the flat-checklist extraction schema, sync the canonical ontology entities to clear schema-drift xfails, and add LLM-extracted per-field evidence snippets.

**Architecture:** Three sequential phases. Phase 1 syncs the canonical Pydantic ontology with the flat extraction schemas, deletes orphan classes/edges, and tags every flat field with `profile_sections` + `profile_subgroup` metadata via `json_schema_extra`. Phase 2 introduces a new `kind="section_properties"` query profile that introspects those tags to project property field-groups instead of doing graph traversal, removes four legacy `/graph/system-*` endpoints, and migrates the persisted active registry. Phase 3 extends the docling-graph extraction prompt to emit per-field source snippets, resolves them to chunk RIDs deterministically, persists them on the entity vertex, and surfaces them in the section/dossier responses with a frontend popover.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, FastAPI, SQLAlchemy + Alembic, ArcadeDB (Cypher + SQL), Celery, React + TypeScript + Cytoscape, docling-graph LLM extraction service.

**Spec:** [docs/superpowers/specs/2026-04-25-flat-schema-profile-refactor-design.md](../specs/2026-04-25-flat-schema-profile-refactor-design.md)

---

## Pre-flight checklist

Run these once at the start of the session and before each chunk to confirm baseline:

- [ ] **P0: Read the spec.**

Run: `wc -l docs/superpowers/specs/2026-04-25-flat-schema-profile-refactor-design.md`
Expected: ≥ 880 lines. If less, the file is truncated — abort.

Use the @superpowers-extended-cc:test-driven-development skill for every code-bearing task.

- [ ] **P1: Confirm baseline test suite status.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline -q --ignore=tests/unit/test_extraction_schemas.py --ignore=tests/unit/test_specification_entity_validation.py 2>&1 | tail -3`
Expected: `1207 passed, 2 skipped, 5 xfailed, 20 warnings` (or near). The 5 xfails are the schema-drift markers Phase 1 removes; their disappearance is a Phase 1 success gate.

- [ ] **P2: Confirm stack is up.**

Run: `docker compose ps --format "table {{.Service}}\t{{.Status}}" | grep -E "api|arcadedb|postgres|worker-graph"`
Expected: all four services `Up`. If not, `./manage.sh --start` and wait 30 s.

- [ ] **P3: Confirm ArcadeDB schema baseline.**

Run:
```bash
AUTH="root:eip_arcadedb_secret"
curl -s -u "$AUTH" -X POST http://localhost:2480/api/v1/query/eip_knowledge_graph \
  -H "Content-Type: application/json" \
  -d '{"language":"sql","command":"SELECT count(*) AS c FROM RADAR_SYSTEM"}'
curl -s -u "$AUTH" -X POST http://localhost:2480/api/v1/query/eip_knowledge_graph \
  -H "Content-Type: application/json" \
  -d '{"language":"sql","command":"SELECT count(*) AS c FROM MISSILE_SYSTEM"}'
```
Expected: both return non-zero counts. If zero, ingest a doc first — Phase 2 verification needs at least one of each.

---

## Chunk 1: Phase 1 — Ontology sync

This chunk covers tasks 1-12: add flat-checklist fields to canonical entities, tag them with profile metadata, delete orphan classes, scrub stale edges on retained classes, update validation/coverage/relationships, drop xfail markers. **No API surface changes.** End state: clean `check_bundle()`, all 5 xfails passing, `ALL_ENTITIES`/`ENTITY_TYPES` only contains classes that have real extraction paths.

### Task 1: Add the four-bucket field-tag conventions

**Files:**
- Modify: `ontology_bundles/air_defense_v3/entities.py:31-65` (the `edge()` helper region)

The spec's §3.3 four-bucket convention (profile-mapped / system_metadata / identity / system_field) needs a small helper to avoid hand-typing the `json_schema_extra` shape on 60+ fields. We add three constructors that wrap `pydantic.Field` with the right `json_schema_extra` block.

- [ ] **Step 1: Read the existing `edge()` helper to mirror its shape.**

Read `ontology_bundles/air_defense_v3/entities.py:31-65`. Note that `edge()` returns `Field(default=..., description=..., examples=..., json_schema_extra={"edge_label": label, ...})`. We follow the same pattern.

- [ ] **Step 2: Write failing test for `profile_field()` helper.**

Create test file `tests/unit/test_entity_field_helpers.py`:

```python
"""Tests for json_schema_extra constructors used by canonical entities.

Exercises the four-bucket convention from the flat-schema profile
refactor spec §3.3: every domain field on RadarSystemEntity /
MissileSystemEntity falls into exactly one of profile-mapped,
system_metadata, identity, or system_field.
"""
import pytest
from pydantic import BaseModel
from ontology_bundles.air_defense_v3.entities import (
    profile_field, metadata_field, identity_field,
)


def test_profile_field_sets_profile_sections_and_subgroup():
    class M(BaseModel):
        gain_dbi: float | None = profile_field(
            sections=["rf_parameters", "performance"],
            subgroup="transmit",
            description="Antenna gain in dBi.",
            examples=[35.0],
            default=None,
        )

    info = M.model_fields["gain_dbi"]
    extra = info.json_schema_extra or {}
    assert extra["profile_sections"] == ["rf_parameters", "performance"]
    assert extra["profile_subgroup"] == "transmit"
    assert info.description == "Antenna gain in dBi."
    assert info.examples == [35.0]


def test_metadata_field_marks_system_metadata():
    class M(BaseModel):
        responsible_agency: str | None = metadata_field(
            description="Agency that owns the parametric record.",
            default=None,
        )

    info = M.model_fields["responsible_agency"]
    extra = info.json_schema_extra or {}
    assert extra["profile_sections"] == []
    assert extra["system_metadata"] is True


def test_identity_field_marks_identity():
    class M(BaseModel):
        nomenclature: str | None = identity_field(
            description="Formal designator (e.g. 5N63S).",
            default=None,
        )

    info = M.model_fields["nomenclature"]
    extra = info.json_schema_extra or {}
    assert extra["identity_field"] is True
    assert extra["profile_sections"] == []
```

- [ ] **Step 3: Run the test to verify it fails.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_entity_field_helpers.py -v`
Expected: `ImportError: cannot import name 'profile_field' from ontology_bundles.air_defense_v3.entities`.

- [ ] **Step 4: Add the three helpers to `entities.py`.**

Insert just below the existing `edge()` helper (around line 65):

```python
def profile_field(
    *,
    sections: list[str],
    subgroup: str,
    description: str,
    examples: list | None = None,
    default=None,
    default_factory=None,
    ge=None,
    le=None,
):
    """Field constructor for profile-mapped properties (spec §3.3 bucket 1).

    Tags the field with json_schema_extra={"profile_sections": [...],
    "profile_subgroup": "..."} so query_profiles' _project_field_groups
    can introspect and group by profile + subgroup.
    """
    extra = {"profile_sections": list(sections), "profile_subgroup": subgroup}
    kwargs = {"description": description, "json_schema_extra": extra}
    if examples is not None:
        kwargs["examples"] = examples
    if default_factory is not None:
        kwargs["default_factory"] = default_factory
    else:
        kwargs["default"] = default
    if ge is not None:
        kwargs["ge"] = ge
    if le is not None:
        kwargs["le"] = le
    return Field(**kwargs)


def metadata_field(
    *,
    description: str,
    examples: list | None = None,
    default=None,
):
    """Field constructor for system_metadata (spec §3.3 bucket 2).

    Real, indexed field — never surfaced by a starter profile. Used
    for audit trails, classifier IDs, status flags, review cadence.
    """
    extra = {"profile_sections": [], "system_metadata": True}
    kwargs = {
        "default": default,
        "description": description,
        "json_schema_extra": extra,
    }
    if examples is not None:
        kwargs["examples"] = examples
    return Field(**kwargs)


def identity_field(
    *,
    description: str,
    examples: list | None = None,
    default=None,
):
    """Field constructor for identity adjuncts (spec §3.3 bucket 3).

    Used for fields like `nomenclature` and the missile-schema `name`
    that are identity context but not graph_id_fields. The four-bucket
    contract test treats them as identity rather than profile-mapped
    or metadata.
    """
    extra = {"profile_sections": [], "identity_field": True}
    kwargs = {
        "default": default,
        "description": description,
        "json_schema_extra": extra,
    }
    if examples is not None:
        kwargs["examples"] = examples
    return Field(**kwargs)
```

- [ ] **Step 5: Re-run the test to verify it passes.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_entity_field_helpers.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit.**

```bash
git add ontology_bundles/air_defense_v3/entities.py tests/unit/test_entity_field_helpers.py
git commit -m "feat(ontology): four-bucket field tagging helpers (profile_field, metadata_field, identity_field)

Phase 1 task 1 of the flat-schema profile refactor. Adds the json_schema_extra
constructors used to tag flat-checklist fields on RadarSystemEntity /
MissileSystemEntity per spec §3.3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Migrate RadarSystemEntity flat fields onto canonical

**Files:**
- Modify: `ontology_bundles/air_defense_v3/entities.py:411-560` (RadarSystemEntity body)
- Reference: `ontology_bundles/air_defense_v3/extraction_schemas/radar_domain.py:87-540` (source field declarations)

Each flat field declared on the extraction-side `RadarSystemEntity` (in `extraction_schemas/radar_domain.py`) gets copied to the canonical class, with the right helper. The mapping from spec §3.3 is:

| Subgroup | Sections | Fields |
|---|---|---|
| `waveform` | `["rf_parameters"]` | `nominal_rf_mhz`, `frequency_excursion_mhz`, `nominal_pri_usec`, `nominal_pd_usec`, `inter_pulse`, `pulses_per_dwell`, `dwell_time`, `intra_pulse_mop`, `num_bits_in_code` |
| `antenna` | `["rf_parameters","components"]` | `antenna_dim_az_m`, `antenna_dim_el_m`, `beamwidth_az_deg`, `beamwidth_el_deg`, `gain_dbi`, `antenna_photo`, `spoiled`, `coverage_limits_el_deg` |
| `transmit` | `["rf_parameters","performance"]` | `tx_peak_power_kw`, `erp_dbw` |
| `scan` | `["rf_parameters","performance"]` | `scan_type`, `scan_period_sec` |
| `classification` | `["rf_parameters"]` | `emitter_function` |
| identity | — | `nomenclature` (`identity_field`) |
| metadata | — | `elnot`, `dieqp`, `asrd`, `system_status`, `responsible_agency`, `review_cycle`, `next_review_date` |

`system_name` is already declared on the canonical class as a graph_id_field — no change.

- [ ] **Step 1: Write failing contract test for RadarSystemEntity field bucketing.**

Append to `tests/unit/test_entity_field_helpers.py`:

```python
def test_radar_system_entity_every_field_is_bucketed():
    """Every domain field on canonical RadarSystemEntity falls into
    profile-mapped / system_metadata / identity / system_field
    (spec §3.3 four-bucket contract)."""
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity

    graph_id_fields = set(
        RadarSystemEntity.model_config.get("graph_id_fields", []) or []
    )

    misclassified = []
    for fname, finfo in RadarSystemEntity.model_fields.items():
        if fname in graph_id_fields:
            continue   # graph_id_fields are identity by definition
        extra = finfo.json_schema_extra or {}
        if not isinstance(extra, dict):
            extra = {}
        is_profile = bool(extra.get("profile_sections"))
        is_metadata = extra.get("system_metadata") is True
        is_identity = extra.get("identity_field") is True
        is_system = extra.get("system_field") is True
        is_edge = bool(extra.get("edge_label"))
        if not (is_profile or is_metadata or is_identity or is_system or is_edge):
            misclassified.append(fname)

    assert not misclassified, (
        f"RadarSystemEntity fields not in any of the four buckets "
        f"(profile-mapped / system_metadata / identity / system_field) "
        f"or edges: {misclassified}"
    )
```

- [ ] **Step 2: Run test to verify it fails.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_entity_field_helpers.py::test_radar_system_entity_every_field_is_bucketed -v`
Expected: FAIL — currently the canonical RadarSystemEntity has no flat fields, so the test passes vacuously OR fails on missing imports. If it passes vacuously, that's fine — the next step adds fields and exercises the bucketing.

- [ ] **Step 3: Open `entities.py:411` (RadarSystemEntity).**

Read the existing class body. Note the `Optional[List["AntennaEntity"]] = edge(...)` style fields — they get deleted in Task 4.5 (stale-edge cleanup, after orphan class deletion).

For now, **only add new flat fields**. Do not delete anything in this task.

- [ ] **Step 4: Add the flat fields with appropriate helper calls.**

Insert just before the `confidence` system field at the bottom of the class:

```python
    # ===== Flat-checklist fields (spec §3.3) =====
    # Waveform group — RF Parameters only
    nominal_rf_mhz: Optional[float] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Nominal radio frequency in MHz at which the radar transmits.",
        examples=[3000.0, 9300.0],
    )
    frequency_excursion_mhz: Optional[float] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Total instantaneous frequency excursion during a coherent processing interval, in MHz.",
        examples=[5.0, 50.0],
    )
    nominal_pri_usec: Optional[float] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Pulse repetition interval in microseconds.",
        examples=[1000.0],
    )
    nominal_pd_usec: Optional[float] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Pulse duration in microseconds.",
        examples=[1.0],
    )
    inter_pulse: Optional[str] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Inter-pulse modulation pattern.",
        examples=["staggered PRI", "fixed"],
    )
    pulses_per_dwell: Optional[int] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Pulses per coherent dwell.",
        examples=[16],
    )
    dwell_time: Optional[float] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Coherent dwell time, seconds.",
        examples=[0.05],
    )
    intra_pulse_mop: Optional[str] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Intra-pulse modulation on pulse.",
        examples=["LFM", "BPSK"],
    )
    num_bits_in_code: Optional[int] = profile_field(
        sections=["rf_parameters"], subgroup="waveform",
        description="Number of bits in the phase-code (when phase-coded MOP).",
        examples=[13],
    )

    # Antenna group — RF Parameters and Components
    antenna_dim_az_m: Optional[float] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Antenna aperture, azimuth dimension, meters.",
        examples=[6.0],
    )
    antenna_dim_el_m: Optional[float] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Antenna aperture, elevation dimension, meters.",
        examples=[2.0],
    )
    beamwidth_az_deg: Optional[float] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="One-way 3 dB azimuth beamwidth, degrees.",
        examples=[1.5],
    )
    beamwidth_el_deg: Optional[float] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="One-way 3 dB elevation beamwidth, degrees.",
        examples=[2.0],
    )
    gain_dbi: Optional[float] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Antenna gain, dBi.",
        examples=[35.0],
    )
    antenna_photo: Optional[bool] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Whether a photo of the antenna is available in source documents.",
    )
    spoiled: Optional[bool] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Whether the antenna pattern is spoiled (broadened) for surveillance.",
    )
    coverage_limits_el_deg: Optional[str] = profile_field(
        sections=["rf_parameters", "components"], subgroup="antenna",
        description="Elevation coverage limits, degrees (e.g. '0–60').",
        examples=["0–60"],
    )

    # Transmit group — RF Parameters and Performance
    tx_peak_power_kw: Optional[float] = profile_field(
        sections=["rf_parameters", "performance"], subgroup="transmit",
        description="Transmitter peak power, kilowatts.",
        examples=[600.0],
    )
    erp_dbw: Optional[float] = profile_field(
        sections=["rf_parameters", "performance"], subgroup="transmit",
        description="Effective radiated power, dBW.",
        examples=[88.0],
    )

    # Scan group — RF Parameters and Performance
    scan_type: Optional[str] = profile_field(
        sections=["rf_parameters", "performance"], subgroup="scan",
        description="Scan mechanism / mode (e.g. 'mechanical', 'phased-array', 'electronic').",
        examples=["phased-array"],
    )
    scan_period_sec: Optional[float] = profile_field(
        sections=["rf_parameters", "performance"], subgroup="scan",
        description="Scan revisit / repeat period, seconds.",
        examples=[10.0],
    )

    # Classification group — RF Parameters
    emitter_function: Optional[str] = profile_field(
        sections=["rf_parameters"], subgroup="classification",
        description="Primary emitter function (e.g. 'acquisition', 'tracking', 'engagement').",
        examples=["tracking"],
    )

    # Identity adjuncts
    nomenclature: Optional[str] = identity_field(
        description=(
            "Official military nomenclature — formal alphanumeric designator "
            "(JETDS for US, GRAU index for Russian). Distinct from system_name."
        ),
        examples=["AN/MPQ-65", "5N63S", "30N6E"],
    )

    # System metadata
    elnot: Optional[str] = metadata_field(
        description="Emitter library number (ELNOT) — IC enumeration.",
        examples=["E0123"],
    )
    dieqp: Optional[str] = metadata_field(
        description="Digital Intelligence Equipment Parameters cross-reference identifier.",
    )
    asrd: Optional[str] = metadata_field(
        description="ASRD identifier — IC source-of-record reference.",
    )
    system_status: Optional[str] = metadata_field(
        description="Operational status (e.g. 'in service', 'retired', 'in development').",
        examples=["in service"],
    )
    responsible_agency: Optional[str] = metadata_field(
        description="Agency or organization that owns the parametric record.",
        examples=["NASIC"],
    )
    review_cycle: Optional[str] = metadata_field(
        description="Review cadence for the parametric record.",
        examples=["annual"],
    )
    next_review_date: Optional[str] = metadata_field(
        description="Next scheduled review date for the record (YYYY-MM-DD).",
        examples=["2027-04-01"],
    )
```

Add the `profile_field`, `metadata_field`, `identity_field` names to the module's existing import / definition sweep — they live in the same file, so no import gymnastics.

- [ ] **Step 5: Run the contract test.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_entity_field_helpers.py::test_radar_system_entity_every_field_is_bucketed -v`
Expected: PASS — every flat field has a `json_schema_extra` block matching one of the four buckets.

- [ ] **Step 6: Run the import-clean check.**

Run: `.venv/bin/python -c "from ontology_bundles.air_defense_v3.entities import RadarSystemEntity; print(len(RadarSystemEntity.model_fields))"`
Expected: a count larger than the original (~50+ fields). If it fails with a forward-ref error referencing an orphan class, that's expected — we still have the old `edge()` fields pointing at to-be-deleted classes. Don't delete them yet; that's Task 4.

- [ ] **Step 7: Commit.**

```bash
git add ontology_bundles/air_defense_v3/entities.py tests/unit/test_entity_field_helpers.py
git commit -m "feat(ontology): add flat-checklist fields to canonical RadarSystemEntity

Phase 1 task 2 of the flat-schema profile refactor. Mirrors the field
list from extraction_schemas/radar_domain.py onto the canonical class,
tagged with profile_sections + profile_subgroup per spec §3.3 mapping.
Old edge fields pointing at orphan classes are kept for now; Task 4
deletes them after the orphans are removed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Migrate MissileSystemEntity flat fields onto canonical

**Files:**
- Modify: `ontology_bundles/air_defense_v3/entities.py:568-660` (MissileSystemEntity body)
- Reference: `ontology_bundles/air_defense_v3/extraction_schemas/missile_domain.py:95-497` (source field declarations)

Mapping from spec §3.3 (more granular than radar — multiple subgroups across `components` and `performance`):

| Subgroup | Sections | Fields |
|---|---|---|
| `airframe` | `["components","performance"]` | `body_length_m`, `body_diameter_m`, `total_mass_kg`, `missile_photo` |
| `seeker` | `["components","performance"]` | `seeker_type` |
| `booster` | `["components","performance"]` | `booster_time_sec`, `booster_thrust`, `booster_mass_kg` |
| `sustain` | `["components","performance"]` | `sustain_time_sec`, `sustain_thrust`, `sustain_mass_kg` |
| `ejector` | `["components","performance"]` | `ejector_time_sec`, `ejector_thrust`, `ejector_mass_kg` |
| `engagement` | `["performance"]` | `min_intercept_km`, `max_intercept_km`, `min_altitude_km`, `max_altitude_km`, `max_launch_angle_deg` |
| `kinematics` | `["performance"]` | `average_speed_mps`, `max_speed_mps`, `max_flyout_time_sec`, `flight_time_sec`, `coast_time_sec`, `total_burn_time_sec`, `intra_salvo_time_sec` |
| `guidance` | `["performance"]` | `guidance_type` |
| `classification` | `["performance"]` | `emitter_function` |
| identity | — | `nomenclature`, `name` (both `identity_field`) |
| metadata | — | `dieqp`, `asrd`, `system_status`, `responsible_agency`, `review_cycle`, `next_review_date` |

- [ ] **Step 1: Write failing contract test for MissileSystemEntity bucketing (mirror of radar version).**

Append to `tests/unit/test_entity_field_helpers.py`:

```python
def test_missile_system_entity_every_field_is_bucketed():
    """Every domain field on canonical MissileSystemEntity falls into
    profile-mapped / system_metadata / identity / system_field
    (spec §3.3 four-bucket contract). Mirror of the radar test."""
    from ontology_bundles.air_defense_v3.entities import MissileSystemEntity

    graph_id_fields = set(
        MissileSystemEntity.model_config.get("graph_id_fields", []) or []
    )

    misclassified = []
    for fname, finfo in MissileSystemEntity.model_fields.items():
        if fname in graph_id_fields:
            continue
        extra = finfo.json_schema_extra or {}
        if not isinstance(extra, dict):
            extra = {}
        is_profile = bool(extra.get("profile_sections"))
        is_metadata = extra.get("system_metadata") is True
        is_identity = extra.get("identity_field") is True
        is_system = extra.get("system_field") is True
        is_edge = bool(extra.get("edge_label"))
        if not (is_profile or is_metadata or is_identity or is_system or is_edge):
            misclassified.append(fname)

    assert not misclassified, (
        f"MissileSystemEntity fields not in any of the four buckets: {misclassified}"
    )
```

- [ ] **Step 2: Run the test, expect failure.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_entity_field_helpers.py::test_missile_system_entity_every_field_is_bucketed -v`
Expected: PASS vacuously OR fails on missing fields, depending on current canonical state. If it passes here, the next step still adds fields and the assertion stays meaningful as fields land.

- [ ] **Step 3: Add the flat fields to MissileSystemEntity.**

Insert just before the `confidence` field at the bottom of the class (similar to the radar block — full code below):

```python
    # ===== Flat-checklist fields (spec §3.3) =====
    # Airframe group — Components + Performance
    body_length_m: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="airframe",
        description="Missile body length, meters.",
        examples=[10.6],
    )
    body_diameter_m: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="airframe",
        description="Missile body diameter, meters.",
        examples=[0.5],
    )
    total_mass_kg: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="airframe",
        description="Total missile mass at launch, kilograms.",
        examples=[2300.0],
    )
    missile_photo: Optional[bool] = profile_field(
        sections=["components", "performance"], subgroup="airframe",
        description="Whether a photo of the missile is available in source documents.",
    )

    # Seeker group — Components + Performance
    seeker_type: Optional[str] = profile_field(
        sections=["components", "performance"], subgroup="seeker",
        description="Seeker type (e.g. 'semi-active radar', 'IR', 'inertial+command').",
        examples=["semi-active radar"],
    )

    # Booster group — Components + Performance
    booster_time_sec: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="booster",
        description="Booster burn duration, seconds.",
        examples=[5.0],
    )
    booster_thrust: Optional[str] = profile_field(
        sections=["components", "performance"], subgroup="booster",
        description="Booster thrust (string — units may vary in source documents).",
        examples=["50 kN"],
    )
    booster_mass_kg: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="booster",
        description="Booster section mass, kilograms.",
        examples=[200.0],
    )

    # Sustain group — Components + Performance
    sustain_time_sec: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="sustain",
        description="Sustain motor burn duration, seconds.",
        examples=[60.0],
    )
    sustain_thrust: Optional[str] = profile_field(
        sections=["components", "performance"], subgroup="sustain",
        description="Sustain motor thrust (string — units may vary).",
        examples=["10 kN"],
    )
    sustain_mass_kg: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="sustain",
        description="Sustain motor section mass, kilograms.",
        examples=[100.0],
    )

    # Ejector group — Components + Performance
    ejector_time_sec: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="ejector",
        description="Ejector charge burn duration, seconds.",
        examples=[0.2],
    )
    ejector_thrust: Optional[str] = profile_field(
        sections=["components", "performance"], subgroup="ejector",
        description="Ejector charge thrust (string — units may vary).",
    )
    ejector_mass_kg: Optional[float] = profile_field(
        sections=["components", "performance"], subgroup="ejector",
        description="Ejector charge mass, kilograms.",
    )

    # Engagement envelope — Performance only
    min_intercept_km: Optional[float] = profile_field(
        sections=["performance"], subgroup="engagement",
        description="Minimum intercept range, kilometers.",
        examples=[3.0],
    )
    max_intercept_km: Optional[float] = profile_field(
        sections=["performance"], subgroup="engagement",
        description="Maximum intercept range, kilometers.",
        examples=[150.0],
    )
    min_altitude_km: Optional[float] = profile_field(
        sections=["performance"], subgroup="engagement",
        description="Minimum engagement altitude, kilometers.",
        examples=[0.05],
    )
    max_altitude_km: Optional[float] = profile_field(
        sections=["performance"], subgroup="engagement",
        description="Maximum engagement altitude, kilometers.",
        examples=[30.0],
    )
    max_launch_angle_deg: Optional[float] = profile_field(
        sections=["performance"], subgroup="engagement",
        description="Maximum launch elevation angle from vertical, degrees.",
        examples=[60.0],
    )

    # Kinematics — Performance only
    average_speed_mps: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Average flight speed, meters per second.",
    )
    max_speed_mps: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Maximum flight speed, meters per second.",
        examples=[1100.0],
    )
    max_flyout_time_sec: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Maximum flyout duration to engagement, seconds.",
    )
    flight_time_sec: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Nominal flight duration, seconds.",
    )
    coast_time_sec: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Coast (unpowered) phase duration, seconds.",
    )
    total_burn_time_sec: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Total powered burn duration across all motors, seconds.",
    )
    intra_salvo_time_sec: Optional[float] = profile_field(
        sections=["performance"], subgroup="kinematics",
        description="Inter-shot interval within a salvo, seconds.",
    )

    # Guidance — Performance only
    guidance_type: Optional[str] = profile_field(
        sections=["performance"], subgroup="guidance",
        description="Guidance approach (e.g. 'command', 'inertial+command+terminal-active').",
        examples=["command + terminal-SARH"],
    )

    # Classification — Performance only
    emitter_function: Optional[str] = profile_field(
        sections=["performance"], subgroup="classification",
        description="Primary emitter function for the missile's seeker / data link.",
    )

    # Identity adjuncts
    nomenclature: Optional[str] = identity_field(
        description="Military designation or NATO reporting name.",
        examples=["MIM-104F"],
    )
    name: Optional[str] = identity_field(
        description=(
            "Secondary alias / common name. The missile schema's secondary "
            "alias field; rendered after nomenclature on the entity header."
        ),
    )

    # System metadata
    dieqp: Optional[str] = metadata_field(
        description="Digital Intelligence Equipment Parameters cross-reference identifier.",
    )
    asrd: Optional[str] = metadata_field(
        description="ASRD identifier — IC source-of-record reference.",
    )
    system_status: Optional[str] = metadata_field(
        description="Operational status.",
        examples=["in service"],
    )
    responsible_agency: Optional[str] = metadata_field(
        description="Agency or organization that owns the parametric record.",
    )
    review_cycle: Optional[str] = metadata_field(
        description="Review cadence for the parametric record.",
    )
    next_review_date: Optional[str] = metadata_field(
        description="Next scheduled review date for the record (YYYY-MM-DD).",
    )
```

- [ ] **Step 4: Run the contract test.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_entity_field_helpers.py::test_missile_system_entity_every_field_is_bucketed -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add ontology_bundles/air_defense_v3/entities.py tests/unit/test_entity_field_helpers.py
git commit -m "feat(ontology): add flat-checklist fields to canonical MissileSystemEntity

Phase 1 task 3 of the flat-schema profile refactor. Mirrors the field
list from extraction_schemas/missile_domain.py onto the canonical class,
tagged per spec §3.3 mapping. ~33 new fields across airframe / seeker /
booster / sustain / ejector / engagement / kinematics / guidance /
classification subgroups, plus identity adjuncts (nomenclature, name)
and 6 system_metadata fields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Delete orphan canonical classes

**Files:**
- Modify: `ontology_bundles/air_defense_v3/entities.py` (delete classes per spec §3.4)
- Modify: `ontology_bundles/air_defense_v3/entities.py:1346` (`ENTITY_TYPES` registry — remove deleted keys)

Spec §3.4 lists 27 classes for deletion: `FrequencyBandEntity`, `ModulationEntity`, `RfSignatureEntity`, `RfEmissionEntity`, `WaveformEntity`, `ScanPatternEntity`, `AntennaEntity`, `TransmitterEntity`, `ReceiverEntity`, `IfAmplifierEntity`, `SignalProcessingChainEntity`, `GuidanceMethodEntity`, `SeekerEntity`, `MissilePerformanceEntity`, `MissilePhysicalCharacteristicsEntity`, `PropulsionStackEntity`, `PropulsionStageEntity`, `CapabilityEntity`, `RadarPerformanceEntity`, `EngagementTimelineEntity`, `ForceStructureEntity`, `AssemblyEntity`, `SpecificationEntity`, `StandardEntity`, `ProcedureEntity`, `FailureModeEntity`, `TestEventEntity`.

This task **only deletes the standalone class definitions and ENTITY_TYPES entries**. The `edge(...)` fields on retained classes that point at these deleted classes are dropped in Task 5 (a focused stale-edge cleanup). Splitting the work makes failures more diagnosable — if `python -c "from ontology_bundles.air_defense_v3 import entities"` fails after Task 4, it's because Task 5 hasn't run yet, and the error message points at the still-broken edge fields.

- [ ] **Step 1: Audit for live references before deletion.**

Run for each candidate class:

```bash
for cls in FrequencyBandEntity ModulationEntity RfSignatureEntity RfEmissionEntity \
           WaveformEntity ScanPatternEntity AntennaEntity TransmitterEntity \
           ReceiverEntity IfAmplifierEntity SignalProcessingChainEntity \
           GuidanceMethodEntity SeekerEntity MissilePerformanceEntity \
           MissilePhysicalCharacteristicsEntity PropulsionStackEntity \
           PropulsionStageEntity CapabilityEntity RadarPerformanceEntity \
           EngagementTimelineEntity ForceStructureEntity AssemblyEntity \
           SpecificationEntity StandardEntity ProcedureEntity \
           FailureModeEntity TestEventEntity; do
  echo "=== $cls ==="
  grep -rn "\\b$cls\\b" --include='*.py' --include='*.yaml' --include='*.md' \
    /home/josh/development/EIP-MMDPP/ 2>/dev/null \
    | grep -v "ontology_bundles/air_defense_v3/entities.py" \
    | grep -v "ontology_bundles/air_defense_v3/coverage.yaml" \
    | grep -v "ontology_bundles/air_defense_v3/validation_matrix.py" \
    | grep -v "ontology_bundles/air_defense_v3/relationships.py" \
    | grep -v "app/services/dossier_service.py" \
    | head
done
```

Expected output: empty for all 27 classes (only references inside the bundle itself, plus `dossier_service.py` which is dead-code-but-still-imported until Phase 2). If you find a live consumer outside that list, surface it in this task's comments and decide whether to:

(a) update the consumer to not depend on the orphan, or
(b) reconsider that class for retention.

Don't proceed until the audit is clean against external consumers.

- [ ] **Step 2: Delete the 27 standalone class definitions in `entities.py`.**

Use the line ranges from `grep -n "^class.*Entity\|^class.*Pass" ontology_bundles/air_defense_v3/entities.py`. Each class spans roughly 30-100 lines including its `model_config` and field declarations. Delete them in reverse order (highest line numbers first) so earlier line numbers stay stable.

Tip: search for `^class FrequencyBandEntity` etc. and delete from `class X` through the line preceding the next `class Y`.

- [ ] **Step 3: Remove deleted classes from the `ENTITY_TYPES` registry.**

Open `entities.py:1346` and delete every key/value pair whose value is in the deletion set.

The registry keeps these (spec §3.4 retained list — minus `Alias` per spec §3.4 note):

```python
ALL_ENTITIES: dict[str, type[BaseModel]] = {
    "DOCUMENT": DocumentEntity,
    "SECTION": SectionEntity,
    "FIGURE": FigureEntity,
    "TABLE": TableEntity,
    "IMAGE": ImageEntity,
    "TEXT_BLOCK": TextBlockEntity,
    "ORGANIZATION": OrganizationEntity,
    "PLATFORM": PlatformEntity,
    "WEAPON_SYSTEM": WeaponSystemEntity,
    "EQUIPMENT_SYSTEM": EquipmentSystemEntity,
    "SUBSYSTEM": SubsystemEntity,
    "COMPONENT": ComponentEntity,
    "RADAR_SYSTEM": RadarSystemEntity,
    "MISSILE_SYSTEM": MissileSystemEntity,
    "AIR_DEFENSE_ARTILLERY_SYSTEM": AirDefenseArtillerySystemEntity,
    "ELECTRONIC_WARFARE_SYSTEM": ElectronicWarfareSystemEntity,
    "FIRE_CONTROL_SYSTEM": FireControlSystemEntity,
    "INTEGRATED_AIR_DEFENSE_SYSTEM": IntegratedAirDefenseSystemEntity,
    "LAUNCHER_SYSTEM": LauncherSystemEntity,
}
# Rebuild forward references for retained entity classes.
for _cls in ALL_ENTITIES.values():
    _cls.model_rebuild()
```

- [ ] **Step 4: Verify import fails (intentionally).**

Run: `.venv/bin/python -c "from ontology_bundles.air_defense_v3 import entities"`
Expected: `NameError: name 'AntennaEntity' is not defined` (or similar) — the retained-class edge fields still point at the deleted classes. This is what Task 5 fixes.

- [ ] **Step 5: Commit (deletion-only commit, broken state).**

```bash
git add ontology_bundles/air_defense_v3/entities.py
git commit -m "feat(ontology): delete 27 orphan canonical entity classes (spec §3.4)

Removes class definitions and ENTITY_TYPES registry entries for:
FrequencyBandEntity, ModulationEntity, RfSignatureEntity, RfEmissionEntity,
WaveformEntity, ScanPatternEntity, AntennaEntity, TransmitterEntity,
ReceiverEntity, IfAmplifierEntity, SignalProcessingChainEntity,
GuidanceMethodEntity, SeekerEntity, MissilePerformanceEntity,
MissilePhysicalCharacteristicsEntity, PropulsionStackEntity,
PropulsionStageEntity, CapabilityEntity, RadarPerformanceEntity,
EngagementTimelineEntity, ForceStructureEntity, AssemblyEntity,
SpecificationEntity, StandardEntity, ProcedureEntity, FailureModeEntity,
TestEventEntity.

These have no extraction path under the flat-checklist refactor and no
live consumers outside the bundle's internal cross-references.

Module is intentionally non-importable until Task 5 scrubs stale edge
fields on retained classes (RadarSystemEntity, MissileSystemEntity,
EquipmentSystemEntity, SubsystemEntity, etc.) that still reference the
deleted names.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Scrub stale edge fields on retained classes

**Files:**
- Modify: `ontology_bundles/air_defense_v3/entities.py` (delete `edge(...)` fields whose target is in the Task 4 deletion set, across every retained class)

Spec §3.4 enumerates the offenders. Walk every retained class and delete `edge(...)` fields whose annotation references any deleted class name. Mechanical audit: import succeeds when the cleanup is complete.

- [ ] **Step 1: Generate the per-class scrub list.**

Run:

```bash
DELETED='FrequencyBandEntity|ModulationEntity|RfSignatureEntity|RfEmissionEntity|WaveformEntity|ScanPatternEntity|AntennaEntity|TransmitterEntity|ReceiverEntity|IfAmplifierEntity|SignalProcessingChainEntity|GuidanceMethodEntity|SeekerEntity|MissilePerformanceEntity|MissilePhysicalCharacteristicsEntity|PropulsionStackEntity|PropulsionStageEntity|CapabilityEntity|RadarPerformanceEntity|EngagementTimelineEntity|ForceStructureEntity|AssemblyEntity|SpecificationEntity|StandardEntity|ProcedureEntity|FailureModeEntity|TestEventEntity'

grep -nE "$DELETED" /home/josh/development/EIP-MMDPP/ontology_bundles/air_defense_v3/entities.py | grep -v "^class "
```

Expected: a list of `<line>: <field_name>: List["<DeletedClass>"] = edge(` rows, one per stale edge.

- [ ] **Step 2: Delete each stale edge field block.**

For each line in the Step 1 output, locate the field block (the `name: ... = edge(...)` declaration spans 4-7 lines) and delete it. Don't worry about preserving comments — the canonical class is the source of truth, and the edge field's deletion is the right time to remove its inline doc too.

- [ ] **Step 3: Run the import gate.**

Run: `.venv/bin/python -c "from ontology_bundles.air_defense_v3 import entities; print('ok')"`
Expected: `ok`. If it fails with `NameError: name 'X' is not defined`, the grep in Step 1 missed a class name (probably because the annotation used a `Optional["X"]` form rather than `List["X"]`); add `Optional` patterns to the grep and rescrub.

- [ ] **Step 4: Run the full bucketing test suite.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_entity_field_helpers.py -v`
Expected: 5 passed (3 helper tests + radar bucketing + missile bucketing).

- [ ] **Step 5: Commit.**

```bash
git add ontology_bundles/air_defense_v3/entities.py
git commit -m "feat(ontology): scrub stale edge fields on retained canonical classes (spec §3.4)

Removes edge(...) fields whose target type was deleted in Task 4.
Affects RadarSystemEntity, MissileSystemEntity, EquipmentSystemEntity,
SubsystemEntity, WeaponSystemEntity, PlatformEntity,
AirDefenseArtillerySystemEntity, ElectronicWarfareSystemEntity,
FireControlSystemEntity, IntegratedAirDefenseSystemEntity,
LauncherSystemEntity. Module imports cleanly again.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Update `validation_matrix.py`

**Files:**
- Modify: `ontology_bundles/air_defense_v3/validation_matrix.py`

Drop every tuple whose subject or object is in the Task 4 deletion set. Also remove the `ASSERTION` rows that the file's own TODO comment (lines 35-43 area) marks as dead code from the B-1 commit.

- [ ] **Step 1: Generate the deletion list.**

Run:

```bash
DELETED='FREQUENCY_BAND|MODULATION|RF_SIGNATURE|RF_EMISSION|WAVEFORM|SCAN_PATTERN|ANTENNA|TRANSMITTER|RECEIVER|IF_AMPLIFIER|SIGNAL_PROCESSING_CHAIN|GUIDANCE_METHOD|SEEKER|MISSILE_PERFORMANCE|MISSILE_PHYSICAL_CHARACTERISTICS|PROPULSION_STACK|PROPULSION_STAGE|CAPABILITY|RADAR_PERFORMANCE|ENGAGEMENT_TIMELINE|FORCE_STRUCTURE|ASSEMBLY|SPECIFICATION|STANDARD|PROCEDURE|FAILURE_MODE|TEST_EVENT|ASSERTION'

grep -nE "($DELETED)" /home/josh/development/EIP-MMDPP/ontology_bundles/air_defense_v3/validation_matrix.py
```

- [ ] **Step 2: Delete each row.**

Each tuple is one logical line. Delete every line in the Step 1 output that's a tuple row (the file is mostly tuples — easy to scan). Leave any header / docstring text alone.

- [ ] **Step 3: Run the bundle checker (will fail until Tasks 7-9 catch up, but verify import).**

Run: `.venv/bin/python -c "from ontology_bundles.air_defense_v3 import validation_matrix; print(len(validation_matrix.RELATIONSHIP_TRIPLES))"`
Expected: an integer (the new triple count). If it raises `NameError`, a deleted name slipped through.

- [ ] **Step 4: Commit.**

```bash
git add ontology_bundles/air_defense_v3/validation_matrix.py
git commit -m "feat(ontology): drop validation rows for deleted entity types (spec §3.4)

Removes RELATIONSHIP_TRIPLES rows whose subject or object is one of the
27 orphan types deleted in Task 4. Also strips the ASSERTION rows the
file's own TODO comment marked as dead code from B-1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Update `relationships.py`

**Files:**
- Modify: `ontology_bundles/air_defense_v3/relationships.py`

Delete `RelationshipType` enum members for edge labels that only occurred between deleted classes. Spec §3.4 enumerates: `HAS_ANTENNA`, `HAS_RECEIVER`, `HAS_TRANSMITTER`, `EMITS`, `RADIATES`, `RECEIVES`, `OPERATES_IN_BAND`, `USES_WAVEFORM`, `USES_MODULATION`, `SPECIFIED_BY`, `HAS_SCAN`, `HAS_SEEKER`, `HAS_SIGNATURE`, `HAS_PROCESSING_CHAIN`, `HAS_IF_AMPLIFIER`, `HAS_PROPULSION`, `HAS_GUIDANCE`, `MEASURES` (audit), `MANUFACTURED_BY` (only if no live edges remain).

Keep: `IS_A`, `PART_OF`, `INSTALLED_ON`, `ASSOCIATED_WITH`, `CUES`, `CHILD_OF`, document/structure edges (`HAS_SECTION`, `HAS_FIGURE`, `HAS_TABLE`, `EXTRACTED_FROM`, `CONTAINS_TEXT`, `CONTAINS_IMAGE`, `NEXT_CHUNK`, `SAME_PAGE`, `SAME_SECTION`, `SAME_ARTIFACT`, `DERIVED_FROM`, `ABOUT`, `DESIGNATES`, `ALIAS_OF`, `DEFENDS`, `DEPLOYED_ON`, `DETECTS`, `AFFECTS`).

- [ ] **Step 1: List current `RelationshipType` members.**

Run: `grep -E "^\s+[A-Z_]+ =" ontology_bundles/air_defense_v3/relationships.py | head -60`

- [ ] **Step 2: Cross-reference against `validation_matrix.RELATIONSHIP_TRIPLES` (after Task 6).**

For each `RelationshipType` member, search whether it appears as the predicate in any retained tuple. If it doesn't, it's dead.

```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.validation_matrix import RELATIONSHIP_TRIPLES
from ontology_bundles.air_defense_v3.relationships import RelationshipType
predicates_used = {t[1] for t in RELATIONSHIP_TRIPLES}
for rt in RelationshipType:
    if rt not in predicates_used:
        print(f'unused: {rt.name}')
"
```

Expected: the list of `unused: NAME` rows is the deletion set for this step.

- [ ] **Step 3: Delete each unused enum member from `RelationshipType`.**

Open `relationships.py`, find the enum class, delete the lines listed in Step 2.

- [ ] **Step 4: Verify retained predicates still all resolve.**

Run: `.venv/bin/python -c "from ontology_bundles.air_defense_v3 import relationships, validation_matrix; print('ok', len(list(relationships.RelationshipType)), len(validation_matrix.RELATIONSHIP_TRIPLES))"`
Expected: `ok <N> <M>` with reasonable counts.

- [ ] **Step 5: Commit.**

```bash
git add ontology_bundles/air_defense_v3/relationships.py
git commit -m "feat(ontology): drop relationship types unused after orphan-class deletion

Removes RelationshipType enum members that no longer have any live
RELATIONSHIP_TRIPLES rows after Task 6's matrix cleanup. Retained the
structural edges (HAS_SECTION/FIGURE/TABLE, EXTRACTED_FROM, CONTAINS_*,
NEXT_CHUNK, SAME_*, DERIVED_FROM, ABOUT) and the cross-system
relationships system_links still emits (ASSOCIATED_WITH, CUES, plus
IS_A / PART_OF / INSTALLED_ON / CHILD_OF for structure).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Update `coverage.yaml`

**Files:**
- Modify: `ontology_bundles/air_defense_v3/coverage.yaml`

Drop YAML entries naming any deleted entity type or relationship label.

- [ ] **Step 1: Identify offending lines.**

Run: same `DELETED` regex from Task 6 against `coverage.yaml`:

```bash
grep -nE "($DELETED)" /home/josh/development/EIP-MMDPP/ontology_bundles/air_defense_v3/coverage.yaml
```

Run also for the dropped relationship labels:

```bash
DELETED_REL='HAS_ANTENNA|HAS_RECEIVER|HAS_TRANSMITTER|HAS_PROCESSING_CHAIN|HAS_IF_AMPLIFIER|HAS_SCAN|HAS_SEEKER|HAS_SIGNATURE|HAS_PROPULSION|HAS_GUIDANCE|EMITS|RADIATES|RECEIVES|OPERATES_IN_BAND|USES_WAVEFORM|USES_MODULATION|SPECIFIED_BY|MEASURES'

grep -nE "($DELETED_REL)" /home/josh/development/EIP-MMDPP/ontology_bundles/air_defense_v3/coverage.yaml
```

- [ ] **Step 2: Delete those YAML list entries.**

Each match is a `- TYPE_NAME` line in a YAML sequence. Delete the line; check that the parent key still has at least one entry afterward (if the parent's list becomes empty, leave `[]` or remove the parent depending on schema — spec doesn't constrain this, do whatever keeps the file valid YAML).

- [ ] **Step 3: Validate YAML still parses.**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('ontology_bundles/air_defense_v3/coverage.yaml'))"`
Expected: no exception.

- [ ] **Step 4: Commit.**

```bash
git add ontology_bundles/air_defense_v3/coverage.yaml
git commit -m "feat(ontology): scrub coverage.yaml of deleted entity / relationship types

Coverage lists no longer reference the 27 orphan entity types or the
edge labels removed alongside them. YAML still parses.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Bundle-checker validation

**Files:**
- Run: `check_bundle()` against `ontology_bundles/air_defense_v3/`

This is the integrative gate for Phase 1. After Tasks 1-8, `check_bundle()` should report 0 errors. If it doesn't, the error message points at the file that needs another pass.

- [ ] **Step 1: Locate the bundle checker entrypoint.**

Run: `grep -rn "def check_bundle" app/ ontology_bundles/ 2>/dev/null | head`

The function lives in `ontology_bundles/<something>/coverage_checker.py` or similar — find it and note the import path.

- [ ] **Step 2: Run the checker.**

Run (replace `<import_path>` with what Step 1 found):

```bash
.venv/bin/python -c "
from <import_path> import check_bundle
from pathlib import Path
errors, warnings = check_bundle(Path('ontology_bundles/air_defense_v3'))
print('ERRORS:')
for e in errors: print(f'  {e}')
print('WARNINGS:')
for w in warnings[:20]: print(f'  {w}')
print(f'Total: {len(errors)} errors, {len(warnings)} warnings')
"
```

Expected: `0 errors`. If non-zero, each error message identifies the violation; rebound to the relevant earlier task and fix.

- [ ] **Step 3: No commit (this is a check, not a code change).**

Move to Task 10.

---

### Task 10: Drop the 5 schema-drift `xfail` markers

**Files:**
- Modify: `tests/unit/test_docs_compliance_contracts.py`
- Modify: `tests/unit/contracts/test_extraction_schema_contract.py`
- Modify: `tests/unit/test_coverage_checker.py`

The 5 `xfail(strict=False)` decorators added during the prior debugging pass must come off. If any of them then fails, that's a real regression to fix in Phase 1 before merging.

- [ ] **Step 1: List the markers.**

Run:

```bash
grep -nB 5 "xfail.*Pre-existing schema drift\|xfail.*Pre-existing bundle drift\|xfail.*Obsolete import\|xfail.*pre-existing" \
  tests/unit/test_docs_compliance_contracts.py \
  tests/unit/contracts/test_extraction_schema_contract.py \
  tests/unit/test_coverage_checker.py
```

Expected: 5 `@pytest.mark.xfail(...)` decorator blocks.

- [ ] **Step 2: Delete each `@pytest.mark.xfail(...)` decorator block.**

Each spans 5-9 lines (the `xfail(...)` reason argument is multi-line). Delete the entire decorator down to and including its closing paren.

- [ ] **Step 3: Run the affected tests.**

Run:

```bash
SKIP_COV=1 .venv/bin/pytest \
  tests/unit/test_docs_compliance_contracts.py::test_extraction_views_subset_of_canonical_with_validator_parity \
  tests/unit/test_docs_compliance_contracts.py::test_descriptions_and_examples_on_extraction_relevant_fields \
  tests/unit/test_docs_compliance_contracts.py::test_pass_root_list_dedup_schema_local \
  tests/unit/contracts/test_extraction_schema_contract.py::test_no_nested_property_dicts \
  tests/unit/test_coverage_checker.py::test_check_bundle_passes_on_real_air_defense_v3 \
  -v
```

Expected: 5 passed.

- [ ] **Step 4: Commit.**

```bash
git add tests/unit/test_docs_compliance_contracts.py tests/unit/contracts/test_extraction_schema_contract.py tests/unit/test_coverage_checker.py
git commit -m "test(ontology): drop 5 xfail markers — schema drift resolved by Phase 1

The flat-checklist refactor's missing canonical fields, missing
descriptions/examples, and the obsolete other_systems import were the
xfail reasons. Tasks 1-9 fixed all five contract violations; the
markers come off.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Final Phase 1 regression sweep

- [ ] **Step 1: Run full unit + pipeline test sweep.**

Run:

```bash
SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline -p no:cacheprovider \
  --ignore=tests/unit/test_extraction_schemas.py \
  --ignore=tests/unit/test_specification_entity_validation.py \
  2>&1 | tail -3
```

Expected: `<N> passed, 2 skipped, 0 xfailed, <warnings>` — note **0 xfailed** (was 5 before this phase).

- [ ] **Step 2: If anything fails, classify.**

Pre-existing failures unrelated to Phase 1 are acceptable. New failures introduced by Phase 1 are not — go back and fix.

- [ ] **Step 3: Verify import paths still work end-to-end.**

Run:

```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3 import entities, relationships, validation_matrix
from ontology_bundles.air_defense_v3.entities import RadarSystemEntity, MissileSystemEntity, ALL_ENTITIES
print('ALL_ENTITIES count:', len(ALL_ENTITIES))
print('Radar field count:', len(RadarSystemEntity.model_fields))
print('Missile field count:', len(MissileSystemEntity.model_fields))
print('relationships count:', len(list(relationships.RelationshipType)))
print('validation triples:', len(validation_matrix.RELATIONSHIP_TRIPLES))
"
```

Expected: ALL_ENTITIES ~19-20, Radar ~30+, Missile ~33+, relationship enum ~25+, triples ~50+. Numbers depend on how many fields the canonical originally had.

- [ ] **Step 4: No commit (verification only).**

---

## Chunk 1 (Phase 1) acceptance gate

Before moving on:

- [ ] All 11 Phase 1 tasks committed.
- [ ] `pytest tests/unit -q` shows 0 xfailed.
- [ ] `python -c "from ontology_bundles.air_defense_v3 import entities, relationships, validation_matrix"` succeeds.
- [ ] `check_bundle()` returns 0 errors.
- [ ] `RadarSystemEntity` and `MissileSystemEntity` `model_fields` introspection passes the four-bucket contract test.

If any check fails, return to the relevant task. Phase 2 must not start with a broken Phase 1.


---

## Chunk 2: Phase 2 — Profile refactor

This chunk covers tasks 12-25: introduce `kind="section_properties"`, build the property-projection helpers, refactor `_fetch_section_items` and `execute_dossier_search`, add the alembic migration that updates the four starter profiles in active registries, delete the four legacy `/graph/system-*` endpoints + `dossier_service.py`, and wire up the frontend (TypeScript types, `<FieldGroupTable>`, `<DossierSectionList>`, `QueryPage` switch).

### Task 12: New schemas — `QueryProfileFieldGroup`, `QueryProfileFieldEntry`

**Files:**
- Modify: `app/schemas/query_profiles.py`
- Test: `tests/unit/test_query_profile_schemas.py` *(new)*

- [ ] **Step 1: Write failing test for the new shapes.**

Create `tests/unit/test_query_profile_schemas.py`:

```python
"""Schemas added by the flat-schema profile refactor (spec §6)."""
import pytest
from pydantic import ValidationError


def test_query_profile_field_entry_minimal():
    from app.schemas.query_profiles import QueryProfileFieldEntry
    entry = QueryProfileFieldEntry(
        name="gain_dbi", label="Gain (dBi)", value=35.0,
    )
    assert entry.name == "gain_dbi"
    assert entry.value == 35.0
    assert entry.evidence == []   # default empty


def test_query_profile_field_entry_with_metadata():
    from app.schemas.query_profiles import QueryProfileFieldEntry
    entry = QueryProfileFieldEntry(
        name="scan_type", label="Scan Type", value="phased-array",
        description="Scan mechanism / mode.",
        examples=["phased-array", "mechanical"],
    )
    assert entry.description == "Scan mechanism / mode."
    assert "mechanical" in entry.examples


def test_query_profile_field_group_groups_entries():
    from app.schemas.query_profiles import QueryProfileFieldGroup, QueryProfileFieldEntry
    grp = QueryProfileFieldGroup(
        subgroup="antenna", subgroup_label="Antenna",
        fields=[
            QueryProfileFieldEntry(name="gain_dbi", label="Gain", value=35.0),
            QueryProfileFieldEntry(name="beamwidth_az_deg", label="BW Az", value=1.5),
        ],
    )
    assert len(grp.fields) == 2
    assert grp.subgroup == "antenna"
```

- [ ] **Step 2: Run, expect ImportError.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profile_schemas.py -v`
Expected: ImportError.

- [ ] **Step 3: Add the schemas to `app/schemas/query_profiles.py`.**

Insert after `class QueryProfileTraversal`:

```python
class QueryProfileFieldEntry(APIModel):
    """One row in a property table — a single canonical field's value
    with its metadata, plus optional per-field evidence (Phase 3)."""
    name: str
    label: str
    value: Any
    description: Optional[str] = None
    examples: Optional[list[Any]] = None
    enum: Optional[list[str]] = None
    evidence: list["QueryProfileFieldEvidence"] = Field(default_factory=list)


class QueryProfileFieldGroup(APIModel):
    """A subgroup of fields rendered as one collapsible card on the
    section UI. `subgroup` is the canonical key from
    json_schema_extra['profile_subgroup']; `subgroup_label` is the
    title-cased display name."""
    subgroup: Optional[str] = None
    subgroup_label: Optional[str] = None
    fields: list[QueryProfileFieldEntry]
```

`QueryProfileFieldEvidence` is added in Task 32 (Phase 3); for now declare it as a forward reference and define a stub:

```python
class QueryProfileFieldEvidence(APIModel):
    """Phase 3 stub. Populated with snippet + element_uid + chunk
    metadata once the docling-graph extraction emits per-field
    provenance. Empty default for Phase 2."""
    supporting_snippet: str = ""
    element_uid: Optional[str] = None
    # Phase 3 adds the chunk-meta fields (chunk_id, document_name,
    # content_text, etc.); Phase 2 leaves the stub minimal so the
    # evidence list type-checks in TypeScript with empty objects.
```

Append `QueryProfileFieldEntry.model_rebuild()` at the bottom of the schema module to resolve the forward ref.

- [ ] **Step 4: Run tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profile_schemas.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit.**

```bash
git add app/schemas/query_profiles.py tests/unit/test_query_profile_schemas.py
git commit -m "feat(schemas): QueryProfileFieldEntry + QueryProfileFieldGroup (spec §6.1)

Phase 2 task 12. Adds the property-projection result shapes used by
section_properties profiles. QueryProfileFieldEvidence is a Phase 3
forward reference — declared here with a minimal stub so Phase 2
type-checks; expanded in Task 32.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Extend `QueryProfileSectionResponse` and `QueryProfileDossierSection` shapes

**Files:**
- Modify: `app/schemas/query_profiles.py`
- Modify: `tests/unit/test_query_profile_schemas.py`

- [ ] **Step 1: Add tests for the response-shape additions.**

Append to `tests/unit/test_query_profile_schemas.py`:

```python
def test_section_response_default_field_groups_empty():
    from app.schemas.query_profiles import QueryProfileSectionResponse
    from app.schemas.graph_store import GraphEntityResult

    resp = QueryProfileSectionResponse(
        registry_id=None, profile_id="p", profile_label="P",
        resolved_root=GraphEntityResult(name="X", entity_type="RADAR_SYSTEM"),
        total=0,
    )
    assert resp.field_groups == []
    assert resp.related_systems == []
    assert resp.items == []


def test_dossier_section_carries_kind_discriminator():
    from app.schemas.query_profiles import QueryProfileDossierSection
    sec = QueryProfileDossierSection(
        profile_id="system_rf_parameters", profile_label="RF",
        kind="section_properties",
    )
    assert sec.kind == "section_properties"
    assert sec.field_groups == []
    assert sec.items == []
```

- [ ] **Step 2: Update the schemas.**

In `app/schemas/query_profiles.py`:

```python
class QueryProfileSectionResponse(APIModel):
    registry_id: Optional[uuid.UUID] = None
    profile_id: str
    profile_label: str
    resolved_root: GraphEntityResult
    field_groups: list[QueryProfileFieldGroup] = Field(default_factory=list)
    related_systems: list[GraphEntityResult] = Field(default_factory=list)
    items: list[GraphEntityResult] = Field(default_factory=list)
    total: int


class QueryProfileDossierSection(APIModel):
    profile_id: str
    profile_label: str
    kind: Literal["section", "section_properties"]
    field_groups: list[QueryProfileFieldGroup] = Field(default_factory=list)
    related_systems: list[GraphEntityResult] = Field(default_factory=list)
    items: list[GraphEntityResult] = Field(default_factory=list)


class QueryProfileDossierResponse(APIModel):
    registry_id: Optional[uuid.UUID] = None
    profile_id: str
    profile_label: str
    resolved_root: GraphEntityResult
    aliases: list[str] = Field(default_factory=list)
    sections: list[QueryProfileDossierSection] = Field(default_factory=list)
    total: int = 0
```

(`aliases` preserved per spec rev-4 #M4.)

- [ ] **Step 3: Run the tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profile_schemas.py -v`
Expected: 5 passed.

- [ ] **Step 4: Commit.**

```bash
git add app/schemas/query_profiles.py tests/unit/test_query_profile_schemas.py
git commit -m "feat(schemas): extend section/dossier responses for property profiles (spec §6)

field_groups + related_systems on QueryProfileSectionResponse;
kind discriminator + items + field_groups + related_systems on
QueryProfileDossierSection; aliases preserved on
QueryProfileDossierResponse for back-compat (spec rev-4 #M4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Add `kind="section_properties"`, `_CANONICAL_ROOT_ENTITY_TYPES`, and validator

**Files:**
- Modify: `app/schemas/query_profiles.py`
- Modify: `tests/unit/test_query_profile_schemas.py`

- [ ] **Step 1: Tests for the new kind + validator.**

Append:

```python
def test_section_properties_requires_profile_sections():
    from app.schemas.query_profiles import QueryProfileDefinition
    with pytest.raises(ValidationError, match="profile_sections"):
        QueryProfileDefinition(
            id="bad", label="Bad", kind="section_properties",
            root_entity_types=["RADAR_SYSTEM"],
            profile_sections=[],   # empty — should fail
        )


def test_section_properties_root_must_be_canonical():
    from app.schemas.query_profiles import QueryProfileDefinition
    with pytest.raises(ValidationError, match="root_entity_types"):
        QueryProfileDefinition(
            id="bad", label="Bad", kind="section_properties",
            root_entity_types=["NONSENSE_TYPE"],
            profile_sections=["rf_parameters"],
        )


def test_section_properties_valid():
    from app.schemas.query_profiles import QueryProfileDefinition
    p = QueryProfileDefinition(
        id="ok", label="OK", kind="section_properties",
        root_entity_types=["RADAR_SYSTEM"],
        profile_sections=["rf_parameters"],
    )
    assert p.kind == "section_properties"
```

- [ ] **Step 2: Update `kind` literal and validator.**

Add at module top:

```python
# Module-local single source of truth for which canonical entity classes
# section_properties profiles can target. Kept in sync with
# _CANONICAL_BY_ENTITY_TYPE in app.services.query_profiles via a
# contract test (Task 15) — the schema layer must not import from the
# service layer (would create a circular dep).
_CANONICAL_ROOT_ENTITY_TYPES: frozenset[str] = frozenset({
    "RADAR_SYSTEM", "MISSILE_SYSTEM",
})
```

Update `QueryProfileDefinition`:

```python
class QueryProfileDefinition(APIModel):
    id: str = Field(..., min_length=1, max_length=100)
    label: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    kind: Literal["section", "section_properties", "dossier"] = "section"
    exposed: bool = True
    root_entity_types: list[str] = Field(default_factory=list)
    target_entity_types: list[str] = Field(default_factory=list)
    traversals: list[QueryProfileTraversal] = Field(default_factory=list)
    profile_sections: list[str] = Field(default_factory=list)
    include_associated_systems: bool = False
    section_profile_ids: list[str] = Field(default_factory=list)
    placeholder_query: Optional[str] = None

    # ... existing field_validator stays unchanged ...

    @model_validator(mode="after")
    def validate_shape(self):
        if self.kind == "section" and not self.traversals:
            raise ValueError("Section profiles require at least one traversal")
        if self.kind == "dossier" and not self.section_profile_ids:
            raise ValueError("Dossier profiles require at least one section_profile_id")
        if self.kind == "section_properties":
            if not self.profile_sections:
                raise ValueError(
                    "section_properties profiles require non-empty profile_sections"
                )
            unknown = [
                t for t in self.root_entity_types
                if t not in _CANONICAL_ROOT_ENTITY_TYPES
            ]
            if unknown:
                raise ValueError(
                    f"section_properties profiles' root_entity_types must be "
                    f"in {sorted(_CANONICAL_ROOT_ENTITY_TYPES)}; got unknown: {unknown}"
                )
        return self
```

- [ ] **Step 3: Run tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profile_schemas.py -v`
Expected: 8 passed.

- [ ] **Step 4: Commit.**

```bash
git add app/schemas/query_profiles.py tests/unit/test_query_profile_schemas.py
git commit -m "feat(schemas): kind=section_properties + _CANONICAL_ROOT_ENTITY_TYPES validation (spec §4.2)

Adds section_properties kind, profile_sections + include_associated_systems
fields, and a validator enforcing non-empty profile_sections and
root_entity_types ⊆ _CANONICAL_ROOT_ENTITY_TYPES. The constant lives in
the schema layer per spec rev-3 #M5; service-layer dispatch is kept in
sync via the contract test in Task 15.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: Service-side `_CANONICAL_BY_ENTITY_TYPE` + sync contract

**Files:**
- Modify: `app/services/query_profiles.py`
- Test: `tests/unit/test_query_profiles.py`

- [ ] **Step 1: Write failing sync-contract test.**

Append to `tests/unit/test_query_profiles.py`:

```python
def test_canonical_root_entity_types_in_sync_with_dispatch():
    """Schema layer's _CANONICAL_ROOT_ENTITY_TYPES must match service
    layer's _CANONICAL_BY_ENTITY_TYPE keys. Single source of truth in
    the schema (no circular import); contract test enforces parity."""
    from app.schemas.query_profiles import _CANONICAL_ROOT_ENTITY_TYPES
    from app.services.query_profiles import _CANONICAL_BY_ENTITY_TYPE
    assert set(_CANONICAL_BY_ENTITY_TYPE.keys()) == _CANONICAL_ROOT_ENTITY_TYPES
```

- [ ] **Step 2: Run, expect ImportError on `_CANONICAL_BY_ENTITY_TYPE`.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profiles.py::test_canonical_root_entity_types_in_sync_with_dispatch -v`
Expected: ImportError.

- [ ] **Step 3: Add the dispatch to `app/services/query_profiles.py`.**

At an appropriate spot near the top:

```python
from ontology_bundles.air_defense_v3.entities import (
    RadarSystemEntity, MissileSystemEntity,
)

# Service-side dispatch from entity_type string → canonical Pydantic
# class. Used by _project_field_groups to introspect which class's
# json_schema_extra to walk for a given resolved root. Kept in sync
# with app.schemas.query_profiles._CANONICAL_ROOT_ENTITY_TYPES via the
# contract test in tests/unit/test_query_profiles.py.
_CANONICAL_BY_ENTITY_TYPE: dict[str, type] = {
    "RADAR_SYSTEM": RadarSystemEntity,
    "MISSILE_SYSTEM": MissileSystemEntity,
}


def _canonical_class_for(entity_type: str):
    cls = _CANONICAL_BY_ENTITY_TYPE.get(entity_type)
    if cls is None:
        raise ValueError(
            f"No canonical Pydantic class registered for entity_type={entity_type!r}; "
            "section_properties profiles only run against types listed in "
            "_CANONICAL_BY_ENTITY_TYPE."
        )
    return cls
```

- [ ] **Step 4: Run the sync test.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profiles.py::test_canonical_root_entity_types_in_sync_with_dispatch -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add app/services/query_profiles.py tests/unit/test_query_profiles.py
git commit -m "feat(query_profiles): _CANONICAL_BY_ENTITY_TYPE dispatch + sync contract (spec §4.5)

Service-side dispatch from entity_type string to canonical Pydantic class.
Schema-layer _CANONICAL_ROOT_ENTITY_TYPES is the single source of truth;
contract test asserts parity to prevent drift.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: Implement `_project_field_groups`

**Files:**
- Modify: `app/services/query_profiles.py`
- Test: `tests/unit/test_query_profiles.py`

- [ ] **Step 1: Write failing tests.**

Append:

```python
def test_project_field_groups_groups_by_subgroup():
    """Walks canonical model_fields, picks fields whose
    profile_sections contains the requested section, groups by
    profile_subgroup, sorts deterministically."""
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity
    from app.services.query_profiles import _project_field_groups

    instance_data = {
        "name": "Fan Song",
        "system_name": "Fan Song",
        "gain_dbi": 35.0,
        "beamwidth_az_deg": 1.5,
        "tx_peak_power_kw": 600.0,
        "nominal_rf_mhz": 3000.0,
        "max_speed_mps": None,   # populated null — skipped
    }
    groups = _project_field_groups(RadarSystemEntity, instance_data, "rf_parameters")

    # antenna group has 2 populated rf_parameters fields
    antenna = next(g for g in groups if g.subgroup == "antenna")
    field_names = {f.name for f in antenna.fields}
    assert "gain_dbi" in field_names
    assert "beamwidth_az_deg" in field_names

    # transmit group has 1 populated rf_parameters field
    transmit = next(g for g in groups if g.subgroup == "transmit")
    assert {f.name for f in transmit.fields} == {"tx_peak_power_kw"}


def test_project_field_groups_skips_none_fields():
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity
    from app.services.query_profiles import _project_field_groups

    instance_data = {"name": "X", "gain_dbi": None}
    groups = _project_field_groups(RadarSystemEntity, instance_data, "rf_parameters")

    for g in groups:
        for f in g.fields:
            assert f.value is not None


def test_project_field_groups_only_returns_requested_section():
    """A field tagged ['rf_parameters', 'performance'] appears in both
    sections. A field tagged only ['performance'] appears only in
    performance projections."""
    from ontology_bundles.air_defense_v3.entities import MissileSystemEntity
    from app.services.query_profiles import _project_field_groups

    instance_data = {
        "system_name": "SA-2",
        "max_speed_mps": 1100.0,         # performance only
        "body_length_m": 10.6,           # components + performance
    }
    perf = _project_field_groups(MissileSystemEntity, instance_data, "performance")
    comp = _project_field_groups(MissileSystemEntity, instance_data, "components")

    perf_names = {f.name for g in perf for f in g.fields}
    comp_names = {f.name for g in comp for f in g.fields}

    assert "max_speed_mps" in perf_names
    assert "max_speed_mps" not in comp_names
    assert "body_length_m" in perf_names
    assert "body_length_m" in comp_names


def test_project_field_groups_carries_metadata():
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity
    from app.services.query_profiles import _project_field_groups

    instance_data = {"gain_dbi": 35.0}
    groups = _project_field_groups(RadarSystemEntity, instance_data, "rf_parameters")
    antenna = next(g for g in groups if g.subgroup == "antenna")
    gain = next(f for f in antenna.fields if f.name == "gain_dbi")
    assert gain.description and "gain" in gain.description.lower()
    assert gain.label   # title-cased version of name
```

- [ ] **Step 2: Run, expect ImportError on `_project_field_groups`.**

- [ ] **Step 3: Implement the helper.**

Add to `app/services/query_profiles.py`:

```python
def _human_label(field_name: str) -> str:
    """Convert canonical field name to a display label.
    'gain_dbi' → 'Gain (dBi)' is too cute; default to title-case."""
    return field_name.replace("_", " ").title()


def _project_field_groups(
    canonical_cls: type,
    instance_data: dict,
    profile_section: str,
) -> list:
    """Walk canonical_cls.model_fields, pick fields whose
    json_schema_extra['profile_sections'] contains *profile_section*,
    group by 'profile_subgroup'. Skip fields where instance_data[name]
    is None. Returns deterministically ordered groups (by subgroup
    name asc; fields by name asc within group).

    Spec §4.3.
    """
    from app.schemas.query_profiles import (
        QueryProfileFieldEntry, QueryProfileFieldGroup,
    )

    groups_by_subgroup: dict[str, list[QueryProfileFieldEntry]] = {}

    for fname, finfo in canonical_cls.model_fields.items():
        extra = finfo.json_schema_extra or {}
        if not isinstance(extra, dict):
            continue
        sections = extra.get("profile_sections") or []
        if profile_section not in sections:
            continue
        value = instance_data.get(fname)
        if value is None:
            continue
        subgroup = extra.get("profile_subgroup") or ""
        entry = QueryProfileFieldEntry(
            name=fname,
            label=_human_label(fname),
            value=value,
            description=finfo.description,
            examples=list(finfo.examples) if finfo.examples else None,
            enum=extra.get("enum"),
        )
        groups_by_subgroup.setdefault(subgroup, []).append(entry)

    # Sort fields within each group, then sort groups
    out: list[QueryProfileFieldGroup] = []
    for subgroup_key in sorted(groups_by_subgroup.keys()):
        entries = sorted(groups_by_subgroup[subgroup_key], key=lambda e: e.name)
        out.append(QueryProfileFieldGroup(
            subgroup=subgroup_key or None,
            subgroup_label=_human_label(subgroup_key) if subgroup_key else None,
            fields=entries,
        ))
    return out
```

- [ ] **Step 4: Run tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profiles.py -k project_field_groups -v`
Expected: 4 passed.

- [ ] **Step 5: Commit.**

```bash
git add app/services/query_profiles.py tests/unit/test_query_profiles.py
git commit -m "feat(query_profiles): _project_field_groups (spec §4.3)

Property-projection helper that introspects a canonical Pydantic class's
json_schema_extra to filter and group fields by profile section /
subgroup. Skips None values; sorts deterministically; carries
description/examples/enum metadata to the UI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 17: `get_entity_by_rid` + `get_associated_systems` on ArcadeDB graph store

**Files:**
- Modify: `app/services/arcadedb_graph.py`
- Test: `tests/unit/test_arcadedb_graph.py` (existing)

- [ ] **Step 1: Tests against the live ArcadeDB.**

Append to `tests/unit/test_arcadedb_graph.py`:

```python
class TestGetEntityByRid:
    async def test_returns_entity_by_rid(self):
        from app.services.arcadedb_graph import ArcadeDBGraphStore
        client = _make_client(query_result=[
            {"@rid": "#40:0", "@type": "MISSILE_SYSTEM", "system_name": "SA-2",
             "max_speed_mps": 1100.0, "gain_dbi": None}
        ])
        store = _graph(client)
        result = await store.get_entity_by_rid("#40:0")
        assert result["system_name"] == "SA-2"
        assert result["max_speed_mps"] == 1100.0


class TestGetAssociatedSystems:
    async def test_includes_both_directions(self):
        from app.services.arcadedb_graph import ArcadeDBGraphStore
        # First call: SELECT @type → MISSILE_SYSTEM
        # Then 1 MATCH against ASSOCIATED_WITH+CUES bothE
        client = _make_client()
        call_count = {"n": 0}

        async def _q(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [{"node_type": "MISSILE_SYSTEM"}]
            return [
                {"node_id": "#37:1", "name": "Fan Song",
                 "entity_type": "RADAR_SYSTEM", "score": None,
                 "relationship_types": ["ASSOCIATED_WITH"]},
            ]

        client.query = _q
        store = _graph(client)
        results = await store.get_associated_systems("#40:0")
        assert len(results) == 1
        assert results[0].name == "Fan Song"
```

- [ ] **Step 2: Run, expect AttributeError.**

- [ ] **Step 3: Implement on `arcadedb_graph.py`.**

Add to `ArcadeDBGraphStore`:

```python
async def get_entity_by_rid(self, rid: str) -> dict[str, Any]:
    """Return all properties of the vertex at the given RID, including
    nullable ones. Used by section_properties profiles to feed
    _project_field_groups."""
    if not rid.startswith("#"):
        raise ValueError(f"get_entity_by_rid expects an ArcadeDB RID (got {rid!r})")
    rows = await self._client.query(
        self._database, "sql",
        f"SELECT @rid, @type, * FROM {rid}",
    )
    return rows[0] if rows else {}


async def get_associated_systems(self, node_id: str) -> list[GraphEntityResult]:
    """Return systems linked by ASSOCIATED_WITH or CUES in either direction.

    Spec §4.6. Used by the System Components profile's `related_systems`
    block. Resolves @type for typed MATCH (ArcadeDB 26.5.x throws
    UnsupportedOperationException without it), traverses bothE() across
    the two relevant edge labels, deduplicates by RID, returns up to 25.
    """
    rid = await self._resolve_rid(node_id)
    if not rid:
        return []
    type_rows = await self._client.query(
        self._database, "sql",
        f"SELECT @type AS node_type FROM {rid}",
    )
    if not type_rows:
        return []
    seed_type = type_rows[0].get("node_type")
    if seed_type not in _CANONICAL_ROOT_ENTITY_TYPES_RUNTIME:
        # We could broaden this later, but for now profiles only
        # call this for RADAR_SYSTEM / MISSILE_SYSTEM seeds.
        return []

    sql = (
        f"MATCH {{type: {seed_type}, as: src, where: (@rid = {rid})}}"
        f".bothE('ASSOCIATED_WITH', 'CUES') {{as: e}}"
        f".bothV() {{as: tgt, where: (@rid <> {rid})}} "
        f"RETURN tgt.@rid AS node_id, tgt.name AS name, "
        f"tgt.@type AS entity_type, tgt.canonical_name AS canonical_name, "
        f"e.@type AS rel_type "
        f"LIMIT 25"
    )
    try:
        rows = await self._client.query(self._database, "sql", sql)
    except Exception:
        return []

    seen: set[str] = set()
    out: list[GraphEntityResult] = []
    for r in rows:
        nid = str(r.get("node_id", ""))
        if not nid or nid in seen:
            continue
        seen.add(nid)
        out.append(GraphEntityResult(
            node_id=nid,
            name=r.get("name", ""),
            entity_type=r.get("entity_type", ""),
            canonical_name=r.get("canonical_name"),
            relationship_types=[r.get("rel_type", "")] if r.get("rel_type") else [],
        ))
    return out
```

Add the runtime constant at module top (mirrors `_CANONICAL_BY_ENTITY_TYPE` in the service layer to keep the graph layer free of cross-service imports):

```python
# Local mirror — kept consistent with app.services.query_profiles via
# the same Task 15 sync contract test.
_CANONICAL_ROOT_ENTITY_TYPES_RUNTIME: frozenset[str] = frozenset({
    "RADAR_SYSTEM", "MISSILE_SYSTEM",
})
```

- [ ] **Step 4: Run unit tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_arcadedb_graph.py -v`
Expected: previous tests pass + new ones pass.

- [ ] **Step 5: Smoke test against live ArcadeDB.**

Run:

```bash
docker compose exec -T api python -c "
import asyncio
from app.db.session import get_graph_store

async def go():
    g = get_graph_store()
    rows = await g._client.query(g._database, 'sql', 'SELECT @rid FROM MISSILE_SYSTEM LIMIT 1')
    if rows:
        rid = rows[0]['@rid']
        ent = await g.get_entity_by_rid(rid)
        print('entity:', ent.get('system_name'))
        rel = await g.get_associated_systems(rid)
        print(f'related: {len(rel)} systems')
        for r in rel[:3]:
            print(f'  {r.entity_type} / {r.name} via {r.relationship_types}')

asyncio.run(go())
"
```

Expected: prints the system name and a non-empty related list (Fan Song or similar).

- [ ] **Step 6: Commit.**

```bash
git add app/services/arcadedb_graph.py tests/unit/test_arcadedb_graph.py
git commit -m "feat(graph): get_entity_by_rid + get_associated_systems (spec §4.6)

Two new ArcadeDBGraphStore methods used by section_properties profiles:
* get_entity_by_rid feeds _project_field_groups with the resolved root's
  full property dict.
* get_associated_systems backs System Components' related_systems block
  via ASSOCIATED_WITH + CUES bothE traversal. Resolves @type up front
  to dodge ArcadeDB's MATCH-without-type UnsupportedOperationException.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 18: Refactor `_fetch_section_items` for `section_properties` branch

**Files:**
- Modify: `app/services/query_profiles.py:_fetch_section_items` (around line 572 in the current file)
- Test: `tests/unit/test_query_profiles.py`

- [ ] **Step 1: Tests for the new branching.**

Append:

```python
@pytest.mark.asyncio
async def test_fetch_section_items_property_branch():
    """When profile.kind == section_properties, _fetch_section_items
    returns a list[QueryProfileFieldGroup], not list[GraphEntityResult]."""
    from unittest.mock import AsyncMock
    from app.schemas.graph_store import GraphEntityResult
    from app.schemas.query_profiles import QueryProfileDefinition, QueryProfileSearchRequest
    from app.services.query_profiles import _fetch_section_items

    profile = QueryProfileDefinition(
        id="test_rf", label="Test", kind="section_properties",
        root_entity_types=["RADAR_SYSTEM"],
        profile_sections=["rf_parameters"],
    )
    resolved = GraphEntityResult(
        node_id="#37:0", name="Fan Song", entity_type="RADAR_SYSTEM",
    )
    request = QueryProfileSearchRequest(
        profile_id="test_rf", query_text="Fan Song", top_k=10,
    )
    graph_store = AsyncMock()
    graph_store.get_entity_by_rid = AsyncMock(return_value={
        "system_name": "Fan Song", "gain_dbi": 35.0, "tx_peak_power_kw": 600.0,
    })

    groups = await _fetch_section_items(graph_store, resolved, request, profile)
    assert len(groups) >= 1
    assert any(f.name == "gain_dbi" for g in groups for f in g.fields)
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Refactor `_fetch_section_items`.**

Open `app/services/query_profiles.py` at the function. Add a branch at the top:

```python
async def _fetch_section_items(graph_store, resolved, request, profile):
    """Returns:
      - list[QueryProfileFieldGroup] when profile.kind == "section_properties"
      - list[GraphEntityResult]      when profile.kind == "section" (legacy)
    """
    if profile.kind == "section_properties":
        if not resolved.node_id:
            return []
        instance_data = await graph_store.get_entity_by_rid(resolved.node_id)
        canonical = _canonical_class_for(resolved.entity_type)
        groups: list = []
        for section in profile.profile_sections:
            groups.extend(_project_field_groups(canonical, instance_data, section))
        return groups
    # ... existing traversal branch unchanged below ...
```

The existing traversal-based code stays as the legacy `kind == "section"` path.

- [ ] **Step 4: Run tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profiles.py -v`
Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add app/services/query_profiles.py tests/unit/test_query_profiles.py
git commit -m "feat(query_profiles): _fetch_section_items section_properties branch (spec §4.4)

When profile.kind=='section_properties', resolves the root's full
property dict via get_entity_by_rid and projects via
_project_field_groups for each profile_section. Legacy traversal
branch (kind=='section') unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 19: Wire `field_groups` + `related_systems` through `execute_section_search`

**Files:**
- Modify: `app/services/query_profiles.py:execute_section_search`
- Test: `tests/unit/test_query_profiles.py`

- [ ] **Step 1: Test for response shape.**

Append:

```python
@pytest.mark.asyncio
async def test_execute_section_search_packages_field_groups():
    """For section_properties profile, the response has populated
    field_groups, empty items, and (for Components) populated
    related_systems."""
    from unittest.mock import AsyncMock
    # ... test setup similar to Task 18, plus assert on response shape
    # carries field_groups, items=[], total>0
    pass
```

(Keep the test minimal — main goal here is the wire-up; deep coverage is in Task 20's integration test.)

- [ ] **Step 2: Update `execute_section_search`.**

Open the function. Where it currently builds `QueryProfileSectionResponse`, branch on `profile.kind`:

```python
async def execute_section_search(graph_store, db, request) -> QueryProfileSectionResponse:
    profile = _resolve_profile(request.profile_id)
    resolved = await _resolve_root(graph_store, profile, request)
    if resolved is None:
        raise QueryRootNotFoundError(...)

    raw = await _fetch_section_items(graph_store, resolved, request, profile)

    if profile.kind == "section_properties":
        field_groups = raw   # list[QueryProfileFieldGroup]
        related_systems: list[GraphEntityResult] = []
        if profile.include_associated_systems and resolved.node_id:
            related_systems = await graph_store.get_associated_systems(resolved.node_id)
        total = sum(len(g.fields) for g in field_groups) + len(related_systems)
        return QueryProfileSectionResponse(
            registry_id=request.registry_id,
            profile_id=profile.id,
            profile_label=profile.label,
            resolved_root=resolved,
            field_groups=field_groups,
            related_systems=related_systems,
            items=[],
            total=total,
        )

    # legacy section branch — unchanged
    items = raw   # list[GraphEntityResult]
    return QueryProfileSectionResponse(
        registry_id=request.registry_id,
        profile_id=profile.id,
        profile_label=profile.label,
        resolved_root=resolved,
        field_groups=[],
        related_systems=[],
        items=items,
        total=len(items),
    )
```

- [ ] **Step 3: Run unit suite for this module.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profiles.py -v`
Expected: all pass.

- [ ] **Step 4: Commit.**

```bash
git add app/services/query_profiles.py tests/unit/test_query_profiles.py
git commit -m "feat(query_profiles): execute_section_search packages field_groups + related_systems (spec §4.4)

Branches the response builder on profile.kind. section_properties
populates field_groups + related_systems; legacy section populates
items unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 20: Refactor `execute_dossier_search`

**Files:**
- Modify: `app/services/query_profiles.py:execute_dossier_search`
- Test: `tests/unit/test_query_profiles.py`

- [ ] **Step 1: Add a test asserting the new dossier shape.**

Append:

```python
@pytest.mark.asyncio
async def test_execute_dossier_search_single_root_with_section_blocks():
    """Dossier returns one resolved_root, populated aliases (back-compat),
    and a list of QueryProfileDossierSection blocks each carrying
    field_groups (for section_properties) or items (for section)."""
    # Mock graph_store, mock _resolve_profile to return a dossier profile
    # with section_profile_ids = ["system_rf_parameters", "system_components",
    # "system_performance"]; _resolve_root returns a fixed entity; assert
    # response has 3 sections, each with kind set, field_groups populated.
    pass
```

(Same approach as Task 19's test stub — keep unit test light, push deep coverage into the integration test in Task 26.)

- [ ] **Step 2: Update `execute_dossier_search`.**

```python
async def execute_dossier_search(graph_store, db, request) -> QueryProfileDossierResponse:
    profile = _resolve_profile(request.profile_id)
    if profile.kind != "dossier":
        raise ValueError(f"Profile {profile.id!r} is not kind='dossier'")
    resolved = await _resolve_root(graph_store, profile, request)
    if resolved is None:
        raise QueryRootNotFoundError(...)

    sections: list[QueryProfileDossierSection] = []
    for section_id in profile.section_profile_ids:
        section_profile = _resolve_profile(section_id)
        # Reuse execute_section_search but with the already-resolved root
        # via an internal _override_resolved kwarg (added below).
        sub_request = QueryProfileSearchRequest(
            profile_id=section_id,
            query_text=request.query_text,
            top_k=request.top_k,
        )
        section_resp = await execute_section_search(
            graph_store, db, sub_request, _override_resolved=resolved,
        )
        sections.append(QueryProfileDossierSection(
            profile_id=section_id,
            profile_label=section_profile.label,
            kind=section_profile.kind,
            field_groups=section_resp.field_groups,
            related_systems=section_resp.related_systems,
            items=section_resp.items,
        ))

    total = sum(len(s.field_groups[0].fields) if s.field_groups else 0 for s in sections)
    total += sum(len(s.items) for s in sections)
    return QueryProfileDossierResponse(
        registry_id=request.registry_id,
        profile_id=profile.id,
        profile_label=profile.label,
        resolved_root=resolved,
        aliases=resolved.aliases or [],
        sections=sections,
        total=total,
    )
```

Update `execute_section_search` to accept an optional `_override_resolved` parameter (skips `_resolve_root` if provided). Internal name with leading underscore signals not-for-public-use.

- [ ] **Step 3: Run tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profiles.py -v`
Expected: all pass.

- [ ] **Step 4: Commit.**

```bash
git add app/services/query_profiles.py tests/unit/test_query_profiles.py
git commit -m "feat(query_profiles): execute_dossier_search single-root + per-section blocks (spec §4.7)

Resolves the root once, then composes per-section blocks via
execute_section_search with an _override_resolved parameter to avoid
re-resolving. Aliases preserved on the dossier response for back-compat
(spec rev-4 #M4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 21: Update starter profile definitions in `build_default_registry_template`

**Files:**
- Modify: `app/services/query_profiles.py:build_default_registry_template` (around line 157)
- Test: `tests/unit/test_query_profiles.py`

- [ ] **Step 1: Add a test that the starter profiles are well-formed.**

Append:

```python
def test_starter_profiles_use_section_properties():
    from app.services.query_profiles import build_default_registry_template

    template = build_default_registry_template()
    profiles = {p.id: p for p in template.profiles}

    for sid in ("system_rf_parameters", "system_components", "system_performance"):
        assert profiles[sid].kind == "section_properties", (
            f"Expected {sid} to be kind=section_properties (post-refactor)"
        )

    components = profiles["system_components"]
    assert components.include_associated_systems is True

    rf = profiles["system_rf_parameters"]
    assert rf.profile_sections == ["rf_parameters"]

    dossier = profiles["system_dossier"]
    assert dossier.kind == "dossier"
    assert dossier.section_profile_ids == [
        "system_rf_parameters", "system_components", "system_performance",
    ]
```

- [ ] **Step 2: Update the four starter profile definitions per spec §4.8.**

Replace each `QueryProfileDefinition(id="system_*", ...)` block with the new shape:

```python
QueryProfileDefinition(
    id="system_rf_parameters",
    label="System RF Parameters",
    description="Frequency, antenna, scan, modulation, and other RF descriptors of the resolved system.",
    kind="section_properties",
    root_entity_types=["RADAR_SYSTEM", "MISSILE_SYSTEM"],
    profile_sections=["rf_parameters"],
    placeholder_query="e.g. Fan Song",
),
QueryProfileDefinition(
    id="system_components",
    label="System Components",
    description="Antenna, propulsion, seeker, ejector, body, and other physical components of the resolved system.",
    kind="section_properties",
    root_entity_types=["RADAR_SYSTEM", "MISSILE_SYSTEM"],
    profile_sections=["components"],
    include_associated_systems=True,
    placeholder_query="e.g. SA-2",
),
QueryProfileDefinition(
    id="system_performance",
    label="System Performance",
    description="Engagement envelope, kinematics, transmit power, and propulsion timing for the resolved system.",
    kind="section_properties",
    root_entity_types=["RADAR_SYSTEM", "MISSILE_SYSTEM"],
    profile_sections=["performance"],
    placeholder_query="e.g. SA-2",
),
QueryProfileDefinition(
    id="system_dossier",
    label="System Dossier",
    description="Composite report of RF parameters, components, and performance for the resolved system.",
    kind="dossier",
    root_entity_types=["RADAR_SYSTEM", "MISSILE_SYSTEM"],
    section_profile_ids=["system_rf_parameters", "system_components", "system_performance"],
    placeholder_query="e.g. SA-2",
),
```

- [ ] **Step 3: Run tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profiles.py::test_starter_profiles_use_section_properties -v`
Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add app/services/query_profiles.py tests/unit/test_query_profiles.py
git commit -m "feat(query_profiles): four starter profiles use kind=section_properties (spec §4.8)

system_rf_parameters / system_components / system_performance use the
new property-projection path with the right profile_sections. Components
sets include_associated_systems=True. system_dossier composes the
three.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 22: Active-registry alembic migration

**Files:**
- Create: `alembic/versions/0018_starter_profiles_to_section_properties.py`
- Test: `tests/unit/test_alembic_migrations.py` (existing or new)

- [ ] **Step 1: Find the next alembic revision number.**

Run: `ls alembic/versions/ | grep -E '^00[0-9]+_' | sort | tail`
Note the highest existing number. The new revision goes one above (likely `0018_`).

- [ ] **Step 2: Create the migration.**

```bash
cd /home/josh/development/EIP-MMDPP
.venv/bin/alembic revision -m "starter_profiles_to_section_properties"
```

This creates a stub file. Open it.

- [ ] **Step 3: Implement `upgrade()` and `downgrade()`.**

```python
"""starter_profiles_to_section_properties

Phase 2 of the flat-schema profile refactor. Updates the four well-known
starter profile rows in any active registry's JSON column to the new
kind="section_properties" / kind="dossier" shapes.

Structurally reversible: down() writes back the prior traversal-based
shape from this file. NOT behaviorally compatible by itself — Phase 1
deleted the ontology types those traversals relied on, so a profile-only
rollback returns empty results until Phase 1 is also reverted.

Revision ID: 0018
Revises: 0017
Create Date: 2026-04-26
"""
from alembic import op
import sqlalchemy as sa
import json

# revision identifiers
revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


# Source of truth for the new shape — kept here for both up() and the
# rollback's reference list. Mirrors build_default_registry_template().
NEW_PROFILES = {
    "system_rf_parameters": {
        "id": "system_rf_parameters",
        "label": "System RF Parameters",
        "description": "Frequency, antenna, scan, modulation, and other RF descriptors of the resolved system.",
        "kind": "section_properties",
        "exposed": True,
        "root_entity_types": ["RADAR_SYSTEM", "MISSILE_SYSTEM"],
        "target_entity_types": [],
        "traversals": [],
        "profile_sections": ["rf_parameters"],
        "include_associated_systems": False,
        "section_profile_ids": [],
        "placeholder_query": "e.g. Fan Song",
    },
    "system_components": {
        "id": "system_components",
        "label": "System Components",
        "description": "Antenna, propulsion, seeker, ejector, body, and other physical components of the resolved system.",
        "kind": "section_properties",
        "exposed": True,
        "root_entity_types": ["RADAR_SYSTEM", "MISSILE_SYSTEM"],
        "target_entity_types": [],
        "traversals": [],
        "profile_sections": ["components"],
        "include_associated_systems": True,
        "section_profile_ids": [],
        "placeholder_query": "e.g. SA-2",
    },
    "system_performance": {
        "id": "system_performance",
        "label": "System Performance",
        "description": "Engagement envelope, kinematics, transmit power, and propulsion timing for the resolved system.",
        "kind": "section_properties",
        "exposed": True,
        "root_entity_types": ["RADAR_SYSTEM", "MISSILE_SYSTEM"],
        "target_entity_types": [],
        "traversals": [],
        "profile_sections": ["performance"],
        "include_associated_systems": False,
        "section_profile_ids": [],
        "placeholder_query": "e.g. SA-2",
    },
    "system_dossier": {
        "id": "system_dossier",
        "label": "System Dossier",
        "description": "Composite report of RF parameters, components, and performance for the resolved system.",
        "kind": "dossier",
        "exposed": True,
        "root_entity_types": ["RADAR_SYSTEM", "MISSILE_SYSTEM"],
        "target_entity_types": [],
        "traversals": [],
        "profile_sections": [],
        "include_associated_systems": False,
        "section_profile_ids": ["system_rf_parameters", "system_components", "system_performance"],
        "placeholder_query": "e.g. SA-2",
    },
}

# Old shapes — preserved here so downgrade() restores parseable JSON.
# Behaviorally inert against post-Phase-1 ontology; documented above.
OLD_PROFILES = {
    # ... copy each starter profile's PRE-refactor JSON here.
    # Leaving this empty in the plan; the implementer fills from the
    # previous build_default_registry_template() definition (visible in
    # git history at HEAD~N before Task 21).
}


def upgrade():
    """For every governance.query_profile_registries row with a
    profiles_json column containing one of the four starter IDs,
    overwrite that profile's entry with the NEW_PROFILES shape."""
    conn = op.get_bind()
    rows = conn.execute(sa.text(
        "SELECT id, profiles_json FROM governance.query_profile_registries"
    )).fetchall()

    for row_id, profiles_json in rows:
        if profiles_json is None:
            continue
        profiles = profiles_json if isinstance(profiles_json, list) else json.loads(profiles_json)
        changed = False
        for i, p in enumerate(profiles):
            if isinstance(p, dict) and p.get("id") in NEW_PROFILES:
                profiles[i] = NEW_PROFILES[p["id"]]
                changed = True
        if changed:
            conn.execute(
                sa.text(
                    "UPDATE governance.query_profile_registries "
                    "SET profiles_json = :p WHERE id = :id"
                ),
                {"p": json.dumps(profiles), "id": row_id},
            )


def downgrade():
    """Structurally reverse the upgrade by writing back OLD_PROFILES.
    NOT behaviorally compatible — see migration docstring."""
    if not OLD_PROFILES:
        # Implementer didn't populate OLD_PROFILES from git history.
        # No-op rollback rather than corrupting registry data.
        return
    conn = op.get_bind()
    rows = conn.execute(sa.text(
        "SELECT id, profiles_json FROM governance.query_profile_registries"
    )).fetchall()
    for row_id, profiles_json in rows:
        if profiles_json is None:
            continue
        profiles = profiles_json if isinstance(profiles_json, list) else json.loads(profiles_json)
        changed = False
        for i, p in enumerate(profiles):
            if isinstance(p, dict) and p.get("id") in OLD_PROFILES:
                profiles[i] = OLD_PROFILES[p["id"]]
                changed = True
        if changed:
            conn.execute(
                sa.text(
                    "UPDATE governance.query_profile_registries "
                    "SET profiles_json = :p WHERE id = :id"
                ),
                {"p": json.dumps(profiles), "id": row_id},
            )
```

**Note for the implementer:** populate `OLD_PROFILES` from the pre-Task-21 starter-profile definitions (visible in git history). The exact column name (`profiles_json`) and table name (`governance.query_profile_registries`) need to be verified against the migration `0007` that originally added the registry — confirm before writing the migration body.

- [ ] **Step 4: Run the migration in dev.**

```bash
docker compose exec -T api alembic upgrade head
```

Expected: clean run, no errors.

- [ ] **Step 5: Verify the active registry got updated.**

```bash
docker compose exec -T postgres psql -U eip -d eip -c "SELECT jsonb_array_elements(profiles_json::jsonb) ->> 'kind' AS kind, jsonb_array_elements(profiles_json::jsonb) ->> 'id' AS id FROM governance.query_profile_registries WHERE active_at IS NOT NULL ORDER BY id;"
```

Expected: rows for the four starter IDs show `kind=section_properties` (or `dossier` for system_dossier).

- [ ] **Step 6: Commit.**

```bash
git add alembic/versions/0018_starter_profiles_to_section_properties.py
git commit -m "feat(migration): 0018 — starter profiles → section_properties (spec §4.10)

Updates the four well-known starter profiles in every existing active
registry to the new kind=section_properties / kind=dossier shapes.
Structurally reversible; behaviorally requires Phase 1 rollback too
(documented in migration docstring).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 23: Delete `dossier_service.py` + 4 legacy `/graph/system-*` endpoints

**Files:**
- Delete: `app/services/dossier_service.py`
- Delete: `tests/unit/test_dossier_service.py`
- Modify: `app/api/v1/graph_store.py` (remove routes 177-216)

- [ ] **Step 1: Audit for leftover imports.**

Run:

```bash
grep -rn "from app.services.dossier_service\|/graph/system-dossier\|/graph/system-components\|/graph/system-rf-parameters\|/graph/system-performance\|SystemQueryRequest\|SystemSectionResponse\|SystemDossierResponse\|_system_section\|build_system_dossier" \
  app/ tests/ frontend/src/ 2>/dev/null | head -30
```

Expected: only matches inside `dossier_service.py`, `graph_store.py:177-216`, `tests/unit/test_dossier_service.py`, and possibly `query_profiles.py` (one helper import). No frontend hits.

- [ ] **Step 2: Inline the one helper `query_profiles.py` borrowed from `dossier_service.py`.**

Find the import at `app/services/query_profiles.py:359` (or wherever). Copy the helper body into `query_profiles.py` and delete the import. Run the unit suite to confirm nothing else depends on it.

- [ ] **Step 3: Delete `app/services/dossier_service.py`.**

```bash
git rm app/services/dossier_service.py
```

- [ ] **Step 4: Delete `tests/unit/test_dossier_service.py`.**

```bash
git rm tests/unit/test_dossier_service.py
```

- [ ] **Step 5: Remove the four routes + `_system_section` from `graph_store.py`.**

Open `app/api/v1/graph_store.py`. Delete the route handlers at lines 177, 204, 209, 214 and the `_system_section` helper at line 195. Also remove any imports of `SystemQueryRequest`, `SystemSectionResponse`, `SystemDossierResponse`, and `build_system_dossier`.

- [ ] **Step 6: Confirm those response/request schemas have no other callers.**

Run:

```bash
grep -rn "SystemQueryRequest\|SystemSectionResponse\|SystemDossierResponse" app/ tests/ 2>/dev/null
```

If empty, delete the schema classes from `app/schemas/graph_store.py`.

- [ ] **Step 7: Run the cleanup-gate audit (Phase 2 success criterion).**

```bash
grep -rn "from app.services.dossier_service\|/graph/system-dossier\|/graph/system-components\|/graph/system-rf-parameters\|/graph/system-performance\|SystemQueryRequest\|SystemSectionResponse\|SystemDossierResponse\|_system_section\|build_system_dossier" \
  app/ tests/ 2>/dev/null
```

Expected: zero hits.

- [ ] **Step 8: Run unit + pipeline test suites.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline -p no:cacheprovider --ignore=tests/unit/test_extraction_schemas.py --ignore=tests/unit/test_specification_entity_validation.py 2>&1 | tail -3`
Expected: all passing.

- [ ] **Step 9: Commit.**

```bash
git add -A
git commit -m "feat(api): remove 4 legacy /graph/system-* endpoints + dossier_service.py (spec §4.13)

Phase 2 packaged breaking change: deletes /graph/system-dossier,
/graph/system-components, /graph/system-rf-parameters,
/graph/system-performance, the _system_section helper, and the
dossier_service.py module + its test counterpart. New
/v1/query-profiles/search/{section,dossier} are the replacements.
Frontend has zero references to the deleted routes.

CHANGELOG migration table will live in spec §4.13 for external API
consumers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 24: Frontend — TypeScript type extensions

**Files:**
- Modify: `frontend/src/api/client.ts` (around line 520 — `QueryProfileDefinition.kind`)

- [ ] **Step 1: Extend `kind` literal.**

```typescript
export interface QueryProfileDefinition {
  id: string;
  label: string;
  description?: string | null;
  kind: "section" | "section_properties" | "dossier";
  exposed: boolean;
  root_entity_types: string[];
  target_entity_types: string[];
  traversals: QueryProfileTraversal[];
  profile_sections: string[];
  include_associated_systems: boolean;
  section_profile_ids: string[];
  placeholder_query: string | null;
}
```

- [ ] **Step 2: Add the new field shapes.**

```typescript
export interface QueryProfileFieldEntry {
  name: string;
  label: string;
  value: unknown;
  description?: string | null;
  examples?: unknown[] | null;
  enum?: string[] | null;
  evidence: QueryProfileFieldEvidence[];
}

export interface QueryProfileFieldGroup {
  subgroup?: string | null;
  subgroup_label?: string | null;
  fields: QueryProfileFieldEntry[];
}

export interface QueryProfileFieldEvidence {
  // Phase 3 expands this. Phase 2 keeps it minimal so empty arrays
  // type-check.
  supporting_snippet: string;
  element_uid?: string | null;
}

export interface QueryProfileDossierSection {
  profile_id: string;
  profile_label: string;
  kind: "section" | "section_properties";
  field_groups: QueryProfileFieldGroup[];
  related_systems: GraphEntityResult[];
  items: GraphEntityResult[];
}
```

- [ ] **Step 3: Update `QueryProfileSectionResponse` and `QueryProfileDossierResponse`.**

```typescript
export interface QueryProfileSectionResponse {
  registry_id: string | null;
  profile_id: string;
  profile_label: string;
  resolved_root: GraphEntityResult;
  field_groups: QueryProfileFieldGroup[];
  related_systems: GraphEntityResult[];
  items: GraphEntityResult[];
  total: number;
}

export interface QueryProfileDossierResponse {
  registry_id: string | null;
  profile_id: string;
  profile_label: string;
  resolved_root: GraphEntityResult;
  aliases: string[];
  sections: QueryProfileDossierSection[];
  total: number;
}
```

- [ ] **Step 4: Run tsc.**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean (or whatever pre-existing failures already exist; no new ones).

- [ ] **Step 5: Commit.**

```bash
git add frontend/src/api/client.ts
git commit -m "feat(frontend): TS types for section_properties + field_groups (spec §6.1)

Extends QueryProfileDefinition.kind to include section_properties.
Adds QueryProfileFieldEntry, QueryProfileFieldGroup,
QueryProfileFieldEvidence, QueryProfileDossierSection. Updates
section/dossier response shapes with the new fields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 25: Frontend — `<FieldGroupTable>` and `<DossierSectionList>` components, `<QueryPage>` switch

**Files:**
- Create: `frontend/src/components/FieldGroupTable.tsx`
- Create: `frontend/src/components/DossierSectionList.tsx`
- Modify: `frontend/src/components/QueryPage.tsx` (around line 827, `selectedIsGraphProfile` branch)

- [ ] **Step 1: Build `<FieldGroupTable>`.**

```tsx
import { QueryProfileFieldGroup, QueryProfileFieldEntry } from "../api/client";

interface FieldGroupTableProps {
  groups: QueryProfileFieldGroup[];
}

export function FieldGroupTable({ groups }: FieldGroupTableProps) {
  return (
    <div className="field-group-table">
      {groups.map((g, idx) => (
        <details key={g.subgroup ?? idx} open={idx === 0}>
          <summary>{g.subgroup_label ?? "Other"}</summary>
          <table>
            <tbody>
              {g.fields.map((f) => (
                <tr key={f.name}>
                  <th title={f.description ?? undefined}>{f.label}</th>
                  <td>{formatValue(f.value)}</td>
                  <td className="evidence-cell">
                    {/* Phase 3 wires the chip; Phase 2 leaves empty */}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      ))}
    </div>
  );
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return v.toLocaleString();
  return String(v);
}
```

- [ ] **Step 2: Build `<DossierSectionList>`.**

```tsx
import { QueryProfileDossierSection, GraphEntityResult } from "../api/client";
import { FieldGroupTable } from "./FieldGroupTable";

interface DossierSectionListProps {
  sections: QueryProfileDossierSection[];
}

export function DossierSectionList({ sections }: DossierSectionListProps) {
  return (
    <div className="dossier-section-list">
      {sections.map((s) => (
        <section key={s.profile_id} className="dossier-section">
          <h3>{s.profile_label}</h3>
          {s.kind === "section_properties" ? (
            s.field_groups.length > 0 ? (
              <FieldGroupTable groups={s.field_groups} />
            ) : (
              <p className="empty">No data extracted for this section.</p>
            )
          ) : (
            <p>(legacy section: {s.items.length} items)</p>
          )}
          {s.related_systems.length > 0 && (
            <RelatedSystemsChips systems={s.related_systems} />
          )}
        </section>
      ))}
    </div>
  );
}

function RelatedSystemsChips({ systems }: { systems: GraphEntityResult[] }) {
  return (
    <div className="related-systems">
      <span>Related: </span>
      {systems.map((s) => (
        <span key={s.node_id ?? s.name} className="chip">
          {s.entity_type} / {s.name}
        </span>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Refactor `QueryPage.tsx` switch (~line 827).**

Find the `selectedIsGraphProfile` block. Replace its inner branching:

```tsx
if (selectedIsGraphProfile) {
  if (selected.profileKind === "dossier") {
    const res = await searchQueryProfileDossier({ /* ... */ });
    setDossierResponse(res);
  } else if (selected.profileKind === "section_properties") {
    const res = await searchQueryProfileSection({ /* ... */ });
    setSectionResponse(res);
  } else {
    // legacy section
    const res = await searchQueryProfileSection({ /* ... */ });
    setLegacyItems(res.items);
  }
}
```

Add three render branches:

```tsx
{sectionResponse && (
  <FieldGroupTable groups={sectionResponse.field_groups} />
)}
{dossierResponse && (
  <DossierSectionList sections={dossierResponse.sections} />
)}
{legacyItems && legacyItems.length > 0 && (
  /* existing legacy result-card list */
)}
```

- [ ] **Step 4: tsc + manual smoke test.**

Run:

```bash
cd frontend && npx tsc --noEmit
```

Expected: clean.

Then bring the app up and visit Search Documents → System RF Parameters → search "SA-2". Expected: a property table with antenna / waveform / transmit / scan groups populated for SA-2.

- [ ] **Step 5: Commit.**

```bash
git add frontend/src/components/FieldGroupTable.tsx frontend/src/components/DossierSectionList.tsx frontend/src/components/QueryPage.tsx
git commit -m "feat(frontend): FieldGroupTable + DossierSectionList + QueryPage switch (spec §4.9)

QueryPage branches on profile.kind: section_properties renders
FieldGroupTable, dossier renders DossierSectionList, legacy section
keeps the existing result-card list. FieldGroupTable's evidence column
is inert in Phase 2 — Phase 3 wires the chip to FieldEvidencePopover.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 26: Phase 2 integration test against live ArcadeDB

- [ ] **Step 1: Rebuild the api image.**

```bash
docker compose stop api worker-graph
docker compose build api
docker compose up -d api worker-graph
sleep 18
```

- [ ] **Step 2: Run the alembic migration if not yet applied.**

```bash
docker compose exec -T api alembic current
docker compose exec -T api alembic upgrade head
```

- [ ] **Step 3: Hit each starter profile.**

```bash
for sid in system_rf_parameters system_components system_performance; do
  echo "=== $sid ==="
  curl -sS -X POST "http://localhost:8005/v1/query-profiles/search/section" \
    -H "Content-Type: application/json" \
    -d "{\"profile_id\":\"$sid\",\"query_text\":\"SA-2\",\"top_k\":5}" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('total:', d.get('total'))
print('field_groups:', [(g.get('subgroup'), len(g.get('fields',[]))) for g in d.get('field_groups',[])])
print('related_systems:', len(d.get('related_systems',[])))
"
done
```

Expected: each profile returns non-empty `field_groups` for SA-2. `system_components` additionally shows a non-empty `related_systems` list (Fan Song or similar).

- [ ] **Step 4: Hit the dossier endpoint.**

```bash
curl -sS -X POST "http://localhost:8005/v1/query-profiles/search/dossier" \
  -H "Content-Type: application/json" \
  -d '{"profile_id":"system_dossier","query_text":"SA-2","top_k":5}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('root:', d.get('resolved_root',{}).get('name'))
print('aliases:', d.get('aliases'))
print('sections:', [(s.get('profile_id'), s.get('kind'), len(s.get('field_groups',[]))) for s in d.get('sections',[])])
"
```

Expected: `root: SA-2`, three sections with the correct kinds and populated field_groups.

- [ ] **Step 5: Cleanup-gate audit.**

```bash
grep -rn "from app.services.dossier_service\|/graph/system-dossier\|/graph/system-components\|/graph/system-rf-parameters\|/graph/system-performance\|SystemQueryRequest\|SystemSectionResponse\|SystemDossierResponse\|_system_section\|build_system_dossier" \
  app/ tests/ frontend/src/ 2>/dev/null
```

Expected: zero hits.

- [ ] **Step 6: No commit (verification only).**

---

## Chunk 2 (Phase 2) acceptance gate

- [ ] All 12 Phase 2 tasks committed (Tasks 12-23 plus the alembic migration as Task 22).
- [ ] `pytest tests/unit tests/pipeline -q` shows the unit suite green.
- [ ] `/v1/query-profiles/search/section` for each starter profile returns populated `field_groups` for SA-2 / Fan Song.
- [ ] `/v1/query-profiles/search/dossier` returns `resolved_root` + `aliases` + 3 populated sections.
- [ ] The grep cleanup-gate audit returns 0 hits.
- [ ] Frontend search UI renders the property tables.
- [ ] Alembic migration applied; existing active registries reflect the new starter shape.


---

## Chunk 3: Phase 3 — Field-level evidence

This chunk covers tasks 26-36: extend the docling-graph LLM extraction prompt to emit per-field source snippets, resolve them to chunk RIDs deterministically, persist them on each entity vertex, surface them in the section response, and wire the frontend popover. Phase 3 is the only phase that requires re-ingest to populate evidence — old data gracefully degrades to empty `evidence: []`.

### Task 26: `FieldProvenanceRow` shared module

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/_field_provenance.py`

- [ ] **Step 1: Create the shared model.**

```python
"""Shared FieldProvenanceRow for per-field source snippets emitted by
the LLM. Lives in extraction_schemas because both RadarDomainPass and
MissileDomainPass use it (spec §5.1.1).

Phase 3 — flat-schema profile refactor.
"""
from pydantic import BaseModel, Field


class FieldProvenanceRow(BaseModel):
    """One per (entity_index, field_name) pair the LLM annotated.

    The service post-process (docling-graph) resolves
    supporting_snippet → element_uid by substring-matching against the
    chunks fed to the LLM. A row whose snippet doesn't match any chunk
    keeps element_uid=None and emits an `unverified_source` log row;
    the snippet still ships to the UI with an "Unverified source"
    badge (spec §5.13)."""
    entity_index: int = Field(
        ...,
        description="0-based index into the pass-template's primary entity list "
                    "(e.g. RadarDomainPass.radar_systems).",
    )
    field_name: str = Field(
        ...,
        description="Canonical field name on the entity model (e.g. 'gain_dbi').",
    )
    supporting_snippet: str = Field(
        ...,
        description="Verbatim quote from the input chunks that established the field's value.",
    )
```

- [ ] **Step 2: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/_field_provenance.py
git commit -m "feat(extraction): FieldProvenanceRow shared model (spec §5.1.1)

Phase 3 task 26. Lives in extraction_schemas because both
RadarDomainPass and MissileDomainPass declare a list[FieldProvenanceRow]
field per spec §5.1.1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 27: Add `field_provenance` field to pass-template classes

**Files:**
- Modify: `ontology_bundles/air_defense_v3/extraction_schemas/radar_domain.py:RadarDomainPass`
- Modify: `ontology_bundles/air_defense_v3/extraction_schemas/missile_domain.py:MissileDomainPass`

- [ ] **Step 1: Failing import test.**

Create `tests/unit/test_field_provenance_field_on_passes.py`:

```python
def test_radar_domain_pass_has_field_provenance():
    from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import RadarDomainPass
    assert "field_provenance" in RadarDomainPass.model_fields
    finfo = RadarDomainPass.model_fields["field_provenance"]
    # Default is empty list
    inst = RadarDomainPass()
    assert inst.field_provenance == []


def test_missile_domain_pass_has_field_provenance():
    from ontology_bundles.air_defense_v3.extraction_schemas.missile_domain import MissileDomainPass
    inst = MissileDomainPass()
    assert inst.field_provenance == []
```

- [ ] **Step 2: Add the field on each pass class.**

In `radar_domain.py:RadarDomainPass`:

```python
from ontology_bundles.air_defense_v3.extraction_schemas._field_provenance import (
    FieldProvenanceRow,
)

class RadarDomainPass(BaseModel):
    # ... existing fields ...
    radar_systems: List[RadarSystemEntity] = edge(...)
    field_provenance: list[FieldProvenanceRow] = Field(
        default_factory=list,
        description=(
            "Per-field source snippets the LLM quoted to justify each "
            "field value. The docling-graph service post-processes these "
            "to resolve element_uid by substring-matching the snippet "
            "against the input chunks. Empty by default; populated only "
            "when the prompt asks for it (Phase 3 of the flat-schema "
            "profile refactor)."
        ),
    )
```

Mirror in `missile_domain.py:MissileDomainPass`.

- [ ] **Step 3: Run tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_field_provenance_field_on_passes.py -v`
Expected: 2 passed.

- [ ] **Step 4: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/ tests/unit/test_field_provenance_field_on_passes.py
git commit -m "feat(extraction): field_provenance list on RadarDomainPass / MissileDomainPass (spec §5.1.1)

Phase 3 task 27. Top-level list, parallel to the pass's primary entity
list (radar_systems / missile_systems). Survives extra='ignore' on the
nested entity classes because it lives one level above.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 28: docling-graph wire schema + `_primary_list_field_name` helper

**Files:**
- Modify: `docker/docling-graph/app/schemas.py` (add `ExtractionFieldProvenance`, extend `ExtractPassResponse`)
- Create: `docker/docling-graph/app/_field_provenance_helpers.py` (helper for resolving primary list field)
- Test: `tests/integration/docling_graph/test_field_provenance_wire.py`

- [ ] **Step 1: Add `ExtractionFieldProvenance` to schemas.**

```python
class ExtractionFieldProvenance(BaseModel):
    """Wire-shape per-field provenance row (spec §5.3).

    Built by the service post-process from the pass-template's
    field_provenance list. Joins entity_index → instance_id by
    indexing into the pass's primary entity list (resolved via
    _primary_list_field_name)."""
    instance_id: str
    field_name: str
    value: Any
    supporting_snippet: str
    element_uid: Optional[str] = None
```

Extend `ExtractPassResponse`:

```python
class ExtractPassResponse(BaseModel):
    # ... existing fields ...
    provenance: list[ExtractionProvenance] = Field(default_factory=list)
    field_provenance: list[ExtractionFieldProvenance] = Field(default_factory=list)
```

- [ ] **Step 2: Add `_primary_list_field_name` helper.**

```python
"""Resolves the pass-template's primary entity list field via manifest
metadata. Used by the post-LLM provenance resolver to index entity_index
into the right list when the pass-template uses a custom field name
like radar_systems or missile_systems (spec §5.1.1)."""
from pydantic import BaseModel
from typing import get_args, get_origin


def _primary_list_field_name(template_cls: type[BaseModel], primary_type: str) -> str:
    """Walk template_cls.model_fields, return the first field whose
    annotation is list[<class with model_config['ontology_name']
    matching primary_type>]."""
    for fname, finfo in template_cls.model_fields.items():
        ann = finfo.annotation
        if get_origin(ann) is list:
            (item_type,) = get_args(ann) or (None,)
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                cfg = item_type.model_config or {}
                if cfg.get("ontology_name") == primary_type:
                    return fname
    raise ValueError(
        f"No primary list field for ontology_name={primary_type!r} on {template_cls.__name__}"
    )
```

- [ ] **Step 3: Wire into the post-process — build `field_provenance` rows from the template's own `field_provenance`.**

Open `docker/docling-graph/app/main.py` near `template_instance.model_dump(mode="json")` (line ~850 in current file). After computing `pass_output`, add:

```python
# Phase 3: per-field provenance.
field_provenance_rows: list[ExtractionFieldProvenance] = []
template_field_provenance = getattr(template_instance, "field_provenance", None) or []
if template_field_provenance:
    primary_type = manifest_pass.primary_entity_types[0] if manifest_pass.primary_entity_types else None
    if primary_type:
        try:
            list_field = _primary_list_field_name(type(template_instance), primary_type)
        except ValueError:
            list_field = None
        if list_field:
            primary_entities = getattr(template_instance, list_field, []) or []
            for fp_row in template_field_provenance:
                if 0 <= fp_row.entity_index < len(primary_entities):
                    entity = primary_entities[fp_row.entity_index]
                    instance_id = _instance_ids[fp_row.entity_index] if _instance_ids else ""
                    value = getattr(entity, fp_row.field_name, None)
                    field_provenance_rows.append(ExtractionFieldProvenance(
                        instance_id=instance_id,
                        field_name=fp_row.field_name,
                        value=value,
                        supporting_snippet=fp_row.supporting_snippet,
                        element_uid=None,  # resolved in Task 30
                    ))

response = ExtractPassResponse(
    # ... existing fields ...
    field_provenance=field_provenance_rows,
)
```

(Naming `_instance_ids` mirrors the existing `provenance` row build path; verify the actual variable name in `main.py` and adjust.)

- [ ] **Step 4: Commit.**

```bash
git add docker/docling-graph/app/schemas.py docker/docling-graph/app/_field_provenance_helpers.py docker/docling-graph/app/main.py
git commit -m "feat(docling-graph): wire shape for per-field provenance (spec §5.3, §5.1.1)

ExtractionFieldProvenance row + ExtractPassResponse.field_provenance.
_primary_list_field_name helper resolves radar_systems /
missile_systems from manifest.primary_entity_types so entity_index
joins correctly. element_uid resolution comes in Task 30.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 29: Extend the extraction prompt

**Files:**
- Modify: `docker/docling-graph/app/prompt_rules.py` (add provenance-block instructions)
- Modify: `docker/docling-graph/app/main.py` (or wherever the system prompt is composed)

- [ ] **Step 1: Add the provenance-block prompt text.**

```python
FIELD_PROVENANCE_PROMPT = """\
After populating the pass's primary entity list, fill `field_provenance`. For every field you populated on an entity for which you can quote a source, emit one `field_provenance` row containing:

- entity_index: the 0-based position of the entity in the pass's primary entity list
- field_name: the canonical field name on that entity (e.g. gain_dbi, max_speed_mps)
- supporting_snippet: an exact verbatim quote from the input text that established the field's value. The snippet must appear verbatim somewhere in the chunks provided. Do not paraphrase or summarize. Whitespace differences are acceptable; word substitution is not.

If you cannot quote a source for a field, simply omit that field's row from `field_provenance` — never invent or paraphrase. An empty `field_provenance` array is acceptable.
"""
```

- [ ] **Step 2: Append it to the system prompt for radar_domain and missile_domain passes.**

In whatever assembly function builds the prompt, add the `FIELD_PROVENANCE_PROMPT` block after the per-pass instructions but before the final structured-output schema description.

- [ ] **Step 3: Manual smoke test — run a single extraction.**

Rebuild docling-graph and re-extract one chunk. Inspect the response's `field_provenance` array — should have rows for fields the LLM populated.

```bash
docker compose build docling-graph
docker compose up -d docling-graph
# Trigger re-extraction (manual via celery or by re-uploading a doc)
```

- [ ] **Step 4: Commit.**

```bash
git add docker/docling-graph/app/prompt_rules.py docker/docling-graph/app/main.py
git commit -m "feat(docling-graph): extraction prompt asks for per-field source snippets (spec §5.4)

Adds FIELD_PROVENANCE_PROMPT block instructing the LLM to emit
verbatim quotes per (entity_index, field_name) pair into the new
field_provenance list.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 30: Snippet → `element_uid` resolver

**Files:**
- Modify: `docker/docling-graph/app/provenance.py` (extend post-LLM resolver)

- [ ] **Step 1: Add `resolve_field_provenance_uids`.**

```python
import re
import logging

_log = logging.getLogger(__name__)
_WS_NORM = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_NORM.sub(" ", text).strip().casefold()


def resolve_field_provenance_uids(
    field_provenance: list[ExtractionFieldProvenance],
    input_chunks: list,   # list of (element_uid, text) tuples or similar — match existing input_chunks shape
) -> None:
    """For each row, set element_uid by whitespace-collapsed,
    case-insensitive substring match against input_chunks.
    Tiebreaker: longest unique-prefix match. Ambiguous-after-tiebreaker
    falls back to first match by stable order and emits an
    `ambiguous_snippet` log row carrying all candidate uids
    (spec §5.13). Mutates rows in place; rows with no match keep
    element_uid=None and an `unverified_source` log row is emitted."""
    for row in field_provenance:
        candidates: list[tuple[str, str]] = []   # (element_uid, text)
        snippet_norm = _normalize(row.supporting_snippet)
        if not snippet_norm:
            continue
        for euid, ctext in input_chunks:
            if snippet_norm in _normalize(ctext):
                candidates.append((euid, ctext))
        if not candidates:
            _log.info(
                "unverified_source",
                extra={
                    "instance_id": row.instance_id,
                    "field_name": row.field_name,
                    "snippet": row.supporting_snippet[:100],
                },
            )
            continue   # element_uid stays None
        if len(candidates) == 1:
            row.element_uid = candidates[0][0]
            continue
        # Multiple matches — longest unique-prefix tiebreaker.
        # Each candidate scores by length of unique prefix sequence
        # within snippet_norm that's specific to that chunk.
        # Simplification: pick longest chunk text (proxy for richest
        # context) as tiebreaker; if tied, first by stable order.
        candidates.sort(key=lambda c: -len(c[1]))
        row.element_uid = candidates[0][0]
        _log.info(
            "ambiguous_snippet",
            extra={
                "instance_id": row.instance_id,
                "field_name": row.field_name,
                "candidates": [c[0] for c in candidates],
                "selected": candidates[0][0],
            },
        )
```

- [ ] **Step 2: Call `resolve_field_provenance_uids` after building rows in `main.py`.**

After the Task 28 post-process where `field_provenance_rows` is populated:

```python
input_chunks_for_resolver = [...]  # collect from the same source provenance uses
resolve_field_provenance_uids(field_provenance_rows, input_chunks_for_resolver)
```

- [ ] **Step 3: Unit test.**

Create `tests/unit/test_field_provenance_resolver.py`:

```python
def test_resolver_matches_single_chunk():
    from docker.docling_graph.app.provenance import resolve_field_provenance_uids
    from docker.docling_graph.app.schemas import ExtractionFieldProvenance

    chunks = [("uid-1", "The radar gain is 35 dBi nominal."),
              ("uid-2", "Other unrelated text.")]
    rows = [ExtractionFieldProvenance(
        instance_id="i1", field_name="gain_dbi", value=35.0,
        supporting_snippet="gain is 35 dBi",
    )]
    resolve_field_provenance_uids(rows, chunks)
    assert rows[0].element_uid == "uid-1"


def test_resolver_handles_no_match():
    from docker.docling_graph.app.provenance import resolve_field_provenance_uids
    from docker.docling_graph.app.schemas import ExtractionFieldProvenance

    rows = [ExtractionFieldProvenance(
        instance_id="i1", field_name="x", value=1,
        supporting_snippet="not in any chunk",
    )]
    resolve_field_provenance_uids(rows, [("uid-1", "different text")])
    assert rows[0].element_uid is None


def test_resolver_picks_first_with_log_on_ambiguity():
    # Two chunks contain the snippet. Resolver picks one and logs ambiguous.
    from docker.docling_graph.app.provenance import resolve_field_provenance_uids
    from docker.docling_graph.app.schemas import ExtractionFieldProvenance

    chunks = [("uid-1", "the antenna gain"),
              ("uid-2", "the antenna gain is 35")]
    rows = [ExtractionFieldProvenance(
        instance_id="i1", field_name="gain_dbi", value=35.0,
        supporting_snippet="antenna gain",
    )]
    resolve_field_provenance_uids(rows, chunks)
    # Either uid-1 or uid-2 — schema doesn't promise which, just that
    # one is chosen.
    assert rows[0].element_uid in {"uid-1", "uid-2"}
```

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_field_provenance_resolver.py -v`
Expected: 3 passed.

- [ ] **Step 4: Commit.**

```bash
git add docker/docling-graph/app/provenance.py docker/docling-graph/app/main.py tests/unit/test_field_provenance_resolver.py
git commit -m "feat(docling-graph): snippet → element_uid resolver (spec §5.5, §5.13)

Whitespace-collapsed case-insensitive substring match. Single match
wins; multi-match falls back to longest chunk + ambiguous_snippet log;
no match keeps element_uid=None + unverified_source log.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 31: Worker-side merger update

**Files:**
- Modify: `app/services/extraction_merge.py`

- [ ] **Step 1: Parse `field_provenance` from `ExtractPassResponse` into `MergedEntityRecord`.**

Add to `MergedEntityRecord`:

```python
@dataclass
class FieldEvidenceRow:
    chunk_id: str | None       # None when element_uid couldn't be resolved
    snippet: str
    element_uid: str | None


@dataclass
class MergedEntityRecord:
    # ... existing fields ...
    field_evidence: dict[str, list[FieldEvidenceRow]] = field(default_factory=dict)
```

- [ ] **Step 2: In `_parse_pass_response` (or equivalent), populate `field_evidence`.**

Group `ExtractPassResponse.field_provenance` rows by `instance_id`. For each row, look up the chunk by `element_uid` (the same path that builds `EXTRACTED_FROM`) to get `chunk_id`. Build `FieldEvidenceRow(chunk_id, snippet, element_uid)` and append under `entity.field_evidence[field_name]`.

- [ ] **Step 3: Unit test.**

Create `tests/unit/test_extraction_merge_field_evidence.py`:

```python
def test_field_evidence_grouped_by_instance_and_field():
    from app.services.extraction_merge import _parse_pass_response
    # Mock an ExtractPassResponse with two field_provenance rows for
    # the same entity. Assert MergedEntityRecord.field_evidence has
    # both entries grouped under the right field_name.
    pass   # implementer fills with realistic mock
```

- [ ] **Step 4: Commit.**

```bash
git add app/services/extraction_merge.py tests/unit/test_extraction_merge_field_evidence.py
git commit -m "feat(merge): parse field_provenance into MergedEntityRecord.field_evidence (spec §5.6)

Worker-side counterpart to Task 30. FieldEvidenceRow carries
(chunk_id, snippet, element_uid); chunk_id resolves from element_uid
via the same lookup path EXTRACTED_FROM uses; chunk_id is Optional[str]
because element_uid may be None.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 32: Persist `_field_evidence` on entity vertex

**Files:**
- Modify: `app/services/arcadedb_graph.py:upsert_nodes_batch_sync`

- [ ] **Step 1: Add `_field_evidence` to the upsert payload.**

Inside `upsert_nodes_batch_sync`, where the entity's properties dict is built before the SQL UPSERT, attach `_field_evidence` if the `MergedEntityRecord` carries it:

```python
if entity.field_evidence:
    props["_field_evidence"] = {
        field_name: [
            {"chunk_id": r.chunk_id, "snippet": r.snippet, "element_uid": r.element_uid}
            for r in rows
        ]
        for field_name, rows in entity.field_evidence.items()
    }
```

ArcadeDB serializes the dict as JSON natively.

- [ ] **Step 2: Smoke test — re-ingest one doc and verify the property lands on a vertex.**

```bash
# After re-ingest:
curl -s -u root:eip_arcadedb_secret -X POST http://localhost:2480/api/v1/query/eip_knowledge_graph \
  -H "Content-Type: application/json" \
  -d '{"language":"sql","command":"SELECT _field_evidence FROM MISSILE_SYSTEM WHERE name=\"SA-2\" LIMIT 1"}'
```

Expected: a JSON object with at least one field key.

- [ ] **Step 3: Commit.**

```bash
git add app/services/arcadedb_graph.py
git commit -m "feat(graph): persist _field_evidence JSON on entity vertices (spec §5.7)

upsert_nodes_batch_sync now writes _field_evidence as a JSON property
on RADAR_SYSTEM / MISSILE_SYSTEM vertices when MergedEntityRecord
carries field_evidence. Old data has no _field_evidence; queries that
read it must handle the missing key.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 33: Expand `QueryProfileFieldEvidence` schema

**Files:**
- Modify: `app/schemas/query_profiles.py`

- [ ] **Step 1: Replace the Phase 2 stub with the full Phase 3 shape.**

```python
class QueryProfileFieldEvidence(APIModel):
    """Per-field evidence row (spec §5.8). Combines chunk metadata
    (same surface as GraphEvidenceItem) with field-specific signals
    (supporting_snippet + element_uid)."""
    chunk_id: Optional[uuid.UUID] = None
    chunk_type: Optional[str] = None
    artifact_id: Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None
    document_name: Optional[str] = None
    modality: Optional[str] = None
    page_number: Optional[int] = None
    classification: str = "UNCLASSIFIED"
    content_text: Optional[str] = None
    source_characterization: Optional[str] = None
    date_of_information: Optional[str] = None
    extraction_confidence: Optional[float] = None
    supporting_snippet: str
    element_uid: Optional[str] = None
```

(Replaces the minimal Phase 2 stub. The TS interface counterpart is updated in Task 35.)

- [ ] **Step 2: Update the `QueryProfileFieldEntry.evidence` annotation if needed (already `list[QueryProfileFieldEvidence]` from Task 12).**

- [ ] **Step 3: Run schema tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profile_schemas.py -v`
Expected: all pass.

- [ ] **Step 4: Commit.**

```bash
git add app/schemas/query_profiles.py
git commit -m "feat(schemas): expand QueryProfileFieldEvidence to full Phase 3 shape (spec §5.8)

Adds chunk-metadata fields (chunk_id, document_name, content_text,
classification, etc.) on top of the supporting_snippet + element_uid
the Phase 2 stub had. Mirrors GraphEvidenceItem's fields plus the
field-specific signals.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 34: Surface `_field_evidence` through `_project_field_groups`

**Files:**
- Modify: `app/services/query_profiles.py:_project_field_groups`

- [ ] **Step 1: Read `_field_evidence` from `instance_data` and convert each row.**

In `_project_field_groups`, after constructing the `QueryProfileFieldEntry`:

```python
field_evidence_blob = instance_data.get("_field_evidence") or {}
raw_rows = field_evidence_blob.get(fname) or []
evidence_items: list[QueryProfileFieldEvidence] = []
for r in raw_rows:
    chunk_meta = await _lookup_chunk_meta(r.get("chunk_id")) if r.get("chunk_id") else {}
    evidence_items.append(QueryProfileFieldEvidence(
        chunk_id=r.get("chunk_id"),
        element_uid=r.get("element_uid"),
        supporting_snippet=r.get("snippet") or "",
        **chunk_meta,
    ))
entry.evidence = evidence_items
```

`_lookup_chunk_meta` is the same helper used by retrieval to fetch chunk metadata; reuse it. Note: `_project_field_groups` must become async because of the awaited lookups; update its callers (`_fetch_section_items`) accordingly.

- [ ] **Step 2: Test with mocked `_lookup_chunk_meta`.**

Append to `tests/unit/test_query_profiles.py`:

```python
@pytest.mark.asyncio
async def test_project_field_groups_surfaces_field_evidence(monkeypatch):
    from app.services import query_profiles as qp

    async def fake_lookup_chunk_meta(chunk_id):
        return {"chunk_type": "TextChunk", "document_name": "X.pdf"}

    monkeypatch.setattr(qp, "_lookup_chunk_meta", fake_lookup_chunk_meta)

    instance_data = {
        "gain_dbi": 35.0,
        "_field_evidence": {
            "gain_dbi": [
                {"chunk_id": "00000000-0000-0000-0000-000000000001",
                 "element_uid": "#/texts/12",
                 "snippet": "antenna gain measured at 35 dBi"},
            ]
        },
    }
    from ontology_bundles.air_defense_v3.entities import RadarSystemEntity
    groups = await qp._project_field_groups(RadarSystemEntity, instance_data, "rf_parameters")
    antenna = next(g for g in groups if g.subgroup == "antenna")
    gain = next(f for f in antenna.fields if f.name == "gain_dbi")
    assert len(gain.evidence) == 1
    assert gain.evidence[0].supporting_snippet.startswith("antenna gain")
```

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_query_profiles.py -k field_evidence -v`
Expected: PASS.

- [ ] **Step 3: Commit.**

```bash
git add app/services/query_profiles.py tests/unit/test_query_profiles.py
git commit -m "feat(query_profiles): surface _field_evidence on QueryProfileFieldEntry (spec §5.8)

_project_field_groups becomes async; reads _field_evidence from
instance_data; resolves chunk metadata for each evidence row via the
existing chunk-lookup path. Old data with no _field_evidence yields
evidence=[] gracefully.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 35: Frontend — `<FieldEvidencePopover>` + activate chip

**Files:**
- Create: `frontend/src/components/FieldEvidencePopover.tsx`
- Modify: `frontend/src/components/FieldGroupTable.tsx`
- Modify: `frontend/src/api/client.ts` (extend `QueryProfileFieldEvidence` interface)

- [ ] **Step 1: Extend the TS evidence interface.**

```typescript
export interface QueryProfileFieldEvidence {
  chunk_id?: string | null;
  chunk_type?: string | null;
  artifact_id?: string | null;
  document_id?: string | null;
  document_name?: string | null;
  modality?: string | null;
  page_number?: number | null;
  classification: string;
  content_text?: string | null;
  source_characterization?: string | null;
  date_of_information?: string | null;
  extraction_confidence?: number | null;
  supporting_snippet: string;
  element_uid?: string | null;
}
```

- [ ] **Step 2: Build the popover component.**

```tsx
import { QueryProfileFieldEvidence } from "../api/client";

interface FieldEvidencePopoverProps {
  evidence: QueryProfileFieldEvidence[];
  onClose: () => void;
}

export function FieldEvidencePopover({ evidence, onClose }: FieldEvidencePopoverProps) {
  return (
    <div className="field-evidence-popover" role="dialog" onClick={(e) => e.stopPropagation()}>
      <button className="close" onClick={onClose}>✕</button>
      {evidence.length === 0 ? (
        <p className="empty">No per-field evidence available.</p>
      ) : (
        <ul>
          {evidence.map((e, idx) => (
            <li key={idx} className={e.element_uid ? "" : "unverified"}>
              {!e.element_uid && (
                <span className="badge unverified">Unverified source</span>
              )}
              <blockquote>{e.supporting_snippet}</blockquote>
              <div className="meta">
                {e.document_name && <span>📄 {e.document_name}</span>}
                {e.page_number && <span>p. {e.page_number}</span>}
                {e.element_uid && (
                  <a href={buildDeepLink(e)} target="_blank" rel="noreferrer">
                    Open in document viewer
                  </a>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function buildDeepLink(e: QueryProfileFieldEvidence): string {
  // Document viewer URL pattern — match the existing deep-link path
  // used by retrieval results.
  if (!e.document_id || !e.element_uid) return "#";
  return `/documents/${e.document_id}?element_uid=${encodeURIComponent(e.element_uid)}`;
}
```

- [ ] **Step 3: Wire the chip in `<FieldGroupTable>`.**

Replace the `evidence-cell` placeholder:

```tsx
import { useState } from "react";
import { FieldEvidencePopover } from "./FieldEvidencePopover";

// inside the FieldGroupTable row mapping:
<td className="evidence-cell">
  {f.evidence.length > 0 ? (
    <EvidenceChip evidence={f.evidence} fieldName={f.label} />
  ) : null}
</td>

// new component:
function EvidenceChip({ evidence, fieldName }: {
  evidence: QueryProfileFieldEvidence[];
  fieldName: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        className="evidence-chip"
        onClick={() => setOpen(true)}
        title={`Show ${evidence.length} source(s) for ${fieldName}`}
      >
        📄 {evidence.length}
      </button>
      {open && (
        <FieldEvidencePopover evidence={evidence} onClose={() => setOpen(false)} />
      )}
    </>
  );
}
```

- [ ] **Step 4: tsc + manual smoke test.**

Run: `cd frontend && npx tsc --noEmit`. Expected: clean.

After re-ingest (Task 36), open Search Documents → System RF Parameters → search "SA-2" → click an evidence chip on a populated row. Expected: popover shows the LLM-quoted snippet with a deep link.

- [ ] **Step 5: Commit.**

```bash
git add frontend/src/components/FieldEvidencePopover.tsx frontend/src/components/FieldGroupTable.tsx frontend/src/api/client.ts
git commit -m "feat(frontend): FieldEvidencePopover wires per-field evidence chip (spec §5.9)

Phase 3 task 35. Chip on each row with non-empty evidence opens a
popover listing (supporting_snippet, document_name, page_number,
element_uid deep link). Unverified rows (element_uid=None) get an
'Unverified source' badge.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 36: Phase 3 end-to-end re-ingest test

- [ ] **Step 1: Rebuild docling-graph + api + workers.**

```bash
docker compose stop docling-graph api worker-graph worker-ingest
docker compose build docling-graph api
docker compose up -d docling-graph api worker-graph worker-ingest
sleep 30
```

- [ ] **Step 2: Re-ingest one test document.**

Pick a doc that has SA-2 / Fan Song content. Re-upload via the API or trigger re-extraction via the manage script.

- [ ] **Step 3: Wait for the pipeline to finish.**

Watch the worker logs for `derive_ontology_graph` completion.

- [ ] **Step 4: Verify `_field_evidence` populated.**

```bash
AUTH="root:eip_arcadedb_secret"
curl -s -u "$AUTH" -X POST http://localhost:2480/api/v1/query/eip_knowledge_graph \
  -H "Content-Type: application/json" \
  -d '{"language":"sql","command":"SELECT name, _field_evidence FROM MISSILE_SYSTEM WHERE name=\"SA-2\" LIMIT 1"}' \
  | python3 -m json.tool
```

Expected: `_field_evidence` is a dict with at least 3 field-name keys, each carrying a list of `{chunk_id, snippet, element_uid}` rows.

- [ ] **Step 5: Verify the section endpoint surfaces evidence.**

```bash
curl -sS -X POST "http://localhost:8005/v1/query-profiles/search/section" \
  -H "Content-Type: application/json" \
  -d '{"profile_id":"system_rf_parameters","query_text":"SA-2","top_k":5}' \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
populated = [
    (g['subgroup'], f['name'], len(f.get('evidence',[])))
    for g in d.get('field_groups',[])
    for f in g.get('fields',[])
    if f.get('evidence')
]
print(f'Fields with evidence: {len(populated)}')
for row in populated[:5]:
    print(f'  {row}')
"
```

Expected: at least one row with non-zero evidence count.

- [ ] **Step 6: Verify the unverified-source path.**

Inspect the structured logs for `unverified_source` and `ambiguous_snippet` events emitted during the re-extraction:

```bash
docker compose logs --since 10m docling-graph 2>&1 | grep -E "unverified_source|ambiguous_snippet" | head
```

Both are acceptable — they're informative, not errors.

- [ ] **Step 7: No commit (verification only).**

---

## Chunk 3 (Phase 3) acceptance gate

- [ ] All 11 Phase 3 tasks committed (Tasks 26-35).
- [ ] Re-ingested SA-2 entity has populated `_field_evidence` for ≥3 fields.
- [ ] `/v1/query-profiles/search/section` for `system_rf_parameters` against SA-2 returns at least one populated `evidence` list inside `field_groups[*].fields[*]`.
- [ ] Pre-Phase-3 entities (those not re-ingested) still render correctly with empty `evidence`.
- [ ] Frontend `<FieldEvidencePopover>` opens for chips and shows snippet + deep link.

---

## Final verification

After all three chunks are complete:

- [ ] **F1: Run the full unit + pipeline test sweep.**

```bash
SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline -p no:cacheprovider \
  --ignore=tests/unit/test_extraction_schemas.py \
  --ignore=tests/unit/test_specification_entity_validation.py \
  2>&1 | tail -3
```

Expected: green, 0 xfailed (down from 5 pre-Phase-1).

- [ ] **F2: Run the cleanup-gate audit (Phase 2's Task 23 success criterion).**

```bash
grep -rn "from app.services.dossier_service\|/graph/system-dossier\|/graph/system-components\|/graph/system-rf-parameters\|/graph/system-performance\|SystemQueryRequest\|SystemSectionResponse\|SystemDossierResponse\|_system_section\|build_system_dossier" \
  app/ tests/ frontend/src/ 2>/dev/null
```

Expected: zero hits.

- [ ] **F3: Verify the four starter profiles work end-to-end.**

```bash
for sid in system_rf_parameters system_components system_performance system_dossier; do
  echo "=== $sid ==="
  if [ "$sid" = "system_dossier" ]; then
    endpoint="search/dossier"
  else
    endpoint="search/section"
  fi
  curl -sS -X POST "http://localhost:8005/v1/query-profiles/$endpoint" \
    -H "Content-Type: application/json" \
    -d "{\"profile_id\":\"$sid\",\"query_text\":\"SA-2\",\"top_k\":5}" \
    | python3 -c "import sys, json; d=json.load(sys.stdin); print('total:', d.get('total'), 'shape:', list(d.keys())[:8])"
done
```

Expected: each profile returns non-empty data with the right shape.

- [ ] **F4: Verify the bundle is clean.**

```bash
.venv/bin/python -c "
from ontology_bundles.coverage_checker import check_bundle  # adjust import path
from pathlib import Path
errors, _ = check_bundle(Path('ontology_bundles/air_defense_v3'))
assert not errors, errors
print('bundle clean')
"
```

- [ ] **F5: Push.**

```bash
git push
```

Done.
