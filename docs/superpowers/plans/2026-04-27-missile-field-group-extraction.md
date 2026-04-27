# Missile Field-Group Extraction Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single `missile_domain` extraction pass with six focused sub-passes (`missile_identity`, `missile_kinematics`, `missile_guidance`, `missile_airframe`, `missile_speed_timing`, `missile_propulsion`) so each LLM call sees a smaller schema. Mirrors the radar field-group split pattern exactly.

**Architecture:** Each sub-pass is its own `/extract-pass` call against a 4-9 field schema. All emit `MISSILE_SYSTEM[]` with `system_name` identity; existing `merge_and_resolve` collapses partial records onto one vertex. Reuses `_numeric_evidence.py` shared helper from radar plan (Task 3); refactors `_clear_unsupported_missile_properties` if it exists in `evidence_gate.py` (parallel to radar's §4.8 fix).

**Tech Stack:** Python 3.11/3.12, Pydantic v2, FastAPI, docling-graph LLM extraction service, Ollama gemma4:31b, ArcadeDB, pytest.

**Source pattern:** [`docs/superpowers/plans/2026-04-27-radar-field-group-extraction.md`](2026-04-27-radar-field-group-extraction.md). This plan is intentionally a mechanical mirror — same architecture, same files, same merge contract, only the field partition and forbidden-name set differ.

**Spec context:** [`docs/superpowers/specs/2026-04-27-radar-field-group-extraction-design.md`](../specs/2026-04-27-radar-field-group-extraction-design.md). The same architectural reasoning (smaller schema per LLM call → better numeric extraction) applies. No separate spec written for missile because the structural decisions are identical.

---

## Prerequisite check before starting

**Do NOT start this plan until both are true:**

1. Radar plan (`2026-04-27-radar-field-group-extraction.md`) Tasks 1-21 are complete and committed.
2. Radar smoke harness extracted **≥2/3 numeric values** (per spec §10 fallback gate). If radar smoke is <2/3, mirroring the split for missile would not move numerics either — switch to spec §10 fallback architecture (candidate-mapping) before touching missile.

Verify both before continuing:

```bash
git log --oneline | grep -E "manifest cutover.*radar_domain|smoke harness.*radar" | head -5
```

If the cutover commit is missing, **stop**; complete the radar plan first.

If the smoke harness exists but the most recent run was <2/3, **stop** and re-run it. If still <2/3, switch to spec §10 fallback instead of starting this plan.

---

## Pre-flight checklist

Run these once at the start of the session and before each chunk to confirm baseline:

- [ ] **P0: Read the source pattern.**

Run: `wc -l docs/superpowers/plans/2026-04-27-radar-field-group-extraction.md`
Expected: ≥ 2900 lines. Skim Chunks 1-4 to refresh the architectural pattern before starting.

Use the @superpowers-extended-cc:test-driven-development skill for every code-bearing task.

- [ ] **P1: Confirm baseline test suite status.**

Run:
```bash
SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline -q \
  --ignore=tests/unit/test_extraction_schemas.py \
  --ignore=tests/unit/test_specification_entity_validation.py 2>&1 \
  | grep -E "passed|failed" | tail -3
```
Expected: ≥1240 passed, with at most the 3 known xfails from prior work plus any radar-cutover-introduced numbers from the radar plan baseline. Document any failure as a pre-existing issue not caused by this plan.

- [ ] **P2: Confirm stack is up.**

Run: `docker compose ps --format "table {{.Service}}\t{{.Status}}" | grep -E "api|arcadedb|postgres|worker-graph|docling-graph"`
Expected: all five services Up. If not, `./manage.sh --start` and wait 30 s.

- [ ] **P3: Confirm field counts on extraction-side MissileSystemEntity.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.missile_domain import MissileSystemEntity
fields = list(MissileSystemEntity.model_fields.keys())
print(f'extraction-side MissileSystemEntity fields: {len(fields)}')
"
```
Expected: ≥38 fields. The plan's `MISSILE_FIELD_GROUPS` partition assumes 38 fields. If the count has changed, run the partition sanity-check command in the next step before continuing.

- [ ] **P4: Confirm field partition is still valid.**

Run:
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
Expected: `missing from groups: set()` AND `extra in groups (not on schema): set()`. If either is non-empty, fix the `MISSILE_FIELD_GROUPS` partition in Task 1 before continuing.

- [ ] **P5: Confirm `_clear_unsupported_missile_properties` blast radius.**

Run: `grep -n "_clear_unsupported_missile" docker/docling-graph/app/evidence_gate.py`

Three possible outcomes:
- **Function exists and unconditionally nulls numerics** (parallel to the radar §4.8 bug): Task 10 of this plan refactors it the same way Task 10 of the radar plan refactored radar's version.
- **Function exists but already verifies values against evidence** (perhaps refactored alongside radar): Task 10 becomes a no-op verification step; document and skip.
- **Function doesn't exist**: missile has no equivalent erasure path. Task 10 becomes a verification-only step ("confirm no other code path silently nulls missile numerics") and the rest of the plan proceeds unchanged.

Record which outcome applies before starting Chunk 3.

---

## Chunk 1: Field-groups foundation + shared utilities

Tasks 1-2 add the missile partition to the existing `_field_groups.py` and create `_missile_shared.py` (sibling to `_radar_shared.py`). `_numeric_evidence.py` from the radar plan is reused as-is — no new task.

### Task 1: Add `MISSILE_FIELD_GROUPS` to `_field_groups.py`

**Files:**
- Modify: `ontology_bundles/air_defense_v3/extraction_schemas/_field_groups.py` (append constant)
- Test: `tests/unit/test_missile_field_groups_contract.py` (new file)

- [ ] **Step 1: Write the failing contract test.**

Create `tests/unit/test_missile_field_groups_contract.py`:

```python
"""Contract tests for MISSILE_FIELD_GROUPS partition.

The partition must (a) cover every field on MissileSystemEntity except
the meta `confidence` field, (b) place every field in exactly one group,
(c) include `system_name` as the first field in every group (identity).
"""
from __future__ import annotations

import pytest

from ontology_bundles.air_defense_v3.extraction_schemas._field_groups import (
    MISSILE_FIELD_GROUPS,
)
from ontology_bundles.air_defense_v3.extraction_schemas.missile_domain import (
    MissileSystemEntity,
)


def test_missile_field_groups_has_six_groups():
    expected_groups = {
        "missile_identity",
        "missile_kinematics",
        "missile_guidance",
        "missile_airframe",
        "missile_speed_timing",
        "missile_propulsion",
    }
    assert set(MISSILE_FIELD_GROUPS.keys()) == expected_groups


def test_every_group_starts_with_system_name():
    for name, fields in MISSILE_FIELD_GROUPS.items():
        assert fields[0] == "system_name", (
            f"group {name!r} must start with 'system_name' as identity, "
            f"got {fields[0]!r}"
        )


def test_partition_covers_every_missile_field_except_meta():
    declared = set(MissileSystemEntity.model_fields.keys())
    grouped = set()
    for fields in MISSILE_FIELD_GROUPS.values():
        grouped.update(fields)
    meta = {"confidence"}
    missing = declared - grouped - meta
    assert missing == set(), (
        f"MISSILE_FIELD_GROUPS missing fields: {sorted(missing)}. "
        f"Either add to a group or document as meta in test."
    )


def test_partition_has_no_extra_fields():
    declared = set(MissileSystemEntity.model_fields.keys())
    grouped = set()
    for fields in MISSILE_FIELD_GROUPS.values():
        grouped.update(fields)
    extra = grouped - declared
    assert extra == set(), (
        f"MISSILE_FIELD_GROUPS lists fields not on MissileSystemEntity: "
        f"{sorted(extra)}. Schema may have been renamed or removed."
    )


def test_each_field_appears_in_exactly_one_group_except_system_name():
    """system_name is the identity, replicated in every group. Every
    other field must appear in exactly one group."""
    seen: dict[str, int] = {}
    for fields in MISSILE_FIELD_GROUPS.values():
        for f in fields:
            seen[f] = seen.get(f, 0) + 1
    for field, count in seen.items():
        if field == "system_name":
            assert count == len(MISSILE_FIELD_GROUPS), (
                f"system_name must appear in all {len(MISSILE_FIELD_GROUPS)} "
                f"groups, found in {count}"
            )
        else:
            assert count == 1, (
                f"field {field!r} appears in {count} groups; should be 1"
            )
```

- [ ] **Step 2: Run test, expect FAIL with import error.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_missile_field_groups_contract.py -v 2>&1 | tail -15`
Expected: ImportError: cannot import name `MISSILE_FIELD_GROUPS` from `_field_groups`.

- [ ] **Step 3: Append the constant to `_field_groups.py`.**

Append to the existing `_field_groups.py` (after `RADAR_FIELD_GROUPS`):

```python
MISSILE_FIELD_GROUPS: dict[str, list[str]] = {
    "missile_identity": [
        "system_name",
        "nomenclature",
        "dieqp",
        "name",
        "emitter_function",
        "system_status",
        "asrd",
        "responsible_agency",
        "review_cycle",
        "next_review_date",
    ],
    "missile_kinematics": [
        "system_name",
        "min_intercept_km",
        "max_intercept_km",
        "min_altitude_km",
        "max_altitude_km",
        "max_launch_angle_deg",
    ],
    "missile_guidance": [
        "system_name",
        "guidance_type",
        "seeker_type",
        "missile_photo",
    ],
    "missile_airframe": [
        "system_name",
        "body_length_m",
        "body_diameter_m",
        "total_mass_kg",
    ],
    "missile_speed_timing": [
        "system_name",
        "average_speed_mps",
        "max_speed_mps",
        "max_flyout_time_sec",
        "flight_time_sec",
        "coast_time_sec",
        "intra_salvo_time_sec",
        "total_burn_time_sec",
        "ejector_time_sec",
    ],
    "missile_propulsion": [
        "system_name",
        "ejector_thrust",
        "ejector_mass_kg",
        "booster_time_sec",
        "booster_thrust",
        "booster_mass_kg",
        "sustain_time_sec",
        "sustain_thrust",
        "sustain_mass_kg",
    ],
}
```

- [ ] **Step 4: Run tests, expect all 5 passed.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_missile_field_groups_contract.py -v 2>&1 | tail -10`
Expected: exactly 5 passed.

- [ ] **Step 5: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/_field_groups.py tests/unit/test_missile_field_groups_contract.py
git commit -m "$(cat <<'EOF'
feat(extraction): MISSILE_FIELD_GROUPS partition (6 groups, 38 fields)

Adds the partition for splitting missile_domain into 6 focused sub-passes:
- missile_identity (10 string-ish fields)
- missile_kinematics (5 numerics + identity)
- missile_guidance (3 fields + identity)
- missile_airframe (3 numerics + identity)
- missile_speed_timing (8 numerics + identity)
- missile_propulsion (8 mixed fields + identity)

system_name appears in every group as the merge identity (matches radar's
pattern). 5 contract tests pin the invariants: 6 groups, system_name
first in each, every MissileSystemEntity field covered, no extras, no
duplicates outside system_name.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Create `_missile_shared.py` (forbidden set, validators, sanitizer factory)

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/_missile_shared.py`
- Test: `tests/unit/test_missile_shared.py`

Sources `_MISSILE_FORBIDDEN_SYSTEM_NAMES` from existing `missile_domain.py` (Chunk 1 of radar plan established that forbidden enforcement is the SINGLE authority of `make_root_sanitizer`; we mirror that here).

- [ ] **Step 1: Write the failing tests.**

Create `tests/unit/test_missile_shared.py`:

```python
"""Tests for _missile_shared module: forbidden set, validators, sanitizer factory."""
from __future__ import annotations

import pytest


def test_missile_forbidden_system_names_is_frozen_set():
    from ontology_bundles.air_defense_v3.extraction_schemas._missile_shared import (
        MISSILE_FORBIDDEN_SYSTEM_NAMES,
    )
    assert isinstance(MISSILE_FORBIDDEN_SYSTEM_NAMES, frozenset)
    assert len(MISSILE_FORBIDDEN_SYSTEM_NAMES) > 0


def test_missile_forbidden_set_contains_known_radar_names():
    """The missile forbidden set should reject radar names — Fan Song
    is a radar, not a missile, and must not be emitted as MISSILE_SYSTEM."""
    from ontology_bundles.air_defense_v3.extraction_schemas._missile_shared import (
        MISSILE_FORBIDDEN_SYSTEM_NAMES,
    )
    upper = {n.upper() for n in MISSILE_FORBIDDEN_SYSTEM_NAMES}
    assert "FAN SONG" in upper or "FAN_SONG" in upper, (
        "Missile forbidden set should reject radar names like Fan Song"
    )


def test_missile_optional_text_fields_includes_known_text_fields():
    from ontology_bundles.air_defense_v3.extraction_schemas._missile_shared import (
        MISSILE_OPTIONAL_TEXT_FIELDS,
    )
    for name in ("nomenclature", "guidance_type", "seeker_type"):
        assert name in MISSILE_OPTIONAL_TEXT_FIELDS, f"{name} missing"


def test_validate_missile_system_name_normalizes_whitespace():
    from ontology_bundles.air_defense_v3.extraction_schemas._missile_shared import (
        validate_missile_system_name,
    )
    assert validate_missile_system_name("  5V55K  ") == "5V55K"


def test_validate_missile_system_name_rejects_empty():
    from ontology_bundles.air_defense_v3.extraction_schemas._missile_shared import (
        validate_missile_system_name,
    )
    with pytest.raises(ValueError):
        validate_missile_system_name("")
    with pytest.raises(ValueError):
        validate_missile_system_name("   ")


def test_validate_missile_system_name_does_not_enforce_forbidden():
    """Forbidden enforcement lives in make_missile_root_sanitizer, not here.
    validate_missile_system_name only normalizes and rejects empty.
    canonicalize_identity_text() only normalizes whitespace, never case —
    so 'Fan Song' round-trips unchanged."""
    from ontology_bundles.air_defense_v3.extraction_schemas._missile_shared import (
        validate_missile_system_name,
    )
    assert validate_missile_system_name("Fan Song") == "Fan Song"


def test_make_missile_root_sanitizer_returns_callable():
    from ontology_bundles.air_defense_v3.extraction_schemas._missile_shared import (
        make_missile_root_sanitizer,
    )
    fn = make_missile_root_sanitizer(
        list_field="missile_systems",
        optional_text_fields={"nomenclature"},
    )
    assert callable(fn)


def test_make_missile_root_sanitizer_drops_forbidden_identities():
    """Factory must enforce forbidden-name list via sanitize_entity_list."""
    from pydantic import BaseModel, ConfigDict, model_validator
    from ontology_bundles.air_defense_v3.extraction_schemas._missile_shared import (
        make_missile_root_sanitizer,
        MISSILE_FORBIDDEN_SYSTEM_NAMES,
    )

    class FakeRecord(BaseModel):
        system_name: str

        model_config = ConfigDict(
            extra="ignore",
            ontology_name="MISSILE_SYSTEM",
            graph_id_fields=["system_name"],
            is_entity=True,
        )

    class FakePass(BaseModel):
        model_config = ConfigDict(extra="ignore")
        missile_systems: list[FakeRecord] = []

        _sanitize = model_validator(mode="before")(
            make_missile_root_sanitizer(
                list_field="missile_systems",
                optional_text_fields=set(),
            )
        )

    forbidden_name = next(iter(MISSILE_FORBIDDEN_SYSTEM_NAMES))
    inst = FakePass.model_validate({
        "missile_systems": [
            {"system_name": forbidden_name},
            {"system_name": "5V55K"},
        ]
    })
    names = [r.system_name for r in inst.missile_systems]
    assert forbidden_name not in names, (
        f"forbidden name {forbidden_name!r} should be dropped by sanitizer"
    )
    assert "5V55K" in names, "valid name should survive"
```

- [ ] **Step 2: Run tests, expect FAIL with import error.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_missile_shared.py -v 2>&1 | tail -15`
Expected: ImportError on `MISSILE_FORBIDDEN_SYSTEM_NAMES`.

- [ ] **Step 3: Write the module.**

Create `ontology_bundles/air_defense_v3/extraction_schemas/_missile_shared.py`:

```python
"""Shared utilities for missile sub-pass extraction schemas.

Mirrors _radar_shared.py exactly. The 6 missile sub-pass modules all
import from this file rather than copy-pasting forbidden sets, sanitizer
factories, or validator bodies.

Single authority for forbidden-name enforcement: make_missile_root_sanitizer.
Single authority for missile identity normalization: validate_missile_system_name.
"""
from __future__ import annotations

from typing import Any

from pydantic import model_validator

from ..validators import (
    canonicalize_identity_text,
    dedupe_entities_by_identity,
    sanitize_entity_list,
)
from .missile_domain import _MISSILE_FORBIDDEN_SYSTEM_NAMES
from .radar_domain import edge as edge  # re-export the same edge() decorator radar uses


# Frozen so accidental mutation in a sub-pass module fails loudly.
MISSILE_FORBIDDEN_SYSTEM_NAMES: frozenset[str] = frozenset(_MISSILE_FORBIDDEN_SYSTEM_NAMES)


# Superset across all missile sub-passes. Used by sub-pass sanitizer
# wiring to decide which fields qualify for optional-text coercion.
MISSILE_OPTIONAL_TEXT_FIELDS: frozenset[str] = frozenset({
    "nomenclature",
    "dieqp",
    "name",
    "emitter_function",
    "system_status",
    "asrd",
    "responsible_agency",
    "review_cycle",
    "next_review_date",
    "guidance_type",
    "seeker_type",
    "ejector_thrust",
    "booster_thrust",
    "sustain_thrust",
})


def validate_missile_system_name(value: Any) -> Any:
    """field_validator("system_name", mode="before") body for missile passes.

    Scope: normalization + non-empty-identity check only.
    Does NOT enforce the forbidden-names list — that authority lives
    exclusively in make_missile_root_sanitizer / sanitize_entity_list.
    """
    if value is None:
        raise ValueError("system_name is required and cannot be None")
    text = canonicalize_identity_text(value)
    if not text or not text.strip():
        raise ValueError("system_name cannot be empty / whitespace-only")
    return text.strip()


def make_missile_root_sanitizer(
    *,
    list_field: str,
    optional_text_fields: set[str] | frozenset[str],
):
    """Factory returning a model_validator(mode="before") body.

    The returned validator runs BOTH sanitize_entity_list AND
    dedupe_entities_by_identity, mirroring make_root_sanitizer in
    _radar_shared.py. Sanitize-only factories silently break
    duplicate-emission handling.

    Defaults forbidden_identities to MISSILE_FORBIDDEN_SYSTEM_NAMES so
    sub-pass modules don't have to import the constant directly. This
    is the SINGLE authority for forbidden-name enforcement; the
    field_validator on system_name only normalizes.
    """
    def _sanitize_and_dedupe(cls, values: Any) -> Any:
        values = sanitize_entity_list(
            cls,
            values,
            list_field=list_field,
            identity_field="system_name",
            optional_text_fields=set(optional_text_fields),
            forbidden_identities=MISSILE_FORBIDDEN_SYSTEM_NAMES,
        )
        return dedupe_entities_by_identity(cls, values)

    return _sanitize_and_dedupe
```

- [ ] **Step 4: Run tests, expect all passed.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_missile_shared.py -v 2>&1 | tail -10`
Expected: exactly 8 passed. The count is deterministic (no parametrization); a partial pass is a regression.

- [ ] **Step 5: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/_missile_shared.py tests/unit/test_missile_shared.py
git commit -m "$(cat <<'EOF'
feat(extraction): _missile_shared.py with forbidden set + sanitizer factory

Mirrors _radar_shared.py for the missile sub-passes:
- MISSILE_FORBIDDEN_SYSTEM_NAMES (frozen view of _MISSILE_FORBIDDEN_SYSTEM_NAMES)
- MISSILE_OPTIONAL_TEXT_FIELDS (superset across sub-passes)
- validate_missile_system_name (normalization + non-empty check, NOT
  forbidden enforcement)
- make_missile_root_sanitizer (factory; runs sanitize_entity_list +
  dedupe_entities_by_identity; defaults forbidden_identities so
  sub-passes don't import the set directly)
- edge (re-exported from radar_domain.py — same decorator both domains use)

Single authority for forbidden-name enforcement: make_missile_root_sanitizer.
Single authority for identity normalization: validate_missile_system_name.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 2: Sub-pass modules

Tasks 3-9 add the 6 sub-pass Pydantic modules + a description-quality contract test parametrized over them. Each module is ~70-110 lines following the radar plan's §4.4 template. After Chunk 2, all 6 modules are importable but not yet referenced from the manifest — `missile_domain` is still the active missile pass.

**Auto-evidence inheritance (same as radar):** `build_auto_field_evidence` runs per-pass automatically after every LLM call. Each sub-pass produces evidence rows for the fields it extracts; the worker-side merger aggregates `_field_evidence` across passes by `(instance_id, field_name)`. No per-sub-pass hook needed — sub-passes inherit the wiring by virtue of being normal pass-templates with `is_entity=True` records. If a numeric value disappears post-cutover, the cause is `_clear_unsupported_missile_properties` (Chunk 3, Task 10), not missing auto-evidence wiring.

### Task 3: Create `missile_identity.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/missile_identity.py`

- [ ] **Step 1: Write the module.**

```python
"""missile_identity extraction pass — missile identity + administrative metadata.

One of 6 sub-passes splitting the legacy missile_domain into smaller
LLM call boundaries. Emits MISSILE_SYSTEM[] with system_name as the
merge identity; merge_and_resolve collapses partial records from
sibling sub-passes onto one vertex.

Group fields: system_name, nomenclature, dieqp, name, emitter_function,
system_status, asrd, responsible_agency, review_cycle, next_review_date.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_text
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_identity"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]


class MissileIdentityRecord(BaseModel):
    """Subset of MissileSystemEntity covering identity + admin fields."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="MISSILE_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the MISSILE. Accept proper-noun "
            "missile names from prose (e.g. '5V55K', '9M82', 'AIM-120'). "
            "Never emit radar, weapon-system, aircraft, or platform "
            "names — those are filtered deterministically by the root "
            "sanitizer."
        ),
        examples=["5V55K", "AIM-120"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description=(
            "Official military nomenclature — formal alphanumeric "
            "designation. Distinct from system_name."
        ),
    )
    dieqp: Optional[str] = Field(
        default=None,
        description=(
            "Digital Intelligence Equipment Parameters identifier. "
            "Emit verbatim; do not infer."
        ),
    )
    name: Optional[str] = Field(
        default=None,
        description=(
            "Common or NATO reporting name when distinct from system_name. "
            "Free-text; emit verbatim."
        ),
    )
    emitter_function: Optional[str] = Field(
        default=None,
        description=(
            "Operational role of the missile. Emit only when the "
            "document explicitly assigns the role."
        ),
    )
    system_status: Optional[str] = Field(
        default=None,
        description=(
            "Lifecycle status: OPERATIONAL, DEVELOPMENTAL, RETIRED, "
            "UPGRADED, EXPORTED. Emit only when the document states it."
        ),
    )
    asrd: Optional[str] = Field(
        default=None,
        description=(
            "ASRD identifier from the All-Source Reference Document. "
            "Emit verbatim when stated."
        ),
    )
    responsible_agency: Optional[str] = Field(
        default=None,
        description=(
            "Organization responsible for the MDE record. Typically a "
            "3-letter IC acronym (IWC, NASIC, ONI, NGIC)."
        ),
    )
    review_cycle: Optional[str] = Field(
        default=None,
        description=(
            "Scheduled review cadence. Free-text; emit verbatim."
        ),
    )
    next_review_date: Optional[str] = Field(
        default=None,
        description=(
            "Next scheduled MDE review date. ISO 8601 preferred."
        ),
    )

    _v_system_name        = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_nomenclature       = field_validator("nomenclature", mode="before")(coerce_optional_text)
    _v_dieqp              = field_validator("dieqp", mode="before")(coerce_optional_text)
    _v_name               = field_validator("name", mode="before")(coerce_optional_text)
    _v_emitter_function   = field_validator("emitter_function", mode="before")(coerce_optional_text)
    _v_system_status      = field_validator("system_status", mode="before")(coerce_optional_text)
    _v_asrd               = field_validator("asrd", mode="before")(coerce_optional_text)
    _v_responsible_agency = field_validator("responsible_agency", mode="before")(coerce_optional_text)
    _v_review_cycle       = field_validator("review_cycle", mode="before")(coerce_optional_text)
    _v_next_review_date   = field_validator("next_review_date", mode="before")(coerce_optional_text)


class MissileIdentityPass(BaseModel):
    """Pass-root template — wraps missile_systems list."""

    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissileIdentityRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with identity + administrative "
            "metadata extracted from this batch."
        ),
        examples=[["5V55K", "9M82"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_missile_root_sanitizer(
            list_field="missile_systems",
            optional_text_fields={
                "nomenclature", "dieqp", "name", "emitter_function",
                "system_status", "asrd", "responsible_agency",
                "review_cycle", "next_review_date",
            },
        )
    )
```

- [ ] **Step 2: Verify importable.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.missile_identity import (
    MissileIdentityPass, MissileIdentityRecord
)
print('OK', sorted(MissileIdentityRecord.model_fields.keys()))
"
```
Expected: `OK ['asrd', 'dieqp', 'emitter_function', 'name', 'next_review_date', 'nomenclature', 'responsible_agency', 'review_cycle', 'system_name', 'system_status']`.

- [ ] **Step 3: Verify model_validate accepts a 5V55K record.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.missile_identity import MissileIdentityPass
inst = MissileIdentityPass.model_validate({
    'missile_systems': [
        {'system_name': '5V55K', 'nomenclature': '5V55K', 'emitter_function': 'INTERCEPTOR'}
    ]
})
print(inst.missile_systems[0].system_name, inst.missile_systems[0].nomenclature)
"
```
Expected: `5V55K 5V55K`.

- [ ] **Step 4: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/missile_identity.py
git commit -m "$(cat <<'EOF'
feat(extraction): missile_identity sub-pass (10 fields)

First of 6 missile sub-passes splitting the legacy missile_domain into
focused LLM-call boundaries. Emits MISSILE_SYSTEM[] with system_name as
merge identity. Group fields: system_name, nomenclature, dieqp, name,
emitter_function, system_status, asrd, responsible_agency, review_cycle,
next_review_date.

Module is importable but not yet referenced from manifest — missile_domain
is still the active pass until Chunk 4's cutover commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Create `missile_kinematics.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/missile_kinematics.py`

- [ ] **Step 1: Write the module.**

```python
"""missile_kinematics extraction pass — engagement envelope.

Group fields: system_name, min_intercept_km, max_intercept_km,
min_altitude_km, max_altitude_km, max_launch_angle_deg.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_kinematics"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]


class MissileKinematicsRecord(BaseModel):
    """Subset of MissileSystemEntity covering engagement envelope."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="MISSILE_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the MISSILE. Accept proper-noun "
            "missile names. Never emit radar, weapon-system, aircraft, "
            "or platform names — those are filtered deterministically."
        ),
        examples=["5V55K", "AIM-120"],
    )
    min_intercept_km: Optional[float] = Field(
        default=None,
        description=(
            "Minimum intercept range in kilometers. Emit only when the "
            "source states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    max_intercept_km: Optional[float] = Field(
        default=None,
        description=(
            "Maximum intercept range in kilometers. Emit only when the "
            "source states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    min_altitude_km: Optional[float] = Field(
        default=None,
        description=(
            "Minimum engagement altitude in kilometers. Emit only when "
            "the source states value AND unit."
        ),
    )
    max_altitude_km: Optional[float] = Field(
        default=None,
        description=(
            "Maximum engagement altitude in kilometers. Emit only when "
            "the source states value AND unit."
        ),
    )
    max_launch_angle_deg: Optional[float] = Field(
        default=None,
        description=(
            "Maximum launch angle in degrees. Emit only when the source "
            "states value AND unit."
        ),
    )

    _v_system_name           = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_min_intercept_km      = field_validator("min_intercept_km", mode="before")(coerce_optional_float)
    _v_max_intercept_km      = field_validator("max_intercept_km", mode="before")(coerce_optional_float)
    _v_min_altitude_km       = field_validator("min_altitude_km", mode="before")(coerce_optional_float)
    _v_max_altitude_km       = field_validator("max_altitude_km", mode="before")(coerce_optional_float)
    _v_max_launch_angle_deg  = field_validator("max_launch_angle_deg", mode="before")(coerce_optional_float)


class MissileKinematicsPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissileKinematicsRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with engagement-envelope values "
            "extracted from this batch."
        ),
        examples=[["5V55K"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_missile_root_sanitizer(
            list_field="missile_systems",
            optional_text_fields=set(),
        )
    )
```

- [ ] **Step 2: Verify importable.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.missile_kinematics import (
    MissileKinematicsPass, MissileKinematicsRecord
)
print('OK', sorted(MissileKinematicsRecord.model_fields.keys()))
"
```
Expected: `OK ['max_altitude_km', 'max_intercept_km', 'max_launch_angle_deg', 'min_altitude_km', 'min_intercept_km', 'system_name']`.

- [ ] **Step 3: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/missile_kinematics.py
git commit -m "feat(extraction): missile_kinematics sub-pass (6 fields)

system_name + min/max_intercept_km + min/max_altitude_km +
max_launch_angle_deg. Float fields use coerce_optional_float; numeric
descriptions reference DELTA_SYSTEM_PROMPT Unit Policy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Create `missile_guidance.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/missile_guidance.py`

- [ ] **Step 1: Write the module.**

```python
"""missile_guidance extraction pass — guidance + seeker + photo flag.

Group fields: system_name, guidance_type, seeker_type, missile_photo.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_text
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_guidance"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]


class MissileGuidanceRecord(BaseModel):
    """Subset of MissileSystemEntity covering guidance + seeker."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="MISSILE_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the MISSILE. Accept proper-noun "
            "missile names. Never emit radar, weapon-system, aircraft, "
            "or platform names — those are filtered deterministically."
        ),
        examples=["5V55K", "AIM-120"],
    )
    guidance_type: Optional[str] = Field(
        default=None,
        description=(
            "Guidance scheme. Free-text; emit verbatim from the source."
        ),
    )
    seeker_type: Optional[str] = Field(
        default=None,
        description=(
            "Seeker type (e.g. semi-active radar homing, infrared). "
            "Free-text; emit verbatim from the source."
        ),
    )
    # Optional[bool] uses Pydantic's native bool parsing — same pattern
    # as radar_antenna's antenna_photo / spoiled fields.
    missile_photo: Optional[bool] = Field(
        default=None,
        description=(
            "Whether a missile photograph is included in the record. "
            "Use null when not stated."
        ),
    )

    _v_system_name    = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_guidance_type  = field_validator("guidance_type", mode="before")(coerce_optional_text)
    _v_seeker_type    = field_validator("seeker_type", mode="before")(coerce_optional_text)
    # missile_photo: no validator — Pydantic native bool parsing.


class MissileGuidancePass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissileGuidanceRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with guidance + seeker values "
            "extracted from this batch."
        ),
        examples=[["5V55K"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_missile_root_sanitizer(
            list_field="missile_systems",
            optional_text_fields={"guidance_type", "seeker_type"},
        )
    )
```

- [ ] **Step 2: Verify importable.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.missile_guidance import (
    MissileGuidancePass, MissileGuidanceRecord
)
print('OK', sorted(MissileGuidanceRecord.model_fields.keys()))
"
```
Expected: `OK ['guidance_type', 'missile_photo', 'seeker_type', 'system_name']`.

- [ ] **Step 3: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/missile_guidance.py
git commit -m "feat(extraction): missile_guidance sub-pass (4 fields)

system_name + guidance_type + seeker_type + missile_photo. Optional[bool]
uses Pydantic native parsing (no coerce_optional_bool helper).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Create `missile_airframe.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/missile_airframe.py`

- [ ] **Step 1: Write the module.**

```python
"""missile_airframe extraction pass — body geometry + mass.

Group fields: system_name, body_length_m, body_diameter_m, total_mass_kg.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_airframe"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]


class MissileAirframeRecord(BaseModel):
    """Subset of MissileSystemEntity covering body geometry + mass."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="MISSILE_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the MISSILE. Accept proper-noun "
            "missile names. Never emit radar, weapon-system, aircraft, "
            "or platform names — those are filtered deterministically."
        ),
        examples=["5V55K", "AIM-120"],
    )
    body_length_m: Optional[float] = Field(
        default=None,
        description=(
            "Missile body length in meters. Emit only when the source "
            "states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    body_diameter_m: Optional[float] = Field(
        default=None,
        description=(
            "Missile body diameter in meters. Emit only when the source "
            "states value AND unit."
        ),
    )
    total_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Total missile mass at launch in kilograms. Emit only when "
            "the source states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )

    _v_system_name      = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_body_length_m    = field_validator("body_length_m", mode="before")(coerce_optional_float)
    _v_body_diameter_m  = field_validator("body_diameter_m", mode="before")(coerce_optional_float)
    _v_total_mass_kg    = field_validator("total_mass_kg", mode="before")(coerce_optional_float)


class MissileAirframePass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissileAirframeRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with body geometry + mass values "
            "extracted from this batch."
        ),
        examples=[["5V55K"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_missile_root_sanitizer(
            list_field="missile_systems",
            optional_text_fields=set(),
        )
    )
```

- [ ] **Step 2: Verify importable.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.missile_airframe import (
    MissileAirframePass, MissileAirframeRecord
)
print('OK', sorted(MissileAirframeRecord.model_fields.keys()))
"
```
Expected: `OK ['body_diameter_m', 'body_length_m', 'system_name', 'total_mass_kg']`.

- [ ] **Step 3: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/missile_airframe.py
git commit -m "feat(extraction): missile_airframe sub-pass (4 fields)

system_name + body_length_m + body_diameter_m + total_mass_kg. All
floats use coerce_optional_float; numeric descriptions reference
DELTA_SYSTEM_PROMPT Unit Policy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Create `missile_speed_timing.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/missile_speed_timing.py`

- [ ] **Step 1: Write the module.**

```python
"""missile_speed_timing extraction pass — speed + flight-time + burn-time.

Group fields: system_name, average_speed_mps, max_speed_mps,
max_flyout_time_sec, flight_time_sec, coast_time_sec,
intra_salvo_time_sec, total_burn_time_sec, ejector_time_sec.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_speed_timing"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]


class MissileSpeedTimingRecord(BaseModel):
    """Subset of MissileSystemEntity covering speed + flight/burn timing."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="MISSILE_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the MISSILE. Accept proper-noun "
            "missile names. Never emit radar, weapon-system, aircraft, "
            "or platform names — those are filtered deterministically."
        ),
        examples=["5V55K", "AIM-120"],
    )
    average_speed_mps: Optional[float] = Field(
        default=None,
        description=(
            "Average flight speed in meters per second. Emit only when "
            "the source states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    max_speed_mps: Optional[float] = Field(
        default=None,
        description=(
            "Maximum flight speed in meters per second. Emit only when "
            "the source states value AND unit."
        ),
    )
    max_flyout_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Maximum flyout time in seconds. Emit only when the source "
            "states value AND unit."
        ),
    )
    flight_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Total flight time in seconds. Emit only when the source "
            "states value AND unit."
        ),
    )
    coast_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Coast time (post-burn) in seconds. Emit only when the "
            "source states value AND unit."
        ),
    )
    intra_salvo_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Intra-salvo time (between launches in a salvo) in seconds. "
            "Emit only when the source states value AND unit."
        ),
    )
    total_burn_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Total motor burn time in seconds. Emit only when the source "
            "states value AND unit."
        ),
    )
    ejector_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Ejector burn duration in seconds. Emit only when the source "
            "states value AND unit."
        ),
    )

    _v_system_name           = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_average_speed_mps     = field_validator("average_speed_mps", mode="before")(coerce_optional_float)
    _v_max_speed_mps         = field_validator("max_speed_mps", mode="before")(coerce_optional_float)
    _v_max_flyout_time_sec   = field_validator("max_flyout_time_sec", mode="before")(coerce_optional_float)
    _v_flight_time_sec       = field_validator("flight_time_sec", mode="before")(coerce_optional_float)
    _v_coast_time_sec        = field_validator("coast_time_sec", mode="before")(coerce_optional_float)
    _v_intra_salvo_time_sec  = field_validator("intra_salvo_time_sec", mode="before")(coerce_optional_float)
    _v_total_burn_time_sec   = field_validator("total_burn_time_sec", mode="before")(coerce_optional_float)
    _v_ejector_time_sec      = field_validator("ejector_time_sec", mode="before")(coerce_optional_float)


class MissileSpeedTimingPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissileSpeedTimingRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with speed + flight/burn timing "
            "values extracted from this batch."
        ),
        examples=[["5V55K"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_missile_root_sanitizer(
            list_field="missile_systems",
            optional_text_fields=set(),
        )
    )
```

- [ ] **Step 2: Verify importable.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.missile_speed_timing import (
    MissileSpeedTimingPass, MissileSpeedTimingRecord
)
print('OK', sorted(MissileSpeedTimingRecord.model_fields.keys()))
"
```
Expected: `OK ['average_speed_mps', 'coast_time_sec', 'ejector_time_sec', 'flight_time_sec', 'intra_salvo_time_sec', 'max_flyout_time_sec', 'max_speed_mps', 'system_name', 'total_burn_time_sec']`.

- [ ] **Step 3: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/missile_speed_timing.py
git commit -m "feat(extraction): missile_speed_timing sub-pass (9 fields)

system_name + average_speed_mps + max_speed_mps + max_flyout_time_sec +
flight_time_sec + coast_time_sec + intra_salvo_time_sec +
total_burn_time_sec + ejector_time_sec. All numerics use
coerce_optional_float; descriptions reference DELTA_SYSTEM_PROMPT
Unit Policy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Create `missile_propulsion.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/missile_propulsion.py`

- [ ] **Step 1: Write the module.**

```python
"""missile_propulsion extraction pass — staged motor parameters.

Group fields: system_name, ejector_thrust, ejector_mass_kg,
booster_time_sec, booster_thrust, booster_mass_kg, sustain_time_sec,
sustain_thrust, sustain_mass_kg.

Note: ejector_thrust, booster_thrust, sustain_thrust are typed
Optional[str] on the canonical MissileSystemEntity. Promoting them to
numerics (kN, lbf) is schema-correction work tracked separately, NOT
part of this field-group split.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float, coerce_optional_text
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_propulsion"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]


class MissilePropulsionRecord(BaseModel):
    """Subset of MissileSystemEntity covering staged-motor parameters."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="MISSILE_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the MISSILE. Accept proper-noun "
            "missile names. Never emit radar, weapon-system, aircraft, "
            "or platform names — those are filtered deterministically."
        ),
        examples=["5V55K", "AIM-120"],
    )
    ejector_thrust: Optional[str] = Field(
        default=None,
        description=(
            "Ejector-stage thrust description. Free-text; emit verbatim "
            "from the source."
        ),
    )
    ejector_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Ejector-stage mass in kilograms. Emit only when the source "
            "states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    booster_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Booster-stage burn time in seconds. Emit only when the "
            "source states value AND unit."
        ),
    )
    booster_thrust: Optional[str] = Field(
        default=None,
        description=(
            "Booster-stage thrust description. Free-text; emit verbatim."
        ),
    )
    booster_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Booster-stage mass in kilograms. Emit only when the source "
            "states value AND unit."
        ),
    )
    sustain_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Sustainer-stage burn time in seconds. Emit only when the "
            "source states value AND unit."
        ),
    )
    sustain_thrust: Optional[str] = Field(
        default=None,
        description=(
            "Sustainer-stage thrust description. Free-text; emit verbatim."
        ),
    )
    sustain_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Sustainer-stage mass in kilograms. Emit only when the "
            "source states value AND unit."
        ),
    )

    _v_system_name        = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_ejector_thrust     = field_validator("ejector_thrust", mode="before")(coerce_optional_text)
    _v_ejector_mass_kg    = field_validator("ejector_mass_kg", mode="before")(coerce_optional_float)
    _v_booster_time_sec   = field_validator("booster_time_sec", mode="before")(coerce_optional_float)
    _v_booster_thrust     = field_validator("booster_thrust", mode="before")(coerce_optional_text)
    _v_booster_mass_kg    = field_validator("booster_mass_kg", mode="before")(coerce_optional_float)
    _v_sustain_time_sec   = field_validator("sustain_time_sec", mode="before")(coerce_optional_float)
    _v_sustain_thrust     = field_validator("sustain_thrust", mode="before")(coerce_optional_text)
    _v_sustain_mass_kg    = field_validator("sustain_mass_kg", mode="before")(coerce_optional_float)


class MissilePropulsionPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissilePropulsionRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with staged-motor parameters "
            "extracted from this batch."
        ),
        examples=[["5V55K"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_missile_root_sanitizer(
            list_field="missile_systems",
            optional_text_fields={"ejector_thrust", "booster_thrust", "sustain_thrust"},
        )
    )
```

- [ ] **Step 2: Verify importable.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.missile_propulsion import (
    MissilePropulsionPass, MissilePropulsionRecord
)
print('OK', sorted(MissilePropulsionRecord.model_fields.keys()))
"
```
Expected: `OK ['booster_mass_kg', 'booster_thrust', 'booster_time_sec', 'ejector_mass_kg', 'ejector_thrust', 'sustain_mass_kg', 'sustain_thrust', 'sustain_time_sec', 'system_name']`.

- [ ] **Step 3: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/missile_propulsion.py
git commit -m "feat(extraction): missile_propulsion sub-pass (9 fields)

system_name + ejector_{thrust,mass_kg} + booster_{time_sec,thrust,mass_kg} +
sustain_{time_sec,thrust,mass_kg}. Three string thrust fields stay
Optional[str] (canonical schema shape); promotion to numerics is tracked
as schema-correction work, not part of the field-group split.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Description-quality contract test for missile sub-passes

**Files:**
- Create: `tests/unit/test_missile_field_groups_desc_contract.py`

- [ ] **Step 1: Write the parametrized test.**

```python
"""Description-quality contract tests for the 6 missile sub-pass record classes.

Mirrors tests/unit/test_radar_field_groups_contract.py's
test_record_descriptions_well_formed pattern.
"""
from __future__ import annotations

from typing import get_args, get_origin

import pytest

from ontology_bundles.air_defense_v3.extraction_schemas import (
    missile_identity, missile_kinematics, missile_guidance,
    missile_airframe, missile_speed_timing, missile_propulsion,
)


def _allows_numeric(annotation) -> bool:
    """Recursive numeric-allowance check via typing.get_origin/get_args."""
    if annotation in (float, int):
        return True
    return any(_allows_numeric(arg) for arg in get_args(annotation))


def _record_class(module):
    from pydantic import BaseModel
    return next(
        c for c in vars(module).values()
        if isinstance(c, type)
        and issubclass(c, BaseModel)
        and isinstance(c.model_config, dict)
        and c.model_config.get("ontology_name") == "MISSILE_SYSTEM"
    )


# Tokens that should NEVER appear in any sub-pass description outside
# the whitelisted instructive sentence.
_FORBIDDEN_NAME_TOKENS = (
    "fan song", "spoon rest", "tombstone",  # radar names
    "an/mpq",                                # radar nomenclature prefix
)


@pytest.mark.parametrize("module", [
    missile_identity, missile_kinematics, missile_guidance,
    missile_airframe, missile_speed_timing, missile_propulsion,
])
def test_record_descriptions_well_formed(module):
    record_cls = _record_class(module)
    for fname, finfo in record_cls.model_fields.items():
        desc = (finfo.description or "").strip()
        assert desc, f"{record_cls.__name__}.{fname}: empty description"

        if _allows_numeric(finfo.annotation):
            assert not finfo.examples, (
                f"{record_cls.__name__}.{fname}: numeric field has examples "
                f"{finfo.examples} — strip per spec §4.4 sanitization (b)"
            )

        lower = desc.lower()
        for banned in ("typical", "common ranges", "forbidden values"):
            assert banned not in lower, (
                f"{record_cls.__name__}.{fname}: description contains "
                f"{banned!r} — strip per spec §4.4 sanitization"
            )


@pytest.mark.parametrize("module", [
    missile_identity, missile_kinematics, missile_guidance,
    missile_airframe, missile_speed_timing, missile_propulsion,
])
def test_system_name_description_excludes_forbidden_tokens(module):
    """Catch verbatim FORBIDDEN-block leakage on the missile identity field.

    The legitimate system_name description tells the LLM never to emit
    radar/weapon-system/aircraft/platform names; that single instructive
    sentence is allowed. What's NOT allowed is leaking the FORBIDDEN
    list itself (e.g. an enumerated dump of "Fan Song, Spoon Rest, ...").
    """
    record_cls = _record_class(module)
    desc = (record_cls.model_fields["system_name"].description or "").lower()

    # Strip the one whitelisted instructive sentence so its mention of
    # "radar, weapon-system, aircraft, or platform" doesn't trip the check.
    whitelisted = "never emit radar, weapon-system, aircraft, or platform names"
    cleaned = desc.replace(whitelisted, "")

    for token in _FORBIDDEN_NAME_TOKENS:
        assert token not in cleaned, (
            f"{record_cls.__name__}.system_name description leaked "
            f"forbidden-name token {token!r} outside the whitelisted "
            f"instructive sentence — strip the FORBIDDEN-values block."
        )
```

- [ ] **Step 2: Run tests.**

Run:
```bash
SKIP_COV=1 .venv/bin/pytest tests/unit/test_missile_field_groups_desc_contract.py -v 2>&1 | tail -20
```
Expected: 6 + 6 = 12 passed (each parametrized test runs across the 6 sub-passes).

If `test_record_descriptions_well_formed` fires, fix the offending field's description in the relevant sub-pass module. If `test_system_name_description_excludes_forbidden_tokens` fires, you have FORBIDDEN-block leakage outside the whitelisted sentence.

- [ ] **Step 3: Commit.**

```bash
git add tests/unit/test_missile_field_groups_desc_contract.py
git commit -m "$(cat <<'EOF'
test(extraction): description-quality contract for 6 missile sub-passes

Two parametrized tests across the 6 sub-pass record classes:

1. test_record_descriptions_well_formed
   - every field has a non-empty description
   - numeric-typed fields carry no examples
   - descriptions don't contain "typical", "common ranges", or
     "forbidden values"

2. test_system_name_description_excludes_forbidden_tokens
   - whitelists the one instructive sentence about radar/weapon-system
     names; rejects any other appearance of forbidden-name tokens
     (fan song, spoon rest, tombstone, an/mpq)

Mirrors the radar field-group description-quality contract test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 3: Cutover prep (additive code changes)

Tasks 10-15 land additive changes that keep `missile_domain` working while preparing the codebase for the manifest cutover. Each task ships independently with main green.

### Task 10: Refactor `_clear_unsupported_missile_properties` (or verify absence)

**Files:**
- Modify (if function exists): `docker/docling-graph/app/evidence_gate.py`
- Test: `docker/docling-graph/tests/test_clear_unsupported_missile_properties.py` (new file, only if function exists)

**Branching based on Pre-flight P5 outcome:**

- [ ] **Step 1: Re-verify the P5 outcome.**

Run: `grep -nA5 "_clear_unsupported_missile" docker/docling-graph/app/evidence_gate.py`

Three branches:

**Branch A — function exists and unconditionally nulls numerics:**

Refactor to use `value_is_supported_by_text` from `_numeric_evidence.py` (already extracted by radar Task 3). Mirror the radar Task 10 pattern exactly: per-field evidence verification before nulling. Build the `evidence_gate_fields` tuple from the union of missile numeric-bearing groups (`kinematics`, `airframe`, `speed_timing`, `propulsion`). Add a regression test parallel to `test_clear_unsupported_radar_properties.py` covering supported / unsupported / unit-conversion / text-field-preserved / text-field-nulled cases. Add a drift-prevention test (parallel to radar Task 10 Step 4b) that asserts the tuple matches the union of group field sets.

**Branch B — function exists but already verifies values:**

Add a verification test only — assert the function does NOT unconditionally null any field in `evidence_gate_fields`. Document in a comment that Branch B was the post-condition.

**Branch C — function doesn't exist:**

Run a broader audit:
```bash
grep -rn "missile" docker/docling-graph/app/ | grep -iE "null|clear|strip" | head -20
```
If no other code path silently nulls missile numerics, this task is verification-only. Add a comment in `evidence_gate.py` (or a new test file) noting the audit was performed and no equivalent regression exists.

- [ ] **Step 2: Implement chosen branch (Branch A — refactor).**

If Branch B or C applied, jump ahead to Step 3b (Branch B/C verification artifact). For Branch A, edit `_clear_unsupported_missile_properties` so its body is structurally identical to the post-refactor radar function. Two parts: (a) the explicit `evidence_gate_fields` tuple, (b) the per-field verification loop. Inlined here so the implementer doesn't have to open the radar plan to copy the loop body.

```python
from app._numeric_evidence import value_is_supported_by_text

def _clear_unsupported_missile_properties(item: dict, evidence_text: str) -> list[str]:
    """Mirror of _clear_unsupported_radar_properties (spec §4.8 pattern).

    Preserves any numeric/bool field whose value (or a unit-aware
    variant) appears in evidence_text. Values without textual support
    are nulled. Same predicate the auto-evidence resolver uses — single
    source of truth.

    Note on group coverage:
    - missile_identity: no numerics, no entries here.
    - missile_guidance: only the bool missile_photo is gate-relevant
      (string fields stay handled by the unchanged exact-text branch).
    - missile_kinematics, missile_airframe, missile_speed_timing,
      missile_propulsion: all numeric fields enumerated below.
    - confidence: meta, gated identically.
    """
    cleared: list[str] = []

    # Existing exact-text branch for string fields stays unchanged here
    # (mirror radar's _value_is_quoted_in_text loop body if present).
    # ... preserved ...

    evidence_gate_fields = (
        # missile_kinematics numerics
        "min_intercept_km", "max_intercept_km",
        "min_altitude_km", "max_altitude_km", "max_launch_angle_deg",
        # missile_airframe numerics
        "body_length_m", "body_diameter_m", "total_mass_kg",
        # missile_speed_timing numerics
        "average_speed_mps", "max_speed_mps",
        "max_flyout_time_sec", "flight_time_sec", "coast_time_sec",
        "intra_salvo_time_sec", "total_burn_time_sec", "ejector_time_sec",
        # missile_propulsion numerics
        "ejector_mass_kg", "booster_time_sec", "booster_mass_kg",
        "sustain_time_sec", "sustain_mass_kg",
        # missile_guidance bool
        "missile_photo",
        # meta
        "confidence",
    )
    for field_name in evidence_gate_fields:
        value = item.get(field_name)
        if value is None:
            continue
        if not value_is_supported_by_text(value, field_name, evidence_text):
            item[field_name] = None
            cleared.append(field_name)

    return cleared
```

Add the regression test at `docker/docling-graph/tests/test_clear_unsupported_missile_properties.py` covering 5 cases parallel to the radar test:
- supported numeric (e.g. `body_length_m=7.5` with evidence "length 7.5 m") preserved
- unsupported numeric (`max_intercept_km=999.0` with evidence "max range 43 km") nulled
- same-unit-suffix variant (`total_mass_kg=1500.0` with evidence "mass 1500 kg") preserved — same-unit suffix appended; the helper does NOT convert "1.5 tonnes" to 1500 kg, that would require real unit conversion which is out of scope (see helper docstring)
- text-field preserved by exact-text branch (`guidance_type="semi-active"` with evidence "semi-active radar homing")
- text-field nulled by exact-text branch (`seeker_type="active"` with evidence containing no "active")

**Out of scope for Session 1:** Real cross-unit conversion (1.5 tonnes ↔ 1500 kg, 43 km ↔ 43000 m, etc.). The helper only matches the value's stringified form with the field's expected unit suffix appended. If a doc states a value in a non-canonical unit and the LLM doesn't normalize, the value gets nulled. Tracked as Session 2 follow-up if false-negatives become a real problem.

- [ ] **Step 3: Run regression test, expect 5 passed.**

Run: `cd docker/docling-graph && python -m pytest tests/test_clear_unsupported_missile_properties.py -v 2>&1 | tail -10`
Expected (Branch A): 5 passed.

- [ ] **Step 3b: Branch B/C — write verification artifact instead.**

If Branch A doesn't apply (function already verifies, or function doesn't exist), create one of:

- **Branch B test** — `docker/docling-graph/tests/test_missile_numeric_erasure_audit.py` with one assertion: pick any missile numeric field, call the existing function with a value present in evidence text, assert the value survives (function already verifies).
- **Branch C audit comment** — append to `docker/docling-graph/app/evidence_gate.py` (after the missile dispatch, in a `# AUDIT:` comment block) noting the date the audit was performed and the negative result.

Re-run: `cd docker/docling-graph && python -m pytest tests/ -v 2>&1 | tail -10`
Expected: all passed.

- [ ] **Step 4: Add a drift-prevention assertion (Branch A only).**

Append to `docker/docling-graph/tests/test_clear_unsupported_missile_properties.py`:

```python
def test_evidence_gate_fields_matches_field_groups():
    """Drift guard: the evidence_gate_fields tuple in
    _clear_unsupported_missile_properties must equal the union of all
    numeric/bool non-identity fields across the 4 numeric/parameter
    sub-pass groups (kinematics, airframe, speed_timing, propulsion)
    plus 'missile_photo' (the only gate-relevant guidance field) plus
    'confidence' meta. If a new numeric field is added to any group,
    this test must be updated alongside the tuple — otherwise the new
    field is silently nulled forever.
    """
    import inspect
    from app.evidence_gate import _clear_unsupported_missile_properties
    from ontology_bundles.air_defense_v3.extraction_schemas._field_groups import (
        MISSILE_FIELD_GROUPS,
    )

    src = inspect.getsource(_clear_unsupported_missile_properties)
    assert "evidence_gate_fields = (" in src, "tuple name changed; update this test"

    expected = (
        set(MISSILE_FIELD_GROUPS["missile_kinematics"])
        | set(MISSILE_FIELD_GROUPS["missile_airframe"])
        | set(MISSILE_FIELD_GROUPS["missile_speed_timing"])
        | set(MISSILE_FIELD_GROUPS["missile_propulsion"])
    ) - {"system_name"}
    expected.add("missile_photo")  # only gate-relevant guidance field
    expected.add("confidence")     # meta

    for field in expected:
        assert f'"{field}"' in src, (
            f"evidence_gate_fields tuple missing {field!r}; "
            f"new field added to MISSILE_FIELD_GROUPS without updating "
            f"the gate. Numeric values for {field} would be nulled."
        )
```

Re-run: `cd docker/docling-graph && python -m pytest tests/test_clear_unsupported_missile_properties.py -v 2>&1 | tail -10`
Expected (Branch A): 6 passed.

- [ ] **Step 5: Run the full evidence-gate tests to confirm no regression.**

Run: `cd docker/docling-graph && python -m pytest tests/ -v 2>&1 | tail -10`
Expected: all passed. The legacy missile_domain pass continues to work because `apply_bundle_postprocessing` still dispatches on `pass_name == "missile_domain"`.

- [ ] **Step 6: Commit.**

Branch A:

```bash
git add docker/docling-graph/app/evidence_gate.py docker/docling-graph/tests/test_clear_unsupported_missile_properties.py
git commit -m "$(cat <<'EOF'
fix(docling-graph): _clear_unsupported_missile_properties preserves evidenced numerics

Mirrors the radar §4.8 fix for the missile postprocessing path. Previously
the function nulled missile numerics unconditionally, dropping correctly-
extracted values before the response left the service.

Refactor: each numeric value is now checked via the shared
value_is_supported_by_text predicate from _numeric_evidence.py (already
extracted during the radar refactor). Same predicate the auto-evidence
resolver uses — single source of truth.

Drift-prevention test asserts the evidence_gate_fields tuple matches
the union of the 4 numeric missile sub-pass groups plus missile_photo
and confidence — schema additions can't silently null new fields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Branch B (verification test):

```bash
git add docker/docling-graph/app/evidence_gate.py \
        docker/docling-graph/tests/test_missile_numeric_erasure_audit.py
git commit -m "chore(docling-graph): verify _clear_unsupported_missile_properties already verifies values

Audit confirmed the existing function already checks numeric values
against evidence text — no refactor needed. Verification test added
to lock in the post-condition.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Branch C (no function exists):

```bash
git add docker/docling-graph/app/evidence_gate.py
git commit -m "chore(docling-graph): verify no missile-numeric erasure path exists

Audit confirmed _clear_unsupported_missile_properties does not exist
and no other code path silently nulls missile numerics. AUDIT comment
added to evidence_gate.py for future reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Update `apply_bundle_postprocessing` dispatch (additive)

**Files:**
- Modify: `docker/docling-graph/app/evidence_gate.py:307-` (the dispatch on pass_name)

Update the dispatch to recognize the 6 new missile sub-pass names IN ADDITION to `missile_domain`. After this commit, the dispatch fires the missile postprocessor for both old and new names — safe to land before manifest cutover.

- [ ] **Step 1: Find the dispatch site.**

Run: `grep -n "pass_name == \"missile_domain\"\|pass_name in (\"missile_domain\"\|MISSILE_PASS_NAMES" docker/docling-graph/app/evidence_gate.py | head`

- [ ] **Step 2: Update the conditional.**

```python
MISSILE_PASS_NAMES = (
    "missile_domain",
    "missile_identity",
    "missile_kinematics",
    "missile_guidance",
    "missile_airframe",
    "missile_speed_timing",
    "missile_propulsion",
)

# In apply_bundle_postprocessing, replace:
#   if pass_name == "missile_domain":
# with:
#   if pass_name in MISSILE_PASS_NAMES:
```

- [ ] **Step 3: Verify idempotency contract — bracket-access scan + dict-shape probe.**

Each sub-pass invocation hands the post-processor only that group's slice of fields. Two checks:

(a) Grep for any `item[<field>]` bracket access on missile fields anywhere in the file. Bracket access raises `KeyError` on a missing field; `.get()` is safe.

```bash
grep -nE "item\[['\"]([a-z_]+)['\"]\]" docker/docling-graph/app/evidence_gate.py \
  | grep -E "missile|min_intercept|max_intercept|min_altitude|max_altitude|max_launch_angle|body_length|body_diameter|total_mass|average_speed|max_speed|max_flyout_time|flight_time|coast_time|intra_salvo|total_burn|ejector|booster|sustain|guidance_type|seeker_type|missile_photo|nomenclature|emitter_function|system_status|asrd|responsible_agency|review_cycle|next_review_date"
```

Expected: empty output (no bracket-access on missile fields). Any hit means change to `.get()` before continuing.

Also catch generic `assert`/raise patterns that might fire on partial slices:

```bash
grep -nE "assert\b|raise (KeyError|ValueError|TypeError)" \
  docker/docling-graph/app/evidence_gate.py | head -30
```

Visually scan results for any assertion that requires a missile-specific field to be present.

(b) Smoke-test the dispatch directly with a partial dict:

```bash
cd docker/docling-graph && python -c "
from app.evidence_gate import apply_bundle_postprocessing
# Each sub-pass slice — only the fields that group emits
identity_slice = {'system_name': '5V55K', 'nomenclature': '5V55K'}
kinematics_slice = {'system_name': '5V55K', 'max_intercept_km': 43.0}
guidance_slice = {'system_name': '5V55K', 'guidance_type': 'semi-active'}
for pass_name, item in [
    ('missile_identity', identity_slice),
    ('missile_kinematics', kinematics_slice),
    ('missile_guidance', guidance_slice),
]:
    out = apply_bundle_postprocessing(
        bundle_key='air_defense_v3',
        pass_name=pass_name,
        pass_output={'missile_systems': [item]},
        evidence_text='5V55K maximum intercept range 43 km. semi-active homing.',
    )
    print(f'{pass_name}: ok')
"
```

Expected: three `ok` lines, no exception.

- [ ] **Step 4: Run service tests.**

Run: `cd docker/docling-graph && python -m pytest tests/ -v 2>&1 | tail -5`
Expected: all passed.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/evidence_gate.py
git commit -m "feat(docling-graph): apply_bundle_postprocessing recognizes missile sub-passes

Additive: dispatch now matches both legacy missile_domain and the 6 new
sub-pass names. Safe to land before manifest cutover — missile_domain
still fires through the same postprocessor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Update `_DOMAIN_PASS_NAMES` (additive)

**Files:**
- Modify: `app/workers/pipeline.py:381-383` (or wherever `_DOMAIN_PASS_NAMES` lives post-radar-cutover)

- [ ] **Step 0: Confirm current contents and merge accordingly.**

The radar plan's Chunk 4 cutover prunes `radar_domain` from `_DOMAIN_PASS_NAMES`. If those commits are landed before this task starts (per the prerequisite-check at the top of the plan), the set should already contain the 5 radar sub-pass names. If for any reason this missile session is started while radar work is still in progress, the merge below could collide with the radar plan's open edits to the same frozenset.

Verify the set's *current* contents before editing:

```bash
grep -n -A 15 "_DOMAIN_PASS_NAMES" app/workers/pipeline.py | head -30
```

Then construct the Step 2 frozenset literal as `<current contents> ∪ <6 new missile sub-pass names>`. Do NOT blindly paste the literal below if the radar 5 names are missing or `radar_domain` is still present — that would silently revert radar work.

- [ ] **Step 1: Locate and inspect.**

Run: `grep -n "_DOMAIN_PASS_NAMES\|domain_hit" app/workers/pipeline.py | head`

- [ ] **Step 2: Update the frozenset.**

Assuming Step 0 confirmed the post-radar-cutover state (5 radar sub-pass names, no `radar_domain`), add the 6 missile sub-pass names alongside `missile_domain` (additive — `missile_domain` is still active until Chunk 4):

```python
_DOMAIN_PASS_NAMES = frozenset({
    # radar (post-radar-cutover — confirm via Step 0 grep)
    "radar_identity", "radar_power_rf", "radar_antenna",
    "radar_timing", "radar_modulation",
    # missile (additive — missile_domain still active until Chunk 4)
    "missile_domain",
    "missile_identity", "missile_kinematics", "missile_guidance",
    "missile_airframe", "missile_speed_timing", "missile_propulsion",
})
```

If Step 0 showed a different starting state, adjust the radar entries to match what's actually there.

- [ ] **Step 3: Run pipeline tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/pipeline -q 2>&1 | tail -5`
Expected: all passed.

- [ ] **Step 4: Commit.**

```bash
git add app/workers/pipeline.py
git commit -m "feat(pipeline): _DOMAIN_PASS_NAMES recognizes missile sub-passes (additive)

Adds the 6 new missile sub-pass names alongside missile_domain. Safe to
land before manifest cutover — both names route through the same
domain-hit logic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Bundle `__init__.py` exports for sub-pass classes (optional ergonomics)

**Files:**
- Modify (optional): `ontology_bundles/air_defense_v3/extraction_schemas/__init__.py`

**Status:** Not required for correctness. The docling-graph bundle loader at `docker/docling-graph/app/bundles.py:53-70` resolves each pass via `importlib.import_module(f"ontology_bundles.{bundle_key}.{module_name}")` using the manifest's `module` field, then reads `template_class` directly off the module. It does NOT consult the bundle's `__init__.py` for re-exports. The radar plan's equivalent task is also optional.

If the post-radar-cutover `__init__.py` re-exports the radar sub-pass templates as a stylistic convention, mirror it for missile here. Otherwise, skip this task and remove it from `.tasks.json`.

- [ ] **Step 1: Decide whether to skip.**

Run: `grep -nE "RadarIdentityPass|RadarPowerRfPass" ontology_bundles/air_defense_v3/extraction_schemas/__init__.py 2>/dev/null`

- **If found:** the radar cutover added re-exports; mirror the convention for missile (continue to Step 2).
- **If not found:** the radar cutover skipped this task; skip it for missile too. Mark Task 13 done with `git commit --allow-empty -m "chore(plan): skip Task 13 — bundle __init__.py re-exports not used"`.

- [ ] **Step 2 (only if continuing): Add the 6 missile sub-pass exports.**

Mirror whatever pattern the radar sub-pass exports established. If there's a `__all__`, append to it.

- [ ] **Step 3 (only if continuing): Verify importable.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas import (
    MissileIdentityPass, MissileKinematicsPass, MissileGuidancePass,
    MissileAirframePass, MissileSpeedTimingPass, MissilePropulsionPass,
)
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 4 (only if continuing): Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/__init__.py
git commit -m "chore(extraction): export 6 missile sub-pass templates from bundle __init__ (ergonomic)

Optional ergonomics — mirrors the radar __init__ convention. The
docling-graph bundle loader (docker/docling-graph/app/bundles.py)
imports submodules via importlib.import_module by manifest 'module'
field; it does NOT consult __init__ for template_class lookup. So
this re-export is for downstream callers' convenience only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: `check_bundle()` coverage validation passes

**Files:** none modified (verification only)

- [ ] **Step 1: Run the bundle checker.**

Run: `.venv/bin/python -c "from app.bundles.checker import check_bundle; result = check_bundle('air_defense_v3'); print('errors:', len(result.errors)); print('warnings:', len(result.warnings))"`

(If the import path is different, find it: `grep -rn "def check_bundle" --include="*.py" .`)

Expected: errors == 0. Warnings about both `missile_domain` and the new sub-passes existing simultaneously are acceptable in this additive phase.

- [ ] **Step 2: No commit (verification only).**

---

### Task 15: Worker `_parse_pass_response` regression sweep

**Files:** none modified (verification only)

- [ ] **Step 1: Run targeted missile tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit -k "missile or postprocess" -q 2>&1 | tail -5`
Expected: all passed.

- [ ] **Step 2: No commit (verification only).**

---

### Task 16: Test-fixture pass-name updates (`missile_domain` → `missile_identity`)

**Files:**
- Modify (if applicable): `docker/docling-graph/tests/test_extract_pass_endpoint.py`
- Modify (if applicable): `docker/docling-graph/tests/test_service_identity_gate.py`
- Modify (if applicable): any test fixture pinning `pass_name="missile_domain"`

- [ ] **Step 1: Find the literals.**

Run: `grep -rn 'pass_name.*missile_domain\|"missile_domain"' docker/docling-graph/tests/`

- [ ] **Step 2: Update fixtures.**

Replace `pass_name="missile_domain"` with `pass_name="missile_identity"` in request-body fixtures.

If `test_service_identity_gate.py` imports `MissileDomainPass` for legacy-loadability, **keep** the import as a regression check (parallel to radar's pattern). Add a comment noting why.

- [ ] **Step 3: Run the service tests.**

Run: `cd docker/docling-graph && python -m pytest tests/ -v 2>&1 | tail -10`

**Decision rule (don't burn cycles debugging):**

- **Pass:** continue to Step 4 (commit).
- **Fail with manifest-resolution error mentioning `missile_identity`:** stage the fixture changes for Chunk 4's cutover commit (Task 17) instead — `git stash push -m "missile-task16-fixture-updates" -- docker/docling-graph/tests/test_*.py` and pop after the cutover lands.
- **Fail with anything else:** real regression — diagnose before continuing.

- [ ] **Step 4: Commit (only if tests pass).**

```bash
git add docker/docling-graph/tests/test_extract_pass_endpoint.py docker/docling-graph/tests/test_service_identity_gate.py
git commit -m "test(docling-graph): fixture pass_name → missile_identity

Updates fixtures from missile_domain to missile_identity. The legacy
MissileDomainPass import in test_service_identity_gate.py (if present)
is kept intentionally as a legacy-loadability regression check.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 4: Manifest cutover + verification

Tasks 17-21 flip the manifest, prune `missile_domain` from `_DOMAIN_PASS_NAMES`, update manifest-shape tests, and verify end-to-end against the live service.

### Task 17: Manifest cutover (atomic commit)

**Files:**
- Modify: `ontology_bundles/air_defense_v3/manifest.yaml`
- Modify: `app/workers/pipeline.py` (prune `missile_domain` from `_DOMAIN_PASS_NAMES`)
- Modify: `tests/unit/test_ontology_bundles.py` (manifest-shape assertion: 12 passes)
- Modify: `tests/unit/test_extraction_schemas.py` (PASS_MODULES — add 6 missile sub-passes, drop missile_domain)
- Modify: `tests/integration/test_pr1_scaffolding_smoke.py` (literal pass-name lists)
- Modify: `ontology_bundles/air_defense_v3/extraction_schemas/system_links.py` (docstring; if needed)
- Stage if Task 16 was deferred: `docker/docling-graph/tests/test_*.py`. If you stashed Task 16's fixture changes per its decision rule, `git stash pop` them now and they land in this atomic commit.

This commit must land everything together so main stays green between commits but post-commit the active extraction shape is the new 12-pass structure (5 radar sub-passes + 6 missile sub-passes + system_links).

- [ ] **Step 1: Update `manifest.yaml`.**

Remove the `missile_domain` entry; add 6 new entries; update `system_links.depends_on`. The radar entries from the prior cutover stay unchanged.

```yaml
# Remove:
#   - name: missile_domain
#     ...

# Add (after the 5 radar sub-pass entries):
  - name: missile_identity
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.missile_identity
    template_class: MissileIdentityPass
    primary_entity_types: [MISSILE_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: missile_kinematics
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.missile_kinematics
    template_class: MissileKinematicsPass
    primary_entity_types: [MISSILE_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: missile_guidance
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.missile_guidance
    template_class: MissileGuidancePass
    primary_entity_types: [MISSILE_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: missile_airframe
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.missile_airframe
    template_class: MissileAirframePass
    primary_entity_types: [MISSILE_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: missile_speed_timing
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.missile_speed_timing
    template_class: MissileSpeedTimingPass
    primary_entity_types: [MISSILE_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: missile_propulsion
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.missile_propulsion
    template_class: MissilePropulsionPass
    primary_entity_types: [MISSILE_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []
```

Update **only** `system_links.depends_on` to list all 11 entity passes (5 radar + 6 missile). Every other field on the `system_links` entry — `required`, `kind`, `input_mode`, `module`, `template_class`, `primary_entity_types`, `bridge_entity_types`, `extracted_relationship_types`, `skip_if_no_upstream_endpoints`, `skip_justification` — must be left exactly as the post-radar-cutover manifest has them. Verify with `grep -A 14 "name: system_links" ontology_bundles/air_defense_v3/manifest.yaml` before editing.

Patch the `depends_on` list only:

```yaml
  - name: system_links
    # all other keys unchanged from current manifest
    depends_on:
      - radar_identity
      - radar_power_rf
      - radar_antenna
      - radar_timing
      - radar_modulation
      - missile_identity
      - missile_kinematics
      - missile_guidance
      - missile_airframe
      - missile_speed_timing
      - missile_propulsion
```

- [ ] **Step 2: Prune `missile_domain` from `_DOMAIN_PASS_NAMES`.**

In `app/workers/pipeline.py`, remove `"missile_domain"` from the frozenset. The 6 new sub-pass names stay.

- [ ] **Step 3: Update `tests/unit/test_ontology_bundles.py`.**

Update the manifest-shape assertion: 12 passes total (5 radar + 6 missile + 1 system_links). Update the `expected_pass_names` set if present.

- [ ] **Step 4: Update `tests/unit/test_extraction_schemas.py`.**

`PASS_MODULES` should iterate the new 6 missile sub-passes; remove the `missile_domain` entry.

- [ ] **Step 5: Update `tests/integration/test_pr1_scaffolding_smoke.py`.**

Update any literal `"missile_domain"` references to the new 6 sub-pass names.

- [ ] **Step 6: Update `system_links.py` docstring (if it mentions missile_domain).**

Run: `grep -n "missile_domain" ontology_bundles/air_defense_v3/extraction_schemas/system_links.py`. If found, update the docstring to reference the 6 sub-pass names.

- [ ] **Step 7: Run the full unit + pipeline suite.**

Run:
```bash
SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline \
  --ignore=tests/unit/test_extraction_schemas.py \
  --ignore=tests/unit/test_specification_entity_validation.py \
  --tb=no -p no:warnings 2>&1 | grep -E "passed|failed" | tail -3

SKIP_COV=1 .venv/bin/pytest tests/unit/test_extraction_schemas.py -v 2>&1 | tail -10
```
Expected: all passed (vs. baseline established in P1).

- [ ] **Step 8: Run pr1-scaffolding-smoke.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/integration/test_pr1_scaffolding_smoke.py -v 2>&1 | tail -10`
Expected: all passed.

- [ ] **Step 9: Run docling-graph service tests.**

Run: `cd docker/docling-graph && python -m pytest tests/ -v 2>&1 | tail -10`
Expected: all passed.

- [ ] **Step 10: Commit.**

```bash
git add ontology_bundles/air_defense_v3/manifest.yaml \
        app/workers/pipeline.py \
        tests/unit/test_ontology_bundles.py \
        tests/unit/test_extraction_schemas.py \
        tests/integration/test_pr1_scaffolding_smoke.py \
        ontology_bundles/air_defense_v3/extraction_schemas/system_links.py \
        docker/docling-graph/tests/test_extract_pass_endpoint.py \
        docker/docling-graph/tests/test_service_identity_gate.py
git commit -m "$(cat <<'EOF'
feat(extraction): manifest cutover — missile_domain → 6 sub-passes

Atomic cutover commit. Replaces the single missile_domain extraction
pass with six focused sub-passes (missile_identity, missile_kinematics,
missile_guidance, missile_airframe, missile_speed_timing,
missile_propulsion), each with its own /extract-pass call against a
4-9 field schema. All emit MISSILE_SYSTEM[] with system_name identity;
merge_and_resolve collapses partial records onto one vertex.

Mirrors the radar field-group cutover (commit ref). Same architectural
reasoning: smaller schema per LLM call → better numeric extraction.

Manifest changes:
- Remove missile_domain entry
- Add 6 new entity passes (depends_on: [])
- system_links.depends_on lists all 11 entity passes (5 radar + 6 missile)

Code changes (downstream of additive Task 11-16 prep commits):
- app/workers/pipeline.py: prune missile_domain from _DOMAIN_PASS_NAMES
- tests/unit/test_ontology_bundles.py: assert 12 passes, new pass-name set
- tests/unit/test_extraction_schemas.py: PASS_MODULES iterates 6 sub-passes
- tests/integration/test_pr1_scaffolding_smoke.py: literal pass-name lists
- system_links.py docstring: pass-name reference updated (if present)
- docker/docling-graph/tests/*: fixture pass_name → missile_identity

Verification:
- check_bundle() reports 0 errors
- Field-group contract tests (5) pass
- Description-quality contract (6 sub-pass record classes) passes
- pr1-scaffolding-smoke + ontology-bundles + extraction-schemas tests pass

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 18: Full unit + pipeline regression sweep

**Files:** none modified (verification only)

- [ ] **Step 1: Run the full sweep.**

```bash
SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline \
  --ignore=tests/unit/test_extraction_schemas.py \
  --ignore=tests/unit/test_specification_entity_validation.py \
  --tb=no -p no:warnings 2>&1 | grep -E "passed|failed" | tail -3
```

(See radar plan Task 18 for ignore-rationale.)

- [ ] **Step 2: Document results.**

Expected: ≥1240 passed; same skipped/xfailed counts as P1 baseline. Any new failures must be investigated and fixed before proceeding.

- [ ] **Step 3: No commit (verification only).**

---

### Task 19: Smoke harness for missile sub-passes

**Files:**
- Create: `tests/integration/test_missile_field_groups_smoke.py`

Three smoke cases verifying that the field-group split improves missile numeric extraction. Cases must be confirmed against the corpus before this task — see Pre-flight P3/P4 and §"Decisions to confirm" in the missile follow-up TODO. Suggested cases:

| Case | Expected numeric | Source-text bracket | Sub-pass that should hit |
|------|------------------|---------------------|--------------------------|
| 5V55K (SA-2 missile) | `max_intercept_km` in [30, 60] | "max range 43 km" | missile_kinematics |
| 5V28 (SA-5 missile) | `max_speed_mps` in [800, 1500] | "speed Mach 3.5" or stated mps | missile_speed_timing |
| 9M82 (SA-12 missile) | `body_length_m` in [4, 9] | "length 7.5 m" | missile_airframe |

**Confirm before continuing:** the corpus must contain ingest-able sources where these cases appear with the stated numeric values. If not, swap to known-good cases.

- [ ] **Step 1: Write the smoke harness.**

```python
"""Smoke harness for missile field-group extraction (mirror of
tests/integration/test_radar_field_groups_smoke.py).

Three known cases that exercise the numeric extraction the field-group
split is intended to improve. Each test parametrizes:
- pass_name to invoke
- source text containing the proper-noun missile name + numeric value
- target field
- exact-match system_name
- acceptable [lower, upper] range bracketing the source-text value

Marked @pytest.mark.integration; default `pytest tests/unit tests/pipeline`
does not pick this up — run explicitly with the marker.

Range calibration policy: ranges bracket the source-text value with
tolerance for unit-conversion rounding, NOT the model's observed output.
If a future model emits 5000 km for a doc that says 43 km, this test
SHOULD fail. Recalibrating to model output would mask regressions.
"""
import os
import pytest
import requests

DOCLING_GRAPH_URL = os.environ.get(
    "DOCLING_GRAPH_URL", "http://localhost:8002/extract-pass"
)


def _build_doc(text: str) -> dict:
    """Minimal valid DoclingDocument with one paragraph (label='text')."""
    return {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "test-missile-smoke",
        "origin": {
            "mimetype": "text/plain",
            "binary_hash": 1,
            "filename": "smoke.txt",
        },
        "furniture": {
            "name": "_root_", "self_ref": "#/furniture", "children": [],
        },
        "body": {
            "name": "_root_", "self_ref": "#/body",
            "children": [{"$ref": "#/texts/0"}],
        },
        "groups": [], "pictures": [], "tables": [],
        "key_value_items": [], "form_items": [], "pages": {},
        "texts": [{
            "self_ref": "#/texts/0",
            "parent": {"$ref": "#/body"},
            "label": "text",
            "prov": [],
            "orig": text,
            "text": text,
        }],
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    "pass_name,text,system_name,field,lower,upper",
    [
        ("missile_kinematics",
         "The 5V55K missile has a maximum intercept range of 43 km.",
         "5V55K", "max_intercept_km", 30.0, 60.0),
        ("missile_speed_timing",
         "The 5V28 missile achieves a maximum speed of 1200 m/s.",
         "5V28", "max_speed_mps", 800.0, 1500.0),
        ("missile_airframe",
         "The 9M82 missile body length is 7.5 m.",
         "9M82", "body_length_m", 4.0, 9.0),
    ],
    ids=["5V55K-range-43km", "5V28-speed-1200mps", "9M82-length-7.5m"],
)
def test_missile_field_group_numeric_smoke(
    pass_name, text, system_name, field, lower, upper
):
    body = {
        "bundle_key": "air_defense_v3",
        "pass_name": pass_name,
        "document_id": f"smoke-{pass_name}-{field}",
        "docling_document_json": _build_doc(text),
        # NOTE: omit upstream_entities entirely for document_only passes;
        # the endpoint rejects document_only requests when the key is
        # present (even with an empty list).
    }
    try:
        resp = requests.post(DOCLING_GRAPH_URL, json=body, timeout=180)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"docling-graph not available at {DOCLING_GRAPH_URL}")

    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:500]}"

    payload = resp.json()
    pass_output = payload.get("pass_output", {})
    missile_systems = pass_output.get("missile_systems", []) or []
    assert len(missile_systems) >= 1, (
        f"expected ≥1 missile_system; pass_output={pass_output!r}"
    )

    # Exact-match on system_name (not substring) — missile names like
    # 5V55K / 5V28 / 9M82 are unambiguous identifiers, unlike radar's
    # multi-token "Fan Song". Substring matching could collide with
    # near-similar designations.
    entity = next(
        (e for e in missile_systems
         if (e.get("system_name") or "") == system_name),
        None,
    )
    assert entity is not None, (
        f"{system_name} not found; got system_names="
        f"{[e.get('system_name') for e in missile_systems]!r}"
    )

    value = entity.get(field)
    if value is None:
        print(f"\n--- FAILURE DEBUG: pass_output ---\n{pass_output}\n---")
        pytest.fail(
            f"{pass_name}.{field} was None for {system_name}; "
            f"expected value in [{lower}, {upper}]"
        )
    assert isinstance(value, (int, float)), (
        f"{field} is {type(value).__name__}, want number; got {value!r}"
    )
    assert lower <= float(value) <= upper, (
        f"{field}={value} not in [{lower}, {upper}]"
    )
```

- [ ] **Step 2: Verify the test discovers correctly.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/integration/test_missile_field_groups_smoke.py --collect-only 2>&1 | tail -5`
Expected: 3 collected.

- [ ] **Step 3: Verify it skips when docling-graph is offline.**

```bash
DOCLING_GRAPH_URL="http://localhost:9999/extract-pass" \
  SKIP_COV=1 .venv/bin/pytest tests/integration/test_missile_field_groups_smoke.py -v 2>&1 | tail -10
```
Expected: 3 skipped.

- [ ] **Step 4: Run the smoke harness against the live service.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/integration/test_missile_field_groups_smoke.py -v -m integration 2>&1 | tail -15`

**Decision gate (per spec §10 fallback):**

| Result | Action |
|--------|--------|
| 3/3 passed | ✅ Field grouping unblocked numeric extraction. Continue to Task 20. |
| 2/3 passed | ✅ Acceptable per success criteria. Continue to Task 20; document the 1 failure for Session 2. |
| 0/3 or 1/3 | ⚠️ **STOP.** Field-group split did not move missile numerics. Switch to spec §10 fallback architecture (candidate-mapping). Do NOT continue iterating on prompt tuning. |

- [ ] **Step 5: Commit.**

```bash
git add tests/integration/test_missile_field_groups_smoke.py
git commit -m "$(cat <<'EOF'
test(extraction): missile field-group smoke harness

3 cases hit live docling-graph at :8002/extract-pass with minimal
DoclingDocuments and assert numeric extraction lands in source-truth-
bracketed ranges:

- "5V55K maximum intercept range 43 km"  → max_intercept_km in [30, 60]
- "5V28 maximum speed 1200 m/s"          → max_speed_mps in [800, 1500]
- "9M82 body length 7.5 m"               → body_length_m in [4, 9]

Mirrors tests/integration/test_radar_field_groups_smoke.py exactly,
with the radar Fan-Song-substring matcher replaced by exact-match
on missile system_names (5V55K / 5V28 / 9M82 are unambiguous).

Marked @pytest.mark.integration; skipped when service is offline.
Decision gate: ≥2/3 means continue iterating field groups; <2/3 means
switch to spec §10 candidate-mapping fallback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 20: Container rebuild + service health check

**Files:** none modified (operational task).

- [ ] **Step 1: Stop services.**

Run: `docker compose down`
Expected: all containers stopped cleanly.

- [ ] **Step 2: Rebuild docling-graph.**

Run: `docker compose build docling-graph`
Expected: build succeeds. Per project memory, the docling-graph image uses COPY (no volume mount for app code), so a rebuild is required after every code change in `docker/docling-graph/app/`.

- [ ] **Step 3: Start services.**

Run: `./manage.sh --start`
Wait 30s for health checks.

- [ ] **Step 4: Verify service health.**

Run: `curl -s http://localhost:8002/health | jq`
Expected: `{"status": "ok"}` or equivalent. (Port 8002 matches the radar plan and the live docling-graph deployment; the smoke harness uses the same port.)

Verify install patches loaded (prompt_rules + resolver_patch):
```bash
docker compose logs docling-graph 2>&1 | grep -E "prompt_rules.*installed|resolver_patch.*installed" | head -5
```
Expected: both messages present.

- [ ] **Step 5: No commit (operational task).**

---

### Task 21: End-to-end re-ingest verification + acceptance gate

**Files:** none modified (verification only).

- [ ] **Step 1: Pick a missile-bearing fixture.**

Identify an ingest-able document containing one of the smoke harness systems (5V55K, 5V28, or 9M82). Note its `instance_id` for ArcadeDB queries below. Record the `(system_name, expected_field)` pair from the table below — Step 4 asserts on the *specific* field for the chosen system, not "any of four":

| Fixture system | Expected field |
|----------------|----------------|
| 5V55K          | `max_intercept_km` |
| 5V28           | `max_speed_mps` |
| 9M82           | `body_length_m` |

- [ ] **Step 2: Trigger re-ingest.**

Use the same endpoint pattern the radar plan verified at Task 21 (`http://localhost:8005/v1/documents/<DOC_ID>/reingest`):

```bash
DOC_ID="<paste fixture instance_id>"
curl -sS -X POST "http://localhost:8005/v1/documents/$DOC_ID/reingest" \
  -H "Content-Type: application/json" \
  -d '{"mode": "graph_only"}'
```

If the radar plan's Task 21 used a different endpoint (e.g. it changed across sessions), align to whatever the radar cutover commit confirmed working and update this command in a follow-up edit before proceeding.

- [ ] **Step 3: Wait for completion.**

```bash
docker compose logs --since 1m -f worker-graph 2>&1 \
  | grep -E "derive_ontology_graph.*succeeded|FAILURE"
# Ctrl-C when one of the missile sub-passes completes (one chain succeeds = all 6 missile passes done; system_links runs after).
```
Expected: a `succeeded` entry for at least one missile sub-pass. If `FAILURE` appears, diagnose before continuing.

- [ ] **Step 4: Verify the specific MISSILE_SYSTEM vertex has the expected field populated.**

Query ArcadeDB for the vertex matching the system from Step 1. Use the system+field pair you recorded:

```bash
SYSTEM_NAME="<paste from Step 1 — e.g. 5V55K>"
EXPECTED_FIELD="<paste from Step 1 — e.g. max_intercept_km>"

# Replace with the project's actual ArcadeDB query path.
# The radar Task 21 used http://localhost:2480/api/v1/query/<db> with
# basic-auth; mirror whatever pattern that task confirmed.
curl -sS -u root:<pwd> -X POST \
  "http://localhost:2480/api/v1/query/EIP" \
  -H "Content-Type: application/json" \
  -d "{\"language\": \"sql\", \"command\": \"SELECT system_name, ${EXPECTED_FIELD} FROM MISSILE_SYSTEM WHERE system_name = '${SYSTEM_NAME}'\"}"
```

Assert: the result row exists AND `${EXPECTED_FIELD}` is non-null AND its numeric value matches the source-text bracket from the smoke harness.

If the field is null on the vertex but Task 19's smoke harness passed for that case, the `_clear_unsupported_missile_properties` refactor (Task 10) may not have landed correctly OR the merge layer is dropping the value — diagnose before declaring success. If a different missile field is populated but not the expected one, the LLM may be misclassifying the value (record as a Session 2 investigation item but do not block acceptance).

- [ ] **Step 5: Apply acceptance gate.**

| Smoke result + E2E result | Outcome |
|----------------------------|---------|
| Smoke ≥2/3 AND E2E vertex has the expected field populated | **Accept.** Field-group split for missile is working. |
| Smoke ≥2/3 AND E2E vertex's expected field is null | Likely Task 10 regression OR merge-layer issue. Investigate `_clear_unsupported_missile_properties` and `merge_and_resolve` for MISSILE_SYSTEM. Re-run Step 4. |
| Smoke <2/3 | Already stopped at Task 19. Switch to spec §10 fallback. |

- [ ] **Step 6: Document outcome (separate commit).**

Add a short paragraph to the plan file or a separate notes file recording: smoke result, E2E result, which acceptance branch fired, any anomalies. Commit:

```bash
git add docs/superpowers/plans/2026-04-27-missile-field-group-extraction.md
git commit -m "docs(plan): record missile field-group acceptance result

Smoke: <X/3 result>. E2E: <vertex result>. Acceptance: <accepted | needs investigation | fallback triggered>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Notes

- **Session boundary:** This plan is intended for execution in a single session after the radar plan completes. If the user opens a new session, run `/superpowers-extended-cc:executing-plans docs/superpowers/plans/2026-04-27-missile-field-group-extraction.md` to resume.
- **Out-of-scope (deferred to later sessions):**
  - Identity/parameter split for missile (mirrors radar Item #3)
  - Group-scoped retry for missile (mirrors radar Item #7)
  - Promotion of `ejector_thrust`/`booster_thrust`/`sustain_thrust` from `Optional[str]` to numeric — schema-correction work
  - Per-pass diagnostics persistence
  - Golden test harness with per-field recall/precision
- **Cross-references:**
  - Radar plan (source pattern): `docs/superpowers/plans/2026-04-27-radar-field-group-extraction.md`
  - Spec (architecture): `docs/superpowers/specs/2026-04-27-radar-field-group-extraction-design.md`
  - Spec §10 fallback if smoke <2/3: candidate-mapping architecture
