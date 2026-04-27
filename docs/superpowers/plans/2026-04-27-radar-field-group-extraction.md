# Radar Field-Group Extraction Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single `radar_domain` extraction pass with five focused sub-passes (`radar_identity`, `radar_power_rf`, `radar_antenna`, `radar_timing`, `radar_modulation`) so each LLM call sees a smaller schema, while preserving auto-evidence wiring and the merge layer's identity collapse.

**Architecture:** Each sub-pass is its own `/extract-pass` call against a 5-11 field schema. All emit `RADAR_SYSTEM[]` with `system_name` identity; existing `merge_and_resolve` collapses partial records onto one vertex. Refactor `_clear_unsupported_radar_properties` to verify numeric values against batch evidence text via shared helpers extracted from `provenance.py`.

**Tech Stack:** Python 3.11/3.12, Pydantic v2, FastAPI, docling-graph LLM extraction service, Ollama gemma4:31b, ArcadeDB, pytest.

**Spec:** [`docs/superpowers/specs/2026-04-27-radar-field-group-extraction-design.md`](../specs/2026-04-27-radar-field-group-extraction-design.md) (commit `9c60b1b`, signed off after 5 review passes)

---

## Pre-flight checklist

Run these once at the start of the session and before each chunk to confirm baseline:

- [ ] **P0: Read the spec.**

Run: `wc -l docs/superpowers/specs/2026-04-27-radar-field-group-extraction-design.md`
Expected: ≥ 800 lines. If less, the file is truncated — abort.

Use the @superpowers-extended-cc:test-driven-development skill for every code-bearing task.

- [ ] **P1: Confirm baseline test suite status.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline -q --ignore=tests/unit/test_extraction_schemas.py --ignore=tests/unit/test_specification_entity_validation.py 2>&1 | grep -E "passed|failed" | tail -3`
Expected: ≥1240 passed, with at most the 3 known xfails from prior work. Document any failure as a pre-existing issue not caused by this plan.

- [ ] **P2: Confirm stack is up.**

Run: `docker compose ps --format "table {{.Service}}\t{{.Status}}" | grep -E "api|arcadedb|postgres|worker-graph|docling-graph"`
Expected: all five services Up. If not, `./manage.sh --start` and wait 30 s.

- [ ] **P3: Confirm field counts on extraction-side RadarSystemEntity.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import RadarSystemEntity
fields = list(RadarSystemEntity.model_fields.keys())
print(f'extraction-side RadarSystemEntity fields: {len(fields)}')
"
```
Expected: ≥30 fields. The spec's `RADAR_FIELD_GROUPS` partitions these.

---

## Chunk 1: Field-groups foundation + shared utilities

Tasks 1-3 establish the data structures every sub-pass module imports from. After Chunk 1, no manifest changes; existing `radar_domain` pass keeps running. Contract tests pass against the still-active legacy pass because they assert field-set partitioning, independent of the manifest.

### Task 1: Create `_field_groups.py` + group-membership contract tests

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/_field_groups.py`
- Create: `tests/unit/test_radar_field_groups_contract.py`

- [ ] **Step 1: Write failing contract tests.**

Create `tests/unit/test_radar_field_groups_contract.py`:

```python
"""Phase B Session 1 — RADAR_FIELD_GROUPS partitioning contract tests.

The contract is about partitioning the **extraction-side**
RadarSystemEntity's fields, not the canonical entity. The canonical
includes structural / system fields outside extraction scope.
"""
import pytest

from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import (
    RadarSystemEntity,
)
from ontology_bundles.air_defense_v3.extraction_schemas._field_groups import (
    RADAR_FIELD_GROUPS,
)


def test_every_group_includes_system_name():
    for group, fields in RADAR_FIELD_GROUPS.items():
        assert "system_name" in fields, f"{group} missing system_name"


def test_every_listed_field_exists_on_canonical():
    canonical = set(RadarSystemEntity.model_fields.keys())
    for group, fields in RADAR_FIELD_GROUPS.items():
        unknown = [f for f in fields if f not in canonical]
        assert not unknown, f"{group}: unknown fields {unknown}"


def test_no_non_identity_field_in_multiple_groups():
    seen: dict[str, str] = {}
    for group, fields in RADAR_FIELD_GROUPS.items():
        for f in fields:
            if f == "system_name":
                continue
            if f in seen:
                pytest.fail(f"{f!r} in both {seen[f]} and {group}")
            seen[f] = group


def test_every_flat_checklist_field_appears_exactly_once():
    """Every non-system_field on the extraction-side RadarSystemEntity
    is in exactly one group; no field listed in groups is missing."""
    expected: set[str] = {
        fname for fname, finfo in RadarSystemEntity.model_fields.items()
        if not (
            isinstance(finfo.json_schema_extra, dict)
            and finfo.json_schema_extra.get("system_field") is True
        )
    }
    expected |= set(RadarSystemEntity.model_config.get("graph_id_fields", []) or [])

    grouped: set[str] = set()
    for fields in RADAR_FIELD_GROUPS.values():
        grouped.update(fields)

    missing = expected - grouped
    extra = grouped - expected
    assert not missing, f"flat-checklist fields not in any group: {sorted(missing)}"
    assert not extra, f"groups reference non-flat-checklist fields: {sorted(extra)}"


def test_system_fields_are_excluded():
    canonical = RadarSystemEntity.model_fields
    system_fields = {
        f for f, info in canonical.items()
        if isinstance(info.json_schema_extra, dict)
        and info.json_schema_extra.get("system_field") is True
    }
    for group, fields in RADAR_FIELD_GROUPS.items():
        for f in fields:
            assert f not in system_fields, f"{group}: includes system field {f}"
```

- [ ] **Step 2: Run tests, expect ImportError.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_radar_field_groups_contract.py -v 2>&1 | tail -10`
Expected: `ImportError: cannot import name 'RADAR_FIELD_GROUPS' from 'ontology_bundles.air_defense_v3.extraction_schemas._field_groups'` (or the module is missing entirely).

- [ ] **Step 3: Create `_field_groups.py`.**

Write `ontology_bundles/air_defense_v3/extraction_schemas/_field_groups.py`:

```python
"""Single source of truth for radar extraction field groups.

Each group becomes its own /extract-pass call so the LLM sees a
focused subset of the radar checklist instead of all 30+ fields at
once. Spec §4.2.

Hand-authored — partitioning is a task-fit decision, not derivable
from json_schema_extra.profile_subgroup. Contract-tested in
tests/unit/test_radar_field_groups_contract.py.
"""

RADAR_FIELD_GROUPS: dict[str, list[str]] = {
    "radar_identity": [
        "system_name",
        "nomenclature",
        "elnot",
        "dieqp",
        "emitter_function",
        "system_status",
        "asrd",
        "responsible_agency",
        "review_cycle",
        "next_review_date",
        "scan_type",
    ],
    "radar_power_rf": [
        "system_name",
        "erp_dbw",
        "tx_peak_power_kw",
        "nominal_rf_mhz",
    ],
    "radar_antenna": [
        "system_name",
        "antenna_photo",
        "gain_dbi",
        "antenna_dim_az_m",
        "antenna_dim_el_m",
        "beamwidth_az_deg",
        "beamwidth_el_deg",
        "spoiled",
        "coverage_limits_el_deg",
    ],
    "radar_timing": [
        "system_name",
        "nominal_pri_usec",
        "nominal_pd_usec",
        "scan_period_sec",
        "dwell_time",
    ],
    "radar_modulation": [
        "system_name",
        "intra_pulse_mop",
        "inter_pulse",
        "frequency_excursion_mhz",
        "num_bits_in_code",
        "pulses_per_dwell",
    ],
}
```

- [ ] **Step 4: Run tests, expect 5 passed.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_radar_field_groups_contract.py -v 2>&1 | tail -10`
Expected: 5 passed. If `test_every_flat_checklist_field_appears_exactly_once` fails, the partitioning is incomplete — add the missing field to the right group.

- [ ] **Step 5: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/_field_groups.py tests/unit/test_radar_field_groups_contract.py
git commit -m "$(cat <<'EOF'
feat(extraction): RADAR_FIELD_GROUPS partitioning + contract tests (spec §4.2)

Single source of truth for which fields each radar sub-pass extracts.
Hand-authored 5-group partition: radar_identity (11 fields),
radar_power_rf (4), radar_antenna (9), radar_timing (5),
radar_modulation (6). Each group includes system_name as the merge
identity.

5 contract assertions enforce: (1) every group has system_name,
(2) every listed field exists on extraction-side RadarSystemEntity,
(3) no non-identity field in multiple groups, (4) every non-system_field
appears in exactly one group, (5) system_field-tagged fields excluded.

No manifest or runtime changes yet — contract tests pass against the
still-active legacy radar_domain pass because they only assert the
partition is well-formed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Create `_radar_shared.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/_radar_shared.py`
- Test: `tests/unit/test_radar_shared.py` *(new)*

- [ ] **Step 1: Write failing tests for the shared helpers.**

Create `tests/unit/test_radar_shared.py`:

```python
"""Phase B Session 1 — _radar_shared utilities (spec §4.3)."""
import pytest


def test_radar_forbidden_system_names_is_frozen_set():
    from ontology_bundles.air_defense_v3.extraction_schemas._radar_shared import (
        RADAR_FORBIDDEN_SYSTEM_NAMES,
    )
    assert isinstance(RADAR_FORBIDDEN_SYSTEM_NAMES, (frozenset, set))
    # Sanity-check a few known entries
    for name in ("SA-2", "PATRIOT", "U-2", "F-16"):
        assert name in {n.upper() for n in RADAR_FORBIDDEN_SYSTEM_NAMES}, (
            f"{name} missing from forbidden set"
        )


def test_radar_optional_text_fields_set_includes_string_fields():
    from ontology_bundles.air_defense_v3.extraction_schemas._radar_shared import (
        RADAR_OPTIONAL_TEXT_FIELDS,
    )
    for name in ("nomenclature", "scan_type", "intra_pulse_mop"):
        assert name in RADAR_OPTIONAL_TEXT_FIELDS, f"{name} missing"


def test_validate_radar_system_name_normalizes_whitespace():
    from ontology_bundles.air_defense_v3.extraction_schemas._radar_shared import (
        validate_radar_system_name,
    )
    assert validate_radar_system_name("  Fan Song  ") == "Fan Song"


def test_validate_radar_system_name_rejects_empty():
    from ontology_bundles.air_defense_v3.extraction_schemas._radar_shared import (
        validate_radar_system_name,
    )
    with pytest.raises(ValueError):
        validate_radar_system_name("")
    with pytest.raises(ValueError):
        validate_radar_system_name("   ")


def test_validate_radar_system_name_does_not_enforce_forbidden():
    """Forbidden enforcement lives in make_root_sanitizer, not here.
    validate_radar_system_name only normalizes and rejects empty."""
    from ontology_bundles.air_defense_v3.extraction_schemas._radar_shared import (
        validate_radar_system_name,
    )
    # SA-2 is forbidden as a radar identity — but this validator doesn't reject it;
    # the root sanitizer does. canonicalize_identity_text() only normalizes
    # whitespace, never case — so "SA-2" round-trips unchanged.
    assert validate_radar_system_name("SA-2") == "SA-2"


def test_make_root_sanitizer_returns_callable():
    from ontology_bundles.air_defense_v3.extraction_schemas._radar_shared import (
        make_root_sanitizer,
    )
    fn = make_root_sanitizer(list_field="radar_systems", optional_text_fields={"nomenclature"})
    assert callable(fn)


def test_make_root_sanitizer_drops_forbidden_identities():
    """The factory must enforce forbidden-name list via sanitize_entity_list."""
    from pydantic import BaseModel, ConfigDict, model_validator
    from ontology_bundles.air_defense_v3.extraction_schemas._radar_shared import (
        make_root_sanitizer,
    )

    class FakeRecord(BaseModel):
        system_name: str

        model_config = ConfigDict(
            extra="ignore",
            ontology_name="RADAR_SYSTEM",
            graph_id_fields=["system_name"],
            is_entity=True,
        )

    class FakePass(BaseModel):
        radar_systems: list[FakeRecord] = []

        model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

        _sanitize = model_validator(mode="before")(
            make_root_sanitizer(list_field="radar_systems", optional_text_fields=set())
        )

    pass_obj = FakePass.model_validate({
        "radar_systems": [
            {"system_name": "Fan Song"},
            {"system_name": "SA-2"},   # forbidden — must be dropped
        ],
    })
    names = [r.system_name for r in pass_obj.radar_systems]
    assert "Fan Song" in names
    assert "SA-2" not in names, f"SA-2 not dropped; got {names}"


def test_make_root_sanitizer_dedupes_by_identity():
    """The factory must run dedupe_entities_by_identity after sanitize."""
    from pydantic import BaseModel, ConfigDict, model_validator
    from ontology_bundles.air_defense_v3.extraction_schemas._radar_shared import (
        make_root_sanitizer,
    )

    class FakeRecord(BaseModel):
        system_name: str

        model_config = ConfigDict(
            extra="ignore",
            ontology_name="RADAR_SYSTEM",
            graph_id_fields=["system_name"],
            is_entity=True,
        )

    class FakePass(BaseModel):
        radar_systems: list[FakeRecord] = []

        model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

        _sanitize = model_validator(mode="before")(
            make_root_sanitizer(list_field="radar_systems", optional_text_fields=set())
        )

    pass_obj = FakePass.model_validate({
        "radar_systems": [
            {"system_name": "Fan Song"},
            {"system_name": "Fan Song"},   # duplicate — must collapse to one
        ],
    })
    assert len(pass_obj.radar_systems) == 1
```

- [ ] **Step 2: Run tests, expect ImportError.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_radar_shared.py -v 2>&1 | tail -5`
Expected: `ImportError`.

- [ ] **Step 3: Create `_radar_shared.py`.**

Write `ontology_bundles/air_defense_v3/extraction_schemas/_radar_shared.py`:

```python
"""Shared helpers for the radar sub-pass modules.

Centralizes the items every radar_* sub-pass uses identically:
- edge field decorator
- forbidden-identity set + optional-text-field set
- system_name normalization validator
- root sanitizer factory (sanitize + dedupe)

Spec §4.3.
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from ..validators import (
    canonicalize_identity_text,
    dedupe_entities_by_identity,
    sanitize_entity_list,
)

# Re-export the edge field decorator from the legacy radar_domain.py
# (kept in source as a legacy reference per spec §6 step 4). The
# decorator is unchanged — just centralized so sub-passes can import
# from one location.
from .radar_domain import edge as edge   # noqa: F401


RADAR_FORBIDDEN_SYSTEM_NAMES: frozenset[str] = frozenset({
    "SA-2", "SA-3", "SA-5", "SA-6", "SA-10", "SA-12", "SA-15", "SA-17",
    "SA-20", "SA-21", "SA-22", "SA-23", "PATRIOT", "PAC-2", "PAC-3",
    "PAC-3 MSE", "HAWK", "NIKE-HERCULES", "S-75", "S-125", "S-200", "S-300",
    "S-350", "S-400", "S-500", "AEGIS BMD", "SM-2", "SM-3", "SM-6", "THAAD",
    "ARROW", "IRON DOME", "DAVID'S SLING", "U-2", "SR-71", "RF-4C", "F-4",
    "F-15", "F-16", "B-52", "MIG-21", "MIG-23", "MIG-29", "SU-27",
})

# Superset across sub-passes; each make_root_sanitizer call passes only
# the subset its record class declares.
RADAR_OPTIONAL_TEXT_FIELDS: frozenset[str] = frozenset({
    "nomenclature",
    "elnot",
    "dieqp",
    "emitter_function",
    "system_status",
    "asrd",
    "responsible_agency",
    "review_cycle",
    "next_review_date",
    "scan_type",
    "intra_pulse_mop",
    "inter_pulse",
    "dwell_time",
})


def validate_radar_system_name(value: Any) -> Any:
    """field_validator("system_name", mode="before") body.

    Scope: normalization + non-empty-identity check only.
    Does NOT enforce the forbidden-names list — that authority lives
    exclusively in make_root_sanitizer / sanitize_entity_list.
    """
    if value is None:
        raise ValueError("system_name is required and cannot be None")
    text = canonicalize_identity_text(value)
    if not text or not text.strip():
        raise ValueError("system_name cannot be empty / whitespace-only")
    return text.strip()


def make_root_sanitizer(
    *,
    list_field: str,
    optional_text_fields: set[str] | frozenset[str],
):
    """Factory returning a model_validator(mode="before") body.

    The returned validator runs BOTH sanitize_entity_list AND
    dedupe_entities_by_identity, mirroring the legacy
    _sanitize_and_dedupe_root_entities body in radar_domain.py.
    Sanitize-only factories silently break duplicate-emission handling.

    Defaults forbidden_identities to RADAR_FORBIDDEN_SYSTEM_NAMES so
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
            forbidden_identities=RADAR_FORBIDDEN_SYSTEM_NAMES,
        )
        return dedupe_entities_by_identity(cls, values)

    return _sanitize_and_dedupe
```

- [ ] **Step 4: Run tests, expect all passed.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_radar_shared.py -v 2>&1 | tail -10`
Expected: exactly 8 passed. The count is deterministic (no parametrization); a partial pass is a regression, not a collection quirk.

- [ ] **Step 5: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/_radar_shared.py tests/unit/test_radar_shared.py
git commit -m "$(cat <<'EOF'
feat(extraction): _radar_shared.py with forbidden set + sanitizer factory (spec §4.3)

Centralizes the items every radar sub-pass uses:
- RADAR_FORBIDDEN_SYSTEM_NAMES (frozen set of 44 forbidden identities)
- RADAR_OPTIONAL_TEXT_FIELDS (superset across sub-passes)
- validate_radar_system_name (normalization + non-empty check, NOT
  forbidden enforcement)
- make_root_sanitizer (factory; runs sanitize_entity_list +
  dedupe_entities_by_identity; defaults forbidden_identities so
  sub-passes don't import the set directly)
- edge (re-exported from legacy radar_domain.py)

Single authority for forbidden-name enforcement: make_root_sanitizer.
Single authority for identity normalization: validate_radar_system_name.
Splitting concerns prevents drift between the two layers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Extract `_numeric_evidence.py` shared helper

**Files:**
- Create: `docker/docling-graph/app/_numeric_evidence.py`
- Modify: `docker/docling-graph/app/provenance.py:329-368` (move helpers, re-export)
- Test: `tests/unit/test_numeric_evidence.py` *(new)*

The unit-aware substring matcher currently lives in `provenance.py::_value_match_candidates` + `_UNIT_HINTS_BY_SUFFIX` + `_normalize_text`. We extract these into `_numeric_evidence.py` with a public predicate `value_is_supported_by_text(value, field_name, evidence_text)` that BOTH `provenance.build_auto_field_evidence` AND the soon-to-be-refactored `_clear_unsupported_radar_properties` consume.

- [ ] **Step 1: Write failing tests.**

Create `tests/unit/test_numeric_evidence.py`:

```python
"""Phase B Session 1 — shared numeric-evidence predicate (spec §4.8).

Both the auto-evidence resolver and the radar-postprocessing
"unsupported numeric clearer" must use the same logic for deciding
whether a numeric value's stringified form (with unit-aware variants)
appears in batch evidence text.
"""
import importlib.util
import pathlib
import sys

_SERVICE_ROOT = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "docker" / "docling-graph" / "app"
)


def _load(modname: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_ne = _load("_dgp_numeric_evidence", _SERVICE_ROOT / "_numeric_evidence.py")
value_is_supported_by_text = _ne.value_is_supported_by_text
value_match_candidates = _ne.value_match_candidates
normalize_text = _ne.normalize_text


def test_normalize_text_collapses_whitespace_and_casefolds():
    assert normalize_text("  Fan   SONG  ") == "fan song"


def test_value_match_candidates_for_float_with_dbi_suffix():
    forms = value_match_candidates(35.0, "gain_dbi")
    norm = [normalize_text(f) for f in forms]
    assert any("35" in n for n in norm)
    assert any("dbi" in n for n in norm)


def test_value_match_candidates_for_int_with_mhz_suffix():
    """Same-unit candidates only — no cross-unit alternates.

    The helper does NOT generate '3000 GHz' as an alternate for '3000 MHz'
    because '3000 GHz' is a different physical magnitude (3 THz vs 3 GHz);
    matching across units without converting the value would silently
    accept wrong values. If you need cross-unit support, convert at the
    LLM-emission step or add explicit unit-conversion logic — do NOT
    paper over it here. Tracked in the plan's "Out of scope" section.
    """
    forms = value_match_candidates(3000, "nominal_rf_mhz")
    norm = [normalize_text(f) for f in forms]
    assert any("3000 mhz" in n for n in norm)
    # Negative assertion: no cross-unit alternates.
    assert not any("3000 ghz" in n for n in norm), (
        "value_match_candidates must not emit cross-unit alternates "
        "without value conversion — see helper docstring."
    )
    assert not any("3000 khz" in n for n in norm)


def test_value_is_supported_by_text_string_field():
    assert value_is_supported_by_text(
        "PHASED-ARRAY", "scan_type",
        "The radar uses a phased-array scan type.",
    )
    assert not value_is_supported_by_text(
        "ELECTRONIC", "scan_type",
        "The radar uses a phased-array scan type.",
    )


def test_value_is_supported_by_text_numeric_with_unit():
    assert value_is_supported_by_text(
        35.0, "gain_dbi", "The antenna gain is 35 dBi nominal.",
    )
    assert value_is_supported_by_text(
        3000.0, "nominal_rf_mhz", "Operates at 3000 MHz.",
    )
    assert value_is_supported_by_text(
        600.0, "tx_peak_power_kw", "Transmitter peak power is 600 kW.",
    )


def test_value_is_supported_by_text_numeric_no_match():
    """Unsupported numeric values return False so the caller can null them."""
    assert not value_is_supported_by_text(
        9999.0, "gain_dbi", "The antenna gain is 35 dBi.",
    )


def test_value_is_supported_by_text_none_value():
    """None values are vacuously supported (the caller's null-check
    happens upstream of this predicate)."""
    assert value_is_supported_by_text(None, "gain_dbi", "any text")
```

- [ ] **Step 2: Run tests, expect ImportError.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_numeric_evidence.py -v 2>&1 | tail -5`
Expected: import-time error (file doesn't exist).

- [ ] **Step 3: Create `_numeric_evidence.py`.**

Move the helpers from `provenance.py` lines 329-368 into the new shared module. Keep names public (no leading underscore) so two consumers can import.

Write `docker/docling-graph/app/_numeric_evidence.py`:

```python
"""Shared numeric-evidence helpers used by:

- provenance.build_auto_field_evidence (post-extraction evidence-row builder)
- evidence_gate._clear_unsupported_radar_properties (numeric-field
  clearing in the radar postprocessor — refactored per spec §4.8 to
  preserve numeric values that appear in batch evidence text)

Both consumers must use the same predicate so a value the resolver
treats as "supported" isn't simultaneously nulled by the postprocessor.

Spec §4.8.
"""
from __future__ import annotations

import re
from typing import Any

_WS_NORM = re.compile(r"\s+")


# Field-name suffix → list of human-readable unit candidates the
# author may have written. Generated value form is paired with each.
_UNIT_HINTS_BY_SUFFIX: dict[str, list[str]] = {
    "_dbi": ["dBi", "dB"],
    "_dbw": ["dBW", "dB"],
    "_mhz": ["MHz", "GHz"],
    "_khz": ["kHz"],
    "_usec": ["μs", "us", "microseconds"],
    "_sec": ["s", "seconds"],
    "_kw": ["kW"],
    "_mw": ["MW"],
    "_km": ["km", "kilometers"],
    "_m": ["m", "meters"],
    "_kg": ["kg", "kilograms"],
    "_mps": ["m/s", "mps"],
    "_deg": ["°", "deg", "degrees"],
}


def normalize_text(text: str) -> str:
    """Whitespace-collapsed casefold for fuzzy substring matching."""
    return _WS_NORM.sub(" ", text or "").strip().casefold()


def _field_unit_suffix(field_name: str) -> str:
    """Return the longest known unit suffix on a field name, or ''."""
    for suffix in sorted(_UNIT_HINTS_BY_SUFFIX, key=len, reverse=True):
        if field_name.endswith(suffix):
            return suffix
    return ""


def value_match_candidates(value: Any, field_name: str) -> list[str]:
    """Generate likely string forms of a field value for substring matching.

    Numeric values get whole-number, decimal, and same-unit-suffix variants
    derived from the field name's suffix convention (e.g. ``gain_dbi`` →
    "35", "35.0", "35 dBi", "35dBi"). String values pass through as-is
    after stripping. Booleans return [] (not useful for substring match).

    IMPORTANT: only same-unit variants are generated — cross-unit
    alternates like "3000 GHz" for value 3000 in field nominal_rf_mhz are
    NEVER emitted, since matching them would silently accept physically
    wrong values. If `_UNIT_HINTS_BY_SUFFIX[<suffix>]` contains a cross-
    unit entry (e.g. "GHz" listed under "_mhz"), audit and remove it
    before this helper is wired into the evidence gate. The contract test
    `test_value_match_candidates_for_int_with_mhz_suffix` enforces this.
    """
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, (int, float)):
        forms: list[str] = [str(value)]
        if isinstance(value, float) and value == int(value):
            forms.append(str(int(value)))
        units = _UNIT_HINTS_BY_SUFFIX.get(_field_unit_suffix(field_name))
        if units:
            base = forms[-1]
            forms.extend(f"{base} {u}" for u in units)
            forms.extend(f"{base}{u}" for u in units)
        return forms
    return [str(value)]


def value_is_supported_by_text(
    value: Any, field_name: str, evidence_text: str,
) -> bool:
    """Return True iff *value*'s stringified form appears in *evidence_text*.

    "Stringified form" includes the field's expected unit suffix appended
    to the value (e.g. "35 dBi" for value 35.0 in field gain_dbi). It does
    NOT include cross-unit converted forms — "35 dBi" is generated, but
    "37.15 dBd" is NOT. Same-magnitude-different-unit matches like
    "3000 GHz" for value 3000 in field nominal_rf_mhz are explicitly
    rejected (they would be physically wrong: 3000 GHz != 3000 MHz).

    Cross-unit conversion (1.5 tonnes <-> 1500 kg, 43 km <-> 43000 m, etc.)
    is OUT OF SCOPE for Session 1. If a doc states a value in a non-canonical
    unit and the LLM doesn't normalize, this predicate returns False and the
    caller will null the value. Tracked as Session 2 follow-up if false-
    negatives become a real problem in production.

    Whitespace is collapsed and case is folded before comparison.
    None / empty-string values are vacuously supported — the caller's
    null-check happens upstream of this predicate.
    """
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    candidates = value_match_candidates(value, field_name)
    if not candidates:
        return False
    et_norm = normalize_text(evidence_text or "")
    if not et_norm:
        return False
    return any(normalize_text(c) in et_norm for c in candidates)
```

- [ ] **Step 4: Update `provenance.py` to import from the shared module.**

Modify `docker/docling-graph/app/provenance.py`:
- Replace local `_normalize_text`, `_UNIT_HINTS_BY_SUFFIX`, `_value_match_candidates`, `_field_unit_suffix` (lines ~327-368) with imports from `_numeric_evidence`.
- Keep `build_auto_field_evidence` body unchanged; just reference the imported helpers.

```python
# At the top of provenance.py, add:
from app._numeric_evidence import (
    normalize_text as _normalize_text,
    value_match_candidates as _value_match_candidates,
)

# Delete the local _WS_NORM, _UNIT_HINTS_BY_SUFFIX, _field_unit_suffix,
# _normalize_text, _value_match_candidates definitions. Keep the rest
# of build_auto_field_evidence unchanged.
```

- [ ] **Step 4b: Update `tests/unit/test_auto_field_evidence.py` to register `_numeric_evidence` in sys.modules before loading provenance.**

The existing test loads `provenance.py` by file path (lines 18, 27 — `_load("_dgp_provenance", _SERVICE_ROOT / "provenance.py")`). It does not establish `docker/docling-graph/app` as the importable package `app`. After Step 4, `provenance.py` does `from app._numeric_evidence import ...`, which will resolve to the repo-root `app/` package (the worker code) and ImportError, OR shadow the wrong module.

Fix: load `_numeric_evidence` first via the same file-path loader and inject it under the `app._numeric_evidence` key in `sys.modules` so the subsequent `provenance.py` `from app._numeric_evidence import ...` resolves to it.

Edit `tests/unit/test_auto_field_evidence.py` between lines 27 and 28 (before the `_provenance = _load(...)` call):

```python
# Pre-register the docling-graph _numeric_evidence module under the
# 'app._numeric_evidence' key so the file-path-loaded provenance.py's
# `from app._numeric_evidence import ...` resolves to the docling-graph
# version, not the unrelated repo-root `app/` package.
_numeric_evidence = _load(
    "app._numeric_evidence", _SERVICE_ROOT / "_numeric_evidence.py"
)
sys.modules["app._numeric_evidence"] = _numeric_evidence

_provenance = _load("_dgp_provenance", _SERVICE_ROOT / "provenance.py")
```

(`_load` already inserts into `sys.modules` under the name passed as `modname` — see line 22. Passing `"app._numeric_evidence"` as the modname does both the load and the registration in one call. The explicit second-line `sys.modules[...]` assignment is redundant but defensive against future `_load` refactors.)

- [ ] **Step 4c: Run the existing auto-evidence test, expect green.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_auto_field_evidence.py -v 2>&1 | tail -10`
Expected: same number of passed tests as the P1 baseline. If ImportError fires, the `app._numeric_evidence` registration is in the wrong order or missing.

- [ ] **Step 5: Verify no other module imports the deleted private names.**

Run:
```bash
grep -rn -E "_WS_NORM|_UNIT_HINTS_BY_SUFFIX|_field_unit_suffix" \
  docker/docling-graph/ tests/ ontology_bundles/ 2>/dev/null \
  | grep -v "_numeric_evidence.py" \
  | grep -v "test_numeric_evidence.py" \
  | grep -v "/__pycache__/"
```
Expected: empty output. If any module still references the deleted private names from `provenance.py`, fix the import before continuing — silent ImportErrors only surface at runtime.

Also confirm `_normalize_text` and `_value_match_candidates` are only referenced via the new import path:
```bash
grep -rn "_normalize_text\|_value_match_candidates" docker/docling-graph/ tests/ 2>/dev/null \
  | grep -v "/__pycache__/"
```
Expected: only `provenance.py` (the new alias-import) and any test that exercises the helpers directly.

- [ ] **Step 6: Run targeted + radar-adjacent tests, expect green.**

Run:
```bash
SKIP_COV=1 .venv/bin/pytest \
  tests/unit/test_numeric_evidence.py \
  tests/unit/test_auto_field_evidence.py \
  tests/unit/test_field_provenance_resolver.py \
  -v 2>&1 | tail -10
```
Expected: All passed. The auto-evidence tests still pass because we only refactored the helpers; the public API of `build_auto_field_evidence` is unchanged.

Then sweep the broader radar/evidence path:
```bash
SKIP_COV=1 .venv/bin/pytest tests/unit -k "radar or provenance or evidence" -q 2>&1 | tail -5
```
Expected: All passed (any pre-existing xfail is fine; no new failures).

- [ ] **Step 7: Commit.**

```bash
git add docker/docling-graph/app/_numeric_evidence.py docker/docling-graph/app/provenance.py tests/unit/test_numeric_evidence.py
git commit -m "$(cat <<'EOF'
refactor(docling-graph): extract _numeric_evidence.py shared helper (spec §4.8)

The unit-aware substring matcher (normalize_text, _UNIT_HINTS_BY_SUFFIX,
value_match_candidates) used to live in provenance.py. Moving it to a
new shared module so two consumers can use the same predicate:

1. provenance.build_auto_field_evidence (existing — emits evidence rows
   for fields whose values appear in chunks)
2. evidence_gate._clear_unsupported_radar_properties (next task — will
   stop nulling numeric fields when their values appear in evidence_text)

Public API: value_is_supported_by_text(value, field_name, evidence_text)
returns True iff the value's stringified form (or a unit-aware variant)
appears as a substring of evidence_text. Plus value_match_candidates
and normalize_text for callers that need finer control.

provenance.py imports the helpers from the shared module; behavior is
identical — refactor only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 2: Sub-pass modules

Tasks 4-9 add the 5 sub-pass Pydantic modules + a description-quality contract test parametrized over them. Each module is ~80–130 lines following the spec §4.4 template. After Chunk 2, all 5 modules are importable but not yet referenced from the manifest — `radar_domain` is still the active radar pass.

**Auto-evidence is wired generically — no per-sub-pass hook needed.** Per spec §4.7, `build_auto_field_evidence` runs per-pass automatically after every LLM call. Each sub-pass produces evidence rows for the fields it extracts; the worker-side merger aggregates `_field_evidence` across passes by `(instance_id, field_name)`. The numeric sub-passes (`radar_power_rf`, `radar_antenna`, `radar_timing`, `radar_modulation`) inherit this hook by virtue of being normal pass-templates with `is_entity=True` records — they need no extra wiring. If a numeric value disappears post-cutover, the cause is `_clear_unsupported_radar_properties` (Task 10), not missing auto-evidence wiring.

### Task 4: Create `radar_identity.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/radar_identity.py`

- [ ] **Step 1: Write the module.**

```python
"""radar_identity extraction pass — radar identity + administrative metadata.

Spec §4.4. One of 5 sub-passes splitting the legacy radar_domain into
smaller LLM call boundaries. Emits RADAR_SYSTEM[] with system_name as
the merge identity; merge_and_resolve collapses partial records from
sibling sub-passes onto one vertex.

Group fields: system_name, nomenclature, elnot, dieqp, emitter_function,
system_status, asrd, responsible_agency, review_cycle, next_review_date,
scan_type.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_text
from ._field_groups import RADAR_FIELD_GROUPS
from ._radar_shared import edge, make_root_sanitizer, validate_radar_system_name

_GROUP_NAME = "radar_identity"
_FIELDS = RADAR_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class RadarIdentityRecord(BaseModel):
    """Subset of RadarSystemEntity covering identity + admin fields."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the RADAR. Accept proper-noun "
            "radar names from prose (e.g. 'Fan Song', 'Spoon Rest', "
            "'Tombstone', 'AN/MPQ-65'). Never emit weapon, missile, "
            "aircraft, or platform names — those are filtered "
            "deterministically by the root sanitizer."
        ),
        examples=["Fan Song", "AN/MPQ-65"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description=(
            "Official military nomenclature — formal alphanumeric "
            "designation (JETDS / AN-style for US, GRAU index for "
            "Russian/Soviet). Distinct from system_name."
        ),
    )
    elnot: Optional[str] = Field(
        default=None,
        description=(
            "ELINT Notation — community-unique alphabetic code from "
            "intelligence databases. Emit verbatim; do not infer."
        ),
    )
    dieqp: Optional[str] = Field(
        default=None,
        description=(
            "Digital Intelligence Equipment Parameters identifier. "
            "Emit verbatim; do not infer."
        ),
    )
    emitter_function: Optional[str] = Field(
        default=None,
        description=(
            "Operational role: SEARCH, TRACKING, FIRE_CONTROL, "
            "MULTI_FUNCTION, HEIGHT_FINDER, NAV. Emit only when the "
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
    scan_type: Optional[str] = Field(
        default=None,
        description=(
            "How the beam is steered: CIRCULAR, SECTOR, RASTER, "
            "ELECTRONIC, DWELL_AND_SWITCH, HELICAL. Emit as uppercase."
        ),
    )

    _v_system_name        = field_validator("system_name", mode="before")(validate_radar_system_name)
    _v_nomenclature       = field_validator("nomenclature", mode="before")(coerce_optional_text)
    _v_elnot              = field_validator("elnot", mode="before")(coerce_optional_text)
    _v_dieqp              = field_validator("dieqp", mode="before")(coerce_optional_text)
    _v_emitter_function   = field_validator("emitter_function", mode="before")(coerce_optional_text)
    _v_system_status      = field_validator("system_status", mode="before")(coerce_optional_text)
    _v_asrd               = field_validator("asrd", mode="before")(coerce_optional_text)
    _v_responsible_agency = field_validator("responsible_agency", mode="before")(coerce_optional_text)
    _v_review_cycle       = field_validator("review_cycle", mode="before")(coerce_optional_text)
    _v_next_review_date   = field_validator("next_review_date", mode="before")(coerce_optional_text)
    _v_scan_type          = field_validator("scan_type", mode="before")(coerce_optional_text)


class RadarIdentityPass(BaseModel):
    """Pass-root template — wraps radar_systems list."""

    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    radar_systems: List[RadarIdentityRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level radar systems with identity + administrative "
            "metadata extracted from this batch."
        ),
        examples=[["Fan Song", "Spoon Rest"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_root_sanitizer(
            list_field="radar_systems",
            optional_text_fields={
                "nomenclature", "elnot", "dieqp", "emitter_function",
                "system_status", "asrd", "responsible_agency",
                "review_cycle", "next_review_date", "scan_type",
            },
        )
    )
```

- [ ] **Step 2: Verify importable.**

Run: `.venv/bin/python -c "from ontology_bundles.air_defense_v3.extraction_schemas.radar_identity import RadarIdentityPass; print('OK', len(RadarIdentityPass.model_fields))"`
Expected: `OK 1` (just the `radar_systems` list field on the pass-root).

- [ ] **Step 3: Verify model_validate accepts a Fan Song record.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.radar_identity import RadarIdentityPass
inst = RadarIdentityPass.model_validate({
    'radar_systems': [
        {'system_name': 'Fan Song', 'nomenclature': '5N62E', 'emitter_function': 'FIRE_CONTROL'}
    ]
})
print(inst.radar_systems[0].system_name, inst.radar_systems[0].nomenclature)
"
```
Expected: `Fan Song 5N62E`.

- [ ] **Step 4: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/radar_identity.py
git commit -m "$(cat <<'EOF'
feat(extraction): radar_identity sub-pass (11 fields, spec §4.4)

First of 5 radar sub-passes splitting the legacy radar_domain into
focused LLM-call boundaries. Emits RADAR_SYSTEM[] with system_name as
merge identity. Group fields: system_name, nomenclature, elnot, dieqp,
emitter_function, system_status, asrd, responsible_agency, review_cycle,
next_review_date, scan_type.

Descriptions are sanitized at copy time per spec §4.4: FORBIDDEN-values
block stripped from system_name (forbidden enforcement is delegated to
make_root_sanitizer); typical-value-range prose dropped.

Module is importable but not yet referenced from manifest — radar_domain
is still the active pass until Chunk 4's cutover commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Create `radar_power_rf.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/radar_power_rf.py`

- [ ] **Step 1: Write the module.**

```python
"""radar_power_rf extraction pass — RF carrier + transmit power.

Spec §4.4. Group fields: system_name, erp_dbw, tx_peak_power_kw,
nominal_rf_mhz.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float
from ._field_groups import RADAR_FIELD_GROUPS
from ._radar_shared import edge, make_root_sanitizer, validate_radar_system_name

_GROUP_NAME = "radar_power_rf"
_FIELDS = RADAR_FIELD_GROUPS[_GROUP_NAME]


class RadarPowerRfRecord(BaseModel):
    """Subset of RadarSystemEntity covering RF carrier + transmit power."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the RADAR. Accept proper-noun "
            "radar names. Never emit weapon, missile, aircraft, or "
            "platform names — those are filtered deterministically."
        ),
        examples=["Fan Song", "AN/MPQ-65"],
    )
    erp_dbw: Optional[float] = Field(
        default=None,
        description=(
            "Effective Radiated Power in dBW. Emit only when the source "
            "states the value with units; otherwise null. See Unit "
            "Policy in DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    tx_peak_power_kw: Optional[float] = Field(
        default=None,
        description=(
            "Transmitter peak power in kilowatts. Emit only when the "
            "source states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    nominal_rf_mhz: Optional[float] = Field(
        default=None,
        description=(
            "Nominal carrier frequency in MHz. Emit only when the source "
            "states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )

    _v_system_name      = field_validator("system_name", mode="before")(validate_radar_system_name)
    _v_erp_dbw          = field_validator("erp_dbw", mode="before")(coerce_optional_float)
    _v_tx_peak_power_kw = field_validator("tx_peak_power_kw", mode="before")(coerce_optional_float)
    _v_nominal_rf_mhz   = field_validator("nominal_rf_mhz", mode="before")(coerce_optional_float)


class RadarPowerRfPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    radar_systems: List[RadarPowerRfRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level radar systems with RF carrier + transmit power "
            "values extracted from this batch."
        ),
        examples=[["Fan Song"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_root_sanitizer(
            list_field="radar_systems",
            optional_text_fields=set(),
        )
    )
```

- [ ] **Step 2: Verify importable.**

Run: `.venv/bin/python -c "from ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf import RadarPowerRfPass, RadarPowerRfRecord; print('OK', sorted(RadarPowerRfRecord.model_fields.keys()))"`
Expected: `OK ['erp_dbw', 'nominal_rf_mhz', 'system_name', 'tx_peak_power_kw']`.

- [ ] **Step 3: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/radar_power_rf.py
git commit -m "feat(extraction): radar_power_rf sub-pass (4 fields)

system_name + erp_dbw + tx_peak_power_kw + nominal_rf_mhz. Numeric
field descriptions reference the Unit Policy block in
DELTA_SYSTEM_PROMPT for conversions; no inline conversion rules per
spec §4.4 sanitization rule (b).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Create `radar_antenna.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/radar_antenna.py`

- [ ] **Step 1: Write the module.**

```python
"""radar_antenna extraction pass — antenna geometry + beam parameters.

Spec §4.4. Group fields: system_name, antenna_photo, gain_dbi,
antenna_dim_az_m, antenna_dim_el_m, beamwidth_az_deg, beamwidth_el_deg,
spoiled, coverage_limits_el_deg.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float
from ._field_groups import RADAR_FIELD_GROUPS
from ._radar_shared import edge, make_root_sanitizer, validate_radar_system_name

_GROUP_NAME = "radar_antenna"
_FIELDS = RADAR_FIELD_GROUPS[_GROUP_NAME]


class RadarAntennaRecord(BaseModel):
    """Subset of RadarSystemEntity covering antenna geometry + beam shape."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the RADAR. Accept proper-noun "
            "radar names. Never emit weapon, missile, aircraft, or "
            "platform names — those are filtered deterministically."
        ),
        examples=["Fan Song", "AN/MPQ-65"],
    )
    # Optional[bool] fields rely on Pydantic's native bool parsing; no
    # coerce_optional_bool helper exists in ..validators and we don't add
    # one in Session 1 (would be scope creep). Pydantic accepts JSON
    # true/false/null natively.
    antenna_photo: Optional[bool] = Field(
        default=None,
        description=(
            "Whether an antenna photograph is included in the record. "
            "Use null when not stated."
        ),
    )
    gain_dbi: Optional[float] = Field(
        default=None,
        description=(
            "Peak antenna gain in dBi. Emit only when the source states "
            "value AND unit. See Unit Policy in DELTA_SYSTEM_PROMPT for "
            "dB-domain conversions."
        ),
    )
    antenna_dim_az_m: Optional[float] = Field(
        default=None,
        description=(
            "Antenna aperture width (azimuth dimension) in meters. Emit "
            "only when the source states value AND unit."
        ),
    )
    antenna_dim_el_m: Optional[float] = Field(
        default=None,
        description=(
            "Antenna aperture height (elevation dimension) in meters. "
            "Emit only when the source states value AND unit."
        ),
    )
    beamwidth_az_deg: Optional[float] = Field(
        default=None,
        description=(
            "Main-beam 3 dB azimuth beamwidth in degrees. Emit only when "
            "the source states value AND unit."
        ),
    )
    beamwidth_el_deg: Optional[float] = Field(
        default=None,
        description=(
            "Main-beam 3 dB elevation beamwidth in degrees. Emit only "
            "when the source states value AND unit."
        ),
    )
    spoiled: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the beam is spoiled (deliberately broadened). Use "
            "null when not stated."
        ),
    )
    coverage_limits_el_deg: Optional[float] = Field(
        default=None,
        description=(
            "Maximum elevation coverage angle in degrees. Emit only when "
            "the source states value AND unit."
        ),
    )

    _v_system_name             = field_validator("system_name", mode="before")(validate_radar_system_name)
    _v_gain_dbi                = field_validator("gain_dbi", mode="before")(coerce_optional_float)
    _v_antenna_dim_az_m        = field_validator("antenna_dim_az_m", mode="before")(coerce_optional_float)
    _v_antenna_dim_el_m        = field_validator("antenna_dim_el_m", mode="before")(coerce_optional_float)
    _v_beamwidth_az_deg        = field_validator("beamwidth_az_deg", mode="before")(coerce_optional_float)
    _v_beamwidth_el_deg        = field_validator("beamwidth_el_deg", mode="before")(coerce_optional_float)
    _v_coverage_limits_el_deg  = field_validator("coverage_limits_el_deg", mode="before")(coerce_optional_float)
    # antenna_photo, spoiled: no validator — Pydantic native bool parsing.


class RadarAntennaPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    radar_systems: List[RadarAntennaRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level radar systems with antenna geometry + beam "
            "parameters extracted from this batch."
        ),
        examples=[["Fan Song"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_root_sanitizer(
            list_field="radar_systems",
            optional_text_fields=set(),  # no text fields in this group beyond system_name
        )
    )
```

- [ ] **Step 2: Verify importable.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.radar_antenna import (
    RadarAntennaPass, RadarAntennaRecord
)
print('OK', sorted(RadarAntennaRecord.model_fields.keys()))
"
```
Expected: `OK ['antenna_dim_az_m', 'antenna_dim_el_m', 'antenna_photo', 'beamwidth_az_deg', 'beamwidth_el_deg', 'coverage_limits_el_deg', 'gain_dbi', 'spoiled', 'system_name']`.

- [ ] **Step 3: Verify Optional[bool] parses natively.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.radar_antenna import RadarAntennaPass
inst = RadarAntennaPass.model_validate({
    'radar_systems': [
        {'system_name': 'Fan Song', 'gain_dbi': 35.0, 'antenna_photo': False, 'spoiled': True}
    ]
})
r = inst.radar_systems[0]
print(r.system_name, r.gain_dbi, r.antenna_photo, r.spoiled)
"
```
Expected: `Fan Song 35.0 False True`. Confirms Pydantic handles `Optional[bool]` without an explicit coercer.

- [ ] **Step 4: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/radar_antenna.py
git commit -m "feat(extraction): radar_antenna sub-pass (9 fields)

system_name + antenna_photo + gain_dbi + antenna_dim_{az,el}_m +
beamwidth_{az,el}_deg + spoiled + coverage_limits_el_deg.

Optional[bool] fields use Pydantic native parsing (no coerce_optional_bool
helper added — would be scope creep). Float fields use existing
coerce_optional_float from ..validators. Numeric descriptions reference
the DELTA_SYSTEM_PROMPT Unit Policy block; no inline conversion rules
per spec §4.4 sanitization rule (b).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Create `radar_timing.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/radar_timing.py`

- [ ] **Step 1: Write the module.**

```python
"""radar_timing extraction pass — pulse + scan timing.

Spec §4.4. Group fields: system_name, nominal_pri_usec, nominal_pd_usec,
scan_period_sec, dwell_time.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float, coerce_optional_text
from ._field_groups import RADAR_FIELD_GROUPS
from ._radar_shared import edge, make_root_sanitizer, validate_radar_system_name

_GROUP_NAME = "radar_timing"
_FIELDS = RADAR_FIELD_GROUPS[_GROUP_NAME]


class RadarTimingRecord(BaseModel):
    """Subset of RadarSystemEntity covering pulse + scan timing."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the RADAR. Accept proper-noun "
            "radar names. Never emit weapon, missile, aircraft, or "
            "platform names — those are filtered deterministically."
        ),
        examples=["Fan Song", "AN/MPQ-65"],
    )
    nominal_pri_usec: Optional[float] = Field(
        default=None,
        description=(
            "Nominal Pulse Repetition Interval in microseconds. Emit "
            "only when the source states value AND unit. See Unit Policy "
            "in DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    nominal_pd_usec: Optional[float] = Field(
        default=None,
        description=(
            "Nominal Pulse Duration in microseconds. Emit only when the "
            "source states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    scan_period_sec: Optional[float] = Field(
        default=None,
        description=(
            "Time to complete one full scan in seconds. Emit only when "
            "the source states value AND unit."
        ),
    )
    dwell_time: Optional[str] = Field(
        default=None,
        description=(
            "Time spent at a single beam position. Free-text; emit "
            "verbatim from the source."
        ),
    )

    _v_system_name      = field_validator("system_name", mode="before")(validate_radar_system_name)
    _v_nominal_pri_usec = field_validator("nominal_pri_usec", mode="before")(coerce_optional_float)
    _v_nominal_pd_usec  = field_validator("nominal_pd_usec", mode="before")(coerce_optional_float)
    _v_scan_period_sec  = field_validator("scan_period_sec", mode="before")(coerce_optional_float)
    _v_dwell_time       = field_validator("dwell_time", mode="before")(coerce_optional_text)


class RadarTimingPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    radar_systems: List[RadarTimingRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level radar systems with pulse + scan timing values "
            "extracted from this batch."
        ),
        examples=[["Fan Song"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_root_sanitizer(
            list_field="radar_systems",
            optional_text_fields={"dwell_time"},
        )
    )
```

- [ ] **Step 2: Verify importable.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.radar_timing import (
    RadarTimingPass, RadarTimingRecord
)
print('OK', sorted(RadarTimingRecord.model_fields.keys()))
"
```
Expected: `OK ['dwell_time', 'nominal_pd_usec', 'nominal_pri_usec', 'scan_period_sec', 'system_name']`.

- [ ] **Step 3: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/radar_timing.py
git commit -m "feat(extraction): radar_timing sub-pass (5 fields)

system_name + nominal_pri_usec + nominal_pd_usec + scan_period_sec +
dwell_time. Numeric descriptions reference DELTA_SYSTEM_PROMPT Unit
Policy; no inline conversion rules per spec §4.4 sanitization rule (b).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Create `radar_modulation.py`

**Files:**
- Create: `ontology_bundles/air_defense_v3/extraction_schemas/radar_modulation.py`

- [ ] **Step 1: Write the module.**

```python
"""radar_modulation extraction pass — pulse modulation + coding.

Spec §4.4. Group fields: system_name, intra_pulse_mop, inter_pulse,
frequency_excursion_mhz, num_bits_in_code, pulses_per_dwell.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float, coerce_optional_int, coerce_optional_text
from ._field_groups import RADAR_FIELD_GROUPS
from ._radar_shared import edge, make_root_sanitizer, validate_radar_system_name

_GROUP_NAME = "radar_modulation"
_FIELDS = RADAR_FIELD_GROUPS[_GROUP_NAME]


class RadarModulationRecord(BaseModel):
    """Subset of RadarSystemEntity covering pulse modulation + coding."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="RADAR_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the RADAR. Accept proper-noun "
            "radar names. Never emit weapon, missile, aircraft, or "
            "platform names — those are filtered deterministically."
        ),
        examples=["Fan Song", "AN/MPQ-65"],
    )
    intra_pulse_mop: Optional[str] = Field(
        default=None,
        description=(
            "Intra-pulse modulation type. Accept one of: CW, LFM_CHIRP, "
            "NLFM, BARKER_CODE, POLYPHASE, BIPHASE. Emit as uppercase."
        ),
    )
    inter_pulse: Optional[str] = Field(
        default=None,
        description=(
            "Inter-pulse modulation type. Accept one of: CONSTANT_PRI, "
            "PRI_STAGGER, PRI_JITTER, FREQ_AGILE. Emit as uppercase."
        ),
    )
    frequency_excursion_mhz: Optional[float] = Field(
        default=None,
        description=(
            "Frequency excursion (chirp bandwidth) in MHz. Emit only "
            "when the source states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    num_bits_in_code: Optional[int] = Field(
        default=None,
        description=(
            "Number of chips in the phase-code sequence. Integer count; "
            "emit only when the source states it."
        ),
    )
    pulses_per_dwell: Optional[int] = Field(
        default=None,
        description=(
            "Pulses integrated per beam-position dwell. Integer count; "
            "emit only when the source states it."
        ),
    )

    _v_system_name              = field_validator("system_name", mode="before")(validate_radar_system_name)
    _v_intra_pulse_mop          = field_validator("intra_pulse_mop", mode="before")(coerce_optional_text)
    _v_inter_pulse              = field_validator("inter_pulse", mode="before")(coerce_optional_text)
    _v_frequency_excursion_mhz  = field_validator("frequency_excursion_mhz", mode="before")(coerce_optional_float)
    _v_num_bits_in_code         = field_validator("num_bits_in_code", mode="before")(coerce_optional_int)
    _v_pulses_per_dwell         = field_validator("pulses_per_dwell", mode="before")(coerce_optional_int)


class RadarModulationPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    radar_systems: List[RadarModulationRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level radar systems with pulse modulation + coding "
            "values extracted from this batch."
        ),
        examples=[["Fan Song"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_root_sanitizer(
            list_field="radar_systems",
            optional_text_fields={"intra_pulse_mop", "inter_pulse"},
        )
    )
```

- [ ] **Step 2: Verify importable.**

Run:
```bash
.venv/bin/python -c "
from ontology_bundles.air_defense_v3.extraction_schemas.radar_modulation import (
    RadarModulationPass, RadarModulationRecord
)
print('OK', sorted(RadarModulationRecord.model_fields.keys()))
"
```
Expected: `OK ['frequency_excursion_mhz', 'inter_pulse', 'intra_pulse_mop', 'num_bits_in_code', 'pulses_per_dwell', 'system_name']`.

- [ ] **Step 3: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/radar_modulation.py
git commit -m "feat(extraction): radar_modulation sub-pass (6 fields)

system_name + intra_pulse_mop + inter_pulse + frequency_excursion_mhz +
num_bits_in_code + pulses_per_dwell. Float/int fields use existing
coerce_optional_float / coerce_optional_int from ..validators. Numeric
description references DELTA_SYSTEM_PROMPT Unit Policy; no inline
conversion rules per spec §4.4 sanitization rule (b).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Description-quality contract test

**Files:**
- Modify: `tests/unit/test_radar_field_groups_contract.py` (append parametrized test)

- [ ] **Step 1: Append the description-quality test.**

Append to `tests/unit/test_radar_field_groups_contract.py`. The file already imports `pytest` from the Chunk 1 RADAR_FIELD_GROUPS tests, so no new import needed for the parametrize decorator.

```python
from typing import get_args, get_origin

from ontology_bundles.air_defense_v3.extraction_schemas import (
    radar_identity, radar_power_rf, radar_antenna, radar_timing, radar_modulation,
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
        and c.model_config.get("ontology_name") == "RADAR_SYSTEM"
    )


# Tokens that should NEVER appear in any sub-pass description. Catches
# verbatim FORBIDDEN-block leakage even when the canonical text doesn't
# carry the literal "forbidden values" header.
_FORBIDDEN_NAME_TOKENS = (
    "missile", "weapon", "aircraft", "platform",
    "sa-2", "sa-5", "fragment", "bomber",
)


@pytest.mark.parametrize("module", [
    radar_identity, radar_power_rf, radar_antenna, radar_timing, radar_modulation,
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
        for banned in ("typical", "common radar bands", "forbidden values"):
            assert banned not in lower, (
                f"{record_cls.__name__}.{fname}: description contains "
                f"{banned!r} — strip per spec §4.4 sanitization"
            )


@pytest.mark.parametrize("module", [
    radar_identity, radar_power_rf, radar_antenna, radar_timing, radar_modulation,
])
def test_system_name_description_excludes_forbidden_tokens(module):
    """Catch verbatim FORBIDDEN-block leakage on the identity field.

    The legitimate system_name description tells the LLM never to emit
    weapon/missile/aircraft/platform names; that single instructive
    sentence is allowed. What's NOT allowed is leaking the FORBIDDEN
    list itself (e.g. an enumerated dump of "SA-2, SA-5, ..."). We
    guard against that by checking for forbidden-name *tokens* outside
    the one whitelisted instructive sentence.
    """
    record_cls = _record_class(module)
    desc = (record_cls.model_fields["system_name"].description or "").lower()

    # Strip the one whitelisted instructive sentence so its mention of
    # "weapon, missile, aircraft, platform" doesn't trip the check.
    whitelisted = "never emit weapon, missile, aircraft, or platform names"
    cleaned = desc.replace(whitelisted, "")

    for token in _FORBIDDEN_NAME_TOKENS:
        assert token not in cleaned, (
            f"{record_cls.__name__}.system_name description leaked "
            f"forbidden-name token {token!r} outside the whitelisted "
            f"instructive sentence — strip the FORBIDDEN-values block "
            f"per spec §4.4 sanitization rule (a)."
        )
```

- [ ] **Step 2: Run tests.**

Run:
```bash
SKIP_COV=1 .venv/bin/pytest tests/unit/test_radar_field_groups_contract.py -v 2>&1 | tail -20
```
Expected: original Chunk-1 tests (count established by Chunk 1 review — confirm before flagging a regression) + 5 parametrized `test_record_descriptions_well_formed` + 5 parametrized `test_system_name_description_excludes_forbidden_tokens` = 10 new tests, all passed.

If `test_record_descriptions_well_formed` fires, fix the offending field's description in the relevant sub-pass module (strip "Typical X-band ground radars: …" etc., remove numeric examples). If `test_system_name_description_excludes_forbidden_tokens` fires, you have FORBIDDEN-block leakage outside the whitelisted instructive sentence — strip the offending text per spec §4.4 sanitization rule (a).

- [ ] **Step 3: Commit.**

```bash
git add tests/unit/test_radar_field_groups_contract.py
git commit -m "$(cat <<'EOF'
test(extraction): description-quality contract for 5 radar sub-passes (spec §5.1)

Two parametrized tests across the 5 sub-pass record classes:

1. test_record_descriptions_well_formed
   - every field has a non-empty description
   - numeric-typed fields (recursive get_origin/get_args) carry no
     examples (numeric examples confuse gemma4 per Phase A diagnosis)
   - descriptions don't contain "typical", "common radar bands", or
     "forbidden values" — sanitization markers from spec §4.4

2. test_system_name_description_excludes_forbidden_tokens
   - catches verbatim FORBIDDEN-block leakage even when the canonical
     text doesn't carry the literal "forbidden values" header
   - whitelists the one instructive sentence that legitimately mentions
     weapon/missile/aircraft/platform; rejects any other appearance of
     forbidden-name tokens (sa-2, sa-5, fragment, bomber, etc.)

Lighter than full description-drift tracking; catches the actual
failure modes from Phase A.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Chunk 3: Cutover prep (additive code changes)

Tasks 10-16 land additive changes that keep `radar_domain` working while preparing the codebase for the manifest cutover. Each task ships independently with main green.

### Task 10: Refactor `_clear_unsupported_radar_properties` to use shared helper

**Files:**
- Modify: `docker/docling-graph/app/evidence_gate.py:398-444`

This is the **CORRECTNESS BLOCKER from spec §4.8.** Currently the function unconditionally nulls 18 numeric fields on every radar pass output, even when the values are evidenced in batch text. Refactor to call `value_is_supported_by_text` from the shared helper.

- [ ] **Step 1: Write a regression test for the new behavior.**

Create a new dedicated test file `docker/docling-graph/tests/test_clear_unsupported_radar_properties.py` (do NOT add to `test_service_identity_gate.py` — keep evidence-gate numeric clearing isolated so the test runs independently and the failure surface is unambiguous):

```python
"""Regression test for _clear_unsupported_radar_properties (spec §4.8).

The function was unconditionally nulling 18 numeric fields. After the
refactor, numeric values that appear in evidence_text (with unit-aware
variants) are preserved; only unsupported values are nulled.
"""
import importlib.util
import pathlib
import sys

_SERVICE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "app"


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_eg = _load("_dgp_evidence_gate", _SERVICE_ROOT / "evidence_gate.py")
_clear = _eg._clear_unsupported_radar_properties


def test_supported_numeric_is_preserved():
    item = {"system_name": "Fan Song", "gain_dbi": 35.0}
    evidence = "The Fan Song antenna gain is 35 dBi nominal."
    cleared = _clear(item, evidence)
    assert "gain_dbi" not in cleared, f"gain_dbi should be preserved; cleared={cleared}"
    assert item["gain_dbi"] == 35.0


def test_unsupported_numeric_is_nulled():
    item = {"system_name": "Fan Song", "gain_dbi": 9999.0}
    evidence = "The Fan Song antenna gain is 35 dBi nominal."
    cleared = _clear(item, evidence)
    assert "gain_dbi" in cleared
    assert item["gain_dbi"] is None


def test_supported_value_with_unit_conversion():
    item = {"system_name": "Fan Song", "nominal_rf_mhz": 3000.0}
    # Source says "3 GHz" but the field is in MHz.
    # Unit-aware variants generate "3000 GHz" — won't match. But
    # "3000 MHz" matches the value 3000.0's primary form.
    evidence = "Fan Song operates at 3000 MHz."
    cleared = _clear(item, evidence)
    assert "nominal_rf_mhz" not in cleared


def test_text_field_branch_unchanged():
    """Text-field clearing (lines 401-417 in original) must still work."""
    item = {"system_name": "Fan Song", "nomenclature": "5N62E"}
    evidence = "Fan Song E (5N62E) — Soviet Fire-Control Radar"
    cleared = _clear(item, evidence)
    assert "nomenclature" not in cleared
    assert item["nomenclature"] == "5N62E"


def test_text_field_unsupported_still_nulled():
    item = {"system_name": "Fan Song", "nomenclature": "ABC-999"}
    evidence = "Fan Song operates at 3000 MHz."
    cleared = _clear(item, evidence)
    assert "nomenclature" in cleared
    assert item["nomenclature"] is None
```

- [ ] **Step 2: Run test, expect FAIL** (current code nulls everything).

Run: `cd docker/docling-graph && python -m pytest tests/test_clear_unsupported_radar_properties.py -v 2>&1 | tail -10`
Expected: failures on `test_supported_numeric_is_preserved` and `test_supported_value_with_unit_conversion` (current code unconditionally nulls).

- [ ] **Step 3: Refactor the function.**

Modify `docker/docling-graph/app/evidence_gate.py`. Add the import at the top:

```python
from app._numeric_evidence import value_is_supported_by_text
```

Replace the body of `_clear_unsupported_radar_properties` (lines 398-444):

```python
def _clear_unsupported_radar_properties(
    item: dict[str, Any], evidence_text: str,
) -> list[str]:
    """Null radar properties whose values aren't supported by batch text.

    Spec §4.8 refactor. Previously unconditionally nulled 18 numeric
    fields; now uses value_is_supported_by_text to preserve values
    that appear in evidence_text (with unit-aware variants).
    """
    cleared: list[str] = []

    # Text fields use the existing exact-quote check.
    exact_text_fields = (
        "nomenclature", "elnot", "dieqp", "asrd",
        "responsible_agency", "review_cycle", "next_review_date",
        "dwell_time", "scan_type", "intra_pulse_mop", "inter_pulse",
    )
    for field_name in exact_text_fields:
        value = item.get(field_name)
        if value is not None and not _value_is_quoted_in_text(value, evidence_text):
            item[field_name] = None
            cleared.append(field_name)

    # Numeric (and the bool / coverage-limits) fields are preserved when
    # value_is_supported_by_text accepts them; nulled otherwise.
    evidence_gate_fields = (
        "erp_dbw", "tx_peak_power_kw", "gain_dbi",
        "antenna_photo", "antenna_dim_az_m", "antenna_dim_el_m",
        "beamwidth_az_deg", "beamwidth_el_deg", "spoiled",
        "coverage_limits_el_deg",
        "nominal_rf_mhz", "nominal_pri_usec", "nominal_pd_usec",
        "scan_period_sec",
        "frequency_excursion_mhz", "num_bits_in_code", "pulses_per_dwell",
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

- [ ] **Step 4: Run test, expect PASS.**

Run: `cd docker/docling-graph && python -m pytest tests/test_clear_unsupported_radar_properties.py -v 2>&1 | tail -10`
Expected: 5 passed.

- [ ] **Step 4b: Add a drift-prevention assertion.**

Append to `tests/test_clear_unsupported_radar_properties.py`:

```python
def test_evidence_gate_fields_matches_field_groups():
    """Drift guard: the evidence_gate_fields tuple in
    _clear_unsupported_radar_properties must equal the union of all
    non-identity fields across the 4 numeric/parameter sub-pass groups
    (power_rf, antenna, timing, modulation) plus 'confidence'. If a new
    field is added to any group, this test must be updated alongside the
    tuple — otherwise the new field is silently nulled forever.
    """
    import inspect
    from app.evidence_gate import _clear_unsupported_radar_properties
    from ontology_bundles.air_defense_v3.extraction_schemas._field_groups import (
        RADAR_FIELD_GROUPS,
    )

    src = inspect.getsource(_clear_unsupported_radar_properties)
    # Grep the literal tuple body — implementation detail, but stable.
    assert "evidence_gate_fields = (" in src, "tuple name changed; update this test"

    expected = (
        set(RADAR_FIELD_GROUPS["radar_power_rf"])
        | set(RADAR_FIELD_GROUPS["radar_antenna"])
        | set(RADAR_FIELD_GROUPS["radar_timing"])
        | set(RADAR_FIELD_GROUPS["radar_modulation"])
    ) - {"system_name"}
    expected.add("confidence")

    for field in expected:
        assert f'"{field}"' in src, (
            f"evidence_gate_fields tuple missing {field!r}; "
            f"new field added to RADAR_FIELD_GROUPS without updating "
            f"the gate. Numeric values for {field} would be nulled."
        )
```

Re-run: `cd docker/docling-graph && python -m pytest tests/test_clear_unsupported_radar_properties.py -v 2>&1 | tail -10`
Expected: 6 passed.

- [ ] **Step 5: Run the full evidence-gate tests to confirm no regression.**

Run: `cd docker/docling-graph && python -m pytest tests/ -v 2>&1 | tail -10`
Expected: all passed. The legacy radar_domain pass continues to work because `_postprocess_air_defense_radars` still dispatches on `pass_name == "radar_domain"`.

- [ ] **Step 6: Commit.**

```bash
git add docker/docling-graph/app/evidence_gate.py docker/docling-graph/tests/test_clear_unsupported_radar_properties.py
git commit -m "$(cat <<'EOF'
fix(docling-graph): _clear_unsupported_radar_properties preserves evidenced numerics (spec §4.8)

CORRECTNESS BLOCKER fix. The function previously unconditionally
nulled 18 numeric fields on every radar pass output (lines 419-442:
erp_dbw, tx_peak_power_kw, gain_dbi, antenna_dim_*, beamwidth_*,
nominal_rf_mhz, nominal_pri_usec, nominal_pd_usec, scan_period_sec,
frequency_excursion_mhz, num_bits_in_code, pulses_per_dwell, plus
antenna_photo, spoiled, coverage_limits_el_deg, confidence). Even a
perfectly-extracted gain_dbi=35.0 from a doc that says "antenna gain
is 35 dBi" was nulled before the response left the service.

Refactor: each numeric value is now checked via the shared
value_is_supported_by_text predicate from _numeric_evidence.py.
Values whose stringified form (or a unit-aware variant) appears in
evidence_text are preserved; others are nulled. Same predicate the
auto-evidence resolver uses — single source of truth.

The exact-text branch for string fields (nomenclature, elnot, etc.)
keeps its current _value_is_quoted_in_text check unchanged.

Without this fix, Session 1's smoke harness would fail even when the
LLM extracts correctly. Tested with 5 cases covering supported,
unsupported, unit-conversion, and the unchanged text-field path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Update `apply_bundle_postprocessing` dispatch (additive)

**Files:**
- Modify: `docker/docling-graph/app/evidence_gate.py:307-`  (the dispatch on pass_name)

Update the dispatch to recognize the 5 new sub-pass names IN ADDITION to `radar_domain`. After this commit, the dispatch fires `_postprocess_air_defense_radars` for both old and new names — safe to land before manifest cutover.

- [ ] **Step 1: Find the dispatch site.**

Run: `grep -n "pass_name == \"radar_domain\"\|pass_name in (\"radar_domain\"" docker/docling-graph/app/evidence_gate.py | head`

- [ ] **Step 2: Update the conditional.**

```python
RADAR_PASS_NAMES = (
    "radar_domain",
    "radar_identity",
    "radar_power_rf",
    "radar_antenna",
    "radar_timing",
    "radar_modulation",
)

# In apply_bundle_postprocessing, replace:
#   if pass_name == "radar_domain":
# with:
#   if pass_name in RADAR_PASS_NAMES:
```

- [ ] **Step 3: Verify idempotency contract.**

Each sub-pass invocation hands the post-processor only that group's slice of fields. Verify `_postprocess_air_defense_radars` doesn't `assert` presence of fields outside the group it's processing. Run this audit grep:

```bash
grep -nE "assert\b|raise (KeyError|ValueError|TypeError)" \
  docker/docling-graph/app/evidence_gate.py \
  | grep -A0 -B0 "" | head -30
```

Then visually scan the `_postprocess_air_defense_radars` body (lines 401-417) and `_clear_unsupported_radar_properties` body for any `item[field]` (bracket access without `.get`) — bracket access raises `KeyError` on a missing field; `.get()` is safe.

Specifically, audit:
- `_clear_unsupported_radar_properties` — already iterates `item.get(field_name)` and only modifies when present. Idempotent.
- Any identity sanitization or recall-pattern logic — verify it tolerates missing fields.

If any bracket access on a sub-pass-omitted field is found, change to `.get()` before continuing.

- [ ] **Step 4: Run service tests.**

Run: `cd docker/docling-graph && python -m pytest tests/ -v 2>&1 | tail -5`
Expected: all passed.

- [ ] **Step 5: Commit.**

```bash
git add docker/docling-graph/app/evidence_gate.py
git commit -m "feat(docling-graph): apply_bundle_postprocessing recognizes radar sub-passes (spec §4.8)

Additive: dispatch now matches both legacy radar_domain and the 5 new
sub-pass names (radar_identity, radar_power_rf, radar_antenna,
radar_timing, radar_modulation). Safe to land before manifest cutover —
radar_domain still fires through the same postprocessor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Update `_DOMAIN_PASS_NAMES` (additive)

**Files:**
- Modify: `app/workers/pipeline.py:381-383`

- [ ] **Step 1: Locate and inspect.**

Run: `grep -n "_DOMAIN_PASS_NAMES\|domain_hit" app/workers/pipeline.py | head`

- [ ] **Step 2: Update the frozenset.**

Replace the current value:

```python
_DOMAIN_PASS_NAMES = frozenset({
    "radar_domain", "missile_domain", "other_systems", "system_links",
})
```

with the union of old + new (additive — keeps `radar_domain` until cutover):

```python
_DOMAIN_PASS_NAMES = frozenset({
    "radar_domain",
    "radar_identity", "radar_power_rf", "radar_antenna",
    "radar_timing", "radar_modulation",
    "missile_domain", "system_links",
})
```

(Drop `"other_systems"` — no longer a manifest pass; dead in the existing set.)

- [ ] **Step 3: Run pipeline tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_classify_extraction_quality.py -v 2>&1 | tail -10`
Expected: all passed.

- [ ] **Step 4: Commit.**

```bash
git add app/workers/pipeline.py
git commit -m "feat(workers): _DOMAIN_PASS_NAMES recognizes radar sub-passes (spec §4.8)

Additive: includes both legacy radar_domain and the 5 new sub-pass
names. Drops dead 'other_systems' entry. system_links and
missile_domain unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Add upstream-ref dedupe (mandatory unit test)

**Files:**
- Modify: `app/workers/pipeline.py::_extend_upstream_refs` (or call-site that aggregates across passes)
- Modify: `tests/unit/test_pipeline_upstream_refs.py` (add the dedupe test)

Per spec §4.5: "the upstream-ref builder MUST dedupe by `(entity_type, normalized identity_values)` across all 6 dependency pass outputs."

The existing `_extend_upstream_refs` accumulates refs into a `dict` keyed by ref-id (E001, E002, …). After radar's cutover, 5 sub-passes each emit a partial RADAR_SYSTEM record with `system_name="Fan Song"`; without dedupe, the relationship pass receives 5 distinct ref-ids for the same logical entity, which inflates prompt size, dilutes the relationship LLM's attention, and breaks downstream merge semantics. This is too important to leave to a smoke harness — write the test first.

- [ ] **Step 1: Locate the dedupe site.**

Run: `grep -rn "upstream_entities\|_extend_upstream_refs" app/workers/ docker/docling-graph/app/ 2>/dev/null | grep -v __pycache__ | head -15`

`_extend_upstream_refs` is the per-pass extender in `app/workers/pipeline.py:10`. It is called once per upstream pass; the dedupe must happen either inside `_extend_upstream_refs` (skip-if-already-seen) or at the loop site that calls it sequentially across passes.

- [ ] **Step 2: Write the failing test.**

Append to `tests/unit/test_pipeline_upstream_refs.py` (the file already imports `_extend_upstream_refs`, `SimpleNamespace`, `_FakePassResult`, and `ONTOLOGY` — reuse them):

```python
class TestExtendUpstreamRefsDedupe:
    """After the radar field-group cutover, 5 sub-passes each emit a
    partial RADAR_SYSTEM with system_name='Fan Song'. They must collapse
    to a single upstream ref before the relationship pass sees them."""

    def _pass_def(self, name: str, primary_types):
        return SimpleNamespace(name=name, primary_entity_types=primary_types)

    def test_five_partial_radars_collapse_to_one_upstream_ref(self):
        refs: dict = {}
        for pass_name in (
            "radar_identity", "radar_power_rf", "radar_antenna",
            "radar_timing", "radar_modulation",
        ):
            pass_result = _FakePassResult({
                "RADAR_SYSTEM": [SimpleNamespace(system_name="Fan Song")],
            })
            _extend_upstream_refs(
                refs, pass_result,
                self._pass_def(pass_name, ["RADAR_SYSTEM"]),
                ONTOLOGY,
            )
        # Exactly one ref for Fan Song, regardless of how many sub-passes
        # emitted it.
        fan_song_refs = [
            r for r in refs.values()
            if r.identity_values.get("system_name") == "Fan Song"
        ]
        assert len(fan_song_refs) == 1, (
            f"expected 1 dedup'd ref for Fan Song; got {len(fan_song_refs)}: "
            f"{fan_song_refs!r}"
        )

    def test_dedupe_is_per_identity_not_per_pass(self):
        """Different system_names from different passes must NOT collapse."""
        refs: dict = {}
        _extend_upstream_refs(
            refs,
            _FakePassResult({"RADAR_SYSTEM": [SimpleNamespace(system_name="Fan Song")]}),
            self._pass_def("radar_identity", ["RADAR_SYSTEM"]),
            ONTOLOGY,
        )
        _extend_upstream_refs(
            refs,
            _FakePassResult({"RADAR_SYSTEM": [SimpleNamespace(system_name="Spoon Rest")]}),
            self._pass_def("radar_power_rf", ["RADAR_SYSTEM"]),
            ONTOLOGY,
        )
        names = {r.identity_values.get("system_name") for r in refs.values()}
        assert names == {"Fan Song", "Spoon Rest"}
```

- [ ] **Step 3: Run, expect FAIL on the first test (no dedupe yet).**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py::TestExtendUpstreamRefsDedupe -v 2>&1 | tail -10`
Expected: `test_five_partial_radars_collapse_to_one_upstream_ref` FAILS (5 refs, not 1) — that's the bug. `test_dedupe_is_per_identity_not_per_pass` may pass already (2 distinct names → 2 distinct refs is the no-dedupe default behavior).

- [ ] **Step 4: Implement dedupe in `_extend_upstream_refs`.**

Add a check at the top of the per-entity loop: build the identity tuple `(entity_type, tuple(sorted(identity_values.items())))`, scan existing `refs.values()` for a match, and skip-if-seen. Or maintain an auxiliary `seen` set keyed alongside `refs` (passed in as another arg, or via a wrapper helper).

- [ ] **Step 5: Run, expect both tests PASS.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_pipeline_upstream_refs.py -v 2>&1 | tail -10`
Expected: all tests pass (existing + 2 new).

- [ ] **Step 6: Commit.**

```bash
git add app/workers/pipeline.py tests/unit/test_pipeline_upstream_refs.py
git commit -m "$(cat <<'EOF'
fix(workers): _extend_upstream_refs dedupes by identity across passes (spec §4.5)

After the radar field-group cutover, 5 sub-passes each emit a partial
RADAR_SYSTEM with the same system_name. Without dedupe, the downstream
relationship pass (system_links) receives 5 distinct E### ref-ids for
the same logical entity, inflating prompt size and breaking merge.

Two new tests pin the contract: 5 partial radars from 5 sub-passes
collapse to 1 ref; distinct system_names from different sub-passes do
NOT collapse.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Run the relevant tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/ -k "upstream" -v 2>&1 | tail -10`

- [ ] **Step 6: Commit (or no-op commit if already deduped).**

```bash
# If changes made:
git add <files>
git commit -m "feat(orchestrator): upstream-ref builder dedupes by identity (spec §4.5)

5 sub-passes emit RADAR_SYSTEM with the same system_name. Without
dedupe, system_links would receive 5 upstream refs for one Fan Song —
wasting tokens and encouraging duplicate emissions. Dedup keys by
(entity_type, sorted identity_values).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

# If already deduped:
echo "verified — no changes needed" > /tmp/dedupe-check.txt
# Add a note to the cutover commit message documenting the verification.
```

---

### Task 14: Update `schemas.py` docstring

**Files:**
- Modify: `docker/docling-graph/app/schemas.py:55` (docstring example)

- [ ] **Step 1: Find and replace the literal.**

Run: `grep -n "radar_domain" docker/docling-graph/app/schemas.py`

Replace the docstring example's `pass_name="radar_domain"` with `pass_name="radar_identity"`.

- [ ] **Step 2: Commit.**

```bash
git add docker/docling-graph/app/schemas.py
git commit -m "docs(docling-graph): schemas.py docstring uses radar_identity example

Low-risk pass-name literal update per spec §4.8. No functional change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: Mark `radar_domain.py` legacy in module docstring

**Files:**
- Modify: `ontology_bundles/air_defense_v3/extraction_schemas/radar_domain.py` (top of file)

- [ ] **Step 1: Update the module docstring.**

Prepend to the existing docstring at the top of `radar_domain.py`:

```python
"""**LEGACY** — Not in the active manifest as of the
2026-04-27 radar field-group refactor. Replaced by:

- radar_identity.py
- radar_power_rf.py
- radar_antenna.py
- radar_timing.py
- radar_modulation.py

This module is kept in source as a reference for description text and
for legacy-loadability tests (e.g. test_service_identity_gate.py).
Do not add manifest entries pointing here. Will be removed in a
future cleanup once the new structure has been operationally proven.

---

[existing docstring content continues below]
"""
```

- [ ] **Step 2: Commit.**

```bash
git add ontology_bundles/air_defense_v3/extraction_schemas/radar_domain.py
git commit -m "docs(extraction): mark radar_domain.py legacy

Per spec §6 step 4. File stays in source but is removed from the
manifest in the cutover commit (Task 17). Module docstring now
flags it as legacy with pointers to the 5 sub-pass replacements.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: Update test fixtures referencing radar_domain pass-name

**Files:**
- Modify: `docker/docling-graph/tests/test_extract_pass_endpoint.py`
- Modify: `docker/docling-graph/tests/test_service_identity_gate.py`

- [ ] **Step 1: Find the literal.**

Run: `grep -n 'pass_name.*radar_domain\|"radar_domain"' docker/docling-graph/tests/test_extract_pass_endpoint.py docker/docling-graph/tests/test_service_identity_gate.py`

- [ ] **Step 2: Update fixtures.**

In both files, replace `pass_name="radar_domain"` with `pass_name="radar_identity"` in the request body fixtures.

In `test_service_identity_gate.py`, **keep** the `from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import RadarDomainPass` line — this is a legacy-loadability check, not a pass-name reference. Add a one-line comment above the import:

```python
# Legacy-loadability check: radar_domain.py is kept in source even
# after the field-group cutover. If this import breaks, that's a
# regression even though the legacy module isn't in the manifest.
from ontology_bundles.air_defense_v3.extraction_schemas.radar_domain import RadarDomainPass
```

- [ ] **Step 3: Run the service tests.**

Run: `cd docker/docling-graph && python -m pytest tests/ -v 2>&1 | tail -10`

**Decision rule (don't burn cycles debugging):**

- **Pass:** continue to Step 4 (commit).
- **Fail with manifest-resolution error mentioning `radar_identity` (`ManifestError`, `KeyError: radar_identity`, "unknown pass", or similar):** the fixtures depend on the live `manifest.yaml` which doesn't yet know `radar_identity`. **Stop here.** Stage the fixture changes for Chunk 4's cutover commit (Task 17) instead — `git stash push -m "task16-fixture-updates" -- docker/docling-graph/tests/test_extract_pass_endpoint.py docker/docling-graph/tests/test_service_identity_gate.py` and pop after the cutover lands. Do **not** revert `radar_domain.py` or relax the manifest to make these pass before cutover.
- **Fail with anything else:** real regression — diagnose before continuing.

If the tests pass-through a custom manifest fixture (not loading the real `manifest.yaml`), they'll work now and you proceed normally.

- [ ] **Step 4: Commit (only if tests pass).**

```bash
git add docker/docling-graph/tests/test_extract_pass_endpoint.py docker/docling-graph/tests/test_service_identity_gate.py
git commit -m "test(docling-graph): fixture pass_name → radar_identity (spec §4.8)

Updates fixtures from radar_domain to radar_identity. The legacy
RadarDomainPass import in test_service_identity_gate.py is kept
intentionally as a legacy-loadability regression check.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 4: Manifest cutover + verification

Tasks 17-21 flip the manifest, prune `_DOMAIN_PASS_NAMES`, update manifest-shape tests, and verify end-to-end against the live service.

### Task 17: Manifest cutover (atomic commit)

**Files:**
- Modify: `ontology_bundles/air_defense_v3/manifest.yaml`
- Modify: `app/workers/pipeline.py:381-383` (prune `radar_domain`)
- Modify: `tests/unit/test_ontology_bundles.py:5-19` (manifest-shape assertion)
- Modify: `tests/unit/test_extraction_schemas.py:8,14` (PASS_MODULES)
- Modify: `tests/integration/test_pr1_scaffolding_smoke.py:36-45,85-95`
- Modify: `ontology_bundles/air_defense_v3/extraction_schemas/system_links.py` (docstring; if needed)
- Stage if Task 16 was deferred: `docker/docling-graph/tests/test_extract_pass_endpoint.py`, `docker/docling-graph/tests/test_service_identity_gate.py` (`pass_name` fixture swap from `radar_domain` → `radar_identity`). If you stashed Task 16's fixture changes per its decision rule, `git stash pop` them now and they land in this atomic commit.

This commit must land everything together so main stays green between commits but post-commit the active extraction shape is the new 7-pass structure.

- [ ] **Step 1: Update `manifest.yaml`.**

Remove the `radar_domain` entry; add 5 new entries; update `system_links.depends_on`:

```yaml
bundle_key: air_defense_v3
manifest_schema_version: "1.0.0"
ontology_name: "EIP Military Equipment Ontology"
ontology_version: "3.0.0"
extraction_profile_version: "1.0.0"

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
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.radar_power_rf
    template_class: RadarPowerRfPass
    primary_entity_types: [RADAR_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: radar_antenna
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.radar_antenna
    template_class: RadarAntennaPass
    primary_entity_types: [RADAR_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: radar_timing
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.radar_timing
    template_class: RadarTimingPass
    primary_entity_types: [RADAR_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: radar_modulation
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.radar_modulation
    template_class: RadarModulationPass
    primary_entity_types: [RADAR_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

  - name: missile_domain
    required: false
    kind: entities
    input_mode: document_only
    module: extraction_schemas.missile_domain
    template_class: MissileDomainPass
    primary_entity_types: [MISSILE_SYSTEM]
    bridge_entity_types: []
    extracted_relationship_types: []
    depends_on: []

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
    skip_justification: >
      When none of the upstream domain passes produce any linkable system
      entities, there is nothing for system_links to link. Skipping is
      preferable to dispatching an LLM call that can only fail.
```

- [ ] **Step 2: Prune `_DOMAIN_PASS_NAMES`.**

Modify `app/workers/pipeline.py`:

```python
# Drop "radar_domain" from the set:
_DOMAIN_PASS_NAMES = frozenset({
    "radar_identity", "radar_power_rf", "radar_antenna",
    "radar_timing", "radar_modulation",
    "missile_domain", "system_links",
})
```

- [ ] **Step 3: Update `tests/unit/test_ontology_bundles.py`.**

Modify lines 5-19 (the manifest-shape assertion):

```python
def test_manifest_pass_count_and_names():
    from ontology_bundles.air_defense_v3 import manifest as m_module
    m = m_module.MANIFEST  # or whatever the entry symbol is
    assert len(m.passes) == 7
    pass_names = {p.name for p in m.passes}
    assert pass_names == {
        "radar_identity", "radar_power_rf", "radar_antenna",
        "radar_timing", "radar_modulation",
        "missile_domain", "system_links",
    }
```

- [ ] **Step 4: Update `tests/unit/test_extraction_schemas.py`.**

Replace lines 8 + 14:

```python
from ontology_bundles.air_defense_v3.extraction_schemas import (
    radar_identity, radar_power_rf, radar_antenna, radar_timing, radar_modulation,
    missile_domain, system_links,
)

PASS_MODULES = [
    (radar_identity, "RadarIdentityPass"),
    (radar_power_rf, "RadarPowerRfPass"),
    (radar_antenna, "RadarAntennaPass"),
    (radar_timing, "RadarTimingPass"),
    (radar_modulation, "RadarModulationPass"),
    (missile_domain, "MissileDomainPass"),
    (system_links, "SystemLinksPass"),
]
```

- [ ] **Step 5: Update `tests/integration/test_pr1_scaffolding_smoke.py`.**

Replace the literal pass-name lists at lines 36-45 and 85-95 with the new 7-pass shape, OR rewrite to load from the manifest dynamically:

```python
expected_passes = {
    "radar_identity", "radar_power_rf", "radar_antenna",
    "radar_timing", "radar_modulation",
    "missile_domain", "system_links",
}
```

- [ ] **Step 6: Update `system_links.py` docstring** (if it mentions `radar_domain`).

Run: `grep -n "radar_domain" ontology_bundles/air_defense_v3/extraction_schemas/system_links.py`

If present, update to reference `radar_identity` (or generic phrasing).

- [ ] **Step 7: Update test fixtures from Task 16** (if deferred).

If Task 16 was a no-op because tests required the new manifest, apply those updates here as part of the same cutover commit.

- [ ] **Step 8: Run the relevant tests.**

```bash
SKIP_COV=1 .venv/bin/pytest \
  tests/unit/test_ontology_bundles.py \
  tests/unit/test_extraction_schemas.py \
  tests/unit/test_radar_field_groups_contract.py \
  tests/unit/test_radar_shared.py \
  tests/integration/test_pr1_scaffolding_smoke.py \
  -v 2>&1 | tail -20
```
Expected: all passed.

- [ ] **Step 9: Run `check_bundle()`.**

```bash
.venv/bin/python -c "
from tools.extraction_coverage.rules import check_bundle
from pathlib import Path
errors, warnings = check_bundle(Path('ontology_bundles/air_defense_v3'))
print(f'errors={len(errors)} warnings={len(warnings)}')
for e in errors: print(' E:', e)
for w in warnings[:5]: print(' W:', w)
"
```
Expected: 0 errors. (Bundle-checker should be content with the new 7-pass shape.)

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
feat(extraction): manifest cutover — radar_domain → 5 sub-passes (spec §4.5)

Atomic cutover commit. Replaces the single radar_domain extraction pass
with five focused sub-passes (radar_identity, radar_power_rf,
radar_antenna, radar_timing, radar_modulation), each with its own
/extract-pass call against a 4-11 field schema. All emit RADAR_SYSTEM[]
with system_name identity; merge_and_resolve collapses partial records
onto one vertex.

Manifest changes:
- Remove radar_domain entry
- Add 5 new entity passes (depends_on: [])
- system_links.depends_on lists all 6 entity passes (5 radar + missile_domain)

Code changes (downstream of the additive Task 11-16 prep commits):
- app/workers/pipeline.py: prune radar_domain from _DOMAIN_PASS_NAMES
- tests/unit/test_ontology_bundles.py: assert 7 passes, new pass-name set
- tests/unit/test_extraction_schemas.py: PASS_MODULES iterates new sub-passes
- tests/integration/test_pr1_scaffolding_smoke.py: literal pass-name lists updated
- system_links.py docstring: pass-name reference updated (if present)
- docker/docling-graph/tests/*: fixture pass_name updated to radar_identity

Verification:
- check_bundle() reports 0 errors
- Field-group contract tests (5) pass
- Description-quality contract (5 sub-pass record classes) passes
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

**Why these two are ignored from the broad sweep:**

- `tests/unit/test_extraction_schemas.py` — the conftest in this repo shadows it with a library-side import path; running it inside the broad sweep produces collection-time conflicts. Run it standalone (next command) so the field-group changes are still verified.
- `tests/unit/test_specification_entity_validation.py` — pre-existing skip from prior schema-compliance work, unrelated to Session 1. The pre-flight P1 check at line 28 uses the same `--ignore` set, so this matches the baseline rather than introducing new noise.

Run the standalone version:

```bash
SKIP_COV=1 .venv/bin/pytest tests/unit/test_extraction_schemas.py -v 2>&1 | tail -10
```

- [ ] **Step 2: Document results.**

Expected: 1240+ passed; same skipped/xfailed counts as P1 baseline. Any new failures must be investigated and fixed before proceeding.

- [ ] **Step 3: No commit (verification only).**

If failures surface, fix them in a separate commit per failure (or roll back the cutover commit and address before re-cutting over). Document any pre-existing unrelated failures.

---

### Task 19: Add smoke harness

**Files:**
- Create: `tests/integration/test_radar_field_groups_smoke.py`

- [ ] **Step 1: Write the smoke harness.**

```python
"""Phase B Session 1 — smoke harness for the field-group split (spec §5.2).

Hits live docling-graph at http://localhost:8002/extract-pass with 3
minimal DoclingDocuments. Each test parametrizes:
- pass_name to invoke
- source text
- target field
- acceptable [lower, upper] range bracketing the source-text value

Skipped when docling-graph is unreachable. Marked @pytest.mark.integration
so it stays out of the default pytest tests/unit invocation.

Range calibration policy (spec §5.2): ranges bracket the source-text
value with tolerance for unit-conversion rounding, NOT the model's
observed output. If a future model emits 3500 MHz for a doc that says
3000 MHz, this test SHOULD fail. Recalibrating to model output would
mask regressions.
"""
import os
import pytest
import requests

DOCLING_GRAPH_URL = os.environ.get(
    "DOCLING_GRAPH_URL", "http://localhost:8002/extract-pass"
)


def _build_doc(text: str) -> dict:
    """Minimal valid DoclingDocument with one paragraph.

    label: "text" — existing fixtures and service-injected text use
    this. "paragraph" may be accepted by some Docling versions but
    leads to debugging document-shape issues instead of measuring
    extraction.
    """
    return {
        "schema_name": "DoclingDocument",
        "version": "1.0.0",
        "name": "test-fansong-smoke",
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
    "pass_name,text,field,lower,upper",
    [
        ("radar_power_rf",
         "Fan Song transmitter peak power is 600 kW.",
         "tx_peak_power_kw", 400.0, 800.0),
        ("radar_power_rf",
         "Fan Song operates at 3000 MHz.",
         "nominal_rf_mhz", 2900.0, 3100.0),
        ("radar_antenna",
         "Fan Song antenna gain is 35 dBi.",
         "gain_dbi", 33.0, 37.0),
    ],
    ids=["power-600kW", "freq-3000MHz", "gain-35dBi"],
)
def test_radar_field_group_numeric_smoke(pass_name, text, field, lower, upper):
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
    radar_systems = pass_output.get("radar_systems", []) or []
    assert len(radar_systems) >= 1, (
        f"expected ≥1 radar_system; pass_output={pass_output!r}"
    )

    entity = next(
        (e for e in radar_systems
         if "Fan Song" in (e.get("system_name") or "")),
        None,
    )
    assert entity is not None, f"Fan Song not found; got {radar_systems!r}"

    value = entity.get(field)
    if value is None:
        print(f"\n--- FAILURE DEBUG: pass_output ---\n{pass_output}\n---")
        pytest.fail(
            f"{pass_name}.{field} was None; expected value in [{lower}, {upper}]"
        )
    assert isinstance(value, (int, float)), (
        f"{field} is {type(value).__name__}, want number; got {value!r}"
    )
    assert lower <= float(value) <= upper, (
        f"{field}={value} not in [{lower}, {upper}]"
    )
```

- [ ] **Step 2: Verify the test discovers correctly.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/integration/test_radar_field_groups_smoke.py --collect-only 2>&1 | tail -5`
Expected: 3 collected.

- [ ] **Step 3: Verify it skips when docling-graph is offline.**

```bash
DOCLING_GRAPH_URL="http://localhost:9999/extract-pass" \
  SKIP_COV=1 .venv/bin/pytest tests/integration/test_radar_field_groups_smoke.py -v 2>&1 | tail -10
```
Expected: 3 skipped.

- [ ] **Step 4: Commit.**

```bash
git add tests/integration/test_radar_field_groups_smoke.py
git commit -m "$(cat <<'EOF'
test(extraction): radar field-group smoke harness (spec §5.2)

3 known-failing-in-Phase-A cases hit live docling-graph at
:8002/extract-pass with minimal DoclingDocuments and assert numeric
extraction lands in source-truth-bracketed ranges:

- "Fan Song transmitter peak power is 600 kW" → tx_peak_power_kw in [400, 800]
- "Fan Song operates at 3000 MHz"             → nominal_rf_mhz in [2900, 3100]
- "Fan Song antenna gain is 35 dBi"           → gain_dbi in [33, 37]

Marked @pytest.mark.integration; skipped when service is offline.
Range calibration policy: ranges bracket source-text value, NOT the
model's observed output. If a future model emits 3500 MHz for a doc
that says 3000 MHz, this test SHOULD fail.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 20: Rebuild docling-graph + worker-graph

**Files:** none modified (operational step)

- [ ] **Step 1: Stop and rebuild.**

```bash
docker compose stop docling-graph worker-graph
docker compose build docling-graph
docker compose up -d docling-graph worker-graph
```

- [ ] **Step 2: Wait for readiness.**

```bash
until curl -sf -o /dev/null http://localhost:8002/health 2>/dev/null; do sleep 2; done
docker compose ps --format "table {{.Service}}\t{{.Status}}" | grep -E "docling-graph|worker-graph"
```
Expected: both Up.

- [ ] **Step 3: Verify the install patches are loaded.**

```bash
docker compose logs --tail 30 docling-graph 2>&1 | grep -E "prompt_rules: installed|semantic-guide"
```
Expected: log line confirming `prompt_rules: installed delta system-prompt rewrite + semantic-guide budget expansion`.

- [ ] **Step 4: No commit (operational only).**

---

### Task 21: Re-ingest SNR-75 + run smoke harness

**Files:** none modified (verification only)

- [ ] **Step 1: Run the smoke harness against the live service.**

```bash
SKIP_COV=1 .venv/bin/pytest tests/integration/test_radar_field_groups_smoke.py -v 2>&1 | tail -20
```

Expected outcomes per spec §7:

| Result | Interpretation | Next step |
|---|---|---|
| 3/3 pass | ✅ Field grouping unblocked numeric extraction | Continue to Step 2 (re-ingest verification) |
| 2/3 pass | ✅ Acceptable per success criteria | Continue to Step 2; document the 1 failure |
| 0/3 or 1/3 | ⚠️ Field grouping insufficient | **Stop iterating prompts; switch to spec §10 fallback track (Session 2: identity-fed parameter passes + group-scoped retry + candidate-mapping)** |

- [ ] **Step 2: Re-ingest the SNR-75 Wikipedia PDF.**

```bash
DOC_ID="f0329023-0c54-4ea1-a485-6c693f55bfed"   # SNR-75 - Wikipedia.pdf (Fan Song spec content)
curl -sS -X POST "http://localhost:8005/v1/documents/$DOC_ID/reingest" \
  -H "Content-Type: application/json" \
  -d '{"mode": "graph_only"}'
```

- [ ] **Step 3: Wait for completion.**

```bash
# Watch worker-graph logs until derive_ontology_graph succeeds:
docker compose logs --since 1m -f worker-graph 2>&1 | grep -E "derive_ontology_graph.*succeeded|FAILURE"
# Ctrl-C when one of the new sub-passes completes (one chain succeeds = all 6 entity passes done; system_links runs after).
```

- [ ] **Step 4: Verify the FAN SONG vertex.**

```bash
AUTH="root:eip_arcadedb_secret"
curl -s -u "$AUTH" -X POST http://localhost:2480/api/v1/query/eip_knowledge_graph \
  -H "Content-Type: application/json" \
  -d '{"language":"sql","command":"SELECT * FROM RADAR_SYSTEM WHERE name=\"FAN SONG\""}' \
  | python3 -m json.tool | head -50
```

Expected per spec §7: ≥1 numeric flat-checklist field (`gain_dbi`, `nominal_rf_mhz`, or `tx_peak_power_kw`) populated.

- [ ] **Step 5: Verify the section endpoint.**

```bash
curl -sS -X POST "http://localhost:8005/v1/query-profiles/search/section" \
  -H "Content-Type: application/json" \
  -d '{"profile_id":"system_rf_parameters","query_text":"Fan Song","top_k":5}' \
  | python3 -m json.tool | head -50
```

Expected: `field_groups` populated with at least one row carrying a numeric value.

- [ ] **Step 6: Document the outcome.**

Add a brief outcome note to `docs/superpowers/plans/2026-04-27-radar-field-group-extraction.md` (this file) — a "## Session 1 outcome" section at the bottom — capturing:
- Smoke harness result (X/3)
- Number of populated flat-checklist fields on FAN SONG vertex
- Section endpoint field_groups count
- Whether spec §10 fallback track is now triggered

If 0/3 or 1/3: stop here and brainstorm Session 2 per spec §10. **Do NOT continue iterating on prompt tuning.** Field grouping was the leverage; if it didn't move the needle, the architecture needs to change (identity-fed parameter passes, candidate-mapping).

- [ ] **Step 7: No commit (verification + outcome documentation).**

If you do add the outcome note, commit separately:

```bash
git add docs/superpowers/plans/2026-04-27-radar-field-group-extraction.md
git commit -m "docs(plan): radar field-group Session 1 outcome — N/3 smoke, X numeric fields on FAN SONG"
```

---

## Session 1 acceptance gate

Per spec §7:

- [ ] All 5 group-membership contract tests pass
- [ ] Description-quality contract passes for all 5 sub-pass record classes
- [ ] Full relevant unit + pipeline regression sweep green; pre-existing unrelated failures documented
- [ ] At least 2 of 3 smoke cases extract a numeric value within the expected range — OR a Session 2 plan triggered per §10 fallback track
- [ ] Section endpoint smoke test on FAN SONG vertex shows ≥1 numeric flat-checklist field populated post re-ingest

If the smoke gate doesn't pass, the field-group split is functionally complete (the structural code is correct, contract tests pass, etc.) but the numeric-extraction goal is not met. Trigger spec §10 fallback track for Session 2.
