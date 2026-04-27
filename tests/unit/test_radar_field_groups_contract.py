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
