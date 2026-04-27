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
