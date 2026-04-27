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
