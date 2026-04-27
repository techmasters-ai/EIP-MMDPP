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
