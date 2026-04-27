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
