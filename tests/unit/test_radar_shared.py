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
