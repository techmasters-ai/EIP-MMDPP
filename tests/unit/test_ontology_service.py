"""Task 1: live ontology service — no stored/registry copy.

get_live_ontology() must be derived purely from the air_defense_v3
introspection SSoT (build_ontology_dict()) plus the canonical Pydantic
classes' json_schema_extra["profile_sections"] tags — not hardcoded.
"""


def test_get_live_ontology_entity_types_include_radar_and_missile():
    from app.services.ontology_service import get_live_ontology

    ontology = get_live_ontology()
    assert ontology["entity_types"]
    names = {entry["name"] for entry in ontology["entity_types"]}
    assert "RADAR_SYSTEM" in names
    assert "MISSILE_SYSTEM" in names
    # every entry carries a label
    for entry in ontology["entity_types"]:
        assert entry["label"]


def test_get_live_ontology_relationship_types_non_empty():
    from app.services.ontology_service import get_live_ontology

    ontology = get_live_ontology()
    assert ontology["relationship_types"]
    assert all("name" in entry for entry in ontology["relationship_types"])


def test_get_live_ontology_profile_sections_derived_not_hardcoded():
    from app.services.ontology_service import get_live_ontology

    ontology = get_live_ontology()
    sections = ontology["profile_sections"]
    assert "rf_parameters" in sections
    assert "components" in sections
    assert "performance" in sections
    # deduped + sorted
    assert sections == sorted(set(sections))


def test_get_live_ontology_version_matches_bundle():
    from app.services.ontology_service import get_live_ontology
    from ontology_bundles.air_defense_v3.introspect import ONTOLOGY_VERSION

    ontology = get_live_ontology()
    assert ontology["version"] == ONTOLOGY_VERSION
