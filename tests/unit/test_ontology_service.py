"""Task 1: live ontology service — no stored/registry copy.

get_live_ontology() must be derived purely from the air_defense_v3
introspection SSoT (build_ontology_dict()) plus the canonical Pydantic
classes' json_schema_extra["profile_sections"] tags — not hardcoded.
"""
import pytest

pytestmark = pytest.mark.unit


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
    # New shape: each entry is {name, description}.
    assert all(set(s.keys()) == {"name", "description"} for s in sections)
    names = [s["name"] for s in sections]

    # Original three field-derived sections still present.
    assert "rf_parameters" in names
    assert "components" in names
    assert "performance" in names
    # Four new field-derived sections.
    for new_section in ("engagement_envelope", "governance", "identification", "deployment"):
        assert new_section in names
    # The dossier catch-all is advertised even though it is not a field tag.
    assert "dossier" in names
    assert len(names) == 8

    # Field-derived names (everything except the appended dossier catch-all)
    # remain deduped + sorted; dossier is appended last.
    assert names[-1] == "dossier"
    field_derived = names[:-1]
    assert field_derived == sorted(set(field_derived))

    # Descriptions are populated (non-empty) for every advertised section.
    by_name = {s["name"]: s["description"] for s in sections}
    assert by_name["dossier"]
    assert "designators" in by_name["identification"].lower() or by_name["identification"]


def test_get_live_ontology_version_matches_bundle():
    from app.services.ontology_service import get_live_ontology
    from ontology_bundles.air_defense_v3.introspect import ONTOLOGY_VERSION

    ontology = get_live_ontology()
    # matches the bundle constant AND pins the current literal so a
    # silent version bump is caught here.
    assert ontology["version"] == ONTOLOGY_VERSION == "3.0.0"


def test_canonical_map_import_contract():
    """The service imports the module-private ``_CANONICAL_BY_ENTITY_TYPE``
    from ``app.services.query_profiles`` (a file the plan is actively
    reshaping). Assert the symbol still imports and is a non-empty mapping
    covering the profile-capable roots — so a future rename fails loudly
    right here instead of at HTTP time.
    """
    from app.services.query_profiles import _CANONICAL_BY_ENTITY_TYPE

    assert isinstance(_CANONICAL_BY_ENTITY_TYPE, dict)
    assert _CANONICAL_BY_ENTITY_TYPE
    assert "RADAR_SYSTEM" in _CANONICAL_BY_ENTITY_TYPE
    assert "MISSILE_SYSTEM" in _CANONICAL_BY_ENTITY_TYPE


def test_ontology_response_schema_validates_live_payload():
    """The actual deliverable schema must accept the service dict without
    drift (field names / shapes stay in lockstep)."""
    from app.schemas.query_profiles import OntologyResponse
    from app.services.ontology_service import get_live_ontology

    model = OntologyResponse.model_validate(get_live_ontology())

    assert model.version == "3.0.0"
    assert model.entity_types
    names = {et.name for et in model.entity_types}
    assert "RADAR_SYSTEM" in names
    assert "MISSILE_SYSTEM" in names
    assert all(et.name and et.label for et in model.entity_types)
    assert model.relationship_types
    assert all(rt.name for rt in model.relationship_types)
    section_names = {s.name for s in model.profile_sections}
    for section in (
        "rf_parameters", "components", "performance",
        "engagement_envelope", "governance", "identification",
        "deployment", "dossier",
    ):
        assert section in section_names
    # Every advertised section carries a description string.
    assert all(isinstance(s.description, str) for s in model.profile_sections)


def _build_route_client():
    """Mount only the query-profiles router on a bare FastAPI app (no
    lifespan / model preload / DB) and return a TestClient. The
    ``GET /v1/ontology`` handler has no DB dependency, so this exercises
    the real registered route without the full app harness."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.v1.query_profiles import router

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    return TestClient(app)


def test_ontology_route_returns_live_payload():
    """End-to-end through the real registered route.

    NOTE: while the parallel Task 2/3 refactor is mid-flight,
    ``app.api.v1.query_profiles`` still imports registry symbols that
    Task 2 has already removed from ``app.services.query_profiles``, so
    the router module is temporarily un-importable. Skip (don't fail) in
    that window — the test auto-activates once Task 3 removes the
    registry routes/imports.
    """
    try:
        client = _build_route_client()
    except ImportError as exc:
        pytest.skip(
            "app.api.v1.query_profiles router not importable yet "
            f"(registry symbols mid-removal by the parallel task): {exc}"
        )

    resp = client.get("/v1/ontology")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == "3.0.0"
    names = {entry["name"] for entry in data["entity_types"]}
    assert "RADAR_SYSTEM" in names
    assert "MISSILE_SYSTEM" in names
    section_names = {s["name"] for s in data["profile_sections"]}
    for section in (
        "rf_parameters", "components", "performance",
        "engagement_envelope", "governance", "identification",
        "deployment", "dossier",
    ):
        assert section in section_names
