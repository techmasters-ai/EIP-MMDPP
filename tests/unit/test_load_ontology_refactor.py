"""Tests for the refactored load_ontology / load_registry_ontology split.
Spec §7.3 + §4.6."""
import pytest
from pathlib import Path

pytestmark = pytest.mark.unit


def test_load_ontology_no_args_uses_system_default_bundle():
    """With no args, load_ontology returns the system default bundle's ontology."""
    from app.services.ontology_templates import load_ontology
    ont = load_ontology()
    assert "entity_types" in ont
    # Air-defense bundle has RADAR_SYSTEM
    assert any(e.get("name") == "RADAR_SYSTEM" for e in ont["entity_types"])


def test_load_ontology_with_bundle_key():
    from app.services.ontology_templates import load_ontology
    ont = load_ontology(bundle_key="air_defense_v3")
    assert "entity_types" in ont


def test_load_ontology_unknown_bundle_raises():
    from app.services.ontology_templates import load_ontology
    with pytest.raises(Exception):
        load_ontology(bundle_key="does_not_exist")


def test_load_ontology_with_path(tmp_path):
    """Explicit path still works for tests / admin tools."""
    import yaml
    from app.services.ontology_templates import load_ontology
    p = tmp_path / "fake.yaml"
    p.write_text(yaml.safe_dump({
        "entity_types": [{"name": "FOO"}],
        "relationship_types": [],
        "validation_matrix": [],
    }))
    ont = load_ontology(path=p)
    assert ont["entity_types"][0]["name"] == "FOO"


def test_load_ontology_no_longer_accepts_prefer_active():
    """prefer_active was dropped from the public signature."""
    from app.services.ontology_templates import load_ontology
    with pytest.raises(TypeError):
        load_ontology(prefer_active=True)


def test_load_registry_ontology_exists():
    """load_registry_ontology is a separate function for version-pinned loads."""
    from app.services.ontology_templates import load_registry_ontology
    import inspect
    sig = inspect.signature(load_registry_ontology)
    assert "version_id" in sig.parameters
