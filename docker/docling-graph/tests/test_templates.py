"""Tests for the YAML-to-Pydantic template generator."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

# Resolve paths relative to this file: tests/ -> docker/docling-graph/ -> docker/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ONTOLOGY_PATH = _REPO_ROOT / "ontology" / "ontology.yaml"

# Import the module under test — add the app dir to sys.path if needed
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from templates import GROUP_MAP, build_templates, load_ontology, _build_entity_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ontology():
    return load_ontology(_ONTOLOGY_PATH)


@pytest.fixture()
def templates(ontology):
    return build_templates(ontology)


# ---------------------------------------------------------------------------
# GROUP_MAP tests
# ---------------------------------------------------------------------------

def test_group_map_has_five_groups():
    assert len(GROUP_MAP) == 5
    assert set(GROUP_MAP.keys()) == {
        "reference",
        "equipment",
        "rf_signal",
        "weapon",
        "operational",
    }


@pytest.mark.skipif(
    not _ONTOLOGY_PATH.exists(),
    reason="Ontology YAML not found on host",
)
def test_all_ontology_types_covered():
    """Every entity type in the ontology must appear in exactly one group."""
    with open(_ONTOLOGY_PATH) as fh:
        ont = yaml.safe_load(fh)

    ontology_names = {et["name"] for et in ont.get("entity_types", [])}
    group_names: set[str] = set()
    for members in GROUP_MAP.values():
        for name in members:
            assert name not in group_names, f"{name} appears in more than one group"
            group_names.add(name)

    assert group_names == ontology_names, (
        f"Mismatch — in groups but not ontology: {group_names - ontology_names}, "
        f"in ontology but not groups: {ontology_names - group_names}"
    )


# ---------------------------------------------------------------------------
# build_templates() return shape
# ---------------------------------------------------------------------------

def test_returns_dict_of_dicts_keyed_by_group(templates):
    assert isinstance(templates, dict)
    assert set(templates.keys()) == set(GROUP_MAP.keys())
    for group_name, group_models in templates.items():
        assert isinstance(group_models, dict), f"{group_name} should be a dict"


def test_each_entity_model_is_pydantic(templates):
    for group_name, group_models in templates.items():
        for entity_name, model in group_models.items():
            assert isinstance(model, type), f"{group_name}/{entity_name} is not a type"
            assert issubclass(model, BaseModel), f"{group_name}/{entity_name} is not a BaseModel"


# ---------------------------------------------------------------------------
# Group model structure
# ---------------------------------------------------------------------------

def test_equipment_group_has_all_entity_types(templates):
    equipment = templates["equipment"]
    for entity_name in GROUP_MAP["equipment"]:
        assert entity_name in equipment, f"Missing entity {entity_name} in equipment group"


def test_entity_models_have_optional_fields(templates):
    """Instantiating any entity model with no args should work (all fields optional)."""
    for group_name, group_models in templates.items():
        for entity_name, model_cls in group_models.items():
            instance = model_cls()  # Should not raise
            for field_name in model_cls.model_fields:
                assert getattr(instance, field_name) is None, (
                    f"{group_name}/{entity_name}.{field_name} should default to None"
                )


def test_entity_models_have_graph_metadata(templates):
    for group_name, group_models in templates.items():
        for entity_name, model_cls in group_models.items():
            assert model_cls.model_config.get("is_entity") is True, (
                f"{group_name}/{entity_name} missing is_entity=True"
            )
