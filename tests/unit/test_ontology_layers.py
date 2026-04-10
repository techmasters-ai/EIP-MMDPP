"""Tests for ontology layer subset builder."""

import pytest
from app.services.ontology_layers import (
    build_layer_subset,
    build_cross_layer_subset,
    build_all_layer_subsets,
    load_layer_map,
)


@pytest.fixture
def sample_ontology():
    return {
        "entity_types": [
            {"name": "DOCUMENT", "properties": {"type": "object", "properties": {}}},
            {"name": "SECTION", "properties": {"type": "object", "properties": {}}},
            {"name": "PLATFORM", "properties": {"type": "object", "properties": {}}},
            {"name": "RADAR_SYSTEM", "properties": {"type": "object", "properties": {}}},
            {"name": "ANTENNA", "properties": {"type": "object", "properties": {}}},
        ],
        "relationship_types": [
            {
                "name": "CONTAINS",
                "validation_matrix": [
                    {"source_type": "DOCUMENT", "target_type": "SECTION"},
                ],
            },
            {
                "name": "HAS_ANTENNA",
                "validation_matrix": [
                    {"source_type": "RADAR_SYSTEM", "target_type": "ANTENNA"},
                ],
            },
            {
                "name": "INSTALLED_ON",
                "validation_matrix": [
                    {"source_type": "RADAR_SYSTEM", "target_type": "PLATFORM"},
                ],
            },
        ],
    }


@pytest.fixture
def sample_layer_map():
    return {
        1: ["DOCUMENT", "SECTION"],
        2: ["PLATFORM", "RADAR_SYSTEM"],
        3: ["ANTENNA"],
    }


def test_build_layer_subset_entities(sample_ontology, sample_layer_map):
    subset = build_layer_subset(sample_ontology, 1, sample_layer_map)
    names = {e["name"] for e in subset["entity_types"]}
    assert names == {"DOCUMENT", "SECTION"}


def test_build_layer_subset_within_layer_rels(sample_ontology, sample_layer_map):
    subset = build_layer_subset(sample_ontology, 1, sample_layer_map)
    rel_names = {r["name"] for r in subset["relationship_types"]}
    assert "CONTAINS" in rel_names  # DOCUMENT→SECTION both in layer 1
    assert "HAS_ANTENNA" not in rel_names  # cross-layer
    assert "INSTALLED_ON" not in rel_names  # cross-layer


def test_build_layer_subset_empty_layer(sample_ontology, sample_layer_map):
    subset = build_layer_subset(sample_ontology, 99, sample_layer_map)
    assert subset["entity_types"] == []
    assert subset["relationship_types"] == []


def test_build_cross_layer_subset(sample_ontology, sample_layer_map):
    subset = build_cross_layer_subset(sample_ontology, sample_layer_map)
    rel_names = {r["name"] for r in subset["relationship_types"]}
    # INSTALLED_ON: RADAR_SYSTEM(layer2) → PLATFORM(layer2) — same layer, excluded
    # HAS_ANTENNA: RADAR_SYSTEM(layer2) → ANTENNA(layer3) — cross-layer, included
    assert "HAS_ANTENNA" in rel_names
    assert "CONTAINS" not in rel_names  # both in layer 1


def test_build_all_layer_subsets(sample_ontology, sample_layer_map):
    subsets = build_all_layer_subsets(sample_ontology, sample_layer_map)
    names = [name for name, _ in subsets]
    assert "layer_1" in names
    assert "layer_2" in names
    assert "layer_3" in names
    assert "cross_layer" in names


def test_load_layer_map():
    layer_map = load_layer_map()
    assert 1 in layer_map
    assert 5 in layer_map
    assert "DOCUMENT" in layer_map[1]
    assert "PLATFORM" in layer_map[2]
