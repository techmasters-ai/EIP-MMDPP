"""Tests for layered extraction merge, dedup, and edge validation."""

import pytest
from app.services.layered_extraction import (
    merge_extraction_results,
    validate_edges,
    _canonical_key,
    _entity_fingerprint,
    _edge_key,
)


def test_canonical_key():
    assert _canonical_key({"entity_type": "RADAR_SYSTEM", "name": "Fan Song"}) == "RADAR_SYSTEM::fan song"
    assert _canonical_key({"entity_type": "radar_system", "name": "Fan Song"}) == "RADAR_SYSTEM::fan song"


def test_entity_dedup_across_passes():
    results = [
        ("layer_1", {
            "entities": [
                {"name": "SA-2", "entity_type": "MISSILE_SYSTEM", "confidence": 0.8, "properties": {}},
            ],
            "relationships": [],
        }),
        ("layer_2", {
            "entities": [
                {"name": "SA-2", "entity_type": "MISSILE_SYSTEM", "confidence": 0.9, "properties": {"range": "50km"}},
            ],
            "relationships": [],
        }),
    ]
    merged = merge_extraction_results(results)
    assert len(merged["entities"]) == 1
    assert merged["entities"][0]["confidence"] == 0.9  # higher confidence kept
    assert merged["entities"][0]["properties"]["range"] == "50km"  # property merged


def test_edge_dedup():
    results = [
        ("layer_1", {
            "entities": [],
            "relationships": [
                {"from_name": "SA-2", "to_name": "Fan Song", "relationship_type": "CUES", "confidence": 0.7},
            ],
        }),
        ("layer_2", {
            "entities": [],
            "relationships": [
                {"from_name": "SA-2", "to_name": "Fan Song", "relationship_type": "CUES", "confidence": 0.9},
            ],
        }),
    ]
    merged = merge_extraction_results(results)
    assert len(merged["relationships"]) == 1
    assert merged["relationships"][0]["confidence"] == 0.9


def test_pass_provenance():
    results = [
        ("layer_1", {
            "entities": [{"name": "Doc1", "entity_type": "DOCUMENT", "confidence": 0.8, "properties": {}}],
            "relationships": [],
        }),
    ]
    merged = merge_extraction_results(results)
    assert merged["entities"][0]["_extraction_pass"] == "layer_1"
    assert merged["metadata"]["extraction_mode"] == "layered"
    assert len(merged["metadata"]["passes"]) == 1


def test_validate_edges_endpoint_existence():
    entities = [
        {"name": "SA-2", "entity_type": "MISSILE_SYSTEM"},
        {"name": "Fan Song", "entity_type": "RADAR_SYSTEM"},
    ]
    rels = [
        {"from_name": "SA-2", "to_name": "Fan Song", "relationship_type": "CUES",
         "from_type": "MISSILE_SYSTEM", "to_type": "RADAR_SYSTEM"},
        {"from_name": "SA-2", "to_name": "DOES_NOT_EXIST", "relationship_type": "RELATED_TO",
         "from_type": "MISSILE_SYSTEM", "to_type": "UNKNOWN"},
    ]
    ontology = {"relationship_types": []}
    valid, rejected = validate_edges(rels, entities, ontology)
    assert len(valid) == 1
    assert len(rejected) == 1
    assert rejected[0]["_reject_reason"] == "endpoint_not_found"


def test_validate_edges_matrix():
    entities = [
        {"name": "SA-2", "entity_type": "MISSILE_SYSTEM"},
        {"name": "Fan Song", "entity_type": "RADAR_SYSTEM"},
    ]
    rels = [
        {"from_name": "SA-2", "to_name": "Fan Song", "relationship_type": "CUES",
         "from_type": "MISSILE_SYSTEM", "to_type": "RADAR_SYSTEM"},
    ]
    ontology = {
        "relationship_types": [
            {
                "name": "CUES",
                "validation_matrix": [
                    {"source_type": "RADAR_SYSTEM", "target_type": "MISSILE_SYSTEM"},
                ],
            },
        ],
    }
    # MISSILE_SYSTEM→RADAR_SYSTEM not in matrix (matrix has RADAR→MISSILE)
    valid, rejected = validate_edges(rels, entities, ontology, reject_invalid=True)
    assert len(valid) == 0
    assert len(rejected) == 1
    assert rejected[0]["_reject_reason"] == "invalid_triple"


def test_validate_edges_no_matrix_allows_all():
    entities = [
        {"name": "SA-2", "entity_type": "MISSILE_SYSTEM"},
        {"name": "Fan Song", "entity_type": "RADAR_SYSTEM"},
    ]
    rels = [
        {"from_name": "SA-2", "to_name": "Fan Song", "relationship_type": "CUES",
         "from_type": "MISSILE_SYSTEM", "to_type": "RADAR_SYSTEM"},
    ]
    ontology = {"relationship_types": []}
    valid, rejected = validate_edges(rels, entities, ontology, reject_invalid=True)
    assert len(valid) == 1
    assert len(rejected) == 0
