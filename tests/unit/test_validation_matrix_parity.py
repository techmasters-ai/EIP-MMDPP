"""Parity tests: VALIDATION_MATRIX + SCORING_WEIGHTS mirror ontology.yaml 1:1.

Gates Plan v32 Tasks 12 (VALIDATION_MATRIX frozenset) and 13 (SCORING_WEIGHTS).
"""
from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from ontology_bundles.air_defense_v3.relationships import RelationshipType
from ontology_bundles.air_defense_v3.validation_matrix import (
    VALIDATION_MATRIX,
    SCORING_WEIGHTS,
)


@pytest.fixture(scope="module")
def yaml_ontology() -> dict:
    path = (
        Path(__file__).parent.parent.parent
        / "tests"
        / "fixtures"
        / "ontology"
        / "air_defense_v3_snapshot.yaml"
    )
    with path.open() as f:
        return yaml.safe_load(f)


def test_validation_matrix_mirrors_yaml_exactly(yaml_ontology):
    """VALIDATION_MATRIX must exactly equal the unique set of YAML triples."""
    yaml_triples = {
        (t["source"], RelationshipType(t["relationship"]), t["target"])
        for t in yaml_ontology["validation_matrix"]
    }
    missing_in_python = yaml_triples - VALIDATION_MATRIX
    extra_in_python = VALIDATION_MATRIX - yaml_triples
    assert not missing_in_python, (
        f"VALIDATION_MATRIX missing YAML triples: {sorted(missing_in_python)}"
    )
    assert not extra_in_python, (
        f"VALIDATION_MATRIX has triples not in YAML: {sorted(extra_in_python)}"
    )


def test_validation_matrix_is_frozenset():
    assert isinstance(VALIDATION_MATRIX, frozenset), (
        "VALIDATION_MATRIX must be a frozenset to guarantee immutability "
        "and hash stability."
    )


def test_validation_matrix_relationship_column_is_enum():
    """Every triple's middle column must be a RelationshipType member, not str."""
    for src, rel, tgt in VALIDATION_MATRIX:
        assert isinstance(rel, RelationshipType), (
            f"({src}, {rel!r}, {tgt}): middle column is {type(rel).__name__}, "
            "not RelationshipType"
        )


def test_validation_matrix_triple_count(yaml_ontology):
    """Sanity: 127 unique triples (YAML lists 128 with one duplicate)."""
    assert len(VALIDATION_MATRIX) == 127, (
        f"VALIDATION_MATRIX has {len(VALIDATION_MATRIX)} triples; "
        f"expected 127 unique (YAML lists {len(yaml_ontology['validation_matrix'])} "
        "with one known duplicate RADAR_SYSTEM × INSTALLED_ON × PLATFORM)."
    )


def test_scoring_weights_mirrors_yaml_exactly(yaml_ontology):
    yaml_weights = dict(yaml_ontology["scoring_weights"])
    assert SCORING_WEIGHTS == yaml_weights, (
        f"SCORING_WEIGHTS drift:\n"
        f"  only in Python: {set(SCORING_WEIGHTS) - set(yaml_weights)}\n"
        f"  only in YAML:   {set(yaml_weights) - set(SCORING_WEIGHTS)}\n"
        f"  value drift:    "
        + repr({k: (SCORING_WEIGHTS.get(k), yaml_weights.get(k))
                for k in set(SCORING_WEIGHTS) & set(yaml_weights)
                if SCORING_WEIGHTS.get(k) != yaml_weights.get(k)})
    )


def test_scoring_weights_default_key_present():
    """A 'default' key is load-bearing — retrieval fallback uses it when no
    relationship-specific weight exists. Asserting it explicitly makes the
    contract visible to any future refactor."""
    assert "default" in SCORING_WEIGHTS
    assert 0.0 < SCORING_WEIGHTS["default"] <= 1.0
