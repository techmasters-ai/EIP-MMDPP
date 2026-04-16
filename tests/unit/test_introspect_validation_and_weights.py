"""Parity tests for ``build_validation_matrix_list`` + ``build_scoring_weights``.

Plan v32 Task 22 (Phase 3). Pydantic introspection must match the YAML
tables after the canonicalizer step (sorted triples for the matrix;
dict equality for the weights). The YAML has one known duplicate in
``validation_matrix`` (``RADAR_SYSTEM, INSTALLED_ON, PLATFORM``); the
frozenset dedupes it, so the YAML side is deduped before comparison.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ontology_bundles.air_defense_v3.introspect import (
    build_scoring_weights,
    build_validation_matrix_list,
)

ONTOLOGY_YAML = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "ontology"
    / "air_defense_v3_snapshot.yaml"
)


@pytest.fixture(scope="module")
def yaml_ont() -> dict:
    with ONTOLOGY_YAML.open() as f:
        return yaml.safe_load(f)


def _canonicalize_matrix(entries: list[dict]) -> list[tuple[str, str, str]]:
    """Sort + dedup a validation_matrix list for stable comparison."""
    unique = {
        (e["source"], e["relationship"], e["target"]) for e in entries
    }
    return sorted(unique)


def test_validation_matrix_parity(yaml_ont):
    yaml_matrix = yaml_ont["validation_matrix"]
    pyd_matrix = build_validation_matrix_list()

    yaml_canon = _canonicalize_matrix(yaml_matrix)
    pyd_canon = _canonicalize_matrix(pyd_matrix)

    assert yaml_canon == pyd_canon


def test_validation_matrix_dedup_count():
    """The frozenset has 127 unique entries (YAML has 128 with 1 dup)."""
    assert len(build_validation_matrix_list()) == 127


def test_validation_matrix_sorted_by_triple():
    """Output is sorted by (source, relationship, target)."""
    entries = build_validation_matrix_list()
    triples = [(e["source"], e["relationship"], e["target"]) for e in entries]
    assert triples == sorted(triples)


def test_validation_matrix_entry_shape():
    """Each entry has exactly keys: source, relationship, target."""
    for e in build_validation_matrix_list():
        assert set(e.keys()) == {"source", "relationship", "target"}


def test_scoring_weights_parity(yaml_ont):
    yaml_weights = yaml_ont["scoring_weights"]
    pyd_weights = build_scoring_weights()
    assert pyd_weights == yaml_weights


def test_scoring_weights_default_present():
    weights = build_scoring_weights()
    assert "default" in weights
    assert weights["default"] == 0.70
