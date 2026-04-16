"""Canonical-JSON parity test for ``build_ontology_dict`` vs YAML load.

Plan v32 Task 23 (Phase 3). Both sides pass through
``canonicalize_ontology_dict`` (sort keys, sort list elements by stable
key, dedupe validation_matrix) then compare via ``json.dumps``.
Eliminates flake from frozenset ordering, dict iteration order, or
the known YAML duplicate triple.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ontology_bundles.air_defense_v3.introspect import (
    build_ontology_dict,
    canonicalize_ontology_dict,
)

ONTOLOGY_YAML = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "ontology"
    / "air_defense_v3_snapshot.yaml"
)


@pytest.fixture(scope="module")
def yaml_ontology() -> dict:
    with ONTOLOGY_YAML.open() as f:
        return yaml.safe_load(f)


def test_ontology_dict_has_all_top_level_keys():
    d = build_ontology_dict()
    assert set(d.keys()) == {
        "version",
        "entity_types",
        "relationship_types",
        "validation_matrix",
        "scoring_weights",
    }


def test_ontology_dict_version_matches_manifest():
    """ONTOLOGY_VERSION must mirror manifest.yaml:ontology_version."""
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "ontology_bundles"
        / "air_defense_v3"
        / "manifest.yaml"
    )
    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)
    assert build_ontology_dict()["version"] == manifest["ontology_version"]


def test_canonical_json_parity(yaml_ontology):
    """Both dicts canonicalize to byte-identical JSON."""
    pyd = canonicalize_ontology_dict(build_ontology_dict())
    yml = canonicalize_ontology_dict(yaml_ontology)

    pyd_json = json.dumps(pyd, sort_keys=True)
    yml_json = json.dumps(yml, sort_keys=True)

    assert pyd_json == yml_json, (
        "canonical-JSON parity failed. First divergent slice:\n"
        f"pydantic[:200]: {pyd_json[:200]}\n"
        f"yaml[:200]:     {yml_json[:200]}"
    )


def test_canonicalizer_sorts_entity_types(yaml_ontology):
    canon = canonicalize_ontology_dict(yaml_ontology)
    names = [e["name"] for e in canon["entity_types"]]
    assert names == sorted(names)


def test_canonicalizer_sorts_relationship_types(yaml_ontology):
    canon = canonicalize_ontology_dict(yaml_ontology)
    names = [e["name"] for e in canon["relationship_types"]]
    assert names == sorted(names)


def test_canonicalizer_dedupes_validation_matrix(yaml_ontology):
    """YAML has one duplicate triple; canonicalizer must collapse it."""
    canon = canonicalize_ontology_dict(yaml_ontology)
    triples = [
        (e["source"], e["relationship"], e["target"]) for e in canon["validation_matrix"]
    ]
    assert len(triples) == len(set(triples))
    assert len(triples) == 127


def test_canonicalizer_sorts_property_fields(yaml_ontology):
    """Each entity's properties.properties sorted by field name."""
    canon = canonicalize_ontology_dict(yaml_ontology)
    for entity in canon["entity_types"]:
        props = entity["properties"]["properties"]
        assert list(props.keys()) == sorted(props.keys())
