"""Tests for the ``ONTOLOGY_SOURCE`` feature flag in ``load_ontology``.

Plan v32 Task 24 (Phase 3). The flag selects Pydantic introspection vs
legacy YAML for the default-lookup path. Explicit ``path=`` and
non-default ``bundle_key`` always use YAML regardless of the flag
(loader contract must stay stable for those code paths).

NOTE (Task 24b): this suite is transitional. Task 45 removes the
env-var read from ``load_ontology``; this file will be rewritten to
assert the two surviving code paths (default lookup vs explicit path)
at that time.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app.services.ontology_templates import (
    SYSTEM_DEFAULT_BUNDLE_KEY,
    invalidate_ontology_cache,
    load_ontology,
)
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


@pytest.fixture(autouse=True)
def _flush_cache():
    invalidate_ontology_cache()
    yield
    invalidate_ontology_cache()


def test_default_lookup_returns_introspection_matching_snapshot():
    """``load_ontology()`` with no args returns canonical-JSON-equivalent
    output to the frozen snapshot fixture (the oracle since plan Task 51
    deleted ontology_bundles/air_defense_v3/ontology.yaml)."""
    result = load_ontology()
    with ONTOLOGY_YAML.open() as f:
        yaml_truth = yaml.safe_load(f)

    canon_result = canonicalize_ontology_dict(result)
    canon_truth = canonicalize_ontology_dict(yaml_truth)
    assert json.dumps(canon_result, sort_keys=True) == json.dumps(
        canon_truth, sort_keys=True
    )


def test_explicit_bundle_key_matches_snapshot():
    """``bundle_key='air_defense_v3'`` mirrors the no-args behavior."""
    result = load_ontology(bundle_key=SYSTEM_DEFAULT_BUNDLE_KEY)
    with ONTOLOGY_YAML.open() as f:
        yaml_truth = yaml.safe_load(f)

    assert canonicalize_ontology_dict(result) == canonicalize_ontology_dict(yaml_truth)


def test_default_bundle_returns_introspection_not_yaml():
    """The default bundle path returns the Pydantic introspection dict
    — there is no YAML file to fall back to after Task 51."""
    introspected = build_ontology_dict()
    loaded = load_ontology()
    assert canonicalize_ontology_dict(loaded) == canonicalize_ontology_dict(introspected)


def test_explicit_path_loads_referenced_yaml_file():
    """Loader contract preserved: an explicit path= always reads that
    file as YAML. Fixtures, tooling, and migration helpers all rely
    on this path."""
    loaded = load_ontology(path=ONTOLOGY_YAML)
    with ONTOLOGY_YAML.open() as f:
        expected = yaml.safe_load(f)
    assert loaded == expected


def test_ontology_source_env_var_is_obsolete_for_default_bundle(monkeypatch):
    """Post-Task-51: ONTOLOGY_SOURCE has no effect on the default bundle
    path — introspection is unconditional. Setting it to 'yaml' does
    NOT re-engage a YAML loader (the bundle yaml no longer exists)."""
    monkeypatch.setenv("ONTOLOGY_SOURCE", "yaml")
    introspected = build_ontology_dict()
    loaded = load_ontology()
    # Same introspection dict whether env says 'yaml' or is unset.
    assert canonicalize_ontology_dict(loaded) == canonicalize_ontology_dict(introspected)


def test_non_default_bundle_key_falls_through_to_yaml(monkeypatch, tmp_path):
    """A non-default bundle_key must hit the YAML loader even when
    ONTOLOGY_SOURCE=pydantic — introspection only covers
    air_defense_v3."""
    monkeypatch.setenv("ONTOLOGY_SOURCE", "pydantic")

    # Construct a throwaway bundle at tmp_path to verify the YAML fallback
    # path. We repoint _BUNDLE_ROOT by symlink-copying the structure.
    from app.services import ontology_templates as otm

    fake_bundle_dir = tmp_path / "fake_bundle"
    fake_bundle_dir.mkdir()
    # Minimal YAML stub — just enough to load
    (fake_bundle_dir / "ontology.yaml").write_text(
        "version: '0.0.1'\nentity_types: []\nrelationship_types: []\n"
        "validation_matrix: []\nscoring_weights: {}\n"
    )

    monkeypatch.setattr(otm, "_BUNDLE_ROOT", tmp_path)
    otm.invalidate_ontology_cache()

    loaded = load_ontology(bundle_key="fake_bundle")
    assert loaded["version"] == "0.0.1"
    assert loaded["entity_types"] == []


def test_default_when_env_unset_is_pydantic(monkeypatch):
    """When ONTOLOGY_SOURCE is unset, the default is pydantic (flipped
    in Task 42). We assert via ordering: introspection emits entities in
    ALL_ENTITIES registry order (ORGANIZATION at position ~7), while YAML
    declared order has ORGANIZATION at ~18."""
    monkeypatch.delenv("ONTOLOGY_SOURCE", raising=False)
    loaded = load_ontology()
    names = [e["name"] for e in loaded["entity_types"]]
    org_index = names.index("ORGANIZATION")
    assert org_index < 10, (
        f"ORGANIZATION at index {org_index} suggests YAML order, "
        "not introspection order — default flag may not be 'pydantic'"
    )
