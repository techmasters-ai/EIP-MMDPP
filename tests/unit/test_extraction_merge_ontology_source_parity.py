"""Parity tests for ``extraction_merge`` consumers under both
``ONTOLOGY_SOURCE`` modes.

Plan v32 Task 25 (Phase 4). The helpers
``_build_logical_identity`` / ``logical_identity_from_dict`` /
``_is_valid_triple`` all read the ontology dict. Since
``build_ontology_dict()`` is canonical-JSON-equivalent to the YAML
load, no code change is needed — but the parity contract demands an
explicit test that exercises both sources and asserts identical
output.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.extraction_merge import (
    _build_logical_identity,
    _is_valid_triple,
    logical_identity_from_dict,
)
from app.services.ontology_templates import (
    invalidate_ontology_cache,
    load_ontology,
)


@pytest.fixture(autouse=True)
def _flush_cache():
    invalidate_ontology_cache()
    yield
    invalidate_ontology_cache()


@pytest.fixture(params=["yaml", "pydantic"])
def ontology(request, monkeypatch):
    monkeypatch.setenv("ONTOLOGY_SOURCE", request.param)
    return load_ontology()


def test_build_logical_identity_section(ontology):
    """SECTION has identity_fields=[section_number], scope=document (post-B-4).
    Logical identity extraction must be identical under both sources."""
    instance = SimpleNamespace(section_number="3.2.1")
    identity = _build_logical_identity("SECTION", instance, ontology, document_id="doc-1")
    assert identity is not None
    assert identity.entity_type == "SECTION"
    assert identity.identity_field_names == ("section_number",)
    assert identity.identity_tuple == ("3.2.1",)
    assert identity.scope == "document"
    assert identity.document_id == "doc-1"


def test_build_logical_identity_platform(ontology):
    """PLATFORM has identity_fields=[name], scope=global → no document_id."""
    instance = SimpleNamespace(name="SA-20 TEL")
    identity = _build_logical_identity("PLATFORM", instance, ontology, document_id="doc-1")
    assert identity is not None
    assert identity.scope == "global"
    assert identity.document_id is None


def test_build_logical_identity_unknown_type_returns_none(ontology):
    assert (
        _build_logical_identity("NOT_A_TYPE", SimpleNamespace(), ontology, "doc-1")
        is None
    )


def test_logical_identity_from_dict_section(ontology):
    identity = logical_identity_from_dict(
        "SECTION",
        {"section_number": "3.2.1"},
        ontology,
        document_id="doc-1",
    )
    assert identity is not None
    assert identity.identity_tuple == ("3.2.1",)


def test_logical_identity_from_dict_missing_key_returns_none(ontology):
    assert (
        logical_identity_from_dict(
            "SECTION",
            {"heading": "Maintenance"},  # missing section_number
            ontology,
            document_id="doc-1",
        )
        is None
    )


def test_is_valid_triple_known_pairs(ontology):
    assert _is_valid_triple(ontology, "RADAR_SYSTEM", "INSTALLED_ON", "PLATFORM")
    assert _is_valid_triple(ontology, "MISSILE_SYSTEM", "HAS_SEEKER", "SEEKER")


def test_is_valid_triple_rejects_unknown(ontology):
    assert not _is_valid_triple(ontology, "FOO", "BAR", "BAZ")
    assert not _is_valid_triple(ontology, "RADAR_SYSTEM", "NOT_A_REL", "PLATFORM")
