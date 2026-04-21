"""Enum + matrix additions for the docling-anchor set.

See docs/plans/2026-04-21-document-structure-pass-design.md §3.
"""
from __future__ import annotations

from ontology_bundles.air_defense_v3.relationships import (
    RelationshipType,
    RELATIONSHIP_METADATA,
)
from ontology_bundles.air_defense_v3.validation_matrix import VALIDATION_MATRIX


def test_anchor_enum_members_present():
    for name in (
        "HAS_SECTION", "HAS_FIGURE", "HAS_TABLE",
        "CHILD_OF", "HAS_IMAGE", "NEAR_TEXT",
    ):
        assert hasattr(RelationshipType, name), f"RelationshipType.{name} missing"


def test_anchor_metadata_descriptors_present():
    for name in (
        "HAS_SECTION", "HAS_FIGURE", "HAS_TABLE",
        "CHILD_OF", "HAS_IMAGE", "NEAR_TEXT",
    ):
        assert RelationshipType[name] in RELATIONSHIP_METADATA, (
            f"RELATIONSHIP_METADATA missing descriptor for {name}"
        )


def test_validation_matrix_has_10_new_triples():
    expected = {
        ("DOCUMENT", RelationshipType.HAS_SECTION, "SECTION"),
        ("DOCUMENT", RelationshipType.HAS_FIGURE,  "FIGURE"),
        ("DOCUMENT", RelationshipType.HAS_TABLE,   "TABLE"),
        ("SECTION",  RelationshipType.CHILD_OF,    "SECTION"),
        ("SECTION",  RelationshipType.HAS_FIGURE,  "FIGURE"),
        ("SECTION",  RelationshipType.HAS_TABLE,   "TABLE"),
        ("SECTION",  RelationshipType.HAS_IMAGE,   "IMAGE"),
        ("DOCUMENT", RelationshipType.HAS_IMAGE,   "IMAGE"),
        ("FIGURE",   RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
        ("IMAGE",    RelationshipType.NEAR_TEXT,   "TEXT_BLOCK"),
    }
    assert expected <= VALIDATION_MATRIX, (
        f"Missing triples: {expected - VALIDATION_MATRIX}"
    )
