"""Fix A — RelationshipRecord carries provenance field."""
from __future__ import annotations


def test_relationship_record_carries_provenance():
    from app.services.graph_store import RelationshipRecord

    r = RelationshipRecord(
        from_type="RADAR",
        from_identity={"id": "R1"},
        to_type="MISSILE",
        to_identity={"id": "M1"},
        rel_type="ENGAGES",
        extraction_confidence=0.9,
        provenance={"evidence_ids": ["#/texts/0"]},
    )
    assert r.provenance == {"evidence_ids": ["#/texts/0"]}


def test_relationship_record_provenance_defaults_none():
    """Existing callers that omit provenance still work."""
    from app.services.graph_store import RelationshipRecord

    r = RelationshipRecord(
        from_type="RADAR",
        from_identity={"id": "R1"},
        to_type="MISSILE",
        to_identity={"id": "M1"},
        rel_type="DETECTS",
    )
    assert r.provenance is None
