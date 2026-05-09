"""Fix A — RelationshipRecord carries provenance field.

Also covers the composite-key upgrade: per-edge provenance matching using
(from_identity, rel_type, to_identity) instead of rel_type alone.
"""
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


# ---------------------------------------------------------------------------
# Composite-key matching tests (Fix A upgrade)
# ---------------------------------------------------------------------------

class _FakeEntityProv:
    """Minimal stand-in for ExtractionProvenance."""
    def __init__(self, instance_id, ontology_name, identity_values):
        self.instance_id = instance_id
        self.ontology_name = ontology_name
        self.identity_values = identity_values


def test_instance_to_identity_map_basic():
    """_instance_to_identity_map builds valid LogicalIdentity entries."""
    from app.workers.pipeline import _instance_to_identity_map
    from app.services.extraction_merge import LogicalIdentity

    rows = [
        _FakeEntityProv("i_r1", "RADAR_SYSTEM", {"id": "R1"}),
        _FakeEntityProv("i_r2", "RADAR_SYSTEM", {"id": "R2"}),
        _FakeEntityProv("i_m1", "MISSILE", {"id": "M1"}),
    ]
    id_map = _instance_to_identity_map(rows)

    assert set(id_map.keys()) == {"i_r1", "i_r2", "i_m1"}
    assert isinstance(id_map["i_r1"], LogicalIdentity)
    assert id_map["i_r1"].entity_type == "RADAR_SYSTEM"
    assert id_map["i_m1"].entity_type == "MISSILE"
    # Same entity_type but different identity values must yield different identities.
    assert id_map["i_r1"] != id_map["i_r2"]


def test_instance_to_identity_map_skips_missing_fields():
    """Rows without instance_id or ontology_name are silently skipped."""
    from app.workers.pipeline import _instance_to_identity_map

    rows = [
        _FakeEntityProv(None, "RADAR_SYSTEM", {"id": "R1"}),   # no instance_id
        _FakeEntityProv("i_r2", None, {"id": "R2"}),            # no ontology_name
        _FakeEntityProv("i_ok", "MISSILE", {"id": "M1"}),       # valid
    ]
    id_map = _instance_to_identity_map(rows)
    assert list(id_map.keys()) == ["i_ok"]


def test_two_distinct_edges_get_separate_provenance():
    """Regression: all ENGAGES edges previously shared one rel_type bucket.

    After the composite-key fix, R1→M1 and R2→M2 each resolve to their own
    provenance bucket, so each RelationshipRecord carries distinct evidence.
    """
    from types import SimpleNamespace
    from app.workers.pipeline import _instance_to_identity_map
    from app.services.extraction_merge import LogicalIdentity

    # Build entity provenance rows for 4 entities.
    entity_rows = [
        _FakeEntityProv("i_r1", "RADAR_SYSTEM", {"id": "R1"}),
        _FakeEntityProv("i_r2", "RADAR_SYSTEM", {"id": "R2"}),
        _FakeEntityProv("i_m1", "MISSILE", {"id": "M1"}),
        _FakeEntityProv("i_m2", "MISSILE", {"id": "M2"}),
    ]
    id_map = _instance_to_identity_map(entity_rows)

    # Two relationship provenance rows for ENGAGES with distinct instance pairs.
    class _FakeRelProv:
        def __init__(self, rel_type, src_id, tgt_id, evidence_ids):
            self.relationship_type = rel_type
            self.source_instance_id = src_id
            self.target_instance_id = tgt_id
            self.evidence_ids = evidence_ids
            self.self_refs = []
            self.page_numbers = []

    rel_rows = [
        _FakeRelProv("ENGAGES", "i_r1", "i_m1", ["#/texts/10"]),
        _FakeRelProv("ENGAGES", "i_r2", "i_m2", ["#/texts/20"]),
    ]

    # Build provenance_by_triple the same way _import_graph_phase_domain_edges does.
    _FALLBACK = "__rel_type_fallback__"
    provenance_by_triple: dict = {}
    for row in rel_rows:
        src_id_str = str(row.source_instance_id) if row.source_instance_id else ""
        tgt_id_str = str(row.target_instance_id) if row.target_instance_id else ""
        src_identity = id_map.get(src_id_str)
        tgt_identity = id_map.get(tgt_id_str)
        if src_identity is not None and tgt_identity is not None:
            key = (src_identity, row.relationship_type, tgt_identity)
        else:
            key = (_FALLBACK, row.relationship_type)
        bucket = provenance_by_triple.setdefault(key, {
            "evidence_ids": [], "self_refs": [], "page_numbers": [],
        })
        bucket["evidence_ids"] = sorted(set(
            bucket["evidence_ids"] + list(row.evidence_ids or [])
        ))

    # Simulate two MergedEdgeRecord-like objects.
    r1_id = id_map["i_r1"]
    m1_id = id_map["i_m1"]
    r2_id = id_map["i_r2"]
    m2_id = id_map["i_m2"]

    prov_r1_m1 = provenance_by_triple.get((r1_id, "ENGAGES", m1_id))
    prov_r2_m2 = provenance_by_triple.get((r2_id, "ENGAGES", m2_id))

    assert prov_r1_m1 is not None, "R1→M1 provenance bucket not found"
    assert prov_r2_m2 is not None, "R2→M2 provenance bucket not found"
    assert prov_r1_m1["evidence_ids"] == ["#/texts/10"]
    assert prov_r2_m2["evidence_ids"] == ["#/texts/20"]
    # Crucially, they must be different objects (not the same merged bucket).
    assert prov_r1_m1 is not prov_r2_m2


def test_unresolvable_rows_fall_back_to_rel_type_bucket():
    """Rows whose instance_ids are not in the entity map use the fallback bucket."""
    from app.workers.pipeline import _instance_to_identity_map

    # No entity provenance → empty id_map → all rows go to fallback.
    id_map = _instance_to_identity_map([])

    class _FakeRelProv:
        def __init__(self, rel_type, src_id, tgt_id, evidence_ids):
            self.relationship_type = rel_type
            self.source_instance_id = src_id
            self.target_instance_id = tgt_id
            self.evidence_ids = evidence_ids
            self.self_refs = []
            self.page_numbers = []

    _FALLBACK = "__rel_type_fallback__"
    rel_rows = [
        _FakeRelProv("ENGAGES", "i_r1", "i_m1", ["#/texts/10"]),
        _FakeRelProv("ENGAGES", "i_r2", "i_m2", ["#/texts/20"]),
    ]

    provenance_by_triple: dict = {}
    for row in rel_rows:
        src_id_str = str(row.source_instance_id) if row.source_instance_id else ""
        tgt_id_str = str(row.target_instance_id) if row.target_instance_id else ""
        src_identity = id_map.get(src_id_str)
        tgt_identity = id_map.get(tgt_id_str)
        if src_identity is not None and tgt_identity is not None:
            key = (src_identity, row.relationship_type, tgt_identity)
        else:
            key = (_FALLBACK, row.relationship_type)
        bucket = provenance_by_triple.setdefault(key, {
            "evidence_ids": [], "self_refs": [], "page_numbers": [],
        })
        bucket["evidence_ids"] = sorted(set(
            bucket["evidence_ids"] + list(row.evidence_ids or [])
        ))

    # Both rows should have merged into the same fallback bucket.
    fallback = provenance_by_triple.get((_FALLBACK, "ENGAGES"))
    assert fallback is not None
    assert sorted(fallback["evidence_ids"]) == ["#/texts/10", "#/texts/20"]
