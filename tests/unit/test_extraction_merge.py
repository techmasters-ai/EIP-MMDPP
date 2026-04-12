"""Tests for app/services/extraction_merge.py.
Spec §3.6 + §3.7 + §3.9 + §6.2 + §6.7, and satisfies checker rule 14
(every RelationshipRejectionReason enum value has at least one test fixture)."""
import pytest

from app.services.extraction_merge import (
    ExtractionMetadata,
    RelationshipRejectionReason,
    LogicalIdentity,
    PassResult,
    MergedEntityRecord,
    MergedEdgeRecord,
    MergedExtraction,
    ChunkForDerivation,
    DerivedEdge,
    merge_and_resolve,
    build_display_label,
    classify_yield,
    classify_yield_from_counts,
    YieldStatus,
)


# --- Shared fixtures -------------------------------------------------------

MINIMAL_ONTOLOGY = {
    "entity_types": [
        {
            "name": "RADAR_SYSTEM",
            "identity_fields": ["system_name"],
            "identity_scope": "global",
            "properties": ["system_name", "nomenclature"],
        },
        {
            "name": "PLATFORM",
            "identity_fields": ["name"],
            "identity_scope": "global",
            "properties": ["name"],
        },
        {
            "name": "SECTION",
            "identity_fields": ["heading", "page_start"],
            "identity_scope": "document",
            "properties": ["heading", "page_start", "page_end"],
        },
    ],
    "validation_matrix": [
        {"source": "RADAR_SYSTEM", "relationship": "INSTALLED_ON", "target": "PLATFORM"},
    ],
}


def _make_pass_result(pass_name, entities, relationships, rejections=None):
    """Build a PassResult with a SimpleNamespace-based template stub.

    `entities` is a list of (type_name, identity_dict, properties_dict).
    `relationships` is a list of dicts convertible to SimpleNamespace.
    """
    from types import SimpleNamespace

    entities_by_list_attr: dict[str, list] = {}
    for ent_type, identity, props in entities:
        # Use the test-fixture convention: '<lower>_list'
        list_attr = f"{ent_type.lower()}_list"
        entities_by_list_attr.setdefault(list_attr, []).append(
            SimpleNamespace(**{**identity, **props})
        )

    template = SimpleNamespace(
        **entities_by_list_attr,
        relationships=[SimpleNamespace(**r) for r in relationships],
    )

    return PassResult(
        pass_name=pass_name,
        template_instance=template,
        metadata=ExtractionMetadata(
            schema_size_chars=1000,
            structured_output_mode="strict",
        ),
        pre_merge_rejections=rejections or [],
    )


def _fake_manifest(pass_names):
    """Minimal BundleManifest-ish stub; merge_and_resolve only needs
    the passes iterable with name attributes."""
    from types import SimpleNamespace
    return SimpleNamespace(
        bundle_key="air_defense_v3",
        passes=[
            SimpleNamespace(
                name=name,
                primary_entity_types=[],
                bridge_entity_types=[],
                extracted_relationship_types=[],
                kind="entities_and_relationships",
            )
            for name in pass_names
        ],
    )


# --- Rule 14 coverage: one test per RelationshipRejectionReason enum value ---

def test_rejection_invalid_triple():
    pass_result = _make_pass_result(
        pass_name="radar_domain",
        entities=[
            ("RADAR_SYSTEM", {"system_name": "S-400"}, {"nomenclature": "Triumf"}),
            ("PLATFORM", {"name": "Truck-1"}, {}),
        ],
        relationships=[{
            "rel_type": "LOVES",  # not in validation_matrix
            "from_identity": {"system_name": "S-400"},
            "to_identity": {"name": "Truck-1"},
            "from_type": "RADAR_SYSTEM",
            "to_type": "PLATFORM",
            "confidence": 0.9,
        }],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_result},
        manifest=_fake_manifest(["radar_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    assert len(merged.edges) == 0
    assert any(
        reason == RelationshipRejectionReason.INVALID_TRIPLE
        for _, _, reason in merged.rejected_edges
    )
    assert merged.rejections_by_pass["radar_domain"] == 1


def test_rejection_missing_rel_type():
    pass_result = _make_pass_result(
        pass_name="radar_domain",
        entities=[
            ("RADAR_SYSTEM", {"system_name": "S-400"}, {}),
            ("PLATFORM", {"name": "Truck-1"}, {}),
        ],
        relationships=[{
            "rel_type": None,
            "from_identity": {"system_name": "S-400"},
            "to_identity": {"name": "Truck-1"},
            "from_type": "RADAR_SYSTEM",
            "to_type": "PLATFORM",
            "confidence": 0.9,
        }],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_result},
        manifest=_fake_manifest(["radar_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    assert len(merged.edges) == 0
    assert any(
        reason == RelationshipRejectionReason.MISSING_REL_TYPE
        for _, _, reason in merged.rejected_edges
    )


def test_rejection_invalid_identity_payload():
    """from_identity missing the required identity_field key."""
    pass_result = _make_pass_result(
        pass_name="radar_domain",
        entities=[
            ("RADAR_SYSTEM", {"system_name": "S-400"}, {}),
            ("PLATFORM", {"name": "Truck-1"}, {}),
        ],
        relationships=[{
            "rel_type": "INSTALLED_ON",
            "from_identity": {},  # missing 'system_name'
            "to_identity": {"name": "Truck-1"},
            "from_type": "RADAR_SYSTEM",
            "to_type": "PLATFORM",
            "confidence": 0.9,
        }],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_result},
        manifest=_fake_manifest(["radar_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    assert any(
        reason == RelationshipRejectionReason.INVALID_IDENTITY_PAYLOAD
        for _, _, reason in merged.rejected_edges
    )


def test_rejection_from_endpoint_not_found():
    """The from-side entity is never emitted in any pass."""
    pass_result = _make_pass_result(
        pass_name="radar_domain",
        entities=[
            # Only the PLATFORM is emitted — the RADAR_SYSTEM is not.
            ("PLATFORM", {"name": "Truck-1"}, {}),
        ],
        relationships=[{
            "rel_type": "INSTALLED_ON",
            "from_identity": {"system_name": "Unemitted"},
            "to_identity": {"name": "Truck-1"},
            "from_type": "RADAR_SYSTEM",
            "to_type": "PLATFORM",
            "confidence": 0.9,
        }],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_result},
        manifest=_fake_manifest(["radar_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    assert any(
        reason == RelationshipRejectionReason.FROM_ENDPOINT_NOT_FOUND
        for _, _, reason in merged.rejected_edges
    )


def test_rejection_to_endpoint_not_found():
    """The to-side entity is never emitted."""
    pass_result = _make_pass_result(
        pass_name="radar_domain",
        entities=[
            ("RADAR_SYSTEM", {"system_name": "S-400"}, {}),
            # No PLATFORM emitted.
        ],
        relationships=[{
            "rel_type": "INSTALLED_ON",
            "from_identity": {"system_name": "S-400"},
            "to_identity": {"name": "Unemitted-Platform"},
            "from_type": "RADAR_SYSTEM",
            "to_type": "PLATFORM",
            "confidence": 0.9,
        }],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_result},
        manifest=_fake_manifest(["radar_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    assert any(
        reason == RelationshipRejectionReason.TO_ENDPOINT_NOT_FOUND
        for _, _, reason in merged.rejected_edges
    )


def test_rejection_unknown_ref_id():
    """system_links-style: from_ref_id/to_ref_id without matching upstream."""
    # Use from_ref_id / to_ref_id (NOT from_identity / to_identity) on the
    # relationship. The pass_result needs upstream_refs to be empty so
    # the ref_id lookup fails.
    pass_result = _make_pass_result(
        pass_name="system_links",
        entities=[],
        relationships=[{
            "rel_type": "INSTALLED_ON",
            "from_ref_id": "E01",  # no upstream_refs at all
            "to_ref_id": "E02",
            "confidence": 0.9,
        }],
    )
    # upstream_refs empty — use a new PassResult field or sidestep via manifest
    pass_result.upstream_refs = {}
    merged = merge_and_resolve(
        pass_results={"system_links": pass_result},
        manifest=_fake_manifest(["system_links"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    assert any(
        reason == RelationshipRejectionReason.UNKNOWN_REF_ID
        for _, _, reason in merged.rejected_edges
    )


def test_every_rejection_reason_has_a_test():
    """Checker rule 14: every enum value has at least one test fixture above.

    This is a meta-check — if a new enum value is added without a test,
    this test fails loudly.
    """
    enum_values = set(RelationshipRejectionReason)
    # The rejection tests above exercise these:
    covered = {
        RelationshipRejectionReason.INVALID_TRIPLE,
        RelationshipRejectionReason.MISSING_REL_TYPE,
        RelationshipRejectionReason.INVALID_IDENTITY_PAYLOAD,
        RelationshipRejectionReason.FROM_ENDPOINT_NOT_FOUND,
        RelationshipRejectionReason.TO_ENDPOINT_NOT_FOUND,
        RelationshipRejectionReason.UNKNOWN_REF_ID,
    }
    missing = enum_values - covered
    assert not missing, f"Missing test fixtures for: {missing}"


# --- Merge key properties --------------------------------------------------

def test_merge_bridge_entity_collapses_across_passes():
    """Same LogicalIdentity from two passes collapses into one MergedEntityRecord."""
    pass_a = _make_pass_result(
        pass_name="radar_domain",
        entities=[("PLATFORM", {"name": "Truck-1"}, {})],
        relationships=[],
    )
    pass_b = _make_pass_result(
        pass_name="missile_domain",
        entities=[("PLATFORM", {"name": "Truck-1"}, {})],
        relationships=[],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_a, "missile_domain": pass_b},
        manifest=_fake_manifest(["radar_domain", "missile_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    platforms = [e for e in merged.entities if e.identity.entity_type == "PLATFORM"]
    assert len(platforms) == 1
    assert platforms[0].pass_origins == {"radar_domain", "missile_domain"}


def test_merge_confidence_defaults_to_08_when_none():
    """confidence = 0.8 when relationship.confidence is None."""
    pass_result = _make_pass_result(
        pass_name="radar_domain",
        entities=[
            ("RADAR_SYSTEM", {"system_name": "S-400"}, {}),
            ("PLATFORM", {"name": "Truck-1"}, {}),
        ],
        relationships=[{
            "rel_type": "INSTALLED_ON",
            "from_identity": {"system_name": "S-400"},
            "to_identity": {"name": "Truck-1"},
            "from_type": "RADAR_SYSTEM",
            "to_type": "PLATFORM",
            "confidence": None,
        }],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_result},
        manifest=_fake_manifest(["radar_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    assert len(merged.edges) == 1
    assert merged.edges[0].confidence == 0.8


def test_merge_confidence_preserves_explicit_zero():
    """Regression: explicit 0.0 must NOT be coerced to 0.8."""
    pass_result = _make_pass_result(
        pass_name="radar_domain",
        entities=[
            ("RADAR_SYSTEM", {"system_name": "S-400"}, {}),
            ("PLATFORM", {"name": "Truck-1"}, {}),
        ],
        relationships=[{
            "rel_type": "INSTALLED_ON",
            "from_identity": {"system_name": "S-400"},
            "to_identity": {"name": "Truck-1"},
            "from_type": "RADAR_SYSTEM",
            "to_type": "PLATFORM",
            "confidence": 0.0,
        }],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_result},
        manifest=_fake_manifest(["radar_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    assert len(merged.edges) == 1
    assert merged.edges[0].confidence == 0.0


def test_merge_happy_path():
    """A valid single-pass extraction produces a MergedExtraction with
    entities, edges, and no rejections."""
    pass_result = _make_pass_result(
        pass_name="radar_domain",
        entities=[
            ("RADAR_SYSTEM", {"system_name": "S-400"}, {"nomenclature": "Triumf"}),
            ("PLATFORM", {"name": "Truck-1"}, {}),
        ],
        relationships=[{
            "rel_type": "INSTALLED_ON",
            "from_identity": {"system_name": "S-400"},
            "to_identity": {"name": "Truck-1"},
            "from_type": "RADAR_SYSTEM",
            "to_type": "PLATFORM",
            "confidence": 0.95,
        }],
    )
    merged = merge_and_resolve(
        pass_results={"radar_domain": pass_result},
        manifest=_fake_manifest(["radar_domain"]),
        ontology=MINIMAL_ONTOLOGY,
        document_id="doc-1",
        pipeline_run_id="run-1",
    )
    assert len(merged.entities) == 2
    assert len(merged.edges) == 1
    assert merged.edges[0].confidence == 0.95
    assert merged.edges[0].rel_type == "INSTALLED_ON"
    assert merged.rejected_edges == []
    assert merged.document_id == "doc-1"
    assert merged.pipeline_run_id == "run-1"


# --- LogicalIdentity helpers -----------------------------------------------

def test_logical_identity_as_upsert_dict_global_scope():
    li = LogicalIdentity(
        entity_type="RADAR_SYSTEM",
        identity_field_names=("system_name",),
        identity_tuple=("S-400",),
        scope="global",
        document_id=None,
    )
    d = li.as_upsert_identity_dict()
    assert d == {"system_name": "S-400"}


def test_logical_identity_as_upsert_dict_document_scope_adds_document_id():
    li = LogicalIdentity(
        entity_type="SECTION",
        identity_field_names=("heading", "page_start"),
        identity_tuple=("Intro", 1),
        scope="document",
        document_id="doc-xyz",
    )
    d = li.as_upsert_identity_dict()
    assert d == {"heading": "Intro", "page_start": 1, "document_id": "doc-xyz"}


def test_logical_identity_document_scope_requires_document_id():
    li = LogicalIdentity(
        entity_type="SECTION",
        identity_field_names=("heading",),
        identity_tuple=("Intro",),
        scope="document",
        document_id=None,
    )
    with pytest.raises(AssertionError):
        li.as_upsert_identity_dict()


def test_logical_identity_is_hashable():
    """LogicalIdentity must be frozen so it can be used as a dict key
    in the entity merge index."""
    li = LogicalIdentity(
        entity_type="RADAR_SYSTEM",
        identity_field_names=("system_name",),
        identity_tuple=("S-400",),
        scope="global",
        document_id=None,
    )
    d = {li: "value"}
    assert d[li] == "value"


# --- build_display_label ---------------------------------------------------

def test_display_label_prefers_system_name_in_identity():
    assert build_display_label("RADAR_SYSTEM", {"system_name": "S-400"}, {}) == "S-400"


def test_display_label_falls_back_to_joined_identity():
    label = build_display_label("SECTION", {"heading": "Intro", "page_start": 1}, {})
    assert label == "Intro"  # heading is in NAME_LIKE_KEYS


def test_display_label_uses_properties_when_identity_empty():
    label = build_display_label("FOO", {}, {"name": "FooName"})
    assert label == "FooName"


def test_display_label_hash_fallback():
    label = build_display_label("FOO", {"opaque_field": ""}, {})
    assert label.startswith("FOO_")
    assert len(label) > len("FOO_")


# --- Yield classification --------------------------------------------------

def test_classify_yield_hit():
    assert classify_yield_from_counts(
        primary=5, bridge=0, extracted_rels=3, rejected_rels=0,
    ) == YieldStatus.HIT


def test_classify_yield_empty():
    assert classify_yield_from_counts(
        primary=0, bridge=0, extracted_rels=0, rejected_rels=0,
    ) == YieldStatus.EMPTY


def test_classify_yield_bridges_only():
    assert classify_yield_from_counts(
        primary=0, bridge=2, extracted_rels=0, rejected_rels=0,
    ) == YieldStatus.BRIDGES_ONLY


def test_classify_yield_degraded_threshold():
    # 4 total with 3 rejected = 75% — exactly at threshold.
    assert classify_yield_from_counts(
        primary=5, bridge=0, extracted_rels=1, rejected_rels=3,
    ) == YieldStatus.DEGRADED


def test_classify_yield_degraded_below_threshold():
    # 3 total with 3 rejected = 100% but total_rels < 4 — NOT DEGRADED.
    assert classify_yield_from_counts(
        primary=5, bridge=0, extracted_rels=0, rejected_rels=3,
    ) == YieldStatus.HIT


# --- ChunkForDerivation / DerivedEdge dataclasses --------------------------

def test_chunk_for_derivation_fields():
    chunk = ChunkForDerivation(rid="#5:1", text_normalized="some text")
    assert chunk.rid == "#5:1"
    assert chunk.text_normalized == "some text"


def test_derived_edge_fields():
    edge = DerivedEdge(from_id="#10:1", to_id="#5:1", rel_type="MENTIONED_IN", confidence=0.9)
    assert edge.from_id == "#10:1"
    assert edge.to_id == "#5:1"
    assert edge.rel_type == "MENTIONED_IN"
    assert edge.confidence == 0.9
