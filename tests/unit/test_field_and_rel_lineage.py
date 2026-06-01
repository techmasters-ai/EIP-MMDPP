"""Task 5: precision lineage completions on the merge phase.

(A) FIELD: every ``_field_evidence`` row must carry a RESOLVED ``chunk_id``
    (today hardcoded None). Resolution happens IN THE MERGE PHASE using the
    SAME ``_resolve_mention_chunks`` resolver Task 4 built. A field whose own
    ``self_ref`` doesn't resolve falls back to the ENTITY's resolved chunk-set
    (never None when the entity has resolvable lineage, never all-document).

(B) RELATIONSHIP: committed relationship edges must carry
    ``source_chunk_ids`` / ``source_pages`` / ``source_self_refs`` properties,
    resolved from the relationship's positional self_refs via the same
    resolver and written via ``RelationshipRecord.properties`` (no SQL change).

These tests exercise the PURE resolution — no live DB. They assert on the
produced row / RelationshipRecord, not on any committed graph state.
"""
from app.services.extraction_merge import (
    ExtractionProvenance,
    ExtractionRelationshipProvenance,
    FieldEvidenceRow,
    LogicalIdentity,
    MergedEdgeRecord,
    MergedEntityRecord,
)
from app.workers import pipeline as pipeline_mod


# --- shared resolver fixtures ------------------------------------------------
# identity_map bridges Docling self_refs (#/texts/N) -> DocumentElement
# element_uid; element_uid_chunk_map maps element_uid -> concrete chunk ids.
ELEMENT_UID_CHUNK_MAP = {
    "p1-2-text-abcd": ["chunkC"],   # the field's own self_ref resolves here
    "p1-7-text-ef01": ["chunkE"],   # the entity's provenance self_ref
    "p1-9-text-9999": ["chunkUnrelated"],  # never referenced -> must not leak
}
IDENTITY_MAP = {
    "#/texts/3": "p1-2-text-abcd",   # -> chunkC
    "#/texts/7": "p1-7-text-ef01",   # -> chunkE  (entity-level fallback)
}


def _identity(name: str) -> LogicalIdentity:
    return LogicalIdentity(
        entity_type="System",
        identity_field_names=("name",),
        identity_tuple=(name,),
        scope="global",
        document_id=None,
    )


# ---------------------------------------------------------------------------
# (a) FIELD chunk_id resolution + entity-fallback
# ---------------------------------------------------------------------------
def test_field_self_ref_resolves_to_specific_chunk():
    """A field row whose own self_ref resolves to chunk C gets chunk_id == C."""
    row = FieldEvidenceRow(
        chunk_id=None,
        snippet="range is 43 km",
        element_uid="#/texts/3",
        value="43 km",
        self_refs=["#/texts/3"],
    )
    entity = MergedEntityRecord(
        identity=_identity("SA-2"),
        properties={"name": "SA-2"},
        confidence=0.9,
        pass_origins={"p"},
        display_label="SA-2",
        provenance=[
            ExtractionProvenance(
                instance_id="i1",
                ontology_name="System",
                identity_values={"name": "SA-2"},
                element_uid="#/texts/7",
                self_refs=["#/texts/7"],
            )
        ],
        field_evidence={"range": [row]},
    )

    pipeline_mod._resolve_field_evidence_chunk_ids(
        entity, ELEMENT_UID_CHUNK_MAP, IDENTITY_MAP, chunk_page_map={"chunkC": 1},
    )

    assert row.chunk_id == "chunkC"
    # No fan-out: unrelated chunks never appear.
    assert row.chunk_id != "chunkUnrelated"
    # page back-filled from the resolved chunk when not already set.
    assert row.page == 1


def test_field_unresolved_self_ref_falls_back_to_entity_chunk_set():
    """A field whose own self_ref doesn't resolve falls back to the ENTITY's
    resolved chunk-set — never None, never all-document."""
    row = FieldEvidenceRow(
        chunk_id=None,
        snippet="guidance is radio command",
        element_uid="#/texts/404",
        value="radio command",
        self_refs=["#/texts/404"],  # unmapped -> does NOT resolve
    )
    entity = MergedEntityRecord(
        identity=_identity("SA-2"),
        properties={"name": "SA-2"},
        confidence=0.9,
        pass_origins={"p"},
        display_label="SA-2",
        provenance=[
            ExtractionProvenance(
                instance_id="i1",
                ontology_name="System",
                identity_values={"name": "SA-2"},
                element_uid="#/texts/7",
                self_refs=["#/texts/7"],  # entity resolves -> chunkE
            )
        ],
        field_evidence={"guidance": [row]},
    )

    pipeline_mod._resolve_field_evidence_chunk_ids(
        entity, ELEMENT_UID_CHUNK_MAP, IDENTITY_MAP, chunk_page_map={},
    )

    # Falls back to the entity's resolved chunk-set (chunkE), not None.
    assert row.chunk_id is not None
    assert row.chunk_id == "chunkE"
    # Never all-document: the unrelated chunk is not selected.
    assert row.chunk_id != "chunkUnrelated"


def test_field_chunk_id_stays_none_when_nothing_resolves():
    """When neither the field nor the entity has resolvable lineage, chunk_id
    stays None (we never fabricate a chunk_id, never fan out)."""
    row = FieldEvidenceRow(
        chunk_id=None,
        snippet="unverifiable",
        element_uid="#/texts/404",
        value="x",
        self_refs=["#/texts/404"],
    )
    entity = MergedEntityRecord(
        identity=_identity("Ghost"),
        properties={"name": "Ghost"},
        confidence=0.5,
        pass_origins={"p"},
        display_label="Ghost",
        provenance=[
            ExtractionProvenance(
                instance_id="i9",
                ontology_name="System",
                identity_values={"name": "Ghost"},
                element_uid="#/texts/404",
                self_refs=["#/texts/404"],  # also unmapped
            )
        ],
        field_evidence={"f": [row]},
    )

    pipeline_mod._resolve_field_evidence_chunk_ids(
        entity, ELEMENT_UID_CHUNK_MAP, IDENTITY_MAP, chunk_page_map={},
    )

    assert row.chunk_id is None


# ---------------------------------------------------------------------------
# (b) RELATIONSHIP source_chunk_ids / source_pages / source_self_refs
# ---------------------------------------------------------------------------
def test_relationship_record_carries_source_chunk_ids():
    """A RelationshipRecord built from a rel whose provenance self_refs
    resolve to chunk C carries record.properties['source_chunk_ids'] == [C]
    plus source_pages + source_self_refs."""
    from_id = _identity("SA-2")
    to_id = LogicalIdentity(
        entity_type="System",
        identity_field_names=("name",),
        identity_tuple=("Fan Song",),
        scope="global",
        document_id=None,
    )
    edge = MergedEdgeRecord(
        from_identity=from_id,
        to_identity=to_id,
        rel_type="GUIDED_BY",
        confidence=0.8,
        pass_origins={"p"},
    )
    rel_prov = [
        ExtractionRelationshipProvenance(
            relationship_type="GUIDED_BY",
            source_instance_id=None,
            target_instance_id=None,
            self_refs=["#/texts/3"],
            page_numbers=[1, 2],
            evidence_ids=["ev1"],
        )
    ]

    records = pipeline_mod._build_relationship_records(
        edges=[edge],
        relationship_provenance_rows=rel_prov,
        entity_provenance_rows=[],
        element_uid_chunk_map=ELEMENT_UID_CHUNK_MAP,
        identity_map=IDENTITY_MAP,
    )

    assert len(records) == 1
    rec = records[0]
    assert rec.rel_type == "GUIDED_BY"
    assert "chunkC" in rec.properties["source_chunk_ids"]
    assert "chunkUnrelated" not in rec.properties["source_chunk_ids"]
    assert rec.properties["source_pages"] == [1, 2]
    assert rec.properties["source_self_refs"] == ["#/texts/3"]


def test_relationship_record_no_self_refs_omits_props():
    """A rel with no resolvable self_refs does not get a source_chunk_ids
    property (we never fabricate / fan out)."""
    edge = MergedEdgeRecord(
        from_identity=_identity("A"),
        to_identity=_identity("B"),
        rel_type="NEAR",
        confidence=0.5,
        pass_origins={"p"},
    )
    records = pipeline_mod._build_relationship_records(
        edges=[edge],
        relationship_provenance_rows=[],
        entity_provenance_rows=[],
        element_uid_chunk_map=ELEMENT_UID_CHUNK_MAP,
        identity_map=IDENTITY_MAP,
    )
    assert len(records) == 1
    assert "source_chunk_ids" not in records[0].properties
