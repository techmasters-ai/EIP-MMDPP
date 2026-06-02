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


# ---------------------------------------------------------------------------
# (b2) PER-EDGE precision: from_ref_id / to_ref_id resolved through
#      upstream_refs → precise (from_identity, rel_type, to_identity) triple,
#      so two edges of the SAME rel_type each get ONLY their own chunks
#      (no coarse __rel_type_fallback__ smear). evidence_ids (per-edge granular)
#      is the precise anchor, favored over the coarse batch self_refs.
# ---------------------------------------------------------------------------
# distinct per-edge chunk anchors
_PE_ELEMENT_UID_CHUNK_MAP = {
    "p1-1-text-aaaa": ["chunkX"],
    "p1-2-text-bbbb": ["chunkY"],
}
_PE_IDENTITY_MAP = {
    "#/texts/100": "p1-1-text-aaaa",   # -> chunkX  (edge A->B evidence)
    "#/texts/200": "p1-2-text-bbbb",   # -> chunkY  (edge A->C evidence)
}


def test_relationship_records_per_edge_from_ref_id_no_smear():
    """TWO ASSOCIATED_WITH edges A->B and A->C share rel_type but have distinct
    to_identity. With upstream_refs mapping E_A->A, E_B->B, E_C->C and two
    provenance rows carrying (from_ref_id, to_ref_id, evidence_ids), each edge
    must bucket to its OWN precise triple and get ONLY its own chunk — NOT the
    union (the coarse __rel_type_fallback__ smear bug)."""
    a, b, c = _identity("A"), _identity("B"), _identity("C")
    edge_ab = MergedEdgeRecord(
        from_identity=a, to_identity=b, rel_type="ASSOCIATED_WITH",
        confidence=0.9, pass_origins={"p"},
    )
    edge_ac = MergedEdgeRecord(
        from_identity=a, to_identity=c, rel_type="ASSOCIATED_WITH",
        confidence=0.9, pass_origins={"p"},
    )
    upstream_refs = {"E_A": a, "E_B": b, "E_C": c}
    rel_prov = [
        ExtractionRelationshipProvenance(
            relationship_type="ASSOCIATED_WITH",
            from_ref_id="E_A", to_ref_id="E_B",
            evidence_ids=["#/texts/100"],   # -> chunkX
            self_refs=["#/texts/100", "#/texts/200"],  # coarse batch union
        ),
        ExtractionRelationshipProvenance(
            relationship_type="ASSOCIATED_WITH",
            from_ref_id="E_A", to_ref_id="E_C",
            evidence_ids=["#/texts/200"],   # -> chunkY
            self_refs=["#/texts/100", "#/texts/200"],  # coarse batch union
        ),
    ]

    records = pipeline_mod._build_relationship_records(
        edges=[edge_ab, edge_ac],
        relationship_provenance_rows=rel_prov,
        entity_provenance_rows=[],
        element_uid_chunk_map=_PE_ELEMENT_UID_CHUNK_MAP,
        identity_map=_PE_IDENTITY_MAP,
        upstream_refs=upstream_refs,
    )

    by_to = {rec.to_identity["name"]: rec for rec in records}
    # A->B gets ONLY chunkX (its own evidence_id), NOT chunkY
    assert by_to["B"].properties["source_chunk_ids"] == ["chunkX"]
    # A->C gets ONLY chunkY (its own evidence_id), NOT chunkX
    assert by_to["C"].properties["source_chunk_ids"] == ["chunkY"]


def test_relationship_records_per_edge_survives_identity_alias_canonicalization():
    """The committed edge's from/to identities are ALIAS-CANONICALIZED (built via
    ``_resolve_identity_alias`` in ``_resolve_relationship``), but the
    ``upstream_refs`` values are the RAW, pre-canonicalization identities. When an
    endpoint is an alias (e.g. designation alias ``PAC-3`` → canonical ``MIM-104F``),
    the precise triple key built from RAW upstream_refs (PAC-3, rel, B) would MISS
    the canonicalized edge key (MIM-104F, rel, B) → edge silently orphaned to None.

    ``_build_relationship_records`` must apply the SAME ``identity_aliases``
    canonicalization to the resolved src/tgt identities BEFORE keying, so the
    precise per-edge match survives alias resolution (each edge gets ONLY its own
    chunk, no smear, no orphan)."""
    # RAW upstream_refs identity for E_A is the alias PAC-3 (pre-canonicalization).
    pac3 = _identity("PAC-3")
    mim104f = _identity("MIM-104F")  # canonical
    b, c = _identity("B"), _identity("C")
    identity_aliases = {pac3: mim104f}

    # Edges are POST-canonical: from_identity is MIM-104F (as committed by merge).
    edge_ab = MergedEdgeRecord(
        from_identity=mim104f, to_identity=b, rel_type="ASSOCIATED_WITH",
        confidence=0.9, pass_origins={"p"},
    )
    edge_ac = MergedEdgeRecord(
        from_identity=mim104f, to_identity=c, rel_type="ASSOCIATED_WITH",
        confidence=0.9, pass_origins={"p"},
    )
    # upstream_refs hold the RAW pre-alias identity for E_A.
    upstream_refs = {"E_A": pac3, "E_B": b, "E_C": c}
    rel_prov = [
        ExtractionRelationshipProvenance(
            relationship_type="ASSOCIATED_WITH",
            from_ref_id="E_A", to_ref_id="E_B",
            evidence_ids=["#/texts/100"],   # -> chunkX
            self_refs=["#/texts/100", "#/texts/200"],  # coarse batch union
        ),
        ExtractionRelationshipProvenance(
            relationship_type="ASSOCIATED_WITH",
            from_ref_id="E_A", to_ref_id="E_C",
            evidence_ids=["#/texts/200"],   # -> chunkY
            self_refs=["#/texts/100", "#/texts/200"],  # coarse batch union
        ),
    ]

    records = pipeline_mod._build_relationship_records(
        edges=[edge_ab, edge_ac],
        relationship_provenance_rows=rel_prov,
        entity_provenance_rows=[],
        element_uid_chunk_map=_PE_ELEMENT_UID_CHUNK_MAP,
        identity_map=_PE_IDENTITY_MAP,
        upstream_refs=upstream_refs,
        identity_aliases=identity_aliases,
    )

    by_to = {rec.to_identity["name"]: rec for rec in records}
    # MIM-104F->B gets ONLY chunkX (precise match survives canonicalization).
    assert by_to["B"].properties["source_chunk_ids"] == ["chunkX"]
    # MIM-104F->C gets ONLY chunkY.
    assert by_to["C"].properties["source_chunk_ids"] == ["chunkY"]


def test_relationship_record_refless_row_falls_back_and_warns(caplog):
    """A provenance row with NO from_ref_id/to_ref_id (and no resolvable
    instance ids) lands in the __rel_type_fallback__ bucket AND emits a WARN
    naming the rel_type + why (no ref)."""
    import logging

    a, b = _identity("A"), _identity("B")
    edge = MergedEdgeRecord(
        from_identity=a, to_identity=b, rel_type="ASSOCIATED_WITH",
        confidence=0.7, pass_origins={"p"},
    )
    rel_prov = [
        ExtractionRelationshipProvenance(
            relationship_type="ASSOCIATED_WITH",
            evidence_ids=["#/texts/100"],   # -> chunkX
        )
    ]
    with caplog.at_level(logging.WARNING):
        records = pipeline_mod._build_relationship_records(
            edges=[edge],
            relationship_provenance_rows=rel_prov,
            entity_provenance_rows=[],
            element_uid_chunk_map=_PE_ELEMENT_UID_CHUNK_MAP,
            identity_map=_PE_IDENTITY_MAP,
            upstream_refs={},
        )
    # row still resolves chunks via the fallback bucket
    assert records[0].properties["source_chunk_ids"] == ["chunkX"]
    # and a WARN was emitted naming the rel_type
    assert any(
        "ASSOCIATED_WITH" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    ), [r.message for r in caplog.records]


def test_relationship_record_ref_not_in_upstream_refs_falls_back_and_warns(caplog):
    """A row carrying ref ids that are NOT present in upstream_refs cannot
    resolve a precise triple → lands in fallback AND WARNs (ref-not-in-upstream)."""
    import logging

    a, b = _identity("A"), _identity("B")
    edge = MergedEdgeRecord(
        from_identity=a, to_identity=b, rel_type="ASSOCIATED_WITH",
        confidence=0.7, pass_origins={"p"},
    )
    rel_prov = [
        ExtractionRelationshipProvenance(
            relationship_type="ASSOCIATED_WITH",
            from_ref_id="E_MISSING", to_ref_id="E_ALSO_MISSING",
            evidence_ids=["#/texts/100"],
        )
    ]
    with caplog.at_level(logging.WARNING):
        records = pipeline_mod._build_relationship_records(
            edges=[edge],
            relationship_provenance_rows=rel_prov,
            entity_provenance_rows=[],
            element_uid_chunk_map=_PE_ELEMENT_UID_CHUNK_MAP,
            identity_map=_PE_IDENTITY_MAP,
            upstream_refs={"E_A": a},  # neither ref present
        )
    assert records[0].properties["source_chunk_ids"] == ["chunkX"]
    assert any(
        "ASSOCIATED_WITH" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    ), [r.message for r in caplog.records]


# ---------------------------------------------------------------------------
# (c) _build_lineage_resolver_maps — exercises the REAL function body
#     (the select(...) calls + artifact_id -> element_uid -> chunk_id join).
#     Regression guard: this function previously called select() WITHOUT a
#     local import, so EVERY production merge raised NameError before any
#     entity/edge was committed. The other tests above passed PRE-BUILT maps,
#     so they never touched this body. These two tests would fail (NameError)
#     without the `from sqlalchemy import select` fix.
# ---------------------------------------------------------------------------
import uuid as _uuid

import pytest


class _StubScalarResult:
    """Mimics the SQLAlchemy ``.scalars()`` proxy: ``.all()`` -> rows."""

    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _StubExecuteResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _StubScalarResult(self._rows)


class _StubDB:
    """Stub Session whose ``.execute(stmt).scalars().all()`` returns the next
    queued result set. ``_build_lineage_resolver_maps`` issues exactly three
    execute() calls in order: DocumentElement, TextChunk, ImageChunk."""

    def __init__(self, result_sets):
        self._queue = list(result_sets)
        self.statements = []

    def execute(self, statement):
        # Force evaluation of the statement — this is where a bare select(...)
        # (no import) would already have blown up with NameError at call time,
        # but capturing it here also proves the select() expression was built.
        self.statements.append(statement)
        return _StubExecuteResult(self._queue.pop(0))


class _Elem:
    def __init__(self, artifact_id, element_uid, element_order=0):
        self.artifact_id = artifact_id
        self.element_uid = element_uid
        self.element_order = element_order


class _Chunk:
    def __init__(self, id, artifact_id=None, page_number=None):
        self.id = id
        self.artifact_id = artifact_id
        self.page_number = page_number


def test_build_lineage_resolver_maps_joins_artifact_to_chunk(monkeypatch):
    """Exercise the REAL ``_build_lineage_resolver_maps`` body with a mock db.

    Would raise ``NameError: name 'select' is not defined`` without the local
    ``from sqlalchemy import select`` import — i.e. this test is the regression
    guard the pre-built-map tests above could never provide. Asserts the
    artifact_id -> element_uid -> chunk_id join + the chunk page map.
    """
    # identity_map comes from the persisted docling json; isolate from any DB.
    monkeypatch.setattr(
        pipeline_mod, "_load_identity_map", lambda doc_id: {"#/texts/3": "euid-A"}
    )

    elements = [
        _Elem(artifact_id="art-1", element_uid="euid-A", element_order=0),
        _Elem(artifact_id="art-2", element_uid="euid-B", element_order=1),
        # rows with no artifact_id / element_uid must be skipped, not crash.
        _Elem(artifact_id=None, element_uid="euid-C", element_order=2),
    ]
    text_chunks = [
        _Chunk(id="chunk-1", artifact_id="art-1", page_number=5),
        _Chunk(id="chunk-2", artifact_id="art-2", page_number=None),
    ]
    image_chunks = [
        _Chunk(id="img-1", artifact_id="art-1", page_number=7),
    ]
    db = _StubDB([elements, text_chunks, image_chunks])

    identity_map, element_uid_chunk_map, chunk_page_map = (
        pipeline_mod._build_lineage_resolver_maps(db, _uuid.uuid4())
    )

    # all three select(...) statements were built + executed (proves no NameError).
    assert len(db.statements) == 3
    # identity_map threaded through from the patched loader.
    assert identity_map == {"#/texts/3": "euid-A"}
    # artifact_id -> element_uid -> chunk_id join, across text + image chunks.
    assert element_uid_chunk_map["euid-A"] == ["chunk-1", "img-1"]
    assert element_uid_chunk_map["euid-B"] == ["chunk-2"]
    # element with no element_uid/artifact contributes nothing.
    assert "euid-C" not in element_uid_chunk_map
    # page map back-fill (None pages omitted).
    assert chunk_page_map["chunk-1"] == 5
    assert chunk_page_map["img-1"] == 7
    assert "chunk-2" not in chunk_page_map


# ---------------------------------------------------------------------------
# (d) _build_element_uid_chunk_map — the shared helper that BOTH the merge
#     phase (_build_lineage_resolver_maps) and derive_structure_links
#     (EXTRACTED_FROM) now call. Direct coverage so the byte-identical map
#     semantics can't drift between the two call sites.
# ---------------------------------------------------------------------------
def test_build_element_uid_chunk_map_joins_text_and_image_chunks():
    """artifact_id -> element_uid -> chunk_id join across BOTH text + image
    chunks; rows missing artifact_id/element_uid contribute nothing; no
    page logic in the map (that lives in _build_lineage_resolver_maps)."""
    elements = [
        _Elem(artifact_id="art-1", element_uid="euid-A"),
        _Elem(artifact_id="art-2", element_uid="euid-B"),
        # no artifact_id -> skipped, no crash.
        _Elem(artifact_id=None, element_uid="euid-C"),
        # no element_uid -> skipped.
        _Elem(artifact_id="art-3", element_uid=None),
    ]
    text_chunks = [
        _Chunk(id="chunk-1", artifact_id="art-1", page_number=5),
        _Chunk(id="chunk-2", artifact_id="art-2"),
        # artifact with no element mapping -> dropped.
        _Chunk(id="chunk-orphan", artifact_id="art-unknown"),
        # chunk with no artifact_id -> dropped.
        _Chunk(id="chunk-none", artifact_id=None),
    ]
    image_chunks = [
        _Chunk(id="img-1", artifact_id="art-1", page_number=7),
    ]

    result = pipeline_mod._build_element_uid_chunk_map(
        elements, text_chunks, image_chunks,
    )

    # text + image chunks for the same element collapse onto one element_uid,
    # text chunk first (insertion order preserved).
    assert result == {
        "euid-A": ["chunk-1", "img-1"],
        "euid-B": ["chunk-2"],
    }
    assert "euid-C" not in result
    assert "art-unknown" not in result


def test_build_element_uid_chunk_map_empty_inputs():
    """Empty inputs -> empty map, no crash."""
    assert pipeline_mod._build_element_uid_chunk_map([], [], []) == {}


def test_build_element_uid_chunk_map_matches_lineage_resolver(monkeypatch):
    """The shared helper must produce the SAME element_uid_chunk_map that
    _build_lineage_resolver_maps returns from the identical row sets — the
    whole point of the refactor (no divergence between the two call sites)."""
    monkeypatch.setattr(pipeline_mod, "_load_identity_map", lambda doc_id: {})

    elements = [
        _Elem(artifact_id="art-1", element_uid="euid-A", element_order=0),
        _Elem(artifact_id="art-2", element_uid="euid-B", element_order=1),
    ]
    text_chunks = [
        _Chunk(id="chunk-1", artifact_id="art-1", page_number=5),
        _Chunk(id="chunk-2", artifact_id="art-2", page_number=None),
    ]
    image_chunks = [
        _Chunk(id="img-1", artifact_id="art-1", page_number=7),
    ]

    direct = pipeline_mod._build_element_uid_chunk_map(
        elements, text_chunks, image_chunks,
    )
    db = _StubDB([elements, text_chunks, image_chunks])
    _, via_resolver, _ = pipeline_mod._build_lineage_resolver_maps(
        db, _uuid.uuid4(),
    )
    assert direct == via_resolver


def test_build_lineage_resolver_maps_empty_db_no_nameerror(monkeypatch):
    """Smoke guard: empty result sets must yield three empty-ish maps WITHOUT
    raising NameError (the minimal repro of the missing-import bug)."""
    monkeypatch.setattr(pipeline_mod, "_load_identity_map", lambda doc_id: {})
    db = _StubDB([[], [], []])

    identity_map, element_uid_chunk_map, chunk_page_map = (
        pipeline_mod._build_lineage_resolver_maps(db, _uuid.uuid4())
    )

    assert identity_map == {}
    assert element_uid_chunk_map == {}
    assert chunk_page_map == {}


# ---------------------------------------------------------------------------
# (e) self_ref -> chunk_id BRIDGE from ArcadeDB TextChunk.self_refs.
#
#     Native HybridChunker gives prose/table chunks artifact_id=None, so the
#     artifact_id join in _build_element_uid_chunk_map reaches ONLY the ~34
#     image-derived chunks (zero text/table). The authoritative
#     self_ref -> chunk_id mapping IS persisted onto the ArcadeDB TextChunk
#     vertex `self_refs` property; seeding the RAW `#/...` self_ref keys into
#     element_uid_chunk_map lets _resolve_mention_chunks resolve them on the
#     DIRECT .get(ref) hop (before the identity_map hop).
#
#     A field whose own self_ref is `#/tables/1` (text/table-only provenance,
#     no image self_ref) must resolve to its REAL chunk — NOT None, NOT an
#     image-fallback chunk. A self_ref spanning MULTIPLE chunks must pick the
#     LOWEST chunk_index deterministically for the scalar chunk_id.
# ---------------------------------------------------------------------------


def test_build_self_ref_chunk_map_pure_sorts_by_chunk_index():
    """The pure builder takes rows (chunk_id, self_refs, chunk_index) and
    returns self_ref -> [chunk_id] with each list ordered by chunk_index.

    A table spanning two chunks (#/tables/1 -> c_idx30, c_idx31) lists the
    chunk_index-30 id FIRST regardless of input row order, so the scalar
    `resolved[0]` consumers pick the lowest-index chunk deterministically.
    """
    rows = [
        # arbitrary order: c_idx31 row BEFORE c_idx30 row
        ("uuidC31", ["#/tables/1"], 31),
        ("uuidC30", ["#/tables/1"], 30),
        ("uuidX", ["#/texts/99"], 21),
        # a chunk carrying multiple self_refs contributes to each.
        ("uuidM", ["#/texts/5", "#/texts/6"], 3),
        # empty / None self_refs contribute nothing, no crash.
        ("uuidEmpty", [], 99),
        ("uuidNone", None, 100),
    ]
    result = pipeline_mod._build_self_ref_chunk_map(rows)

    # multi-chunk table self_ref: lowest chunk_index FIRST.
    assert result["#/tables/1"] == ["uuidC30", "uuidC31"]
    # single-chunk text self_ref.
    assert result["#/texts/99"] == ["uuidX"]
    # one chunk feeding multiple self_refs.
    assert result["#/texts/5"] == ["uuidM"]
    assert result["#/texts/6"] == ["uuidM"]


def test_field_table_self_ref_resolves_to_real_chunk_not_image_or_none():
    """Flat-Face-shaped entity: its ONLY field self_ref is `#/tables/1` and its
    ENTITY provenance is text/table-only (no image self_ref). With the
    self_ref->chunk_id bridge seeded (table spans c_idx30 + c_idx31), the field
    resolves to the REAL chunk (lowest chunk_index, c_idx30), NOT None and NOT
    an image-description chunk; page back-fills to 7."""
    UUID_C30 = "ac77ed0b-0000-0000-0000-000000000030"
    UUID_C31 = "478f0d84-0000-0000-0000-000000000031"
    UUID_IMG = "image-desc-chunk-id"

    # element_uid_chunk_map seeded with the RAW self_ref keys from ArcadeDB,
    # plus an image element_uid that the OLD artifact_id join could reach.
    seeded_map = {
        "#/tables/1": [UUID_C30, UUID_C31],  # table spans two chunks
        "euid-image": [UUID_IMG],            # image-derived chunk (artifact join)
    }
    chunk_page_map = {UUID_C30: 7, UUID_C31: 7, UUID_IMG: 99}

    row = FieldEvidenceRow(
        chunk_id=None,
        snippet="associated radar",
        element_uid="#/tables/1",
        value="Flat Face",
        self_refs=["#/tables/1"],
    )
    entity = MergedEntityRecord(
        identity=_identity("Flat Face"),
        properties={"name": "Flat Face"},
        confidence=0.9,
        pass_origins={"p"},
        display_label="Flat Face",
        provenance=[
            ExtractionProvenance(
                instance_id="i1",
                ontology_name="System",
                identity_values={"name": "Flat Face"},
                # ENTITY provenance is text/table-only — NO image self_ref.
                element_uid="#/tables/1",
                self_refs=["#/tables/1"],
            )
        ],
        field_evidence={"associated_radar": [row]},
    )

    pipeline_mod._resolve_field_evidence_chunk_ids(
        entity, seeded_map, {}, chunk_page_map=chunk_page_map,
    )

    # Resolves to a REAL chunk, the lowest-chunk_index one for the table.
    assert row.chunk_id == UUID_C30
    # NOT None, NOT the image-description chunk.
    assert row.chunk_id is not None
    assert row.chunk_id != UUID_IMG
    # page back-fills to the table's page (7).
    assert row.page == 7


def test_field_multi_chunk_table_self_ref_deterministic_lowest_index():
    """Determinism: even with the seeded list in arbitrary order, the field's
    scalar chunk_id is the chunk_index-30 id (built sorted by chunk_index in the
    pure builder, so resolved[0] is deterministic)."""
    UUID_C30 = "uuid-c30"
    UUID_C31 = "uuid-c31"
    # Build via the pure builder with rows in REVERSED chunk_index order.
    seeded_map = pipeline_mod._build_self_ref_chunk_map([
        ("uuid-c31", ["#/tables/1"], 31),
        ("uuid-c30", ["#/tables/1"], 30),
    ])
    row = FieldEvidenceRow(
        chunk_id=None,
        snippet="x",
        element_uid="#/tables/1",
        value="v",
        self_refs=["#/tables/1"],
    )
    entity = MergedEntityRecord(
        identity=_identity("Squat Eye"),
        properties={"name": "Squat Eye"},
        confidence=0.9,
        pass_origins={"p"},
        display_label="Squat Eye",
        provenance=[
            ExtractionProvenance(
                instance_id="i1",
                ontology_name="System",
                identity_values={"name": "Squat Eye"},
                element_uid="#/tables/1",
                self_refs=["#/tables/1"],
            )
        ],
        field_evidence={"f": [row]},
    )
    pipeline_mod._resolve_field_evidence_chunk_ids(
        entity, seeded_map, {}, chunk_page_map={},
    )
    assert row.chunk_id == UUID_C30
    assert row.chunk_id != UUID_C31


def test_resolve_mention_chunks_unions_multi_chunk_table():
    """The multi-chunk case must NOT lose the second chunk where the resolver
    unions (relationship/entity source_chunk_ids). #/tables/1 -> both chunks,
    in chunk_index order."""
    seeded_map = {"#/tables/1": ["uuid-c30", "uuid-c31"]}
    resolved, is_coarse = pipeline_mod._resolve_mention_chunks(
        ["#/tables/1"], seeded_map, {},
    )
    assert resolved == ["uuid-c30", "uuid-c31"]
    assert is_coarse is False


def test_build_lineage_resolver_maps_seeds_self_ref_keys(monkeypatch):
    """_build_lineage_resolver_maps must seed the RAW self_ref keys from the
    ArcadeDB TextChunk.self_refs bridge into element_uid_chunk_map (so the
    direct .get(ref) hop in _resolve_mention_chunks resolves prose/table refs),
    WITHOUT clobbering the existing artifact_id-join (image) keys."""
    monkeypatch.setattr(pipeline_mod, "_load_identity_map", lambda doc_id: {})
    # Inject the ArcadeDB-sourced self_ref map (thin wrapper is mocked out).
    monkeypatch.setattr(
        pipeline_mod,
        "_load_self_ref_chunk_map_from_arcadedb",
        lambda doc_id: {
            "#/texts/99": ["chunk-text-99"],
            "#/tables/1": ["chunk-tbl-30", "chunk-tbl-31"],
        },
    )

    elements = [_Elem(artifact_id="art-1", element_uid="euid-A", element_order=0)]
    text_chunks = [_Chunk(id="chunk-1", artifact_id="art-1", page_number=5)]
    image_chunks = []
    db = _StubDB([elements, text_chunks, image_chunks])

    _, element_uid_chunk_map, _ = pipeline_mod._build_lineage_resolver_maps(
        db, _uuid.uuid4(),
    )

    # artifact_id-join (image/text-with-artifact) keys still present.
    assert element_uid_chunk_map["euid-A"] == ["chunk-1"]
    # RAW self_ref keys seeded from ArcadeDB.
    assert element_uid_chunk_map["#/texts/99"] == ["chunk-text-99"]
    assert element_uid_chunk_map["#/tables/1"] == ["chunk-tbl-30", "chunk-tbl-31"]
