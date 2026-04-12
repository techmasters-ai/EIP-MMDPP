"""Integration test for ArcadeDBGraphStore.delete_extraction_layer_graph_sync.

Spec §6.8 + residual check #1 + Task 3.2.

Seeding strategy
----------------
We create a minimal but realistic mixed-scope graph under two distinct
document_ids (DOC_A and DOC_B).  DOC_A is the rollback target; DOC_B is
a bystander whose data MUST survive.

Graph layout
~~~~~~~~~~~~
  Document(DOC_A)   Document(DOC_B)
       |                  |
  [HAS_PROVENANCE]  [HAS_PROVENANCE]
       |                  |
  RADAR_SYSTEM(RS1)  ← global entity tied to BOTH documents
  PLATFORM(PLT1)     ← global entity tied to DOC_A
  SECTION(SEC_A)     ← document-scoped entity, DOC_A only
  TextChunk(TC_A)    ← chunk vertex, document_id=DOC_A (MUST survive rollback)
  TextChunk(TC_B)    ← chunk vertex, document_id=DOC_B (bystander)
  INSTALLED_ON edge  ← domain edge RS1→PLT1, document_ids=[DOC_A] (single-doc)

After rollback of DOC_A:
  - SECTION(SEC_A) DELETED
  - HAS_PROVENANCE from RS1 to Document(DOC_A) DELETED
  - HAS_PROVENANCE from RS1 to Document(DOC_B) PRESERVED
  - INSTALLED_ON.document_ids shrinks to [DOC_B] (not deleted — other doc)
  - TextChunk(TC_A) PRESERVED (chunks are owned by upstream stages)
  - RADAR_SYSTEM(RS1) PRESERVED (global entity)
  - Document(DOC_A) PRESERVED (structural vertex, not our concern)
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from app.services.arcadedb_graph import ArcadeDBGraphStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_doc_id() -> str:
    return f"test-rollback-{uuid.uuid4().hex[:12]}"


def _count_vertices(store: "ArcadeDBGraphStore", vertex_class: str, doc_id: str) -> int:
    """Count vertices of class with document_id = doc_id."""
    rows = store._client.query_sync(
        store._database, "sql",
        f"SELECT count(*) AS n FROM {vertex_class} WHERE document_id = :doc_id",
        {"doc_id": doc_id},
    )
    return int(rows[0]["n"]) if rows else 0


def _count_vertices_by_name(
    store: "ArcadeDBGraphStore", vertex_class: str, name: str
) -> int:
    """Count vertices of class with a given name (for global entities)."""
    rows = store._client.query_sync(
        store._database, "sql",
        f"SELECT count(*) AS n FROM {vertex_class} WHERE name = :name",
        {"name": name},
    )
    return int(rows[0]["n"]) if rows else 0


def _count_domain_edges_for_doc(
    store: "ArcadeDBGraphStore", edge_class: str, doc_id: str
) -> int:
    """Count domain edges whose document_ids list CONTAINS doc_id."""
    rows = store._client.query_sync(
        store._database, "sql",
        f"SELECT count(*) AS n FROM {edge_class} WHERE document_ids CONTAINS :doc_id",
        {"doc_id": doc_id},
    )
    return int(rows[0]["n"]) if rows else 0


def _count_has_provenance_to_doc(store: "ArcadeDBGraphStore", doc_id: str) -> int:
    """Count HAS_PROVENANCE edges pointing at the Document vertex for doc_id.

    ArcadeDB requires ``@in.<prop>`` (not plain ``in.<prop>``) to traverse
    edge endpoint vertex properties in WHERE clauses.
    """
    rows = store._client.query_sync(
        store._database, "sql",
        "SELECT count(*) AS n FROM HAS_PROVENANCE WHERE @in.document_id = :doc_id",
        {"doc_id": doc_id},
    )
    return int(rows[0]["n"]) if rows else 0


def _upsert_document(store: "ArcadeDBGraphStore", doc_id: str) -> str:
    """Upsert a Document vertex, return its RID."""
    result = store._client.command_sync(
        store._database, "sql",
        "UPDATE Document SET document_id = :doc_id, updated_at = sysdate() "
        "UPSERT RETURN AFTER @rid WHERE document_id = :doc_id",
        {"doc_id": doc_id},
    )
    if result and isinstance(result[0], dict):
        return str(result[0].get("@rid", ""))
    return ""


def _cleanup(store: "ArcadeDBGraphStore", *doc_ids: str) -> None:
    """Best-effort cleanup of test data after test completion."""
    for doc_id in doc_ids:
        try:
            store.delete_document_graph_sync(doc_id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fixture: seeded mixed-scope graph
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def seeded_graph(arcadedb_store: "ArcadeDBGraphStore"):
    """Seed a mixed-scope graph and yield context dict.  Cleans up after test."""
    store = arcadedb_store
    doc_a = _mk_doc_id()
    doc_b = _mk_doc_id()
    radar_name = f"rs1-{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # 1. Structural Document vertices
    # ------------------------------------------------------------------
    doc_a_rid = _upsert_document(store, doc_a)
    doc_b_rid = _upsert_document(store, doc_b)
    assert doc_a_rid, f"Failed to upsert Document for {doc_a}"
    assert doc_b_rid, f"Failed to upsert Document for {doc_b}"

    # ------------------------------------------------------------------
    # 2. Global entities: RADAR_SYSTEM and PLATFORM tied to BOTH documents
    # ------------------------------------------------------------------
    from app.services.graph_store import NodeRecord, ProvenanceMetadata

    platform_name = f"plt1-{uuid.uuid4().hex[:8]}"

    radar_record = NodeRecord(
        entity_type="RADAR_SYSTEM",
        identity_fields={"name": radar_name},
        name=radar_name,
        extraction_confidence=0.9,
    )
    platform_record = NodeRecord(
        entity_type="PLATFORM",
        identity_fields={"name": platform_name},
        name=platform_name,
        extraction_confidence=0.9,
    )
    prov_a = ProvenanceMetadata(document_id=doc_a, page_numbers=[1])
    prov_b = ProvenanceMetadata(document_id=doc_b, page_numbers=[2])

    radar_rid = store.upsert_node_sync(radar_record, prov_a)
    assert radar_rid, "Failed to upsert RADAR_SYSTEM node"
    # Also attach HAS_PROVENANCE to DOC_B
    store._create_provenance_edge_sync(radar_rid, prov_b)

    platform_rid = store.upsert_node_sync(platform_record, prov_a)
    assert platform_rid, "Failed to upsert PLATFORM node"

    # ------------------------------------------------------------------
    # 3. Document-scoped entity: SECTION tied only to DOC_A
    # ------------------------------------------------------------------
    section_record = NodeRecord(
        entity_type="SECTION",
        identity_fields={"name": f"sec-{doc_a}", "document_id": doc_a},
        name=f"sec-{doc_a}",
        properties={"document_id": doc_a},
        extraction_confidence=0.85,
    )
    section_rid = store.upsert_node_sync(section_record, prov_a)
    assert section_rid, "Failed to upsert SECTION node"

    # ------------------------------------------------------------------
    # 4. TextChunk vertices — one per document
    # ------------------------------------------------------------------
    tc_a_rid = store.create_text_chunk_vertex_sync(
        chunk_id=f"chunk-{doc_a}",
        text="The radar system operates in S-band.",
        document_id=doc_a,
    )
    assert tc_a_rid, "Failed to create TextChunk for DOC_A"

    tc_b_rid = store.create_text_chunk_vertex_sync(
        chunk_id=f"chunk-{doc_b}",
        text="The radar system has a range of 200 km.",
        document_id=doc_b,
    )
    assert tc_b_rid, "Failed to create TextChunk for DOC_B"

    # ------------------------------------------------------------------
    # 5. Domain edge: INSTALLED_ON with document_ids=[doc_a]
    #    RADAR_SYSTEM → PLATFORM is a valid ontology triple.
    # ------------------------------------------------------------------
    from app.services.graph_store import RelationshipRecord

    store.upsert_relationships_batch_sync(
        [
            RelationshipRecord(
                from_type="RADAR_SYSTEM",
                from_identity={"name": radar_name},
                to_type="PLATFORM",
                to_identity={"name": platform_name},
                rel_type="INSTALLED_ON",
                extraction_confidence=0.75,
            )
        ],
        prov_a,
    )

    # ------------------------------------------------------------------
    # 6. Structural MENTIONED_IN edge from RADAR_SYSTEM to TextChunk (TC_A)
    #    via create_structural_edge_sync (the derive_rules path)
    # ------------------------------------------------------------------
    store.create_structural_edge_sync(
        from_id=radar_rid,
        to_id=tc_a_rid,
        rel_type="MENTIONED_IN",
    )

    yield {
        "store": store,
        "doc_a": doc_a,
        "doc_b": doc_b,
        "doc_a_rid": doc_a_rid,
        "doc_b_rid": doc_b_rid,
        "radar_name": radar_name,
        "platform_name": platform_name,
        "radar_rid": radar_rid,
        "platform_rid": platform_rid,
        "section_rid": section_rid,
        "tc_a_rid": tc_a_rid,
        "tc_b_rid": tc_b_rid,
    }

    # Cleanup: purge both test documents fully
    _cleanup(store, doc_a, doc_b)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_delete_extraction_layer_graph_sync(seeded_graph):
    """Verifies the MUST-delete and MUST-NOT-delete sets from spec §6.8."""
    store: "ArcadeDBGraphStore" = seeded_graph["store"]
    doc_a: str = seeded_graph["doc_a"]
    doc_b: str = seeded_graph["doc_b"]
    radar_name: str = seeded_graph["radar_name"]
    tc_a_rid: str = seeded_graph["tc_a_rid"]

    # ---- Pre-conditions -------------------------------------------------

    # SECTION exists for DOC_A
    assert _count_vertices(store, "SECTION", doc_a) >= 1, (
        "Pre-condition: SECTION for DOC_A not found"
    )

    # HAS_PROVENANCE from RADAR_SYSTEM to DOC_A
    assert _count_has_provenance_to_doc(store, doc_a) >= 1, (
        "Pre-condition: HAS_PROVENANCE to DOC_A not found"
    )

    # HAS_PROVENANCE from RADAR_SYSTEM to DOC_B
    assert _count_has_provenance_to_doc(store, doc_b) >= 1, (
        "Pre-condition: HAS_PROVENANCE to DOC_B not found"
    )

    # INSTALLED_ON with doc_a in document_ids
    assert _count_domain_edges_for_doc(store, "INSTALLED_ON", doc_a) >= 1, (
        "Pre-condition: INSTALLED_ON edge for DOC_A not found"
    )

    # TextChunk TC_A exists
    assert _count_vertices(store, "TextChunk", doc_a) >= 1, (
        "Pre-condition: TextChunk for DOC_A not found"
    )

    # RADAR_SYSTEM exists
    assert _count_vertices_by_name(store, "RADAR_SYSTEM", radar_name) >= 1, (
        "Pre-condition: RADAR_SYSTEM not found"
    )

    # ---- Execute rollback -----------------------------------------------

    result = store.delete_extraction_layer_graph_sync(doc_a)
    assert isinstance(result, int), "delete_extraction_layer_graph_sync must return int"
    assert result >= 0

    # ---- MUST-delete assertions -----------------------------------------

    assert _count_vertices(store, "SECTION", doc_a) == 0, (
        "SECTION for DOC_A must be deleted by rollback"
    )

    assert _count_has_provenance_to_doc(store, doc_a) == 0, (
        "HAS_PROVENANCE edges to Document(DOC_A) must be deleted"
    )

    assert _count_domain_edges_for_doc(store, "INSTALLED_ON", doc_a) == 0, (
        "INSTALLED_ON edge must no longer list DOC_A in document_ids after rollback"
    )

    # ---- MUST-NOT-delete assertions -------------------------------------

    assert _count_vertices(store, "TextChunk", doc_a) >= 1, (
        "TextChunk for DOC_A must be PRESERVED (chunks owned by upstream stage)"
    )

    assert _count_vertices_by_name(store, "RADAR_SYSTEM", radar_name) >= 1, (
        "Global RADAR_SYSTEM vertex must be PRESERVED (not an orphan-cleanup target)"
    )

    # Document vertex must survive
    doc_vertex_rows = store._client.query_sync(
        store._database, "sql",
        "SELECT count(*) AS n FROM Document WHERE document_id = :doc_id",
        {"doc_id": doc_a},
    )
    doc_vertex_count = int(doc_vertex_rows[0]["n"]) if doc_vertex_rows else 0
    assert doc_vertex_count >= 1, (
        "Structural Document vertex for DOC_A must be PRESERVED"
    )

    # HAS_PROVENANCE to DOC_B must survive
    assert _count_has_provenance_to_doc(store, doc_b) >= 1, (
        "HAS_PROVENANCE edges to DOC_B must be PRESERVED (different document)"
    )

    # MENTIONED_IN derive_rules structural edge:
    # The edge from RADAR_SYSTEM to TC_A should be deleted since TC_A is a
    # TextChunk with document_id=doc_a.  ArcadeDB uses @in.<prop> for
    # edge-endpoint property traversal in WHERE clauses.
    rows = store._client.query_sync(
        store._database, "sql",
        "SELECT count(*) AS n FROM MENTIONED_IN WHERE @in.document_id = :doc_id",
        {"doc_id": doc_a},
    )
    mentioned_in_count = int(rows[0]["n"]) if rows else 0
    assert mentioned_in_count == 0, (
        "MENTIONED_IN structural edges (derive_rules) pointing at TC_A "
        "must be deleted"
    )
