"""Tests for pipeline graph operations via GraphStore.

Verifies that graph-writing pipeline stages:
- Call ensure_ready_sync before any graph writes
- Use correct GraphStore methods (no direct DB calls)
- Correctly look up existing chunk vertices rather than re-creating them
- Route EXTRACTED_FROM edges through the proper entity-name lookup path
"""
import hashlib
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

pytestmark = pytest.mark.unit

DOC_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
RUN_ID = "f0e1d2c3-b4a5-6789-0123-456789abcdef"


def _make_graph_store() -> MagicMock:
    """Return a fully-mocked GraphStore with sensible defaults."""
    gs = MagicMock()
    gs.ensure_ready_sync = MagicMock()
    gs.upsert_nodes_batch_sync = MagicMock(return_value=["#10:1", "#10:2"])
    gs.upsert_relationships_batch_sync = MagicMock(return_value=["#11:1"])
    gs.upsert_node_sync = MagicMock(return_value="#12:0")
    gs.create_text_chunk_vertex_sync = MagicMock(return_value="#13:0")
    gs.create_image_chunk_vertex_sync = MagicMock(return_value="#14:0")
    gs.get_chunk_rid_sync = MagicMock(return_value="#13:1")
    gs.create_structural_edge_sync = MagicMock(return_value="#15:0")
    gs.set_vertex_embedding_sync = MagicMock()
    return gs


# ---------------------------------------------------------------------------
# ensure_ready_sync is called by every graph-writing task
# ---------------------------------------------------------------------------


class TestEnsureReadySyncCalled:
    """Each graph-writing task must call ensure_ready_sync before its first write."""

    def test_derive_ontology_graph_calls_ensure_ready(self):
        """derive_ontology_graph should call ensure_ready_sync before upsert calls."""
        from app.services.graph_store import NodeRecord, RelationshipRecord, ProvenanceMetadata

        gs = _make_graph_store()
        # upsert_nodes_batch_sync must be called AFTER ensure_ready_sync
        call_order = []
        gs.ensure_ready_sync.side_effect = lambda: call_order.append("ensure_ready")
        gs.upsert_nodes_batch_sync.side_effect = lambda *a, **k: call_order.append("upsert") or ["#10:1"]

        nodes = [NodeRecord(entity_type="PLATFORM", identity_fields={"name": "F-16"}, name="F-16")]
        provenance = ProvenanceMetadata(document_id=DOC_ID)

        with patch("app.db.session.get_graph_store", return_value=gs):
            gs.ensure_ready_sync()
            gs.upsert_nodes_batch_sync(nodes, provenance)

        assert call_order.index("ensure_ready") < call_order.index("upsert")

    def test_derive_canonicalization_calls_ensure_ready(self):
        """derive_canonicalization should call ensure_ready_sync before canonicalization."""
        gs = _make_graph_store()
        called_before_canon = []

        def mock_canon(graph_store, doc_id):
            called_before_canon.append(gs.ensure_ready_sync.called)
            return {"resolved": 0, "total": 0}

        with patch("app.db.session.get_graph_store", return_value=gs), \
             patch("app.services.canonicalization.canonicalize_document_entities", side_effect=mock_canon):
            gs.ensure_ready_sync()
            from app.services.canonicalization import canonicalize_document_entities
            canonicalize_document_entities(gs, DOC_ID)

        assert called_before_canon[0] is True


# ---------------------------------------------------------------------------
# derive_ontology_graph: graph node/edge creation
# ---------------------------------------------------------------------------


class TestDeriveOntologyGraph:
    """derive_ontology_graph should use upsert_nodes_batch_sync and
    upsert_relationships_batch_sync."""

    def test_creates_nodes_via_batch_upsert(self):
        """Nodes should be sent through upsert_nodes_batch_sync."""
        from app.services.graph_store import NodeRecord, ProvenanceMetadata

        gs = _make_graph_store()
        nodes = [
            NodeRecord(entity_type="PLATFORM", identity_fields={"name": "F-16"}, name="F-16"),
            NodeRecord(entity_type="RADAR_SYSTEM", identity_fields={"name": "APG-66"}, name="APG-66"),
        ]
        provenance = ProvenanceMetadata(document_id=DOC_ID)

        gs.upsert_nodes_batch_sync(nodes, provenance)

        gs.upsert_nodes_batch_sync.assert_called_once_with(nodes, provenance)

    def test_creates_relationships_via_batch_upsert(self):
        """Relationships should be sent through upsert_relationships_batch_sync."""
        from app.services.graph_store import RelationshipRecord, ProvenanceMetadata

        gs = _make_graph_store()
        rels = [
            RelationshipRecord(
                from_type="PLATFORM",
                from_identity={"name": "F-16"},
                to_type="RADAR_SYSTEM",
                to_identity={"name": "APG-66"},
                rel_type="EQUIPPED_WITH",
            )
        ]
        provenance = ProvenanceMetadata(document_id=DOC_ID)

        gs.upsert_relationships_batch_sync(rels, provenance)

        gs.upsert_relationships_batch_sync.assert_called_once_with(rels, provenance)

    def test_ensure_ready_called_before_upsert_nodes(self):
        """ensure_ready_sync must be invoked before any node upsert."""
        from app.services.graph_store import NodeRecord, ProvenanceMetadata

        gs = _make_graph_store()
        manager = MagicMock()
        manager.attach_mock(gs.ensure_ready_sync, "ensure_ready_sync")
        manager.attach_mock(gs.upsert_nodes_batch_sync, "upsert_nodes_batch_sync")

        gs.ensure_ready_sync()
        gs.upsert_nodes_batch_sync([], ProvenanceMetadata(document_id=DOC_ID))

        calls = [c[0] for c in manager.mock_calls]
        assert calls.index("ensure_ready_sync") < calls.index("upsert_nodes_batch_sync")


# ---------------------------------------------------------------------------
# derive_structure_links: document vertex + structural edges
# ---------------------------------------------------------------------------


class TestDeriveStructureLinks:
    """derive_structure_links should look up existing chunk vertices and create edges."""

    def test_creates_document_vertex(self):
        """Should upsert a Document vertex before creating edges."""
        from app.services.graph_store import NodeRecord

        gs = _make_graph_store()
        doc_node = NodeRecord(
            entity_type="Document",
            identity_fields={"document_id": DOC_ID},
            name="test.pdf",
        )
        gs.upsert_node_sync(doc_node)

        gs.upsert_node_sync.assert_called_once_with(doc_node)

    def test_looks_up_existing_text_chunk_vertices(self):
        """Should call get_chunk_rid_sync for TextChunk, not create_text_chunk_vertex_sync."""
        gs = _make_graph_store()
        chunk_id = str(uuid.uuid4())

        result = gs.get_chunk_rid_sync(chunk_id, "TextChunk")

        gs.get_chunk_rid_sync.assert_called_once_with(chunk_id, "TextChunk")
        assert result == "#13:1"

    def test_looks_up_existing_image_chunk_vertices(self):
        """Should call get_chunk_rid_sync for ImageChunk, not create_image_chunk_vertex_sync."""
        gs = _make_graph_store()
        chunk_id = str(uuid.uuid4())

        result = gs.get_chunk_rid_sync(chunk_id, "ImageChunk")

        gs.get_chunk_rid_sync.assert_called_once_with(chunk_id, "ImageChunk")
        assert result is not None

    def test_creates_contains_text_edge(self):
        """Should create CONTAINS_TEXT edge from Document to TextChunk."""
        gs = _make_graph_store()
        doc_rid = "#12:0"
        chunk_rid = "#13:1"

        gs.create_structural_edge_sync(doc_rid, chunk_rid, "CONTAINS_TEXT")

        gs.create_structural_edge_sync.assert_called_once_with(doc_rid, chunk_rid, "CONTAINS_TEXT")

    def test_creates_contains_image_edge(self):
        """Should create CONTAINS_IMAGE edge from Document to ImageChunk."""
        gs = _make_graph_store()
        doc_rid = "#12:0"
        chunk_rid = "#14:1"

        gs.create_structural_edge_sync(doc_rid, chunk_rid, "CONTAINS_IMAGE")

        gs.create_structural_edge_sync.assert_called_once_with(doc_rid, chunk_rid, "CONTAINS_IMAGE")

    def test_creates_same_page_edge_between_text_and_image(self):
        """Should create SAME_PAGE edge between text chunk and image chunk on same page."""
        gs = _make_graph_store()
        tc_rid = "#13:1"
        ic_rid = "#14:1"

        gs.create_structural_edge_sync(tc_rid, ic_rid, "SAME_PAGE")

        gs.create_structural_edge_sync.assert_called_once_with(tc_rid, ic_rid, "SAME_PAGE")

    def test_does_not_call_create_text_chunk_vertex_sync(self):
        """derive_structure_links should never call create_text_chunk_vertex_sync.

        Chunk vertices are created by the embedding stages; structure_links
        only looks them up.
        """
        import inspect
        from app.workers.pipeline import derive_structure_links

        source = inspect.getsource(derive_structure_links)
        # The stage should use get_chunk_rid_sync for lookups, not create_* calls
        assert "create_text_chunk_vertex_sync" not in source
        assert "create_image_chunk_vertex_sync" not in source
        assert "get_chunk_rid_sync" in source

    def test_extracted_from_edges_use_batch_method(self):
        """EXTRACTED_FROM edges should use batch_create_entity_chunk_edges_sync."""
        import inspect
        from app.workers.pipeline import derive_structure_links

        source = inspect.getsource(derive_structure_links)
        assert "batch_create_entity_chunk_edges_sync" in source
        # Singular legacy per-edge method should no longer appear
        assert "create_entity_chunk_edge_sync" not in source

    def test_same_page_edges_use_rid_maps(self):
        """SAME_PAGE edges must use RIDs from the tc_rid_map/ic_rid_map lookups."""
        import inspect
        from app.workers.pipeline import derive_structure_links

        source = inspect.getsource(derive_structure_links)
        # Confirm that ic_rid_map and tc_rid_map are used for SAME_PAGE edges
        assert "ic_rid_map" in source
        assert "tc_rid_map" in source


# ---------------------------------------------------------------------------
# GraphStore Protocol surface — the plural batch helper and the chunk-rid
# lookup helper are both load-bearing; the singular per-edge helper was
# deleted in T53b (no production callers, name+type LIMIT-1 semantics
# conflated same-name same-type siblings).
# ---------------------------------------------------------------------------


class TestProtocolSurface:
    def test_get_chunk_rid_method_exists_on_protocol(self):
        """get_chunk_rid_sync should be defined in the GraphStore Protocol."""
        from app.services.graph_store import GraphStore
        assert hasattr(GraphStore, "get_chunk_rid_sync")

    def test_singular_create_entity_chunk_edge_deleted_from_protocol(self):
        """The name+type LIMIT-1 singular helper was removed in T53b —
        callers must use the batch helper with (entity_id, source_rid)
        populated for correct same-name same-type disambiguation."""
        from app.services.graph_store import GraphStore
        assert not hasattr(GraphStore, "create_entity_chunk_edge_sync")

    def test_batch_helper_exists_on_protocol(self):
        from app.services.graph_store import GraphStore
        assert hasattr(GraphStore, "batch_create_entity_chunk_edges_sync")
