"""Unit tests for the GraphStore Protocol and associated data classes.

Tests focus on dataclass defaults and the runtime-checkable Protocol contract.
No database or network required.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestDataClasses:
    def test_provenance_metadata_defaults(self):
        from app.services.graph_store import ProvenanceMetadata

        p = ProvenanceMetadata(document_id="doc-1")
        assert p.document_id == "doc-1"
        assert p.page_numbers == []
        assert p.upload_datetime is None
        assert p.document_datetime is None

    def test_provenance_metadata_full(self):
        from app.services.graph_store import ProvenanceMetadata

        p = ProvenanceMetadata(
            document_id="doc-2",
            page_numbers=[1, 2, 3],
            upload_datetime="2024-01-01T00:00:00Z",
            document_datetime="2023-06-15",
        )
        assert p.page_numbers == [1, 2, 3]
        assert p.upload_datetime == "2024-01-01T00:00:00Z"
        assert p.document_datetime == "2023-06-15"

    def test_node_record_defaults(self):
        from app.services.graph_store import NodeRecord

        n = NodeRecord(entity_type="RADAR_SYSTEM", identity_fields={"name": "Test"}, name="Test")
        assert n.entity_type == "RADAR_SYSTEM"
        assert n.identity_fields == {"name": "Test"}
        assert n.name == "Test"
        assert n.properties == {}
        assert n.extraction_confidence == 1.0

    def test_node_record_full(self):
        from app.services.graph_store import NodeRecord

        n = NodeRecord(
            entity_type="PLATFORM",
            identity_fields={"name": "F-16"},
            name="F-16",
            properties={"role": "fighter"},
            extraction_confidence=0.95,
        )
        assert n.properties == {"role": "fighter"}
        assert n.extraction_confidence == 0.95

    def test_relationship_record(self):
        from app.services.graph_store import RelationshipRecord

        r = RelationshipRecord(
            from_type="RADAR_SYSTEM",
            from_identity={"name": "A"},
            to_type="PLATFORM",
            to_identity={"name": "B"},
            rel_type="INSTALLED_ON",
        )
        assert r.from_type == "RADAR_SYSTEM"
        assert r.to_type == "PLATFORM"
        assert r.rel_type == "INSTALLED_ON"
        assert r.properties == {}
        assert r.extraction_confidence == 1.0

    def test_relationship_record_full(self):
        from app.services.graph_store import RelationshipRecord

        r = RelationshipRecord(
            from_type="RADAR_SYSTEM",
            from_identity={"name": "APG-77"},
            to_type="PLATFORM",
            to_identity={"name": "F-22"},
            rel_type="INSTALLED_ON",
            properties={"source": "doc-1"},
            extraction_confidence=0.88,
        )
        assert r.properties == {"source": "doc-1"}
        assert r.extraction_confidence == 0.88

    def test_graph_entity_result(self):
        from app.services.graph_store import GraphEntityResult

        g = GraphEntityResult(node_id="rid", name="Test", entity_type="PLATFORM")
        assert g.node_id == "rid"
        assert g.name == "Test"
        assert g.entity_type == "PLATFORM"
        assert g.canonical_name is None
        assert g.extraction_confidence is None
        assert g.properties == {}

    def test_graph_entity_result_full(self):
        from app.services.graph_store import GraphEntityResult

        g = GraphEntityResult(
            node_id="rid-2",
            name="APG-77",
            entity_type="RADAR_SYSTEM",
            canonical_name="APG-77 Radar",
            extraction_confidence=0.9,
            properties={"band": "X"},
        )
        assert g.canonical_name == "APG-77 Radar"
        assert g.extraction_confidence == 0.9
        assert g.properties == {"band": "X"}

    def test_schema_sync_report_defaults(self):
        from app.services.graph_store import SchemaSyncReport

        s = SchemaSyncReport()
        assert s.types_created == 0
        assert s.properties_added == 0
        assert s.indexes_created == 0
        assert s.errors == []

    def test_schema_sync_report_full(self):
        from app.services.graph_store import SchemaSyncReport

        s = SchemaSyncReport(
            types_created=3,
            properties_added=7,
            indexes_created=2,
            errors=["index already exists"],
        )
        assert s.types_created == 3
        assert s.errors == ["index already exists"]

    def test_protocol_is_runtime_checkable(self):
        from app.services.graph_store import GraphStore

        # GraphStore should be a runtime-checkable Protocol
        assert hasattr(GraphStore, "__protocol_attrs__") or hasattr(GraphStore, "_is_protocol")


class TestDeletePrimitives:
    """Spec residual check #1: the new narrower rollback primitive
    delete_extraction_layer_graph_sync must appear on the backend-agnostic
    Protocol declaration so any alternative backend that implements
    GraphStore is forced to honor the contract. The existing broader
    delete_document_graph_sync is kept unchanged for purge callers."""

    def test_delete_extraction_layer_graph_sync_is_declared(self):
        """The new narrower rollback primitive must be part of the protocol.

        `from __future__ import annotations` makes return-type annotations
        strings at runtime, so we use `eval_str=True` to resolve them.
        """
        import inspect

        from app.services.graph_store import GraphStore

        assert hasattr(GraphStore, "delete_extraction_layer_graph_sync")
        sig = inspect.signature(
            GraphStore.delete_extraction_layer_graph_sync, eval_str=True
        )
        params = list(sig.parameters)
        assert params == ["self", "document_id"], (
            f"Unexpected signature: {params}"
        )
        # Return type matches the existing delete_document_graph_sync for
        # logging parity — both return int (count of deletions).
        assert sig.return_annotation is int, (
            f"Expected -> int, got {sig.return_annotation}"
        )

    def test_delete_document_graph_sync_unchanged(self):
        """The existing broader primitive is kept unchanged for purge callers."""
        import inspect

        from app.services.graph_store import GraphStore

        assert hasattr(GraphStore, "delete_document_graph_sync")
        sig = inspect.signature(
            GraphStore.delete_document_graph_sync, eval_str=True
        )
        params = list(sig.parameters)
        assert params == ["self", "document_id"], (
            f"Unexpected signature: {params}"
        )
        assert sig.return_annotation is int, (
            f"delete_document_graph_sync regression: return type changed "
            f"from int to {sig.return_annotation}"
        )
