"""Integration tests for the ontology graph stage of the ingest pipeline.

These tests call derive_ontology_graph directly (bypassing Celery), using a
real test database session. They verify that:
  1. NER extraction produces entities/relationships from text elements
  2. Graph data is stored in document_graph_extractions
  3. upsert_node/upsert_relationship are called for AGE import
"""

import uuid
from unittest.mock import patch, MagicMock

import pytest

pytestmark = pytest.mark.integration

MILITARY_TEXT = """
TECHNICAL MANUAL: Patriot PAC-3 Guidance Computer

NSN: 1410-01-234-5678
Part Number: GC-4521-A
CAGE Code: 12345

The MK-4 guidance computer is a subsystem of the Patriot PAC-3 missile system.
Compliance: MIL-STD-1553B, MIL-DTL-38999.

Specifications: 28 VDC input, 1000 MHz processor, 5000 hours MTBF.
"""


@pytest.fixture
def sample_document_id(db_session) -> str:
    """Create a minimal Source + Document record and return the document UUID."""
    from app.models.ingest import Source, Document

    source = Source(
        id=uuid.uuid4(),
        name=f"Pipeline Graph Test Source {uuid.uuid4().hex[:8]}",
        created_by=uuid.UUID("00000000-0000-0000-0000-000000000001"),
    )
    db_session.add(source)
    db_session.flush()

    document = Document(
        id=uuid.uuid4(),
        source_id=source.id,
        filename="test_manual.pdf",
        storage_bucket="test-bucket",
        storage_key="test-key.pdf",
        uploaded_by=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        pipeline_status="PROCESSING",
    )
    db_session.add(document)
    db_session.flush()
    return str(document.id)


@pytest.fixture
def sample_document_element(db_session, sample_document_id) -> "DocumentElement":
    """Create a text document element with military content."""
    from app.models.ingest import DocumentElement
    import hashlib

    element_hash = hashlib.sha256(
        f"{sample_document_id}:elem-0:{MILITARY_TEXT}".encode()
    ).hexdigest()

    element = DocumentElement(
        id=uuid.uuid4(),
        document_id=uuid.UUID(sample_document_id),
        element_uid="elem-0",
        element_type="text",
        element_order=0,
        content_text=MILITARY_TEXT,
        element_metadata={},
        element_hash=element_hash,
    )
    db_session.add(element)
    db_session.flush()
    return element


class TestDeriveOntologyGraph:
    """Tests for the derive_ontology_graph task (called directly)."""

    def test_ner_extraction_produces_entities(
        self, db_session, sample_document_id, sample_document_element
    ):
        """NER should extract entities from military text."""
        from app.workers.pipeline import derive_ontology_graph

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.services.graph.upsert_node", return_value="mock-node-id") as mock_node,
            patch("app.services.graph.upsert_relationship", return_value=True) as mock_rel,
        ):
            result = derive_ontology_graph.run(sample_document_id)

        assert result["status"] == "ok"
        assert result["nodes"] > 0
        assert mock_node.call_count > 0

    def test_graph_data_stored_in_extraction_table(
        self, db_session, sample_document_id, sample_document_element
    ):
        """Graph extraction should be stored in document_graph_extractions."""
        from app.workers.pipeline import derive_ontology_graph
        from app.models.ingest import DocumentGraphExtraction
        from sqlalchemy import select

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.services.graph.upsert_node", return_value="mock-id"),
            patch("app.services.graph.upsert_relationship", return_value=True),
        ):
            derive_ontology_graph.run(sample_document_id)

        db_session.expire_all()
        extraction = db_session.execute(
            select(DocumentGraphExtraction)
            .where(DocumentGraphExtraction.document_id == uuid.UUID(sample_document_id))
        ).scalar_one_or_none()

        assert extraction is not None
        assert extraction.status == "COMPLETE"
        assert extraction.provider == "ner"
        graph_json = extraction.graph_json
        assert "nodes" in graph_json
        assert "edges" in graph_json
        assert len(graph_json["nodes"]) > 0

    def test_entity_dicts_have_required_keys(
        self, db_session, sample_document_id, sample_document_element
    ):
        """Each node dict must have entity_type, name, confidence."""
        from app.workers.pipeline import derive_ontology_graph
        from app.models.ingest import DocumentGraphExtraction
        from sqlalchemy import select

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.services.graph.upsert_node", return_value="mock-id"),
            patch("app.services.graph.upsert_relationship", return_value=True),
        ):
            derive_ontology_graph.run(sample_document_id)

        db_session.expire_all()
        extraction = db_session.execute(
            select(DocumentGraphExtraction)
            .where(DocumentGraphExtraction.document_id == uuid.UUID(sample_document_id))
        ).scalar_one_or_none()

        for node in extraction.graph_json["nodes"]:
            assert "entity_type" in node
            assert "name" in node
            assert "confidence" in node

    def test_no_elements_is_noop(self, db_session, sample_document_id):
        """No document elements → no extraction (ok status, 0 nodes)."""
        from app.workers.pipeline import derive_ontology_graph

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
        ):
            result = derive_ontology_graph.run(sample_document_id)

        assert result["status"] == "ok"
        assert result["nodes"] == 0
        assert result["edges"] == 0

    def test_upsert_relationship_called_for_edges(
        self, db_session, sample_document_id, sample_document_element
    ):
        """upsert_relationship should be called for discovered relationships."""
        from app.workers.pipeline import derive_ontology_graph

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.services.graph.upsert_node", return_value="mock-id"),
            patch("app.services.graph.upsert_relationship", return_value=True) as mock_rel,
        ):
            result = derive_ontology_graph.run(sample_document_id)

        if result["edges"] > 0:
            assert mock_rel.call_count > 0
            call_kwargs = mock_rel.call_args.kwargs
            assert "from_name" in call_kwargs
            assert "to_name" in call_kwargs
            assert "rel_type" in call_kwargs
