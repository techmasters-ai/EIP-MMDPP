"""Integration tests for the ontology graph stage of the ingest pipeline.

These tests call derive_ontology_graph directly (bypassing Celery), using a
real test database session. They verify that:
  1. Docling-Graph service extraction produces entities/relationships
  2. Graph data is stored in document_graph_extractions
  3. upsert_nodes_batch/upsert_relationships_batch are called for Neo4j import
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

# Mock response from the Docling-Graph service
MOCK_EXTRACTION_RESULT = {
    "entities": [
        {"name": "Patriot PAC-3", "entity_type": "EQUIPMENT_SYSTEM", "confidence": 0.95, "properties": {"designation": "MIM-104F"}},
        {"name": "MK-4", "entity_type": "COMPONENT", "confidence": 0.9, "properties": {}},
        {"name": "MIL-STD-1553B", "entity_type": "STANDARD", "confidence": 0.85, "properties": {}},
    ],
    "relationships": [
        {"from_name": "MK-4", "from_type": "COMPONENT", "rel_type": "SUBSYSTEM_OF", "to_name": "Patriot PAC-3", "to_type": "EQUIPMENT_SYSTEM", "confidence": 0.88},
        {"from_name": "Patriot PAC-3", "from_type": "EQUIPMENT_SYSTEM", "rel_type": "COMPLIES_WITH", "to_name": "MIL-STD-1553B", "to_type": "STANDARD", "confidence": 0.8},
    ],
    "ontology_version": "3.0.0",
    "model": "llama3.2",
    "provider": "ollama",
}


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

    def test_extraction_produces_entities(
        self, db_session, sample_document_id, sample_document_element
    ):
        """Docling-Graph service should extract entities from text."""
        from app.workers.pipeline import derive_ontology_graph

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_nodes_batch", return_value=3) as mock_nodes,
            patch("app.services.neo4j_graph.upsert_relationships_batch", return_value=2),
            patch(
                "app.services.docling_graph_service.extract_graph",
                return_value=MOCK_EXTRACTION_RESULT,
            ),
        ):
            result = derive_ontology_graph.run(sample_document_id)

        assert result["status"] == "ok"
        assert result["nodes"] == 3
        assert mock_nodes.call_count == 1

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
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_nodes_batch", return_value=3),
            patch("app.services.neo4j_graph.upsert_relationships_batch", return_value=2),
            patch(
                "app.services.docling_graph_service.extract_graph",
                return_value=MOCK_EXTRACTION_RESULT,
            ),
        ):
            derive_ontology_graph.run(sample_document_id)

        db_session.expire_all()
        extraction = db_session.execute(
            select(DocumentGraphExtraction)
            .where(DocumentGraphExtraction.document_id == uuid.UUID(sample_document_id))
        ).scalar_one_or_none()

        assert extraction is not None
        assert extraction.status == "COMPLETE"
        assert extraction.provider == "ollama"
        graph_json = extraction.graph_json
        assert "nodes" in graph_json
        assert "edges" in graph_json
        assert len(graph_json["nodes"]) == 3

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
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_nodes_batch", return_value=3),
            patch("app.services.neo4j_graph.upsert_relationships_batch", return_value=2),
            patch(
                "app.services.docling_graph_service.extract_graph",
                return_value=MOCK_EXTRACTION_RESULT,
            ),
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
        """No document elements -> no extraction (ok status, 0 nodes)."""
        from app.workers.pipeline import derive_ontology_graph

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
        ):
            result = derive_ontology_graph.run(sample_document_id)

        assert result["status"] == "ok"
        assert result["nodes"] == 0
        assert result["edges"] == 0

    def test_relationships_batch_called_for_edges(
        self, db_session, sample_document_id, sample_document_element
    ):
        """upsert_relationships_batch should be called for discovered relationships."""
        from app.workers.pipeline import derive_ontology_graph

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_nodes_batch", return_value=3),
            patch("app.services.neo4j_graph.upsert_relationships_batch", return_value=2) as mock_rels,
            patch(
                "app.services.docling_graph_service.extract_graph",
                return_value=MOCK_EXTRACTION_RESULT,
            ),
        ):
            result = derive_ontology_graph.run(sample_document_id)

        assert result["edges"] == 2
        assert mock_rels.call_count == 1
        # Verify edge dicts have correct keys
        edges_arg = mock_rels.call_args[0][1]  # second positional arg
        for edge in edges_arg:
            assert "from_name" in edge
            assert "to_name" in edge
            assert "rel_type" in edge
            assert "artifact_id" in edge


class TestDoclingGraphPath:
    """Tests for the Docling-Graph service extraction path."""

    def test_provider_set_from_response(
        self, db_session, sample_document_id, sample_document_element
    ):
        """Provider in extraction record should come from service response."""
        from app.workers.pipeline import derive_ontology_graph
        from app.models.ingest import DocumentGraphExtraction
        from sqlalchemy import select

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_nodes_batch", return_value=3),
            patch("app.services.neo4j_graph.upsert_relationships_batch", return_value=2),
            patch(
                "app.services.docling_graph_service.extract_graph",
                return_value=MOCK_EXTRACTION_RESULT,
            ),
        ):
            result = derive_ontology_graph.run(sample_document_id)

        assert result["nodes"] >= 1

        db_session.expire_all()
        extraction = db_session.execute(
            select(DocumentGraphExtraction)
            .where(DocumentGraphExtraction.document_id == uuid.UUID(sample_document_id))
        ).scalar_one_or_none()
        assert extraction is not None
        assert extraction.provider == "ollama"
        assert extraction.model_name == "llama3.2"

    def test_confidence_gating_filters_low_confidence(
        self, db_session, sample_document_id, sample_document_element
    ):
        """Entities below min confidence should be rejected."""
        from app.workers.pipeline import derive_ontology_graph

        low_conf_result = {
            "entities": [
                {"name": "LowConf", "entity_type": "UNKNOWN", "confidence": 0.1, "properties": {}},
                {"name": "HighConf", "entity_type": "EQUIPMENT_SYSTEM", "confidence": 0.95, "properties": {}},
            ],
            "relationships": [],
            "ontology_version": "3.0.0",
            "model": "llama3.2",
            "provider": "ollama",
        }

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_nodes_batch", return_value=1) as mock_nodes,
            patch("app.services.neo4j_graph.upsert_relationships_batch", return_value=0),
            patch(
                "app.services.docling_graph_service.extract_graph",
                return_value=low_conf_result,
            ),
        ):
            result = derive_ontology_graph.run(sample_document_id)

        # Only HighConf should pass the confidence gate
        assert mock_nodes.call_count == 1
        batch_arg = mock_nodes.call_args[0][1]
        assert len(batch_arg) == 1
        assert batch_arg[0]["name"] == "HighConf"

    def test_ingest_filter_metadata_stored(
        self, db_session, sample_document_id, sample_document_element
    ):
        """graph_json should contain _ingest_filter metadata."""
        from app.workers.pipeline import derive_ontology_graph
        from app.models.ingest import DocumentGraphExtraction
        from sqlalchemy import select

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_nodes_batch", return_value=3),
            patch("app.services.neo4j_graph.upsert_relationships_batch", return_value=2),
            patch(
                "app.services.docling_graph_service.extract_graph",
                return_value=MOCK_EXTRACTION_RESULT,
            ),
        ):
            derive_ontology_graph.run(sample_document_id)

        db_session.expire_all()
        extraction = db_session.execute(
            select(DocumentGraphExtraction)
            .where(DocumentGraphExtraction.document_id == uuid.UUID(sample_document_id))
        ).scalar_one_or_none()

        assert "_ingest_filter" in extraction.graph_json
        filt = extraction.graph_json["_ingest_filter"]
        assert "node_min_confidence" in filt
        assert "rel_min_confidence" in filt

    def test_http_error_triggers_retry(
        self, db_session, sample_document_id, sample_document_element
    ):
        """httpx.HTTPStatusError should trigger a Celery retry."""
        import httpx
        from app.workers.pipeline import derive_ontology_graph
        from celery.exceptions import Retry as CeleryRetry

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.request = MagicMock()

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch(
                "app.services.docling_graph_service.extract_graph",
                side_effect=httpx.HTTPStatusError(
                    "Service Unavailable", request=mock_response.request, response=mock_response,
                ),
            ),
        ):
            with pytest.raises(CeleryRetry):
                derive_ontology_graph.run(sample_document_id)

    def test_deterministic_error_no_retry(
        self, db_session, sample_document_id, sample_document_element
    ):
        """DeterministicExtractionError should NOT retry."""
        from app.workers.pipeline import derive_ontology_graph
        from app.services.docling_graph_service import DeterministicExtractionError

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch(
                "app.services.docling_graph_service.extract_graph",
                side_effect=DeterministicExtractionError("empty response"),
            ),
        ):
            result = derive_ontology_graph.run(sample_document_id)

        assert result["status"] == "failed"
        assert "empty response" in result["error"]
