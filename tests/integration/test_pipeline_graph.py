"""Integration tests for the ontology graph stage of the ingest pipeline.

These tests call derive_ontology_graph directly (bypassing Celery), using a
real test database session. They verify that:
  1. NER extraction produces entities/relationships from text elements
  2. Graph data is stored in document_graph_extractions
  3. upsert_node/upsert_relationship are called for Neo4j import
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
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_node", return_value="mock-node-id") as mock_node,
            patch("app.services.neo4j_graph.upsert_relationship", return_value=True) as mock_rel,
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
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_node", return_value="mock-id"),
            patch("app.services.neo4j_graph.upsert_relationship", return_value=True),
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
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_node", return_value="mock-id"),
            patch("app.services.neo4j_graph.upsert_relationship", return_value=True),
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
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
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
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_node", return_value="mock-id"),
            patch("app.services.neo4j_graph.upsert_relationship", return_value=True) as mock_rel,
        ):
            result = derive_ontology_graph.run(sample_document_id)

        if result["edges"] > 0:
            assert mock_rel.call_count > 0
            call_kwargs = mock_rel.call_args.kwargs
            assert "from_name" in call_kwargs
            assert "to_name" in call_kwargs
            assert "rel_type" in call_kwargs


class TestDoclingGraphPath:
    """Tests for the docling-graph LLM extraction path (mocked LLM)."""

    def test_docling_graph_provider_set_correctly(
        self, db_session, sample_document_id, sample_document_element
    ):
        """When LLM extraction succeeds, provider should be 'docling-graph'."""
        import networkx as nx
        from app.workers.pipeline import derive_ontology_graph
        from app.models.ingest import DocumentGraphExtraction
        from sqlalchemy import select

        # Build a fake graph that _extract_single_pass would return
        fake_graph = nx.DiGraph()
        fake_graph.add_node(
            "EQUIPMENT_SYSTEM:Patriot PAC-3",
            entity_type="EQUIPMENT_SYSTEM", name="Patriot PAC-3",
            properties={"designation": "MIM-104F"}, confidence=0.95,
        )

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_node", return_value="mock-id"),
            patch("app.services.neo4j_graph.upsert_relationship", return_value=True),
            patch(
                "app.services.docling_graph_service.extract_graph_from_text_chunked",
                return_value=(fake_graph, "docling-graph"),
            ),
            patch("app.workers.pipeline.settings") as mock_settings,
        ):
            mock_settings.llm_provider = "ollama"
            mock_settings.graph_extraction_chunk_size = 7000
            mock_settings.graph_extraction_chunk_overlap = 500
            mock_settings.graph_node_min_confidence = 0.6
            mock_settings.graph_rel_min_confidence = 0.55
            mock_settings.docling_graph_model = "llama3.2"
            mock_settings.graph_max_retries = 2
            mock_settings.graph_retry_delay = 60
            mock_settings.graph_soft_time_limit = 600
            mock_settings.graph_time_limit = 660
            result = derive_ontology_graph.run(sample_document_id)

        assert result["nodes"] >= 1

        db_session.expire_all()
        extraction = db_session.execute(
            select(DocumentGraphExtraction)
            .where(DocumentGraphExtraction.document_id == uuid.UUID(sample_document_id))
        ).scalar_one_or_none()
        assert extraction is not None
        assert extraction.provider == "docling-graph"

    def test_silent_fallback_sets_provider_ner(
        self, db_session, sample_document_id, sample_document_element
    ):
        """When LLM fails silently, provider should be 'ner'."""
        import networkx as nx
        from app.workers.pipeline import derive_ontology_graph
        from app.models.ingest import DocumentGraphExtraction
        from sqlalchemy import select

        # Simulate NER fallback returning a graph with "ner" provider
        fake_graph = nx.DiGraph()
        fake_graph.add_node(
            "STANDARD:MIL-STD-1553B",
            entity_type="STANDARD", name="MIL-STD-1553B",
            properties={}, confidence=0.7,
        )

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_node", return_value="mock-id"),
            patch("app.services.neo4j_graph.upsert_relationship", return_value=True),
            patch(
                "app.services.docling_graph_service.extract_graph_from_text_chunked",
                return_value=(fake_graph, "ner"),
            ),
            patch("app.workers.pipeline.settings") as mock_settings,
        ):
            mock_settings.llm_provider = "ollama"
            mock_settings.graph_extraction_chunk_size = 7000
            mock_settings.graph_extraction_chunk_overlap = 500
            mock_settings.graph_node_min_confidence = 0.6
            mock_settings.graph_rel_min_confidence = 0.55
            mock_settings.docling_graph_model = "llama3.2"
            mock_settings.graph_max_retries = 2
            mock_settings.graph_retry_delay = 60
            mock_settings.graph_soft_time_limit = 600
            mock_settings.graph_time_limit = 660
            result = derive_ontology_graph.run(sample_document_id)

        db_session.expire_all()
        extraction = db_session.execute(
            select(DocumentGraphExtraction)
            .where(DocumentGraphExtraction.document_id == uuid.UUID(sample_document_id))
        ).scalar_one_or_none()
        assert extraction is not None
        assert extraction.provider == "ner"

    def test_node_properties_passed_to_upsert(
        self, db_session, sample_document_id, sample_document_element
    ):
        """Node properties dict should be passed to upsert_node."""
        import networkx as nx
        from app.workers.pipeline import derive_ontology_graph

        fake_graph = nx.DiGraph()
        fake_graph.add_node(
            "EQUIPMENT_SYSTEM:Patriot",
            entity_type="EQUIPMENT_SYSTEM", name="Patriot",
            properties={"designation": "MIM-104F", "nsn": "1410-01-234-5678"},
            confidence=0.95,
        )

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.workers.pipeline._get_pipeline_run_id", return_value=None),
            patch("app.db.session.get_neo4j_driver", return_value=MagicMock()),
            patch("app.services.neo4j_graph.upsert_node", return_value="mock-id") as mock_node,
            patch("app.services.neo4j_graph.upsert_relationship", return_value=True),
            patch(
                "app.services.docling_graph_service.extract_graph_from_text_chunked",
                return_value=(fake_graph, "docling-graph"),
            ),
            patch("app.workers.pipeline.settings") as mock_settings,
        ):
            mock_settings.llm_provider = "ollama"
            mock_settings.graph_extraction_chunk_size = 7000
            mock_settings.graph_extraction_chunk_overlap = 500
            mock_settings.graph_node_min_confidence = 0.6
            mock_settings.graph_rel_min_confidence = 0.55
            mock_settings.docling_graph_model = "llama3.2"
            mock_settings.graph_max_retries = 2
            mock_settings.graph_retry_delay = 60
            mock_settings.graph_soft_time_limit = 600
            mock_settings.graph_time_limit = 660
            derive_ontology_graph.run(sample_document_id)

        assert mock_node.call_count >= 1
        call_kwargs = mock_node.call_args.kwargs
        assert "properties" in call_kwargs
        assert call_kwargs["properties"]["designation"] == "MIM-104F"
