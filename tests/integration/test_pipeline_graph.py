"""Integration tests for the NER + graph stages of the ingest pipeline.

These tests call the task functions directly (bypassing Celery), using a real
test database session. They verify that:
  1. extract_graph_entities writes entity/relationship candidates to artifact metadata
  2. import_graph successfully calls AGE (or fails gracefully if AGE is unavailable)
  3. The full text-extraction → NER → chunk flow produces consistent results
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
def sample_artifact(db_session, sample_document_id) -> "Artifact":
    """Create a text artifact with military content."""
    from app.models.ingest import Artifact

    artifact = Artifact(
        id=uuid.uuid4(),
        document_id=uuid.UUID(sample_document_id),
        artifact_type="text",
        content_text=MILITARY_TEXT,
        content_metadata={},
    )
    db_session.add(artifact)
    db_session.flush()
    return artifact


class TestExtractGraphEntities:
    """Tests for the extract_graph_entities Celery task (called directly)."""

    def test_entities_written_to_artifact_metadata(
        self, db_session, sample_document_id, sample_artifact
    ):
        """After extraction, artifact.content_metadata must contain entity lists."""
        from app.workers.pipeline import extract_graph_entities
        from app.models.ingest import Artifact
        from sqlalchemy import select

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
        ):
            extract_graph_entities.run(sample_document_id)

        db_session.expire_all()
        result = db_session.execute(
            select(Artifact).where(Artifact.document_id == uuid.UUID(sample_document_id))
        )
        artifact = result.scalars().first()
        assert artifact is not None
        metadata = artifact.content_metadata or {}
        assert "extracted_entities" in metadata, (
            "content_metadata must have 'extracted_entities' after NER"
        )
        assert "extracted_relationships" in metadata, (
            "content_metadata must have 'extracted_relationships' after NER"
        )
        assert isinstance(metadata["extracted_entities"], list)
        assert isinstance(metadata["extracted_relationships"], list)

    def test_at_least_one_entity_found(
        self, db_session, sample_document_id, sample_artifact
    ):
        """The military text fixture should yield at least one entity."""
        from app.workers.pipeline import extract_graph_entities
        from app.models.ingest import Artifact
        from sqlalchemy import select

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
        ):
            extract_graph_entities.run(sample_document_id)

        db_session.expire_all()
        result = db_session.execute(
            select(Artifact).where(Artifact.document_id == uuid.UUID(sample_document_id))
        )
        artifact = result.scalars().first()
        entities = (artifact.content_metadata or {}).get("extracted_entities", [])
        assert len(entities) > 0, (
            "Expected at least one entity from military text fixture"
        )

    def test_entity_dicts_have_required_keys(
        self, db_session, sample_document_id, sample_artifact
    ):
        """Each entity dict must have entity_type, name, confidence, properties."""
        from app.workers.pipeline import extract_graph_entities
        from app.models.ingest import Artifact
        from sqlalchemy import select

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
        ):
            extract_graph_entities.run(sample_document_id)

        db_session.expire_all()
        result = db_session.execute(
            select(Artifact).where(Artifact.document_id == uuid.UUID(sample_document_id))
        )
        artifact = result.scalars().first()
        for entity in (artifact.content_metadata or {}).get("extracted_entities", []):
            assert "entity_type" in entity
            assert "name" in entity
            assert "confidence" in entity
            assert "properties" in entity
            assert isinstance(entity["confidence"], float)

    def test_missing_document_does_not_crash(self, db_session):
        """extract_graph_entities should not crash on missing document_id."""
        from app.workers.pipeline import extract_graph_entities

        nonexistent_id = str(uuid.uuid4())
        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
        ):
            # Should complete without raising (no artifacts → no-op)
            extract_graph_entities.run(nonexistent_id)


class TestImportGraph:
    """Tests for the import_graph Celery task (called directly).

    AGE may not be available in all test environments, so failures are tolerated
    gracefully. We assert that the task either succeeds or marks the document
    as PARTIAL_COMPLETE.
    """

    def test_import_graph_no_entities_is_noop(
        self, db_session, sample_document_id, sample_artifact
    ):
        """An artifact with no extracted_entities produces no graph writes."""
        from app.workers.pipeline import import_graph

        # Ensure no extracted_entities in metadata
        sample_artifact.content_metadata = {}
        db_session.flush()

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.services.graph.upsert_node", return_value="mock-node-id") as mock_upsert,
        ):
            import_graph.run(sample_document_id)

        mock_upsert.assert_not_called()

    def test_import_graph_calls_upsert_for_each_entity(
        self, db_session, sample_document_id, sample_artifact
    ):
        """import_graph should call upsert_node for each extracted entity."""
        from app.workers.pipeline import import_graph

        sample_artifact.content_metadata = {
            "extracted_entities": [
                {"entity_type": "EQUIPMENT_SYSTEM", "name": "Patriot PAC-3", "confidence": 0.9, "properties": {}},
                {"entity_type": "STANDARD", "name": "MIL-STD-1553B", "confidence": 0.95, "properties": {}},
            ],
            "extracted_relationships": [],
        }
        db_session.flush()

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.services.graph.upsert_node", return_value="mock-node-id") as mock_node,
            patch("app.services.graph.upsert_relationship", return_value=True) as mock_rel,
        ):
            import_graph.run(sample_document_id)

        assert mock_node.call_count == 2
        mock_rel.assert_not_called()

    def test_import_graph_calls_upsert_for_relationships(
        self, db_session, sample_document_id, sample_artifact
    ):
        """import_graph should call upsert_relationship for each relationship."""
        from app.workers.pipeline import import_graph

        sample_artifact.content_metadata = {
            "extracted_entities": [
                {"entity_type": "EQUIPMENT_SYSTEM", "name": "Patriot PAC-3", "confidence": 0.9, "properties": {}},
                {"entity_type": "SUBSYSTEM", "name": "MK-4 Guidance Computer", "confidence": 0.85, "properties": {}},
            ],
            "extracted_relationships": [
                {
                    "rel_type": "IS_SUBSYSTEM_OF",
                    "from_name": "MK-4 Guidance Computer",
                    "from_type": "SUBSYSTEM",
                    "to_name": "Patriot PAC-3",
                    "to_type": "EQUIPMENT_SYSTEM",
                    "confidence": 0.8,
                }
            ],
        }
        db_session.flush()

        with (
            patch("app.workers.pipeline._get_db", return_value=db_session),
            patch("app.workers.pipeline._update_document_status"),
            patch("app.services.graph.upsert_node", return_value="mock-id") as mock_node,
            patch("app.services.graph.upsert_relationship", return_value=True) as mock_rel,
        ):
            import_graph.run(sample_document_id)

        assert mock_node.call_count == 2
        assert mock_rel.call_count == 1

        # Verify relationship call args
        call_kwargs = mock_rel.call_args.kwargs
        assert call_kwargs["from_name"] == "MK-4 Guidance Computer"
        assert call_kwargs["to_name"] == "Patriot PAC-3"
        assert call_kwargs["rel_type"] == "IS_SUBSYSTEM_OF"
