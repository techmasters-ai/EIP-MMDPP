"""Unit tests for EXTRACTED_FROM edge creation in derive_structure_links.

Tests the new graph_json.mentions path and backward-compat artifact metadata fallback.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("celery", reason="celery not installed")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_element(element_uid, artifact_id, section_path=None, content_text=None):
    elem = MagicMock()
    elem.element_uid = element_uid
    elem.artifact_id = artifact_id
    elem.section_path = section_path
    elem.content_text = content_text
    elem.element_order = 0
    return elem


def _make_text_chunk(chunk_id, artifact_id, page_number=1, chunk_index=0):
    tc = MagicMock()
    tc.id = chunk_id
    tc.artifact_id = artifact_id
    tc.document_id = uuid.uuid4()
    tc.page_number = page_number
    tc.chunk_index = chunk_index
    return tc


def _make_graph_extraction(document_id, graph_json):
    ge = MagicMock()
    ge.document_id = document_id
    ge.graph_json = graph_json
    return ge


def _make_artifact(artifact_id, document_id, content_metadata=None):
    art = MagicMock()
    art.id = artifact_id
    art.document_id = document_id
    art.content_metadata = content_metadata
    return art


def _setup_db_mock(doc, elements, text_chunks, image_chunks,
                   graph_extraction=None, artifacts_with_entities=None):
    """Build a db mock that returns the right results for derive_structure_links queries.

    The function issues these selects in order:
    1. TextChunk (order_by page_number, chunk_index)
    2. ImageChunk
    3. DocumentElement (order_by element_order)
    Then later (EXTRACTED_FROM section):
    4. DocumentGraphExtraction (.first())
    5. Artifact with content_metadata (fallback, .all())
    """
    db = MagicMock()
    call_count = {"n": 0}

    def fake_execute(stmt, params=None):
        result = MagicMock()

        # Advisory lock uses sqlalchemy.text — has a .text attribute
        if hasattr(stmt, 'text') and not hasattr(stmt, 'froms'):
            return result

        call_count["n"] += 1
        n = call_count["n"]

        if n == 1:
            result.scalars.return_value.all.return_value = text_chunks
        elif n == 2:
            result.scalars.return_value.all.return_value = image_chunks
        elif n == 3:
            result.scalars.return_value.all.return_value = elements
        elif n == 4:
            result.scalars.return_value.first.return_value = graph_extraction
        elif n == 5:
            result.scalars.return_value.all.return_value = artifacts_with_entities or []
        else:
            # ChunkLink upserts, commits, etc.
            pass
        return result

    db.execute.side_effect = fake_execute
    db.get.return_value = doc
    return db


# Shared patches for all tests — patches at the source modules
_COMMON_PATCHES = [
    "app.services.neo4j_graph.create_entity_chunk_edge",
    "app.services.neo4j_graph.create_structural_edge",
    "app.services.neo4j_graph.upsert_chunk_ref_node",
    "app.services.neo4j_graph.upsert_document_node",
    "app.db.session.get_neo4j_driver",
]


def _default_settings(mock_settings):
    mock_settings.retrieval_weight_next_chunk = 0.8
    mock_settings.retrieval_weight_same_page = 0.6
    mock_settings.retrieval_weight_same_section = 0.5
    mock_settings.retrieval_weight_same_artifact = 0.4


# ---------------------------------------------------------------------------
# Tests: graph_json mentions path
# ---------------------------------------------------------------------------

class TestMentionsFromGraphJson:
    @patch("app.db.session.get_neo4j_driver")
    @patch("app.services.neo4j_graph.create_entity_chunk_edge", return_value=True)
    @patch("app.services.neo4j_graph.create_structural_edge")
    @patch("app.services.neo4j_graph.upsert_chunk_ref_node")
    @patch("app.services.neo4j_graph.upsert_document_node")
    @patch("app.workers.pipeline._update_stage_run")
    @patch("app.workers.pipeline._get_pipeline_run_id", return_value=None)
    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline.settings")
    def test_mentions_creates_extracted_from_edges(
        self, mock_settings, mock_get_db, mock_status, mock_run_id,
        mock_stage, mock_upsert_doc, mock_upsert_chunk, mock_struct_edge,
        mock_entity_edge, mock_driver,
    ):
        from app.workers.pipeline import derive_structure_links

        doc_id = str(uuid.uuid4())
        art_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        euid = "elem-001"

        doc = MagicMock()
        doc.filename = "test.pdf"
        doc.source_id = uuid.uuid4()

        elements = [_make_element(euid, art_id)]
        text_chunks = [_make_text_chunk(chunk_id, art_id)]
        mentions = [
            {"entity_name": "M1 Abrams", "entity_type": "WEAPON_SYSTEM", "element_uid": euid},
            {"entity_name": "Fort Hood", "entity_type": "LOCATION", "element_uid": euid},
        ]
        graph_extraction = _make_graph_extraction(doc_id, {"mentions": mentions})

        _default_settings(mock_settings)

        db = _setup_db_mock(doc, elements, text_chunks, [], graph_extraction)
        mock_get_db.return_value = db

        result = derive_structure_links(None, doc_id)

        assert result["status"] == "ok"
        # 2 mentions × 1 chunk = 2 calls
        assert mock_entity_edge.call_count == 2
        names = {c.args[1] for c in mock_entity_edge.call_args_list}
        assert "M1 Abrams" in names
        assert "Fort Hood" in names

    @patch("app.db.session.get_neo4j_driver")
    @patch("app.services.neo4j_graph.create_entity_chunk_edge", return_value=True)
    @patch("app.services.neo4j_graph.create_structural_edge")
    @patch("app.services.neo4j_graph.upsert_chunk_ref_node")
    @patch("app.services.neo4j_graph.upsert_document_node")
    @patch("app.workers.pipeline._update_stage_run")
    @patch("app.workers.pipeline._get_pipeline_run_id", return_value=None)
    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline.settings")
    def test_empty_mentions_no_edges(
        self, mock_settings, mock_get_db, mock_status, mock_run_id,
        mock_stage, mock_upsert_doc, mock_upsert_chunk, mock_struct_edge,
        mock_entity_edge, mock_driver,
    ):
        from app.workers.pipeline import derive_structure_links

        doc_id = str(uuid.uuid4())
        doc = MagicMock()
        doc.filename = "test.pdf"
        doc.source_id = uuid.uuid4()

        graph_extraction = _make_graph_extraction(doc_id, {"mentions": []})
        _default_settings(mock_settings)

        db = _setup_db_mock(doc, [], [], [], graph_extraction)
        mock_get_db.return_value = db

        result = derive_structure_links(None, doc_id)

        assert result["status"] == "ok"
        mock_entity_edge.assert_not_called()

    @patch("app.db.session.get_neo4j_driver")
    @patch("app.services.neo4j_graph.create_entity_chunk_edge", return_value=True)
    @patch("app.services.neo4j_graph.create_structural_edge")
    @patch("app.services.neo4j_graph.upsert_chunk_ref_node")
    @patch("app.services.neo4j_graph.upsert_document_node")
    @patch("app.workers.pipeline._update_stage_run")
    @patch("app.workers.pipeline._get_pipeline_run_id", return_value=None)
    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline.settings")
    def test_element_uid_to_chunk_mapping(
        self, mock_settings, mock_get_db, mock_status, mock_run_id,
        mock_stage, mock_upsert_doc, mock_upsert_chunk, mock_struct_edge,
        mock_entity_edge, mock_driver,
    ):
        """Mentions referencing element_uid map to correct chunks via artifact_id."""
        from app.workers.pipeline import derive_structure_links

        doc_id = str(uuid.uuid4())
        art_id_1 = uuid.uuid4()
        art_id_2 = uuid.uuid4()
        chunk_a = uuid.uuid4()
        chunk_b = uuid.uuid4()
        chunk_c = uuid.uuid4()

        doc = MagicMock()
        doc.filename = "test.pdf"
        doc.source_id = uuid.uuid4()

        elements = [
            _make_element("elem-1", art_id_1),
            _make_element("elem-2", art_id_2),
        ]
        text_chunks = [
            _make_text_chunk(chunk_a, art_id_1, chunk_index=0),
            _make_text_chunk(chunk_b, art_id_1, chunk_index=1),
            _make_text_chunk(chunk_c, art_id_2, chunk_index=0),
        ]

        mentions = [
            {"entity_name": "Tank", "entity_type": "EQUIPMENT", "element_uid": "elem-1"},
        ]
        graph_extraction = _make_graph_extraction(doc_id, {"mentions": mentions})
        _default_settings(mock_settings)

        db = _setup_db_mock(doc, elements, text_chunks, [], graph_extraction)
        mock_get_db.return_value = db

        result = derive_structure_links(None, doc_id)

        assert result["status"] == "ok"
        # elem-1 → art_id_1 → chunk_a, chunk_b  (2 calls)
        assert mock_entity_edge.call_count == 2
        chunk_ids_called = {c.args[3] for c in mock_entity_edge.call_args_list}
        assert str(chunk_a) in chunk_ids_called
        assert str(chunk_b) in chunk_ids_called


# ---------------------------------------------------------------------------
# Tests: fallback to Artifact.content_metadata
# ---------------------------------------------------------------------------

class TestFallbackToArtifactMetadata:
    @patch("app.db.session.get_neo4j_driver")
    @patch("app.services.neo4j_graph.create_entity_chunk_edge", return_value=True)
    @patch("app.services.neo4j_graph.create_structural_edge")
    @patch("app.services.neo4j_graph.upsert_chunk_ref_node")
    @patch("app.services.neo4j_graph.upsert_document_node")
    @patch("app.workers.pipeline._update_stage_run")
    @patch("app.workers.pipeline._get_pipeline_run_id", return_value=None)
    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline.settings")
    def test_no_graph_extraction_uses_artifact_metadata(
        self, mock_settings, mock_get_db, mock_status, mock_run_id,
        mock_stage, mock_upsert_doc, mock_upsert_chunk, mock_struct_edge,
        mock_entity_edge, mock_driver,
    ):
        from app.workers.pipeline import derive_structure_links

        doc_id = str(uuid.uuid4())
        art_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        doc = MagicMock()
        doc.filename = "test.pdf"
        doc.source_id = uuid.uuid4()

        elements = [_make_element("elem-1", art_id)]
        text_chunks = [_make_text_chunk(chunk_id, art_id)]

        artifact = _make_artifact(art_id, doc_id, content_metadata={
            "extracted_entities": [
                {"name": "Apache", "entity_type": "HELICOPTER"},
            ]
        })
        _default_settings(mock_settings)

        db = _setup_db_mock(doc, elements, text_chunks, [],
                           graph_extraction=None,
                           artifacts_with_entities=[artifact])
        mock_get_db.return_value = db

        result = derive_structure_links(None, doc_id)

        assert result["status"] == "ok"
        mock_entity_edge.assert_called_once()
        assert mock_entity_edge.call_args.args[1] == "Apache"
        assert mock_entity_edge.call_args.args[2] == "HELICOPTER"

    @patch("app.db.session.get_neo4j_driver")
    @patch("app.services.neo4j_graph.create_entity_chunk_edge", return_value=True)
    @patch("app.services.neo4j_graph.create_structural_edge")
    @patch("app.services.neo4j_graph.upsert_chunk_ref_node")
    @patch("app.services.neo4j_graph.upsert_document_node")
    @patch("app.workers.pipeline._update_stage_run")
    @patch("app.workers.pipeline._get_pipeline_run_id", return_value=None)
    @patch("app.workers.pipeline._update_document_status")
    @patch("app.workers.pipeline._get_db")
    @patch("app.workers.pipeline.settings")
    def test_docling_graph_data_fallback(
        self, mock_settings, mock_get_db, mock_status, mock_run_id,
        mock_stage, mock_upsert_doc, mock_upsert_chunk, mock_struct_edge,
        mock_entity_edge, mock_driver,
    ):
        """Fallback reads docling_graph_data.nodes from artifact metadata."""
        from app.workers.pipeline import derive_structure_links

        doc_id = str(uuid.uuid4())
        art_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        doc = MagicMock()
        doc.filename = "test.pdf"
        doc.source_id = uuid.uuid4()

        elements = [_make_element("elem-1", art_id)]
        text_chunks = [_make_text_chunk(chunk_id, art_id)]

        artifact = _make_artifact(art_id, doc_id, content_metadata={
            "docling_graph_data": {
                "nodes": [{"name": "Radar", "entity_type": "SENSOR"}],
            }
        })
        _default_settings(mock_settings)

        db = _setup_db_mock(doc, elements, text_chunks, [],
                           graph_extraction=None,
                           artifacts_with_entities=[artifact])
        mock_get_db.return_value = db

        result = derive_structure_links(None, doc_id)

        assert result["status"] == "ok"
        mock_entity_edge.assert_called_once()
        assert mock_entity_edge.call_args.args[1] == "Radar"
        assert mock_entity_edge.call_args.args[2] == "SENSOR"
