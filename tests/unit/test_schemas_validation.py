"""Unit tests for Pydantic schema validation across all schema modules.

Tests governance, graph_store, sources, memory, text_store, image_store,
and common schemas for valid/invalid inputs, defaults, and bounds.
"""

import uuid

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Governance schemas
# ---------------------------------------------------------------------------

class TestGovernanceSchemas:
    def test_feedback_type_enum_values(self):
        from app.schemas.governance import FeedbackType
        expected = {
            "WRONG_TEXT", "WRONG_CLASSIFICATION", "INCORRECT_ENTITY",
            "MISSING_RELATIONSHIP", "MISSING_ENTITY", "DELETE_ENTITY", "MERGE_ENTITY",
        }
        actual = {ft.value for ft in FeedbackType}
        assert actual == expected

    def test_graph_mutation_types_set(self):
        from app.schemas.governance import GRAPH_MUTATION_TYPES, FeedbackType
        assert FeedbackType.incorrect_entity in GRAPH_MUTATION_TYPES
        assert FeedbackType.missing_relationship in GRAPH_MUTATION_TYPES
        assert FeedbackType.missing_entity in GRAPH_MUTATION_TYPES
        assert FeedbackType.delete_entity in GRAPH_MUTATION_TYPES
        assert FeedbackType.merge_entity in GRAPH_MUTATION_TYPES
        assert FeedbackType.wrong_text not in GRAPH_MUTATION_TYPES
        assert FeedbackType.wrong_classification not in GRAPH_MUTATION_TYPES

    def test_feedback_create_valid(self):
        from app.schemas.governance import FeedbackCreate
        fb = FeedbackCreate(feedback_type="WRONG_TEXT", notes="test")
        assert fb.feedback_type.value == "WRONG_TEXT"
        assert fb.notes == "test"

    def test_patch_state_enum_values(self):
        from app.schemas.governance import PatchState
        expected = {
            "DRAFT", "UNDER_REVIEW", "APPROVED", "DUAL_APPROVED",
            "REJECTED", "APPLIED", "REVERTED",
        }
        actual = {ps.value for ps in PatchState}
        assert actual == expected

    def test_patch_state_has_seven_members(self):
        from app.schemas.governance import PatchState
        assert len(PatchState) == 7

    def test_patch_approval_create_valid(self):
        from app.schemas.governance import PatchApprovalCreate
        pac = PatchApprovalCreate(notes="LGTM")
        assert pac.notes == "LGTM"

    def test_patch_approval_create_notes_optional(self):
        from app.schemas.governance import PatchApprovalCreate
        pac = PatchApprovalCreate()
        assert pac.notes is None


# ---------------------------------------------------------------------------
# Graph store schemas
# ---------------------------------------------------------------------------

class TestGraphStoreSchemas:
    def test_graph_entity_ingest_valid(self):
        from app.schemas.graph_store import GraphEntityIngest
        body = GraphEntityIngest(entity_type="RadarSystem", name="AN/SPY-1")
        assert body.entity_type == "RadarSystem"
        assert body.name == "AN/SPY-1"

    def test_graph_entity_ingest_empty_name_rejected(self):
        from app.schemas.graph_store import GraphEntityIngest
        with pytest.raises(ValidationError):
            GraphEntityIngest(entity_type="RadarSystem", name="")

    def test_graph_relationship_ingest_valid(self):
        from app.schemas.graph_store import GraphRelationshipIngest
        body = GraphRelationshipIngest(
            from_entity="AN/SPY-1", from_type="RadarSystem",
            to_entity="Aegis", to_type="MissileSystem",
            relationship_type="associated_with",
        )
        assert body.relationship_type == "associated_with"

    def test_graph_relationship_ingest_empty_fields_rejected(self):
        from app.schemas.graph_store import GraphRelationshipIngest
        with pytest.raises(ValidationError):
            GraphRelationshipIngest(
                from_entity="", from_type="A",
                to_entity="B", to_type="C",
                relationship_type="r",
            )

    def test_graph_query_request_hop_count_bounds(self):
        from app.schemas.graph_store import GraphQueryRequest
        # Valid bounds
        r = GraphQueryRequest(query="test", hop_count=1)
        assert r.hop_count == 1
        r = GraphQueryRequest(query="test", hop_count=4)
        assert r.hop_count == 4
        # Out of bounds
        with pytest.raises(ValidationError):
            GraphQueryRequest(query="test", hop_count=0)
        with pytest.raises(ValidationError):
            GraphQueryRequest(query="test", hop_count=5)

    def test_graph_query_request_top_k_bounds(self):
        from app.schemas.graph_store import GraphQueryRequest
        with pytest.raises(ValidationError):
            GraphQueryRequest(query="test", top_k=0)
        with pytest.raises(ValidationError):
            GraphQueryRequest(query="test", top_k=101)

    def test_graph_query_request_defaults(self):
        from app.schemas.graph_store import GraphQueryRequest
        r = GraphQueryRequest(query="test")
        assert r.hop_count == 2
        assert r.top_k == 20


# ---------------------------------------------------------------------------
# Sources schemas
# ---------------------------------------------------------------------------

class TestSourcesSchemas:
    def test_source_create_valid(self):
        from app.schemas.sources import SourceCreate
        sc = SourceCreate(name="My Source")
        assert sc.name == "My Source"

    def test_source_create_empty_name_rejected(self):
        from app.schemas.sources import SourceCreate
        with pytest.raises(ValidationError):
            SourceCreate(name="")

    def test_watch_dir_create_defaults(self):
        from app.schemas.sources import WatchDirCreate
        wdc = WatchDirCreate(path="/data/watch", source_id=uuid.uuid4())
        assert wdc.poll_interval_seconds == 30
        assert wdc.file_patterns == ["*.pdf", "*.docx", "*.txt", "*.png", "*.jpg", "*.tiff"]


# ---------------------------------------------------------------------------
# Memory schemas
# ---------------------------------------------------------------------------

class TestMemorySchemas:
    def test_memory_proposal_create_valid(self):
        from app.schemas.memory import MemoryProposalCreate
        mpc = MemoryProposalCreate(
            content="The SA-11 uses a phased array radar.",
            source="document:abc123",
        )
        assert "SA-11" in mpc.content

    def test_memory_query_request_valid(self):
        from app.schemas.memory import MemoryQueryRequest
        mqr = MemoryQueryRequest(query="radar systems")
        assert mqr.query == "radar systems"

    def test_memory_query_request_top_k_default(self):
        from app.schemas.memory import MemoryQueryRequest
        mqr = MemoryQueryRequest(query="test")
        assert mqr.top_k == 10


# ---------------------------------------------------------------------------
# Text store schemas
# ---------------------------------------------------------------------------

class TestTextStoreSchemas:
    def test_text_chunk_ingest_valid(self):
        from app.schemas.text_store import TextChunkIngest
        tci = TextChunkIngest(text="Hello world.")
        assert tci.text == "Hello world."
        assert tci.modality == "text"
        assert tci.classification == "UNCLASSIFIED"

    def test_text_chunk_ingest_empty_text_rejected(self):
        from app.schemas.text_store import TextChunkIngest
        with pytest.raises(ValidationError):
            TextChunkIngest(text="")

    def test_text_query_request_valid(self):
        from app.schemas.text_store import TextQueryRequest
        tqr = TextQueryRequest(query="radar frequency")
        assert tqr.query == "radar frequency"


# ---------------------------------------------------------------------------
# Image store schemas
# ---------------------------------------------------------------------------

class TestImageStoreSchemas:
    def test_image_chunk_ingest_valid(self):
        from app.schemas.image_store import ImageChunkIngest
        ici = ImageChunkIngest(
            image="aGVsbG8=",
            content_type="image/png",
        )
        assert ici.content_type == "image/png"

    def test_image_query_request_valid(self):
        from app.schemas.image_store import ImageQueryRequest
        iqr = ImageQueryRequest(query_text="schematic diagram")
        assert iqr.query_text == "schematic diagram"

    def test_image_query_request_text_and_image_optional(self):
        from app.schemas.image_store import ImageQueryRequest
        iqr = ImageQueryRequest(query_image="base64data")
        assert iqr.query_image == "base64data"
        assert iqr.query_text is None


# ---------------------------------------------------------------------------
# Common schemas
# ---------------------------------------------------------------------------

class TestCommonSchemas:
    def test_api_model_from_attributes(self):
        from app.schemas.common import APIModel
        assert APIModel.model_config.get("from_attributes") is True

    def test_cursor_page_defaults(self):
        from app.schemas.common import CursorPage
        page = CursorPage(items=[])
        assert page.next_cursor is None
        assert page.total_hint is None
