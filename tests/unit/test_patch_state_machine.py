"""Unit tests for the patch state machine logic."""

import pytest

from app.schemas.governance import (
    GRAPH_MUTATION_TYPES,
    FeedbackType,
    PatchState,
)


pytestmark = pytest.mark.unit


class TestGraphMutationClassification:
    """Verify that the correct feedback types require dual-curator approval."""

    def test_wrong_text_does_not_require_dual_approval(self):
        assert FeedbackType.wrong_text not in GRAPH_MUTATION_TYPES

    def test_wrong_classification_does_not_require_dual_approval(self):
        assert FeedbackType.wrong_classification not in GRAPH_MUTATION_TYPES

    def test_incorrect_entity_requires_dual_approval(self):
        assert FeedbackType.incorrect_entity in GRAPH_MUTATION_TYPES

    def test_missing_relationship_requires_dual_approval(self):
        assert FeedbackType.missing_relationship in GRAPH_MUTATION_TYPES

    def test_missing_entity_requires_dual_approval(self):
        assert FeedbackType.missing_entity in GRAPH_MUTATION_TYPES

    def test_delete_entity_requires_dual_approval(self):
        assert FeedbackType.delete_entity in GRAPH_MUTATION_TYPES

    def test_merge_entity_requires_dual_approval(self):
        assert FeedbackType.merge_entity in GRAPH_MUTATION_TYPES

    def test_all_graph_mutation_types_accounted_for(self):
        """All FeedbackTypes should be explicitly classified."""
        all_types = set(FeedbackType)
        classified = GRAPH_MUTATION_TYPES | {
            FeedbackType.wrong_text,
            FeedbackType.wrong_classification,
        }
        unclassified = all_types - classified
        assert not unclassified, f"Unclassified feedback types: {unclassified}"


class TestFeedbackToPatchTranslation:
    """Test that feedback types translate to correct patch payloads."""

    def _translate(self, feedback_type: FeedbackType, proposed_value=None):
        from app.api.v1.governance import _feedback_to_patch_payload
        from unittest.mock import MagicMock

        feedback = MagicMock()
        feedback.feedback_type = feedback_type.value
        feedback.proposed_value = proposed_value or {}
        return _feedback_to_patch_payload(feedback)

    def test_wrong_text_targets_chunks_table(self):
        patch_type, payload = self._translate(
            FeedbackType.wrong_text, {"text": "corrected text"}
        )
        assert patch_type == "chunk_text_correction"
        assert payload["target_table"] == "retrieval.chunks"
        ops = payload["operations"]
        assert any(op["op"] == "replace" and op["path"] == "/chunk_text" for op in ops)

    def test_missing_relationship_targets_age_graph(self):
        patch_type, payload = self._translate(FeedbackType.missing_relationship)
        assert patch_type == "relationship_add"
        assert "age_graph" in payload["target_table"]

    def test_wrong_classification_replaces_classification(self):
        patch_type, payload = self._translate(
            FeedbackType.wrong_classification, {"classification": "SECRET"}
        )
        assert patch_type == "classification_correction"
        ops = payload["operations"]
        assert any(op["op"] == "replace" and "/classification" in op["path"] for op in ops)
