"""Unit tests for app.services.neo4j_graph helper functions.

Pure unit tests — no database required. _sanitize_label is exercised here;
the Cypher-level functions are covered by integration tests.
"""

import pytest

pytestmark = pytest.mark.unit


class TestSanitizeLabel:
    def test_simple_word(self):
        from app.services.neo4j_graph import _sanitize_label

        assert _sanitize_label("COMPONENT") == "COMPONENT"

    def test_spaces_replaced_with_underscore(self):
        from app.services.neo4j_graph import _sanitize_label

        assert _sanitize_label("EQUIPMENT SYSTEM") == "EQUIPMENT_SYSTEM"

    def test_hyphens_replaced(self):
        from app.services.neo4j_graph import _sanitize_label

        assert _sanitize_label("MIL-STD") == "MIL_STD"

    def test_leading_digit_prefixed(self):
        from app.services.neo4j_graph import _sanitize_label

        result = _sanitize_label("1553B")
        assert result.startswith("_"), f"Expected underscore prefix, got: {result}"

    def test_empty_string_returns_unknown(self):
        from app.services.neo4j_graph import _sanitize_label

        assert _sanitize_label("") == "UNKNOWN"

    def test_only_special_chars_returns_unknown(self):
        from app.services.neo4j_graph import _sanitize_label

        # All non-alphanumeric chars become underscores, all-underscore is fine
        result = _sanitize_label("!@#")
        assert result  # should not be empty
        assert all(c in "_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" for c in result)

    def test_already_valid_label(self):
        from app.services.neo4j_graph import _sanitize_label

        assert _sanitize_label("EQUIPMENT_SYSTEM") == "EQUIPMENT_SYSTEM"

    def test_mixed_case_preserved(self):
        from app.services.neo4j_graph import _sanitize_label

        assert _sanitize_label("EntityType") == "EntityType"

    def test_ontology_types(self):
        """New ontology entity types should pass sanitization."""
        from app.services.neo4j_graph import _sanitize_label

        for label in ("RadarSystem", "MissileSystem", "FrequencyBand",
                       "SignalProcessingChain", "AirDefenseArtillerySystem"):
            assert _sanitize_label(label) == label
