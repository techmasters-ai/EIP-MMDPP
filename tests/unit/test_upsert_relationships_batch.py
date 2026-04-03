"""Unit tests for upsert_relationships_batch (name, entity_type) matching."""
from unittest.mock import MagicMock
import pytest

pytestmark = pytest.mark.unit


class TestUpsertRelationshipsBatch:
    def test_cypher_matches_by_name_and_entity_type(self, mock_neo4j_driver):
        """MATCH clause should use Entity {name: ..., entity_type: ...}."""
        from app.services.neo4j_graph import upsert_relationships_batch

        driver, session = mock_neo4j_driver
        session.run.return_value.single.return_value = {"cnt": 1}

        edges = [{
            "from_name": "SA-2 Guideline",
            "from_type": "EQUIPMENT_SYSTEM",
            "to_name": "Max Range",
            "to_type": "SPECIFICATION",
            "rel_type": "SPECIFIED_BY",
            "artifact_id": "art-1",
            "confidence": 0.9,
            "props": {"artifact_id": "art-1", "confidence": 0.9},
        }]
        upsert_relationships_batch(driver, edges)

        query = session.run.call_args.args[0]
        # Entity type labels should NOT appear in the MATCH clause
        assert ":EQUIPMENT_SYSTEM" not in query
        assert ":SPECIFICATION" not in query
        # Matching uses parameterized (name, entity_type) pair
        assert "edge.from_name" in query
        assert "edge.from_type" in query
        assert "edge.to_name" in query
        assert "edge.to_type" in query

    def test_groups_by_rel_type_only(self, mock_neo4j_driver):
        """Edges with same rel_type but different entity types go in one batch."""
        from app.services.neo4j_graph import upsert_relationships_batch

        driver, session = mock_neo4j_driver
        session.run.return_value.single.return_value = {"cnt": 2}

        edges = [
            {"from_name": "A", "from_type": "RADAR_SYSTEM", "to_name": "X",
             "to_type": "SPECIFICATION", "rel_type": "SPECIFIED_BY",
             "artifact_id": "a1", "confidence": 0.9, "props": {}},
            {"from_name": "B", "from_type": "MISSILE_SYSTEM", "to_name": "Y",
             "to_type": "SPECIFICATION", "rel_type": "SPECIFIED_BY",
             "artifact_id": "a1", "confidence": 0.9, "props": {}},
        ]
        result = upsert_relationships_batch(driver, edges)
        assert session.run.call_count == 1

    def test_returns_count(self, mock_neo4j_driver):
        from app.services.neo4j_graph import upsert_relationships_batch
        driver, session = mock_neo4j_driver
        session.run.return_value.single.return_value = {"cnt": 3}
        result = upsert_relationships_batch(driver, [
            {"from_name": "A", "from_type": "T", "to_name": "B", "to_type": "T",
             "rel_type": "REL", "artifact_id": "a", "confidence": 0.5, "props": {}},
        ])
        assert result == 3

    def test_exception_returns_zero(self, mock_neo4j_driver):
        from app.services.neo4j_graph import upsert_relationships_batch
        driver, session = mock_neo4j_driver
        session.run.side_effect = Exception("connection lost")
        result = upsert_relationships_batch(driver, [
            {"from_name": "A", "from_type": "T", "to_name": "B", "to_type": "T",
             "rel_type": "REL", "artifact_id": "a", "confidence": 0.5, "props": {}},
        ])
        assert result == 0
