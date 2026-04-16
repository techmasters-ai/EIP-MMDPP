"""Unit tests for ontology-driven ArcadeDB schema sync."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import yaml
from pathlib import Path

ONTOLOGY_PATH = (
    Path(__file__).parent.parent.parent
    / "tests" / "fixtures" / "ontology" / "air_defense_v3_snapshot.yaml"
)


def _load_ontology():
    with open(ONTOLOGY_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    return client


class TestSchemaSync:
    async def test_creates_vertex_types(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        report = await sync_schema_from_ontology(mock_client, "testdb", ontology)

        # Should create types for all entity types + structural types
        assert report.types_created > 0

        # Check that CREATE VERTEX TYPE was called for RADAR_SYSTEM
        calls = [str(c) for c in mock_client.command.call_args_list]
        create_calls = [c for c in calls if "CREATE VERTEX TYPE" in c]
        assert any("RADAR_SYSTEM" in c for c in create_calls)

    async def test_creates_edge_types(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        report = await sync_schema_from_ontology(mock_client, "testdb", ontology)

        calls = [str(c) for c in mock_client.command.call_args_list]
        edge_calls = [c for c in calls if "CREATE EDGE TYPE" in c]
        assert any("HAS_COMPONENT" in c for c in edge_calls)
        assert any("INSTALLED_ON" in c for c in edge_calls)

    async def test_creates_structural_types(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        await sync_schema_from_ontology(mock_client, "testdb", ontology)

        calls = [str(c) for c in mock_client.command.call_args_list]
        assert any("TextChunk" in c for c in calls)
        assert any("ImageChunk" in c for c in calls)
        assert any("Document" in c for c in calls)
        assert any("CommunityReport" in c for c in calls)

    async def test_creates_vector_indexes(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        report = await sync_schema_from_ontology(mock_client, "testdb", ontology)

        # With DDL batching, vector index CREATEs are all issued in one sqlscript
        # call, so count LSM_VECTOR occurrences across the joined script text.
        joined = "\n".join(str(c) for c in mock_client.command.call_args_list)
        assert joined.count("LSM_VECTOR") >= 4  # text, image, community, trusted

    async def test_reserved_word_table(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        await sync_schema_from_ontology(mock_client, "testdb", ontology)

        calls = [str(c) for c in mock_client.command.call_args_list]
        create_vertex_calls = [c for c in calls if "CREATE VERTEX TYPE" in c]
        # TABLE should be mapped to TABLE_REF
        assert any("TABLE_REF" in c for c in create_vertex_calls)
        # Should NOT create a type called just "TABLE" (without REF suffix)
        bare_table = [c for c in create_vertex_calls if " TABLE " in c and "TABLE_REF" not in c]
        assert bare_table == []

    async def test_idempotent(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        # Run twice — should not error
        r1 = await sync_schema_from_ontology(mock_client, "testdb", ontology)
        r2 = await sync_schema_from_ontology(mock_client, "testdb", ontology)
        assert r1.errors == []
        assert r2.errors == []

    async def test_returns_sync_report(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        report = await sync_schema_from_ontology(mock_client, "testdb", ontology)

        assert isinstance(report.types_created, int)
        assert isinstance(report.properties_added, int)
        assert isinstance(report.indexes_created, int)
        assert isinstance(report.errors, list)

    async def test_properties_added_for_entity_types(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        report = await sync_schema_from_ontology(mock_client, "testdb", ontology)

        # Should have added many properties (common props per type + ontology-specific)
        assert report.properties_added > 0

        calls = [str(c) for c in mock_client.command.call_args_list]
        prop_calls = [c for c in calls if "CREATE PROPERTY" in c]
        # Common entity props include 'name', check it was added
        assert any("name" in c for c in prop_calls)

    async def test_structural_edge_types_created(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        await sync_schema_from_ontology(mock_client, "testdb", ontology)

        calls = [str(c) for c in mock_client.command.call_args_list]
        edge_calls = [c for c in calls if "CREATE EDGE TYPE" in c]
        structural = ["CONTAINS_TEXT", "CONTAINS_IMAGE", "SAME_PAGE", "EXTRACTED_FROM", "HAS_ALIAS"]
        for stype in structural:
            assert any(stype in c for c in edge_calls), f"Missing structural edge type: {stype}"

    async def test_unique_indexes_on_structural_types(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        report = await sync_schema_from_ontology(mock_client, "testdb", ontology)

        calls = [str(c) for c in mock_client.command.call_args_list]
        unique_calls = [c for c in calls if "UNIQUE" in c]
        assert any("TextChunk" in c for c in unique_calls)
        assert any("Document" in c for c in unique_calls)

    async def test_fulltext_indexes_on_entity_types(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        report = await sync_schema_from_ontology(mock_client, "testdb", ontology)

        calls = [str(c) for c in mock_client.command.call_args_list]
        ft_calls = [c for c in calls if "FULL_TEXT" in c]
        assert len(ft_calls) > 0

    async def test_indexes_created_count(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()
        report = await sync_schema_from_ontology(mock_client, "testdb", ontology)

        # At minimum: 4 vector + 4 unique + N fulltext (one per entity type)
        assert report.indexes_created >= 8

    async def test_errors_collected_on_failure(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()

        # Simulate the sqlscript batch failing (connection reset), then in the
        # per-statement fallback, fail one specific statement with a real error
        # so it gets recorded.
        async def failing_command(db, lang, sql):
            if lang == "sqlscript":
                raise RuntimeError("connection reset during batch")
            if "CREATE VERTEX TYPE RADAR_SYSTEM" in sql:
                raise RuntimeError("connection refused")
            return []

        mock_client.command = failing_command
        report = await sync_schema_from_ontology(mock_client, "testdb", ontology)
        # The specific failing statement should surface as an error
        assert any("connection refused" in e for e in report.errors)

    async def test_already_exists_not_an_error(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        ontology = _load_ontology()

        async def already_exists_command(db, lang, sql):
            if "CREATE VERTEX TYPE" in sql or "CREATE EDGE TYPE" in sql:
                raise RuntimeError("Type already exists in the database")
            return []

        mock_client.command = already_exists_command
        report = await sync_schema_from_ontology(mock_client, "testdb", ontology)
        # "already exists" errors should be swallowed
        assert report.errors == []

    async def test_empty_ontology(self, mock_client):
        from app.services.arcadedb_schema import sync_schema_from_ontology
        report = await sync_schema_from_ontology(mock_client, "testdb", {})

        # Should still create structural types even with empty ontology
        assert report.types_created > 0
        calls = [str(c) for c in mock_client.command.call_args_list]
        assert any("TextChunk" in c for c in calls)
