"""Native ArcadeDB integration tests.

These tests exercise the REAL ArcadeDBGraphStore against a running ArcadeDB
instance. They validate SQL queries, MATCH traversals, vector searches, and
edge semantics actually work — NOT via mocks.

**Requires:** Running ArcadeDB with eip_knowledge_graph database and schema synced.

    pytest tests/native/ -m native --timeout=30

Skipped automatically if ArcadeDB is not reachable.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = [pytest.mark.native, pytest.mark.slow]

_ARCADEDB_URL = os.environ.get("ARCADEDB_URL", "http://localhost:2480")
_ARCADEDB_USER = os.environ.get("ARCADEDB_USER", "root")
_ARCADEDB_PASSWORD = os.environ.get("ARCADEDB_PASSWORD", "eip_arcadedb_secret")
_ARCADEDB_DATABASE = os.environ.get("ARCADEDB_DATABASE", "eip_knowledge_graph")
_TEST_ID = uuid.uuid4().hex[:8]


def _check_arcadedb() -> bool:
    try:
        import httpx
        return httpx.get(f"{_ARCADEDB_URL}/api/v1/ready", timeout=3).status_code == 204
    except Exception:
        return False


_AVAILABLE = _check_arcadedb()
skip_no_arcadedb = pytest.mark.skipif(not _AVAILABLE, reason=f"ArcadeDB not available at {_ARCADEDB_URL}")


def _make_store():
    from app.services.arcadedb_client import ArcadeDBClient
    from app.services.arcadedb_graph import ArcadeDBGraphStore
    client = ArcadeDBClient(_ARCADEDB_URL, _ARCADEDB_USER, _ARCADEDB_PASSWORD)
    return ArcadeDBGraphStore(client=client, database=_ARCADEDB_DATABASE)


# ---------------------------------------------------------------------------
# Schema and readiness
# ---------------------------------------------------------------------------

@skip_no_arcadedb
class TestSchemaAndReadiness:
    @pytest.mark.asyncio
    async def test_ensure_ready(self):
        store = _make_store()
        await store.ensure_ready()

    @pytest.mark.asyncio
    async def test_schema_query(self):
        store = _make_store()
        rows = await store._client.query(_ARCADEDB_DATABASE, "sql", "SELECT name FROM schema:database")
        assert rows is not None


# ---------------------------------------------------------------------------
# Entity CRUD
# ---------------------------------------------------------------------------

@skip_no_arcadedb
class TestEntityCRUD:
    @pytest.mark.asyncio
    async def test_upsert_and_resolve_entity(self):
        from app.services.graph_store import NodeRecord
        store = _make_store()
        name = f"TestRadar-{_TEST_ID}"

        record = NodeRecord(
            entity_type="RADAR_SYSTEM",
            identity_fields={"name": name, "entity_type": "RADAR_SYSTEM"},
            name=name,
            properties={"band": "X"},
            extraction_confidence=0.95,
        )
        rid = await store.upsert_node(record)
        assert rid is not None

        entity = await store.resolve_root_entity(name, "RADAR_SYSTEM")
        assert entity is not None
        assert entity.name == name
        assert entity.entity_type == "RADAR_SYSTEM"

    @pytest.mark.asyncio
    async def test_fulltext_search(self):
        store = _make_store()
        name = f"TestRadar-{_TEST_ID}"
        results = await store.fulltext_search(name, limit=5)
        names = [r.name for r in results]
        assert name in names

    @pytest.mark.asyncio
    async def test_fulltext_search_carries_score(self):
        store = _make_store()
        name = f"TestRadar-{_TEST_ID}"
        results = await store.fulltext_search(name, limit=5)
        assert results
        assert results[0].score_type == "fulltext"
        assert results[0].score is not None


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

@skip_no_arcadedb
class TestAliasResolution:
    @pytest.mark.asyncio
    async def test_create_and_search_alias(self):
        store = _make_store()
        name = f"TestRadar-{_TEST_ID}"
        alias = f"TestAlias-{_TEST_ID}"

        entity = await store.resolve_root_entity(name, "RADAR_SYSTEM")
        assert entity is not None

        await store.create_alias(entity.node_id, alias)
        results = await store.search_by_alias(alias)
        names = [r.name for r in results]
        assert name in names


# ---------------------------------------------------------------------------
# Structural edges
# ---------------------------------------------------------------------------

@skip_no_arcadedb
class TestStructuralEdges:
    @pytest.mark.asyncio
    async def test_create_and_traverse_neighborhood(self):
        from app.services.graph_store import NodeRecord, RelationshipRecord
        store = _make_store()

        radar_name = f"Radar2-{_TEST_ID}"
        platform_name = f"Platform2-{_TEST_ID}"

        rid1 = await store.upsert_node(NodeRecord(
            entity_type="RADAR_SYSTEM",
            identity_fields={"name": radar_name, "entity_type": "RADAR_SYSTEM"},
            name=radar_name,
        ))
        rid2 = await store.upsert_node(NodeRecord(
            entity_type="PLATFORM",
            identity_fields={"name": platform_name, "entity_type": "PLATFORM"},
            name=platform_name,
        ))

        edge_rid = await store.upsert_relationship(RelationshipRecord(
            from_type="RADAR_SYSTEM",
            from_identity={"name": radar_name, "entity_type": "RADAR_SYSTEM"},
            to_type="PLATFORM",
            to_identity={"name": platform_name, "entity_type": "PLATFORM"},
            rel_type="INSTALLED_ON",
        ))

        entity = await store.resolve_root_entity(radar_name, "RADAR_SYSTEM")
        neighbors = await store.get_neighborhood(entity.node_id, depth=1)
        neighbor_names = [n.name for n in neighbors]
        assert platform_name in neighbor_names

    @pytest.mark.asyncio
    async def test_neighborhood_graph_returns_edges(self):
        store = _make_store()
        radar_name = f"Radar2-{_TEST_ID}"

        entity = await store.resolve_root_entity(radar_name, "RADAR_SYSTEM")
        if not entity:
            pytest.skip("Entity not found")

        graph = await store.get_neighborhood_graph(entity.node_id, depth=1)
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0


# ---------------------------------------------------------------------------
# Community reports
# ---------------------------------------------------------------------------

@skip_no_arcadedb
class TestCommunityReports:
    @pytest.mark.asyncio
    async def test_upsert_and_list_community_report(self):
        store = _make_store()
        rid = await store.upsert_community_report(
            community_id=99999,
            title=f"Test Community {_TEST_ID}",
            summary="Test summary for native integration test",
            member_count=3,
            membership_hash=f"testhash-{_TEST_ID}",
            model_name="test",
            source_documents=[{"document_id": "test-doc", "page_number": 1}],
        )
        assert rid

        reports = await store.list_community_reports(limit=200)
        titles = [r.get("title", "") for r in reports]
        assert f"Test Community {_TEST_ID}" in titles


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

@skip_no_arcadedb
class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_test_entities(self):
        store = _make_store()
        for sql in [
            "DELETE VERTEX FROM CommunityReport WHERE community_id = 99999",
            f"DELETE VERTEX FROM RADAR_SYSTEM WHERE name LIKE '%{_TEST_ID}%'",
            f"DELETE VERTEX FROM PLATFORM WHERE name LIKE '%{_TEST_ID}%'",
            f"DELETE VERTEX FROM Alias WHERE alias_name LIKE '%{_TEST_ID}%'",
        ]:
            try:
                await store._client.command(_ARCADEDB_DATABASE, "sql", sql)
            except Exception:
                pass
