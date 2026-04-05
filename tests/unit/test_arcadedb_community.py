"""Unit tests for community detection service.

All tests mock the ArcadeDB client and LLM calls — no real server required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_store(query_results=None, command_results=None):
    """Build a minimal graph-store mock."""
    client = MagicMock()
    client.query = AsyncMock(return_value=query_results or [])
    client.command = AsyncMock(return_value=command_results or [{"@rid": "#10:0"}])
    gs = MagicMock()
    gs._client = client
    gs._database = "testdb"
    gs.vector_search = AsyncMock(return_value=[])
    gs.set_vertex_embedding = AsyncMock(return_value=None)
    return gs


# ---------------------------------------------------------------------------
# _compute_membership_hash
# ---------------------------------------------------------------------------

def test_membership_hash_is_deterministic():
    from app.services.arcadedb_community import _compute_membership_hash

    members = [
        {"entity_type": "RADAR_SYSTEM", "name": "APG-77"},
        {"entity_type": "MISSILE", "name": "AIM-120"},
    ]
    h1 = _compute_membership_hash(members)
    # Same members in different order → same hash (sorted internally)
    h2 = _compute_membership_hash(list(reversed(members)))
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_membership_hash_changes_with_members():
    from app.services.arcadedb_community import _compute_membership_hash

    h1 = _compute_membership_hash([{"entity_type": "RADAR_SYSTEM", "name": "APG-77"}])
    h2 = _compute_membership_hash([{"entity_type": "RADAR_SYSTEM", "name": "APG-63"}])
    assert h1 != h2


# ---------------------------------------------------------------------------
# run_community_detection — full mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_mode_generates_reports_for_all_communities():
    """Full mode should regenerate all communities regardless of hash."""
    from app.services.arcadedb_community import run_community_detection

    detection_rows = [
        {"community_id": 1, "name": "APG-77", "entity_type": "RADAR_SYSTEM"},
        {"community_id": 1, "name": "APG-63", "entity_type": "RADAR_SYSTEM"},
        {"community_id": 2, "name": "S-400", "entity_type": "SAM_SYSTEM"},
    ]
    # Existing report has matching hash for community 1 — full mode must ignore it
    existing_hash = __import__(
        "app.services.arcadedb_community", fromlist=["_compute_membership_hash"]
    )._compute_membership_hash(
        [{"entity_type": "RADAR_SYSTEM", "name": "APG-77"},
         {"entity_type": "RADAR_SYSTEM", "name": "APG-63"}]
    )
    existing_rows = [{"community_id": 1, "membership_hash": existing_hash}]

    gs = _make_graph_store()
    gs._client.query = AsyncMock(side_effect=[detection_rows, existing_rows, []])

    result = await run_community_detection(gs, mode="full")

    assert result["status"] == "COMPLETE"
    assert result["total_communities"] == 2
    assert result["reports_generated"] == 2
    assert result["reports_reused"] == 0


@pytest.mark.asyncio
async def test_incremental_mode_skips_unchanged_communities():
    """Incremental mode reuses reports whose membership hash is unchanged."""
    from app.services.arcadedb_community import run_community_detection, _compute_membership_hash

    members_c1 = [
        {"entity_type": "RADAR_SYSTEM", "name": "APG-77"},
    ]
    existing_hash = _compute_membership_hash(members_c1)

    detection_rows = [
        {"community_id": 1, "name": "APG-77", "entity_type": "RADAR_SYSTEM"},
        {"community_id": 2, "name": "S-400", "entity_type": "SAM_SYSTEM"},
    ]
    existing_rows = [{"community_id": 1, "membership_hash": existing_hash}]

    gs = _make_graph_store()
    gs._client.query = AsyncMock(side_effect=[detection_rows, existing_rows, []])

    result = await run_community_detection(gs, mode="incremental")

    assert result["status"] == "COMPLETE"
    assert result["total_communities"] == 2
    assert result["reports_generated"] == 1   # only community 2
    assert result["reports_reused"] == 1       # community 1 unchanged


@pytest.mark.asyncio
async def test_structural_types_are_excluded():
    """Structural vertex types (Document, TextChunk, etc.) must be filtered out."""
    from app.services.arcadedb_community import run_community_detection

    detection_rows = [
        {"community_id": 1, "name": "doc.pdf", "entity_type": "Document"},
        {"community_id": 1, "name": "chunk-1", "entity_type": "TextChunk"},
        {"community_id": 1, "name": "APG-77", "entity_type": "RADAR_SYSTEM"},
    ]

    gs = _make_graph_store()
    gs._client.query = AsyncMock(side_effect=[detection_rows, [], []])

    result = await run_community_detection(gs, mode="full")

    # Only community 1 with one domain entity should remain
    assert result["total_communities"] == 1
    assert result["reports_generated"] == 1


@pytest.mark.asyncio
async def test_detection_algorithm_failure_returns_failed_status():
    """When the algorithm CALL fails, status should be FAILED."""
    from app.services.arcadedb_community import run_community_detection

    gs = _make_graph_store()
    gs._client.query = AsyncMock(side_effect=Exception("ArcadeDB algo not found"))

    result = await run_community_detection(gs, mode="incremental")

    assert result["status"] == "FAILED"
    assert "ArcadeDB algo not found" in result["error"]


# ---------------------------------------------------------------------------
# search_community_reports
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_community_reports_delegates_to_vector_search():
    from app.services.arcadedb_community import search_community_reports

    mock_reports = [
        {"community_id": 1, "summary": "radar cluster", "score": 0.9},
    ]
    gs = _make_graph_store()
    gs.vector_search = AsyncMock(return_value=mock_reports)

    results = await search_community_reports(gs, query_vector=[0.1, 0.2], top_k=5)

    gs.vector_search.assert_called_once_with(
        "CommunityReport", "report_embedding", [0.1, 0.2], 5
    )
    assert results == mock_reports
