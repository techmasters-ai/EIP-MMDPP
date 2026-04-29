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
    """Build a minimal graph-store mock.

    Exposes the high-level GraphStore methods used by the community module,
    plus legacy _client/_database attributes for tests that want to override
    client-level behavior.
    """
    client = MagicMock()
    client.query = AsyncMock(return_value=query_results or [])
    client.command = AsyncMock(return_value=command_results or [{"@rid": "#10:0"}])
    gs = MagicMock()
    gs._client = client
    gs._database = "testdb"
    gs.run_community_algorithm = AsyncMock(return_value=[])
    gs.get_community_reports = AsyncMock(return_value=[])
    gs.upsert_community_report = AsyncMock(return_value="#99:0")
    gs.get_relationships_between_entities = AsyncMock(return_value=[])
    gs.search_community_reports_by_vector = AsyncMock(return_value=[])
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
    from app.services.arcadedb_community import _compute_membership_hash
    existing_hash = _compute_membership_hash(
        [{"entity_type": "RADAR_SYSTEM", "name": "APG-77"},
         {"entity_type": "RADAR_SYSTEM", "name": "APG-63"}]
    )

    gs = _make_graph_store()
    gs.run_community_algorithm = AsyncMock(return_value=detection_rows)
    gs.get_community_reports = AsyncMock(return_value=[
        {"community_id": 1, "membership_hash": existing_hash},
    ])

    with (
        patch("app.services.arcadedb_community._call_llm_for_report",
              new_callable=AsyncMock,
              return_value={"title": "t", "summary": "s"}),
        patch("app.services.arcadedb_community._embed_report",
              new_callable=AsyncMock, return_value=None),
    ):
        result = await run_community_detection(gs, mode="full")

    assert result["status"] == "COMPLETE"
    assert result["total_communities"] == 2
    assert result["reports_generated"] == 2
    assert result["reports_reused"] == 0


@pytest.mark.asyncio
async def test_incremental_mode_skips_unchanged_communities():
    """Incremental mode reuses reports whose membership hash is unchanged."""
    from app.services.arcadedb_community import run_community_detection, _compute_membership_hash

    members_c1 = [{"entity_type": "RADAR_SYSTEM", "name": "APG-77"}]
    existing_hash = _compute_membership_hash(members_c1)

    detection_rows = [
        {"community_id": 1, "name": "APG-77", "entity_type": "RADAR_SYSTEM"},
        {"community_id": 2, "name": "S-400", "entity_type": "SAM_SYSTEM"},
    ]

    gs = _make_graph_store()
    gs.run_community_algorithm = AsyncMock(return_value=detection_rows)
    gs.get_community_reports = AsyncMock(return_value=[
        {"community_id": 1, "membership_hash": existing_hash},
    ])

    with (
        patch("app.services.arcadedb_community._call_llm_for_report",
              new_callable=AsyncMock,
              return_value={"title": "t", "summary": "s"}),
        patch("app.services.arcadedb_community._embed_report",
              new_callable=AsyncMock, return_value=None),
    ):
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
    gs.run_community_algorithm = AsyncMock(return_value=detection_rows)

    with (
        patch("app.services.arcadedb_community._call_llm_for_report",
              new_callable=AsyncMock,
              return_value={"title": "t", "summary": "s"}),
        patch("app.services.arcadedb_community._embed_report",
              new_callable=AsyncMock, return_value=None),
    ):
        result = await run_community_detection(gs, mode="full")

    # Only community 1 with one domain entity should remain
    assert result["total_communities"] == 1
    assert result["reports_generated"] == 1


@pytest.mark.asyncio
async def test_detection_algorithm_failure_returns_failed_status():
    """When the algorithm CALL fails, status should be FAILED."""
    from app.services.arcadedb_community import run_community_detection

    gs = _make_graph_store()
    gs.run_community_algorithm = AsyncMock(
        side_effect=Exception("ArcadeDB algo not found")
    )

    result = await run_community_detection(gs, mode="incremental")

    assert result["status"] == "FAILED"
    assert "ArcadeDB algo not found" in result["error"]


@pytest.mark.asyncio
async def test_embedding_is_attached_when_available():
    """When _embed_report returns a vector, it must be persisted."""
    from app.services.arcadedb_community import run_community_detection

    detection_rows = [
        {"community_id": 1, "name": "APG-77", "entity_type": "RADAR_SYSTEM"},
    ]
    gs = _make_graph_store()
    gs.run_community_algorithm = AsyncMock(return_value=detection_rows)
    gs.upsert_community_report = AsyncMock(return_value="#99:1")

    embedding = [0.1] * 1024
    with (
        patch("app.services.arcadedb_community._call_llm_for_report",
              new_callable=AsyncMock,
              return_value={"title": "Radars", "summary": "body"}),
        patch("app.services.arcadedb_community._embed_report",
              new_callable=AsyncMock, return_value=embedding),
    ):
        result = await run_community_detection(gs, mode="full")

    assert result["reports_generated"] == 1
    gs.set_vertex_embedding.assert_awaited_once_with(
        "CommunityReport", "#99:1", "report_embedding", embedding,
    )


# ---------------------------------------------------------------------------
# _generate_community_report
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_report_passes_relationships_into_prompt():
    """_generate_community_report should query edges and include them in the prompt."""
    from app.services.arcadedb_community import _generate_community_report

    gs = _make_graph_store()
    gs.get_relationships_between_entities = AsyncMock(return_value=[
        {
            "rel_type": "USES",
            "from_name": "APG-77",
            "from_type": "RADAR_SYSTEM",
            "to_name": "AIM-120",
            "to_type": "MISSILE",
        },
    ])

    captured: dict[str, str] = {}

    async def fake_llm(prompt: str, model: str) -> dict[str, str]:
        captured["prompt"] = prompt
        return {"title": "x", "summary": "y"}

    members = [
        {"name": "APG-77", "entity_type": "RADAR_SYSTEM"},
        {"name": "AIM-120", "entity_type": "MISSILE"},
    ]

    with patch("app.services.arcadedb_community._call_llm_for_report", side_effect=fake_llm):
        report = await _generate_community_report(gs, 1, members)

    assert report == {"title": "x", "summary": "y"}
    gs.get_relationships_between_entities.assert_awaited_once()
    assert "APG-77" in captured["prompt"]
    assert "USES" in captured["prompt"]
    assert "AIM-120" in captured["prompt"]


@pytest.mark.asyncio
async def test_generate_report_falls_back_when_llm_raises():
    """When the LLM call raises, a fallback report is returned instead of None."""
    from app.services.arcadedb_community import _generate_community_report

    gs = _make_graph_store()
    members = [{"name": "APG-77", "entity_type": "RADAR_SYSTEM"}]

    with patch(
        "app.services.arcadedb_community._call_llm_for_report",
        new_callable=AsyncMock,
        side_effect=RuntimeError("ollama down"),
    ):
        report = await _generate_community_report(gs, 42, members)

    assert report is not None
    assert report["title"] == "Community 42"
    assert "APG-77" in report["summary"]


# ---------------------------------------------------------------------------
# _call_llm_for_report
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_llm_for_report_parses_json_response():
    """Valid JSON response is parsed into {title, summary}."""
    from app.services.arcadedb_community import _call_llm_for_report

    fake_client = MagicMock()
    fake_client.chat.return_value = (
        '{"title": "Radar Cluster", "summary": "Details here."}'
    )

    with patch("app.services.ollama_clients.get_community_report_client",
               return_value=fake_client):
        result = await _call_llm_for_report("prompt", "gpt-oss:20b")

    assert result == {"title": "Radar Cluster", "summary": "Details here."}


@pytest.mark.asyncio
async def test_call_llm_for_report_line_fallback_when_json_missing():
    """Non-JSON output falls back to first-line-as-title parsing."""
    from app.services.arcadedb_community import _call_llm_for_report

    fake_client = MagicMock()
    fake_client.chat.return_value = "Radar Cluster\nBody paragraph."

    with patch("app.services.ollama_clients.get_community_report_client",
               return_value=fake_client):
        result = await _call_llm_for_report("prompt", "gpt-oss:20b")

    assert result["title"] == "Radar Cluster"
    assert "Body paragraph." in result["summary"]


@pytest.mark.asyncio
async def test_call_llm_for_report_reads_reasoning_content_when_content_empty():
    """gpt-oss with COMMUNITY_REPORT_LLM_THINK=high puts the answer in reasoning_content.

    The pool client's `chat()` already falls back to reasoning_content when
    content is empty (see OllamaChatClient._post_chat_with_retry). So from
    _call_llm_for_report's perspective, the reasoning text simply arrives as
    the chat() return value.
    """
    from app.services.arcadedb_community import _call_llm_for_report

    fake_client = MagicMock()
    fake_client.chat.return_value = (
        '{"title": "From Reasoning", "summary": "body"}'
    )

    with patch("app.services.ollama_clients.get_community_report_client",
               return_value=fake_client):
        result = await _call_llm_for_report("prompt", "gpt-oss:20b")

    assert result == {"title": "From Reasoning", "summary": "body"}


# ---------------------------------------------------------------------------
# _embed_report
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_report_returns_vector_from_embed_texts():
    from app.services.arcadedb_community import _embed_report

    with patch("app.services.embedding.embed_texts", return_value=[[0.1] * 1024]):
        vec = await _embed_report("some report text")

    assert vec is not None
    assert len(vec) == 1024


@pytest.mark.asyncio
async def test_embed_report_returns_none_for_empty_text():
    from app.services.arcadedb_community import _embed_report

    assert await _embed_report("") is None
    assert await _embed_report("   ") is None


@pytest.mark.asyncio
async def test_embed_report_returns_none_on_failure():
    from app.services.arcadedb_community import _embed_report

    with patch("app.services.embedding.embed_texts", side_effect=RuntimeError("fail")):
        vec = await _embed_report("text")

    assert vec is None


# ---------------------------------------------------------------------------
# search_community_reports
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_community_reports_delegates_to_graph_store():
    from app.services.arcadedb_community import search_community_reports

    mock_reports = [
        {"community_id": 1, "summary": "radar cluster", "score": 0.9},
    ]
    gs = _make_graph_store()
    gs.search_community_reports_by_vector = AsyncMock(return_value=mock_reports)

    results = await search_community_reports(gs, query_vector=[0.1, 0.2], top_k=5)

    gs.search_community_reports_by_vector.assert_awaited_once_with([0.1, 0.2], 5)
    assert results == mock_reports
