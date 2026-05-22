"""Unit tests for the VR C.4 janitor task: purge_terminated_extraction_chunks.

Tests the cross-store purge algorithm (rev 9 H3 + rev 10 H3):
  1. ArcadeDB: find old run_ids
  2. Postgres: batch-fetch statuses
  3. Compute purge set = terminal ∪ orphan
  4. ArcadeDB: bulk DELETE

All tests mock external services (ArcadeDB, Postgres) — no live services needed.

Run with: pytest tests/unit/test_janitor.py -v
"""
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helper: build mock ArcadeDB store
# ---------------------------------------------------------------------------


def _make_store(old_run_ids: list[str]) -> MagicMock:
    """Return a mock ArcadeDB store whose query_sync returns the given run_ids."""
    store = MagicMock()
    store._database = "eip"
    store._client = MagicMock()
    store._client.query_sync.return_value = [
        {"pipeline_run_id": rid} for rid in old_run_ids
    ]
    return store


def _make_db_rows(statuses: dict[str, str]):
    """Build mock Postgres fetchall rows: list of (id, status) tuples."""
    return [(rid, status) for rid, status in statuses.items()]


# ---------------------------------------------------------------------------
# Test 1: Purges terminal runs
# ---------------------------------------------------------------------------


def test_janitor_purges_terminal_runs():
    """3 old run IDs in ArcadeDB; 2 are COMPLETE in Postgres → those 2 are purged.

    IMPORTANT #3: also asserts cleanup_extraction_index was called with the CORRECT
    run_ids, not just the right total count.
    """
    old_ids = [
        "aaaaaaaa-0000-0000-0000-000000000001",
        "aaaaaaaa-0000-0000-0000-000000000002",
        "aaaaaaaa-0000-0000-0000-000000000003",
    ]
    pg_statuses = {
        "aaaaaaaa-0000-0000-0000-000000000001": "COMPLETE",
        "aaaaaaaa-0000-0000-0000-000000000002": "PROCESSING",
        "aaaaaaaa-0000-0000-0000-000000000003": "FAILED",
    }

    store = _make_store(old_ids)

    with patch("app.workers.pipeline.get_graph_store", return_value=store), \
         patch("app.workers.pipeline._get_db") as mock_get_db, \
         patch("app.services.extraction_chunk_index.cleanup_extraction_index") as mock_cleanup:

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        # Simulate Postgres fetchall returning status rows
        mock_db.execute.return_value.fetchall.return_value = _make_db_rows(pg_statuses)

        mock_cleanup.return_value = 5  # deleted 5 chunks per call

        from app.workers.pipeline import purge_terminated_extraction_chunks
        result = purge_terminated_extraction_chunks()

    # COMPLETE and FAILED are terminal → both get purged
    # PROCESSING is active → NOT purged
    assert result["purge_set_size"] == 2, (
        "COMPLETE + FAILED → 2 purged; PROCESSING preserved"
    )
    assert result["purged_chunks"] == 10, "5 chunks per call × 2 calls = 10"

    # IMPORTANT #3: verify the CORRECT run_ids were passed (not just the right count)
    called_ids = [c.args[0] for c in mock_cleanup.call_args_list]
    assert "aaaaaaaa-0000-0000-0000-000000000001" in called_ids, (
        "COMPLETE run must be in cleanup_extraction_index call args"
    )
    assert "aaaaaaaa-0000-0000-0000-000000000003" in called_ids, (
        "FAILED run must be in cleanup_extraction_index call args"
    )
    assert "aaaaaaaa-0000-0000-0000-000000000002" not in called_ids, (
        "PROCESSING run must NOT be in cleanup_extraction_index call args"
    )


def test_janitor_purges_partial_complete_runs():
    """PARTIAL_COMPLETE is terminal — must be purged.

    IMPORTANT #3: also asserts the correct run_id was passed to cleanup.
    """
    expected_id = "aaaaaaaa-0000-0000-0000-000000000001"
    old_ids = [expected_id]
    pg_statuses = {expected_id: "PARTIAL_COMPLETE"}

    store = _make_store(old_ids)

    with patch("app.workers.pipeline.get_graph_store", return_value=store), \
         patch("app.workers.pipeline._get_db") as mock_get_db, \
         patch("app.services.extraction_chunk_index.cleanup_extraction_index") as mock_cleanup:

        mock_cleanup.return_value = 3
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.execute.return_value.fetchall.return_value = _make_db_rows(pg_statuses)

        from app.workers.pipeline import purge_terminated_extraction_chunks
        result = purge_terminated_extraction_chunks()

    assert result["purge_set_size"] == 1
    # IMPORTANT #3: assert the correct run_id was passed
    mock_cleanup.assert_any_call(expected_id, store=store)


# ---------------------------------------------------------------------------
# Test 2: Purges orphan runs (no Postgres row)
# ---------------------------------------------------------------------------


def test_janitor_purges_orphans():
    """Old run_id in ArcadeDB but NOT in Postgres → orphan → included in purge_set."""
    old_ids = [
        "aaaaaaaa-0000-0000-0000-000000000001",  # orphan
        "aaaaaaaa-0000-0000-0000-000000000002",  # PROCESSING
    ]
    # Postgres only knows about run 2
    pg_statuses = {"aaaaaaaa-0000-0000-0000-000000000002": "PROCESSING"}

    store = _make_store(old_ids)

    with patch("app.workers.pipeline.get_graph_store", return_value=store), \
         patch("app.workers.pipeline._get_db") as mock_get_db, \
         patch("app.services.extraction_chunk_index.cleanup_extraction_index",
               return_value=2):

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.execute.return_value.fetchall.return_value = _make_db_rows(pg_statuses)

        from app.workers.pipeline import purge_terminated_extraction_chunks
        result = purge_terminated_extraction_chunks()

    assert result["purge_set_size"] == 1, "Orphan run must be purged"


# ---------------------------------------------------------------------------
# Test 3: Preserves PROCESSING runs
# ---------------------------------------------------------------------------


def test_janitor_preserves_processing_runs():
    """Old run_id in ArcadeDB; status=PROCESSING in Postgres → NOT purged."""
    old_ids = ["aaaaaaaa-0000-0000-0000-000000000001"]
    pg_statuses = {"aaaaaaaa-0000-0000-0000-000000000001": "PROCESSING"}

    store = _make_store(old_ids)

    with patch("app.workers.pipeline.get_graph_store", return_value=store), \
         patch("app.workers.pipeline._get_db") as mock_get_db, \
         patch("app.services.extraction_chunk_index.cleanup_extraction_index") as mock_cleanup:

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.execute.return_value.fetchall.return_value = _make_db_rows(pg_statuses)

        from app.workers.pipeline import purge_terminated_extraction_chunks
        result = purge_terminated_extraction_chunks()

    assert result["purge_set_size"] == 0, "PROCESSING run must NOT be purged"
    mock_cleanup.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Empty old set
# ---------------------------------------------------------------------------


def test_janitor_empty_old_set():
    """No chunks older than 24h → 0 deletes, no Postgres lookup."""
    store = _make_store([])  # ArcadeDB returns no old run_ids

    with patch("app.workers.pipeline.get_graph_store", return_value=store), \
         patch("app.workers.pipeline._get_db") as mock_get_db, \
         patch("app.services.extraction_chunk_index.cleanup_extraction_index") as mock_cleanup:

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        from app.workers.pipeline import purge_terminated_extraction_chunks
        result = purge_terminated_extraction_chunks()

    assert result["old_run_ids_found"] == 0
    assert result["purge_set_size"] == 0
    assert result["purged_chunks"] == 0
    # DB session is obtained but not queried — when no old_run_ids, we return early
    mock_cleanup.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: ArcadeDB query failure
# ---------------------------------------------------------------------------


def test_janitor_arcadedb_query_failure_returns_error():
    """ArcadeDB query failure → function returns with error field set."""
    store = MagicMock()
    store._database = "eip"
    store._client = MagicMock()
    store._client.query_sync.side_effect = Exception("ArcadeDB unavailable")

    with patch("app.workers.pipeline.get_graph_store", return_value=store), \
         patch("app.workers.pipeline._get_db") as mock_get_db:

        from app.workers.pipeline import purge_terminated_extraction_chunks
        result = purge_terminated_extraction_chunks()

    assert result["error"] is not None, "ArcadeDB failure must be captured in error"
    assert result["purged_chunks"] == 0


# ---------------------------------------------------------------------------
# Test 6: Janitor is registered in Celery beat schedule
# ---------------------------------------------------------------------------


def test_janitor_registered_in_celery_beat():
    """purge_terminated_extraction_chunks must be in the Celery beat schedule."""
    from app.workers.celery_app import celery_app

    beat_schedule = celery_app.conf.beat_schedule or {}
    schedule_tasks = [v.get("task") for v in beat_schedule.values()]
    assert "vr.purge_terminated_extraction_chunks" in schedule_tasks, (
        "Janitor must be registered in celery_app.conf.beat_schedule"
    )


def test_janitor_beat_schedule_uses_graph_queue():
    """Janitor beat entry must route to the graph queue."""
    from app.workers.celery_app import celery_app

    beat_schedule = celery_app.conf.beat_schedule or {}
    for entry in beat_schedule.values():
        if entry.get("task") == "vr.purge_terminated_extraction_chunks":
            opts = entry.get("options") or {}
            assert opts.get("queue") == "graph", "Janitor must run on graph queue"
            return
    pytest.fail("Janitor beat entry not found")


# ---------------------------------------------------------------------------
# IMPORTANT #4: UUID case normalization
# ---------------------------------------------------------------------------


def test_janitor_uppercase_uuid_from_arcadedb_is_normalized():
    """ArcadeDB may return uppercase UUIDs in theory.

    IMPORTANT #4: The janitor must normalize ArcadeDB-returned and Postgres-returned
    run_ids before the lookup so the status lookup doesn't fail silently.

    Without normalization: uppercase ArcadeDB ID fails dict lookup against lowercase
    Postgres key → treated as 'orphan' and purged (happens to be correct but
    for the wrong reason — a PROCESSING run with uppercase ID would also be purged).

    With normalization: both sides normalized before comparison — PROCESSING run
    is correctly preserved even if ArcadeDB returned uppercase.

    This test verifies the PROCESSING case is preserved with uppercase ArcadeDB IDs:
    without normalization, a PROCESSING run with an uppercase ArcadeDB ID would
    be incorrectly classified as orphan and purged.
    """
    uppercase_id = "AAAAAAAA-0000-0000-0000-000000000001"
    lowercase_id = "aaaaaaaa-0000-0000-0000-000000000001"

    # ArcadeDB returns uppercase; Postgres returns lowercase with PROCESSING status
    store = _make_store([uppercase_id])
    pg_statuses = {lowercase_id: "PROCESSING"}

    with patch("app.workers.pipeline.get_graph_store", return_value=store), \
         patch("app.workers.pipeline._get_db") as mock_get_db, \
         patch("app.services.extraction_chunk_index.cleanup_extraction_index") as mock_cleanup:

        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_db.execute.return_value.fetchall.return_value = _make_db_rows(pg_statuses)

        from app.workers.pipeline import purge_terminated_extraction_chunks
        result = purge_terminated_extraction_chunks()

    # With normalization: PROCESSING run is preserved (purge_set_size=0)
    # Without normalization: uppercase ID doesn't match Postgres lowercase key →
    # treated as orphan → incorrectly purged (purge_set_size=1).
    assert result["purge_set_size"] == 0, (
        "Uppercase UUID from ArcadeDB must normalize to lowercase for Postgres lookup. "
        "PROCESSING run must be preserved (purge_set_size=0), not misidentified as orphan."
    )
    mock_cleanup.assert_not_called()
