"""Unit tests for cleanup_extraction_index — VR Phase C.2.

All tests are mocked (no ArcadeDB).

TDD discipline: tests written BEFORE implementation.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def _make_mock_store(deleted_count: int = 0) -> MagicMock:
    """Return a mock ArcadeDBGraphStore with command_sync stubbed to return a count."""
    store = MagicMock()
    store._database = "test_db"
    # ArcadeDB DELETE returns a list with a dict containing 'count'
    store._client.command_sync.return_value = [{"count": deleted_count}]
    return store


class TestCleanupReturnsDeletedCount:
    """test_cleanup_returns_deleted_count"""

    def test_returns_42_when_store_reports_42(self):
        """cleanup_extraction_index returns the count from the store response."""
        from app.services.extraction_chunk_index import cleanup_extraction_index

        store = _make_mock_store(deleted_count=42)
        result = cleanup_extraction_index("run-abc", store=store)
        assert result == 42

    def test_returns_zero_when_nothing_deleted(self):
        """cleanup_extraction_index returns 0 when no rows matched."""
        from app.services.extraction_chunk_index import cleanup_extraction_index

        store = _make_mock_store(deleted_count=0)
        result = cleanup_extraction_index("run-empty", store=store)
        assert result == 0

    def test_returns_int_type(self):
        """Return value is always an int."""
        from app.services.extraction_chunk_index import cleanup_extraction_index

        store = _make_mock_store(deleted_count=7)
        result = cleanup_extraction_index("run-x", store=store)
        assert isinstance(result, int)

    def test_invokes_command_sync_once(self):
        """Exactly one SQL command is issued to the store."""
        from app.services.extraction_chunk_index import cleanup_extraction_index

        store = _make_mock_store(deleted_count=3)
        cleanup_extraction_index("run-y", store=store)
        assert store._client.command_sync.call_count == 1


class TestCleanupSwallowsException:
    """test_cleanup_swallows_exception_returns_zero"""

    def test_exception_returns_zero(self):
        """When store raises, cleanup returns 0 instead of propagating."""
        from app.services.extraction_chunk_index import cleanup_extraction_index

        store = MagicMock()
        store._database = "test_db"
        store._client.command_sync.side_effect = RuntimeError("ArcadeDB offline")

        result = cleanup_extraction_index("run-bad", store=store)
        assert result == 0

    def test_exception_logs_warning(self, caplog):
        """When store raises, a WARNING-level log is emitted."""
        from app.services.extraction_chunk_index import cleanup_extraction_index

        store = MagicMock()
        store._database = "test_db"
        store._client.command_sync.side_effect = RuntimeError("connection refused")

        with caplog.at_level(logging.WARNING, logger="app.services.extraction_chunk_index"):
            cleanup_extraction_index("run-warn", store=store)

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, (
            "cleanup_extraction_index must emit a WARNING when the store raises"
        )

    def test_exception_does_not_reraise(self):
        """cleanup_extraction_index never raises — best-effort contract."""
        from app.services.extraction_chunk_index import cleanup_extraction_index

        store = MagicMock()
        store._database = "test_db"
        store._client.command_sync.side_effect = Exception("unexpected failure")

        # Must not raise
        try:
            cleanup_extraction_index("run-noraise", store=store)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"cleanup_extraction_index raised unexpectedly: {exc}")


class TestCleanupScopeMatchingRunIdOnly:
    """test_cleanup_scope_only_matching_run_id"""

    def test_delete_sql_contains_where_pipeline_run_id(self):
        """DELETE SQL must contain 'WHERE pipeline_run_id = :run_id' (or equivalent)."""
        from app.services.extraction_chunk_index import cleanup_extraction_index

        store = _make_mock_store(deleted_count=5)
        cleanup_extraction_index("specific-run-id", store=store)

        call = store._client.command_sync.call_args
        # Positional: (database, language, sql[, params])
        args = call[0]
        sql = args[2]
        params = args[3] if len(args) > 3 else call[1].get("params", {})

        assert "ExtractionChunk" in sql, f"SQL must reference ExtractionChunk: {sql!r}"
        assert "pipeline_run_id" in sql, f"SQL must filter by pipeline_run_id: {sql!r}"

        # Parameterized — value must be in params, not baked into the SQL literal
        run_id_value = (params or {}).get("run_id")
        assert run_id_value == "specific-run-id", (
            f"run_id bind param must match the argument; "
            f"got params={params!r}"
        )

    def test_delete_sql_uses_bind_param_not_string_interpolation(self):
        """SQL uses :run_id bind parameter, not f-string interpolation (SQL injection safety)."""
        from app.services.extraction_chunk_index import cleanup_extraction_index

        store = _make_mock_store(deleted_count=0)
        cleanup_extraction_index("'; DROP TABLE ExtractionChunk; --", store=store)

        call = store._client.command_sync.call_args
        args = call[0]
        sql = args[2]

        # The malicious string must NOT appear literally in the SQL
        assert "DROP TABLE" not in sql, (
            "SQL must use parameterized query, not string interpolation"
        )
        assert ":run_id" in sql, "SQL must use :run_id bind parameter"

    def test_different_run_ids_pass_different_params(self):
        """Two cleanup calls with different run_ids each use correct bind params."""
        from app.services.extraction_chunk_index import cleanup_extraction_index

        store = _make_mock_store(deleted_count=1)

        cleanup_extraction_index("run-aaa", store=store)
        first_call = store._client.command_sync.call_args_list[0]
        first_params = first_call[0][3] if len(first_call[0]) > 3 else first_call[1].get("params", {})

        cleanup_extraction_index("run-bbb", store=store)
        second_call = store._client.command_sync.call_args_list[1]
        second_params = second_call[0][3] if len(second_call[0]) > 3 else second_call[1].get("params", {})

        assert (first_params or {}).get("run_id") == "run-aaa"
        assert (second_params or {}).get("run_id") == "run-bbb"
