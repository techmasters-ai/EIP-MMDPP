"""Unit tests for ArcadeDBGraphStore.vector_search() filters parameter.

VR Phase C.1 — rev 10: adds `filters: dict[str, Any] | None = None` alongside
the existing `document_ids` parameter. Both compile to WHERE clauses; ANDed
when both are provided. Unknown filter keys raise ValueError to catch typos.

These tests are unit (mocked) — no ArcadeDB server required.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(query_result: list | None = None) -> MagicMock:
    """Build a mock ArcadeDBClient."""
    client = MagicMock()
    client.command = AsyncMock(return_value=[{"@rid": "#1:0"}])
    client.query = AsyncMock(return_value=query_result or [])
    client.command_sync = MagicMock(return_value=[{"@rid": "#1:0"}])
    client.query_sync = MagicMock(return_value=[])
    return client


def _graph(client: MagicMock | None = None):
    """Build ArcadeDBGraphStore with mocked client, validation matrix disabled."""
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    store = ArcadeDBGraphStore(client=client or _make_client(), database="testdb")
    store._validation_matrix = set()
    return store


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _captured_sql(client: MagicMock) -> str:
    """Return the SQL string passed to client.query (first positional arg after database + type)."""
    assert client.query.await_count >= 1, "client.query was never called"
    # call args: (database, query_type, sql, params)
    args = client.query.call_args
    # positional: (database, "sql", sql_string, params_dict)
    if args.args and len(args.args) >= 3:
        return args.args[2]
    raise AssertionError(f"Unexpected call signature: {args}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVectorSearchFilters:
    """Verify that the `filters` parameter generates correct WHERE clauses."""

    def test_vector_search_with_filters_includes_where_clause(self):
        """filters={"pipeline_run_id": "abc"} → WHERE clause includes pipeline_run_id = 'abc'."""
        client = _make_client()
        store = _graph(client)

        _run(
            store.vector_search(
                vertex_type="ExtractionChunk",
                embedding_property="embedding",
                query_vector=[0.1] * 1024,
                top_k=10,
                score_threshold=None,
                filters={"pipeline_run_id": "abc"},
            )
        )

        sql = _captured_sql(client)
        # The WHERE clause must reference pipeline_run_id with value 'abc'
        # Accept both parameterized (:pipeline_run_id) and literal ('abc') forms.
        assert (
            "pipeline_run_id" in sql
        ), f"Expected 'pipeline_run_id' in SQL WHERE clause, got:\n{sql}"
        assert (
            "WHERE" in sql
        ), f"Expected WHERE keyword in SQL, got:\n{sql}"

    def test_vector_search_with_both_document_ids_and_filters(self):
        """Both document_ids and filters → WHERE clause ANDs them together."""
        client = _make_client()
        store = _graph(client)

        _run(
            store.vector_search(
                vertex_type="ExtractionChunk",
                embedding_property="embedding",
                query_vector=[0.1] * 1024,
                top_k=10,
                score_threshold=None,
                document_ids=["doc-a", "doc-b"],
                filters={"pipeline_run_id": "run-xyz"},
            )
        )

        sql = _captured_sql(client)
        assert "document_id" in sql, f"Expected 'document_id' in SQL, got:\n{sql}"
        assert "pipeline_run_id" in sql, f"Expected 'pipeline_run_id' in SQL, got:\n{sql}"
        assert "AND" in sql, (
            f"Expected AND to join document_ids clause and filters clause, got:\n{sql}"
        )

    def test_vector_search_filters_none_is_unfiltered(self):
        """filters=None → no extra WHERE clause emitted (unfiltered vector search)."""
        client = _make_client()
        store = _graph(client)

        _run(
            store.vector_search(
                vertex_type="ExtractionChunk",
                embedding_property="embedding",
                query_vector=[0.1] * 1024,
                top_k=10,
                score_threshold=None,
                filters=None,
            )
        )

        sql = _captured_sql(client)
        # Should NOT add a pipeline_run_id constraint when filters=None
        assert "pipeline_run_id" not in sql, (
            f"filters=None should not add pipeline_run_id to SQL, got:\n{sql}"
        )

    def test_vector_search_filters_empty_dict_is_unfiltered(self):
        """filters={} → same as None (no extra WHERE clause emitted)."""
        client = _make_client()
        store = _graph(client)

        _run(
            store.vector_search(
                vertex_type="ExtractionChunk",
                embedding_property="embedding",
                query_vector=[0.1] * 1024,
                top_k=10,
                score_threshold=None,
                filters={},
            )
        )

        sql = _captured_sql(client)
        # Empty filters dict = no additional constraints
        assert "pipeline_run_id" not in sql, (
            f"filters={{}} should not add pipeline_run_id to SQL, got:\n{sql}"
        )

    def test_vector_search_filters_unknown_key_raises(self):
        """filters with an unknown property key raises ValueError (typo catcher)."""
        store = _graph()

        with pytest.raises(ValueError, match="bogus_property"):
            _run(
                store.vector_search(
                    vertex_type="ExtractionChunk",
                    embedding_property="embedding",
                    query_vector=[0.1] * 1024,
                    top_k=10,
                    score_threshold=None,
                    filters={"bogus_property": "x"},
                )
            )

    def test_vector_search_filters_multi_key_AND(self):
        """Multiple filter keys must all combine via AND in the WHERE clause.

        Code path exists in arcadedb_graph.py:vector_search around lines 1633-1637
        (the loop: `for k, v in filters.items(): where_parts.append(...)`).
        This test was added in C.1 review nit #10 to cover the multi-key loop that
        had no dedicated test.

        Verifies:
          - Both filter keys appear in the emitted SQL.
          - Both bind parameters appear in the params dict (via parameterized form).
          - The AND keyword joins the clauses.
        """
        client = _make_client()
        store = _graph(client)

        _run(
            store.vector_search(
                vertex_type="ExtractionChunk",
                embedding_property="embedding",
                query_vector=[0.0] * 1024,
                top_k=10,
                score_threshold=None,
                filters={
                    "pipeline_run_id": "run-A",
                    "document_id": "doc-X",
                },
            )
        )

        sql = _captured_sql(client)

        # Both filter columns must appear in the SQL.
        assert "pipeline_run_id" in sql, (
            f"Expected 'pipeline_run_id' in SQL for multi-key filter, got:\n{sql}"
        )
        assert "document_id" in sql, (
            f"Expected 'document_id' in SQL for multi-key filter, got:\n{sql}"
        )

        # The two conditions must be joined by AND.
        assert "AND" in sql, (
            f"Expected AND keyword to join multi-key filter conditions, got:\n{sql}"
        )

        # Both parameterized bind keys must appear (parameterized form prevents
        # SQL injection — see arcadedb_graph.py comment at the filter loop).
        assert ":filter_pipeline_run_id" in sql, (
            f"Expected parameterized ':filter_pipeline_run_id' in SQL, got:\n{sql}"
        )
        assert ":filter_document_id" in sql, (
            f"Expected parameterized ':filter_document_id' in SQL, got:\n{sql}"
        )

        # Confirm the actual bind params dict was populated with both values.
        assert client.query.await_count >= 1, "client.query was never called"
        call_args = client.query.call_args
        # Signature: (database, "sql", sql_string, params_dict)
        params = call_args.args[3] if len(call_args.args) >= 4 else {}
        assert params.get("filter_pipeline_run_id") == "run-A", (
            f"Expected filter_pipeline_run_id='run-A' in params, got: {params}"
        )
        assert params.get("filter_document_id") == "doc-X", (
            f"Expected filter_document_id='doc-X' in params, got: {params}"
        )


class TestProtocolStubIncludesFilters:
    """Verify the Protocol stub in graph_store.py declares the filters parameter."""

    def test_protocol_vector_search_signature_has_filters(self):
        """GraphStore Protocol stub must include `filters` param (synced to concrete impl)."""
        import inspect
        from app.services.graph_store import GraphStore

        sig = inspect.signature(GraphStore.vector_search)
        param_names = list(sig.parameters.keys())
        assert "filters" in param_names, (
            f"GraphStore.vector_search Protocol stub is missing 'filters' parameter. "
            f"Current params: {param_names}. "
            f"Sync the Protocol stub in graph_store.py with the concrete implementation."
        )

    def test_protocol_vector_search_signature_has_document_ids(self):
        """GraphStore Protocol stub must retain existing `document_ids` param (backward compat)."""
        import inspect
        from app.services.graph_store import GraphStore

        sig = inspect.signature(GraphStore.vector_search)
        param_names = list(sig.parameters.keys())
        assert "document_ids" in param_names, (
            f"GraphStore.vector_search Protocol stub is missing 'document_ids' parameter. "
            f"Current params: {param_names}."
        )
