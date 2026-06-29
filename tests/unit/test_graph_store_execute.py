"""Unit tests for the GraphStore raw-SQL escape hatch (TODO #76).

These public methods replace external callers reaching into the private
``graph_store._client`` attribute. They must hide the backend client AND the
database name, delegating to the underlying ArcadeDB client with the fixed
``"sql"`` language.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


async def test_execute_query_delegates_to_client():
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    client = MagicMock()
    client.query = AsyncMock(return_value=[{"total_chars": 42}])
    store = ArcadeDBGraphStore(client, "mydb")

    out = await store.execute_query("SELECT 1 FROM X WHERE y = :y", {"y": 3})

    client.query.assert_awaited_once_with("mydb", "sql", "SELECT 1 FROM X WHERE y = :y", {"y": 3})
    assert out == [{"total_chars": 42}]


async def test_execute_command_delegates_to_client():
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    store = ArcadeDBGraphStore(client, "mydb")

    await store.execute_command("DELETE VERTEX FROM C WHERE id = :cid", {"cid": 7})

    client.command.assert_awaited_once_with("mydb", "sql", "DELETE VERTEX FROM C WHERE id = :cid", {"cid": 7})


def test_execute_command_sync_delegates_to_client():
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    client = MagicMock()
    client.command_sync = MagicMock(return_value=[{"ok": True}])
    store = ArcadeDBGraphStore(client, "mydb")

    out = store.execute_command_sync("UPDATE T SET a = 1 UPSERT WHERE chunk_id = :c", {"c": "abc"})

    client.command_sync.assert_called_once_with(
        "mydb", "sql", "UPDATE T SET a = 1 UPSERT WHERE chunk_id = :c", {"c": "abc"},
    )
    assert out == [{"ok": True}]


def test_execute_query_sync_delegates_to_client():
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    client = MagicMock()
    client.query_sync = MagicMock(return_value=[{"rid": "#1:0"}])
    store = ArcadeDBGraphStore(client, "mydb")

    out = store.execute_query_sync("SELECT @rid AS rid FROM T WHERE d = :d", {"d": "x"})

    client.query_sync.assert_called_once_with(
        "mydb", "sql", "SELECT @rid AS rid FROM T WHERE d = :d", {"d": "x"},
    )
    assert out == [{"rid": "#1:0"}]


def test_execute_command_sync_supports_sqlscript_language():
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    client = MagicMock()
    client.command_sync = MagicMock(return_value=[])
    store = ArcadeDBGraphStore(client, "mydb")

    store.execute_command_sync("CREATE EDGE...;\nCREATE EDGE...", {"p": 1}, language="sqlscript")

    client.command_sync.assert_called_once_with(
        "mydb", "sqlscript", "CREATE EDGE...;\nCREATE EDGE...", {"p": 1},
    )


def test_execute_methods_default_params_to_none():
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    client = MagicMock()
    client.command_sync = MagicMock(return_value=[])
    store = ArcadeDBGraphStore(client, "db")
    store.execute_command_sync("DELETE FROM X")
    client.command_sync.assert_called_once_with("db", "sql", "DELETE FROM X", None)
