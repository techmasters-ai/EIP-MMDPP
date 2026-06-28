"""Unit tests for ArcadeDB HTTP/JSON API client.

Tests auth flow, query/command/batch/transaction endpoints, retry-on-401,
sync variants, and error propagation — all using unittest.mock.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import httpx
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int, body: dict | None = None, headers: dict | None = None):
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = body or {}
    resp.headers = headers or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _login_response():
    return _make_response(200, {"token": "test-bearer-token"})


def _query_response(result=None):
    return _make_response(200, {"result": result or [{"@rid": "#1:0"}]})


# ---------------------------------------------------------------------------
# Test: query sends correct request
# ---------------------------------------------------------------------------

class TestQuerySendsCorrectRequest:
    @pytest.mark.asyncio
    async def test_query_url_and_body(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        async_post = AsyncMock(side_effect=[_login_response(), _query_response()])

        with patch.object(httpx.AsyncClient, "post", async_post):
            result = await client.query("mydb", "sql", "SELECT * FROM V", {"p": 1})

        # First call: login
        login_call = async_post.call_args_list[0]
        assert "/api/v1/login" in login_call.args[0]

        # Second call: query
        query_call = async_post.call_args_list[1]
        assert query_call.args[0] == "http://localhost:2480/api/v1/query/mydb"
        sent_body = query_call.kwargs.get("json", {})
        assert sent_body["language"] == "sql"
        assert sent_body["command"] == "SELECT * FROM V"
        assert sent_body["params"] == {"p": 1}

        assert result == [{"@rid": "#1:0"}]

    @pytest.mark.asyncio
    async def test_query_includes_auth_header(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        async_post = AsyncMock(side_effect=[_login_response(), _query_response()])

        with patch.object(httpx.AsyncClient, "post", async_post):
            await client.query("mydb", "sql", "SELECT 1")

        query_call = async_post.call_args_list[1]
        headers = query_call.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer test-bearer-token"

    @pytest.mark.asyncio
    async def test_query_no_params_omits_params_key(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        async_post = AsyncMock(side_effect=[_login_response(), _query_response()])

        with patch.object(httpx.AsyncClient, "post", async_post):
            await client.query("mydb", "sql", "SELECT 1")

        query_call = async_post.call_args_list[1]
        sent_body = query_call.kwargs.get("json", {})
        assert "params" not in sent_body


# ---------------------------------------------------------------------------
# Test: command uses POST to /api/v1/command/{database}
# ---------------------------------------------------------------------------

class TestCommandSendsPost:
    @pytest.mark.asyncio
    async def test_command_url(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        async_post = AsyncMock(side_effect=[
            _login_response(),
            _make_response(200, {"result": []}),
        ])

        with patch.object(httpx.AsyncClient, "post", async_post):
            result = await client.command("mydb", "sql", "CREATE VERTEX V SET name='x'")

        cmd_call = async_post.call_args_list[1]
        assert cmd_call.args[0] == "http://localhost:2480/api/v1/command/mydb"
        assert result == []

    @pytest.mark.asyncio
    async def test_command_body_format(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        async_post = AsyncMock(side_effect=[
            _login_response(),
            _make_response(200, {"result": [{"count": 1}]}),
        ])

        with patch.object(httpx.AsyncClient, "post", async_post):
            await client.command("db", "gremlin", "g.V().count()", {"x": "y"})

        cmd_call = async_post.call_args_list[1]
        body = cmd_call.kwargs.get("json", {})
        assert body == {"language": "gremlin", "command": "g.V().count()", "params": {"x": "y"}}


# ---------------------------------------------------------------------------
# Test: auth token is cached across calls
# ---------------------------------------------------------------------------

class TestAuthTokenCached:
    @pytest.mark.asyncio
    async def test_second_call_reuses_token(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        # login once + two query responses
        async_post = AsyncMock(side_effect=[
            _login_response(),
            _query_response([{"a": 1}]),
            _query_response([{"b": 2}]),
        ])

        with patch.object(httpx.AsyncClient, "post", async_post):
            r1 = await client.query("db", "sql", "SELECT 1")
            r2 = await client.query("db", "sql", "SELECT 2")

        # Only 3 calls total: 1 login + 2 queries (not 2 logins)
        assert async_post.call_count == 3
        assert r1 == [{"a": 1}]
        assert r2 == [{"b": 2}]


# ---------------------------------------------------------------------------
# Test: 401 triggers re-login and retry
# ---------------------------------------------------------------------------

class TestAuthReloginOn401:
    @pytest.mark.asyncio
    async def test_relogin_and_retry_on_401(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        # Pre-seed a "stale" token so no initial login is triggered
        client._token = "stale-token"
        import time
        client._token_time = time.monotonic()

        stale_401 = _make_response(401)
        stale_401.raise_for_status.return_value = None  # don't raise on 401 — client checks status_code

        new_login = _login_response()
        success = _query_response([{"ok": True}])

        # order: first query attempt → 401, re-login, retry → success
        async_post = AsyncMock(side_effect=[stale_401, new_login, success])

        with patch.object(httpx.AsyncClient, "post", async_post):
            result = await client.query("db", "sql", "SELECT 1")

        assert result == [{"ok": True}]
        # calls: initial query attempt (401), re-login, retry
        assert async_post.call_count == 3

        login_url = async_post.call_args_list[1].args[0]
        assert "/api/v1/login" in login_url

    @pytest.mark.asyncio
    async def test_command_relogin_on_401(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        client._token = "stale-token"
        import time
        client._token_time = time.monotonic()

        stale_401 = _make_response(401)
        stale_401.raise_for_status.return_value = None

        async_post = AsyncMock(side_effect=[
            stale_401,
            _login_response(),
            _make_response(200, {"result": []}),
        ])

        with patch.object(httpx.AsyncClient, "post", async_post):
            result = await client.command("db", "sql", "DELETE VERTEX")

        assert result == []
        assert async_post.call_count == 3


# ---------------------------------------------------------------------------
# Test: batch sends NDJSON Content-Type
# ---------------------------------------------------------------------------

class TestBatchSendsNdjsonContentType:
    @pytest.mark.asyncio
    async def test_batch_content_type(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        ndjson = '{"@type":"v","@cat":"Foo","name":"test"}\n'

        async_post = AsyncMock(side_effect=[
            _login_response(),
            _make_response(200, {"result": 1, "importedVertices": 1}),
        ])

        with patch.object(httpx.AsyncClient, "post", async_post):
            result = await client.batch("mydb", ndjson)

        batch_call = async_post.call_args_list[1]
        headers = batch_call.kwargs.get("headers", {})
        assert headers.get("Content-Type") == "application/x-ndjson"
        assert batch_call.kwargs.get("content") == ndjson

    @pytest.mark.asyncio
    async def test_batch_url_with_light_edges(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        async_post = AsyncMock(side_effect=[
            _login_response(),
            _make_response(200, {}),
        ])

        with patch.object(httpx.AsyncClient, "post", async_post):
            await client.batch("mydb", "", light_edges=True)

        url = async_post.call_args_list[1].args[0]
        assert "lightEdges=true" in url

    @pytest.mark.asyncio
    async def test_batch_url_without_light_edges(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        async_post = AsyncMock(side_effect=[
            _login_response(),
            _make_response(200, {}),
        ])

        with patch.object(httpx.AsyncClient, "post", async_post):
            await client.batch("mydb", "", light_edges=False)

        url = async_post.call_args_list[1].args[0]
        assert "lightEdges" not in url


# ---------------------------------------------------------------------------
# Test: sync query works
# ---------------------------------------------------------------------------

class TestSyncQueryWorks:
    def test_query_sync_url_and_result(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        sync_post = MagicMock(side_effect=[
            _login_response(),
            _query_response([{"id": "1"}]),
        ])

        with patch.object(httpx.Client, "post", sync_post):
            result = client.query_sync("db", "sql", "SELECT FROM V")

        assert result == [{"id": "1"}]
        query_call = sync_post.call_args_list[1]
        assert "/api/v1/query/db" in query_call.args[0]

    def test_command_sync_url(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        sync_post = MagicMock(side_effect=[
            _login_response(),
            _make_response(200, {"result": []}),
        ])

        with patch.object(httpx.Client, "post", sync_post):
            result = client.command_sync("db", "sql", "INSERT INTO V SET x=1")

        assert result == []
        cmd_call = sync_post.call_args_list[1]
        assert "/api/v1/command/db" in cmd_call.args[0]

    def test_batch_sync_ndjson_header(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        ndjson = '{"@type":"v"}\n'

        sync_post = MagicMock(side_effect=[
            _login_response(),
            _make_response(200, {"result": 1}),
        ])

        with patch.object(httpx.Client, "post", sync_post):
            client.batch_sync("db", ndjson)

        batch_call = sync_post.call_args_list[1]
        assert batch_call.kwargs.get("headers", {}).get("Content-Type") == "application/x-ndjson"

    def test_sync_401_triggers_relogin(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        client._token = "stale"
        import time
        client._token_time = time.monotonic()

        stale_401 = _make_response(401)
        stale_401.raise_for_status.return_value = None

        sync_post = MagicMock(side_effect=[
            stale_401,
            _login_response(),
            _query_response([{"x": 1}]),
        ])

        with patch.object(httpx.Client, "post", sync_post):
            result = client.query_sync("db", "sql", "SELECT 1")

        assert result == [{"x": 1}]
        assert sync_post.call_count == 3


# ---------------------------------------------------------------------------
# Test: error handling — non-2xx raises HTTPStatusError
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_404_raises_http_status_error(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")

        # Pre-seed token so we skip login
        client._token = "tok"
        import time
        client._token_time = time.monotonic()

        error_resp = _make_response(404, {"error": "Database not found"})
        # 404 is not 401, so the client should call raise_for_status after the request
        async_post = AsyncMock(side_effect=[error_resp])

        with patch.object(httpx.AsyncClient, "post", async_post):
            with pytest.raises(httpx.HTTPStatusError):
                await client.query("nonexistent", "sql", "SELECT 1")

    @pytest.mark.asyncio
    async def test_500_raises_http_status_error(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        client._token = "tok"
        import time
        client._token_time = time.monotonic()

        error_resp = _make_response(500, {"error": "Internal server error"})

        async_post = AsyncMock(side_effect=[error_resp])

        with patch.object(httpx.AsyncClient, "post", async_post):
            with pytest.raises(httpx.HTTPStatusError):
                await client.command("db", "sql", "INVALID SYNTAX")

    def test_sync_500_raises(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        client._token = "tok"
        import time
        client._token_time = time.monotonic()

        error_resp = _make_response(500)

        with patch.object(httpx.Client, "post", MagicMock(return_value=error_resp)):
            with pytest.raises(httpx.HTTPStatusError):
                client.query_sync("db", "sql", "SELECT 1")


# ---------------------------------------------------------------------------
# Test: transaction helpers (begin/commit/rollback)
# ---------------------------------------------------------------------------

class TestTransactionHelpers:
    @pytest.mark.asyncio
    async def test_begin_returns_session_id(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        client._token = "tok"
        import time
        client._token_time = time.monotonic()

        begin_resp = _make_response(200, {}, headers={"arcadedb-session-id": "sess-abc"})

        async_post = AsyncMock(return_value=begin_resp)

        with patch.object(httpx.AsyncClient, "post", async_post):
            session_id = await client.begin("mydb")

        assert session_id == "sess-abc"
        call_url = async_post.call_args.args[0]
        assert "/api/v1/begin/mydb" in call_url

    @pytest.mark.asyncio
    async def test_commit_sends_session_header(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        client._token = "tok"
        import time
        client._token_time = time.monotonic()

        commit_resp = _make_response(200, {})

        async_post = AsyncMock(return_value=commit_resp)

        with patch.object(httpx.AsyncClient, "post", async_post):
            await client.commit("mydb", "sess-xyz")

        call_url = async_post.call_args.args[0]
        assert "/api/v1/commit/mydb" in call_url
        headers = async_post.call_args.kwargs.get("headers", {})
        assert headers.get("arcadedb-session-id") == "sess-xyz"

    @pytest.mark.asyncio
    async def test_rollback_sends_session_header(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        client._token = "tok"
        import time
        client._token_time = time.monotonic()

        rollback_resp = _make_response(200, {})

        async_post = AsyncMock(return_value=rollback_resp)

        with patch.object(httpx.AsyncClient, "post", async_post):
            await client.rollback("mydb", "sess-xyz")

        call_url = async_post.call_args.args[0]
        assert "/api/v1/rollback/mydb" in call_url
        headers = async_post.call_args.kwargs.get("headers", {})
        assert headers.get("arcadedb-session-id") == "sess-xyz"


# ---------------------------------------------------------------------------
# Test: close lifecycle
# ---------------------------------------------------------------------------

class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_close_async_client(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        mock_aclient = AsyncMock(spec=httpx.AsyncClient)
        mock_aclient.is_closed = False
        client._async_client = mock_aclient

        await client.close()

        mock_aclient.aclose.assert_called_once()

    def test_close_sync_client(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        mock_sclient = MagicMock(spec=httpx.Client)
        mock_sclient.is_closed = False
        client._sync_client = mock_sclient

        client.close_sync()

        mock_sclient.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_skips_if_already_closed(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        mock_aclient = AsyncMock(spec=httpx.AsyncClient)
        mock_aclient.is_closed = True
        client._async_client = mock_aclient

        await client.close()

        mock_aclient.aclose.assert_not_called()


# ---------------------------------------------------------------------------
# Test: command_sync retries transient ConcurrentModificationException (503)
# ---------------------------------------------------------------------------

class TestCommandSyncConcurrentModificationRetry:
    """command_sync re-issues on ArcadeDB MVCC write-conflicts (503 CM)."""

    @staticmethod
    def _cm_503():
        r = _make_response(503, {"error": "Cannot execute command"})
        r.text = ('{"error":"Cannot execute command","detail":"Concurrent '
                  'modification on page PageId(...) ... '
                  'ConcurrentModificationException"}')
        return r

    def test_retries_then_succeeds(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        ok = _make_response(200, {"result": [{"@rid": "#1:0"}]})
        sync_post = MagicMock(side_effect=[
            _login_response(),     # _ensure_auth_sync login
            self._cm_503(),        # initial attempt -> 503 CM
            self._cm_503(),        # retry 1 -> 503 CM
            ok,                    # retry 2 -> success
        ])
        with patch.object(httpx.Client, "post", sync_post), \
             patch("app.services.arcadedb_client.time.sleep") as mock_sleep:
            result = client.command_sync(
                "db", "sql", "CREATE EDGE HAS_IMAGE FROM #4:0 TO #13:0")

        assert result == [{"@rid": "#1:0"}]
        assert mock_sleep.call_count == 2       # slept before each retry
        assert sync_post.call_count == 4        # login + 3 command attempts

    def test_non_concurrent_503_raises_without_retry(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        other = _make_response(503, {"error": "some other failure"})
        other.text = '{"detail":"unrelated 503, not a write conflict"}'
        sync_post = MagicMock(side_effect=[_login_response(), other])
        with patch.object(httpx.Client, "post", sync_post), \
             patch("app.services.arcadedb_client.time.sleep") as mock_sleep:
            with pytest.raises(httpx.HTTPStatusError):
                client.command_sync("db", "sql", "INSERT INTO V SET x=1")

        assert mock_sleep.call_count == 0       # no retry for non-CM 503
        assert sync_post.call_count == 2        # login + 1 attempt

    def test_exhausts_retries_then_raises(self):
        from app.services.arcadedb_client import ArcadeDBClient

        client = ArcadeDBClient("http://localhost:2480", "root", "pass")
        responses = [_login_response()] + [self._cm_503() for _ in range(6)]
        sync_post = MagicMock(side_effect=responses)
        with patch.object(httpx.Client, "post", sync_post), \
             patch("app.services.arcadedb_client.time.sleep") as mock_sleep:
            with pytest.raises(httpx.HTTPStatusError):
                client.command_sync("db", "sql", "CREATE EDGE HAS_IMAGE ...")

        assert mock_sleep.call_count == 5       # 5 bounded retries
        assert sync_post.call_count == 7        # login + initial + 5 retries
