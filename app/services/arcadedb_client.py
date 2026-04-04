"""ArcadeDB HTTP/JSON API client with token-based auth."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ArcadeDBError(Exception):
    """Base exception for ArcadeDB client errors."""

    def __init__(self, message: str, status_code: int | None = None, detail: str | None = None):
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class ArcadeDBClient:
    """Async/sync HTTP client for ArcadeDB server."""

    def __init__(self, base_url: str, username: str, password: str):
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._token: str | None = None
        self._token_time: float = 0
        self._token_ttl: float = 1500  # 25 min (server default 30 min, refresh early)
        self._async_client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

    # --- Auth ---

    def _token_expired(self) -> bool:
        return self._token is None or (time.monotonic() - self._token_time) > self._token_ttl

    async def _login_async(self) -> None:
        client = self._get_async_client()
        resp = await client.post(
            f"{self._base_url}/api/v1/login",
            auth=(self._username, self._password),
        )
        resp.raise_for_status()
        self._token = resp.json().get("token") or resp.headers.get("arcadedb-session-id")
        self._token_time = time.monotonic()

    def _login_sync(self) -> None:
        client = self._get_sync_client()
        resp = client.post(
            f"{self._base_url}/api/v1/login",
            auth=(self._username, self._password),
        )
        resp.raise_for_status()
        self._token = resp.json().get("token") or resp.headers.get("arcadedb-session-id")
        self._token_time = time.monotonic()

    def _auth_headers(self) -> dict[str, str]:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    # --- Client lifecycle ---

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(timeout=60.0)
        return self._async_client

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(timeout=60.0)
        return self._sync_client

    async def _ensure_auth_async(self) -> None:
        if self._token_expired():
            await self._login_async()

    def _ensure_auth_sync(self) -> None:
        if self._token_expired():
            self._login_sync()

    # --- Async API ---

    async def query(
        self,
        database: str,
        language: str,
        command: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read-only query against ArcadeDB."""
        await self._ensure_auth_async()
        client = self._get_async_client()
        body: dict[str, Any] = {"language": language, "command": command}
        if params:
            body["params"] = params
        resp = await client.post(
            f"{self._base_url}/api/v1/query/{database}",
            json=body,
            headers=self._auth_headers(),
        )
        if resp.status_code == 401:
            await self._login_async()
            resp = await client.post(
                f"{self._base_url}/api/v1/query/{database}",
                json=body,
                headers=self._auth_headers(),
            )
        resp.raise_for_status()
        return resp.json().get("result", [])

    async def command(
        self,
        database: str,
        language: str,
        command: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a write command against ArcadeDB."""
        await self._ensure_auth_async()
        client = self._get_async_client()
        body: dict[str, Any] = {"language": language, "command": command}
        if params:
            body["params"] = params
        resp = await client.post(
            f"{self._base_url}/api/v1/command/{database}",
            json=body,
            headers=self._auth_headers(),
        )
        if resp.status_code == 401:
            await self._login_async()
            resp = await client.post(
                f"{self._base_url}/api/v1/command/{database}",
                json=body,
                headers=self._auth_headers(),
            )
        resp.raise_for_status()
        return resp.json().get("result", [])

    async def batch(
        self,
        database: str,
        data: str,
        light_edges: bool = True,
    ) -> dict[str, Any]:
        """Bulk import NDJSON data via the batch endpoint."""
        await self._ensure_auth_async()
        client = self._get_async_client()
        url = f"{self._base_url}/api/v1/batch/{database}"
        if light_edges:
            url += "?lightEdges=true"
        resp = await client.post(
            url,
            content=data,
            headers={**self._auth_headers(), "Content-Type": "application/x-ndjson"},
        )
        resp.raise_for_status()
        return resp.json()

    async def begin(self, database: str) -> str:
        """Begin a transaction; returns the arcadedb-session-id."""
        await self._ensure_auth_async()
        client = self._get_async_client()
        resp = await client.post(
            f"{self._base_url}/api/v1/begin/{database}",
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.headers.get("arcadedb-session-id", "")

    async def commit(self, database: str, session_id: str) -> None:
        """Commit an open transaction."""
        client = self._get_async_client()
        resp = await client.post(
            f"{self._base_url}/api/v1/commit/{database}",
            headers={**self._auth_headers(), "arcadedb-session-id": session_id},
        )
        resp.raise_for_status()

    async def rollback(self, database: str, session_id: str) -> None:
        """Roll back an open transaction."""
        client = self._get_async_client()
        resp = await client.post(
            f"{self._base_url}/api/v1/rollback/{database}",
            headers={**self._auth_headers(), "arcadedb-session-id": session_id},
        )
        resp.raise_for_status()

    # --- Sync API ---

    def query_sync(
        self,
        database: str,
        language: str,
        command: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronous read-only query."""
        self._ensure_auth_sync()
        client = self._get_sync_client()
        body: dict[str, Any] = {"language": language, "command": command}
        if params:
            body["params"] = params
        resp = client.post(
            f"{self._base_url}/api/v1/query/{database}",
            json=body,
            headers=self._auth_headers(),
        )
        if resp.status_code == 401:
            self._login_sync()
            resp = client.post(
                f"{self._base_url}/api/v1/query/{database}",
                json=body,
                headers=self._auth_headers(),
            )
        resp.raise_for_status()
        return resp.json().get("result", [])

    def command_sync(
        self,
        database: str,
        language: str,
        command: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronous write command."""
        self._ensure_auth_sync()
        client = self._get_sync_client()
        body: dict[str, Any] = {"language": language, "command": command}
        if params:
            body["params"] = params
        resp = client.post(
            f"{self._base_url}/api/v1/command/{database}",
            json=body,
            headers=self._auth_headers(),
        )
        if resp.status_code == 401:
            self._login_sync()
            resp = client.post(
                f"{self._base_url}/api/v1/command/{database}",
                json=body,
                headers=self._auth_headers(),
            )
        resp.raise_for_status()
        return resp.json().get("result", [])

    def batch_sync(
        self,
        database: str,
        data: str,
        light_edges: bool = True,
    ) -> dict[str, Any]:
        """Synchronous bulk NDJSON import."""
        self._ensure_auth_sync()
        client = self._get_sync_client()
        url = f"{self._base_url}/api/v1/batch/{database}"
        if light_edges:
            url += "?lightEdges=true"
        resp = client.post(
            url,
            content=data,
            headers={**self._auth_headers(), "Content-Type": "application/x-ndjson"},
        )
        resp.raise_for_status()
        return resp.json()

    # --- Lifecycle ---

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()

    def close_sync(self) -> None:
        """Close the sync HTTP client."""
        if self._sync_client and not self._sync_client.is_closed:
            self._sync_client.close()
