"""Integration-test fixtures for tests/integration/.

Provides a live ArcadeDBGraphStore fixture when ArcadeDB is reachable,
and pytest.skip()s otherwise so host-dev runs that lack the compose
stack don't false-fail.
"""
from __future__ import annotations

import os
import socket

import pytest


def _arcadedb_reachable() -> tuple[str, int] | None:
    """Try candidate hosts, mirroring the pattern from test_migration_0015.py."""
    candidates = [
        ("arcadedb", 2480),
        ("localhost", 2480),
        ("127.0.0.1", 2480),
    ]
    for host, port in candidates:
        try:
            with socket.create_connection((host, port), timeout=1):
                return host, port
        except (OSError, socket.gaierror):
            continue
    return None


@pytest.fixture(scope="module")
def arcadedb_store():
    """Return a live ArcadeDBGraphStore, or skip if ArcadeDB is unreachable."""
    reach = _arcadedb_reachable()
    if reach is None:
        pytest.skip("ArcadeDB not reachable on arcadedb:2480 / localhost:2480")

    host, _port = reach

    # Point the client at the reachable host.  The default URL in config
    # uses the compose service name "arcadedb"; on the host machine we need
    # "localhost" instead.
    os.environ.setdefault("ARCADEDB_URL", f"http://{host}:2480")

    from app.config import get_settings
    settings = get_settings()

    from app.services.arcadedb_client import ArcadeDBClient
    from app.services.arcadedb_graph import ArcadeDBGraphStore

    client = ArcadeDBClient(
        base_url=f"http://{host}:2480",
        username=settings.arcadedb_user,
        password=settings.arcadedb_password,
    )
    store = ArcadeDBGraphStore(client, settings.arcadedb_database)

    try:
        store.ensure_ready_sync()
    except Exception as exc:
        pytest.skip(f"ArcadeDBGraphStore.ensure_ready_sync failed: {exc}")

    yield store
    # No teardown: each test uses a unique document_id (uuid4) for isolation.
