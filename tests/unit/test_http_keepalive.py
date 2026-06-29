"""Unit tests for the shared TCP-keepalive httpx client builders (TODO #82)."""
from __future__ import annotations

import socket

import httpx

from app.services._http_keepalive import (
    build_keepalive_async_client,
    build_keepalive_client,
    keepalive_socket_options,
)


def test_keepalive_socket_options_includes_so_keepalive():
    opts = keepalive_socket_options()
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in opts


def test_keepalive_socket_options_includes_tcp_knobs_on_linux():
    # Linux exposes the per-connection TCP_KEEP* constants; assert the
    # ~150s dead-peer detection tuning is present.
    if not hasattr(socket, "TCP_KEEPIDLE"):
        return
    opts = keepalive_socket_options()
    assert (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60) in opts
    assert (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 15) in opts
    assert (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6) in opts


def test_build_keepalive_client_applies_split_timeout():
    t = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=30.0)
    c = build_keepalive_client(t)
    try:
        assert isinstance(c, httpx.Client)
        assert c._timeout.connect == 10.0
        assert c._timeout.read == 60.0
        assert c._timeout.write == 60.0
        assert c._timeout.pool == 30.0
    finally:
        c.close()


def test_build_keepalive_async_client_type_and_timeout():
    t = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=30.0)
    c = build_keepalive_async_client(t)
    assert isinstance(c, httpx.AsyncClient)
    assert c._timeout.connect == 10.0
    assert c._timeout.read == 600.0


def test_build_keepalive_client_with_limits_builds_client():
    limits = httpx.Limits(
        max_keepalive_connections=10, max_connections=20, keepalive_expiry=30.0,
    )
    c = build_keepalive_client(
        httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=30.0), limits,
    )
    try:
        assert isinstance(c, httpx.Client)
    finally:
        c.close()


def test_arcadedb_sync_client_uses_keepalive_split_timeout():
    """The ArcadeDB sync client must use the split timeout (connect=10s),
    not a single flat 60s blanket, so a dead socket surfaces quickly."""
    from app.services.arcadedb_client import ArcadeDBClient

    client = ArcadeDBClient("http://localhost:2480", "root", "pass")
    sc = client._get_sync_client()
    try:
        assert sc._timeout.connect == 10.0
        assert sc._timeout.read == 60.0
    finally:
        sc.close()


def test_arcadedb_async_client_uses_keepalive_split_timeout():
    from app.services.arcadedb_client import ArcadeDBClient

    client = ArcadeDBClient("http://localhost:2480", "root", "pass")
    ac = client._get_async_client()
    # Constructed outside a running loop (current_loop=None); assert the
    # timeout split without awaiting aclose (GC reclaims it).
    assert ac._timeout.connect == 10.0
    assert ac._timeout.read == 60.0
