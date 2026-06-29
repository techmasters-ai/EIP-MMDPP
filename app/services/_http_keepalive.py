"""Shared TCP-keepalive httpx client builders.

Every long-lived ``httpx`` client in the services layer (the Ollama pool
clients and the ArcadeDB client) needs the same defense against silently
dropped TCP connections: a single blanket read timeout lets a dead socket
hang the caller for as long as that timeout, with no error to trigger a
retry. Setting SO_KEEPALIVE + the TCP_KEEP* knobs at the socket level makes
the kernel detect a truly-dead peer in
``TCP_KEEPIDLE + KEEPCNT*KEEPINTVL = 60 + 6*15 = ~150s`` — well before any
application read timeout — so the connection surfaces as a retryable
``httpx`` error instead of hanging.

This helper (TODO #82) is used by the ArcadeDB client. The Ollama pool
clients keep their own byte-identical inline copy of the same socket tuning
in ``ollama_pool_client.py`` — they cannot import this module because the
docling-graph service runs a mirrored copy of ``ollama_pool_client.py``
(enforced by ``tests/test_pool_client_mirror.py``) and this main-app module
is not present in the docling-graph image. If the keepalive knobs below ever
change, update the inline copy in ``ollama_pool_client.py`` to match.

Each consumer supplies its own ``httpx.Timeout`` because their read profiles
differ (LLM generations run for minutes; ArcadeDB queries finish in well
under a second).
"""
from __future__ import annotations

import socket

import httpx


def keepalive_socket_options() -> list[tuple[int, int, int]]:
    """Return the SO_KEEPALIVE + TCP_KEEP* socket options for httpx transports.

    Falls back gracefully on platforms that lack the per-connection TCP_KEEP*
    constants (e.g. macOS has no ``TCP_KEEPIDLE``): the basic ``SO_KEEPALIVE``
    toggle is still applied, and if even that is unavailable an empty list is
    returned so the caller builds a default transport.
    """
    try:
        return [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60),   # idle 60s before first probe
            (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 15),  # 15s between probes
            (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6),     # 6 probes -> ~150s detection
        ]
    except AttributeError:
        try:
            return [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        except AttributeError:
            return []


def build_keepalive_client(
    timeout: httpx.Timeout, limits: httpx.Limits | None = None,
) -> httpx.Client:
    """Build a sync ``httpx.Client`` with TCP keepalive on its transport."""
    opts = keepalive_socket_options()
    transport: httpx.HTTPTransport | None = None
    if opts or limits is not None:
        kwargs: dict[str, object] = {}
        if opts:
            kwargs["socket_options"] = opts
        if limits is not None:
            kwargs["limits"] = limits
        transport = httpx.HTTPTransport(**kwargs)
    return httpx.Client(timeout=timeout, transport=transport)


def build_keepalive_async_client(
    timeout: httpx.Timeout, limits: httpx.Limits | None = None,
) -> httpx.AsyncClient:
    """Build an async ``httpx.AsyncClient`` with TCP keepalive on its transport."""
    opts = keepalive_socket_options()
    transport: httpx.AsyncHTTPTransport | None = None
    if opts or limits is not None:
        kwargs: dict[str, object] = {}
        if opts:
            kwargs["socket_options"] = opts
        if limits is not None:
            kwargs["limits"] = limits
        transport = httpx.AsyncHTTPTransport(**kwargs)
    return httpx.AsyncClient(timeout=timeout, transport=transport)
