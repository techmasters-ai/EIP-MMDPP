"""Shared Redis client factory and locking helpers.

Provides a module-level singleton Redis client so all callers share a single
connection pool, and a standard ``redis_lock`` helper that wraps
``redis.Redis.lock()`` with auto-release semantics.
"""

from __future__ import annotations

import logging
from typing import Any

import redis as redis_lib

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: redis_lib.Redis | None = None


def get_redis() -> redis_lib.Redis:
    """Return a process-wide Redis client, creating it on first use.

    The client uses the broker URL from settings and a built-in connection
    pool. Callers should not close the returned client; use ``close_redis()``
    at process shutdown instead.
    """
    global _client
    if _client is None:
        _client = redis_lib.Redis.from_url(get_settings().celery_broker_url)
    return _client


def close_redis() -> None:
    """Close the shared client on process shutdown."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception as exc:
            logger.warning("Error closing shared Redis client: %s", exc)
        _client = None


def redis_lock(
    key: str,
    *,
    timeout: float,
    blocking: bool = False,
) -> Any:
    """Acquire a Redis lock via ``redis.Redis.lock()``.

    Returns the lock object on successful acquisition, or ``None`` if the lock
    could not be acquired (only in non-blocking mode). Callers must call
    ``lock.release()`` when done, ideally in a try/finally.

    Use in preference to bare ``SET NX`` so acquisition, TTL, and release all
    flow through the same idiom.
    """
    lock = get_redis().lock(key, timeout=timeout, blocking=blocking)
    if lock.acquire(blocking=blocking):
        return lock
    return None
