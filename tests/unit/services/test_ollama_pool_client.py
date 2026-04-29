"""Unit tests for OllamaPool routing core.

Pool tests cover acquire/release semantics, least-in-flight selection,
and counter integrity under concurrent acquire (proxy for thread-safety).
HTTP behavior is covered separately in test_ollama_pool_client_http.py.
"""
import json
import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.services.ollama_pool_client import OllamaPool


def test_pool_picks_lowest_inflight():
    pool = OllamaPool(urls=["http://a", "http://b", "http://c"])
    pool._inflight["http://a"] = 2
    pool._inflight["http://b"] = 1
    pool._inflight["http://c"] = 3
    url = pool.acquire()
    assert url == "http://b"
    assert pool._inflight["http://b"] == 2


def test_pool_round_robin_on_ties():
    """When all inflight counts are equal (the dominant case for serial
    workloads), successive acquisitions must rotate, not always pick urls[0]."""
    pool = OllamaPool(urls=["http://a", "http://b", "http://c"])
    seen: list[str] = []
    for _ in range(6):
        u = pool.acquire()
        pool.release(u)
        seen.append(u)
    # In 6 tied acquisitions across 3 URLs, each URL should appear at least
    # once (round-robin gives exactly 2 each).
    assert set(seen) == {"http://a", "http://b", "http://c"}


def test_pool_release_decrements():
    pool = OllamaPool(urls=["http://a"])
    pool.acquire()
    assert pool._inflight["http://a"] == 1
    pool.release("http://a")
    assert pool._inflight["http://a"] == 0


def test_pool_acquire_release_balanced_concurrent():
    """Sanity check on counter integrity under thread races. We don't assert
    a specific distribution — only that no count goes negative and the total
    matches the number of releases."""
    pool = OllamaPool(urls=["http://a", "http://b"])
    N = 200

    def worker():
        for _ in range(N):
            url = pool.acquire()
            pool.release(url)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert pool._inflight["http://a"] == 0
    assert pool._inflight["http://b"] == 0


def test_pool_rejects_empty_urls():
    with pytest.raises(ValueError, match="at least one URL"):
        OllamaPool(urls=[])


def test_pool_with_single_url_always_picks_it():
    pool = OllamaPool(urls=["http://only"])
    for _ in range(5):
        url = pool.acquire()
        assert url == "http://only"
        pool.release(url)
