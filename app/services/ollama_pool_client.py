"""Direct-Ollama pool client (canonical).

Replaces LiteLLM (used inside docling_graph) and the per-callsite httpx
calls scattered across app/services/. Two clients sit on top of a shared
routing core:

    OllamaPool                — acquire() / release() / least-in-flight
    OllamaChatClient          — /v1/chat/completions; implements docling_graph's
                                LLMClientProtocol so it can be plugged into
                                PipelineConfig(llm_client=...)
    OllamaEmbeddingClient     — /v1/embeddings; thin helper for embedding.py

Constructed by `app.config.Settings.get_ollama_*_urls()` callers; one pool
per role (LLM / VLM / embedding) keyed off the matching env vars.

MIRROR: docker/docling-graph/app/ollama_pool_client.py. The two files are
byte-for-byte identical below the SHARED CODE marker; the docstring is the
only difference. tests/test_pool_client_mirror.py enforces this invariant.
"""
# === SHARED CODE BELOW THIS LINE ===
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Iterator, Literal, Mapping, Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaPool:
    """URL pool with least-in-flight routing.

    Tracks per-URL request counts behind a lock; `acquire()` returns the URL
    with the lowest current count and increments it; `release()` decrements.
    Always wrap acquire+release in try/finally so a failing call still
    releases its slot.
    """

    def __init__(self, urls: list[str]) -> None:
        if not urls:
            raise ValueError("OllamaPool requires at least one URL")
        seen: set[str] = set()
        ordered: list[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                ordered.append(u)
        self._urls: list[str] = ordered
        # Precomputed URL→index map; avoids self._urls.index(u) inside the lock.
        self._url_index: dict[str, int] = {u: i for i, u in enumerate(ordered)}
        self._inflight: dict[str, int] = {u: 0 for u in ordered}
        # Round-robin cursor — used to break ties when multiple URLs share
        # the lowest in-flight count. Without this, serial workloads always
        # pick urls[0] (min() returns first match on ties), defeating fan-out.
        # Initialized to -1 so the first tied acquisition (after the
        # cursor++ inside acquire) lands on urls[0].
        self._rr_cursor: int = -1
        self._lock = threading.Lock()
        # Per-URL routing counter for diagnostics / Gate 5 fan-out check.
        # Atomic with the lock that protects _inflight.
        self._served: dict[str, int] = {u: 0 for u in ordered}

    @property
    def urls(self) -> list[str]:
        return list(self._urls)

    @property
    def routing_metrics(self) -> dict[str, int]:
        """Snapshot of per-URL request counts (cumulative since pool creation)."""
        with self._lock:
            return dict(self._served)

    def acquire(self, exclude: set[str] | None = None) -> str:
        """Pick the URL with the lowest in-flight count (excluding any URL
        listed in `exclude`); increment in-flight + served counters and
        return the URL.

        Tie-break: round-robin across URLs sharing the minimum in-flight
        count. Cursor advances monotonically.
        """
        with self._lock:
            candidates = [u for u in self._urls if not exclude or u not in exclude]
            if not candidates:
                raise RuntimeError(
                    f"No URLs available (all {len(self._urls)} excluded)"
                )
            min_inflight = min(self._inflight[u] for u in candidates)
            tied = [u for u in candidates if self._inflight[u] == min_inflight]
            if len(tied) == 1:
                url = tied[0]
            else:
                # Round-robin among ties. Use cursor mod len(_urls) to keep
                # rotation stable even when `exclude` shrinks the candidate
                # set on retries.
                self._rr_cursor = (self._rr_cursor + 1) % len(self._urls)
                # Pick the tied URL whose index is closest to (but not below)
                # the cursor position; wrap if needed.
                tied_indexed = sorted(
                    (self._url_index[u], u) for u in tied
                )
                pick = next(
                    (u for idx, u in tied_indexed if idx >= self._rr_cursor),
                    tied_indexed[0][1],
                )
                url = pick
            self._inflight[url] += 1
            self._served[url] += 1
            return url

    def release(self, url: str) -> None:
        with self._lock:
            self._inflight[url] = max(0, self._inflight[url] - 1)
