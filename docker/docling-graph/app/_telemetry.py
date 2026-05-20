"""C0 telemetry helpers for the docling-graph service.

`PhaseTimer` is a tiny context manager that records the wall-time of a code
block into a dict under a named key. Used by ``main.py`` to capture the 6
service-side timing metrics required by Phase 0 of the wall-time-reduction
plan:

    with PhaseTimer(diagnostics, "sanitize_ms"):
        docling_document_json = _sanitize_docling_document(...)

The timer fires on the ``__exit__`` boundary regardless of whether the block
raised, so failure paths still record how much time they consumed. The
overhead is one ``time.perf_counter()`` call on each boundary — negligible
against the LLM-bound phases this is instrumenting.
"""
from __future__ import annotations

import time


class PhaseTimer:
    """Context manager that writes elapsed milliseconds into ``sink[key]``."""

    __slots__ = ("_sink", "_key", "_t0")

    def __init__(self, sink: dict, key: str) -> None:
        self._sink = sink
        self._key = key
        self._t0 = 0.0

    def __enter__(self) -> "PhaseTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._sink[self._key] = (time.perf_counter() - self._t0) * 1000.0
        # Returning None / falsy → exceptions propagate, which is the intent.
