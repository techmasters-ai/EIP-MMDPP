"""Retrieval strategy performance benchmarks.

These tests are intended for **nightly CI** execution, not per-PR runs.
They require a running ArcadeDB instance with a seeded corpus of 1000+
documents to produce meaningful latency baselines.

Run with::

    pytest tests/benchmarks/ -m benchmark --timeout=300

All tests are skipped by default unless the corpus is available.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

_SKIP_REASON = "Requires running ArcadeDB with seeded corpus"


class TestBasicStrategyPerformance:
    """Latency benchmarks for the basic (text-only vector) retrieval strategy."""

    def test_basic_text_search_latency(self):
        pytest.skip(_SKIP_REASON)

    def test_basic_text_search_recall_at_k(self):
        pytest.skip(_SKIP_REASON)


class TestHybridStrategyPerformance:
    """Latency benchmarks for the hybrid (text + image + expansion) strategy."""

    def test_hybrid_search_latency(self):
        pytest.skip(_SKIP_REASON)

    def test_hybrid_expansion_overhead(self):
        pytest.skip(_SKIP_REASON)


class TestGlobalStrategyPerformance:
    """Latency benchmarks for the global (community report synthesis) strategy."""

    def test_global_community_search_latency(self):
        pytest.skip(_SKIP_REASON)

    def test_global_synthesis_latency(self):
        pytest.skip(_SKIP_REASON)
