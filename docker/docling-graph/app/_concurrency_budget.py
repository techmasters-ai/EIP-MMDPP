"""C1 concurrency-budget helper (walltime-reduction Phase 1, warn-only).

The docling-graph service has three concurrency knobs:

  * ``DOCLING_GRAPH_MAX_CONCURRENT_EXTRACTIONS`` (default 2) — asyncio
    semaphore in ``main.py``; caps concurrent ``/extract-pass`` requests
    handled inside the service.
  * ``DOCLING_GRAPH_PARALLEL_WORKERS`` (default 4) — per-extract-pass
    chunk-batch fan-out to Ollama (docling-graph library ThreadPool).
  * ``DOCLING_GRAPH_LLM_MAX_IN_FLIGHT`` (default 0 = disabled) — process-
    wide BoundedSemaphore in ``ollama_clients.py``; caps outbound chat
    generations regardless of the above two.

Effective per-document in-flight LLM calls inside this service =
``MAX_CONCURRENT_EXTRACTIONS × PARALLEL_WORKERS`` (default 8).

When the LLM backend has a fixed slot budget (e.g. Ollama's
``OLLAMA_NUM_PARALLEL`` per host × number of hosts), oversubscribing
inflates request-queue depth — every call's tail latency grows without
adding throughput. This helper computes the effective in-flight and
emits a startup line so an operator can see at a glance whether the
backend slot count and the docling-graph config agree.

C1 ships warn-only: no behavior change beyond a single log line at
service startup. Enforcement at the LLM-call boundary already exists
(``ollama_clients.py:50-55``); this helper is purely observational.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_effective_in_flight(
    *,
    max_concurrent_extractions: int,
    parallel_workers: int,
) -> int:
    """Effective per-document in-flight LLM calls in the docling-graph
    service. Multi-document concurrent ingest multiplies this by the
    worker's PASS_CONCURRENCY_PER_DOCUMENT."""
    return int(max_concurrent_extractions) * int(parallel_workers)


def emit_concurrency_budget_warning(
    *,
    max_concurrent_extractions: int,
    parallel_workers: int,
    llm_max_in_flight: int,
) -> None:
    """Log one line at startup describing the concurrency budget.

    WARNING when either:
      (a) the LLM cap is unset (0) AND effective in-flight is >1 (likely
          oversubscribe vs a typical single-slot Ollama host), or
      (b) the LLM cap is set but below the effective in-flight (workers
          will block on the semaphore instead of doing work).

    INFO when the cap is set and >= the effective in-flight (healthy).
    """
    effective = compute_effective_in_flight(
        max_concurrent_extractions=max_concurrent_extractions,
        parallel_workers=parallel_workers,
    )
    formula = (
        "effective_in_flight = DOCLING_GRAPH_MAX_CONCURRENT_EXTRACTIONS "
        "* DOCLING_GRAPH_PARALLEL_WORKERS"
    )

    if llm_max_in_flight <= 0:
        if effective > 1:
            logger.warning(
                "concurrency-budget: %s = %d * %d = %d, "
                "DOCLING_GRAPH_LLM_MAX_IN_FLIGHT is unset (0). With no cap, "
                "the service may oversubscribe the Ollama backend and "
                "inflate tail latency. Set DOCLING_GRAPH_LLM_MAX_IN_FLIGHT "
                "to the real backend slot count "
                "(e.g. OLLAMA_NUM_PARALLEL * number_of_hosts).",
                formula, max_concurrent_extractions, parallel_workers,
                effective,
            )
        else:
            logger.info(
                "concurrency-budget: effective_in_flight=%d, no cap "
                "configured (effective_in_flight is low enough that this "
                "is fine).",
                effective,
            )
        return

    if llm_max_in_flight < effective:
        logger.warning(
            "concurrency-budget: %s = %d * %d = %d, but "
            "DOCLING_GRAPH_LLM_MAX_IN_FLIGHT=%d caps below effective. "
            "Workers will block on the in-flight semaphore. Either raise "
            "the cap (if backend has spare slots) or lower "
            "MAX_CONCURRENT_EXTRACTIONS / PARALLEL_WORKERS so they don't "
            "compete.",
            formula, max_concurrent_extractions, parallel_workers,
            effective, llm_max_in_flight,
        )
        return

    logger.info(
        "concurrency-budget: effective_in_flight=%d, "
        "DOCLING_GRAPH_LLM_MAX_IN_FLIGHT=%d (cap meets or exceeds "
        "effective — healthy).",
        effective, llm_max_in_flight,
    )
