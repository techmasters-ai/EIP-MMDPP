"""Process-cached docling-graph LLM client factory.

Mirrors app/services/ollama_clients.py but lives in the docling-graph
container. Concurrent extraction passes share one OllamaPool + one
OllamaChatClient per process so in-flight counters are accurate and
GET /debug/routing-metrics reports real fan-out.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache

from app.config_builder import DoclingGraphSettings
from app.ollama_pool_client import OllamaChatClient, OllamaPool


@lru_cache(maxsize=1)
def get_docling_graph_client() -> OllamaChatClient:
    """Return the process-cached OllamaChatClient for docling-graph extraction.

    First call constructs the pool from DoclingGraphSettings + service-level
    settings (force_json_mode, structured_output_threshold_chars). Subsequent
    calls return the same instance. lru_cache.cache_clear() is exposed so
    tests can rebuild against patched env.

    Limitations:
      The following values are read on first call and frozen for the
      lifetime of the process — env-var changes after import won't take
      effect without an explicit `get_docling_graph_client.cache_clear()`
      (or a service restart):
        - force_json_mode
        - structured_output_threshold_chars
        - DOCLING_GRAPH_LLM_THINK
        - every field on DoclingGraphSettings (model, timeout,
          temperature, max_tokens, top_p / top_k / seed / stop / etc.,
          and the URL pool from get_ollama_llm_urls())

      Tests rebuild via `get_docling_graph_client.cache_clear()` after
      patching env. Operators rotating Ollama endpoints in production
      must restart the docling-graph service to pick up new URLs.
    """
    from app.config import settings as _service_settings
    from app.prompt_rules import sanitize_schema_for_llm
    from docling_graph.exceptions import ClientError
    from app.llm_json import parse_llm_json_loose

    settings = DoclingGraphSettings()
    pool = OllamaPool(urls=settings.get_ollama_llm_urls())
    max_in_flight = max(0, int(settings.docling_graph_llm_max_in_flight or 0))
    request_semaphore = (
        threading.BoundedSemaphore(max_in_flight)
        if max_in_flight > 0
        else None
    )
    if max_in_flight > 0:
        # One explicit startup line makes it easy to verify that high
        # per-pass parallelism is bounded by the fixed Ollama slot budget.
        import logging
        logging.getLogger(__name__).info(
            "docling-graph LLM in-flight limiter enabled capacity=%d",
            max_in_flight,
        )

    default_extra_params = {
        "top_p": getattr(settings, "docling_graph_llm_top_p", None),
        "top_k": getattr(settings, "docling_graph_llm_top_k", None),
        "frequency_penalty": getattr(settings, "docling_graph_llm_frequency_penalty", None),
        "presence_penalty": getattr(settings, "docling_graph_llm_presence_penalty", None),
        "seed": getattr(settings, "docling_graph_llm_seed", None),
        "stop": getattr(settings, "docling_graph_llm_stop", None),
    }

    return OllamaChatClient(
        pool=pool,
        model=settings.docling_graph_llm_model,
        timeout_s=float(settings.docling_graph_llm_timeout),
        temperature=settings.docling_graph_llm_temperature,
        max_tokens=settings.docling_graph_llm_max_tokens,
        think=os.environ.get("DOCLING_GRAPH_LLM_THINK", "") or None,
        schema_transform=sanitize_schema_for_llm,
        force_json_mode=_service_settings.force_json_mode,
        structured_output_threshold_chars=_service_settings.structured_output_threshold_chars,
        default_extra_params=default_extra_params,
        truncation_retry_max_tokens=(
            settings.docling_graph_llm_truncation_retry_max_tokens
        ),
        client_error_cls=ClientError,
        parse_json_fn=parse_llm_json_loose,
        request_semaphore=request_semaphore,
        request_semaphore_capacity=max_in_flight or None,
    )
