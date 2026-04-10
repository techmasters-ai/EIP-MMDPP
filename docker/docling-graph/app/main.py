"""Docling-Graph extraction service -- thin wrapper around run_pipeline()."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import networkx as nx
import yaml
from fastapi import FastAPI, HTTPException, Request

# ---------------------------------------------------------------------------
# Monkey-patch docling-graph's LiteLLMClient to fix two defects:
#
# 1. _build_request() runs a support-filter that strips Ollama-native params
#    (format, think) before the request reaches litellm.completion().
#    Fix: preserve these params through the filter for Ollama providers.
#
# 2. _call_api() only reads message.content — empty for thinking models like
#    gpt-oss:120b where reasoning goes to message.thinking and content can
#    be empty.  Fix: richer error with diagnostic fields, no silent failure.
#
# 3. For Ollama: send format=<schema> (structured) or format="json" (fallback),
#    think="low" for gpt-oss, stream=False for reliable structured output.
# ---------------------------------------------------------------------------
import litellm as _litellm

_logger = logging.getLogger("docling_graph.llm_clients.litellm.patch")


def _patched_build_request(
    self,
    messages,
    schema_json=None,
    structured_output=True,
    response_top_level="object",
    response_schema_name="extraction_result",
):
    from docling_graph.llm_clients.schema_utils import normalize_schema_for_response_format
    from docling_graph.exceptions import ClientError

    gen = self.generation
    max_tokens = gen.max_tokens or self._max_output_tokens
    model_name = self.model_config.litellm_model

    request = {
        "model": model_name,
        "messages": messages,
        "temperature": gen.temperature,
        "max_tokens": max_tokens,
        "timeout": self.timeout,
        "drop_params": True,
        "stream": False,
    }

    connection = self.connection
    api_key = connection.api_key.get_secret_value() if connection.api_key else None
    if api_key:
        request["api_key"] = api_key
    if connection.base_url:
        request["api_base"] = connection.base_url
    if connection.organization:
        request["organization"] = connection.organization
    if connection.headers:
        request["headers"] = dict(connection.headers)

    provider_id = str(getattr(self._config, "provider_id", "") or "").lower()
    is_ollama = (
        provider_id == "ollama"
        or str(model_name).startswith("ollama/")
        or ("11434" in str(request.get("api_base", "")))
    )

    if structured_output:
        try:
            schema_dict = json.loads(schema_json or "{}")
        except json.JSONDecodeError as e:
            raise ClientError(
                "Invalid schema_json passed for structured output.",
                details={"error": str(e), "schema_json_preview": (schema_json or "")[:200]},
                cause=e,
            ) from e
        normalized = normalize_schema_for_response_format(
            schema_dict,
            top_level=response_top_level,
            name=response_schema_name,
        )
        # OpenAI-style response_format uses the normalized envelope
        request["response_format"] = {"type": "json_schema", "json_schema": normalized}
        if is_ollama:
            # Ollama format= wants the RAW JSON Schema, not the OpenAI envelope.
            # Guard: if schema is very large (>50 properties), use simple json mode
            # to avoid degenerate constrained decoding with huge schemas.
            schema_str = json.dumps(schema_dict)
            if len(schema_str) > 8000:
                _logger.info("Schema too large for Ollama format= (%d chars), using format='json'", len(schema_str))
                request["format"] = "json"
            else:
                request["format"] = schema_dict
    else:
        request["response_format"] = {"type": "json_object"}
        if is_ollama:
            request["format"] = "json"

    if gen.top_p is not None:
        request["top_p"] = gen.top_p
    if gen.top_k is not None:
        request["top_k"] = gen.top_k
    if gen.frequency_penalty is not None:
        request["frequency_penalty"] = gen.frequency_penalty
    if gen.presence_penalty is not None:
        request["presence_penalty"] = gen.presence_penalty
    if gen.seed is not None:
        request["seed"] = gen.seed
    if gen.stop is not None:
        request["stop"] = gen.stop

    if is_ollama and "gpt-oss" in str(model_name).lower():
        request["think"] = "low"

    supported_fn = getattr(_litellm, "get_supported_openai_params", None)
    if callable(supported_fn):
        try:
            supported = supported_fn(model=model_name)
            if supported:
                required = {
                    "model", "messages", "api_base", "api_key", "headers",
                    "organization", "timeout", "drop_params", "response_format",
                    "stream",
                }
                provider_required = set()
                if is_ollama:
                    provider_required.update({"format", "think"})
                allowed = required | provider_required | set(supported)
                request = {k: v for k, v in request.items() if k in allowed}
        except Exception:
            _logger.debug("LiteLLM supported params lookup failed for %s", model_name)

    return request


def _patched_call_api(self, messages, **params):
    from docling_graph.exceptions import ClientError

    try:
        request = self._build_request(messages, **params)
        response = _litellm.completion(**request)

        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        elif isinstance(response, dict):
            response_dict = response
        else:
            try:
                response_dict = dict(response)
            except Exception:
                response_dict = {"raw": str(response)}

        choices = response_dict.get("choices", []) or []
        if not choices:
            raise ClientError("LiteLLM returned no choices", details={"model": self.model})

        message = choices[0].get("message", {}) or {}
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        thinking = message.get("thinking")
        top_reasoning = response_dict.get("reasoning_content")
        top_thinking = response_dict.get("thinking")

        metadata = {
            "finish_reason": choices[0].get("finish_reason"),
            "model": response_dict.get("model", self.model),
            "usage": response_dict.get("usage"),
            "has_content": bool(content),
            "has_reasoning_content": bool(reasoning_content or top_reasoning),
            "has_thinking": bool(thinking or top_thinking),
        }

        if content:
            return str(content), metadata

        raise ClientError(
            "LiteLLM returned empty content",
            details={
                "model": self.model,
                "finish_reason": choices[0].get("finish_reason"),
                "message_keys": sorted(message.keys()),
                "message_preview": str(message)[:1000],
                "has_reasoning_content": bool(reasoning_content or top_reasoning),
                "has_thinking": bool(thinking or top_thinking),
                "reasoning_preview": str(
                    reasoning_content or top_reasoning or thinking or top_thinking or ""
                )[:500],
                "request_keys": sorted(list(request.keys())),
            },
        )
    except Exception as e:
        if isinstance(e, ClientError):
            raise
        if params.get("structured_output", True):
            self.last_call_diagnostics.update({
                "structured_failed": True,
                "fallback_error_class": type(e).__name__,
            })
            raise ClientError(
                "Structured output request failed.",
                details={
                    "model": self.model,
                    "provider": self._config.provider_id,
                    "error": str(e),
                },
                cause=e,
            ) from e
        raise ClientError(
            f"LiteLLM API call failed: {type(e).__name__}",
            details={"model": self.model, "error": str(e)},
            cause=e,
        ) from e


def _apply_litellm_client_patches():
    """Apply patches to LiteLLMClient after docling_graph is imported."""
    try:
        from docling_graph.llm_clients.litellm import LiteLLMClient
        LiteLLMClient._build_request = _patched_build_request
        LiteLLMClient._call_api = _patched_call_api
        _logger.info("LiteLLMClient patched for Ollama structured output support")
    except ImportError:
        _logger.warning("Could not patch LiteLLMClient — docling_graph.llm_clients.litellm not available")

    # Fix TABLE_REF node ID collision: split("_")[0] yields "TABLE" for
    # "TABLE_REF_<fingerprint>", causing false collision.  Use rsplit.
    try:
        from docling_graph.core.converters.node_id_registry import NodeIDRegistry

        _original_get_node_id = NodeIDRegistry.get_node_id

        def _patched_get_node_id(self, model_instance, auto_register=True):
            fingerprint = self._generate_fingerprint(model_instance)
            class_name = model_instance.__class__.__name__

            if fingerprint in self.fingerprint_to_id:
                existing_id = self.fingerprint_to_id[fingerprint]
                existing_class = existing_id.rsplit("_", 1)[0] if "_" in existing_id else existing_id
                if existing_class != class_name:
                    raise ValueError(
                        f"Node ID collision: fingerprint {fingerprint} maps to both "
                        f"{existing_id} (class: {existing_class}) and {class_name}_... (new class)"
                    )
                return existing_id

            if class_name not in self.seen_classes:
                self.seen_classes[class_name] = set()

            node_id = f"{class_name}_{fingerprint}"

            if auto_register:
                self.fingerprint_to_id[fingerprint] = node_id
                self.id_to_fingerprint[node_id] = fingerprint
                self.seen_classes[class_name].add(fingerprint)

            return node_id

        NodeIDRegistry.get_node_id = _patched_get_node_id
        _logger.info("NodeIDRegistry patched for underscore class name collision fix")
    except ImportError:
        _logger.warning("Could not patch NodeIDRegistry")


_apply_litellm_client_patches()

from app.config_builder import build_pipeline_config
from app.schemas import (
    ExtractionMetadata,
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
)
from app.template_builder import build_templates_with_edges

logger = logging.getLogger(__name__)

ONTOLOGY_PATH = os.environ.get("ONTOLOGY_PATH", "/ontology/ontology.yaml")
MAX_CONCURRENT = int(os.environ.get("DOCLING_GRAPH_MAX_CONCURRENT_EXTRACTIONS", "2"))


def _validate_library_surface() -> str:
    """Validate docling-graph library API. Returns version string."""
    try:
        from docling_graph import run_pipeline, PipelineConfig  # noqa: F401
        import docling_graph
        return getattr(docling_graph, "__version__", "unknown")
    except ImportError as e:
        logger.warning("docling-graph library not fully available: %s", e)
        return "unavailable"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline_version = _validate_library_surface()
    logger.info("docling-graph library version: %s", app.state.pipeline_version)

    app.state.templates = {}
    app.state.ontology_version = None
    app.state.ontology_cache = {}

    if os.path.exists(ONTOLOGY_PATH):
        with open(ONTOLOGY_PATH) as f:
            ontology = yaml.safe_load(f)
        app.state.ontology_version = ontology.get("version")
        app.state.templates = build_templates_with_edges(ontology)
        # Build a unified template for PipelineConfig.template (single model)
        from app.template_builder import build_unified_template
        app.state.unified_template = build_unified_template(ontology)
        logger.info("Loaded ontology v%s (%d templates, unified=%s)",
                     app.state.ontology_version, len(app.state.templates),
                     app.state.unified_template.__name__ if app.state.unified_template else "None")
    else:
        logger.warning("Ontology not found at %s", ONTOLOGY_PATH)

    app.state.extraction_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    yield
    logger.info("Shutting down")


app = FastAPI(title="Docling-Graph Extraction Service", version="2.0.0", lifespan=lifespan)


def _resolve_templates(request: Request, ontology_definition: dict[str, Any] | None) -> dict[str, Any]:
    if ontology_definition is None:
        return request.app.state.templates
    ont_hash = hashlib.sha256(json.dumps(ontology_definition, sort_keys=True).encode()).hexdigest()[:16]
    cache = request.app.state.ontology_cache
    if ont_hash in cache:
        return cache[ont_hash]
    templates = build_templates_with_edges(ontology_definition)
    cache[ont_hash] = templates
    if len(cache) > 2:
        del cache[next(iter(cache))]
    return templates


def run_extraction_pipeline(
    docling_document_json: dict[str, Any],
    templates: dict[str, Any],
    unified_template: type | None = None,
) -> Any:
    """Run docling-graph pipeline synchronously (called via asyncio.to_thread).

    Uses the unified_template (single Pydantic model with all entity types)
    as PipelineConfig.template so the library extracts all entity types in
    one pass, conforming to its canonical API.
    """
    import tempfile
    from docling_graph import run_pipeline

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(docling_document_json, tmp, ensure_ascii=False, default=str)
        tmp_path = tmp.name

    try:
        template_cls = unified_template or (
            next(iter(templates.values())) if templates else None
        )
        config = build_pipeline_config(source=tmp_path, template_class=template_cls)
        return run_pipeline(config)
    finally:
        os.unlink(tmp_path)


def _should_run_validation_pass() -> bool:
    val = os.environ.get("DOCLING_GRAPH_VALIDATION_PASS_ENABLED", "true")
    return val.lower() in ("true", "1", "yes")


def _apply_validation_edges(graph: nx.DiGraph, new_edges: list[dict[str, Any]]) -> int:
    added = 0
    for edge in new_edges:
        source, target = edge.get("source"), edge.get("target")
        if source and target and graph.has_node(source) and graph.has_node(target) and not graph.has_edge(source, target):
            graph.add_edge(source, target, label=edge.get("label", "RELATED_TO"),
                          confidence=edge.get("confidence", 0.5), _source="validation_pass")
            added += 1
    return added


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    return HealthResponse(
        ontology_version=request.app.state.ontology_version,
        template_count=len(request.app.state.templates),
        extraction_contract=os.environ.get("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
        pipeline_version=request.app.state.pipeline_version,
    )


@app.post("/extract-all", response_model=ExtractionResponse)
async def extract_all(request: Request, body: ExtractionRequest):
    semaphore = request.app.state.extraction_semaphore
    if semaphore is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    templates = _resolve_templates(request, body.ontology_definition)
    if not templates:
        raise HTTPException(status_code=422, detail="No templates available")

    if body.ontology_version and body.ontology_version != request.app.state.ontology_version:
        logger.warning("Ontology version mismatch: request=%s server=%s", body.ontology_version, request.app.state.ontology_version)

    async with semaphore:
        try:
            # Use per-request ontology if provided, otherwise app-level default
            if body.ontology_definition is not None:
                from app.template_builder import build_unified_template
                unified = build_unified_template(body.ontology_definition)
            else:
                unified = getattr(request.app.state, "unified_template", None)
            context = await asyncio.to_thread(
                run_extraction_pipeline, body.docling_document_json, templates, unified,
            )
        except Exception as exc:
            logger.exception("Pipeline failed for %s", body.document_id)
            raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    graph = context.knowledge_graph
    graph_data = nx.node_link_data(graph, edges="links")

    meta = context.graph_metadata
    metadata = ExtractionMetadata(
        node_count=getattr(meta, "node_count", graph.number_of_nodes()),
        edge_count=getattr(meta, "edge_count", graph.number_of_edges()),
        node_types=getattr(meta, "node_types", {}),
        edge_types=getattr(meta, "edge_types", {}),
        extraction_contract=os.environ.get("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
    )

    return ExtractionResponse(
        graph=graph_data,
        metadata=metadata,
        model=os.environ.get("DOCLING_GRAPH_LLM_MODEL", "granite3-dense:8b"),
        provider=os.environ.get("DOCLING_GRAPH_LLM_PROVIDER", "ollama"),
        ontology_version=request.app.state.ontology_version,
    )
