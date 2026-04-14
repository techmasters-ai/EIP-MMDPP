"""Docling-Graph extraction service -- thin wrapper around run_pipeline()."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import networkx as nx
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
            from app.config import settings as _service_settings
            schema_str = json.dumps(schema_dict)
            threshold = _service_settings.structured_output_threshold_chars
            if len(schema_str) > threshold:
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

from app.bundles import load_bundle_manifest, load_pass_template, preload_all_templates
from app.config_builder import build_pipeline_config
from app.schemas import (
    ExtractionMetadata,
    HealthResponse,
    ExtractPassRequest,
    ExtractPassResponse,
    EntityRef,
)

logger = logging.getLogger(__name__)

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
    # Pre-import every bundle's extraction schemas so per-request dispatch via
    # load_pass_template is constant-time.
    try:
        preload_all_templates()
        logger.info("Preloaded all bundle extraction schemas")
    except Exception as exc:
        logger.warning("preload_all_templates failed: %s", exc)

    app.state.pipeline_version = _validate_library_surface()
    logger.info("docling-graph library version: %s", app.state.pipeline_version)

    app.state.extraction_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    yield
    logger.info("Shutting down")


app = FastAPI(title="Docling-Graph Extraction Service", version="2.0.0", lifespan=lifespan)


def run_extraction_pass(
    docling_document_json: dict[str, Any],
    template_cls: type,
    upstream_entities: list | None = None,
) -> Any:
    """Run docling-graph pipeline for a SINGLE fixed-template pass.

    Mirrors the deleted run_extraction_pipeline() exactly but takes the template
    class directly from the bundle loader instead of resolving it from a dynamic
    definition blob. upstream_entities is accepted but is NOT threaded
    into docling_graph.run_pipeline in PR 1 — the integration of upstream
    refs into the service prompt preamble is handled by PR 2 alongside the
    worker-side refactor. For now, upstream_entities is logged and passed
    through as metadata only.
    """
    import tempfile
    from docling_graph import run_pipeline

    if upstream_entities:
        logger.info(
            "extract-pass: received %d upstream entity refs (not yet threaded into prompt in PR 1)",
            len(upstream_entities),
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(docling_document_json, tmp, ensure_ascii=False, default=str)
        tmp_path = tmp.name

    try:
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
    from app.bundles import _template_cache  # noqa: PLC0415 — local import to avoid circular
    schema_count = len(_template_cache)
    return HealthResponse(
        schema_count=schema_count,
        extraction_contract=os.environ.get("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
        pipeline_version=request.app.state.pipeline_version,
    )


@app.post("/extract-pass", response_model=ExtractPassResponse)
async def extract_pass(request: Request, body: ExtractPassRequest):
    """Fixed-template extraction for ONE pass from a bundle. Spec §5.9."""
    semaphore = request.app.state.extraction_semaphore
    if semaphore is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # 1. Resolve bundle + pass
    try:
        manifest = load_bundle_manifest(body.bundle_key)
    except (KeyError, FileNotFoundError):
        raise HTTPException(status_code=404, detail=f"Unknown bundle_key: {body.bundle_key}")

    pass_def = None
    for p in manifest.get("passes", []):
        if p.get("name") == body.pass_name:
            pass_def = p
            break
    if pass_def is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown pass_name: {body.pass_name} in bundle {body.bundle_key}",
        )

    # 2. Validate input_mode compatibility
    input_mode = pass_def.get("input_mode")
    if input_mode == "document_only" and body.upstream_entities:
        raise HTTPException(
            status_code=400,
            detail=f"document_only pass {body.pass_name} received upstream_entities",
        )
    if input_mode == "document_plus_entity_refs" and not body.upstream_entities:
        raise HTTPException(
            status_code=400,
            detail=f"document_plus_entity_refs pass {body.pass_name} missing upstream_entities",
        )

    # 3. Load the pre-imported fixed template
    try:
        template_cls = load_pass_template(body.bundle_key, body.pass_name)
    except (ImportError, AttributeError, KeyError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load template for {body.bundle_key}/{body.pass_name}: {exc}",
        )

    # 4. Run the pipeline
    #
    # The docling-graph library emits validation-salvage warnings
    # (e.g. `specifications -> N -> value: Input should be a valid string`)
    # without naming the pass. Bracket the call with bundle/pass context
    # so operators can correlate those warnings to the offending pass.
    upstream_ref_count = len(body.upstream_entities) if body.upstream_entities else 0
    logger.info(
        "extract-pass: START bundle=%s pass=%s input_mode=%s document_id=%s upstream_ref_count=%d",
        body.bundle_key, body.pass_name, pass_def.get("input_mode"),
        body.document_id, upstream_ref_count,
    )
    async with semaphore:
        try:
            context = await asyncio.to_thread(
                run_extraction_pass,
                body.docling_document_json,
                template_cls,
                body.upstream_entities,
            )
        except Exception as exc:
            logger.exception(
                "extract-pass pipeline failed for document_id=%s bundle=%s pass=%s",
                body.document_id, body.bundle_key, body.pass_name,
            )
            raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")
    # 5. Build response — mirror the /extract-all metadata shape
    graph = context.knowledge_graph
    meta = context.graph_metadata
    metadata = ExtractionMetadata(
        node_count=getattr(meta, "node_count", graph.number_of_nodes()),
        edge_count=getattr(meta, "edge_count", graph.number_of_edges()),
        node_types=getattr(meta, "node_types", {}),
        edge_types=getattr(meta, "edge_types", {}),
        extraction_contract=os.environ.get("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
        # --- Plan 1 — appended below. -----------------------------------
        upstream_ref_count=upstream_ref_count,
        upstream_preamble_applied=False,  # flipped to True in Task 5b
    )
    logger.info(
        "extract-pass: END bundle=%s pass=%s document_id=%s node_count=%d edge_count=%d",
        body.bundle_key, body.pass_name, body.document_id,
        metadata.node_count, metadata.edge_count,
    )

    # pass_output is the dumped template instance so the worker can re-parse it
    pass_output: dict[str, Any] = {}
    template_instance = getattr(context, "template_instance", None)
    if template_instance is not None and hasattr(template_instance, "model_dump"):
        pass_output = template_instance.model_dump(mode="json")
    else:
        # Fallback: serialize the graph as the pass_output until Chunk 3
        # wires template_instance through the docling-graph pipeline
        import networkx as nx
        pass_output = {"graph": nx.node_link_data(graph, edges="links")}

    return ExtractPassResponse(
        bundle_key=body.bundle_key,
        pass_name=body.pass_name,
        pass_output=pass_output,
        metadata=metadata,
        model=os.environ.get("DOCLING_GRAPH_LLM_MODEL", "granite3-dense:8b"),
        provider=os.environ.get("DOCLING_GRAPH_LLM_PROVIDER", "ollama"),
    )
