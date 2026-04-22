"""Docling-Graph extraction service -- thin wrapper around run_pipeline()."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

try:
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - test-host fallback only
    class _FallbackDiGraph:
        def number_of_nodes(self) -> int:
            return 0

        def number_of_edges(self) -> int:
            return 0

        def has_node(self, _node: Any) -> bool:
            return False

        def has_edge(self, _source: Any, _target: Any) -> bool:
            return False

        def add_edge(self, _source: Any, _target: Any, **_kwargs: Any) -> None:
            return None

    class _FallbackNetworkX:
        DiGraph = _FallbackDiGraph

        @staticmethod
        def node_link_data(_graph: Any, edges: str = "links") -> dict[str, Any]:
            return {"nodes": [], edges: []}

    nx = _FallbackNetworkX()

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

# Delta system-prompt rewrite. Replaces Rules 2-4 in get_delta_batch_prompt
# so the LLM accepts BOTH structure-backed evidence AND explicit named
# mentions in prose, while still rejecting section titles + unnamed
# descriptions. See docker/docling-graph/app/prompt_rules.py and
# ontology_bundles/_shared/prompt_rules.py for the rule text and rationale.
# Section-title slippage is handled by the library's own post-extraction
# filter_entity_nodes_by_identity (delta_identity_filter_enabled=True).
from app.prompt_rules import install as _install_prompt_rules
from app.resolver_patch import install as _install_resolver_patch

_install_prompt_rules()
_install_resolver_patch()

from app.bundles import load_bundle_manifest, load_pass_template, preload_all_templates
from app.config_builder import build_pipeline_config
from app.evidence_gate import (
    collect_batch_evidence_text as _collect_batch_evidence_text,
    filter_pass_output_by_batch_text as _filter_pass_output_by_batch_text,
    filter_provenance_rows_by_allowed_identities as _filter_provenance_rows_by_allowed_identities,
)
from app.provenance import build_provenance_from_context
from app.schemas import (
    ExtractionMetadata,
    ExtractionProvenance,
    HealthResponse,
    ExtractPassRequest,
    ExtractPassResponse,
    EntityRef,  # for typing only; runtime uses attribute access
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


def _render_upstream_entities_preamble(upstream_entities: list | None) -> str:
    """Render a plain-text preamble listing upstream entity refs for the LLM.

    Returns an empty string when:
    - upstream_entities is None or empty
    - DOCLING_GRAPH_UPSTREAM_PREAMBLE env var is "false" / "0" / "no"

    The returned string is designed to be prepended to the document body so all
    three docling-graph extraction contracts (direct, delta, staged) see it.
    """
    flag = os.environ.get("DOCLING_GRAPH_UPSTREAM_PREAMBLE", "true")
    if flag.lower() in ("false", "0", "no"):
        return ""

    if not upstream_entities:
        return ""

    lines = ["Upstream entities:"]
    for entity in upstream_entities:
        ref_id = getattr(entity, "ref_id", None) or entity.get("ref_id", "")
        entity_type = getattr(entity, "entity_type", None) or entity.get("entity_type", "")
        display_label = getattr(entity, "display_label", None)
        if display_label is None and isinstance(entity, dict):
            display_label = entity.get("display_label")

        if display_label:
            lines.append(f"  [{ref_id}] {entity_type} \u2014 {display_label}")
        else:
            lines.append(f"  [{ref_id}] {entity_type}")

    lines.append(
        "Only emit from_ref_id and to_ref_id values from the list above "
        "when referencing these upstream entities."
    )
    return "\n".join(lines)


def run_extraction_pass(
    docling_document_json: dict[str, Any],
    template_cls: type,
    upstream_entities: list | None = None,
    pass_name: str | None = None,
) -> Any:
    """Run docling-graph pipeline for a SINGLE fixed-template pass.

    Mirrors the deleted run_extraction_pipeline() exactly but takes the template
    class directly from the bundle loader instead of resolving it from a dynamic
    definition blob.

    ``pass_name`` is threaded through to :func:`build_pipeline_config` so
    per-pass quality-gate overrides (spec §4.6) apply — e.g. system_links
    drops min_instances to 1.

    Path B preamble injection: if DOCLING_GRAPH_UPSTREAM_PREAMBLE is enabled and
    upstream_entities is non-empty, the preamble is appended to the document's
    texts array and prepended to body.children so that export_to_markdown()
    includes it at the top of the document body for all three extraction contracts.
    """
    import shutil
    import tempfile
    from docling_graph import run_pipeline

    # --- Empty-source short-circuit --------------------------------------
    # If the DoclingDocument has no body content AND no registered items
    # (texts / pictures / tables), the library's LlmBackend fails fast with
    # "Markdown is empty for DoclingDocument. Cannot proceed." and bubbles a
    # 500. That thrashes the worker's retry loop for true-negative cases
    # (image-only inputs Docling didn't decompose — see the Docling service's
    # synthetic-picture fallback). Return a clean empty response instead.
    def _is_empty(doc: dict) -> bool:
        body_children = (doc.get("body") or {}).get("children") or []
        return (
            not body_children
            and not doc.get("texts")
            and not doc.get("pictures")
            and not doc.get("tables")
        )

    if _is_empty(docling_document_json):
        logger.warning(
            "extract-pass short-circuit: DoclingDocument has no extractable content "
            "(body.children/texts/pictures/tables all empty). Returning empty pass_output "
            "with diagnostic marker instead of calling run_pipeline."
        )
        import networkx as _nx

        class _EmptySourceContext:
            template_instance = None
            extracted_models: list = []
            knowledge_graph = _nx.DiGraph()

            class _Meta:
                node_count = 0
                edge_count = 0
                node_types: dict = {}
                edge_types: dict = {}

            graph_metadata = _Meta()

        ctx = _EmptySourceContext()
        ctx._upstream_preamble_applied = False  # no chance to apply — source was empty
        ctx._chunk_to_self_refs = None  # no doc, nothing to map
        ctx._delta_trace = {
            "empty_source": True,
            "reason": "docling_document_has_no_extractable_content",
            "body_children": 0,
            "texts": 0,
            "pictures": 0,
            "tables": 0,
            "library_log": "",
            "suggestion": (
                "Upstream Docling conversion produced an empty body. Image-only "
                "inputs that the PDF pipeline didn't decompose are the typical "
                "cause; see docker/docling/app/converter.py synthetic-picture "
                "fallback and TODO #29 for VLM routing."
            ),
        }
        return ctx
    # ----------------------------------------------------------------------

    # --- Path B injection -------------------------------------------------
    preamble = _render_upstream_entities_preamble(upstream_entities)
    preamble_applied = False

    if preamble:
        new_index = len(docling_document_json.get("texts", []))
        preamble_item = {
            "self_ref": f"#/texts/{new_index}",
            "parent": {"$ref": "#/body"},
            "children": [],
            "content_layer": "body",
            "label": "text",
            "prov": [],
            "orig": preamble,
            "text": preamble,
        }
        # Append to texts (preserves existing indices so RefItem lookups don't break)
        new_texts = list(docling_document_json.get("texts", [])) + [preamble_item]
        # Prepend a body-child ref so the preamble appears first in markdown
        existing_body = docling_document_json.get("body", {})
        new_body = dict(existing_body)
        new_body["children"] = [
            {"$ref": f"#/texts/{new_index}"}
        ] + list(new_body.get("children", []))
        # Copy-on-write: never mutate the caller's dict
        docling_document_json = dict(docling_document_json)
        docling_document_json["texts"] = new_texts
        docling_document_json["body"] = new_body
        preamble_applied = True
    # ----------------------------------------------------------------------

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(docling_document_json, tmp, ensure_ascii=False, default=str)
        tmp_path = tmp.name

    # Per-call debug dir; read back delta_trace.json afterwards and stash
    # on context so extract_pass can surface it as response.diagnostics.
    debug_dir = tempfile.mkdtemp(prefix="docgraph-debug-")

    try:
        config = build_pipeline_config(
            source=tmp_path,
            template_class=template_cls,
            pass_name=pass_name,
            debug_dir=debug_dir,
        )

        # Capture the library's print() + logging output to stdout/stderr during
        # run_pipeline so the response can surface the exact fallback chain
        # ([DeltaExtraction] → "Warning: ...produced no JSON" → "falling back
        # to direct" → "LiteLLMClient returned empty or all-null JSON" → etc.)
        # to the notebook and worker. A Tee keeps the original streams flowing
        # to the container logs so uvicorn's log aggregation doesn't go dark.
        library_log_buf = io.StringIO()

        class _Tee:
            def __init__(self, *streams):
                self._streams = streams
            def write(self, data):
                for s in self._streams:
                    try:
                        s.write(data)
                    except Exception:
                        pass
                return len(data) if isinstance(data, str) else 0
            def flush(self):
                for s in self._streams:
                    try:
                        s.flush()
                    except Exception:
                        pass

        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(original_stdout, library_log_buf)
        sys.stderr = _Tee(original_stderr, library_log_buf)
        pipeline_error: Exception | None = None
        try:
            context = run_pipeline(config)
        except Exception as exc:
            # Library raised PipelineError (or anything else). Build a stub
            # context so the service returns a clean 200 with diagnostics
            # instead of a 500 that tanks the worker's retry loop. The
            # notebook + worker see why the library bailed via the captured
            # log + pipeline_error marker in _delta_trace.
            pipeline_error = exc
            logger.warning(
                "run_pipeline raised %s: %s — returning stub context with "
                "diagnostic marker so the pass doesn't hard-fail.",
                type(exc).__name__, exc,
            )
            import networkx as _nx

            class _PipelineFailureContext:
                template_instance = None
                extracted_models: list = []
                knowledge_graph = _nx.DiGraph()
                docling_document = None

                class _Meta:
                    node_count = 0
                    edge_count = 0
                    node_types: dict = {}
                    edge_types: dict = {}

                graph_metadata = _Meta()

            context = _PipelineFailureContext()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        # docling-graph's stages don't set ``context.template_instance``
        # — they populate ``extracted_models``. Promote the single
        # extracted pass-root to ``template_instance`` so the response
        # builder serializes the populated Pydantic object (not the
        # node_link_data fallback).
        extracted = getattr(context, "extracted_models", None)
        if isinstance(extracted, list) and extracted:
            context.template_instance = extracted[0]

        try:
            context._upstream_preamble_applied = preamble_applied
        except AttributeError:
            pass

        # Build chunk_index → [doc_item.self_ref, ...] mapping so the
        # provenance resolver can turn chunk indexes (the only location
        # identity the delta-IR normalizer attaches) into real
        # DoclingDocument self_refs. Runs HybridChunker again on the doc —
        # the embedding model is already warm so this is milliseconds.
        # Kept best-effort: on chunker failure we log and continue with
        # None, which drops provenance rows cleanly rather than crashing.
        chunk_to_self_refs = _build_chunk_to_self_refs_map(
            getattr(context, "docling_document", None)
        )
        try:
            context._chunk_to_self_refs = chunk_to_self_refs
        except AttributeError:
            pass

        # Read library-level trace (batch_errors, quality_gate, identity_filter,
        # path_counts, merge_stats, diagnostics) for response surfacing.
        trace: dict | None = None
        for candidate in (
            os.path.join(debug_dir, "debug", "delta_trace.json"),
            os.path.join(debug_dir, "delta_trace.json"),
            os.path.join(debug_dir, "debug", "trace_data.json"),
        ):
            if os.path.exists(candidate):
                try:
                    with open(candidate, encoding="utf-8") as f:
                        trace = json.load(f)
                    break
                except Exception as exc:
                    logger.warning("Failed to load debug trace %s: %s", candidate, exc)
        if trace is None:
            trace = {}

        trace["library_log"] = library_log_buf.getvalue()
        if pipeline_error is not None:
            trace["pipeline_error"] = {
                "type": type(pipeline_error).__name__,
                "message": str(pipeline_error),
            }
        try:
            context._delta_trace = trace
        except AttributeError:
            pass
        return context
    finally:
        os.unlink(tmp_path)
        shutil.rmtree(debug_dir, ignore_errors=True)


def _build_chunk_to_self_refs_map(docling_document: Any) -> dict[int, list[str]] | None:
    """Re-chunk ``docling_document`` with HybridChunker and return a
    ``{chunk_index: [doc_item.self_ref, ...]}`` mapping.

    The library throws away chunk metadata after delta extraction, so we
    can't read it off ``context``. A re-run of the chunker is cheap
    because the sentence-transformers model is already loaded. Returns
    ``None`` when:
      - docling_document is missing/None
      - the chunker raises (logged at WARNING, never propagated)
    """
    if docling_document is None:
        return None
    try:
        from docling.chunking import HybridChunker
    except ImportError as exc:
        logger.warning("HybridChunker import failed; provenance self_refs unavailable: %s", exc)
        return None

    try:
        chunker = HybridChunker()
        out: dict[int, list[str]] = {}
        for i, chunk in enumerate(chunker.chunk(dl_doc=docling_document)):
            meta = getattr(chunk, "meta", None)
            doc_items = getattr(meta, "doc_items", None) if meta is not None else None
            refs = []
            for item in doc_items or []:
                ref = getattr(item, "self_ref", None)
                if isinstance(ref, str) and ref:
                    refs.append(ref)
            out[i] = refs
        return out
    except Exception as exc:
        logger.warning("Re-chunking for provenance self_refs failed: %s", exc)
        return None


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
    # Sentinels: -1 means extraction did not complete (failure path).
    node_count_for_log = -1
    edge_count_for_log = -1
    try:
        async with semaphore:
            try:
                context = await asyncio.to_thread(
                    run_extraction_pass,
                    body.docling_document_json,
                    template_cls,
                    body.upstream_entities,
                    body.pass_name,
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
            upstream_preamble_applied=getattr(context, "_upstream_preamble_applied", False),
        )
        node_count_for_log = metadata.node_count
        edge_count_for_log = metadata.edge_count

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

        # Phase 8 Task 51: per-entity-instance provenance payload. Nodes
        # whose element_uid cannot be resolved are dropped with WARNING
        # inside the helper — the response only carries chunk-linkable
        # rows.
        try:
            provenance_rows = build_provenance_from_context(
                context, ExtractionProvenance,
                chunk_to_self_refs=getattr(context, "_chunk_to_self_refs", None),
            )
        except Exception as exc:
            logger.warning(
                "extract-pass: provenance builder failed for document_id=%s bundle=%s pass=%s: %s",
                body.document_id, body.bundle_key, body.pass_name, exc,
            )
            provenance_rows = []

        evidence_text = _collect_batch_evidence_text(body.docling_document_json)
        filtered_pass_output, service_filter_stats, allowed_identities = _filter_pass_output_by_batch_text(
            pass_output,
            template_cls,
            evidence_text,
        )
        pass_output = filtered_pass_output
        provenance_rows = _filter_provenance_rows_by_allowed_identities(
            provenance_rows,
            allowed_identities,
        )
        if service_filter_stats:
            diagnostics = getattr(context, "_delta_trace", None) or {}
            diagnostics["service_identity_gate"] = service_filter_stats
            diagnostics["service_identity_gate"]["evidence_text_nonempty"] = bool(evidence_text)
            context._delta_trace = diagnostics

        return ExtractPassResponse(
            bundle_key=body.bundle_key,
            pass_name=body.pass_name,
            pass_output=pass_output,
            metadata=metadata,
            model=os.environ.get("DOCLING_GRAPH_LLM_MODEL", "granite3-dense:8b"),
            provider=os.environ.get("DOCLING_GRAPH_LLM_PROVIDER", "ollama"),
            provenance=provenance_rows,
            diagnostics=getattr(context, "_delta_trace", None),
        )
    finally:
        logger.info(
            "extract-pass: END bundle=%s pass=%s document_id=%s node_count=%d edge_count=%d",
            body.bundle_key, body.pass_name, body.document_id,
            node_count_for_log, edge_count_for_log,
        )
