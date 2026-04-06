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
