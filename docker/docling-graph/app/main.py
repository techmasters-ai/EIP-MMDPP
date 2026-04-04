"""Docling-Graph extraction service — thin wrapper around run_pipeline()."""

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
from fastapi import FastAPI, HTTPException

from app.config_builder import build_pipeline_config
from app.schemas import (
    ExtractionMetadata,
    ExtractionRequest,
    ExtractionResponse,
    HealthResponse,
)
from app.template_builder import build_templates_with_edges

logger = logging.getLogger(__name__)

# Module-level state
_templates: dict[str, Any] = {}
_ontology_version: str | None = None
_extraction_semaphore: asyncio.Semaphore | None = None
_ontology_cache: dict[str, dict[str, Any]] = {}
_pipeline_version: str = "unknown"

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
    global _templates, _ontology_version, _extraction_semaphore, _pipeline_version

    _pipeline_version = _validate_library_surface()
    logger.info("docling-graph library version: %s", _pipeline_version)

    if os.path.exists(ONTOLOGY_PATH):
        with open(ONTOLOGY_PATH) as f:
            ontology = yaml.safe_load(f)
        _ontology_version = ontology.get("version")
        _templates = build_templates_with_edges(ontology)
        logger.info("Loaded ontology v%s (%d templates)", _ontology_version, len(_templates))
    else:
        logger.warning("Ontology not found at %s", ONTOLOGY_PATH)

    _extraction_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    yield
    logger.info("Shutting down")


app = FastAPI(title="Docling-Graph Extraction Service", version="2.0.0", lifespan=lifespan)


def _resolve_templates(ontology_definition: dict[str, Any] | None) -> dict[str, Any]:
    if ontology_definition is None:
        return _templates
    ont_hash = hashlib.sha256(json.dumps(ontology_definition, sort_keys=True).encode()).hexdigest()[:16]
    if ont_hash in _ontology_cache:
        return _ontology_cache[ont_hash]
    templates = build_templates_with_edges(ontology_definition)
    _ontology_cache[ont_hash] = templates
    if len(_ontology_cache) > 2:
        del _ontology_cache[next(iter(_ontology_cache))]
    return templates


def run_extraction_pipeline(docling_document_json: dict[str, Any], templates: dict[str, Any]) -> Any:
    """Run docling-graph pipeline synchronously (called via asyncio.to_thread)."""
    import tempfile
    from docling_graph import run_pipeline

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(docling_document_json, tmp, ensure_ascii=False, default=str)
        tmp_path = tmp.name

    try:
        root_template = next(iter(templates.values())) if templates else None
        config = build_pipeline_config(source=tmp_path, template_class=root_template)
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
async def health():
    return HealthResponse(
        ontology_version=_ontology_version,
        template_count=len(_templates),
        extraction_contract=os.environ.get("DOCLING_GRAPH_EXTRACTION_CONTRACT", "delta"),
        pipeline_version=_pipeline_version,
    )


@app.post("/extract-all", response_model=ExtractionResponse)
async def extract_all(request: ExtractionRequest):
    if _extraction_semaphore is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    templates = _resolve_templates(request.ontology_definition)
    if not templates:
        raise HTTPException(status_code=422, detail="No templates available")

    if request.ontology_version and request.ontology_version != _ontology_version:
        logger.warning("Ontology version mismatch: request=%s server=%s", request.ontology_version, _ontology_version)

    async with _extraction_semaphore:
        try:
            context = await asyncio.to_thread(run_extraction_pipeline, request.docling_document_json, templates)
        except Exception as exc:
            logger.exception("Pipeline failed for %s", request.document_id)
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
        ontology_version=_ontology_version,
    )
