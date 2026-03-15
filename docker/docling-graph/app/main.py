"""Docling-Graph entity/relationship extraction service.

FastAPI wrapper around the docling-graph package.  Generates Pydantic
templates from a volume-mounted ontology YAML and routes LLM calls via
LiteLLM to Ollama or an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.schemas import (
    ExtractedEntityResponse,
    ExtractedRelationshipResponse,
    ExtractionRequest,
    ExtractionResponse,
)
from app.templates import build_templates, get_ontology_version, load_ontology

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
ONTOLOGY_PATH = os.environ.get("ONTOLOGY_PATH", "/ontology/ontology.yaml")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "granite3-dense:8b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

# ---------------------------------------------------------------------------
# Module-level state (populated at startup)
# ---------------------------------------------------------------------------
_templates: dict[str, type[BaseModel]] | None = None
_ontology_version: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ontology and build templates at startup."""
    global _templates, _ontology_version

    logger.info("Loading ontology from %s", ONTOLOGY_PATH)
    try:
        ontology = load_ontology(ONTOLOGY_PATH)
        _ontology_version = get_ontology_version(ontology)
        _templates = build_templates(ontology)
        logger.info(
            "Docling-Graph service ready — ontology v%s, %d templates",
            _ontology_version,
            len(_templates),
        )
    except Exception:
        logger.exception("Failed to load ontology — service will report unhealthy")

    yield
    logger.info("Docling-Graph service shutting down.")


app = FastAPI(
    title="EIP-MMDPP Docling-Graph Service",
    description="Entity/relationship extraction via docling-graph + LiteLLM",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    if _templates is None:
        raise HTTPException(status_code=503, detail="Templates not loaded")
    return {
        "status": "ok",
        "ontology_version": _ontology_version,
        "template_count": len(_templates),
    }


# ---------------------------------------------------------------------------
# Extraction endpoint
# ---------------------------------------------------------------------------
def _build_litellm_model_string() -> str:
    """Construct the model string LiteLLM expects for the configured provider."""
    if LLM_PROVIDER == "ollama":
        return f"ollama/{LLM_MODEL}"
    if LLM_PROVIDER == "openai":
        return LLM_MODEL
    # Generic fallback: provider/model
    return f"{LLM_PROVIDER}/{LLM_MODEL}"


def _mock_extraction_response() -> ExtractionResponse:
    """Return a canned response for testing (LLM_PROVIDER=mock)."""
    return ExtractionResponse(
        entities=[
            ExtractedEntityResponse(
                name="Mock Radar System",
                entity_type="RADAR_SYSTEM",
                confidence=0.95,
                properties={"designation": "AN/APG-00"},
            ),
            ExtractedEntityResponse(
                name="Mock Platform",
                entity_type="PLATFORM",
                confidence=0.90,
                properties={"platform_type": "aircraft"},
            ),
        ],
        relationships=[
            ExtractedRelationshipResponse(
                from_name="Mock Platform",
                from_type="PLATFORM",
                rel_type="HAS_COMPONENT",
                to_name="Mock Radar System",
                to_type="RADAR_SYSTEM",
                confidence=0.85,
            ),
        ],
        ontology_version=_ontology_version,
        model="mock",
        provider="mock",
    )


def _run_extraction(text: str) -> Any:
    """Run docling-graph pipeline synchronously (called in threadpool).

    The ``docling_graph`` import is deferred because the package is only
    available inside the Docker container.
    """
    import networkx as nx  # noqa: F811
    from docling_graph import run_pipeline  # type: ignore[import-untyped]

    # Configure LiteLLM environment before calling the pipeline
    if LLM_PROVIDER == "ollama":
        os.environ.setdefault("OLLAMA_API_BASE", OLLAMA_BASE_URL)
    if OPENAI_API_KEY:
        os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
    if OPENAI_BASE_URL:
        os.environ.setdefault("OPENAI_API_BASE", OPENAI_BASE_URL)

    model_string = _build_litellm_model_string()

    graph: nx.DiGraph = run_pipeline(
        text=text,
        templates=_templates,
        model=model_string,
    )
    return graph


def _graph_to_response(graph: Any) -> tuple[list[ExtractedEntityResponse], list[ExtractedRelationshipResponse]]:
    """Convert a NetworkX DiGraph to lists of entity/relationship responses."""
    entities: list[ExtractedEntityResponse] = []
    relationships: list[ExtractedRelationshipResponse] = []

    # Nodes carry entity data
    for node_id, data in graph.nodes(data=True):
        entities.append(
            ExtractedEntityResponse(
                name=data.get("name", str(node_id)),
                entity_type=data.get("entity_type", data.get("type", "UNKNOWN")),
                confidence=float(data.get("confidence", 1.0)),
                properties={
                    k: v
                    for k, v in data.items()
                    if k not in {"name", "entity_type", "type", "confidence"}
                },
            )
        )

    # Edges carry relationship data
    for src, dst, data in graph.edges(data=True):
        src_data = graph.nodes[src]
        dst_data = graph.nodes[dst]
        relationships.append(
            ExtractedRelationshipResponse(
                from_name=src_data.get("name", str(src)),
                from_type=src_data.get("entity_type", src_data.get("type", "UNKNOWN")),
                rel_type=data.get("rel_type", data.get("type", "RELATED_TO")),
                to_name=dst_data.get("name", str(dst)),
                to_type=dst_data.get("entity_type", dst_data.get("type", "UNKNOWN")),
                confidence=float(data.get("confidence", 1.0)),
            )
        )

    return entities, relationships


@app.post("/extract", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest):
    """Extract entities and relationships from document text."""
    if _templates is None:
        raise HTTPException(status_code=503, detail="Service not ready — templates not loaded")

    # Warn on ontology version mismatch
    if request.ontology_version and request.ontology_version != _ontology_version:
        logger.warning(
            "Ontology version mismatch: request=%s, loaded=%s",
            request.ontology_version,
            _ontology_version,
        )

    logger.info(
        "Extracting from document %s (%d chars)",
        request.document_id,
        len(request.text),
    )

    # Mock mode: return canned response without importing docling_graph
    if LLM_PROVIDER == "mock":
        logger.info("Mock mode — returning canned extraction for document %s", request.document_id)
        return _mock_extraction_response()

    try:
        graph = await run_in_threadpool(_run_extraction, request.text)
    except Exception:
        logger.exception("Extraction failed for document %s", request.document_id)
        raise HTTPException(status_code=503, detail="Extraction pipeline failed")

    entities, relationships = _graph_to_response(graph)

    logger.info(
        "Extracted %d entities, %d relationships from document %s",
        len(entities),
        len(relationships),
        request.document_id,
    )

    return ExtractionResponse(
        entities=entities,
        relationships=relationships,
        ontology_version=_ontology_version,
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
    )
