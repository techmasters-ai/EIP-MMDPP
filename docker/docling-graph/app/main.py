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
from app.templates import GROUP_MAP, build_templates, get_ontology_version, load_ontology

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
_templates: dict[str, dict[str, type[BaseModel]]] | None = None
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
    total_entity_types = sum(len(group) for group in _templates.values())
    return {
        "status": "ok",
        "ontology_version": _ontology_version,
        "groups": list(_templates.keys()),
        "template_count": total_entity_types,
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


def _mock_extraction_response(
    mode: str = "entities", template_group: str | None = None
) -> ExtractionResponse:
    """Return a canned response for testing (LLM_PROVIDER=mock).

    When *mode* is ``"relationships"``, returns mock relationships only (empty
    entities).  When ``"entities"``, returns mock entities only (empty
    relationships).
    """
    mock_entities = [
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
    ]
    mock_relationships = [
        ExtractedRelationshipResponse(
            from_name="Mock Platform",
            from_type="PLATFORM",
            rel_type="HAS_COMPONENT",
            to_name="Mock Radar System",
            to_type="RADAR_SYSTEM",
            confidence=0.85,
        ),
    ]

    if mode == "relationships":
        return ExtractionResponse(
            entities=[],
            relationships=mock_relationships,
            ontology_version=_ontology_version,
            model="mock",
            provider="mock",
        )

    # mode == "entities" (or legacy default)
    return ExtractionResponse(
        entities=mock_entities,
        relationships=[],
        ontology_version=_ontology_version,
        model="mock",
        provider="mock",
    )


def _run_extraction(
    text: str,
    template_group: str | None = None,
    mode: str = "entities",
    entities_context: list[dict] | None = None,
) -> Any:
    """Run docling-graph pipeline synchronously (called in threadpool).

    When *template_group* is provided, iterates through each entity model
    in that group and merges the resulting graphs.  Each entity type gets
    its own ``run_pipeline`` call so docling-graph can use a focused schema.

    The ``docling_graph`` import is deferred because the package is only
    available inside the Docker container.
    """
    import tempfile
    from pathlib import Path

    import networkx as nx
    from docling_graph import run_pipeline  # type: ignore[import-untyped]

    model_string = _build_litellm_model_string()

    # Resolve which entity model(s) to use
    entity_models_to_run: list[tuple[str, type]] = []
    if template_group and _templates and template_group in _templates:
        group_models = _templates[template_group]
        entity_models_to_run = list(group_models.items())
    elif _templates:
        # Legacy: pick the first entity model from the first group
        first_group = next(iter(_templates.values()))
        first_name, first_model = next(iter(first_group.items()))
        entity_models_to_run = [(first_name, first_model)]

    # Write text to a temp .md file — docling-graph's input handler tries
    # Path(source).exists() first, which raises ENAMETOOLONG for long text
    tmp = tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8")
    try:
        tmp.write(text)
        tmp.close()

        merged_graph = nx.DiGraph()

        for entity_name, template_cls in entity_models_to_run:
            config: dict[str, Any] = {
                "source": tmp.name,
                "template": template_cls,
                "backend": "llm",
                "inference": "remote",
                "model_override": model_string,
                "provider_override": LLM_PROVIDER,
                "dump_to_disk": False,
                "llm_overrides": {
                    "connection": {"base_url": OLLAMA_BASE_URL},
                },
            }
            try:
                context = run_pipeline(config=config, mode="api")
                kg = context.knowledge_graph
                if kg is not None:
                    merged_graph = nx.compose(merged_graph, kg)
                    logger.info(
                        "Extracted %d nodes, %d edges for entity type %s",
                        kg.number_of_nodes(), kg.number_of_edges(), entity_name,
                    )
            except Exception as exc:
                logger.warning(
                    "docling-graph pipeline error for %s: %s — skipping",
                    entity_name, exc,
                )

        return merged_graph
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _graph_to_response(graph: Any) -> tuple[list[ExtractedEntityResponse], list[ExtractedRelationshipResponse]]:
    """Convert a NetworkX DiGraph to lists of entity/relationship responses."""
    entities: list[ExtractedEntityResponse] = []
    relationships: list[ExtractedRelationshipResponse] = []

    # Nodes carry entity data.
    # docling-graph stores the entity class in 'label' or '__class__' within
    # node properties; 'entity_type' may just be 'entity' (generic).
    for node_id, data in graph.nodes(data=True):
        raw_type = data.get("entity_type", data.get("type", "UNKNOWN"))
        # Prefer 'label' or '__class__' for the actual ontology entity type
        if raw_type in ("entity", "UNKNOWN"):
            raw_type = data.get("label", data.get("__class__", raw_type))
        entities.append(
            ExtractedEntityResponse(
                name=data.get("name", str(node_id)),
                entity_type=raw_type,
                confidence=float(data.get("confidence", 1.0)),
                properties={
                    k: v
                    for k, v in data.items()
                    if k not in {"name", "entity_type", "type", "confidence", "label", "__class__"}
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


_VALID_MODES = {"entities", "relationships"}


@app.post("/extract", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest):
    """Extract entities and relationships from document text."""
    if _templates is None:
        raise HTTPException(status_code=503, detail="Service not ready — templates not loaded")

    # Validate mode
    if request.mode not in _VALID_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown mode '{request.mode}'; valid modes: {sorted(_VALID_MODES)}",
        )

    # Validate template_group
    if request.template_group is not None and request.template_group not in _templates:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown template_group '{request.template_group}'; "
                f"valid groups: {sorted(_templates.keys())}"
            ),
        )

    # Warn on ontology version mismatch
    if request.ontology_version and request.ontology_version != _ontology_version:
        logger.warning(
            "Ontology version mismatch: request=%s, loaded=%s",
            request.ontology_version,
            _ontology_version,
        )

    logger.info(
        "Extracting from document %s (%d chars, group=%s, mode=%s)",
        request.document_id,
        len(request.text),
        request.template_group or "(default)",
        request.mode,
    )

    # Mock mode: return canned response without importing docling_graph
    if LLM_PROVIDER == "mock":
        logger.info("Mock mode — returning canned extraction for document %s", request.document_id)
        return _mock_extraction_response(mode=request.mode, template_group=request.template_group)

    try:
        graph = await run_in_threadpool(
            _run_extraction,
            request.text,
            template_group=request.template_group,
            mode=request.mode,
            entities_context=request.entities_context,
        )
    except Exception:
        logger.exception("Extraction failed for document %s", request.document_id)
        raise HTTPException(status_code=503, detail="Extraction pipeline failed")

    entities, relationships = _graph_to_response(graph)

    logger.info(
        "Extracted %d entities, %d relationships from document %s (group=%s, mode=%s)",
        len(entities),
        len(relationships),
        request.document_id,
        request.template_group or "(default)",
        request.mode,
    )

    return ExtractionResponse(
        entities=entities,
        relationships=relationships,
        ontology_version=_ontology_version,
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
    )
