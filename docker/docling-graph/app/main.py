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


def _run_entity_extraction(
    text: str,
    template_group: str | None = None,
) -> Any:
    """Run docling-graph pipeline for entity extraction (called in threadpool).

    Iterates through each entity model in the group and merges graphs.
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
        first_group = next(iter(_templates.values()))
        first_name, first_model = next(iter(first_group.items()))
        entity_models_to_run = [(first_name, first_model)]

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


def _run_relationship_extraction(
    text: str,
    entities_context: list[dict],
) -> list[dict]:
    """Extract relationships via direct LLM call with structured output.

    docling-graph's run_pipeline doesn't support relationship-only extraction,
    so we call LiteLLM directly with the relationship prompt and ask for JSON.
    """
    import json as json_mod

    import litellm

    from app.prompts import get_relationship_prompt

    model_string = _build_litellm_model_string()
    system_prompt = get_relationship_prompt(entities_context)

    user_prompt = (
        f"Analyze this text and extract relationships between the known entities:\n\n"
        f"=== TEXT ===\n{text}\n=== END TEXT ===\n\n"
        "Return a JSON object with a single key 'relationships' containing an array. "
        "Each relationship object must have: from_name, from_type, rel_type, to_name, to_type, confidence (0.0-1.0).\n"
        "Example: {\"relationships\": [{\"from_name\": \"AN/MPQ-53\", \"from_type\": \"RADAR_SYSTEM\", "
        "\"rel_type\": \"INSTALLED_ON\", \"to_name\": \"Patriot\", \"to_type\": \"PLATFORM\", \"confidence\": 0.9}]}\n"
        "Return ONLY valid JSON."
    )

    llm_kwargs: dict[str, Any] = {
        "model": model_string,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    if LLM_PROVIDER == "ollama":
        llm_kwargs["api_base"] = OLLAMA_BASE_URL

    try:
        response = litellm.completion(**llm_kwargs)
        raw = response.choices[0].message.content
        if raw is None:
            logger.warning("Relationship extraction: LLM returned None content")
            return []
        raw = raw.strip()
        logger.debug("Relationship LLM raw output (%d chars): %s", len(raw), raw[:500])

        # Try to parse JSON — handle markdown-wrapped responses
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                inner = parts[1]
                if inner.startswith("json"):
                    inner = inner[4:]
                raw = inner.strip()

        # Try to find JSON object in the response
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            raw = raw[json_start:json_end]

        parsed = json_mod.loads(raw)
        relationships = parsed.get("relationships", [])
        logger.info("Relationship extraction returned %d relationships", len(relationships))
        return relationships
    except Exception as exc:
        logger.warning("Relationship extraction failed: %s — raw: %s", exc, raw[:200] if raw else "(empty)")
        return []


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
        node_name = data.get("name") or data.get("label") or str(node_id)
        entities.append(
            ExtractedEntityResponse(
                name=node_name,
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

    if request.mode == "relationships":
        # Relationship extraction via direct LLM call (not docling-graph pipeline)
        try:
            rel_dicts = await run_in_threadpool(
                _run_relationship_extraction,
                request.text,
                request.entities_context or [],
            )
        except Exception:
            logger.exception("Relationship extraction failed for document %s", request.document_id)
            raise HTTPException(status_code=503, detail="Relationship extraction failed")

        relationships = [
            ExtractedRelationshipResponse(
                from_name=r.get("from_name", ""),
                from_type=r.get("from_type", "UNKNOWN"),
                rel_type=r.get("rel_type", "RELATED_TO"),
                to_name=r.get("to_name", ""),
                to_type=r.get("to_type", "UNKNOWN"),
                confidence=float(r.get("confidence", 0.5)),
            )
            for r in rel_dicts
            if r.get("from_name") and r.get("to_name")
        ]

        logger.info(
            "Extracted %d relationships from document %s",
            len(relationships), request.document_id,
        )

        return ExtractionResponse(
            entities=[],
            relationships=relationships,
            ontology_version=_ontology_version,
            model=LLM_MODEL,
            provider=LLM_PROVIDER,
        )

    # Entity extraction via docling-graph pipeline
    try:
        graph = await run_in_threadpool(
            _run_entity_extraction,
            request.text,
            request.template_group,
        )
    except Exception:
        logger.exception("Entity extraction failed for document %s", request.document_id)
        raise HTTPException(status_code=503, detail="Extraction pipeline failed")

    entities, graph_relationships = _graph_to_response(graph)

    logger.info(
        "Extracted %d entities, %d relationships from document %s (group=%s)",
        len(entities),
        len(graph_relationships),
        request.document_id,
        request.template_group or "(default)",
    )

    return ExtractionResponse(
        entities=entities,
        relationships=graph_relationships,
        ontology_version=_ontology_version,
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
    )
