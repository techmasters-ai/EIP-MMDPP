"""LLM-powered entity/relationship extraction via ontology templates.

Uses LiteLLM for provider-agnostic routing (Ollama, OpenAI, etc.) controlled
by the system-wide LLM_PROVIDER setting.  Falls back to regex NER
(app.services.ner) when the LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import networkx as nx

from app.config import get_settings
from app.services.ontology_templates import (
    DocumentExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    build_extraction_prompt,
    load_ontology,
)

logger = logging.getLogger(__name__)


def extract_graph_from_text(
    text: str,
    document_id: str,
    *,
    ontology_path=None,
) -> nx.DiGraph:
    """Extract entities and relationships from text, returning a NetworkX DiGraph.

    The LLM provider is controlled by ``settings.llm_provider``:
      - ``ollama``: routes through local Ollama instance
      - ``openai``: routes through OpenAI API
      - ``mock``: returns an empty graph (for tests)

    Falls back to regex NER if the LLM call fails.
    """
    settings = get_settings()

    if settings.llm_provider == "mock":
        logger.info("LLM provider is 'mock'; returning empty graph")
        return nx.DiGraph()

    ontology = load_ontology(ontology_path)
    prompt = build_extraction_prompt(ontology, text)

    try:
        result = _call_llm(prompt, settings)
        extraction = _parse_llm_response(result)
    except Exception as e:
        logger.warning("LLM extraction failed, falling back to regex NER: %s", e)
        extraction = _fallback_regex_extraction(text)

    return _build_networkx_graph(extraction, document_id)


def _call_llm(prompt: str, settings) -> str:
    """Call LLM via LiteLLM for entity/relationship extraction."""
    import litellm

    provider = settings.llm_provider
    model = settings.docling_graph_model

    if provider == "ollama":
        model_str = f"ollama/{model}"
        litellm.api_base = settings.ollama_base_url
    elif provider == "openai":
        model_str = model
        litellm.api_key = settings.openai_api_key
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    response = litellm.completion(
        model=model_str,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a military/defense document analysis expert. "
                    "Extract entities and relationships as structured JSON. "
                    "Return ONLY valid JSON, no markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
        timeout=settings.docling_graph_timeout,
    )

    return response.choices[0].message.content


def _parse_llm_response(response_text: str) -> DocumentExtractionResult:
    """Parse LLM JSON output into a DocumentExtractionResult."""
    # Strip markdown fences if the LLM included them despite instructions
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    data = json.loads(cleaned)
    return DocumentExtractionResult(**data)


def _fallback_regex_extraction(text: str) -> DocumentExtractionResult:
    """Use regex NER as fallback when LLM is unavailable."""
    from app.services.ner import extract_entities, extract_relationships

    entity_candidates = extract_entities(text)
    rel_candidates = extract_relationships(text, entity_candidates)

    entities = [
        ExtractedEntity(
            entity_type=ec.entity_type,
            name=ec.name,
            properties=ec.properties,
            confidence=ec.confidence,
        )
        for ec in entity_candidates
    ]

    relationships = [
        ExtractedRelationship(
            relationship_type=rc.rel_type,
            from_name=rc.from_name,
            from_type=rc.from_type,
            to_name=rc.to_name,
            to_type=rc.to_type,
            confidence=rc.confidence,
        )
        for rc in rel_candidates
    ]

    return DocumentExtractionResult(entities=entities, relationships=relationships)


def _build_networkx_graph(
    extraction: DocumentExtractionResult,
    document_id: str,
) -> nx.DiGraph:
    """Convert extraction results into a NetworkX DiGraph for AGE import."""
    G = nx.DiGraph()

    for entity in extraction.entities:
        node_id = f"{entity.entity_type}:{entity.name}"
        G.add_node(
            node_id,
            entity_type=entity.entity_type,
            name=entity.name,
            properties=entity.properties,
            confidence=entity.confidence,
            document_id=document_id,
        )

    for rel in extraction.relationships:
        from_id = f"{rel.from_type}:{rel.from_name}"
        to_id = f"{rel.to_type}:{rel.to_name}"

        # Ensure source/target nodes exist (LLM might reference entities
        # not in the entities list)
        if from_id not in G:
            G.add_node(
                from_id,
                entity_type=rel.from_type,
                name=rel.from_name,
                properties={},
                confidence=0.3,
                document_id=document_id,
            )
        if to_id not in G:
            G.add_node(
                to_id,
                entity_type=rel.to_type,
                name=rel.to_name,
                properties={},
                confidence=0.3,
                document_id=document_id,
            )

        G.add_edge(
            from_id,
            to_id,
            relationship_type=rel.relationship_type,
            confidence=rel.confidence,
            document_id=document_id,
        )

    return G


def networkx_to_age_import(
    graph: nx.DiGraph,
    artifact_id: str,
) -> dict[str, Any]:
    """Convert NetworkX graph to AGE-compatible import format.

    Returns a dict with 'nodes' and 'edges' lists ready for the
    graph.py upsert_node() / upsert_relationship() helpers.
    """
    nodes = []
    for node_id, data in graph.nodes(data=True):
        nodes.append({
            "entity_type": data.get("entity_type", "UNKNOWN"),
            "name": data.get("name", str(node_id)),
            "properties": data.get("properties", {}),
            "confidence": data.get("confidence", 0.5),
            "artifact_id": artifact_id,
        })

    edges = []
    for from_id, to_id, data in graph.edges(data=True):
        from_data = graph.nodes[from_id]
        to_data = graph.nodes[to_id]
        edges.append({
            "from_name": from_data.get("name", str(from_id)),
            "from_type": from_data.get("entity_type", "UNKNOWN"),
            "to_name": to_data.get("name", str(to_id)),
            "to_type": to_data.get("entity_type", "UNKNOWN"),
            "rel_type": data.get("relationship_type", "RELATED_TO"),
            "confidence": data.get("confidence", 0.5),
            "artifact_id": artifact_id,
        })

    return {"nodes": nodes, "edges": edges}
