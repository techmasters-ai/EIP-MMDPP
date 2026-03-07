"""LLM-powered entity/relationship extraction via ontology templates.

Uses LiteLLM for provider-agnostic routing (Ollama, OpenAI, etc.) controlled
by the system-wide LLM_PROVIDER setting.  Falls back to regex NER
(app.services.ner) when the LLM is unavailable.

Extraction results are imported directly into Neo4j (replaces AGE).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import networkx as nx
import redis as redis_lib

from app.config import get_settings
from app.services.ontology_templates import (
    DocumentExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    build_extraction_prompt,
    load_ontology,
    load_validation_matrix,
)

logger = logging.getLogger(__name__)


def extract_graph_from_text(
    text: str,
    document_id: str,
    *,
    ontology_path=None,
) -> nx.DiGraph:
    """Extract entities and relationships from text, returning a NetworkX DiGraph.

    Backward-compatible wrapper around ``_extract_single_pass``.
    """
    graph, _provider = _extract_single_pass(text, document_id, ontology_path=ontology_path)
    return graph


def _extract_single_pass(
    text: str,
    document_id: str,
    *,
    ontology_path=None,
) -> tuple[nx.DiGraph, str]:
    """Single-pass extraction with retry/backoff. Returns (graph, actual_provider).

    The provider string reflects what actually ran: "docling-graph" if the LLM
    succeeded, "ner" if it fell back to regex, or "mock" in test mode.

    When ``docling_graph_require_llm`` is True (default), raises on final
    failure instead of falling back to NER — fail-closed behavior.
    """
    settings = get_settings()

    if settings.llm_provider == "mock":
        logger.info("LLM provider is 'mock'; returning empty graph")
        return nx.DiGraph(), "mock"

    ontology = load_ontology(ontology_path)
    prompt = build_extraction_prompt(ontology, text)

    max_attempts = settings.docling_graph_retry_attempts
    backoff = settings.docling_graph_retry_backoff_seconds

    for attempt in range(max_attempts):
        try:
            result = _call_llm(prompt, settings)
            extraction = _parse_llm_response(result)
            extraction = _validate_triples(extraction, ontology_path)
            return _build_networkx_graph(extraction, document_id), "docling-graph"
        except Exception as e:
            if attempt < max_attempts - 1:
                wait = backoff * (attempt + 1)
                logger.info(
                    "LLM extraction attempt %d/%d failed, retrying in %ds: %s",
                    attempt + 1, max_attempts, wait, e,
                )
                time.sleep(wait)
                continue
            if settings.docling_graph_require_llm:
                logger.error(
                    "LLM extraction failed after %d attempts (fail-closed): %s",
                    max_attempts, e,
                )
                raise
            logger.warning(
                "LLM extraction failed after %d attempts, falling back to NER: %s",
                max_attempts, e,
            )
            extraction = _fallback_regex_extraction(text)
            return _build_networkx_graph(extraction, document_id), "ner"

    # Should not be reached, but satisfy type checker
    raise RuntimeError("Extraction loop exited without result")


def extract_graph_from_text_chunked(
    text: str,
    document_id: str,
    *,
    chunk_size: int = 7000,
    chunk_overlap: int = 500,
    ontology_path=None,
) -> tuple[nx.DiGraph, str]:
    """Extract entities/relationships using chunked windows for large text.

    For short text (<= chunk_size), delegates to a single pass.
    For longer text, splits into overlapping chunks, extracts from each,
    deduplicates entities/relationships, and returns merged results.

    Returns (graph, provider) where provider reflects the actual method used.
    """
    if len(text) <= chunk_size:
        return _extract_single_pass(text, document_id, ontology_path=ontology_path)

    # Build overlapping chunks
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - chunk_overlap

    logger.info(
        "extract_graph_from_text_chunked: %d chars → %d chunks (size=%d, overlap=%d)",
        len(text), len(chunks), chunk_size, chunk_overlap,
    )

    # Extract from each chunk and merge
    merged = DocumentExtractionResult()
    actual_provider: str | None = None

    for i, chunk_text in enumerate(chunks):
        chunk_graph, chunk_provider = _extract_single_pass(
            chunk_text, document_id, ontology_path=ontology_path,
        )
        if actual_provider is None:
            actual_provider = chunk_provider

        for node_id, data in chunk_graph.nodes(data=True):
            merged.entities.append(ExtractedEntity(
                entity_type=data.get("entity_type", "UNKNOWN"),
                name=data.get("name", str(node_id)),
                properties=data.get("properties", {}),
                confidence=data.get("confidence", 0.5),
            ))
        for u, v, data in chunk_graph.edges(data=True):
            merged.relationships.append(ExtractedRelationship(
                relationship_type=data.get("relationship_type", "RELATED_TO"),
                from_name=chunk_graph.nodes[u].get("name", str(u)),
                from_type=chunk_graph.nodes[u].get("entity_type", "UNKNOWN"),
                to_name=chunk_graph.nodes[v].get("name", str(v)),
                to_type=chunk_graph.nodes[v].get("entity_type", "UNKNOWN"),
                properties=data.get("properties", {}),
                confidence=data.get("confidence", 0.5),
            ))

    merged = _deduplicate_extraction(merged)
    return _build_networkx_graph(merged, document_id), actual_provider or "ner"


def _deduplicate_extraction(
    result: DocumentExtractionResult,
) -> DocumentExtractionResult:
    """Deduplicate entities and relationships from multi-chunk extraction.

    Entities: keyed by (entity_type, name), keeps highest confidence,
    merges properties from all occurrences.
    Relationships: keyed by (from_name, to_name, from_type, relationship_type),
    keeps highest confidence.
    """
    entity_map: dict[tuple[str, str], ExtractedEntity] = {}
    for e in result.entities:
        key = (e.entity_type, e.name)
        existing = entity_map.get(key)
        if existing is None:
            entity_map[key] = e
        elif e.confidence > existing.confidence:
            merged_props = {**existing.properties, **e.properties}
            entity_map[key] = ExtractedEntity(
                entity_type=e.entity_type,
                name=e.name,
                properties=merged_props,
                confidence=e.confidence,
            )
        else:
            # Keep existing but merge in any new properties
            merged_props = {**e.properties, **existing.properties}
            entity_map[key] = ExtractedEntity(
                entity_type=existing.entity_type,
                name=existing.name,
                properties=merged_props,
                confidence=existing.confidence,
            )

    rel_map: dict[tuple[str, str, str, str], ExtractedRelationship] = {}
    for r in result.relationships:
        key = (r.from_name, r.to_name, r.from_type, r.relationship_type)
        existing = rel_map.get(key)
        if existing is None or r.confidence > existing.confidence:
            rel_map[key] = r

    return DocumentExtractionResult(
        entities=list(entity_map.values()),
        relationships=list(rel_map.values()),
    )


def extract_and_import_to_neo4j(
    text: str,
    document_id: str,
    artifact_id: str,
    driver,
    *,
    ontology_path=None,
) -> dict[str, Any]:
    """Extract entities/relationships and import directly into Neo4j.

    Returns import stats: {nodes_upserted, edges_upserted, nodes_failed, edges_failed}.
    """
    from app.services.neo4j_graph import upsert_node, upsert_relationship

    graph = extract_graph_from_text(text, document_id, ontology_path=ontology_path)
    import_data = networkx_to_neo4j_import(graph, artifact_id)

    stats = {"nodes_upserted": 0, "edges_upserted": 0, "nodes_failed": 0, "edges_failed": 0}

    for node in import_data["nodes"]:
        result = upsert_node(
            driver,
            entity_type=node["entity_type"],
            name=node["name"],
            artifact_id=node["artifact_id"],
            confidence=node["confidence"],
            properties=node["properties"],
        )
        if result:
            stats["nodes_upserted"] += 1
        else:
            stats["nodes_failed"] += 1

    for edge in import_data["edges"]:
        success = upsert_relationship(
            driver,
            from_name=edge["from_name"],
            from_type=edge["from_type"],
            to_name=edge["to_name"],
            to_type=edge["to_type"],
            rel_type=edge["rel_type"],
            artifact_id=edge["artifact_id"],
            confidence=edge["confidence"],
            properties=edge.get("properties", {}),
        )
        if success:
            stats["edges_upserted"] += 1
        else:
            stats["edges_failed"] += 1

    logger.info(
        "Neo4j import for document %s: %d nodes, %d edges (%d/%d failed)",
        document_id,
        stats["nodes_upserted"],
        stats["edges_upserted"],
        stats["nodes_failed"],
        stats["edges_failed"],
    )
    return stats


def _call_llm(prompt: str, settings) -> str:
    """Call LLM via LiteLLM for entity/relationship extraction.

    Serializes Ollama calls via a Redis lock to prevent concurrent requests
    from overwhelming the model server (same pattern as docling concurrency gate).
    Guards against empty/None responses.
    """
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

    logger.info(
        "LLM graph extraction: model=%s, prompt_len=%d, max_tokens=%d",
        model_str, len(prompt), settings.docling_graph_max_tokens,
    )

    # Serialize LLM calls via Redis lock (prevents concurrent Ollama overload)
    r = redis_lib.from_url(settings.redis_url)
    lock_timeout = int(settings.docling_graph_timeout) + 60
    lock = r.lock(
        "ollama:llm_extract",
        timeout=lock_timeout,
        blocking_timeout=lock_timeout,
    )
    if not lock.acquire(blocking=True):
        raise TimeoutError("Could not acquire Ollama LLM lock")

    try:
        kwargs: dict[str, Any] = dict(
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
            max_tokens=settings.docling_graph_max_tokens,
            timeout=settings.docling_graph_timeout,
        )
        if provider == "ollama":
            kwargs["num_ctx"] = settings.ollama_num_ctx
        response = litellm.completion(**kwargs)
    finally:
        try:
            lock.release()
        except redis_lib.exceptions.LockNotOwnedError:
            pass

    content = response.choices[0].message.content
    finish_reason = getattr(response.choices[0], "finish_reason", None)
    if not content or not content.strip():
        logger.error(
            "LLM returned empty response: finish_reason=%s, model=%s, prompt_len=%d",
            finish_reason, model_str, len(prompt),
        )
        raise ValueError("LLM returned empty response")

    logger.info("LLM graph extraction response: %d chars", len(content))
    return content


def _parse_llm_response(response_text: str) -> DocumentExtractionResult:
    """Parse LLM JSON output into a DocumentExtractionResult.

    Handles common LLM output issues: markdown fences, preamble text before
    JSON, and truncated output.
    """
    cleaned = response_text.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # First attempt: direct parse
    try:
        data = json.loads(cleaned)
        return DocumentExtractionResult(**data)
    except json.JSONDecodeError:
        pass

    # Recovery: extract JSON object between first { and last }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        subset = cleaned[first_brace:last_brace + 1]
        try:
            data = json.loads(subset)
            logger.info("JSON recovery succeeded (extracted object from position %d-%d)", first_brace, last_brace)
            return DocumentExtractionResult(**data)
        except json.JSONDecodeError:
            pass

    # All recovery failed
    snippet = response_text[:200].replace("\n", "\\n")
    raise ValueError(f"Failed to parse LLM response as JSON. Response starts with: {snippet}")


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


def _validate_triples(
    extraction: DocumentExtractionResult,
    ontology_path=None,
) -> DocumentExtractionResult:
    """Filter out relationships that violate the ontology validation matrix."""
    matrix = load_validation_matrix(ontology_path)
    if not matrix:
        return extraction

    valid_rels = []
    for rel in extraction.relationships:
        triple = (rel.from_type, rel.relationship_type, rel.to_type)
        if triple in matrix:
            valid_rels.append(rel)
        else:
            logger.debug(
                "Rejected invalid triple: %s -[%s]-> %s",
                rel.from_type, rel.relationship_type, rel.to_type,
            )

    return DocumentExtractionResult(
        entities=extraction.entities,
        relationships=valid_rels,
    )


def _build_networkx_graph(
    extraction: DocumentExtractionResult,
    document_id: str,
) -> nx.DiGraph:
    """Convert extraction results into a NetworkX DiGraph."""
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
            properties=rel.properties,
            document_id=document_id,
        )

    return G


def networkx_to_neo4j_import(
    graph: nx.DiGraph,
    artifact_id: str,
) -> dict[str, Any]:
    """Convert NetworkX graph to Neo4j import format.

    Returns a dict with 'nodes' and 'edges' lists ready for
    neo4j_graph.upsert_node() / upsert_relationship().
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
            "properties": data.get("properties", {}),
            "artifact_id": artifact_id,
        })

    return {"nodes": nodes, "edges": edges}
