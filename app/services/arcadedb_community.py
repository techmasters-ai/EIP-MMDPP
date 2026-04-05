"""Community detection via ArcadeDB native algorithms + LLM report generation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Domain entity types for community detection (exclude structural types)
_STRUCTURAL_TYPES = {
    "Document", "TextChunk", "ImageChunk", "TrustedTextChunk",
    "Alias", "CommunityReport", "TABLE_REF",
}


def _compute_membership_hash(members: list[dict[str, str]]) -> str:
    """Hash sorted (entity_type, name) tuples for change detection."""
    key = json.dumps(
        sorted([(m["entity_type"], m["name"]) for m in members]),
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()


async def run_community_detection(
    graph_store: Any,
    mode: str = "incremental",
) -> dict[str, Any]:
    """Run community detection on domain entities.

    Args:
        graph_store: GraphStore instance
        mode: "incremental" (only regenerate changed reports) or "full" (regenerate all)

    Returns:
        Detection results with stats.
    """
    from app.config import get_settings
    settings = get_settings()

    algorithm = settings.community_detection_algorithm  # "leiden" or "louvain"
    resolution = settings.community_detection_resolution
    max_iterations = settings.community_detection_max_iterations

    # Build algorithm params
    algo_params: dict[str, Any] = {"maxIterations": max_iterations}
    if algorithm == "leiden":
        algo_params["resolution"] = resolution

    # Step 1: Run community detection algorithm on domain entities
    try:
        results = await graph_store.run_community_algorithm(algorithm, algo_params)
    except Exception as exc:
        logger.error("Community detection algorithm failed: %s", exc)
        return {"status": "FAILED", "error": str(exc)}

    # Step 2: Group by community, filtering out structural types
    communities: dict[int, list[dict[str, str]]] = {}
    for row in results:
        cid = row.get("community_id")
        if cid is None:
            continue
        etype = row.get("entity_type", "")
        if etype in _STRUCTURAL_TYPES:
            continue
        communities.setdefault(cid, []).append({
            "name": row.get("name", ""),
            "entity_type": etype,
        })

    # Step 3: Load existing hashes for incremental diffing
    try:
        existing_rows = await graph_store.get_community_reports()
        existing_hashes: dict[int, str] = {
            r["community_id"]: r["membership_hash"] for r in existing_rows
        }
    except Exception as exc:
        logger.warning("Could not load existing community reports: %s", exc)
        existing_hashes = {}

    reports_generated = 0
    reports_reused = 0

    for cid, members in communities.items():
        new_hash = _compute_membership_hash(members)

        if mode == "incremental" and existing_hashes.get(cid) == new_hash:
            reports_reused += 1
            continue

        # Generate LLM report
        report = await _generate_community_report(graph_store, cid, members)
        if not report:
            continue

        # Embed the report summary
        report_embedding = await _embed_report(report["summary"])

        # Upsert CommunityReport vertex
        try:
            report_rid = await graph_store.upsert_community_report(
                community_id=cid,
                title=report["title"],
                summary=report["summary"],
                member_count=len(members),
                membership_hash=new_hash,
                model_name=os.environ.get(
                    "COMMUNITY_REPORT_LLM_MODEL",
                    settings.community_report_llm_model,
                ),
            )
        except Exception as exc:
            logger.warning("Failed to upsert community report %d: %s", cid, exc)
            continue

        # Attach embedding if we have one
        if report_embedding and report_rid:
            try:
                await graph_store.set_vertex_embedding(
                    "CommunityReport",
                    report_rid,
                    "report_embedding",
                    report_embedding,
                )
            except Exception as exc:
                logger.warning("Failed to set embedding for community %d: %s", cid, exc)

        reports_generated += 1

    return {
        "status": "COMPLETE",
        "total_communities": len(communities),
        "reports_generated": reports_generated,
        "reports_reused": reports_reused,
    }


async def _generate_community_report(
    graph_store: Any,
    community_id: int,
    members: list[dict[str, str]],
) -> dict[str, str] | None:
    """Generate an LLM summary report for a community."""
    from app.config import get_settings
    settings = get_settings()

    prompt_template = settings.community_report_llm_prompt or _DEFAULT_PROMPT

    entities_text = "\n".join(
        f"- {m['name']} ({m['entity_type']})" for m in members
    )

    prompt = (
        prompt_template
        .replace("{entities}", entities_text)
        .replace("{relationships}", "(relationships would be fetched from graph)")
        .replace("{evidence}", "")
    )

    try:
        return await _call_llm_for_report(prompt, settings.community_report_llm_model)
    except Exception as exc:
        logger.warning(
            "Failed to generate LLM report for community %d: %s; using fallback",
            community_id, exc,
        )
        return {"title": f"Community {community_id}", "summary": entities_text}


async def _call_llm_for_report(prompt: str, model: str) -> dict[str, str]:
    """Call LLM to generate community report.

    TODO: Implement actual LLM call via Ollama once community report prompting
    is finalised. For now returns a stub so the pipeline can run end-to-end.
    """
    return {
        "title": "Community Report",
        "summary": prompt[:500],
    }


async def _embed_report(text: str) -> list[float] | None:
    """Embed report text for vector search.

    TODO: Wire up Ollama embedding once report embeddings are needed for
    global-query vector search.
    """
    return None


_DEFAULT_PROMPT = """You are analyzing a cluster of related military equipment entities.

Community members:
{entities}

Relationships:
{relationships}

Generate:
1. Title (short, descriptive)
2. Summary (2-4 paragraphs covering systems, relationships, and operational significance)
"""


async def search_community_reports(
    graph_store: Any,
    query_vector: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search community reports by vector similarity."""
    return await graph_store.vector_search(
        "CommunityReport", "report_embedding",
        query_vector, top_k,
    )
