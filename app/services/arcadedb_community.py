"""Community detection via ArcadeDB native algorithms + LLM report generation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Domain entity types for community detection (exclude structural types).
# Derived from the schema module's single source of truth.
from app.services.arcadedb_schema import STRUCTURAL_TYPES as _STRUCTURAL_TYPES


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

        # Generate LLM report (fetches relationships internally)
        report = await _generate_community_report(graph_store, cid, members)
        if not report:
            continue

        # Embed the report (title + summary) for vector search
        report_embedding = await _embed_report(
            f"{report['title']}\n\n{report['summary']}"
        )

        # Upsert CommunityReport vertex
        try:
            report_rid = await graph_store.upsert_community_report(
                community_id=cid,
                title=report["title"],
                summary=report["summary"],
                member_count=len(members),
                membership_hash=new_hash,
                model_name=settings.community_report_llm_model,
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

    # Fetch real relationships between community members
    relationships_text = "(no relationships found among members)"
    try:
        names = [m["name"] for m in members if m.get("name")]
        edges = await graph_store.get_relationships_between_entities(names)
        if edges:
            relationships_text = "\n".join(
                f"- {e.get('from_name', '?')} "
                f"({e.get('from_type', '?')}) "
                f"--[{e.get('rel_type', '?')}]--> "
                f"{e.get('to_name', '?')} "
                f"({e.get('to_type', '?')})"
                for e in edges
            )
    except Exception as exc:
        logger.warning(
            "Failed to fetch relationships for community %d: %s",
            community_id, exc,
        )

    prompt = (
        prompt_template
        .replace("{entities}", entities_text)
        .replace("{relationships}", relationships_text)
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
    """Call Ollama chat API to generate a community report.

    Expects JSON output ``{"title": "...", "summary": "..."}``; falls back to
    line-based parsing when the model doesn't return valid JSON.
    """
    from app.config import get_settings
    from app.services.llm_json import parse_llm_json_loose

    settings = get_settings()
    url = f"{settings.get_ollama_llm_url()}/v1/chat/completions"
    timeout = settings.doc_analysis_timeout

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            url,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a knowledge-graph analyst. "
                            "Respond with a single JSON object containing "
                            '"title" (short, descriptive) and "summary" '
                            "(2-4 paragraphs). Do not include any other text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": settings.llm_max_tokens,
            },
        )
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
        content = (message.get("content") or message.get("reasoning_content") or "").strip()

    parsed = parse_llm_json_loose(content)
    if isinstance(parsed, dict) and parsed.get("title") and parsed.get("summary"):
        return {
            "title": str(parsed["title"]).strip(),
            "summary": str(parsed["summary"]).strip(),
        }

    # Fallback: first line is the title, remainder is the summary
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return {"title": "Community Report", "summary": content or ""}
    return {
        "title": lines[0][:200],
        "summary": "\n".join(lines[1:]) if len(lines) > 1 else lines[0],
    }


async def _embed_report(text: str) -> list[float] | None:
    """Embed report text for vector search.

    Uses the same BGE embedding pipeline as document text. Runs the sync
    ``embed_texts`` call in a thread so it doesn't block the event loop.
    """
    if not text or not text.strip():
        return None
    try:
        from app.services.embedding import embed_texts
        vectors = await asyncio.to_thread(embed_texts, [text])
        return vectors[0] if vectors else None
    except Exception as exc:
        logger.warning("Failed to embed community report: %s", exc)
        return None


_DEFAULT_PROMPT = """You are analyzing a cluster of related entities in a knowledge graph.

Community members:
{entities}

Relationships among members:
{relationships}

{evidence}

Produce a JSON object with:
- "title": short, descriptive (under 80 characters)
- "summary": 2-4 paragraphs covering the key systems, their relationships, and operational significance.
"""


async def search_community_reports(
    graph_store: Any,
    query_vector: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search community reports by vector similarity."""
    return await graph_store.search_community_reports_by_vector(
        query_vector, top_k,
    )
