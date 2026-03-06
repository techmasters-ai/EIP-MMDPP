"""Pure helper functions for the agent context endpoint.

Kept in a separate module so unit tests can import them without triggering
the DB session / asyncpg dependency chain.
"""

from app.schemas.common import APIModel
from app.schemas.retrieval import QueryResultItem
from typing import Optional


class AgentSource(APIModel):
    chunk_id: Optional[str] = None
    score: float
    modality: str
    classification: str


class AgentContextResponse(APIModel):
    query: str
    mode: str
    total_results: int
    context: str          # markdown string ready for LLM prompt injection
    sources: list[AgentSource]


def build_markdown(query: str, results: list[QueryResultItem]) -> str:
    """Format retrieval results as a markdown string for LLM injection."""
    if not results:
        return f"## Retrieved Context\n\nNo results found for query: *{query}*\n"

    lines: list[str] = ["## Retrieved Context\n"]
    for i, item in enumerate(results, start=1):
        score_pct = f"{item.score * 100:.0f}%"
        header = f"### Result {i} (score: {score_pct}, modality: {item.modality})"
        lines.append(header)

        if item.classification and item.classification != "UNCLASSIFIED":
            lines.append(f"**Classification**: {item.classification}")

        if item.page_number is not None:
            lines.append(f"**Page**: {item.page_number}")

        if item.content_text:
            lines.append(f"\n{item.content_text.strip()}\n")

        if item.context and item.context.get("source") == "ontology":
            rel_type = item.context.get("rel_type", "")
            entity_name = item.context.get("entity_name", "")
            related_name = item.context.get("related_name", "")
            if entity_name and rel_type and related_name:
                lines.append(f"**Via ontology**: {entity_name} --[{rel_type}]--> {related_name}")
            lines.append("")
        elif item.context and item.context.get("source") == "cross_modal":
            edge_type = item.context.get("edge_type", "")
            if edge_type:
                lines.append(f"**Via graph bridge**: {edge_type}")
            lines.append("")
        elif not item.content_text:
            lines.append("")

    return "\n".join(lines)


def build_sources(results: list[QueryResultItem]) -> list[AgentSource]:
    return [
        AgentSource(
            chunk_id=str(item.chunk_id) if item.chunk_id else None,
            score=item.score,
            modality=item.modality,
            classification=item.classification,
        )
        for item in results
    ]
