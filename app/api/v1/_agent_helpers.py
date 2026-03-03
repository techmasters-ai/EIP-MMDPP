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
        elif item.context and isinstance(item.context.get("entity"), dict):
            entity = item.context["entity"]
            name = (
                entity.get("properties", {}).get("name", "")
                or entity.get("name", "")
            )
            rel_type = item.context.get("rel_type", "")
            neighbor = item.context.get("neighbor")
            neighbor_name = ""
            if isinstance(neighbor, dict):
                neighbor_name = (
                    neighbor.get("properties", {}).get("name", "")
                    or neighbor.get("name", "")
                )
            if name:
                lines.append(f"\n**Entity**: {name}")
            if rel_type and neighbor_name:
                lines.append(f"**Relationship**: {name} \u2013[{rel_type}]\u2192 {neighbor_name}")
            lines.append("")
        else:
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
