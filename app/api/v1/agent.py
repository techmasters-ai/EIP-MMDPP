"""LangGraph-compatible context retrieval endpoint.

GET /v1/agent/context returns a pre-formatted markdown context string that a
LangGraph (or any LLM agent) can inject directly into a system or human
message without further processing.

Design notes:
- GET not POST so it can be registered as an HTTP tool with query-string args.
- Calls the same internal query helpers as the main /v1/retrieval/query
  endpoint — no duplicated logic.
- The `context` field is markdown, not JSON, so the agent never needs to
  iterate over a results list.
- Pure helper functions live in _agent_helpers.py so unit tests can import
  them without pulling in the asyncpg/DB dependency chain.
"""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1._agent_helpers import (
    AgentContextResponse,
    AgentSource,  # noqa: F401 — re-exported for API consumers
    build_markdown,
    build_sources,
)
from app.db.session import get_async_session
from app.schemas.retrieval import QueryMode, UnifiedQueryRequest

router = APIRouter(tags=["agent"])
logger = logging.getLogger(__name__)


@router.get("/agent/context", response_model=AgentContextResponse)
async def get_agent_context(
    query: str = Query(..., min_length=1, max_length=4096, description="Search query"),
    mode: QueryMode = Query(QueryMode.text_basic, description="Retrieval mode"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results"),
    include_sources: bool = Query(True, description="Include sources list in response"),
    db: AsyncSession = Depends(get_async_session),
) -> AgentContextResponse:
    """Return a markdown-formatted retrieval context for LangGraph agents.

    The `context` field can be injected directly into an LLM prompt:

        system_message = f"Use this context to answer the question:\\n\\n{resp['context']}"

    Query parameters
    ----------------
    - **query**: Search query string
    - **mode**: `text_basic` | `text_only` | `images_only` | `multi_modal` | `memory`
    - **top_k**: Number of results to include (1–50, default: 10)
    - **include_sources**: Whether to include the `sources` list (default: true)
    """
    body = UnifiedQueryRequest(
        query_text=query,
        mode=mode,
        top_k=top_k,
        include_context=True,
    )

    from app.api.v1.retrieval import (
        _memory_query,
        _multi_modal_pipeline,
        _text_vector_search,
    )

    if mode == QueryMode.memory:
        results = await _memory_query(body)
    elif mode == QueryMode.text_basic:
        results = await _text_vector_search(db, body)
    elif mode in (QueryMode.text_only, QueryMode.images_only, QueryMode.multi_modal):
        results = await _multi_modal_pipeline(db, body)
    else:
        results = await _text_vector_search(db, body)

    context_md = build_markdown(query, results)
    sources = build_sources(results) if include_sources else []

    return AgentContextResponse(
        query=query,
        mode=mode.value,
        total_results=len(results),
        context=context_md,
        sources=sources,
    )
