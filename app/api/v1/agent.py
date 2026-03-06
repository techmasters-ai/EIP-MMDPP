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
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1._agent_helpers import (
    AgentContextResponse,
    AgentSource,  # noqa: F401 — re-exported for API consumers
    build_markdown,
    build_sources,
)
from app.db.session import get_async_session
from app.schemas.retrieval import ModalityFilter, QueryStrategy, UnifiedQueryRequest, _MODE_MAP

router = APIRouter(tags=["agent"])
logger = logging.getLogger(__name__)


@router.get("/agent/context", response_model=AgentContextResponse)
async def get_agent_context(
    query: str = Query(..., min_length=1, max_length=4096, description="Search query"),
    strategy: QueryStrategy = Query(QueryStrategy.basic, description="Retrieval strategy"),
    modality_filter: ModalityFilter = Query(ModalityFilter.all, description="Filter results by modality"),
    mode: Optional[str] = Query(None, description="Deprecated: use strategy + modality_filter"),
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
    - **strategy**: `basic` | `hybrid` | `memory` | `graphrag_local` | `graphrag_global`
    - **modality_filter**: `all` | `text` | `image`
    - **top_k**: Number of results to include (1–50, default: 10)
    - **include_sources**: Whether to include the `sources` list (default: true)
    """
    # Backward-compat: map legacy mode to strategy + modality_filter
    if mode:
        if mode not in _MODE_MAP:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown mode: '{mode}'. Valid modes: {', '.join(sorted(_MODE_MAP.keys()))}",
            )
        if strategy == QueryStrategy.basic and modality_filter == ModalityFilter.all:
            strategy, modality_filter = _MODE_MAP[mode]

    body = UnifiedQueryRequest(
        query_text=query,
        strategy=strategy,
        modality_filter=modality_filter,
        top_k=top_k,
        include_context=True,
    )

    from app.api.v1.retrieval import (
        _graphrag_global_query,
        _graphrag_local_query,
        _memory_query,
        _multi_modal_pipeline,
        _text_vector_search,
    )

    if strategy == QueryStrategy.memory:
        results = await _memory_query(body)
    elif strategy == QueryStrategy.basic:
        results = await _text_vector_search(db, body)
    elif strategy == QueryStrategy.hybrid:
        results = await _multi_modal_pipeline(db, body)
    elif strategy == QueryStrategy.graphrag_local:
        results = await _graphrag_local_query(db, body)
    elif strategy == QueryStrategy.graphrag_global:
        results = await _graphrag_global_query(db, body)
    else:
        results = await _text_vector_search(db, body)

    context_md = build_markdown(query, results)
    sources = build_sources(results) if include_sources else []

    return AgentContextResponse(
        query=query,
        strategy=strategy.value,
        modality_filter=modality_filter.value,
        total_results=len(results),
        context=context_md,
        sources=sources,
    )
