"""Aggregate all v1 API routers under the /v1 prefix."""

from fastapi import APIRouter

from app.api.v1 import (
    agent,
    governance,
    graph_store,
    health,
    image_store,
    memory,
    retrieval,
    sources,
    text_store,
)

api_router = APIRouter(prefix="/v1")

api_router.include_router(health.router)
api_router.include_router(sources.router)
api_router.include_router(text_store.router)
api_router.include_router(image_store.router)
api_router.include_router(graph_store.router)
api_router.include_router(memory.router)
api_router.include_router(retrieval.router)
api_router.include_router(governance.router)
api_router.include_router(agent.router)
