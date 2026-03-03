"""Aggregate all v1 API routers under the /v1 prefix."""

from fastapi import APIRouter

from app.api.v1 import agent, governance, health, retrieval, sources

api_router = APIRouter(prefix="/v1")

api_router.include_router(health.router)
api_router.include_router(sources.router)
api_router.include_router(retrieval.router)
api_router.include_router(governance.router)
api_router.include_router(agent.router)
