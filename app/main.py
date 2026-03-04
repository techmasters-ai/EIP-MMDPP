"""FastAPI application factory."""

import logging
import os

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if settings.app_env == "development"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(message)s",
)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    application = FastAPI(
        title="EIP-MMDPP",
        description=(
            "Multi-modal document processing and retrieval platform. "
            "Ingest, parse, embed, and query military/defense documents."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router)

    # Serve the React SPA when the built frontend exists.
    # In development (no dist/), the Vite dev server proxies to the API directly.
    _frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
    _frontend_dist = os.path.abspath(_frontend_dist)
    if os.path.isdir(_frontend_dist):
        # /assets served first so the catch-all SPA mount never intercepts asset requests
        _assets = os.path.join(_frontend_dist, "assets")
        if os.path.isdir(_assets):
            application.mount("/assets", StaticFiles(directory=_assets), name="assets")
        # Catch-all: serve index.html for any unmatched path (client-side routing)
        application.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="spa")

    @application.on_event("startup")
    async def on_startup() -> None:
        log = structlog.get_logger()
        log.info("EIP-MMDPP API starting", env=settings.app_env)

    @application.on_event("shutdown")
    async def on_shutdown() -> None:
        from app.db.session import async_engine
        await async_engine.dispose()

    return application


app = create_app()
