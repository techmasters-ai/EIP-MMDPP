from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Async engine + session factory (FastAPI)
# ---------------------------------------------------------------------------
async_engine = create_async_engine(
    settings.async_database_url,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=settings.sql_echo,
)

AsyncSessionFactory = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Sync engine + session factory (Celery workers)
# ---------------------------------------------------------------------------
sync_engine = create_engine(
    settings.sync_database_url,
    pool_size=10,
    max_overflow=5,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
)

SyncSessionFactory = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


def get_sync_session() -> Session:
    return SyncSessionFactory()


# ---------------------------------------------------------------------------
# ArcadeDB GraphStore singleton
# ---------------------------------------------------------------------------
_graph_store = None


def get_graph_store():
    """Returns the singleton GraphStore instance.

    The singleton pattern ensures httpx connection pools are reused across
    all FastAPI requests and Celery tasks within the same process, avoiding
    per-request TCP overhead.
    """
    global _graph_store
    if _graph_store is None:
        from app.services.arcadedb_client import ArcadeDBClient
        from app.services.arcadedb_graph import ArcadeDBGraphStore
        client = ArcadeDBClient(
            base_url=settings.arcadedb_url,
            username=settings.arcadedb_user,
            password=settings.arcadedb_password,
        )
        _graph_store = ArcadeDBGraphStore(client, settings.arcadedb_database)
        logger.info(
            "ArcadeDB GraphStore singleton created (url=%s, db=%s, pool=httpx-default)",
            settings.arcadedb_url,
            settings.arcadedb_database,
        )
    return _graph_store
