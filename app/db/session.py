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
# Neo4j driver singletons (sync for Celery, async for FastAPI)
# ---------------------------------------------------------------------------
_neo4j_driver = None
_neo4j_async_driver = None


def get_neo4j_driver():
    """Return a sync Neo4j driver (for Celery workers)."""
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase

        _neo4j_driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Neo4j sync driver created: %s", settings.neo4j_uri)
    return _neo4j_driver


def get_neo4j_async_driver():
    """Return an async Neo4j driver (for FastAPI endpoints)."""
    global _neo4j_async_driver
    if _neo4j_async_driver is None:
        from neo4j import AsyncGraphDatabase

        _neo4j_async_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info("Neo4j async driver created: %s", settings.neo4j_uri)
    return _neo4j_async_driver


# ---------------------------------------------------------------------------
# Qdrant client singletons (sync for Celery, async for FastAPI)
# ---------------------------------------------------------------------------
_qdrant_client = None
_qdrant_async_client = None


def get_qdrant_client():
    """Return a sync Qdrant client (for Celery workers)."""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient

        _qdrant_client = QdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)
        logger.info("Qdrant sync client created: %s (timeout=%ss)", settings.qdrant_url, settings.qdrant_timeout_seconds)
    return _qdrant_client


def get_qdrant_async_client():
    """Return an async Qdrant client (for FastAPI endpoints)."""
    global _qdrant_async_client
    if _qdrant_async_client is None:
        from qdrant_client import AsyncQdrantClient

        _qdrant_async_client = AsyncQdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_seconds)
        logger.info("Qdrant async client created: %s (timeout=%ss)", settings.qdrant_url, settings.qdrant_timeout_seconds)
    return _qdrant_async_client
