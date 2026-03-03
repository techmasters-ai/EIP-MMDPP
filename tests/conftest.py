"""Shared pytest fixtures for all test layers."""

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Override settings before any app imports resolve settings
# ---------------------------------------------------------------------------
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("TEXT_EMBEDDING_MODEL", "paraphrase-MiniLM-L6-v2")
os.environ.setdefault("TEXT_EMBEDDING_DIM", "384")


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_db_url() -> str:
    return os.environ.get(
        "DATABASE_URL_SYNC",
        "postgresql+psycopg2://eip_test:eip_test_secret@localhost:5433/eip_test",
    )


@pytest.fixture(scope="session")
def test_async_db_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://eip_test:eip_test_secret@localhost:5433/eip_test",
    )


@pytest.fixture(scope="session")
def sync_engine(test_db_url):
    engine = create_engine(test_db_url, echo=False)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def async_engine(test_async_db_url):
    engine = create_async_engine(test_async_db_url, echo=False)
    yield engine


@pytest.fixture
def db_session(sync_engine) -> Generator[Session, None, None]:
    """Synchronous DB session that rolls back after each test."""
    connection = sync_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest_asyncio.fixture
async def async_db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Async DB session that rolls back after each test."""
    async with async_engine.connect() as conn:
        await conn.begin()
        session_factory = async_sessionmaker(
            bind=conn, class_=AsyncSession, expire_on_commit=False
        )
        async with session_factory() as session:
            try:
                yield session
            finally:
                await session.rollback()


# ---------------------------------------------------------------------------
# FastAPI app fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    from app.main import create_app
    return create_app()


@pytest.fixture
def client(app, async_db_session) -> TestClient:
    """Synchronous test client with DB override."""
    from app.db.session import get_async_session

    async def override_get_db():
        yield async_db_session

    app.dependency_overrides[get_async_session] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(app, async_db_session) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client with DB override."""
    from app.db.session import get_async_session

    async def override_get_db():
        yield async_db_session

    app.dependency_overrides[get_async_session] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Mocked external services
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_minio(monkeypatch):
    """Mock MinIO storage calls."""
    monkeypatch.setattr(
        "app.services.storage.upload_bytes_sync",
        lambda data, bucket, key, content_type="application/octet-stream": key,
    )
    monkeypatch.setattr(
        "app.services.storage.download_bytes_sync",
        lambda bucket, key: b"%PDF-1.4 mock pdf content",
    )
    monkeypatch.setattr(
        "app.services.storage.stream_upload_async",
        lambda stream, bucket, key, content_type="application/octet-stream": (
            key,
            1024,
            "abc123" * 10 + "abcd",
        ),
    )
    return MagicMock()


@pytest.fixture
def mock_celery(monkeypatch):
    """Mock Celery task submission (don't actually enqueue tasks)."""
    mock_result = MagicMock()
    mock_result.id = "mock-task-id-12345"

    monkeypatch.setattr(
        "app.workers.pipeline.start_ingest_pipeline",
        lambda document_id: "mock-task-id-12345",
    )
    return mock_result


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Mock embedding model calls with deterministic zero vectors."""
    import random

    def fake_embed_texts(texts, batch_size=64):
        # Return deterministic pseudo-random 384-dim vectors (test model dim)
        result = []
        for text in texts:
            rng = random.Random(hash(text) % (2**31))
            result.append([rng.uniform(-1, 1) for _ in range(384)])
        return result

    def fake_embed_query(query):
        return fake_embed_texts([query])[0]

    monkeypatch.setattr("app.services.embedding.embed_texts", fake_embed_texts)
    monkeypatch.setattr("app.services.embedding.embed_query", fake_embed_query)
    return MagicMock()


# ---------------------------------------------------------------------------
# Fixture document helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixture_dir() -> "Path":
    from pathlib import Path
    return Path(__file__).parent / "fixtures" / "documents"


@pytest.fixture(scope="session")
def sample_pdf_bytes(fixture_dir) -> bytes:
    return (fixture_dir / "sample_technical_manual.pdf").read_bytes()


@pytest.fixture(scope="session")
def sample_png_bytes(fixture_dir) -> bytes:
    return (fixture_dir / "sample_schematic.png").read_bytes()


@pytest.fixture(scope="session")
def sample_docx_bytes(fixture_dir) -> bytes:
    return (fixture_dir / "sample_docx.docx").read_bytes()
