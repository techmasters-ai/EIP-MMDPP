"""Shared pytest fixtures for all test layers."""

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

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
os.environ.setdefault("IMAGE_EMBEDDING_DIM", "128")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("MEMORY_ENABLED", "true")


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
    """Mock all embedding model calls with deterministic vectors."""
    import random

    text_dim = int(os.environ.get("TEXT_EMBEDDING_DIM", "384"))
    image_dim = int(os.environ.get("IMAGE_EMBEDDING_DIM", "128"))

    def fake_embed_texts(texts, batch_size=64, *, query=False):
        result = []
        for t in texts:
            rng = random.Random(hash(t) % (2**31))
            result.append([rng.uniform(-1, 1) for _ in range(text_dim)])
        return result

    def fake_embed_query(query):
        return fake_embed_texts([query], query=True)[0]

    def fake_embed_images(images):
        result = []
        for img in images:
            rng = random.Random(42)
            result.append([rng.uniform(-1, 1) for _ in range(image_dim)])
        return result

    def fake_embed_text_for_clip(text_input):
        rng = random.Random(hash(text_input) % (2**31))
        return [rng.uniform(-1, 1) for _ in range(image_dim)]

    monkeypatch.setattr("app.services.embedding.embed_texts", fake_embed_texts)
    monkeypatch.setattr("app.services.embedding.embed_query", fake_embed_query)
    monkeypatch.setattr("app.services.embedding.embed_images", fake_embed_images)
    monkeypatch.setattr("app.services.embedding.embed_text_for_clip", fake_embed_text_for_clip)
    return MagicMock()



@pytest.fixture
def mock_graph_store():
    """Mock GraphStore for tests.

    Returns an AsyncMock that implements all GraphStore Protocol methods.
    """
    store = AsyncMock()
    store.upsert_node = AsyncMock(return_value="test-rid")
    store.upsert_nodes_batch = AsyncMock(return_value=[])
    store.upsert_relationship = AsyncMock(return_value="test-rid")
    store.upsert_relationships_batch = AsyncMock(return_value=[])
    store.fulltext_search = AsyncMock(return_value=[])
    store.search_by_alias = AsyncMock(return_value=[])
    store.resolve_root_entity = AsyncMock(return_value=None)
    store.get_neighborhood = AsyncMock(return_value=[])
    store.get_neighborhood_graph = AsyncMock(return_value={"nodes": [], "edges": []})
    store.get_ontology_linked_chunks = AsyncMock(return_value=[])
    store.get_graph_stats = AsyncMock(return_value={"vertex_count": 0, "edge_count": 0})
    store.delete_document_graph = AsyncMock(return_value=0)
    store.create_alias = AsyncMock()
    store.set_canonical_name = AsyncMock()
    # Sync variants
    store.upsert_node_sync = MagicMock(return_value="test-rid")
    store.upsert_nodes_batch_sync = MagicMock(return_value=[])
    store.upsert_relationships_batch_sync = MagicMock(return_value=[])
    store.create_structural_edge_sync = MagicMock(return_value="test-rid")
    store.delete_document_graph_sync = MagicMock(return_value=0)
    store.fulltext_search_sync = MagicMock(return_value=[])
    store.search_by_alias_sync = MagicMock(return_value=[])
    store.create_alias_sync = MagicMock()
    store.set_canonical_name_sync = MagicMock()
    store.create_text_chunk_vertex_sync = MagicMock(return_value="test-rid")
    store.create_image_chunk_vertex_sync = MagicMock(return_value="test-rid")
    return store


@pytest.fixture
def mock_docling_graph(monkeypatch):
    """Mock docling-graph HTTP client to return a canned extraction response."""

    def fake_extract(text, document_id, *, ontology_version=None):
        return {
            "entities": [
                {
                    "name": "Test System",
                    "entity_type": "EQUIPMENT_SYSTEM",
                    "confidence": 0.9,
                    "properties": {"designation": "TS-001"},
                },
                {
                    "name": "Test Component",
                    "entity_type": "COMPONENT",
                    "confidence": 0.85,
                    "properties": {"part_number": "TC-001"},
                },
            ],
            "relationships": [
                {
                    "from_name": "Test System",
                    "from_type": "EQUIPMENT_SYSTEM",
                    "rel_type": "CONTAINS",
                    "to_name": "Test Component",
                    "to_type": "COMPONENT",
                    "confidence": 0.8,
                },
            ],
            "ontology_version": "3.0.0",
            "model": "mock",
            "provider": "mock",
        }

    monkeypatch.setattr(
        "app.services.docling_graph_service.extract_graph",
        fake_extract,
    )
    return MagicMock()


# ---------------------------------------------------------------------------
# Shared pipeline-run factory fixture
# ---------------------------------------------------------------------------

_FACTORY_DUMMY_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def pipeline_run_factory(db_session):
    """Return a callable that creates one Source → Document → PipelineRun chain.

    Usage::

        def test_something(pipeline_run_factory):
            run_id = pipeline_run_factory()
            # or with custom status:
            run_id = pipeline_run_factory(status="FAILED")

    All rows are rolled back after the test by the ``db_session`` fixture.
    """
    from sqlalchemy import text as sa_text

    def _make(status: str = "PROCESSING") -> "uuid.UUID":
        source_id = uuid.uuid4()
        document_id = uuid.uuid4()
        pipeline_run_id = uuid.uuid4()

        db_session.execute(
            sa_text(
                "INSERT INTO ingest.sources (id, name, created_by) "
                "VALUES (:id, :name, :created_by)"
            ),
            {
                "id": source_id,
                "name": f"test-source-{source_id}",
                "created_by": _FACTORY_DUMMY_USER_ID,
            },
        )
        db_session.execute(
            sa_text(
                "INSERT INTO ingest.documents "
                "(id, source_id, filename, storage_bucket, storage_key, uploaded_by, retry_count) "
                "VALUES (:id, :source_id, :filename, :bucket, :key, :user_id, :retry_count)"
            ),
            {
                "id": document_id,
                "source_id": source_id,
                "filename": "test.pdf",
                "bucket": "test-bucket",
                "key": f"test/{document_id}.pdf",
                "user_id": _FACTORY_DUMMY_USER_ID,
                "retry_count": 0,
            },
        )
        db_session.execute(
            sa_text(
                "INSERT INTO ingest.pipeline_runs (id, document_id, pipeline_version, status) "
                "VALUES (:id, :document_id, :version, :status)"
            ),
            {
                "id": pipeline_run_id,
                "document_id": document_id,
                "version": "v2",
                "status": status,
            },
        )
        db_session.flush()
        return pipeline_run_id

    return _make


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
