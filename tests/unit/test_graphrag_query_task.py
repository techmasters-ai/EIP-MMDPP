"""Unit tests for async GraphRAG query task, endpoints, and helpers."""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# Stub graphrag/pandas if not available (same pattern as test_query_coverage.py).
# Only install stubs when the real packages are genuinely unavailable to avoid
# contaminating other test modules that rely on real imports.
class _AutoStubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        mock = MagicMock()
        setattr(self, name, mock)
        return mock


def _try_import(name: str) -> bool:
    """Return True if the top-level package can be imported."""
    top = name.split(".")[0]
    try:
        __import__(top)
        return True
    except ImportError:
        return False


for mod_name in [
    "pandas", "graphrag", "graphrag.api", "graphrag.api.prompt_tune",
    "graphrag.config", "graphrag.config.enums",
    "graphrag.config.models", "graphrag.config.models.cluster_graph_config",
    "graphrag.config.models.extract_graph_config",
    "graphrag.config.models.drift_search_config",
    "graphrag.config.models.graph_rag_config",
    "graphrag.config.models.local_search_config",
    "graphrag.config.models.reporting_config",
    "graphrag.config.models.llm_config",
    "graphrag.config.models.llm_parameters_config",
    "graphrag.config.models.parallelization_parameters_config",
    "graphrag.config.models.embeddings_config",
    "graphrag.config.models.text_embedding_config",
    "graphrag.query", "litellm",
    "nest_asyncio2", "graphrag.index", "graphrag.index.update",
    "graphrag.index.update.incremental_index",
    "graphrag_llm", "graphrag_llm.config", "graphrag_llm.config.model_config",
    "graphrag_llm.embedding", "graphrag_llm.embedding.lite_llm_embedding",
    "graphrag_cache", "graphrag_cache.cache_config",
    "graphrag_storage", "graphrag_storage.storage_config",
    "graphrag_vectors", "graphrag_vectors.vector_store_config",
    "lancedb",
]:
    if not _try_import(mod_name):
        sys.modules.setdefault(mod_name, _AutoStubModule(mod_name))

# Pre-import graphrag_service so @patch can resolve it, and stub Redis so the
# module-level cleanup_stale_locks() call in graphrag_tasks doesn't fail.
with patch("redis.from_url", return_value=MagicMock()):
    import app.services.graphrag_service  # noqa: E402  (must come after stubs)
    import app.workers.graphrag_tasks  # noqa: E402


# ---------------------------------------------------------------------------
# Celery task tests
# ---------------------------------------------------------------------------

class TestRunGraphRAGQueryTask:
    """Tests for run_graphrag_query_task Celery task."""

    @patch("app.services.graphrag_service.local_search")
    def test_local_search_success(self, mock_search):
        mock_search.return_value = {
            "response": "Fan Song is a radar system.",
            "context": {"entities": []},
        }
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_local", "query_text": "Fan Song"}
        )
        assert result["strategy"] == "graphrag_local"
        assert result["total"] == 1
        assert result["results"][0]["content_text"] == "Fan Song is a radar system."
        assert result["results"][0]["modality"] == "graphrag_response"
        assert "error" not in result

    @patch("app.services.graphrag_service.global_search")
    def test_global_search_success(self, mock_search):
        mock_search.return_value = {
            "response": "Community summary.",
            "context": {},
        }
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_global", "query_text": "overview"}
        )
        assert result["strategy"] == "graphrag_global"
        assert result["total"] == 1

    @patch("app.services.graphrag_service.drift_search")
    def test_drift_search_success(self, mock_search):
        mock_search.return_value = {
            "response": "Drift result.",
            "context": {"communities": []},
        }
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_drift", "query_text": "radar"}
        )
        assert result["strategy"] == "graphrag_drift"
        assert result["total"] == 1
        assert result["results"][0]["context"]["source"] == "graphrag_drift"

    @patch("app.services.graphrag_service.basic_search")
    def test_basic_search_success(self, mock_search):
        mock_search.return_value = {
            "response": "Basic text unit match.",
            "context": {},
        }
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_basic", "query_text": "missile"}
        )
        assert result["strategy"] == "graphrag_basic"
        assert result["total"] == 1
        assert result["results"][0]["context"]["source"] == "graphrag_basic"

    @patch("app.services.graphrag_service.local_search")
    def test_communities_not_indexed_error(self, mock_search):
        mock_search.return_value = {"response": "", "error": "communities_not_indexed"}
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_local", "query_text": "test"}
        )
        assert "error" in result
        assert "indexing has not completed" in result["error"]

    @patch("app.services.graphrag_service.drift_search")
    def test_drift_empty_returns_no_error(self, mock_search):
        mock_search.return_value = {"response": ""}
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_drift", "query_text": "test"}
        )
        assert "error" not in result
        assert result["total"] == 0

    @patch("app.services.graphrag_service.basic_search")
    def test_basic_empty_returns_no_error(self, mock_search):
        mock_search.return_value = {"response": ""}
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_basic", "query_text": "test"}
        )
        assert "error" not in result
        assert result["total"] == 0

    @patch("app.services.graphrag_service.global_search")
    def test_global_empty_returns_error(self, mock_search):
        mock_search.return_value = {"response": ""}
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_global", "query_text": "test"}
        )
        assert "error" in result
        assert "no results found" in result["error"]

    def test_invalid_strategy(self):
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "basic", "query_text": "test"}
        )
        assert "error" in result
        assert "Invalid" in result["error"]

    def test_missing_query_text(self):
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_local", "query_text": ""}
        )
        assert "error" in result

    @patch("app.services.graphrag_service.local_search")
    def test_min_confidence_filter(self, mock_search):
        mock_search.return_value = {
            "response": "Answer.",
            "context": {},
        }
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_local", "query_text": "test",
             "min_confidence": 1.5}
        )
        assert result["total"] == 0  # score=1.0 < 1.5

    @patch("app.services.graphrag_service.local_search")
    def test_search_exception(self, mock_search):
        mock_search.side_effect = RuntimeError("LLM timeout")
        from app.workers.graphrag_tasks import run_graphrag_query_task
        result = run_graphrag_query_task(
            {"strategy": "graphrag_local", "query_text": "test"}
        )
        assert "error" in result
        assert "LLM timeout" in result["error"]


# ---------------------------------------------------------------------------
# _build_response helper
# ---------------------------------------------------------------------------

class TestBuildResponse:
    def test_builds_correct_shape(self):
        from app.workers.graphrag_tasks import _build_response
        result = _build_response("graphrag_local", "test query", [{"score": 1.0}])
        assert result["query_text"] == "test query"
        assert result["query_image"] is None
        assert result["strategy"] == "graphrag_local"
        assert result["modality_filter"] == "all"
        assert result["results"] == [{"score": 1.0}]
        assert result["total"] == 1

    def test_empty_results(self):
        from app.workers.graphrag_tasks import _build_response
        result = _build_response("graphrag_drift", "q", [])
        assert result["total"] == 0
        assert result["results"] == []


# ---------------------------------------------------------------------------
# Endpoint helpers (retrieval.py)
# ---------------------------------------------------------------------------

class TestCheckJobExists:
    def test_raises_404_when_key_missing(self):
        from fastapi import HTTPException
        import app.api.v1.retrieval as retrieval_mod

        mock_redis = MagicMock()
        mock_redis.exists.return_value = False
        retrieval_mod._graphrag_redis = mock_redis
        try:
            with pytest.raises(HTTPException) as exc_info:
                retrieval_mod._check_job_exists("nonexistent-id")
            assert exc_info.value.status_code == 404
        finally:
            retrieval_mod._graphrag_redis = None

    def test_passes_when_key_exists(self):
        import app.api.v1.retrieval as retrieval_mod

        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        retrieval_mod._graphrag_redis = mock_redis
        try:
            retrieval_mod._check_job_exists("valid-id")  # should not raise
        finally:
            retrieval_mod._graphrag_redis = None

    def test_passes_when_redis_unavailable(self):
        import app.api.v1.retrieval as retrieval_mod

        mock_redis = MagicMock()
        mock_redis.exists.side_effect = ConnectionError("Redis down")
        retrieval_mod._graphrag_redis = mock_redis
        try:
            retrieval_mod._check_job_exists("any-id")  # should not raise
        finally:
            retrieval_mod._graphrag_redis = None


class TestCeleryStatusMap:
    def test_maps_all_known_states(self):
        from app.api.v1.retrieval import _CELERY_STATUS_MAP
        assert _CELERY_STATUS_MAP["PENDING"] == "pending"
        assert _CELERY_STATUS_MAP["STARTED"] == "running"
        assert _CELERY_STATUS_MAP["SUCCESS"] == "completed"
        assert _CELERY_STATUS_MAP["FAILURE"] == "failed"
        assert _CELERY_STATUS_MAP["REVOKED"] == "failed"

    def test_unknown_state_defaults_to_pending(self):
        from app.api.v1.retrieval import _CELERY_STATUS_MAP
        assert _CELERY_STATUS_MAP.get("UNKNOWN_STATE", "pending") == "pending"


# ---------------------------------------------------------------------------
# Endpoint tests (using FastAPI TestClient)
# ---------------------------------------------------------------------------

pytest.importorskip("asyncpg", reason="asyncpg not installed")


class TestSubmitEndpoint:
    @patch("app.api.v1.retrieval._get_graphrag_redis")
    @patch("app.workers.graphrag_tasks.run_graphrag_query_task")
    def test_submit_returns_202(self, mock_task_cls, mock_redis_fn):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_task = MagicMock()
        mock_task.id = "test-job-id-123"
        mock_task_cls.delay.return_value = mock_task
        mock_redis_fn.return_value = MagicMock()

        client = TestClient(app)
        resp = client.post("/v1/retrieval/graphrag/submit", json={
            "query_text": "Fan Song",
            "strategy": "graphrag_local",
        })
        assert resp.status_code == 202
        data = resp.json()
        assert data["job_id"] == "test-job-id-123"
        assert data["status"] == "pending"

    @patch("app.api.v1.retrieval._get_graphrag_redis")
    @patch("app.workers.graphrag_tasks.run_graphrag_query_task")
    def test_submit_rejects_non_graphrag_strategy(self, mock_task_cls, mock_redis_fn):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        resp = client.post("/v1/retrieval/graphrag/submit", json={
            "query_text": "test",
            "strategy": "basic",
        })
        assert resp.status_code == 400


class TestStatusEndpoint:
    @patch("app.workers.celery_app.celery_app")
    @patch("app.api.v1.retrieval._get_graphrag_redis")
    def test_status_completed(self, mock_redis_fn, mock_celery):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis_fn.return_value = mock_redis

        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"results": [], "total": 0}
        mock_celery.AsyncResult.return_value = mock_result

        client = TestClient(app)
        resp = client.get("/v1/retrieval/graphrag/status/job-123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["error"] is None

    @patch("app.workers.celery_app.celery_app")
    @patch("app.api.v1.retrieval._get_graphrag_redis")
    def test_status_failed_with_error_in_result(self, mock_redis_fn, mock_celery):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis_fn.return_value = mock_redis

        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"error": "indexing not complete"}
        mock_celery.AsyncResult.return_value = mock_result

        client = TestClient(app)
        resp = client.get("/v1/retrieval/graphrag/status/job-123")
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error"] == "indexing not complete"

    @patch("app.api.v1.retrieval._get_graphrag_redis")
    def test_status_404_for_bogus_id(self, mock_redis_fn):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_redis = MagicMock()
        mock_redis.exists.return_value = False
        mock_redis_fn.return_value = mock_redis

        client = TestClient(app)
        resp = client.get("/v1/retrieval/graphrag/status/bogus")
        assert resp.status_code == 404


class TestResultEndpoint:
    @patch("app.workers.celery_app.celery_app")
    @patch("app.api.v1.retrieval._get_graphrag_redis")
    def test_result_returns_data(self, mock_redis_fn, mock_celery):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis_fn.return_value = mock_redis

        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {
            "query_text": "test", "strategy": "graphrag_local",
            "modality_filter": "all", "results": [], "total": 0,
        }
        mock_celery.AsyncResult.return_value = mock_result

        client = TestClient(app)
        resp = client.get("/v1/retrieval/graphrag/result/job-123")
        assert resp.status_code == 200
        assert resp.json()["strategy"] == "graphrag_local"

    @patch("app.workers.celery_app.celery_app")
    @patch("app.api.v1.retrieval._get_graphrag_redis")
    def test_result_409_when_not_complete(self, mock_redis_fn, mock_celery):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis_fn.return_value = mock_redis

        mock_result = MagicMock()
        mock_result.state = "STARTED"
        mock_celery.AsyncResult.return_value = mock_result

        client = TestClient(app)
        resp = client.get("/v1/retrieval/graphrag/result/job-123")
        assert resp.status_code == 409

    @patch("app.workers.celery_app.celery_app")
    @patch("app.api.v1.retrieval._get_graphrag_redis")
    def test_result_422_when_task_returned_error(self, mock_redis_fn, mock_celery):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_redis = MagicMock()
        mock_redis.exists.return_value = True
        mock_redis_fn.return_value = mock_redis

        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"error": "no results found"}
        mock_celery.AsyncResult.return_value = mock_result

        client = TestClient(app)
        resp = client.get("/v1/retrieval/graphrag/result/job-123")
        assert resp.status_code == 422

    @patch("app.api.v1.retrieval._get_graphrag_redis")
    def test_result_404_for_bogus_id(self, mock_redis_fn):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_redis = MagicMock()
        mock_redis.exists.return_value = False
        mock_redis_fn.return_value = mock_redis

        client = TestClient(app)
        resp = client.get("/v1/retrieval/graphrag/result/bogus")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Monkey-patch test
# ---------------------------------------------------------------------------

class TestPatchConcatDataframes:
    def test_patch_applied(self):
        """Verify the monkey-patch replaced concat_dataframes."""
        from graphrag.index.update import incremental_index
        # The patched function is named _safe_concat
        assert incremental_index.concat_dataframes.__name__ == "_safe_concat"
