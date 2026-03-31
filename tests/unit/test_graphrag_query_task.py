"""Unit tests for async GraphRAG query task and endpoints."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

# Stub graphrag/pandas if not available (same pattern as test_query_coverage.py)
class _AutoStubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        mock = MagicMock()
        setattr(self, name, mock)
        return mock

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
    "nest_asyncio2", "graphrag.index.update",
    "graphrag.index.update.incremental_index",
    "graphrag_llm", "graphrag_llm.config", "graphrag_llm.config.model_config",
    "graphrag_llm.embedding", "graphrag_llm.embedding.lite_llm_embedding",
    "graphrag_cache", "graphrag_cache.cache_config",
    "graphrag_storage", "graphrag_storage.storage_config",
    "graphrag_vectors", "graphrag_vectors.vector_store_config",
    "lancedb",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _AutoStubModule(mod_name)

# Pre-import graphrag_service so @patch can resolve it, and stub Redis so the
# module-level cleanup_stale_locks() call in graphrag_tasks doesn't fail.
with patch("redis.from_url", return_value=MagicMock()):
    import app.services.graphrag_service  # noqa: E402  (must come after stubs)
    import app.workers.graphrag_tasks  # noqa: E402


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
