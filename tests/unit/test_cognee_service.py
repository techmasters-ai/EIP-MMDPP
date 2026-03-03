"""Unit tests for app.services.cognee_service.

All tests run with LLM_PROVIDER=mock so no real Cognee/LLM calls are made.
The module-level cognee import is mocked where needed to prevent import errors
when the cognee package is not installed.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_configured():
    """Reset the module-level _configured flag between tests."""
    import app.services.cognee_service as svc
    svc._configured = False


# ---------------------------------------------------------------------------
# _configure_cognee (mock mode)
# ---------------------------------------------------------------------------


class TestConfigureCogneeMock:
    """With LLM_PROVIDER=mock, _configure_cognee sets _configured=True immediately."""

    def setup_method(self):
        _reset_configured()

    def test_mock_provider_sets_configured(self):
        import app.services.cognee_service as svc

        with patch("app.services.cognee_service.get_settings") as mock_settings:
            mock_settings.return_value.llm_provider = "mock"
            asyncio.run(svc._configure_cognee())
        assert svc._configured is True

    def test_idempotent_second_call(self):
        """Calling _configure_cognee twice should not raise."""
        import app.services.cognee_service as svc

        with patch("app.services.cognee_service.get_settings") as mock_settings:
            mock_settings.return_value.llm_provider = "mock"
            asyncio.run(svc._configure_cognee())
            asyncio.run(svc._configure_cognee())  # second call is a no-op
        assert svc._configured is True


# ---------------------------------------------------------------------------
# cognee_search — mock provider
# ---------------------------------------------------------------------------


class TestCogneeSearchMockProvider:
    """With LLM_PROVIDER=mock, cognee_search always returns []."""

    def setup_method(self):
        _reset_configured()

    def _search(self, query="test query", top_k=10):
        import app.services.cognee_service as svc
        with patch("app.services.cognee_service.get_settings") as mock_settings:
            mock_settings.return_value.llm_provider = "mock"
            return asyncio.run(svc.cognee_search(query, top_k))

    def test_returns_empty_list(self):
        assert self._search() == []

    def test_returns_list_type(self):
        result = self._search()
        assert isinstance(result, list)

    def test_respects_different_queries(self):
        """Multiple queries all return []."""
        for q in ["missile", "guidance computer", "radar system"]:
            assert self._search(query=q) == []


# ---------------------------------------------------------------------------
# cognee_search — exception handling
# ---------------------------------------------------------------------------


class TestCogneeSearchExceptionHandling:
    """cognee_search must degrade to [] on any exception."""

    def setup_method(self):
        _reset_configured()

    def test_import_error_returns_empty(self):
        """If cognee package is not installed, return []."""
        import app.services.cognee_service as svc

        with patch("app.services.cognee_service.get_settings") as mock_settings:
            mock_settings.return_value.llm_provider = "openai"
            mock_settings.return_value.cognee_graph_engine = "networkx"
            mock_settings.return_value.cognee_vector_engine = "lancedb"
            mock_settings.return_value.cognee_data_dir = "/tmp/test"
            mock_settings.return_value.openai_api_key = "sk-fake"
            # Simulate cognee not installed
            with patch.dict("sys.modules", {"cognee": None}):
                result = asyncio.run(svc.cognee_search("missile system"))
        assert result == []

    def test_search_exception_returns_empty(self):
        """Runtime exception from cognee.search returns []."""
        import app.services.cognee_service as svc
        svc._configured = True  # skip re-configure

        mock_cognee = MagicMock()
        mock_cognee.search = AsyncMock(side_effect=RuntimeError("connection refused"))

        # Mock SearchType enum
        from enum import Enum
        class MockSearchType(str, Enum):
            GRAPH_COMPLETION = "GRAPH_COMPLETION"
            RAG_COMPLETION = "RAG_COMPLETION"
            CHUNKS = "CHUNKS"
            SUMMARIES = "SUMMARIES"

        mock_search_module = MagicMock()
        mock_search_module.SearchType = MockSearchType

        with patch("app.services.cognee_service.get_settings") as mock_settings:
            mock_settings.return_value.llm_provider = "openai"
            with patch.dict("sys.modules", {
                "cognee": mock_cognee,
                "cognee.api.v1.search.search": mock_search_module,
            }):
                result = asyncio.run(svc.cognee_search("test"))
        assert result == []


# ---------------------------------------------------------------------------
# _result_to_text
# ---------------------------------------------------------------------------


class TestResultToText:
    def _call(self, result):
        from app.services.cognee_service import _result_to_text
        return _result_to_text(result)

    def test_string_result(self):
        assert self._call("hello world") == "hello world"

    def test_empty_string_returns_none(self):
        assert self._call("") is None

    def test_none_returns_none(self):
        assert self._call(None) is None

    def test_object_with_text_attr(self):
        obj = MagicMock()
        obj.text = "chunk text"
        obj.chunk_text = None
        obj.content = None
        obj.summary = None
        obj.answer = None
        assert self._call(obj) == "chunk text"

    def test_object_with_summary_attr(self):
        obj = MagicMock(spec=[])
        obj.summary = "doc summary"
        assert self._call(obj) == "doc summary"

    def test_dict_with_text_key(self):
        assert self._call({"text": "dict text"}) == "dict text"

    def test_dict_with_content_key(self):
        assert self._call({"content": "content text"}) == "content text"

    def test_dict_no_known_key_returns_none(self):
        assert self._call({"unknown": "value"}) is None

    def test_whitespace_only_returns_none(self):
        assert self._call("   ") is None


# ---------------------------------------------------------------------------
# _result_to_score
# ---------------------------------------------------------------------------


class TestResultToScore:
    def _call(self, result, fallback=0.5):
        from app.services.cognee_service import _result_to_score
        return _result_to_score(result, fallback)

    def test_object_with_score_attr(self):
        obj = MagicMock()
        obj.score = 0.87
        obj.similarity = None
        obj.distance = None
        assert abs(self._call(obj) - 0.87) < 1e-6

    def test_dict_with_score(self):
        assert abs(self._call({"score": 0.72}) - 0.72) < 1e-6

    def test_distance_converted_to_similarity(self):
        obj = MagicMock(spec=[])
        obj.distance = 0.3
        score = self._call(obj)
        assert abs(score - 0.7) < 1e-6

    def test_fallback_when_no_score(self):
        assert self._call({}, fallback=0.55) == 0.55

    def test_score_clamped_to_one(self):
        assert self._call({"score": 1.5}) == 1.0

    def test_score_clamped_to_zero(self):
        assert self._call({"score": -0.1}) == 0.0


# ---------------------------------------------------------------------------
# cognee_add / cognee_cognify — mock provider
# ---------------------------------------------------------------------------


class TestCogneeAddCognifyMock:
    def setup_method(self):
        _reset_configured()

    def _run_add(self, text="some text", dataset_name="test_ds"):
        import app.services.cognee_service as svc
        with patch("app.services.cognee_service.get_settings") as mock_settings:
            mock_settings.return_value.llm_provider = "mock"
            asyncio.run(svc.cognee_add(text, dataset_name))

    def _run_cognify(self, dataset_name="test_ds"):
        import app.services.cognee_service as svc
        with patch("app.services.cognee_service.get_settings") as mock_settings:
            mock_settings.return_value.llm_provider = "mock"
            asyncio.run(svc.cognee_cognify(dataset_name))

    def test_add_does_not_raise(self):
        self._run_add()  # should complete silently

    def test_cognify_does_not_raise(self):
        self._run_cognify()  # should complete silently

    def test_add_exception_does_not_propagate(self):
        """Even if cognee.add raises, cognee_add catches and logs."""
        import app.services.cognee_service as svc
        svc._configured = True

        mock_cognee = MagicMock()
        mock_cognee.add = AsyncMock(side_effect=Exception("storage error"))

        with patch("app.services.cognee_service.get_settings") as mock_settings:
            mock_settings.return_value.llm_provider = "openai"
            with patch.dict("sys.modules", {"cognee": mock_cognee}):
                asyncio.run(svc.cognee_add("text", "dataset"))
        # No exception propagated — test passes if we reach here
