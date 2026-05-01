"""Tests for PipelineConfig builder from environment variables."""
import os
import pytest
from unittest.mock import patch, MagicMock
import importlib.util
from pathlib import Path


_CONFIG_BUILDER_PATH = Path(__file__).resolve().parent.parent / "app" / "config_builder.py"


def _load_config_builder():
    """Load the docling-graph config_builder directly from its file path.

    Host-side pytest runs from the repo root, where the API package also uses
    the top-level name ``app``. Loading by path avoids importing the wrong app.
    """
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_config_builder",
        _CONFIG_BUILDER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Mock PipelineConfig since docling_graph may not be installed
@pytest.fixture(autouse=True)
def mock_pipeline_config(monkeypatch):
    """Mock docling_graph.PipelineConfig to capture kwargs."""
    mock_cls = MagicMock()
    mock_cls.side_effect = lambda **kwargs: MagicMock(**kwargs)

    import sys
    mock_module = MagicMock()
    mock_module.PipelineConfig = mock_cls
    monkeypatch.setitem(sys.modules, "docling_graph", mock_module)

    return mock_cls


class TestConfigBuilderDefaults:
    def test_builds_config_with_defaults(self, mock_pipeline_config):
        build_pipeline_config = _load_config_builder().build_pipeline_config
        config = build_pipeline_config(source="/tmp/test.json", template_class=None)

        call_kwargs = mock_pipeline_config.call_args[1]
        assert call_kwargs["extraction_contract"] == "delta"
        assert call_kwargs["backend"] == "llm"
        assert call_kwargs["processing_mode"] == "many-to-one"
        assert call_kwargs["use_chunking"] is True
        assert call_kwargs["gleaning_enabled"] is True
        assert call_kwargs["gleaning_max_passes"] == 1
        assert call_kwargs["dump_to_disk"] is False

    def test_delta_resolvers_enabled_by_default(self, mock_pipeline_config):
        build_pipeline_config = _load_config_builder().build_pipeline_config
        config = build_pipeline_config(source="/tmp/test.json", template_class=None)

        call_kwargs = mock_pipeline_config.call_args[1]
        assert call_kwargs["delta_resolvers_enabled"] is True
        assert call_kwargs["delta_resolvers_mode"] == "semantic"

    def test_structured_output_enabled_by_default(self, mock_pipeline_config):
        build_pipeline_config = _load_config_builder().build_pipeline_config
        config = build_pipeline_config(source="/tmp/test.json", template_class=None)

        call_kwargs = mock_pipeline_config.call_args[1]
        assert call_kwargs["structured_output"] is True

    def test_extraction_max_tokens_default_is_bounded(self, mock_pipeline_config):
        build_pipeline_config = _load_config_builder().build_pipeline_config
        build_pipeline_config(source="/tmp/test.json", template_class=None)

        generation = mock_pipeline_config.call_args[1]["llm_overrides"]["generation"]
        assert generation["max_tokens"] == 8192


class TestConfigBuilderOverrides:
    def test_extraction_contract_override(self, mock_pipeline_config):
        build_pipeline_config = _load_config_builder().build_pipeline_config
        with patch.dict(os.environ, {"DOCLING_GRAPH_EXTRACTION_CONTRACT": "direct"}):
            build_pipeline_config(source="/tmp/test.json", template_class=None)
        assert mock_pipeline_config.call_args[1]["extraction_contract"] == "direct"

    def test_parallel_workers_override(self, mock_pipeline_config):
        build_pipeline_config = _load_config_builder().build_pipeline_config
        with patch.dict(os.environ, {"DOCLING_GRAPH_PARALLEL_WORKERS": "4"}):
            build_pipeline_config(source="/tmp/test.json", template_class=None)
        assert mock_pipeline_config.call_args[1]["parallel_workers"] == 4

    def test_gleaning_disabled_override(self, mock_pipeline_config):
        build_pipeline_config = _load_config_builder().build_pipeline_config
        with patch.dict(os.environ, {"DOCLING_GRAPH_GLEANING_ENABLED": "false"}):
            build_pipeline_config(source="/tmp/test.json", template_class=None)
        assert mock_pipeline_config.call_args[1]["gleaning_enabled"] is False

    def test_gleaning_max_passes_boolean_override(self, mock_pipeline_config):
        build_pipeline_config = _load_config_builder().build_pipeline_config
        with patch.dict(os.environ, {"DOCLING_GRAPH_GLEANING_MAX_PASSES": "false"}):
            build_pipeline_config(source="/tmp/test.json", template_class=None)
        assert mock_pipeline_config.call_args[1]["gleaning_max_passes"] == 0

    def test_llm_batch_token_size_override(self, mock_pipeline_config):
        build_pipeline_config = _load_config_builder().build_pipeline_config
        with patch.dict(os.environ, {"DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE": "4096"}):
            build_pipeline_config(source="/tmp/test.json", template_class=None)
        assert mock_pipeline_config.call_args[1]["llm_batch_token_size"] == 4096
