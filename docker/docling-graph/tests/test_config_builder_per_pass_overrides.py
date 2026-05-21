"""C2 — build_pipeline_config per-pass override tests.

Verifies that the new chunk_max_tokens_override and max_tokens_override
kwargs are applied correctly, and that None falls back to env defaults.

These tests run from the host venv where the docling_graph library is not
installed. The try/except ImportError in build_pipeline_config returns
llm_client=None in that case (because app.ollama_clients is not available
in the host venv), so we mock it.

Run standalone:
    python3 -m pytest docker/docling-graph/tests/test_config_builder_per_pass_overrides.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from contextlib import contextmanager


def _load_config_builder():
    """Load docker/docling-graph/app/config_builder.py by file path.

    Bypasses the repo-root app/ package resolution on sys.path. Uses the
    _dg_config_builder module name to avoid conflicts with prior loads.
    """
    key = "_dg_config_builder"
    if key in sys.modules:
        return sys.modules[key]
    path = Path(__file__).resolve().parent.parent / "app" / "config_builder.py"
    spec = importlib.util.spec_from_file_location(key, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _fake_pipeline_env(captured: dict, mock_client=None):
    """Context manager that mocks away docling_graph and app.ollama_clients.

    - PipelineConfig kwargs are captured into `captured`.
    - If mock_client is provided, app.ollama_clients.get_docling_graph_client
      returns it, so the llm_client override path activates.
    - If mock_client is None, the import still fails gracefully and
      llm_client stays None (env-default path).
    """
    class FakePipelineConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_dg = MagicMock()
    fake_dg.PipelineConfig = FakePipelineConfig

    # Build a fake app.ollama_clients module.
    fake_ollama_clients = MagicMock()
    if mock_client is not None:
        fake_ollama_clients.get_docling_graph_client = MagicMock(return_value=mock_client)
    else:
        # Make the import succeed but the call raise so llm_client stays None.
        fake_ollama_clients.get_docling_graph_client = MagicMock(
            side_effect=ImportError("no client in host venv")
        )

    with patch.dict("sys.modules", {
        "docling_graph": fake_dg,
        "app.ollama_clients": fake_ollama_clients,
        # Provide a stub for app.config so DoclingGraphSettings can import
        # pydantic_settings without pulling in the real service config.
    }):
        yield


class TestBuildPipelineConfigChunkMaxTokensOverride:

    def test_chunk_max_tokens_override_replaces_env_default(self):
        """When chunk_max_tokens_override=256, PipelineConfig gets chunk_max_tokens=256."""
        cb = _load_config_builder()
        captured: dict = {}

        with _fake_pipeline_env(captured):
            cb.build_pipeline_config(
                source="/fake/source.json",
                template_class=None,
                chunk_max_tokens_override=256,
            )

        assert captured.get("chunk_max_tokens") == 256

    def test_chunk_max_tokens_override_none_uses_env_default(self):
        """When chunk_max_tokens_override=None, PipelineConfig gets the env default (512)."""
        cb = _load_config_builder()
        captured: dict = {}

        with _fake_pipeline_env(captured):
            cb.build_pipeline_config(
                source="/fake/source.json",
                template_class=None,
                chunk_max_tokens_override=None,
            )

        # Default from DoclingGraphSettings.docling_graph_chunk_max_tokens = 512
        assert captured.get("chunk_max_tokens") == 512


class TestBuildPipelineConfigMaxTokensOverride:

    def test_max_tokens_override_reaches_with_runtime_defaults(self):
        """max_tokens_override is plumbed into llm_client.with_runtime_defaults."""
        cb = _load_config_builder()
        captured: dict = {}
        captured_runtime_defaults: dict = {}

        mock_client = MagicMock(spec=["with_runtime_defaults"])
        new_client = MagicMock()

        def _with_runtime_defaults(**kwargs):
            captured_runtime_defaults.update(kwargs)
            return new_client

        mock_client.with_runtime_defaults = _with_runtime_defaults

        with _fake_pipeline_env(captured, mock_client=mock_client):
            cb.build_pipeline_config(
                source="/fake/source.json",
                template_class=None,
                max_tokens_override=8192,
            )

        # max_tokens should be in the with_runtime_defaults kwargs
        assert captured_runtime_defaults.get("max_tokens") == 8192

    def test_max_tokens_override_none_does_not_call_with_runtime_defaults(self):
        """When all overrides are None, with_runtime_defaults is not called."""
        cb = _load_config_builder()
        captured: dict = {}
        was_called = []

        mock_client = MagicMock(spec=["with_runtime_defaults"])

        def _with_runtime_defaults(**kwargs):
            was_called.append(kwargs)
            return mock_client

        mock_client.with_runtime_defaults = _with_runtime_defaults

        with _fake_pipeline_env(captured, mock_client=mock_client):
            cb.build_pipeline_config(
                source="/fake/source.json",
                template_class=None,
                # All override kwargs absent / None
            )

        assert not was_called, "with_runtime_defaults should not be called when no overrides"


class TestBuildPipelineConfigTemperatureExistingPath:

    def test_temperature_override_goes_through_with_runtime_defaults(self):
        """Existing behavior: temperature_override uses with_runtime_defaults."""
        cb = _load_config_builder()
        captured: dict = {}
        captured_runtime_defaults: dict = {}

        mock_client = MagicMock(spec=["with_runtime_defaults"])
        new_client = MagicMock()

        def _with_runtime_defaults(**kwargs):
            captured_runtime_defaults.update(kwargs)
            return new_client

        mock_client.with_runtime_defaults = _with_runtime_defaults

        with _fake_pipeline_env(captured, mock_client=mock_client):
            cb.build_pipeline_config(
                source="/fake/source.json",
                template_class=None,
                temperature_override=0.3,
            )

        assert captured_runtime_defaults.get("temperature") == 0.3

    def test_temperature_and_max_tokens_both_reach_runtime_defaults(self):
        """Both temperature_override and max_tokens_override go to with_runtime_defaults."""
        cb = _load_config_builder()
        captured: dict = {}
        captured_runtime_defaults: dict = {}

        mock_client = MagicMock(spec=["with_runtime_defaults"])
        new_client = MagicMock()

        def _with_runtime_defaults(**kwargs):
            captured_runtime_defaults.update(kwargs)
            return new_client

        mock_client.with_runtime_defaults = _with_runtime_defaults

        with _fake_pipeline_env(captured, mock_client=mock_client):
            cb.build_pipeline_config(
                source="/fake/source.json",
                template_class=None,
                temperature_override=0.3,
                max_tokens_override=8192,
            )

        assert captured_runtime_defaults.get("temperature") == 0.3
        assert captured_runtime_defaults.get("max_tokens") == 8192
