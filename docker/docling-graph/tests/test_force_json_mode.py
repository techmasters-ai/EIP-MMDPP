"""Tests for DOCLING_GRAPH_FORCE_JSON_MODE kill switch.

Validates that force_json_mode defaults on, and that setting the env var
can still override it. _patched_build_request uses this to bypass
Ollama's constrained grammar (format=<schema>) and fall through to loose
format="json".
"""
import importlib.util
from pathlib import Path


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "app" / "config.py"


def _load_service_config():
    """Load the docling-graph service config module directly from its
    file path. Tests share a filesystem with the worker-side app.config
    and would otherwise pick the wrong one."""
    spec = importlib.util.spec_from_file_location("docling_graph_service_config", _CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_force_json_mode_default_true(monkeypatch):
    monkeypatch.delenv("DOCLING_GRAPH_FORCE_JSON_MODE", raising=False)
    config = _load_service_config()
    settings = config.ServiceSettings()
    assert settings.force_json_mode is True


def test_force_json_mode_reads_env(monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_FORCE_JSON_MODE", "true")
    config = _load_service_config()
    settings = config.ServiceSettings()
    assert settings.force_json_mode is True


def test_force_json_mode_accepts_false_string(monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_FORCE_JSON_MODE", "false")
    config = _load_service_config()
    settings = config.ServiceSettings()
    assert settings.force_json_mode is False
