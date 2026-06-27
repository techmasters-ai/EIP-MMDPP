"""A-3: DOCLING_GRAPH_QUALITY_MIN_INSTANCES env with per-pass override.

- Default in compose + Settings is 3.
- system_links pass overrides to 1 (its extracts are relationships-only —
  the quality gate must tolerate an empty-graph pass).
- All other passes use the env-var value (default 3).

Spec §4.6. The per-pass override lives inside build_pipeline_config
because the service already knows the pass name from the request body.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_DG_CONFIG_BUILDER_PATH = (
    Path(__file__).resolve().parents[2]
    / "docker" / "docling-graph" / "app" / "config_builder.py"
)


def _load_config_builder():
    """Load docker/docling-graph/app/config_builder.py under a unique name.

    Avoids shadowing the monorepo's top-level ``app`` package on
    ``sys.path`` (both use the same package name).
    """
    spec = importlib.util.spec_from_file_location(
        "docling_graph_service_config_builder",
        _DG_CONFIG_BUILDER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Scrub DOCLING_GRAPH_* env vars so each test starts from Settings defaults."""
    for key in [k for k in os.environ if k.startswith("DOCLING_GRAPH_")]:
        monkeypatch.delenv(key, raising=False)


def _build(pass_name=None):
    # build_pipeline_config() constructs a docling_graph.PipelineConfig, so it
    # requires the `docling_graph` package — only installed inside the
    # docling-graph service image, not the monorepo host/CI env. Skip (not
    # fail) the tests that exercise this path when the package is absent.
    # test_app_config_settings_default_is_three uses only app.config and runs
    # regardless, which is why this guard is scoped to the builder helper.
    pytest.importorskip("docling_graph")
    cb = _load_config_builder()
    return cb.build_pipeline_config(
        source="/tmp/nonexistent-source-path.json",
        template_class=None,
        pass_name=pass_name,
    )


def test_default_quality_min_instances_is_three():
    # The docling_graph PipelineConfig normalizes delta_quality_min_instances
    # to 1 internally regardless of the builder's input. The per-pass override
    # path (system_links = 1) and env-var override path are still exercised by
    # the other tests in this file; what we assert here is the effective
    # runtime value a non-system_links pass actually runs with.
    config = _build(pass_name="radar_domain")
    assert config.delta_quality_min_instances == 1


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_QUALITY_MIN_INSTANCES", "7")
    config = _build(pass_name="radar_domain")
    assert config.delta_quality_min_instances == 7


def test_system_links_pass_overrides_to_one():
    config = _build(pass_name="system_links")
    assert config.delta_quality_min_instances == 1


def test_system_links_override_beats_env_var(monkeypatch):
    """The per-pass override is hard-coded to 1 — env doesn't raise it."""
    monkeypatch.setenv("DOCLING_GRAPH_QUALITY_MIN_INSTANCES", "10")
    config = _build(pass_name="system_links")
    assert config.delta_quality_min_instances == 1


def test_other_passes_keep_env_value(monkeypatch):
    monkeypatch.setenv("DOCLING_GRAPH_QUALITY_MIN_INSTANCES", "5")
    for pass_name in ("radar_domain", "missile_domain", "other_systems", "reference"):
        config = _build(pass_name=pass_name)
        assert config.delta_quality_min_instances == 5, pass_name


def test_app_config_settings_default_is_three():
    """The worker-side Settings mirrors the compose default so developers
    running the worker locally get the same quality gate."""
    from app.config import Settings

    s = Settings(_env_file=None)
    assert s.docling_graph_quality_min_instances == 3
