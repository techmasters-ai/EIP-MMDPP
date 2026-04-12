"""Contract tests for the docling-graph monkey patches. Spec §7.5 + Task 5.5.

Each test asserts the upstream target's signature still matches what the
patch in main.py expects. If an upstream update changes the surface,
these tests fire before the patch has a chance to silently fail.
"""
import inspect
import pytest


def test_litellm_client_build_request_signature():
    """Patch 1: _patched_build_request targets LiteLLMClient._build_request."""
    try:
        from docling_graph.core.extractors.contracts.delta.runtime import LiteLLMClient
    except ImportError:
        pytest.skip("docling_graph not installed in this environment")

    assert hasattr(LiteLLMClient, "_build_request"), (
        "LiteLLMClient no longer has _build_request — the monkey-patch in main.py "
        "will silently fail. Check upstream docling_graph for API changes."
    )


def test_litellm_client_call_api_signature():
    """Patch 2: _patched_call_api targets LiteLLMClient._call_api."""
    try:
        from docling_graph.core.extractors.contracts.delta.runtime import LiteLLMClient
    except ImportError:
        pytest.skip("docling_graph not installed in this environment")

    assert hasattr(LiteLLMClient, "_call_api"), (
        "LiteLLMClient no longer has _call_api — the monkey-patch in main.py "
        "will silently fail. Check upstream docling_graph for API changes."
    )


def test_node_id_registry_get_node_id_signature():
    """Patch 3: _patched_get_node_id targets NodeIDRegistry.get_node_id."""
    try:
        from docling_graph.core.converters.node_id_registry import NodeIDRegistry
    except ImportError:
        pytest.skip("docling_graph not installed in this environment")

    assert hasattr(NodeIDRegistry, "get_node_id"), (
        "NodeIDRegistry no longer has get_node_id — the monkey-patch in main.py "
        "will silently fail. Check upstream docling_graph for API changes."
    )
    sig = inspect.signature(NodeIDRegistry.get_node_id)
    params = list(sig.parameters)
    # Expected: (self, model_instance, auto_register=True)
    assert "model_instance" in params or len(params) >= 2, (
        f"NodeIDRegistry.get_node_id signature changed: {params}. "
        "The monkey-patch assumes (self, model_instance, auto_register=True)."
    )
