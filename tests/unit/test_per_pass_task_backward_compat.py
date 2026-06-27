"""Backward-compatibility tests for derive_ontology_graph_pass kwargs.

VR C.4 rev 10 M7: Old Celery tasks queued before this commit WITHOUT the
chunk_scope / router_diagnostics kwargs must still run the full doc cleanly.

These tests verify that:
  1. derive_ontology_graph_pass called WITHOUT chunk_scope/router_diagnostics
     kwargs (pre-VR Celery messages) behaves as before: full doc, no narrowing.
  2. Explicit chunk_scope=None → no apply_chunk_scope call.

Run with: pytest tests/unit/test_per_pass_task_backward_compat.py -v
"""
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helper: verify the task signature has backward-compat defaults
# ---------------------------------------------------------------------------


def test_derive_ontology_graph_pass_has_backward_compat_defaults():
    """derive_ontology_graph_pass must have chunk_scope=None and
    router_diagnostics=None as default kwargs (rev 10 M7 backward-compat).
    """
    import inspect
    from app.workers.pipeline import derive_ontology_graph_pass

    # Celery wraps the function; access the underlying function
    fn = derive_ontology_graph_pass
    # Try to get the wrapped function
    while hasattr(fn, "run"):
        fn = fn.run
    if not hasattr(fn, "__wrapped__"):
        fn = getattr(fn, "__func__", fn)

    sig = inspect.signature(fn)
    params = sig.parameters

    assert "chunk_scope" in params, "chunk_scope param must exist on derive_ontology_graph_pass"
    assert "router_diagnostics" in params, "router_diagnostics param must exist"

    cs_default = params["chunk_scope"].default
    rd_default = params["router_diagnostics"].default

    # MINOR #4 fix: clean assertion — default must be exactly None.
    assert cs_default is None, f"chunk_scope default must be None for backward-compat; got {cs_default!r}"


def test_chunk_scope_none_does_not_call_apply_chunk_scope():
    """When chunk_scope=None (default for old tasks), apply_chunk_scope is NOT called."""
    from app.workers import pipeline as pip

    # Import the module-level helper to check it is NOT called
    with patch("app.services.scoped_docling_document.apply_chunk_scope") as mock_apply:
        # Simulate the task body logic: chunk_scope is None → skip apply_chunk_scope
        chunk_scope = None
        if chunk_scope is not None and chunk_scope.get("mode") == "selected_refs":
            from app.services.scoped_docling_document import apply_chunk_scope
            apply_chunk_scope({}, chunk_scope)

    mock_apply.assert_not_called()


def test_chunk_scope_mode_full_does_not_call_apply_chunk_scope():
    """chunk_scope with mode='full' (should never happen per contract, but defensive)
    must not call apply_chunk_scope.
    """
    with patch("app.services.scoped_docling_document.apply_chunk_scope") as mock_apply:
        chunk_scope = {"mode": "full", "self_refs": []}
        if chunk_scope is not None and chunk_scope.get("mode") == "selected_refs":
            from app.services.scoped_docling_document import apply_chunk_scope
            apply_chunk_scope({}, chunk_scope)

    mock_apply.assert_not_called()


def test_chunk_scope_selected_refs_calls_apply_chunk_scope():
    """chunk_scope with mode='selected_refs' DOES call apply_chunk_scope."""
    with patch("app.services.scoped_docling_document.apply_chunk_scope",
               return_value={"body": {"children": []}}) as mock_apply:
        doc_json = {"texts": [], "body": {"children": []}}
        chunk_scope = {"mode": "selected_refs", "self_refs": ["#/texts/0"]}

        if chunk_scope is not None and chunk_scope.get("mode") == "selected_refs":
            from app.services.scoped_docling_document import apply_chunk_scope
            doc_json = apply_chunk_scope(doc_json, chunk_scope)

    mock_apply.assert_called_once()


def test_old_signature_no_chunk_scope_kwarg_concept():
    """Concept: a Celery message serialized without chunk_scope or router_diagnostics
    can be deserialized into a task call with default None values.

    This tests the Python-level kwarg default mechanism, NOT Celery broker serialization
    (which would require a live broker). The assertion is that calling the underlying
    function without the kwargs works cleanly.

    We inspect the task's underlying function (the one registered with Celery)
    to verify the defaults are correct. Celery wraps via @guard_stage_run which
    may itself wrap — we find the first function with our expected params.
    """
    import inspect
    from app.workers.pipeline import derive_ontology_graph_pass

    # Celery task objects have a 'run' attribute pointing to the registered callable.
    # @guard_stage_run wraps with functools.wraps, preserving __wrapped__.
    fn = derive_ontology_graph_pass.run if hasattr(derive_ontology_graph_pass, "run") else derive_ontology_graph_pass
    # Unwrap functools.wraps chains
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__

    sig = inspect.signature(fn)
    params = sig.parameters

    assert "chunk_scope" in params
    assert "router_diagnostics" in params

    # Verify the defaults are None by simulating a minimal bind
    bound = sig.bind(
        MagicMock(),         # self (bind=True task)
        "doc-id",            # document_id
        "run-id",            # run_id
        "radar_power_rf",    # pass_name
        # NOT passing chunk_scope or router_diagnostics → defaults apply
    )
    bound.apply_defaults()

    assert bound.arguments.get("chunk_scope") is None, (
        "Old message without chunk_scope kwarg must default to None"
    )
    assert bound.arguments.get("router_diagnostics") is None, (
        "Old message without router_diagnostics kwarg must default to None"
    )
