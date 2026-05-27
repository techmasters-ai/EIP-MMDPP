"""Unit tests for VR C.4 dispatcher wiring.

Tests the effective_chunk_scope decision logic in
_compute_effective_chunk_scope and the _call_chunk_scope_endpoint helper,
as well as the short-circuit for non-field_group passes.

All tests mock external I/O (httpx, ArcadeDB, Celery).

Run with: pytest tests/unit/test_dispatcher_vr_wiring.py -v
"""
import pytest
from unittest.mock import MagicMock, patch

from app.workers.pipeline import (
    _compute_effective_chunk_scope,
    _call_chunk_scope_endpoint,
)


# ---------------------------------------------------------------------------
# _compute_effective_chunk_scope unit tests
# ---------------------------------------------------------------------------


class TestComputeEffectiveChunkScope:
    """Tests for _compute_effective_chunk_scope(router_response, mode)."""

    def _make_response(self, mode, self_refs=None, diag_extra=None):
        """Build a mock router_response dict."""
        resp = {
            "mode": mode,
            "self_refs": self_refs or [],
            "diagnostics": dict(diag_extra or {}),
        }
        return resp

    def test_disabled_mode_passes_none_chunk_scope(self):
        """mode=disabled → effective_chunk_scope=None always, regardless of router."""
        router_response = self._make_response("selected_refs", self_refs=["#/texts/1"])
        eff, diag = _compute_effective_chunk_scope(router_response, "disabled")
        assert eff is None, "disabled mode must never narrow"

    def test_shadow_mode_always_passes_none_chunk_scope_even_when_selected_refs(self):
        """mode=shadow + router returns selected_refs → effective_chunk_scope=None.

        This is the LOAD-BEARING assertion: shadow mode must NEVER narrow.
        shadow_skipped_narrowing=True must be captured in diagnostics.
        """
        router_response = self._make_response("selected_refs", self_refs=["#/texts/1", "#/tables/0"])
        eff, diag = _compute_effective_chunk_scope(router_response, "shadow")

        assert eff is None, "shadow mode MUST NOT narrow (even with selected_refs)"
        assert diag.get("shadow_skipped_narrowing") is True, (
            "shadow_skipped_narrowing must be True when router returned selected_refs"
        )

    def test_shadow_mode_shadow_skipped_narrowing_false_when_full(self):
        """mode=shadow + router returns full → shadow_skipped_narrowing=False."""
        router_response = self._make_response("full", self_refs=[])
        eff, diag = _compute_effective_chunk_scope(router_response, "shadow")

        assert eff is None
        assert diag.get("shadow_skipped_narrowing") is False

    def test_narrow_only_mode_passes_chunk_scope_on_selected_refs(self):
        """mode=narrow_only + router returns selected_refs → effective_chunk_scope set."""
        self_refs = ["#/texts/3", "#/tables/1"]
        router_response = self._make_response("selected_refs", self_refs=self_refs)
        eff, diag = _compute_effective_chunk_scope(router_response, "narrow_only")

        assert eff is not None, "narrow_only must narrow on selected_refs"
        assert eff["mode"] == "selected_refs"
        assert eff["self_refs"] == self_refs

    def test_narrow_only_mode_passes_none_on_full(self):
        """mode=narrow_only + router returns full → effective_chunk_scope=None (run full)."""
        router_response = self._make_response("full", self_refs=[])
        eff, diag = _compute_effective_chunk_scope(router_response, "narrow_only")

        assert eff is None
        assert "fail_open_reason" not in diag

    def test_narrow_only_mode_fails_open_on_would_skip(self):
        """mode=narrow_only + router returns would_skip → effective_chunk_scope=None
        + fail_open_reason captured in diagnostics.
        """
        router_response = self._make_response("would_skip", self_refs=[])
        eff, diag = _compute_effective_chunk_scope(router_response, "narrow_only")

        assert eff is None, "would_skip must fail open to full doc"
        assert diag.get("fail_open_reason") == "would_skip_in_narrow_only_mode", (
            "fail_open_reason must be captured in diagnostics"
        )

    def test_endpoint_http_error_falls_back_to_run_full(self):
        """router_response=None (HTTP error) → effective_chunk_scope=None."""
        eff, diag = _compute_effective_chunk_scope(None, "narrow_only")

        assert eff is None, "HTTP error must fall back to full doc"
        assert "http_error" in diag or "fallback_reason" in diag, (
            "HTTP error must be captured in diagnostics"
        )

    def test_narrow_only_threads_text_by_ref_for_selected_refs(self):
        """Selected chunk text must cross the worker boundary with self_refs."""
        router_response = self._make_response(
            "selected_refs",
            self_refs=["#/texts/3", "#/texts/4"],
        )
        router_response["text_by_ref"] = {
            "#/texts/3": "post-filter radar evidence",
            "#/texts/999": "not selected",
        }

        eff, _diag = _compute_effective_chunk_scope(router_response, "narrow_only")

        assert eff is not None
        assert eff["text_by_ref"] == {"#/texts/3": "post-filter radar evidence"}

    def test_narrow_only_empty_self_refs_returns_none(self):
        """mode=narrow_only + router returns selected_refs with empty self_refs
        → effective_chunk_scope=None (no valid refs to narrow on).
        """
        router_response = self._make_response("selected_refs", self_refs=[])
        eff, diag = _compute_effective_chunk_scope(router_response, "narrow_only")
        # No self_refs → can't narrow meaningfully; should be None
        assert eff is None


# ---------------------------------------------------------------------------
# _call_chunk_scope_endpoint unit tests
# ---------------------------------------------------------------------------


class TestCallChunkScopeEndpoint:
    """Tests for _call_chunk_scope_endpoint (HTTP call helper)."""

    def test_successful_call_returns_response(self):
        """Successful HTTP call returns the parsed JSON response."""
        expected = {"mode": "selected_refs", "self_refs": ["#/texts/1"], "diagnostics": {}}
        mock_response = MagicMock()
        mock_response.json.return_value = expected
        mock_response.raise_for_status.return_value = None

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = _call_chunk_scope_endpoint(
                "run-123", "air_defense_v3", "radar_power_rf",
                "http://api:8000",
            )

        assert result == expected

    def test_http_timeout_returns_none(self):
        """HTTP timeout → returns None (caller applies fail-open)."""
        import httpx

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value = mock_client

            result = _call_chunk_scope_endpoint(
                "run-123", "air_defense_v3", "radar_power_rf",
                "http://api:8000",
            )

        assert result is None

    def test_http_500_returns_none(self):
        """HTTP 500 → returns None (caller applies fail-open)."""
        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = _call_chunk_scope_endpoint(
                "run-123", "air_defense_v3", "radar_power_rf",
                "http://api:8000",
            )

        assert result is None

    def test_connection_refused_returns_none(self):
        """Connection refused → returns None."""
        import httpx

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_client_cls.return_value = mock_client

            result = _call_chunk_scope_endpoint(
                "run-123", "air_defense_v3", "radar_power_rf",
                "http://api:8000",
            )

        assert result is None


# ---------------------------------------------------------------------------
# Short-circuit tests (no HTTP call for identity/relationship passes)
# ---------------------------------------------------------------------------


class TestPassPhaseShortCircuit:
    """Identity and relationship passes must not trigger the router.

    The short-circuit is implemented in _claim_and_dispatch_pass: only
    field_group passes with retrieval profiles get the HTTP call.
    We test _compute_effective_chunk_scope directly with the mode checks.
    """

    def test_identity_pass_short_circuit_concept(self):
        """Conceptual: identity pass dispatch should not call chunk-scope endpoint.

        The dispatcher gates the HTTP call on pass_def.phase == "field_group".
        We verify that when the caller bypasses the HTTP call and passes
        router_response=None, the result is always full-doc.
        """
        eff, diag = _compute_effective_chunk_scope(None, "narrow_only")
        assert eff is None, "Identity pass (no HTTP call) must dispatch full doc"

    def test_relationship_pass_short_circuit_concept(self):
        """Relationship passes bypass the router entirely."""
        eff, diag = _compute_effective_chunk_scope(None, "narrow_only")
        assert eff is None


class TestBuildFailureDisablesVR:
    """When build_extraction_index failed, all passes must dispatch full doc.

    IMPORTANT #2 fix: Tests call _claim_and_dispatch_pass with build_index_failed=True,
    exercising the actual short-circuit branch, not just _compute_effective_chunk_scope.
    """

    def _make_pass_def(self, phase="field_group", required=False):
        """Build a minimal mock pass_def."""
        pd = MagicMock()
        pd.phase = phase
        pd.required = required
        pd.name = "radar_power_rf"
        return pd

    def test_build_failure_makes_no_http_call(self):
        """With build_index_failed=True, _claim_and_dispatch_pass must NOT call the endpoint."""
        from app.workers.pipeline import _claim_and_dispatch_pass

        mock_db = MagicMock()
        # Simulate claim_phase succeeding
        with patch("app.workers.pipeline.claim_phase", return_value=True), \
             patch("app.workers.pipeline.mark_phase_dispatched"), \
             patch("app.workers.pipeline.derive_ontology_graph_pass") as mock_task, \
             patch("app.workers.pipeline._call_chunk_scope_endpoint") as mock_http, \
             patch("app.workers.pipeline.settings") as mock_settings:

            mock_settings.vector_router_mode = "narrow_only"
            mock_settings.internal_api_base_url = "http://api:8000"
            mock_settings.pass_concurrency_per_document = 4
            mock_task.delay.return_value = MagicMock(id="task-abc")

            _claim_and_dispatch_pass(
                mock_db,
                document_id="doc-1",
                run_id="run-1",
                pass_name="radar_power_rf",
                pass_def=self._make_pass_def(),
                bundle_key="air_defense_v3",
                build_index_failed=True,
            )

        # Must not call the HTTP endpoint
        mock_http.assert_not_called()

    def test_build_failure_router_diagnostics_has_fallback_reason(self):
        """With build_index_failed=True, the Celery task is dispatched with
        router_diagnostics containing fallback_reason='index_build_failed'.
        """
        from app.workers.pipeline import _claim_and_dispatch_pass

        mock_db = MagicMock()
        dispatched_kwargs = {}

        def capture_delay(document_id, run_id, pass_name, **kwargs):
            dispatched_kwargs.update(kwargs)
            result = MagicMock()
            result.id = "task-abc"
            return result

        with patch("app.workers.pipeline.claim_phase", return_value=True), \
             patch("app.workers.pipeline.mark_phase_dispatched"), \
             patch("app.workers.pipeline._call_chunk_scope_endpoint"), \
             patch("app.workers.pipeline.derive_ontology_graph_pass") as mock_task, \
             patch("app.workers.pipeline.settings") as mock_settings:

            mock_settings.vector_router_mode = "narrow_only"
            mock_settings.internal_api_base_url = "http://api:8000"
            mock_settings.pass_concurrency_per_document = 4
            mock_task.delay.side_effect = capture_delay

            _claim_and_dispatch_pass(
                mock_db,
                document_id="doc-1",
                run_id="run-1",
                pass_name="radar_power_rf",
                pass_def=self._make_pass_def(),
                bundle_key="air_defense_v3",
                build_index_failed=True,
            )

        diag = dispatched_kwargs.get("router_diagnostics") or {}
        assert diag.get("fallback_reason") == "index_build_failed", (
            f"Expected fallback_reason='index_build_failed'; got router_diagnostics={diag!r}"
        )

    def test_build_failure_effective_chunk_scope_is_none(self):
        """With build_index_failed=True, effective_chunk_scope kwarg on the task is None."""
        from app.workers.pipeline import _claim_and_dispatch_pass

        mock_db = MagicMock()
        dispatched_kwargs = {}

        def capture_delay(document_id, run_id, pass_name, **kwargs):
            dispatched_kwargs.update(kwargs)
            result = MagicMock()
            result.id = "task-abc"
            return result

        with patch("app.workers.pipeline.claim_phase", return_value=True), \
             patch("app.workers.pipeline.mark_phase_dispatched"), \
             patch("app.workers.pipeline._call_chunk_scope_endpoint"), \
             patch("app.workers.pipeline.derive_ontology_graph_pass") as mock_task, \
             patch("app.workers.pipeline.settings") as mock_settings:

            mock_settings.vector_router_mode = "narrow_only"
            mock_settings.internal_api_base_url = "http://api:8000"
            mock_settings.pass_concurrency_per_document = 4
            mock_task.delay.side_effect = capture_delay

            _claim_and_dispatch_pass(
                mock_db,
                document_id="doc-1",
                run_id="run-1",
                pass_name="radar_power_rf",
                pass_def=self._make_pass_def(),
                bundle_key="air_defense_v3",
                build_index_failed=True,
            )

        assert dispatched_kwargs.get("chunk_scope") is None, (
            f"chunk_scope must be None when build_index_failed=True; "
            f"got {dispatched_kwargs.get('chunk_scope')!r}"
        )


# ---------------------------------------------------------------------------
# MINOR #5 — shadow + would_skip branch
# ---------------------------------------------------------------------------


class TestShadowModeWouldSkip:
    """Shadow mode with router returning would_skip."""

    def test_shadow_mode_with_would_skip_returns_none_chunk_scope(self):
        """shadow + would_skip → effective_chunk_scope=None.

        would_skip is NOT a 'skipped narrowing' event in shadow's sense:
        shadow_skipped_narrowing should be False (shadow skips narrowing only
        when router returned selected_refs).
        """
        router_response = {
            "mode": "would_skip",
            "self_refs": [],
            "diagnostics": {},
        }
        eff, diag = _compute_effective_chunk_scope(router_response, "shadow")

        assert eff is None, "shadow must never narrow regardless of router mode"
        assert diag.get("shadow_skipped_narrowing") is False, (
            "shadow_skipped_narrowing must be False for would_skip "
            "(only True when router returned selected_refs)"
        )
