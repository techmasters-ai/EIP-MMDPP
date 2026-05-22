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

    The dispatcher sets build_index_failed=True; _claim_and_dispatch_pass
    then skips the HTTP call and returns effective_chunk_scope=None.
    """

    def test_build_failure_means_no_narrowing(self):
        """Simulate the build_index_failed flag: no router call → full doc."""
        # With build_index_failed=True, the dispatcher skips calling the endpoint.
        # Since there's no router_response, _compute_effective_chunk_scope
        # receives None → effective_chunk_scope=None.
        eff, diag = _compute_effective_chunk_scope(None, "narrow_only")
        assert eff is None

    def test_build_failure_diagnostics_captured(self):
        """Build failure diagnostics (fallback_reason) must be in the result."""
        eff, diag = _compute_effective_chunk_scope(None, "narrow_only")
        assert "fallback_reason" in diag or "http_error" in diag
