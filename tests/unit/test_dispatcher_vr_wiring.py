"""Unit tests for VR C.4 dispatcher wiring.

Tests the effective_chunk_scope decision logic in
_compute_effective_chunk_scope and the _call_chunk_scope_endpoint helper,
as well as the short-circuit for non-field_group passes.

All tests mock external I/O (httpx, ArcadeDB, Celery).

Run with: pytest tests/unit/test_dispatcher_vr_wiring.py -v
"""
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.workers.pipeline import (
    _compute_effective_chunk_scope,
    _call_chunk_scope_endpoint,
    _collect_committed_identity_anchors,
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

    def test_narrow_only_fails_open_on_degraded_fallback(self):
        """mode=narrow_only + selected_refs BUT fallback_level=degraded → fall open
        to full-doc. RECALL-SAFETY GUARD (2026-06-21): the degraded fallback pool is
        starved with no gate floor (gate_unit_keeps=0); narrowing to its self_refs
        drops recall. Must run full instead, even though self_refs are present.
        """
        router_response = self._make_response(
            "selected_refs",
            self_refs=["#/texts/0", "#/texts/1", "#/texts/2"],
            diag_extra={"fallback_level": "degraded"},
        )
        eff, diag = _compute_effective_chunk_scope(router_response, "narrow_only")

        assert eff is None, "degraded fallback MUST fall open to full doc (no gate floor)"
        assert diag.get("fail_open_reason") == "degraded_fallback_no_gate_floor"

    def test_narrow_only_still_narrows_on_nondegraded_fallback(self):
        """Guard against over-broad fall-open: a non-degraded fallback rung (e.g.
        relaxed_dense) that still produced a real pool MUST still narrow."""
        self_refs = ["#/texts/3", "#/tables/1"]
        router_response = self._make_response(
            "selected_refs", self_refs=self_refs,
            diag_extra={"fallback_level": "relaxed_dense"},
        )
        eff, diag = _compute_effective_chunk_scope(router_response, "narrow_only")

        assert eff is not None, "non-degraded fallback must still narrow"
        assert eff["self_refs"] == self_refs
        assert diag.get("fail_open_reason") != "degraded_fallback_no_gate_floor"

    def test_narrow_only_fails_open_on_small_doc(self):
        """mode=narrow_only + selected_refs BUT full_doc_token_estimate below the
        narrow_min_doc_tokens gate → fall open to full-doc. RECALL-SAFETY GUARD
        (2026-06-21): narrowing a small doc is all recall-risk for negligible
        wall-time savings (observed: NMUSAF 3459 tokens lost 33% recall when
        narrowed). Must run full instead, even though self_refs are present.
        """
        router_response = self._make_response(
            "selected_refs",
            self_refs=["#/texts/0", "#/texts/1", "#/texts/2"],
            diag_extra={"full_doc_token_estimate": 3459, "fallback_level": "none"},
        )
        with patch("app.workers.pipeline.settings.narrow_min_doc_tokens", 6000):
            eff, diag = _compute_effective_chunk_scope(router_response, "narrow_only")

        assert eff is None, "small doc MUST fall open to full doc (narrowing risks recall)"
        assert diag.get("fail_open_reason") == "small_doc_no_narrow_benefit"

    def test_narrow_only_narrows_on_large_doc(self):
        """Guard against over-broad fall-open: a doc at/above the size gate that
        returned selected_refs MUST still narrow."""
        self_refs = ["#/texts/3", "#/tables/1"]
        router_response = self._make_response(
            "selected_refs", self_refs=self_refs,
            diag_extra={"full_doc_token_estimate": 7118, "fallback_level": "none"},
        )
        with patch("app.workers.pipeline.settings.narrow_min_doc_tokens", 6000):
            eff, diag = _compute_effective_chunk_scope(router_response, "narrow_only")

        assert eff is not None, "doc above the size gate must still narrow"
        assert eff["self_refs"] == self_refs
        assert diag.get("fail_open_reason") != "small_doc_no_narrow_benefit"

    def test_narrow_only_size_gate_disabled_when_zero(self):
        """narrow_min_doc_tokens=0 disables the gate: a tiny doc still narrows."""
        self_refs = ["#/texts/0"]
        router_response = self._make_response(
            "selected_refs", self_refs=self_refs,
            diag_extra={"full_doc_token_estimate": 100, "fallback_level": "none"},
        )
        with patch("app.workers.pipeline.settings.narrow_min_doc_tokens", 0):
            eff, diag = _compute_effective_chunk_scope(router_response, "narrow_only")

        assert eff is not None, "size gate disabled (0) must allow narrowing any doc"
        assert eff["self_refs"] == self_refs

    def test_narrow_only_degraded_takes_precedence_over_size_gate(self):
        """degraded fall-open and size-gate fall-open both yield None; degraded is
        checked first, so its reason wins when both apply."""
        router_response = self._make_response(
            "selected_refs", self_refs=["#/texts/0"],
            diag_extra={"full_doc_token_estimate": 100, "fallback_level": "degraded"},
        )
        with patch("app.workers.pipeline.settings.narrow_min_doc_tokens", 6000):
            eff, diag = _compute_effective_chunk_scope(router_response, "narrow_only")

        assert eff is None
        assert diag.get("fail_open_reason") == "degraded_fallback_no_gate_floor"

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


# ---------------------------------------------------------------------------
# _call_chunk_scope_endpoint — identity_anchors forwarding (C8 worker delivery)
# ---------------------------------------------------------------------------


def _capture_post_body():
    """Return (mock_client_cls_ctx_factory, captured) where captured['body'] is
    the JSON body the worker POSTs. Mirrors the existing httpx.Client mock style.
    """
    captured: dict = {}
    mock_response = MagicMock()
    mock_response.json.return_value = {"mode": "full", "self_refs": [], "diagnostics": {}}
    mock_response.raise_for_status.return_value = None

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    def _post(url, json=None):  # noqa: A002 - mirror httpx signature
        captured["url"] = url
        captured["body"] = json
        return mock_response

    mock_client.post.side_effect = _post
    return mock_client, captured


class TestCallChunkScopeEndpointIdentityAnchors:
    """The worker must forward committed identity-entity names as
    ``identity_anchors`` in the chunk-scope request body so the endpoint's C8
    channel fires.  Empty/None anchors must leave the body byte-identical to
    today (no ``identity_anchors`` key)."""

    def test_anchors_included_in_request_body_when_supplied(self):
        mock_client, captured = _capture_post_body()
        with patch("httpx.Client", return_value=mock_client):
            _call_chunk_scope_endpoint(
                "run-123", "air_defense_v3", "radar_power_rf",
                "http://api:8000",
                identity_anchors=["Fan Song", "SA-2"],
            )
        assert captured["body"].get("identity_anchors") == ["Fan Song", "SA-2"]
        # Existing keys must still be present and unchanged.
        assert captured["body"]["pipeline_run_id"] == "run-123"
        assert captured["body"]["bundle_key"] == "air_defense_v3"
        assert captured["body"]["pass_name"] == "radar_power_rf"

    def test_empty_anchors_omitted_from_body(self):
        """Empty list → no identity_anchors key (body byte-identical to today)."""
        mock_client, captured = _capture_post_body()
        with patch("httpx.Client", return_value=mock_client):
            _call_chunk_scope_endpoint(
                "run-123", "air_defense_v3", "radar_power_rf",
                "http://api:8000",
                identity_anchors=[],
            )
        assert "identity_anchors" not in captured["body"]
        assert set(captured["body"].keys()) == {
            "pipeline_run_id", "bundle_key", "pass_name",
        }

    def test_none_anchors_omitted_from_body(self):
        """Default None (no anchors) → body byte-identical to pre-C8 shape."""
        mock_client, captured = _capture_post_body()
        with patch("httpx.Client", return_value=mock_client):
            _call_chunk_scope_endpoint(
                "run-123", "air_defense_v3", "radar_power_rf",
                "http://api:8000",
            )
        assert "identity_anchors" not in captured["body"]
        assert set(captured["body"].keys()) == {
            "pipeline_run_id", "bundle_key", "pass_name",
        }


# ---------------------------------------------------------------------------
# _collect_committed_identity_anchors — worker-side fetch of identity names
# ---------------------------------------------------------------------------


def _identity_manifest():
    """Manifest with one identity pass and one field_group pass (generalized:
    identity scope is driven by phase == 'identity', not hardcoded types)."""
    id_pass = SimpleNamespace(
        name="radar_identity",
        phase="identity",
        primary_entity_types=["RADAR_SYSTEM"],
        module="extraction_schemas.radar_identity",
        template_class="RadarIdentityPass",
        input_mode="document",
        depends_on=[],
    )
    fg_pass = SimpleNamespace(
        name="radar_power_rf",
        phase="field_group",
        primary_entity_types=["RADAR_SYSTEM"],
    )
    return SimpleNamespace(passes=[id_pass, fg_pass], bundle_key="air_defense_v3")


class TestCollectCommittedIdentityAnchors:
    """Worker-side anchor fetch: read identity-phase passes' persisted outputs
    and surface their entity names (+ aliases) for the routing request."""

    def test_returns_names_from_persisted_identity_passes(self):
        manifest = _identity_manifest()
        ontology = {"entity_types": [
            {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"]},
        ]}

        dep_row = SimpleNamespace(
            execution_status="COMPLETE",
            extract_pass_response_json={"stub": True},
        )

        # Re-parsed pass result yields refs via _extend_upstream_refs; we patch
        # _extend_upstream_refs to inject two refs with display_label + aliases.
        def _fake_extend(upstream_refs, pass_result, pass_def, ont):
            upstream_refs["E001"] = SimpleNamespace(
                entity_type="RADAR_SYSTEM",
                identity_values={"system_name": "Fan Song"},
                display_label="Fan Song",
                aliases=["SNR-75"],
            )
            upstream_refs["E002"] = SimpleNamespace(
                entity_type="RADAR_SYSTEM",
                identity_values={"system_name": "Spoon Rest"},
                display_label="Spoon Rest",
                aliases=[],
            )

        with (
            patch("app.workers.pipeline.load_pass_output", return_value=dep_row),
            patch("app.workers.pipeline._parse_pass_response", return_value=MagicMock()),
            patch("app.workers.pipeline._build_pre_merge_walk_summary", return_value=None),
            patch("app.workers.pipeline._extend_upstream_refs", side_effect=_fake_extend),
        ):
            anchors = _collect_committed_identity_anchors(
                db=MagicMock(),
                run_id="run-1",
                manifest=manifest,
                ontology=ontology,
                document_id="doc-1",
            )

        # Display labels AND aliases are surfaced as anchors, deduped.
        assert "Fan Song" in anchors
        assert "Spoon Rest" in anchors
        assert "SNR-75" in anchors
        assert len(anchors) == len(set(anchors))

    def test_empty_when_no_identity_passes(self):
        """A manifest with no identity-phase pass → no anchors, no error."""
        manifest = SimpleNamespace(
            passes=[SimpleNamespace(name="radar_power_rf", phase="field_group",
                                    primary_entity_types=["RADAR_SYSTEM"])],
            bundle_key="air_defense_v3",
        )
        anchors = _collect_committed_identity_anchors(
            db=MagicMock(), run_id="run-1", manifest=manifest,
            ontology={"entity_types": []}, document_id="doc-1",
        )
        assert anchors == []

    def test_empty_when_identity_pass_not_complete(self):
        """Identity pass present but its row is not COMPLETE → empty anchors."""
        manifest = _identity_manifest()
        not_done = SimpleNamespace(execution_status="FAILED",
                                   extract_pass_response_json=None)
        with patch("app.workers.pipeline.load_pass_output", return_value=not_done):
            anchors = _collect_committed_identity_anchors(
                db=MagicMock(), run_id="run-1", manifest=manifest,
                ontology={"entity_types": [
                    {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"]},
                ]}, document_id="doc-1",
            )
        assert anchors == []

    def test_never_raises_on_internal_error(self):
        """Any internal failure → returns [] (opportunistic, non-blocking)."""
        manifest = _identity_manifest()
        with patch("app.workers.pipeline.load_pass_output",
                   side_effect=RuntimeError("db down")):
            anchors = _collect_committed_identity_anchors(
                db=MagicMock(), run_id="run-1", manifest=manifest,
                ontology={"entity_types": [
                    {"name": "RADAR_SYSTEM", "identity_fields": ["system_name"]},
                ]}, document_id="doc-1",
            )
        assert anchors == []
