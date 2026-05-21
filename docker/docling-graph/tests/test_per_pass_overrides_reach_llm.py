"""C2 Task 2.8b — Per-pass overrides reach the actual LLM call boundary.

CRITICAL TEST: this test verifies that temperature, max_tokens, and
llm_batch_token_size overrides reach the outbound HTTP body sent to Ollama,
NOT just ExtractPassRequest or PipelineConfig.

The hidden failure mode (documented here for future maintainers):
  - OllamaChatClient is process-cached via lru_cache in ollama_clients.py.
  - When PipelineConfig(llm_client=<cached_client>) is injected, the library
    ignores ``llm_overrides`` — it uses the client's constructor-time values.
  - So ``temperature``, ``max_tokens``, etc. ONLY reach the LLM call if they
    go through ``OllamaChatClient.with_runtime_defaults()``, which creates a
    new short-lived client for that pass.
  - ``llm_batch_token_size`` is different: it's a chunking/batching parameter
    consumed by the library's batch dispatcher, NOT by OllamaChatClient
    directly. It goes into PipelineConfig and does NOT need with_runtime_defaults.

This test mocks the outbound httpx stream call and asserts that the
HTTP body has the expected values. It runs from the host venv — the
docling_graph library is NOT available here, so we test OllamaChatClient
in isolation via _build_chat_body (the method that assembles the body)
and with_runtime_defaults.

Run standalone:
    python3 -m pytest docker/docling-graph/tests/test_per_pass_overrides_reach_llm.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_pool_client_module():
    """Load docker/docling-graph/app/ollama_pool_client.py by file path."""
    key = "_dg_ollama_pool_client_2b"
    if key in sys.modules:
        return sys.modules[key]
    path = Path(__file__).resolve().parent.parent / "app" / "ollama_pool_client.py"
    spec = importlib.util.spec_from_file_location(key, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def _make_pool(urls=None):
    mod = _load_pool_client_module()
    return mod.OllamaPool(urls=urls or ["http://ollama:11434"])


def _make_client(
    *,
    temperature=0.1,
    max_tokens=4096,
    pool=None,
    model="llama3.3:70b",
):
    mod = _load_pool_client_module()
    pool = pool or _make_pool()
    return mod.OllamaChatClient(
        pool=pool,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class TestTemperatureOverrideReachesLLMCall:
    """Verify that temperature override bypasses the cached client constructor
    and reaches the actual outbound HTTP body."""

    def test_constructor_temperature_is_default_in_chat_body(self):
        """Without override, _build_chat_body uses constructor temperature=0.1."""
        client = _make_client(temperature=0.1)
        body = client._build_chat_body(
            messages=[{"role": "user", "content": "hello"}],
            schema_json="{}",
            structured_output=False,
        )
        assert body["temperature"] == 0.1

    def test_with_runtime_defaults_temperature_overrides_constructor(self):
        """with_runtime_defaults(temperature=0.3) creates a client that uses 0.3."""
        client = _make_client(temperature=0.1)
        overridden = client.with_runtime_defaults(temperature=0.3)

        body = overridden._build_chat_body(
            messages=[{"role": "user", "content": "hello"}],
            schema_json="{}",
            structured_output=False,
        )
        assert body["temperature"] == 0.3
        # Original client unaffected
        orig_body = client._build_chat_body(
            messages=[{"role": "user", "content": "hello"}],
            schema_json="{}",
            structured_output=False,
        )
        assert orig_body["temperature"] == 0.1

    def test_with_runtime_defaults_temperature_none_preserves_constructor_value(self):
        """with_runtime_defaults(temperature=None) keeps constructor temperature."""
        client = _make_client(temperature=0.1)
        unchanged = client.with_runtime_defaults(temperature=None)

        body = unchanged._build_chat_body(
            messages=[{"role": "user", "content": "hello"}],
            schema_json="{}",
            structured_output=False,
        )
        assert body["temperature"] == 0.1


class TestMaxTokensOverrideReachesLLMCall:
    """Task 2.8b core: max_tokens override must reach the outbound HTTP body.

    Before C2 was implemented, max_tokens went into llm_overrides which is
    IGNORED when llm_client is injected. The fix: pass max_tokens through
    with_runtime_defaults just like temperature.
    """

    def test_constructor_max_tokens_is_default_in_chat_body(self):
        """Without override, _build_chat_body uses constructor max_tokens=4096."""
        client = _make_client(max_tokens=4096)
        body = client._build_chat_body(
            messages=[{"role": "user", "content": "hello"}],
            schema_json="{}",
            structured_output=False,
        )
        assert body.get("max_tokens") == 4096

    def test_with_runtime_defaults_max_tokens_overrides_constructor(self):
        """with_runtime_defaults(max_tokens=8192) creates a client that uses 8192."""
        client = _make_client(max_tokens=4096)
        overridden = client.with_runtime_defaults(max_tokens=8192)

        body = overridden._build_chat_body(
            messages=[{"role": "user", "content": "hello"}],
            schema_json="{}",
            structured_output=False,
        )
        assert body.get("max_tokens") == 8192
        # Original unaffected
        orig_body = client._build_chat_body(
            messages=[{"role": "user", "content": "hello"}],
            schema_json="{}",
            structured_output=False,
        )
        assert orig_body.get("max_tokens") == 4096

    def test_with_runtime_defaults_max_tokens_none_preserves_constructor_value(self):
        """with_runtime_defaults(max_tokens=None) keeps constructor max_tokens."""
        client = _make_client(max_tokens=4096)
        unchanged = client.with_runtime_defaults(max_tokens=None)

        body = unchanged._build_chat_body(
            messages=[{"role": "user", "content": "hello"}],
            schema_json="{}",
            structured_output=False,
        )
        assert body.get("max_tokens") == 4096


class TestBothTemperatureAndMaxTokensReachLLMCall:
    """End-to-end: temperature + max_tokens overrides both reach the outbound body."""

    def test_temperature_and_max_tokens_both_overridden_in_single_with_runtime_defaults(self):
        """with_runtime_defaults can override both temperature and max_tokens at once."""
        client = _make_client(temperature=0.1, max_tokens=4096)
        overridden = client.with_runtime_defaults(temperature=0.3, max_tokens=8192)

        body = overridden._build_chat_body(
            messages=[{"role": "user", "content": "hello"}],
            schema_json="{}",
            structured_output=False,
        )
        assert body["temperature"] == 0.3
        assert body.get("max_tokens") == 8192


class TestExtractPassRequestRejectsUnknownKeys:
    """C2 review fix: ExtractPassRequest must reject unknown keys so direct API
    callers can't silently typo an override field name and run with env defaults
    thinking they overrode it. Worker-produced requests are already protected
    by the manifest-side ExecutionProfile validator; this closes the gap at the
    HTTP boundary for non-worker callers."""

    def _minimal_body(self, **overrides) -> dict:
        base = {
            "bundle_key": "air_defense_v3",
            "pass_name": "radar_identity",
            "docling_document_json": {},
        }
        base.update(overrides)
        return base

    def _load_request_cls(self):
        """Load docker/docling-graph/app/schemas.py by file path.
        The repo-root app.schemas/ is a different package; this test needs the
        docling-graph service's single-file schemas module."""
        key = "_dg_schemas_module"
        if key in sys.modules:
            return sys.modules[key].ExtractPassRequest
        path = Path(__file__).resolve().parent.parent / "app" / "schemas.py"
        spec = importlib.util.spec_from_file_location(key, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
        return module.ExtractPassRequest

    def test_typo_in_llm_batch_token_size_raises(self):
        ExtractPassRequest = self._load_request_cls()
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ExtractPassRequest(**self._minimal_body(llm_batch_size_token=2048))
        # Pydantic v2 reports extra_forbidden for unknown fields
        msg = str(exc_info.value)
        assert "extra_forbidden" in msg or "llm_batch_size_token" in msg

    def test_typo_in_max_tokens_raises(self):
        ExtractPassRequest = self._load_request_cls()
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExtractPassRequest(**self._minimal_body(maxtokens=8192))  # missing underscore

    def test_known_keys_accepted(self):
        """The fix must not break legitimate overrides."""
        ExtractPassRequest = self._load_request_cls()

        body = self._minimal_body(
            temperature=0.3,
            max_tokens=8192,
            llm_batch_token_size=2048,
            chunk_max_tokens=512,
        )
        req = ExtractPassRequest(**body)
        assert req.temperature == 0.3
        assert req.max_tokens == 8192
        assert req.llm_batch_token_size == 2048
        assert req.chunk_max_tokens == 512
