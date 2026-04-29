"""Unit tests for OllamaPool routing core.

Pool tests cover acquire/release semantics, least-in-flight selection,
and counter integrity under concurrent acquire (proxy for thread-safety).
HTTP behavior is covered separately in test_ollama_pool_client_http.py.
"""
import json
import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.services.ollama_pool_client import OllamaPool


def test_pool_picks_lowest_inflight():
    pool = OllamaPool(urls=["http://a", "http://b", "http://c"])
    pool._inflight["http://a"] = 2
    pool._inflight["http://b"] = 1
    pool._inflight["http://c"] = 3
    url = pool.acquire()
    assert url == "http://b"
    assert pool._inflight["http://b"] == 2


def test_pool_round_robin_on_ties():
    """When all inflight counts are equal (the dominant case for serial
    workloads), successive acquisitions must rotate, not always pick urls[0]."""
    pool = OllamaPool(urls=["http://a", "http://b", "http://c"])
    seen: list[str] = []
    for _ in range(6):
        u = pool.acquire()
        pool.release(u)
        seen.append(u)
    # In 6 tied acquisitions across 3 URLs, each URL should appear at least
    # once (round-robin gives exactly 2 each).
    assert set(seen) == {"http://a", "http://b", "http://c"}


def test_pool_release_decrements():
    pool = OllamaPool(urls=["http://a"])
    pool.acquire()
    assert pool._inflight["http://a"] == 1
    pool.release("http://a")
    assert pool._inflight["http://a"] == 0


def test_pool_acquire_release_balanced_concurrent():
    """Sanity check on counter integrity under thread races. We don't assert
    a specific distribution — only that no count goes negative and the total
    matches the number of releases."""
    pool = OllamaPool(urls=["http://a", "http://b"])
    N = 200

    def worker():
        for _ in range(N):
            url = pool.acquire()
            pool.release(url)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert pool._inflight["http://a"] == 0
    assert pool._inflight["http://b"] == 0


def test_pool_rejects_empty_urls():
    with pytest.raises(ValueError, match="at least one URL"):
        OllamaPool(urls=[])


def test_pool_with_single_url_always_picks_it():
    pool = OllamaPool(urls=["http://only"])
    for _ in range(5):
        url = pool.acquire()
        assert url == "http://only"
        pool.release(url)


from app.services.ollama_pool_client import OllamaChatClient


def test_chat_client_satisfies_protocol_attrs():
    """Library-side LlmBackend reads `model`, `provider`, `streaming`,
    `last_call_diagnostics` off the client. They must exist."""
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="llama3.3:70b")
    assert client.model == "llama3.3:70b"
    assert client.provider == "ollama"
    assert client.streaming is False
    assert client.last_call_diagnostics is None


def test_chat_client_calls_v1_chat_completions():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "choices": [{"message": {"content": '{"a": 1}'}}]
    }
    fake_resp.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake_resp) as mock_post:
        out = client.get_json_response(
            prompt={"system": "s", "user": "u"},
            schema_json="{}",
            structured_output=False,
        )
    assert out == {"a": 1}
    assert mock_post.call_args.args[0] == "http://only/v1/chat/completions"


def test_chat_client_releases_on_failure():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    with patch(
        "httpx.Client.post",
        side_effect=httpx.ConnectError("boom"),
    ):
        with pytest.raises(httpx.ConnectError):
            client.get_json_response(prompt="hi", schema_json="{}")
    assert pool._inflight["http://only"] == 0


def test_chat_client_retries_on_different_url():
    pool = OllamaPool(urls=["http://a", "http://b"])
    client = OllamaChatClient(pool=pool, model="m")
    good = MagicMock()
    good.json.return_value = {
        "choices": [{"message": {"content": '{"x": 1}'}}]
    }
    good.raise_for_status.return_value = None
    seq = [httpx.ConnectError("boom"), good]
    with patch("httpx.Client.post", side_effect=seq) as mock_post:
        out = client.get_json_response(prompt="hi", schema_json="{}")
    assert out == {"x": 1}
    # Both URLs were tried; second call landed on the survivor
    assert mock_post.call_count == 2
    urls = [call.args[0] for call in mock_post.call_args_list]
    assert urls[0] != urls[1]


def test_chat_client_no_retry_with_single_url():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    with patch("httpx.Client.post", side_effect=httpx.ConnectError("boom")):
        with pytest.raises(httpx.ConnectError):
            client.get_json_response(prompt="hi", schema_json="{}")


def test_chat_client_format_schema_when_structured_output_true():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.get_json_response(
            prompt="hi",
            schema_json='{"type":"object"}',
            structured_output=True,
        )
    body = mock_post.call_args.kwargs["json"]
    assert body["format"] == {"type": "object"}


def test_chat_client_format_json_when_structured_output_false():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.get_json_response(
            prompt="hi", schema_json="{}", structured_output=False,
        )
    body = mock_post.call_args.kwargs["json"]
    assert body["format"] == "json"


# ----- Tests for the v3 additions: ClientError wrapping, empty-content
#       rejection, legacy schema strip, parse_json_fn injection. -----


class _FakeClientError(Exception):
    def __init__(self, msg, details=None):
        super().__init__(msg)
        self.details = details


def test_get_json_response_wraps_parse_failure_as_client_error():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m", client_error_cls=_FakeClientError,
    )
    fake = MagicMock()
    fake.json.return_value = {
        "choices": [{"message": {"content": "not valid json"}}]
    }
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake):
        with pytest.raises(_FakeClientError):
            client.get_json_response(prompt="hi", schema_json="{}")
    assert client.last_call_diagnostics["structured_failed"] is True


def test_get_json_response_rejects_empty_content_with_only_reasoning():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m", client_error_cls=_FakeClientError,
    )
    fake = MagicMock()
    fake.json.return_value = {"choices": [{
        "message": {"content": "", "reasoning_content": "thinking..."}
    }]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake):
        with pytest.raises(_FakeClientError, match="empty content"):
            client.get_json_response(prompt="hi", schema_json="{}")


def test_chat_falls_back_to_reasoning_content_when_empty():
    """app-side chat() must preserve current reasoning_content fallback."""
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    fake = MagicMock()
    fake.json.return_value = {"choices": [{
        "message": {"content": "", "reasoning_content": "the answer"}
    }]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake):
        out = client.chat(messages=[{"role": "user", "content": "x"}])
    assert out == "the answer"


def test_legacy_schema_strip_fires_when_force_json_and_unstructured():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m", force_json_mode=True)
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    fake.raise_for_status.return_value = None
    huge_schema = "{}" * 1000
    legacy_user = (
        "Extract from this:\n=== DOC ===\nbody\n=== END DOC ===\n\n"
        f"=== TARGET SCHEMA ===\n{huge_schema}\n=== END SCHEMA ===\n\n"
        "Return ONLY a JSON object."
    )
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.get_json_response(
            prompt={"system": "s", "user": legacy_user},
            schema_json="{}",
            structured_output=False,
        )
    sent = mock_post.call_args.kwargs["json"]["messages"]
    user_msg = next(m["content"] for m in sent if m["role"] == "user")
    assert "TARGET SCHEMA" not in user_msg
    assert "Return ONLY a JSON object" in user_msg


def test_legacy_strip_does_not_fire_on_structured_output():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m", force_json_mode=True)
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    fake.raise_for_status.return_value = None
    user_with_schema = (
        "body\n\n=== TARGET SCHEMA ===\n{}\n=== END SCHEMA ===\n\nReturn JSON."
    )
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.get_json_response(
            prompt={"user": user_with_schema},
            schema_json="{}",
            structured_output=True,
        )
    sent = mock_post.call_args.kwargs["json"]["messages"]
    user_msg = next(m["content"] for m in sent if m["role"] == "user")
    # On structured calls we DON'T strip — the library uses a different
    # prompt format for those, and the strip would corrupt them.
    assert "TARGET SCHEMA" in user_msg


def test_parse_json_fn_used_when_provided():
    pool = OllamaPool(urls=["http://only"])
    parse_calls: list[str] = []
    def loose_parse(s: str):
        parse_calls.append(s)
        # Strip code fences.
        s = s.replace("```json", "").replace("```", "").strip()
        return json.loads(s)
    client = OllamaChatClient(
        pool=pool, model="m", parse_json_fn=loose_parse,
    )
    fake = MagicMock()
    fake.json.return_value = {
        "choices": [{"message": {"content": '```json\n{"x": 1}\n```'}}]
    }
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake):
        out = client.get_json_response(prompt="hi", schema_json="{}")
    assert out == {"x": 1}
    assert parse_calls  # the custom parser was used


def test_parse_json_fn_returning_none_wraps_as_client_error():
    """parse_llm_json_loose returns None on failure; the client must
    convert that into a ClientError so LlmBackend's fallback triggers."""
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m",
        parse_json_fn=lambda _s: None,  # always fails
        client_error_cls=_FakeClientError,
    )
    fake = MagicMock()
    fake.json.return_value = {
        "choices": [{"message": {"content": "garbage that doesn't parse"}}]
    }
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake):
        with pytest.raises(_FakeClientError, match="returned None"):
            client.get_json_response(prompt="hi", schema_json="{}")
    assert client.last_call_diagnostics["structured_failed"] is True


def test_get_json_response_with_real_docling_client_error():
    """Integration check: when docling_graph.exceptions.ClientError is wired
    in, we get a real ClientError (not _FakeClientError) on failures."""
    pytest.importorskip("docling_graph")
    from docling_graph.exceptions import ClientError as RealClientError
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m", client_error_cls=RealClientError,
    )
    fake = MagicMock()
    fake.json.return_value = {
        "choices": [{"message": {"content": "not json"}}]
    }
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake):
        with pytest.raises(RealClientError):
            client.get_json_response(prompt="hi", schema_json="{}")


def test_chat_per_call_timeout_overrides_default():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m", timeout_s=10.0)
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.chat(messages=[{"role": "user", "content": "x"}], timeout_s=600.0)
    assert mock_post.call_args.kwargs["timeout"] == 600.0


def test_pool_routing_metrics_accumulate():
    pool = OllamaPool(urls=["http://a", "http://b"])
    for _ in range(4):
        u = pool.acquire()
        pool.release(u)
    metrics = pool.routing_metrics
    assert sum(metrics.values()) == 4
    # Both URLs got picked thanks to round-robin tie-break.
    assert metrics["http://a"] >= 1
    assert metrics["http://b"] >= 1


# ----- Coverage gaps from v6: load-bearing knobs of OllamaChatClient -----


def test_chat_per_call_model_override():
    """Different roles share one cached client and override model per call."""
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="default-model")
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.chat(
            messages=[{"role": "user", "content": "x"}], model="other-model",
        )
    body = mock_post.call_args.kwargs["json"]
    assert body["model"] == "other-model"


def test_chat_force_json_sets_format_json():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.chat(
            messages=[{"role": "user", "content": "x"}], force_json=True,
        )
    body = mock_post.call_args.kwargs["json"]
    assert body["format"] == "json"


def test_schema_transform_applied_before_format_schema():
    """The schema_transform callback must run before serializing to format=."""
    pool = OllamaPool(urls=["http://only"])
    transformed: list[dict] = []
    def xform(schema_dict: dict) -> dict:
        out = {**schema_dict, "x-stripped": True}
        transformed.append(out)
        return out
    client = OllamaChatClient(pool=pool, model="m", schema_transform=xform)
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.get_json_response(
            prompt="hi",
            schema_json='{"type":"object"}',
            structured_output=True,
        )
    assert transformed and transformed[0]["x-stripped"] is True
    body = mock_post.call_args.kwargs["json"]
    # The schema sent to Ollama is the transformed one, not the original.
    assert body["format"]["x-stripped"] is True


def test_threshold_forces_format_json_for_oversize_schema():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m", structured_output_threshold_chars=50,
    )
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    fake.raise_for_status.return_value = None
    big_schema = json.dumps({
        "type": "object",
        "properties": {f"f{i}": {"type": "string"} for i in range(20)},
    })
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.get_json_response(
            prompt="hi", schema_json=big_schema, structured_output=True,
        )
    body = mock_post.call_args.kwargs["json"]
    # Schema > 50 chars → falls back to plain "json".
    assert body["format"] == "json"


def test_force_json_mode_overrides_structured():
    """force_json_mode=True must beat structured_output=True (Ollama's
    constrained decoder degrades on big schemas; force_json_mode says
    'never use schema-format, even when structured)."""
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m", force_json_mode=True)
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.get_json_response(
            prompt="hi",
            schema_json='{"type":"object"}',
            structured_output=True,
        )
    body = mock_post.call_args.kwargs["json"]
    assert body["format"] == "json"


def test_default_extra_params_pass_through():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m",
        default_extra_params={
            "top_p": 0.9, "top_k": 40, "seed": 42, "stop": ["END"],
        },
    )
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.chat(messages=[{"role": "user", "content": "x"}])
    body = mock_post.call_args.kwargs["json"]
    assert body["top_p"] == 0.9
    assert body["top_k"] == 40
    assert body["seed"] == 42
    assert body["stop"] == ["END"]


def test_default_extra_params_on_get_json_response():
    """docling-graph extraction goes through get_json_response(), not chat().
    Verify default_extra_params reach the body on that path too — otherwise
    seed/stop/top_p never affect extraction calls."""
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m",
        default_extra_params={"seed": 42, "top_p": 0.9},
    )
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.get_json_response(
            prompt="hi", schema_json="{}", structured_output=False,
        )
    body = mock_post.call_args.kwargs["json"]
    assert body["seed"] == 42
    assert body["top_p"] == 0.9


def test_think_low_passed_for_gpt_oss():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="gpt-oss:120b", think="low")
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.chat(messages=[{"role": "user", "content": "x"}])
    body = mock_post.call_args.kwargs["json"]
    assert body["think"] == "low"


def test_think_low_dropped_for_non_gpt_oss():
    """low/medium/high think levels are gpt-oss-only — silently drop on
    other Ollama models so they don't error on an unsupported value."""
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="gemma4:31b", think="low")
    fake = MagicMock()
    fake.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        client.chat(messages=[{"role": "user", "content": "x"}])
    body = mock_post.call_args.kwargs["json"]
    assert "think" not in body


def test_malformed_response_envelope_wraps_as_client_error():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m", client_error_cls=_FakeClientError,
    )
    fake = MagicMock()
    fake.raise_for_status.return_value = None
    fake.json.side_effect = json.JSONDecodeError("expecting value", "", 0)
    fake.text = "<html>nginx 502</html>"
    with patch("httpx.Client.post", return_value=fake):
        with pytest.raises(_FakeClientError, match="malformed JSON envelope"):
            client.get_json_response(prompt="hi", schema_json="{}")


from app.services.ollama_pool_client import OllamaEmbeddingClient


def test_embedding_client_calls_v1_embeddings():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaEmbeddingClient(pool=pool, model="bge-m3")
    fake = MagicMock()
    fake.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4]},
        ]
    }
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake) as mock_post:
        out = client.embed(["hello", "world"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]
    assert mock_post.call_args.args[0] == "http://only/v1/embeddings"
    body = mock_post.call_args.kwargs["json"]
    assert body == {"model": "bge-m3", "input": ["hello", "world"]}


def test_embedding_client_preserves_input_order():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaEmbeddingClient(pool=pool, model="bge-m3")
    fake = MagicMock()
    # Server returns out-of-order; client must sort by index.
    fake.json.return_value = {
        "data": [
            {"index": 1, "embedding": [9.0]},
            {"index": 0, "embedding": [1.0]},
        ]
    }
    fake.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=fake):
        out = client.embed(["a", "b"])
    assert out == [[1.0], [9.0]]
