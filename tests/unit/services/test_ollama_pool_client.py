"""Unit tests for OllamaPool routing core.

Pool tests cover acquire/release semantics, least-in-flight selection,
and counter integrity under concurrent acquire (proxy for thread-safety).
HTTP behavior is covered separately in test_ollama_pool_client_http.py.
"""
import contextlib
import json
import logging
import socket
import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.services.ollama_pool_client import (
    OllamaChatClient,
    OllamaEmbeddingClient,
    OllamaPool,
)


@contextlib.contextmanager
def _attach_caplog_to_pool_logger(caplog):
    """Attach caplog's handler to app.services.ollama_pool_client's logger.

    The pool client logger sets propagate=False (so its INFO logs don't
    duplicate up to the root logger when the host process configures
    logging itself), which means caplog's root-level handler never sees
    its records. Tests that need to assert on its log lines must attach
    caplog.handler directly to that named logger for the duration of the
    test. This helper handles attach + restore so individual tests stay
    a single `with` block.
    """
    pool_logger = logging.getLogger("app.services.ollama_pool_client")
    prev_level = pool_logger.level
    pool_logger.setLevel(logging.INFO)
    pool_logger.addHandler(caplog.handler)
    try:
        yield
    finally:
        pool_logger.removeHandler(caplog.handler)
        pool_logger.setLevel(prev_level)


class _FakeClientError(Exception):
    def __init__(self, msg, details=None):
        super().__init__(msg)
        self.details = details


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


def test_chat_client_satisfies_protocol_attrs():
    """Library-side LlmBackend reads `model`, `provider`, `streaming`,
    `last_call_diagnostics` off the client. They must exist."""
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="llama3.3:70b")
    assert client.model == "llama3.3:70b"
    assert client.provider == "ollama"
    assert client.streaming is False
    assert client.last_call_diagnostics is None


def _fake_stream_payload(content: str) -> dict:
    """Synthesize the dict shape _stream_chat_with_watchdog returns."""
    return {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


def test_chat_client_calls_v1_chat_completions():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    captured: list[tuple[str, dict]] = []

    def fake_stream(self, url, body, **kw):
        captured.append((url, body))
        return _fake_stream_payload('{"a": 1}')

    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", fake_stream):
        out = client.get_json_response(
            prompt={"system": "s", "user": "u"},
            schema_json="{}",
            structured_output=False,
        )
    assert out == {"a": 1}
    assert captured[0][0] == "http://only"


def test_chat_client_releases_on_failure():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    with patch.object(
        OllamaChatClient, "_stream_chat_with_watchdog",
        side_effect=httpx.ConnectError("boom"),
    ):
        with pytest.raises(httpx.ConnectError):
            client.get_json_response(prompt="hi", schema_json="{}")
    assert pool._inflight["http://only"] == 0


def test_chat_client_retries_on_different_url():
    pool = OllamaPool(urls=["http://a", "http://b"])
    client = OllamaChatClient(pool=pool, model="m")
    seen_urls: list[str] = []

    def fake_stream(self, url, body, **kw):
        seen_urls.append(url)
        if len(seen_urls) == 1:
            raise httpx.ConnectError("boom")
        return _fake_stream_payload('{"x": 1}')

    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", fake_stream):
        out = client.get_json_response(prompt="hi", schema_json="{}")
    assert out == {"x": 1}
    assert len(seen_urls) == 2
    assert seen_urls[0] != seen_urls[1]


def test_chat_client_no_retry_with_single_url():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    with patch.object(
        OllamaChatClient, "_stream_chat_with_watchdog",
        side_effect=httpx.ConnectError("boom"),
    ):
        with pytest.raises(httpx.ConnectError):
            client.get_json_response(prompt="hi", schema_json="{}")


def test_stream_watchdog_parses_chunked_sse_response():
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def chunk(payload: bytes) -> bytes:
        return f"{len(payload):x}\r\n".encode() + payload + b"\r\n"

    def serve_sse_response():
        conn, _addr = server.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
            )
            conn.sendall(chunk(
                b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
            ))
            conn.sendall(chunk(
                b'data: {"choices":[{"delta":{"content":"lo"},'
                b'"finish_reason":"stop"}]}\n\n',
            ))
            conn.sendall(chunk(b"data: [DONE]\n\n"))
            conn.sendall(b"0\r\n\r\n")

    thread = threading.Thread(target=serve_sse_response, daemon=True)
    thread.start()

    pool = OllamaPool(urls=[f"http://127.0.0.1:{port}"])
    client = OllamaChatClient(pool=pool, model="m")
    try:
        payload = client._stream_chat_with_watchdog(
            f"http://127.0.0.1:{port}",
            {"model": "m", "messages": [{"role": "user", "content": "x"}]},
            no_progress_seconds=1.0,
        )
    finally:
        client.close()
        server.close()

    choice = payload["choices"][0]
    assert choice["message"]["content"] == "hello"
    assert choice["finish_reason"] == "stop"


def test_stream_watchdog_times_out_when_sse_headers_are_followed_by_silence():
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    stop = threading.Event()

    def serve_silent_sse_response():
        conn, _addr = server.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
            )
            stop.wait(5.0)

    thread = threading.Thread(target=serve_silent_sse_response, daemon=True)
    thread.start()

    pool = OllamaPool(urls=[f"http://127.0.0.1:{port}"])
    client = OllamaChatClient(pool=pool, model="m")
    started = time.monotonic()
    try:
        with pytest.raises(httpx.ReadTimeout, match="no SSE bytes from Ollama"):
            client._stream_chat_with_watchdog(
                f"http://127.0.0.1:{port}",
                {"model": "m", "messages": [{"role": "user", "content": "x"}]},
                no_progress_seconds=0.3,
            )
        assert time.monotonic() - started < 1.0
    finally:
        stop.set()
        client.close()
        server.close()


def test_stream_watchdog_times_out_on_total_wall_clock_even_with_progress():
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    stop = threading.Event()

    def chunk(data: bytes) -> bytes:
        return f"{len(data):X}\r\n".encode() + data + b"\r\n"

    def serve_slow_streaming_response():
        conn, _addr = server.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
            )
            while not stop.wait(0.05):
                conn.sendall(chunk(
                    b'data: {"choices":[{"delta":{"content":" "}}]}\n\n',
                ))

    thread = threading.Thread(target=serve_slow_streaming_response, daemon=True)
    thread.start()

    pool = OllamaPool(urls=[f"http://127.0.0.1:{port}"])
    client = OllamaChatClient(pool=pool, model="m")
    started = time.monotonic()
    try:
        with pytest.raises(httpx.ReadTimeout, match="wall-clock limit"):
            client._stream_chat_with_watchdog(
                f"http://127.0.0.1:{port}",
                {"model": "m", "messages": [{"role": "user", "content": "x"}]},
                no_progress_seconds=2.0,
                max_wall_seconds=0.25,
            )
        assert time.monotonic() - started < 1.0
    finally:
        stop.set()
        client.close()
        server.close()


def test_chat_client_format_schema_when_structured_output_true():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    captured_bodies: list[dict] = []

    def fake_stream(self, url, body, **kw):
        captured_bodies.append(body)
        return _fake_stream_payload("{}")

    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", fake_stream):
        client.get_json_response(
            prompt="hi",
            schema_json='{"type":"object"}',
            structured_output=True,
        )
    assert captured_bodies[0]["format"] == {"type": "object"}


def test_chat_client_format_json_when_structured_output_false():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    fake_payload = {"choices": [{"message": {"content": "{}"}}]}
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.get_json_response(
            prompt="hi", schema_json="{}", structured_output=False,
        )
    body = mock_post.call_args.args[1]
    assert body["format"] == "json"


# ----- Tests for the v3 additions: ClientError wrapping, empty-content
#       rejection, legacy schema strip, parse_json_fn injection. -----


def test_get_json_response_wraps_parse_failure_as_client_error():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m", client_error_cls=_FakeClientError,
    )
    fake_payload = {
        "choices": [{"message": {"content": "not valid json"}}]
    }
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload):
        with pytest.raises(_FakeClientError):
            client.get_json_response(prompt="hi", schema_json="{}")
    assert client.last_call_diagnostics["structured_failed"] is True


def test_get_json_response_rejects_empty_content_with_only_reasoning():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m", client_error_cls=_FakeClientError,
    )
    fake_payload = {"choices": [{
        "message": {"content": "", "reasoning_content": "thinking..."}
    }]}
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload):
        with pytest.raises(_FakeClientError, match="empty content"):
            client.get_json_response(prompt="hi", schema_json="{}")


def test_chat_falls_back_to_reasoning_content_when_empty():
    """app-side chat() must preserve current reasoning_content fallback."""
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    fake_payload = {"choices": [{
        "message": {"content": "", "reasoning_content": "the answer"}
    }]}
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload):
        out = client.chat(messages=[{"role": "user", "content": "x"}])
    assert out == "the answer"


def test_legacy_schema_strip_fires_when_force_json_and_unstructured():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m", force_json_mode=True)
    fake_payload = {"choices": [{"message": {"content": "{}"}}]}
    huge_schema = "{}" * 1000
    legacy_user = (
        "Extract from this:\n=== DOC ===\nbody\n=== END DOC ===\n\n"
        f"=== TARGET SCHEMA ===\n{huge_schema}\n=== END SCHEMA ===\n\n"
        "Return ONLY a JSON object."
    )
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.get_json_response(
            prompt={"system": "s", "user": legacy_user},
            schema_json="{}",
            structured_output=False,
        )
    sent = mock_post.call_args.args[1]["messages"]
    user_msg = next(m["content"] for m in sent if m["role"] == "user")
    assert "TARGET SCHEMA" not in user_msg
    assert "Return ONLY a JSON object" in user_msg


def test_legacy_strip_does_not_fire_on_structured_output():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m", force_json_mode=True)
    fake_payload = {"choices": [{"message": {"content": "{}"}}]}
    user_with_schema = (
        "body\n\n=== TARGET SCHEMA ===\n{}\n=== END SCHEMA ===\n\nReturn JSON."
    )
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.get_json_response(
            prompt={"user": user_with_schema},
            schema_json="{}",
            structured_output=True,
        )
    sent = mock_post.call_args.args[1]["messages"]
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
    fake_payload = {
        "choices": [{"message": {"content": '```json\n{"x": 1}\n```'}}]
    }
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload):
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
    fake_payload = {
        "choices": [{"message": {"content": "garbage that doesn't parse"}}]
    }
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload):
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
    fake_payload = {
        "choices": [{"message": {"content": "not json"}}]
    }
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload):
        with pytest.raises(RealClientError):
            client.get_json_response(prompt="hi", schema_json="{}")


# Removed: test_chat_per_call_timeout_overrides_default — the per-call
# timeout_s argument is no longer plumbed into the httpx call after the
# 2026-05-01 streaming-watchdog refactor. The streaming helper enforces a
# fixed 120s no-progress timeout; total wall-time is bounded by the
# orchestrator-level BATCH_HARD_TIMEOUT instead.


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
    fake_payload = {"choices": [{"message": {"content": "hi"}}]}
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.chat(
            messages=[{"role": "user", "content": "x"}], model="other-model",
        )
    body = mock_post.call_args.args[1]
    assert body["model"] == "other-model"


def test_chat_force_json_sets_format_json():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m")
    fake_payload = {"choices": [{"message": {"content": "{}"}}]}
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.chat(
            messages=[{"role": "user", "content": "x"}], force_json=True,
        )
    body = mock_post.call_args.args[1]
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
    fake_payload = {"choices": [{"message": {"content": "{}"}}]}
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.get_json_response(
            prompt="hi",
            schema_json='{"type":"object"}',
            structured_output=True,
        )
    assert transformed and transformed[0]["x-stripped"] is True
    body = mock_post.call_args.args[1]
    # The schema sent to Ollama is the transformed one, not the original.
    assert body["format"]["x-stripped"] is True


def test_threshold_forces_format_json_for_oversize_schema():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m", structured_output_threshold_chars=50,
    )
    fake_payload = {"choices": [{"message": {"content": "{}"}}]}
    big_schema = json.dumps({
        "type": "object",
        "properties": {f"f{i}": {"type": "string"} for i in range(20)},
    })
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.get_json_response(
            prompt="hi", schema_json=big_schema, structured_output=True,
        )
    body = mock_post.call_args.args[1]
    # Schema > 50 chars → falls back to plain "json".
    assert body["format"] == "json"


def test_force_json_mode_overrides_structured():
    """force_json_mode=True must beat structured_output=True (Ollama's
    constrained decoder degrades on big schemas; force_json_mode says
    'never use schema-format, even when structured)."""
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="m", force_json_mode=True)
    fake_payload = {"choices": [{"message": {"content": "{}"}}]}
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.get_json_response(
            prompt="hi",
            schema_json='{"type":"object"}',
            structured_output=True,
        )
    body = mock_post.call_args.args[1]
    assert body["format"] == "json"


def test_default_extra_params_pass_through():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(
        pool=pool, model="m",
        default_extra_params={
            "top_p": 0.9, "top_k": 40, "seed": 42, "stop": ["END"],
        },
    )
    fake_payload = {"choices": [{"message": {"content": "ok"}}]}
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.chat(messages=[{"role": "user", "content": "x"}])
    body = mock_post.call_args.args[1]
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
    fake_payload = {"choices": [{"message": {"content": "{}"}}]}
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.get_json_response(
            prompt="hi", schema_json="{}", structured_output=False,
        )
    body = mock_post.call_args.args[1]
    assert body["seed"] == 42
    assert body["top_p"] == 0.9


def test_think_low_passed_for_gpt_oss():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="gpt-oss:120b", think="low")
    fake_payload = {"choices": [{"message": {"content": "ok"}}]}
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.chat(messages=[{"role": "user", "content": "x"}])
    body = mock_post.call_args.args[1]
    assert body["think"] == "low"


def test_think_low_dropped_for_non_gpt_oss():
    """low/medium/high think levels are gpt-oss-only — silently drop on
    other Ollama models so they don't error on an unsupported value."""
    pool = OllamaPool(urls=["http://only"])
    client = OllamaChatClient(pool=pool, model="gemma4:31b", think="low")
    fake_payload = {"choices": [{"message": {"content": "ok"}}]}
    with patch.object(OllamaChatClient, "_stream_chat_with_watchdog", return_value=fake_payload) as mock_post:
        client.chat(messages=[{"role": "user", "content": "x"}])
    body = mock_post.call_args.args[1]
    assert "think" not in body


# Removed: test_malformed_response_envelope_wraps_as_client_error.
# After the 2026-05-01 streaming-watchdog refactor, the client no longer
# does a single resp.json() call that can fail with JSONDecodeError on a
# bad envelope. Instead each SSE chunk is parsed independently and
# malformed chunks are skipped. The "no usable content" failure mode is
# already exercised by the empty-content / parse_json_fn tests above.


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
    assert mock_post.call_args.kwargs["json"] == {
        "model": "bge-m3", "input": ["hello", "world"],
    }


def test_embedding_client_preserves_input_order():
    pool = OllamaPool(urls=["http://only"])
    client = OllamaEmbeddingClient(pool=pool, model="bge-m3")
    fake = MagicMock()
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


# ----- Task 6.5b: success-URL INFO logging (per-call observability) -----


def test_chat_logs_success_url_at_info(caplog):
    with _attach_caplog_to_pool_logger(caplog):
        pool = OllamaPool(urls=["http://only"])
        client = OllamaChatClient(pool=pool, model="m")
        fake_payload = {"choices": [{"message": {"content": "ok"}}]}
        with patch.object(
            OllamaChatClient, "_stream_chat_with_watchdog",
            return_value=fake_payload,
        ):
            client.chat(messages=[{"role": "user", "content": "x"}])
    assert any(
        "OllamaChatClient: ok" in rec.message and "http://only" in rec.message
        for rec in caplog.records
    ), [r.message for r in caplog.records]


def test_embedding_logs_success_url_at_info(caplog):
    with _attach_caplog_to_pool_logger(caplog):
        pool = OllamaPool(urls=["http://only"])
        client = OllamaEmbeddingClient(pool=pool, model="bge-m3")
        fake = MagicMock()
        fake.json.return_value = {"data": [{"index": 0, "embedding": [0.1]}]}
        fake.raise_for_status.return_value = None
        with patch("httpx.Client.post", return_value=fake):
            client.embed(["hello"])
    assert any(
        "OllamaEmbeddingClient: ok" in rec.message and "http://only" in rec.message
        for rec in caplog.records
    ), [r.message for r in caplog.records]
