"""Direct-Ollama pool client (canonical).

Replaces LiteLLM (used inside docling_graph) and the per-callsite httpx
calls scattered across app/services/. Two clients sit on top of a shared
routing core:

    OllamaPool                — acquire() / release() / least-in-flight
    OllamaChatClient          — /v1/chat/completions; implements docling_graph's
                                LLMClientProtocol so it can be plugged into
                                PipelineConfig(llm_client=...)
    OllamaEmbeddingClient     — /v1/embeddings; thin helper for embedding.py

Constructed by `app.config.Settings.get_ollama_*_urls()` callers; one pool
per role (LLM / VLM / embedding) keyed off the matching env vars.

MIRROR: docker/docling-graph/app/ollama_pool_client.py. The two files are
byte-for-byte identical below the SHARED CODE marker; the docstring is the
only difference. tests/test_pool_client_mirror.py enforces this invariant.
"""
# === SHARED CODE BELOW THIS LINE ===
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Iterator, Literal, Mapping, Optional

import httpx


# TODO #88: Opt-in fail-fast for the legacy schema-strip retry path. When the
# pool exhausts and the docling-graph LLM backend re-issues the call with
# structured_output=False, the client today strips the structured-output
# schema and re-prompts — silently producing JSON that is not validated
# against the original Pydantic schema. Setting this env to "false" turns
# that into a hard failure so the caller can reschedule under structured
# output rather than absorbing schema-unvalidated output into the graph.
# Default "true" preserves existing behavior.
_LEGACY_RETRY_ENABLED = (
    os.environ.get("OLLAMA_LEGACY_RETRY_ENABLED", "true").strip().lower()
    not in {"false", "0", "no", "off"}
)

logger = logging.getLogger(__name__)


def _build_keepalive_http_client(timeout_s: float) -> httpx.Client:
    """Build an httpx.Client with TCP keepalive + a sane read-timeout split.

    Why: with a single blanket timeout (DOCLING_GRAPH_LLM_TIMEOUT can be 20h),
    a silently-dropped TCP connection hangs the read for hours. This client
    splits the timeouts so a dead socket surfaces in minutes:

      * connect=10s    — DNS + TCP handshake should complete fast
      * write=60s      — POST body upload
      * pool=30s       — wait for a free connection slot
      * read=min(timeout_s, 600s) — application-level "no bytes at all" cap.
        gemma4's longest legitimate single-batch generation we've observed
        is ~377s (content-heavy radar batch, 4kB JSON output); 600s is a
        safe ceiling. Tightened from 1800s on 2026-05-01 after the Ollama
        silent-request-loss class hung an extract-pass for 14+min before
        any timeout fired — see OLLAMA_REQUEST_LOSS_DETECTION below.

    Plus TCP keepalive at the socket level so the kernel detects truly-
    dead connections in TCP_KEEPIDLE + KEEPCNT*KEEPINTVL = 60 + 6*15 =
    ~150s, well before the read timeout. Both signals trigger
    OllamaChatClient._post_with_retry's retry-on-different-URL path
    instead of hanging silently.

    Defense-in-depth: actual /v1/chat/completions calls run in streaming
    mode via _stream_chat_with_watchdog, which applies a per-recv
    no-progress timeout of 120s — catches Ollama's silent-request-loss
    class (peer alive at TCP, request abandoned at app layer) without
    waiting for the 600s ceiling.
    """
    socket_options: list[tuple[int, int, int]] = []
    try:
        import socket as _sock
        socket_options = [
            (_sock.SOL_SOCKET, _sock.SO_KEEPALIVE, 1),
            (_sock.IPPROTO_TCP, _sock.TCP_KEEPIDLE, 60),    # idle 60s before first probe
            (_sock.IPPROTO_TCP, _sock.TCP_KEEPINTVL, 15),   # 15s between probes
            (_sock.IPPROTO_TCP, _sock.TCP_KEEPCNT, 6),      # 6 probes -> ~150s detection
        ]
    except (ImportError, AttributeError):
        try:
            import socket as _sock
            socket_options = [(_sock.SOL_SOCKET, _sock.SO_KEEPALIVE, 1)]
        except ImportError:
            socket_options = []
    transport = httpx.HTTPTransport(socket_options=socket_options) \
        if socket_options else None
    return httpx.Client(
        timeout=httpx.Timeout(
            connect=10.0,
            read=min(float(timeout_s), 600.0),
            write=60.0,
            pool=30.0,
        ),
        transport=transport,
    )


# Per-recv watchdog applied to streaming /v1/chat/completions calls. If
# Ollama goes silent for this long mid-response (or never sends a byte after
# accepting the request — the silent-request-loss class), httpx raises
# ReadTimeout from inside _stream_chat_with_watchdog and the retry path
# engages on a different pool URL. Raised to 1200s (was 120s) to tolerate
# first-token latency under heavy GPU contention with gemma4:31b.
_STREAM_NO_PROGRESS_SECONDS = 1200.0

# Absolute wall-clock ceiling for one streamed chat completion. The
# no-progress watchdog catches silent sockets; this catches pathological
# generations that keep streaming low-value JSON forever. Must remain
# strictly greater than _STREAM_NO_PROGRESS_SECONDS or the no-progress
# watchdog never fires.
_STREAM_MAX_WALL_SECONDS = 1800.0
# Attach a stream handler when the host process hasn't configured logging.
# Without this, _maybe_strip_legacy_schema and _post_chat_with_retry log
# silently in production (the api and docling-graph processes don't call
# logging.basicConfig), leaving us blind to schema-strip / retry behavior.
# Mirrors the pattern at docker/docling-graph/app/main.py:57-63.
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Ollama format-mode literal — used for `format="json"` in request bodies.
# Extracted as a module-level constant so the four occurrences below stay
# in sync if Ollama ever introduces a new mode.
_FORMAT_JSON = "json"


class OllamaPool:
    """URL pool with least-in-flight routing.

    Tracks per-URL request counts behind a lock; `acquire()` returns the URL
    with the lowest current count and increments it; `release()` decrements.
    Always wrap acquire+release in try/finally so a failing call still
    releases its slot.
    """

    def __init__(self, urls: list[str]) -> None:
        if not urls:
            raise ValueError("OllamaPool requires at least one URL")
        seen: set[str] = set()
        ordered: list[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                ordered.append(u)
        self._urls: list[str] = ordered
        # Precomputed URL→index map; avoids self._urls.index(u) inside the lock.
        self._url_index: dict[str, int] = {u: i for i, u in enumerate(ordered)}
        self._inflight: dict[str, int] = {u: 0 for u in ordered}
        # Round-robin cursor — used to break ties when multiple URLs share
        # the lowest in-flight count. Without this, serial workloads always
        # pick urls[0] (min() returns first match on ties), defeating fan-out.
        # Initialized to -1 so the first tied acquisition (after the
        # cursor++ inside acquire) lands on urls[0].
        self._rr_cursor: int = -1
        self._lock = threading.Lock()
        # Per-URL routing counter for diagnostics / Gate 5 fan-out check.
        # Atomic with the lock that protects _inflight.
        self._served: dict[str, int] = {u: 0 for u in ordered}

    @property
    def urls(self) -> list[str]:
        return list(self._urls)

    @property
    def routing_metrics(self) -> dict[str, int]:
        """Snapshot of per-URL request counts (cumulative since pool creation)."""
        with self._lock:
            return dict(self._served)

    def acquire(self, exclude: set[str] | None = None) -> str:
        """Pick the URL with the lowest in-flight count (excluding any URL
        listed in `exclude`); increment in-flight + served counters and
        return the URL.

        Tie-break: round-robin across URLs sharing the minimum in-flight
        count. Cursor advances monotonically.
        """
        with self._lock:
            candidates = [u for u in self._urls if not exclude or u not in exclude]
            if not candidates:
                raise RuntimeError(
                    f"No URLs available (all {len(self._urls)} excluded)"
                )
            min_inflight = min(self._inflight[u] for u in candidates)
            tied = [u for u in candidates if self._inflight[u] == min_inflight]
            if len(tied) == 1:
                url = tied[0]
            else:
                # Round-robin among ties. Use cursor mod len(_urls) to keep
                # rotation stable even when `exclude` shrinks the candidate
                # set on retries.
                self._rr_cursor = (self._rr_cursor + 1) % len(self._urls)
                # Pick the tied URL whose index is closest to (but not below)
                # the cursor position; wrap if needed.
                tied_indexed = sorted(
                    (self._url_index[u], u) for u in tied
                )
                pick = next(
                    (u for idx, u in tied_indexed if idx >= self._rr_cursor),
                    tied_indexed[0][1],
                )
                url = pick
            self._inflight[url] += 1
            self._served[url] += 1
            return url

    def release(self, url: str) -> None:
        with self._lock:
            self._inflight[url] = max(0, self._inflight[url] - 1)


class OllamaChatClient:
    """Implements docling_graph's LLMClientProtocol against an Ollama backend
    (or a pool of Ollama backends).

    The library reads these attributes off the client:
      - `model`: the model name string
      - `provider`: provider tag (we always set "ollama")
      - `streaming`: bool; we always set False
      - `last_call_diagnostics`: dict | None populated after each call

    Constructor knobs (all optional except pool and model):
      - timeout_s: default per-call HTTP timeout; overridable per call
      - temperature, max_tokens, think: default generation params
      - schema_transform: callback applied to the JSON Schema dict before it
        becomes `format=<schema>`. docling-graph wires in `sanitize_schema_for_llm`;
        api-side leaves it None.
      - structured_output_threshold_chars: when the (post-transform) schema
        serializes longer than this, fall back to `format="json"` instead of
        `format=<schema>`. Mirrors the threshold gate that
        `_patched_build_request` had — large schemas degrade Ollama's
        constrained decoder.
      - force_json_mode: when True, always send `format="json"` (never the
        full schema), even for structured_output calls. Mirrors the
        DOCLING_GRAPH_FORCE_JSON_MODE behavior.
      - default_extra_params: dict of extra body fields to merge into every
        request (top_p, top_k, frequency_penalty, presence_penalty, seed,
        stop). Per-call kwargs override.
      - request_semaphore: optional process-wide limiter for chat requests.
        This lets callers set high per-pass worker counts while still capping
        total in-flight Ollama generations to the fixed backend slot budget.
    """

    provider: str = "ollama"
    streaming: bool = False

    def __init__(
        self,
        pool: OllamaPool,
        model: str,
        *,
        timeout_s: float = 7200.0,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        think: str | bool | None = None,
        schema_transform: Optional[Callable[[dict], dict]] = None,
        structured_output_threshold_chars: int | None = None,
        force_json_mode: bool = False,
        default_extra_params: dict[str, Any] | None = None,
        truncation_retry_max_tokens: int | None = None,
        client_error_cls: type[Exception] | None = None,
        parse_json_fn: Callable[[str], Any] | None = None,
        request_semaphore: threading.BoundedSemaphore | None = None,
        request_semaphore_capacity: int | None = None,
        legacy_strip_marker_start: str = "\n\n=== TARGET SCHEMA ===\n",
        legacy_strip_marker_end: str = "=== END SCHEMA ===\n",
    ) -> None:
        self.pool = pool
        self.model = model
        self.model_id = model
        self._default_timeout = timeout_s
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._default_think = think
        self._schema_transform = schema_transform
        self._threshold = structured_output_threshold_chars
        self._force_json_mode = force_json_mode
        self._default_extra: dict[str, Any] = dict(default_extra_params or {})
        self._truncation_retry_max_tokens = truncation_retry_max_tokens
        # Optional ClientError class — when set, parse / HTTP / empty-content
        # failures are wrapped as instances of this class so the upstream
        # LlmBackend's structured-output fallback path triggers correctly.
        # docling-graph wires in `docling_graph.exceptions.ClientError`;
        # app-side leaves it None (raises raw exceptions).
        self._client_error_cls = client_error_cls
        # Optional loose JSON parser. When set, used for get_json_response()
        # parsing; lets docling-graph callers handle fenced/prose-wrapped JSON
        # without a hard json.loads failure. Falls back to json.loads.
        self._parse_json = parse_json_fn or json.loads
        self._request_semaphore = request_semaphore
        self._request_semaphore_capacity = request_semaphore_capacity
        # Schema-embedding markers used by upstream's legacy prompt-builder.
        # When force_json_mode=True AND structured_output=False, the client
        # strips this block from prompt['user'] before sending — replaces the
        # _patched_get_json_response monkey-patch.
        self._legacy_marker_start = legacy_strip_marker_start
        self._legacy_marker_end = legacy_strip_marker_end
        self.last_call_diagnostics: dict | None = None
        # Retained for deterministic close() behavior and any future
        # non-streaming chat path. Streaming chat calls use a short-lived
        # per-request client so a stalled reader can be abandoned without
        # closing transport state underneath other concurrent calls.
        self._http = _build_keepalive_http_client(timeout_s)

    def with_runtime_defaults(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        truncation_retry_max_tokens: int | None = None,
        model: str | None = None,
        think: bool | str | None = None,
    ) -> "OllamaChatClient":
        """Return a client sharing this pool with per-request generation knobs.

        docling-graph ignores ``llm_overrides`` when a concrete ``llm_client``
        is injected. The service uses this to keep the process-wide pool while
        still honoring notebook/API overrides such as temperature or model.
        """
        return OllamaChatClient(
            pool=self.pool,
            model=model if model is not None else self.model,
            timeout_s=self._default_timeout,
            temperature=(
                temperature
                if temperature is not None
                else self._default_temperature
            ),
            max_tokens=(
                max_tokens
                if max_tokens is not None
                else self._default_max_tokens
            ),
            think=think if think is not None else self._default_think,
            schema_transform=self._schema_transform,
            structured_output_threshold_chars=self._threshold,
            force_json_mode=self._force_json_mode,
            default_extra_params=self._default_extra,
            truncation_retry_max_tokens=(
                truncation_retry_max_tokens
                if truncation_retry_max_tokens is not None
                else self._truncation_retry_max_tokens
            ),
            client_error_cls=self._client_error_cls,
            parse_json_fn=self._parse_json,
            request_semaphore=self._request_semaphore,
            request_semaphore_capacity=self._request_semaphore_capacity,
            legacy_strip_marker_start=self._legacy_marker_start,
            legacy_strip_marker_end=self._legacy_marker_end,
        )

    def close(self) -> None:
        """Close the underlying httpx.Client. Safe to call multiple times.

        Prefer explicit close() (or `with contextlib.closing(client):`) over
        relying on __del__ — interpreter shutdown order is undefined and
        connections may leak silently if __del__ doesn't run.
        """
        self._http.close()

    def __del__(self) -> None:
        try:
            self._http.close()
            logger.debug("OllamaChatClient.__del__ closed http client")
        except Exception:
            # __del__ runs at interpreter shutdown; swallowing here avoids
            # noisy tracebacks. Use close() for deterministic cleanup.
            pass

    # ----- LLMClientProtocol surface -----

    def get_json_response(
        self,
        prompt: str | Mapping[str, str],
        schema_json: str,
        structured_output: bool = True,
        response_top_level: Literal["object", "array"] = "object",
        response_schema_name: str = "extraction_result",
    ) -> dict | list:
        # Apply the legacy schema-strip transform when this is a non-structured
        # retry under force_json_mode. Replaces the _patched_get_json_response
        # monkey-patch so behavior is portable across docling-graph versions.
        prompt = self._maybe_strip_legacy_schema(prompt, structured_output)
        messages = self._messages_from_prompt(prompt)
        body = self._build_chat_body(
            messages=messages,
            schema_json=schema_json,
            structured_output=structured_output,
        )
        # Strict content semantics for extraction path: empty content is an
        # error even if reasoning_content is present (chain-of-thought must
        # not leak into structured output).
        content = self._post_chat_with_retry(body, require_content=True)
        try:
            parsed = self._parse_json(content)
        except (json.JSONDecodeError, ValueError) as exc:
            diag = {
                "raw_response": content,
                "json_decode_error": str(exc),
                "model": self.model,
                "provider": self.provider,
                "structured_attempted": structured_output,
                "structured_failed": structured_output,
                "fallback_used": False,
                "fallback_error_class": type(exc).__name__,
            }
            self.last_call_diagnostics = diag
            if self._client_error_cls is not None:
                raise self._client_error_cls(
                    f"Failed to parse JSON response: {exc}",
                    details=diag,
                ) from exc
            raise
        # parse_json_fn (e.g. parse_llm_json_loose) returns None on failure
        # rather than raising. Treat that as a structured-output failure so
        # LlmBackend's sparse-result fallback path triggers.
        if parsed is None:
            diag = {
                "raw_response": content,
                "model": self.model,
                "provider": self.provider,
                "structured_attempted": structured_output,
                "structured_failed": structured_output,
                "fallback_used": False,
                "error": "parse_json_fn returned None",
            }
            self.last_call_diagnostics = diag
            if self._client_error_cls is not None:
                raise self._client_error_cls(
                    "JSON parser returned None (parse failed)", details=diag,
                )
            raise ValueError("JSON parser returned None (parse failed)")
        return parsed

    def _maybe_strip_legacy_schema(
        self, prompt: str | Mapping[str, str], structured_output: bool,
    ) -> str | Mapping[str, str]:
        """When force_json_mode is on AND this is a non-structured retry
        (i.e., the legacy fallback path), strip the schema-embedding tail
        from prompt['user']. Replaces the _get_json_response patch.
        """
        if structured_output or not self._force_json_mode:
            return prompt
        if not isinstance(prompt, Mapping):
            return prompt
        user = prompt.get("user")
        if not isinstance(user, str):
            return prompt
        idx = user.find(self._legacy_marker_start)
        if idx == -1:
            return prompt
        end = user.find(self._legacy_marker_end, idx)
        if end != -1:
            tail = user[end + len(self._legacy_marker_end):]
            stripped = user[:idx].rstrip() + "\n\n" + tail.lstrip()
        else:
            stripped = user[:idx].rstrip()
        # TODO #88: opt-in fail-fast — refuse the legacy retry when the
        # operator has disabled it via OLLAMA_LEGACY_RETRY_ENABLED=false.
        # Forces the caller (LlmBackend) to surface the failure instead of
        # producing schema-unvalidated output.
        if not _LEGACY_RETRY_ENABLED:
            msg = (
                "OLLAMA_LEGACY_RETRY_ENABLED=false: refusing to strip "
                "structured-output schema and re-prompt without validation"
            )
            logger.error(msg)
            if self._client_error_cls is not None:
                raise self._client_error_cls(
                    msg,
                    details={
                        "schema_validated": False,
                        "legacy_retry_blocked": True,
                        "model": self.model,
                        "provider": self.provider,
                    },
                )
            raise RuntimeError(msg)
        # TODO #88: severity bumped from INFO → WARNING. Schema strip is a
        # recall regression (no Pydantic validation) and deserves the same
        # log tier as OllamaPool failures so it's easy to grep.
        logger.warning(
            "OllamaChatClient: stripped %d-char schema embedding from "
            "legacy retry prompt (schema_validated=False)",
            len(user) - len(stripped),
        )
        return {**prompt, "user": stripped}

    def get_json_response_stream(
        self,
        prompt: str | Mapping[str, str],
        schema_json: str,
        structured_output: bool = True,
        response_top_level: Literal["object", "array"] = "object",
        response_schema_name: str = "extraction_result",
    ) -> Iterator[dict | list]:
        yield self.get_json_response(
            prompt, schema_json, structured_output,
            response_top_level, response_schema_name,
        )

    # ----- Plain chat helper for app-side callers -----

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: str | bool | None = None,
        timeout_s: float | None = None,
        force_json: bool = False,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat-completion call; return assistant content (stripped).

        Per-call overrides for model, temperature, max_tokens, think, timeout.
        `force_json` sets format="json"; `extra_params` merges into the body
        (e.g. {"seed": 42, "top_p": 0.9}).
        """
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self._default_temperature
            ),
        }
        eff_max = max_tokens if max_tokens is not None else self._default_max_tokens
        if eff_max is not None:
            body["max_tokens"] = eff_max
        eff_think = think if think is not None else self._default_think
        eff_think = self._coerce_think(eff_think, body["model"])
        if eff_think is not None:
            body["think"] = eff_think
        if force_json:
            body["format"] = _FORMAT_JSON
        # Merge default extras then per-call extras (per-call wins).
        for k, v in self._default_extra.items():
            if v is not None and k not in body:
                body[k] = v
        if extra_params:
            for k, v in extra_params.items():
                if v is not None:
                    body[k] = v
        return self._post_chat_with_retry(
            body, timeout_s=timeout_s,
        )

    # ----- internals -----

    @staticmethod
    def _messages_from_prompt(prompt: str | Mapping[str, str]) -> list[dict]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        msgs: list[dict] = []
        sys_msg = prompt.get("system")
        if sys_msg:
            msgs.append({"role": "system", "content": sys_msg})
        user_msg = prompt.get("user", "")
        msgs.append({"role": "user", "content": user_msg})
        return msgs

    @staticmethod
    def _coerce_think(value: str | bool | None, model: str) -> str | bool | None:
        """gpt-oss accepts low/medium/high; other Ollama models only true/false.
        Mirror the gate from _patched_build_request."""
        if value is None or isinstance(value, bool):
            return value
        v = str(value).strip().lower()
        if v in {"true", "on", "enabled"}:
            return True
        if v in {"false", "off", "disabled"}:
            return False
        if v in {"low", "medium", "high"}:
            return v if "gpt-oss" in (model or "").lower() else None
        return None

    def _build_chat_body(
        self,
        *,
        messages: list[dict],
        schema_json: str,
        structured_output: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._default_temperature,
        }
        if self._default_max_tokens is not None:
            body["max_tokens"] = self._default_max_tokens
        eff_think = self._coerce_think(self._default_think, self.model)
        if eff_think is not None:
            body["think"] = eff_think
        # Merge default extras (top_p, top_k, seed, stop, etc.).
        for k, v in self._default_extra.items():
            if v is not None and k not in body:
                body[k] = v

        # Decide format=
        if not structured_output or not schema_json:
            body["format"] = _FORMAT_JSON
            return body
        if self._force_json_mode:
            body["format"] = _FORMAT_JSON
            return body
        try:
            schema_dict = json.loads(schema_json)
        except json.JSONDecodeError:
            body["format"] = _FORMAT_JSON
            return body
        if self._schema_transform:
            schema_dict = self._schema_transform(schema_dict)
        schema_serialized = json.dumps(schema_dict)
        if (
            self._threshold is not None
            and len(schema_serialized) > self._threshold
        ):
            body["format"] = _FORMAT_JSON
        else:
            body["format"] = schema_dict
        return body

    def _stream_chat_with_watchdog(
        self,
        url: str,
        body: dict[str, Any],
        *,
        no_progress_seconds: float = _STREAM_NO_PROGRESS_SECONDS,
        max_wall_seconds: float = _STREAM_MAX_WALL_SECONDS,
    ) -> dict[str, Any]:
        """Stream a /v1/chat/completions response with a per-recv watchdog.

        Switches the request to ``stream=True`` and reads SSE bytes in a
        daemon reader thread. This method monitors raw-byte progress itself,
        so the caller is released after ``no_progress_seconds`` even if the
        underlying httpcore read parks below Python and never raises its own
        timeout. The raised ``ReadTimeout`` engages the caller's retry path
        on a different pool URL.

        Returns a synthesized non-streaming-shaped dict so the existing
        choices/message/content parser in _post_chat_with_retry doesn't
        change.
        """
        streamed = dict(body)
        streamed["stream"] = True
        watchdog = httpx.Timeout(
            connect=10.0,
            # Keep httpx's native read timeout as a near-term fallback, but
            # make the explicit watchdog below the first line of defense.
            read=no_progress_seconds + 1.0,
            write=60.0,
            pool=30.0,
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: str | None = None
        stream_request: httpx.Request | None = None

        def _read_timeout(cause: BaseException | None = None) -> httpx.ReadTimeout:
            msg = (
                "no SSE bytes from Ollama for "
                f"{no_progress_seconds:.1f}s while streaming chat completion"
            )
            exc = httpx.ReadTimeout(msg, request=stream_request)
            if cause is not None:
                raise exc from cause
            return exc

        def _consume_sse_line(line: str) -> bool:
            nonlocal finish_reason
            if not line:
                return False
            data = line[6:].strip() if line.startswith("data: ") else line.strip()
            if data == "[DONE]":
                return True
            try:
                chunk = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                return False
            choices = chunk.get("choices") or []
            if not choices:
                return False
            delta = choices[0].get("delta") or {}
            ct = delta.get("content")
            if ct:
                content_parts.append(ct)
            rc = delta.get("reasoning_content") or delta.get("thinking")
            if rc:
                reasoning_parts.append(rc)
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr
            return False

        stream_http = _build_keepalive_http_client(no_progress_seconds + 1.0)
        stream_done = threading.Event()
        progress_lock = threading.Lock()
        start_time = time.monotonic()
        last_byte_time = time.monotonic()
        read_error: list[Exception] = []

        def _mark_progress() -> None:
            nonlocal last_byte_time
            with progress_lock:
                last_byte_time = time.monotonic()

        def _read_stream() -> None:
            nonlocal stream_request
            pending = b""
            saw_done = False
            try:
                with stream_http.stream(
                    "POST",
                    f"{url}/v1/chat/completions",
                    json=streamed,
                    timeout=watchdog,
                ) as resp:
                    stream_request = resp.request
                    if resp.is_error:
                        err_body = resp.read().decode("utf-8", errors="replace")
                        # Construct a fake non-streaming response shape on the resp
                        # object so HTTPStatusError carries the body for diagnostics.
                        resp._content = err_body.encode("utf-8")  # noqa: SLF001
                        raise httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}",
                            request=resp.request,
                            response=resp,
                        )
                    for chunk in resp.iter_raw():
                        if not chunk:
                            continue
                        _mark_progress()
                        pending += chunk
                        while b"\n" in pending:
                            line_bytes, pending = pending.split(b"\n", 1)
                            line = line_bytes.rstrip(b"\r").decode(
                                "utf-8", errors="replace",
                            )
                            if _consume_sse_line(line):
                                saw_done = True
                                break
                        if saw_done:
                            break
                    if pending and not saw_done:
                        line = pending.decode("utf-8", errors="replace")
                        _consume_sse_line(line)
            except Exception as exc:
                # Some HTTP stacks close/reset immediately after the terminal
                # chunk. If we already saw a finish_reason, we have a complete
                # assistant message; do not turn post-finish socket noise into
                # a failed extraction batch.
                if finish_reason is None:
                    read_error.append(exc)
            finally:
                stream_done.set()
                stream_http.close()

        reader = threading.Thread(
            target=_read_stream,
            name="ollama-stream-reader",
            daemon=True,
        )
        reader.start()
        poll_s = min(1.0, max(0.05, no_progress_seconds / 10.0))
        while not stream_done.wait(poll_s):
            with progress_lock:
                elapsed = time.monotonic() - last_byte_time
            wall_elapsed = time.monotonic() - start_time
            if wall_elapsed >= max_wall_seconds:
                logger.error(
                    "OLLAMA_STREAM_WALL_TIMEOUT url=%s model=%s "
                    "elapsed_s=%.1f threshold_s=%.1f — closing stream "
                    "so structured-output fallback can engage.",
                    url, body["model"], wall_elapsed, max_wall_seconds,
                )
                stream_http.close()
                raise httpx.ReadTimeout(
                    "streamed chat completion exceeded wall-clock limit "
                    f"of {max_wall_seconds:.1f}s",
                    request=stream_request,
                )
            if elapsed >= no_progress_seconds:
                logger.error(
                    "OLLAMA_STREAM_NO_PROGRESS url=%s model=%s "
                    "elapsed_s=%.1f threshold_s=%.1f — closing stream "
                    "so retry path can fail over.",
                    url, body["model"], elapsed, no_progress_seconds,
                )
                stream_http.close()
                raise _read_timeout()
        if read_error:
            raise read_error[0]
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts),
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        return {
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
        }

    def _stream_with_truncation_retry(
        self,
        url: str,
        body: dict[str, Any],
        *,
        require_content: bool,
        t0: float,
    ) -> tuple[dict[str, Any], bool]:
        """Run a streaming chat call; if the model hit ``max_tokens`` (Ollama
        ``finish_reason="length"``), log a TRUNCATION warning and one-shot
        retry on the **same URL** with the cap doubled. Returns
        ``(payload, truncation_persisted)`` — payload has the same shape
        ``_stream_chat_with_watchdog`` produces; ``truncation_persisted`` is
        True only when the bumped retry also got cut off, signaling that the
        operator should raise ``DOCLING_GRAPH_LLM_MAX_TOKENS`` for this
        model+prompt class.

        Why one-shot only: if a chunk genuinely needs >2× the configured cap,
        an unbounded retry loop is worse than a single ERROR log surfacing
        the case to an operator. The ``finish_reason="length"`` ERROR is the
        signal to act on.

        Why same URL: truncation is a model+prompt deterministic property —
        retrying on a different URL won't help (same model). URL-failover
        is for transport-level errors and lives one layer up in
        ``_post_chat_with_retry``.
        """
        payload = self._stream_chat_with_watchdog(url, body)
        choices = payload.get("choices") or []
        if not choices:
            return payload, False
        finish_reason = choices[0].get("finish_reason")
        if finish_reason != "length":
            return payload, False

        # First call truncated. Always log; retry only when the caller
        # actually needs the content (extraction path) and we haven't
        # already bumped on this call.
        message_first = choices[0].get("message", {}) or {}
        content_first = (message_first.get("content") or "").strip()
        current_cap = body.get("max_tokens") or 4096
        logger.warning(
            "OllamaChatClient: TRUNCATION_AT_NUM_PREDICT url=%s model=%s "
            "elapsed=%.2fs len(content)=%d max_tokens=%d — output cap hit; "
            "response may be incomplete.",
            url, body["model"], time.time() - t0, len(content_first), current_cap,
        )
        if not require_content:
            # Caller is non-strict (app-side chat()). Pass the truncated
            # content through without an extra round-trip.
            logger.info(
                "OllamaChatClient: TRUNCATION_OUTCOME url=%s model=%s "
                "first_content_len=%d first_max_tokens=%d retry_ran=False "
                "retry_content_len=0 retry_finish_reason=n/a "
                "final=truncated_no_retry_caller_non_strict",
                url, body["model"], len(content_first), current_cap,
            )
            return payload, False

        # Always-bump retry on require_content=True. The earlier spin-skip
        # branch (skip retry when content=0) was reverted on 2026-05-01
        # after a regression run showed it cost both recall (15 vs 30
        # nodes) and wall-time (88 min vs 67 min). Some content=0 first
        # calls were recoverable via the bumped retry — the model needed
        # the extra token budget to produce a clean empty response or to
        # finish a structured terminator instead of running out mid-state.
        # Skipping that retry caused permanent batch losses (ClientError)
        # that the bumped retry would otherwise have prevented.
        max_retry_cap = self._truncation_retry_max_tokens
        next_cap = current_cap * 2
        last_payload = payload
        last_content_len = len(content_first)
        while True:
            if max_retry_cap is not None:
                next_cap = min(next_cap, max_retry_cap)
            if next_cap <= current_cap:
                logger.error(
                    "OllamaChatClient: TRUNCATION_RETRY_CAP_EXHAUSTED url=%s "
                    "model=%s first_max_tokens=%d retry_max_tokens=%d "
                    "last_content_len=%d — returning known-partial content.",
                    url, body["model"], body.get("max_tokens") or 4096,
                    current_cap, last_content_len,
                )
                return last_payload, True

            bumped_body = dict(body)
            bumped_body["max_tokens"] = next_cap
            logger.info(
                "OllamaChatClient: retrying with max_tokens=%d to recover "
                "potentially-truncated content (url=%s model=%s).",
                bumped_body["max_tokens"], url, body["model"],
            )
            retry_payload = self._stream_chat_with_watchdog(url, bumped_body)
            retry_choices = retry_payload.get("choices") or []
            if not retry_choices:
                logger.info(
                    "OllamaChatClient: TRUNCATION_OUTCOME url=%s model=%s "
                    "first_content_len=%d first_max_tokens=%d retry_ran=True "
                    "retry_content_len=0 retry_finish_reason=n/a "
                    "retry_max_tokens=%d final=retry_no_choices",
                    url, body["model"], len(content_first), current_cap,
                    bumped_body["max_tokens"],
                )
                return payload, True

            retry_finish = retry_choices[0].get("finish_reason")
            retry_msg = retry_choices[0].get("message", {}) or {}
            retry_content = (retry_msg.get("content") or "").strip()
            if retry_finish != "length":
                outcome_label = (
                    "recovered_content" if retry_content else "recovered_clean_empty"
                )
                logger.info(
                    "OllamaChatClient: TRUNCATION_OUTCOME url=%s model=%s "
                    "first_content_len=%d first_max_tokens=%d retry_ran=True "
                    "retry_content_len=%d retry_finish_reason=%s "
                    "retry_max_tokens=%d final=%s",
                    url, body["model"], len(content_first), current_cap,
                    len(retry_content), retry_finish, bumped_body["max_tokens"],
                    outcome_label,
                )
                return retry_payload, False

            last_payload = retry_payload
            last_content_len = len(retry_content)
            if max_retry_cap is None or bumped_body["max_tokens"] >= max_retry_cap:
                logger.error(
                    "OllamaChatClient: TRUNCATION_PERSISTS_AFTER_RETRY url=%s "
                    "model=%s len(content)=%d max_tokens=%d — the bumped retry "
                    "also truncated. Consider raising "
                    "DOCLING_GRAPH_LLM_TRUNCATION_RETRY_MAX_TOKENS for this "
                    "model+prompt class, or chunking smaller. Returning the "
                    "bumped-retry content (may still be partial).",
                    url, body["model"], len(retry_content), bumped_body["max_tokens"],
                )
                logger.info(
                    "OllamaChatClient: TRUNCATION_OUTCOME url=%s model=%s "
                    "first_content_len=%d first_max_tokens=%d retry_ran=True "
                    "retry_content_len=%d retry_finish_reason=length "
                    "retry_max_tokens=%d final=persisted_truncation",
                    url, body["model"], len(content_first), current_cap,
                    len(retry_content), bumped_body["max_tokens"],
                )
                return retry_payload, True

            logger.warning(
                "OllamaChatClient: TRUNCATION_PERSISTS_RETRYING url=%s "
                "model=%s len(content)=%d max_tokens=%d next_max_tokens=%d",
                url, body["model"], len(retry_content),
                bumped_body["max_tokens"], min(bumped_body["max_tokens"] * 2, max_retry_cap),
            )
            current_cap = bumped_body["max_tokens"]
            next_cap = current_cap * 2

    def _post_chat_with_retry(
        self,
        body: dict[str, Any],
        *,
        timeout_s: float | None = None,
        require_content: bool = False,
    ) -> str:
        """Pick a URL via the pool, POST; on connect/timeout error, retry once
        on a different URL. Always release the inflight slot.

        `require_content=True` (extraction path) raises ClientError when the
        assistant message has only `reasoning_content` / `thinking` and no
        `content` — prevents chain-of-thought from polluting structured output.
        `require_content=False` (app-side `chat()`) falls back to
        `reasoning_content` if `content` is empty (preserves current behavior
        for community reports + global synthesis on thinking models).

        Uses streaming (_stream_chat_with_watchdog) so the per-recv
        no-progress watchdog (_STREAM_NO_PROGRESS_SECONDS) catches Ollama's
        silent-request-loss class (peer alive at TCP, request abandoned at
        app layer).
        """
        excluded: set[str] = set()
        last_exc: Exception | None = None
        # The streaming watchdog enforces per-recv timeouts, so the caller's
        # timeout_s is no longer plumbed into the httpx call. The orchestrator-
        # level BATCH_HARD_TIMEOUT (default 3600s) remains the upper bound on
        # total batch wall-time.
        del timeout_s  # unused after the streaming switch
        slot_wait_started = time.time()
        if self._request_semaphore is not None:
            self._request_semaphore.acquire()
            waited_s = time.time() - slot_wait_started
            if waited_s >= 1.0:
                logger.info(
                    "OllamaChatClient: waited %.2fs for LLM slot "
                    "(capacity=%s model=%s)",
                    waited_s, self._request_semaphore_capacity, self.model,
                )
        try:
            for attempt in range(2):
                url = self.pool.acquire(exclude=excluded)
                t0 = time.time()
                try:
                    payload, _truncation_persisted = self._stream_with_truncation_retry(
                        url, body, require_content=require_content, t0=t0,
                    )
                    choices = payload.get("choices") or []
                    if not choices:
                        diag = {
                            "url": url, "model": self.model, "provider": self.provider,
                            "elapsed_s": time.time() - t0,
                            "structured_failed": True, "fallback_used": False,
                            "error": "no choices in response",
                        }
                        self.last_call_diagnostics = diag
                        if self._client_error_cls is not None:
                            raise self._client_error_cls(
                                "LLM returned no choices", details=diag,
                            )
                        raise RuntimeError("LLM returned no choices")
                    message = choices[0].get("message", {}) or {}
                    content = (message.get("content") or "").strip()
                    reasoning = (
                        message.get("reasoning_content")
                        or message.get("thinking")
                        or ""
                    )
                    self.last_call_diagnostics = {
                        "url": url,
                        "model": self.model,
                        "provider": self.provider,
                        "elapsed_s": time.time() - t0,
                        "raw_response": content,
                        "has_reasoning_content": bool(reasoning),
                        "structured_attempted": True,
                        "structured_failed": False,
                        "fallback_used": False,
                    }
                    logger.info(
                        "OllamaChatClient: ok url=%s model=%s elapsed=%.2fs len(content)=%d",
                        url, body["model"], time.time() - t0, len(content),
                    )
                    if content:
                        return content
                    if require_content:
                        diag = {
                            **self.last_call_diagnostics,
                            "structured_failed": True,
                            "error": "empty content; only reasoning available",
                            "reasoning_preview": str(reasoning)[:500],
                        }
                        self.last_call_diagnostics = diag
                        if self._client_error_cls is not None:
                            raise self._client_error_cls(
                                "LLM returned empty content",
                                details=diag,
                            )
                        raise RuntimeError("LLM returned empty content")
                    # Non-strict: app-side caller; fall back to reasoning.
                    return str(reasoning).strip()
                except (
                    httpx.TimeoutException,    # ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout
                    httpx.NetworkError,         # ConnectError, ReadError, WriteError, CloseError
                    httpx.RemoteProtocolError,  # truncated response, mid-stream disconnect
                ) as exc:
                    last_exc = exc
                    excluded.add(url)
                    # ERROR not WARNING — silent socket drops (the
                    # OLLAMA_POOL_TCP_DROP class) need to surface loudly so an
                    # operator monitoring logs sees one entry per drop, not just
                    # buried INFO routine. The retry-on-different-URL still happens;
                    # this is purely visibility.
                    logger.error(
                        "OLLAMA_POOL_CONN_FAILURE exc_type=%s url=%s model=%s "
                        "elapsed_s=%.1f attempt=%d/2 exc_msg=%s — retrying on "
                        "remaining pool URLs.",
                        type(exc).__name__, url, self.model,
                        time.time() - t0, attempt + 1, exc,
                    )
                    # Guards acquire() from raising on attempt 2 — must run before
                    # next iteration. Once every URL is excluded, retrying would
                    # call pool.acquire(exclude=all_urls), which raises RuntimeError.
                    if len(excluded) >= len(self.pool.urls):
                        break
                except httpx.HTTPStatusError as exc:
                    # 4xx/5xx — wrap and re-raise; don't retry (the server
                    # responded, the issue is the request body or model state).
                    diag = {
                        "url": url, "model": self.model, "provider": self.provider,
                        "status_code": exc.response.status_code,
                        "structured_failed": True, "fallback_used": False,
                        "error": str(exc),
                        "raw_response": exc.response.text[:1000],
                    }
                    self.last_call_diagnostics = diag
                    if self._client_error_cls is not None:
                        raise self._client_error_cls(
                            f"HTTP {exc.response.status_code} from Ollama",
                            details=diag,
                        ) from exc
                    raise
                finally:
                    self.pool.release(url)
        finally:
            if self._request_semaphore is not None:
                self._request_semaphore.release()
        assert last_exc is not None
        # All pool URLs failed transport-wise. Loud surface BEFORE raising so
        # the failure is visible even if the caller swallows the exception
        # downstream.
        logger.error(
            "OLLAMA_POOL_ALL_FAILED model=%s tried_urls=%s last_exc_type=%s "
            "last_exc_msg=%s — every pool URL dropped or timed out.",
            self.model, sorted(excluded),
            type(last_exc).__name__, last_exc,
        )
        if self._client_error_cls is not None:
            raise self._client_error_cls(
                f"All pool URLs failed: {type(last_exc).__name__}: {last_exc}",
                details={
                    "model": self.model, "provider": self.provider,
                    "tried_urls": sorted(excluded),
                    "fallback_error_class": type(last_exc).__name__,
                },
            ) from last_exc
        raise last_exc


class OllamaEmbeddingClient:
    """Pool-backed embedding client. Calls /v1/embeddings on the picked URL
    and returns the embedding vectors (sorted by input index)."""

    def __init__(
        self,
        pool: OllamaPool,
        model: str,
        *,
        timeout_s: float = 120.0,
    ) -> None:
        self.pool = pool
        self.model = model
        # Symmetric with OllamaChatClient — populated after each successful
        # embed() POST with {url, elapsed_s, model, batch_size}. Useful for
        # routing diagnostics and slow-host detection.
        self.last_call_diagnostics: dict | None = None
        # Same TCP-keepalive + read-timeout-split treatment as the chat client.
        self._http = _build_keepalive_http_client(timeout_s)

    def close(self) -> None:
        """Close the underlying httpx.Client. Safe to call multiple times."""
        self._http.close()

    def __del__(self) -> None:
        try:
            self._http.close()
            logger.debug("OllamaEmbeddingClient.__del__ closed http client")
        except Exception:
            # __del__ runs at interpreter shutdown; swallowing here avoids
            # noisy tracebacks. Use close() for deterministic cleanup.
            pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts` in one call. Returns vectors in the same order as the
        input. Caller is responsible for batching if texts is too large for
        a single request."""
        excluded: set[str] = set()
        last_exc: Exception | None = None
        for _ in range(2):
            url = self.pool.acquire(exclude=excluded)
            t0 = time.time()
            try:
                resp = self._http.post(
                    f"{url}/v1/embeddings",
                    json={"model": self.model, "input": texts},
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
                items = sorted(data, key=lambda x: x.get("index", 0))
                self.last_call_diagnostics = {
                    "url": url,
                    "elapsed_s": time.time() - t0,
                    "model": self.model,
                    "batch_size": len(texts),
                }
                logger.info(
                    "OllamaEmbeddingClient: ok url=%s model=%s batch_size=%d elapsed=%.2fs",
                    url, self.model, len(texts), time.time() - t0,
                )
                return [item["embedding"] for item in items]
            except (
                httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError,
            ) as exc:
                last_exc = exc
                excluded.add(url)
                # ERROR not silent — embedding-side connection drops also
                # need loud surface so an operator sees per-drop entries.
                logger.error(
                    "OLLAMA_EMBED_CONN_FAILURE exc_type=%s url=%s model=%s "
                    "batch=%d exc_msg=%s — retrying on remaining pool URLs.",
                    type(exc).__name__, url, self.model, len(texts), exc,
                )
                if len(excluded) >= len(self.pool.urls):
                    break
            finally:
                self.pool.release(url)
        assert last_exc is not None
        logger.error(
            "OLLAMA_EMBED_ALL_FAILED model=%s tried_urls=%s last_exc_type=%s "
            "last_exc_msg=%s — every embedding pool URL dropped or timed out.",
            self.model, sorted(excluded),
            type(last_exc).__name__, last_exc,
        )
        raise last_exc
