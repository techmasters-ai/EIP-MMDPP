# OllamaPool Client Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Revision history:**
- **v10 (2026-04-29, post-ninth-review — last polish before implementation):** Final cleanups, all minor: P1 stale-step-reference (`Step 9` → `Step 7`); Task 6.3 Step 7's patch-target instruction reworded to "patch the import path actually looked up by the code AFTER migration" with explicit guidance on module-scope vs in-function imports; Task 6.5b says **replace** the existing DEBUG success log (not "promote") to avoid leaving both lines emitting; Task 6.6 Step 1 env snippet now includes the model vars (`DOC_ANALYSIS_LLM_MODEL`, `DOCLING_GRAPH_LLM_MODEL`, etc.) so the validation recipe is self-contained. Reviewer rated v9 implementable; v10 is final polish.
- **v9 (2026-04-29, post-eighth-review — Chunk-6-only fixes):** **High:** Resolved contradiction between Task 6.5b (which modifies `app/services/ollama_pool_client.py` + mirror) and the "Files Chunk 6 must NOT touch" section. Reworded that section to "must NOT touch (routing-behavior-wise)" and explicitly carve out Task 6.5b's logging-only edits as allowed. **Medium:** Task 6.3 Step 1 still contained the old narrow grep (`from app.services.ollama_clients import (get_llm_client|...)`) which misses module-qualified calls and `get_docling_llm_client`. Replaced with the comprehensive `rg` from P1. **Medium:** Task 6.5b's success log used `self.model` for the chat client; switched to `body.get("model", self.model)` so per-call model overrides via `chat(model=...)` are visible in the log (Gates 6.C/6.D's filter rely on this). Embedding client keeps `self.model` (no per-call override exists today). **Medium:** Gate 6.C grep was ambiguous when functions share a model; added a concrete `grep "model=gpt-oss:120b"` filter and noted the test config should set `DOC_ANALYSIS_LLM_MODEL` distinct from `DOCLING_GRAPH_LLM_MODEL` so the disambiguation works cleanly. Also documented the time-window-narrowing fallback for same-model configs. **Low:** Task 6.4 Step 5 expected output reworded — `routing_metrics` is a dict keyed by URLs with zero counts at startup, not literally `{}`. **Low:** Risk-table mitigation reworded to reflect Step 1's now-comprehensive audit.
- **v8 (2026-04-29, post-seventh-review — Chunk-6-only fixes):** Addressed reviewer findings on the Chunk 6 fleshed-out plan. **High:** Gates 6.C / 6.D were not verifiable — current client only logs URLs at WARNING (failure path) and `OllamaEmbeddingClient.embed()` had no success URL log at all. Added new Task 6.5b that promotes the success log to INFO with `url=...` field on both chat and embedding clients (with parity tests + mirror-drift sync). Updated Gates 6.C / 6.D with concrete grep commands against the new log lines. **Medium:** P1 audit grep was too narrow — couldn't catch module-qualified calls (`ollama_clients.get_llm_client(...)`) or `get_docling_llm_client`. Replaced with a comprehensive `rg` over both production and test trees and a separate test-patch sweep section. **Medium:** Task 6.4 Step 5 overstated what the host-venv test proves — that test swallows the `app.ollama_clients` ImportError and would NOT catch a renamed symbol inside the container. Added explicit `docker exec ... python -c "from app.ollama_clients import get_docling_graph_client"` smoke check. **Medium:** Test mock-path sweep was implicit; made it an explicit Step 7 with the comprehensive `rg tests --type py` invocation, and called out two known stale targets (`test_document_analysis.py:14` and `test_arcadedb_community.py:290`). **Low:** Task 6.1 wording said "6 fields" but only added 5 (the 6th is `DOCLING_GRAPH_LLM_BASE_URLS` in Task 6.4) — renamed to "api-side 5 per-function fields" with explicit pointer to Task 6.4. **Low:** Task 6.3 Step 8 said "six commits — one per file" but listed 5 — fixed to 5 migration commits + optional 6th test-sweep commit only if Steps 1-5's per-file commits don't cover all the patches.
- **v7 (2026-04-28, post-sixth-review):** Fixed invalid `docker compose -e` syntax in Gate 5 (the `-e` flag exists on `compose run/exec`, not on `up`) — switched to shell env interpolation `DOCLING_GRAPH_DEBUG_ENDPOINTS=true docker compose up -d ...`. Added `docker-compose.yml` to Task 3.2b commit (the compose env passthrough was being modified without being staged). Reordered Task 3.2b Step 2 to (a) verify the endpoint returns 404 with the flag off (default), then (b) enable the flag and verify a real metrics dict comes back. Replaced the last "validator" reference in the P3 helper output. Initialized `_rr_cursor=-1` so the first tied acquisition picks `urls[0]` (cosmetic — fan-out worked either way). Added `test_default_extra_params_on_get_json_response` to cover the extraction path that actually uses `get_json_response()`.
- **v6 (2026-04-28, post-fifth-review):** Fixed API port in Task 4.1 reingest call (`${API_PORT:-8003}`, not the local `.env` override of 8005). Fixed commit commands in Tasks 3.1 and 3.2 to stage all modified files (mirror `llm_json.py` + the canonical's marker line; `docker-compose.yml` alongside `config_builder.py`). Replaced fragile exact-count assertions with "all tests in file pass" guidance — Task 1.2 / 1.3 / 1.4 / 1.5 expected counts were drifting as new tests landed each review pass. Added the missing `test_chat_per_call_model_override` and `test_chat_force_json_sets_format_json` tests Task 2.2 was assuming. Added focused coverage for the load-bearing chat-body knobs that were unit-test gaps: `schema_transform` applied before `format=<schema>`, `structured_output_threshold_chars` forces `format="json"` for oversized schemas, `force_json_mode=True` overrides structured calls, `default_extra_params` passes through (`top_p`/`seed`/etc.), `think="low"` gpt-oss vs non-gpt-oss gating, malformed `resp.json()` envelope wrapped as `ClientError`. Corrected Chunk 1 file map: docling mirror is created in Chunk 3 (Task 3.1), not Chunk 1. Updated P3 wording away from "validator". Gated `/debug/routing-metrics` behind a `DOCLING_GRAPH_DEBUG_ENDPOINTS` env flag (default off) so the published port doesn't leak backend URLs in production.
- **v5 (2026-04-28, post-fourth-review):** Deleted leftover stale v3 `list[str]` getter block in Task 1.5 (would have called `list(json_string)` → list-of-chars). Added missing `OLLAMA_LLM_BASE_URLS` passthrough to docker-compose.yml docling-graph environment block (Task 3.2 Step 0). Fixed `/debug/routing-metrics` URL in Gate 5: port is `${DOCLING_GRAPH_PORT:-8002}`, not `8005`. Added missing `import json, httpx` at top of test_ollama_pool_client.py (Task 1.1 Step 1). Broadened retry catch from `(ConnectError, ReadTimeout, WriteTimeout)` to `(httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)`. Wrapped `resp.json()` envelope decode in try/except so malformed Ollama responses go through the ClientError path. Replaced `self._urls.index(u)` inside the lock with a precomputed `_url_index` dict. Simplified `build_pipeline_config` to a one-liner `llm_client = get_docling_llm_client()` (the dead `default_extra_params` block was leftover from v3). Reject blank entries inside parsed pools. Updated P3 preflight + Risks table wording away from the validator/`list[str]` story.
- **v4 (2026-04-28, post-third-review):** Verified `field_validator(mode="before")` does NOT intercept blank-string for `list[str]` — pydantic-settings raises `SettingsError` during JSON decode before validators run. **Switched plural URL fields to raw `str` storage**; JSON parsing happens in `get_ollama_*_urls()` helper methods. Validator approach removed. Fixed docling-graph import path (`from app.llm_json import parse_llm_json_loose`, not `app.services.llm_json` — docling-graph container layout is `/app/app/`). Added explicit `None`-check after `parse_json_fn` since `parse_llm_json_loose` returns `None` on failure (not exception); now wraps as `ClientError`. Replaced Gate 5 mechanism (a fresh `docker exec python -c` can't see the running uvicorn process's module state) with a diagnostic FastAPI endpoint (`GET /debug/routing-metrics`). Added a process-cached docling-graph LLM client factory (mirror of `ollama_clients.py`) so concurrent extraction passes share in-flight counters instead of each pass building a fresh pool. Dropped vestigial `client_unused, url_unused` translation params. Refreshed stale test counts in Task 1.2/1.3/1.4/1.5. Removed redundant Task 2.2 (per-call model/force_json overrides already implemented in Task 1.3). Corrected Task 1.6's wording ("the custom parser handles it" — pydantic-settings does NOT treat blank as default).
- **v3 (2026-04-28, post-second-review):** verified `list[str]` blank-string raises `SettingsError` — added `field_validator(mode='before')` to coerce. Verified docling-graph version drift (container 1.5.0 has `_get_json_response`; host venv doesn't) — moved schema-strip logic INTO `OllamaChatClient` so behavior is portable; the existing patch becomes deletable in Chunk 5. Added `client_error_cls` + `parse_json_fn` injection so the client can wrap `JSONDecodeError`/`httpx` failures as `docling_graph.exceptions.ClientError` for fallback compatibility. Split content semantics: `get_json_response()` requires assistant `content` and raises `ClientError` on empty (no chain-of-thought leaks); `chat()` falls back to `reasoning_content` (preserves app-side semantics). Added round-robin tie-breaking to `OllamaPool` so serial workloads fan out. Fixed every Chunk 2 migration to pass per-call `timeout_s=`. Replaced Gate 5 (log-grep was infeasible — no info logs on success) with a `last_call_diagnostics` URL counter or per-instance routing_metrics dict. Resolved Chunk 3 self-contradiction (file map said "delete LiteLLM patches"; text said keep — text wins, file map updated). Pinned `docling-graph` container fallback to `http://ollama:11434` (compose service name), not `localhost:11434`.
- **v2 (2026-04-28, post-first-review):** added `schema_transform`/`force_json_mode`/`structured_output_threshold_chars` knobs, gpt-oss think mapping, gen-param passthrough, per-call timeout. Restructured: Chunk 3 keeps LiteLLM patches as safety net; new Chunk 5 deletes them post-validation. Mirror-drift test uses marker line. Out-of-scope listing for similar non-Ollama httpx call sites.
- **v1 (2026-04-28):** initial draft.

**Goal:** Replace LiteLLM (in docling-graph) and per-site `httpx` Ollama calls (in app/) with a single direct-Ollama pool client that load-balances across a configurable bank of Ollama instances per role (LLM / VLM / embedding).

**Architecture:** A small reusable routing core (`OllamaPool`) tracks in-flight request counts per URL behind a lock with round-robin tie-breaking; client classes layered on top (`OllamaChatClient` implementing docling_graph's `LLMClientProtocol`, `OllamaEmbeddingClient` for `/v1/embeddings`) provide the actual HTTP wire calls. The chat client preserves all the load-bearing logic that `_patched_build_request` does today: optional `schema_transform` hook (so docling-graph can wire in `sanitize_schema_for_llm`), `force_json_mode` flag, schema-size threshold gating (`structured_output_threshold_chars` — large schemas fall back to `format="json"` because constrained decoding degrades on big schemas), gpt-oss think-level pass-through (`low|medium|high`), and full generation-param pass-through (`top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `seed`, `stop`). The chat client also absorbs the legacy schema-strip behavior previously done by the `_patched_get_json_response` monkey-patch: when `force_json_mode=True` and `structured_output=False` (the sparse-result-detection retry path), the client strips the `=== TARGET SCHEMA === ... === END SCHEMA ===` block from `prompt["user"]` before sending. This makes the behavior portable across docling-graph versions and lets Chunk 5 delete the upstream patch entirely. Each role-specific pool is constructed from a JSON-array env var (`OLLAMA_LLM_BASE_URLS`, `OLLAMA_VLM_BASE_URLS`, `OLLAMA_EMBEDDING_BASE_URLS`), with plural-first / singular-fallback / `OLLAMA_BASE_URL`-fallback semantics so existing `.env` files keep working unchanged. Routing is least-in-flight; failure handling is one retry on a different URL for connection/timeout errors. The same client is wired into the docling-graph extraction service via `PipelineConfig(llm_client=...)`, eliminating LiteLLM from the call path entirely. The chat client accepts an injectable `client_error_cls` so `JSONDecodeError`/`httpx`/empty-content failures get wrapped as `docling_graph.exceptions.ClientError`, which is what `LlmBackend` catches to trigger its sparse-result fallback. App-side callers leave `client_error_cls=None` and get raw exceptions.

**Tech Stack:** Python 3.11/3.12, pydantic-settings v2, httpx (sync + async wrappers), docling_graph's `LLMClientProtocol`, FastAPI, Celery.

**Spec:** Architecture agreed in conversation 2026-04-28 (no separate spec doc — see chat history for design rationale and tradeoff analysis).

**Out of scope — non-Ollama httpx call sites you might be tempted to migrate:**
- `app/services/docling_client.py` — POSTs to the Docling conversion service (not Ollama). Leave alone.
- `app/workers/pipeline.py::_call_extract_pass` — POSTs to docling-graph `/extract-pass` (not Ollama). Leave alone.
- `app/services/arcadedb_client.py` — POSTs to ArcadeDB. Leave alone.
- `app/services/embedding.py::embed_images` — runs OpenCLIP locally (no HTTP). Leave alone.
- `app/api/v1/retrieval.py::_apply_reranker` — runs CrossEncoder locally. Leave alone.

---

## Pre-flight checklist

Run these once at the start of the session and before each chunk to confirm baseline:

- [ ] **P0: Confirm baseline test suite status.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline -q 2>&1 | tail -3`
Expected: all green (`N passed, ...`). Capture the count; later chunks must not regress.

- [ ] **P1: Confirm stack is up.**

Run: `docker compose ps --format "table {{.Service}}\t{{.Status}}" | grep -E "api|arcadedb|postgres|worker-graph|docling-graph"`
Expected: all five services `Up (healthy)`. If not, `docker compose up -d` and wait 30 s.

- [ ] **P2: Capture current LiteLLM patch fire counters as baseline.**

Run: `docker logs eip-mmdpp-docling-graph-1 2>&1 | grep -cE "force_json_mode: stripped|LiteLLMClient patched|LlmBackend.*patched"`
Note the value. After Chunk 5 (post-validation) all LiteLLM patches AND the `_get_json_response` patch are deleted; only `NodeIDRegistry patched ...` should appear.

- [ ] **P3: Verify pydantic-settings env-array behavior — including the empty-string trap.**

Run:
```bash
.venv/bin/python -c "
from pydantic_settings import BaseSettings
import os
class S(BaseSettings):
    my_list: list[str] = []
# JSON array — works
os.environ['MY_LIST'] = '[\"a\",\"b\"]'
assert S().my_list == ['a', 'b']
# Empty string — RAISES SettingsError by default. Plural env vars must
# be stored as raw strings and parsed by helper methods (Task 1.5).
os.environ['MY_LIST'] = ''
try:
    S()
    print('UNEXPECTED: blank parsed without error')
except Exception as e:
    assert 'error parsing value' in str(e).lower(), e
    print('OK: blank correctly raises (handled by raw-string parsing in Task 1.5)')
# Unset — default kicks in
os.environ.pop('MY_LIST')
assert S().my_list == []
print('OK')
"
```
Expected: `OK: blank correctly raises ...` then `OK`. The plan stores plural URL fields as raw `str` precisely because a `field_validator(mode="before")` does NOT intercept this — pydantic-settings raises during JSON-decode of the env value, before any validator runs. Verified locally; if a future pydantic-settings version changes this and starts accepting blank strings, the raw-string design still works (the helper just sees `""` and returns `[]`).

Use the @superpowers-extended-cc:test-driven-development skill for every code-bearing task.

---

## Chunk 1: Pool client + settings

This chunk builds the foundation: a sync `OllamaPool` routing core, the chat + embedding client classes, and the settings expansion (`get_ollama_*_urls()` returning `list[str]`). **No call site is migrated yet — at end of chunk the new client exists but nothing uses it.** Tests are unit-level + a thin smoke harness against a stub HTTP server.

**File map for this chunk:**
- Create: `app/services/ollama_pool_client.py` (the canonical implementation, ~250 lines)
- (Mirror at `docker/docling-graph/app/ollama_pool_client.py` is created in **Chunk 3**, Task 3.1 — not in this chunk. The canonical lives in `app/services/`; mirroring is downstream wiring.)
- Modify: `app/config.py:82-117` (add plural URL fields + new `get_*_urls()` methods, keep singular getters as `urls[0]`)
- Modify: `env.example:96-107` (add new plural vars with examples)
- Create: `tests/unit/services/test_ollama_pool_client.py`
- Create: `tests/smoke/ollama_pool_routing.py`
- Create: `tests/smoke/ollama_pool_failover.py`

### Task 1.1: Write the failing pool routing test

**Files:**
- Create: `tests/unit/services/test_ollama_pool_client.py`

- [ ] **Step 1: Write the failing test for least-in-flight routing.**

Create `tests/unit/services/test_ollama_pool_client.py`:

```python
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
```

- [ ] **Step 2: Run test, confirm fail.**

Run: `.venv/bin/pytest tests/unit/services/test_ollama_pool_client.py -q`
Expected: `ImportError: No module named 'app.services.ollama_pool_client'`

### Task 1.2: Implement OllamaPool routing core

**Files:**
- Create: `app/services/ollama_pool_client.py`

- [ ] **Step 1: Write minimal implementation.**

Create `app/services/ollama_pool_client.py`:

```python
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
import threading
import time
from typing import Any, Callable, Iterator, Literal, Mapping, Optional

import httpx

logger = logging.getLogger(__name__)


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
```

- [ ] **Step 2: Run tests, confirm pass.**

Run: `.venv/bin/pytest tests/unit/services/test_ollama_pool_client.py -q`
Expected: all tests in `tests/unit/services/test_ollama_pool_client.py` pass with no failures or errors. (Exact count drifts as new tests land — don't pin it.)

- [ ] **Step 3: Commit.**

```bash
git add app/services/ollama_pool_client.py tests/unit/services/test_ollama_pool_client.py
git commit -m "feat(ollama-pool): least-in-flight routing core w/ round-robin tie-break"
```

### Task 1.3: Add OllamaChatClient implementing LLMClientProtocol

**Files:**
- Modify: `app/services/ollama_pool_client.py`
- Modify: `tests/unit/services/test_ollama_pool_client.py`

- [ ] **Step 1: Write failing tests for chat client.**

Append to `tests/unit/services/test_ollama_pool_client.py` (the imports at the top of the file already cover `json`, `httpx`, `pytest`, `MagicMock`, `patch`; only the new symbol `OllamaChatClient` needs to be added):

```python
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
```

- [ ] **Step 2: Run, confirm fail.**

Run: `.venv/bin/pytest tests/unit/services/test_ollama_pool_client.py -q`
Expected: `ImportError: cannot import name 'OllamaChatClient'`.

- [ ] **Step 3: Implement OllamaChatClient.**

Append to `app/services/ollama_pool_client.py`:

```python
from typing import Callable, Optional


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
        client_error_cls: type[Exception] | None = None,
        parse_json_fn: Callable[[str], Any] | None = None,
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
        # Schema-embedding markers used by upstream's legacy prompt-builder.
        # When force_json_mode=True AND structured_output=False, the client
        # strips this block from prompt['user'] before sending — replaces the
        # _patched_get_json_response monkey-patch.
        self._legacy_marker_start = legacy_strip_marker_start
        self._legacy_marker_end = legacy_strip_marker_end
        self.last_call_diagnostics: dict | None = None
        # One httpx.Client per OllamaChatClient. httpx.Client is thread-safe
        # for concurrent .post() calls.
        self._http = httpx.Client(timeout=timeout_s)

    def __del__(self) -> None:
        try:
            self._http.close()
        except Exception:
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
        logger.info(
            "OllamaChatClient: stripped %d-char schema embedding from "
            "legacy retry prompt",
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
            body["format"] = "json"
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
            body["format"] = "json"
            return body
        if self._force_json_mode:
            body["format"] = "json"
            return body
        try:
            schema_dict = json.loads(schema_json)
        except json.JSONDecodeError:
            body["format"] = "json"
            return body
        if self._schema_transform:
            schema_dict = self._schema_transform(schema_dict)
        schema_serialized = json.dumps(schema_dict)
        if (
            self._threshold is not None
            and len(schema_serialized) > self._threshold
        ):
            body["format"] = "json"
        else:
            body["format"] = schema_dict
        return body

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
        """
        excluded: set[str] = set()
        last_exc: Exception | None = None
        eff_timeout = timeout_s if timeout_s is not None else self._default_timeout
        for attempt in range(2):
            url = self.pool.acquire(exclude=excluded)
            t0 = time.time()
            try:
                resp = self._http.post(
                    f"{url}/v1/chat/completions",
                    json=body,
                    timeout=eff_timeout,
                )
                resp.raise_for_status()
                try:
                    payload = resp.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    diag = {
                        "url": url, "model": self.model, "provider": self.provider,
                        "elapsed_s": time.time() - t0,
                        "structured_failed": True, "fallback_used": False,
                        "error": f"malformed response envelope: {exc}",
                        "raw_response": resp.text[:1000],
                    }
                    self.last_call_diagnostics = diag
                    if self._client_error_cls is not None:
                        raise self._client_error_cls(
                            "Ollama returned malformed JSON envelope",
                            details=diag,
                        ) from exc
                    raise
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
                logger.debug(
                    "OllamaChatClient: ok url=%s elapsed=%.2fs len(content)=%d",
                    url, time.time() - t0, len(content),
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
                logger.warning(
                    "OllamaChatClient: %s on %s (attempt %d/2): %s",
                    type(exc).__name__, url, attempt + 1, exc,
                )
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
        assert last_exc is not None
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
```

> **Note on schema_transform and the byte-for-byte mirror:** the canonical client at `app/services/ollama_pool_client.py` does NOT import `sanitize_schema_for_llm` (that helper lives only inside the docling-graph container). Instead it accepts a callback. The docling-graph wiring layer (`docker/docling-graph/app/config_builder.py`) constructs the client with `schema_transform=sanitize_schema_for_llm`. App-side callers leave it as `None`. The mirror stays byte-for-byte identical.

- [ ] **Step 4: Run, confirm pass.**

Run: `.venv/bin/pytest tests/unit/services/test_ollama_pool_client.py -q`
Expected: all tests in the file pass. (Exact count drifts as new tests land. The file should now exercise: pool routing, round-robin tie-break, chat client protocol attrs, format=schema, format=json, ClientError-wrap, empty-content reject, reasoning_content fallback, legacy strip ON, legacy strip OFF, parse_json_fn, parse-None wrap, real-ClientError integration, per-call timeout, per-call model override, force_json override, schema_transform, threshold gate, force_json_mode override, default_extra_params, gpt-oss think mapping, malformed-envelope wrap, routing_metrics.)

- [ ] **Step 5: Commit.**

```bash
git add app/services/ollama_pool_client.py tests/unit/services/test_ollama_pool_client.py
git commit -m "feat(ollama-pool): chat client implementing LLMClientProtocol"
```

### Task 1.4: Add OllamaEmbeddingClient

**Files:**
- Modify: `app/services/ollama_pool_client.py`
- Modify: `tests/unit/services/test_ollama_pool_client.py`

- [ ] **Step 1: Write failing tests.**

Append to `tests/unit/services/test_ollama_pool_client.py`:

```python
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
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement OllamaEmbeddingClient.**

Append to `app/services/ollama_pool_client.py`:

```python
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
        self._http = httpx.Client(timeout=timeout_s)

    def __del__(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed `texts` in one call. Returns vectors in the same order as the
        input. Caller is responsible for batching if texts is too large for
        a single request."""
        excluded: set[str] = set()
        last_exc: Exception | None = None
        for _ in range(2):
            url = self.pool.acquire(exclude=excluded)
            try:
                resp = self._http.post(
                    f"{url}/v1/embeddings",
                    json={"model": self.model, "input": texts},
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
                items = sorted(data, key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in items]
            except (
                httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError,
            ) as exc:
                last_exc = exc
                excluded.add(url)
                if len(excluded) >= len(self.pool.urls):
                    break
            finally:
                self.pool.release(url)
        assert last_exc is not None
        raise last_exc
```

- [ ] **Step 4: Run, confirm pass.**

Run: `.venv/bin/pytest tests/unit/services/test_ollama_pool_client.py -q`
Expected: all tests in the file pass. (Embedding tests add to the chat-client tests from Task 1.3. Re-run on every task to catch regressions; don't pin a count.)

- [ ] **Step 5: Commit.**

```bash
git add app/services/ollama_pool_client.py tests/unit/services/test_ollama_pool_client.py
git commit -m "feat(ollama-pool): embedding client"
```

### Task 1.5: Settings — plural URL fields with fallback

**Files:**
- Modify: `app/config.py:82-117`

- [ ] **Step 1: Write failing test for settings expansion.**

Create `tests/unit/test_config_ollama_pools.py`:

```python
"""Test the new plural OLLAMA_*_BASE_URLS env vars and getters."""
import importlib
import os
import pytest


def _reload_settings(monkeypatch, **env: str):
    monkeypatch.setattr(os, "environ", {**os.environ, **env}, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import app.config as cfg
    importlib.reload(cfg)
    cfg.get_settings.cache_clear()
    return cfg.get_settings()


def test_plural_takes_precedence_over_singular(monkeypatch):
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URL="http://singular:11434",
        OLLAMA_LLM_BASE_URLS='["http://h1:11434","http://h2:11434"]',
    )
    assert s.get_ollama_llm_urls() == [
        "http://h1:11434", "http://h2:11434",
    ]


def test_singular_used_when_plural_empty(monkeypatch):
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URL="http://singular:11434",
        OLLAMA_LLM_BASE_URLS="",
    )
    assert s.get_ollama_llm_urls() == ["http://singular:11434"]


def test_base_url_used_when_both_empty(monkeypatch):
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URL="",
        OLLAMA_LLM_BASE_URLS="",
    )
    assert s.get_ollama_llm_urls() == ["http://fallback:11434"]


def test_singular_getter_returns_first_url(monkeypatch):
    """Back-compat: existing call sites that only know about singular keep
    working — get_ollama_llm_url() returns urls[0]."""
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URLS='["http://h1:11434","http://h2:11434"]',
    )
    assert s.get_ollama_llm_url() == "http://h1:11434"


def test_vlm_and_embedding_pools_independent(monkeypatch):
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URLS='["http://llm-1"]',
        OLLAMA_VLM_BASE_URLS='["http://vlm-1","http://vlm-2"]',
        OLLAMA_EMBEDDING_BASE_URLS='["http://emb-1"]',
    )
    assert s.get_ollama_llm_urls() == ["http://llm-1"]
    assert s.get_ollama_vlm_urls() == ["http://vlm-1", "http://vlm-2"]
    assert s.get_ollama_embedding_urls() == ["http://emb-1"]


def test_blank_plural_env_var_does_not_crash_startup(monkeypatch):
    """Production .env files commonly leave OLLAMA_LLM_BASE_URLS= blank when
    the operator uses the singular var instead. Storing the plural as a raw
    str (not list[str]) sidesteps pydantic-settings' SettingsError trap."""
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URL="http://singular:11434",
        OLLAMA_LLM_BASE_URLS="",  # the trap case — must NOT raise
        OLLAMA_VLM_BASE_URLS="",
        OLLAMA_EMBEDDING_BASE_URLS="",
    )
    # Stored as raw string, not list.
    assert s.ollama_llm_base_urls == ""
    # Helper falls through to the singular var.
    assert s.get_ollama_llm_urls() == ["http://singular:11434"]


def test_malformed_plural_url_var_raises_at_read_time(monkeypatch):
    """Misconfigured JSON in plural env var must fail loudly when first
    consumed (rather than silently falling through to singular)."""
    import pytest
    s = _reload_settings(
        monkeypatch,
        OLLAMA_BASE_URL="http://fallback:11434",
        OLLAMA_LLM_BASE_URLS='["http://h1', # malformed (unclosed)
    )
    # Construction succeeds; failure is at first call to the helper.
    with pytest.raises(ValueError, match="not valid JSON"):
        s.get_ollama_llm_urls()


def test_non_array_plural_url_var_raises_at_read_time(monkeypatch):
    import pytest
    s = _reload_settings(
        monkeypatch,
        OLLAMA_LLM_BASE_URLS='"http://just-a-string"',
    )
    with pytest.raises(ValueError, match="JSON array of strings"):
        s.get_ollama_llm_urls()
```

- [ ] **Step 2: Run, confirm fail.**

Run: `.venv/bin/pytest tests/unit/test_config_ollama_pools.py -q`
Expected: `AttributeError: ... has no attribute 'get_ollama_llm_urls'`

- [ ] **Step 3: Modify `app/config.py:82-117`.**

The plural URL fields are stored as **raw strings**, not `list[str]`. pydantic-settings tries to JSON-decode `list[str]` env values during source-loading and raises `SettingsError` on blank string — *before* any field-validator runs. So we keep the env value as a `str`, and the helper methods parse it. This sidesteps the decode trap entirely and keeps the parsing logic obvious.

Replace the existing block:

```python
    # Ollama connection
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_base_url: str = ""
    ollama_vlm_base_url: str = ""
    ollama_embedding_base_url: str = ""
    ollama_num_ctx: int = 16384
```

with:

```python
    # Ollama connection
    # Singular fallbacks (kept for back-compat with existing .env files).
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_base_url: str = ""
    ollama_vlm_base_url: str = ""
    ollama_embedding_base_url: str = ""
    # Plural pools — JSON arrays as raw strings, e.g.
    # '["http://host1:11434","http://host2:11434"]'.
    # Stored as `str` (not `list[str]`) because pydantic-settings raises
    # SettingsError on blank-string values for list[str] before any
    # validator runs. Parsed in the get_ollama_*_urls() helpers below.
    # When set, these take precedence over the singular variants.
    ollama_llm_base_urls: str = ""
    ollama_vlm_base_urls: str = ""
    ollama_embedding_base_urls: str = ""
    ollama_num_ctx: int = 16384
```

And replace the existing getter block with parse-on-read helpers:

```python
    @staticmethod
    def _parse_url_pool(raw: str) -> list[str]:
        """Parse a JSON-array env value into list[str].

        Returns [] for blank/unset. Raises ValueError for malformed JSON
        or non-array values so misconfiguration fails loudly at startup
        rather than silently falling through to the singular fallback.
        """
        import json
        s = (raw or "").strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"OLLAMA_*_BASE_URLS env value is not valid JSON: {exc}; "
                f"expected a JSON array like '[\"http://h1:11434\",...]', "
                f"got: {s!r}"
            ) from exc
        if not isinstance(parsed, list) or not all(
            isinstance(x, str) for x in parsed
        ):
            raise ValueError(
                f"OLLAMA_*_BASE_URLS must be a JSON array of strings; got: {parsed!r}"
            )
        # Reject blank entries — they'd silently break the pool's
        # least-in-flight invariant (an empty URL would still get acquire'd
        # and posted to, raising httpx.UnsupportedProtocol on every request).
        if not all(x.strip() for x in parsed):
            raise ValueError(
                f"OLLAMA_*_BASE_URLS contains blank entries; got: {parsed!r}"
            )
        return parsed

    def get_ollama_llm_urls(self) -> list[str]:
        """Return the LLM (chat/reasoning) URL pool.

        Priority: ollama_llm_base_urls (plural JSON) > ollama_llm_base_url
        (singular) > ollama_base_url. Always returns a non-empty list.
        """
        plural = self._parse_url_pool(self.ollama_llm_base_urls)
        if plural:
            return plural
        if self.ollama_llm_base_url:
            return [self.ollama_llm_base_url]
        return [self.ollama_base_url]

    def get_ollama_vlm_urls(self) -> list[str]:
        plural = self._parse_url_pool(self.ollama_vlm_base_urls)
        if plural:
            return plural
        if self.ollama_vlm_base_url:
            return [self.ollama_vlm_base_url]
        return [self.ollama_base_url]

    def get_ollama_embedding_urls(self) -> list[str]:
        plural = self._parse_url_pool(self.ollama_embedding_base_urls)
        if plural:
            return plural
        if self.ollama_embedding_base_url:
            return [self.ollama_embedding_base_url]
        return [self.ollama_base_url]

    # Back-compat singular getters: return urls[0] from the new pools.
    def get_ollama_llm_url(self) -> str:
        return self.get_ollama_llm_urls()[0]

    def get_ollama_vlm_url(self) -> str:
        return self.get_ollama_vlm_urls()[0]

    def get_ollama_embedding_url(self) -> str:
        return self.get_ollama_embedding_urls()[0]
```

(Replace the existing `get_ollama_llm_url` / `get_ollama_vlm_url` / `get_ollama_embedding_url` methods at app/config.py:109-116 with the block above. Do not keep both definitions.)

And replace the existing getter block:

```python
    def get_ollama_llm_url(self) -> str:
        return self.ollama_llm_base_url or self.ollama_base_url

    def get_ollama_vlm_url(self) -> str:
        return self.ollama_vlm_base_url or self.ollama_base_url

    def get_ollama_embedding_url(self) -> str:
        return self.ollama_embedding_base_url or self.ollama_base_url
```

**(The replacement block — `_parse_url_pool` helper plus parse-on-read getters — is shown above this paragraph in this same Step. Don't add a second copy.)** Make sure the OLD `get_ollama_llm_url` / `get_ollama_vlm_url` / `get_ollama_embedding_url` methods are removed; the new pool-aware versions replace them, and the singular getters become thin wrappers around `urls[0]`.

- [ ] **Step 4: Run, confirm pass.**

Run: `.venv/bin/pytest tests/unit/test_config_ollama_pools.py -q`
Expected: all tests in `tests/unit/test_config_ollama_pools.py` pass. (Exact count drifts as new tests land — don't pin it.)

- [ ] **Step 5: Confirm full suite still green.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit -q 2>&1 | tail -3`
Expected: same passing count as P0 baseline (no regressions).

- [ ] **Step 6: Commit.**

```bash
git add app/config.py tests/unit/test_config_ollama_pools.py
git commit -m "feat(config): plural OLLAMA_*_BASE_URLS pool vars w/ singular fallback"
```

### Task 1.6: env.example — document the new vars

**Files:**
- Modify: `env.example:96-107`

- [ ] **Step 1: Edit env.example to add plural lines next to the singular ones.**

After the existing `OLLAMA_LLM_BASE_URL` / `OLLAMA_VLM_BASE_URL` / `OLLAMA_EMBEDDING_BASE_URL` lines (around lines 105-107), add:

```bash
# Pool URLs (NEW). JSON array form. When set, takes precedence over the
# singular OLLAMA_*_BASE_URL above. The OllamaPool client load-balances
# across all listed instances using least-in-flight routing.
# Example: '["http://10.0.1.121:11434","http://10.0.1.122:11434"]'
OLLAMA_LLM_BASE_URLS=
OLLAMA_VLM_BASE_URLS=
OLLAMA_EMBEDDING_BASE_URLS=
```

- [ ] **Step 2: Verify env.example values parse cleanly.**

Run:
```bash
.venv/bin/python -c "
from app.config import Settings
import os
# Empty string must NOT crash startup. Stored as '' on the model;
# get_ollama_*_urls() helpers parse on read and fall through to singular.
for k in ('OLLAMA_LLM_BASE_URLS','OLLAMA_VLM_BASE_URLS','OLLAMA_EMBEDDING_BASE_URLS'):
    os.environ[k] = ''
s = Settings()
assert s.ollama_llm_base_urls == ''
print('OK')
"
```
Expected: `OK`. The plural fields are stored as raw `str` precisely because pydantic-settings raises `SettingsError` on blank-string values for `list[str]` before any validator runs (verified in P3 preflight). The custom JSON parser in `Settings._parse_url_pool` handles the read-time conversion.

- [ ] **Step 3: Commit.**

```bash
git add env.example
git commit -m "docs(env): document OLLAMA_*_BASE_URLS pool vars"
```

### Task 1.7: Smoke harness — pool routing fan-out

**Files:**
- Create: `tests/smoke/ollama_pool_routing.py`

- [ ] **Step 1: Write the harness.**

Create `tests/smoke/ollama_pool_routing.py`:

```python
"""Smoke harness: 4-URL pool with stub HTTP servers.

Fires 16 concurrent requests via OllamaChatClient; asserts each URL gets
exactly 4 (least-in-flight under uniform service time = even distribution).

Run: .venv/bin/python -m tests.smoke.ollama_pool_routing
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.services.ollama_pool_client import OllamaChatClient, OllamaPool


_hits: dict[int, int] = {}
_hits_lock = threading.Lock()


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        port = self.server.server_address[1]
        with _hits_lock:
            _hits[port] = _hits.get(port, 0) + 1
        # Simulate ~50ms generation so multiple requests overlap.
        time.sleep(0.05)
        body = b'{"choices":[{"message":{"content":"{}"}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kw):
        pass  # silence


def _start_stub() -> tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), _StubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def main() -> None:
    servers = [_start_stub() for _ in range(4)]
    urls = [u for _, u in servers]
    pool = OllamaPool(urls=urls)
    client = OllamaChatClient(pool=pool, model="stub")

    def call(_):
        return client.get_json_response(
            prompt="hi", schema_json="{}", structured_output=False,
        )

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(call, range(16)))

    for srv, _ in servers:
        srv.shutdown()

    distribution = {url: _hits[int(url.rsplit(":", 1)[1])] for url in urls}
    print("Distribution:", distribution)
    counts = sorted(distribution.values())
    # Least-in-flight isn't perfectly even, but with uniform service times
    # the spread should be small. Allow up to 2x the lowest.
    assert counts[0] >= 2, f"Some URL got <2 requests: {counts}"
    assert counts[-1] <= 8, f"Some URL got >8 requests: {counts}"
    assert sum(counts) == 16
    print("PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run.**

Run: `.venv/bin/python -m tests.smoke.ollama_pool_routing`
Expected: prints `Distribution: {...}` followed by `PASS`.

- [ ] **Step 3: Commit.**

```bash
git add tests/smoke/ollama_pool_routing.py
git commit -m "test(ollama-pool): smoke harness for fan-out distribution"
```

### Task 1.8: Smoke harness — failover

**Files:**
- Create: `tests/smoke/ollama_pool_failover.py`

- [ ] **Step 1: Write the harness.**

Create `tests/smoke/ollama_pool_failover.py`:

```python
"""Smoke harness: 2-URL pool, kill one mid-stream, assert survivor handles all.

Run: .venv/bin/python -m tests.smoke.ollama_pool_failover
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.services.ollama_pool_client import OllamaChatClient, OllamaPool


def _make_handler():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            time.sleep(0.05)
            body = b'{"choices":[{"message":{"content":"{}"}}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_a, **_kw):
            pass
    return H


def _start() -> tuple[HTTPServer, str]:
    srv = HTTPServer(("127.0.0.1", 0), _make_handler())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def main() -> None:
    srv1, u1 = _start()
    srv2, u2 = _start()
    pool = OllamaPool(urls=[u1, u2])
    client = OllamaChatClient(pool=pool, model="stub", timeout_s=5.0)

    # Kill srv1 after firing 2 requests; remaining 18 must succeed on srv2.
    def call(i):
        if i == 2:
            srv1.shutdown()
        try:
            client.get_json_response(
                prompt="hi", schema_json="{}", structured_output=False,
            )
            return True
        except Exception as exc:
            return f"FAIL: {type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(call, range(20)))

    srv2.shutdown()

    failures = [r for r in results if r is not True]
    print(f"Successes: {sum(1 for r in results if r is True)}/20")
    if failures:
        print("Failures:", failures[:5])
    # Allow a couple of mid-flight calls on srv1 to fail; the retry-once
    # pattern only handles connect/timeout, not in-flight kill. The
    # majority must succeed.
    assert sum(1 for r in results if r is True) >= 16, results
    print("PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run.**

Run: `.venv/bin/python -m tests.smoke.ollama_pool_failover`
Expected: prints `Successes: N/20` (N≥16) followed by `PASS`.

- [ ] **Step 3: Commit.**

```bash
git add tests/smoke/ollama_pool_failover.py
git commit -m "test(ollama-pool): smoke harness for single-URL failover"
```

---

## Chunk 2: Migrate api-side call sites

This chunk swaps every app-side `httpx`-to-Ollama call onto the new pool client. End state: `app/services/document_analysis.py`, `app/services/translation.py`, `app/services/embedding.py`, `app/services/arcadedb_community.py`, and `app/api/v1/retrieval.py::_synthesize_global_answer` all import from `app.services.ollama_pool_client` and use `OllamaChatClient` / `OllamaEmbeddingClient`. Each task migrates one file; integration tests catch regressions.

**Cross-task helpers we'll add:**
- `app/services/ollama_clients.py` — module-level `lru_cache`d factories: `get_llm_client()`, `get_vlm_client()`, `get_embedding_client()`. Each builds the pool from `settings.get_ollama_*_urls()` and the role-specific model env var. Centralizes wiring so every call site imports the factory rather than reconstructing.

### Task 2.1: Module-level client factories

**Files:**
- Create: `app/services/ollama_clients.py`
- Create: `tests/unit/services/test_ollama_clients_factory.py`

- [ ] **Step 1: Write failing tests.**

```python
"""Tests for the module-level pool/client factory cache."""
from app.services.ollama_clients import (
    get_llm_client, get_vlm_client, get_embedding_client,
)


def test_llm_client_is_cached_singleton():
    c1 = get_llm_client()
    c2 = get_llm_client()
    assert c1 is c2


def test_factories_use_role_specific_pools(monkeypatch):
    # Order matters: clear caches BEFORE patching env, otherwise a previous
    # test's cached singleton wins and the new env values are ignored.
    from app.config import get_settings
    from app.services import ollama_clients
    get_settings.cache_clear()
    ollama_clients.get_llm_client.cache_clear()
    ollama_clients.get_vlm_client.cache_clear()
    ollama_clients.get_embedding_client.cache_clear()
    monkeypatch.setenv("OLLAMA_LLM_BASE_URLS", '["http://llm-1"]')
    monkeypatch.setenv("OLLAMA_VLM_BASE_URLS", '["http://vlm-1"]')
    monkeypatch.setenv("OLLAMA_EMBEDDING_BASE_URLS", '["http://emb-1"]')

    assert get_llm_client().pool.urls == ["http://llm-1"]
    assert get_vlm_client().pool.urls == ["http://vlm-1"]
    assert get_embedding_client().pool.urls == ["http://emb-1"]
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement `app/services/ollama_clients.py`:**

```python
"""Module-level cached factories for the role-scoped Ollama clients.

One pool per role (LLM / VLM / embedding). Cached via `lru_cache` so each
process has at most one client per role; tests can `clear_cache()` to
rebuild against patched env.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.ollama_pool_client import (
    OllamaChatClient, OllamaEmbeddingClient, OllamaPool,
)


@lru_cache(maxsize=1)
def get_llm_client() -> OllamaChatClient:
    s = get_settings()
    return OllamaChatClient(
        pool=OllamaPool(urls=s.get_ollama_llm_urls()),
        model=s.doc_analysis_llm_model,  # default; per-call overrides via .chat(model=...)
        timeout_s=float(s.doc_analysis_timeout),
        max_tokens=s.llm_max_tokens,
    )


@lru_cache(maxsize=1)
def get_vlm_client() -> OllamaChatClient:
    s = get_settings()
    return OllamaChatClient(
        pool=OllamaPool(urls=s.get_ollama_vlm_urls()),
        model=s.picture_description_model,
        timeout_s=float(s.picture_description_timeout),
        max_tokens=s.llm_max_tokens,
    )


@lru_cache(maxsize=1)
def get_embedding_client() -> OllamaEmbeddingClient:
    s = get_settings()
    return OllamaEmbeddingClient(
        pool=OllamaPool(urls=s.get_ollama_embedding_urls()),
        model=s.text_embedding_model,
        timeout_s=120.0,
    )
```

- [ ] **Step 4: Run, confirm pass.**

- [ ] **Step 5: Commit.**

```bash
git add app/services/ollama_clients.py tests/unit/services/test_ollama_clients_factory.py
git commit -m "feat(ollama-pool): cached factories per role (llm/vlm/embedding)"
```

> **Important:** the factory caches `model` from the settings at first construction. Different roles (translation, doc-analysis, community-report) want **different models**, all served by the same LLM pool. Per-call `model=` override is via `OllamaChatClient.chat(messages, model=...)` (added in Task 2.2). The same client instance is reused across roles; only the pool routing benefit is shared. We do NOT spin up a separate client per model — that would defeat the pool-sharing.

### Task 2.2: ~~Per-call model/force_json override~~ — REMOVED in v4

The per-call `model=`, `force_json=`, `extra_params=`, and `timeout_s=` overrides on `OllamaChatClient.chat()` are already implemented in Task 1.3 (the v3 chat() signature includes all of them). No new work in this task; numbering preserved so downstream task IDs stay stable. Skip to Task 2.3.

To verify the existing implementation handles per-call overrides correctly, the unit-test added in v3/v4 (`test_chat_per_call_model_override`) is sufficient. Confirm it's in `tests/unit/services/test_ollama_pool_client.py` before moving on.

### Task 2.3: Migrate document_analysis.py

**Files:**
- Modify: `app/services/document_analysis.py:19-45, 60-66, 138-232`

- [ ] **Step 1: Write integration test (skipped by default, runs against live Ollama only when `OLLAMA_LIVE=1`).**

`tests/integration/test_document_analysis_pool.py`:

```python
import os
import pytest


@pytest.mark.skipif(os.environ.get("OLLAMA_LIVE") != "1",
                    reason="requires live Ollama")
def test_extract_document_metadata_via_pool():
    from app.services.document_analysis import extract_document_metadata
    md = "# Hello\n\nThis is a test document about radar systems."
    result = extract_document_metadata(md)
    assert "document_summary" in result
    assert isinstance(result["document_summary"], str)
```

- [ ] **Step 2: Migrate `extract_document_metadata`.**

Replace the body of `_ollama_chat`, `_llm_call`, and the `with ThreadPoolExecutor(...)` orchestration so it goes through `get_llm_client().chat(...)`. Specifically:

1. Delete the local `_ollama_chat` helper (lines 19-45). It's replaced by the pool client.
2. Rewrite `extract_document_metadata` to:

```python
def extract_document_metadata(markdown: str, classification_text: str | None = None) -> dict:
    settings = get_settings()
    from app.services.ollama_clients import get_llm_client
    client = get_llm_client()
    model = settings.doc_analysis_llm_model
    think = settings.get_doc_analysis_llm_think()
    timeout = settings.doc_analysis_timeout

    max_chars = settings.ollama_num_ctx * 3
    doc_text = markdown[:max_chars] if len(markdown) > max_chars else markdown
    raw_class_text = classification_text if classification_text is not None else markdown
    class_text = raw_class_text[:max_chars] if len(raw_class_text) > max_chars else raw_class_text

    def _llm_call(system_prompt: str, user_text: str) -> str:
        return client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            model=model,
            temperature=0.1,
            max_tokens=settings.llm_max_tokens,
            think=think,
            timeout_s=float(timeout),  # role-specific: doc_analysis_timeout
        )

    results: dict[str, str] = {}
    non_class_prompts = {
        "document_summary": settings.doc_analysis_summary_prompt,
        "date_of_information": settings.doc_analysis_date_prompt,
        "source_characterization": settings.doc_analysis_source_prompt,
    }

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures: dict = {
            pool.submit(_llm_call, prompt, doc_text): key
            for key, prompt in non_class_prompts.items()
        }
        futures[pool.submit(_llm_call, settings.doc_analysis_classification_prompt, class_text)] = "classification"
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logger.warning("Document metadata '%s' failed: %s", key, e)
                results[key] = "Unknown" if key != "classification" else "UNCLASSIFIED"

    valid_classes = {"UNCLASSIFIED", "CUI", "FOUO", "SECRET", "TOP SECRET"}
    classification = results.get("classification", "UNCLASSIFIED").upper().strip()
    if classification not in valid_classes:
        classification = "UNCLASSIFIED"

    return {
        "document_summary": results.get("document_summary", ""),
        "date_of_information": results.get("date_of_information", "Unknown"),
        "classification": classification,
        "source_characterization": results.get("source_characterization", "Unknown"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
```

3. Migrate `describe_pictures` and `_describe_single_image` to use `get_vlm_client()`:

```python
def _describe_single_image(image_b64: str, prompt: str, model: str, timeout: int, settings) -> str | None:
    try:
        from app.services.ollama_clients import get_vlm_client
        client = get_vlm_client()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }]
        content = client.chat(
            messages=messages, model=model,
            temperature=0.2, max_tokens=settings.llm_max_tokens,
            think=settings.get_picture_description_think(),
            timeout_s=float(timeout),  # picture_description_timeout per call
        )
        return content
    except Exception as e:
        logger.warning("Picture description failed: %s", e)
        return None
```

4. Remove the `import httpx` at top of file.

- [ ] **Step 3: Run unit suite, confirm green.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit -q 2>&1 | tail -3`
Expected: same passing count as P0.

- [ ] **Step 4: Run live integration test against local Ollama.**

```bash
OLLAMA_LIVE=1 .venv/bin/pytest tests/integration/test_document_analysis_pool.py -q
```
Expected: `1 passed`. (If your dev machine doesn't have Ollama, skip this step and let CI catch it.)

- [ ] **Step 5: Commit.**

```bash
git add app/services/document_analysis.py tests/integration/test_document_analysis_pool.py
git commit -m "refactor(doc-analysis): use OllamaChatClient/VLM pool client"
```

### Task 2.4: Migrate translation.py

**Files:**
- Modify: `app/services/translation.py`

- [ ] **Step 1: Replace `_ollama_translate` body to call `get_llm_client().chat(...)`.**

The current code imports `_ollama_chat` from `document_analysis` (line 207); after Task 2.3 that helper is gone. Rewrite `_ollama_translate` to use the pool client directly:

```python
def _ollama_translate(
    model: str, prompt: str, text: str,
    *, temperature: float, max_tokens: int, timeout: int,
    think: str | bool | None = None,
) -> str:
    """Single-element translation via pool client."""
    from app.services.ollama_clients import get_llm_client
    pool_client = get_llm_client()
    return pool_client.chat(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        model=model, temperature=temperature, max_tokens=max_tokens,
        think=think,
        timeout_s=float(timeout),  # role-specific: translation_timeout
    )
```

Update the call sites at translation.py:145, 157, 188-194 to drop the now-removed `client` and `url` positional args. Old: `_ollama_translate(client, url, model, prompt, text, ...)`. New: `_ollama_translate(model, prompt, text, ...)`. Run pytest after the edit so any missed caller fails fast (the function will raise `TypeError` on the extra positional args).

- [ ] **Step 2: Remove the `httpx.Client(...)` block (lines 141-200ish) and the `httpx` import.**

- [ ] **Step 3: Run unit suite + the existing translation pipeline test.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/services/test_translation.py tests/pipeline -k translation -q`
Expected: same passing count.

- [ ] **Step 4: Commit.**

```bash
git add app/services/translation.py
git commit -m "refactor(translation): use OllamaChatClient pool client"
```

### Task 2.5: Migrate arcadedb_community.py

**Files:**
- Modify: `app/services/arcadedb_community.py:228-285`

- [ ] **Step 1: Rewrite `_call_llm_for_report` to use the pool client via `asyncio.to_thread` (callers are async).**

```python
async def _call_llm_for_report(prompt: str, model: str) -> dict[str, str]:
    """Call Ollama via pool client; expects JSON {title, summary}."""
    from app.config import get_settings
    from app.services.llm_json import parse_llm_json_loose
    from app.services.ollama_clients import get_llm_client

    settings = get_settings()
    client = get_llm_client()
    think = settings.get_community_report_llm_think()
    # Community reports historically used doc_analysis_timeout; keep that
    # for v1 to avoid a behavior change. If we add a dedicated
    # community_report_timeout later, swap it in here.
    timeout = settings.doc_analysis_timeout

    def _sync_call() -> str:
        return client.chat(
            messages=[
                {"role": "system", "content": (
                    "You are a knowledge-graph analyst. "
                    "Respond with a single JSON object containing "
                    '"title" (short, descriptive) and "summary" '
                    "(2-4 paragraphs). Do not include any other text."
                )},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.1,
            max_tokens=settings.llm_max_tokens,
            think=think,
            force_json=True,
            timeout_s=float(timeout),
        )

    content = await asyncio.to_thread(_sync_call)

    parsed = parse_llm_json_loose(content)
    if isinstance(parsed, dict) and parsed.get("title") and parsed.get("summary"):
        return {"title": str(parsed["title"]).strip(),
                "summary": str(parsed["summary"]).strip()}
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return {"title": "Community Report", "summary": content or ""}
    return {
        "title": lines[0][:200],
        "summary": "\n".join(lines[1:]) if len(lines) > 1 else lines[0],
    }
```

- [ ] **Step 2: Remove the `import httpx` if no other code in this file uses it. (Check first; arcadedb_community has lots of code.)**

- [ ] **Step 3: Run community-report unit tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit -k community -q`
Expected: same passing count.

- [ ] **Step 4: Commit.**

```bash
git add app/services/arcadedb_community.py
git commit -m "refactor(community): use OllamaChatClient pool via asyncio.to_thread"
```

### Task 2.6: Migrate retrieval.py::_synthesize_global_answer

**Files:**
- Modify: `app/api/v1/retrieval.py:1171-1217`

- [ ] **Step 1: Same `asyncio.to_thread` pattern as Task 2.5.**

```python
async def _synthesize_global_answer(query: str, reports: list[dict]) -> str:
    from app.config import get_settings
    from app.services.ollama_clients import get_llm_client
    settings = get_settings()
    client = get_llm_client()

    template = settings.community_global_synthesis_prompt or _DEFAULT_GLOBAL_SYNTHESIS_PROMPT
    reports_text = "\n\n".join(
        f"[community {r.get('community_id', '?')}] "
        f"{r.get('title', '')}\n{r.get('summary', '')}"
        for r in reports
    )
    prompt = template.replace("{query}", query).replace("{reports}", reports_text)

    timeout = settings.doc_analysis_timeout  # historical reuse; see Task 2.5

    def _sync_call() -> str:
        return client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=settings.community_report_llm_model,
            temperature=0.2,
            max_tokens=settings.llm_max_tokens,
            think=settings.get_community_report_llm_think(),
            timeout_s=float(timeout),
        )

    try:
        content = await asyncio.to_thread(_sync_call)
        return content or _fallback_concatenated_reports(reports)
    except Exception as exc:
        logger.warning("Global synthesis LLM call failed: %s; returning concatenated reports", exc)
        return _fallback_concatenated_reports(reports)
```

- [ ] **Step 2: Remove the `import httpx` inside the function (line 1176) since it's no longer needed.**

- [ ] **Step 3: Confirm retrieval tests green.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit -k retrieval -q`

- [ ] **Step 4: Commit.**

```bash
git add app/api/v1/retrieval.py
git commit -m "refactor(retrieval): use OllamaChatClient pool for global synthesis"
```

### Task 2.7: Migrate embedding.py

**Files:**
- Modify: `app/services/embedding.py:22-83`

- [ ] **Step 1: Rewrite `embed_texts` to delegate to `OllamaEmbeddingClient.embed`.**

Replace the body:

```python
def embed_texts(texts: list[str], batch_size: int = 64, *, query: bool = False) -> list[list[float]]:
    if not texts:
        return []
    settings = get_settings()
    if "bge" in settings.text_embedding_model.lower():
        if query:
            texts = [f"Represent this query for searching relevant passages: {t}" for t in texts]
        else:
            texts = [f"Represent this sentence: {t}" for t in texts]

    from app.services.ollama_clients import get_embedding_client
    client = get_embedding_client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        all_embeddings.extend(client.embed(batch))

    arr = np.array(all_embeddings)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    arr = arr / norms
    return arr.tolist()
```

- [ ] **Step 2: Remove `_thread_local`, `_get_http_client`, `import httpx`, `import threading` (verify nothing else uses them in this file).**

- [ ] **Step 3: Run embedding tests.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit -k embedding -q`

- [ ] **Step 4: Commit.**

```bash
git add app/services/embedding.py
git commit -m "refactor(embedding): use OllamaEmbeddingClient pool"
```

### Task 2.8: Full-suite regression check

- [ ] **Step 1: Run.**

Run: `SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline -q 2>&1 | tail -3`
Expected: same passing count as P0 baseline; no regressions.

- [ ] **Step 2: If any regression, fix before proceeding to Chunk 3.**

---

## Chunk 3: Wire OllamaChatClient into docling-graph (patches kept as safety net)

This chunk swaps the docling-graph service onto the pool client. **The LiteLLM patches stay in place during this chunk** — once `PipelineConfig(llm_client=...)` is set, the library bypasses LiteLLMClient entirely (verified in `pipeline/stages.py:470`), so the `LiteLLMClient._build_request` / `_call_api` patches are dormant but harmless. They get deleted in Chunk 5, after Chunk 4 validation confirms the new client works in production. This ordering keeps a working fallback path the entire time.

**File map for this chunk:**
- Create: `docker/docling-graph/app/ollama_pool_client.py` (mirror of `app/services/ollama_pool_client.py`)
- Modify: `docker/docling-graph/app/config_builder.py` (add `llm_client` to PipelineConfig kwargs; expand DoclingGraphSettings to read OLLAMA_LLM_BASE_URLS)
- Modify: `docker/docling-graph/app/main.py` (NO patch deletions in this chunk — LiteLLM patches stay as safety net through end of Chunk 4. Patch deletion happens in Chunk 5.)

### Task 3.1: Mirror the pool client AND the loose JSON parser into docker/docling-graph/

**Files:**
- Create: `docker/docling-graph/app/ollama_pool_client.py`
- Create: `docker/docling-graph/app/llm_json.py`

The docling-graph container runs in a separate Python tree and does NOT import from `app/services/`. Both files must be physically present in `docker/docling-graph/app/`. The mirror-drift CI test enforces sync.

- [ ] **Step 1: Copy.**

```bash
cp app/services/ollama_pool_client.py docker/docling-graph/app/ollama_pool_client.py
cp app/services/llm_json.py docker/docling-graph/app/llm_json.py
```

- [ ] **Step 2: Edit the top-of-file docstrings of BOTH copies** to mark them as mirrors:

For `docker/docling-graph/app/ollama_pool_client.py`:
```python
"""Direct-Ollama pool client (docling-graph mirror).

MIRRORED FROM app/services/ollama_pool_client.py — keep byte-for-byte
identical below the SHARED CODE marker. Two copies exist because
docker/docling-graph/ runs in a separate container with its own Python
tree; both must implement the same routing core. If you change the
canonical file, copy it here too.
"""
# === SHARED CODE BELOW THIS LINE ===
... (rest of file unchanged)
```

For `docker/docling-graph/app/llm_json.py`:
```python
"""Tolerant JSON parser (docling-graph mirror).

MIRRORED FROM app/services/llm_json.py — keep byte-for-byte identical
below the SHARED CODE marker.
"""
# === SHARED CODE BELOW THIS LINE ===
... (rest of file unchanged)
```

Add the `# === SHARED CODE BELOW THIS LINE ===` marker to both canonicals (`app/services/ollama_pool_client.py` and `app/services/llm_json.py`) immediately after their docstrings if not already present. Task 1.2 places the marker in `ollama_pool_client.py`; you may need to add it to `llm_json.py` as a one-line edit.

- [ ] **Step 3: Add a CI check that fails if the two copies drift.**

Both files must contain a marker line `# === SHARED CODE BELOW THIS LINE ===` immediately after the docstring; everything below the marker must be byte-for-byte identical.

Add to `tests/test_pool_client_mirror.py`:

```python
"""Fail loudly if the docling-graph mirror drifts from the canonical client."""
from pathlib import Path

_MARKER = "# === SHARED CODE BELOW THIS LINE ===\n"


def _shared_body(text: str, file_label: str) -> str:
    parts = text.split(_MARKER, 1)
    if len(parts) != 2:
        raise AssertionError(
            f"{file_label} is missing the marker line {_MARKER!r}; the "
            "mirror invariant requires it immediately after the docstring."
        )
    return parts[1]


_MIRROR_PAIRS = [
    ("app/services/ollama_pool_client.py",
     "docker/docling-graph/app/ollama_pool_client.py"),
    ("app/services/llm_json.py",
     "docker/docling-graph/app/llm_json.py"),
]


def test_pool_client_mirror_in_sync():
    for canonical_path, mirror_path in _MIRROR_PAIRS:
        canonical = Path(canonical_path).read_text()
        mirror = Path(mirror_path).read_text()
        canon_body = _shared_body(canonical, canonical_path)
        mirror_body = _shared_body(mirror, mirror_path)
        assert canon_body == mirror_body, (
            f"{mirror_path} drifted from {canonical_path} — copy the "
            "canonical file's shared section."
        )
```

When implementing Task 1.2 / 1.3 / 1.4, ensure the marker line appears exactly once in `app/services/ollama_pool_client.py`, immediately after the docstring. Same in `docker/docling-graph/app/ollama_pool_client.py`.

- [ ] **Step 4: Run.**

Run: `.venv/bin/pytest tests/test_pool_client_mirror.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Commit. Stage every file touched in this task — both mirror copies, the canonical-side marker line edits if they're new, the new mirror test, AND the llm_json mirror.**

```bash
git add \
  app/services/ollama_pool_client.py \
  app/services/llm_json.py \
  docker/docling-graph/app/ollama_pool_client.py \
  docker/docling-graph/app/llm_json.py \
  tests/test_pool_client_mirror.py
git commit -m "feat(ollama-pool): mirror pool client + llm_json into docling-graph service"
```

### Task 3.1b: Cached LLM-client factory for docling-graph

**Files:**
- Create: `docker/docling-graph/app/ollama_clients.py`

The existing `build_pipeline_config()` is called per extraction pass — if we instantiated `OllamaPool` + `OllamaChatClient` inside it, each concurrent pass would build its own pool with empty in-flight counters, defeating the routing logic and making the `/debug/routing-metrics` endpoint useless. Add a process-cached factory (mirroring `app/services/ollama_clients.py`) so all extraction passes inside the same uvicorn process share one pool and one client.

- [ ] **Step 1: Create the factory.**

```python
"""Process-cached docling-graph LLM client factory.

Mirrors app/services/ollama_clients.py but lives in the docling-graph
container. Concurrent extraction passes share one OllamaPool + one
OllamaChatClient per process so in-flight counters are accurate and
GET /debug/routing-metrics reports real fan-out.
"""
from __future__ import annotations

import os
from functools import lru_cache

from app.config_builder import DoclingGraphSettings
from app.ollama_pool_client import OllamaChatClient, OllamaPool


@lru_cache(maxsize=1)
def get_docling_llm_client() -> OllamaChatClient:
    """Return the process-cached OllamaChatClient for docling-graph extraction.

    First call constructs the pool from DoclingGraphSettings + service-level
    settings (force_json_mode, structured_output_threshold_chars). Subsequent
    calls return the same instance. lru_cache.cache_clear() is exposed so
    tests can rebuild against patched env.
    """
    from app.config import settings as _service_settings
    from app.prompt_rules import sanitize_schema_for_llm
    from docling_graph.exceptions import ClientError
    from app.llm_json import parse_llm_json_loose  # mirror of app/services/llm_json.py

    settings = DoclingGraphSettings()
    pool = OllamaPool(urls=settings.get_ollama_llm_urls())

    default_extra_params = {
        "top_p": getattr(settings, "docling_graph_llm_top_p", None),
        "top_k": getattr(settings, "docling_graph_llm_top_k", None),
        "frequency_penalty": getattr(settings, "docling_graph_llm_frequency_penalty", None),
        "presence_penalty": getattr(settings, "docling_graph_llm_presence_penalty", None),
        "seed": getattr(settings, "docling_graph_llm_seed", None),
        "stop": getattr(settings, "docling_graph_llm_stop", None),
    }

    return OllamaChatClient(
        pool=pool,
        model=settings.docling_graph_llm_model,
        timeout_s=float(settings.docling_graph_llm_timeout),
        temperature=settings.docling_graph_llm_temperature,
        max_tokens=settings.docling_graph_llm_max_tokens,
        think=os.environ.get("DOCLING_GRAPH_LLM_THINK", "") or None,
        schema_transform=sanitize_schema_for_llm,
        force_json_mode=_service_settings.force_json_mode,
        structured_output_threshold_chars=_service_settings.structured_output_threshold_chars,
        default_extra_params=default_extra_params,
        client_error_cls=ClientError,
        parse_json_fn=parse_llm_json_loose,
    )
```

- [ ] **Step 2: Commit.**

```bash
git add docker/docling-graph/app/ollama_clients.py
git commit -m "feat(docling-graph): process-cached OllamaChatClient factory"
```

### Task 3.2: DoclingGraphSettings — read pool URLs (and pass them in via compose)

**Files:**
- Modify: `docker/docling-graph/app/config_builder.py:17-100`
- Modify: `docker-compose.yml` (docling-graph environment block, lines ~168-173)

- [ ] **Step 0: Pass `OLLAMA_LLM_BASE_URLS` through to the container.**

The docling-graph service does NOT use `env_file: .env`; only the env vars explicitly listed in the `environment:` block reach the container. The new plural var must be added or `DoclingGraphSettings` reads `""` and falls back to the singular.

In `docker-compose.yml`, locate the docling-graph `environment:` block (search for `OLLAMA_LLM_BASE_URL:` near line 170). Insert directly below the existing singular line:

```yaml
      # Pool URLs — JSON-array string. When set, takes precedence over
      # the singular OLLAMA_LLM_BASE_URL above.
      OLLAMA_LLM_BASE_URLS: ${OLLAMA_LLM_BASE_URLS:-}
```

Note the empty default (`:-}`): `DoclingGraphSettings.ollama_llm_base_urls` is stored as raw `str` and an empty value is fine.



- [ ] **Step 1: Add fields to `DoclingGraphSettings`.**

Inside the class, add (next to `ollama_llm_base_url` if it exists, else add the singular too). The plural field is stored as raw `str` (same reasoning as Task 1.5: pydantic-settings raises `SettingsError` on blank-string for `list[str]` before any validator runs).

The fallback default is `http://ollama:11434` (compose service name) — NOT `localhost:11434`. Inside the docling-graph container, `localhost` resolves to the container itself, not the Ollama host.

```python
    # Singular / fallback URLs (back-compat with existing .env files).
    ollama_base_url: str = "http://ollama:11434"
    ollama_llm_base_url: str = ""
    # Plural pool — raw JSON-array string. Parsed in get_ollama_llm_urls().
    ollama_llm_base_urls: str = ""

    def get_ollama_llm_urls(self) -> list[str]:
        """Parse priority: ollama_llm_base_urls (plural JSON) >
        ollama_llm_base_url (singular) > ollama_base_url. Always non-empty.
        """
        import json
        s = (self.ollama_llm_base_urls or "").strip()
        if s:
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"OLLAMA_LLM_BASE_URLS is not valid JSON: {exc}; "
                    f"got: {s!r}"
                ) from exc
            if not isinstance(parsed, list) or not all(
                isinstance(x, str) for x in parsed
            ):
                raise ValueError(
                    f"OLLAMA_LLM_BASE_URLS must be a JSON array of strings; "
                    f"got: {parsed!r}"
                )
            if not all(x.strip() for x in parsed):
                raise ValueError(
                    f"OLLAMA_LLM_BASE_URLS contains blank entries; got: {parsed!r}"
                )
            if parsed:
                return parsed
        if self.ollama_llm_base_url:
            return [self.ollama_llm_base_url]
        return [self.ollama_base_url]
```

- [ ] **Step 2: Modify `build_pipeline_config` to construct the pool client with FULL parity to `_patched_build_request`.**

The new client must preserve every load-bearing knob the patch had: schema sanitization (`sanitize_schema_for_llm`), `force_json_mode`, `structured_output_threshold_chars`, gpt-oss think-level mapping (handled inside the client via `_coerce_think`), and generation params (`top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `seed`, `stop`).

```python
def build_pipeline_config(...) -> Any:
    from docling_graph import PipelineConfig
    from app.ollama_clients import get_docling_llm_client

    settings = DoclingGraphSettings()
    quality_min_instances = ...  # (unchanged)

    # Build via the process-cached factory in app/ollama_clients.py. All
    # generation knobs (top_p / top_k / seed / stop / etc.), the schema
    # transform, force_json_mode, structured_output_threshold_chars, and
    # ClientError + parse_json_fn wiring live inside that factory — this
    # function is just a one-line consumer.
    llm_client = get_docling_llm_client()

    config_kwargs: dict[str, Any] = {
        "source": source,
        "llm_client": llm_client,  # NEW: bypass library's LiteLLM
        "backend": settings.docling_graph_backend,
        ...  # rest unchanged
    }
    # NOTE: keep `provider_override` and `model_override` — they're vestigial
    # when llm_client is set (pipeline/stages.py:470 short-circuits) but
    # leaving them avoids touching the rest of the kwargs.
    # Drop the `connection.base_url` line — the pool client owns connection
    # state now.
    config_kwargs["llm_overrides"] = {
        "generation": {
            "temperature": settings.docling_graph_llm_temperature,
            "max_tokens": settings.docling_graph_llm_max_tokens,
        },
        "reliability": {
            "timeout_s": settings.docling_graph_llm_timeout,
        },
        "context_limit": settings.docling_graph_llm_context_limit,
        "max_output_tokens": settings.docling_graph_llm_max_output_tokens,
    }
    ...
```

Note: the gen knobs (`top_p`, `top_k`, etc.) referenced inside `get_docling_llm_client()` may not exist on `DoclingGraphSettings` today. Task 3.1b uses `getattr(settings, "docling_graph_llm_top_p", None)` so missing fields fall through to `None` (and the chat body skips them). Add real fields to `DoclingGraphSettings` only if you actually want to control them — not required for v1.

- [ ] **Step 3: Build and restart.**

```bash
docker compose build docling-graph
docker compose up -d --force-recreate docling-graph
```

- [ ] **Step 4: Verify the new client wires in.**

```bash
docker logs eip-mmdpp-docling-graph-1 2>&1 | tail -30
# Expect: no errors at startup; the [LlmBackend] Initialized with: line
# should show "OllamaChatClient" instead of "LiteLLMClient".
```

`docling_graph/pipeline/stages.py:470` is unambiguous: `if context.config.llm_client is not None: llm_client = context.config.llm_client else: llm_client = self._initialize_llm_client(...)`. So passing `llm_client=` skips LiteLLM construction entirely; `provider_override` / `model_override` are only consulted on the else branch. They become vestigial when `llm_client` is set, but leaving them is harmless. If the log still shows LiteLLMClient anyway, escalate — something in the library wiring changed.

- [ ] **Step 5: Commit. Stage docker-compose.yml alongside config_builder.py — Step 0 added the env passthrough and Step 1+2 modified the settings/builder.**

```bash
git add docker-compose.yml docker/docling-graph/app/config_builder.py
git commit -m "refactor(docling-graph): inject OllamaChatClient via PipelineConfig"
```

### Task 3.2b: Add `/debug/routing-metrics` endpoint to docling-graph FastAPI

**Files:**
- Modify: `docker/docling-graph/app/main.py`

The Gate 5 validation in Chunk 4 needs to query the running uvicorn process's routing-metrics state. A `docker exec ... python -c ...` shortcut spawns a fresh Python process that imports the factory fresh and sees an empty cache. The endpoint reads the live in-memory state.

- [ ] **Step 1: Add the endpoint to `main.py`.**

Locate the FastAPI app definition (line ~476: `app = FastAPI(title="Docling-Graph Extraction Service", ...)`). Add below the lifespan + app construction:

```python
from app.ollama_clients import get_docling_llm_client


@app.get("/debug/routing-metrics", tags=["diagnostics"])
def debug_routing_metrics():
    """Return per-URL request counts for the in-process LLM pool.

    Gated behind DOCLING_GRAPH_DEBUG_ENDPOINTS=true (default off) — this
    endpoint exposes backend Ollama URLs in its response, which is a small
    leak on a port that's published in compose. Enable only when running
    Gate 5 validation (Chunk 4).

    Used by Chunk 4's Gate 5 to verify fan-out across all configured Ollama
    URLs. Returns {} for the LLM role only in v1; VLM/embedding pools
    aren't used inside docling-graph.
    """
    import os
    from fastapi import HTTPException
    if os.environ.get("DOCLING_GRAPH_DEBUG_ENDPOINTS", "false").lower() not in (
        "true", "1", "yes", "on"
    ):
        raise HTTPException(status_code=404, detail="Not Found")
    try:
        client = get_docling_llm_client()
        return {"llm": client.pool.routing_metrics}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
```

To enable for Gate 5 validation, set in `.env`:
```bash
DOCLING_GRAPH_DEBUG_ENDPOINTS=true
```
And add to `docker-compose.yml`'s docling-graph environment block:
```yaml
DOCLING_GRAPH_DEBUG_ENDPOINTS: ${DOCLING_GRAPH_DEBUG_ENDPOINTS:-false}
```
After Gate 5 passes, set back to `false` (or unset) before any production deployment.

- [ ] **Step 2a: Restart docling-graph (flag default OFF) and confirm the endpoint is gated.**

```bash
docker compose up -d --force-recreate docling-graph
sleep 3
curl -s -o /dev/null -w "%{http_code}\n" \
  "http://localhost:${DOCLING_GRAPH_PORT:-8002}/debug/routing-metrics"
```

Expected: `404` — the endpoint should be invisible by default to avoid leaking backend URLs on the published port.

- [ ] **Step 2b: Enable the flag, restart, and confirm the endpoint returns metrics.**

```bash
DOCLING_GRAPH_DEBUG_ENDPOINTS=true docker compose up -d --force-recreate docling-graph
sleep 3
curl -s "http://localhost:${DOCLING_GRAPH_PORT:-8002}/debug/routing-metrics" | python3 -m json.tool
```

Expected: `{"llm": {"http://...": 0, ...}}` (zeros initially since no calls yet, but every URL in the pool appears).

- [ ] **Step 2c: Disable the flag again before continuing** (so we don't leave a dev-only endpoint exposed during the rest of the chunk).

```bash
docker compose up -d --force-recreate docling-graph    # falls back to .env, where the flag is unset/false
```

- [ ] **Step 3: Commit. Stage `docker-compose.yml` alongside `main.py` — Step 1 added the compose env-passthrough line.**

```bash
git add docker/docling-graph/app/main.py docker-compose.yml
git commit -m "feat(docling-graph): /debug/routing-metrics diagnostic endpoint (gated)"
```

### Task 3.3: Restart all services + sanity-check

- [ ] **Step 1: Recreate api + workers + jupyter so they pick up the new config.**

```bash
docker compose up -d --force-recreate api worker worker-graph worker-ingest worker-embed beat
docker compose -f docker-compose.jupyter.yml up -d --force-recreate
```

- [ ] **Step 2: Confirm api/worker import paths work.**

```bash
docker exec eip-mmdpp-api-1 python -c "
from app.services.ollama_clients import get_llm_client, get_vlm_client, get_embedding_client
print('llm urls:', get_llm_client().pool.urls)
print('vlm urls:', get_vlm_client().pool.urls)
print('emb urls:', get_embedding_client().pool.urls)
"
```
Expected: prints three URL lists; no ImportError.

- [ ] **Step 3: Commit any final adjustments.**

---

## Chunk 4: End-to-end validation

This is the quality gate. Re-ingest a known document and verify the extraction pipeline still produces the right output, with no LiteLLM in the call path.

### Task 4.1: End-to-end reingest of radar PDF

- [ ] **Step 1: Pick a doc.**

Use `Microsoft Word - lf99-4297_final-combineddoc.doc - 20080014230.pdf` or any radar PDF still in the DB (per session history). Confirm `document_id`:

```bash
docker exec eip-mmdpp-api-1 python -c "
from app.db.session import get_sync_session
from app.models.ingest import Document
db = get_sync_session()
docs = db.query(Document).filter(Document.original_filename.ilike('%radar%')).limit(5).all()
for d in docs:
    print(f'{d.id}\t{d.original_filename}\t{d.created_at}')
"
```

- [ ] **Step 2: Trigger reingest in graph_only mode.**

```bash
DOC_ID=...    # from Step 1
curl -s -X POST "http://localhost:${API_PORT:-8003}/v1/documents/$DOC_ID/reingest?mode=graph_only" \
  | python3 -m json.tool
```

- [ ] **Step 3: Watch docling-graph logs for errors.**

```bash
docker logs eip-mmdpp-docling-graph-1 -f 2>&1 | grep -iE "error|fail|warn|stripped|delta_batch" &
# Wait until the run finishes or until logs stabilize
```

Look for:
- ✅ Successful `Extracted: ...` lines
- ❌ ZERO `Structured output failed for ... falling back to legacy prompt-schema mode` warnings
- ❌ ZERO `LiteLLMClient JSON parse failed` errors
- ✅ The new `OllamaChatClient: stripped %d-char schema embedding ...` log line MAY appear if sparse-result detection fired and the retry came back through `get_json_response(structured_output=False)` — that is OK, it's the in-client strip working as intended (replaces the old patch). Zero firings is also fine and more common

- [ ] **Step 4: Compare output to the pre-refactor baseline.**

Query the graph for the radar's extracted fields:

```bash
AUTH="root:eip_arcadedb_secret"
curl -s -u "$AUTH" -X POST http://localhost:2480/api/v1/query/eip_knowledge_graph \
  -H "Content-Type: application/json" \
  -d '{"language":"sql","command":"SELECT system_name, peak_power_kw, antenna_gain_dbi, frequency_min_mhz, frequency_max_mhz FROM RADAR_SYSTEM ORDER BY @rid DESC LIMIT 5"}' \
  | python3 -m json.tool
```

Expected: at least the `system_name` populates; `peak_power_kw`/`antenna_gain_dbi`/`frequency_min_mhz` populate when present in the source. (Spot-check against expected values from session history's known-good runs.)

If output is degraded relative to baseline: hand back to user; do NOT proceed to delete the `_get_json_response` patch.

### Task 4.2: Document the new env shape

- [ ] **Step 1: Update `README.md` (or wherever the project docs the LLM env vars) with a short section on `OLLAMA_*_BASE_URLS`.**

Find the existing "LLM configuration" section and add:

> **Pool URLs (NEW):** Each role (LLM / VLM / embedding) accepts a JSON array of Ollama base URLs via `OLLAMA_LLM_BASE_URLS`, `OLLAMA_VLM_BASE_URLS`, `OLLAMA_EMBEDDING_BASE_URLS`. When set, the corresponding pool load-balances using least-in-flight routing (one retry on a different instance for connection/timeout errors). The singular `OLLAMA_*_BASE_URL` vars are still honored when the plural is unset.

- [ ] **Step 2: Commit.**

```bash
git add README.md
git commit -m "docs(readme): document OLLAMA_*_BASE_URLS pool vars"
```

### Task 4.3: Validation gate — DECISION POINT

This is the explicit gate before Chunk 5. Do NOT proceed unless ALL of the following are true on the Task 4.1 reingest run:

- [ ] **Gate 1:** Extraction completed without errors (no exceptions in worker-graph or docling-graph logs).
- [ ] **Gate 2:** Output graph quality matches pre-refactor baseline (Task 4.1 Step 4 spot-check passes — known-good fields populate as before).
- [ ] **Gate 3:** Zero `LiteLLMClient JSON parse failed` errors in docling-graph logs (the new client doesn't go through LiteLLM at all, so any such log line means our wiring is wrong).
- [ ] **Gate 4:** Zero `Structured output failed for ... falling back to legacy prompt-schema mode` warnings — OR if any did fire, the in-client `OllamaChatClient: stripped %d-char schema embedding from legacy retry prompt` log followed and the retry succeeded. This proves the in-client strip handles sparse-result detection correctly (replacing what `_patched_get_json_response` used to do).
- [ ] **Gate 5:** Pool routing fanned out across all configured URLs (if `OLLAMA_LLM_BASE_URLS` has >1 entry). Use the `GET /debug/routing-metrics` endpoint added to the docling-graph FastAPI app in Task 3.2b.

    The endpoint is gated behind `DOCLING_GRAPH_DEBUG_ENDPOINTS=true` so it returns 404 by default (it leaks backend URLs). Before running Gate 5:

    ```bash
    # Enable temporarily for Gate 5 validation only. (Note: `docker compose up`
    # does NOT accept `-e`; that flag only exists on `run`/`exec`. Use shell
    # env interpolation instead — compose interpolates ${VAR} from the shell
    # at config-load time.)
    DOCLING_GRAPH_DEBUG_ENDPOINTS=true docker compose up -d --force-recreate docling-graph
    # OR: edit .env to set DOCLING_GRAPH_DEBUG_ENDPOINTS=true,
    # restart docling-graph, then unset after validation passes.
    ```

    Then query:

    ```bash
    curl -s http://localhost:${DOCLING_GRAPH_PORT:-8002}/debug/routing-metrics | python3 -m json.tool
    # Expected output:
    # {
    #   "llm": {
    #     "http://10.0.1.121:11434": 412,
    #     "http://10.0.1.122:11434": 405,
    #     "http://10.0.1.123:11434": 398,
    #     ...
    #   }
    # }
    ```

    A `docker exec ... python -c "..."` shortcut DOES NOT WORK here: it spawns a fresh Python process that imports the factory module fresh and gets `served={}` — it can't observe the running uvicorn process's module state. The HTTP endpoint queries the in-memory state of the live process, which is what we need.

    Expected: every URL in `OLLAMA_LLM_BASE_URLS` appears in `routing_metrics` with a non-zero count. If any URL has zero, fan-out is broken (cursor wedged, exclude set never released, etc.) — investigate before Chunk 5.

    If `OLLAMA_LLM_BASE_URLS` has only 1 URL, Gate 5 is automatically satisfied (single-URL pool always picks it).

If all 5 gates pass: proceed to Chunk 5. If any fail: stop, fix the root cause, re-run Task 4.1, re-evaluate. Do NOT delete the LiteLLM patches while gates are red — they're our rollback path.

---

## Chunk 5: Delete LiteLLM patches AND `_get_json_response` patch (POST-VALIDATION)

This chunk runs only after Chunk 4's Gate 5 is green. End state: `docker/docling-graph/app/main.py` no longer references `LiteLLMClient`, `_patched_build_request`, `_patched_call_api`, or `_patched_get_json_response`.

The `_get_json_response` strip patch is now redundant because the schema-strip logic lives inside `OllamaChatClient.get_json_response` (via `_maybe_strip_legacy_schema`). When `LlmBackend` triggers the sparse-result legacy retry by calling `self.client.get_json_response(..., structured_output=False)`, our client's `_maybe_strip_legacy_schema` does the strip in-process — no monkey-patch needed.

The `NodeIDRegistry` patch stays (unrelated to LLM calls; fixes a class-name parsing bug in the library's collision detection).

### Task 5.1: Delete LiteLLM patches from main.py

**Files:**
- Modify: `docker/docling-graph/app/main.py:45-296`

- [ ] **Step 1: Delete the LiteLLMClient and `_get_json_response` patch blocks.**

In `_apply_litellm_client_patches()` (line ~288), remove:
1. The try/except block that imports `LiteLLMClient` and assigns `LiteLLMClient._build_request` / `LiteLLMClient._call_api`.
2. The try/except block that imports `LlmBackend` and assigns `LlmBackend._get_json_response = _patched_get_json_response`.

Delete the function definitions at module level:
- `_patched_build_request` (lines ~65–220)
- `_patched_call_api` (lines ~210–290)
- `_patched_get_json_response` (lines ~310–360)

KEEP:
- The `_logger` setup at the top (still used elsewhere for diagnostic logs)
- The `NodeIDRegistry` patch (unrelated to LLM calls)
- The `_apply_litellm_client_patches` function itself, but reduce it to just the NodeIDRegistry block. (Or rename it to `_apply_node_id_registry_patch` — the litellm name is misleading once that work is gone.)

Update the file's top-of-file comment block (lines 45–60) to reflect the new state:

```python
# We bypass LiteLLM entirely via PipelineConfig(llm_client=OllamaChatClient(...))
# (see config_builder.build_pipeline_config). The OllamaChatClient also
# absorbs the legacy-fallback schema-strip behavior in-process, so no
# upstream LlmBackend patches are required for that either.
#
# Only one upstream patch remains:
#   - NodeIDRegistry: fixes a class-name parsing bug in the library's
#     collision detection (TABLE_REF_<fp> being misread as TABLE).
#     Unrelated to LLM call paths.
```

- [ ] **Step 2: Build, restart docling-graph.**

```bash
docker compose build docling-graph
docker compose up -d --force-recreate docling-graph
```

- [ ] **Step 3: Verify the patch logs.**

```bash
docker logs eip-mmdpp-docling-graph-1 2>&1 | grep -iE "patched|patch installed"
```

Expected: exactly one line — `NodeIDRegistry patched ...`. Both `LiteLLMClient patched ...` and `LlmBackend._get_json_response patched ...` MUST be gone (deleted in Step 1).

- [ ] **Step 4: Commit.**

```bash
git add docker/docling-graph/app/main.py
git commit -m "refactor(docling-graph): drop LiteLLM patches (replaced by OllamaChatClient)"
```

### Task 5.2: Re-run end-to-end validation

- [ ] **Step 1: Re-run Task 4.1 against the same document.** Use the same gate criteria as Task 4.3.

- [ ] **Step 2: If gates pass, push.** If any gate fails, revert the patch-deletion commit and investigate.

```bash
git push
```

---

## Chunk 6: Per-function URL pools (POST-CHUNK-5; user-requested 2026-04-29)

This chunk extends the role-level pools (`OLLAMA_LLM_BASE_URLS` / `OLLAMA_VLM_BASE_URLS` / `OLLAMA_EMBEDDING_BASE_URLS`) with per-function granularity so each LLM-using function can route to its own bank of Ollama instances. Use case: doc analysis runs `gpt-oss:120b` on one host, graph extraction runs `gemma4:31b` on another bank, translation runs `llama3.3:70b` on a third. Each function uses a different model, so pinning each function to the host that has its model loaded is the natural split.

The routing core (`OllamaPool`, `OllamaChatClient`, `OllamaEmbeddingClient`) is unchanged — Chunk 6 just multiplies the number of pool instances using the same plumbing.

### Cascade (4-tier, additive — all existing config keeps working)

```
DOCLING_GRAPH_LLM_BASE_URLS  →  OLLAMA_LLM_BASE_URLS  →  OLLAMA_LLM_BASE_URL  →  OLLAMA_BASE_URL
DOC_ANALYSIS_LLM_BASE_URLS   →  OLLAMA_LLM_BASE_URLS  →  OLLAMA_LLM_BASE_URL  →  OLLAMA_BASE_URL
TRANSLATION_LLM_BASE_URLS    →  OLLAMA_LLM_BASE_URLS  →  OLLAMA_LLM_BASE_URL  →  OLLAMA_BASE_URL
COMMUNITY_REPORT_LLM_BASE_URLS → OLLAMA_LLM_BASE_URLS →  OLLAMA_LLM_BASE_URL  →  OLLAMA_BASE_URL
PICTURE_DESCRIPTION_BASE_URLS  → OLLAMA_VLM_BASE_URLS →  OLLAMA_VLM_BASE_URL  →  OLLAMA_BASE_URL
TEXT_EMBEDDING_BASE_URLS       → OLLAMA_EMBEDDING_BASE_URLS → OLLAMA_EMBEDDING_BASE_URL → OLLAMA_BASE_URL
```

(Plus `_synthesize_global_answer` — uses the community-report config since it's part of the global query strategy.)

### Pre-flight checklist

- [ ] **P0: Confirm baseline test suite status.**

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline tests/test_pool_client_mirror.py -q 2>&1 | tail -3`
  Expected: `1399 passed, 3 failed, 3 skipped, 3 xfailed` (the 3 baseline failures are pre-existing: `test_docling_graph_client_defaults`, `test_default_quality_min_instances_is_three`, `test_system_name_description_excludes_forbidden_tokens[radar_identity]`). Don't regress this.

- [ ] **P1: Audit ALL callers of the role-level factories AND the docling-graph factory.**

  The role-level names (`get_llm_client`, `get_vlm_client`, `get_embedding_client`) and the docling-graph name (`get_docling_llm_client`) are ALL being renamed/removed. Catch every reference, including module-qualified ones (`ollama_clients.get_llm_client(...)`) and string references in tests/fixtures:

  ```bash
  cd /home/josh/development/EIP-MMDPP

  echo "=== Production callers (must be migrated in Task 6.3 / 6.4) ==="
  rg -nE 'get_(llm|vlm|embedding)_client\b|get_docling_llm_client\b|ollama_clients\.get_' \
    app docker --type py | grep -v 'ollama_clients\.py' | grep -v 'test_' | grep -v '\.tasks\.json'

  echo
  echo "=== Test patch targets / mock paths (must be updated in Task 6.3 Step 7) ==="
  rg -nE 'get_(llm|vlm|embedding)_client\b|get_docling_llm_client\b' tests --type py
  ```

  Expected production matches (capture exact list before starting Task 6.3):
  - `app/services/document_analysis.py` — 2 call sites
  - `app/services/translation.py` — 1
  - `app/services/arcadedb_community.py` — 1
  - `app/api/v1/retrieval.py` — 1 (`_synthesize_global_answer`)
  - `app/services/embedding.py` — 1
  - `docker/docling-graph/app/main.py` — 1 (`get_docling_llm_client`)
  - `docker/docling-graph/app/config_builder.py` — 1 (`get_docling_llm_client`)

  Plus an unknown number of test patch targets (mock-path strings like `'app.services.document_analysis.get_llm_client'`) — typically 5-10 across the test suite.

  If any new caller has been added since this plan was written, ADD it to your task scope and migrate it. Anything you miss will fail loudly at next process boot since Task 6.2 deliberately removes the old factory names (no shims).

- [ ] **P2: Confirm stack is up.**

  Run: `docker compose ps --format "table {{.Service}}\t{{.Status}}" | grep -E "api|arcadedb|postgres|worker-graph|docling-graph"` — all 5 must be `Up (healthy)`.

### Task 6.1: api-side Settings — add 5 per-function URL fields + 5 helpers with 4-tier cascade

(Note: this task covers the **5 api-side** functions. The 6th per-function pool — `DOCLING_GRAPH_LLM_BASE_URLS` for docling-graph extraction — is added in **Task 6.4** because it lives in the docling-graph container's `DoclingGraphSettings`, not the api-side `Settings`. Cascade pattern is identical between the two.)

**Files:**
- Modify: `app/config.py` (add fields next to existing `ollama_llm_base_urls`; add helpers next to existing `get_ollama_*_urls()`)
- Create: `tests/unit/test_config_per_function_pools.py`

- [ ] **Step 1: Write failing test for per-function cascade priority.**

  Create `tests/unit/test_config_per_function_pools.py`:

  ```python
  """Test the per-function plural URL env vars and their 4-tier cascade.

  Cascade priority (function-specific > role-level > singular > base):
    DOCLING_GRAPH_LLM_BASE_URLS    > OLLAMA_LLM_BASE_URLS       > OLLAMA_LLM_BASE_URL       > OLLAMA_BASE_URL
    DOC_ANALYSIS_LLM_BASE_URLS     > OLLAMA_LLM_BASE_URLS       > OLLAMA_LLM_BASE_URL       > OLLAMA_BASE_URL
    TRANSLATION_LLM_BASE_URLS      > OLLAMA_LLM_BASE_URLS       > OLLAMA_LLM_BASE_URL       > OLLAMA_BASE_URL
    COMMUNITY_REPORT_LLM_BASE_URLS > OLLAMA_LLM_BASE_URLS       > OLLAMA_LLM_BASE_URL       > OLLAMA_BASE_URL
    PICTURE_DESCRIPTION_BASE_URLS  > OLLAMA_VLM_BASE_URLS       > OLLAMA_VLM_BASE_URL       > OLLAMA_BASE_URL
    TEXT_EMBEDDING_BASE_URLS       > OLLAMA_EMBEDDING_BASE_URLS > OLLAMA_EMBEDDING_BASE_URL > OLLAMA_BASE_URL
  """
  import pytest

  from app.config import Settings


  def _build_settings(monkeypatch, **env: str) -> Settings:
      """Construct a fresh Settings without polluting the LRU-cached singleton."""
      for k, v in env.items():
          monkeypatch.setenv(k, v)
      return Settings(_env_file=None)


  # ----- Function-specific overrides win over role-level -----

  def test_doc_analysis_function_pool_overrides_role(monkeypatch):
      s = _build_settings(
          monkeypatch,
          OLLAMA_BASE_URL="http://base:11434",
          OLLAMA_LLM_BASE_URLS='["http://role-1:11434"]',
          DOC_ANALYSIS_LLM_BASE_URLS='["http://gpt-oss-host-1:11434","http://gpt-oss-host-2:11434"]',
      )
      assert s.get_doc_analysis_llm_urls() == [
          "http://gpt-oss-host-1:11434",
          "http://gpt-oss-host-2:11434",
      ]


  def test_translation_function_pool_overrides_role(monkeypatch):
      s = _build_settings(
          monkeypatch,
          OLLAMA_LLM_BASE_URLS='["http://role-host:11434"]',
          TRANSLATION_LLM_BASE_URLS='["http://llama-host:11434"]',
      )
      assert s.get_translation_llm_urls() == ["http://llama-host:11434"]


  def test_community_report_function_pool_overrides_role(monkeypatch):
      s = _build_settings(
          monkeypatch,
          OLLAMA_LLM_BASE_URLS='["http://role-host:11434"]',
          COMMUNITY_REPORT_LLM_BASE_URLS='["http://gpt-oss-host:11434"]',
      )
      assert s.get_community_report_llm_urls() == ["http://gpt-oss-host:11434"]


  def test_picture_description_function_pool_overrides_role(monkeypatch):
      s = _build_settings(
          monkeypatch,
          OLLAMA_VLM_BASE_URLS='["http://role-host:11434"]',
          PICTURE_DESCRIPTION_BASE_URLS='["http://gemma-vlm:11434"]',
      )
      assert s.get_picture_description_urls() == ["http://gemma-vlm:11434"]


  def test_text_embedding_function_pool_overrides_role(monkeypatch):
      s = _build_settings(
          monkeypatch,
          OLLAMA_EMBEDDING_BASE_URLS='["http://role-host:11434"]',
          TEXT_EMBEDDING_BASE_URLS='["http://bge-host:11434"]',
      )
      assert s.get_text_embedding_urls() == ["http://bge-host:11434"]


  # ----- Role-level pools serve as fallback when function-specific is empty -----

  def test_function_falls_back_to_role_level(monkeypatch):
      s = _build_settings(
          monkeypatch,
          OLLAMA_BASE_URL="http://base:11434",
          OLLAMA_LLM_BASE_URLS='["http://role-1:11434","http://role-2:11434"]',
          DOC_ANALYSIS_LLM_BASE_URLS="",  # blank → fall through to role
      )
      assert s.get_doc_analysis_llm_urls() == [
          "http://role-1:11434",
          "http://role-2:11434",
      ]


  # ----- Singular fallback when both plurals are empty -----

  def test_function_falls_back_through_singular(monkeypatch):
      s = _build_settings(
          monkeypatch,
          OLLAMA_BASE_URL="http://base:11434",
          OLLAMA_LLM_BASE_URL="http://singular:11434",
          OLLAMA_LLM_BASE_URLS="",
          TRANSLATION_LLM_BASE_URLS="",
      )
      assert s.get_translation_llm_urls() == ["http://singular:11434"]


  # ----- Base URL when nothing else is set -----

  def test_function_falls_back_to_base(monkeypatch):
      s = _build_settings(
          monkeypatch,
          OLLAMA_BASE_URL="http://base:11434",
          OLLAMA_LLM_BASE_URL="",
          OLLAMA_LLM_BASE_URLS="",
          COMMUNITY_REPORT_LLM_BASE_URLS="",
      )
      assert s.get_community_report_llm_urls() == ["http://base:11434"]


  # ----- Role isolation: chat-role function vs vlm-role function vs embedding-role function -----

  def test_picture_description_uses_vlm_cascade_not_llm(monkeypatch):
      """PICTURE_DESCRIPTION must cascade through OLLAMA_VLM_*, NOT OLLAMA_LLM_*."""
      s = _build_settings(
          monkeypatch,
          OLLAMA_BASE_URL="http://base:11434",
          OLLAMA_LLM_BASE_URLS='["http://llm-pool:11434"]',
          OLLAMA_VLM_BASE_URLS='["http://vlm-pool:11434"]',
          PICTURE_DESCRIPTION_BASE_URLS="",
      )
      assert s.get_picture_description_urls() == ["http://vlm-pool:11434"]


  def test_text_embedding_uses_embedding_cascade_not_llm(monkeypatch):
      s = _build_settings(
          monkeypatch,
          OLLAMA_LLM_BASE_URLS='["http://llm-pool:11434"]',
          OLLAMA_EMBEDDING_BASE_URLS='["http://embed-pool:11434"]',
          TEXT_EMBEDDING_BASE_URLS="",
      )
      assert s.get_text_embedding_urls() == ["http://embed-pool:11434"]


  # ----- Malformed / blank-entry rejection (parity with role-level) -----

  def test_malformed_function_pool_raises_at_read(monkeypatch):
      s = _build_settings(
          monkeypatch,
          DOC_ANALYSIS_LLM_BASE_URLS='["http://h1',  # unclosed
      )
      with pytest.raises(ValueError, match="not valid JSON"):
          s.get_doc_analysis_llm_urls()


  def test_blank_entry_in_function_pool_raises(monkeypatch):
      s = _build_settings(
          monkeypatch,
          TEXT_EMBEDDING_BASE_URLS='["http://h1:11434",""]',
      )
      with pytest.raises(ValueError, match="contains blank entries"):
          s.get_text_embedding_urls()
  ```

- [ ] **Step 2: Run, confirm fail with `AttributeError: 'Settings' object has no attribute 'doc_analysis_llm_base_urls'`.**

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_config_per_function_pools.py -q`

- [ ] **Step 3: Modify `app/config.py` — add fields and helpers.**

  Locate the existing block:
  ```python
      # Plural pools — JSON arrays as raw strings, e.g.
      # '["http://host1:11434","http://host2:11434"]'.
      ollama_llm_base_urls: str = ""
      ollama_vlm_base_urls: str = ""
      ollama_embedding_base_urls: str = ""
  ```

  Add immediately after:
  ```python
      # Per-function URL pools (Chunk 6, NEW). Each function gets its own
      # pool so the operator can route different LLM functions to different
      # banks of Ollama instances. When set, function-specific takes
      # precedence over role-level pools above. Cascade depth is 4-tier
      # for chat functions and embedding/vlm:
      #   DOC_ANALYSIS_LLM_BASE_URLS     > OLLAMA_LLM_BASE_URLS > OLLAMA_LLM_BASE_URL > OLLAMA_BASE_URL
      #   TRANSLATION_LLM_BASE_URLS      > OLLAMA_LLM_BASE_URLS > OLLAMA_LLM_BASE_URL > OLLAMA_BASE_URL
      #   COMMUNITY_REPORT_LLM_BASE_URLS > OLLAMA_LLM_BASE_URLS > OLLAMA_LLM_BASE_URL > OLLAMA_BASE_URL
      #   PICTURE_DESCRIPTION_BASE_URLS  > OLLAMA_VLM_BASE_URLS > OLLAMA_VLM_BASE_URL > OLLAMA_BASE_URL
      #   TEXT_EMBEDDING_BASE_URLS       > OLLAMA_EMBEDDING_BASE_URLS > OLLAMA_EMBEDDING_BASE_URL > OLLAMA_BASE_URL
      doc_analysis_llm_base_urls: str = ""
      translation_llm_base_urls: str = ""
      community_report_llm_base_urls: str = ""
      picture_description_base_urls: str = ""
      text_embedding_base_urls: str = ""
  ```

  And after the existing `get_ollama_*_urls()` methods, add:

  ```python
      def get_doc_analysis_llm_urls(self) -> list[str]:
          plural = self._parse_url_pool(self.doc_analysis_llm_base_urls)
          if plural:
              return plural
          return self.get_ollama_llm_urls()

      def get_translation_llm_urls(self) -> list[str]:
          plural = self._parse_url_pool(self.translation_llm_base_urls)
          if plural:
              return plural
          return self.get_ollama_llm_urls()

      def get_community_report_llm_urls(self) -> list[str]:
          plural = self._parse_url_pool(self.community_report_llm_base_urls)
          if plural:
              return plural
          return self.get_ollama_llm_urls()

      def get_picture_description_urls(self) -> list[str]:
          plural = self._parse_url_pool(self.picture_description_base_urls)
          if plural:
              return plural
          return self.get_ollama_vlm_urls()

      def get_text_embedding_urls(self) -> list[str]:
          plural = self._parse_url_pool(self.text_embedding_base_urls)
          if plural:
              return plural
          return self.get_ollama_embedding_urls()
  ```

  Each function-specific helper falls through to its corresponding role-level helper via composition — that's how the 4-tier cascade is implemented without duplicating the singular/base fallback logic.

- [ ] **Step 4: Run, confirm pass.**

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_config_per_function_pools.py tests/unit/test_config_ollama_pools.py -q`
  Expected: all tests in both files pass.

- [ ] **Step 5: Commit.**

  ```bash
  git add app/config.py tests/unit/test_config_per_function_pools.py
  git commit -m "feat(config): per-function URL pool env vars w/ 4-tier cascade"
  ```

### Task 6.2: Per-function client factories in `app/services/ollama_clients.py`

**Files:**
- Modify: `app/services/ollama_clients.py` (replace 3 role-level factories with 5 per-function factories)
- Modify: `tests/unit/services/test_ollama_clients_factory.py` (update tests for new factory names)

- [ ] **Step 1: Write failing tests for per-function factories.**

  Replace the body of `tests/unit/services/test_ollama_clients_factory.py` with:

  ```python
  """Tests for the per-function cached factory module.

  Each factory is @lru_cache(maxsize=1) so each process has at most one
  client per function. Tests clear caches around env mutation to stay
  hermetic.
  """
  import pytest

  from app.config import get_settings
  from app.services import ollama_clients


  @pytest.fixture(autouse=True)
  def _clear_caches():
      """Clear all factory caches before AND after each test so polluted
      singletons don't leak across tests in the file."""
      def _clear():
          get_settings.cache_clear()
          ollama_clients.get_doc_analysis_client.cache_clear()
          ollama_clients.get_translation_client.cache_clear()
          ollama_clients.get_community_report_client.cache_clear()
          ollama_clients.get_picture_description_client.cache_clear()
          ollama_clients.get_text_embedding_client.cache_clear()
      _clear()
      yield
      _clear()


  def test_each_factory_is_cached_singleton():
      c1 = ollama_clients.get_doc_analysis_client()
      c2 = ollama_clients.get_doc_analysis_client()
      assert c1 is c2


  def test_each_factory_uses_function_specific_pool(monkeypatch):
      monkeypatch.setenv("DOC_ANALYSIS_LLM_BASE_URLS", '["http://da:11434"]')
      monkeypatch.setenv("TRANSLATION_LLM_BASE_URLS", '["http://tr:11434"]')
      monkeypatch.setenv("COMMUNITY_REPORT_LLM_BASE_URLS", '["http://cr:11434"]')
      monkeypatch.setenv("PICTURE_DESCRIPTION_BASE_URLS", '["http://pd:11434"]')
      monkeypatch.setenv("TEXT_EMBEDDING_BASE_URLS", '["http://te:11434"]')

      assert ollama_clients.get_doc_analysis_client().pool.urls == ["http://da:11434"]
      assert ollama_clients.get_translation_client().pool.urls == ["http://tr:11434"]
      assert ollama_clients.get_community_report_client().pool.urls == ["http://cr:11434"]
      assert ollama_clients.get_picture_description_client().pool.urls == ["http://pd:11434"]
      assert ollama_clients.get_text_embedding_client().pool.urls == ["http://te:11434"]


  def test_factories_pin_to_role_specific_models(monkeypatch):
      """Per-function factories use the role's model setting at construction."""
      monkeypatch.setenv("DOC_ANALYSIS_LLM_MODEL", "gpt-oss:120b")
      monkeypatch.setenv("TRANSLATION_MODEL", "llama3.3:70b")
      monkeypatch.setenv("COMMUNITY_REPORT_LLM_MODEL", "llama3.2")
      monkeypatch.setenv("PICTURE_DESCRIPTION_MODEL", "gemma3:27b")
      monkeypatch.setenv("TEXT_EMBEDDING_MODEL", "bge-m3:latest")
      monkeypatch.setenv("OLLAMA_LLM_BASE_URLS", '["http://h:11434"]')
      monkeypatch.setenv("OLLAMA_VLM_BASE_URLS", '["http://h:11434"]')
      monkeypatch.setenv("OLLAMA_EMBEDDING_BASE_URLS", '["http://h:11434"]')

      assert ollama_clients.get_doc_analysis_client().model == "gpt-oss:120b"
      assert ollama_clients.get_translation_client().model == "llama3.3:70b"
      assert ollama_clients.get_community_report_client().model == "llama3.2"
      assert ollama_clients.get_picture_description_client().model == "gemma3:27b"
      assert ollama_clients.get_text_embedding_client().model == "bge-m3:latest"
  ```

- [ ] **Step 2: Run, confirm fail with `AttributeError`.**

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/services/test_ollama_clients_factory.py -q`

- [ ] **Step 3: Replace `app/services/ollama_clients.py` body.**

  Replace the file with:

  ```python
  """Process-cached per-function Ollama client factories.

  Each LLM-using function in the system gets its own factory that builds
  a dedicated `OllamaChatClient` (chat/VLM) or `OllamaEmbeddingClient`
  (embedding) backed by a function-specific URL pool from `Settings`.
  This lets the operator route different LLM functions to different banks
  of Ollama instances — e.g. doc analysis on a gpt-oss:120b host, graph
  extraction on a gemma4:31b bank, embeddings on a CPU node.

  Each factory is @lru_cache(maxsize=1); the first call constructs the
  client, subsequent calls return it. Tests must call `<factory>.cache_clear()`
  to rebuild against patched env. Env values frozen at first call:
  function-specific URLs, role-level URLs, model name, timeout, and any
  per-function `*_THINK` setting.
  """
  from __future__ import annotations

  from functools import lru_cache

  from app.config import get_settings
  from app.services.ollama_pool_client import (
      OllamaChatClient, OllamaEmbeddingClient, OllamaPool,
  )


  @lru_cache(maxsize=1)
  def get_doc_analysis_client() -> OllamaChatClient:
      s = get_settings()
      return OllamaChatClient(
          pool=OllamaPool(urls=s.get_doc_analysis_llm_urls()),
          model=s.doc_analysis_llm_model,
          timeout_s=float(s.doc_analysis_timeout),
          max_tokens=s.llm_max_tokens,
      )


  @lru_cache(maxsize=1)
  def get_translation_client() -> OllamaChatClient:
      s = get_settings()
      return OllamaChatClient(
          pool=OllamaPool(urls=s.get_translation_llm_urls()),
          model=s.translation_model,
          timeout_s=float(s.translation_timeout),
          max_tokens=s.llm_max_tokens,
      )


  @lru_cache(maxsize=1)
  def get_community_report_client() -> OllamaChatClient:
      """Used by both community-report generation and global-query synthesis
      (the latter is part of the global-query strategy, which uses the
      same model + timeout settings).
      """
      s = get_settings()
      return OllamaChatClient(
          pool=OllamaPool(urls=s.get_community_report_llm_urls()),
          model=s.community_report_llm_model,
          timeout_s=float(s.doc_analysis_timeout),  # historical reuse
          max_tokens=s.llm_max_tokens,
      )


  @lru_cache(maxsize=1)
  def get_picture_description_client() -> OllamaChatClient:
      s = get_settings()
      return OllamaChatClient(
          pool=OllamaPool(urls=s.get_picture_description_urls()),
          model=s.picture_description_model,
          timeout_s=float(s.picture_description_timeout),
          max_tokens=s.llm_max_tokens,
      )


  @lru_cache(maxsize=1)
  def get_text_embedding_client() -> OllamaEmbeddingClient:
      s = get_settings()
      # Embeddings are fast; 120s covers worst-case batch.
      return OllamaEmbeddingClient(
          pool=OllamaPool(urls=s.get_text_embedding_urls()),
          model=s.text_embedding_model,
          timeout_s=120.0,
      )
  ```

  **The role-level factories (`get_llm_client`, `get_vlm_client`, `get_embedding_client`) are GONE.** Don't keep deprecated shims — they would mask any caller we missed in Task 6.3. Forced removal makes any stale import fail loudly at import time, which is what we want.

- [ ] **Step 4: Run, confirm tests pass.**

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/services/test_ollama_clients_factory.py -q`

- [ ] **Step 5: Commit.**

  ```bash
  git add app/services/ollama_clients.py tests/unit/services/test_ollama_clients_factory.py
  git commit -m "feat(ollama-pool): per-function factories replace role-level shared clients"
  ```

### Task 6.3: Migrate every call site to its function-specific factory

**Files:**
- Modify: `app/services/document_analysis.py` (2 sites: `extract_document_metadata`, `_describe_single_image`)
- Modify: `app/services/translation.py` (1 site: `_ollama_translate`)
- Modify: `app/services/arcadedb_community.py` (1 site: `_call_llm_for_report`)
- Modify: `app/api/v1/retrieval.py` (1 site: `_synthesize_global_answer`)
- Modify: `app/services/embedding.py` (1 site: `embed_texts`)

- [ ] **Step 1: Re-run the comprehensive audit from the Pre-flight checklist (P1) to confirm the scope, including module-qualified calls and `get_docling_llm_client`.**

  ```bash
  cd /home/josh/development/EIP-MMDPP

  echo "=== Production callers (must be migrated) ==="
  rg -nE 'get_(llm|vlm|embedding)_client\b|get_docling_llm_client\b|ollama_clients\.get_' \
    app docker --type py | grep -v 'ollama_clients\.py' | grep -v 'test_' | grep -v '\.tasks\.json'

  echo
  echo "=== Test patch targets (must be updated in Step 7) ==="
  rg -nE 'get_(llm|vlm|embedding)_client\b|get_docling_llm_client\b' tests --type py
  ```

  Each production match must be migrated in this task. Each test match must be updated in Step 7's sweep. The narrower `from app.services.ollama_clients import ...` grep used in earlier plan revisions misses module-qualified calls (`ollama_clients.get_llm_client(...)`), patch-target strings, and `get_docling_llm_client` — use the `rg` form above. If you find a new caller not enumerated in P1, ADD it to your task scope and report it back.

- [ ] **Step 2: Migrate `app/services/document_analysis.py`.**

  Replace `from app.services.ollama_clients import get_llm_client` with `from app.services.ollama_clients import get_doc_analysis_client` and update the call inside `extract_document_metadata` from `client = get_llm_client()` to `client = get_doc_analysis_client()`.

  In `_describe_single_image`, replace `from app.services.ollama_clients import get_vlm_client` with `from app.services.ollama_clients import get_picture_description_client` and `client = get_vlm_client()` → `client = get_picture_description_client()`.

  The `chat()` calls inside both functions stay the same (model passed per-call, timeout per-call) — only the factory import changes.

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/test_document_analysis.py -q`
  Expected: all pre-existing tests still pass. Mock targets in the test file may need updating from `get_llm_client` / `get_vlm_client` to the new function-specific names — fix any mock paths that break.

- [ ] **Step 3: Migrate `app/services/translation.py`.**

  Replace `from app.services.ollama_clients import get_llm_client` with `from app.services.ollama_clients import get_translation_client` and update `_ollama_translate` to use it.

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/services/test_translation.py tests/pipeline -k translation -q`
  Expected: same passing count as baseline.

- [ ] **Step 4: Migrate `app/services/arcadedb_community.py`.**

  Replace `from app.services.ollama_clients import get_llm_client` with `from app.services.ollama_clients import get_community_report_client` in `_call_llm_for_report`. Update the cached client variable.

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit -k community -q`

- [ ] **Step 5: Migrate `app/api/v1/retrieval.py::_synthesize_global_answer`.**

  Replace `from app.services.ollama_clients import get_llm_client` with `from app.services.ollama_clients import get_community_report_client`. Update the call. (Global synthesis is part of the global query strategy; it shares the community-report config.)

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit -k retrieval -q`

- [ ] **Step 6: Migrate `app/services/embedding.py`.**

  Replace `from app.services.ollama_clients import get_embedding_client` with `from app.services.ollama_clients import get_text_embedding_client`. Update the call inside `embed_texts`.

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit -k embedding -q`

- [ ] **Step 7: Sweep the test suite for stale `get_llm_client` / `get_vlm_client` / `get_embedding_client` patch targets.**

  Many test files mock the pool client at the import path used by the production code under test. Examples that are known stale today:

  - `tests/unit/test_document_analysis.py:14` patches `app.services.document_analysis.get_llm_client`
  - `tests/unit/test_arcadedb_community.py:290` patches `app.services.arcadedb_community.get_llm_client`

  After Task 6.2 deletes the old factories, EVERY mock-patch string referencing them MUST be updated to the new function-specific name. Run a comprehensive sweep:

  ```bash
  cd /home/josh/development/EIP-MMDPP
  rg -nE 'get_(llm|vlm|embedding)_client\b' tests --type py
  ```

  **For each match: patch the import path actually looked up by the code AFTER migration**, not whatever path was patched before. The right path depends on where the import landed in the production code:

  - If the production code imports the factory at module scope (e.g. `from app.services.ollama_clients import get_doc_analysis_client` at the top of `document_analysis.py`), patch `app.services.document_analysis.get_doc_analysis_client`.
  - If the production code imports the factory **inside a function** (e.g. `def extract(): from app.services.ollama_clients import get_doc_analysis_client; ...`), the local lookup binds in the source module — patch `app.services.ollama_clients.get_doc_analysis_client` instead.

  Look at where the migration actually placed the import (Task 6.3 Steps 2-6), THEN update each test's patch path to match. Both styles work; consistency between production and test is what matters. If you change the import location during migration, update the test in the same commit.

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit tests/pipeline tests/test_pool_client_mirror.py -q 2>&1 | tail -3`
  Expected: same `1399+ passed, 3 failed (pre-existing), 3 skipped, 3 xfailed` baseline. No NEW failures.

- [ ] **Step 8: Commit each migration as its own commit. Push after each.**

  Five migration commits + one test-sweep commit (six total):

  1. `refactor(doc-analysis): use per-function get_doc_analysis_client + get_picture_description_client` (covers `app/services/document_analysis.py` — both sites — plus the patch updates in `tests/unit/test_document_analysis.py`)
  2. `refactor(translation): use get_translation_client` (covers `app/services/translation.py` plus its mock patches)
  3. `refactor(community): use get_community_report_client` (covers `app/services/arcadedb_community.py` plus mocks in `tests/unit/test_arcadedb_community.py`)
  4. `refactor(retrieval): use get_community_report_client for global synthesis` (covers `app/api/v1/retrieval.py::_synthesize_global_answer`)
  5. `refactor(embedding): use get_text_embedding_client` (covers `app/services/embedding.py` plus its mocks)
  6. `chore(tests): sweep remaining patch targets for new per-function factory names` (only if Step 7's grep surfaces patch sites NOT covered by commits 1-5)

  If commit #6 ends up empty (Steps 1-5 caught all the patches in their respective files), drop it.

### Task 6.4: Mirror into docling-graph + add `DOCLING_GRAPH_LLM_BASE_URLS`

**Files:**
- Modify: `docker/docling-graph/app/config_builder.py` (extend `DoclingGraphSettings.get_ollama_llm_urls` cascade)
- Modify: `docker/docling-graph/app/ollama_clients.py` (rename `get_docling_llm_client` to clarify it's the graph-extraction-specific factory; or leave the name and just update the URL source)
- Modify: `docker-compose.yml` (add `DOCLING_GRAPH_LLM_BASE_URLS` passthrough to docling-graph environment block)
- Modify: `docker/docling-graph/app/main.py` (the `get_docling_llm_client` import reference, if renamed)

- [ ] **Step 1: Add `docling_graph_llm_base_urls` field + extend cascade in `DoclingGraphSettings`.**

  In `docker/docling-graph/app/config_builder.py`, locate the existing `ollama_llm_base_urls: str = ""` field. Add immediately after:

  ```python
      # Per-function pool for graph extraction (Chunk 6, NEW). When set,
      # overrides OLLAMA_LLM_BASE_URLS for THIS service only — other
      # LLM-using functions (doc analysis, translation, etc.) consume
      # their own *_LLM_BASE_URLS via the api-side Settings.
      docling_graph_llm_base_urls: str = ""
  ```

  Update `get_ollama_llm_urls()` to honor the new field at the top of the cascade:

  ```python
      def get_ollama_llm_urls(self) -> list[str]:
          """Parse priority (4-tier):
            docling_graph_llm_base_urls (function-specific JSON)
            > ollama_llm_base_urls (role-level JSON)
            > ollama_llm_base_url (singular)
            > ollama_base_url (base).
          Always returns a non-empty list.
          """
          import json
          for raw in (self.docling_graph_llm_base_urls, self.ollama_llm_base_urls):
              s = (raw or "").strip()
              if not s:
                  continue
              try:
                  parsed = json.loads(s)
              except json.JSONDecodeError as exc:
                  raise ValueError(
                      f"Pool URL env var is not valid JSON: {exc}; got: {s!r}"
                  ) from exc
              if not isinstance(parsed, list) or not all(
                  isinstance(x, str) for x in parsed
              ):
                  raise ValueError(
                      f"Pool URL env var must be a JSON array of strings; got: {parsed!r}"
                  )
              if not all(x.strip() for x in parsed):
                  raise ValueError(
                      f"Pool URL env var contains blank entries; got: {parsed!r}"
                  )
              if parsed:
                  return parsed
          if self.ollama_llm_base_url:
              return [self.ollama_llm_base_url]
          return [self.ollama_base_url]
  ```

- [ ] **Step 2: Add the env passthrough to `docker-compose.yml`.**

  Locate the docling-graph `environment:` block (search for `OLLAMA_LLM_BASE_URLS:` near line 173 — added in Chunk 3). Add directly below:

  ```yaml
        # Per-function pool for graph extraction (Chunk 6). When set, overrides
        # OLLAMA_LLM_BASE_URLS for this service only.
        DOCLING_GRAPH_LLM_BASE_URLS: ${DOCLING_GRAPH_LLM_BASE_URLS:-}
  ```

- [ ] **Step 3: Rename `get_docling_llm_client` to `get_docling_graph_client` for naming consistency with the api-side per-function factories. Update its single call site in `main.py`.**

  Edit `docker/docling-graph/app/ollama_clients.py`:
  - Rename `get_docling_llm_client` → `get_docling_graph_client` (function-specific name aligns with `get_doc_analysis_client`, `get_translation_client`, etc.)

  Edit `docker/docling-graph/app/main.py`:
  - `from app.ollama_clients import get_docling_llm_client` → `from app.ollama_clients import get_docling_graph_client`
  - `client = get_docling_llm_client()` → `client = get_docling_graph_client()`

  Edit `docker/docling-graph/app/config_builder.py`:
  - `from app.ollama_clients import get_docling_llm_client` → `from app.ollama_clients import get_docling_graph_client`
  - `llm_client: Any | None = get_docling_llm_client()` → `llm_client: Any | None = get_docling_graph_client()`

- [ ] **Step 4: Rebuild and restart docling-graph.**

  ```bash
  docker compose build docling-graph
  DOCLING_GRAPH_DEBUG_ENDPOINTS=true docker compose up -d --force-recreate docling-graph
  sleep 5
  docker logs eip-mmdpp-docling-graph-1 --tail 20 2>&1 | grep -E "patched|ERROR|Started"
  curl -s http://localhost:${DOCLING_GRAPH_PORT:-8002}/debug/routing-metrics | python3 -m json.tool
  ```

  Expected: `NodeIDRegistry patched ...` log line, no errors, debug endpoint returns the new pool's URLs (zeros initially).

- [ ] **Step 5: Verify the rename inside the running docling-graph container.**

  `tests/unit/test_docling_graph_quality_config.py` runs in the host venv and `build_pipeline_config` swallows the `ImportError` from `app.ollama_clients` (see `config_builder.py:160`-ish), so that host-side test will NOT prove the renamed `get_docling_graph_client` symbol exists or works inside the docling-graph package. The test is still useful (it confirms `build_pipeline_config` still constructs a valid config dict in the host venv), but it's not a substitute for in-container verification.

  Run BOTH:

  ```bash
  # 1. Host-venv unit suite — config-shape validation (lazy import path)
  SKIP_COV=1 .venv/bin/pytest tests/unit/test_docling_graph_quality_config.py -q
  ```

  Expected: same passing count as baseline.

  ```bash
  # 2. Container smoke check — proves the rename inside the container
  docker exec eip-mmdpp-docling-graph-1 python -c "
  from app.ollama_clients import get_docling_graph_client
  client = get_docling_graph_client()
  print('model:', client.model)
  print('pool urls:', client.pool.urls)
  print('routing_metrics:', client.pool.routing_metrics)
  print('OK')
  "
  ```

  Expected: prints the configured model name, the list of pool URLs, and `routing_metrics` as a dict keyed by those same URLs with zero counts (e.g. `{'http://10.0.1.121:11434': 0, 'http://10.0.1.109:11434': 0}`), then `OK`. If `ImportError: cannot import name 'get_docling_graph_client'` — the rename didn't land in the container; rebuild docling-graph and retry.

  ```bash
  # 3. Confirm the OLD name is gone (in-container)
  docker exec eip-mmdpp-docling-graph-1 python -c "
  try:
      from app.ollama_clients import get_docling_llm_client
      print('FAIL: old name still importable')
  except ImportError:
      print('OK: old name removed')
  "
  ```

  Expected: `OK: old name removed`. Catches the case where the rename added the new symbol but left the old one as a vestigial alias.

- [ ] **Step 6: Commit (one commit covering all four files since they're tightly coupled by the rename).**

  ```bash
  git add docker/docling-graph/app/config_builder.py docker/docling-graph/app/ollama_clients.py docker/docling-graph/app/main.py docker-compose.yml
  git commit -m "feat(docling-graph): DOCLING_GRAPH_LLM_BASE_URLS pool var w/ per-function factory"
  ```

### Task 6.5: Document new env vars in `env.example` + `README.md`

**Files:**
- Modify: `env.example`
- Modify: `README.md`

- [ ] **Step 1: Add new plural vars to `env.example` near the existing pool URLs.**

  After the existing `OLLAMA_LLM_BASE_URLS=` / `OLLAMA_VLM_BASE_URLS=` / `OLLAMA_EMBEDDING_BASE_URLS=` lines, add:

  ```bash
  # Per-function URL pools (NEW). When set, override the role-level
  # OLLAMA_*_BASE_URLS above for that specific function. Use this to route
  # different LLM functions to different banks of Ollama instances — e.g.
  # doc analysis on a gpt-oss:120b host, graph extraction on a gemma4:31b
  # bank, translation on a llama3.3:70b host.
  # Format: JSON array of base URLs.
  # Example: ["http://10.0.1.121:11434","http://10.0.1.122:11434"]
  DOCLING_GRAPH_LLM_BASE_URLS=
  DOC_ANALYSIS_LLM_BASE_URLS=
  TRANSLATION_LLM_BASE_URLS=
  COMMUNITY_REPORT_LLM_BASE_URLS=
  PICTURE_DESCRIPTION_BASE_URLS=
  TEXT_EMBEDDING_BASE_URLS=
  ```

- [ ] **Step 2: Extend the README's "Pool URLs (NEW)" section with the per-function table.**

  Find the existing Pool URLs section (added in Chunk 4 Task 4.2). After the existing role-level table, add:

  ```markdown
  #### Per-function pool URLs (NEW in Chunk 6)

  For finer-grained control, each LLM-using function can specify its own pool. When set, the function-specific pool overrides the role-level pool above.

  | Function Variable                 | Falls back to (in order)                                                                       |
  |-----------------------------------|------------------------------------------------------------------------------------------------|
  | `DOCLING_GRAPH_LLM_BASE_URLS`     | `OLLAMA_LLM_BASE_URLS` → `OLLAMA_LLM_BASE_URL` → `OLLAMA_BASE_URL`                             |
  | `DOC_ANALYSIS_LLM_BASE_URLS`      | `OLLAMA_LLM_BASE_URLS` → `OLLAMA_LLM_BASE_URL` → `OLLAMA_BASE_URL`                             |
  | `TRANSLATION_LLM_BASE_URLS`       | `OLLAMA_LLM_BASE_URLS` → `OLLAMA_LLM_BASE_URL` → `OLLAMA_BASE_URL`                             |
  | `COMMUNITY_REPORT_LLM_BASE_URLS`  | `OLLAMA_LLM_BASE_URLS` → `OLLAMA_LLM_BASE_URL` → `OLLAMA_BASE_URL` (also used by global-query) |
  | `PICTURE_DESCRIPTION_BASE_URLS`   | `OLLAMA_VLM_BASE_URLS` → `OLLAMA_VLM_BASE_URL` → `OLLAMA_BASE_URL`                             |
  | `TEXT_EMBEDDING_BASE_URLS`        | `OLLAMA_EMBEDDING_BASE_URLS` → `OLLAMA_EMBEDDING_BASE_URL` → `OLLAMA_BASE_URL`                 |

  Common patterns:

  ```bash
  # Pattern 1: single bank for everything (default — leave the per-function vars empty)
  OLLAMA_LLM_BASE_URLS=["http://10.0.1.121:11434","http://10.0.1.122:11434"]

  # Pattern 2: graph extraction on bank A (gemma4:31b), other chat functions on bank B (gpt-oss:120b)
  DOCLING_GRAPH_LLM_BASE_URLS=["http://gemma-host-1:11434","http://gemma-host-2:11434"]
  OLLAMA_LLM_BASE_URLS=["http://gpt-oss-host-1:11434","http://gpt-oss-host-2:11434"]

  # Pattern 3: every function pinned to its own host
  DOCLING_GRAPH_LLM_BASE_URLS=["http://gemma-bank:11434"]
  DOC_ANALYSIS_LLM_BASE_URLS=["http://gpt-oss:11434"]
  TRANSLATION_LLM_BASE_URLS=["http://llama:11434"]
  COMMUNITY_REPORT_LLM_BASE_URLS=["http://gpt-oss:11434"]
  PICTURE_DESCRIPTION_BASE_URLS=["http://gemma-vlm:11434"]
  TEXT_EMBEDDING_BASE_URLS=["http://bge-host:11434"]
  ```
  ```

- [ ] **Step 3: Commit.**

  ```bash
  git add env.example README.md
  git commit -m "docs: document per-function URL pools (DOCLING_GRAPH_LLM_BASE_URLS et al.)"
  ```

### Task 6.5b: Add success-URL logging to client (api-side observability for Task 6.6)

The current client only logs URLs at WARNING level (on retry/failure) — there's no INFO-level proof of which URL handled a successful call. The api-side `/debug/routing-metrics` endpoint doesn't exist (only docling-graph has it), and adding a Redis-backed counter is overkill for v1. Cheapest reliable fix: add a single INFO log line per successful chat/embedding call so worker-graph and worker-embed logs show exactly which URL got each request. This makes Gates 6.C / 6.D in Task 6.6 grep-verifiable.

**Files:**
- Modify: `app/services/ollama_pool_client.py` (canonical) — add INFO log on success in both `OllamaChatClient._post_chat_with_retry` and `OllamaEmbeddingClient.embed`
- Modify: `docker/docling-graph/app/ollama_pool_client.py` (mirror — mirror-drift test enforces parity)
- Modify: `tests/unit/services/test_ollama_pool_client.py` (assert the log line fires on success)

- [ ] **Step 1: Add a focused unit test for the success log.**

  ```python
  def test_chat_logs_success_url_at_info(caplog):
      pool = OllamaPool(urls=["http://only"])
      client = OllamaChatClient(pool=pool, model="m")
      fake = MagicMock()
      fake.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
      fake.raise_for_status.return_value = None
      with caplog.at_level("INFO", logger="app.services.ollama_pool_client"):
          with patch("httpx.Client.post", return_value=fake):
              client.chat(messages=[{"role": "user", "content": "x"}])
      assert any(
          "OllamaChatClient: ok" in rec.message and "http://only" in rec.message
          for rec in caplog.records
      ), [r.message for r in caplog.records]


  def test_embedding_logs_success_url_at_info(caplog):
      pool = OllamaPool(urls=["http://only"])
      client = OllamaEmbeddingClient(pool=pool, model="bge-m3")
      fake = MagicMock()
      fake.json.return_value = {"data": [{"index": 0, "embedding": [0.1]}]}
      fake.raise_for_status.return_value = None
      with caplog.at_level("INFO", logger="app.services.ollama_pool_client"):
          with patch("httpx.Client.post", return_value=fake):
              client.embed(["hello"])
      assert any(
          "OllamaEmbeddingClient: ok" in rec.message and "http://only" in rec.message
          for rec in caplog.records
      ), [r.message for r in caplog.records]
  ```

- [ ] **Step 2: Confirm fail.**

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/services/test_ollama_pool_client.py::test_chat_logs_success_url_at_info tests/unit/services/test_ollama_pool_client.py::test_embedding_logs_success_url_at_info -q`

- [ ] **Step 3: Implement.**

  In `OllamaChatClient._post_chat_with_retry` (the success branch — locate the existing `logger.debug("OllamaChatClient: ok ...")` line, which exists today after Chunk 1's cleanup pass), **replace that DEBUG line entirely with the INFO version below** (don't keep both — duplicate logs would clutter the worker output and could double-count if a future tool grep-counts them). The INFO version reads `body["model"]` so per-call model overrides via `chat(model=...)` are visible in the log (the grep filters in Gates 6.C/6.D rely on this):

  ```python
  logger.info(
      "OllamaChatClient: ok url=%s model=%s elapsed=%.2fs len(content)=%d",
      url, body.get("model", self.model), time.time() - t0, len(content),
  )
  ```

  In `OllamaEmbeddingClient.embed`, after the successful `resp.json()` parse and just before `return [item["embedding"] for item in items]`, add:

  ```python
  logger.info(
      "OllamaEmbeddingClient: ok url=%s model=%s batch_size=%d elapsed=%.2fs",
      url, self.model, len(texts), time.time() - t0,
  )
  ```

  (The embedding client doesn't have a per-call `model=` override today — `self.model` is sufficient. If we add per-call model override later, mirror the chat-client pattern.)

  (Ensure `t0 = time.time()` is captured at the start of each retry attempt.)

  Both files (canonical + docling-graph mirror) must change identically below the marker. The mirror-drift test at `tests/test_pool_client_mirror.py` enforces this.

- [ ] **Step 4: Confirm pass.**

  Run: `SKIP_COV=1 .venv/bin/pytest tests/unit/services/test_ollama_pool_client.py tests/test_pool_client_mirror.py -q`

- [ ] **Step 5: Commit.**

  ```bash
  git add app/services/ollama_pool_client.py docker/docling-graph/app/ollama_pool_client.py tests/unit/services/test_ollama_pool_client.py
  git commit -m "feat(ollama-pool): INFO-log per-call success URL for routing observability"
  ```

### Task 6.6: End-to-end validation — heterogeneous config

**Files:**
- None modified — this is a validation pass. (Task 6.5b above adds the observability hooks needed to make Gates 6.C / 6.D grep-verifiable.)

- [ ] **Step 1: Set up a heterogeneous config in `.env`.**

  Pick TWO Ollama URLs that both have `gemma4:31b`, `gpt-oss:120b` (or whatever doc-analysis model you choose), and `bge-m3:latest` available (the hosts used in Chunk 4 — `10.0.1.121` and `10.0.1.109` — qualify if loaded). Set the URL pools AND **distinct models per function** so the Gate 6.C / 6.D log greps disambiguate cleanly:

  ```bash
  # --- Pool URLs ---
  # Role-level fallback — doc analysis, translation, community inherit from this
  OLLAMA_LLM_BASE_URLS=["http://10.0.1.121:11434"]

  # Function-specific — graph extraction routed to a 2-URL pool (THE test)
  DOCLING_GRAPH_LLM_BASE_URLS=["http://10.0.1.121:11434","http://10.0.1.109:11434"]

  # Embedding stays on a single URL (per user policy: not 121)
  OLLAMA_EMBEDDING_BASE_URLS=["http://10.0.1.109:11434"]

  # Picture-description VLM cascade — fall through to OLLAMA_VLM_BASE_URLS
  OLLAMA_VLM_BASE_URLS=["http://10.0.1.121:11434"]

  # --- Models (distinct per function so log greps disambiguate) ---
  DOC_ANALYSIS_LLM_MODEL=gpt-oss:120b
  DOCLING_GRAPH_LLM_MODEL=gemma4:31b
  TRANSLATION_MODEL=llama3.3:70b
  COMMUNITY_REPORT_LLM_MODEL=llama3.2
  PICTURE_DESCRIPTION_MODEL=gemma3:27b
  TEXT_EMBEDDING_MODEL=bge-m3:latest
  ```

  If the chosen models aren't loaded on both `10.0.1.121` and `10.0.1.109`, either pull them first (`curl -X POST <host>/api/pull -d '{"name":"<model>"}'`) OR pick different distinct-model assignments. The validation only requires that doc-analysis and graph-extraction use DIFFERENT models so the worker-graph log filter in Gate 6.C is unambiguous.

- [ ] **Step 2: Restart all services so each reads the new env.**

  ```bash
  DOCLING_GRAPH_DEBUG_ENDPOINTS=true docker compose up -d --force-recreate docling-graph api worker worker-graph worker-ingest worker-embed
  sleep 10
  ```

- [ ] **Step 3: Trigger a fresh reingest of the SNR-75 PDF (used in Chunks 4-5 validation).**

  ```bash
  DOC_ID=4db44228-62b4-4930-a367-1398b3cd05b9    # SNR-75 - Wikipedia.pdf
  curl -s -X POST "http://localhost:${API_PORT:-8003}/v1/documents/$DOC_ID/reingest?mode=graph_only" | python3 -m json.tool
  ```

- [ ] **Step 4: Watch the run; when it completes, evaluate the heterogeneous-config gates.**

  Same gates as Chunk 4 (1: zero errors, 2: graph populated, 3: zero LiteLLM parse fails, 4: zero unhandled legacy fallbacks, 5: routing fan-out across all configured docling-graph URLs):

  ```bash
  curl -s http://localhost:${DOCLING_GRAPH_PORT:-8002}/debug/routing-metrics | python3 -m json.tool
  # Expected: BOTH "http://10.0.1.121:11434" and "http://10.0.1.109:11434" with non-zero counts.
  ```

  PLUS Chunk-6-specific gates:

- [ ] **Gate 6.A:** docling-graph routing-metrics shows fan-out within the **graph-extraction** pool (both URLs incremented) — i.e., `DOCLING_GRAPH_LLM_BASE_URLS` is being honored.

- [ ] **Gate 6.B:** No api-side function (doc analysis, translation, community, embedding) leaked into the docling-graph routing-metrics. The `/debug/routing-metrics` endpoint reports only the docling-graph LLM pool, so this gate is mostly self-evident; double-check by inspecting the endpoint's response shape and ensuring it shows ONLY `"llm"` (not `"vlm"` or `"embedding"` keys).

- [ ] **Gate 6.C:** Doc analysis routed to `10.0.1.121` only (inherited from `OLLAMA_LLM_BASE_URLS` since `DOC_ANALYSIS_LLM_BASE_URLS` is unset in this test). Verifiable thanks to Task 6.5b's success log including `model=`.

  **Choose a doc-analysis model that's distinct from the graph-extraction model in the test config** so the log lines disambiguate cleanly. For this validation, set `DOC_ANALYSIS_LLM_MODEL=gpt-oss:120b` in `.env` while `DOCLING_GRAPH_LLM_MODEL=gemma4:31b` (the production split) — the heterogeneous-config test step in Task 6.6 Step 1 already implies different models per function.

  Then:

  ```bash
  # All doc-analysis calls during this run (filter by model)
  docker logs eip-mmdpp-worker-graph-1 --since 60m 2>&1 \
      | grep "OllamaChatClient: ok" \
      | grep "model=gpt-oss:120b" \
      | grep -oE "url=http://[^ ]+" | sort -u
  ```

  Expected: only `url=http://10.0.1.121:11434`. If `10.0.1.109` appears, doc analysis is leaking into the graph-extraction pool — investigate.

  If your test config uses the SAME model for doc analysis and graph extraction (rare but possible), narrow by time window instead. The `extraction-status` API on the document gives you exact `started_at`/`finished_at` timestamps for the `derive_document_metadata` stage — grep with `--since <minutes>` set to the metadata stage's window only.

- [ ] **Gate 6.D:** Embedding ran on `10.0.1.109` only. Verifiable via Task 6.5b's success log on `OllamaEmbeddingClient`:

  ```bash
  docker logs eip-mmdpp-worker-embed-1 --since 60m 2>&1 \
      | grep "OllamaEmbeddingClient: ok" | grep -oE "url=http://[^ ]+" | sort -u
  ```

  Expected: only `url=http://10.0.1.109:11434`. Bonus check via httpx's own INFO logging:

  ```bash
  docker logs eip-mmdpp-worker-embed-1 --since 60m 2>&1 \
      | grep -E "POST http://[^ ]+/v1/embeddings" | grep -oE "http://[^ ]+/v1/embeddings" | sort -u
  ```

  Same expectation — only `http://10.0.1.109:11434/v1/embeddings`. If both greps return zero matches, the worker may not be running embeddings yet (run not started or stuck).

- [ ] **Step 5: Disable the debug endpoint after validation passes.**

  Unset `DOCLING_GRAPH_DEBUG_ENDPOINTS` in the running shell; restart docling-graph without the flag so production posture is restored.

  ```bash
  docker compose up -d --force-recreate docling-graph
  curl -s -o /dev/null -w "%{http_code}\n" http://localhost:${DOCLING_GRAPH_PORT:-8002}/debug/routing-metrics
  # Expected: 404
  ```

- [ ] **Step 6: Commit any documentation updates surfaced during validation; close out Chunk 6.**

  No new code commits expected from Step 6. If any small fix lands during validation (e.g., a typo in a comment or a missed env var), commit it separately with `chore(ollama-pool): post-validation fix — <description>`.

### Files Chunk 6 must NOT touch (routing-behavior-wise)

- `app/services/ollama_pool_client.py` — **routing behavior** unchanged. The only allowed edit is the logging-only change in Task 6.5b (promote success log to INFO, add `url=` field). No other class/method/signature edits.
- `app/config.py:_parse_url_pool` — reused as-is (only NEW helpers and fields are added; `_parse_url_pool` itself is not modified).
- `docker/docling-graph/app/ollama_pool_client.py` and `llm_json.py` — must stay byte-for-byte synced with their canonicals below the `# === SHARED CODE BELOW THIS LINE ===` marker. Task 6.5b's logging edit MUST be applied to both files identically; the mirror-drift CI test enforces this.

### Estimated effort

Single implementer, ~1 day. Most work is mechanical (settings expansion + factory rename + 6 import-line swaps). Validation in Task 6.6 is the longest single step (full reingest, ~30-60 min).

### Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Removed role-level factories (`get_llm_client` etc.) leave a stale import somewhere we missed | Task 6.3 Step 1 (re-runs the comprehensive P1 audit, including module-qualified calls and `get_docling_llm_client`) enumerates all callers; we migrate every match. Anything missed fails loudly at import time on next worker boot since Task 6.2 deletes the old names with no shim. |
| Mock paths in unit tests reference `get_llm_client` / `get_vlm_client` / `get_embedding_client` and break after the rename | Task 6.3 Step 2 calls this out explicitly; fix mock paths as part of each per-file migration commit. |
| `DoclingGraphSettings.get_ollama_llm_urls` cascade gets out of sync with `Settings.get_*_llm_urls` cascades | Both are tested separately. Keep the JSON-parsing logic mirrored exactly between `_parse_url_pool` (api-side) and the inline parser (docling-graph-side). |
| docling-graph rename of `get_docling_llm_client` → `get_docling_graph_client` breaks notebooks | The `raw_libraries_walkthrough.ipynb` constructs its own `OllamaChatClient` directly (doesn't use the factory) — verified during Chunk 5 notebook update. No notebook should be affected. |

---

## Out of scope (deferred to follow-up)

- **Image embedding (OpenCLIP)** — runs locally; no Ollama, no pooling needed.
- **Reranker (CrossEncoder)** — runs locally; no Ollama.
- **Health-tracking / circuit-breaker** on instances — YAGNI; least-in-flight + one-retry handles the dominant failure modes. If we observe one instance going hot-dead in practice, add health-tracking then.
- **Per-instance model differences** — the pool assumes a homogeneous bank (same model on every URL). If you ever need a heterogeneous pool, this becomes a real design question.
- **Async-native pool** — community/global-query callers wrap sync calls in `asyncio.to_thread`. If we observe contention on the worker thread pool, switch the chat client to `httpx.AsyncClient`. Not now.

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Library auto-builds a LiteLLMClient even when `llm_client` is passed (e.g., from `provider_override` / `model_override`) | Task 3.2 Step 4 verifies via log inspection. If it does, drop the overrides. |
| `format=<schema>` Ollama JSON-Schema mode produces "Unterminated string" on long generations (observed before with gemma4:31b) | This is a model + schema problem, not a transport problem. The pool client doesn't help here. If we hit it, raise it as a separate workstream — possibly fall back to `format="json"` only and rely on retry-on-parse-failure. |
| Two-copy mirror drifts | `tests/test_pool_client_mirror.py` (Task 3.1) catches this in CI. |
| Pydantic-settings raises `SettingsError` on blank-string for `list[str]` (verified locally) | Plural URL fields stored as raw `str`; JSON parsing happens in `Settings._parse_url_pool` at read time. Validator approach abandoned in v4 because pydantic-settings raises during source-decode, before any field validator runs. |
| `lru_cache` factory returns a stale client when env changes (e.g., URL rotation in production) | Process restart picks up new env. Not a concern in v1; documented as a known limitation. Production rotates Ollama via DNS or LB, not by mutating .env. |
| Existing `_ollama_chat` import in translation.py becomes dangling after Task 2.3 deletes it from doc_analysis.py | Task 2.4 Step 1 explicitly rewrites the translation call site to not import the deleted helper. Verified by running unit suite at end of Chunk 2. |
