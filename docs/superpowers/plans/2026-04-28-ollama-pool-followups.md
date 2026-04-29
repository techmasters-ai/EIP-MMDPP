# OllamaPool Refactor — Cleanup Follow-ups

Running tally of NIT-level findings from each chunk's code-quality review. Address as a single post-implementation cleanup pass after Chunk 5 lands.

**Source plan:** `docs/superpowers/plans/2026-04-28-ollama-pool-client-refactor.md`

---

## From Chunk 1 (commits `0b87101..fa91705`)

### Code (`app/services/ollama_pool_client.py`)

1. **`_post_chat_with_retry` retry-after-acquire-fails comment.** The `if len(excluded) >= len(self.pool.urls): break` guard prevents `pool.acquire()` from raising on attempt 2, but the safety is implicit. Add a one-line comment near the break: `# Guards acquire() from raising on attempt 2 — must run before next iteration.`

2. **`__del__` swallows all exceptions.** `OllamaChatClient.__del__` and `OllamaEmbeddingClient.__del__` both run `self._http.close()` inside `except Exception: pass`, hiding shutdown errors and leaking connections silently. Expose a real `close()` method, recommend `contextlib.closing()`/`with` usage. Keep `__del__` as safety net but log at DEBUG when it fires.

3. **Re-import of `Callable, Optional`** inside the file body (~line 119) — already imported at top (~line 27). Trivial: delete the duplicate.

4. **Magic-string `"json"` literal (4 occurrences).** Extract `_FORMAT_JSON = "json"` as a module-level constant near the top; replace the four string literals. Makes intent self-documenting and grep-friendly when Ollama adds new format modes.

5. **`OllamaEmbeddingClient.embed` has no `last_call_diagnostics`.** Asymmetric with `OllamaChatClient` (which populates rich diagnostics for debug). Either add a minimal diagnostics dict (URL, elapsed_s, batch_size) or document why embeddings don't need it.

### Config (`app/config.py`)

6. **`Settings._parse_url_pool` has a local `import json`** at line ~131. `json` is stdlib used elsewhere; hoist to module top.

7. **`_parse_url_pool` doesn't warn on duplicate URLs.** `OllamaPool.__init__` silently dedupes (good defense). But operators who set `OLLAMA_LLM_BASE_URLS='["http://h1","http://h1"]'` get no signal that their config is wrong. Add `logger.warning` on dupes at parse time.

### Tests

8. **Mid-file imports in `test_ollama_pool_client.py`** (lines 84, 119, 188, 546). `OllamaChatClient`, `OllamaEmbeddingClient`, `_FakeClientError` — hoist to top of file per PEP 8.

9. **Smoke harness slack assertions over-permissive.** `tests/smoke/ollama_pool_routing.py:63-69` accepts `counts[0] >= 2` and `counts[-1] <= 8` for what should be ~4 each. Tighten the comment to: "Allow [2..8] per URL; uniform service time should give ~4 each, but thread scheduling jitter on under-resourced CI can skew."

### Docs

10. **`env.example` pre-existing typo** at `OLLAMA_EMBEDDING_BASE_URL=http://ollama11434` (missing colon, should be `http://ollama:11434`). Pre-existed before this refactor — drive-by fix candidate.

---

## From Chunk 2 (commits `bde1646..f85ff04`)

### Tests

11. **`test_factories_use_role_specific_pools` doesn't restore caches on teardown.** After the test runs, `get_llm_client()`/`get_vlm_client()`/`get_embedding_client()` cached singletons hold mock URLs (`http://llm-1` etc.). `monkeypatch` restores env but not the cached instances; subsequent tests in the same session get the polluted singletons. Add a `try/finally` block (or autouse fixture) that calls `cache_clear()` on all three factories on teardown.

12. **`test_llm_client_is_cached_singleton` doesn't `cache_clear()` at the top.** If test ordering changes and the factory test runs first, this test sees a singleton populated with mock URLs. Identity check still passes but the test isn't hermetic. Add cache_clear at top.

### Code

13. **`app/services/ollama_clients.py:45` magic `timeout_s=120.0`** for the embedding client — unannotated. Other factories pull from settings. Either add a comment ("embeddings are fast; 120s covers worst-case batch") or thread an `embedding_timeout` setting through `app/config.py` for parity.

14. **`app/api/v1/retrieval.py:1191` comment references "Task 2.5"** — meaningless to future readers post-merge. Rephrase to: "reused from doc_analysis_timeout pending a dedicated community_global_synthesis_timeout setting" — same shape as the comment in `arcadedb_community.py:236-238`.

### Pre-existing (drive-by candidates)

15. **`app/services/arcadedb_community.py:155-159` reaches through `graph_store._client.command(...)`** to issue raw SQL. Pre-existing — bypasses GraphStore's interface and breaks if internal moves. Out of scope for OllamaPool refactor but worth flagging.

---

## From Chunk 3 (commits `adb0666..1815fad`)

### Regression caught during verification (already fixed)

**(Resolved in `1815fad`)** The Chunk 3 implementer missed that `from app.ollama_clients import get_docling_llm_client` works inside the docling-graph container (where `/app/app/` is the package root) but fails in the host venv where `app` resolves to the api-side package. `tests/unit/test_docling_graph_quality_config.py` calls `build_pipeline_config()` directly to verify pass-name override behavior, and 5 tests started failing with `ModuleNotFoundError`. Wrapped the import in `try/except ImportError` so the function still returns a valid config dict in the host venv (with `llm_client` unset; library falls back to its own client at runtime, which is fine for the config-shape unit tests).

**Lesson for future:** when a test calls into a container-side module that imports from a different package layout, the import must be lazy + tolerant. Spec reviewer didn't catch this because they were focused on file-shape parity, not host-venv import resolution. Worth a baseline-test-run gate after each chunk.

### Code-quality NITs

16. **`tests/test_pool_client_mirror.py:8`** — assert `text.count(_MARKER) == 1` to fail loudly on accidental marker duplication (e.g., someone adds the marker string inside a docstring example).
17. **`tests/test_pool_client_mirror.py:25`** — anchor paths via `Path(__file__).resolve().parents[1]` instead of relying on pytest invocation cwd. (Other tests in repo use the same convention; not blocking.)
18. **`tests/test_pool_client_mirror.py`** — add a negative test of `_shared_body` (string lacking marker → `AssertionError`).
19. **`docker/docling-graph/app/ollama_clients.py:17`** — `get_docling_llm_client` cache-staleness limitation undocumented. Add a "Limitations:" block in the docstring listing the values frozen at first call (`force_json_mode`, `structured_output_threshold_chars`, `DOCLING_GRAPH_LLM_THINK`, all `DoclingGraphSettings` fields) and noting `get_docling_llm_client.cache_clear()` is the test escape hatch.
20. **`docker/docling-graph/app/main.py:497`** — `/debug/routing-metrics` reads `DOCLING_GRAPH_DEBUG_ENDPOINTS` per request (good) but the cached `client.pool` URLs are frozen at process start. Add docstring note: "Pool URL list is cached per-process; restart docling-graph to refresh."
21. **`docker/docling-graph/app/main.py:495-496`** — drop dead inline `import os` / `from fastapi import HTTPException` (already imported at module top).
22. **`docker/docling-graph/app/config_builder.py:191-195`** — lazy import rationale is in a code comment but not the function docstring. Append to docstring: "When run from the host venv (unit tests), `app.ollama_clients` import is suppressed and library falls back to LiteLLMClient; in-container the import always succeeds."
23. **`docker/docling-graph/app/config_builder.py:201-202`** — vestigial `provider_override`/`model_override` left in `config_kwargs`. Remove or tag with `# TODO` referencing this followups doc.
24. **`docker/docling-graph/app/ollama_clients.py:29`** — drop trailing `# mirror of app/services/llm_json.py` import comment (the module docstring already explains the mirror; inconsistent with the `ollama_pool_client` import).

### Observability gap surfaced during Chunk 4 validation

25b. **Compose Ollama (`open-webui-stack-ollama-1`, reachable as `http://ollama:11434`) crashed mid-Chunk-5 validation** — returned `{"error":{"message":"llama runner process has terminated: %!w(<nil>)"}}` on every embedding request. Workaround applied: switched `OLLAMA_EMBEDDING_BASE_URL` from `http://ollama:11434` to `http://10.0.1.121:11434` (which has `bge-m3:latest`). Worker-embed restarted to pick up the new URL via `lru_cache.cache_clear`. Surfaces a real concern: the cached factory caches the URL pool at first call; if an operator rotates Ollama endpoints, only a worker restart reflects the change. Already documented in followups #19. The compose Ollama probably needs investigation/restart in the open-webui-stack project — but that's out of scope for this refactor.

25. **`OllamaChatClient` logger has no handler in production runtime** — `logger = logging.getLogger(__name__)` inherits root config which has no handler attached in either the api or docling-graph app processes (verified: `oc_logger.handlers=[]`, `root.handlers=[]`, `root.level=30`). So `_maybe_strip_legacy_schema`'s `logger.info("OllamaChatClient: stripped ...")` and `_post_chat_with_retry`'s `logger.warning(...)` produce no visible output. Consequence: during Chunk 4 validation we couldn't tell whether the in-client schema-strip fired during the library's legacy-fallback retry. The OLD `_get_json_response` patch (still in main.py through Chunk 4) DOES have a configured handler from Chunk 3 — its log line IS visible — but neither it nor the new client's strip logged a single line during the SNR-75 reingest, despite the library logging "Structured output failed for delta_batch_0; retrying with legacy prompt-schema mode" once. Either both strips are no-ops on this code path, or the new strip ran silently, or the OLD patch ran silently (its handler should preclude this). Either way: add explicit logging-handler configuration to the canonical `ollama_pool_client.py` (mirroring the pattern at `main.py:62-68`), so production traces actually show what the strip path is doing. Without observability we can't validate Gate 4 cleanly.

---

## From Chunk 4 (validation)

_(populated after end-to-end validation)_

---

## From Chunk 5 (LiteLLM patch deletion)

_(populated after patch deletion)_

---

## Cleanup pass plan

After Chunk 5 lands and the OllamaPool refactor is fully shipped:

1. Group items by file (consolidates touch surface)
2. One commit per file: `chore(ollama-pool): cleanup nits from refactor — <file>`
3. Push without ceremony — these are mechanical low-risk changes

Estimated effort: ~30 min single sitting, no review ceremony needed.

---

## Post-validation env config switch (user-requested 2026-04-29)

Once Chunks 4 and 5 are complete and validated, switch `.env` and `env.example` from the singular `OLLAMA_*_BASE_URL` form to the plural pool form. User explicitly requested this once the refactor is done.

### Current state (singular)
```bash
OLLAMA_LLM_BASE_URL=http://10.0.1.121:11434
OLLAMA_VLM_BASE_URL=http://ollama:11434
OLLAMA_EMBEDDING_BASE_URL=http://ollama:11434
```

### Target state (plural — JSON array form)
```bash
OLLAMA_LLM_BASE_URLS=["http://10.0.1.121:11434"]
OLLAMA_VLM_BASE_URLS=["http://10.0.1.121:11434"]
OLLAMA_EMBEDDING_BASE_URLS=["http://10.0.1.121:11434"]
```

When the user provisions the bank of 8 gemma4 instances, expand `OLLAMA_LLM_BASE_URLS` to enumerate all 8 IPs. The other roles (VLM/embedding) can stay single-URL until they need scaling.

Both forms are accepted (plural takes precedence over singular when set), but the plural is now canonical and is what we want documented in `env.example` for new users. Leave the singular vars commented out in `env.example` as legacy fallbacks.
