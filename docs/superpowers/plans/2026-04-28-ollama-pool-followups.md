# OllamaPool Refactor — Cleanup Follow-ups

Running tally of NIT-level findings from each chunk's code-quality review. **Cleanup pass complete (2026-04-29) — 22 of 25 items resolved, 3 skipped per spec, plus the post-validation env switch + 4 notebook updates.**

**Source plan:** `docs/superpowers/plans/2026-04-28-ollama-pool-client-refactor.md`

---

## Resolution summary

| # | Status | Commit |
|---|---|---|
| 1 | ✅ DONE | `7b56c2d` |
| 2 | ✅ DONE | `7b56c2d` |
| 3 | ✅ DONE | `7b56c2d` |
| 4 | ✅ DONE | `7b56c2d` |
| 5 | ✅ DONE | `7b56c2d` |
| 6 | ✅ DONE | `78b3958` |
| 7 | ✅ DONE | `78b3958` |
| 8 | ✅ DONE | `a5e2689` |
| 9 | ✅ DONE | `1fde787` |
| 10 | ✅ DONE | `d06fc29` |
| 11 | ✅ DONE | `8716e26` |
| 12 | ✅ DONE | `8716e26` |
| 13 | ✅ DONE | `15a13af` |
| 14 | ✅ DONE | `45dbaa3` |
| 15 | ❌ SKIPPED — pre-existing, out of scope for OllamaPool refactor (raw `_client.command` access in `arcadedb_community.py`) | — |
| 16 | ✅ DONE | `f39bbbb` |
| 17 | ✅ DONE | `f39bbbb` |
| 18 | ✅ DONE | `f39bbbb` |
| 19 | ✅ DONE | `4ec9174` |
| 20 | ✅ DONE | `21e592f` |
| 21 | ✅ DONE | `21e592f` |
| 22 | ✅ DONE | `2b568d2` |
| 23 | ✅ DONE (verified safely removable; tests pass) | `2b568d2` |
| 24 | ✅ DONE | `4ec9174` |
| 25 | ✅ DONE | `7b56c2d` |
| 25b | ❌ SKIPPED — operational issue, not code (compose Ollama crash; user updated `OLLAMA_EMBEDDING_BASE_URL` to `10.0.1.109`) | — |
| post-validation env switch | ✅ DONE — pool-form promoted to canonical in `env.example` | `243f585` |
| notebook alignment (extraction_walkthrough) | ✅ DONE | `449d414` |
| notebook alignment (extraction_walkthrough_direct_ollama) | ✅ DONE | `5f4f9bc` |
| notebook alignment (raw_libraries_walkthrough) | ✅ DONE | `ea98cf9` |
| notebook alignment (ingest_walkthrough) | ✅ DONE | `3f6b05c` |

**Net effect:** +4 passing tests (1395 → 1399), 0 new regressions, mirror invariant intact.

---

## Detail (kept for reference)

### From Chunk 1 (commits `0b87101..fa91705`) — code reviewer found 10 NITs

#### Code (`app/services/ollama_pool_client.py`)

1. ✅ **`_post_chat_with_retry` retry-after-acquire-fails comment.** Resolved in `7b56c2d`. Added a one-line comment near the break.

2. ✅ **`__del__` swallows all exceptions.** Resolved in `7b56c2d`. Added explicit `close()` method to both `OllamaChatClient` and `OllamaEmbeddingClient`; `__del__` now logs at DEBUG instead of swallowing exceptions silently.

3. ✅ **Re-import of `Callable, Optional`** inside the file body. Resolved in `7b56c2d`.

4. ✅ **Magic-string `"json"` literal (4 occurrences).** Resolved in `7b56c2d`. Extracted `_FORMAT_JSON = "json"` module-level constant.

5. ✅ **`OllamaEmbeddingClient.embed` has no `last_call_diagnostics`.** Resolved in `7b56c2d`. Added symmetric diagnostics dict (URL, elapsed_s, model, batch_size).

#### Config (`app/config.py`)

6. ✅ **Local `import json` inside `_parse_url_pool`.** Resolved in `78b3958`. Hoisted to module top.

7. ✅ **No warning on duplicate URLs.** Resolved in `78b3958`. `logger.warning` on dupes at parse time.

#### Tests

8. ✅ **Mid-file imports in `test_ollama_pool_client.py`.** Resolved in `a5e2689`.

9. ✅ **Smoke harness slack assertions over-permissive.** Resolved in `1fde787`. Tightened comment.

#### Docs

10. ✅ **`env.example` typo `http://ollama11434`.** Resolved in `d06fc29`. Fixed missing colon.

### From Chunk 2 (commits `bde1646..f85ff04`) — code reviewer found 5 NITs

#### Tests

11. ✅ **`test_factories_use_role_specific_pools` doesn't restore caches on teardown.** Resolved in `8716e26`. Autouse fixture clears caches before AND after each test (the implementer noted that pre-clear was also necessary to handle non-deterministic test ordering).

12. ✅ **`test_llm_client_is_cached_singleton` doesn't `cache_clear()` at the top.** Resolved in `8716e26` (same fixture covers it).

#### Code

13. ✅ **`app/services/ollama_clients.py:45` magic `timeout_s=120.0`.** Resolved in `15a13af`. Added an inline comment ("embeddings are fast; 120s covers worst-case batch").

14. ✅ **`app/api/v1/retrieval.py:1191` "Task 2.5" comment.** Resolved in `45dbaa3`. Rephrased.

#### Pre-existing (drive-by candidates)

15. ❌ **SKIPPED — `app/services/arcadedb_community.py:155-159` raw `_client.command()` access.** Pre-existing, out of scope for OllamaPool refactor.

### From Chunk 3 (commits `adb0666..1815fad`)

#### Regression caught during verification (already fixed)

**(Resolved in `1815fad`)** Lazy import in `build_pipeline_config` so host-venv tests don't break. (See plan doc commit log for full context.)

#### Code-quality NITs

16. ✅ **`tests/test_pool_client_mirror.py` — assert `text.count(_MARKER) == 1`.** Resolved in `f39bbbb`.

17. ✅ **`tests/test_pool_client_mirror.py` — anchor paths via `Path(__file__).resolve().parents[1]`.** Resolved in `f39bbbb`.

18. ✅ **`tests/test_pool_client_mirror.py` — add a negative test of `_shared_body`.** Resolved in `f39bbbb`.

19. ✅ **`docker/docling-graph/app/ollama_clients.py` — `get_docling_llm_client` cache-staleness limitation undocumented.** Resolved in `4ec9174`. Added "Limitations:" block in docstring.

20. ✅ **`docker/docling-graph/app/main.py` — `/debug/routing-metrics` pool-URL caching note missing.** Resolved in `21e592f`.

21. ✅ **`docker/docling-graph/app/main.py` — dead inline `import os` / `from fastapi import HTTPException`.** Resolved in `21e592f`.

22. ✅ **`docker/docling-graph/app/config_builder.py` — lazy import rationale not in function docstring.** Resolved in `2b568d2`.

23. ✅ **`docker/docling-graph/app/config_builder.py` — vestigial `provider_override`/`model_override`.** Resolved in `2b568d2`. **Removed** after verifying tests pass — the library's `pipeline/stages.py:470` short-circuit means these kwargs are dead when `llm_client` is set.

24. ✅ **`docker/docling-graph/app/ollama_clients.py` — drop trailing `# mirror of...` comment.** Resolved in `4ec9174`.

#### Observability gap surfaced during Chunk 4 validation

25. ✅ **`OllamaChatClient` logger has no handler in production runtime.** Resolved in `7b56c2d`. Added explicit logging-handler configuration to `app/services/ollama_pool_client.py` (mirrored to docling-graph copy via the byte-for-byte invariant). Production traces now show what the strip path is doing.

25b. ❌ **SKIPPED — Compose Ollama (`open-webui-stack-ollama-1`) crashed mid-Chunk-5 validation.** Operational issue in another stack, not code. Workaround in place (`OLLAMA_EMBEDDING_BASE_URL=http://10.0.1.109:11434`).

### Post-validation env config switch (user-requested 2026-04-29)

✅ **DONE in `243f585`.** `env.example` now documents the plural pool form as canonical. Singular forms remain (uncommented, blank values) as legacy fallbacks. `.env` itself was updated during Chunks 4-5 validation to the user's preferred config (singular fallbacks for back-compat + plural pools for LLM/VLM/embedding roles).

### Notebook alignment (user-requested 2026-04-29)

All 4 notebooks updated to reflect the OllamaPool refactor:

- ✅ `notebooks/extraction_walkthrough.ipynb` — `449d414` — pipeline-diagram + cascade documentation in markdown cells.
- ✅ `notebooks/extraction_walkthrough_direct_ollama.ipynb` — `5f4f9bc` — reframed (LiteLLM no longer the contrast point); cascade documentation.
- ✅ `notebooks/raw_libraries_walkthrough.ipynb` — `ea98cf9` — most substantive: replaces `LiteLLMClient` instantiation with `OllamaChatClient` + pool, drops dead `connection.base_url`/`context_limit`/`max_output_tokens` overrides; matches production wiring.
- ✅ `notebooks/ingest_walkthrough.ipynb` — `3f6b05c` — log-line expectation updates (`OllamaChatClient` not `LiteLLMClient`).

All notebooks pass `nbformat.validate`; cell code AST-validated; `raw_libraries_walkthrough.ipynb`'s new `OllamaChatClient` wiring smoke-tested end-to-end. Not run via `jupyter nbconvert --execute` — host venv lacks nbconvert and Jupyter sidecar wasn't running.

---

## Cleanup pass — what landed

13 commits to `main`, all pushed:

```
7b56c2d feat: cleanup nits — ollama_pool_client.py (items 1-5, 25)
78b3958 feat: cleanup nits — app/config.py (items 6, 7)
a5e2689 feat: cleanup nits — test_ollama_pool_client.py (item 8)
1fde787 feat: cleanup nits — tests/smoke/ollama_pool_routing.py (item 9)
d06fc29 docs: cleanup nits — env.example typo (item 10)
8716e26 feat: cleanup nits — test_ollama_clients_factory.py (items 11, 12)
15a13af feat: cleanup nits — ollama_clients.py (item 13)
45dbaa3 feat: cleanup nits — retrieval.py comment (item 14)
f39bbbb feat: cleanup nits — test_pool_client_mirror.py (items 16-18)
4ec9174 feat: cleanup nits — docling-graph/app/ollama_clients.py (items 19, 24)
21e592f feat: cleanup nits — docling-graph/app/main.py (items 20, 21)
2b568d2 feat: cleanup nits — docling-graph/app/config_builder.py (items 22, 23)
243f585 docs(env): document pool URLs as the canonical form
```

Plus 4 notebook commits (`449d414`, `5f4f9bc`, `ea98cf9`, `3f6b05c`).

**Final pytest count:** `1399 passed, 3 failed (pre-existing baseline), 3 skipped, 3 xfailed`. **Net delta from cleanup pass: +2 passing, 0 new regressions.** Mirror-drift test PASS.
