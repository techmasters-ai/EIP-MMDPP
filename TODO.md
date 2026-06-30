# TODO — Remaining Work

**Last updated:** 2026-06-29

---

## Backlog Reconciliation — 2026-06-29

Three-agent audit of every open item against HEAD `9950deb`. **NOTE: line numbers in items written before 2026-05 are stale** — `docling-graph/app/main.py` was refactored (filters → `evidence_gate.py`, table logic → `app/services/table_normalization/`) and `app/config.py` shifted. Corrected refs are inline below.

**Resolved / obsolete (no longer actionable):**
- **#84** (`_table_facts.py` zero `sustain_mass_kg`) — OBSOLETE. The fact-synthesis path was reverted behind `DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS` (off in prod; `.env:494`); production uses `app/services/table_normalization/`. Sustainer aliases now exist (`_alias_map.py:129-141`). See item for detail.
- **#85** (`5Ya23` dropped) — OBSOLETE as written (same reverted path). If the symptom recurs, the live suspect is the evidence-gate identity filter (#83), not the parser.
- **VARIANT_OF coverage gap** (memory-tracked) — FIXED in `a9f1028` (2026-06-17); manifest/coverage/validation_matrix all agree on `VARIANT_OF`. Residual cosmetic: stray `IS_VARIANT_OF` key in `validation_matrix.py:126` `SCORING_WEIGHTS` (0.7 default vs intended 0.95) — harmless to the coverage test.
- **HNSW post-filter starvation** (memory-tracked) — FIXED; production defaults to direct per-run cosine (`vector_router_retrieval_mode=direct`, `.env:530`). Legacy HNSW path retained behind a flag (optional cleanup).
- **`run_tests.sh` false-green** (memory-tracked) — FIXED; now `scripts/run_tests.sh` with `set -euo pipefail` + non-zero exit on failure.
- **Schema-wide retrieval/routing plan** (memory-tracked) — largely EXECUTED (post-rerank boost, identity anchors §8, subset-schema §9 all landed); only Phase A wire re-enable + D3 `table_boost` remain (deferred).
- **Lineage follow-up (e) checklist/README** — ADDRESSED in `VERIFICATION_CHECKLIST.md` + README.

---

## Open Items

### Frontend / UX

**#73. Render LaTeX in image-description panel**
**Status:** DONE 2026-06-29 (deploy = api image rebuild). The DoclingViewer "AI Image Analysis" panel now renders inline (`$...$`) and display (`$$...$$`) LaTeX via the `katex` npm package: new `renderImageDescriptionWithMath` helper (`DoclingViewer.tsx`) splits `content_text` on math delimiters and renders each math segment with `katex.renderToString` (CSS via `import "katex/dist/katex.min.css"`). Conservative guard — only segments containing LaTeX-ish chars `[\\^_{}]` are treated as math, so prose like `$5 and $10` stays literal; invalid LaTeX falls back to the raw text. `tsc --noEmit` clean + `vite build` succeeds (KaTeX fonts/CSS bundle). **Scope note:** the old `QueryPage.tsx` native-`title=`-tooltip subtask is DROPPED as stale — image descriptions are no longer surfaced via result-tile tooltips there; the panel is the live locus. Deploy: api image bakes the frontend (multi-stage `npm run build`), so it ships on the next `docker compose build api` + recreate.
**Files:** `frontend/src/components/DoclingViewer.tsx` (`renderImageDescriptionWithMath`)

**Observation:**
Picture descriptions emitted by `derive_picture_descriptions` frequently contain inline LaTeX (e.g. `$\sigma_0$`, `$E = mc^2$`, equation blocks). Today they appear in two places:
1. The native browser tooltip on result-tile images in `QueryPage.tsx` — set via the `alt=` / `title=` attributes — which can only render plain text. LaTeX shows literally as `$\sigma$`.
2. The "AI Image Analysis" panel in `DoclingViewer.tsx` — rendered into a `<div>` with `whiteSpace: "pre-line"` — same plain-text limitation.

`katex@^0.16.45` and `react-markdown@^9.0.3` are already in `frontend/package.json`, so no new heavy dependencies are needed.

**What needs to be done:**
1. Install `remark-math` + `rehype-katex` (peer deps of the existing katex install).
2. Replace the native `title=` tooltip on result-tile images with a small custom hover popover (positioned `<div>`, toggled by `onMouseEnter`/`onMouseLeave`, dismissable on focus-out) that renders the description through `react-markdown` with `[remarkMath]` and `[rehypeKatex]` plugins.
3. Apply the same renderer to the description block in `DoclingViewer.tsx`'s AI Image Analysis panel.
4. Keep the plain-text description in `alt=` so screen readers still get something useful, and load `katex.min.css` once at app shell so equations get the standard typeset look.

**Why this matters:**
Several radar/missile parameter docs use inline math for cross-section, gain, beamwidth, and pulse-parameter formulae. Those are exactly the descriptions analysts hover to confirm an image relates to the surrounding parameters; rendering them as `$\sigma_0 = ...$` text defeats the purpose.

**Acceptance:**
- Hovering an image with `$E = mc^2$` in its description shows a typeset equation, not the raw source.
- Block math (`$$...$$`) renders centered in the popover and the `DoclingViewer` panel.
- Native browser tooltip (or alt attribute) still carries the plain-text version for accessibility.
- No regression on images whose descriptions contain no math.

---

### Infrastructure / Worker Queue Isolation (Deferred)

**#28. Isolate `scan_watch_directories` from pipeline worker queue**
**Status:** DONE + DEPLOYED 2026-06-29. `scan_watch_directories` now routes to a dedicated `watcher` queue (`celery_app.py` task_routes); the catch-all `worker` consumes `celery,ingest,extract,embed,graph,graph_extract,trusted,watcher` while `worker-ingest` keeps `celery,ingest,extract` (NO watcher) — so scan polling can never sit in the `ingest` FIFO ahead of pipeline ingest tasks. Item 3 (stop beat during re-ingest) was already satisfied by `scripts/full_purge_and_reingest.py` (stops/restarts beat in steps 2/8). Item 4 (queue-depth ceiling) skipped as optional. Note: the periodic DB-polled `dispatch_pending_pipeline_stages` already mitigates the original "chain-break illusion"; this is defense-in-depth and removes the contention entirely. 2 routing unit tests added; worker recreated + beat restarted; verified live (beat→watcher→worker-1 receipt, ingest depth 0). **Caveat:** `watcher` is consumed only by the catch-all `worker` — a future split-only deployment without it would leave `watcher` unconsumed.
**Files:** `app/workers/celery_app.py`, `docker-compose.yml`

**Observed failure mode (2026-04-17 corpus re-ingest):**
During the docs-alignment migration run, the beat-driven
`scan_watch_directories` poller (every 30s → 120 tasks/hour) piled up
77+ unprocessed tasks in the `ingest` queue while pipeline workers
were blocked for hours on long-running `derive_ontology_graph` tasks
(LLM passes at 20–60 min each on llama3.3:70b thinking=high). With
`worker_prefetch_multiplier=1` and FIFO queue consumption, chain
continuations (`collect_derivations`, `derive_structure_links`,
`derive_canonicalization`, `finalize_document`) sat behind the
watcher backlog and appeared — incorrectly — as a chain break. The
chain WAS firing; continuations were just deep in the queue.

**What needs to be done:**
1. Add a dedicated `watcher` queue. Route
   `app.workers.watcher.scan_watch_directories` to it in
   `celery_app.py::task_routes`.
2. Update `docker-compose.yml` worker subscriptions so the primary
   pipeline worker does NOT consume from `watcher`. Either start a
   separate, low-priority watcher worker OR accept that watch-dir
   polling pauses during heavy ingest runs.
3. Update `scripts/full_purge_and_reingest.py` so the migration
   driver `docker compose stop beat` after step 2 (containers stop)
   and restarts beat only at the end of step 8 — during the ingest
   run, no new scan_watch_directories tasks should be generated at
   all.
4. Optional: add a ceiling on beat-generated tasks (drop-if-queue-
   depth-already-contains-same-task pattern) so intermittent worker
   restarts can't accumulate unbounded backlog.

**Why this matters:**
Without isolation, any long-running pipeline task starves every
downstream chain continuation by queue position, even when the
chain callback fires correctly. The symptom looks identical to a
chain break, which sent debugging off the rails for an hour during
the 2026-04-17 run. Proper queue isolation eliminates this failure
mode entirely.

**Acceptance:**
- `scan_watch_directories` tasks do not appear in the `ingest` queue.
- Migration script stops beat for the duration of a re-ingest.
- A load test with both stages running doesn't starve pipeline
  continuations regardless of queue depth.

---

### Feature Additions (Deferred)

**#27. LLM-based entity mention resolution**
**Status:** Deferred (valid; ref refreshed 2026-06-29). Intentionally deferred on cost/latency. NOTE: the cited `_build_entity_mentions` no longer exists — mention handling now lives around `_resolve_mention_chunks` (`app/workers/pipeline.py:1929`). Feature (LLM-based mention resolution as an enhancement over the current deterministic path) is NOT obsolete, just unimplemented.
**Files:** `app/workers/pipeline.py` (`_resolve_mention_chunks` area), new module TBD

**Current state:**
- `_build_entity_mentions` uses regex/substring matching to link extracted entities to document chunks.
- This catches exact text matches but misses: paraphrases ("the fire control radar" -> APG-77), abbreviations ("FCR" -> APG-77), misspellings, coreferences ("the system" -> S-400).

**What needs to be done:**
1. Add an LLM-based mention resolution mode, configurable via `ENTITY_MENTION_RESOLUTION_MODE` env var (values: `regex`, `llm`, default `regex`).
2. When `llm` mode is active, send each chunk text along with the entity list to an LLM to determine which entities are mentioned (even implicitly).
3. Batch chunks to reduce LLM call count (e.g., 10 chunks per call with boundary markers).
4. Fall back to regex mode on LLM failure.
5. Benchmark accuracy improvement vs. latency/cost tradeoff.

**Tradeoffs:**
| | Regex | LLM |
|---|---|---|
| Exact mentions | Yes | Yes |
| Paraphrases | No | Yes |
| Abbreviations/coreferences | No | Yes |
| Speed | ~50ms per doc | Minutes (thousands of LLM calls) |
| Cost | Free | Significant |
| Determinism | 100% | Non-deterministic |

**Acceptance:**
- Configurable via env var (regex remains the default)
- LLM mode produces more complete EXTRACTED_FROM edges
- Latency is bounded via batching

---

**#29. Route image-heavy / handwritten docs through VLM backend + one-to-one**
**Status:** Deferred (valid; trigger refreshed 2026-06-29). Still a live TODO — referenced at `docker/docling-graph/app/main.py:925` ("TODO #29 for VLM routing"). NOTE: the old trigger condition (`llama3.1:8b` + `batch=1024` + `gleaning=2` baseline) is obsolete — production is now gemma4:31b + absolute_union with no gleaning (only a stray `.env:118` context-size comment still mentions llama3.1:8b). Re-trigger criterion = when image-heavy/handwritten docs show measurably poor extraction vs the current gemma4 baseline.
**Files:** `app/workers/pipeline.py` (`_derive_ontology_graph_bundle_passes`), `docker/docling-graph/app/main.py` (per-request backend dispatch), `app/models/ingest.py` (optional `backend_hint` column)

**Observation (2026-04-18):**
The docs-sanctioned pattern for image-heavy / handwritten / form-style documents is `backend="vlm"` + `processing_mode="one-to-one"` + `numind/NuExtract-2.0-8B` — see `docling-graph-docs.md` §Extraction Backends ("VLM Backend Criteria") and §The Extraction Process ("Choose the Right Backend"). Four docs in the current corpus fit that profile:
- `cw_radar.jpg`
- `Fan_Song_Radar.jpeg`
- `chinese_handwritten_notes.pdf`
- `chinese_handwritten_notes_2.pdf`

On the current LLM-only path these docs either run the ontology pass on very sparse markdown (low recall) or fail the quality gate entirely (empty LLM output).

**What needs to be done:**
1. Add a per-document `backend_hint` at ingest (heuristic: `mime_type in image/*` OR `page_count==1 and ocr_text_ratio<X` → `"vlm"`, else `"llm"`).
2. Extend `/extract-pass` (or add `/extract-pass-vlm`) so the worker can request the VLM backend per call without changing the global service config.
3. Pull `numind/NuExtract-2.0-8B` on the Ollama host (or run via vLLM/local GPU runtime supported by docling-graph's `VlmBackend`).
4. Make the worker send the raw source bytes (not DoclingDocument JSON) for VLM passes, since VLM reads images directly.
5. Add `backend.cleanup()` call in the docling-graph service handler so VLM GPU memory is freed between requests (per docs best-practices note).

**Why this matters:**
Keeps prose extraction on the fast LLM+delta path while giving the image/handwritten minority of the corpus the docs-aligned VLM treatment. Without this, these docs consistently produce sparse or empty ontology output regardless of how well the LLM side is tuned.

**Acceptance:**
- `cw_radar.jpg` and `Fan_Song_Radar.jpeg` produce non-empty ontology entities end-to-end.
- The two `chinese_handwritten_notes*.pdf` docs either extract real entities or are explicitly marked "no-extract" (e.g. by a hint that says "image-only, no target domain content").
- No regression on the LLM/delta path for prose PDFs.

---

### Infrastructure / Ollama Routing (Deferred)

**#74. Centralize Ollama pool routing in a sidecar service**
**Status:** Open. Deferred to follow up on the in-process pool client refactor (`docs/superpowers/plans/2026-04-28-ollama-pool-client-refactor.md`).

**Observation:**
The OllamaPool client refactor (planned 2026-04-28) ships an in-process pool that lives inside every consumer container (api, worker, worker-graph, worker-ingest, worker-embed, docling-graph). Each process maintains its own least-in-flight counters. Round-robin tie-breaking handles the "all processes pick URL[0] on cold start" pathology, but the design has a structural blind spot: process A can't see process B's in-flight calls. With ~6 LLM-calling processes and 8+ Ollama instances, fan-out is good enough on paper, but the routing accuracy degrades as we add worker processes or as bursts overlap across containers.

**What needs to be done:**
Build a standalone `ollama-pool` Docker service:
1. New container at `docker/ollama-pool/` (Dockerfile + FastAPI app + requirements.txt).
2. Service exposes Ollama-compatible endpoints under role-prefixed paths: `/llm/v1/chat/completions`, `/vlm/v1/chat/completions`, `/embed/v1/embeddings`. Each path has its own backend pool driven by a separate env var (e.g. `OLLAMA_POOL_LLM_BACKENDS`, `OLLAMA_POOL_VLM_BACKENDS`, `OLLAMA_POOL_EMBED_BACKENDS`).
3. Service owns the existing `OllamaPool` routing core (already factored out by the in-process refactor — port unchanged). Single process = globally accurate least-in-flight + round-robin tie-break.
4. Health check + graceful shutdown + per-role metrics endpoint (`/metrics` or `/v1/diagnostics` showing per-backend served counts).
5. Add to `docker-compose.yml` with healthcheck and `depends_on` from api/workers/docling-graph.
6. Migrate consumer config: `OLLAMA_LLM_BASE_URLS=["http://ollama-pool:8001/llm"]` (single URL pointing at the sidecar). The `OllamaChatClient` still works unchanged — it's now a single-URL "pool" with the actual fan-out happening downstream.

**When to do it:**
Trigger on observed pathology — production traces showing one Ollama instance with sustained queue depth while others idle, OR scaling past ~20 LLM-calling worker processes (process-local blindness compounds with process count). Until then, in-process pool + `routing_metrics` instrumentation is enough.

**Estimated lift:** ~2.5–3 days (single implementer).

**Why this matters:**
Globally-accurate routing eliminates the "process A piles onto URL X while process B does the same independently" failure mode. Also lets ops swap routing strategies (least-busy → token-aware → sticky-by-document) without redeploying the app code in 6 containers — only the sidecar.

**Tradeoffs to weigh at trigger time:**
- Adds one container to monitor + deploy + debug
- Adds one HTTP hop (~1–3ms) on the hot path of every LLM call
- Creates a single point of failure unless replicated; replication needs shared in-flight state (Redis-backed counters as Option 2 between in-process and sidecar)

---

### Infrastructure / LLM Provider Support (Deferred)

**#75. Add vLLM as a peer provider to the OllamaPool**
**Status:** Open. Builds on the per-function pool refactor (`docs/superpowers/plans/2026-04-28-ollama-pool-client-refactor.md` Chunks 1–6).

**Observation:**
The current pool client speaks only Ollama's wire format. vLLM serves the same OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings` endpoints, but its structured-output and JSON-mode payloads differ:

- **Schema-constrained output:** Ollama uses `format=<schema_dict>`; vLLM uses `extra_body={"guided_json": <schema>, "guided_decoding_backend": "outlines"}`.
- **JSON mode:** Ollama uses `format="json"`; vLLM uses `response_format={"type": "json_object"}`.
- **`think` parameter:** Ollama-only (gpt-oss low/medium/high); vLLM models reject it.
- **Auth:** vLLM typically requires `Authorization: Bearer <key>`; Ollama doesn't.

The `OllamaPool` routing core (least-in-flight + round-robin tie-break + retry) is provider-agnostic and reused as-is. The per-function factory pattern from Chunk 6 is the natural extension point: each function picks its provider via env, the factory dispatches.

**What needs to be done:**

1. **Refactor (1 commit, ~2 hours):** Extract `BaseChatClient` and `BaseEmbeddingClient` from the current Ollama clients with `_post_chat_with_retry` + retry/diagnostics shared, leave `_build_chat_body` abstract. Both clients become ~50 lines each and the new vLLM clients only need to implement the body-builder.
2. **`app/services/vllm_client.py` (~150 lines, 1 day):** `VLLMChatClient` + `VLLMEmbeddingClient` implementing the same `LLMClientProtocol`. Differences from Ollama: `extra_body.guided_json` for structured output, `response_format` for JSON mode, Bearer auth header from `VLLM_API_KEY`, no `think` (`_coerce_think` becomes a no-op for vLLM).
3. **Settings + factory dispatch (~0.5 day):**
   - New env vars: `LLM_PROVIDER_DEFAULT` (process-wide ollama|vllm fallback), `VLLM_API_KEY`, plus per-function `<FUNCTION>_LLM_PROVIDER` env vars (parallel to per-function URL pools — `DOC_ANALYSIS_LLM_PROVIDER`, `TRANSLATION_LLM_PROVIDER`, `COMMUNITY_REPORT_LLM_PROVIDER`, `PICTURE_DESCRIPTION_PROVIDER`, `TEXT_EMBEDDING_PROVIDER`, `DOCLING_GRAPH_LLM_PROVIDER`).
   - Factory cascade: per-function provider → `LLM_PROVIDER_DEFAULT` → `"ollama"`. URL list cascade unchanged (vLLM consumes `<FUNCTION>_LLM_BASE_URLS` the same way).
   - Each `get_<func>_client()` factory dispatches: `if provider == "vllm": return VLLMChatClient(...) else: return OllamaChatClient(...)`.
4. **docling-graph mirror (~0.5 day):** Mirror `vllm_client.py` into `docker/docling-graph/app/`, add to `tests/test_pool_client_mirror.py`'s known pairs, update `get_docling_graph_client` factory to dispatch on `DOCLING_GRAPH_LLM_PROVIDER`.
5. **Tests (~bundled into 1+2 above):** Body-shape tests for `VLLMChatClient` mirroring the Ollama suite; factory-dispatch tests; cascade tests for `<FUNCTION>_LLM_PROVIDER`.
6. **Docs (~1–2 hours):** `env.example` and `README.md` section on provider selection + 2-3 common heterogeneous-provider patterns (e.g., extraction on vLLM, doc analysis on Ollama).
7. **End-to-end validation (~0.5 day, requires a real vLLM instance):** Heterogeneous reingest where graph extraction routes to vLLM and doc analysis routes to Ollama. Same gate structure as Chunk 6's heterogeneous-config validation.

**When to do it:**
Trigger when:
- An operator wants to deploy a vLLM instance (e.g., for a model not well-supported by Ollama, or for higher throughput)
- A specific LLM function gets pinned to vLLM (e.g., gpt-oss:120b on vLLM for doc analysis while gemma4:31b stays on Ollama for extraction)

Until then, the OllamaPool design serves the existing single-provider deployment.

**Estimated lift:** ~2.5 days (single implementer). The routing core, per-function factory pattern, settings cascade, mirror invariant, and observability hooks all already exist — this chunk is mechanical extension.

**Why this matters:**
- Different LLMs have different strengths. vLLM excels at high-throughput batched inference for fixed-model workloads; Ollama wins on flexibility (model swapping, easier model management). Letting each function pick its provider matches each function to its best backend without forcing a global choice.
- vLLM's structured-output backend (outlines/lm-format-enforcer) is more reliable than Ollama's `format=<schema>` for some model+schema combinations. If we hit gemma4:31b "Unterminated string" issues again on a future schema, vLLM is the natural escape hatch.

**Out of scope for v1:**
- Mixed-provider pool (some URLs in a single pool are Ollama, others vLLM). Requires per-URL provider tagging in `OllamaPool` — real design work, ~half day on top. Strongly recommend separate pools per provider; you almost certainly run homogeneous banks.
- Streaming. Neither current client streams; both providers support it but we've decided we don't need it.
- OpenAI / Anthropic clients (same pattern would apply; defer until needed).

---

### Code Quality / Tech Debt (Deferred)

**#76. Replace private `graph_store._client.command(...)` access with a public `GraphStore` method**
**Status:** DONE + DEPLOYED 2026-06-29. Added a public escape hatch to the `GraphStore` Protocol + `ArcadeDBGraphStore` impl — `execute_query` / `execute_query_sync` / `execute_command` / `execute_command_sync` (the command pair takes `language="sql"`, accepting `"sqlscript"` for multi-statement scripts) — each hiding the backend client AND the database name. Migrated **all 13** external private-access sites (the audit had named 3): `arcadedb_community.py`, `trusted_data_tasks.py`, `api/v1/extraction_routing.py`, `extraction_chunk_search.py` (×3), `extraction_chunk_index.py` (×4), `pipeline.py` (×5 incl. the sqlscript structural-edge writer). Grep confirms **zero** remaining external `_client.(command|query|command_sync|query_sync)` access. 6 unit tests added; workers+api restarted, in-container imports verified. (`arcadedb_graph.py` internal `self._client.*` usage is the impl and stays.)

**Observation:**
`app/services/arcadedb_community.py:155-159` reaches through the private attributes `graph_store._client` and `graph_store._database` to execute raw SQL when cleaning up stale community reports:
```python
await graph_store._client.command(
    graph_store._database, "sql",
    "DELETE VERTEX FROM CommunityReport WHERE community_id = :cid",
    {"cid": stale_cid},
)
```
The leading underscore marks both attributes as private by Python convention. This caller bypasses `GraphStore`'s public API and depends on the private interface. If `GraphStore`'s internals ever change (e.g., migrate from the HTTP client to gRPC, or wrap the client differently), this caller breaks silently.

**What needs to be done:**
1. Add a public method to `GraphStore` for the operation. Options:
   - `delete_vertex_by_predicate(class_name: str, where: str, params: dict)` — narrow, captures the specific use case
   - `execute_command(language: str, command: str, params: dict)` — broad, leaks the underlying SDK semantics back to callers
2. Migrate `arcadedb_community.py:155-159` to the new method.
3. (Optional) `grep -rn "graph_store\._" app/` to find any other callers reaching through private attributes; migrate them too.

**When to do it:**
Bundle with the next change to `app/services/arcadedb_community.py` for ANY reason, OR when `GraphStore`'s internals get reworked (a refactor away from the ArcadeDB HTTP client toward gRPC has been discussed). Don't do it standalone — too small a change to justify its own review cycle.

**Estimated lift:** ~30 minutes if `GraphStore` already has a similar public method to extend; ~1 hour if a new public method is needed end-to-end including a unit test.

**Why this matters:** encapsulation hygiene. Private-attribute access points are landmines for future refactors of `GraphStore` — any rename or restructure has to either preserve the exact attribute names verbatim or coordinate the change with every external caller.

---

### Production Readiness (Decision Needed)

**#79. Scope and execute production-readiness pass for the platform**
**Status:** Open. **Blocks on two decisions from product/eng leadership before brainstorming starts.**

**Observation:**
The user asked "I want this production ready" on 2026-04-30 during the post-OllamaPool-refactor monitoring run. That's a substantial scope expansion beyond the soft-fail cleanup of #77/#78. Off-the-cuff, the production-readiness surface for this stack spans at minimum:

- **Hard-failure handling.** Many failures today soft-fall to stub contexts, empty translations, or `WARNING` log lines (see #77, #78, plus the full list of stub paths in `docker/docling-graph/app/main.py` and the warning paths in `app/services/translation.py`). Production needs explicit policy: which failures are recoverable, which fail the run, which raise paging alerts.
- **Data-integrity gates.** Silent translation drops, empty extraction results, NULL flat fields on RADAR_SYSTEM/MISSILE_SYSTEM (the merge bug class — see commit `3296bf1`) — all these escape today's tests. Need post-stage assertions (e.g., "no doc completes derive_ontology_graph with zero entities AND zero relationships" → either a real error or a recorded skip-reason).
- **Observability.** Today: stdout logs and a `/debug/routing-metrics` endpoint behind an env flag. Production needs structured logs, per-stage error counters, percentile latency histograms, GPU-utilization fan-out per Ollama host, queue-depth dashboards, and on-call alerts with runbooks.
- **Open critical-path TODO items that block production.** At minimum: #28 (`scan_watch_directories` queue isolation — hides as chain breakage under load), #74 (sidecar pool — process-local routing blindness compounds with worker count), #76 (private `graph_store._client` access — refactor landmine), #77/#78 (soft-fail cleanup), plus the full `## Open Items` list reviewed for production-criticality.
- **Test coverage on live integration paths.** `tests/conftest.py:212` globally stubs `GraphStore`, so backend/query-profile tests don't exercise concrete ArcadeDB alias/chunk/vector behavior. Production needs a real-ArcadeDB integration-test tier (separate suite, runs in CI against a containerized ArcadeDB).
- **Deployment / runtime hardening.** Secrets out of `.env` and into a vault, no hardcoded URLs in compose, healthchecks on all services (currently only some), graceful shutdown verified across worker pools, backup/restore tested end-to-end (#32 added the scheduler — has the restore path been verified?).
- **Compliance posture.** This dataset includes CUI / FCI markers on remote hosts (`10.0.1.121` SSH banner). Production needs a documented authorization scope, audit logging, access controls on the API, and possibly an air-gapped deployment path.
- **SLOs.** Define ingest-rate target, query-latency p50/p95, retrieval-recall floor, and uptime target. Today none of these exist as committed numbers. Without SLOs, "production ready" is ambiguous.

**Decisions needed before brainstorming starts:**

1. **Deployment target and audience.** Drives what "production ready" actually means.
   - Single-tenant on-prem for the SA-2 use case?
   - Multi-tenant SaaS?
   - Air-gapped CUI environment?
   - Hybrid (gov + commercial)?

   These choices have very different scopes for compliance, secrets handling, multi-tenancy isolation, and deployment automation.

2. **Trigger and timeline.** Determines scope-cutting posture.
   - Hard date or stakeholder ask? → cut scope ruthlessly to a v1, defer hardening to v1.5.
   - "Next major milestone, scope it properly"? → full hardening pass over 4–8 weeks.
   - "Continuous" (production is the new working mode)? → ongoing program, prioritize by severity.

**What needs to be done (blocked on the two decisions above):**

1. Get answers to decisions 1 and 2 from the user.
2. Run a focused production-readiness audit: current state vs. typical bar for the chosen deployment target. Output: gaps list with severity (P0/P1/P2/P3) and rough effort estimate per gap.
3. Invoke the brainstorming skill on the audit output to scope the v1 production cut. Likely produces 1–3 sub-project specs (data integrity + observability is one spec; deployment hardening is another; compliance is a third).
4. Each sub-project gets its own spec → plan → implementation cycle (per the brainstorming skill's decomposition guidance).

**Why this matters:**
The platform has solid bones (ArcadeDB migration done, OllamaPool refactor shipped, merge bug fixed) but several "hides degradation silently" paths and zero committed SLOs. Calling it "production ready" without resolving those is the kind of declaration that gets reversed by the first incident. Scoping it properly up front avoids that.

**Estimated lift:** Cannot estimate until decisions 1 and 2 are answered. Audit step alone is ~1 day. Full hardening pass for a single-tenant on-prem deployment could be 2–4 weeks; for an air-gapped CUI environment, 2–3× that.

**When to revisit:** As soon as the user is ready to answer decisions 1 and 2.

---

### Pipeline / Soft-Fail Cleanup (Deferred)

**#77. Bump `TRANSLATION_TIMEOUT` to match worst-case graph-extraction queueing**
**Status:** Open. Surfaced during the 2026-04-30 fresh-ingest monitoring run.
**Files:** `.env` (line 160), `app/config.py:414`

**Observation:**
During the 2026-04-30 21-doc reingest, three translation calls hard-timed out with the pattern:
```
[WARNING] OllamaChatClient: ReadTimeout on http://10.0.1.109:11434 (attempt 2/2): timed out
[WARNING] Translation failed: timed out
translate batch N: input=2138 chars, output=0 chars, has_boundary=False
```
The translation call returns `output=0 chars` (silent translation drop) and the document continues on un-translated for that batch. `TRANSLATION_TIMEOUT=180` in `.env`, but graph-extraction calls on the same host (.109) take 191–368s end-to-end (gemma4:31b at `OLLAMA_NUM_PARALLEL=4`). When the pool routes a translation request to .109 behind even one in-flight extraction call, the translation can't complete inside 180s. The retry to a different URL counts as `attempt 2/2`, but if that URL is ALSO busy the second attempt times out the same way.

**What needs to be done:**
1. Bump `TRANSLATION_TIMEOUT=180` → `600` in `.env`. 600s comfortably exceeds the observed worst-case extraction call (368s) plus headroom.
2. Bump the default in `app/config.py:414` (`translation_timeout: int = 180`) → `600` so a fresh checkout doesn't reproduce the bug.
3. (Optional) Lower the floor: instead of bumping the timeout, dedicate a separate URL pool for translation that doesn't share with graph-extraction. This is what we already do for embeddings (.122 isolated). Pattern: add `OLLAMA_TRANSLATION_BASE_URLS` env var; route translation there. Avoids head-of-line blocking entirely. Larger change but eliminates the contention class.
4. Apply on next worker restart — env change only takes effect when the worker process restarts. Don't restart mid-ingest.

**Why this matters:**
Each timeout silently drops a batch of element translations. The doc proceeds with the original-language text in those elements, which:
- breaks downstream search recall (translated text was supposed to be the bridge between non-English source and English-trained embeddings),
- leaves a non-deterministic gap that won't show up as a hard failure,
- isn't caught by any existing regression test (the warning is `WARNING`, not `ERROR`).

**Estimated lift:** ~5 minutes for option 1+2; ~2 hours for option 3.

---

**#78. Short-circuit `extract-pass` for tiny-markdown DoclingDocuments**
**Status:** OBSOLETE — superseded by absolute_union (decided 2026-06-29, user-confirmed). The extract-pass short-circuit operates on the PER-PASS scoped DoclingDocument. Under `absolute_union` (narrow_only), a field pass's scoped doc IS the signal-selected chunks — legitimately short but meaningful (e.g. a `"Range: 45 km"` chunk selected precisely for its measurement signal). A blanket markdown-length short-circuit would drop exactly those chunks, regressing the field recall #83 protects. The original waste case (166-char junk chunk, 2026-04-30) predates absolute_union — back then tiny chunks were chunking artifacts; now they're deliberately selected. Not implementing. `_is_empty()` (truly-empty docs) at `main.py` stays as-is.

**Observation:**
During the 2026-04-30 reingest, three `/extract-pass` calls hit the soft-fail path with markdown_length=166 (twice) and 12662 (once):
```
2026-04-30 03:30:54 [INFO] OllamaChatClient: ok url=http://10.0.1.121:11434 model=gemma4:31b elapsed=58.93s len(content)=14
Warning: No valid JSON returned from LLM for DoclingDocument
[Extraction] Error extracting from DoclingDocument: Failed to extract data from DoclingDocument
Details: markdown_length=166
run_pipeline raised PipelineError: Pipeline failed at stage 'Extraction'
```
The 166-char markdown chunk is too short for gemma4:31b to find structured entities, so the model returns ~14 chars (likely `{}` or `null`), the docling-graph library's parser rejects it, and our wrapper at `docker/docling-graph/app/main.py:418` catches the `PipelineError` and returns a stub context with diagnostic marker — soft-fail, doc moves on.

The 58-second LLM round-trip on a 166-char input is pure waste: there's nothing meaningful to extract from a chunk that small, the model knows it, the parser knows it, but we still pay the latency + GPU time AND the worker counts it as an Extraction stage error in observability.

The existing `_is_empty()` short-circuit at line 281 only catches docs with NO body content / no texts / no pictures / no tables. It does not catch docs that have content but whose total markdown is too small to be worth a graph-extraction LLM call.

**What needs to be done:**
1. Add a `_is_too_small()` check parallel to `_is_empty()`. Compute total markdown length (sum of `texts[].text` lengths plus table/picture caption lengths). If below a threshold (e.g., 256 chars), short-circuit just like `_is_empty()` does — return `_EmptySourceContext` with a `_delta_trace` reason of `"markdown_too_small_for_extraction"` and the actual length.
2. Threshold should be configurable via env var (`DOCLING_GRAPH_MIN_MARKDOWN_CHARS`, default 256) so it's tunable without a code change.
3. Add a unit test in `docker/docling-graph/tests/` covering: 100-char doc → short-circuit; 1000-char doc → calls run_pipeline as before.
4. Verify the 12662-char case is unrelated — that one is a real failure (large enough that the model SHOULD extract entities), which means investigating why gemma4:31b returned 14 chars on a 12.6KB input. Probably a separate bug — file as a follow-up if so.

**Why this matters:**
- Eliminates ~60s of GPU time per tiny chunk that we KNOW won't yield extractable entities.
- Cleans up the soft-fail noise in observability — only true extraction failures (mid-size and large docs that fail) remain in the error log.
- Reduces worker queue pressure on the slow-pool host (.109) where these calls are accumulating.

**When to do it:**
Bundle with the next `docker/docling-graph/app/main.py` change. Standalone is fine if no other docling-graph work is queued.

**Estimated lift:** ~2 hours including the unit test + env-var wiring.

---

**#80. Sanitizer Rule 3 — drop encoded blobs (base64 + percent-encoded URL fragments)** — **DONE 2026-05-01**
**Files:** `docker/docling-graph/app/main.py:_contains_encoded_blob`, `_looks_like_nav_or_tracking`. Test: `docker/docling-graph/tests/test_sanitizer.py`.

**Why:** A 2026-05-01 batch dump from the radar_identity pass showed Rule 1 (ad-tracking domain) catching only the *first* fragment of a tracker URL after docling fragmented it on a line break. The continuation fragments (bare percent-encoded params like `0%26kv7%3DBA%26kv10%3D%5BISP%5D%26kv11%3D...`) and trailing base64 ad-payload tokens (`adroll_ad_payload=__HIA9QBkwHFA8HIA70AAZ1...`) lacked the `adroll.com` substring and slipped through, costing 60–180s of GPU time per such fragment to confirm "no entities."

**What landed:**
1. New `_contains_encoded_blob(text)` predicate with two sub-rules, both gated on a 64-char length floor:
   - **3a (base64):** match `[A-Za-z0-9+/_-]{64,}={0,2}`; require either explicit padding (`+`/`/`/`=`) or mixed-case + digit composition. Excludes hex hashes, UUIDs, and all-lowercase identifiers.
   - **3b (percent-encoded):** any whitespace-delimited token ≥ 64 chars with ≥ 6 `%XX` triplets.
2. Wired as Rule 3 in `_looks_like_nav_or_tracking()`; preserves `label='caption'` unconditionally per existing design.
3. Standalone false-positive guard verified against: SHA-256 hex hashes, UUIDs, short serial numbers (`RP-12345-A6B7C8-D9E0F1`), all-decimal runs, prose with embedded short encoded URLs, sentences with embedded UUIDs.
4. Unit test `tests/test_sanitizer.py` covers all three rules + the in-place blanking behavior. (Test infra requires running inside the docling-graph container; standalone Rule 3 verification ran on the host with all 10 cases passing.)

**Verification command (inside container after rebuild):**
```
docker exec eip-mmdpp-docling-graph-1 python -m pytest /app/tests/test_sanitizer.py -v
```

**Rebuild required:** `docker-compose build docling-graph && docker-compose up -d docling-graph`.

---

**#81. Surface sanitize stats in notebook outcome tracker** — **DONE 2026-05-01**
**Files:** `docker/docling-graph/app/main.py:660` (the `# TODO #81` marker is now removed). `notebooks/extraction_walkthrough.ipynb` cells `call-helper` (outcome tracker) and `section-inspect-markdown` + adjacent code cell (§2b).

**Why:** `trace["input_sanitize"]` was already being written into the response diagnostics, but the notebook outcome tracker didn't read it, so an operator couldn't see how aggressive the cruft filter was on a per-pass basis. With Rule 3 added (#80) the sanitizer drop rate becomes a primary signal for whether crud-removal is doing its job — surfacing it in the same table as `json_failed`, `quality_gate_fail`, etc. makes that visible.

**What landed:**
1. `_record_extraction_outcome()` now reads `diag["input_sanitize"]` and stores `texts_in` / `texts_dropped` on each outcome record.
2. `print_outcome_summary()` adds a `sanit` column showing `dropped/in` per pass plus a corpus-wide aggregate line: `sanitize_dropped / texts_in (all passes): X/Y = Z.Z%`.
3. §2b (`inspect_doc_markdown`) now applies a local mirror of `_sanitize_docling_document` before chunking, so the rendered markdown is byte-identical to what the LLM receives post-sanitizer. A header line shows the per-doc `texts_dropped/texts_in` ratio. Pass `apply_sanitizer=False` to inspect the raw pre-sanitizer view for diff comparison.
4. §2b markdown text updated to spell out the three sanitizer rules explicitly so notebook readers see what the filter is doing without reading `main.py`.

**Note:** the §2b sanitizer mirror is a code copy — the source of truth is `main.py`. Leave a "keep in sync" comment so future edits don't drift.

---

**#82. Apply TCP keepalive + read-timeout-split hardening to the ArcadeDB httpx client**
**Status:** DONE + DEPLOYED 2026-06-29. New shared helper `app/services/_http_keepalive.py` (`keepalive_socket_options` + `build_keepalive_client` / `build_keepalive_async_client`) applies SO_KEEPALIVE + TCP_KEEPIDLE/INTVL/CNT (~150s dead-peer detection) and a split `httpx.Timeout(connect=10, read=60, write=60, pool=30)` on a custom transport (limits carried on the transport). `arcadedb_client.py` `_get_async_client`/`_get_sync_client` now use it (read cap stays 60s — ArcadeDB queries are sub-second). The Ollama pool clients keep their byte-identical inline copy (they cannot import this main-app module — the docling-graph mirror, enforced by `tests/test_pool_client_mirror.py`, lacks it). 7 unit tests in `tests/unit/test_http_keepalive.py` (green); workers+api restarted, in-container wiring + ArcadeDB reachability verified.
**Files:** `app/services/_http_keepalive.py` (new), `app/services/arcadedb_client.py` (`_get_async_client`/`_get_sync_client`)

**#86. Reconcile pre-existing `ollama_pool_client.py` mirror drift (`think:` per-call override)**
**Status:** DONE + DEPLOYED 2026-06-29. The docling-graph mirror's `with_runtime_defaults` carried a per-call `think: bool | str | None = None` keyword (needed for thinking-model extraction) that the canonical main-app copy lacked. Synced the canonical to match (added the keyword-only param + `think=think if think is not None else self._default_think`) — backward-compatible, byte-identical to the mirror's shared section now. `tests/test_pool_client_mirror.py` is GREEN again (both pairs); 20 ollama-pool config tests still pass; workers+api restarted, param verified live in-container. The mirror tripwire can catch new drift again.
**Files:** `app/services/ollama_pool_client.py` (canonical), `docker/docling-graph/app/ollama_pool_client.py` (mirror).

**Observation:** The mirror invariant test is RED at HEAD. The docling-graph mirror's shared section has a per-call `think: bool | str | None = None` parameter (and `think=think if think is not None else self._default_think`) that the main-app canonical lacks (`think=self._default_think`) — 3 differing lines. One side was updated without the other.

**What needs to be done:** Decide which side is correct and sync the other. Likely the docling-graph extraction client legitimately needs the per-call `think` override (gpt-oss/thinking-model handling), in which case ADD it to the canonical `app/services/ollama_pool_client.py` (don't strip it from docling-graph — extraction may depend on it). Then `tests/test_pool_client_mirror.py::test_pool_client_mirror_in_sync` goes green. Verify no caller in the main app relies on the absence.

**Why this matters:** the mirror test is a tripwire against the two LLM-client copies diverging silently; while it's red it can't catch NEW drift, and the two services may already invoke gemma4/gpt-oss with different think semantics. Small, contained (3 lines) but it's a live behavioral divergence on the critical extraction client.

**Observation:**
On 2026-04-30 a /extract-pass call hung 35+ minutes against a healthy Ollama because docling-graph's `httpx.Client` had no `SO_KEEPALIVE` set and a single 20-hour blanket timeout — when a NAT/firewall/conntrack table aged out the idle TCP state mid-generation, the kernel never noticed the connection was dead. The fix in `49c2e43` added `_build_keepalive_http_client()` (TCP_KEEPIDLE=60, KEEPINTVL=15, KEEPCNT=6 → ~150s dead-socket detection; `httpx.Timeout(connect=10s, read=min(timeout_s, 1800s), write=60s, pool=30s)`) to both `OllamaChatClient` and `OllamaEmbeddingClient`.

The same hazard class technically exists on `arcadedb_client.py`'s two long-lived `httpx.Client` instances. They speak HTTP+SQL to ArcadeDB on the docker network, so the practical risk is dramatically lower:
- ArcadeDB queries typically complete in <1s — the socket rarely sits idle long enough for a middle-box to drop state
- Both endpoints are inside the same docker network, so no NAT/firewall idle timeout sits between them
- ArcadeDB itself doesn't have a multi-minute idle generation phase like Ollama+gemma4 at `stream:false`

But "low risk" ≠ "no risk." If a future deployment moves ArcadeDB out-of-cluster, or if `nf_conntrack_tcp_timeout_established` gets shortened, the same silent-stuck-connection failure could happen with no signal to the worker.

**What needs to be done:**
1. Hoist `_build_keepalive_http_client()` from `app/services/ollama_pool_client.py` to a shared utility (`app/services/_http_keepalive.py` or similar) — both pool clients and the ArcadeDB client should consume the same helper so any future tuning propagates.
2. There's an async variant needed too (`httpx.AsyncClient` for `_async_client`). Add `_build_keepalive_async_http_client()` mirroring the sync version (httpx accepts the same `transport=` and `timeout=` shapes).
3. Replace `httpx.AsyncClient(...)` and `httpx.Client(...)` constructions in `arcadedb_client.py` with the new helpers.
4. Update the existing `tests/test_pool_client_mirror.py` (or add a new unit test) to assert the helper produces the expected `socket_options` and `httpx.Timeout` shape.

**When to do it:**
Bundle with the next significant change to `arcadedb_client.py`, OR proactively as part of TODO #79's production-readiness pass — observability/SLO design will surface the question of "what happens when an ArcadeDB connection silently dies?" naturally.

**Estimated lift:** ~2 hours (helper extraction + async variant + 4 call-site updates + test).

**Why this matters:**
A worker hung on a dead ArcadeDB socket is invisible: no error log, no task timeout, no retry. With keepalive, the OS detects within ~150s and the SQLAlchemy/httpx stack surfaces a connection error that Celery's existing retry logic catches. It's a small change with disproportionate failure-mode coverage.

---

**#88. Retrieval pipeline is nondeterministic (hybrid result set varies run-to-run)**
**Status:** DONE 2026-06-30. Two root causes fixed: (1) `get_structural_neighbors` had a SQL `LIMIT` inside the MATCH that returned an arbitrary `limit`-sized subset of neighbors *before* ORDER BY — replaced with fetch-all + deterministic Python sort (`-weight`, `chunk_id`) then `[:limit]`; (2) THE primary cause — `_expand_seeds` ran up to 16 concurrent `_expand_one` coroutines all sharing the pipeline's single SQLAlchemy `AsyncSession` via `_expand_via_doc_structure`/`_batch_lookup_chunks`; concurrent ops on one session intermittently failed and were swallowed by `except`, so doc_structure candidates flipped run-to-run. Each expansion now opens its own `AsyncSessionFactory` session. Also added `chunk_id` secondary keys to all expansion sorts/dedups (`get_ontology_linked_chunks`, `get_related_entity_chunks`, `deduplicate_results`, `_apply_reserved_slots`, `_text/_image_vector_search` sorts, Postgres chunk_links ORDER BY). Verified: 10/10 + 5/5 byte-identical chunk_id lists across two distinct queries; unit tests 11/11 green incl. new `test_get_related_entity_chunks_deterministic.py`.

**Fix-review follow-ups (2026-06-30):** the per-expansion-session fix raised concurrent DB sessions/query from 1 → `_EXPAND_CONCURRENCY`. (a) DONE — lowered `_EXPAND_CONCURRENCY` 16→8 so ~3 concurrent hybrid queries fit the 30-conn async pool (was a pool-exhaustion risk). (b) DONE — wrapped the `get_structural_neighbors` MATCH in an outer `SELECT ... ORDER BY weight DESC, chunk_id ASC LIMIT cap` (cap = `max(limit*40, 100)`) so the cap applies POST-traversal; verified stable across repeated live ArcadeDB calls, and the Python `(-weight, chunk_id)` sort + `[:limit]` stays the final authority. (c) DONE — promoted the expansion-path `except` fallbacks (ArcadeDB→Postgres doc-structure, Postgres→[], `_merge_seed_results`, and the previously-silent `_expand_seeds` gather Exception filter) from `logger.debug` to `logger.warning`, control flow unchanged. (d) DONE — added `tests/unit/test_expand_seeds_own_session.py` asserting each `_expand_one` gets a fresh per-expansion `AsyncSessionFactory` session distinct from the pipeline `db`. Live 5× determinism re-verified IDENTICAL after all three changes.

**Status (historical):** Open. Surfaced 2026-06-30 during the ontology-aware-hybrid-retrieval Task 9 gate (NOT caused by it — verified on `main` with no feature code).
**Files:** `app/api/v1/retrieval.py` (`_expand_seeds` async gather, `_deduplicate_results`, `_diversify_results`, `_multi_modal_pipeline`).

**Observation:** `POST /v1/retrieval/query` `strategy=hybrid` for the same query (`"SA-2 guidance radar"`, top_k=10) returned **6 results one run, 5 the next** on `main` — set intersection 4, symmetric difference 3. Result **set membership** (not just ordering) varies run-to-run. This made the planned "byte-identical rollback" gate criterion unachievable (no stable baseline); the gate was dispositioned to an inert-rollback criterion (user-approved). Broadly undermines retrieval reproducibility (A/B comparisons, regression gates).

**Likely sources (unverified):** `_expand_seeds` runs per-seed expansion via `asyncio.gather` under a semaphore — async completion ordering feeding `_deduplicate_results`/`_diversify_results` could change which near-duplicate survives; or a flaky graph/vector query under concurrency; or set/dict iteration order in dedup. Count *varying* points at expansion/dedup, not pure sort.

**What needs to be done:** make the expansion gather + dedup order-stable (deterministically sort expanded items before dedup; confirm vector/graph queries are deterministic), then add a determinism regression test (same query twice → identical chunk_id list). See [[project_retrieval_nondeterminism]].

---

**#83. Relax post-extraction IDENTITY_FILTER (recoverable false-positive drops at high temperature)**
**Status:** Tier A DONE + DEPLOYED 2026-06-29. `identity_is_supported_by_batch_text` (`evidence_gate.py`) now builds a separator-tolerant pattern: the normalized identity is split on `[\s\-_/.]+` and rejoined with an optional-separator class so "SA-2 C" matches "SA-2C"/"SA 2 C" while alphanumeric boundary lookarounds still block over-match ("S-75"≠"SA-75") and commas/semicolons/colons are NOT bridged (no list-glue false positives). Verified on the 2026-06-29 fresh ingest: 17/19 distinct dropped missile names were recoverable; the 2 genuine hallucinations (9K33M3, HQ-2P) are still dropped. 9 unit tests added to `test_service_identity_gate.py` (all green); docling-graph rebuilt + recreated (healthy); in-container behavior confirmed. **Tier B (numeric-aware match) + Tier C (temperature-aware advisory + `DOCLING_GRAPH_IDENTITY_GATE_MODE` kill-switch) remain optional follow-ups.** See [[project_identity_filter_recoverable_drops]]. Original detail below.

**Status (historical):** Open (refs corrected + ELEVATED 2026-06-29). Still valid and unrelaxed — the filter now lives in `evidence_gate.py` and still drops any entity whose normalized identity (uppercase/whitespace-collapsed via `normalize_evidence_text`, `evidence_gate.py:118`) fails a literal regex/substring match against batch text (`evidence_gate.py:169-172`), no fuzzy/temperature loosening. **This is now the prime suspect for the #85 `5Ya23` drop** — if entities are still disappearing, this gate (not the reverted parser path) is the cause.
**Files:** `docker/docling-graph/app/evidence_gate.py:176` (`filter_pass_output_by_batch_text`), `:278` (`filter_provenance_rows_by_allowed_identities`); call sites `main.py:1824` / `:1880`, IDENTITY_FILTER log `main.py:1908`.

**Observation:**
The post-extraction "service identity gate" at `main.py:1121` substring-matches each extracted entity's identity values against the batch's evidence text and drops any that don't appear verbatim. At higher LLM sampling temperatures (T=1.0 in particular), gemma4:31b extracted entities whose identity values ARE in the source document but whose surface form differed from the evidence text by literal characters — the gate rejected them as if they were hallucinations.

Concrete failure modes the substring check misses:
1. **Unit normalization** — model emits `"30 kilometres"` while batch has `"30 km"` (or model converts `"29000 m"` → `"29"` for a `_km` field, mathematically correct but invisible to substring match).
2. **Whitespace / punctuation** — model emits `"SA-2 Guideline"` while batch has `"SA-2  Guideline"` (table-extracted double space) or `"SA-2/Guideline"`.
3. **Case** — current check appears case-sensitive depending on the cell.
4. **Cross-cell concatenation** — value spans a table cell boundary the evidence-text collector flattened differently from the model's reconstruction.
5. **Numeric formatting** — `"1,135"` vs `"1135"` vs `"1.135 t"` for the same booster mass.

The gate is correct in spirit (drops obvious hallucinations), but its substring-equality implementation is too literal — at T≥0.5 it costs us *real* extractions while still admitting some false positives. At T=0.0 the model's surface form usually matches the input verbatim, so the gate is approximately a no-op; at T=1.0 the same gate drops correct entities along with hallucinations.

**What needs to be done:**

Three relaxation tiers, each independently shippable. Land Tier A first; measure delta vs R17/alias-patch baseline before deciding on B/C.

1. **Tier A (cheap, ~2 hours):** Normalize both sides before substring check. Lowercase, collapse whitespace runs, strip ASCII punctuation `[-_/.,;:]`. Recovers (2) and (3) immediately. Add a unit test in `docker/docling-graph/tests/` covering each of (2)/(3) variants. Acceptance: `IDENTITY_FILTER` `service_identity_dropped` count at T=1.0 drops by ≥30% on the SA-2 corpus without admitting any new hallucinations (verify by spot-check of `dropped_examples` log before/after).

2. **Tier B (moderate, ~4 hours):** Numeric-aware match. When the field's Pydantic type is `int`/`float`, parse both the extracted value and any number-shaped tokens in the same evidence cell as numbers; admit if `abs(extracted - any_token) / max(any_token, 1) < 0.10` AND the unit-class is consistent with the field name (`*_km` accepts evidence-cell numbers near the value × {1, 1000} for m↔km conversion; `*_m` accepts × {1, 1/1000}; `*_kg` only accepts × 1, etc.). Recovers (1) and (5). Reuse the `grade()` tolerance dict already in `notebooks/extraction_walkthrough.ipynb` §20 as a starting source-of-truth for per-field tolerances. Acceptance: aggregate ✓+~ count on the §20 GT scorecard increases at T=1.0 vs Tier A baseline by ≥20%.

3. **Tier C (advisory, ~1 hour):** Make the gate temperature-aware. Read the call's `temperature` (already plumbed through `extract_pass` body) and switch behavior:
   - T ≤ 0.3: drop on miss (current behavior — keeps the strict gate where it works).
   - T ≥ 0.5: log a `WARNING` with `gate_advisory=true` but admit the entity. The `service_identity_dropped` count becomes `service_identity_advisory` so observability still tracks how often the gate would have fired.
   - Add a kill-switch env var `DOCLING_GRAPH_IDENTITY_GATE_MODE` ∈ `{strict, normalized, numeric, advisory}` defaulting to `normalized` (Tier A) so an operator can pin behavior without a code change.

**Why this matters:**
At T=1.0 the model has higher recall (more entities found) AND higher field-fill (more numeric fields populated per entity) — exactly what the §20 sweep is trying to measure — but those gains are partially being clawed back by an over-strict post-extraction gate. The R17 vs T=1.0 comparison currently understates the temperature effect because we're measuring `extracted ∩ admitted_by_gate`, not `extracted`.

This is the same class of soft-fail that #77 (silent translation drops) and #78 (tiny-markdown stub) address: a real signal being lost to a too-aggressive filter, with no surfacing in the operator's primary dashboard. Per the project's "soft-fails belong in TODO" rule, fixing it visibly closes one more silent-degradation path.

**Estimated lift:** Tier A ~2h. Tier B ~4h. Tier C ~1h. Total ~1 day if all three land, or ~2h for the high-leverage Tier A alone.

**When to do it:**
Bundle with the next `docker/docling-graph/app/main.py` change. Specifically: if the alias-patch sweep (currently running) shows field-fill plateau at T=0.3 and the working hypothesis shifts to "T=1.0 has more raw signal but the gate is eating it," ship Tier A immediately and re-run §20 — that's the cheapest test of the hypothesis.

**Acceptance:**
- `IDENTITY_FILTER` log shows reduced `service_identity_dropped` count at T=1.0 (Tier A: ≥30%).
- `dropped_examples` spot-check confirms no new hallucinations admitted.
- §20 GT scorecard at T=1.0 shows ≥20% more ✓+~ entries vs current behavior (Tier B).
- Optional: §20 cells re-run identical → cache-friendly diff demonstrates the gate change is the only delta.

---

**#84. Parser-side `_table_facts.py` emits zero `sustain_mass_kg` facts on SA-2 Sustainer band (silent regression vs 2026-05-06 baseline)**
**Status:** OBSOLETE (closed 2026-06-29 audit). The `synthesize_table_facts` path this item targets was built+validated then reverted behind `is_experimental_table_facts_enabled()` (`main.py:699-707, 870-883`), defaults off (`table_normalization/config.py:59-60`), and is off in prod (`DOCLING_GRAPH_USE_EXPERIMENTAL_TABLE_FACTS=false`, `.env:494`). Production uses `app/services/table_normalization/` instead, and the missing sustainer aliases now exist (`_alias_map.py:129-141`, `detect_section_context` at `_table_facts.py:396`; commits `b8a9ccd`/`767d62f`, 2026-05-17). The original detail is retained below for history.
**Files:** `docker/docling-graph/app/_table_facts.py` (section detection — `detect_section_context` from commit `b888a3b`, `extract_label_rows`, fact emission path), `docker/docling-graph/app/_alias_map.py` (ALIAS_MAP keyed on `(label_normalized, section_ctx, pass_name)`), `pipeline_pass_outputs.extract_pass_response_json->'table_overlay'->'facts'` (post-run inspection).

**Observation:**
ALIAS_MAP entries for sustain_mass_kg require `section_ctx ∈ {"2nd Stage", "Sustainer"}`. In the post-merge `table_overlay.facts[]` JSON for missile_propulsion in this run, every fact has `section_ctx: null` and **zero facts have `schema_field == "sustain_mass_kg"`** (vs 9 expected per the 2026-05-06 baseline at `/tmp/baseline_2026-05-06_pre_overlay/expected_overrides.md`):

| schema_field | facts emitted (this run) | expected per baseline |
|---|---|---|
| booster_mass_kg | 9 | 9 |
| sustain_mass_kg | **0** | **9** |
| total_mass_kg | 10 | 10 |
| body_length_m | 10 | 10 |

Booster facts still emit despite `section_ctx=null` because there is an effective default-to-booster fallback path. Sustain has no such fallback, so when section detection produces null the second "Weight" row in the SA-2 table (the Sustainer band) is silently dropped at lookup time. Net effect: **all 8 expected `sustain_mass_kg` overlay corrections + 1 silent apply never fire**, and ArcadeDB ends up with the LLM's raw sustain values (mostly equal to each variant's booster value because the LLM filled both fields from the same booster row).

Confirmed via:
- 0 `FIELD_OVERLAY_OVERRIDE` log lines with `field=sustain_mass_kg` in this run (vs 8 expected).
- `TABLE_OVERLAY_APPLIED ... field_overlay_applied=144 conflicts_overridden=6` — the 6 conflicts are all on entity=1D (booster_mass_kg, max_intercept_km×3, total_mass_kg×2). No sustain in any overlay log.
- ArcadeDB MISSILE_SYSTEM rows for SA-2 variants show sustain_mass_kg=1032 for 13DM/13DA/13DAM (matches their booster_mass_kg values), 1011 for 20D/20DP/20DSU (same pattern), 1007 for 1D (5Ya23's LLM value via cross-variant assignment), and `null` for 13D.

**What needs to be done:**

1. **Add a `git show 8d91986 -- docker/docling-graph/app/_table_facts.py docker/docling-graph/app/_alias_map.py` diff against current HEAD** to identify what changed in section detection between the 2026-05-06 baseline and now. The 9 baseline sustain facts proved section detection USED to work; the regression is somewhere in this diff (or in a `detect_section_context` path that was added/altered after).
2. **Add an integration test** under `docker/docling-graph/tests/test_table_overlay_*` that exercises the SA-2 PDF table (or a synthetic version with two `Weight` rows in distinct sections) and asserts the parser emits ≥1 `sustain_mass_kg` fact. Currently no test in that directory specifically guards against this — the drift was invisible until live ingest.
3. **Implement fix.** Most likely either: (a) restore the prior section-header detection logic, (b) add fallback ALIAS_MAP entries with `section_ctx=None` that disambiguate by row-position-within-table (e.g., second `Weight` row → sustain), or (c) emit a WARNING when ALIAS_MAP lookup misses on a recognized label so future regressions surface immediately.
4. **Regenerate the merge notes baseline** at `/tmp/baseline_2026-05-06_pre_overlay/expected_overrides.md` from a known-good run (post-fix) so future acceptance runs have a current reference.

**Why this matters:**
A1's overlay phase is supposed to correct LLM mis-extraction of numeric fields by writing the table-derived "ground truth" over the LLM's value. With sustain_mass_kg facts missing, half of the propulsion correction is silently disabled. The pipeline still completes and writes to ArcadeDB — but the data is the LLM's raw output, defeating the entire point of Mechanism A1 for that field on every SA-2-style ingest.

This is the same soft-fail class as #77/#78/#83: a real degradation hidden behind a successful pipeline run, with no operator-visible signal. Per the standing rule, it deserves a TODO with a concrete fix sketch.

**Estimated lift:** Investigation ~1h (diff `8d91986..HEAD`). Test ~2h. Fix ~2-4h depending on whether section detection or fallback ALIAS_MAP is the right intervention. Total ~half a day.

**When to do it:**
Soon — every SA-2 ingest is silently producing wrong sustain_mass_kg values until this is fixed. Bundle with the next `docker/docling-graph/app/_table_facts.py` change, or pull forward as its own focused commit if the data-quality impact is operationally meaningful.

**Acceptance:**
- `pipeline_pass_outputs.extract_pass_response_json->'table_overlay'->'facts'` for missile_propulsion contains ≥9 facts with `schema_field="sustain_mass_kg"` on a fresh SA-2 ingest.
- Re-running the SA-2 acceptance produces a 1D MISSILE_SYSTEM vertex with sustain_mass_kg matching the table value for the Sustainer-band Weight row (verify exact value against fresh baseline).
- ≥8 `FIELD_OVERLAY_OVERRIDE pass=missile_propulsion ... field=sustain_mass_kg` log lines fire (count varies with LLM stochasticity but >0 is the contract).
- Drift-guard test in `docker/docling-graph/tests/` fails before fix, passes after.

---

**#85. `5Ya23` MISSILE_SYSTEM dropped between parser facts and ArcadeDB write (cross-bug companion to #84)**
**Status:** OBSOLETE as written (2026-06-29 audit) — the "parser-facts → graph-write" path it blames is the same reverted `synthesize_table_facts` flow that does not run in production (`.env:494`). If `5Ya23`-class entity drops recur on a fresh ingest, the live cause is the evidence-gate identity filter (#83, `evidence_gate.py:176`), which drops entities whose identity doesn't literally match batch text. Being verified against the current 21-doc ingest; if confirmed it folds into #83. Original detail retained below.
**Files:** `app/services/extraction_merge.py` (identity canonicalization, entity-merge phase), `app/services/table_overlay.py` (`apply_identity_rewrite`), `app/services/arcadedb_graph.py` (graph write path), and parser-side `_alias_map.py` (`MISSILE_IDENTITY_LABELS`, `CANONICAL_PRIORITY`).

**Observation:**
The parser's `table_overlay.facts[]` includes a fact `entity=5Ya23 schema_field=booster_mass_kg value=1007.0` (one of 9 SA-2 variants the parser correctly extracts). But ArcadeDB has **zero** MISSILE_SYSTEM vertices with `system_name="5Ya23"` after merge — even broad search (`system_name LIKE "5%" OR name LIKE "5%"`) returns empty. The other 8 expected variants (1D, 13D, 13DM, 13DA, 13DAM, 20D, 20DP, 20DSU) all wrote successfully.

So `5Ya23` is being dropped somewhere between (a) the parser's fact emission and (b) the ArcadeDB graph write. Most likely culprits:
1. **Identity rewrite (Phase 0 of A1)** — `IDENTITY_REWRITE rewrites=56 unique_canonicals=10 passes_touched=6` fires globally. If `5Ya23` is being rewritten to a canonical form the rest of the pipeline doesn't handle, or collapsed into another variant by mistake, it'd disappear here.
2. **Entity-merge canonicalization** — `merge_and_resolve` builds a `LogicalIdentity` per entity instance and merges across passes. If 5Ya23 ends up with a LogicalIdentity that collides with another variant (e.g., maps to "20DSU" via some alias rule), its instance gets merged away.
3. **Identity gate (`IDENTITY_FILTER`)** — substring-matching the variant name against batch evidence text could drop 5Ya23 if the doc never contains the literal string "5Ya23" verbatim and the LLM was emitting it from canonicalization rather than verbatim quote (similar to #83).
4. **Validation / Pydantic coercion** — the 5Ya23 value of 1007.0 might trip a coerce-to-int validator that rejects exactly that value. Less likely but worth a quick check.

**What needs to be done:**

1. **Trace `5Ya23` through merge logs.** Re-run with `LOG_LEVEL=DEBUG` on worker-graph (or grep merge-phase logs from this run for "5Ya23") and identify the phase where it disappears: parser → identity rewrite → instance build → LogicalIdentity merge → graph write.
2. **Inspect `_alias_map.py:MISSILE_IDENTITY_LABELS` and `CANONICAL_PRIORITY`** for any rule that maps "5Ya23" to another variant (e.g., a Cyrillic-aware alias or a "5Ya23 → 20DP" rule that's overzealous).
3. **Write a focused unit test** that constructs a synthetic PassResult containing a 5Ya23 entity and asserts `merge_and_resolve` produces a vertex with `system_name="5Ya23"` (parametrize for all 9 SA-2 variants — a drift-guard against cross-collapse regressions).
4. **Implement fix at the identified phase** (most likely in identity rewrite or canonicalization, depending on what step 1 reveals).

**Why this matters:**
5Ya23 is a real SA-2 variant (the export designation for some configurations of S-75). Silently dropping a canonical entity from the graph means downstream queries ("show me all SA-2 variants with sustain mass < 1100 kg") will be wrong by exactly one row per ingest. Combined with #84, this is the second silent data-quality regression in the SA-2 propulsion path. Both are invisible from the pipeline-status field — they require comparing emitted parser facts against the ArcadeDB final state, which no current automated check does.

**Estimated lift:** Investigation ~30min (focused log grep + alias-map read). Fix likely ~1-2h. Test ~1h. Total ~half a day.

**When to do it:**
Pair with #84 — same run produced both deltas, same investigation tooling, same acceptance regenerator (the fresh baseline). Filing as a separate item because the fix probably touches different files (worker-side merge vs. parser-side fact extraction).

**Acceptance:**
- Fresh SA-2 ingest produces a MISSILE_SYSTEM vertex with `system_name="5Ya23"` (and matching booster/sustain mass values from the table).
- All 9 SA-2 variants present in ArcadeDB post-ingest (1D, 13D, 13DM, 13DA, 13DAM, 20D, 20DP, 20DSU, 5Ya23).
- Drift-guard unit test in `tests/unit/` constructs synthetic per-variant PassResults and asserts each variant survives `merge_and_resolve` with the expected `system_name` populated.

---

## Completed Items (Reference)

### Gaps/Bugs Fixed

- **#35.** Fixed upstream repo clone/update contract for Docling and Docling-Graph (verification checklist/design spec updated to reflect actual PyPI package contract)
- **#36.** Fixed vector_search() dropping chunk metadata (`chunk_id`, `document_id`, `artifact_id`, `modality`, `text`) that retrieval depends on
- **#37.** Fixed ontology/evidence traversal edge direction and identifier mismatch (superseded by #49)
- **#38.** Fixed alias resolution property name inconsistency (standardized on `alias_name`; `entity_type` filter moved to linked entity)
- **#39.** Aligned docling-graph wrapper with canonical template/pipeline API (`is_entity=True`/`edge()` pattern)
- **#40.** Wired retrieval `filters` (classification, document_id, modality constraints) that were silently ignored
- **#41.** Fixed document canonicalization scope (superseded by #50)
- **#42.** Fixed orphan cleanup system field (`@cat` changed to `@class`)
- **#43.** Added `$distance` projection to community-report vector search (scores were collapsing to `0.0`)
- **#44.** Fixed stale test harness for docling-graph integration tests (patched `app.state.templates` instead of deleted global)
- **#45.** Reconciled Stage 1 worker/service request+response contract (worker/service request and response shape aligned)
- **#46.** Made derive_ontology_graph() consume persisted DoclingDocument JSON from MinIO instead of reconstructed plain text
- **#47.** Normalized extraction output shape before graph import (adapter maps Docling-Graph output to `{nodes, edges}` format)
- **#48.** Fixed entity-to-chunk mention wiring to be element-complete (image chunks included; partial fallback for zero-mention entities)
- **#49.** Fixed EXTRACTED_FROM traversal direction and identifier type across all callers (UUID-to-RID lookup added)
- **#50.** Fixed canonicalization to use graph traversal for document-entity discovery (replaced LUCENE search with Document->chunk->entity edges)
- **#5.** Implemented validation_matrix enforcement in upsert_relationship (reject/warn on invalid triples)
- **#51.** Fixed query profiles to honor their own directed traversal model (native MATCH patterns from each step's direction/hops)
- **#52.** Fixed dossier/query-profile evidence attachment (separate entity-centric vs chunk-centric helpers)
- **#53.** Fixed hybrid cross-modal fallback UUID/RID mismatch (UUID-to-RID resolution added)
- **#54.** Fixed ontology expansion to preserve graph context from returned chunks (enriched traversal results)
- **#55.** Implemented retrieval filter push-down into native queries (document_id as WHERE clause on outer SELECT)
- **#56.** Aligned API defaults with advertised retrieval settings (top_k=20, min_confidence=0.1)
- **#57.** Fixed graph fulltext search to carry Lucene $score through result model (score priority: $distance > $score > extraction_confidence)
- **#58.** Wired co-extracted entity discovery into the query stack (fallback in entity resolution)
- **#59.** Updated stale integration tests to assert current response shape (strategy/modality_filter)

### Features Implemented

- **#1.** Implemented real LLM community report generation (`_call_llm_for_report` with Ollama, JSON parsing, thinking-model handling)
- **#2.** Implemented community report embedding (`_embed_report` wraps `embed_texts` via `asyncio.to_thread`)
- **#3.** Wired LLM synthesis into global query response (single synthesized answer with raw reports in context)
- **#4.** Added batch HTTP operations to replace N+1 patterns (sqlscript batches for nodes, edges, chunks with embeddings)
- **#6.** Implemented post-ingest hook counter + threshold trigger for community detection (Redis counter with configurable threshold)
- **#7.** Added batch_create_entity_chunk_edges to GraphStore Protocol (single sqlscript call replaces per-edge loop)
- **#21.** Added VLM extraction backend option to docling-graph service (`DOCLING_GRAPH_BACKEND` env var with `llm`/`vlm` options)
- **#22.** Implemented dual-approval workflow for graph mutations (`approvals` list, duplicate-curator prevention, config flag)
- **#25.** Added benchmark suite for retrieval strategies (`tests/benchmarks/` with latency assertions)
- **#28.** Native ArcadeDB vector functions for cross-model queries (efSearch parameter, graph_vector_search with vectorCosineSimilarity, Python-side fusion scoring)
- **#29.** MATCH syntax for graph traversal queries (get_neighborhood and get_neighborhood_graph rewritten to MATCH pattern syntax)
- **#72.** Restored Global Search indexing UI on the Ontology page (GlobalSearchIndexingPanel with schedule display, manual trigger, status, reports browser)

### Code Quality / Refactoring

- **#8.** Optimized schema sync to batch DDL statements (~200 calls reduced to 7 phase-scoped sqlscript batches)
- **#9.** Implemented time-based ensure_ready caching (configurable TTL via `ARCADEDB_READY_CACHE_SECONDS`)
- **#10.** Extracted shared SQL building to reduce async/sync duplication (`_build_*_sql()` helpers return `(sql, params)` tuples)
- **#11.** Consolidated Redis client factory (singleton `get_redis()` in `redis_utils.py` with `close_redis()` shutdown hook)
- **#12.** Standardized Redis locking idiom (`r.lock()` pattern; `redis_lock()` helper; bare `SET NX` patterns replaced)
- **#13.** Parameterized vector_search instead of duplicating for image (unified method; `image_vector_search` removed)
- **#14.** Unified RESERVED_WORD_MAP definitions (shared `ontology/arcadedb_reserved_words.json` contract file)
- **#15.** Replaced manual env var helpers with pydantic_settings in docling-graph (`DoclingGraphSettings(BaseSettings)`)
- **#16.** Moved module-level globals to FastAPI `app.state` in docling-graph service
- **#17.** Derived `_STRUCTURAL_TYPES` from `arcadedb_schema.STRUCTURAL_TYPES` export (single source of truth)
- **#18.** Scoped resolve_root_entity queries to specific types (optional `entity_type` parameter with `V` fallback)
- **#19.** Pre-compiled regex patterns in `_build_entity_mentions` (one compilation per entity instead of per element)
- **#20.** Collapsed pipeline stage section comments (comments now explain WHY, not WHAT)
- **#23.** Added ArcadeDB connection pooling (singleton GraphStore; httpx `Limits` tuned)
- **#24.** Added observability for ArcadeDB operations (latency histograms, error counters, slow-query logging, `/metrics` endpoint)
- **#26.** Migrated docling-graph `templates.py` to use `template_builder.py` logic (removed duplication; single source of truth)
- **#30.** Added BucketSelectionStrategy `'thread'` for write-heavy types (TextChunk, ImageChunk, high-write entity types)
- **#31.** Enabled ArcadeDB Prometheus metrics plugin (`PrometheusMetricsPlugin` in docker-compose JAVA_OPTS)
- **#32.** Configured automatic backup scheduler (`backup.json` with cron schedule and retention)
- **#33.** Used `text.levenshteinDistance()` for fuzzy entity matching (server-side fuzzy matching in canonicalization)
- **#34.** Added EXPLAIN/PROFILE tooling for query plan validation (health-check asserts no full scans on critical paths)

### Architecture (Native-First) Implemented

- **#60.** Made DoclingDocument the authoritative mutable artifact through the pipeline
- **#61.** Replaced custom chunker with native Docling chunkers
- **#62.** Used Docling's native enrichment path for translation and picture descriptions
- **#66.** Compiled query-profile traversals into native MATCH instead of generic undirected walk
- **#67.** Enriched GraphStore result model to carry native ArcadeDB semantics
- **#68.** Pushed retrieval filters into native queries
- **#69.** Returned enriched graph context from ontology expansion traversal
- **#71.** Decided: hybrid approach for Docling-Graph templates (auto-generate from YAML, conform to canonical patterns). Decision feeds into #63.

### Completed During Migration (Pre-merge Fixes)

- Fixed `set_vertex_embedding` signature mismatch between Protocol and implementation
- Fixed Protocol encapsulation violations (direct `_client`/`_database` access) -- added proper methods
- Fixed `upsert_relationships_batch_sync` dropped `record.properties` (copy-paste bug)
- Fixed Redis client leak in `community_tasks.py` (no `r.close()`)
- Fixed direct `os.environ.get` bypassing settings in community module
- Parallelized `cross_model_search` and `get_graph_stats` sequential queries with `asyncio.gather`
- Removed stale references (GraphRAG, Neo4j, Qdrant, ChunkRef, AGE) across 8 files
- Removed narrating step-number comments in `delete_document_graph` and `sync_schema_from_ontology`
- Fixed `delete_document_graph` to remove document_id from relationship edge document_ids lists and delete empty-list edges

---

## Verbatim Reviews (Reference)

### Verbatim Native-First Review (2026-04-06)

> No code changes made. The branch does call native `DocumentConverter`, Docling-Graph `run_pipeline()`, and ArcadeDB `MATCH`/`vectorNeighbors()`, but the dominant pattern is to immediately flatten or wrap those native objects into app-specific contracts and then reimplement upstream behavior in Python.
>
> **Findings**
>
> 1. Critical: `DoclingDocument` is not kept as the authoritative working object after conversion. The Docling wrapper converts natively, then immediately flattens the result into custom `ConvertedElement`/`ExtractedChunk` DTOs and later stages mutate `DocumentElement` rows plus hand-built markdown instead of mutating and reserializing the `DoclingDocument` itself. This diverges from Docling's documented post-conversion mutation/enrichment flow and is the root cause of the stale JSON vs current-text split.
>
> 2. High: native Docling chunkers are bypassed in favor of a custom heuristic chunker over flattened rows. Upstream Docling documents two native approaches and explicitly recommends chunking directly from `DoclingDocument` via `BaseChunker`/`HybridChunker`; this branch instead uses `structure_aware_chunk()` with approximate token counting and manual overlap logic.
>
> 3. High: Docling's own enrichment path is replaced with custom translation and picture-description stages that do not update the canonical Docling JSON. The converter explicitly disables picture description, then later stages write translated text to `DocumentElement.translated_text`, append image descriptions to markdown, and inject `_enriched_text` into the extraction payload. Native Docling examples instead mutate the `DoclingDocument` and regenerate output from that object.
>
> 4. High: Docling-Graph template authoring is replaced by a custom ontology-to-Pydantic generator. The upstream docs show explicit Pydantic models using `graph_id_fields`, `is_entity=False` for components, and `edge()` for relationships, while this branch auto-derives identity fields, never emits component semantics, and encodes edges with `json_schema_extra={"edge_label": ...}`.
>
> 5. High: the wrapper introduces a custom "unified template" container instead of using canonical authored template classes. This is a custom accommodation of Docling-Graph's singular `PipelineConfig.template` surface, but it moves semantics out of explicit domain models into a generated wrapper model with list fields for every entity type.
>
> 6. High: native Docling-Graph graph output, provenance, and metadata are normalized into an app-specific `{entities, relationships}` contract and then partially discarded. The client flattens the node-link graph, the pipeline keeps only `properties`, ignores most response `metadata`, and rebuilds mention grounding with regex matching instead of consuming upstream provenance/resolver output.
>
> 7. High: ArcadeDB is not the primary structure graph for retrieval even though the pipeline writes structural edges there. The main hybrid expansion path reads Postgres `retrieval.chunk_links` first and only falls back to ArcadeDB neighborhood traversal for legacy cases.
>
> 8. High: query profiles and dossiers do not compile their own traversal model into native ArcadeDB `MATCH` traversals. The profile schema supports directed multi-step paths with `direction`, `min_hops`, and `max_hops`, but execution collapses that into one generic undirected neighborhood walk using `.both(...)`.
>
> 9. Medium: the backend-agnostic `GraphStore` abstraction strips native ArcadeDB result semantics and encourages lowest-common-denominator query usage.
>
> 10. Medium: retrieval filters are handled in Python after search/expansion instead of being pushed into native Postgres/ArcadeDB queries, even though SQL-side filter builders already exist.
>
> 11. Medium: evidence and ontology expansion are routed through custom helper contracts instead of native ArcadeDB graph semantics.
>
> 12. Low: native integration coverage is weak, so many of these custom-vs-native drifts are not protected by tests.
>
> **Handoff Priorities**
> 1. Make `DoclingDocument` the authoritative mutable artifact through translation and picture enrichment, then regenerate markdown/JSON from it.
> 2. Replace the custom Docling chunker where feasible with native Docling chunkers, or document why retrieval chunking must intentionally differ.
> 3. Decide whether Docling-Graph templates should be explicit canonical Pydantic models instead of generated wrappers.
> 4. Preserve Docling-Graph provenance and metadata instead of rebuilding mention grounding in Python.
> 5. Move retrieval, query-profile traversal, and dossier evidence onto native ArcadeDB graph traversal instead of Postgres-first/generic-wrapper execution.
>
> **Sources**
> - ArcadeDB Manual.pdf
> - https://docling-project.github.io/docling/concepts/chunking/
> - https://docling-project.github.io/docling/examples/translate/
> - https://docling-project.github.io/docling/examples/enrich_doclingdocument/
> - https://ibm.github.io/docling-graph/usage/examples/docling-document-input/
> - https://ibm.github.io/docling-graph/fundamentals/schema-definition/entities-vs-components/

---

### Verbatim Retrieval Query Mechanisms Review (2026-04-06)

> ArcadeDB usage itself is mostly canonical here; the main problems are in the custom wrappers around it. I checked this against ArcadeDB Manual.pdf. Docling and Docling-Graph are upstream to graph quality, but they are not directly in the active query path, so the retrieval review is mostly an ArcadeDB and application-layer review.
>
> **Findings**
>
> 1. High: query profiles do not honor their own traversal model. The schema supports per-step `direction`, `min_hops`, and `max_hops` at query_profiles.py (line 13), but execution collapses a profile to `rel_types + max_depth` and runs one undirected neighborhood traversal at query_profiles.py (lines 556, 569). The backend traversal is `MATCH ... .both(...)` at arcadedb_graph.py (line 792). So profiles are effectively "any of these rel types within N hops," not the configured directed path model.
>
> 2. High: dossier and query-profile evidence attachment is wired through a chunk-centric helper using entity IDs. Both services call `get_ontology_linked_chunks(item.node_id)` for entities at dossier_service.py (line 338) and query_profiles.py (line 645). But `get_ontology_linked_chunks()` is explicitly implemented as `chunk <- EXTRACTED_FROM - entities -> other chunks` at arcadedb_graph.py (line 856). That is correct for a seed chunk, not for an entity RID, so dossier/query-profile evidence is likely empty or wrong.
>
> 3. High: the hybrid cross-modal fallback for legacy documents is probably broken by UUID/RID mismatch. `_expand_via_cross_modal()` passes `str(seed.chunk_id)` into `get_neighborhood()` at retrieval.py (lines 339, 646). `get_neighborhood()` then interpolates that value directly into `@rid = {node_id}` at arcadedb_graph.py (line 792). Seed chunk IDs are UUIDs, not ArcadeDB RIDs, and this path does not resolve them first.
>
> 4. High: hybrid "ontology expansion" is not preserving enough native graph information. `_expand_via_ontology()` expects `target_chunk_type`, `rel_type`, and entity context at retrieval.py (line 710), but `get_ontology_linked_chunks()` returns raw chunk rows from `expand(out('EXTRACTED_FROM'))` at arcadedb_graph.py (line 889). That means relation typing collapses to the fallback `RELATED_TO`, entity names are blank, and image chunks default to `text_chunk` lookup and can be dropped.
>
> 5. Medium: retrieval filters are only partially implemented, and only after search/expansion. The request schema still exposes `classification`, `modalities`, `source_ids`, and `document_ids` at retrieval.py (line 37). But `unified_query()` applies filters only after strategy execution at retrieval.py (line 78), and `_apply_query_filters()` explicitly skips `source_ids` at retrieval.py (line 56). SQL-side filter builders exist at _retrieval_helpers.py (line 120), but they are not used by the live retrieval path.
>
> 6. Medium: the API defaults do not match the advertised retrieval settings. `UnifiedQueryRequest.top_k` defaults to `10` and `min_confidence` defaults to `None` at retrieval.py (line 58), while `/v1/settings/retrieval` exposes `20` and `0.1` from config at retrieval.py (line 1124) and config.py (line 152). Raw API clients that omit these fields do not get the documented defaults.
>
> 7. Medium: graph query responses surface the wrong score semantics and strip too much node data. Fulltext search is correctly ordered by Lucene `$score` at arcadedb_graph.py (line 328), but `_to_entity()` ignores `$score` and only maps vector distance or `extraction_confidence` at arcadedb_graph.py (line 53). `/graph/query` then returns `match.extraction_confidence` as the score at graph_store.py (line 66). `/graph/neighborhood` also returns minimal node/edge payloads rather than fuller native properties at arcadedb_graph.py (line 833).
>
> 8. Low: co-extracted discovery exists in the backend, but it is not actually wired into the query stack. The backend method exists at arcadedb_graph.py (line 921), and `resolve_root_entity()` in query profiles even comments about a "co-extracted fallback" at query_profiles.py (line 536). But the implementation falls back directly to `resolve_root_entity()` exact lookup and never calls `get_co_extracted_entities()`.
>
> 9. Low: verification is weaker than the branch suggests. The concrete ArcadeDB retrieval path is not what most passing tests validate, because the shared fixture replaces GraphStore with mocks at conftest.py (line 206). There are also stale tests that still assert `response["mode"]` at test_retrieval_api.py (line 15) and test_full_pipeline.py (line 66), while the live response shape uses `strategy` and `modality_filter` at retrieval.py (line 114).
>
> **Native-First Direction**
> - Keep the current native ArcadeDB primitives. `vectorNeighbors(...)->expand(...)` and `MATCH` traversal are the right foundation.
> - Generate native `MATCH` from query-profile traversal steps instead of collapsing them into a custom undirected neighborhood wrapper.
> - Split chunk-centric expansion from entity-centric evidence lookup instead of forcing both through `get_ontology_linked_chunks()`.
> - Push filters into native Postgres/ArcadeDB queries where possible instead of post-filtering in Python.
> - Surface native relevance where it exists. For graph fulltext, that means carrying Lucene `$score` through instead of substituting extraction confidence.

---

### Verbatim Graph Extraction Pipeline Review (2026-04-06)

The following is the complete graph extraction review for reference when addressing items #45-#50.

> **Graph Extraction Review**
>
> The stage ordering is sound: extract entities/relationships and chunks in parallel, wire only after both exist, then canonicalize. The implementation problem is not the DAG shape. It is that the graph extraction contract has drifted across the worker, the Docling-Graph service, and the downstream consumers.
>
> **Stage 1 is built against a stale Docling-Graph API.**
> derive_ontology_graph() reconstructs full_text from DocumentElement rows and calls extract_graph_all(full_text, document_id) at pipeline.py (line 2226) and pipeline.py (line 2257). The client sends text in docling_graph_service.py (line 172), but the service now requires docling_document_json in schemas.py (line 10) and consumes that in main.py (line 141). The service also returns graph and metadata, not entities and relationships, but the worker still reads result.get("entities") and result.get("relationships") at pipeline.py (line 2258). If this code path is live, successful extraction is likely being interpreted as zero nodes and zero edges.
>
> **The pipeline already has the canonical structured artifact, but the graph stage ignores it.**
> prepare_document persists docling_document.json to object storage at pipeline.py (line 915). Another stage later downloads that same JSON at pipeline.py (line 1401). derive_ontology_graph() does neither. It rebuilds plain text from normalized elements instead. That throws away layout, structure, and native provenance before the extraction service even runs.
>
> **The current "primary" mention path is not LLM grounding; it is lexical matching.**
> _build_entity_mentions() uses word-boundary regex for short names and substring matching for longer names at pipeline.py (line 2116). It does not resolve paraphrase, coreference, metonymy, abbreviation expansion, or implicit references. So your #27 concern is valid, but more precisely: the implementation never attempts semantic mention grounding in the first place.
>
> **Partial mention miss cases are never repaired.**
> derive_structure_links() only falls back to artifact-wide entity->chunk linking when mentions is empty, not when mentions are incomplete, at pipeline.py (line 2696) and pipeline.py (line 2708). That means if lexical matching finds 2 real mentions and misses 3 implicit ones, the stage keeps the 2 and silently drops the 3. The fallback does not help recall unless the primary path fails completely.
>
> **The entity->chunk wiring is text-centric, not fully element-centric.**
> The element_uid -> chunk_id map is built only from text_chunks via artifact_id at pipeline.py (line 2672). image_chunks are wired to the document and same-page neighbors, but they are not included in the mention map used for EXTRACTED_FROM. So even if extraction identifies entities grounded in image/schematic elements, the primary wiring path does not appear to attach them to image chunks.
>
> **Even perfect EXTRACTED_FROM edges would not currently pay off fully, because traversal is broken downstream.**
> The graph writer creates EXTRACTED_FROM as entity -> chunk at arcadedb_graph.py (line 1453), but lookup traverses in('EXTRACTED_FROM') in arcadedb_graph.py (line 724). Retrieval also passes chunk UUIDs into that helper at retrieval.py (line 314) and retrieval.py (line 696), while the helper interpolates directly into ArcadeDB SQL as if it were a RID. So the downstream consumers you listed are currently degraded both by missing edges and by broken traversal semantics.
>
> **Canonicalization is in the right place in the DAG, but the implementation is not giving the graph a reliable dedup pass.**
> derive_canonicalization runs after wiring, which is the correct point architecturally. But canonicalize_document_entities() discovers "document entities" by calling fulltext_search_sync(document_id) at canonicalization.py (line 64), and that search is just WHERE name LUCENE :query at arcadedb_graph.py (line 1568). Alias lookup is also broken on alias vs alias_name. So the dedup stage is not operating on solid document-local provenance.
>
> **What I'd Tell Another Agent**
>
> Reconcile the Stage 1 contract first.
> The worker, HTTP client, and service do not agree on request or response shape. Until that is fixed, all discussion about mention precision is second-order.
>
> Make derive_ontology_graph() consume persisted docling_document.json.
> The canonical structured document already exists. Using reconstructed full_text is both lossy and out of sync with the actual service API.
>
> Normalize the extraction output shape before import.
> The worker needs a stable adapter from Docling-Graph output to nodes/edges/provenance records. Right now it assumes an old shape.
>
> Reassess mention grounding only after the contract is stable.
> At that point the real question becomes whether to keep lexical mention building, enrich it with LLM/entity-resolution logic, or consume provenance directly from Docling-Graph output if available.
>
> Fix traversal before measuring graph-extraction quality.
> Otherwise better entity->chunk edges will still not show up properly in retrieval, dossier evidence, or ontology expansion.
>
> **Test Gaps**
>
> The client tests still assert the old text-based interface in test_docling_graph_client.py (line 119).
> The docling-graph integration tests are stale and patch a removed _templates global at test_pipeline_integration.py (line 43).
> There is no strong end-to-end test that proves: DoclingDocument JSON -> Docling-Graph extraction -> graph_json mentions -> EXTRACTED_FROM edges -> retrieval traversal all agree on the same identifiers and schema.
>
> My bottom-line assessment is: the architecture is defensible, but the current graph extraction process is not trustworthy until the worker/service contract is repaired. After that, #27 becomes a meaningful optimization target; before that, it is not the primary blocker.

---

### Verbatim Code Analysis and Review (2026-04-05)

The following is the complete standalone review for reference when addressing the above items.

> Standalone review of the current implementation only; no base-branch comparison.
>
> **Findings**
>
> 1. Critical: the "clone/update upstream repos on every build" contract is not actually satisfied for Docling or Docling-Graph, and it is not part of `docker compose build` itself. [manage.sh](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/manage.sh#L123), [manage.sh](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/manage.sh#L174), [docker-compose.yml](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker-compose.yml#L111), [docker-compose.yml](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker-compose.yml#L143), [docker/arcadedb/Dockerfile](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/arcadedb/Dockerfile#L6), [docker/docling/Dockerfile](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling/Dockerfile#L102), [docker/docling-graph/Dockerfile](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/Dockerfile#L83), [docker/docling/requirements.txt](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling/requirements.txt#L1), [docker/docling-graph/requirements.txt](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/requirements.txt#L5). Only ArcadeDB consumes a cloned `repo/`; Docling and Docling-Graph install released packages. A plain `docker compose build` clones nothing, and even `manage.sh` continues on `git pull` failure with only a warning at [manage.sh](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/manage.sh#L115). This also conflicts with the repo's stated verification/design contract in [VERIFICATION_CHECKLIST.md](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/VERIFICATION_CHECKLIST.md#L236) and [2026-04-04-arcadedb-migration-design.md](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docs/superpowers/specs/2026-04-04-arcadedb-migration-design.md#L395).
>
> 2. Critical: `vector_search()` drops the chunk metadata that retrieval depends on. [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L858), [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L69), [app/api/v1/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/api/v1/retrieval.py#L449), [app/api/v1/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/api/v1/retrieval.py#L523). The API expects `chunk_id`, `document_id`, `artifact_id`, `modality`, and chunk text in `hit.properties`, but the ArcadeDB query only projects entity fields plus distance/RID. The ArcadeDB manual says `expand(vectorNeighbors(...))` returns all document properties; the current query throws those away.
>
> 3. Critical: ontology/evidence traversal is broken by both edge direction and identifier mismatch. EXTRACTED_FROM edges are written entity -> chunk in [app/workers/pipeline.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/workers/pipeline.py#L2742) and [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1453), but lookup traverses `in('EXTRACTED_FROM')` in [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L730). Retrieval also passes `str(seed.chunk_id)` into that helper in [app/api/v1/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/api/v1/retrieval.py#L314) and [app/api/v1/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/api/v1/retrieval.py#L326), but the helper interpolates directly into `FROM {node_id}`, which is RID-oriented SQL. Impact: ontology expansion and evidence attachment are likely empty or reversed in [app/services/query_profiles.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/query_profiles.py#L659) and [app/services/dossier_service.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/dossier_service.py#L352).
>
> 4. High: alias resolution is internally inconsistent and likely nonfunctional against ArcadeDB. Alias schema/creation use `alias_name` in [app/services/arcadedb_schema.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_schema.py#L60) and [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L786), but lookup queries `WHERE alias = :alias` in [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L813) and [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1603). The optional `entity_type` filter is also applied on the `Alias` vertex query itself. That directly affects root resolution in [app/services/query_profiles.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/query_profiles.py#L519), [app/services/dossier_service.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/dossier_service.py#L166), and canonicalization in [app/services/canonicalization.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/canonicalization.py#L129).
>
> 5. High: the docling-graph wrapper is not following canonical docling-graph template usage. [docker/docling-graph/app/main.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/app/main.py#L56), [docker/docling-graph/app/main.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/app/main.py#L93), [docker/docling-graph/app/template_builder.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/app/template_builder.py#L101), [docker/docling-graph/app/template_builder.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/app/template_builder.py#L218), [config.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/.venv/lib/python3.12/site-packages/docling_graph/config.py#L113). The library expects a singular `PipelineConfig.template`; this service builds many templates and passes only the first one. The generated models also use `graph_id_fields` but not the documented `is_entity=True`/`edge()` pattern. That is a real misalignment with docling-graph's canonical API surface, not just a style difference.
>
> 6. High: retrieval `filters` are still public API but are silently ignored. [app/schemas/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/schemas/retrieval.py#L64), [app/schemas/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/schemas/retrieval.py#L37). `app/api/v1/retrieval.py` does not read `body.filters` anywhere, so callers can request classification/document/modality constraints and get unfiltered results with no warning.
>
> 7. Medium: document canonicalization is not actually document-scoped. [app/services/canonicalization.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/canonicalization.py#L68), [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1582). The code claims it finds entities linked to a document, but it really does `WHERE name LUCENE :query` using the document ID string. That makes the canonicalization pass logically disconnected from document/chunk/entity linkage.
>
> 8. Medium: orphan cleanup likely uses the wrong ArcadeDB system field. [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1107), [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1527). The manual defines `@class` as the type name and `@cat` as the type category, so filtering `@cat NOT IN ['Document', 'TextChunk', 'ImageChunk', 'Alias']` is almost certainly targeting the wrong field.
>
> 9. Medium: community-report vector search loses similarity scores. [app/services/arcadedb_graph.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/services/arcadedb_graph.py#L1035), [app/api/v1/retrieval.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/app/api/v1/retrieval.py#L993). The search query does not project distance/score, but global retrieval assumes `score` exists, so synthesis metadata collapses to `0.0`.
>
> 10. Medium: the highest-risk integration points are not currently verified. GraphStore is globally stubbed in [tests/conftest.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/tests/conftest.py#L212), so passing backend/query-profile tests do not validate the concrete ArcadeDB alias/chunk/vector behavior. The docling-graph integration suite currently fails because it still patches `_templates` at [docker/docling-graph/tests/test_pipeline_integration.py](/home/josh/development/EIP-MMDPP/.worktrees/arcadedb/docker/docling-graph/tests/test_pipeline_integration.py#L43), which no longer exists.
>
> **Verification**
> - Backend suites passed: `test_arcadedb_graph`, `test_query_coverage`, `test_community_tasks`, `test_startup_bootstrap`, plus `test_query_profiles` and `test_query_profiles_api`.
> - Those passing suites are not strong evidence for the concrete backend behavior because the shared fixture replaces GraphStore methods with mocks.
> - `docker/docling-graph` tests partly pass, but `tests/test_pipeline_integration.py` currently errors at setup because the test harness is stale relative to `app.main`.
>
> **Sources**
> - [ArcadeDB Manual.pdf](/home/josh/development/EIP-MMDPP/ArcadeDB%20Manual.pdf)
> - [IBM/docling-graph README](https://raw.githubusercontent.com/IBM/docling-graph/main/README.md)
> - [docling-project/docling README](https://raw.githubusercontent.com/docling-project/docling/main/README.md)
> - [ArcadeData/arcadedb repo](https://github.com/ArcadeData/arcadedb)

---

### Code Quality Cleanups

**#86. Replace stringly-typed StageRun status / skip_reason values with enums**
**Status:** Open. Dedicated cleanup PR — should NOT be mixed with feature work.
**Files:** `app/workers/pipeline.py` (heavy), `app/models/ingest.py` (StageRun + PipelinePassOutput), test files.

**Observation:**
StageRun.status, StageRun.execution_status, StageRun.skip_reason, and PipelinePassOutput.execution_status are `String(N)` columns populated with raw string literals scattered across pipeline.py. The strings are matched at multiple call sites:
- Status values: `"PENDING"`, `"RUNNING"`, `"COMPLETE"`, `"FAILED"` (legacy celery-level column)
- Execution status: `"COMPLETE"`, `"SKIPPED"`, `"FAILED"`
- Skip reasons: `"NO_UPSTREAM_ENDPOINTS"`, `"EMPTY_ANCHOR_SET"`, `"disabled"`, `"no_elements"`, `"no_markdown"`, `"no_text_elements"`, `"no_markdown_with_text_elements"`

A casing inconsistency (uppercase `"EMPTY_ANCHOR_SET"` for the column vs lowercase `"empty_anchor_set"` in the metrics JSON) was caught during 2026-05-10 review — exactly the class of bug enums prevent.

The doc-level status already uses module constants (`STATUS_PROCESSING`, `STATUS_COMPLETE`, `STATUS_PARTIAL_COMPLETE`, `STATUS_FAILED`, `STATUS_PENDING_REVIEW`). Extending the same pattern to StageRun/PipelinePassOutput is the natural next step.

**What needs to be done:**
1. Add `class StageRunStatus(str, Enum)`, `class ExecutionStatus(str, Enum)`, `class SkipReason(str, Enum)` (or string-constant equivalents) in a new `app/models/enums.py` (or alongside the existing `STATUS_*` constants in pipeline.py).
2. Update `_write_stage_run`, `_update_stage_run`, `check_required_pass_gate`, and the synthetic-StageRun short-circuit to accept/emit enum values.
3. Replace every raw-string literal in SQL strings, dict literals, and gate `set` membership checks with the enum.
4. Update tests that hardcode `"COMPLETE"`/`"SKIPPED"`/etc.
5. Decide whether to migrate the DB columns to `Enum(...)` (stricter, requires alembic) or keep `String(N)` + Python-side validation (simpler).

**Why this matters:**
- Catches casing/typo mismatches at type-check time.
- Single source of truth for the authorized skip-reason set; today it's duplicated between writers and the gate's `set` literal.
- Better IDE autocompletion and refactor safety.

**Acceptance:**
- No raw `"COMPLETE"`/`"SKIPPED"`/`"FAILED"` string literals in pipeline.py outside the enum definitions.
- `check_required_pass_gate` uses an enum-backed authorized set.
- Adding a new authorized skip reason is a one-line enum change, not a multi-file find-and-replace.
- Existing tests pass without touching their string literals if the enum is `StrEnum` (auto-coerces); otherwise tests are updated.
