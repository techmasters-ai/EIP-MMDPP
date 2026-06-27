# Structured-Output / Constrained-Decoding Review — verified findings + deferred remediation

Status: **REVIEW ARTIFACT — verified, remediation DEFERRED.** No code changed.
Date: 2026-06-18
Verified against: live worktree `.worktrees/walltime-c0-telemetry` (the deployment bind-mounts it).
Origin: a ~2-week-old analysis (grounded in a structured-outputs best-practices doc: 4 methods + common-failures + perf-tips) was re-verified claim-by-claim against current code; stale items corrected below.
Related: `production-reliability-2026-06.md` §R3 (truncation / json_schema migration) — this is the deep-dive for that deferred item.
Pick-up point: after the chunk-selection mechanism is validated live + merged to main, turn the "Remediation plan" below into an **A/B test plan** (see "Next action").

## TL;DR

Schema-constrained decoding is **inert in production**. The extraction schema is also a **generic IR**, so even if turned on, a grammar would only enforce the JSON *envelope*, not field values. The highest-ROI fixes are (1) actually send a constraint on the live `/v1` path, and (2) give the grammar typed fields to bite on. Both are deferred behind an A/B harness.

## The reframe (why method order matters here)

The best-practices doc's examples are *tight* schemas (`Literal[...]`, bounded `float`), where grammar can reject any invalid token. **Our live extraction schema is the opposite — a generic graph IR:**

- `DeltaNode.properties: dict[str, Any]` (open map), `ids: dict[str,str]`, `property_evidence: dict[str,list[str]]` — `…/repo/docling_graph/core/extractors/contracts/delta/models.py:29-77`
- The schema sent to the model is `DeltaGraph.model_json_schema()` — `…/delta/orchestrator.py:195`
- The actual entity/field structure (enums, numeric specs, designations) lives in the **prompt's semantic guide / catalog**, not the Pydantic types.

→ A grammar built from `DeltaGraph` can only enforce `{nodes:[…], relationships:[…]}` + each node's key set; `properties: dict[str,Any]` decodes as "any JSON object." This single fact determines which levers pay off and in what order.

## Verified findings (claim → verdict → evidence)

| # | Claim | Verdict | Evidence (live worktree) |
|---|-------|---------|--------------------------|
| 1 | Extraction schema is a generic IR (`dict[str,Any]` properties, no typed enums/numerics) | **TRUE** | `…/delta/models.py:34-38,76-77`; schema = `DeltaGraph.model_json_schema()` `orchestrator.py:195` |
| 2 | Live client sends Ollama-native `format` to OpenAI-compat `/v1/chat/completions` → constrained decoding inert | **TRUE** | `app/ollama_pool_client.py:730` (POST `/v1/chat/completions`), `:617-636` (`body["format"]`) |
| 2b | **(NEW, missed by the original analysis)** `force_json_mode=True` default sends `format="json"` unconditionally and short-circuits the schema-size gate — decoding is *doubly* unconstrained | **TRUE** | default `config.py:32`; short-circuit `ollama_pool_client.py:619-621` |
| 3 | `normalize_schema_for_response_format` exists but is dead on the live path (LiteLLM bypassed) | **TRUE (dead)** | def `…/repo/docling_graph/llm_clients/schema_utils.py:11-39`, sole caller `llm_clients/litellm.py:286-291`; bypass `app/main.py:44-49` |
| 4 | `num_ctx` never set on the wire (can't on /v1, no Modelfile, `OLLAMA_CONTEXT_LENGTH` unset) → Ollama default context → silent input-truncation risk | **TRUE** | no `num_ctx`/`options` in `ollama_pool_client.py:600-637`; no Modelfile in repo |
| 4-caveat | `OLLAMA_NUM_CTX`/`ollama_num_ctx` exists but is **only** client-side prompt-char budgeting (`max_chars = ollama_num_ctx*3`), never sent to Ollama — don't mistake its presence for context being set | **TRUE** | `.env.example:115`, `app/config.py`; use in `document_analysis.py` |
| 4b | A schema-size threshold gate (`structured_output_threshold_chars`, default 20000) chooses `format="json"` vs `format=<schema>`; it measures the small `DeltaGraph` schema, not the ~20k prompt — and is currently moot (2b short-circuits it) | **TRUE (moot)** | `ollama_pool_client.py:629-636`; `ontology_bundles/_shared/limits.py:21-26` |
| 5 | "Lift `Literal` enums straight from `ontology.yaml`'s `enum:`" | **STALE (prod)** | `air_defense_v3/ontology.yaml` **deleted** (commit `59e0dbe`, Pydantic-SSoT/Task 51). Enums now in Python: `ontology_bundles/air_defense_v3/relationships.py:20-60` (`RelationshipType(str,Enum)`), field enums via `entities.py` `json_schema_extra={"enum":[...]}`. `ontology.yaml` survives only in non-default bundles (incl. the regression subset). |
| 5b | Flagship example `classification: Literal["UNCLASSIFIED",…]` is liftable today | **FALSE** | classification levels are **not a formal enum** anywhere — only a runtime validation set + non-constraining examples (`document_analysis.py:78`). Would have to be *created*. |
| 6 | Retry injects the prior `ValidationError` into the next prompt (not blind retry) | **TRUE** | `…/delta/orchestrator.py:210-222` (`=== FIX ===` block), errors collected `:249-253` |
| 7 | `DOCLING_GRAPH_LLM_THINK` exists; `think:true` returns empty content on `/v1` (only `reasoning_content`) | **TRUE** | flag `app/ollama_clients.py:80`; send `ollama_pool_client.py:546-549,607-609`; empty-content raise `:1192-1205`; default `false`/`None` |
| 8 | Per-field `examples=[...]` injected, but **no single complete worked input→output exemplar** | **TRUE** | per-field `schema_utils.py:73-75`, per-path `delta/schema_mapper.py:73-77`; prompt has only the shape template `delta/prompts.py:68-74` |
| 9 | A salvage/repair stack absorbs malformed JSON (response_format would reduce its load) | **TRUE** | docling-graph `response_handler.py:32-104` + `:423-578` (7 repair strategies); worker `app/services/llm_json.py:64-111` (5 strategies). Validation = `DeltaGraph.model_validate` `orchestrator.py:235`. |

## Corrections to the original analysis

1. **Stronger than stated:** decoding isn't merely "format ignored on /v1" — `force_json_mode=True` (default) sends `format="json"` and short-circuits before the schema gate. **Step 1 must flip `force_json_mode` too**, not just swap `format`→`response_format`.
2. **`ontology.yaml` is stale for prod (Claim 5).** The SSoT is now the Pydantic entity/relationship models. *Silver lining:* this makes Step 2 (typed per-pass schemas) **more tractable** — the enums the analysis wanted to "generate from the ontology" already exist in Python (`RelationshipType`, `entities.py` `json_schema_extra`). The mechanism changes (read Pydantic SSoT, not YAML), not the goal.
3. **`classification` enum doesn't exist yet** — the analysis's flagship Step-2 example must be authored, not lifted.
4. **`num_ctx` caveat:** `OLLAMA_NUM_CTX` exists but is client-side char budgeting only; verify true server context separately (`OLLAMA_CONTEXT_LENGTH` server-side or a Modelfile `num_ctx`).

## Remediation plan (DEFERRED — A/B each step)

| Step | Lever | Action (corrected) | Risk | Payoff |
|---|---|---|---|---|
| 0 | prereq | Verify effective server `num_ctx`; stand up a thin A/B harness over a baseline bundle (e.g. `air_defense_v3_baseline_subset`) diffing recall / coverage / parse-failure / empty-record across: current `format` vs `response_format json_object` vs `json_schema` | none | makes every later step provable, not hopeful; rules out silent truncation masking results |
| 1 | Method 2 (constrain) | On the `/v1` path, send OpenAI `response_format` (lift the dead `normalize_schema_for_response_format` logic) **and flip `force_json_mode`** so the `DeltaGraph` schema actually constrains the envelope | low | kills malformed-envelope failures; reduces the salvage-stack (#9) load; schema-size gate (#4b) starts mattering |
| 2 | Method 3 (tight schema) | Add concrete typed fields per pass from the **Pydantic SSoT** (not ontology.yaml): `Literal` enums (`RelationshipType`, field `json_schema_extra` enums), numeric/`QuantityWithUnit`. **Architecture gate:** generate per-pass concrete models vs keep generic IR with envelope-only enforcement. Scope to highest-value passes (enums + numeric specs) first. Author the missing `classification` enum if wanted. | med | the real grammar leverage `dict[str,Any]` currently throws away |
| 3 | perf-tip (few-shot) | Add one complete worked document-snippet→exact-`DeltaGraph`-JSON exemplar per contract — strongest lever for the generic-IR parts grammar can't touch (#8) | low | "dramatically improves reliability" where the schema is loosest; A/B the token cost vs recall gain |
| 4 | Method 4 (retry/think) | Retry already injects validation feedback (#6) — optionally add a small temp bump (0.1→0.25) on the **final** retry only. For `think` on hard passes, route through `/api/chat` (native) where `think` works — never flip it on the `/v1` path (#7 empty-output trap) | low | marginal robustness |

### Skip
- `extra="forbid"` globally (Pydantic ignores extras by default — safer for LLM output; use `forbid` only in tests to catch schema/prompt drift).
- Deeper nesting / bigger single-call schemas — would regress the existing token-batching.

## Next action (per the owner's sequence)
Deferred until: shadow generalization eval holds → chunk-selection flipped live → validated on small docs → **all code committed + merged to main**. THEN circle back here and convert the Remediation plan into a concrete A/B test plan (start with Step 0 harness, which makes Steps 1–3 measurable).
