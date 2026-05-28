# Schema-Wide Retrieval/Routing Upgrade Implementation Plan

> **For agentic workers:** REQUIRED — use `superpowers-extended-cc:subagent-driven-development` (preferred) or `executing-plans` to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking. The implementation decisions are locked in this document; do not stop for additional discussion gates unless code inspection reveals a new blocker that cannot be resolved from repo context.

**Goal:** Convert single-query merged-chunk routing into a structured, multi-channel retrieval system with per-field queries, lexical/pattern channels, table-aware chunks, and a worker→docling-graph selected-chunk handoff that delivers the LLM the exact merged chunks the router selected.

**Architecture:** Adds structured retrieval signals (entity query + per-field queries + lexical aliases + negative terms + section hints + evidence patterns) generated from extraction-schema Pydantic field metadata (`json_schema_extra={"retrieval": ...}`). Multi-channel candidate generation merges dense + lexical + pattern signals: **dense similarity drives recall**, the **cross-encoder reranker orders by semantics**, and a per-field **keyword/pattern precision boost is applied AFTER rerank, before the top_k cut** (keyword presence is a strong precision signal that the current pre-rerank-blend design throws away at final selection). The extraction LLM then receives only the fields that had retrieval evidence (subset-schema extraction, opt-in). Selected merged chunks flow worker→docling-graph as a first-class request field (`ExtractPassRequest.selected_chunks`), so the LLM receives the exact merged chunk text the router/reranker scored instead of a worker-side scoped-document reconstruction. (Note: this is a *fidelity* change, not a wire-payload reduction — `docling_document_json` remains a required request field; the win is skipping worker-side `apply_chunk_scope` CPU and the receiver's internal re-chunking, plus exact text parity.) Table-aware chunk rendering makes table rows discoverable by lexical/pattern channels.

**Tech Stack:** Python 3.11, Pydantic v2 (model_fields + json_schema_extra + ConfigDict), FastAPI router endpoint (`/v1/extraction/chunk-scope`), ArcadeDB ExtractionChunk vertices + b-tree index on `pipeline_run_id`, bge-m3 embeddings via Ollama (`embed_texts` query-prefixed), bge-reranker-v2-m3 cross-encoder (`rerank()`), gemma4:31b extraction LLM via `OllamaChatClient` pool.

**Branch:** `walltime/c0-telemetry` (this worktree). Backward compatible: `EXTRACTION_INDEX_MODE` default is now `merged` (flipped in `48302f2`); the binding gate that keeps production behavior unchanged is `VECTOR_ROUTER_MODE=shadow` (still the `.env.example`/`config.py` default). New behavior activates on the single remaining flip `VECTOR_ROUTER_MODE=narrow_only`. (Note: this worktree's live `.env` already has `narrow_only` + `merged`, so both gates are active here.)

---

## Review Revisions (2026-05-28)

This plan was revised after a three-dimension subagent review (gaps/performance, technical correctness, code quality) that audited every claim against the code in this worktree. Folded-in fixes, with their origin:

- **[Blocker 1 — Technical]** `MergedCandidate`/`merge_candidates` must read chunk fields from `GraphEntityResult.properties` via the existing `read_chunk_*` accessors — they are NOT attributes on `GraphEntityResult`. (Task C4.)
- **[Blocker 2 — Technical/Quality]** The builder test file is `tests/unit/test_extraction_query_builder.py` (NOT `test_retrieval_query_builder.py`, which does not exist). That file is **already red on this branch** (5 failures); a new pre-task **B0** re-baselines them before Phase B work begins.
- **[Blocker 3 — Gaps]** The `c10` recall harness defines ground truth from the dense+reranker ordering and therefore **cannot** detect lexical/pattern/table recall gains. The blocking merge gate is now an **executed end-to-end extracted-entity-count comparison** on `merged_v1`; the `c10` sweep is retained only as a non-gating coefficient-tuning tool. (Test Plan + Acceptance #2.)
- **[Blocker 4 — Gaps, resolved per user direction]** The retrieval metadata (B3/B4) AND config defaults (B6) apply to **all four bundles, including the production `air_defense_v3`**, which routes all 9 field-group passes. The narrowed bundles (`merged_v1`, `narrowing_v1`, `baseline_subset`) carry only `radar_power_rf` + `missile_kinematics`; production `air_defense_v3` exercises all 9 — so none of the annotated schemas are dead metadata. Recall/wall acceptance is measured per bundle under test; running multi-channel across 9 production passes amplifies per-pass cost (~4.5× the 2-pass narrowed bundles), raising the importance of the Blocker-6 rerank cap.
- **[Blocker 5 — Gaps/Technical]** Phase A is real worker wiring, not "verify-only" plumbing: `selected_chunks` must be threaded through a 3-function chain and `apply_chunk_scope` must be skipped in merged mode. A1/A2/A3 rewritten accordingly.
- **[Blocker 6 — Gaps/Perf]** Cap the candidate pool to `cfg.top_n_candidates` **before** the cross-encoder `rerank()`, and require the multi-channel row fetch to reuse the `search_extraction_chunks_direct` per-run SQL — never the HNSW dispatcher / `vector_search(filters=)` — to avoid reintroducing the known HNSW post-filter starvation bug. Starvation-guard assertion added to acceptance.
- **Should-fix (folded in):** `page_number` preserved through `MergedCandidate` (lineage); blended-score coefficients promoted to `RetrievalProfile` config fields; D1 must not call `chunker.contextualize()` on the table adapter; C6 local variable renamed to `signals` to avoid the `RetrievalProfile`/`retrieval_profile` footgun; TDD test-first ordering enforced in C1/C4/C5/C6/D1/D3; C8 identity source made concrete; named config for fallback relaxation; NFC normalization in the pattern path; B7 propagation made mandatory; extra edge-case tests added.

### Second revision — post re-review + scope expansion (2026-05-28)

A second review round confirmed all 6 blockers resolved against live code and surfaced refinements; the user then directed scope changes. Folded in:

- **Phase C re-architecture — keyword boost is now POST-rerank.** The first revision blended keyword/dense signals *before* rerank, which meant the final top_k was chosen purely by the cross-encoder and the keyword precision signal was discarded at selection. New flow: dense (recall) → cap to `top_n_candidates` → cross-encoder rerank scores the **whole capped pool** (no internal top_k slice) → per-field keyword/pattern/section/table boost − negative penalty applied **after rerank** → re-sort → **then** take top_k. Dense = recall, rerank = semantic precision, keyword = lexical precision, each independently weighted. Rewrites C5/C6, Phase C locked decisions, and the weight config (B5: `dense_weight` retired in favor of `rerank_weight` base; `section_boost` removed as redundant with `section_weight`).
- **§9 subset-schema extraction added as Phase F.** Send the extraction LLM only fields that had retrieval evidence (`supported_field_hints`), opt-in via `RetrievalProfile.subset_schema_extraction`, conservative (never drops an evidenced or identity/required field). BM25 and LLM field-aware reranking remain **deferred** — the post-rerank keyword boost is itself a deterministic, free field-aware rerank, and BM25 would reintroduce the per-run/global-index starvation mismatch (see Open Risks).
- **§8 identity-anchor source corrected (opportunistic, not guaranteed).** C8's prior prose assumed a Postgres session the `chunk_scope` endpoint does not hold. The channel now reads run-scoped committed identity entities from the ArcadeDB graph store the endpoint already obtains via `get_graph_store` (worker-supplied `ChunkScopeRequest.identity_anchors` is an optional augmentation). Because **C1.6r removed identity-first serialization** (`pipeline.py:7041-7057`, concurrency cap 2), the channel is **opportunistic** — it fires when identity entities are already committed at route time and no-ops otherwise (early field-group passes under concurrent dispatch frequently route before identity commits). A guaranteed-populated channel would require reintroducing serialization (wall cost) and stays out of scope; default-on promotion remains deferred (§8).
- **§10 deferred** (strict quality / quantity normalization / conflict diagnostics / caching) — confirmed out of scope for this slice.
- **Re-review fixes:** (a) the runtime-reach reality is now stated — "applies to all bundles" is config-only; the new path fires only when an operator sets `EXTRACTION_INDEX_MODE=merged` + `VECTOR_ROUTER_MODE=narrow_only`, which are NOT production defaults; (b) the production-bundle wall cap is anchored to the existing **bdde417 ~303m full-bundle SA-2** baseline, not the unverifiable 274m narrowed-bundle number; (c) B6 writes **only intentional overrides** (defaults come from the B5 model) and is **add-only** (never overwrites the narrowed bundles' deliberate `0.25`/`300`/`500` base values); (d) D3 reordered test-first; (e) minor doc corrections — C8 graph/body source, `page_number` has no `read_chunk_*` accessor (use `r.properties.get`), B0 attributes the `0.35` default to `air_defense_v3`, B2 uses `(json_schema_extra or {})`, `_pipeline_hooks.py` path prefixed `app/services/table_normalization/`.

### Code-sync — partial implementation landed (2026-05-28)

Verified against the current branch after three feature commits. **Phase A and Phase D's table-text work are largely DONE; merged-mode is now the default index mode.** What changed and what remains:

- **Phase A — largely implemented in `27a0c64`.** `_compute_effective_chunk_scope` now retains `selected_chunks` (gated on `extraction_index_mode=='merged'`), threaded through `_execute_pass_attempt(chunk_scope=...)` → `_build_extract_pass_request(selected_chunks=...)`; 6 unit tests in `tests/unit/test_phase2_selected_chunks_forwarding.py` (green, live-smoke confirmed). A1/A2/A3-threading/A4 are DONE. **REMAINING (decision: implement):** the guarded `apply_chunk_scope` skip is NOT done — the worker still calls it at `pipeline.py:7585-7586`. It is currently harmless (docling-graph ignores `docling_document_json` in chunked mode) but wastes per-pass CPU; skip it ONLY when non-empty `selected_chunks` are forwarded in merged mode, keep it on the empty/legacy path. Add the "apply_chunk_scope not invoked on merged forward" test assertion. Diagnostic field names actually emitted: `selected_chunks_forwarded` + `selected_chunks_forwarded_count` (worker `diag`); `selected_chunk_count`/`selected_chunk_token_estimate` live on the endpoint `ChunkScopeDiagnostics`, not the worker diag.
- **Phase D — table TEXT done UPSTREAM in `ae5f501`, but the metadata column is NOT.** `build_extraction_index_hybrid` now applies `web_cruft_sanitizer.sanitize_docling_document` (always) + table normalization (`normalize_tables` + `render_for_graph`, synthesizing per-row TextItems) BEFORE HybridChunker, gated on the pre-existing default-off `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED`. This **supersedes** the plan's `hybrid_chunking.py` / `_substitute_table_chunks` approach (D1/D2) — `MergedChunk` was NOT extended. **REMAINING + REQUIRED (decision: keep `table_boost`):** D3 — table rows are now indistinguishable from prose once indexed (no `content_type` column on `ExtractionChunk`), so **Phase C's `table_boost` is currently unwired (`is_table` always false)**. D3 must add `content_type`/`table_refs` columns and set `content_type="table"` by intersecting a chunk's `source_refs` with the synth-table-ref mapping (`extraction_chunk_index.py:1176-1197`). Phase C `table_boost` is a no-op until D3 lands. Note: Phase C lexical/pattern channels now scan **sanitized** text.
- **Config / bundles (`48302f2`).** `EXTRACTION_INDEX_MODE` default flipped to `merged` (`.env`/`.env.example`; `config.py` Field default still `per_element` — env-file-wins). Narrowed bundles were propagated to **uniform `min_similarity=0.35, top_n_candidates=50, top_k=15`**, so B6's "preserve the narrowed bundles' `0.25`/`300`/`500` base values" is **obsolete** — B6 now layers the new B5 fields onto the uniform `0.35/50/15` base (omit any new field equal to the model default). The two `DOCLING_GRAPH_*` table-norm env vars are pre-existing (not new); `NORMALIZATION_ENABLED` is default-off (the binding gate), `SUPPRESS_RAW_TABLE_MARKDOWN` is default-on. `per_element` retirement clock started 2026-05-28, earliest deletion **2026-06-11** subject to Phase 2 A/B (`docs/handoffs/2026-05-28-merged-mode-bundle-propagation.md`).
- **Decisions locked:** (1) implement the guarded `apply_chunk_scope` skip; (2) keep `table_boost` — D3's `content_type` column is required.
- **Line-ref drift** (after `27a0c64`'s +35 lines): `_execute_pass_attempt`@603 (unchanged), `_build_extract_pass_request`@**3476**, `_compute_effective_chunk_scope`@**7276**, `derive_ontology_graph_pass`@**7464**, `apply_chunk_scope`@**7585-7586**.
- **Post-sync 3-agent review folded in:** D3 now also closes the **table page-lineage hole** (synth rows carry `prov:[]` → `page_number=None`) and flags the **`SUPPRESS_RAW_TABLE_MARKDOWN` co-dependency** + the flattened-`#/texts/M`-values intersection / var-hoist; new **D6** (sanitizer recall audit + cross-tree SYNC back-pointer); new **Performance enhancements** section; B0 notes the env-dependent `test_extraction_index_mode_default_is_per_element` failure; Test-Plan diagnostic field names reconciled (`selected_chunks_forwarded_count` worker-native; `selected_chunk_count`/`_token_estimate` arrive via the envelope flatten); A3 skip scoped to `apply_chunk_scope` only.

---

## Scope

**IN SCOPE — spec sections 1-7, plus §9 subset-schema and the §8 identity-anchor fix:**

1. Selected-chunk worker→docling-graph handoff (Phase A)
2. Structured retrieval signals + `build_retrieval_profile()` (Phase B)
3. Retrieval metadata across all 9 radar + missile field-group passes (Phase B), applied to **all four bundles including production `air_defense_v3`**. Production routes all 9 passes; the narrowed bundles route 2.
4. RetrievalProfile config extension with new defaults (Phase B)
5. Multi-channel candidate generation (dense recall + lexical + pattern + functional identity-anchor) (Phase C)
6. **Post-rerank** keyword/pattern precision boost with negative-term penalty — applied after the cross-encoder, before the top_k cut (Phase C)
7. Table-aware router chunks at index time (Phase D)
8. Graduated fallback before full-document fallback (Phase E)
9. **§9 — Subset-schema extraction:** send the extraction LLM only the fields that had retrieval evidence (Phase F, opt-in via `RetrievalProfile.subset_schema_extraction`)
10. **§8 fix:** the identity-anchor channel (C8) reads run-scoped identity entities from the ArcadeDB graph store (correcting the inert Postgres-session source) so it fires **opportunistically** — when identity data is committed at route time; a guaranteed-populated channel would need serialization (out of scope)

**DEFERRED to follow-up plan:**

- Adding the other 7 field-group passes to the *narrowed* bundles (`merged_v1`/`narrowing_v1`/`baseline_subset`) — those stay 2-pass by design; production `air_defense_v3` already routes all 9.
- **§8 (remainder):** promoting the now-functional identity-anchor channel to a **default-on mode change** — a separate decision once recall data is in.
- **§9 (remainder):** BM25 lexical index and LLM field-aware reranking — BM25 would reintroduce the per-run/global-index starvation mismatch, and the post-rerank keyword boost already delivers deterministic field-aware reranking; both stay deferred unless measurements demand.
- **§10 (all):** strict quality mode with grounded field evidence; centralized quantity normalization; conflict diagnostics; extraction caching.

---

## File Structure

Files modified or created, grouped by responsibility:

**Worker → docling-graph handoff (Phase A):**
- Modify: `app/schemas/extraction_routing.py` — `SelectedChunk` already exists; verify shape (A1)
- Modify: `app/workers/pipeline.py` — thread `router_response["selected_chunks"]` through `_compute_effective_chunk_scope` → Celery `chunk_scope` dict → `derive_ontology_graph_pass` → `_execute_pass_attempt` (NEW param) → `_build_extract_pass_request` (NEW param); skip `apply_chunk_scope` in merged mode
- Verify only unless tests fail: `docker/docling-graph/app/schemas.py` and `docker/docling-graph/app/main.py` — `SelectedChunkInput`, `ExtractPassRequest.selected_chunks`, and the `main.py:1374-1376/1596-1613` handler support already exist
- Verify only unless tests fail: `docker/docling-graph/repo/docling_graph/core/extractors/strategies/many_to_one.py` — `_pre_built_chunks` threading already exists
- Test: `tests/unit/test_phase2_selected_chunks_forwarding.py` (DONE in `27a0c64`; add the `apply_chunk_scope`-skip assertion when A3's skip lands)

**Structured retrieval signals + schema metadata (Phase B):**
- Modify: `app/services/extraction_query_builder.py` — add `FieldRetrievalQuery`, `PassRetrievalSignals`, `build_retrieval_profile()`
- Modify: `app/services/ontology_bundles.py` — extend `RetrievalProfile` with new config fields (incl. post-rerank boost weights + `subset_schema_extraction`)
- Modify the 9 field-group schema modules under `ontology_bundles/air_defense_v3/extraction_schemas/` — add field-level `retrieval` metadata:
  - `radar_power_rf.py`, `radar_antenna.py`, `radar_timing.py`, `radar_modulation.py`
  - `missile_kinematics.py`, `missile_guidance.py`, `missile_airframe.py`, `missile_speed_timing.py`, `missile_propulsion.py`
- Modify ALL four bundle manifests (B6) — set new retrieval config defaults across every field-group pass each contains: `air_defense_v3/manifest.yaml` (9 passes) plus `air_defense_v3_merged_v1`/`_narrowing_v1`/`_baseline_subset` manifests (2 passes each)
- Propagate the modified schema modules to all bundles (B7) — these are **physical copies** (distinct inodes), so propagation is mandatory, not optional
- Test (extend, do NOT create new): `tests/unit/test_extraction_query_builder.py`
- Test: `tests/unit/test_retrieval_profile_config.py` (NEW)

**Multi-channel candidate generation + scoring (Phase C):**
- Modify: `app/services/extraction_chunk_search.py` — add `search_extraction_chunks_multi_channel()` and `search_extraction_chunks_dense_multi_query()`; reuse the `search_extraction_chunks_direct` per-run row fetch
- New: `app/services/extraction_lexical_search.py` — in-memory lexical + pattern search over `ExtractionChunk.chunk_text`
- New: `app/services/extraction_candidate_scoring.py` — post-rerank precision scorer combining the reranker score with lexical/pattern/section/table/negative signals (+ `active_fields` for §9)
- Modify: `app/api/v1/extraction_routing.py` — wire multi-channel into `chunk_scope` orchestration
- Modify: `app/schemas/extraction_routing.py` — diagnostics fields for multi-channel counts
- Test: `tests/unit/test_extraction_lexical_search.py` (NEW)
- Test: `tests/unit/test_extraction_candidate_scoring.py` (NEW)
- Test: `tests/integration/test_chunk_scope_multi_channel.py` (NEW)

**Table-aware router chunks (Phase D):**
- Modify: `app/services/hybrid_chunking.py` — reuse table-normalization rendering before emitting merged chunks
- Modify: `app/services/extraction_chunk_index.py` — pass table-aware text + metadata into the merged row; add new properties to the explicit `ExtractionChunk` schema (`arcadedb_schema.py`)
- Modify: `app/services/extraction_chunk_search.py` — project the new columns in the direct-fetch SELECT
- Test: `tests/unit/test_hybrid_chunking_tables.py` (NEW)
- Test: `tests/integration/test_table_chunks_indexed.py` (NEW)

**Graduated fallback (Phase E):**
- Modify: `app/api/v1/extraction_routing.py` — attempt relaxed/lexical/table fallback before full-doc fallback
- Modify: `app/schemas/extraction_routing.py` — diagnostics for fallback level/reason/coverage
- Test: `tests/unit/test_extraction_routing_fallback.py` (NEW)

---

## Sequencing + Dependencies

```
Phase B0 — Re-baseline the already-red builder tests (PREREQUISITE for Phase B)
   │
Phase A — Selected-chunk worker handoff
   │   (independent — worker wiring; can ship alone and validate via integration test)
   ▼
Phase B — Structured retrieval signals + schema metadata
   │   (independent of A — pure data types + schema annotation)
   │   (provides field_queries / lexical_terms / etc. that Phase C consumes)
   ▼
Phase C — Multi-channel candidate generation + scoring
   │   (depends on B's PassRetrievalSignals.field_queries + lexical_terms + negative_terms)
   │   (can develop scoring scaffolding in parallel with B but cannot land without B's types)
   ▼
Phase D — Table-aware router chunks
   │   (independent — chunk-rendering pre-step; can run in parallel with A/B/C)
   │   (improves retrieval quality once C is shipped, but ships safely on its own)
   ▼
Phase E — Graduated fallback
   │   (depends on Phase C diagnostics/candidate merge shape; can land before or after D)
   ▼
Phase F — Subset-schema extraction (§9)
       (depends on Phase C field_coverage/supported_field_hints + Phase A request handoff)
```

**Recommended parallel ordering:** B0 first (it gates B). Then A and B and D start concurrently. C lands after B. E lands after C. F lands after C (needs field hints) and A (needs the request handoff). Final integration test exercises A+B+C+D+E+F together.

**No new env vars required.** New behavior keys off existing `VECTOR_ROUTER_MODE` + `EXTRACTION_INDEX_MODE`; new defaults live in `RetrievalProfile` config and bundle manifest YAML. Note: the upstream table-normalization (Phase D, done) keys off the **pre-existing** `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED` / `DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN` — these are not new. `NORMALIZATION_ENABLED` is **default-off** in `.env.example` (live `.env` here is on), so a fresh checkout indexes tables as raw markdown until it is set; `SUPPRESS_RAW_TABLE_MARKDOWN` is **default-on**. So `NORMALIZATION_ENABLED` is the binding gate for table-aware indexing.

---

## Phase A — Selected-chunk worker handoff

> **Status (code-sync 2026-05-28):** A1, A2, A4, and A3's threading are **DONE** in `27a0c64` (6 tests in `tests/unit/test_phase2_selected_chunks_forwarding.py`, live-smoke confirmed). **Remaining:** (i) the guarded `apply_chunk_scope` skip — A3's last step, decision: implement; and (ii) the "apply_chunk_scope not invoked on merged forward" assertion (A5). Diagnostics actually emitted are `selected_chunks_forwarded` + `selected_chunks_forwarded_count` (worker `diag`). Line refs below are corrected for current code.

### Locked implementation decisions

- Assert byte-identical selected chunk text from router response to worker request JSON. Hashes may be logged, but the required assertion is direct string equality.
- `selected_chunks=None` or an absent field preserves the existing scoped-document path. An explicit empty list is never forwarded because docling-graph rejects it (`main.py:1374-1376`).
- `selected_chunks` are forwarded only for `narrow_only` + `mode=selected_refs` + non-empty `selected_chunks`. Shadow mode records diagnostics but does not narrow or forward chunks.
- When `selected_chunks` is forwarded, the worker MUST skip the legacy `apply_chunk_scope(doc_json, chunk_scope)` reconstruction for that pass. Forwarding chunks AND applying scope to the doc would double-narrow.
- The endpoint already limits selected chunks through `profile.top_k`; keep that cap and add a defensive assertion in tests rather than adding a second cap in the worker.
- `docling_document_json` remains a required request field even when `selected_chunks` is set. This handoff does not reduce wire payload; it changes which text the LLM sees and skips redundant worker/receiver chunking.

### Task A1 — Verify SelectedChunk schemas exist and match across boundary

**Files:**
- Read: `app/schemas/extraction_routing.py` (`SelectedChunk` — already added; expect fields `chunk_index`, `chunk_key`, `text`, `source_refs`, `token_count`)
- Read: `docker/docling-graph/app/schemas.py` (`SelectedChunkInput` with `ConfigDict(extra="ignore")`; `ExtractPassRequest.selected_chunks: Optional[list[SelectedChunkInput]] = None`)
- Confirm: receiver-required fields are a subset of worker fields — `chunk_index`, `text`, `source_refs`, `token_count`. Worker `SelectedChunk.chunk_key` is intentionally extra; `SelectedChunkInput`'s `extra="ignore"` accepts it.

**Steps:**
- [ ] Read both files and confirm receiver-required field names and types match worker output.
- [ ] Do not remove `chunk_key` from worker payload; receiver-side extra-ignore is the compatibility mechanism.
- [ ] Add a short contract comment only if the current docstrings are incomplete after inspection.

**Acceptance:** A worker `SelectedChunk.model_dump()` validates as docling-graph `SelectedChunkInput` without warnings; the receiver ignores `chunk_key`.

### Task A2 — Capture router's selected_chunks in the worker scope

**Files:**
- Modify: `app/workers/pipeline.py` — `_compute_effective_chunk_scope(router_response: dict | None, mode: str)` (~line 7276) — **already implemented in `27a0c64`**

**Context (verified):** `_compute_effective_chunk_scope` currently retains only `self_refs` and `text_by_ref` and never reads `selected_chunks`. `selected_chunks` arrives inside the endpoint's `router_response` dict — it is NOT a function parameter.

**Steps:**
- [ ] (DONE in `27a0c64` — covered by `tests/unit/test_phase2_selected_chunks_forwarding.py`.) The original task: a failing unit test building a `router_response` dict containing `selected_chunks=[<two chunks>]` (mode `selected_refs`, non-empty `self_refs`) and asserting `_compute_effective_chunk_scope(router_response, "narrow_only")` returns a scope carrying both `self_refs` and `selected_chunks`.
- [ ] In `_compute_effective_chunk_scope`, when `mode == "narrow_only"`, `router_response["mode"] == "selected_refs"`, `self_refs` is non-empty, and `router_response.get("selected_chunks")` is a non-empty list, retain the list on the returned `effective_chunk_scope`.
- [ ] In `shadow`, `disabled`, `full`, `would_skip`, and empty-list cases, leave `selected_chunks` absent so legacy behavior remains byte-identical.
- [ ] Run the focused test, confirm pass.

**Acceptance:** Test passes. No behavior change when router mode is shadow or response mode is full.

### Task A3 — Forward selected_chunks through the per-pass extraction request

**Files:**
- Modify: `app/workers/pipeline.py` — the threading chain (**already wired in `27a0c64`**, verify-only):
  - `derive_ontology_graph_pass` (~line 7464; passes `chunk_scope=`)
  - `_execute_pass_attempt` (~line 603; now takes a `chunk_scope` kwarg → builds `forwarded_selected_chunks`)
  - `_build_extract_pass_request` (~line 3476; now takes `selected_chunks` and serializes it)
- Verify: `docker/docling-graph/app/schemas.py` — `ExtractPassRequest.selected_chunks: Optional[list[SelectedChunkInput]] = None` (already present)

**Context (verified — REMAINING WORK):** The worker still calls `apply_chunk_scope(doc_json, chunk_scope)` at `pipeline.py:7585-7586` whenever `mode=="selected_refs"`, even when forwarding `selected_chunks`. This is harmless today (docling-graph ignores `docling_document_json` in chunked mode) but wastes per-pass CPU. The remaining work is to **skip** that call when non-empty `selected_chunks` are forwarded in merged mode (decision: implement), keeping it on the empty/legacy path. Scope the skip to the `apply_chunk_scope` call ONLY — do NOT skip the separate `filter_docling_document` doc-filter call just above it (~`pipeline.py:7533-7534`), which is an unrelated concern. `docling_document_json` is always populated (`_build_extract_pass_request`, ~`pipeline.py:3504`), so skipping the narrowing never drops the required field.

**Steps:**
- [ ] FIRST add a failing integration-style unit test that constructs the extraction request from a scope object carrying `selected_chunks` and asserts the request JSON includes `selected_chunks` with byte-identical text and expected `source_refs`.
- [ ] Add `selected_chunks` param to `_execute_pass_attempt` and `_build_extract_pass_request`; pass it from `derive_ontology_graph_pass` when `chunk_scope` carries selected chunks.
- [ ] When merged-mode `selected_chunks` is forwarded, SKIP `apply_chunk_scope(doc_json, chunk_scope)` for that pass (do not double-narrow).
- [ ] Serialize plain dicts only with keys accepted by docling-graph (`chunk_index`, `chunk_key`, `text`, `source_refs`, `token_count`); do not instantiate docling-graph models in the worker.
- [ ] When `selected_chunks` is None or empty, omit the field and keep the legacy `apply_chunk_scope` path — request shape unchanged.
- [ ] Run test, confirm pass.

**Acceptance:** Test passes. Request JSON shape matches docling-graph's `ExtractPassRequest`. No behavior change when no selected_chunks; `apply_chunk_scope` still runs on the legacy path.

### Task A4 — Add forwarding diagnostics

**Files:**
- Modify: `app/workers/pipeline.py` — router diagnostics dict returned by `_compute_effective_chunk_scope()` and persisted with pass diagnostics

**Steps:**
- [ ] Set `diag["selected_chunks_forwarded"] = True` only when chunks are added to the extract-pass request; otherwise set False. This is worker-side telemetry, not a chunk-scope endpoint field.
- [ ] Field names (verified): the worker emits `selected_chunks_forwarded` + `selected_chunks_forwarded_count`. The endpoint's `selected_chunk_count` / `selected_chunk_token_estimate` already reach `diagnostics_json->router` via the envelope flatten (`_compute_effective_chunk_scope` copies `router_response["diagnostics"]`) — the worker does NOT re-emit them. A test asserting a worker-native count must use `selected_chunks_forwarded_count`.
- [ ] Add a unit test asserting router diagnostics contain these keys when merged-mode handoff fires.

**Acceptance:** Diagnostics surface in `pipeline_pass_outputs` and can be queried with `diagnostics_json->'router'->>'selected_chunks_forwarded'`.

### Task A5 — Integration test: worker → docling-graph byte-identity

**Files:**
- Existing: `tests/unit/test_phase2_selected_chunks_forwarding.py` — 6 tests from `27a0c64` already cover capture, request build, and byte-identity round-trip. **Add** the missing "`apply_chunk_scope` not invoked on merged forward" assertion here once the A3 skip lands (a separate `tests/integration/...` file is optional).

**Steps:**
- [ ] Prefer a fast integration-style worker test that mocks `_call_extract_pass` and captures the request JSON sent by the worker. Full service startup is optional, not required for this phase.
- [ ] Feed a synthetic router response with deterministic merged `selected_chunks`.
- [ ] Dispatch the pass path with `VECTOR_ROUTER_MODE=narrow_only` and `EXTRACTION_INDEX_MODE=merged`.
- [ ] Assert captured request JSON contains byte-identical `selected_chunks[*].text` and expected `source_refs`, and assert `apply_chunk_scope` was NOT invoked on that pass.
- [ ] Keep the existing docling-graph selected-chunk receiver tests as the receiver-side contract.

**Acceptance:** Integration test passes. Forwarded chunk text is byte-identical to router-selected chunk text.

---

## Phase B0 — Re-baseline the already-red builder tests (PREREQUISITE)

**Context (verified):** `tests/unit/test_extraction_query_builder.py` is currently **5 failed, 13 passed** on this branch. The plan's downstream "all existing tests pass" gates are unsatisfiable until these are addressed. Note the correct filename is `test_extraction_query_builder.py` — `test_retrieval_query_builder.py` does NOT exist.

**Files:**
- Modify: `tests/unit/test_extraction_query_builder.py`

**Known failures to resolve:**
- `test_radar_power_rf_query_snapshot` / `test_missile_kinematics_query_snapshot` — stale frozen snapshots vs the edited `air_defense_v3` descriptions.
- `test_conservative_defaults_applied` — expects `min_similarity == 0.45`; the loaded `air_defense_v3` manifest now sets `0.35` (the test loads production `air_defense_v3`, not `merged_v1`).
- Two `RetrievalProfile` max-bound validation tests — "DID NOT RAISE".
- **Separately (introduced by `48302f2`, not in the above 5):** `test_extraction_index_mode_default_is_per_element` now FAILS — it `monkeypatch.delenv`s the OS var but `.env` sets `EXTRACTION_INDEX_MODE=merged`, which wins (`config.py` reads `env_file=".env"`), so it resolves `merged != per_element`. Update the test to assert the env-file default (or isolate the `config.py` Field default). Fold into the green-baseline goal.

**Steps:**
- [ ] Run `pytest tests/unit/test_extraction_query_builder.py -v` and capture the 5 failures.
- [ ] For each: determine whether it encodes a real contract or a stale expectation. Re-freeze snapshots from current builder output; update the `min_similarity` expectation to match the current manifest default; fix the two max-bound validation tests — the `le=2000` validator EXISTS (`ontology_bundles.py:96,102`); the tests assert a tighter bound, so update the test expectations.
- [ ] Confirm `pytest tests/unit/test_extraction_query_builder.py` is fully green BEFORE starting Phase B.

**Acceptance:** Builder test file is green on the unmodified branch, establishing a trustworthy baseline for B2's parity assertions.

---

## Phase B — Structured retrieval signals + schema metadata

### Locked implementation decisions

- Canonical metadata shape is `json_schema_extra={"retrieval": {"aliases": [...], "negative_terms": [...], "evidence_patterns": [...], "likely_sections": [...], "units": [...]}}`.
- `description` remains extraction/prompt guidance (prompt-authoritative). The `retrieval` block is routing metadata only (routing-authoritative for aliases/patterns). The two may overlap in vocabulary; `retrieval.aliases` is the source of truth for the lexical channel.
- `evidence_patterns` are literal phrase patterns by default. A value prefixed with `re:` is treated as a regular expression. This keeps most schema metadata simple and avoids accidental regex errors.
- **Generalization guardrail:** the dataclasses and `build_retrieval_profile` contain ZERO literal domain/equipment terms. Every alias/pattern/section/unit value originates from `json_schema_extra` on the schema field. Domain terms live ONLY in `ontology_bundles/air_defense_v3/extraction_schemas/*.py` (sanctioned config).
- New `RetrievalProfile` config defaults: `field_query_top_k=8`, `lexical_top_k=40`, `pattern_hit_limit=50`, `fallback_min_field_coverage=1`, `fallback_similarity_relaxation=0.07`; post-rerank boost weights `rerank_weight=1.0`, `lexical_weight=0.20`, `pattern_weight=0.15`, `section_weight=0.10`, `table_boost=0.08`, `negative_weight=0.20`; and `subset_schema_extraction=False` (§9 opt-in).

### Task B1 — Add FieldRetrievalQuery + PassRetrievalSignals types

**Files:**
- Modify: `app/services/extraction_query_builder.py`

**Code skeleton:**

```python
from dataclasses import dataclass, field
from typing import Sequence

@dataclass(frozen=True)
class FieldRetrievalQuery:
    """A single field's retrieval signal — used in multi-channel candidate gen.
    All values originate from the schema field's json_schema_extra['retrieval'];
    this type contains NO literal domain terms."""
    field_name: str               # e.g. "min_intercept_km"
    query_text: str               # doc + field label + description + aliases
    aliases: tuple[str, ...]      # lexical hits
    negative_terms: tuple[str, ...]
    evidence_patterns: tuple[str, ...]
    likely_sections: tuple[str, ...]
    units: tuple[str, ...]

@dataclass(frozen=True)
class PassRetrievalSignals:
    """Structured retrieval signals for a pass (distinct from the pydantic
    RetrievalProfile *config* in ontology_bundles.py)."""
    pass_name:          str
    entity_doc:         str
    entity_query:       str                                # backward-compat single-string query
    field_queries:      tuple[FieldRetrievalQuery, ...]
    lexical_terms:      tuple[str, ...]                    # union of all field aliases (dedup)
    negative_terms:     tuple[str, ...]                    # union of all field negative_terms (dedup)
    likely_sections:    tuple[str, ...]                    # union (dedup)
    evidence_patterns:  tuple[str, ...]                    # union (dedup)
```

**Steps:**
- [ ] Write the dataclasses at the top of `extraction_query_builder.py`.
- [ ] Export from `__init__.py` if needed.

**Acceptance:** Types importable; no callers yet. (Naming note: `PassRetrievalSignals` = signal bundle; `RetrievalProfile` = pydantic config; never name a local variable `retrieval_profile`.)

### Task B2 — Implement build_retrieval_profile()

**Files:**
- Modify: `app/services/extraction_query_builder.py`

**Context (verified):** The existing walker (`extraction_query_builder.py:55-119`) already skips `system_name` (`_SKIP_FIELDS`), skips `INTERNAL`-prefixed descriptions, and reads `field_info.description`. Two callers pass `pass_def=None` (`test_vector_router_live_recall.py:141`, `scripts/generate_baseline_recall_fixtures.py:180`), so `build_retrieval_profile` must tolerate `pass_def=None`. Field metadata access is `template_cls.model_fields[name].json_schema_extra` (Pydantic v2; returns the dict as-is).

**Code skeleton:**

```python
def build_retrieval_profile(pass_def, template_cls) -> PassRetrievalSignals:
    """Build structured retrieval signals from the pydantic extraction schema.

    1. Resolve the Record class (existing logic). Tolerate pass_def=None.
    2. Build entity_query via the existing walker (= prior build_retrieval_query output).
    3. For each non-skipped, non-INTERNAL field with a description:
       - Read (model_fields[name].json_schema_extra or {}).get("retrieval", {}) — json_schema_extra is None until B3/B4 populate it, so the `or {}` guard is required.
       - Construct a FieldRetrievalQuery from description + retrieval block.
    4. Union all field-level aliases/negative_terms/etc. into the top-level tuples.
    """
    # ... full implementation ...

def build_retrieval_query(pass_def, template_cls) -> str:
    """Backward-compat shim — returns PassRetrievalSignals.entity_query (str)."""
    return build_retrieval_profile(pass_def, template_cls).entity_query
```

**Steps:**
- [ ] FIRST write a failing test in `tests/unit/test_extraction_query_builder.py::test_build_retrieval_profile_returns_field_queries` — assert at least one `FieldRetrievalQuery` per non-identity field on `MissileKinematicsRecord`.
- [ ] Add a parity test `test_build_retrieval_query_shim_byte_identical` — assert `build_retrieval_query(...)` output equals the (now re-baselined) snapshot for both `radar_power_rf` and `missile_kinematics`, proving the shim refactor changed nothing.
- [ ] Implement `build_retrieval_profile`; make `build_retrieval_query` a shim returning `.entity_query`.
- [ ] All existing tests pass (`pytest tests/unit/test_extraction_query_builder.py -v`).

**Acceptance:** Signals contain `entity_query` (byte-identical to prior `build_retrieval_query`) + per-field queries with empty retrieval metadata (until B3-B4 populate them).

### Task B3 — Add retrieval metadata to radar field-group schemas

**Files:** Edit 4 schema modules under `ontology_bundles/air_defense_v3/extraction_schemas/` (all live in production `air_defense_v3`):
- `radar_power_rf.py`, `radar_antenna.py`, `radar_timing.py`, `radar_modulation.py`

**Per-field pattern (radar example):**

```python
tx_peak_power_kw: Optional[float] = Field(
    default=None,
    description=(
        "Transmitter peak power in kilowatts. Use transmitter output power, "
        "not effective radiated power or generator power."
    ),
    json_schema_extra={
        "retrieval": {
            "aliases": ["peak power", "transmitter power", "tx power", "transmit power"],
            "negative_terms": ["effective radiated power", "ERP", "generator power"],
            "evidence_patterns": ["peak power", "transmitter power", r"re:TX\s+power"],
            "likely_sections": ["specifications", "performance", "transmitter", "radar"],
            "units": ["kW", "W", "MW", "dBW"],
        }
    },
)
```

**Steps for each schema file:**
- [ ] For every non-identity, non-INTERNAL field, add `json_schema_extra={"retrieval": {...}}` populated from the existing description + spec domain knowledge. All domain terms stay in these config files (guardrail).
- [ ] Re-run the bundle's pydantic loader (`python -c "from ontology_bundles.air_defense_v3.extraction_schemas.radar_power_rf import RadarPowerRfRecord; print(RadarPowerRfRecord.model_fields)"`) — confirm no validation errors.
- [ ] Run `pytest tests/unit/test_extraction_query_builder.py::test_field_metadata_loaded` (extends B2 test) to confirm aliases now populate.

**Acceptance:** All 4 radar field-group passes have rich `retrieval` blocks. `PassRetrievalSignals.lexical_terms` is non-empty for each.

### Task B4 — Add retrieval metadata to missile field-group schemas

**Files:** Edit 5 schema modules (all live in production `air_defense_v3`):
- `missile_kinematics.py`, `missile_guidance.py`, `missile_airframe.py`, `missile_speed_timing.py`, `missile_propulsion.py`

**Steps:**
- Same pattern as B3.

**Acceptance:** All 5 missile field-group passes have `retrieval` blocks.

### Task B5 — Extend RetrievalProfile pydantic with new config fields

**Files:**
- Modify: `app/services/ontology_bundles.py` — `class RetrievalProfile(BaseModel)` (~line 67)

**Context (verified):** Current fields are `min_similarity`, `top_n_candidates`, `top_k`, `fallback_to_full` with `ConfigDict(extra="forbid")`, parsed from manifest YAML via `load_bundle_manifest` → `BundleManifest` → `PassManifest.retrieval`.

**New fields (all with defaults so existing manifests don't break):**

```python
class RetrievalProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # existing defaults stay exactly as currently defined in code:
    min_similarity:     float
    top_n_candidates:   int
    top_k:              int
    fallback_to_full:   bool
    # NEW — channel sizing / limits:
    field_query_top_k:           int   = 8
    lexical_top_k:               int   = 40
    pattern_hit_limit:           int   = 50
    fallback_min_field_coverage: int   = 1
    fallback_similarity_relaxation: float = 0.07
    # NEW — POST-rerank precision-boost weights (applied AFTER the cross-encoder, before the
    #        top_k cut; promoted from hardcoded literals so calibration is a manifest edit):
    rerank_weight:   float = 1.0    # base term = normalized cross-encoder score
    lexical_weight:  float = 0.20
    pattern_weight:  float = 0.15
    section_weight:  float = 0.10
    table_boost:     float = 0.08
    negative_weight: float = 0.20
    # NEW — §9 subset-schema extraction (opt-in; default off = full schema sent to the LLM):
    subset_schema_extraction: bool = False
```

**Steps:**
- [ ] FIRST write a failing test in `tests/unit/test_retrieval_profile_config.py` — load a manifest with the new fields, assert they round-trip; load a manifest WITHOUT them, assert defaults apply; assert an unknown field is still rejected (`extra="forbid"`).
- [ ] Add fields with defaults. Keep `extra="forbid"`.
- [ ] All existing manifest-loading tests must still pass.

**Acceptance:** New fields load with defaults; explicit overrides in manifest YAML are accepted; unknown fields still rejected.

### Task B6 — Update ALL bundle manifests with new retrieval defaults

**Files:**
- Modify: `ontology_bundles/air_defense_v3/manifest.yaml` (production — all 9 field-group passes)
- Modify: `ontology_bundles/air_defense_v3_merged_v1/manifest.yaml` (2 passes)
- Modify: `ontology_bundles/air_defense_v3_narrowing_v1/manifest.yaml` (2 passes)
- Modify: `ontology_bundles/air_defense_v3_baseline_subset/manifest.yaml` (2 passes)

**Steps:**
- [ ] Write ONLY intentional per-bundle **overrides** for the NEW B5 fields under each field-group pass's `retrieval:` block — the defaults come from the B5 `RetrievalProfile` model, so do NOT re-state default values in YAML (that would create 4×N silently-divergent copies). **If a pass's value for a new field equals the B5 model default, OMIT it entirely.** **Base values are already uniform:** `48302f2` propagated `min_similarity=0.35, top_n_candidates=50, top_k=15, fallback_to_full=true` to all narrowed passes across all 4 bundles (the earlier `0.25`/`300`/`500` values are gone), so layer the new fields on top of that uniform base — there are no longer per-bundle base values to preserve. Touch all 9 field-group passes in `air_defense_v3`; the 2 (`radar_power_rf`, `missile_kinematics`) in each narrowed bundle. Do not add or remove passes from any bundle.
- [ ] Comment block referencing this plan + the calibration source in each manifest.
- [ ] Run bundle-loader smoke for all four keys **from the worktree root** (so `app`/`ontology_bundles` resolve to this worktree, not a stale install): `python -c "from app.services.ontology_bundles import load_bundle_manifest; [print(k, {p.name: bool(p.retrieval) for p in load_bundle_manifest(k).passes}) for k in ['air_defense_v3','air_defense_v3_merged_v1','air_defense_v3_narrowing_v1','air_defense_v3_baseline_subset']]"` — confirm no errors.

**Acceptance:** Every field-group pass in all four bundles loads with its retrieval config (defaults from the model, overrides from YAML). Base values are already uniform `0.35/50/15` post-`48302f2`; the new B5 fields layer on top (nothing to "preserve").

### Task B7 — Propagate to sibling bundles (MANDATORY)

**Context (verified):** Bundle schema files are independent physical copies (distinct inodes across all four bundles). Edits to `air_defense_v3/extraction_schemas/` do NOT auto-propagate. The regression source (`SA-2_Sources` → `air_defense_v3_baseline_subset`) makes this load-bearing — skipping it silently leaves siblings unmodified.

**Files:**
- Copy the modified schema modules into `air_defense_v3_baseline_subset`, `air_defense_v3_narrowing_v1`, and `air_defense_v3_merged_v1`.
- Do not replace entire directories blindly; copy only the schema modules changed in B3/B4 plus any new helper module.

**Steps:**
- [ ] Compare the nine modified field-group schema modules across sibling bundles (diff by content, not inode).
- [ ] Copy changed modules from `air_defense_v3/extraction_schemas/` to all siblings that carry physical copies.
- [ ] Run bundle-loader smoke tests for all four bundle keys.

**Acceptance:** All four bundles share identical `extraction_schemas/` content.

---

## Phase C — Multi-channel candidate generation + scoring

### Locked implementation decisions

- Lexical and pattern search use an in-memory scan over the per-run chunk rows already needed for direct cosine. Do not add SQL `LIKE` queries or BM25 in this slice.
- **Row fetch MUST reuse the `search_extraction_chunks_direct` per-run SQL** (the `WHERE pipeline_run_id = :run_id` b-tree path, `extraction_chunk_search.py:318-330`). The multi-channel path MUST NOT call the HNSW dispatcher or `vector_search(filters=...)` — doing so reintroduces the known HNSW post-filter starvation bug (global top-k then post-filter makes `min_similarity` a no-op).
- Dense retrieval must fetch rows once per pass, embed all query strings in one batch (`embed_texts(..., query=True)`), and compute all query-by-chunk similarities from one chunk embedding matrix. Do not issue one ArcadeDB row pull per field query.
- **Pipeline order (locked): dense (recall) → cap → rerank (semantic) → keyword boost (precision) → top_k.**
  1. Dense retrieval populates the candidate pool — this is the recall net.
  2. Order the merged pool by best dense score and **cap to `cfg.top_n_candidates`** — this bounds the cross-encoder cost (`rerank()` scores every candidate on CPU with no internal cap, `reranker.py:60-67`; multi-channel + fallback must not balloon the pair count).
  3. Run `rerank()` over the **whole capped pool** and keep a score for **every** candidate — do NOT let the reranker slice to the final top_k yet (call it with `top_k = len(pool)`).
  4. Apply the **post-rerank precision boost**, re-sort, and **then** take `cfg.top_k`.
- Post-rerank scoring formula (all weights are `RetrievalProfile` config fields; the reranker score is the semantic base — there is no `dense_weight` term post-rerank because the reranker subsumes dense semantics):
  ```
  final = cfg.rerank_weight   * rerank_norm          # normalized cross-encoder score (base)
        + cfg.lexical_weight  * lexical_hit_norm
        + cfg.pattern_weight  * pattern_hit_norm
        + cfg.section_weight  * section_hit_norm
        + cfg.table_boost     * is_table             # flat boost when content_type == "table" (requires D3's column; no-op until D3 lands)
        - cfg.negative_weight * negative_hit_norm
  ```
  Clamp `final >= 0.0`; negative terms penalize ordering but never remove a candidate. Rationale: keyword presence is a precision signal — applying it after rerank lets it influence the *selected* chunks, instead of being discarded at final top_k as a pre-rerank blend would be.
- Keyword search is **precision, not recall**: lexical/pattern signals boost candidates already in the reranked pool; they do NOT admit keyword-only chunks into the primary path. The recall safety net for "dense missed it entirely" is Phase E's `lexical_table` fallback level.
- Identity anchors are opportunistic and nonblocking. Identity names are supplied by the worker in the chunk-scope request (C8); if none are present, emit `identity_anchor_count=0` and continue.

### Task C1 — Batched dense entity + per-field queries

**Files:**
- Modify: `app/services/extraction_chunk_search.py`

**Context (verified):** `search_extraction_chunks` takes a `query_vector` (not text). `embed_texts(texts: list[str], batch_size=64, *, query=False)` batches and L2-normalizes. The row-fetch + matrix-multiply pattern already exists in `search_extraction_chunks_direct` (`embeddings @ q` at line 363; chunk fields projected into `GraphEntityResult.properties` at 419-431).

**Steps:**
- [ ] FIRST write a test with synthetic rows and mocked embeddings asserting: rows fetched exactly once (one SELECT, zero `vector_search`), identity fields absent, per-field result sets returned.
- [ ] Extract a reusable helper that fetches all `ExtractionChunk` rows for one `pipeline_run_id` once via the direct-fetch SQL, including `vertex_id`, `self_ref`, `chunk_text`, `embedding`, `chunk_index`, `source_refs`, `token_count`, `page_number`, `modality`, and any table/section metadata added by Phase D.
- [ ] Add `search_extraction_chunks_dense_multi_query(retrieval_signals, rows, cfg)` that embeds `[entity_query] + [field.query_text ...]` in one `embed_texts(..., query=True)` call.
- [ ] Convert row embeddings to one numpy matrix and compute all query/chunk cosine scores in one matrix operation.
- [ ] Return one entity-dense candidate set plus `dict[field_name, candidates]`, capped by `cfg.top_n_candidates` for entity and `cfg.field_query_top_k` per field.

**Acceptance:** Dense multi-query retrieval does not perform per-field ArcadeDB row pulls, issues exactly one per-run SELECT, and never calls the HNSW path. Query embedding is batched. Identity (`system_name`) and INTERNAL fields are skipped.

### Task C2 — Lexical alias search

**Files:**
- New: `app/services/extraction_lexical_search.py`

**Code skeleton:**

```python
def lexical_hit_counts(
    rows: list[dict],
    field_queries: Sequence[FieldRetrievalQuery],
) -> dict[str, dict]:
    """Return {candidate_key: {"alias_hits": int, "negative_hits": int, "supported_fields": set[str], ...}}."""
    # Normalize text with unicodedata.normalize("NFC", text).casefold(). Candidate key is vertex_id if present, else self_ref.
    # Track field-level alias hits so supported_field_hints are available before rerank.
```

**Steps:**
- [ ] FIRST write the failing test: per-chunk hit counts for synthetic data; negative terms tracked separately; Cyrillic/Latin casefold behavior is stable (pinned to NFC).
- [ ] Implement deterministic `unicodedata.normalize("NFC", text).casefold()` substring matching.
- [ ] Track alias matches, negative matches, and field names whose aliases matched.
- [ ] Key results by stable candidate key (`vertex_id` preferred; fallback `self_ref`).

**Acceptance:** Returns correct per-chunk hit counts; case-insensitive; no false negatives on accented/Cyrillic characters (NFC normalization for the Russian-language SA-2 doc).

### Task C3 — Pattern (regex) search

**Files:**
- Modify: `app/services/extraction_lexical_search.py` — add `pattern_hit_counts()`

**Steps:**
- [ ] FIRST write tests: literal phrase matching, one `re:` pattern case, AND one Cyrillic pattern case mirroring C2.
- [ ] Normalize the chunk haystack with the SAME `unicodedata.normalize("NFC", text)` step as C2 before matching (composed/decomposed Cyrillic parity).
- [ ] Treat evidence pattern strings as literal phrase matches unless prefixed with `re:`. Compile only `re:` patterns with `re.IGNORECASE | re.MULTILINE`; invalid regexes raise in unit tests, not at runtime.
- [ ] Per chunk, track pattern hits and field names whose patterns matched; cap diagnostic samples at `cfg.pattern_hit_limit`.

**Acceptance:** Patterns from `PassRetrievalSignals.evidence_patterns` correctly flag chunks containing matches, including Cyrillic.

### Task C4 — Candidate merging

**Files:**
- New: `app/services/extraction_candidate_scoring.py`

**Context (verified) — CRITICAL:** `GraphEntityResult` (`graph_store.py:64-81`) has only `node_id, name, entity_type, canonical_name, extraction_confidence, score, score_type, properties, relationship_types`. The chunk fields (`chunk_text`, `chunk_index`, `self_ref`, `source_refs`, `token_count`, `page_number`) live INSIDE `.properties`. Read them via the `read_chunk_*` accessors from `extraction_chunk_index.py` **where one exists** — `read_chunk_index`, `read_chunk_source_refs`, `read_chunk_token_count`; `chunk_text`, `self_ref`, and `page_number` have NO accessor, so read them with `r.properties.get(...)`. Mirror `app/api/v1/extraction_routing.py:335-342`. Do NOT read any of them as attributes on `GraphEntityResult` — that will `AttributeError`.

**Code skeleton:**

```python
@dataclass
class MergedCandidate:
    candidate_key: str                 # vertex_id preferred; self_ref fallback
    chunk_index:   int
    self_ref:      str
    chunk_text:    str
    source_refs:   list[str]
    token_count:   int
    page_number:   int | None          # lineage — read from properties; do NOT drop
    vector_score:  float | None
    field_scores:  dict[str, float]      # per-field dense scores (best)
    alias_hits:    int
    pattern_hits:  int
    negative_hits: int
    section_hits:  int
    content_type:  str | None          # e.g. "table" when Phase D metadata exists
    retrieval_sources: set[str]   # {"dense", "field:min_intercept_km", "lexical", "pattern"}
    supported_field_hints: set[str]  # field_names that contributed signal

def merge_candidates(
    entity_dense: list[GraphEntityResult],       # read chunk fields via r.properties + read_chunk_*
    field_dense:  dict[str, list[GraphEntityResult]],
    lexical_hits: dict[str, dict],
    pattern_hits: dict[str, dict],
    section_meta: dict[str, dict],   # from index metadata keyed by candidate_key
    table_meta:   dict[str, str],    # content_type keyed by candidate_key
) -> list[MergedCandidate]:
    """Merge by candidate_key; aggregate provenance. Chunk fields come from
    GraphEntityResult.properties via read_chunk_index/read_chunk_source_refs/etc."""
```

**Steps:**
- [ ] FIRST write tests: duplicate chunk across channels merges by candidate key; sources preserved; field hints aggregated; per-element fallback keys still work; `page_number` survives; AND two merged chunks sharing a `self_ref` but with distinct `vertex_id` do NOT merge (collision guard, since merged-mode `self_ref` can repeat — `extraction_chunk_search.py:385-393`).
- [ ] Implement `merge_candidates`, reading chunk fields from `GraphEntityResult.properties` via the `read_chunk_*` accessors.

**Acceptance:** No duplicates; provenance complete; field hints populated; `page_number` preserved end-to-end.

### Task C5 — Post-rerank precision scoring

**Files:**
- Modify: `app/services/extraction_candidate_scoring.py`

**Context:** This runs **after** `rerank()` (see C6). Each candidate carries its cross-encoder score under the key **`reranker_score`** — that is the key `rerank()` actually writes (`reranker.py:75`; the existing endpoint reads `c["reranker_score"]` at `extraction_routing.py:99`), NOT `rerank_score`. **Unscorable candidates** (empty `content_text`) are returned by `rerank()` with NO score key — `score_candidates` must tolerate that. Each candidate also carries the keyword/pattern/section/negative/table signals from C2-C4. `score_candidates` combines them into the final ordering — keyword search is precision applied on top of the reranked semantic order.

**Steps:**
- [ ] FIRST write tests: rerank-only (no keyword signal) → final order equals rerank order; a chunk with strong alias/unit hits is promoted ABOVE a higher-rerank chunk that has none (precision boost demonstrably changes selection); only-negatives → demoted but NOT removed; all-zero keyword signal → order unchanged from rerank; a candidate missing `reranker_score` (unscorable) is handled, not a `KeyError`; sort stability.
- [ ] Implement `score_candidates(candidates, cfg) -> list[(MergedCandidate, final_score)]` using the locked POST-rerank formula, reading ALL weights (incl. `rerank_weight`) from `cfg` — no hardcoded coefficients. Read the base score via `candidate.get("reranker_score", <floor>)`.
- [ ] Normalize within the pool: `rerank_norm = minmax(reranker_score)` (or sigmoid — pick one and pin it in the test); candidates without a `reranker_score` get `rerank_norm = 0.0`. `lexical_norm = alias_hits / max(1, max_alias_hits_in_pool)`; same for pattern, section, negative. Guard empty and all-zero pools against divide-by-zero. Add boosts, subtract the negative term, clamp `final >= 0.0`.
- [ ] Sort by `final desc`, then `reranker_score desc`, then stable candidate key for deterministic ties.

**Acceptance:** With no keyword signal the order equals the reranker's. A strong exact-term/unit match can lift a chunk above a higher-rerank chunk lacking it. Negative terms demote but never hard-filter. All weights (incl. `rerank_weight`) read from `RetrievalProfile` config.

### Task C6 — Wire multi-channel into chunk_scope endpoint

**Files:**
- Modify: `app/api/v1/extraction_routing.py`
- Modify: `app/services/extraction_chunk_search.py` — add `search_extraction_chunks_multi_channel()` orchestrator

**Steps:**
- [ ] FIRST write the integration test (below) so it fails before wiring.
- [ ] Add `search_extraction_chunks_multi_channel(retrieval_signals, pipeline_run_id, cfg)` that fetches rows once (direct SQL, no HNSW), runs **C1-C4** (dense + lexical + pattern + merge), orders the merged pool by best dense score, **caps to `cfg.top_n_candidates`**, and returns that capped pool plus diagnostics. Scoring (C5) is post-rerank and happens in the endpoint, NOT here.
- [ ] In `extraction_routing.py`, name variables to avoid the footgun: `signals = build_retrieval_profile(...)` (the `PassRetrievalSignals` dataclass); `cfg = pass_def.retrieval or RetrievalProfile(...)` (the pydantic config). Do NOT use `retrieval_profile` as a variable name.
- [ ] Branch only for merged-mode multi-channel:
  ```python
  if settings.extraction_index_mode == "merged" and cfg.field_query_top_k > 0:
      pool, search_diag = await search_extraction_chunks_multi_channel(signals, run_id, cfg)
      query_text = signals.entity_query
  else:
      query_text = build_retrieval_query(pass_def, template_cls)
      pool, search_diag = await search_extraction_chunks(...)
  ```
- [ ] **Rerank the ENTIRE capped pool** — call `rerank(query_text, pool, top_k=len(pool))` so every scoreable candidate keeps its `reranker_score` (the key `rerank()` writes, `reranker.py:75`); do NOT let the reranker slice to the final top_k here. Note `rerank()` returns `scored + unscorable`, and unscorable (empty `content_text`) candidates carry NO score key — C5 must handle that. Preserve reranker input keys `content_text`, `self_ref`, `vector_score`, `chunk_index`, `source_refs`, `token_count`, `page_number`; carry `alias_hits`, `pattern_hits`, `negative_hits`, `section_hits`, `content_type`, `retrieval_sources`, `supported_field_hints` through as extra keys (reranker copies the dict and passes unknown keys through — `reranker.py:72`).
- [ ] Apply C5 `score_candidates(reranked_pool, cfg)` (the POST-rerank keyword/pattern/table boost), re-sort by `final`, and **then** take `cfg.top_k`. Attach `score_components` to the selected candidates for diagnostics.

**Acceptance:** Per-element mode unchanged. Merged mode runs through multi-channel with exactly one per-run row fetch and no HNSW call. The cross-encoder scores at most `cfg.top_n_candidates` pairs (Blocker 6 preserved). The post-rerank keyword boost is applied before the top_k cut and CAN change which chunks land in the final selection.

### Task C7 — Diagnostics: candidate source counts, field coverage, post-rerank scores

**Files:**
- Modify: `app/schemas/extraction_routing.py` — extend `ChunkScopeDiagnostics`
- Modify: `app/api/v1/extraction_routing.py` — populate (only the success path must populate the non-`None` values; all new fields default to `None`)

**New diagnostic fields:**

```python
class ChunkScopeDiagnostics(BaseModel):
    # ... existing ...
    # NEW:
    channel_counts: dict[str, int] | None = None  # {"dense": 36, "field:min_intercept_km": 12, ...}
    field_coverage: dict[str, int] | None = None  # {field_name: chunks_with_evidence}
    score_components: list[dict] | None = None    # selected chunks only; cap length at top_k
    fallback_level: str | None = None             # "none" | "relaxed_dense" | "lexical_table" | "identity_anchor" | "full"
    fallback_reason: str | None = None
```

**Steps:**
- [ ] Populate in `chunk_scope` endpoint success path.
- [ ] Test: assert diagnostics_json keys after a real chunk-scope call.

**Acceptance:** Diagnostics queryable via `diagnostics_json->'router'->'channel_counts'` etc.

### Task C8 — Opportunistic identity-anchor channel (§8 inert-path fix)

**Files:**
- Modify: `app/services/extraction_chunk_search.py` — add `identity_anchor_queries()` helper
- Modify: `app/schemas/extraction_routing.py` — add OPTIONAL `identity_anchors: list[str] | None = None` to `ChunkScopeRequest` (an augmentation the worker MAY supply; the channel does not depend on it)

**Context (corrected — the prior source was inert, and a guaranteed channel is NOT cheaply achievable):** The endpoint is `async def chunk_scope(body: ChunkScopeRequest)` — it holds NO Postgres session (so "read `pipeline_pass_outputs` from the endpoint" was inert), but it DOES obtain the ArcadeDB graph store via `get_graph_store`. **The right source is run-scoped committed identity entities from the graph store.** Critical caveat verified in code: **C1.6r deliberately REMOVED identity-first serialization** — under concurrent dispatch (`pass_concurrency_per_document=2`, `app/config.py:319`; rationale at `app/workers/pipeline.py:7041-7057`), a field-group pass can route BEFORE its run's identity pass has committed entities. The channel is therefore necessarily **opportunistic**: it fires when identity entities are already committed/queryable at route time and no-ops otherwise (early passes frequently route before identity commits; it populates for later passes / retries as the run progresses). Do NOT reintroduce an identity-before-field-group serialization gate — that re-imposes the wall cost C1.6r removed.

**Steps:**
- [ ] FIRST write tests: with run-scoped identity entities present in the (mocked) graph store → anchor channel fires (dense + lexical anchors added, `identity_anchor_count > 0`); with none present and `identity_anchors=None` → silently skipped (`identity_anchor_count=0`, no regression).
- [ ] In `identity_anchor_queries()`, query the graph store for the run's committed identity-type entities (cheap, run-scoped); union any worker-supplied `body.identity_anchors`; add the names as additional dense and lexical anchors with source `identity_anchor`. Never block or fail the pass.
- [ ] Add OPTIONAL `identity_anchors` to `ChunkScopeRequest` (default None) so the worker MAY pass names it cheaply has; the channel works without it via the graph store.
- [ ] Add `identity_anchor_count` to diagnostics.

**Acceptance:** The identity-anchor channel and Phase E's `identity_anchor` level are functional **when identity entities are committed/queryable at route time**, and silently no-op otherwise — opportunistic, never blocking, no serialization gate. Backward-compatible. **Known limitation (documented, not a defect):** under C1.6r concurrent dispatch, early field-group passes often route before identity commits, so the channel is frequently empty for them; a guaranteed-populated channel would require reintroducing serialization (wall cost) — out of scope. Promoting to a default-on mode change remains deferred (§8).

---

## Phase D — Table-aware router chunks

### Locked implementation decisions

- **Table normalization is applied UPSTREAM in `build_extraction_index_hybrid` (DONE in `ae5f501`)**, NOT via `hybrid_chunking.py`. Before HybridChunker, the index builder runs `web_cruft_sanitizer.sanitize_docling_document` (always) then `normalize_tables` + `render_for_graph(..., emit_unit_hint=True)` (synthesizing per-row TextItems for ALL tables), gated on the pre-existing **default-off** `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED`. This **supersedes** the original `_substitute_table_chunks`/`_NormalizedTableChunkAdapter` approach — there are no adapter chunks in this path.
- Router-visible table rows are therefore plain merged `chunk_text` (lexically discoverable) — table TEXT is DONE. What REMAINS is table METADATA (`content_type="table"`, `table_refs`) so the scorer can apply `table_boost` (D3). Use optional/nullable properties; no destructive migration.

### Task D1 — Table-normalization substitution before merged chunk emission (DONE — `ae5f501`)

**Status: implemented.** `build_extraction_index_hybrid` (`app/services/extraction_chunk_index.py:1136-1222`) sanitizes + normalizes tables upstream of HybridChunker, synthesizing per-row TextItems via `normalize_tables` + `render_for_graph`. `MergedChunk` was intentionally NOT extended — synthesized rows flow through the chunker as ordinary text. The original plan's `hybrid_chunking.py` / `_substitute_table_chunks` / adapter-branch approach is **superseded** and not needed.

**Verify-only:**
- [ ] Confirm `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=true` in the environment under test (default-off in `.env.example`; `true` in this worktree's live `.env`).
- [ ] Confirm synthesized table rows reach `chunk_text` (sanitize covered by `tests/unit/test_web_cruft_sanitizer.py`; add a "table rows present in merged index" assertion if absent).

**Acceptance:** Table rows are lexically discoverable in `chunk_text` (text-level table awareness). Metadata for scoring is D3.

### Task D2 — Verify table text is self-contained and header-aware

**Files:**
- Reuse: `app/services/table_normalization/` helpers — the upstream path uses `render_for_graph` (`render_graph.py`, `emit_unit_hint=True`), NOT `render_for_embedding`.

**Steps:**
- [ ] Add tests around the upstream normalized render output: each emitted row chunk must include table/header context, row labels, units, and values.
- [ ] Add an explicit test that a >512-token table splits with header context carried into EVERY emitted fragment (no header-less rows).
- [ ] If the render output omits required context, extend the `render_for_graph` renderer rather than adding a parallel renderer.

**Acceptance:** Rendered chunks are lexically discoverable; a search for a row label or unit-bearing column header finds the table chunk without surrounding prose; large tables split with headers on every fragment.

### Task D3 — Preserve table metadata + page on ExtractionChunk (REQUIRED — keeps `table_boost` functional AND fixes table page-lineage)

**Context (three verified gaps):**
1. After D1, table rows reach the index as ordinary merged `chunk_text` with NO marker (no `content_type` column on `ExtractionChunk`: `arcadedb_schema.py:38-57`), so **Phase C's `table_boost` is unwired (`is_table` always false)**.
2. **Synthesized table rows carry `prov: []`** (`app/services/table_normalization/_text_item.py:37-50`), so `_resolve_first_page_no` returns `None` (`hybrid_chunking.py:220-234`) and a table-derived merged chunk stores `page_number=None` — **table results currently have NO page (a data-lineage hole)**. This is the stronger reason for D3, beyond the boost.
3. **Two flags gate the synth path, not one.** Synth `#/texts/M` rows are swapped into `body.children` (so HybridChunker actually chunks them) only when `DOCLING_GRAPH_SUPPRESS_RAW_TABLE_MARKDOWN=true` (`_replace_raw_table_refs_in_body_children`, run inside the suppress gate, `extraction_chunk_index.py:1200-1204`). With suppress OFF, the raw `#/tables/N` markdown is chunked instead, so a merged chunk's `source_refs` hold `#/tables/N` (a KEY of `_synth_only_table_refs`), not the `#/texts/M` values — and the intersection below misses. **D3 requires NORMALIZATION=true AND SUPPRESS=true** — SUPPRESS is already default-on, so `NORMALIZATION_ENABLED` is the binding gate to turn on.

**Files:**
- Modify: `app/services/table_normalization/_text_item.py` (or the synth call site) — carry the parent table's page into the synth TextItem `prov` so table chunks resolve a real `page_number`.
- Modify: `app/services/extraction_chunk_index.py` — **hoist `_synth_only_table_refs` to function scope** (it is currently local to the `if _norm_on:` block, ~1176-1222, and goes out of scope before the insert loop). Its shape is `{table_ref: [synth_ref, ...]}`. On a merged row, set `content_type="table"` + `table_refs=[parent #/tables/N]` by intersecting the row's `source_refs` with the **flattened `#/texts/M` values** (invert the dict to recover the parent table ref). Thread through the insert. Add `content_type`/`table_refs` to the explicit `ExtractionChunk` schema (`arcadedb_schema.py`, `CREATE PROPERTY ... IF NOT EXISTS`, NULL-backfill pattern at `arcadedb_schema.py:208-237`).
- Modify: `app/services/extraction_chunk_search.py` — add the new columns to the `search_extraction_chunks_direct` SELECT projection (`extraction_chunk_search.py:322-324`).

**Steps:**
- [ ] FIRST write the failing test (NORMALIZATION + SUPPRESS both on): index a synthetic table doc, query `ExtractionChunk`, assert `content_type="table"` on table-derived chunks, `page_number` is non-null, and prose/legacy rows default to `"text"`.
- [ ] Carry parent-table page onto synth rows; hoist `_synth_only_table_refs`; set `content_type`/`table_refs` via the flattened-values intersection; extend insert SQL + schema dict + direct SELECT.
- [ ] Add a `read_chunk_content_type` accessor defaulting to `"text"`; default missing `table_refs` to `[]`. (`section_path` deferred — section scoring uses query-time `section_hits`, not a stored column.)

**Acceptance:** Table-derived chunks carry `content_type="table"`, `table_refs`, AND a real `page_number` (closes the lineage hole); Phase C `table_boost` (decision: kept) is functional. Works only with `NORMALIZATION=true` + `SUPPRESS=true`. Legacy rows unaffected.

### Task D4 — Re-index test docs + verify

**Files:**
- (No code change — operational task)

**Steps:**
- [ ] Trigger graph_only reingest of Dvina + SA-2 with `merged_v1` and `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=true`.
- [ ] After D3 lands: `SELECT count(*) FROM ExtractionChunk WHERE content_type='table' AND pipeline_run_id=<run>` — expect non-zero for SA-2 (explicit performance tables). (Before D3 the column does not exist.)
- [ ] Inspect rendered chunk_text for one table chunk — confirm row sentences are self-contained.

**Acceptance:** Table chunks exist in DB; rendered text passes manual readability check.

### Task D5 — Integration test: table chunks discoverable via lexical/pattern

**Files:**
- New: `tests/integration/test_table_chunks_indexed.py`

**Steps:**
- [ ] Build a synthetic doc with one performance table.
- [ ] Index via `build_extraction_index_hybrid` with `DOCLING_GRAPH_TABLE_NORMALIZATION_ENABLED=true`.
- [ ] Call multi-channel search with `aliases=["min range"]` and `evidence_patterns=["min range", r"re:\bmin\s+range\b"]`.
- [ ] Assert the table chunk is in the merged candidate list with both `alias_hits > 0` and `pattern_hits > 0`. (The `content_type=="table"` assertion requires D3.)

**Acceptance:** Table chunks are findable through non-dense channels — proves Phase D + C are wired correctly.

### Task D6 — Sanitizer recall audit + cross-tree sync back-pointer

**Context:** `ae5f501` added `app/services/web_cruft_sanitizer.py` (a hand-fork of `docker/docling-graph/app/main.py`) and runs it on EVERY index build, blanking ~27% of texts on Dvina (49/182). Whole-item blanking triggers on any single blob-like token — a false positive would silently drop extractable content.

**Steps:**
- [ ] On a gate doc (Dvina + SA-2), dump the blanked text items and confirm none carry extractable spec values (the blanking is meant for ad/nav/tracking cruft only). Tighten the predicate if a false positive is found.
- [ ] Add a reciprocal `# SYNC: app/services/web_cruft_sanitizer.py` back-pointer comment at `docker/docling-graph/app/main.py` (the worker copy already documents the obligation; the original does not), and correct the worker docstring's "byte-identical" → "functionally equivalent rule set" (the copies differ in imports/signatures).
- [ ] Consider making the `web_cruft_sanitizer` import in `extraction_chunk_index.py` fail-soft (try/except like the table-norm import) so a future break degrades to raw text rather than failing the index build.

**Acceptance:** No real content is silently blanked on the gate docs; the two sanitizer copies are discoverable from both sides.

---

## Phase E — Graduated fallback before full-document fallback

### Task E1 — Define field-coverage and enough-candidates helpers

**Files:**
- Modify: `app/services/extraction_candidate_scoring.py`

**Steps:**
- [ ] FIRST unit test empty, sparse, all-noise (non-empty but no real retrieval signal), and well-covered candidate sets.
- [ ] Add `field_coverage(candidates) -> dict[str, int]` using `supported_field_hints`.
- [ ] Add `enough_candidates(candidates, cfg) -> bool`. Do NOT count length alone — an all-noise pool must NOT report "enough". Use a count of candidates with a real retrieval signal (non-empty `retrieval_sources` beyond a bare dense floor) `>= min(cfg.top_k, 10)`. (Fallback is a recall decision over the merged pool, made independent of the post-rerank `final` score.)
- [ ] Add `enough_field_coverage(candidates, cfg) -> bool` using `len(nonzero_fields) >= cfg.fallback_min_field_coverage`.

**Acceptance:** Fallback decisions can be made without calling the LLM or reranker, and an all-noise pool correctly triggers fallback rather than suppressing it.

### Task E2 — Add fallback ladder in chunk-scope orchestration

**Files:**
- Modify: `app/api/v1/extraction_routing.py`

**Steps:**
- [ ] Run normal multi-channel retrieval first (`fallback_level="none"`).
- [ ] If under-covered, run relaxed dense retrieval with `min_similarity=max(0.0, cfg.min_similarity - cfg.fallback_similarity_relaxation)` (`fallback_level="relaxed_dense"`). Reuse the already-computed query embedding matrix AND the per-run chunk-embedding matrix — do NOT re-embed or re-fetch rows; relaxed dense is a threshold change over the same matrices.
- [ ] If still under-covered, include lexical/pattern/table-only candidates even when dense score is absent (`fallback_level="lexical_table"`).
- [ ] If identity anchors are available (C8) and coverage is still low, add them (`fallback_level="identity_anchor"`). If C8 found no identity source, this level no-ops.
- [ ] Only after those levels fail, use existing full-document fallback when `cfg.fallback_to_full=True`; otherwise return `would_skip` as today.

**Acceptance:** Full-document fallback remains available but is no longer the first recovery path after sparse retrieval. With `fallback_to_full=False` and a genuinely empty result, the response is still `would_skip` (not silently `full`).

### Task E3 — Fallback diagnostics and tests

**Files:**
- Modify: `app/schemas/extraction_routing.py`
- New: `tests/unit/test_extraction_routing_fallback.py`

**Steps:**
- [ ] Populate `fallback_level`, `fallback_reason`, `field_coverage_before_fallback`, `candidate_count_before_fallback`, and `candidate_count_after_fallback`.
- [ ] Test each ladder level with mocked retrieval functions, including zero-field-coverage-with-non-empty-candidates escalation.
- [ ] Verify legacy full fallback behavior remains unchanged when all cheaper levels return no candidates, and that `would_skip` survives when `fallback_to_full=False`.

**Acceptance:** Operators can tell which fallback level fired and whether it avoided full-document extraction.

---

## Phase F — Subset-schema extraction (§9)

### Locked implementation decisions

- Opt-in via `RetrievalProfile.subset_schema_extraction` (default `False` → full schema sent to the LLM, behavior byte-identical to today).
- The **active field set** for a pass = fields with retrieval evidence in the selected chunks (`supported_field_hints`, i.e. any non-zero dense/lexical/pattern signal) **UNION** always-include fields — identity fields (from `template_cls.model_config['graph_id_fields']`, NOT a hardcoded name like `system_name`) and pydantic-required fields (`model_fields[name].is_required()`). Conservative: subsetting only DROPS fields with ZERO evidence anywhere in the selected chunks — it NEVER drops an evidenced or required field. This protects recall.
- The subset is advisory metadata on the extract request; docling-graph builds the per-pass LLM schema/prompt from it. `None`/absent → full schema.
- Rationale: this builds directly on Phase C's per-field signals (we already know which fields had evidence). It is precision/token-efficiency, not a retrieval change. BM25 and LLM field-aware reranking remain deferred (the post-rerank keyword boost already provides deterministic field-aware reranking).

### Task F1 — Derive the active field set

**Files:**
- Modify: `app/services/extraction_candidate_scoring.py` — add `active_fields(candidates, template_cls, cfg) -> list[str]`

**Steps:**
- [ ] FIRST write tests: a field with evidence in ≥1 selected chunk is active; a field with zero evidence anywhere is dropped; identity/required fields are ALWAYS active even with zero evidence; with `subset_schema_extraction=False` the function returns ALL fields (no-op).
- [ ] Compute the union of `supported_field_hints` across the selected candidates, plus always-include fields: identity fields from `template_cls.model_config['graph_id_fields']` and required fields via `model_fields[name].is_required()`. Do NOT hardcode any field name (e.g. `system_name`) — the guardrail requires deriving them from model metadata.
- [ ] Return active field names in schema order.

**Acceptance:** Never drops an evidenced or required field; returns the full set when the flag is off.

### Task F2 — Thread the field subset to the extract request

**Files:**
- Modify: `app/workers/pipeline.py` — `_build_extract_pass_request(..., field_subset: list[str] | None = None)` (alongside the Phase A `selected_chunks` param)
- Modify: `docker/docling-graph/app/schemas.py` — confirm/add `ExtractPassRequest.field_subset: Optional[list[str]] = None`

**Steps:**
- [ ] FIRST write a failing test asserting the extract request JSON carries `field_subset` when the flag is on and the set is non-empty, and omits it otherwise.
- [ ] Verify whether `ExtractPassRequest` already supports a field subset; if not, add `field_subset: Optional[list[str]] = None` (backward-compatible).
- [ ] Worker: when `cfg.subset_schema_extraction` is on, pass `active_fields(...)` through to `_build_extract_pass_request`; when off or empty, omit the field.

**Acceptance:** Request JSON carries `field_subset` only when enabled; request shape unchanged otherwise.

### Task F3 — Apply the subset in docling-graph extraction

**Files:**
- Modify the docling-graph LLM schema build site — verified net-new: the per-pass schema is generated from the FULL template via `template.model_json_schema()` in `docker/docling-graph/repo/docling_graph/core/extractors/backends/llm_backend.py` (~lines 124/130/295) and the template threads through `strategies/many_to_one.py`. There is no field-subset hook today. Restrict to `field_subset` at this build site.

**Steps:**
- [ ] FIRST write/extend a receiver-side test: with `field_subset=[...]`, the LLM schema/prompt includes only those fields (plus required); with `None`, the full schema is used.
- [ ] Build the subset as a NEW schema/submodel — filter the `model_json_schema()` output or `pydantic.create_model` a sub-record. Do NOT prune the live record class, or final full-record pydantic validation will fail on the dropped fields. Never drop required/identity fields the record needs to validate.
- [ ] Keep full-schema behavior byte-identical when `field_subset` is `None`.

**Acceptance:** With a subset, the extraction prompt/schema is restricted to the active fields; without one, behavior is unchanged.

### Task F4 — Diagnostics + recall guard

**Files:**
- Modify: `app/schemas/extraction_routing.py` / worker diagnostics — add `active_field_count`, `dropped_field_count`

**Steps:**
- [ ] Populate `active_field_count` and `dropped_field_count` in diagnostics.
- [ ] Add a guard test: a field that had any evidence is NEVER in the dropped set.
- [ ] Recall check (folds into the end-to-end gate): subset-on extracted-entity counts must be `>=` subset-off on both gate docs — subsetting must not reduce recall.

**Acceptance:** Operators can see how many fields were dropped; subsetting never reduces extracted-entity recall vs full-schema on the gate docs.

---

## Public Interfaces / Types Summary

**New:**
- `app.services.extraction_query_builder.FieldRetrievalQuery` (frozen dataclass)
- `app.services.extraction_query_builder.PassRetrievalSignals` (frozen dataclass; renamed from the earlier `PassRetrievalProfile` to avoid collision with the `RetrievalProfile` config)
- `app.services.extraction_query_builder.build_retrieval_profile(pass_def, template_cls) -> PassRetrievalSignals`
- `app.services.extraction_lexical_search.lexical_hit_counts(...)`, `pattern_hit_counts(...)`
- `app.services.extraction_candidate_scoring.MergedCandidate`, `merge_candidates(...)`, `score_candidates(...)` (post-rerank), `field_coverage(...)`, `enough_candidates(...)`, `enough_field_coverage(...)`, `active_fields(...)` (§9)
- `app.services.extraction_chunk_search.search_extraction_chunks_multi_channel(...)`, `search_extraction_chunks_dense_multi_query(...)`, `identity_anchor_queries(...)`

**Extended:**
- `app.services.ontology_bundles.RetrievalProfile` — new fields `field_query_top_k`, `lexical_top_k`, `pattern_hit_limit`, `fallback_min_field_coverage`, `fallback_similarity_relaxation`; post-rerank boost weights `rerank_weight`, `lexical_weight`, `pattern_weight`, `section_weight`, `table_boost`, `negative_weight`; and `subset_schema_extraction` (§9 opt-in). (No `dense_weight`/`section_boost` — the reranker is the semantic base and `section_weight` covers section scoring.)
- `app.schemas.extraction_routing.ChunkScopeDiagnostics` — new endpoint fields `channel_counts`, `field_coverage`, `score_components`, `fallback_level`, `fallback_reason`, `active_field_count`, `dropped_field_count`
- `app.schemas.extraction_routing.ChunkScopeRequest` — new `identity_anchors: list[str] | None` (worker-supplied identity names, §8 fix)
- Worker router diagnostics — new worker-populated field `selected_chunks_forwarded`
- `app.workers.pipeline._execute_pass_attempt` and `_build_extract_pass_request` — accept optional `selected_chunks` and `field_subset`
- `docker/docling-graph/app/schemas.ExtractPassRequest` — `selected_chunks: list[SelectedChunkInput] | None` (already present) and `field_subset: list[str] | None` (§9; verify/add)

**Backward-compat shims:**
- `build_retrieval_query(...)` still returns `str` (= `signals.entity_query`)
- All new `RetrievalProfile` fields have defaults; existing manifests load unchanged
- `selected_chunks=None` preserves current scoped-doc fallback in docling-graph and keeps `apply_chunk_scope` on the legacy worker path

**Eventual cleanup (not this slice):** the per-element retrieval path and the `build_retrieval_query` str shim are retained for compatibility. The `per_element` retirement clock already started **2026-05-28** (per `48302f2`), with earliest deletion **2026-06-11** subject to Phase 2 A/B completion (`docs/handoffs/2026-05-28-merged-mode-bundle-propagation.md`). Remove the per-element path + `build_retrieval_query` shim on/after that date once A/B confirms, so the dual path does not become permanently load-bearing.

---

## Test Plan

**Phase B0 (prerequisite):**
- `test_extraction_query_builder.py` green on the unmodified branch (5 prior failures re-baselined)

**Unit tests (must pass before integration):**
- `test_extraction_query_builder.py` — extended for `build_retrieval_profile`, field-query enumeration, `system_name` skipping, INTERNAL skipping, backward-compat string output, AND `build_retrieval_query` byte-identity parity
- `test_retrieval_profile_config.py` — new config fields load with defaults; explicit overrides accepted; unknown fields rejected
- `test_extraction_lexical_search.py` — case-insensitive substring match; per-chunk hit counts; negative terms tracked separately; NFC Cyrillic stability
- `test_extraction_candidate_scoring.py` — merge dedup; `page_number` preserved; self_ref-collision-with-distinct-vertex_id guard; provenance preserved; **post-rerank scoring** (rerank-only order preserved when no keyword signal; a keyword/unit hit promotes a chunk above a higher-rerank chunk lacking it; negative penalty demotes but never eliminates; all-zero keyword signal leaves rerank order unchanged); sort stability; all weights incl. `rerank_weight` read from cfg; **`active_fields`** (§9: drops zero-evidence fields, keeps evidenced + identity/required, no-op when flag off)
- `test_hybrid_chunking_tables.py` — table detection; adapter `.text` used (not `contextualize`); self-contained row rendering; header repetition across split tables
- `test_phase2_selected_chunks_forwarding.py` (DONE in `27a0c64`) — worker captures `router_response["selected_chunks"]`; legacy path unchanged when criteria not met. **Add** the `apply_chunk_scope`-skipped-only-on-merged-forward assertion when the A3 skip lands.
- `test_extraction_routing_fallback.py` — each ladder level; all-noise triggers fallback; `would_skip` survives with `fallback_to_full=False`
- identity-anchor channel (in `extraction_chunk_search`/routing tests) — fires when `identity_anchors` present in the request; silent no-op when `None` (§8 fix)
- subset-schema (`active_fields` + request) — `field_subset` carried only when `subset_schema_extraction` on; a field with evidence is never dropped (§9)

**Integration tests:**
- Phase A byte-identity + `apply_chunk_scope`-not-invoked coverage lives in `test_phase2_selected_chunks_forwarding.py` (byte-identity DONE in `27a0c64`; the skip assertion is added with the A3 skip). A separate `tests/integration/...` file is optional.
- `test_chunk_scope_multi_channel.py` — multi-channel pulls more candidates than single-query alone; exactly one per-run SELECT; zero `vector_search` calls (Phase C; mirror `tests/integration/test_extraction_chunk_filter_starvation.py` guard)
- `test_table_chunks_indexed.py` — table chunks discoverable via lexical + pattern (Phase D)

**Recall validation — BLOCKING GATE (executed end-to-end):**
- The blocking gate is an **end-to-end extraction run** on `merged_v1` for Dvina + SA-2, comparing extracted entity counts (single-query/shadow baseline vs multi-channel/narrow_only) against the bdde417 baseline floors. The gate is on extracted-entity counts, NOT on the `c10` gt_coverage metric.
- Acceptance: multi-channel extracted-entity counts are **>= the baseline floors** (Dvina missile_kinematics ≥ 2, SA-2 radar_power_rf ≥ 22, SA-2 missile_kinematics ≥ 16) AND **>= single-query counts**; `selected_token_estimate` within 1.2× of single-query (recall gains, not LLM-input bloat).

**Coefficient calibration — NON-GATING (exploratory only):**
- `scripts/c10_phase1_merged_chunk_calibration.py` may be extended to sweep the post-rerank boost coefficients, BUT its ground truth is defined by the dense+reranker ordering and therefore CANNOT measure recall gains from the new lexical/pattern/table channels. Use it only to explore relative coefficient behavior; never as the recall merge gate.

**Telemetry/diagnostics tests:**
- After a real chunk-scope call, assert these JSON paths exist:
  - `diagnostics_json->router->channel_counts`
  - `diagnostics_json->router->field_coverage`
  - `diagnostics_json->router->score_components` (non-empty array for `mode=selected_refs`)
  - `diagnostics_json->router->fallback_level`
  - `diagnostics_json->router->selected_chunks_forwarded`
  - `diagnostics_json->router->selected_chunks_forwarded_count` (worker-native)
  - `diagnostics_json->router->selected_chunk_count` (endpoint field; reaches router via the envelope flatten)
  - `diagnostics_json->router->selected_chunk_token_estimate` (endpoint field; via flatten)

---

## Locked Decisions Recap

The former discussion gates are resolved in this plan. Implementers should proceed with the locked defaults unless direct code inspection reveals a blocker: byte-identical selected-chunk forwarding (with `apply_chunk_scope` skipped on merged forward), retrieval metadata shape as documented, in-memory lexical/pattern scanning over ONE per-run direct-SQL row pull (never HNSW), candidate pool capped before rerank, **keyword/pattern precision boost applied AFTER rerank and before the top_k cut** with weights as `RetrievalProfile` config fields (BM25 deferred in lieu of this keyword channel), table-normalization reuse, graduated fallback before full-document fallback, a functional worker-supplied identity-anchor channel (§8), and opt-in subset-schema extraction (§9).

---

## Acceptance Criteria for Plan Completion

A subagent-driven execution of this plan is **done** when all of the following hold:

1. Phase B0 plus all phases (A, B, C, D, E, F) have completed all tasks with green tests.
2. Every bundle exercised in merged mode shows **non-decreasing recall** vs its current Phase 1 baseline, measured by the **executed end-to-end entity-count gate** (not gt_coverage). Minimum named floors: Dvina missile_kinematics ≥ 2 ents, SA-2 radar_power_rf ≥ 22 ents, SA-2 missile_kinematics ≥ 16 ents, AND multi-channel ≥ single-query on both docs. If production `air_defense_v3` (all 9 passes) is run in merged mode, no live pass may regress vs its baseline.
3. **No regression on wall** beyond 120% of the *relevant* solo baseline. Narrowed bundles: Dvina ≤ 17.5 m, SA-2 ≤ 274 m. Production `air_defense_v3` (9 field-group passes): anchor the cap to the existing **bdde417 ~303m full-bundle SA-2** baseline (and the corresponding Dvina full-bundle baseline) — re-measure on a solo run; do NOT apply the 2-pass 274m number to the 9-pass bundle (multi-channel cost amplifies ~4.5×).
4. **Runtime-reach reality (documented, not a defect):** multi-channel fires only when `EXTRACTION_INDEX_MODE=merged` + `VECTOR_ROUTER_MODE=narrow_only`. After `48302f2`, `EXTRACTION_INDEX_MODE=merged` is now the `.env.example` default, so the **binding gate** keeping production behavior unchanged is `VECTOR_ROUTER_MODE=shadow` (still default) — merging this plan changes production only when an operator flips the single remaining gate to `narrow_only`. The "applies to all bundles" scope is config/metadata reach, not automatic runtime activation. (This worktree's live `.env` already has both gates active.)
5. Per-pass diagnostics contain the new multi-channel, selected-chunk-forwarding, field-coverage, fallback, and subset-schema fields and round-trip through postgres correctly.
6. Multi-channel retrieval fetches `ExtractionChunk` rows once per pass via the direct per-run SQL, batches all query embeddings, introduces no per-field database row pull, makes zero HNSW/`vector_search` calls, caps the candidate pool to `cfg.top_n_candidates` before the cross-encoder rerank, and applies the keyword/pattern boost AFTER rerank but before the top_k cut.
7. §9 subset-schema (when `subset_schema_extraction` is enabled) never drops a field that had retrieval evidence, and subset-on extracted-entity counts are `>=` subset-off on both gate docs.
8. Final code-reviewer subagent approves the entire branch.

---

## Open Risks

- **Lexical/pattern search on Russian-language content** — the SA-2 doc has Cyrillic page-2+ content. Both the lexical channel (C2) and the pattern channel (C3) use NFC-normalized matching to avoid false negatives on composed/decomposed Cyrillic.
- **Per-field dense queries cost embedding** — each narrowed pass currently embeds 1 query; multi-channel embeds N (one per field, ~5-6). Batched query embedding (one `embed_texts` call), one chunk-row fetch/matrix multiply, and the pre-rerank candidate cap mitigate the cost; net latency should stay small relative to rerank/extraction.
- **Rerank pair-count growth is the dominant new CPU cost** — `rerank()` scores every candidate in the capped pool. The `cfg.top_n_candidates` cap before rerank (C6) is the key control keeping SA-2 under its wall cap (274m narrowed / ~303m full-bundle bdde417).
- **Post-rerank boost weights are conjecture** until calibrated on real data. They are now config fields (`rerank_weight`/`lexical_weight`/…), so calibration is a manifest edit; the `c10` sweep is exploratory only (its GT cannot measure true recall).
- **Table rendering token-budget overrun** — large tables (SA-2 has tables with 20+ rows) may exceed 512 tokens. D2 tests enforce header propagation into every split fragment.
- **HNSW post-filter starvation reintroduction** — the multi-channel path must reuse the direct per-run SQL and never call HNSW; a starvation-guard test mirrors the existing filter-starvation integration guard. (This is also why **BM25 stays deferred**: a Lucene/BM25 chunk index ranks globally then post-filters by run — the same starvation mismatch — so the curated keyword channel is used in its place.)
- **Schema-wide rollout amplifies cost on the production bundle** — applying multi-channel to all 9 `air_defense_v3` passes (vs 2 in the narrowed bundles) multiplies the new embedding/lexical/rerank work per document by ~4.5×. The pre-rerank candidate cap (Blocker 6) and batched embedding keep this bounded, but the production-bundle wall MUST be re-baselined against bdde417 (~303m), not assumed equal to the 2-pass 274m number (acceptance #3).
- **Runtime reach is opt-in** — `EXTRACTION_INDEX_MODE=merged` is now the `.env.example` default (`48302f2`), but `VECTOR_ROUTER_MODE=shadow` is still default, and `shadow` forces RUN_FULL with no narrowing/forwarding. So none of the new B/C/D/E/F runtime path fires until an operator flips the single remaining gate `VECTOR_ROUTER_MODE=narrow_only`. The schema/manifest edits ship to all bundles, but the retrieval upgrade reaches production only on that flip (acceptance #4). Don't assume merging this plan changes production extraction.
- **Subset-schema over-pruning (§9)** — too-aggressive field subsetting could drop a field that is actually present, reducing recall. Mitigations: opt-in (`subset_schema_extraction=False` default), conservative active-set (any evidence ⇒ keep; identity/required always kept), and a recall guard (F4) requiring subset-on entity counts `>=` subset-off on the gate docs.
- **Table synthesis inflates embedding count / index size** — `build_extraction_index_hybrid` synthesizes a per-row TextItem for every table row and embeds them in the index batch (a 20-row SA-2 table → ~20 extra chunks). Once-per-run so amortized, but it enlarges the embedding batch and ArcadeDB row count. Mitigation: log `synth_count` into index diagnostics + a per-table row cap (see Performance enhancements).
- **`web_cruft_sanitizer` blanks ~27% of texts and is a hand-forked copy** — on Dvina it blanked 49/182 texts; whole-item blanking triggers on any single blob-like token, so a prose paragraph containing one long encoded token is dropped entirely (silent recall risk). The module is also a manual fork of `docker/docling-graph/app/main.py` that must stay in sync (the worker copy notes the obligation; the docling-graph original has no reciprocal pointer). Mitigation: D6 (blanked-text audit + reciprocal SYNC back-pointer). Note the worker import is NOT fail-soft (bare `import`), unlike the table-norm import.

---

## Performance enhancements (folded from review — optional, do alongside the phases)

These are free or near-free wins the current code leaves on the table; none are blockers.

1. **Memoize `build_retrieval_profile(pass_def, template_cls)`** per `(bundle_key, pass_name)` via `lru_cache` — it is deterministic from schema metadata but is rebuilt on every `chunk_scope` call (and every retry). Mirror the `_get_tokenizer` cache pattern (`hybrid_chunking.py:57`).
2. **Skip the reranker ONLY when the capped pool ≤ `cfg.top_k`** (C6 fast-path) — then every candidate is selected regardless of order, so reranking changes nothing observable. Do NOT generalize to pools > `top_k`: the post-rerank `final` score has no dense term, so skipping rerank there zeroes `rerank_norm` for the whole pool and collapses ordering onto keyword boosts alone.
3. **Reuse both matrices in the relaxed-dense fallback** (E2) — already folded in: reuse the query embedding AND the per-run chunk-embedding matrix; relaxed dense is a threshold change, no re-embed/re-fetch.
4. **Table-synthesis embedding-count diagnostic + soft cap** — log `synth_count` into the index-build diagnostics; add a per-table row cap so a pathological 100+-row table doesn't balloon the embedding batch.
5. **Quantify the `apply_chunk_scope`-skip win (A3)** — log `apply_chunk_scope` wall before/after the skip on one prod run so the CPU saving is on record (justifies the added skip branch).

---

## Out-of-scope reminders (sections 8-10, follow-up plan)

The following are NOT in this plan and should be punted to a follow-up:
- Adding the other 7 field-group passes to the *narrowed* bundles (`merged_v1`/`narrowing_v1`/`baseline_subset`) — they stay 2-pass by design (production `air_defense_v3` already routes all 9)
- **§8 (remainder):** promoting the now-functional identity-anchor channel to a **default-on mode change** (the channel itself is in scope and functional; only the default promotion is deferred)
- **§9 (remainder):** BM25 lexical index — **deferred in lieu of the curated keyword channel** (a Lucene/BM25 chunk index ranks globally then post-filters by run, reintroducing the per-run starvation mismatch); and LLM field-aware reranking (the post-rerank keyword boost already provides deterministic field-aware reranking)
- **§10 (all):** strict quality mode requiring grounded field evidence; centralized quantity normalization across radar + missile schemas; conflict diagnostics for divergent values; extraction-result caching

These layer onto the infrastructure this plan establishes once retrieval quality is measured and validated.
