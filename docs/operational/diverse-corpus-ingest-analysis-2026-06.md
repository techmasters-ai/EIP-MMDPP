# Diverse-Corpus Ingest Analysis — graceful-completion review

Status: **FINALIZED** (ingest complete 2026-06-20 00:39Z). No code changed — analysis only.
Collection: `guarded_ranker_eval_2026-06` (source `5e31fa7c…`, bundle `air_defense_v3`, `selection_mode=topk`, router shadow). 14 docs (4 SA-2 carried from the prior run + the Fandom doc + 9 new types).
Owner goal: **every doc should end `COMPLETE`, even with zero extraction.** Un-extractable / off-domain docs must ingest gracefully, not error.

## Per-doc outcomes (final)

C/S/F = pass execution_status counts (COMPLETE / SKIPPED / FAILED of 12). SCA = passes with `score_components_all`.

| Doc | run status | C | S | F | SCA | Mode | Desired? |
|---|---|---:|---:|---:|---:|---|---|
| Images_Demo_Doc | COMPLETE | 12 | 0 | 0 | 9 | A full | ✅ |
| SNR-75 (Wikipedia) | COMPLETE | 12 | 0 | 0 | 9 | A | ✅ |
| S-75 Dvina | COMPLETE | 12 | 0 | 0 | 9 | A | ✅ |
| V-75 SA-2 Guideline | COMPLETE | 12 | 0 | 0 | 9 | A | ✅ |
| S-75 Dvina (Fandom) | COMPLETE | 12 | 0 | 0 | 9 | A | ✅ |
| chinese_handwritten_notes | COMPLETE | 0 | 0 | 0 | 0 | B clean-empty | ✅ (target behavior) |
| chinese_handwritten_notes_2 | COMPLETE | 0 | 0 | 0 | 0 | B | ✅ |
| NASA combined (lf99-4297) | COMPLETE | 7 | 0 | 5 | 9 | A/C mix | ⚠️ run COMPLETE but 5 off-domain radar passes FAILED |
| radar_textbook_chapter7 | COMPLETE | 3 | 1 | 8 | 9 | A/C mix | ⚠️ run COMPLETE but 8 passes FAILED |
| Radar Basics | **PARTIAL_COMPLETE** | 6 | 0 | 6 | 9 | C | ❌ should be COMPLETE |
| Handwritten_Text | **PARTIAL_COMPLETE** | 4 | 0 | 8 | 9 | C | ❌ |
| radar2_waveform1 | **PARTIAL_COMPLETE** | 0 | 1 | 11 | 9 | C | ❌ |
| chinese_research_paper | **PARTIAL_COMPLETE** | 0 | 1 | 11 | 9 | C | ❌ |
| EWIRDB_Production | **FAILED** | 5 | 0 | 7 | 9 | D required-pass fail | ❌ should be COMPLETE |

Result: 9 COMPLETE, 4 PARTIAL_COMPLETE, 1 FAILED. Robustness held (all terminal, no hang/crash, `error_message` empty, driver healthy).

## Root cause (unified)

**Passes that find nothing — off-domain or empty — are recorded `FAILED` (`ExtractionError` / "Quality gate failed: missing_root_instance, empty_output") instead of a clean `SKIPPED`/`ZERO_YIELD`.** Evidence: the FAILED passes track the doc's *domain*, not random failure —
- Radar Basics (radar doc): radar passes ✅, **all 6 missile passes FAILED** (off-domain).
- NASA combined (missile doc): all missile passes ✅, **all 5 radar passes FAILED** (off-domain).
- radar_textbook: 3 radar ✅, rest FAILED.
- radar2_waveform / chinese_research_paper (no text layer): **all 11 entity passes FAILED** (with the extra "no chunk metadata available from doc_processor" signature).

The system already has `_is_clean_empty_pipeline_error` meant to turn off-domain/empty into `ZERO_YIELD` (SKIPPED, no error). It is **not catching these** — so empty passes land FAILED.

### Why the three bad terminal labels differ
- **C → PARTIAL_COMPLETE** (Radar Basics, Handwritten, radar2_waveform, chinese_research_paper): entity passes FAILED but `system_links` was cleanly `SKIPPED` (`NO_UPSTREAM_ENDPOINTS` — 0 entities to link), so the `derive_ontology_graph` **stage** stayed COMPLETE and the run escalated to PARTIAL via the empty/failed-pass path.
- **D → FAILED** (EWIRDB): the doc *did* extract 5 passes, so `system_links` (a **required** relationship pass) actually ran — and **FAILED with ExtractionError, retried 4× (00:17–00:38Z)**. A failed *required* pass → `check_required_pass_gate` fails → `_update_summary_stage_run(FAILED)` → `derive_ontology_graph` **stage FAILED** → **run FAILED** (`pipeline.py:9136-9145`).
- **A/C mix → COMPLETE-with-FAILED-passes** (NASA, radar_textbook): enough passes succeeded + `system_links` succeeded → gate passed → stage COMPLETE → run COMPLETE, *despite* off-domain passes being FAILED. (So even "COMPLETE" docs carry spurious pass-level FAILEDs.)

So the run-level label (COMPLETE / PARTIAL / FAILED) is incidental to **whether `system_links` happened to skip, succeed, or fail** — all three trace back to the same defect: empty/off-domain passes错误-classified as FAILED.

## Two empty sub-signatures (distinguish during remediation)
1. **Off-domain on a text doc** (e.g., missile passes on Radar Basics): the doc HAS chunks (sibling radar passes extracted fine); the pass simply finds no in-domain entities → empty → "missing_root_instance, empty_output". **Legitimately empty → must clean-skip.**
2. **No text layer** (scanned/handwritten/diagram: radar2_waveform, chinese_*): docling-graph's `doc_processor` reports **"no chunk metadata available … provenance will be empty"**. Ambiguous: (a) genuinely no text layer (legit) vs (b) a chunk-metadata **propagation bug** (the worker's chunk-scope saw chunks — SCA=9 — yet docling-graph saw none). Diagnose before masking: do NOT blanket-skip if it's (b), or it hides a real defect (cf. the `EXTRACTED_FROM` `chunk_id=None` history).

## Remediation target (DEFERRED — to address after chunk-selection testing)
Goal: **all four ❌/⚠️ outcomes become graceful `COMPLETE`.** Three coordinated fixes:

1. **Empty/off-domain entity pass → `SKIPPED`/`ZERO_YIELD`, not `FAILED`.** Extend `_is_clean_empty_pipeline_error` to recognize the `missing_root_instance + empty_output` signature (sub-case 1) as a clean zero-yield. → off-domain passes stop counting against the gate; single-domain docs (Radar Basics, NASA) → COMPLETE; pass-level FAILEDs disappear from the COMPLETE docs too.
2. **`system_links` must handle "no/partial relationships" gracefully** (the EWIRDB FAILED driver). A required relationship pass that finds nothing should `SKIP`/zero-yield, not `ExtractionError` → run COMPLETE instead of FAILED. **UPDATE (2026-06-22): EWIRDB's failure is CONFIRMED volume-driven, not empty** — a clean re-run on healthy infra reproduced it (FAILED, 2/12 passes). It's the *oversized-input* scaling defect, documented separately below ("Deferred defect: oversized-table extraction batching").
3. **Diagnose the "no chunk metadata" sub-case (2b)** for scanned docs: confirm legit-empty vs propagation bug by comparing the worker chunk count (had chunks → SCA=9) against docling-graph's `doc_processor.last_chunk_metadata`. If bug → fix the hand-off (these docs would then actually extract); if legit → clean-skip per fix #1.

Acceptance: re-ingest the 5 ❌/⚠️ docs → all terminate `COMPLETE`; no pass-level `FAILED` for off-domain/empty passes (they show `SKIPPED`); EWIRDB completes.

## Deferred defect: oversized-table extraction batching (EWIRDB — CONFIRMED volume-driven)

**Status:** DEFERRED — separate from graceful-completion (fixes #1–#3 above) AND orthogonal to the chunk-selector. Confirmed 2026-06-22 by a clean `reingest graph_only` of EWIRDB on healthy infra (docling-graph up, LLM hosts 200): reproduced FAILED, 2/12 passes, `narrowed=9`. So it is not the docling-graph outage and not narrowing — it bites full-doc extraction the same way.

**Symptom.** EWIRDB_Production (a production-database-dump PDF) fails most passes with `ExtractionError` → `retry_exhausted` → pass FAILED → run FAILED. Quality gate: `empty_output`. The LLM (`gemma4:31b`) returns nothing usable because the prompt is enormous.

**Evidence.** docling-graph log for the failing pass: `markdown_chars=8,609,421` (~8.6M characters of markdown in a single extraction input) while the worker's `full_doc_token_estimate=27,195` — i.e. ~**316 chars/token** vs the normal ~4, an **~80× undercount**. The doc indexed 72 chunks; EWIRDB's content is huge raw tables (table-norm is disabled upstream: `EXTRACTION_INDEX_UPSTREAM_TABLE_NORM=false` → raw markdown tables land verbatim).

**Root cause (two compounding flaws in the delta batcher):**
1. **No intra-chunk splitting.** `chunk_batches_by_token_limit()` (`docling_graph/core/extractors/contracts/delta/helpers.py:132`) only decides where to cut *between* chunks — it packs whole chunks until the running sum would exceed `max_batch_tokens`, then starts a new batch, but it **always appends the current chunk even if that single chunk alone exceeds the limit**. A chunk == one document element, and docling-graph keeps a table as one element. So a single multi-million-char table element becomes **one over-budget batch = one giant prompt**. (`max_batch_tokens` = `DOCLING_GRAPH_LLM_BATCH_TOKEN_SIZE=512` for field passes, `…SYSTEM_LINKS…=4096` for `system_links`.)
2. **Token counts undercount tables.** The batcher's `token_counts` come from `chunk_metadata["token_count"]` or fall back to `len(chunk.split())` (`orchestrator.py:530-538`, `runtime.py:339-344`). Both badly undercount dense numeric tables (the ~80× gap above), so the batcher (and the worker's `narrow_min_doc_tokens` size-gate, which keys on the same estimate) are blind to the true volume — they "see" a small doc and never force enough batches.

**Why most docs are fine.** Normal prose chunks are well under 512 tokens, so the batcher produces many small prompts. Only docs with pathologically large single elements (DB-dump tables) trip the gap. EWIRDB is the only doc in the corpus that does.

**Fix sketch (when addressed):**
1. **Row-wise / intra-element table splitting before batching:** split a large table element into multiple bounded chunks (header repeated + N rows each) so no single chunk exceeds `max_batch_tokens`.
2. **Accurate size accounting:** count by characters (or a real tokenizer), not `len(split)` / metadata `token_count`, so both the batcher and the worker size-gate see true volume. Propagate the same char-based measure to `full_doc_token_estimate` / `narrow_min_doc_tokens`.
3. **Defensive cap:** if a single chunk still exceeds `max_batch_tokens` after splitting, truncate-with-diagnostic or clean-skip that chunk rather than emitting a multi-MB prompt that deterministically returns `empty_output`.

**Acceptance:** re-ingest EWIRDB → no single extraction prompt exceeds ~`max_batch_tokens`; passes COMPLETE (or clean-skip per graceful-completion fix #1/#2) instead of FAILED-on-oversized-input.

**Cross-ref:** the char-based size accounting (fix #2) is the same blind spot behind the worker-side `narrow_min_doc_tokens` recall-safety gate — both trust `full_doc_token_estimate`, which undercounts table-heavy docs.

## Cross-refs
- `ISSUES.md` (this collection) — raw per-doc flags.
- `structured-output-decoding-review-2026-06.md` — the constrained-decoding deep-dive (separate deferred item).
- `production-reliability-2026-06.md` §R4 — failure-cleanup behavior (related).
