# Diverse-Corpus Ingest Analysis — graceful-completion review

Status: **PRELIMINARY (ingest in progress).** No code changed. Finalize when the run completes (radar_textbook / NASA combined / EWIRDB still pending).
Date: 2026-06-19
Collection: `guarded_ranker_eval_2026-06` (source `5e31fa7c…`, bundle `air_defense_v3`, `selection_mode=topk`, router shadow)
Goal (owner): **every doc should end gracefully with status `COMPLETE`, even when no extraction takes place.** Un-extractable docs (scanned/handwritten/diagram/non-English) must ingest cleanly, not error.

## Per-doc outcomes (docs processed so far)

C/S/F = pass execution_status counts (COMPLETE / SKIPPED / FAILED of 12); SCA = passes with `score_components_all`.

| Doc | run status | C | S | F | SCA | Mode | Desired? |
|---|---|---:|---:|---:|---:|---|---|
| Images_Demo_Doc | COMPLETE | 12 | 0 | 0 | 9 | A — full extraction | ✅ |
| SNR-75 (Wikipedia) | COMPLETE | 12 | 0 | 0 | 9 | A | ✅ |
| S-75 Dvina | COMPLETE | 12 | 0 | 0 | 9 | A | ✅ |
| V-75 SA-2 Guideline | COMPLETE | 12 | 0 | 0 | 9 | A | ✅ |
| S-75 Dvina (Fandom) | COMPLETE | 12 | 0 | 0 | 9 | A | ✅ |
| chinese_handwritten_notes | COMPLETE | 0 | 0 | 0 | 0 | **B — clean empty (no passes ran)** | ✅ (this is the target behavior) |
| chinese_handwritten_notes_2 | COMPLETE | 0 | 0 | 0 | 0 | B | ✅ |
| radar2_waveform1 | **PARTIAL_COMPLETE** | 0 | 1 | **11** | 9 | **C — empty-but-errored** | ❌ should be COMPLETE |
| chinese_research_paper | **PARTIAL_COMPLETE** | 0 | 1 | **11** | 9 | C | ❌ should be COMPLETE |
| Handwritten_Text | **PARTIAL_COMPLETE** | 4 | 0 | **8** | 9 | C (partial) | ❌ should be COMPLETE |
| Radar Basics | (in progress) | 5 | 0 | 3 | 6 | trending C | ⚠️ watch |

## The three modes

- **Mode A — full extraction → COMPLETE.** Text PDFs. All 12 passes yield, run `COMPLETE`. Correct.
- **Mode B — clean empty → COMPLETE.** `chinese_handwritten ×2`: image-only/no-text docs produced **no extraction passes at all** (C=S=F=0), and the run still finalized `COMPLETE`. **This is exactly the owner's target** — graceful completion with no extraction. (0 SCA is fine; nothing to extract.) Driven by the "no text elements but has image elements → legitimate skip, do NOT escalate" branch (`pipeline.py:6196`).
- **Mode C — empty-but-errored → PARTIAL_COMPLETE.** `radar2_waveform`, `chinese_research_paper`, `Handwritten_Text`: the doc produced *enough* to run extraction passes (chunk-scope even captured SCA=9), but docling-graph's `doc_processor` got **no chunk metadata** → "Quality gate failed: missing_root_instance, empty_output" → the pass is recorded **`FAILED` (ExtractionError)** → the required-pass gate fails → `_terminalize_doc_and_run(..., "PARTIAL_COMPLETE")` (`pipeline.py:9136-9145`). The lone `SKIPPED` is `system_links` (`NO_UPSTREAM_ENDPOINTS` — clean, no entities to link).

## Root cause of the ❌ cases (Mode C)

The pipeline **already has** a clean-empty path: `_is_clean_empty_pipeline_error` is meant to treat an off-domain/empty extraction as `ZERO_YIELD` → `SKIPPED`/COMPLETE-EMPTY (no retry, no FAILED). Mode C docs slip past it: their failure signature is **"no chunk metadata available from doc_processor … provenance will be empty"** + **"Quality gate failed: missing_root_instance, empty_output"**, which is **not** recognized as a clean-empty, so the pass lands `FAILED`. One `FAILED` required pass is enough to flip the run to `PARTIAL_COMPLETE`.

So Mode C and Mode B are the *same reality* ("nothing extractable here") with two different classifications — only Mode B is caught as a clean skip.

### Important nuance before "just mark it clean"
The "no chunk metadata available from `doc_processor` (strategy_ops did not set `last_chunk_metadata` after chunking)" signature is ambiguous:
- **(a) Legitimately empty** — the doc has no text layer (scanned image/diagram) → no chunks → SHOULD clean-skip → run `COMPLETE`.
- **(b) Metadata-passing bug** — chunks existed but `last_chunk_metadata` wasn't propagated (cf. the `EXTRACTED_FROM` `chunk_id=None` root-cause history) → this would be a real defect that blanket "treat as clean-empty" would **mask**.

The fact that chunk-scope captured `SCA=9` (so the *worker* saw chunks) while docling-graph reported "no chunk metadata" leans toward **(b) a propagation gap** for these doc types, not pure emptiness. Must distinguish before changing classification.

## Target state + remediation sketch (DEFERRED — post-ingest)

Goal: **Mode C → behave like Mode B** (graceful `COMPLETE`, even with zero extraction).

1. **Diagnose the "no chunk metadata" signature** for Mode C docs: is `last_chunk_metadata` genuinely empty because the doc has no text layer (legit), or is it a propagation bug (chunks exist upstream but don't reach `doc_processor`)? Compare the worker's chunk count (it captured SCA) vs docling-graph's `doc_processor` view.
2. **If legit-empty:** extend `_is_clean_empty_pipeline_error` to recognize the `missing_root_instance + empty_output + no-chunk-metadata` signature as a clean `ZERO_YIELD` → pass `SKIPPED` (not `FAILED`) → required-pass gate passes → run `COMPLETE`.
3. **If propagation bug:** fix the chunk-metadata hand-off (so these docs actually extract) — separate, higher-value fix.
4. Either way, **a doc with genuinely nothing to extract must terminate `COMPLETE`**, never `PARTIAL_COMPLETE`, per the owner's bar.

## Verdict so far
- Robustness (no crash/hang/stuck): **PASS** — every doc reached a terminal state, `error_message` empty, driver healthy.
- Owner's "graceful `COMPLETE` even with no extraction": **PARTIAL** — Mode B docs meet it; Mode C docs (3 so far) land `PARTIAL_COMPLETE` due to empty-passes-classified-as-`FAILED`. Remediation above.

(To finalize: append radar_textbook / NASA combined / EWIRDB outcomes when the run completes; confirm the (a)-vs-(b) diagnosis.)
