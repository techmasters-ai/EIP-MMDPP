# Large-Scale Extraction Data-Collection Plan

**Status:** PLANNING (discussion gate — not started)
**Branch:** walltime/c0-telemetry (on hold; this is investigation, not a merge blocker)
**Date:** 2026-05-30

## Goal

Collect extraction + retrieval data across the full `notebooks/` document corpus
(~20 docs) to (a) give the **document-shape variety** the SA-2-only ingested
corpus lacks — needed to design entity-instance-aware chunk-coverage
(see `project_threshold_selection_vs_topk`), and (b) **test the recent
graph-extraction modifications** (gemma4 death-spiral fix + the B/C/E/F retrieval
stack) through the **full ingest pipeline** end-to-end (every gate run so far was
`graph_only` against a cached parse — this is the first full parse→docling→embed→graph
exercise of the new code).

## Data collected per (document × pass)

1. **Per-pass recall:** primary_entities_extracted, relationships_extracted, execution_status.
2. **Retrieval/coverage signals** (chunk-scope replay): field_coverage, channel_counts,
   score_components (final/rerank), min-set-cover size, selected_ref_count vs total chunks.
3. **Chunk/index stats:** total ExtractionChunk count, tokens, chunk-size dist;
   distinct entity/system count per doc (the multi-instance axis).
4. **Wall-time + gemma4 truncation events** per pass.

## Configuration (DECIDED)

- **Bundle:** `air_defense_v3` (production, 9 field-group passes). NOT merged_v1 (5 passes).
- **Router A/B:** every doc collected under BOTH:
  - `VECTOR_ROUTER_MODE=shadow` → no narrowing → **ground-truth recall** (full-doc extraction).
  - `VECTOR_ROUTER_MODE=narrow_only` → current narrowed behavior + coverage signals.
  - The shadow vs narrow_only delta per (doc×pass) is the definitive "does narrowing lose recall" measurement.
- Deployed baseline env: `EXTRACTION_INDEX_MODE=merged`, `WORKER_FORWARD_SELECTED_CHUNKS=false`,
  `DEFAULT_ONTOLOGY_BUNDLE_KEY=air_defense_v3`. Switching router mode = container env change + `up -d --force-recreate` (from worktree path, `-p eip-mmdpp`).
- **Idle LLM pool is mandatory** for every wall-time number (probe active-inference latency, not /api/ps).

## Document buckets (verified against ingest.documents)

- **Bucket A — 8 NEVER ingested → `mode: full` (upload first):** chinese_research_paper,
  "Engagement and Fire Control Radars (S/X-band)" (~30MB), **EWIRDB_Production (likely multi-distinct-system — the key missing case)**, Handwritten_Text, radar2_waveform1, Radar Basics, radar_textbook_chapter7, "S-75 Dvina _ Military Wiki _ Fandom".
- **Bucket B — 13 already ingested (stages_done=12) → `graph_only`** reuses cached parse.
  (S-75 Dvina + SA-2 Guideline show 10 stages only because their last run was a graph_only gate run — collect_derivations + derive_canonicalization are legitimately skipped in that mode; parse/embed intact. NOT a deficiency.)

## Phases

**P0 — Harness + dry-run (no LLM):** write the collection script (extends the durable
`gate_monitor.py` pattern: setsid-detached, CSV + per-pass rows, postgres + ArcadeDB
as source of truth). Dry-run the chunk-scope/stats collectors against the 2 existing
live indexes to validate output schema. Cheap.

**P1 — Full-ingest the 8 (Bucket A), narrow_only:** upload via `POST /v1/sources/{SA-2_Sources}/documents`
(multipart), then full ingest on `air_defense_v3`. **This is the end-to-end test of the new
graph-extraction code.** Watch parse/OCR on the 30MB radar PDF + handwritten/Chinese docs
(stress cases). Collect recall+coverage+stats+wall/truncation as each completes.

**P2 — graph_only re-run the 13 (Bucket B), narrow_only:** cheaper (cached parse); collect same.

**P3 — Shadow A/B pass:** flip `VECTOR_ROUTER_MODE=shadow`, re-run all ~20 (graph_only now that
all are ingested) to capture ground-truth recall; pair with the narrow_only numbers.

**P4 — Synthesis:** per-doc + per-pass tables; recall(shadow) vs recall(narrow_only) delta;
coverage/min-set-cover vs chunks-sent; flag docs where narrowing lost recall (the coverage-design
targets) and docs where a schema was absent (would-skip candidates).

## Cost & risks (honest)

- **Large.** 8 full ingests + 20 graph_only (narrow) + ~20 graph_only (shadow). Each
  field-group pass ran 30–280 min on SA-2 even idle. Expect MANY hours total; run detached, checkpoint per doc.
- The 30MB radar PDF + handwritten/Chinese docs may stress parse/OCR or fail — capture failures as data, don't let them block the batch.
- gemma4 truncation will vary by doc; the death-spiral fix is deployed but truncations still occur (recover, don't collapse) — track per doc.
- Data persists in postgres `ingest.pipeline_pass_outputs` + ArcadeDB `ExtractionChunk` regardless of session; the detached collector is a convenience layer, not the system of record.

## Open question for synthesis
Ground-truth recall (shadow) requires the LLM to extract from the FULL doc — which itself
isn't perfect recall (LLM misses things). True "did narrowing lose a real fact" needs a
human/pseudo-GT per doc. Shadow is the best automated proxy; note this limitation in P4.
