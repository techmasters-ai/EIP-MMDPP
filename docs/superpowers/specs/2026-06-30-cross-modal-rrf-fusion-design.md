# Cross-Modal RRF Fusion for Multi-Modal Retrieval — Design

- **Date:** 2026-06-30
- **Status:** Approved (brainstorming) → pending implementation plan
- **Scope:** `_multi_modal_pipeline` (hybrid retrieval) in `app/api/v1/retrieval.py`

## Problem

Multi-modal (hybrid) retrieval merges results from **incompatible scorers** onto one
0–1 axis and sorts them together:

- **cross-encoder** text relevance (`bge-reranker-v2-m3`, ~0.1–0.99)
- **SigLIP** visual match probability (~0–0.93)
- **cross-modal expansion** decay score (text-proximity, *not* a query-relevance measure)

A SigLIP `0.5` is not comparable to a cross-encoder `0.9`, so the ordering of a mixed
list is not meaningful. True calibration to a common relevance probability would require
labeled cross-modal relevance data we do not have.

## Goal

One **merged, comparably-ordered** result list for hybrid queries via **Reciprocal Rank
Fusion (RRF)** — a calibration-free rank-aggregation method (industry standard for fusing
ranked lists from heterogeneous retrievers). Realize: *"an image is relevant if it matches
visually OR by description."*

## Non-goals

- Calibrated relevance probabilities (not achievable without labels).
- Any change to **Text Basic** (single-modality `strategy=basic`).
- A VLM cross-modal reranker (possible future "deep search" toggle).
- Faceted/grouped UI (considered, rejected in favor of the merged list).

## Design

### Integration point

Only `_multi_modal_pipeline` (`strategy=hybrid`) changes. RRF **replaces the final
ordering** — today's `sort by fused score → _apply_reserved_slots → single _apply_reranker`.
Everything upstream is unchanged: seeds (`_text_vector_search`, `_image_vector_search`),
graph expansion, dedup/diversify, and the per-signal quality gates built earlier this session
(reranker floor 0.05, image SigLIP floor).

### Signals (each a ranked list, produced after its own existing gate)

1. **S_text** — cross-encoder relevance over every chunk that has text: `text`, `table`,
   `image_description`, **and `image` chunks scored via their caption (`content_text`)**.
   Junk removed by the floor-0.05 reranker gate. Ranked desc by reranker score.
2. **S_visual** — SigLIP match probability over `image`/`schematic` chunks (image floor
   applied). Ranked desc by SigLIP prob.
3. **S_expand** — cross-modal/ontology expansion association (decay score). Ranked desc.
4. **S_ontology** — qualifying `ontology_relation` chunks (relation weight ≥ min, raw
   cosine ≥ min), ranked by relation weight. **This folds the former reserved-slots
   guarantee into RRF as a soft signal** (per design decision: no hard guarantee).

A chunk may appear in multiple signals (an image in S_visual *and* S_text via its caption;
an ontology chunk in S_text *and* S_ontology).

**Gate semantics (important):** a per-signal gate controls **membership in that signal**,
not removal from the candidate set. If an image's caption scores below the floor-0.05
reranker gate, it is simply **absent from S_text** but still carried by S_visual — so it is
not dropped. This *supersedes* the earlier "images bypass the reranker" patch (commit
`33063fe`): under RRF, images are not bypassed, they participate in S_text via caption and
in S_visual via SigLIP, and survive as long as they rank in **any** signal. The
`RETRIEVAL_RRF_FUSION_ENABLED=false` path retains the bypass behavior unchanged.

### Fusion

```
RRF(c) = Σ over signals S where c ∈ S:  w_S / (k + rank_S(c))
```

- `k` = `RETRIEVAL_RRF_K` (default **60**, the standard constant)
- `w_S` per-signal weights: `w_text=1.0`, `w_visual=1.0`, `w_expand=0.5`, `w_ontology=0.5` (tunable)
- `rank_S(c)` = 1-based rank of `c` within signal `S` (best = 1)

Sort by `RRF(c)` desc; tiebreak by `chunk_id` (deterministic). Trim to `top_k`.

### "Best of visual + description" — emergent, not explicit

An image present in S_visual and/or S_text accrues RRF from each list it ranks in. RRF
rewards multi-list presence, so a Fan Song photo surfaces if SigLIP sees it **or** its
caption matches — **without ever comparing a SigLIP score to a cross-encoder score**. No
explicit `max()` needed.

### Display score (fixed, absolute)

```
display(c) = RRF(c) / (RRF(c) + C)      C = RETRIEVAL_RRF_DISPLAY_SCALE (default 0.02)
```

Monotonic, **stable across queries** (a given RRF always maps to the same display value),
0–1 range. Labeled as a fused relevance, not a probability. `min_confidence` is applied to
`display(c)`.

### Config / rollback

- `RETRIEVAL_RRF_FUSION_ENABLED` (default **true**) — off restores today's reserved-slots +
  single-rerank ordering. Instant rollback, no redeploy.
- `RETRIEVAL_RRF_K` (60), `RETRIEVAL_RRF_W_TEXT/VISUAL/EXPAND/ONTOLOGY`,
  `RETRIEVAL_RRF_DISPLAY_SCALE` (0.02).
- All mirrored into `.env` and `.env.example` (per project convention).

### Edge cases

- **Single-modality hybrid** (e.g. `modality_filter=text`): RRF over one surviving signal =
  identity ranking. Harmless.
- **Empty signals**: skipped in the sum.
- **Determinism**: builds on exact-kNN vector search; the `chunk_id` tiebreak keeps RRF
  output deterministic run-to-run.
- **Text Basic**: not touched — `strategy=basic` does not enter `_multi_modal_pipeline`.

## Verification

- `"Fan Song"` / All → text + image interleaved, sensible order, deterministic across runs.
- `"radar antenna"` / All → strong images rank near strong text (both signals contribute).
- Text Basic `"Fan Song"` → byte-identical to pre-change.
- `RETRIEVAL_RRF_FUSION_ENABLED=false` → identical to current behavior.

## Testing

- **Unit:** RRF math (known ranks → known fused order), the display transform, tiebreak
  determinism, multi-list accumulation.
- **Integration:** the live queries above through `POST /v1/retrieval/query`.
