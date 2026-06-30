# Cross-Modal RRF Fusion for Multi-Modal Retrieval — Design (v2)

- **Date:** 2026-06-30
- **Status:** Revised after 3 independent reviews → pending implementation plan
- **Scope:** hybrid retrieval ordering in `app/api/v1/retrieval.py`

## Problem

Multi-modal (hybrid) retrieval merges results from **incompatible scorers** onto one
0–1 axis and sorts them together:

- **cross-encoder** text relevance (`bge-reranker-v2-m3`, ~0.1–0.99)
- **SigLIP** visual match probability (~0–0.93)
- **cross-modal expansion** decay (text-proximity, *not* a query-relevance measure)

A SigLIP `0.5` is not comparable to a cross-encoder `0.9`, so the merged ordering is not
meaningful. True calibration to a common relevance probability requires labeled cross-modal
data we do not have.

## Goal

One merged, well-ordered hybrid result list via **Reciprocal Rank Fusion (RRF)** — the
**industry-standard, calibration-free** rank-aggregation method for fusing heterogeneous
retrievers (Cormack et al. 2009; the default "hybrid search" fusion in Elasticsearch,
OpenSearch, Qdrant, Weaviate, Vespa, Milvus). RRF is **agreement-based** (CombSUM-family):
an item endorsed by multiple signals ranks above one endorsed by a single weak signal —
the empirically-better behavior (Fox & Shaw 1994: CombSUM/MNZ > CombMAX). We explicitly
**reject** "best-of"/MAX fusion; it underperforms and is not standard.

## Non-goals

- Calibrated relevance probabilities (not achievable without labels).
- Any change to **Text Basic** (`strategy=basic`) ranking — must stay byte-identical.
- A VLM cross-modal reranker (possible future "deep search" toggle).
- Faceted/grouped UI.

---

## Design

### Fusion signals (RRF inputs)

RRF fuses **relevance signals only**. Each is an independently-ranked list, built **before**
the candidate merge/dedup so per-signal ranks survive:

1. **S_text** — cross-encoder relevance over text-bearing candidates (`text`, `table`,
   `image_description`, and image captions — see per-image unit below). Ranked desc by
   reranker score.
2. **S_visual** — SigLIP match probability over `image`/`schematic` candidates, **admitted
   only when the visual signal is real** (separation gate, below). Ranked desc by prob.
3. **S_ontology** — qualifying `ontology_relation` chunks (`rel_weight ≥ min`,
   `raw_cosine ≥ min`; reuses `_apply_reserved_slots.qualifies()`). Ranked desc by rel_weight.

**Cross-modal expansion is NOT a fusion signal.** It was peer-weighted in v1; reviews showed
a text-proximity signal at peer weight floats co-page/derived images above the canonical text
answer. Expansion remains **candidate generation only** (recall): an expansion chunk ranks
*only if it independently earns a place in S_text / S_visual / S_ontology*. (Trade-off: a
co-page figure with no caption match and weak SigLIP no longer auto-surfaces — the honest
outcome when no genuine image-relevance signal exists.)

### Per-image unit (de-duplication of the *picture*)

A single picture can exist as up to three chunks: the `image` chunk (SigLIP vector + caption
`content_text`), and a separate `image_description` TextChunk (VLM prose). These are
**collapsed into one logical result unit keyed by `artifact_id`** for both fusion and display:

- the unit's **S_visual rank** = its `image` chunk's rank;
- the unit's **S_text rank** = the *best* rank among {caption, image_description};
- displayed as **one card** ("matched visually and/or by description").

This removes the double-counting and double-card problems and is where agreement between an
image's visual and textual evidence is rewarded (it appears in both S_visual and S_text).

### Visual separation gate (suppress SigLIP noise)

For codename queries (e.g. "Fan Song") SigLIP has no visual handle; its within-band ordering
is noise. An image is admitted to **S_visual** only if its SigLIP prob ≥
`retrieval_rrf_visual_min_prob` (default **0.30**) — distinct from, and stricter than, the
`retrieval_image_min_score_threshold` (0.0) used for Images-Only display. On a flat/noise
distribution this empties S_visual, so noise contributes nothing to fusion. (Images-Only mode
is unaffected — it does not use fusion.)

### Fusion

```
RRF(u) = Σ over signals S where unit u ∈ S:  w_S / (k + rank_S(u))
```

- `k` = `RETRIEVAL_RRF_K` (default **20**, not 60 — these lists are short (tens of items);
  k=60 flattens rank so "presence" dominates and the cross-encoder's confident top is lost).
- `w_S`: `w_text=1.0`, `w_visual=1.0`, `w_ontology=0.5` (tunable). No `w_expand` (dropped).
- `rank_S(u)`: 1-based rank within S, assigned after sorting **(score desc, then `chunk_id`/
  `artifact_id` asc)** so ties are deterministic.

Sort fused units by `RRF(u)` desc, tiebreak by unit id. Then apply the ontology floor and trim.

### Ontology minimal floor (preserve the live-verified guarantee)

Folding ontology fully into a soft signal would drop the shipped, live-verified reserved-slot
guarantee (CUES→Amazonka/SNR-75). Keep a **minimal hard floor**: guarantee up to
`retrieval_rrf_ontology_min_slots` (default **1**) qualifying ontology unit(s) in the final
top_k if any qualify; RRF orders everything else. (v1's full reserved-slot count is replaced
by this minimal floor + the S_ontology signal.)

### Filtering vs. display (decouple them)

- **Filtering** is done by the **per-signal gates** on their native scales — the existing
  reranker floor (0.05) for S_text, the visual separation gate (0.30) for S_visual — *before*
  fusion. The request `min_confidence` is **not** applied to the fused score in hybrid mode
  (the fused score is not a relevance probability; applying a 0.1–0.5 threshold to it is
  meaningless and breaks portability). `min_confidence` continues to gate **Text Basic** and
  is documented as a per-signal relevance floor for hybrid.
- **Display score** is informational only: `display(u) = RRF(u) / (RRF(u) + C)`,
  `C = RETRIEVAL_RRF_DISPLAY_SCALE` (default **0.05** so a strong single-signal top item reads
  ≈ 0.5 with k=20). Monotonic, stable across queries. **Labeled "fusion score," not a
  probability**, with the understood band (~0.3–0.85), not "0–1".

---

## Integration (rewritten per reviewer 2)

The current pipeline collapses signal identity (`_merge_seed_results` → `_deduplicate_results`
→ `_diversify_results`) **before** any final ordering, and `_text_vector_search` reranks +
trims internally and is **shared with Text Basic**. So RRF is not "just replace the final
sort." Concretely:

1. **Signal capture (pre-merge).**
   - **S_text:** add a `for_fusion: bool=False` parameter to `_text_vector_search`. When true
     (hybrid only) it returns the **wide reranked pool without the final top_k trim** and
     treats the reranker floor as **signal membership, not candidate removal** (sub-floor text
     chunks are simply absent from S_text, not deleted). When false → **today's exact behavior**
     (Text Basic byte-identical). Caption text for image chunks is fetched for reranking
     regardless of the response-level `include_context`.
   - **S_visual:** capture `_image_vector_search` output (already a ranked SigLIP list) before
     merge; apply the separation gate.
   - **S_ontology / candidates:** partition `expanded` by `context["source"]`.
2. **Build units** (collapse images by `artifact_id`), assign per-signal ranks.
3. **Fuse** (RRF), apply ontology floor, then **dedup/diversify on the fused output**, trim top_k.
4. **Rollback:** the entire RRF path is behind `RETRIEVAL_RRF_FUSION_ENABLED` (default true).
   When false, `_multi_modal_pipeline` runs the **untouched** legacy path (reserved-slots +
   single rerank, image-bypass intact). All RRF edits to shared functions are strictly branched
   so the off-path is bit-for-bit the pre-change behavior.

`_apply_reranker`'s `_NON_RERANK_MODALITIES` image-bypass (commit 33063fe) stays for the
legacy/off path; the RRF path handles images via S_visual + caption-in-S_text instead.

---

## Config / env (mirrored into `.env` + `.env.example`)

| Key | Default | Purpose |
|---|---|---|
| `RETRIEVAL_RRF_FUSION_ENABLED` | `true` | master flag; false = legacy ordering |
| `RETRIEVAL_RRF_K` | `20` | RRF constant (tuned for short lists) |
| `RETRIEVAL_RRF_W_TEXT` | `1.0` | S_text weight |
| `RETRIEVAL_RRF_W_VISUAL` | `1.0` | S_visual weight |
| `RETRIEVAL_RRF_W_ONTOLOGY` | `0.5` | S_ontology weight |
| `RETRIEVAL_RRF_VISUAL_MIN_PROB` | `0.30` | SigLIP admit threshold for S_visual |
| `RETRIEVAL_RRF_ONTOLOGY_MIN_SLOTS` | `1` | minimal ontology hard floor |
| `RETRIEVAL_RRF_DISPLAY_SCALE` | `0.05` | display transform constant C |

## Edge cases

- **Single-modality hybrid** (`modality_filter=text`/`image`): the filter runs **before**
  signal building, so S_visual (or S_text) is simply empty; RRF over the remaining signal =
  identity ranking. Harmless.
- **Empty signals:** skipped in the sum.
- **No qualifying ontology unit:** floor does nothing.
- **Determinism:** RRF + stable within-signal sort + `chunk_id` tiebreak gives **deterministic
  ordering for a fixed candidate set**. Exact-kNN (commit ec363e6) stabilizes vector retrieval;
  full byte-identical run-to-run for hybrid additionally depends on the expansion-gather set
  (separate TODO), so it is **not** claimed as a gate here. Text Basic byte-identical **is** a gate.

## Verification

- `"radar antenna"` / All → strong images (SigLIP ≥0.3) rank near strong text; agreement items
  (good SigLIP + good caption) top the list.
- `"Fan Song"` / All → text leads; S_visual largely empty (codename → no visual separation), so
  no spurious images; any image present has a real caption match. Deterministic across runs.
- Known ontology case (CUES→Amazonka/SNR-75) still appears in top_k (floor + S_ontology).
- Text Basic `"Fan Song"` → **byte-identical** to pre-change.
- `RETRIEVAL_RRF_FUSION_ENABLED=false` → identical to current behavior (hard gate).

## Testing

- **Unit:** RRF math (known ranks → known fused order); within-signal stable-rank under ties;
  per-image unit collapse; display transform monotonicity; visual separation gate.
- **Integration:** the verification queries via `POST /v1/retrieval/query`, plus a
  flag-off equality test and a Text-Basic equality test.

---

## Appendix — resolution of the 3 reviews

| Finding (reviewer) | Resolution |
|---|---|
| SUM ≠ "best-of" (R1, R3) | Goal corrected to agreement/RRF (industry standard); "best-of" rejected. |
| k=60 flattens cross-encoder (R1, R3) | k default 20; documented rationale for short lists. |
| Per-signal lists don't exist; upstream collapses them (R2) | Integration rewritten: capture signals pre-merge; dedup/diversify after fusion. |
| `_text_vector_search` shared w/ Text Basic (R2) | `for_fusion` param; off-path byte-identical; Text-Basic equality test. |
| Images-in-S_text reverses 33063fe + `include_context` dep (R2) | Per-image unit; caption fetched regardless of `include_context`; bypass kept for legacy path. |
| Display band compressed / `min_confidence` breaks (R1, R2, R3) | Filtering via per-signal gates pre-fusion; `min_confidence` not applied to fused score in hybrid; C retuned to 0.05; band documented. |
| S_expand = non-relevance peer signal (R1, R3) | Expansion dropped as a fusion signal; candidate-generation only. |
| Same image counted up to 4× / two cards (R1, R2, R3) | Per-image unit collapse by `artifact_id`. |
| SigLIP noise floods codename queries (R3) | Visual separation gate `retrieval_rrf_visual_min_prob`=0.30. |
| Ontology guarantee regression (R3) | Minimal hard floor `retrieval_rrf_ontology_min_slots`=1 + S_ontology signal. |
| Within-signal tie determinism (R1, R2) | Stable within-signal sort (score desc, id asc) before rank assignment. |
| Run-to-run determinism over-claimed (R2) | Down-scoped to "deterministic ordering for a fixed set"; not a gate for hybrid. |
| Rollback not clean if shared fns edited (R2) | All RRF edits strictly branched behind the flag; off-path untouched + equality test. |
