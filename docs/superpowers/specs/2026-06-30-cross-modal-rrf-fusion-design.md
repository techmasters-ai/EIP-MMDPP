# Cross-Modal RRF Fusion for Multi-Modal Retrieval — Design (v3.1)

- **Date:** 2026-06-30
- **Status:** Revised after **3 rounds × 3 independent reviews** → ready for implementation plan
  (final round: 2 GO-conditional + 1 narrow NO-GO, all patched below)
- **Open action (carry into plan, do NOT ship without):** measure the live
  CUES→Amazonka/SNR-75 qualifying-ontology count and set `RETRIEVAL_RRF_ONTOLOGY_MIN_SLOTS` to it.
- **Scope:** hybrid retrieval ordering in `app/api/v1/retrieval.py` (+ `_apply_reranker`, `unified_query`)

## Problem

Multi-modal (hybrid) retrieval merges results from **incompatible scorers** onto one
0–1 axis and sorts them together: cross-encoder text relevance (`bge-reranker-v2-m3`,
~0.1–0.99), SigLIP visual match probability (~0–0.93), and cross-modal expansion decay
(text-proximity, *not* a relevance measure). A SigLIP `0.5` is not comparable to a
cross-encoder `0.9`, so the merged ordering is meaningless. Calibrating to a common
relevance probability needs labeled cross-modal data we do not have.

## Goal

One merged, well-ordered hybrid list via **Reciprocal Rank Fusion (RRF)** — the
industry-standard, calibration-free, **agreement-based** (CombSUM-family) rank-aggregation
method (Cormack et al. 2009; default hybrid-search fusion in Elasticsearch, OpenSearch,
Qdrant, Weaviate, Vespa, Milvus). "Best-of"/MAX is explicitly rejected (Fox & Shaw 1994:
CombSUM/MNZ > CombMAX).

## Non-goals

- Calibrated relevance probabilities. Any change to **Text Basic** ranking (must stay
  byte-identical). A VLM cross-modal reranker. Faceted UI.

---

## Design

### Fusion signals (relevance only)

Each is an independently-ranked list, built **before** the candidate merge/dedup:

1. **S_text** — cross-encoder relevance over text-bearing units (`text`, `table`,
   `image_description`, and image **captions**, see caption pass + per-image unit).
2. **S_visual** — SigLIP probability over `image`/`schematic` units, admitted only when the
   visual signal is real (separation gate).
3. **S_ontology** — qualifying `ontology_relation` chunks (`qualifies()`: `rel_weight ≥ min`,
   `raw_cosine ≥ min`).

**Per-signal dedup:** within each signal, collapse duplicate `chunk_id`s (the expansion fan-out
can reach the same chunk from multiple seeds) **before** assigning ranks.

**Within-signal rank assignment:** sort `(score desc, id asc)`, then assign **contiguous**
1-based ranks (no holes).

### Per-image unit (collapse the *picture*)

A picture exists as up to three chunks: the `image` chunk (SigLIP vector + caption
`content_text`) and a separate `image_description` TextChunk. **Collapse into one unit keyed
by `artifact_id`, gated to `modality ∈ {image, image_description, schematic}` with non-null
`artifact_id`** (plain `text`/`table` chunks also carry an `artifact_id` and must NOT be
merged):

- unit **visual_score** = its image chunk's SigLIP prob → its S_visual rank;
- unit **text_score** = `max(caption_score, image_description_score)` (same-signal dedup, a
  MAX *within* S_text, not a fusion-level MAX) → its S_text rank;
- **Ranks are assigned over units by score, contiguously** (collapse-by-score first, then
  rank — resolves the v2 wording ambiguity; no rank holes).
- Displayed as **one card** retaining **both source `chunk_id`s and page numbers** (project
  lineage rule — the merged card must list image + description provenance, not drop one).

### Caption → S_text pass (explicit producer)

Captions live on `image` chunks returned by `_image_vector_search`, **not** on any TextChunk
from `_text_vector_search`. So S_text is built from **two** producers: (a) the text/table/
image_description TextChunks from the text vector search, and (b) a dedicated cross-encoder
pass over the **captions of the S_visual image chunks**. Both feed one combined S_text ranking
(after per-image-unit collapse). Caption text is read from the image chunk's `chunk_text`
**regardless of `include_context`**.

### Visual separation gate (suppress SigLIP noise)

Admit an image to S_visual only if SigLIP prob ≥ `retrieval_rrf_visual_min_prob` (default
**0.35**). Measured separation ("Fan Song" one image at 0.51 vs rest ~0; "radar antenna" ~18
images ≥0.5) leaves headroom, so 0.35 keeps genuine hits while excluding the noise floor.
Distinct from `retrieval_image_min_score_threshold` (0.0, Images-Only display). On a flat
distribution S_visual is effectively empty.

### Expansion floor (bounded, non-leading — restores codename diagrams)

Dropping expansion entirely (v2) removed the **only** mechanism that surfaces an on-page
schematic for a **codename** query — where SigLIP can't read the codename, the caption is
often "Figure 3", and the VLM description names shape/function not the codename, so all three
relevance signals fail together. Re-admit expansion as a **bounded, non-leading** path, NOT a
peer RRF signal:

- **Fill-if-spare, never evict (per round-3 R1+R3, blocking):** the floor adds up to
  `retrieval_rrf_expansion_floor_slots` (default **2**) **additional** slots — i.e. the result
  may return up to `top_k + floor_slots` — OR fills only the unused tail when fewer than `top_k`
  genuine units were fused. It must **never displace a genuinely-scored fused item** (a dense
  query with ≥`top_k` real results gets zero floored items, not two evicted answers).
- Candidates = best `cross_modal`/`doc_structure` expansion units **not already present**,
  ordered by decay score, distinct decay-ordered display values.
- **Hard constraint:** an expansion-floor unit's display score is **capped strictly below the
  lowest fused (non-floor) item** (may read **< 0.30** — a documented sub-band), so a co-page
  item can *appear* but provably **cannot outrank** a genuinely-scored result.
- **Determinism caveat:** floored diagrams ride the expansion-gather path, which is run-to-run
  unstable (`project_retrieval_nondeterminism`), so a floored diagram may appear in one run and
  not the next until that gather is stabilized. Surfacing it with a "co-page (proximity)" UI
  label is recommended (it carries no relevance signal — it could be a co-page figure of a
  *different* system).
- **Injection happens AFTER display-level filtering** so the deliberately-low-capped item is
  not immediately filtered back out.

### Fusion

```
RRF(u) = Σ over signals S where unit u ∈ S:  w_S / (k + rank_S(u))
```

- `k` = `RETRIEVAL_RRF_K` (default **20**; verified separation: rank-1 = 0.0476, rank-10 =
  0.0333, rank-30 = 0.0200 — 2.38× r1/r30, vs 1.48× at k=60).
- `w_S`: `w_text=1.0`, `w_visual=1.0`, `w_ontology=0.5` (tunable).
- Sort units by a **total-order key on EXACT RRF equality** (a tolerance/"near-equal" compare
  is non-transitive → breaks determinism, per round-3 R1): `sort by (-RRF, -num_signals,
  -text_bearing, id_asc)`. Exact ties are common (two single-signal rank-1 units both = w/(k+1)),
  so exact-equality handling suffices — a lone single-signal image ties, then loses to a
  text-bearing unit, so it cannot edge the primary text answer at slot #1.

### Ontology minimal floor (preserve the live-verified guarantee)

Guarantee up to `retrieval_rrf_ontology_min_slots` qualifying ontology units **in top_k**.
**Action item for the plan:** the live CUES→Amazonka/SNR-75 case must be re-run to confirm how
many qualifying ontology chunks it produced; set the default to that count (≥1) so the
guarantee is not silently weaker than v1's reserved slots. The floor guarantees **membership**;
a floored unit keeps its RRF-ordered position (annotate that a tail-floored unit may display
below the item above it — accepted minor non-monotonicity).

### Filtering vs. display

- **Filtering** is per-signal, on native scales, **before** fusion: reranker floor (0.05) for
  S_text, visual gate (0.35) for S_visual.
- **`min_confidence` in hybrid** is **mapped onto the per-signal floors** (not applied to the
  fused score, which is not a probability): the effective S_text floor = `max(0.05,
  min_confidence)` on reranker score, and the effective S_visual gate = `max(0.35,
  min_confidence)` on SigLIP prob. This keeps the API knob meaningful instead of inert. Text
  Basic keeps applying `min_confidence` to its score unchanged.
- **Display score** is informational: `display(u) = RRF(u) / (RRF(u) + C)`,
  `C = RETRIEVAL_RRF_DISPLAY_SCALE` (default **0.05**). Strong single-signal top reads ≈ **0.49**,
  agreement top ≈ **0.66**. Signals are modality-partitioned, so **no unit can be in all three**
  → realistic ceiling is the 2-signal agreement (image+caption) ≈ **0.66** (round-3 R1).
  **Documented band ~0.3–0.66** for fused items (expansion-floor items form a sub-band < 0.30); NOT 0–1.
  Labeled "fusion score, not a probability." Note: not comparable across queries (a
  single-signal-correct answer reads ~0.49 while an agreement answer reads ~0.66) — UI should
  not threshold on it.

---

## Integration (per reviewer-2, round 2)

The pipeline currently collapses signal identity (`_merge_seed_results :352` →
`_deduplicate_results :365` → `_diversify_results :366`) **before** ordering; `_text_vector_search`
reranks+trims via `_apply_reranker` (the `top_k` trim lives at `_apply_reranker :299`, **not** in
`_text_vector_search`); and `min_confidence` is applied strategy-agnostically in
`unified_query :102-103`. So the RRF path touches **three** shared surfaces, all flag-branched:

1. **`_text_vector_search(for_fusion=False)`** — default = today's exact behavior (Text Basic
   byte-identical). `for_fusion=True` returns the **wide reranked pool without the final trim**.
2. **`_apply_reranker`** — `for_fusion` branch must widen **BOTH** trims (round-3 R2 blocker):
   the `top_k` argument passed into `cross_encoder_rerank` at `:264` (which returns
   `result[:top_k]` in `reranker.py:80` — this is what actually caps the pool, *before* the
   `:299` slice) **and** the `:299` `output[:top_k]`. Widen the `:264` arg to
   `len(rerankable)` / `retrieval_rerank_pool_size`, else S_text silently stays ~`top_k`, not
   ~128, and the whole seed/pool design no-ops. Also skip the `passthrough`/`remainder`
   re-assembly in this branch.
3. **`unified_query`** — branches `:102-103` so hybrid (when fusion enabled) maps
   `min_confidence` onto per-signal floors instead of cutting the fused score. (Note: one 0–1
   knob maps to two incommensurable scales — reranker ~0.1–0.99 vs SigLIP 0–0.93 — with
   different activation points; document this asymmetry on the API surface.)
4. **`_image_vector_search`** (a 4th surface, round-3 R2) — caption `content_text` is **nulled
   when `include_context=False`** (`:688`); the caption pass needs it regardless, so read the
   image chunk's caption independent of `include_context`. **The caption (Docling `chunk_text`,
   often empty/"Figure 3") is NOT the VLM `image_description`** (a separate TextChunk); both can
   feed S_text for the same picture, so the **per-image MAX collapse must run BEFORE S_text rank
   assignment** to prevent double-counting one picture across two S_text entries.

**Merged-card lineage carrier (round-3 R2):** `QueryResultItem` has a **singular** `chunk_id`/
`page_number`. The merged image card carries the image chunk as primary `chunk_id`, puts the
second source (`image_description` chunk_id + source label) in `context` keys captured **at
collapse time** (downstream backfills key off the single `chunk_id` and won't populate it), and
lists both pages in `page_numbers`. No schema change required.

**Seed fan-out fix (HIGH):** the wide S_text pool (≈128) must **not** become the expansion seed
set — `_expand_seeds` is sized for ~`top_k` seeds (`_EXPAND_CONCURRENCY=8`, each seed opens DB
sessions). Take the **top ~`top_k`** of the wide pool as expansion seeds; keep the full pool only
for S_text. No second search (the reranker already scores ≤`retrieval_rerank_pool_size` pairs, so
this does not increase cross-encoder cost).

**Flow:** capture S_visual (pre-merge, gated) and the wide text pool → caption pass over S_visual
images → build per-image units → per-signal dedup + contiguous ranks → RRF → ontology floor →
expansion floor → dedup/diversify on the fused output → trim `top_k`.

**Rollback:** `RETRIEVAL_RRF_FUSION_ENABLED` (default true). When false, all three functions run
their untouched legacy branch (reserved-slots + single rerank + image-bypass `_NON_RERANK_MODALITIES`
`:248-250`). The flag-off equality test must cover **all three** functions.

---

## Config / env (mirrored into `.env` + `.env.example`)

| Key | Default | Purpose |
|---|---|---|
| `RETRIEVAL_RRF_FUSION_ENABLED` | `true` | master flag; false = legacy ordering |
| `RETRIEVAL_RRF_K` | `20` | RRF constant (tuned for short lists) |
| `RETRIEVAL_RRF_W_TEXT` / `_W_VISUAL` / `_W_ONTOLOGY` | `1.0` / `1.0` / `0.5` | signal weights |
| `RETRIEVAL_RRF_VISUAL_MIN_PROB` | `0.35` | SigLIP admit threshold for S_visual |
| `RETRIEVAL_RRF_ONTOLOGY_MIN_SLOTS` | `1` (verify vs live case) | ontology hard floor |
| `RETRIEVAL_RRF_EXPANSION_FLOOR_SLOTS` | `2` | bounded non-leading co-page slots |
| `RETRIEVAL_RRF_DISPLAY_SCALE` | `0.05` | display transform constant C |

## Edge cases

- **Single-modality hybrid** (`modality_filter`): filter runs before signal building → the other
  signal is empty; RRF over one signal = identity ranking.
- **Empty signals / no qualifying ontology:** skipped / floor no-ops.
- **Determinism:** RRF + per-signal dedup + contiguous stable ranks + id tiebreak →
  **deterministic ordering for a fixed candidate set** (exact-kNN `ec363e6` stabilizes vectors;
  full run-to-run hybrid determinism also needs expansion-gather stability — separate TODO, not a
  gate here). **Text Basic byte-identical IS a gate.**

## Verification

- `"radar antenna"` / All → strong images (≥0.35) near strong text; agreement units (good SigLIP
  + good caption) lead; weak-but-real images mid-list.
- `"Fan Song"` / All → text leads (RRF over the dominant signal ≈ identity); S_visual admits only
  the one separated image (0.51) if genuine; the **on-page schematic appears via the expansion
  floor at the tail, never above text**; deterministic across runs.
- CUES→Amazonka/SNR-75 still in top_k (ontology floor sized to the live case).
- Text Basic `"Fan Song"` → **byte-identical** to pre-change.
- `RETRIEVAL_RRF_FUSION_ENABLED=false` → identical to current behavior across all three functions.

## Testing

- **Unit:** RRF math (known ranks → fused order); contiguous-rank assignment under ties and
  unit-collapse; per-image collapse modality gate; caption pass; visual gate; expansion-floor cap
  (floored item < lowest fused item); `min_confidence`→floor mapping.
- **Integration:** the verification queries; flag-off equality (3 functions); Text-Basic equality;
  lineage present on merged image cards.

---

## Appendix — review resolutions (rounds 1 & 2)

| Finding | Resolution |
|---|---|
| SUM ≠ best-of (R1) | Agreement/RRF adopted as standard; best-of rejected. |
| k=60 flattens (R1) | k=20 (numbers verified: 2.38× r1/r30). |
| Per-signal lists collapsed upstream (R2) | Capture pre-merge; dedup/diversify after fusion. |
| `_text_vector_search` shared (R2) | `for_fusion` param; off-path byte-identical. |
| **Trim is in `_apply_reranker`, not `_text_vector_search` (R2-r2)** | Flag branch spans `_apply_reranker` too. |
| **`min_confidence` in `unified_query` (R2-r2)** | Branch `:102-103`; map onto per-signal floors. |
| **Seed fan-out: wide pool as 128 seeds (R2-r2, HIGH)** | Expansion seeds = top ~`top_k`; wide pool only for S_text. |
| **Caption→S_text has no producer (R2-r2, HIGH)** | Explicit caption rerank pass over S_visual images. |
| **Per-image collapse needs modality guard + lineage (R2-r2)** | Gate to image modalities w/ non-null artifact_id; retain both chunk_ids/pages. |
| **Per-signal dedup missing (R2-r2)** | Dedup each signal before rank assignment. |
| Display band / min_confidence coupling (R1, R2, R3) | Filter pre-fusion; map min_confidence to floors; C=0.05; band ~0.3–0.70. |
| **"collapse by max rank" vs "max score" (R1-r2)** | Collapse by max **score**, then contiguous ranks. |
| **Visual gate binary cliff / lone-0.51 leads (R1-r2, R3-r2)** | Gate 0.30→0.35; leading-slot tiebreak prefers agreement/text. |
| **Codename diagrams disappear (R3-r2, blocking)** | Bounded non-leading expansion floor (display-capped). |
| **Ontology floor=1 may be < live case (R3-r2)** | Size floor to verified live cardinality; guarantees membership. |
| S_expand non-relevance peer (R1, R3) | Not a fusion signal; only the bounded floor. |
| SigLIP noise floods codename (R3) | Visual separation gate. |
| Determinism over-claimed (R2) | Down-scoped to fixed-set ordering. |
| "S_visual largely empty" prose wrong (R3-r2) | Corrected: noise floor excluded, separated images admitted. |
| **Round 3 — tiebreak "near-equal" non-transitive (R1)** | Exact-equality total-order key `(-RRF,-num_signals,-text_bearing,id)`. |
| **Round 3 — expansion floor evicts genuine results (R1+R3, blocking)** | Fill-if-spare / additive `top_k+floor_slots`; never evict; floor-aware trim. |
| **Round 3 — internal reranker trim `:264` un-handled (R2 blocker)** | `for_fusion` widens the `cross_encoder_rerank` top_k arg, not just `:299`. |
| **Round 3 — caption producer = 4th surface; caption ≠ image_description (R2)** | Read caption regardless of include_context; per-image MAX collapse BEFORE S_text rank. |
| **Round 3 — merged-card lineage carrier unspecified (R2)** | Primary=image chunk_id; secondary in `context` at collapse time; both pages in `page_numbers`. |
| **Round 3 — structural max display 0.70 unreachable (R1)** | Corrected to ≈0.66 (no unit in all 3 partitioned signals). |
| **Round 3 — ontology floor=1 still unverified (R1+R3, OPEN)** | Carried as a hard pre-ship action item; measure live CUES count. |
| Round 3 — min_confidence two-scale asymmetry (R3) | Documented on API surface. |
| Round 3 — determinism of restored diagram (R3) | Noted: rides expansion-gather (separate nondeterminism TODO). |
