# Cross-Modal RRF Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge cross-encoder text, SigLIP visual, and ontology signals into one hybrid result list with Reciprocal Rank Fusion (RRF), behind a default-on flag, without changing Text Basic.

**Architecture:** A pure, unit-tested fusion module (`app/services/rrf_fusion.py`) computes per-signal ranks, RRF scores, the display transform, per-image unit collapse, the leading-slot tiebreak, and the fill-if-spare expansion floor. `_multi_modal_pipeline` captures the per-signal ranked lists *before* the existing merge/dedup, calls the module, then dedups/trims. All edits to shared functions (`_text_vector_search`, `_apply_reranker`, `_image_vector_search`, `unified_query`) are branched behind `for_fusion` / `RETRIEVAL_RRF_FUSION_ENABLED` so the off-path is byte-identical to today and Text Basic is untouched.

**Tech Stack:** Python 3.11, FastAPI, pydantic-settings, pytest, NumPy, ArcadeDB, `bge-reranker-v2-m3` (cross-encoder, GPU), SigLIP2 (OpenCLIP).

**User decisions (already made):**
- "Lets do RRF if it is industry standard" — agreement-based RRF, not best-of/MAX.
- "fold in reserved ontology slots" — ontology is a soft signal + a minimal hard floor (not the old reserved-slots count).
- "Give me a fixed, absolute value for the display score" — `display = RRF/(RRF+C)`.
- Spec authority: `docs/superpowers/specs/2026-06-30-cross-modal-rrf-fusion-design.md` (v3.1), vetted by 3 rounds × 3 independent reviewers.

---

## Reference: key current code anchors (read before editing)

- `app/api/v1/retrieval.py:83-84` — `strategy==basic` dispatch (Text Basic entry; must stay untouched).
- `app/api/v1/retrieval.py:102-103` — `unified_query` applies `min_confidence` to `r.score` strategy-agnostically.
- `app/api/v1/retrieval.py:_apply_reranker` — calls `cross_encoder_rerank(query, rerank_input, top_k=body.top_k)`; `app/services/reranker.py:80` returns `result[:top_k]` (the **real** pool cap), then `_apply_reranker` returns `output[:body.top_k]` (the second cap). BOTH must widen under `for_fusion`.
- `app/api/v1/retrieval.py:_text_vector_search` — builds `results[:retrieval_rerank_pool_size]` then `_apply_reranker(results, body)`; shared with Text Basic.
- `app/api/v1/retrieval.py:_image_vector_search` — sets `content_text = props.get("text") or props.get("chunk_text") if body.include_context else None`; SigLIP prob via `clip_cosine_to_prob`.
- `app/api/v1/retrieval.py:_multi_modal_pipeline` — `_text_vector_search` + `_image_vector_search` → `_merge_seed_results` → `_expand_seeds` → `_rescore_expanded_chunks` → `_deduplicate_results` → `_diversify_results` → modality filter → `_apply_reserved_slots` → `_apply_reranker`.
- `app/api/v1/retrieval.py:_apply_reserved_slots.qualifies()` — the ontology `rel_weight ≥ min, raw_cosine ≥ min` predicate; `context["source"]=="ontology_relation"`.
- `app/schemas/retrieval.py` — `QueryResultItem` has singular `chunk_id`/`page_number`, plus `page_numbers: list`, `context: dict`, `evidence_ids`, `self_refs`.
- Expansion sources stamped in `context["source"]`: `doc_structure`, `cross_modal`, `ontology`, `ontology_relation`.

---

## Task 0: Config knobs + env mirroring

**Goal:** Add the 8 `retrieval_rrf_*` settings with defaults, mirrored into `.env` and `.env.example`.

**Files:**
- Modify: `app/config.py` (Settings class, after `retrieval_reranker_score_floor`)
- Modify: `.env`, `.env.example` (near `RETRIEVAL_RERANK_BLEND_ALPHA`)
- Test: `tests/test_config_rrf.py` (create)

**Acceptance Criteria:**
- [ ] `get_settings()` exposes `retrieval_rrf_fusion_enabled=True`, `retrieval_rrf_k=20`, `retrieval_rrf_w_text=1.0`, `retrieval_rrf_w_visual=1.0`, `retrieval_rrf_w_ontology=0.5`, `retrieval_rrf_visual_min_prob=0.35`, `retrieval_rrf_ontology_min_slots=1`, `retrieval_rrf_expansion_floor_slots=2`, `retrieval_rrf_display_scale=0.05`.
- [ ] Both `.env` and `.env.example` contain all 8 vars with comments (project rule: every new env var in both files).

**Verify:** `python -c "from app.config import get_settings as g; s=g(); print(s.retrieval_rrf_k, s.retrieval_rrf_visual_min_prob)"` → `20 0.35`

**Steps:**

- [ ] **Step 1: Write failing test** — `tests/test_config_rrf.py`:
```python
from app.config import get_settings

def test_rrf_defaults():
    s = get_settings()
    assert s.retrieval_rrf_fusion_enabled is True
    assert s.retrieval_rrf_k == 20
    assert s.retrieval_rrf_w_text == 1.0
    assert s.retrieval_rrf_w_visual == 1.0
    assert s.retrieval_rrf_w_ontology == 0.5
    assert s.retrieval_rrf_visual_min_prob == 0.35
    assert s.retrieval_rrf_ontology_min_slots == 1
    assert s.retrieval_rrf_expansion_floor_slots == 2
    assert s.retrieval_rrf_display_scale == 0.05
```
- [ ] **Step 2: Run** `pytest tests/test_config_rrf.py -v` → FAIL (AttributeError).
- [ ] **Step 3: Add to `app/config.py`** after the `retrieval_reranker_score_floor` field:
```python
    # Cross-modal RRF fusion (spec 2026-06-30-cross-modal-rrf-fusion v3.1)
    retrieval_rrf_fusion_enabled: bool = True
    retrieval_rrf_k: int = 20            # RRF constant; tuned for short lists
    retrieval_rrf_w_text: float = 1.0
    retrieval_rrf_w_visual: float = 1.0
    retrieval_rrf_w_ontology: float = 0.5
    retrieval_rrf_visual_min_prob: float = 0.35   # SigLIP admit floor for S_visual
    retrieval_rrf_ontology_min_slots: int = 1     # minimal ontology hard floor (verify vs live CUES — Task 1)
    retrieval_rrf_expansion_floor_slots: int = 2  # bounded, non-leading co-page slots
    retrieval_rrf_display_scale: float = 0.05     # display = RRF/(RRF+C)
```
- [ ] **Step 4: Add to `.env` and `.env.example`** (both), near `RETRIEVAL_RERANK_BLEND_ALPHA`:
```bash
# Cross-modal RRF fusion (hybrid result ordering). Master flag false = legacy ordering.
RETRIEVAL_RRF_FUSION_ENABLED=true
RETRIEVAL_RRF_K=20
RETRIEVAL_RRF_W_TEXT=1.0
RETRIEVAL_RRF_W_VISUAL=1.0
RETRIEVAL_RRF_W_ONTOLOGY=0.5
RETRIEVAL_RRF_VISUAL_MIN_PROB=0.35
RETRIEVAL_RRF_ONTOLOGY_MIN_SLOTS=1
RETRIEVAL_RRF_EXPANSION_FLOOR_SLOTS=2
RETRIEVAL_RRF_DISPLAY_SCALE=0.05
```
- [ ] **Step 5: Run** `pytest tests/test_config_rrf.py -v` → PASS.
- [ ] **Step 6: Commit** `git add app/config.py .env.example tests/test_config_rrf.py && git commit -m "feat(retrieval): RRF fusion config knobs"` (note: `.env` is gitignored — stage only `.env.example`).

---

## Task 1: Measure live ontology cardinality (PRE-SHIP GATE)

**USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user (spec: "do NOT ship without"). It MUST NOT be closed by walking around it or substituting a cheaper check. Close only after the qualifying-ontology count for the live case has been measured with captured output.

**Goal:** Measure how many qualifying ontology chunks the live CUES→Amazonka/SNR-75 case produces, and set `RETRIEVAL_RRF_ONTOLOGY_MIN_SLOTS` to that count so the floor is not silently weaker than v1's reserved slots.

**Files:**
- Modify: `.env`, `.env.example` (`RETRIEVAL_RRF_ONTOLOGY_MIN_SLOTS`), `app/config.py` default if needed.

**Acceptance Criteria:**
- [ ] The CUES-style hybrid query is run with `RETRIEVAL_DOMAIN_EXPANSION_ENABLED=true`; the number of results with `context["source"]=="ontology_relation"` that pass `qualifies()` is captured.
- [ ] `RETRIEVAL_RRF_ONTOLOGY_MIN_SLOTS` is set to `max(1, that count)` in both env files (and the config default updated to match).

**Verify:** Run the probe below; output shows the qualifying count; env value matches it.

**Steps:**

- [ ] **Step 1: Probe in-container** (the API container has the graph + code):
```bash
docker exec eip-mmdpp-api-1 python -c "
import asyncio, json
from app.schemas.retrieval import UnifiedQueryRequest, QueryStrategy, ModalityFilter
from app.api.v1 import retrieval as R
from app.db.session import AsyncSessionFactory
from app.config import get_settings
from app.api.v1._retrieval_helpers import get_retrieval_relation_weights
async def main():
    s=get_settings()
    async with AsyncSessionFactory() as db:
        body=UnifiedQueryRequest(query_text='CUES', strategy=QueryStrategy.hybrid, modality_filter=ModalityFilter.all, top_k=20, min_confidence=0.0, include_context=True, ontology_reserved_slots=3)
        txt=await R._text_vector_search(db, body); img=await R._image_vector_search(db, body)
        seeds=R._merge_seed_results([txt,img])
        exp=await R._expand_seeds(db, seeds, True, body.query_text)
        exp=await R._rescore_expanded_chunks(exp, body.query_text)
        rw=get_retrieval_relation_weights()
        def qualifies(it):
            ctx=it.context or {}
            if ctx.get('source')!='ontology_relation': return False
            w=rw.get(str(ctx.get('rel_type')), rw.get('default',0.70))
            return w>=s.retrieval_ontology_reserve_min_rel_weight and float(ctx.get('raw_cosine',0.0))>=s.retrieval_ontology_reserve_min_cosine
        n=sum(1 for it in (seeds+exp) if qualifies(it))
        print('QUALIFYING_ONTOLOGY_UNITS', n)
asyncio.run(main())
" 2>&1 | grep QUALIFYING
```
- [ ] **Step 2:** Set `RETRIEVAL_RRF_ONTOLOGY_MIN_SLOTS` (both env files) and the `app/config.py` default to `max(1, n)` from Step 1. If the real production CUES query differs from `'CUES'`, use the actual query the live case used (check `project_retrieval_nondeterminism` / the ontology spec for the verified example).
- [ ] **Step 3: Commit** `git add app/config.py .env.example && git commit -m "chore(retrieval): set RRF ontology floor to live CUES cardinality"`.

---

## Task 2: RRF fusion core module (pure functions) + unit tests

**Goal:** A dependency-free module computing ranks, RRF, display, tiebreak, and the expansion floor — fully unit-tested.

**Files:**
- Create: `app/services/rrf_fusion.py`
- Test: `tests/services/test_rrf_fusion.py`

**Acceptance Criteria:**
- [ ] `assign_ranks` produces contiguous 1-based ranks, stable under ties (score desc, id asc).
- [ ] `rrf_score` sums `w_S/(k+rank)` across only the signals a unit appears in.
- [ ] `display_score(rrf, c)` is monotonic; `display_score(1/21, 0.05)≈0.488`.
- [ ] `fuse` orders by total-order key `(-rrf, -num_signals, -text_bearing, id)`; a lone single-signal image loses slot #1 to a text unit at equal RRF.
- [ ] `apply_expansion_floor` never evicts a fused item; floored items' display capped strictly below the lowest fused display.

**Verify:** `pytest tests/services/test_rrf_fusion.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write `tests/services/test_rrf_fusion.py`:**
```python
from app.services.rrf_fusion import (
    assign_ranks, rrf_score, display_score, fuse, apply_expansion_floor, FusedUnit,
)

def test_assign_ranks_contiguous_stable():
    items = [("a", 0.9), ("b", 0.5), ("c", 0.9)]  # a,c tie
    ranks = assign_ranks(items)  # -> {id: rank}
    assert ranks == {"a": 1, "c": 2, "b": 3}  # tie broken by id asc, contiguous

def test_rrf_score_sums_present_signals():
    # unit in text rank1 and visual rank1, weights 1.0/1.0, k=20
    r = rrf_score({"text": 1, "visual": 1}, {"text": 1.0, "visual": 1.0}, k=20)
    assert abs(r - (1/21 + 1/21)) < 1e-9

def test_display_monotonic_and_anchor():
    assert abs(display_score(1/21, 0.05) - 0.4878) < 1e-3
    assert display_score(0.10, 0.05) > display_score(0.05, 0.05)

def test_fuse_text_wins_tie_over_lone_image():
    units = [
        FusedUnit(id="img", signals={"visual": 1}, text_bearing=False),
        FusedUnit(id="txt", signals={"text": 1}, text_bearing=True),
    ]
    out = fuse(units, {"text": 1.0, "visual": 1.0, "ontology": 0.5}, k=20)
    assert out[0].id == "txt" and out[1].id == "img"  # equal RRF, text leads

def test_expansion_floor_never_evicts_and_caps_below():
    fused = [FusedUnit(id="t1", signals={"text": 1}, text_bearing=True)]
    for u in fuse(fused, {"text":1.0,"visual":1.0,"ontology":0.5}, k=20):
        pass
    floored = apply_expansion_floor(
        fused_units=fuse(fused, {"text":1.0,"visual":1.0,"ontology":0.5}, k=20),
        expansion_candidates=[("e1", 0.40)],
        top_k=20, floor_slots=2, display_scale=0.05,
    )
    # t1 still present; e1 appended; e1 display < t1 display
    ids = [u.id for u in floored]
    assert "t1" in ids and "e1" in ids
    t1 = next(u for u in floored if u.id == "t1")
    e1 = next(u for u in floored if u.id == "e1")
    assert e1.display < t1.display

def test_expansion_floor_no_evict_on_full_topk():
    fused = fuse([FusedUnit(id=f"t{i}", signals={"text": i+1}, text_bearing=True) for i in range(20)],
                 {"text":1.0,"visual":1.0,"ontology":0.5}, k=20)
    out = apply_expansion_floor(fused, [("e1", 0.4)], top_k=20, floor_slots=2, display_scale=0.05)
    # additive: 20 fused + up to 2 floor = 22; no fused item dropped
    assert len([u for u in out if u.id.startswith("t")]) == 20
```
- [ ] **Step 2: Run** `pytest tests/services/test_rrf_fusion.py -v` → FAIL (module missing).
- [ ] **Step 3: Write `app/services/rrf_fusion.py`:**
```python
"""Reciprocal Rank Fusion for cross-modal hybrid retrieval.

Pure functions only (no I/O). See docs/superpowers/specs/2026-06-30-cross-modal-rrf-fusion-design.md.
"""
from __future__ import annotations
from dataclasses import dataclass, field


def assign_ranks(items: list[tuple[str, float]]) -> dict[str, int]:
    """Contiguous 1-based ranks, sorted (score desc, id asc) for determinism."""
    ordered = sorted(items, key=lambda t: (-t[1], t[0]))
    return {id_: i + 1 for i, (id_, _score) in enumerate(ordered)}


def rrf_score(signal_ranks: dict[str, int], weights: dict[str, float], k: int) -> float:
    """Sum of w_S/(k+rank) over the signals the unit appears in."""
    return sum(weights.get(s, 0.0) / (k + rank) for s, rank in signal_ranks.items())


def display_score(rrf: float, c: float) -> float:
    """Monotonic display transform RRF/(RRF+C), bounded (0,1)."""
    return rrf / (rrf + c) if (rrf + c) > 0 else 0.0


@dataclass
class FusedUnit:
    id: str
    signals: dict[str, int] = field(default_factory=dict)  # signal name -> rank
    text_bearing: bool = False
    rrf: float = 0.0
    display: float = 0.0
    payload: object = None  # the underlying QueryResultItem(s); opaque here


def fuse(units: list[FusedUnit], weights: dict[str, float], k: int) -> list[FusedUnit]:
    """Score and order units by the total-order key (-rrf, -num_signals, -text_bearing, id)."""
    for u in units:
        u.rrf = rrf_score(u.signals, weights, k)
    units.sort(key=lambda u: (-u.rrf, -len(u.signals), -int(u.text_bearing), u.id))
    return units


def apply_expansion_floor(
    fused_units: list[FusedUnit],
    expansion_candidates: list[tuple[str, float]],  # (id, decay_score), not already present
    top_k: int,
    floor_slots: int,
    display_scale: float,
) -> list[FusedUnit]:
    """Fill-if-spare / additive expansion floor that NEVER evicts a fused item.

    Returns up to top_k fused units PLUS up to floor_slots expansion units whose
    display is capped strictly below the lowest fused display (a sub-band).
    """
    kept = fused_units[:top_k]
    if not expansion_candidates or floor_slots <= 0:
        return kept
    lowest = min((u.display for u in kept), default=display_scale)  # cap anchor
    # decay-ordered, distinct display values strictly below `lowest`
    floored: list[FusedUnit] = []
    ranked = sorted(expansion_candidates, key=lambda t: (-t[1], t[0]))[:floor_slots]
    for i, (eid, _decay) in enumerate(ranked):
        cap = lowest * (0.9 ** (i + 1))  # strictly decreasing, < lowest
        floored.append(FusedUnit(id=eid, signals={}, text_bearing=False, rrf=0.0, display=cap))
    return kept + floored
```
- [ ] **Step 4: Run** `pytest tests/services/test_rrf_fusion.py -v` → PASS. (Compute `display` on `kept` units via `display_score` inside `fuse` or the caller — add `u.display = display_score(u.rrf, c)` in `fuse` by threading `c`; update the test/signature if you prefer `fuse(units, weights, k, c)`. Keep the chosen signature consistent across Tasks 2 and 6.)
- [ ] **Step 5: Commit** `git add app/services/rrf_fusion.py tests/services/test_rrf_fusion.py && git commit -m "feat(retrieval): RRF fusion core module"`.

> **Type-consistency note for later tasks:** `fuse()` and `apply_expansion_floor()` take/return `FusedUnit`. Task 6 wraps each `QueryResultItem` (or per-image unit) in a `FusedUnit` with `payload=` the item(s), then maps the ordered `FusedUnit`s back to `QueryResultItem`s, setting `item.score = unit.display`.

---

## Task 3: Per-image unit collapse + tests

**Goal:** Collapse a candidate list into fusion units, merging a picture's `image` + `image_description` chunks (by `artifact_id`, modality-gated) into one unit that carries both lineages and `text_score = max(caption, description)`.

**Files:**
- Modify: `app/services/rrf_fusion.py` (add `build_units`)
- Test: `tests/services/test_rrf_fusion.py` (extend)

**Acceptance Criteria:**
- [ ] Only `modality ∈ {image, image_description, schematic}` with non-null `artifact_id` collapse together; plain `text`/`table` never merge even if they share an `artifact_id`.
- [ ] A merged unit's `text_score` = max of its text-bearing members' scores; `visual_score` = the image member's SigLIP prob.
- [ ] The merged unit retains BOTH source `chunk_id`s and both page numbers.

**Verify:** `pytest tests/services/test_rrf_fusion.py -k units -v` → pass.

**Steps:**

- [ ] **Step 1: Add tests** (text not merged; image+description merged; lineage retained):
```python
from app.services.rrf_fusion import build_units

class _Item:  # stand-in for QueryResultItem
    def __init__(self, chunk_id, modality, artifact_id=None, score=0.0, page_number=None):
        self.chunk_id, self.modality, self.artifact_id = chunk_id, modality, artifact_id
        self.score, self.page_number = score, page_number

def test_build_units_collapses_image_and_description():
    items = [
        _Item("img1", "image", artifact_id="A", score=0.51, page_number=7),
        _Item("desc1", "image_description", artifact_id="A", score=0.80, page_number=7),
        _Item("txt1", "text", artifact_id="A", score=0.95, page_number=2),  # same artifact, must NOT merge
    ]
    units = build_units(items)
    by = {tuple(sorted(u.member_chunk_ids)): u for u in units}
    assert ("desc1", "img1") in by  # collapsed
    merged = by[("desc1", "img1")]
    assert merged.text_score == 0.80 and merged.visual_score == 0.51
    assert set(merged.member_chunk_ids) == {"img1", "desc1"} and set(merged.pages) == {7}
    assert ("txt1",) in by  # text stayed separate
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Add to `rrf_fusion.py`:**
```python
_IMAGE_MODALITIES = {"image", "schematic"}
_TEXT_BEARING = {"text", "table", "image_description"}

@dataclass
class CandidateUnit:
    member_chunk_ids: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    text_score: float | None = None
    visual_score: float | None = None
    primary_chunk_id: str | None = None        # image chunk if present, else the text chunk
    image_description_chunk_id: str | None = None
    text_bearing: bool = False
    payload: object = None

def build_units(items: list) -> list["CandidateUnit"]:
    """Collapse image+image_description (same artifact_id) into one unit; everything else 1:1."""
    units: dict[str, CandidateUnit] = {}
    singles: list[CandidateUnit] = []
    for it in items:
        mod = getattr(it, "modality", "text")
        aid = getattr(it, "artifact_id", None)
        collapsible = aid is not None and mod in (_IMAGE_MODALITIES | {"image_description"})
        if not collapsible:
            u = CandidateUnit(member_chunk_ids=[str(it.chunk_id)],
                              pages=[it.page_number] if it.page_number is not None else [],
                              text_score=it.score if mod in _TEXT_BEARING else None,
                              visual_score=it.score if mod in _IMAGE_MODALITIES else None,
                              primary_chunk_id=str(it.chunk_id),
                              text_bearing=mod in _TEXT_BEARING, payload=it)
            singles.append(u)
            continue
        u = units.setdefault(aid, CandidateUnit())
        u.member_chunk_ids.append(str(it.chunk_id))
        if it.page_number is not None and it.page_number not in u.pages:
            u.pages.append(it.page_number)
        if mod in _IMAGE_MODALITIES:
            u.visual_score = it.score if u.visual_score is None else max(u.visual_score, it.score)
            u.primary_chunk_id = str(it.chunk_id)  # image is primary
            u.payload = it
        if mod in _TEXT_BEARING:  # image_description (or caption-bearing)
            u.text_score = it.score if u.text_score is None else max(u.text_score, it.score)
            u.image_description_chunk_id = str(it.chunk_id)
            u.text_bearing = True
            if u.primary_chunk_id is None:
                u.primary_chunk_id = str(it.chunk_id)
                u.payload = it
    return list(units.values()) + singles
```
- [ ] **Step 4: Run** `pytest tests/services/test_rrf_fusion.py -k units -v` → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(retrieval): per-image unit collapse for RRF"`.

> Note: caption text relevance (the Docling caption on the `image` chunk) is folded into `text_score` by Task 5's caption pass, which scores the caption and feeds it as a text-bearing member of the same artifact unit BEFORE `build_units` runs. `build_units`'s MAX is what prevents double-counting caption + description.

---

## Task 4: `for_fusion` on `_text_vector_search` and `_apply_reranker` (+ Text Basic equality)

**Goal:** Let the hybrid path obtain the WIDE reranked text pool (no `top_k` trim, floor as membership) while keeping `strategy=basic` byte-identical.

**Files:**
- Modify: `app/api/v1/retrieval.py` (`_apply_reranker`, `_text_vector_search`)
- Test: `tests/api/test_rrf_integration.py` (create — Text Basic equality)

**Acceptance Criteria:**
- [ ] `_apply_reranker(results, body, for_fusion=True)` returns the full reranked+gated set with NO `top_k` trim (widens the `cross_encoder_rerank` `top_k` arg to `len(rerankable)` AND skips `output[:body.top_k]`).
- [ ] `_text_vector_search(db, body, for_fusion=True)` returns the wide pool; `for_fusion=False` (default) is unchanged.
- [ ] Text Basic `"Fan Song"` returns byte-identical chunk_id+score list before and after this task.

**Verify:** `pytest tests/api/test_rrf_integration.py::test_text_basic_unchanged -v` → PASS.

**Steps:**

- [ ] **Step 1: Capture the Text Basic baseline** (snapshot file) and write the equality test:
```python
import json, subprocess
def _query(payload):
    out = subprocess.check_output(["curl","-s","-X","POST","http://localhost:8005/v1/retrieval/query",
        "-H","Content-Type: application/json","-d",json.dumps(payload)])
    d = json.loads(out)
    return [(r["chunk_id"], round(r["score"], 6)) for r in d["results"]]

def test_text_basic_unchanged():
    payload = {"query_text":"Fan Song","strategy":"basic","modality_filter":"all",
               "top_k":20,"reranker_top_n":20,"min_confidence":0.1,"include_context":True}
    got = _query(payload)
    # baseline captured pre-change into tests/api/fixtures/text_basic_fan_song.json
    expected = json.load(open("tests/api/fixtures/text_basic_fan_song.json"))
    assert got == [tuple(x) for x in expected]
```
(Capture the fixture first, before editing code: run the query, write `[[chunk_id, score], ...]` to the fixture path.)
- [ ] **Step 2:** Add `for_fusion: bool = False` to `_apply_reranker(results, body, for_fusion=False)`. Branch:
  - Replace `top_k=body.top_k` in the `cross_encoder_rerank(...)` call with `top_k=(len(rerankable) if for_fusion else body.top_k)`.
  - At the final return, `return output if for_fusion else output[:body.top_k]`.
  - In `for_fusion`, keep the floor gate (membership) but do NOT drop image passthrough specially (the RRF path doesn't call this for images — it only widens text). Leave `_NON_RERANK_MODALITIES` handling intact for the legacy path.
- [ ] **Step 3:** Add `for_fusion: bool = False` to `_text_vector_search(db, body, for_fusion=False)`. At its rerank call, pass `_apply_reranker(results, body, for_fusion=for_fusion)`; when `for_fusion`, skip the final `results[:body.top_k]` if any and return the wide pool (the `results[:retrieval_rerank_pool_size]` cap stays — that bounds compute).
- [ ] **Step 4:** Confirm `strategy=basic` calls `_text_vector_search(db, body)` with default `for_fusion=False` (`retrieval.py:84`). Run `pytest tests/api/test_rrf_integration.py::test_text_basic_unchanged -v` → PASS.
- [ ] **Step 5: Commit** `git commit -am "feat(retrieval): for_fusion wide-pool path; Text Basic unchanged"`.

---

## Task 5: Signal capture + caption pass in `_multi_modal_pipeline`

**Goal:** Build S_text (wide), S_visual (gated, pre-merge), S_ontology, and the caption→S_text pass; keep expansion seeds bounded to ~`top_k`.

**Files:**
- Modify: `app/api/v1/retrieval.py` (`_multi_modal_pipeline`, `_image_vector_search`)

**Acceptance Criteria:**
- [ ] S_visual = `_image_vector_search` output filtered to SigLIP prob ≥ `retrieval_rrf_visual_min_prob`, captured before `_merge_seed_results`.
- [ ] Expansion seeds = top ~`top_k` of the SORTED wide text pool (not the wide 128) — `_expand_seeds` fan-out unchanged in magnitude.
- [ ] Caption text is read from the image chunk regardless of `include_context`; captions are scored by the cross-encoder and attached as text-bearing members of their artifact unit.

**Verify:** Probe (manual) shows S_visual non-empty for "radar antenna", ≤1 for "Fan Song"; expansion still receives ~`top_k` seeds.

**Steps:**

- [ ] **Step 1:** In `_image_vector_search`, read caption regardless of `include_context`: change `content_text = (props.get("text") or props.get("chunk_text")) if body.include_context else None` to always set `content_text = props.get("text") or props.get("chunk_text")` when the RRF flag is on (guard: `if get_settings().retrieval_rrf_fusion_enabled or body.include_context`). Legacy path unchanged when flag off.
- [ ] **Step 2:** In `_multi_modal_pipeline`, when `settings.retrieval_rrf_fusion_enabled and strategy==hybrid`, branch into the RRF path (else the existing legacy body):
```python
# --- RRF path ---
wide_text = await _text_vector_search(db, body, for_fusion=True)   # S_text source (text/table/image_description)
images = await _image_vector_search(db, body)                      # S_visual source
visual = [r for r in images if (r.score or 0.0) >= _s.retrieval_rrf_visual_min_prob]
# caption pass: rerank image captions (Docling chunk_text) against the query, attach as text-bearing members
caption_items = _build_caption_items(visual)        # see Step 3
caption_scored = _apply_reranker(caption_items, body, for_fusion=True) if body.query_text else []
# expansion: bounded seeds = top ~top_k of the sorted wide text pool
seeds = _merge_seed_results([wide_text[:body.top_k], images[:body.top_k]])
expanded = await _expand_seeds(db, seeds, body.include_context, body.query_text)
expanded = await _rescore_expanded_chunks(expanded, body.query_text)
```
- [ ] **Step 3:** Add `_build_caption_items(visual)` — copy each image chunk into a lightweight item carrying `content_text=<caption>`, `modality="image_description"` semantics for reranking, and the SAME `artifact_id` (so Task 3's `build_units` folds it into the image's unit). Captions with empty text are skipped (they can't be scored).
- [ ] **Step 4:** Verify with the probe from earlier sessions: `"radar antenna"` → `len(visual) ≈ many`; `"Fan Song"` → `len(visual) ≤ 1`.
- [ ] **Step 5: Commit** `git commit -am "feat(retrieval): RRF signal capture + caption pass"`.

---

## Task 6: Wire fusion, floors, and min_confidence mapping

**Goal:** Replace the hybrid final ordering with RRF over the captured signals; apply ontology floor + fill-if-spare expansion floor; map `min_confidence` to per-signal floors; all behind the master flag.

**Files:**
- Modify: `app/api/v1/retrieval.py` (`_multi_modal_pipeline` RRF branch, `unified_query`)

**Acceptance Criteria:**
- [ ] Ranks assigned per signal (contiguous, stable) over: S_text = wide_text ∪ caption_scored ∪ (image_description from expansion); S_visual = visual; S_ontology = qualifying `ontology_relation` from `expanded`.
- [ ] Units built via `build_units`, fused via `fuse`, ontology floor guarantees ≤`retrieval_rrf_ontology_min_slots` qualifying units in top_k, expansion floor adds ≤`retrieval_rrf_expansion_floor_slots` non-leading items.
- [ ] `item.score` set to the unit's `display`; results dedup/diversified AFTER fusion, then returned.
- [ ] `unified_query` does NOT apply `min_confidence` to the fused score when hybrid+flag-on; instead the hybrid path uses effective floors `max(0.05, min_confidence)` (reranker) and `max(0.35, min_confidence)` (visual).

**Verify:** `"Fan Song"/All` deterministic ×3 (identical chunk_id list); text leads; image present only if genuine; `pytest tests/api/test_rrf_integration.py -v`.

**Steps:**

- [ ] **Step 1:** Assemble per-signal `(id, score)` lists and `assign_ranks` each (per-signal dedup by chunk_id first). Build `FusedUnit`s from `build_units` output, mapping each `CandidateUnit` to its signal ranks (text rank from S_text, visual rank from S_visual, ontology rank from S_ontology), `text_bearing` from the unit.
- [ ] **Step 2:** `fuse(units, {"text":w_text,"visual":w_visual,"ontology":w_ontology}, k)`, set `display` via `display_score(rrf, C)`. Apply ontology floor (ensure up to N qualifying ontology units survive the top_k cut — inject before trim, floor-aware). Apply `apply_expansion_floor` with the best `cross_modal`/`doc_structure` expansion units not already present.
- [ ] **Step 3:** Map ordered units back to `QueryResultItem`s: `item = unit.payload (primary)`, `item.score = unit.display`, attach lineage (Task 7). Run `_deduplicate_results` + `_diversify_results` on the final list, then `[:top_k (+floor)]`.
- [ ] **Step 4:** In `unified_query:102-103`, guard: `if body.min_confidence is not None and not (hybrid and fusion_enabled): results = [r for r in results if r.score >= body.min_confidence]`. The hybrid floor mapping lives in the pipeline (Steps 1/Task 5 gate values use `max(floor, min_confidence)`).
- [ ] **Step 5:** Determinism + smoke check:
```bash
for i in 1 2 3; do curl -s -X POST localhost:8005/v1/retrieval/query -H 'Content-Type: application/json' \
 -d '{"query_text":"Fan Song","strategy":"hybrid","modality_filter":"all","top_k":20,"min_confidence":0.1,"ontology_reserved_slots":3}' \
 | python3 -c "import sys,json;d=json.load(sys.stdin);print([r['chunk_id'] for r in d['results']])"; done
```
Expected: 3 identical lists.
- [ ] **Step 6: Commit** `git commit -am "feat(retrieval): wire RRF fusion + floors + min_confidence mapping"`.

---

## Task 7: Merged-card lineage carrier

**Goal:** The merged image card retains both source `chunk_id`s and pages (project hard-lineage rule).

**Files:**
- Modify: `app/api/v1/retrieval.py` (unit→item mapping in the RRF branch)

**Acceptance Criteria:**
- [ ] A merged image result's `context["merged_chunk_ids"]` lists both the image and image_description chunk_ids; `context["merged_sources"]` notes "visual"+"description"; `page_numbers` includes both pages.
- [ ] Existing backfills still resolve the image (they key off the primary image `chunk_id`/`artifact_id`).

**Verify:** Query "radar antenna"/All; a merged card has `context.merged_chunk_ids` length 2 and `page_numbers` populated.

**Steps:**

- [ ] **Step 1:** When mapping a collapsed `CandidateUnit` back to its `QueryResultItem`, set `item.context = {**(item.context or {}), "merged_chunk_ids": unit.member_chunk_ids, "merged_sources": ["visual","description"]}` and `item.page_numbers = sorted(set((item.page_numbers or []) + unit.pages))`.
- [ ] **Step 2:** Verify lineage present:
```bash
curl -s -X POST localhost:8005/v1/retrieval/query -H 'Content-Type: application/json' \
 -d '{"query_text":"radar antenna","strategy":"hybrid","modality_filter":"all","top_k":20,"min_confidence":0.1,"ontology_reserved_slots":3}' \
 | python3 -c "import sys,json;d=json.load(sys.stdin);print([ (len(r.get('context',{}).get('merged_chunk_ids',[])), r.get('page_numbers')) for r in d['results'] if r['modality']=='image'])"
```
- [ ] **Step 3: Commit** `git commit -am "feat(retrieval): retain both lineages on merged image cards"`.

---

## Task 8: Integration verification (PRE-MERGE GATE)

**USER-ORDERED GATE — NON-SKIPPABLE.** Requested by the user (3-round review demanded these gates). Close only after every acceptance criterion is independently re-validated with captured output.

**Goal:** Prove the feature behaves per spec and that the flag-off and Text-Basic paths are unchanged.

**Files:**
- Test: `tests/api/test_rrf_integration.py` (extend)

**Acceptance Criteria:**
- [ ] `"radar antenna"/All` → text + image + image_description interleaved; agreement units (good SigLIP + good caption/description) lead.
- [ ] `"Fan Song"/All` → text leads; ≤1 genuine image; on-page schematic appears only at the tail (display < lowest fused), never above text; identical across 3 runs.
- [ ] Text Basic `"Fan Song"` byte-identical (Task 4 test still green).
- [ ] `RETRIEVAL_RRF_FUSION_ENABLED=false` → hybrid output identical to pre-change legacy (capture a legacy baseline first).
- [ ] CUES live case still returns its qualifying ontology unit(s) in top_k.

**Verify:** `pytest tests/api/test_rrf_integration.py -v` → all pass; manual probes captured.

**Steps:**

- [ ] **Step 1:** Capture a legacy baseline for hybrid "Fan Song" with `RETRIEVAL_RRF_FUSION_ENABLED=false` (force-recreate api), save fixture.
- [ ] **Step 2:** Add tests: flag-off equality (re-run with flag false → equals fixture), determinism (3 identical runs flag-on), modality-mix assertions for "radar antenna", "no image above text on Fan Song" assertion, CUES ontology presence.
- [ ] **Step 3:** Recreate api with flag on; run `pytest tests/api/test_rrf_integration.py -v`; capture the manual probe outputs into the task close notes.
- [ ] **Step 4:** Run the full suite and VERIFICATION_CHECKLIST.md per the post-code-change workflow; update README if user-facing behavior changed.
- [ ] **Step 5: Commit** `git commit -am "test(retrieval): RRF fusion integration + flag-off/Text-Basic equality gates"`.

---

## Rollout note

Default `RETRIEVAL_RRF_FUSION_ENABLED=true`. Instant rollback = set false + force-recreate api (no redeploy). All shared-function edits are flag/`for_fusion`-branched; the off-path is the pre-change behavior, guarded by the Task 8 equality test.
