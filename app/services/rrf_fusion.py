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
    return sum(weights.get(s, 0.0) / (k + rank) for s, rank in signal_ranks.items())


def display_score(rrf: float, c: float) -> float:
    return rrf / (rrf + c) if (rrf + c) > 0 else 0.0


@dataclass
class FusedUnit:
    id: str
    signals: dict[str, int] = field(default_factory=dict)
    text_bearing: bool = False
    rrf: float = 0.0
    display: float = 0.0
    payload: object = None


def fuse(units: list[FusedUnit], weights: dict[str, float], k: int, c: float) -> list[FusedUnit]:
    for u in units:
        u.rrf = rrf_score(u.signals, weights, k)
        u.display = display_score(u.rrf, c)
    units.sort(key=lambda u: (-u.rrf, -len(u.signals), -int(u.text_bearing), u.id))
    return units


def apply_expansion_floor(
    fused_units: list[FusedUnit],
    expansion_candidates: list[tuple[str, float]],
    top_k: int,
    floor_slots: int,
    display_scale: float,
) -> list[FusedUnit]:
    """Fill-if-spare / additive expansion floor that NEVER evicts a fused item.

    Returns up to top_k fused units PLUS up to floor_slots expansion units whose
    display is capped strictly below the lowest fused display.
    """
    kept = fused_units[:top_k]
    if not expansion_candidates or floor_slots <= 0:
        return kept
    lowest = min((u.display for u in kept), default=display_scale)
    floored: list[FusedUnit] = []
    ranked = sorted(expansion_candidates, key=lambda t: (-t[1], t[0]))[:floor_slots]
    for i, (eid, _decay) in enumerate(ranked):
        cap = lowest * (0.9 ** (i + 1))
        floored.append(FusedUnit(id=eid, signals={}, text_bearing=False, rrf=0.0, display=cap))
    return kept + floored


_IMAGE_MODALITIES = {"image", "schematic"}
_TEXT_BEARING = {"text", "table", "image_description"}


@dataclass
class CandidateUnit:
    member_chunk_ids: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    text_score: float | None = None
    visual_score: float | None = None
    primary_chunk_id: str | None = None
    image_description_chunk_id: str | None = None
    text_bearing: bool = False
    payload: object = None


def build_units(items: list) -> list["CandidateUnit"]:
    """Collapse image + image_description (same artifact_id) into one unit; everything else 1:1.

    text_score = max over text-bearing members; visual_score = image member's score.
    Both source chunk_ids and pages are retained.
    """
    units: dict[str, CandidateUnit] = {}
    singles: list[CandidateUnit] = []
    for it in items:
        mod = getattr(it, "modality", "text")
        aid = getattr(it, "artifact_id", None)
        collapsible = aid is not None and mod in (_IMAGE_MODALITIES | {"image_description"})
        if not collapsible:
            singles.append(CandidateUnit(
                member_chunk_ids=[str(it.chunk_id)],
                pages=[it.page_number] if it.page_number is not None else [],
                text_score=it.score if mod in _TEXT_BEARING else None,
                visual_score=it.score if mod in _IMAGE_MODALITIES else None,
                primary_chunk_id=str(it.chunk_id),
                text_bearing=mod in _TEXT_BEARING,
                payload=it,
            ))
            continue
        u = units.setdefault(aid, CandidateUnit())
        u.member_chunk_ids.append(str(it.chunk_id))
        if it.page_number is not None and it.page_number not in u.pages:
            u.pages.append(it.page_number)
        if mod in _IMAGE_MODALITIES:
            u.visual_score = it.score if u.visual_score is None else max(u.visual_score, it.score)
            u.primary_chunk_id = str(it.chunk_id)
            u.payload = it
        if mod in _TEXT_BEARING:
            u.text_score = it.score if u.text_score is None else max(u.text_score, it.score)
            u.image_description_chunk_id = str(it.chunk_id)
            u.text_bearing = True
            if u.primary_chunk_id is None:
                u.primary_chunk_id = str(it.chunk_id)
                u.payload = it
    return list(units.values()) + singles
