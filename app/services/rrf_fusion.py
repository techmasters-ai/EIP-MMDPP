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
