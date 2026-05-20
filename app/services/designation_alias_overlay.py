"""Step 6: deterministic designation alias expansion from labeled
designation-row tables.

Tables of the SA-2 master-designations shape carry one row per identity
class (Industry / Military / NATO) plus a canonical-entity row (Missile
Type). Each column maps a tuple of designations to one canonical entity.
This module produces, per column, an alias bag attached to the
canonical entity — to be merged onto the matching missile_identity
entity in postprocess.

Generic — no equipment names anywhere. Operates on identity-row labels
and column position; works on any table that uses the same label
vocabulary (e.g. SAM, AAM, BM corpora).

Schema mapping (per user spec, Step 6):
  * Industry / Military Designation  → `nomenclature_aliases`
  * NATO Designation                 → `name_aliases`
  * `dieqp` is NOT used (kept for ICD-specific identifiers).

Guardrails:
  * Skip a table that has no canonical-entity row (don't synthesize
    new top-level entities just because designation rows exist).
  * Skip a column whose canonical cell is empty.
  * Skip an alias whose value equals the canonical entity name (no
    self-loops).
  * Skip an alias cell that's empty.
  * Process only matrix tables — list tables don't have the same
    column semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from app.services.table_normalization.models import NormalizedTable


class DesignationRowKind(Enum):
    INDUSTRY = "industry"
    MILITARY = "military"
    NATO = "nato"
    CANONICAL_ENTITY = "canonical_entity"


# Mapping of normalized identity-row labels to their kind. Lookup is
# case-insensitive and whitespace-collapsed. Order doesn't matter.
_IDENTITY_LABEL_TO_KIND: dict[str, DesignationRowKind] = {
    "industry designation": DesignationRowKind.INDUSTRY,
    "military designation": DesignationRowKind.MILITARY,
    "nato designation": DesignationRowKind.NATO,
    "missile type": DesignationRowKind.CANONICAL_ENTITY,
    "missile designation": DesignationRowKind.CANONICAL_ENTITY,
    "round designation": DesignationRowKind.CANONICAL_ENTITY,
}


def _normalize_label(s: str) -> str:
    return " ".join(s.lower().split())


def _classify_identity_label(label: str) -> DesignationRowKind | None:
    if not isinstance(label, str):
        return None
    return _IDENTITY_LABEL_TO_KIND.get(_normalize_label(label))


@dataclass(frozen=True)
class DesignationAliases:
    """Per-canonical-entity alias bag derived from one table column."""
    canonical_entity: str
    nomenclature_aliases: tuple[str, ...]
    name_aliases: tuple[str, ...]
    source_table_index: int = -1


def _column_aliases(
    column_identity: dict[str, str],
    table_index: int,
) -> DesignationAliases | None:
    """Project a single column's identity dict into an alias bag.

    Returns None when the column lacks a canonical entity or contributes
    no aliases. Order of nomenclature_aliases is Military first, then
    Industry — matches the user's "Military > Industry" preference and
    keeps Military as the primary nomenclature alias when both exist.
    """
    canonical_value: str | None = None
    by_kind: dict[DesignationRowKind, str] = {}
    for raw_label, value in column_identity.items():
        if not isinstance(value, str) or not value.strip():
            continue
        kind = _classify_identity_label(raw_label)
        if kind is None:
            continue
        v = value.strip()
        # Guard: merged label cells in the source can leak the literal
        # identity-row label as a cell VALUE for adjacent columns (e.g.
        # `{"Industry Designation": "Industry Designation"}`). Drop
        # values that themselves are recognized identity row labels —
        # they're never meaningful aliases.
        if _classify_identity_label(v) is not None:
            continue
        if kind is DesignationRowKind.CANONICAL_ENTITY:
            canonical_value = v
        else:
            by_kind[kind] = v
    if canonical_value is None:
        return None
    nomenclature: list[str] = []
    for kind in (DesignationRowKind.MILITARY, DesignationRowKind.INDUSTRY):
        v = by_kind.get(kind)
        if v is None or v == canonical_value or v in nomenclature:
            continue
        nomenclature.append(v)
    names: list[str] = []
    v = by_kind.get(DesignationRowKind.NATO)
    if v is not None and v != canonical_value and v not in names:
        names.append(v)
    if not nomenclature and not names:
        return None
    return DesignationAliases(
        canonical_entity=canonical_value,
        nomenclature_aliases=tuple(nomenclature),
        name_aliases=tuple(names),
        source_table_index=table_index,
    )


def expand_designation_aliases(
    tables: Iterable[NormalizedTable],
) -> list[DesignationAliases]:
    """Walk normalized tables; for each table that has BOTH a canonical-
    entity identity row AND at least one designation identity row, emit
    one DesignationAliases per column that yields a canonical entity.

    Returns aliases in column-iteration order across tables.
    """
    out: list[DesignationAliases] = []
    for table in tables:
        if not isinstance(table, NormalizedTable):
            continue
        # Quick filter: skip tables whose columns carry no designation
        # identity labels at all.
        has_canonical = False
        has_designation = False
        for col in table.columns:
            for raw_label in col.identity:
                kind = _classify_identity_label(raw_label)
                if kind is DesignationRowKind.CANONICAL_ENTITY:
                    has_canonical = True
                elif kind is not None:
                    has_designation = True
        if not (has_canonical and has_designation):
            continue
        for col in table.columns:
            bag = _column_aliases(col.identity, table.table_index)
            if bag is not None:
                out.append(bag)
    return out


def merge_alias_bags_by_canonical(
    bags: Iterable[DesignationAliases],
) -> dict[str, DesignationAliases]:
    """Collapse multiple bags for the same canonical entity (rare — when
    a designation table appears across sections). Preserves order: first
    occurrence wins, later additions append uniquely. The returned
    mapping is keyed by canonical_entity name."""
    merged: dict[str, dict[str, list[str]]] = {}
    first_source: dict[str, int] = {}
    for bag in bags:
        slot = merged.setdefault(
            bag.canonical_entity,
            {"nomenclature": [], "name": []},
        )
        first_source.setdefault(bag.canonical_entity, bag.source_table_index)
        for v in bag.nomenclature_aliases:
            if v not in slot["nomenclature"]:
                slot["nomenclature"].append(v)
        for v in bag.name_aliases:
            if v not in slot["name"]:
                slot["name"].append(v)
    return {
        name: DesignationAliases(
            canonical_entity=name,
            nomenclature_aliases=tuple(v["nomenclature"]),
            name_aliases=tuple(v["name"]),
            source_table_index=first_source[name],
        )
        for name, v in merged.items()
    }
