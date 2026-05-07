"""Worker-side overlay application (spec §5.3, Mechanism A1).

This module is also the canonical home of the worker-side TableOverlay,
TableFact, CrossEntityHint Pydantic classes. The parser-side mirror in
docker/docling-graph/app/schemas.py is a structurally-identical
declaration; JSON travels between them. A drift-guard test in Task 9
asserts field-shape equivalence by JSON round-trip.

Two functions operate on Pydantic instances reachable via
PassResult.iter_entities_of_type:

  apply_identity_rewrite — entity-type-scoped system_name alias
    collapse, runs inside canonicalize_cross_pass_identities BEFORE
    the existing token-overlap pass.

  apply_field_overlay — per-cell field overlay with
    table_wins_for_table_facts policy (default), full
    cls.model_validate(...) gate, fan-out to all matching post-rewrite
    instances, per-(fact, instance) atomicity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Wire types (mirror of docker/docling-graph/app/schemas.py declarations).
# Drift guard: docker/docling-graph/tests/test_table_overlay_schemas.py
#   ::test_parser_and_worker_table_overlay_classes_round_trip asserts
# field-shape equivalence between this declaration and the parser side.
# ----------------------------------------------------------------------


class TableFact(BaseModel):
    """Spec §5.4."""
    model_config = ConfigDict(frozen=True)
    canonical_entity: str
    entity_type: str
    schema_field: str
    value: Any
    source_label: str
    section_ctx: Optional[str] = None
    pass_name: str
    raw_text: str


class CrossEntityHint(BaseModel):
    """Spec §5.4."""
    model_config = ConfigDict(frozen=True)
    source_canonical: str
    source_entity_type: str
    target_alias: str
    target_entity_type: str
    relationship_kind: str


class TableOverlay(BaseModel):
    """Spec §5.4."""
    alias_map_by_entity_type: dict[str, dict[str, str]] = Field(default_factory=dict)
    facts: list[TableFact] = Field(default_factory=list)
    cross_entity_hints: list[CrossEntityHint] = Field(default_factory=list)


TableFact.model_rebuild()
CrossEntityHint.model_rebuild()
TableOverlay.model_rebuild()


@dataclass
class RewriteStats:
    rewrites: int = 0
    unique_canonicals: int = 0
    passes_touched: int = 0
    def as_dict(self) -> dict: return asdict(self)


def apply_identity_rewrite(
    pass_results: dict,           # dict[str, PassResult]
    alias_map_by_entity_type: dict[str, dict[str, str]],
    ontology: dict,
) -> RewriteStats:
    """Mutate Pydantic instances in-place: where system_name is in the
    alias map for the instance's entity_type, replace with canonical.
    Idempotent (alias_map[canonical] == canonical short-circuits).
    Spec §5.3.
    """
    stats = RewriteStats()
    if not alias_map_by_entity_type:
        return stats

    canonicals: set[str] = set()
    for entity_def in ontology.get("entity_types", []) or []:
        entity_type = entity_def.get("name")
        if not entity_type:
            continue
        sub_map = alias_map_by_entity_type.get(entity_type) or {}
        if not sub_map:
            continue
        for pass_name, pass_result in pass_results.items():
            touched_this_pass = False
            try:
                instances = list(pass_result.iter_entities_of_type(entity_type))
            except Exception as exc:
                logger.warning(
                    "apply_identity_rewrite: iter_entities_of_type failed for "
                    "pass=%s entity_type=%s: %s", pass_name, entity_type, exc,
                )
                continue
            for inst in instances:
                current = getattr(inst, "system_name", None)
                if not current or current not in sub_map:
                    continue
                canonical = sub_map[current]
                if current == canonical:
                    canonicals.add(canonical)
                    continue
                try:
                    inst.system_name = canonical
                    stats.rewrites += 1
                    canonicals.add(canonical)
                    touched_this_pass = True
                except Exception as exc:
                    logger.warning(
                        "apply_identity_rewrite: cannot set system_name on "
                        "%s instance: %s", entity_type, exc,
                    )
            if touched_this_pass:
                stats.passes_touched += 1
    stats.unique_canonicals = len(canonicals)
    return stats
