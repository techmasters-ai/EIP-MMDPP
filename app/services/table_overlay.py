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


@dataclass
class OverlayStats:
    applied: int = 0                  # fact-instance count (fan-out)
    matches_touched: int = 0          # fact count that landed on >=1 inst
    skipped_no_entity: int = 0
    skipped_unknown_field: int = 0
    skipped_validation_fail: int = 0
    conflicts_overridden: int = 0
    policy_active: str = "table_wins_for_table_facts"
    def as_dict(self) -> dict: return asdict(self)


def _instances_for_fact(
    pass_results: dict,
    fact: Any,
) -> list[Any]:
    """Enumerate ALL instances in pass_results[fact.pass_name] of type
    fact.entity_type whose system_name == fact.canonical_entity (post-
    rewrite). Empty list if pass_name not in pass_results OR no
    matches."""
    pr = pass_results.get(fact.pass_name)
    if pr is None:
        return []
    try:
        candidates = list(pr.iter_entities_of_type(fact.entity_type))
    except Exception as exc:
        logger.warning(
            "apply_field_overlay: iter_entities_of_type failed for "
            "pass=%s entity_type=%s: %s",
            fact.pass_name, fact.entity_type, exc,
        )
        return []
    return [
        inst for inst in candidates
        if getattr(inst, "system_name", None) == fact.canonical_entity
    ]


def apply_field_overlay(
    pass_results: dict,
    table_facts: list,
    *,
    policy: str = "table_wins_for_table_facts",
) -> OverlayStats:
    """Apply per-cell table facts to Pydantic entity instances. Spec §5.3.

    Per-(fact, instance) atomicity: a model_validate failure on one
    instance leaves that instance UNCHANGED and does NOT block
    fan-out to siblings. The overall loop is NOT a single transaction.
    """
    stats = OverlayStats(policy_active=policy)

    for fact in table_facts:
        matches = _instances_for_fact(pass_results, fact)
        if not matches:
            stats.skipped_no_entity += 1
            continue

        any_landed = False
        for inst in matches:
            cls = type(inst)

            # (a) Pre-validate field name. extra="ignore" would drop
            # unknown keys silently otherwise.
            if not isinstance(inst, BaseModel):
                stats.skipped_unknown_field += 1
                continue
            if fact.schema_field not in cls.model_fields:
                stats.skipped_unknown_field += 1
                logger.info(
                    "FIELD_OVERLAY_UNKNOWN_FIELD pass=%s entity_type=%s "
                    "entity=%s schema_field=%s model=%s — fact dropped",
                    fact.pass_name, fact.entity_type, fact.canonical_entity,
                    fact.schema_field, cls.__name__,
                )
                continue

            # (b) capture original
            original = getattr(inst, fact.schema_field, None)

            # (c) full model validation. The candidate dict carries
            # the LLM's existing values for every other field plus
            # fact.value for fact.schema_field — model_validate runs
            # every field_validator(mode="before") hook AND every
            # Field(...) constraint AND any model_validator(mode=
            # "after"). Coerced output is read back via getattr.
            candidate = {**inst.model_dump(), fact.schema_field: fact.value}
            try:
                revalidated = cls.model_validate(candidate)
            except (ValidationError, ValueError, TypeError):
                stats.skipped_validation_fail += 1
                continue
            coerced = getattr(revalidated, fact.schema_field)

            # (d) atomic single-field setattr. We mutate ONLY
            # fact.schema_field on `inst`, using the validated
            # `coerced` value. Per-pair atomicity: either the field
            # changes to coerced or it doesn't change at all (the
            # try/except below). We deliberately do NOT loop over
            # revalidated.model_dump() and copy every field — that
            # would silently rewrite siblings to whatever shape
            # model_validate produced (string→float coercions on
            # un-touched fields, etc.) and could surprise downstream
            # logic that expects unchanged LLM values for fields the
            # overlay didn't touch.
            try:
                setattr(inst, fact.schema_field, coerced)
            except Exception as exc:
                # Pydantic on these schemas does NOT have
                # validate_assignment=True, so a normal float setattr
                # cannot fail. If a future schema change makes setattr
                # fail (e.g., adding validate_assignment=True with a
                # rejecting validator), failing loudly is the right
                # call: silent-pass would mean we count applied++
                # for an instance whose value didn't actually change.
                logger.warning(
                    "FIELD_OVERLAY_SETATTR_FAILED pass=%s entity_type=%s "
                    "entity=%s field=%s coerced=%r — instance unchanged: %s",
                    fact.pass_name, fact.entity_type, fact.canonical_entity,
                    fact.schema_field, coerced, exc,
                )
                stats.skipped_validation_fail += 1
                continue

            # (e) per-instance bookkeeping
            stats.applied += 1
            any_landed = True
            if original is not None and original != coerced:
                stats.conflicts_overridden += 1
                logger.info(
                    "FIELD_OVERLAY_OVERRIDE pass=%s entity_type=%s "
                    "entity=%s field=%s llm=%r table=%r source=%r",
                    fact.pass_name, fact.entity_type, fact.canonical_entity,
                    fact.schema_field, original, coerced, fact.source_label,
                )

        # step 3: fact-level matches_touched (once per fact that
        # landed on >=1 instance — NOT per instance).
        if any_landed:
            stats.matches_touched += 1

    return stats
