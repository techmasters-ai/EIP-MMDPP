"""Worker-side overlay application (spec §5.3, Mechanism A1).

This module is the canonical worker-side home of:
  - Wire types: TableOverlay, TableFact, CrossEntityHint (mirror of
    docker/docling-graph/app/schemas.py)
  - Stats dataclasses: RewriteStats, OverlayStats
  - Worker-side env-flag check: is_overlay_enabled_worker()
  - Per-pass-result overlay extraction: extract_doc_overlay(pass_results)
  - Per-fact apply functions: apply_identity_rewrite, apply_field_overlay
  - Orchestrator that wires Phase 0 + Phase 0.5 in the right order:
    apply_table_overlay_phases()

The orchestrator is the integration surface. A merge dispatcher
(current merge_and_resolve, future derive_ontology_graph_merge) only
needs to:

  from app.services.table_overlay import apply_table_overlay_phases
  apply_table_overlay_phases(
      pass_results=pass_results,
      ontology=ontology,
      document_id=document_id,
      canonicalize_fn=canonicalize_cross_pass_identities,
  )

Everything else (env-flag check, kill-switch authority over cached
overlays, Phase 0 alias rewrite, Phase 0.5 field overlay, log-line
emission) is encapsulated. Spec §5.5.

Drift guard: docker/docling-graph/tests/test_table_overlay_schemas.py
::test_parser_and_worker_table_overlay_classes_round_trip asserts
field-shape equivalence between this module's TableOverlay/TableFact
and the parser-side declarations via JSON round-trip.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Module-level env-flag helper. Spec §4.3 — worker-side check is
# AUTHORITATIVE over cached overlay payloads loaded from
# pipeline_pass_outputs.metadata_json. Even when a PassResult arrives
# carrying a populated table_overlay, this flag suppresses both Phase 0
# and Phase 0.5 application.
# ----------------------------------------------------------------------
_ENV_VAR = "DOCLING_GRAPH_TABLE_OVERLAY_ENABLED"


def is_overlay_enabled_worker() -> bool:
    """Read DOCLING_GRAPH_TABLE_OVERLAY_ENABLED. Default True.
    Returns False only when the env value lowercases to 'false'.
    Spec §4.3."""
    return os.environ.get(_ENV_VAR, "true").lower() != "false"


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


# ----------------------------------------------------------------------
# Cross-pass overlay extraction. Spec §5.5.
# ----------------------------------------------------------------------


def extract_doc_overlay(pass_results: dict) -> "TableOverlay | None":
    """Find the first non-empty `table_overlay` across all pass_results.

    All passes from the same DoclingDocument should ship structurally
    identical overlays (the parser is deterministic per doc). When two
    passes' overlays diverge, log a WARNING and use the first non-empty
    one. Returns None when no pass has a non-empty overlay.
    """
    first = None
    for pass_name, pr in pass_results.items():
        ov = getattr(pr, "table_overlay", None)
        if ov is None:
            continue
        is_nonempty = bool(
            ov.alias_map_by_entity_type or ov.facts or ov.cross_entity_hints
        )
        if not is_nonempty:
            continue
        if first is None:
            first = ov
            continue
        if ov.model_dump() != first.model_dump():
            logger.warning(
                "extract_doc_overlay: divergent overlays across passes — "
                "using first non-empty. Inspect parser deterministic "
                "behavior. first_facts=%d other_facts=%d",
                len(first.facts), len(ov.facts),
            )
    return first


# ----------------------------------------------------------------------
# Top-level orchestrator. Wires Phase 0 (alias rewrite via the supplied
# canonicalize callable) and Phase 0.5 (per-cell field overlay) in the
# right order, with the worker-side kill switch as the authoritative
# gate. Spec §5.5.
#
# A merge dispatcher (current merge_and_resolve, future
# derive_ontology_graph_merge) only needs to call this function and
# the rest of merge can proceed against post-overlay PassResults.
# ----------------------------------------------------------------------


def apply_table_overlay_phases(
    pass_results: dict,
    *,
    ontology: dict,
    document_id: str,
    canonicalize_fn: Callable,
) -> Optional["OverlayStats"]:
    """Apply Phase 0 + Phase 0.5 of the Mechanism A1 overlay.

    Order:
      1. Read worker-side env flag (is_overlay_enabled_worker).
      2. If disabled AND any pass has a cached overlay → emit
         TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER log. Skip both phases.
      3. Extract doc-level overlay via extract_doc_overlay(pass_results).
      4. Phase 0 (cross-pass identity canonicalization): call
         canonicalize_fn with table_alias_map_by_entity_type set when
         enabled+overlay-present; otherwise None. The canonicalize_fn
         itself runs apply_identity_rewrite (when alias_map provided)
         followed by the existing token-overlap pass.
      5. Phase 0.5 (per-cell field overlay): when enabled+overlay+facts,
         call apply_field_overlay and emit TABLE_OVERLAY_APPLIED log.
         apply_field_overlay exceptions are caught (bounded-degraded
         per spec §7).

    Returns the OverlayStats if Phase 0.5 ran, otherwise None.

    Args:
      pass_results: dict[pass_name, PassResult].
      ontology: ontology dict (passed through to canonicalize_fn).
      document_id: for log line correlation.
      canonicalize_fn: a callable matching the signature
        canonicalize_cross_pass_identities(pass_results, ontology, *,
        table_alias_map_by_entity_type=…) -> int. The orchestrator
        accepts this as a parameter to avoid an import cycle with
        extraction_merge.py and to let future refactors thread their
        own canonicalization.
    """
    overlay_enabled = is_overlay_enabled_worker()
    table_overlay = (
        extract_doc_overlay(pass_results) if overlay_enabled else None
    )

    # Worker-side kill switch is authoritative over cached overlays.
    # Spec §4.3.
    if not overlay_enabled:
        cached_overlay_present = sum(
            1 for pr in pass_results.values()
            if getattr(pr, "table_overlay", None) is not None
        )
        if cached_overlay_present:
            logger.info(
                "TABLE_OVERLAY_KILL_SWITCH_ACTIVE_WORKER doc_id=%s "
                "pass_count=%d cached_overlay_present=%d",
                document_id, len(pass_results), cached_overlay_present,
            )

    # Phase 0: cross-pass identity canonicalization. When overlay is
    # enabled AND we found one, pass the alias map through; otherwise
    # call with None and rely on the existing token-overlap pass.
    canonicalize_fn(
        pass_results,
        ontology,
        table_alias_map_by_entity_type=(
            table_overlay.alias_map_by_entity_type
            if (overlay_enabled and table_overlay is not None)
            else None
        ),
    )

    # Phase 0.5: per-cell field overlay. Only when overlay is enabled,
    # we found one, and it carries facts.
    if overlay_enabled and table_overlay is not None and table_overlay.facts:
        try:
            stats = apply_field_overlay(
                pass_results,
                table_overlay.facts,
                policy="table_wins_for_table_facts",
            )
            logger.info(
                "TABLE_OVERLAY_APPLIED doc_id=%s "
                "field_overlay_applied=%d matches_touched=%d "
                "skipped_no_entity=%d skipped_unknown_field=%d "
                "skipped_validation_fail=%d conflicts_overridden=%d "
                "policy=%s",
                document_id, stats.applied, stats.matches_touched,
                stats.skipped_no_entity, stats.skipped_unknown_field,
                stats.skipped_validation_fail, stats.conflicts_overridden,
                stats.policy_active,
            )
            return stats
        except Exception as exc:
            logger.warning(
                "apply_field_overlay failed mid-loop: %s — proceeding "
                "with merge using whatever (fact, instance) swaps had "
                "already completed. Bounded-degraded per §7. Operator "
                "rollback via kill switch only.", exc,
            )
    return None
