"""System links pass — cross-pass relationships (Decision 4 exception).

This file is the documented multi-pass-architecture exception. Unlike
other passes where relationships move inside entity classes via typed
``edge(label=...)`` fields (Phase 5), cross-pass linking operates on
already-extracted upstream refs (``from_ref_id``/``to_ref_id``) that are
not in-scope Python objects. A DTO record is the only workable shape:
the LLM receives a ref catalog and emits pairs. The merge layer resolves
those ref pairs to ``LogicalIdentity`` via ``PassResult.upstream_refs``.

Key entities: None — ``SystemLinksPass`` carries no entity collections.
Key relationships (docs-valid RelationshipType labels):
- ``ASSOCIATED_WITH`` — generic cross-pass association
- ``CUES`` — one system cues another for target handoff

Plan v32 Task 49 (Phase 5 docs-compliance housekeeping).
Per docs "Template Basics → Edge Helper Function → Required Definition",
the ``edge()`` helper is defined identically here even though this file
does not call it — the DTO pattern replaces typed edges for cross-pass
linking. Adherence to the literal docs matters so introspection and
contract tests treat this file consistently with entity-bearing passes.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..relationships import RelationshipType
from ..validators import (
    coerce_optional_confidence,
    normalize_enum,
)


def edge(
    label: str,
    *,
    description: str | None = None,
    examples: list | None = None,
    **field_kwargs: Any,
) -> Any:
    """Helper: declare a typed entity-to-entity edge field.

    Per docs "Template Basics → Edge Helper Function → Required Definition":
    this function must be defined identically in every template. This
    file does not call it — ``SystemLinksPass.relationships`` is a plain
    DTO list per Decision 4 — but the helper is present verbatim for
    uniform template shape.
    """
    existing_extra = field_kwargs.pop("json_schema_extra", None) or {}
    existing_extra["edge_label"] = label
    if description is not None:
        field_kwargs["description"] = description
    if examples is not None:
        field_kwargs["examples"] = examples
    return Field(json_schema_extra=existing_extra, **field_kwargs)


class SystemLinkRelationship(BaseModel):
    """Cross-pass relationship DTO — Decision 4 exception.

    Carries ``from_ref_id`` / ``to_ref_id`` (strings referencing upstream
    entity refs emitted by earlier passes). The merge layer resolves
    these against ``PassResult.upstream_refs`` to build a
    ``MergedEdgeRecord`` with proper ``LogicalIdentity`` endpoints.
    """
    model_config = ConfigDict(extra="ignore", is_entity=False)

    rel_type: Optional[str] = Field(
        default=None,
        description=(
            "Cross-pass relationship type. Must be one of the allowed "
            "RelationshipType enum values declared in the manifest's "
            "`extracted_relationship_types` for this pass. For "
            "air_defense_v3 the allowed set is: "
            "ASSOCIATED_WITH = generic pairing (e.g. a fire-control "
            "radar paired with the weapon it guides); "
            "CUES = directional handoff (e.g. a search radar cues a "
            "fire-control radar or a missile system, passing target "
            "location and velocity). "
            "Emit ASSOCIATED_WITH when the text describes a persistent "
            "pairing without explicit handoff. Emit CUES when the text "
            "describes one system passing target data to another."
        ),
        examples=["ASSOCIATED_WITH", "CUES"],
    )
    from_ref_id: Optional[str] = Field(
        default=None,
        description=(
            "Upstream ref id of the edge source — a typed token like "
            "'RADAR_SYSTEM:Fan Song' or 'MISSILE_SYSTEM:1D' that refers "
            "to an entity previously emitted by an earlier pass "
            "(radar_identity, missile_identity). The catalog of available "
            "ref ids is provided in the prompt's 'Upstream entities:' "
            "preamble (look for 'REF=<ref_id>' anchors). Emit only ref ids "
            "that appear in that list — the merge layer rejects unknown "
            "ids. Do NOT wrap the ref_id in square brackets."
        ),
        examples=["RADAR_SYSTEM:Fan Song", "MISSILE_SYSTEM:1D"],
    )
    to_ref_id: Optional[str] = Field(
        default=None,
        description=(
            "Upstream ref id of the edge target — same format as "
            "from_ref_id. The merge layer resolves (from_ref_id, "
            "to_ref_id) pairs against PassResult.upstream_refs to build "
            "MergedEdgeRecord with real LogicalIdentity endpoints. "
            "from_ref_id and to_ref_id must be different refs — "
            "self-loops are rejected. Do NOT wrap the ref_id in square "
            "brackets."
        ),
        examples=["MISSILE_SYSTEM:1D", "RADAR_SYSTEM:Spoon Rest"],
    )
    confidence: Optional[float] = Field(
        default=None,
        description=(
            "Extraction confidence for this cross-system relationship, "
            "0-1. Use 0.9-1.0 when the text explicitly names both "
            "endpoints together in a kill-chain or pairing statement; "
            "0.5-0.8 when the relationship is inferred from adjacent "
            "mentions; <0.5 only for speculative links. System-populated "
            "— leave null if unsure."
        ),
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    # Per plan Task 42: rel_type validation via the canonical
    # RelationshipType enum. The normalize_enum helper (validators.py)
    # accepts the full set so cross-pass passes can emit any docs-valid
    # label, and the merge layer filters on the per-pass allowed set
    # declared in manifest.passes[...].extracted_relationship_types.
    _v_rel_type = field_validator("rel_type", mode="before")(
        normalize_enum({member.value for member in RelationshipType}),
    )
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)

    @staticmethod
    def _strip_ref_brackets(v: Any) -> Any:
        """v8a defensive strip: if the LLM wrapped the ref_id in '[…]'
        brackets despite our prompt format using REF= anchors, peel
        exactly one surrounding pair off. Preserves any legitimate
        internal characters."""
        if isinstance(v, str):
            stripped = v.strip()
            if (
                len(stripped) >= 2
                and stripped[0] == "["
                and stripped[-1] == "]"
            ):
                return stripped[1:-1].strip()
            return stripped
        return v

    _v_from_ref_id = field_validator("from_ref_id", mode="before")(_strip_ref_brackets)
    _v_to_ref_id = field_validator("to_ref_id", mode="before")(_strip_ref_brackets)


class SystemLinksPass(BaseModel):
    """Relationships-only pass root. Has NO entity fields — enforced by
    the 'input_mode == document_plus_entity_refs implies no entity-
    collection fields' sub-check in the manifest self-consistency section
    of §2 checker rules. Decision 4: ``relationships`` uses a plain
    ``Field`` (not ``edge()``) because its items are DTOs, not entities.

    is_entity=True per docling-graph-docs.md §Template Basics → Root
    Document Model (line 18859). See radar_identity.py for full rationale.
    graph_id_fields=[] because the DTO-emitting pass-root has no natural
    identity; the walker skips it at at_pass_root=True.
    """
    model_config = ConfigDict(
        extra="ignore",
        is_entity=True,
        graph_id_fields=[],
    )

    relationships: List[SystemLinkRelationship] = Field(
        default_factory=list,
        description=(
            "Cross-pass relationship DTOs emitted by this pass. Each carries "
            "from_ref_id / to_ref_id that the merge layer resolves against "
            "PassResult.upstream_refs. "
            "EMIT when the document describes multiple named systems as part "
            "of one engagement kill-chain. Typical patterns: "
            "(a) early-warning / search radar → fire-control radar → weapon "
            "system (e.g. 'Spoon Rest detected at long range and handed off "
            "to Fan Song, which guided the SA-2 missile'). That narrative "
            "yields TWO edges: Spoon Rest CUES Fan Song (search→track "
            "handoff), and Fan Song ASSOCIATED_WITH SA-2 (radar-weapon "
            "pairing). "
            "(b) fire-control radar → missile. Emit ASSOCIATED_WITH when the "
            "text explicitly pairs them. "
            "(c) search radar → missile (skipping a fire-control radar). "
            "Emit CUES when text emphasizes target handoff. "
            "DO NOT invent edges between systems the document does not "
            "jointly describe. Empty list is only correct when upstream refs "
            "share no narrative context in the document."
        ),
        examples=[
            [
                {"rel_type": "CUES",
                 "from_ref_id": "RADAR_SYSTEM:Spoon Rest",
                 "to_ref_id": "RADAR_SYSTEM:Fan Song"},
                {"rel_type": "ASSOCIATED_WITH",
                 "from_ref_id": "RADAR_SYSTEM:Fan Song",
                 "to_ref_id": "MISSILE_SYSTEM:S-75"},
            ],
        ],
    )

    @model_validator(mode="after")
    def _drop_incomplete_relationships(self) -> "SystemLinksPass":
        """Drop rows missing required fields or self-loops, then dedup on
        canonical-direction (from, to). v8a bonus: when the LLM emits both
        ``A → B ASSOCIATED_WITH`` and ``A → B CUES`` (different rel_types
        for the same logical pair), keep only the first — the merge layer
        creates one edge per (source, target, rel_type) triple downstream,
        but exposing the second to the merge layer needlessly increases
        edge volume and confuses downstream callers."""
        cleaned: list[SystemLinkRelationship] = []
        seen_pairs: set[tuple[str, str]] = set()
        for rel in self.relationships:
            if not rel.rel_type or not rel.from_ref_id or not rel.to_ref_id:
                continue
            if rel.from_ref_id == rel.to_ref_id:
                continue
            pair_key = (rel.from_ref_id, rel.to_ref_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            cleaned.append(rel)
        self.relationships = cleaned
        return self
