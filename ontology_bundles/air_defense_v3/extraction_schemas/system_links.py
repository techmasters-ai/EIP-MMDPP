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

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
        description="Cross-pass relationship type; must be a valid RelationshipType value",
        examples=["ASSOCIATED_WITH", "CUES"],
    )
    from_ref_id: Optional[str] = Field(
        default=None,
        description="Upstream ref id of the edge source (e.g. 'E001')",
        examples=["E001"],
    )
    to_ref_id: Optional[str] = Field(
        default=None,
        description="Upstream ref id of the edge target (e.g. 'E002')",
        examples=["E002"],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this relationship, 0–1.",
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


class SystemLinksPass(BaseModel):
    """Relationships-only pass root. Has NO entity fields — enforced by
    the 'input_mode == document_plus_entity_refs implies no entity-
    collection fields' sub-check in the manifest self-consistency section
    of §2 checker rules. Decision 4: ``relationships`` uses a plain
    ``Field`` (not ``edge()``) because its items are DTOs, not entities.

    is_entity=True per docling-graph-docs.md §Template Basics → Root
    Document Model (line 18859). See radar_domain.py for full rationale.
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
        description="Cross-pass relationship DTOs emitted by this pass; each carries from_ref_id/to_ref_id that the merge layer resolves against PassResult.upstream_refs.",
        examples=[[{"rel_type": "CUES", "from_ref_id": "E001", "to_ref_id": "E002"}], [{"rel_type": "ASSOCIATED_WITH", "from_ref_id": "E003", "to_ref_id": "E004"}]],
    )
