"""Reference pass: document structure.

Narrow extraction view of the canonical document-anchor entities. Parallel
classes with matching ``ontology_name`` — NOT subclasses of canonical — so
pass-specific narrowing stays local while the merge layer rejoins views
by ``ontology_name`` + ``LogicalIdentity``.

Key entities (ontology_name):
- ``SECTION`` — section or heading within the document
- ``FIGURE`` — figure / diagram / image
- ``TABLE`` — data table

Key relationships: None emitted by the LLM. ``MENTIONED_IN`` edges are
derived post-merge by ``derive_rules.derive_structural_edges`` (spec §3.8);
``HAS_PROVENANCE`` edges are auto-created by
``graph_store.upsert_nodes_batch_sync`` in phase 2 (spec §5.6 Phase 2).
Plan v32 Task 45 (Phase 5 docs-compliance rewrite).
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import (
    coerce_optional_int,
    coerce_optional_confidence,
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
    this function must be defined identically in every template. The
    reference pass has no inter-entity relationships so the helper is
    unused here — docs adherence is the reason it's present.
    """
    existing_extra = field_kwargs.pop("json_schema_extra", None) or {}
    existing_extra["edge_label"] = label
    if description is not None:
        field_kwargs["description"] = description
    if examples is not None:
        field_kwargs["examples"] = examples
    return Field(json_schema_extra=existing_extra, **field_kwargs)


def _dedupe_entities_by_identity(cls, values):
    """Merge-preserving dedup for pass-root entity lists (plan Task 9d).

    For each entity list whose inner class carries non-empty
    ``graph_id_fields``, collapse duplicates by identity tuple and union
    non-null non-identity scalar fields (first-non-null wins). Guards
    against the LLM emitting the same logical entity twice with
    complementary fields — merging here preserves both sides instead of
    letting dict-insertion order overwrite data.
    """
    if not isinstance(values, dict):
        return values
    for fname, finfo in cls.model_fields.items():
        raw = values.get(fname)
        if not isinstance(raw, list) or not raw:
            continue
        inner_types = getattr(finfo, "annotation", None)
        # Extract the inner BaseModel type from List[X] / Optional[List[X]].
        inner_cls = None
        from typing import get_args as _ga, get_origin as _go
        def _find_basemodel(ann):
            if ann is None:
                return None
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                return ann
            for a in _ga(ann):
                r = _find_basemodel(a)
                if r is not None:
                    return r
            return None
        inner_cls = _find_basemodel(inner_types)
        if inner_cls is None:
            continue
        id_fields = list((inner_cls.model_config or {}).get("graph_id_fields", []) or [])
        if not id_fields:
            continue
        accum: dict[tuple, dict] = {}
        order: list[tuple] = []
        for item in raw:
            if not isinstance(item, dict):
                # Already-parsed model instance; skip mutation (merger layer handles it).
                continue
            key = tuple(item.get(k) for k in id_fields)
            if key not in accum:
                accum[key] = dict(item)
                order.append(key)
            else:
                merged = accum[key]
                for k, v in item.items():
                    if k in id_fields:
                        continue
                    if v is None:
                        continue
                    if k not in merged or merged[k] is None:
                        merged[k] = v
        if len(accum) != len(raw):
            values[fname] = [accum[k] for k in order]
    return values


# ----------------------------------------------------------------------
# Entities (components would come first per docs; this pass has none)
# ----------------------------------------------------------------------

class SectionEntity(BaseModel):
    """Document Section — narrow extraction view of canonical SECTION.

    Identity: ``heading`` + ``page_start`` (document-scoped).
    """
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="SECTION",
        graph_id_fields=["heading", "page_start"],
        identity_scope="document",
        is_entity=True,
    )

    heading: str = Field(
        ...,
        description="Section heading or title text",
        examples=["Chapter 3: Maintenance Procedures", "Chapter 3: Maintenance Procedures"],
    )
    page_start: int = Field(
        ...,
        description="Starting page number of the section",
        examples=[42, 42],
    )
    page_end: Optional[int] = Field(
        default=None,
        description="Ending page number of the section",
        examples=[67],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_page_start = field_validator("page_start", mode="before")(coerce_optional_int)
    _v_page_end = field_validator("page_end", mode="before")(coerce_optional_int)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class FigureEntity(BaseModel):
    """Figure — narrow extraction view of canonical FIGURE.

    Identity: ``figure_id`` + ``page`` (document-scoped).
    """
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="FIGURE",
        graph_id_fields=["figure_id", "page"],
        identity_scope="document",
        is_entity=True,
    )

    figure_id: str = Field(
        ...,
        description="Figure number or identifier within the document",
        examples=["Figure 3-12", "Figure 3-12"],
    )
    page: int = Field(
        ...,
        description="Page number where the figure appears",
        examples=[55, 55],
    )
    caption: Optional[str] = Field(
        default=None,
        description="Figure caption text",
        examples=["Antenna Feed Assembly Exploded View"],
    )
    figure_type: Optional[str] = Field(
        default=None,
        description="Category of the figure",
        json_schema_extra={
            "enum": ["BLOCK_DIAGRAM", "SCHEMATIC", "PHOTO", "SPECTRUM_PLOT",
                     "WIRING_DIAGRAM", "FLOWCHART"],
        },
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_page = field_validator("page", mode="before")(coerce_optional_int)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class TableEntity(BaseModel):
    """Table — narrow extraction view of canonical TABLE.

    Identity: ``table_id`` + ``page`` (document-scoped).
    """
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="TABLE",
        graph_id_fields=["table_id", "page"],
        identity_scope="document",
        is_entity=True,
    )

    table_id: str = Field(
        ...,
        description="Table number or identifier within the document",
        examples=["Table 4-1", "Table 4-1"],
    )
    page: int = Field(
        ...,
        description="Page number where the table appears",
        examples=[78, 78],
    )
    caption: Optional[str] = Field(
        default=None,
        description="Table caption or title",
        examples=["Radar System Frequency Parameters"],
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0–1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_page = field_validator("page", mode="before")(coerce_optional_int)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


# ----------------------------------------------------------------------
# Pass-root container — not an ontology entity (Decision 4a).
# ----------------------------------------------------------------------

class ReferencePass(BaseModel):
    """Reference pass root. ``is_entity=False`` by Decision 4a — pass
    roots are containers, not entities."""
    model_config = ConfigDict(extra="ignore", is_entity=False)

    sections: List[SectionEntity] = Field(default_factory=list)
    figures: List[FigureEntity] = Field(default_factory=list)
    tables: List[TableEntity] = Field(default_factory=list)

    _dedupe_root_entities = model_validator(mode="before")(_dedupe_entities_by_identity)
