"""Reference pass: document structure.

Extracts document-level anchors. No LLM-extracted relationships. Post-merge,
MENTIONED_IN edges are produced by derive_rules.derive_structural_edges;
HAS_PROVENANCE edges are auto-created by graph_store.upsert_nodes_batch_sync
in phase 2 (see §3.8 and §5.6 Phase 2) — NOT by derive_rules.
"""
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ..validators import (
    coerce_optional_int,
    coerce_optional_confidence,
)


class SectionEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [heading, page_start]
    heading: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    confidence: Optional[float] = None

    _v_page_start = field_validator("page_start", mode="before")(coerce_optional_int)
    _v_page_end = field_validator("page_end", mode="before")(coerce_optional_int)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class FigureEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [figure_id, page]
    figure_id: Optional[str] = None
    page: Optional[int] = None
    caption: Optional[str] = None
    figure_type: Optional[str] = None
    confidence: Optional[float] = None

    _v_page = field_validator("page", mode="before")(coerce_optional_int)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class TableEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [table_id, page]
    table_id: Optional[str] = None
    page: Optional[int] = None
    caption: Optional[str] = None
    confidence: Optional[float] = None

    _v_page = field_validator("page", mode="before")(coerce_optional_int)
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class AssertionEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # identity_fields: [assertion_text]
    assertion_text: Optional[str] = None
    extraction_method: Optional[str] = None
    review_status: Optional[str] = None
    confidence: Optional[float] = None

    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class ReferencePass(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sections: list[SectionEntity] = Field(default_factory=list)
    figures: list[FigureEntity] = Field(default_factory=list)
    tables: list[TableEntity] = Field(default_factory=list)
    assertions: list[AssertionEntity] = Field(default_factory=list)
