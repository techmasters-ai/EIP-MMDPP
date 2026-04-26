"""Shared FieldProvenanceRow for per-field source snippets emitted by
the LLM. Lives in extraction_schemas because both RadarDomainPass and
MissileDomainPass use it (spec §5.1.1).

Phase 3 — flat-schema profile refactor.
"""
from pydantic import BaseModel, Field


class FieldProvenanceRow(BaseModel):
    """One per (entity_index, field_name) pair the LLM annotated.

    The service post-process (docling-graph) resolves
    supporting_snippet → element_uid by substring-matching against the
    chunks fed to the LLM. A row whose snippet doesn't match any chunk
    keeps element_uid=None and emits an ``unverified_source`` log row;
    the snippet still ships to the UI with an "Unverified source"
    badge (spec §5.13)."""
    entity_index: int = Field(
        ...,
        description="0-based index into the pass-template's primary entity list "
                    "(e.g. RadarDomainPass.radar_systems).",
    )
    field_name: str = Field(
        ...,
        description="Canonical field name on the entity model (e.g. 'gain_dbi').",
    )
    supporting_snippet: str = Field(
        ...,
        description="Verbatim quote from the input chunks that established the field's value.",
    )
