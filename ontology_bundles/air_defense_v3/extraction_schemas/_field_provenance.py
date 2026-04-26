"""Shared FieldProvenanceRow for per-field source snippets emitted by
the LLM. Lives in extraction_schemas because both RadarDomainPass and
MissileDomainPass use it (spec §5.1.1).

Phase 3 — flat-schema profile refactor.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FieldProvenanceRow(BaseModel):
    """One per (entity_index, field_name) pair the LLM annotated.

    The service post-process (docling-graph) resolves
    supporting_snippet → element_uid by substring-matching against the
    chunks fed to the LLM. A row whose snippet doesn't match any chunk
    keeps element_uid=None and emits an ``unverified_source`` log row;
    the snippet still ships to the UI with an "Unverified source"
    badge (spec §5.13).

    All three fields are Optional per the bundle-checker rule that
    extraction models must tolerate partial LLM output. Rows missing
    any of (entity_index, field_name, supporting_snippet) are dropped
    by the post-process with an `unverified_source` log row.
    """
    model_config = ConfigDict(is_entity=False)

    entity_index: Optional[int] = Field(
        default=None,
        description="0-based index into the pass-template's primary entity list "
                    "(e.g. RadarDomainPass.radar_systems).",
        examples=[0, 1, 2],
    )
    field_name: Optional[str] = Field(
        default=None,
        description="Canonical field name on the entity model (e.g. 'gain_dbi').",
        examples=["gain_dbi", "max_speed_mps", "nominal_rf_mhz"],
    )
    supporting_snippet: Optional[str] = Field(
        default=None,
        description="Verbatim quote from the input chunks that established the field's value.",
        examples=["antenna gain measured at 35 dBi"],
    )
