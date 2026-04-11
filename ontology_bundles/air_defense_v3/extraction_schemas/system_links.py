"""System links pass: cross-domain relationships only."""
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ..validators import (
    coerce_optional_confidence,
    normalize_enum,
)


class SystemLinkRelationship(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rel_type: Optional[str] = None
    from_ref_id: Optional[str] = None
    to_ref_id: Optional[str] = None
    confidence: Optional[float] = None

    _v_rel_type = field_validator("rel_type", mode="before")(
        normalize_enum({"ASSOCIATED_WITH", "CUES"})
    )
    _v_confidence = field_validator("confidence", mode="before")(coerce_optional_confidence)


class SystemLinksPass(BaseModel):
    """Relationships-only pass. Consumes upstream_entities from the request body.
    Has NO entity fields — enforced by the 'input_mode == document_plus_entity_refs
    implies no entity-collection fields' sub-check in the manifest self-consistency
    section of §2 checker rules."""
    model_config = ConfigDict(extra="ignore")

    relationships: list[SystemLinkRelationship] = Field(default_factory=list)
