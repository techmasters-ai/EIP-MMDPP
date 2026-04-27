"""missile_guidance extraction pass — guidance + seeker + photo flag.

Spec §4.4. One of 6 sub-passes splitting the legacy missile_domain.
Group fields: system_name, guidance_type, seeker_type, missile_photo.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_text
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_guidance"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class MissileGuidanceRecord(BaseModel):
    """Subset of MissileSystemEntity covering guidance + seeker."""

    model_config = ConfigDict(
        extra="ignore",
        ontology_name="MISSILE_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the MISSILE. Accept proper-noun "
            "missile names. Never emit radar, weapon-system, aircraft, "
            "or platform names — those are filtered deterministically."
        ),
        examples=["5V55K", "AIM-120"],
    )
    guidance_type: Optional[str] = Field(
        default=None,
        description=(
            "Guidance scheme. Free-text; emit verbatim from the source."
        ),
    )
    seeker_type: Optional[str] = Field(
        default=None,
        description=(
            "Seeker type (e.g. semi-active radar homing, infrared). "
            "Free-text; emit verbatim from the source."
        ),
    )
    # Optional[bool] uses Pydantic's native bool parsing — same pattern
    # as radar_antenna's antenna_photo / spoiled fields.
    missile_photo: Optional[bool] = Field(
        default=None,
        description=(
            "Whether a missile photograph is included in the record. "
            "Use null when not stated."
        ),
    )

    _v_system_name    = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_guidance_type  = field_validator("guidance_type", mode="before")(coerce_optional_text)
    _v_seeker_type    = field_validator("seeker_type", mode="before")(coerce_optional_text)
    # missile_photo: no validator — Pydantic native bool parsing.


class MissileGuidancePass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissileGuidanceRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with guidance + seeker values "
            "extracted from this batch."
        ),
        examples=[["5V55K"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_missile_root_sanitizer(
            list_field="missile_systems",
            optional_text_fields={"guidance_type", "seeker_type"},
        )
    )
