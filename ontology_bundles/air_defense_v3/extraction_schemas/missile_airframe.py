"""missile_airframe extraction pass — body geometry + mass.

Spec §4.4. One of 6 sub-passes splitting the legacy missile_domain.
Group fields: system_name, body_length_m, body_diameter_m, total_mass_kg.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_airframe"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class MissileAirframeRecord(BaseModel):
    """Subset of MissileSystemEntity covering body geometry + mass."""

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
    body_length_m: Optional[float] = Field(
        default=None,
        description=(
            "Missile body length in meters. Emit only when the source "
            "states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    body_diameter_m: Optional[float] = Field(
        default=None,
        description=(
            "Missile body diameter in meters. Emit only when the source "
            "states value AND unit."
        ),
    )
    total_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Total missile mass at launch in kilograms. Emit only when "
            "the source states value AND unit. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )

    _v_system_name      = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_body_length_m    = field_validator("body_length_m", mode="before")(coerce_optional_float)
    _v_body_diameter_m  = field_validator("body_diameter_m", mode="before")(coerce_optional_float)
    _v_total_mass_kg    = field_validator("total_mass_kg", mode="before")(coerce_optional_float)


class MissileAirframePass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissileAirframeRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with body geometry + mass values "
            "extracted from this batch."
        ),
        examples=[["5V55K"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_missile_root_sanitizer(
            list_field="missile_systems",
            optional_text_fields=set(),
        )
    )
