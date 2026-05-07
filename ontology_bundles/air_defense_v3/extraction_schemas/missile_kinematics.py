"""missile_kinematics extraction pass — engagement envelope.

Spec §4.4. One of 6 sub-passes splitting the legacy missile_domain.
Group fields: system_name, min_intercept_km, max_intercept_km,
min_altitude_km, max_altitude_km, max_launch_angle_deg.

Field descriptions are sanitized at copy time per spec §4.4: numeric
fields reference DELTA_SYSTEM_PROMPT's Unit Policy block instead of
inlining conversion rules.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_kinematics"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class MissileKinematicsRecord(BaseModel):
    """Subset of MissileSystemEntity covering engagement envelope."""

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
            "Canonical designation of the MISSILE — the SHORTEST "
            "canonical token from prose (e.g. '5V55K', '9M82', "
            "'AIM-120', 'PAC-3', 'SM-6'). When the document gives a "
            "formal designation PLUS a common/NATO name PLUS a "
            "block/variant (e.g. 'MIM-104F Patriot Advanced "
            "Capability-3 (PAC-3)'), emit ONLY the primary identifier "
            "('PAC-3' or 'MIM-104F'); DO NOT compress them all into "
            "system_name. CRITICAL: this name MUST match the "
            "system_name used by `missile_identity` for the same "
            "missile, so all field-group passes deduplicate onto one "
            "MISSILE_SYSTEM vertex during merge. Never emit radar, "
            "weapon-system, aircraft, or platform names — those are "
            "filtered deterministically."
        ),
        examples=["5V55K", "AIM-120"],
    )
    min_intercept_km: Optional[float] = Field(
        default=None,
        description=(
            "Minimum intercept range in kilometers. Emit only when the "
            "source states value AND unit. Source labels such as "
            "'Min Range' or 'minimum range' map here when they describe "
            "the missile variant. See Unit Policy in DELTA_SYSTEM_PROMPT "
            "for conversions."
        ),
    )
    max_intercept_km: Optional[float] = Field(
        default=None,
        description=(
            "Maximum intercept range in kilometers. Emit only when the "
            "source states value AND unit. Source labels such as "
            "'Range', 'Max Range', 'maximum range', 'effective range', "
            "or 'engagement range' map here when they describe the "
            "missile variant. See Unit Policy in DELTA_SYSTEM_PROMPT for "
            "conversions."
        ),
    )
    min_altitude_km: Optional[float] = Field(
        default=None,
        description=(
            "Minimum engagement altitude in kilometers. Emit only when "
            "the source states value AND unit. Source labels such as "
            "'Min Altitude' or 'minimum altitude' map here."
        ),
    )
    max_altitude_km: Optional[float] = Field(
        default=None,
        description=(
            "Maximum engagement altitude in kilometers. Emit only when "
            "the source states value AND unit. Source labels such as "
            "'Altitude', 'Max Altitude', 'ceiling', or 'engagement "
            "altitude' map here when they describe the missile variant."
        ),
    )
    max_launch_angle_deg: Optional[float] = Field(
        default=None,
        description=(
            "Maximum launch angle in degrees. Emit only when the source "
            "states value AND unit."
        ),
    )

    _v_system_name           = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_min_intercept_km      = field_validator("min_intercept_km", mode="before")(coerce_optional_float)
    _v_max_intercept_km      = field_validator("max_intercept_km", mode="before")(coerce_optional_float)
    _v_min_altitude_km       = field_validator("min_altitude_km", mode="before")(coerce_optional_float)
    _v_max_altitude_km       = field_validator("max_altitude_km", mode="before")(coerce_optional_float)
    _v_max_launch_angle_deg  = field_validator("max_launch_angle_deg", mode="before")(coerce_optional_float)


class MissileKinematicsPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissileKinematicsRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with engagement-envelope values "
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
