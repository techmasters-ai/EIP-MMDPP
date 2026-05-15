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
    body_length_m: Optional[float] = Field(
        default=None,
        description=(
            "Missile body length in meters. Source labels such as "
            "'Length', 'Overall Length', or 'Missile Length' map here "
            "when they describe the missile variant. See Unit Policy in "
            "DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    body_diameter_m: Optional[float] = Field(
        default=None,
        description=(
            "Missile body diameter in meters. Source labels such as "
            "'Diameter', 'Body Diameter', 'Calibre', or 'Caliber' map "
            "here when they describe the missile variant. "
        ),
    )
    total_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Total whole-missile mass at launch in kilograms. Source "
            "labels such as 'Weight', 'Mass', 'Launch Weight', or "
            "'Launch Mass' map here only when they describe the whole "
            "missile. Do not use booster/sustainer stage weights. See Unit "
            "Policy in DELTA_SYSTEM_PROMPT for conversions."
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
