"""missile_speed_timing extraction pass — speed + flight-time + burn-time.

Spec §4.4. One of 6 sub-passes splitting the legacy missile_domain.
Group fields: system_name, average_speed_mps, max_speed_mps,
max_flyout_time_sec, flight_time_sec, coast_time_sec,
intra_salvo_time_sec, total_burn_time_sec, ejector_time_sec.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_speed_timing"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class MissileSpeedTimingRecord(BaseModel):
    """Subset of MissileSystemEntity covering speed + flight/burn timing."""

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
    average_speed_mps: Optional[float] = Field(
        default=None,
        description=(
            "Average flight speed in meters per second. Emit only when "
            "the source states value AND unit. Source labels such as "
            "'Average Speed' or 'average velocity' map here. See Unit "
            "Policy in DELTA_SYSTEM_PROMPT for conversions."
        ),
    )
    max_speed_mps: Optional[float] = Field(
        default=None,
        description=(
            "Maximum flight speed in meters per second. Source labels "
            "such as 'Speed', 'Velocity', 'Maximum Speed', or 'Maximum "
            "Velocity' map here only when the source implies a maximum. "
            "Emit only when the source states value AND unit."
        ),
    )
    max_flyout_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Maximum flyout time in seconds. Emit only when the source "
            "states value AND unit. Source labels such as 'maximum "
            "flyout time', 'max time of flight', or 'maximum flight "
            "time' map here."
        ),
    )
    flight_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Total flight time in seconds. Source labels such as 'Time "
            "of Flight', 'Flight Time', or 'Flyout Time' map here when "
            "they do not explicitly describe a maximum. Emit only when "
            "the source states value AND unit."
        ),
    )
    coast_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Coast time (post-burn) in seconds. Emit only when the "
            "source states value AND unit."
        ),
    )
    intra_salvo_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Intra-salvo time (between launches in a salvo) in seconds. "
            "Emit only when the source states value AND unit."
        ),
    )
    total_burn_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Total motor burn time in seconds. Source labels such as "
            "'Burn Time' or 'Motor Burn Time' map here only when they "
            "describe the full missile motor burn. If the source is "
            "under a booster/ejector/sustainer stage label, use the "
            "stage-specific field in missile_propulsion instead. Emit "
            "only when the source states value AND unit."
        ),
    )
    ejector_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Ejector burn duration in seconds. Emit only when the source "
            "states value AND unit."
        ),
    )

    _v_system_name           = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_average_speed_mps     = field_validator("average_speed_mps", mode="before")(coerce_optional_float)
    _v_max_speed_mps         = field_validator("max_speed_mps", mode="before")(coerce_optional_float)
    _v_max_flyout_time_sec   = field_validator("max_flyout_time_sec", mode="before")(coerce_optional_float)
    _v_flight_time_sec       = field_validator("flight_time_sec", mode="before")(coerce_optional_float)
    _v_coast_time_sec        = field_validator("coast_time_sec", mode="before")(coerce_optional_float)
    _v_intra_salvo_time_sec  = field_validator("intra_salvo_time_sec", mode="before")(coerce_optional_float)
    _v_total_burn_time_sec   = field_validator("total_burn_time_sec", mode="before")(coerce_optional_float)
    _v_ejector_time_sec      = field_validator("ejector_time_sec", mode="before")(coerce_optional_float)


class MissileSpeedTimingPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissileSpeedTimingRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with speed + flight/burn timing "
            "values extracted from this batch."
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
