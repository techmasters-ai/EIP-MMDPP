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
    """Missile performance timing: average speed, maximum speed, Mach, flyout time, flight time, coast time, salvo interval, and burn time."""

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
            "Average flight speed in meters per second. Relevant source "
            "labels include 'Average Speed', 'average velocity', "
            "'mean speed', 'cruise speed', "
            "'average Mach', or 'velocity average'. Use average or "
            "nominal flight speed, not peak speed."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "average speed", "average velocity", "mean speed",
                    "cruise speed", "typical speed", "average Mach",
                    "velocity average", "nominal speed",
                ],
                "negative_terms": [
                    "maximum speed", "peak speed", "top speed",
                    "boost speed", "terminal velocity",
                ],
                "evidence_patterns": [
                    r"re:\baverage\s+(?:flight\s+)?speed\b",
                    r"re:\bmean\s+speed\b",
                    "cruise speed", "average Mach",
                ],
                "likely_sections": [
                    "specifications", "performance", "speed", "missile",
                    "flight performance",
                ],
                "units": ["m/s", "mps", "km/s", "ft/s", "Mach"],
            }
        },
    )
    max_speed_mps: Optional[float] = Field(
        default=None,
        description=(
            "Maximum flight speed in meters per second. Relevant source "
            "labels include 'Speed', 'Velocity', 'Maximum Speed', "
            "'Max Speed', 'top speed', 'peak speed', 'Maximum "
            "Velocity', 'Mach', or 'maximum Mach'. Use only when the "
            "source states or clearly implies a maximum or peak value."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "speed", "velocity", "maximum speed", "max speed",
                    "top speed", "peak speed", "maximum velocity",
                    "Mach", "maximum Mach", "terminal speed",
                ],
                "negative_terms": [
                    "average speed", "cruise speed", "mean speed",
                    "radar speed", "target speed",
                ],
                "evidence_patterns": [
                    r"re:\bmax(?:imum)?\s+(?:flight\s+)?speed\b",
                    r"re:\bMach\s+\d",
                    r"re:\bpeak\s+speed\b",
                    "Max Speed",
                ],
                "likely_sections": [
                    "specifications", "performance", "speed", "missile",
                    "flight performance",
                ],
                "units": ["m/s", "mps", "km/s", "ft/s", "Mach"],
            }
        },
    )
    max_flyout_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Maximum flyout time in seconds. Relevant source labels include "
            "'maximum flyout time', 'max flyout time', 'max time of "
            "flight', 'maximum flight time', 'flight duration limit', "
            "'maximum intercept time', or 'self-destruct time'."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "maximum flyout time", "max flyout time",
                    "max time of flight", "maximum flight time",
                    "flight duration limit", "maximum intercept time",
                    "self-destruct time", "time of flight limit",
                ],
                "negative_terms": [
                    "flight time", "coast time", "burn time",
                    "nominal flight time", "time to intercept",
                ],
                "evidence_patterns": [
                    r"re:\bmax(?:imum)?\s+flyout\s+time\b",
                    r"re:\bself-destruct\s+time\b",
                    "max flyout", "flight duration limit",
                ],
                "likely_sections": [
                    "specifications", "performance", "timing", "missile",
                    "flight performance",
                ],
                "units": ["s", "sec", "seconds"],
            }
        },
    )
    flight_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Total or nominal flight time in seconds. Relevant source "
            "labels include 'Time of Flight', 'Flight Time', 'Flyout "
            "Time', 'time to intercept', 'intercept time', or "
            "'flight duration'. Use this only when the source does not "
            "explicitly describe a maximum."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "time of flight", "flight time", "flyout time",
                    "time to intercept", "intercept time",
                    "flight duration", "time of flight (TOF)",
                ],
                "negative_terms": [
                    "maximum flyout time", "coast time", "burn time",
                    "maximum flight time", "self-destruct time",
                ],
                "evidence_patterns": [
                    r"re:\btime\s+of\s+flight\b",
                    r"re:\bflight\s+time\b",
                    r"re:\btime\s+to\s+intercept\b",
                    "flyout time",
                ],
                "likely_sections": [
                    "specifications", "performance", "timing", "missile",
                    "flight performance",
                ],
                "units": ["s", "sec", "seconds"],
            }
        },
    )
    coast_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Coast time, post-burn time, or unpowered flight duration in "
            "seconds. Relevant source labels include 'coast time', "
            "'coasting time', 'post-burn coast', 'unpowered flight', "
            "or 'ballistic coast'."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "coast time", "coasting time", "post-burn coast",
                    "unpowered flight", "ballistic coast",
                    "glide time", "post-burnout time",
                ],
                "negative_terms": [
                    "burn time", "motor burn time", "boost time",
                    "sustainer time", "powered flight time",
                ],
                "evidence_patterns": [
                    r"re:\bcoast(?:ing)?\s+time\b",
                    r"re:\bpost-burn(?:out)?\s+(?:coast|time)\b",
                    "unpowered flight", "ballistic coast",
                ],
                "likely_sections": [
                    "specifications", "performance", "propulsion", "missile",
                    "flight performance", "timing",
                ],
                "units": ["s", "sec", "seconds"],
            }
        },
    )
    intra_salvo_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Intra-salvo time between launches in seconds. Relevant source "
            "labels include 'intra-salvo time', 'salvo interval', "
            "'launch interval', 'ripple interval', 'ripple-fire time', "
            "'time between launches', or 'reload between shots'."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "intra-salvo time", "salvo interval", "launch interval",
                    "ripple interval", "ripple-fire time",
                    "time between launches", "reload between shots",
                    "inter-launch interval",
                ],
                "negative_terms": [
                    "flight time", "burn time", "reload time",
                    "reaction time", "engagement time",
                ],
                "evidence_patterns": [
                    r"re:\bsalvo\s+interval\b",
                    r"re:\blaunch\s+interval\b",
                    r"re:\bripple(?:-fire)?\s+(?:interval|time)\b",
                    "intra-salvo",
                ],
                "likely_sections": [
                    "specifications", "performance", "salvo", "missile",
                    "launcher", "fire rate",
                ],
                "units": ["s", "sec", "seconds"],
            }
        },
    )
    total_burn_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Total motor burn time in seconds. Relevant source labels "
            "include 'Burn Time', 'Motor Burn Time', 'total burn time', "
            "'powered flight time', 'propulsion time', or 'rocket "
            "motor duration'. Use only full missile motor burn; if the "
            "source is under a booster, ejector, or sustainer label, use "
            "the stage-specific field in missile_propulsion instead."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "burn time", "motor burn time", "total burn time",
                    "powered flight time", "propulsion time",
                    "rocket motor duration", "total motor burn",
                ],
                "negative_terms": [
                    "booster burn time", "sustainer burn time",
                    "ejector time", "coast time", "flight time",
                ],
                "evidence_patterns": [
                    r"re:\b(?:total\s+)?burn\s+time\b",
                    r"re:\bmotor\s+burn\s+time\b",
                    r"re:\bpowered\s+flight\s+time\b",
                    "rocket motor duration",
                ],
                "likely_sections": [
                    "specifications", "performance", "propulsion", "missile",
                    "timing", "motor",
                ],
                "units": ["s", "sec", "seconds"],
            }
        },
    )
    ejector_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Ejector burn duration in seconds. Relevant source labels "
            "include 'ejector time', 'ejector burn time', 'ejection "
            "time', 'cold launch ejector', 'launch eject motor', "
            "'gas generator time', or 'soft-launch impulse duration'."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "ejector time", "ejector burn time", "ejection time",
                    "cold launch ejector time", "launch eject motor time",
                    "gas generator time", "soft-launch impulse duration",
                    "cold-launch time",
                ],
                "negative_terms": [
                    "booster burn time", "sustainer burn time",
                    "total burn time", "coast time", "flight time",
                ],
                "evidence_patterns": [
                    r"re:\bejector\s+(?:burn\s+)?time\b",
                    r"re:\bejection\s+time\b",
                    r"re:\bgas\s+generator\s+time\b",
                    "soft-launch impulse", "cold launch ejector",
                ],
                "likely_sections": [
                    "specifications", "performance", "propulsion", "missile",
                    "timing", "ejector", "launch",
                ],
                "units": ["s", "sec", "seconds"],
            }
        },
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
