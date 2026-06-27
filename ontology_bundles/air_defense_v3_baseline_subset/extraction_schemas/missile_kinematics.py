"""missile_kinematics extraction pass — engagement envelope.

Spec §4.4. One of 6 sub-passes splitting the legacy missile_domain.
Group fields: system_name, min_intercept_km, max_intercept_km,
min_altitude_km, max_altitude_km, max_launch_angle_deg.

Field descriptions are written as dual-use retrieval and prompt text:
use document-facing anchor terms while preserving extraction constraints.
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
    """Surface-to-air missile engagement envelope: range, altitude, ceiling, floor, and launch-angle limits."""

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
            "Minimum intercept range in kilometers. Relevant source labels include "
            "'minimum effective range', 'minimum range', 'Min Range', "
            "'minimum intercept range', 'inner range', 'range floor', "
            "'near limit', or 'kill zone minimum'. Use missile engagement "
            "envelope limits, not radar range or launcher spacing."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "minimum range", "minimum intercept range",
                    "minimum effective range", "Min Range", "near boundary",
                    "inner range", "range floor", "near limit",
                    "kill zone minimum", "minimum engagement range",
                ],
                "negative_terms": [
                    "maximum range", "radar range", "radar coverage",
                    "launcher spacing", "detection range",
                ],
                "evidence_patterns": [
                    r"re:\bmin(?:imum)?\s+(?:intercept\s+)?range",
                    r"re:\brange\s+floor\b",
                    "Min Range", "near boundary",
                ],
                "likely_sections": [
                    "specifications", "engagement envelope", "performance",
                    "missile", "range", "kill zone",
                ],
                "units": ["km", "m", "nmi", "nm", "miles"],
            }
        },
    )
    max_intercept_km: Optional[float] = Field(
        default=None,
        description=(
            "Maximum intercept range in kilometers. Relevant source labels include "
            "'maximum effective range', 'maximum range', 'Max Range', "
            "'Range', 'effective range', 'engagement range', "
            "'intercept range', 'range against targets', 'range limit', "
            "or 'kill zone range'. Use missile engagement envelope limits, "
            "not radar range or launcher spacing."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "maximum range", "maximum intercept range",
                    "maximum effective range", "Max Range", "engagement range",
                    "effective range", "intercept range", "range against targets",
                    "range limit", "kill zone range", "maximum engagement range",
                    "slant range",
                ],
                "negative_terms": [
                    "minimum range", "radar range", "radar coverage",
                    "launcher spacing", "detection range",
                ],
                "evidence_patterns": [
                    r"re:\bmax(?:imum)?\s+(?:intercept\s+|effective\s+)?range",
                    r"re:\brange\s+limit\b",
                    "Max Range", "engagement range",
                ],
                "likely_sections": [
                    "specifications", "engagement envelope", "performance",
                    "missile", "range", "kill zone",
                ],
                "units": ["km", "m", "nmi", "nm", "miles"],
            }
        },
    )
    min_altitude_km: Optional[float] = Field(
        default=None,
        description=(
            "Minimum engagement altitude in kilometers. Relevant source labels include "
            "'minimum effective altitude', 'minimum altitude', 'Min Altitude', "
            "'Min Alt', 'altitude floor', 'lower altitude limit', "
            "'minimum intercept altitude', or 'kill zone floor'."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "minimum altitude", "minimum engagement altitude",
                    "minimum effective altitude", "Min Altitude", "Min Alt",
                    "altitude floor", "lower altitude limit",
                    "minimum intercept altitude", "kill zone floor",
                    "low altitude limit",
                ],
                "negative_terms": [
                    "maximum altitude", "ceiling", "radar altitude",
                    "aircraft altitude", "terrain clearance",
                ],
                "evidence_patterns": [
                    r"re:\bmin(?:imum)?\s+(?:engagement\s+|intercept\s+)?altitude",
                    r"re:\baltitude\s+floor\b",
                    "Min Alt", "Min Altitude",
                ],
                "likely_sections": [
                    "specifications", "engagement envelope", "performance",
                    "missile", "altitude", "kill zone",
                ],
                "units": ["km", "m", "ft", "kft"],
            }
        },
    )
    max_altitude_km: Optional[float] = Field(
        default=None,
        description=(
            "Maximum engagement altitude in kilometers. Relevant source labels include "
            "'maximum effective altitude', 'maximum altitude', 'Max Altitude', "
            "'Max Alt', 'Altitude', 'altitude ceiling', 'ceiling', "
            "'intercept ceiling', 'engagement altitude', 'launch ceiling', "
            "or 'kill zone ceiling'."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "maximum altitude", "maximum engagement altitude",
                    "maximum effective altitude", "Max Altitude", "Max Alt",
                    "altitude ceiling", "ceiling", "intercept ceiling",
                    "engagement altitude", "kill zone ceiling",
                    "high altitude limit",
                ],
                "negative_terms": [
                    "minimum altitude", "altitude floor", "radar altitude",
                    "aircraft service ceiling", "terrain clearance",
                ],
                "evidence_patterns": [
                    r"re:\bmax(?:imum)?\s+(?:engagement\s+|intercept\s+)?altitude",
                    r"re:\baltitude\s+ceiling\b",
                    "Max Alt", "Max Altitude",
                ],
                "likely_sections": [
                    "specifications", "engagement envelope", "performance",
                    "missile", "altitude", "kill zone",
                ],
                "units": ["km", "m", "ft", "kft"],
            }
        },
    )
    max_launch_angle_deg: Optional[float] = Field(
        default=None,
        description=(
            "Maximum launch angle in degrees. Relevant source labels include "
            "'maximum launch angle', 'launch angle', 'elevation launch angle', "
            "'off-boresight launch angle', 'firing angle', or "
            "'canister elevation angle'."
        ),
        json_schema_extra={
            "retrieval": {
                "aliases": [
                    "maximum launch angle", "launch angle",
                    "elevation launch angle", "off-boresight launch angle",
                    "firing angle", "canister elevation angle",
                    "maximum elevation angle", "launch elevation",
                ],
                "negative_terms": [
                    "radar elevation", "antenna elevation",
                    "depression angle", "dive angle", "impact angle",
                ],
                "evidence_patterns": [
                    r"re:\blaunch\s+angle\b",
                    r"re:\belevation\s+(?:launch\s+)?angle\b",
                    "canister elevation angle", "off-boresight",
                ],
                "likely_sections": [
                    "specifications", "engagement envelope", "performance",
                    "missile", "launcher", "launch parameters",
                ],
                "units": ["deg", "°", "degrees", "mils"],
            }
        },
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
