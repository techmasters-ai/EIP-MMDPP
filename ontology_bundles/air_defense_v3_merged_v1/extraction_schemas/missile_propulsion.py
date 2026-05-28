"""missile_propulsion extraction pass — staged motor parameters.

Spec §4.4. One of 6 sub-passes splitting the legacy missile_domain.
Group fields: system_name, ejector_thrust, ejector_mass_kg,
booster_time_sec, booster_thrust, booster_mass_kg, sustain_time_sec,
sustain_thrust, sustain_mass_kg.

Note: ejector_thrust, booster_thrust, sustain_thrust are typed
Optional[str] on the canonical MissileSystemEntity. Promoting them to
numerics (kN, lbf) is schema-correction work tracked separately, NOT
part of this field-group split.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import coerce_optional_float, coerce_optional_text
from ._field_groups import MISSILE_FIELD_GROUPS
from ._missile_shared import edge, make_missile_root_sanitizer, validate_missile_system_name

_GROUP_NAME = "missile_propulsion"
_FIELDS = MISSILE_FIELD_GROUPS[_GROUP_NAME]   # implicit assertion the group exists


class MissilePropulsionRecord(BaseModel):
    """Missile propulsion stages: ejector, booster, and sustainer thrust, mass, burn time, impulse, and stage separation parameters."""

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
    ejector_thrust: Optional[str] = Field(
        default=None,
        description=(
            "Ejector-stage thrust or impulse description for cold-launch "
            "or soft-launch before the main motor ignites. Relevant source "
            "labels include 'ejector thrust', 'ejection impulse', "
            "'launch ejector', 'gas generator', 'cold-gas ejector', "
            "'soft launch', or 'initial impulse'. Free-text; preserve "
            "source units such as kN, lbf, N-s, or kg-s and emit verbatim "
            "when stated explicitly."
        ),
    )
    ejector_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Ejector-stage mass in kilograms. Relevant source labels "
            "include 'ejector mass', 'ejector weight', 'launch ejector "
            "mass', 'gas generator mass', 'cold-launch motor mass', or "
            "'soft-launch charge weight'. Use ejector-stage mass only, "
            "not total missile mass."
        ),
    )
    booster_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Booster-stage burn time in seconds. Relevant source labels "
            "include 'booster burn time', 'boost time', 'first-stage "
            "burn', '1st Stage Burn Time', 'Booster Time', 'Burn "
            "Time', or 'Time' under a booster or first-stage section."
        ),
    )
    booster_thrust: Optional[str] = Field(
        default=None,
        description=(
            "Booster-stage thrust description for the high-impulse first "
            "stage that accelerates the missile off the rail or launcher. "
            "Relevant source labels include 'booster thrust', 'boost "
            "thrust', 'first-stage thrust', '1st Stage Thrust', "
            "'launch motor thrust', 'peak thrust', or 'initial thrust'. "
            "Free-text; preserve source units such as kN, lbf, or N and "
            "emit verbatim when stated explicitly."
        ),
    )
    booster_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Booster-stage mass in kilograms. Relevant source labels "
            "include 'booster mass', 'booster weight', 'first-stage "
            "mass', '1st Stage Weight', 'boost motor weight', "
            "'stage mass', 'Weight', or 'Mass' under a booster or "
            "first-stage section. Never copy this value to total_mass_kg."
        ),
    )
    sustain_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Sustainer-stage burn time in seconds. Relevant source labels "
            "include 'sustainer burn time', 'sustain time', "
            "'second-stage burn', '2nd Stage Burn Time', 'Sustainer "
            "Time', 'Burn Time', or 'Time' under a sustainer, sustain, "
            "or second-stage section."
        ),
    )
    sustain_thrust: Optional[str] = Field(
        default=None,
        description=(
            "Sustainer-stage thrust description for the second-stage or "
            "cruise motor after booster burnout. Relevant source labels "
            "include 'sustainer thrust', 'sustain thrust', 'second-stage "
            "thrust', '2nd Stage Thrust', 'cruise thrust', 'march "
            "motor thrust', or 'dual-pulse sustainer'. Free-text; "
            "preserve source units such as kN, lbf, or N and emit verbatim "
            "when stated explicitly."
        ),
    )
    sustain_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Sustainer-stage mass in kilograms. Relevant source labels "
            "include 'sustainer mass', 'sustainer weight', "
            "'second-stage mass', '2nd Stage Weight', 'march motor "
            "weight', 'stage mass', 'Weight', or 'Mass' under a "
            "sustainer, sustain, or second-stage section. Never copy this "
            "value to total_mass_kg."
        ),
    )

    _v_system_name        = field_validator("system_name", mode="before")(validate_missile_system_name)
    _v_ejector_thrust     = field_validator("ejector_thrust", mode="before")(coerce_optional_text)
    _v_ejector_mass_kg    = field_validator("ejector_mass_kg", mode="before")(coerce_optional_float)
    _v_booster_time_sec   = field_validator("booster_time_sec", mode="before")(coerce_optional_float)
    _v_booster_thrust     = field_validator("booster_thrust", mode="before")(coerce_optional_text)
    _v_booster_mass_kg    = field_validator("booster_mass_kg", mode="before")(coerce_optional_float)
    _v_sustain_time_sec   = field_validator("sustain_time_sec", mode="before")(coerce_optional_float)
    _v_sustain_thrust     = field_validator("sustain_thrust", mode="before")(coerce_optional_text)
    _v_sustain_mass_kg    = field_validator("sustain_mass_kg", mode="before")(coerce_optional_float)


class MissilePropulsionPass(BaseModel):
    model_config = ConfigDict(extra="ignore", is_entity=True, graph_id_fields=[])

    missile_systems: List[MissilePropulsionRecord] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems with staged-motor parameters "
            "extracted from this batch."
        ),
        examples=[["5V55K"]],
        default_factory=list,
    )

    _sanitize_and_dedupe = model_validator(mode="before")(
        make_missile_root_sanitizer(
            list_field="missile_systems",
            optional_text_fields={"ejector_thrust", "booster_thrust", "sustain_thrust"},
        )
    )
