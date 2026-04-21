"""Missile domain pass — flat, checklist-aligned schema.

Scope aligned to the `(U) MISSILE MDE Checklist.xlsx` deliverable: every
extracted field corresponds 1:1 to a Data Element on that checklist.
Subcomponent classes (GuidanceMethodEntity, SeekerEntity,
PropulsionStackEntity, LauncherSystemEntity, PlatformEntity) are
intentionally removed — the checklist treats all parameters (including
booster / sustainer / ejector propulsion stages) as flat properties of
a single MISSILE_SYSTEM instance. Canonical entities.py retains the
broader taxonomy for other bundles / future expansion.

Key entity (ontology_name):
- ``MISSILE_SYSTEM`` — flattened to carry all checklist fields.

Tier 1 prose-mention rules are preserved on the pass-root list +
identity field.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..validators import (
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_confidence,
    dedupe_entities_by_identity,
)


def edge(
    label: str,
    *,
    description: str | None = None,
    examples: list | None = None,
    **field_kwargs: Any,
) -> Any:
    existing_extra = field_kwargs.pop("json_schema_extra", None) or {}
    existing_extra["edge_label"] = label
    if description is not None:
        field_kwargs["description"] = description
    if examples is not None:
        field_kwargs["examples"] = examples
    return Field(json_schema_extra=existing_extra, **field_kwargs)


# ----------------------------------------------------------------------
# Missile system entity — flat, checklist-driven fields.
# ----------------------------------------------------------------------

class MissileSystemEntity(BaseModel):
    """Missile system — flat extraction view aligned to the MISSILE MDE Checklist.

    Every field below maps to a Data Element on that checklist. Numeric
    fields carry their unit in the field name (e.g. ``max_intercept_km``)
    so the extractor does not need to canonicalize units separately.
    """
    model_config = ConfigDict(
        extra="ignore",
        ontology_name="MISSILE_SYSTEM",
        graph_id_fields=["system_name"],
        identity_scope="global",
        is_entity=True,
    )

    # Identity / Production Information
    system_name: str = Field(
        ...,
        description=(
            "Canonical designation of the missile system. "
            "Accept canonical proper-noun identifiers from prose when "
            "unambiguous (e.g. 'SA-2', 'SA-20', 'PAC-3 MSE', '9M96'). "
            "Reject descriptive phrases ('the missile', 'the interceptor') "
            "and generic noun phrases."
        ),
        examples=["SA-2", "SA-20", "PAC-3 MSE", "SM-6 Block IA", "9M96"],
    )
    nomenclature: Optional[str] = Field(
        default=None,
        description="Military designation or NATO reporting name.",
        examples=["MIM-104F"],
    )

    dieqp: Optional[str] = Field(
        default=None,
        description="Digital Intelligence Equipment Parameters (DIEQP) identifier.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Formal NAME field from the MDE checklist, "
                    "distinct from the common ``system_name``.",
    )
    emitter_function: Optional[str] = Field(
        default=None,
        description="Emitter function from the MDE checklist.",
    )
    system_status: Optional[str] = Field(
        default=None,
        description="Lifecycle status (e.g. OPERATIONAL, DEVELOPMENTAL, RETIRED).",
    )
    asrd: Optional[str] = Field(
        default=None,
        description="ASRD identifier from the MDE checklist.",
    )
    responsible_agency: Optional[str] = Field(
        default=None,
        description="Agency responsible for the MDE record (e.g. 'IWC').",
        examples=["IWC"],
    )
    review_cycle: Optional[str] = Field(
        default=None,
        description="Review cycle cadence for the MDE record.",
    )
    next_review_date: Optional[str] = Field(
        default=None,
        description="Next scheduled review date for the MDE record.",
    )

    # Range / Guidance
    min_intercept_km: Optional[float] = Field(
        default=None,
        description="Minimum intercept range in kilometers.",
    )
    max_intercept_km: Optional[float] = Field(
        default=None,
        description="Maximum intercept range in kilometers.",
    )
    min_altitude_km: Optional[float] = Field(
        default=None,
        description="Minimum engagement altitude in kilometers.",
    )
    max_altitude_km: Optional[float] = Field(
        default=None,
        description="Maximum engagement altitude in kilometers.",
    )
    max_launch_angle_deg: Optional[float] = Field(
        default=None,
        description="Maximum launch angle in degrees.",
    )
    guidance_type: Optional[str] = Field(
        default=None,
        description="Guidance type / system (e.g. SARH, Active Radar, "
                    "Command, IR, Beam-riding, TVM, GPS/INS, Dual-mode).",
        examples=["SARH", "Active Radar", "Command"],
    )
    seeker_type: Optional[str] = Field(
        default=None,
        description="Seeker type (e.g. ACTIVE_RADAR, SEMI_ACTIVE_RADAR, "
                    "IR, DUAL_MODE, ARM, GPS_INS, COMMAND).",
        examples=["ACTIVE_RADAR", "IR"],
    )

    # Physical Characteristics
    missile_photo: Optional[bool] = Field(
        default=None,
        description="Whether a missile photograph is included in the record (Y/N).",
    )
    body_length_m: Optional[float] = Field(
        default=None,
        description="Missile body length in meters.",
    )
    body_diameter_m: Optional[float] = Field(
        default=None,
        description="Missile body diameter in meters.",
    )
    total_mass_kg: Optional[float] = Field(
        default=None,
        description="Total missile mass in kilograms "
                    "(checklist unit column shows 'deg' in row 20 which is a source typo).",
    )

    # Performance Characteristics
    average_speed_mps: Optional[float] = Field(
        default=None,
        description="Average flight speed in meters per second.",
    )
    max_speed_mps: Optional[float] = Field(
        default=None,
        description="Maximum flight speed in meters per second.",
    )
    max_flyout_time_sec: Optional[float] = Field(
        default=None,
        description="Maximum flyout time in seconds.",
    )
    flight_time_sec: Optional[float] = Field(
        default=None,
        description="Typical flight time in seconds.",
    )
    coast_time_sec: Optional[float] = Field(
        default=None,
        description="Coast (unpowered) time in seconds.",
    )
    intra_salvo_time_sec: Optional[float] = Field(
        default=None,
        description="Intra-salvo time between launches, in seconds "
                    "(checklist labels this 'Intra-Solvo Time').",
    )

    # Propulsion
    total_burn_time_sec: Optional[float] = Field(
        default=None,
        description="Total burn time across all propulsion stages, in seconds.",
    )
    ejector_time_sec: Optional[float] = Field(
        default=None,
        description="Ejector stage duration in seconds.",
    )
    ejector_thrust: Optional[str] = Field(
        default=None,
        description="Ejector thrust (free-text; checklist leaves unit column blank).",
    )
    ejector_mass_kg: Optional[float] = Field(
        default=None,
        description="Ejector mass in kilograms.",
    )
    booster_time_sec: Optional[float] = Field(
        default=None,
        description="Booster stage duration in seconds.",
    )
    booster_thrust: Optional[str] = Field(
        default=None,
        description="Booster thrust (free-text; checklist leaves unit column blank).",
    )
    booster_mass_kg: Optional[float] = Field(
        default=None,
        description="Booster mass in kilograms.",
    )
    sustain_time_sec: Optional[float] = Field(
        default=None,
        description="Sustainer stage duration in seconds.",
    )
    sustain_thrust: Optional[str] = Field(
        default=None,
        description="Sustainer thrust (free-text; checklist leaves unit column blank).",
    )
    sustain_mass_kg: Optional[float] = Field(
        default=None,
        description="Sustainer mass in kilograms.",
    )

    # System
    confidence: Optional[float] = Field(
        default=None,
        description="Extraction confidence for this instance, 0-1.",
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    _v_min_intercept_km      = field_validator("min_intercept_km",     mode="before")(coerce_optional_float)
    _v_max_intercept_km      = field_validator("max_intercept_km",     mode="before")(coerce_optional_float)
    _v_min_altitude_km       = field_validator("min_altitude_km",      mode="before")(coerce_optional_float)
    _v_max_altitude_km       = field_validator("max_altitude_km",      mode="before")(coerce_optional_float)
    _v_max_launch_angle_deg  = field_validator("max_launch_angle_deg", mode="before")(coerce_optional_float)
    _v_body_length_m         = field_validator("body_length_m",        mode="before")(coerce_optional_float)
    _v_body_diameter_m       = field_validator("body_diameter_m",      mode="before")(coerce_optional_float)
    _v_total_mass_kg         = field_validator("total_mass_kg",        mode="before")(coerce_optional_float)
    _v_average_speed_mps     = field_validator("average_speed_mps",    mode="before")(coerce_optional_float)
    _v_max_speed_mps         = field_validator("max_speed_mps",        mode="before")(coerce_optional_float)
    _v_max_flyout_time_sec   = field_validator("max_flyout_time_sec",  mode="before")(coerce_optional_float)
    _v_flight_time_sec       = field_validator("flight_time_sec",      mode="before")(coerce_optional_float)
    _v_coast_time_sec        = field_validator("coast_time_sec",       mode="before")(coerce_optional_float)
    _v_intra_salvo_time_sec  = field_validator("intra_salvo_time_sec", mode="before")(coerce_optional_float)
    _v_total_burn_time_sec   = field_validator("total_burn_time_sec",  mode="before")(coerce_optional_float)
    _v_ejector_time_sec      = field_validator("ejector_time_sec",     mode="before")(coerce_optional_float)
    _v_ejector_mass_kg       = field_validator("ejector_mass_kg",      mode="before")(coerce_optional_float)
    _v_booster_time_sec      = field_validator("booster_time_sec",     mode="before")(coerce_optional_float)
    _v_booster_mass_kg       = field_validator("booster_mass_kg",      mode="before")(coerce_optional_float)
    _v_sustain_time_sec      = field_validator("sustain_time_sec",     mode="before")(coerce_optional_float)
    _v_sustain_mass_kg       = field_validator("sustain_mass_kg",      mode="before")(coerce_optional_float)
    _v_confidence            = field_validator("confidence",           mode="before")(coerce_optional_confidence)


# ----------------------------------------------------------------------
# Pass root
# ----------------------------------------------------------------------

class MissileDomainPass(BaseModel):
    """Missile-domain pass root. Emits only ``MISSILE_SYSTEM`` entities
    with flat, checklist-aligned properties. No nested subcomponent
    entities and no typed HAS_* / LAUNCHES edges.

    is_entity=True per docling-graph-docs.md Template Basics -> Root
    Document Model. graph_id_fields=[] because the pass-root is a
    synthetic container.
    """
    model_config = ConfigDict(
        extra="ignore",
        is_entity=True,
        graph_id_fields=[],
    )

    missile_systems: List[MissileSystemEntity] = edge(
        label="CONTAINS",
        description=(
            "Top-level missile systems extracted from this document. "
            "Emit when the batch contains EITHER (a) a defining structure "
            "(table, caption, labeled list, captioned figure) that "
            "identifies the system, OR (b) an explicit named mention in "
            "prose using the system's canonical designation (e.g. 'SA-2', "
            "'SA-20', 'PAC-3 MSE', '9M96'). Do NOT emit from unnamed "
            "descriptions ('the missile', 'the interceptor'). For "
            "mention-only evidence, emit identity plus directly stated "
            "properties; do not infer attachments not explicit in this "
            "batch."
        ),
        examples=[["SA-2", "SA-20"], ["PAC-3 MSE", "SM-6 Block IA"]],
        default_factory=list,
    )

    _dedupe_root_entities = model_validator(mode="before")(dedupe_entities_by_identity)
