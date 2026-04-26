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
    canonicalize_identity_text,
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_text,
    coerce_optional_confidence,
    dedupe_entities_by_identity,
    sanitize_entity_list,
)
from ._field_provenance import FieldProvenanceRow


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

_MISSILE_FORBIDDEN_SYSTEM_NAMES = {
    "FAN SONG", "SPOON REST", "FLAT FACE", "SIDE NET", "FLAP LID",
    "GRAVE STONE", "BIG BIRD", "BACK TRAP", "TOMBSTONE", "AN/MPQ-53",
    "AN/MPQ-65", "AN/SPY-1", "AN/SPY-6", "AN/TPY-2", "U-2", "SR-71",
    "RF-4C", "F-4", "F-15", "F-16", "B-52", "MIG-21", "MIG-23", "MIG-29",
    "SU-27",
}
_MISSILE_OPTIONAL_TEXT_FIELDS = {
    "nomenclature",
    "dieqp",
    "name",
    "emitter_function",
    "system_status",
    "asrd",
    "responsible_agency",
    "review_cycle",
    "next_review_date",
    "guidance_type",
    "seeker_type",
    "ejector_thrust",
    "booster_thrust",
    "sustain_thrust",
}

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
            "Canonical designation of the MISSILE / WEAPON SYSTEM itself. "
            "Accept canonical proper-noun weapon identifiers from prose "
            "when unambiguous (e.g. 'SA-2', 'SA-20', 'PAC-3 MSE', '9M96'). "
            "FORBIDDEN values — never emit any of these as system_name "
            "because they are radars, not missile/weapon systems: "
            "Fan Song, Spoon Rest, Flat Face, Side Net, Flap Lid, "
            "Grave Stone, Big Bird, Back Trap, Tombstone, AN/MPQ-53, "
            "AN/MPQ-65, AN/SPY-1, AN/SPY-6, AN/TPY-2. "
            "Also FORBIDDEN: aircraft / platform / target names (U-2, "
            "SR-71, RF-4C, F-4, F-15, F-16, B-52, MiG-21, MiG-23, "
            "MiG-29, Su-27) — these are aircraft that may be engaged "
            "by a missile, but they are NOT missile systems. "
            "If the text says 'U-2 was shot down by SA-2', only emit "
            "the missile ('SA-2'). Do NOT emit 'U-2'. "
            "Reject descriptive phrases ('the missile', 'the interceptor')."
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
        description=(
            "Digital Intelligence Equipment Parameters (DIEQP) identifier — "
            "a cross-reference ID into the DIEQP database maintained by "
            "the MDE (Mission Data Engineering) community. Only appears "
            "in IC / MDE source documents. Emit verbatim — do not infer."
        ),
    )
    name: Optional[str] = Field(
        default=None,
        description=(
            "Formal NAME field from the MDE checklist, distinct from the "
            "common ``system_name``. Often the full proper name (e.g. "
            "'Patriot Advanced Capability 3 Missile Segment Enhancement'). "
            "Emit when the source provides a formal long-form name "
            "alongside the short system_name."
        ),
        examples=["Patriot Advanced Capability 3 MSE", "S-400 Triumf"],
    )
    emitter_function: Optional[str] = Field(
        default=None,
        description=(
            "MDE-checklist emitter-function field. For a missile weapon "
            "system this is typically null (missiles don't usually have "
            "their own active emitters; their radar is a separate system). "
            "Emit a value only when the source explicitly assigns an "
            "emitter function to the missile itself (e.g. missile seeker "
            "listed as 'ACTIVE_RADAR_HOMING')."
        ),
    )
    system_status: Optional[str] = Field(
        default=None,
        description=(
            "Lifecycle status of the missile system. Typical values: "
            "OPERATIONAL (currently deployed), DEVELOPMENTAL (prototype "
            "or pre-IOC), RETIRED (withdrawn from service), UPGRADED "
            "(modified variant superseding a base model), EXPORTED "
            "(sold to foreign operators only). Emit only when the "
            "document explicitly states the status; do not infer it "
            "from historical or descriptive text."
        ),
        examples=["OPERATIONAL", "RETIRED", "DEVELOPMENTAL"],
    )
    asrd: Optional[str] = Field(
        default=None,
        description=(
            "ASRD identifier — a catalog code from the All-Source "
            "Reference Document, a classified IC catalog. Emit verbatim "
            "when explicitly stated; do not infer."
        ),
    )
    responsible_agency: Optional[str] = Field(
        default=None,
        description=(
            "Organization responsible for maintaining the MDE record for "
            "this missile system. Typically a 3-letter IC acronym (e.g. "
            "'IWC' = Information Warfare Center, 'NASIC' = National Air "
            "and Space Intelligence Center, 'ONI' = Office of Naval "
            "Intelligence, 'NGIC' = National Ground Intelligence Center, "
            "'MSIC' = Missile and Space Intelligence Center)."
        ),
        examples=["IWC", "NASIC", "MSIC"],
    )
    review_cycle: Optional[str] = Field(
        default=None,
        description=(
            "Scheduled cadence at which the MDE record for this missile "
            "is reviewed. Typical values: 'annual', 'biennial', '2-year', "
            "'3-year', or an explicit duration. Free-text; emit verbatim."
        ),
        examples=["annual", "biennial", "3-year"],
    )
    next_review_date: Optional[str] = Field(
        default=None,
        description=(
            "Date of the next scheduled MDE review. Prefer ISO 8601 "
            "(YYYY-MM-DD); otherwise emit the date string verbatim."
        ),
        examples=["2026-06-30", "June 2026"],
    )

    # Range / Guidance
    min_intercept_km: Optional[float] = Field(
        default=None,
        description=(
            "Minimum effective intercept / engagement range, in "
            "kilometers. Below this range the missile's safety arm, "
            "initialization sequence, or terminal-guidance lock-on cannot "
            "complete in time. Typical SAMs: 2-10 km minimum. If source "
            "gives the value in miles, nautical miles, feet, or meters, "
            "convert to km before emitting. Do not copy the raw source "
            "number when the source unit is not already kilometers."
        ),
        examples=[2.0, 5.0],
    )
    max_intercept_km: Optional[float] = Field(
        default=None,
        description=(
            "Maximum effective intercept / engagement range, in "
            "kilometers. The outer edge of the missile's engagement "
            "envelope against an assumed target profile. Use the "
            "document's effective range value here. Do NOT use slant "
            "range, ferry range, or maximum kinematic distance unless the "
            "document explicitly says those are the effective engagement "
            "range. Convert source units to km."
        ),
        examples=[35.0, 400.0],
    )
    min_altitude_km: Optional[float] = Field(
        default=None,
        description=(
            "Minimum engagement altitude, in kilometers. Below this "
            "altitude the missile cannot acquire or intercept. Legacy "
            "SAMs (SA-2) have ~1 km minimum; modern systems reach "
            "sea-skimming altitudes (<0.05 km). Only populate when the "
            "document explicitly states a minimum altitude / floor; do "
            "not infer from generic system knowledge."
        ),
        examples=[0.05, 1.0],
    )
    max_altitude_km: Optional[float] = Field(
        default=None,
        description=(
            "Maximum engagement altitude (ceiling), in kilometers. "
            "The top of the missile's engagement envelope. Classical "
            "high-altitude SAMs (SA-2 / S-75): ~25 km. Exo-atmospheric "
            "interceptors (SM-3, GBI): >100 km. If the source gives "
            "ceiling in feet or meters, convert to km."
        ),
        examples=[18.0, 35.0, 180.0],
    )
    max_launch_angle_deg: Optional[float] = Field(
        default=None,
        description=(
            "Maximum launch angle off vertical / elevation, in degrees. "
            "For vertical-launch systems this may be fixed at 0° (true "
            "vertical) or slightly canted. For tilt-launchers (SA-2, "
            "SA-3) it's typically 50-80°. 90° means the launcher can "
            "fire horizontally."
        ),
        examples=[60.0, 80.0],
    )
    guidance_type: Optional[str] = Field(
        default=None,
        description=(
            "Primary guidance method used to drive the missile to intercept. "
            "Enum values and meanings: "
            "COMMAND = ground station computes aim and uplinks guidance "
            "from an external control unit; "
            "BEAM_RIDING = missile rides the radar beam to target; "
            "SARH = semi-active radar homing (missile homes on target-"
            "illuminated RF energy; target illumination from a separate "
            "radar); "
            "ARH = active radar homing (missile carries its own seeker "
            "radar); "
            "IR = passive infrared homing; "
            "TVM = track-via-missile (missile relays target data back to "
            "ground, hybrid command + SARH); "
            "GPS_INS = inertial-navigation with GPS updates (usually for "
            "mid-course of a longer-range missile); "
            "DUAL_MODE = combines two modes (e.g. IR + radar). "
            "Many modern missiles combine phases: MID_COURSE + TERMINAL."
        ),
    )
    seeker_type: Optional[str] = Field(
        default=None,
        description=(
            "Terminal-phase seeker technology. Enum values: "
            "ACTIVE_RADAR = onboard radar illuminator + receiver (PAC-3, "
            "AMRAAM); "
            "SEMI_ACTIVE_RADAR = receives target echoes from a separate "
            "illuminating radar; "
            "PASSIVE_RADAR = detects target's own RF emissions (anti-"
            "radiation missiles); "
            "IR = infrared / thermal guidance; "
            "DUAL_MODE = IR + radar; "
            "ARM = anti-radiation homing; "
            "GPS_INS = navigation-based (no terminal seeker); "
            "COMMAND = no onboard seeker, ground-uplink guidance only "
            "from an external control unit."
        ),
    )

    # Physical Characteristics
    missile_photo: Optional[bool] = Field(
        default=None,
        description=(
            "Whether a missile photograph is included in the record (Y/N). "
            "Use null when the document does not state this — do NOT "
            "default to false. Emit true only when the text explicitly "
            "indicates a photograph is included."
        ),
    )
    body_length_m: Optional[float] = Field(
        default=None,
        description=(
            "Overall missile body length in meters, nose-tip to tail "
            "(booster included if permanently attached). Typical SAM "
            "ranges vary by class. If source "
            "gives feet, multiply by 0.3048."
        ),
    )
    body_diameter_m: Optional[float] = Field(
        default=None,
        description=(
            "Missile body diameter (airframe cross-section) in meters. "
            "If source gives inches, "
            "multiply by 0.0254; if centimeters, divide by 100."
        ),
    )
    total_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Total missile launch mass in kilograms (all stages + "
            "warhead + fuel). If source gives "
            "pounds, divide by 2.205. "
            "Note: the MDE checklist unit column shows 'deg' in row 20 — "
            "that's a source typo; this field is mass in kg."
        ),
    )

    # Performance Characteristics
    average_speed_mps: Optional[float] = Field(
        default=None,
        description=(
            "Average in-flight speed in meters per second (averaged over "
            "powered + coast phases). If source gives Mach, multiply by ~340 "
            "(sea-level Mach ≈ 340 m/s). If km/h, divide by 3.6."
        ),
    )
    max_speed_mps: Optional[float] = Field(
        default=None,
        description=(
            "Peak in-flight speed in meters per second (typically reached "
            "at end of boost phase). Same unit-conversion rules as "
            "average_speed_mps."
        ),
    )
    max_flyout_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Maximum total time of flight from launch to intercept or "
            "self-destruct, in seconds. Determined by fuel burn + coast "
            "dynamics + range. Typical long-range SAMs: 60-120 s."
        ),
        examples=[60.0, 120.0],
    )
    flight_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Typical / nominal flight time from launch to expected "
            "intercept, in seconds (for a median engagement profile). "
            "Shorter than max_flyout_time_sec, which is the worst-case."
        ),
        examples=[30.0, 60.0],
    )
    coast_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Duration of the unpowered (post-motor-burnout) coast phase, "
            "in seconds. The missile relies on residual kinetic energy + "
            "aerodynamic control. Longer-range missiles have more coast "
            "time; short-range MANPADS may have near-zero coast."
        ),
        examples=[10.0, 45.0],
    )
    intra_salvo_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Time between successive missile launches in a salvo, in "
            "seconds. For multi-missile engagements (shoot-look-shoot "
            "or ripple-fire tactics). Note: the MDE checklist labels "
            "this 'Intra-Solvo Time' — source typo."
        ),
        examples=[6.0, 30.0],
    )

    # Propulsion
    total_burn_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Total powered-flight burn time across all propulsion stages "
            "(boost + sustain, plus ejector if applicable), in seconds. "
            "Excludes any coast phase. Typical long-range two-stage SAMs: "
            "20-40 s total."
        ),
        examples=[22.0, 35.0],
    )
    ejector_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Ejector / launch-eject stage duration in seconds. The ejector "
            "is a low-thrust charge that pushes the missile out of a "
            "vertical launch canister before main motor ignition (cold "
            "launch). Typical: 0.5-2 s. Null if the missile uses hot "
            "launch (no ejector stage)."
        ),
        examples=[1.0, 1.5],
    )
    ejector_thrust: Optional[str] = Field(
        default=None,
        description=(
            "Ejector-stage thrust. Free-text because the checklist unit "
            "column is blank — source may give newtons, kilonewtons, or "
            "pounds-force. Emit verbatim with the unit the source uses."
        ),
        examples=["12 kN", "2700 lbf"],
    )
    ejector_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Mass of the ejector stage + its expended propellant, in "
            "kilograms. Separates from the missile and is discarded "
            "before main motor burn. Typical: 5-30 kg."
        ),
        examples=[20.0, 10.0],
    )
    booster_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Booster (first-stage) motor burn duration in seconds. The "
            "booster provides initial acceleration from rest to a target "
            "velocity at which the sustainer can take over. Typical "
            "SAM boosters: 4-10 s."
        ),
        examples=[6.0, 8.0],
    )
    booster_thrust: Optional[str] = Field(
        default=None,
        description=(
            "Booster-stage thrust. Free-text (see ejector_thrust). Emit "
            "verbatim with units as the source provides."
        ),
        examples=["220 kN", "50000 lbf"],
    )
    booster_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Booster stage mass (hardware + propellant) in kilograms. For "
            "two-stage missiles the booster typically separates after "
            "burnout. Typical: 300-1000 kg for medium / long-range SAMs."
        ),
        examples=[700.0, 1200.0],
    )
    sustain_time_sec: Optional[float] = Field(
        default=None,
        description=(
            "Sustainer (second-stage) motor burn duration in seconds. "
            "Sustains the velocity gained from the booster and maintains "
            "kinetic energy through the intercept geometry. Typical: "
            "10-30 s for long-range SAMs."
        ),
        examples=[18.0, 25.0],
    )
    sustain_thrust: Optional[str] = Field(
        default=None,
        description=(
            "Sustainer-stage thrust. Free-text (see ejector_thrust). "
            "Typically lower than booster thrust because the sustainer "
            "burns longer at lower pressure."
        ),
        examples=["50 kN", "11000 lbf"],
    )
    sustain_mass_kg: Optional[float] = Field(
        default=None,
        description=(
            "Sustainer stage mass (hardware + propellant) in kilograms. "
            "For single-stage missiles this is the whole motor. Typical "
            "range: 100-500 kg for medium / long-range SAMs."
        ),
        examples=[250.0, 400.0],
    )

    # System
    confidence: Optional[float] = Field(
        default=None,
        description=(
            "Overall extraction confidence for this missile instance, "
            "0-1. System-populated field. Leave null unless the document "
            "itself explicitly provides a confidence value."
        ),
        ge=0.0, le=1.0,
        json_schema_extra={"system_field": True},
    )

    @field_validator("system_name", mode="before")
    @classmethod
    def _v_system_name(cls, value: Any) -> str:
        normalized = canonicalize_identity_text(value)
        if normalized is None:
            raise ValueError("system_name must be a non-empty missile identity from the document")
        return normalized

    _v_nomenclature_text       = field_validator("nomenclature",           mode="before")(coerce_optional_text)
    _v_dieqp_text              = field_validator("dieqp",                  mode="before")(coerce_optional_text)
    _v_name_text               = field_validator("name",                   mode="before")(coerce_optional_text)
    _v_emitter_function_text   = field_validator("emitter_function",       mode="before")(coerce_optional_text)
    _v_system_status_text      = field_validator("system_status",          mode="before")(coerce_optional_text)
    _v_asrd_text               = field_validator("asrd",                   mode="before")(coerce_optional_text)
    _v_responsible_agency_text = field_validator("responsible_agency",     mode="before")(coerce_optional_text)
    _v_review_cycle_text       = field_validator("review_cycle",           mode="before")(coerce_optional_text)
    _v_next_review_date_text   = field_validator("next_review_date",       mode="before")(coerce_optional_text)
    _v_guidance_type_text      = field_validator("guidance_type",          mode="before")(coerce_optional_text)
    _v_seeker_type_text        = field_validator("seeker_type",            mode="before")(coerce_optional_text)
    _v_ejector_thrust_text     = field_validator("ejector_thrust",         mode="before")(coerce_optional_text)
    _v_booster_thrust_text     = field_validator("booster_thrust",         mode="before")(coerce_optional_text)
    _v_sustain_thrust_text     = field_validator("sustain_thrust",         mode="before")(coerce_optional_text)
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

    field_provenance: list[FieldProvenanceRow] = Field(
        default_factory=list,
        description=(
            "Per-field source snippets the LLM quoted to justify each "
            "field value. The docling-graph service post-processes these "
            "to resolve element_uid by substring-matching the snippet "
            "against the input chunks. Empty by default; populated only "
            "when the prompt asks for it (Phase 3 of the flat-schema "
            "profile refactor)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _sanitize_and_dedupe_root_entities(cls, values: Any) -> Any:
        values = sanitize_entity_list(
            cls,
            values,
            list_field="missile_systems",
            identity_field="system_name",
            optional_text_fields=_MISSILE_OPTIONAL_TEXT_FIELDS,
            forbidden_identities=_MISSILE_FORBIDDEN_SYSTEM_NAMES,
        )
        return dedupe_entities_by_identity(cls, values)
